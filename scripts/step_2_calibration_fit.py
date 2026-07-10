"""Step 2: T-scaling fit on dev for 4 benchmarks × 3 prompt variants.

设计:
- 用 `no_culture` method 在 dev (valid) split 收集 LLM 概率分布
- 对每个 (benchmark, prompt_variant) 单独 fit T*
- 输出 calibrations.json: 12 个 T* + nll_before/after + n_dev

dev 数量 (Theorist 强调每 bench 独立 + 每 prompt 独立):
- WVB: 7 valid qids × 多 (continent, urb_rur, edu) cells → 用全部 (~ 350 个分布)
- GOQA: 511 valid items → 抽样 80 (节约 cost)
- NormAd: 508 valid → 抽样 80
- BLEnD MC: 100 valid → 全用
- BLEnD SAQ: SAQ 没有 calibration 概念 (是文本输出, T-scaling 不适用) → 跳过

总调用 = (350 + 80 + 80 + 100) × 3 prompts = 1830 调用 ≈ $0.12

3 个 prompt variants (Theorist insight §3 calibration 最敏感, 加密到 3 个):
- v0: 标准 (现有 prompts.render_baseline 直接用)
- v1: 加 reasoning hint ("Think step by step about the cultural context")
- v2: 强调输出格式 ("Output ONLY the JSON. No prose, no markdown.")
"""

from __future__ import annotations

import json
import logging
import os
import random
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.model_client import chat_json, MODEL_NAME  # noqa: E402
from src.calibrate import fit_temperature, negative_log_likelihood_from_probs  # noqa: E402
from src.io import load_bench  # noqa: E402
from src.prompts import (  # noqa: E402
    render_baseline,
    render_norm_judgment,
    render_survey_distribution,
    render_daily_knowledge,
    to_chat_messages,
)
from src.run_predictions import dispatch_items  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)
logger = logging.getLogger("step2")

JST = timezone(timedelta(hours=9))

# ---------------------------------------------------------------------------
# 配置
# ---------------------------------------------------------------------------

SEED = 42
N_DEV = {
    "wvb": None,        # 全部 7 qid × 所有 cells
    "goqa": 80,
    "normad": 80,
    "blend_mc": None,   # 全部 100 valid
}
PROMPT_VARIANTS = [0, 1, 2]

REASONING_HINT = " Think briefly about the relevant cultural context before answering."
FORMAT_HINT = " Output ONLY the JSON. No prose, no markdown fences."


# ---------------------------------------------------------------------------
# Prompt variants 应用
# ---------------------------------------------------------------------------

def apply_prompt_variant(rendered: dict, variant: int) -> dict:
    """对已渲染的 prompt 加 variant 修饰."""
    if variant == 0:
        return rendered
    user = rendered["user"]
    if variant == 1:
        user = user + "\n\n" + REASONING_HINT.strip()
    elif variant == 2:
        user = user + "\n\n" + FORMAT_HINT.strip()
    return {**rendered, "user": user}


# ---------------------------------------------------------------------------
# WVB dev probs
# ---------------------------------------------------------------------------

def collect_wvb_dev(prompt_variant: int) -> tuple[np.ndarray, list[int], list[dict]]:
    """收集 WVB dev (7 qids × all cells) 在 no_culture × prompt_variant 上的 (probs, y_argmax)."""
    sp = json.load(open(
        "${EXPERIMENT_ROOT}/data/splits/wvb_split.json"
        if Path("${EXPERIMENT_ROOT}/data/splits/wvb_split.json").exists()
        else ROOT / "data" / "splits" / "wvb_split.json"
    ))
    valid_qids = set(sp["valid"])
    meta = json.load(open(
        "${EXPERIMENT_ROOT}/data/processed/wvb/probe_question_metadata.json"
        if Path("${EXPERIMENT_ROOT}/data/processed/wvb/probe_question_metadata.json").exists()
        else ROOT / "data" / "processed" / "wvb" / "probe_question_metadata.json"
    ))
    df = load_bench("wvb.probe_distributions")
    df = df[df["qid"].isin(valid_qids)].reset_index(drop=True)
    logger.info("WVB dev: %d cells over %d qids", len(df), len(valid_qids))

    items = []
    for i, row in df.iterrows():
        qid = row["qid"]
        opts_raw = row["answer_options"]
        if isinstance(opts_raw, str):
            opts_raw = json.loads(opts_raw)
        opts = [str(o) for o in opts_raw]
        gold_raw = row["distribution"]
        if isinstance(gold_raw, str):
            gold_raw = json.loads(gold_raw)
        gold = np.array([float(gold_raw[o]) for o in opts], dtype=np.float64)
        gold = gold / gold.sum()
        q_meta = meta.get(qid, {})
        q_text = q_meta.get("question", f"WVS question {qid}")

        rendered = render_baseline(
            "no_culture",
            item_id=f"{qid}_{i}",
            question_text=q_text,
            answer_options=opts,
        )
        rendered = apply_prompt_variant(rendered, prompt_variant)
        items.append({
            "item_id": f"wvb_{qid}_{i}",
            "messages": to_chat_messages(rendered),
            "options": opts,
            "gold": gold,
        })

    return _run_distribution_dispatch(items, bench="wvb")


def _safe_extract_probs(parsed, options):
    if not isinstance(parsed, dict):
        return None
    d = None
    for k in ("probabilities", "probs", "distribution"):
        if k in parsed and isinstance(parsed[k], (dict, list)):
            d = parsed[k]
            break
    if d is None:
        return None
    if isinstance(d, list):
        if len(d) != len(options):
            return None
        try:
            arr = np.array([float(x) for x in d], dtype=np.float64)
        except Exception:
            return None
    else:
        try:
            arr = np.array([float(d.get(str(o), d.get(o, 0.0))) for o in options], dtype=np.float64)
        except Exception:
            return None
    if np.any(arr < -1e-9):
        return None
    s = arr.sum()
    if s <= 0:
        return None
    return arr / s


def _run_distribution_dispatch(items, bench):
    """并发跑 LLM, 收集 (pred_probs, gold_argmax, raw record).

    按 K 分组返回 (因为不同 qid 可能不同 K, np.stack 要求同 shape).
    返回: list[(K, probs(N,K), y(N,))], sanity
    """
    def handler(item):
        res = chat_json(messages=item["messages"], max_retries=2)
        return {
            "item_id": item["item_id"],
            "ok": bool(res.ok),
            "parsed": res.parsed,
            "prompt_tokens": res.prompt_tokens,
            "completion_tokens": res.completion_tokens,
            "latency_seconds": res.latency_seconds,
            "options": item["options"],
            "gold": item["gold"].tolist() if isinstance(item["gold"], np.ndarray) else item["gold"],
            "n_attempts": res.n_attempts,
        }

    results = dispatch_items(items, handler, max_workers=8, on_error="record")

    sanity = []
    for r in results[:200]:
        sanity.append({
            "item_id": r["item_id"], "ok": r.get("ok"),
            "n_attempts": r.get("n_attempts"),
            "prompt_tokens": r.get("prompt_tokens"),
            "completion_tokens": r.get("completion_tokens"),
            "latency": r.get("latency_seconds"),
        })

    # 按 K 分组
    by_k: dict[int, tuple[list, list]] = {}
    for r in results:
        if not r.get("ok"):
            continue
        opts = r["options"]
        p = _safe_extract_probs(r["parsed"], opts)
        if p is None:
            continue
        gold = np.array(r["gold"], dtype=np.float64)
        if gold.shape != p.shape:
            continue
        K = p.shape[0]
        if K not in by_k:
            by_k[K] = ([], [])
        by_k[K][0].append(p)
        by_k[K][1].append(int(np.argmax(gold)))

    grouped = []
    for K in sorted(by_k.keys()):
        probs_list, y_list = by_k[K]
        if len(probs_list) >= 5:
            grouped.append((K, np.stack(probs_list, axis=0), y_list))
    return grouped, sanity


# ---------------------------------------------------------------------------
# GOQA dev probs
# ---------------------------------------------------------------------------

def collect_goqa_dev(prompt_variant: int, sample_n: int = 80) -> tuple[np.ndarray, list[int], list[dict]]:
    sp = json.load(open(
        "${EXPERIMENT_ROOT}/data/splits/goqa_split.json"
        if Path("${EXPERIMENT_ROOT}/data/splits/goqa_split.json").exists()
        else ROOT / "data" / "splits" / "goqa_split.json"
    ))
    valid_ids = set(sp["valid"])
    df = load_bench("goqa")
    df = df[df["question_id"].isin(valid_ids)].reset_index(drop=True)
    if sample_n and len(df) > sample_n:
        df = df.sample(n=sample_n, random_state=SEED).reset_index(drop=True)
    logger.info("GOQA dev: %d items", len(df))

    items = []
    for i, row in df.iterrows():
        qid = row["question_id"]
        q_text = row["question"]
        opts = row["options_parsed"]
        if isinstance(opts, str):
            opts = json.loads(opts)
        sels = row["selections_parsed"]
        if isinstance(sels, str):
            sels = json.loads(sels)
        if not opts or not sels:
            continue
        country = next(iter(sels.keys()))
        gold = np.array([float(x) for x in sels[country]], dtype=np.float64)
        if gold.sum() <= 0:
            continue
        gold = gold / gold.sum()

        rendered = render_baseline(
            "no_culture",
            item_id=f"{qid}_{i}",
            question_text=q_text,
            answer_options=opts,
        )
        rendered = apply_prompt_variant(rendered, prompt_variant)
        items.append({
            "item_id": f"goqa_{qid}_{i}",
            "messages": to_chat_messages(rendered),
            "options": opts,
            "gold": gold,
        })

    return _run_distribution_dispatch(items, bench="goqa")


# ---------------------------------------------------------------------------
# NormAd dev probs (3-class y/n/neutral, calibration on probs)
# ---------------------------------------------------------------------------

NORMAD_OPTIONS = ["yes", "no", "neutral"]


def collect_normad_dev(prompt_variant: int, sample_n: int = 80):
    sp = json.load(open(
        "${EXPERIMENT_ROOT}/data/splits/normad_split.json"
        if Path("${EXPERIMENT_ROOT}/data/splits/normad_split.json").exists()
        else ROOT / "data" / "splits" / "normad_split.json"
    ))
    valid_ids = set(sp["valid"])
    df = load_bench("normad")
    df = df[df["story_id"].isin(valid_ids)].reset_index(drop=True)
    if sample_n and len(df) > sample_n:
        df = df.sample(n=sample_n, random_state=SEED).reset_index(drop=True)
    logger.info("NormAd dev: %d items", len(df))

    items = []
    for i, row in df.iterrows():
        sid = row["story_id"]
        story = row["Story"]
        gold = str(row["Gold Label"]).strip().lower()
        gold_idx = NORMAD_OPTIONS.index(gold) if gold in NORMAD_OPTIONS else 0

        question = (
            f"Below is a social scenario. Decide whether the described behavior is "
            f"socially acceptable in a generic / unspecified cultural context.\n\n"
            f"Scenario:\n{story}\n\n"
            f"Choose exactly one of: yes (acceptable), no (not acceptable), neutral (depends)."
        )
        rendered = render_baseline(
            "no_culture",
            item_id=sid,
            question_text=question,
            answer_options=NORMAD_OPTIONS,
        )
        rendered = apply_prompt_variant(rendered, prompt_variant)
        items.append({
            "item_id": f"normad_{sid}",
            "messages": to_chat_messages(rendered),
            "options": NORMAD_OPTIONS,
            "gold": np.eye(3)[gold_idx],  # 转 one-hot
        })

    return _run_distribution_dispatch(items, bench="normad")


# ---------------------------------------------------------------------------
# BLEnD MC dev probs
# ---------------------------------------------------------------------------

def collect_blend_mc_dev(prompt_variant: int):
    sp = json.load(open(
        "${EXPERIMENT_ROOT}/data/splits/blend_split.json"
        if Path("${EXPERIMENT_ROOT}/data/splits/blend_split.json").exists()
        else ROOT / "data" / "splits" / "blend_split.json"
    ))
    valid_ids = set(sp["valid"])
    df = load_bench("blend.mc")
    df = df[df["ID"].isin(valid_ids)].reset_index(drop=True)
    # 每 ID 取 1 MCQID (避免 18 国 distractor 爆量到 80k+); 100 valid ID → 100 items
    df = df.groupby("ID", as_index=False).first().reset_index(drop=True)
    logger.info("BLEnD MC dev: %d items (1 per valid ID, matched %d valid IDs)",
                len(df), len(valid_ids))

    items = []
    for i, row in df.iterrows():
        mcqid = row["MCQID"]
        choices_raw = row["choices"]
        if isinstance(choices_raw, str):
            choices_raw = json.loads(choices_raw)
        letters = sorted(choices_raw.keys())
        options = [choices_raw[L] for L in letters]
        prompt = row["prompt"]
        gold_letter = row["answer_idx"]
        gold_idx = letters.index(gold_letter) if gold_letter in letters else 0

        rendered = render_baseline(
            "no_culture",
            item_id=mcqid,
            question_text=prompt,
            answer_options=options,
        )
        rendered = apply_prompt_variant(rendered, prompt_variant)
        items.append({
            "item_id": f"blend_mc_{mcqid}",
            "messages": to_chat_messages(rendered),
            "options": options,
            "gold": np.eye(len(letters))[gold_idx],
        })

    return _run_distribution_dispatch(items, bench="blend_mc")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

BENCH_COLLECTORS = {
    "wvb": collect_wvb_dev,
    "goqa": lambda pv: collect_goqa_dev(pv, sample_n=N_DEV["goqa"]),
    "normad": lambda pv: collect_normad_dev(pv, sample_n=N_DEV["normad"]),
    "blend_mc": collect_blend_mc_dev,
}


def main():
    out_dir = ROOT / "experiments" / "calibration_fits"
    out_dir.mkdir(parents=True, exist_ok=True)
    analysis_dir = ROOT / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    all_calibrations = {}
    sanity_records = {}
    total_calls = 0
    total_tokens_in = 0
    total_tokens_out = 0

    for bench in ["wvb", "goqa", "normad", "blend_mc"]:
        all_calibrations[bench] = {}
        sanity_records[bench] = {}
        for pv in PROMPT_VARIANTS:
            logger.info("=== %s × pv%d ===", bench, pv)
            collector = BENCH_COLLECTORS[bench]
            try:
                grouped, sanity = collector(pv)
            except Exception as e:
                logger.exception("collector %s pv%d failed", bench, pv)
                all_calibrations[bench][f"pv{pv}"] = {"error": str(e)}
                continue

            total_n = sum(len(y) for _, _, y in grouped)
            if total_n < 10:
                logger.warning("%s pv%d: 仅 %d 个有效样本, 跳过 fit", bench, pv, total_n)
                all_calibrations[bench][f"pv{pv}"] = {
                    "T_star": None, "n_dev": total_n,
                    "error": "insufficient samples",
                }
                sanity_records[bench][f"pv{pv}"] = sanity[:50]
                continue

            # 按 K 分别 fit
            per_k_results = {}
            for K, probs, y in grouped:
                cal = fit_temperature(probs, y, bench=f"{bench}_K{K}", prompt_variant=pv)
                per_k_results[f"K{K}"] = {
                    "T_star": cal.T_star,
                    "nll_before": cal.nll_before,
                    "nll_after": cal.nll_after,
                    "n_dev": cal.n_dev_samples,
                }
                logger.info("  K=%d T*=%.3f NLL %.4f → %.4f (n=%d)",
                            K, cal.T_star, cal.nll_before, cal.nll_after, cal.n_dev_samples)

            # 也计算"全 K 加权平均 T*" 作为单一 fallback
            all_T = [v["T_star"] * v["n_dev"] for v in per_k_results.values()]
            all_n = sum(v["n_dev"] for v in per_k_results.values())
            T_weighted = sum(all_T) / all_n if all_n > 0 else 1.0
            all_calibrations[bench][f"pv{pv}"] = {
                "per_K": per_k_results,
                "T_star_weighted_avg": T_weighted,
                "n_dev_total": total_n,
            }
            sanity_records[bench][f"pv{pv}"] = sanity[:50]

            # 累计 token 估算 (从 sanity 前 50 估均值再外推)
            valid_san = [s for s in sanity if s.get("ok")]
            if valid_san:
                mean_in = np.mean([s.get("prompt_tokens", 0) for s in valid_san])
                mean_out = np.mean([s.get("completion_tokens", 0) for s in valid_san])
                n_calls_this = len(sanity)
                total_calls += n_calls_this
                total_tokens_in += int(mean_in * n_calls_this)
                total_tokens_out += int(mean_out * n_calls_this)

            logger.info("  T*=%.3f NLL %.4f → %.4f (n=%d)",
                        cal.T_star, cal.nll_before, cal.nll_after, cal.n_dev_samples)

    elapsed = time.time() - t0

    # Cost (gpt-4o-mini 占位)
    cost = (total_tokens_in / 1e6) * 0.15 + (total_tokens_out / 1e6) * 0.60

    log = {
        "step": "step_2_calibration_fit",
        "timestamp_jst": datetime.now(JST).isoformat(),
        "model": MODEL_NAME,
        "machine": "gpu_server",
        "elapsed_seconds": elapsed,
        "total_calls": total_calls,
        "total_prompt_tokens": total_tokens_in,
        "total_completion_tokens": total_tokens_out,
        "cost_usd_estimate": cost,
        "calibrations": all_calibrations,
    }
    out_p = out_dir / "calibrations.json"
    out_p.write_text(json.dumps(log, ensure_ascii=False, indent=2))

    # Sanity 单独 (大)
    sanity_p = out_dir / "step_2_sanity.json"
    sanity_p.write_text(json.dumps(sanity_records, ensure_ascii=False, indent=2))

    # Copy to analysis
    (analysis_dir / "step_2_calibrations.json").write_text(json.dumps(
        {bench: pvs for bench, pvs in all_calibrations.items()},
        ensure_ascii=False, indent=2,
    ))

    logger.info("=== DONE ===")
    logger.info("elapsed=%.1fs total_calls=%d cost~$%.4f", elapsed, total_calls, cost)
    print(json.dumps({
        "elapsed": elapsed,
        "total_calls": total_calls,
        "cost": cost,
        "T_stars": {
            bench: {pv: pv_data.get("T_star") for pv, pv_data in pvs.items()}
            for bench, pvs in all_calibrations.items()
        },
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
