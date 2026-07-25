"""H_anchor_4: 无文化提示 baseline 复现 (Researcher COMPARE 派单).

任务: 跑 readme §14.1 "无文化提示" baseline, 在四个 benchmark 子样上, 用 src/metrics
算指标, 与 literature/baseline_table.md 文献区间对比.

子样设计 (seed=42, 节约成本):
  - WVB:           50 个 (qid, continent, urb_rur, edu) 聚合行 → 分布预测 (W1, JS-D, Top-1)
  - GOQA test:     50 个 (question_id, country) 对 → 分布预测 (JS-D, TV-D, Top-1)
  - NormAd test:   50 story (Country only 模式) → Acc, Macro-F1
  - BLEnD MC:      50 个 MCQ, 各国均衡 → Acc
  - BLEnD SAQ:     25 个 (country, qid) (英语 question_en + US gold) → EM, Token F1
  共 ~225 调用; 按 1.4s/call ~ 5 min, $0.01 量级.

输出:
  - experiments/anchor_4_no_culture_baseline/results.json  (各 bench raw 数字 + 子样详情)
  - analysis/anchor_4_assertions.json                       (PASS/FAIL 逐项)
  - analysis/anchor_4_report.md                             (一页 Researcher 阅读)

运行 (gpu_server):
  AZURE_API_KEY='...' python -m experiments.anchor_4_no_culture_baseline.run
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

ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = Path(os.environ.get("EXPERIMENT_ROOT", ROOT))
sys.path.insert(0, str(ROOT))

from src.io import load_bench
from src.metrics import (
    accuracy,
    exact_match,
    js_distance,
    macro_f1,
    token_f1,
    top1_accuracy,
    tv_distance,
    wasserstein_1,
)
from src.model_client import MODEL_NAME, chat_json
from src.prompts import render_baseline, to_chat_messages

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
# 减少 openai SDK 噪音
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)
logger = logging.getLogger("anchor4")

JST = timezone(timedelta(hours=9))

# ---------------------------------------------------------------------------
# 子样数量 (Researcher 指定)
# ---------------------------------------------------------------------------
N_WVB = 50
N_GOQA = 50
N_NORMAD = 50
N_BLEND_MC = 50
N_BLEND_SAQ = 25
SEED = 42

# ---------------------------------------------------------------------------
# 文献参考区间 (literature/baseline_table.md §5)
# Vanilla "无文化" baseline 的合理量级
# ---------------------------------------------------------------------------
LIT_REF = {
    "wvb": {
        # 取 CuMA 论文 Vanilla 基线作为锚 (gpt-5.4-mini 比 Llama 强, 但 zero-shot 无文化应该在
        # vanilla ~ persona prompting 之间)
        "acc_low": 0.27,   # Vanilla baseline ~ 32% 下浮 5pp
        "acc_high": 0.42,  # Persona prompting 37% 上浮 5pp
        "emd_low": 0.20,   # Persona prompting EMD 0.31 ± 相对差
        "emd_high": 0.45,  # Vanilla EMD ~ 0.40 + 余量
        "note": "anchor_4 我们用 W1 (与 EMD 概念相同), 取 ordinal qid 计算",
    },
    "goqa": {
        # 表 5: JS-Similarity ~0.4-0.5; JS-D = 1 - JS-Sim → 0.5-0.6
        "jsd_low": 0.30,   # 留余量, JS-Sim 0.7 (强)
        "jsd_high": 0.70,  # JS-Sim 0.3 (弱)
        "tvd_low": 0.20,
        "tvd_high": 0.55,
        "note": "GOQA 没有精确文献数, 给宽区间; 用 JS Distance (sqrt-of-JS-D, [0,1])",
    },
    "normad": {
        # Country only: 51-56% (Rao et al.); 多模型 zero-shot 中段
        "acc_low": 0.40,   # 留宽下限 (gpt-5.4-mini 可能比 GPT-3.5 弱)
        "acc_high": 0.70,  # 上限 (强模型上界)
        "note": "Country only, 3-class (yes/no/neutral); 我们用 no_culture (更弱) 应该 ≤ Country only",
    },
    "blend_mc": {
        # 英语高资源 GPT-4 83-94%; GPT-3.5 75-89%; 低资源差距大
        "acc_low": 0.45,   # 混合各国家, 包含低资源拉低均值
        "acc_high": 0.95,
        "note": "混合各国 (含低资源), zero-shot 无 RAG",
    },
    "blend_saq": {
        # GPT-4 65-70%; GPT-3.5 56-65%
        "em_low": 0.05,    # SAQ EM 很低, 表面匹配
        "em_high": 0.50,
        "f1_low": 0.10,
        "f1_high": 0.65,
        "note": "用 US 数据 (高资源英语), 25 个子样, 测 EM + Token F1",
    },
}


# ---------------------------------------------------------------------------
# 工具
# ---------------------------------------------------------------------------

def _parse_json_col(x):
    """parquet 中部分列存为 JSON string, 部分原生 list/dict."""
    if isinstance(x, str):
        return json.loads(x)
    return x


def _sample(df: pd.DataFrame, n: int, seed: int = SEED) -> pd.DataFrame:
    """固定 seed 随机取 n 行."""
    if len(df) <= n:
        return df.copy()
    return df.sample(n=n, random_state=seed).reset_index(drop=True)


def _safe_distribution_from_parsed(parsed: dict, options: list[str]) -> np.ndarray | None:
    """从 LLM 返回的 parsed dict 中提取分布 over options. 兼容多个键名."""
    if not isinstance(parsed, dict):
        return None
    probs_dict = None
    for k in ("probabilities", "probs", "distribution"):
        if k in parsed and isinstance(parsed[k], (dict, list)):
            probs_dict = parsed[k]
            break
    if probs_dict is None:
        return None
    if isinstance(probs_dict, list):
        if len(probs_dict) != len(options):
            return None
        try:
            arr = np.array([float(x) for x in probs_dict], dtype=np.float64)
        except Exception:
            return None
    else:
        # dict
        try:
            arr = np.array(
                [float(probs_dict.get(str(o), probs_dict.get(o, 0.0))) for o in options],
                dtype=np.float64,
            )
        except Exception:
            return None
    if np.any(arr < -1e-9):
        return None
    s = arr.sum()
    if s <= 0:
        return None
    return arr / s


def _safe_answer(parsed: dict, options: list[str] | None = None) -> str | None:
    if not isinstance(parsed, dict):
        return None
    a = parsed.get("answer")
    if isinstance(a, str):
        return a.strip()
    if isinstance(a, list) and a:
        return str(a[0]).strip()
    return None


# ---------------------------------------------------------------------------
# WVB
# ---------------------------------------------------------------------------

def run_wvb(out_records: list, sample_n: int = N_WVB) -> dict:
    logger.info("=== WVB (n=%d) ===", sample_n)
    df = load_bench("wvb.probe_distributions")
    meta = json.loads(
        (
            DATA_ROOT / "data" / "processed" / "wvb" / "probe_question_metadata.json"
        ).read_text()
    )
    sample = _sample(df, sample_n)

    w1_list, jsd_list, top1_list = [], [], []
    fails = 0
    for i, row in sample.iterrows():
        qid = row["qid"]
        opts_raw = _parse_json_col(row["answer_options"])
        # WVB options 是 int (1..K), 当作有序 ordinal
        opts = [str(o) for o in opts_raw]
        gold_dist = _parse_json_col(row["distribution"])
        gold_arr = np.array([float(gold_dist[o]) for o in opts], dtype=np.float64)
        gold_arr = gold_arr / gold_arr.sum() if gold_arr.sum() > 0 else gold_arr

        q_meta = meta.get(qid, {})
        q_text = q_meta.get("question", f"WVS question {qid}")

        rendered = render_baseline(
            "no_culture",
            item_id=f"{qid}_{i}",
            question_text=q_text,
            answer_options=opts,
        )
        msgs = to_chat_messages(rendered)
        res = chat_json(messages=msgs, max_retries=2)

        pred_arr = _safe_distribution_from_parsed(res.parsed, opts) if res.ok else None
        if pred_arr is None:
            fails += 1
            out_records.append({
                "bench": "wvb", "qid": qid, "ok": False, "err": res.error,
            })
            continue

        # support = 1..K (ordinal); options 是 int 字符串
        support = np.array([float(o) for o in opts_raw], dtype=np.float64)
        w1 = wasserstein_1(pred_arr, gold_arr, support=support)
        jsd = js_distance(pred_arr, gold_arr)
        t1 = top1_accuracy(pred_arr, gold_arr)
        w1_list.append(w1); jsd_list.append(jsd); top1_list.append(t1)
        out_records.append({
            "bench": "wvb", "qid": qid, "ok": True,
            "w1": w1, "jsd": jsd, "top1": int(t1),
            "pred_probs": pred_arr.tolist(),
            "gold_probs": gold_arr.tolist(),
            "options": opts,
        })

    return {
        "bench": "wvb",
        "n_sampled": len(sample),
        "n_failed": int(fails),
        "n_evaluated": len(w1_list),
        "metrics": {
            "W1_mean": float(np.mean(w1_list)) if w1_list else None,
            "W1_std": float(np.std(w1_list)) if w1_list else None,
            "JS_distance_mean": float(np.mean(jsd_list)) if jsd_list else None,
            "Top1_Acc": float(np.mean(top1_list)) if top1_list else None,
        },
    }


# ---------------------------------------------------------------------------
# GOQA
# ---------------------------------------------------------------------------

def run_goqa(out_records: list, sample_n: int = N_GOQA) -> dict:
    logger.info("=== GOQA (n=%d) ===", sample_n)
    df = load_bench("goqa")
    test = df[df["split"] == "test"].reset_index(drop=True)
    sample = _sample(test, sample_n)

    jsd_list, tvd_list, top1_list = [], [], []
    fails = 0
    for i, row in sample.iterrows():
        qid = row["question_id"]
        q_text = row["question"]
        opts = _parse_json_col(row["options_parsed"])
        sels = _parse_json_col(row["selections_parsed"])
        if not opts or not sels:
            continue
        # 每行有一个国家 (Anthropic GOQA 已 explode), 取首个
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
        msgs = to_chat_messages(rendered)
        res = chat_json(messages=msgs, max_retries=2)
        pred = _safe_distribution_from_parsed(res.parsed, opts) if res.ok else None
        if pred is None:
            fails += 1
            out_records.append({"bench": "goqa", "qid": qid, "country": country, "ok": False, "err": res.error})
            continue

        jsd = js_distance(pred, gold)
        tvd = tv_distance(pred, gold)
        t1 = top1_accuracy(pred, gold)
        jsd_list.append(jsd); tvd_list.append(tvd); top1_list.append(t1)
        out_records.append({
            "bench": "goqa", "qid": qid, "country": country, "ok": True,
            "jsd": jsd, "tvd": tvd, "top1": int(t1),
            "pred_probs": pred.tolist(), "gold_probs": gold.tolist(),
            "options": opts,
        })

    return {
        "bench": "goqa",
        "n_sampled": len(sample),
        "n_failed": int(fails),
        "n_evaluated": len(jsd_list),
        "metrics": {
            "JS_distance_mean": float(np.mean(jsd_list)) if jsd_list else None,
            "JS_similarity_proxy": float(1.0 - np.mean(jsd_list)) if jsd_list else None,
            "TV_distance_mean": float(np.mean(tvd_list)) if tvd_list else None,
            "Top1_Acc": float(np.mean(top1_list)) if top1_list else None,
        },
    }


# ---------------------------------------------------------------------------
# NormAd
# ---------------------------------------------------------------------------

NORMAD_LABEL_MAP = {"yes": "yes", "no": "no", "neutral": "neutral"}
NORMAD_OPTIONS = ["yes", "no", "neutral"]


def run_normad(out_records: list, sample_n: int = N_NORMAD) -> dict:
    logger.info("=== NormAd (n=%d) ===", sample_n)
    df = load_bench("normad")
    split = pd.read_csv(
        DATA_ROOT / "data" / "processed" / "normad" / "story_id_split.csv"
    )
    test_ids = set(split[split["split"] == "test"]["story_id"])
    test_df = df[df["story_id"].isin(test_ids)].reset_index(drop=True)
    sample = _sample(test_df, sample_n)

    y_true, y_pred = [], []
    fails = 0
    for i, row in sample.iterrows():
        sid = row["story_id"]
        story = row["Story"]
        gold = row["Gold Label"].strip().lower()

        # readme §14.1 "无文化提示" → 不给国家信息
        # 我们的提示让 LLM 判断 story 在 generic context 下是否合适
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
        res = chat_json(messages=to_chat_messages(rendered), max_retries=2)
        ans = _safe_answer(res.parsed) if res.ok else None
        if ans is None:
            fails += 1
            out_records.append({"bench": "normad", "story_id": sid, "ok": False, "err": res.error})
            continue
        ans_norm = ans.strip().lower()
        if ans_norm not in NORMAD_OPTIONS:
            # 试着抓首字
            if ans_norm.startswith("yes"):
                ans_norm = "yes"
            elif ans_norm.startswith("no"):
                ans_norm = "no"
            elif ans_norm.startswith("neutral") or "depend" in ans_norm:
                ans_norm = "neutral"
            else:
                fails += 1
                out_records.append({"bench": "normad", "story_id": sid, "ok": False, "err": f"unmapped: {ans!r}"})
                continue

        # 整数 label for metric (0=yes 1=no 2=neutral)
        m = {"yes": 0, "no": 1, "neutral": 2}
        y_true.append(m[gold])
        y_pred.append(m[ans_norm])
        out_records.append({
            "bench": "normad", "story_id": sid, "ok": True,
            "country": row["Country"], "gold": gold, "pred": ans_norm,
            "correct": int(gold == ans_norm),
        })

    acc = accuracy(y_true, y_pred) if y_true else None
    mf1 = macro_f1(y_true, y_pred, labels=[0, 1, 2]) if y_true else None

    return {
        "bench": "normad",
        "n_sampled": len(sample),
        "n_failed": int(fails),
        "n_evaluated": len(y_true),
        "metrics": {
            "Accuracy": acc,
            "Macro_F1": mf1,
        },
    }


# ---------------------------------------------------------------------------
# BLEnD MC
# ---------------------------------------------------------------------------

def run_blend_mc(out_records: list, sample_n: int = N_BLEND_MC) -> dict:
    logger.info("=== BLEnD MC (n=%d) ===", sample_n)
    df = load_bench("blend.mc")
    # 按 country 分层均衡 (各国家随机取等量), 避免单一国家主导
    countries = sorted(df["country"].unique().tolist())
    per_country = max(1, sample_n // len(countries))
    rng = np.random.default_rng(SEED)
    samples = []
    for c in countries:
        sub = df[df["country"] == c]
        if len(sub) == 0:
            continue
        take = sub.sample(n=min(per_country, len(sub)), random_state=int(rng.integers(1, 1_000_000)))
        samples.append(take)
    sample = pd.concat(samples).sample(n=min(sample_n, sum(len(x) for x in samples)), random_state=SEED).reset_index(drop=True)

    y_true, y_pred = [], []
    fails = 0
    for i, row in sample.iterrows():
        mcqid = row["MCQID"]
        choices_raw = _parse_json_col(row["choices"])  # dict A->.., B->...
        # choices: {"A": "...", "B": "...", "C": "...", "D": "..."}
        letters = sorted(choices_raw.keys())
        options = [choices_raw[L] for L in letters]
        prompt = row["prompt"]
        gold_letter = row["answer_idx"]

        rendered = render_baseline(
            "no_culture",
            item_id=mcqid,
            question_text=prompt,
            answer_options=options,
        )
        res = chat_json(messages=to_chat_messages(rendered), max_retries=2)
        ans = _safe_answer(res.parsed) if res.ok else None
        if ans is None:
            fails += 1
            out_records.append({"bench": "blend_mc", "mcqid": mcqid, "ok": False, "err": res.error})
            continue
        # 把 LLM 答案 (文本) 匹配回 letter
        pred_letter = None
        ans_norm = ans.strip().lower()
        for L, opt in choices_raw.items():
            if ans_norm == opt.strip().lower():
                pred_letter = L
                break
        if pred_letter is None:
            # 也接受 LLM 直接给字母
            ah = ans.strip().upper()
            if ah in choices_raw:
                pred_letter = ah
        if pred_letter is None:
            fails += 1
            out_records.append({"bench": "blend_mc", "mcqid": mcqid, "ok": False, "err": f"unmapped: {ans!r}"})
            continue

        # 编码 letter → idx
        L2I = {L: i for i, L in enumerate(letters)}
        y_true.append(L2I[gold_letter])
        y_pred.append(L2I[pred_letter])
        out_records.append({
            "bench": "blend_mc", "mcqid": mcqid, "ok": True,
            "country": row["country"],
            "gold_letter": gold_letter, "pred_letter": pred_letter,
            "correct": int(gold_letter == pred_letter),
        })

    acc = accuracy(y_true, y_pred) if y_true else None
    return {
        "bench": "blend_mc",
        "n_sampled": len(sample),
        "n_failed": int(fails),
        "n_evaluated": len(y_true),
        "metrics": {"Accuracy": acc},
    }


# ---------------------------------------------------------------------------
# BLEnD SAQ
# ---------------------------------------------------------------------------

def run_blend_saq(out_records: list, sample_n: int = N_BLEND_SAQ) -> dict:
    logger.info("=== BLEnD SAQ (n=%d) ===", sample_n)
    # 用 US 高资源英语数据 + 文本 gold (annotations[].en_answers)
    raw_path = (
        DATA_ROOT
        / "data"
        / "raw"
        / "blend"
        / "hf"
        / "data"
        / "annotations_hf"
        / "US_data.json"
    )
    if not raw_path.exists():
        return {
            "bench": "blend_saq",
            "n_sampled": 0, "n_failed": 0, "n_evaluated": 0,
            "metrics": {"EM": None, "Token_F1": None},
            "skipped_reason": f"raw file not found: {raw_path}",
        }
    annot = json.loads(raw_path.read_text())
    # 取有 gold 的样本
    valid = [
        x for x in annot
        if x.get("annotations") and any(a.get("en_answers") for a in x["annotations"])
    ]
    rng = random.Random(SEED)
    rng.shuffle(valid)
    chosen = valid[:sample_n]

    em_list, f1_list = [], []
    fails = 0
    for x in chosen:
        sid = x["ID"]
        q = x.get("en_question") or x.get("question")
        # gold = 所有 en_answers 列表 (多 gold OK, 取最高 score)
        golds = []
        for a in x["annotations"]:
            for g in a.get("en_answers") or []:
                if isinstance(g, str) and g.strip():
                    golds.append(g.strip().lower())
        if not golds:
            continue

        rendered = render_baseline(
            "no_culture",
            item_id=sid,
            question_text=q,
            answer_options=None,
        )
        res = chat_json(messages=to_chat_messages(rendered), max_retries=2)
        ans = _safe_answer(res.parsed) if res.ok else None
        if ans is None:
            fails += 1
            out_records.append({"bench": "blend_saq", "qid": sid, "ok": False, "err": res.error})
            continue
        em = max(exact_match(ans, g) for g in golds)
        f1 = max(token_f1(ans, g) for g in golds)
        em_list.append(em); f1_list.append(f1)
        out_records.append({
            "bench": "blend_saq", "qid": sid, "ok": True,
            "country": "US", "pred": ans, "golds": golds[:5],
            "em": em, "f1": f1,
        })

    return {
        "bench": "blend_saq",
        "n_sampled": len(chosen),
        "n_failed": int(fails),
        "n_evaluated": len(em_list),
        "metrics": {
            "EM": float(np.mean(em_list)) if em_list else None,
            "Token_F1": float(np.mean(f1_list)) if f1_list else None,
        },
    }


# ---------------------------------------------------------------------------
# 断言: 与文献区间对比
# ---------------------------------------------------------------------------

def _within(x: float | None, lo: float, hi: float) -> bool:
    if x is None:
        return False
    return lo <= x <= hi


def build_assertions(results: dict) -> dict:
    a: dict = {"anchor": "H_anchor_4", "debug_move": "COMPARE", "checks": [], "verdict": None}

    # WVB: W1 应在 EMD 区间; Top-1 应在 Acc 区间
    w = results["wvb"]["metrics"]
    a["checks"].append({
        "bench": "wvb",
        "metric": "W1_mean",
        "value": w["W1_mean"],
        "reference_range": [LIT_REF["wvb"]["emd_low"], LIT_REF["wvb"]["emd_high"]],
        "in_range": _within(w["W1_mean"], LIT_REF["wvb"]["emd_low"], LIT_REF["wvb"]["emd_high"]),
        "note": LIT_REF["wvb"]["note"],
    })
    a["checks"].append({
        "bench": "wvb",
        "metric": "Top1_Acc",
        "value": w["Top1_Acc"],
        "reference_range": [LIT_REF["wvb"]["acc_low"], LIT_REF["wvb"]["acc_high"]],
        "in_range": _within(w["Top1_Acc"], LIT_REF["wvb"]["acc_low"], LIT_REF["wvb"]["acc_high"]),
        "note": "CuMA vanilla ~32%, persona ~37%",
    })

    g = results["goqa"]["metrics"]
    a["checks"].append({
        "bench": "goqa",
        "metric": "JS_distance_mean",
        "value": g["JS_distance_mean"],
        "reference_range": [LIT_REF["goqa"]["jsd_low"], LIT_REF["goqa"]["jsd_high"]],
        "in_range": _within(g["JS_distance_mean"], LIT_REF["goqa"]["jsd_low"], LIT_REF["goqa"]["jsd_high"]),
        "note": LIT_REF["goqa"]["note"],
    })
    a["checks"].append({
        "bench": "goqa",
        "metric": "TV_distance_mean",
        "value": g["TV_distance_mean"],
        "reference_range": [LIT_REF["goqa"]["tvd_low"], LIT_REF["goqa"]["tvd_high"]],
        "in_range": _within(g["TV_distance_mean"], LIT_REF["goqa"]["tvd_low"], LIT_REF["goqa"]["tvd_high"]),
        "note": "tighter than JS-D",
    })

    n = results["normad"]["metrics"]
    a["checks"].append({
        "bench": "normad",
        "metric": "Accuracy",
        "value": n["Accuracy"],
        "reference_range": [LIT_REF["normad"]["acc_low"], LIT_REF["normad"]["acc_high"]],
        "in_range": _within(n["Accuracy"], LIT_REF["normad"]["acc_low"], LIT_REF["normad"]["acc_high"]),
        "note": LIT_REF["normad"]["note"],
    })

    bm = results["blend_mc"]["metrics"]
    a["checks"].append({
        "bench": "blend_mc",
        "metric": "Accuracy",
        "value": bm["Accuracy"],
        "reference_range": [LIT_REF["blend_mc"]["acc_low"], LIT_REF["blend_mc"]["acc_high"]],
        "in_range": _within(bm["Accuracy"], LIT_REF["blend_mc"]["acc_low"], LIT_REF["blend_mc"]["acc_high"]),
        "note": LIT_REF["blend_mc"]["note"],
    })

    bs = results["blend_saq"]["metrics"]
    a["checks"].append({
        "bench": "blend_saq",
        "metric": "EM",
        "value": bs["EM"],
        "reference_range": [LIT_REF["blend_saq"]["em_low"], LIT_REF["blend_saq"]["em_high"]],
        "in_range": _within(bs["EM"], LIT_REF["blend_saq"]["em_low"], LIT_REF["blend_saq"]["em_high"]),
        "note": "US 高资源 SAQ",
    })
    a["checks"].append({
        "bench": "blend_saq",
        "metric": "Token_F1",
        "value": bs["Token_F1"],
        "reference_range": [LIT_REF["blend_saq"]["f1_low"], LIT_REF["blend_saq"]["f1_high"]],
        "in_range": _within(bs["Token_F1"], LIT_REF["blend_saq"]["f1_low"], LIT_REF["blend_saq"]["f1_high"]),
        "note": "US 高资源 SAQ",
    })

    n_in = sum(1 for c in a["checks"] if c["in_range"])
    n_out = sum(1 for c in a["checks"] if not c["in_range"])
    a["summary"] = {"n_checks": len(a["checks"]), "n_in_range": n_in, "n_out_of_range": n_out}
    a["verdict"] = "PASS" if n_out == 0 else ("PARTIAL" if n_in > 0 else "FAIL")
    return a


def build_report(results: dict, assertions: dict, log: dict) -> str:
    lines = [
        "# H_anchor_4 无文化基线复现报告",
        "",
        f"- 时间 (JST): {log['timestamp_jst']}",
        f"- 模型: {log['model']}",
        "- 机器: gpu_server",
        f"- 子样 seed: {SEED}",
        "",
        "## 总览",
        "",
        f"- 总调用: {log['total_calls']}",
        f"- 总成功: {log['total_calls'] - log['total_fails']}",
        f"- 总失败 (api/json): {log['total_fails']}",
        f"- 总耗时: {log['total_seconds']:.1f}s",
        f"- 估计成本: ${log['cost_usd_estimate']:.4f}",
        "",
        "## Verdict",
        "",
        f"- **{assertions['verdict']}** ({assertions['summary']['n_in_range']}/{assertions['summary']['n_checks']} 通过区间检查)",
        "",
        "## 指标对照",
        "",
        "| Bench | Metric | Value | Lit Range | In Range |",
        "|-------|--------|-------|-----------|----------|",
    ]
    for c in assertions["checks"]:
        v = c["value"]
        v_str = f"{v:.4f}" if isinstance(v, float) else str(v)
        lo, hi = c["reference_range"]
        lines.append(f"| {c['bench']} | {c['metric']} | {v_str} | [{lo:.2f}, {hi:.2f}] | {'✅' if c['in_range'] else '❌'} |")

    lines += [
        "",
        "## 各 bench 详情",
        "",
    ]
    for b in ["wvb", "goqa", "normad", "blend_mc", "blend_saq"]:
        r = results[b]
        lines.append(f"### {b}")
        lines.append("")
        lines.append(f"- n_sampled={r['n_sampled']}, n_evaluated={r['n_evaluated']}, n_failed={r['n_failed']}")
        lines.append("- metrics: " + ", ".join(
            f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
            for k, v in r["metrics"].items()
        ))
        lines.append("")

    lines += [
        "## 文献参考",
        "",
        "见 literature/baseline_table.md §5 (跨基准无文化基线汇总).",
        "",
        "## 注意",
        "",
        "- 子样规模 (50/25), 数字仅做量级 sanity check, 不替代全 test 评测.",
        "- 失败样本 = LLM 返回 schema 错误 (parse 不出概率/答案), 已计入 n_failed.",
        "- temperature 不传给 gpt-5 reasoning model, 实际 t 由 model 默认行为承担.",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

# 价格 (gpt-4o-mini 占位)
PRICE_INPUT_PER_M = 0.15
PRICE_OUTPUT_PER_M = 0.60


def main():
    out_dir = Path(__file__).resolve().parent
    out_dir.mkdir(parents=True, exist_ok=True)
    analysis_dir = ROOT / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    records: list = []

    res_wvb = run_wvb(records)
    res_goqa = run_goqa(records)
    res_normad = run_normad(records)
    res_blend_mc = run_blend_mc(records)
    res_blend_saq = run_blend_saq(records)

    results = {
        "wvb": res_wvb,
        "goqa": res_goqa,
        "normad": res_normad,
        "blend_mc": res_blend_mc,
        "blend_saq": res_blend_saq,
    }

    total_calls = sum(r["n_sampled"] for r in results.values())
    total_fails = sum(r["n_failed"] for r in results.values())

    # 粗略 token / cost 估算: 从 records 里没有 token 字段 (省内存), 用 anchor_3 平均值估
    avg_pt = 200; avg_ct = 60  # 比 anchor_3 prompt 长一些
    cost_est = (total_calls * avg_pt / 1e6) * PRICE_INPUT_PER_M + \
               (total_calls * avg_ct / 1e6) * PRICE_OUTPUT_PER_M

    log = {
        "anchor": "H_anchor_4",
        "model": MODEL_NAME,
        "machine": "gpu_server",
        "timestamp_jst": datetime.now(JST).isoformat(),
        "seed": SEED,
        "n_per_bench": {
            "wvb": N_WVB, "goqa": N_GOQA, "normad": N_NORMAD,
            "blend_mc": N_BLEND_MC, "blend_saq": N_BLEND_SAQ,
        },
        "total_calls": total_calls,
        "total_fails": total_fails,
        "total_seconds": time.time() - t0,
        "cost_usd_estimate": cost_est,
        "results": results,
        "n_records": len(records),
    }

    # 写 records 到独立大文件 (留在 gpu_server, 不必 rsync 回 GCP)
    (out_dir / "records.json").write_text(json.dumps(records, ensure_ascii=False, indent=2))
    # 写 results.json
    (out_dir / "results.json").write_text(json.dumps(log, ensure_ascii=False, indent=2))

    assertions = build_assertions(results)
    (analysis_dir / "anchor_4_assertions.json").write_text(json.dumps(assertions, ensure_ascii=False, indent=2))
    (analysis_dir / "anchor_4_report.md").write_text(build_report(results, assertions, log))

    logger.info("=== DONE ===")
    logger.info("verdict: %s (%d/%d in range)", assertions["verdict"],
                assertions["summary"]["n_in_range"], assertions["summary"]["n_checks"])
    logger.info("total calls=%d fails=%d cost~$%.4f", total_calls, total_fails, cost_est)

    print(json.dumps({
        "verdict": assertions["verdict"],
        "summary": assertions["summary"],
        "metrics_per_bench": {b: r["metrics"] for b, r in results.items()},
        "total_seconds": log["total_seconds"],
        "cost_estimate": cost_est,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
