"""Step 4 ablation: 84 runs.

Ablation combos (4 main bench only, 不含 blend_saq):
- fPC (filter OFF): 4 bench x 1 prompt x 3 seeds = 12
- FpC (proto OFF):  4 bench x 3 prompts x 3 seeds = 36
- FPc (calib OFF): 4 bench x 3 prompts x 3 seeds = 36
"""

from __future__ import annotations

import json
import logging
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.step_3_main import (  # noqa: E402
    BENCH_LOADERS, BENCH_N_SAMPLES, _compute_metrics, _fmt_evidence,
    _safe_extract_probs, apply_prompt_variant, get_prototype_card, get_T_star,
    retrieve_general, retrieve_hierarchical,
)
from src.model_client import chat_json  # noqa: E402
from src.calibrate import apply_temperature  # noqa: E402
from src.metrics import js_distance, top1_accuracy, tv_distance, wasserstein_1  # noqa: E402
from src.prompts import render_baseline, to_chat_messages  # noqa: E402
from src.run_predictions import dispatch_items  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)
logger = logging.getLogger("step4")

JST = timezone(timedelta(hours=9))


def render_ablation(*, combo, bench, item_id, country, question_text,
                     answer_options, topic=None, prompt_variant=0):
    use_filter = combo[0] == "F"
    use_proto = combo[1] == "P"
    if not country:
        country = "(unspecified)"
    if use_filter:
        retrieved = retrieve_hierarchical(question_text, country=country, topic=topic, top_k=5)
    else:
        retrieved = retrieve_general(question_text, top_k=5)
    ev_text = _fmt_evidence(retrieved)
    proto = get_prototype_card(bench, country) if use_proto else None
    if use_proto and proto:
        r = render_baseline(
            "prototype", item_id=item_id,
            question_text=question_text + "\n\nRetrieved evidence:\n" + ev_text,
            answer_options=answer_options,
            country_or_region=country, cultural_prototype=proto,
        )
    else:
        r = render_baseline(
            "country", item_id=item_id,
            question_text=question_text + "\n\nRetrieved evidence:\n" + ev_text,
            answer_options=answer_options,
            country_or_region=country,
        )
    return apply_prompt_variant(r, prompt_variant)


def run_ablation_cell(combo, bench, prompt_variant, seed, n_samples=None, max_workers=8):
    logger.info("=== ablation %s x %s x pv%d x seed%d (n=%s) ===",
                combo, bench, prompt_variant, seed, n_samples)
    use_calib = combo[2] == "C"
    items = BENCH_LOADERS[bench](seed, n_samples)
    if not items:
        return {"error": "no items"}

    task_items = []
    for it in items:
        rendered = render_ablation(
            combo=combo, bench=bench, item_id=it["item_id"],
            country=it.get("country"), question_text=it["question_text"],
            answer_options=it.get("answer_options"),
            topic=it.get("topic"), prompt_variant=prompt_variant,
        )
        task_items.append({
            "item_id": it["item_id"],
            "messages": to_chat_messages(rendered),
            "item": it,
        })

    def handler(t):
        res = chat_json(messages=t["messages"], max_retries=2)
        return {
            "item_id": t["item_id"], "ok": bool(res.ok), "parsed": res.parsed,
            "prompt_tokens": res.prompt_tokens, "completion_tokens": res.completion_tokens,
            "latency_seconds": res.latency_seconds, "n_attempts": res.n_attempts,
        }

    results = dispatch_items(task_items, handler, max_workers=max_workers, on_error="record")
    by_item = {r["item_id"]: r for r in results}
    out = _compute_metrics(bench, "ablation_" + combo, prompt_variant, items, by_item)

    # If use_calib and bench is distribution, apply T* and recompute distribution metrics
    if use_calib and bench in ("wvb", "goqa"):
        new_jsd, new_tvd, new_top1, new_w1 = [], [], [], []
        new_derived, new_sanity = [], []
        for it in items:
            r = by_item.get(it["item_id"])
            if not r or not r.get("ok"):
                continue
            pred = _safe_extract_probs(r["parsed"], it["answer_options"])
            if pred is None:
                continue
            gold = np.array(it["gold_dist"], dtype=np.float64)
            T = get_T_star(bench, prompt_variant, it["K"])
            pred = apply_temperature(pred, T)
            new_jsd.append(js_distance(pred, gold))
            new_tvd.append(tv_distance(pred, gold))
            new_top1.append(top1_accuracy(pred, gold))
            if bench == "wvb" and it.get("support") is not None:
                new_w1.append(wasserstein_1(pred, gold, support=it["support"]))
            d = {"item_id": it["item_id"], "country": it.get("country"),
                 "jsd": new_jsd[-1], "tvd": new_tvd[-1], "top1": int(new_top1[-1])}
            if new_w1 and bench == "wvb":
                d["w1"] = new_w1[-1]
            new_derived.append(d)
            if len(new_sanity) < 100:
                new_sanity.append({**d, "pred_probs": pred.tolist(), "gold_probs": gold.tolist()})
        if new_derived:
            summary = out.get("summary", {})
            summary["metrics"]["JS_distance_mean"] = float(np.mean(new_jsd))
            summary["metrics"]["TV_distance_mean"] = float(np.mean(new_tvd))
            summary["metrics"]["Top1_Acc"] = float(np.mean(new_top1))
            if new_w1:
                summary["metrics"]["W1_mean"] = float(np.mean(new_w1))
                summary["metrics"]["W1_std"] = float(np.std(new_w1))
            out = {"summary": summary, "sanity": new_sanity, "derived": new_derived}
    return out


def main_ablation():
    out_dir = ROOT / "experiments" / "step_4_ablation"
    runs_dir = out_dir / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    progress_path = out_dir / "progress.json"

    completed = set()
    if progress_path.exists():
        prog = json.loads(progress_path.read_text())
        completed = set(prog.get("completed", []))
        logger.info("RESUME: %d cells", len(completed))
    else:
        prog = {
            "start_ts_jst": datetime.now(JST).isoformat(),
            "completed": [], "errors": [],
            "n_calls_total": 0, "cost_usd_estimate": 0.0,
        }

    benches = ("wvb", "goqa", "normad", "blend_mc")
    cells = []
    for bench in benches:
        for seed in (42, 123, 456):
            cells.append(("fPC", bench, 0, seed))
    for bench in benches:
        for pv in (0, 1, 2):
            for seed in (42, 123, 456):
                cells.append(("FpC", bench, pv, seed))
    for bench in benches:
        for pv in (0, 1, 2):
            for seed in (42, 123, 456):
                cells.append(("FPc", bench, pv, seed))

    logger.info("=== Step 4 ablation: %d cells ===", len(cells))
    t0 = time.time()

    for idx, (combo, bench, pv, seed) in enumerate(cells):
        cell_id = combo + "_" + bench + "_pv" + str(pv) + "_seed" + str(seed)
        if cell_id in completed:
            continue
        cell_path = runs_dir / (cell_id + ".json")
        if cell_path.exists():
            completed.add(cell_id)
            continue
        logger.info("[%d/%d] %s", idx + 1, len(cells), cell_id)
        n = BENCH_N_SAMPLES[bench]
        try:
            cell = run_ablation_cell(combo, bench, pv, seed, n_samples=n, max_workers=8)
        except Exception as e:
            logger.exception("cell %s failed", cell_id)
            prog["errors"].append({"cell": cell_id, "error": type(e).__name__ + ": " + str(e)})
            cell = {"summary": {"error": str(e)}, "sanity": [], "derived": []}
        cell_path.write_text(json.dumps(cell, ensure_ascii=False, indent=2, default=str))
        completed.add(cell_id)
        prog["completed"] = list(completed)
        n_items = cell.get("summary", {}).get("n_items", 0)
        prog["n_calls_total"] += n_items
        prog["cost_usd_estimate"] = prog["n_calls_total"] * 0.000066
        progress_path.write_text(json.dumps(prog, ensure_ascii=False, indent=2))

    elapsed = time.time() - t0
    prog["done_ts_jst"] = datetime.now(JST).isoformat()
    prog["total_elapsed_seconds"] = elapsed
    progress_path.write_text(json.dumps(prog, ensure_ascii=False, indent=2))
    logger.info("=== ABLATION DONE === %d, $%.4f, %.1fs",
                len(completed), prog["cost_usd_estimate"], elapsed)


if __name__ == "__main__":
    main_ablation()
