"""Step 5 aggregate: 主表 + ablation 数字 → paired bootstrap CI + headroom-normalized + p-value.

输出 (按 Researcher 派单要求):
- analysis/main_results.json (主表 mean ± CI per bench × method)
- analysis/ablation_results.json (fPC/FpC/FPc 三 ablation, 含 high/low resource 分层)
- analysis/running_report.md (一页给 Researcher, 含 goal check matrix)
"""

from __future__ import annotations

import json
import logging
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.aggregate_results import (  # noqa: E402
    LOWER_IS_BETTER, ORACLE_FLOOR, bootstrap_ci, headroom_normalized,
    paired_bootstrap_diff,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("step5")


MAIN_DIR = ROOT / "experiments" / "step_3_main" / "runs"
ABL_DIR = ROOT / "experiments" / "step_4_ablation" / "runs"
ANALYSIS_DIR = ROOT / "analysis"
RESOURCE_GROUPING_PATH = ANALYSIS_DIR / "resource_grouping.json"

METHODS = ("no_culture", "country", "demographic", "prototype", "general_semantic", "culturelens_rc")
BENCHES = ("wvb", "goqa", "normad", "blend_mc", "blend_saq")
ABLATION_COMBOS = ("fPC", "FpC", "FPc")


def load_cells(d: Path) -> list[dict]:
    cells = []
    if not d.exists():
        return cells
    for p in sorted(d.glob("*.json")):
        try:
            cells.append(json.loads(p.read_text()))
        except Exception as e:
            logger.warning("load fail %s: %s", p, e)
    return cells


def aggregate_main_table(cells: list[dict]) -> dict:
    """主表: 每 (method, bench, metric) → 9 个值 (3 prompts × 3 seeds), bootstrap CI."""
    grouped: dict = defaultdict(list)
    for c in cells:
        s = c.get("summary", {})
        m, b = s.get("method"), s.get("bench")
        if not m or not b:
            continue
        for mk, mv in (s.get("metrics") or {}).items():
            if isinstance(mv, (int, float)) and mv is not None:
                grouped[(m, b, mk)].append(float(mv))
    out = {}
    for (method, bench, metric), values in grouped.items():
        mean, lo, hi = bootstrap_ci(values, n_boot=1000, seed=42)
        out.setdefault(bench, {}).setdefault(method, {})[metric] = {
            "mean": mean, "ci_lo": lo, "ci_hi": hi,
            "n_runs": len(values),
            "values": values,
        }
    return out


def aggregate_ablation(cells: list[dict]) -> dict:
    """Ablation: (combo, bench, metric) → 9 个值 (或 3 for fPC); 加 high/low resource 分层."""
    grouped: dict = defaultdict(list)
    for c in cells:
        s = c.get("summary", {})
        method = s.get("method", "")  # ablation_fPC / ablation_FpC / ablation_FPc
        if not method.startswith("ablation_"):
            continue
        combo = method[len("ablation_"):]
        b = s.get("bench")
        if not b:
            continue
        for mk, mv in (s.get("metrics") or {}).items():
            if isinstance(mv, (int, float)) and mv is not None:
                grouped[(combo, b, mk)].append(float(mv))
    out = {}
    for (combo, bench, metric), values in grouped.items():
        mean, lo, hi = bootstrap_ci(values, n_boot=1000, seed=42)
        out.setdefault(bench, {}).setdefault(combo, {})[metric] = {
            "mean": mean, "ci_lo": lo, "ci_hi": hi,
            "n_runs": len(values), "values": values,
        }
    return out


def compute_paired_bootstrap(main_table: dict, methods: list[str],
                              baseline: str = "no_culture") -> dict:
    """对每 (bench, method, metric), paired bootstrap CI of (method - baseline)
    over 9 (prompt, seed) pairs."""
    out = {}
    for bench, ms in main_table.items():
        if baseline not in ms:
            continue
        base = ms[baseline]
        out[bench] = {}
        for m in methods:
            if m == baseline or m not in ms:
                continue
            out[bench][m] = {}
            for metric, mdata in ms[m].items():
                if metric not in base:
                    continue
                a_vals = mdata["values"]
                b_vals = base[metric]["values"]
                if len(a_vals) != len(b_vals):
                    continue
                diff = paired_bootstrap_diff(a_vals, b_vals, n_boot=1000, seed=42)
                hn = headroom_normalized(mdata["mean"], base[metric]["mean"], metric)
                out[bench][m][metric] = {
                    "method_mean": mdata["mean"],
                    "baseline_mean": base[metric]["mean"],
                    "headroom_normalized_improvement": hn,
                    **diff,
                }
    return out


def by_resource_tier(cells_main: list[dict], cells_abl: list[dict]) -> dict:
    """按 high/low resource 分层算 (主要给 H0.method_prototype 用).

    用每 cell 的 derived (per-item) records, 按 item['country'] 在 resource_grouping 里查 tier.
    """
    if not RESOURCE_GROUPING_PATH.exists():
        return {}
    rg = json.loads(RESOURCE_GROUPING_PATH.read_text())

    def country_tier(bench: str, country: str) -> str | None:
        bench_key = "blend" if bench.startswith("blend") else bench
        cd = rg.get(bench_key, {}).get("countries", {})
        if country in cd:
            return cd[country].get("tier")
        # case insensitive
        lookup = {k.lower(): v for k, v in cd.items()}
        return lookup.get(country.lower(), {}).get("tier") if country else None

    out: dict = {"main": {}, "ablation": {}}

    def aggregate_with_tier(cells, target):
        bucket: dict = defaultdict(list)  # (key, bench, tier, metric) → values per cell
        for c in cells:
            s = c.get("summary", {})
            method = s.get("method")
            b = s.get("bench")
            if not method or not b:
                continue
            derived = c.get("derived") or []
            # Per-item metric grouped by tier
            tier_vals: dict = defaultdict(list)
            for d in derived:
                country = d.get("country")
                tier = country_tier(b, country) if country else None
                if tier is None:
                    continue
                for mk, mv in d.items():
                    if mk in ("item_id", "country", "gold", "pred", "gold_letter", "pred_letter"):
                        continue
                    if isinstance(mv, (int, float)):
                        tier_vals[(tier, mk)].append(float(mv))
            # 该 cell 内每 tier 的 mean
            for (tier, mk), vals in tier_vals.items():
                if not vals:
                    continue
                cell_mean = float(np.mean(vals))
                bucket[(method, b, tier, mk)].append(cell_mean)
        for (method, b, tier, mk), vals in bucket.items():
            mean, lo, hi = bootstrap_ci(vals, n_boot=500, seed=42)
            target.setdefault(b, {}).setdefault(method, {}).setdefault(tier, {})[mk] = {
                "mean": mean, "ci_lo": lo, "ci_hi": hi,
                "n_cells": len(vals),
            }

    aggregate_with_tier(cells_main, out["main"])
    aggregate_with_tier(cells_abl, out["ablation"])
    return out


def _fmt_v(v):
    if v is None:
        return "—"
    if isinstance(v, float):
        return f"{v:.4f}"
    return str(v)


def render_report(main_table, ablation, comparisons, tier) -> str:
    lines = ["# RUNNING Phase Report — Step 5 Aggregate", ""]
    lines.append("## Goal Check Matrix (Theorist insight §1 → method assignment)")
    lines.append("")
    lines.append("| Hypothesis | Bench Expected | Bench Observed | Verdict |")
    lines.append("|-----------|----------------|----------------|---------|")

    def get_metric(bench, method, key):
        return main_table.get(bench, {}).get(method, {}).get(key, {}).get("mean")

    # H0.method_filtering: WVB/GOQA filter on (FPC) > general_semantic (no filter)
    wvb_clr = get_metric("wvb", "culturelens_rc", "W1_mean")
    wvb_gs = get_metric("wvb", "general_semantic", "W1_mean")
    goqa_clr = get_metric("goqa", "culturelens_rc", "JS_distance_mean")
    goqa_gs = get_metric("goqa", "general_semantic", "JS_distance_mean")

    def fmt_check(observed_str, ok):
        return ("✅ PASS" if ok else "⚠️ MIXED") + " | " + observed_str

    lines.append(f"| H0.method_filtering | WVB W1: filter<general | filter={_fmt_v(wvb_clr)} vs general={_fmt_v(wvb_gs)} | {'✅' if wvb_clr is not None and wvb_gs is not None and wvb_clr < wvb_gs else '⚠️'} |")
    # H0.method_prototype: GOQA/BLEnD SAQ 高资源 prototype ≥ no_culture
    goqa_proto = get_metric("goqa", "prototype", "JS_distance_mean")
    goqa_noc = get_metric("goqa", "no_culture", "JS_distance_mean")
    lines.append(f"| H0.method_prototype | GOQA JS-D: proto<noc | proto={_fmt_v(goqa_proto)} vs noc={_fmt_v(goqa_noc)} | {'✅' if goqa_proto is not None and goqa_noc is not None and goqa_proto < goqa_noc else '⚠️'} |")
    # H0.method_calibration: NormAd Acc-equivalent unchanged; 但其他 metric (NLL/Brier)/ distribution improved
    normad_clr = get_metric("normad", "culturelens_rc", "Accuracy")
    normad_noc = get_metric("normad", "no_culture", "Accuracy")
    lines.append(f"| H0.method_calibration | NormAd culturelens_rc Acc>no_culture | clr={_fmt_v(normad_clr)} vs noc={_fmt_v(normad_noc)} | {'✅' if normad_clr is not None and normad_noc is not None and normad_clr > normad_noc else '⚠️'} |")

    lines.append("")

    lines.append("## Main Table (mean ± 95% bootstrap CI over 9 = 3 prompts × 3 seeds)")
    lines.append("")
    for bench in BENCHES:
        if bench not in main_table:
            continue
        lines.append(f"### {bench}")
        lines.append("")
        ms = main_table[bench]
        first_method = next(iter(ms.values()))
        metric_keys = list(first_method.keys())
        lines.append("| Method | " + " | ".join(metric_keys) + " |")
        lines.append("|" + "|".join(["---"] * (1 + len(metric_keys))) + "|")
        for m in METHODS:
            if m not in ms:
                continue
            row = [m]
            for mk in metric_keys:
                d = ms[m].get(mk, {})
                row.append(f"{_fmt_v(d.get('mean'))} [{_fmt_v(d.get('ci_lo'))}, {_fmt_v(d.get('ci_hi'))}]")
            lines.append("| " + " | ".join(row) + " |")
        lines.append("")

    lines.append("## Paired Bootstrap vs no_culture (95% CI, p-value 2-sided)")
    lines.append("")
    for bench in BENCHES:
        if bench not in comparisons:
            continue
        lines.append(f"### {bench}")
        lines.append("")
        lines.append("| Method | Metric | Δ (method − noc) | 95% CI | p-value | Headroom-norm |")
        lines.append("|--------|--------|------------------|--------|---------|---------------|")
        for method, mtrs in comparisons[bench].items():
            for mk, d in mtrs.items():
                lines.append(f"| {method} | {mk} | {_fmt_v(d.get('mean_diff'))} | "
                             f"[{_fmt_v(d.get('lo'))}, {_fmt_v(d.get('hi'))}] | "
                             f"{_fmt_v(d.get('p_value_two_sided'))} | "
                             f"{_fmt_v(d.get('headroom_normalized_improvement'))} |")
        lines.append("")

    lines.append("## Ablation (3 combos × 4 bench)")
    lines.append("")
    lines.append("- fPC = filter OFF (general semantic + proto + calib)")
    lines.append("- FpC = proto OFF (filter + calib, retrieval only)")
    lines.append("- FPc = calib OFF (filter + proto, no T-scaling)")
    lines.append("- FPC = full (主表 culturelens_rc)")
    lines.append("")
    for bench in ("wvb", "goqa", "normad", "blend_mc"):
        if bench not in ablation:
            continue
        lines.append(f"### {bench}")
        lines.append("")
        cs = ablation[bench]
        first_combo = next(iter(cs.values()))
        metric_keys = list(first_combo.keys())
        lines.append("| Combo | " + " | ".join(metric_keys) + " |")
        lines.append("|" + "|".join(["---"] * (1 + len(metric_keys))) + "|")
        # FPC row from main_table
        full = main_table.get(bench, {}).get("culturelens_rc", {})
        if full:
            row = ["FPC (full)"]
            for mk in metric_keys:
                d = full.get(mk, {})
                row.append(f"{_fmt_v(d.get('mean'))}")
            lines.append("| " + " | ".join(row) + " |")
        for combo in ABLATION_COMBOS:
            if combo not in cs:
                continue
            row = [combo]
            for mk in metric_keys:
                d = cs[combo].get(mk, {})
                row.append(f"{_fmt_v(d.get('mean'))}")
            lines.append("| " + " | ".join(row) + " |")
        lines.append("")

    lines.append("## Tier-stratified (high vs low resource, prototype H0)")
    lines.append("")
    if tier and tier.get("main"):
        for bench in ("goqa", "blend_mc"):
            if bench not in tier["main"]:
                continue
            lines.append(f"### {bench}")
            lines.append("")
            lines.append("| Method | Tier | Top1/Acc (mean) |")
            lines.append("|--------|------|-----------------|")
            for m in ("no_culture", "prototype", "culturelens_rc"):
                if m not in tier["main"][bench]:
                    continue
                for ttier in ("high", "low"):
                    if ttier not in tier["main"][bench][m]:
                        continue
                    tdata = tier["main"][bench][m][ttier]
                    # 取 top1 / correct 优先
                    candidate = tdata.get("top1") or tdata.get("correct") or tdata.get("em")
                    val = candidate["mean"] if candidate else None
                    lines.append(f"| {m} | {ttier} | {_fmt_v(val)} |")
            lines.append("")

    return "\n".join(lines)


def main():
    main_cells = load_cells(MAIN_DIR)
    abl_cells = load_cells(ABL_DIR)
    logger.info("loaded %d main cells, %d ablation cells", len(main_cells), len(abl_cells))

    main_table = aggregate_main_table(main_cells)
    ablation = aggregate_ablation(abl_cells)
    comparisons = compute_paired_bootstrap(main_table, list(METHODS), baseline="no_culture")
    tier = by_resource_tier(main_cells, abl_cells)

    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    (ANALYSIS_DIR / "main_results.json").write_text(json.dumps({
        "main_table": main_table,
        "paired_vs_no_culture": comparisons,
    }, ensure_ascii=False, indent=2, default=str))
    (ANALYSIS_DIR / "ablation_results.json").write_text(json.dumps({
        "ablation": ablation,
        "tier_stratified": tier,
    }, ensure_ascii=False, indent=2, default=str))
    (ANALYSIS_DIR / "running_report.md").write_text(render_report(main_table, ablation, comparisons, tier))

    logger.info("=== DONE === wrote main_results.json + ablation_results.json + running_report.md")
    print("main_table benches:", list(main_table.keys()))
    print("ablation benches:", list(ablation.keys()))


if __name__ == "__main__":
    main()
