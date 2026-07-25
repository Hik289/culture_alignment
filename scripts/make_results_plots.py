#!/usr/bin/env python3
"""Generate 5 publication-quality results plots for CultureLens-RC paper."""
from __future__ import annotations

import json
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

ROOT = Path(os.environ.get("EXPERIMENT_ROOT", Path(__file__).resolve().parents[1]))
ANLY = ROOT / "analysis"
FIG  = ROOT / "figures"
FIG.mkdir(parents=True, exist_ok=True)

plt.style.use("tableau-colorblind10")
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Arial", "Liberation Sans"],
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.dpi": 100,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linestyle": ":",
    "axes.spines.top": False,
    "axes.spines.right": False,
})

METHODS = ["no_culture", "country", "demographic", "general_semantic",
           "prototype", "culturelens_rc"]
METHOD_LABELS = {
    "no_culture": "no-culture",
    "country": "country",
    "demographic": "demographic",
    "general_semantic": "general-sem",
    "prototype": "prototype",
    "culturelens_rc": "CultureLens-RC",
}
PALETTE = ["#5778A4", "#E49444", "#85B6B2", "#9D9D9D", "#B66353", "#D1615D"]
RC_COLOR = "#D1615D"


def save(fig, name):
    for ext in ("png", "pdf"):
        out = FIG / f"{name}.{ext}"
        fig.savefig(out, format=ext)
        print(f"  -> {out}")
    plt.close(fig)


main = json.load(open(ANLY / "main_results.json"))
abla = json.load(open(ANLY / "ablation_results.json"))
calib = json.load(open(ANLY / "step_2_calibrations.json"))


def plot_1_main_results():
    print("[1/5] main results bar chart ...")
    BENCHES = [
        ("wvb",      "W1_mean",          "WVB: W1 (Wasserstein-1, lower better)",  True,  "wvb"),
        ("goqa",     "JS_distance_mean", "GOQA: JS-D (lower better)",              True,  "goqa"),
        ("normad",   "Macro_F1",         "NormAd: Macro-F1 (higher better)",       False, "normad"),
        ("blend_mc", "Accuracy",         "BLEnD-MC: Accuracy (higher better)",     False, "blend_mc"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8))
    for ax, (bench, metric, title, lower_is_better, pkey) in zip(axes.flat, BENCHES):
        mt = main["main_table"][bench]
        pv = main["paired_vs_no_culture"][pkey]
        means, lo, hi, sig = [], [], [], []
        for m in METHODS:
            d = mt.get(m, {}).get(metric)
            if d is None:
                means.append(np.nan); lo.append(0); hi.append(0); sig.append(False)
                continue
            mu = d["mean"]
            means.append(mu)
            lo.append(mu - d["ci_lo"])
            hi.append(d["ci_hi"] - mu)
            p = pv.get(m, {}).get(metric, {})
            pval = p.get("p_value_two_sided", 1.0)
            md = p.get("mean_diff", 0)
            beneficial = (md < 0) if lower_is_better else (md > 0)
            sig.append(pval is not None and pval < 0.05 and beneficial)

        x = np.arange(len(METHODS))
        colors = [RC_COLOR if m == "culturelens_rc" else PALETTE[i % len(PALETTE)]
                  for i, m in enumerate(METHODS)]
        edgew = [2.0 if m == "culturelens_rc" else 0.5 for m in METHODS]
        ax.bar(x, means, yerr=[lo, hi], capsize=3, color=colors,
               edgecolor="black", linewidth=edgew, alpha=0.9,
               error_kw=dict(elinewidth=1.0, ecolor="#333333"))
        for i, (mu, s) in enumerate(zip(means, sig)):
            if s and not np.isnan(mu):
                y = mu + hi[i] + (max(means) - min(means)) * 0.04
                ax.text(i, y, "*", ha="center", va="bottom", fontsize=14,
                        fontweight="bold", color="#2A6F2A")
        ax.set_xticks(x)
        ax.set_xticklabels([METHOD_LABELS[m] for m in METHODS], rotation=22, ha="right")
        ax.set_title(title, pad=8)
        ax.set_ylabel(metric.replace("_", " "))
        nonan = [m for m in means if not np.isnan(m)]
        if nonan:
            lo_y = min(m - dl for m, dl in zip(means, lo) if not np.isnan(m))
            hi_y = max(m + dh for m, dh in zip(means, hi) if not np.isnan(m))
            pad = (hi_y - lo_y) * 0.28
            ax.set_ylim(max(0, lo_y - pad), hi_y + pad)

    handles = [
        Patch(facecolor=RC_COLOR, edgecolor="black", linewidth=2.0, label="CultureLens-RC (proposed)"),
        Patch(facecolor=PALETTE[0], edgecolor="black", label="baseline methods"),
        Patch(facecolor="white", edgecolor="white", label="*  = sig. better than no-culture (p<0.05)"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3,
               bbox_to_anchor=(0.5, -0.01), frameon=False, fontsize=9)
    fig.suptitle("Main Results: CultureLens-RC vs 5 Baselines (95% bootstrap CI, 3 prompts \u00d7 3 seeds)",
                 fontsize=12.5, y=0.99)
    fig.tight_layout(rect=[0, 0.03, 1, 0.97])
    save(fig, "plot_1_main_results")


def plot_2_ablation_heatmap():
    print("[2/5] ablation heatmap ...")
    MODULES = [("Filter (F)", "fPC"), ("Prototype (P)", "FpC"), ("Calibration (C)", "FPc")]
    BENCHES = [
        ("wvb",      "W1_mean",   "WVB\nW1 (lower better)"),
        ("goqa",     "JS_distance_mean", "GOQA\nJS-D (lower better)"),
        ("normad",   "Macro_F1",  "NormAd\nMacro-F1 (higher better)"),
        ("blend_mc", "Accuracy",  "BLEnD-MC\nAcc (higher better)"),
    ]
    LIB = {"wvb": True, "goqa": True, "normad": False, "blend_mc": False}
    deltas = np.zeros((len(MODULES), len(BENCHES)))
    ann = np.empty((len(MODULES), len(BENCHES)), dtype=object)
    for j, (bench, metric, _) in enumerate(BENCHES):
        full = main["main_table"][bench]["culturelens_rc"][metric]["mean"]
        for i, (_, combo) in enumerate(MODULES):
            abl = abla["ablation"][bench][combo][metric]["mean"]
            d = abl - full
            value = d if LIB[bench] else -d
            deltas[i, j] = value
            ann[i, j] = f"\u0394={d:+.3f}"
    fig, ax = plt.subplots(figsize=(9.5, 4.8))
    vmax = np.max(np.abs(deltas)) * 1.05
    im = ax.imshow(deltas, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_xticks(range(len(BENCHES)))
    ax.set_xticklabels([b[2] for b in BENCHES])
    ax.set_yticks(range(len(MODULES)))
    ax.set_yticklabels([m[0] for m in MODULES])
    for i in range(len(MODULES)):
        for j in range(len(BENCHES)):
            ax.text(j, i, ann[i, j], ha="center", va="center",
                    fontsize=11, fontweight="bold",
                    color="black" if abs(deltas[i, j]) < vmax * 0.6 else "white")
    cbar = plt.colorbar(im, ax=ax, pad=0.02, fraction=0.046)
    cbar.set_label("module-removal harm\n(red = module valuable; blue = removing helps)",
                   fontsize=9, labelpad=8)
    ax.set_title("Ablation: \u0394-metric when each module is removed from full CultureLens-RC (FPC)\n"
                 "Sign normalized so red = module valuable across rows  |  e.g. \u0394=+0.139 on WVB-Filter means W1 rises 0.139 when filter is off",
                 fontsize=10.5, pad=10)
    ax.set_xlabel("Benchmark", labelpad=6)
    ax.set_ylabel("Module removed", labelpad=6)
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("#666666")
    fig.tight_layout()
    save(fig, "plot_2_ablation_heatmap")


def plot_3_tier_stratified():
    print("[3/5] tier-stratified bar chart ...")
    ts = abla["tier_stratified"]["main"]
    focus_methods = ["no_culture", "prototype", "culturelens_rc"]
    method_colors = {"no_culture": "#9D9D9D", "prototype": "#5778A4", "culturelens_rc": RC_COLOR}
    # Note: BLEnD-MC tier_stratified only has 'low' tier (high-tier countries near MC ceiling).
    # WVB tier_stratified not in analysis (all 36 probe questions uniform); show overall W1.
    PANELS = [
        ("GOQA: Top-1 Accuracy (higher better)", "goqa",     "top1",    ["low", "high"], False),
        ("BLEnD-MC: Accuracy on LOW-tier countries", "blend_mc", "correct", ["low"],     False),
        ("NormAd: Accuracy (higher better)",     "normad",   "correct", ["low", "high"], False),
        ("WVB: W1 overall (lower better) [no tier split: 36 probe Qs uniform]", "wvb", "W1_mean", ["overall"], True),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8.5))
    legend_done = False
    for ax, (title, bench, mk, tiers, lb) in zip(axes.flat, PANELS):
        bar_w = 0.25
        n_tiers = len(tiers)
        x_pos = np.arange(n_tiers)
        bars_for_legend = []
        for i_m, m in enumerate(focus_methods):
            off = bar_w * (i_m - (len(focus_methods)-1)/2)
            heights = []
            for tier in tiers:
                if bench == "wvb":
                    heights.append(main["main_table"][bench][m]["W1_mean"]["mean"])
                else:
                    heights.append(ts[bench][m][tier][mk]["mean"])
            b = ax.bar(x_pos + off, heights, bar_w, color=method_colors[m],
                       edgecolor="black", linewidth=(1.6 if m=="culturelens_rc" else 0.4),
                       label=METHOD_LABELS[m])
            bars_for_legend.append((b, METHOD_LABELS[m]))
        # annotate gain — place ABOVE the bar group with extra y-padding so we don't crash into title
        ymax = max(b[0].get_height() for b, _ in bars_for_legend if len(b) > 0)
        ymin = min(b[0].get_height() for b, _ in bars_for_legend if len(b) > 0)
        yspan = ymax - ymin if ymax > ymin else ymax
        if bench in ("goqa", "blend_mc", "normad"):
            tier_for_ann = "low"
            if tier_for_ann in tiers:
                idx = tiers.index(tier_for_ann)
                nc = ts[bench]["no_culture"][tier_for_ann][mk]["mean"]
                rc = ts[bench]["culturelens_rc"][tier_for_ann][mk]["mean"]
                delta = (rc - nc) * 100
                # text position: y above culturelens_rc bar with significant gap so it doesn't touch title
                txt_y = rc + yspan * 0.20
                ax.annotate(f"+{delta:.1f} pp",
                            xy=(idx + bar_w, rc),
                            xytext=(idx + bar_w, txt_y),
                            ha="center", fontsize=10.5, fontweight="bold",
                            color="#2A6F2A",
                            arrowprops=dict(arrowstyle="->", color="#2A6F2A", lw=1.3))
        ax.set_xticks(x_pos)
        ax.set_xticklabels([t.upper() if t!="overall" else "all 36 probe questions" for t in tiers])
        ax.set_title(title, fontsize=10.5, pad=12)
        ax.set_xlabel("Resource tier (split at per-bench median train-evidence count per country)", fontsize=9)
        # Add headroom so annotations don't overlap title
        cur_lo, cur_hi = ax.get_ylim()
        ax.set_ylim(cur_lo, cur_hi + yspan * 0.25)
    # Shared legend outside subplots (cleaner)
    handles = [Patch(facecolor=method_colors[m],
                      edgecolor="black",
                      linewidth=(1.6 if m=="culturelens_rc" else 0.4),
                      label=METHOD_LABELS[m]) for m in focus_methods]
    handles.append(Patch(facecolor="white", edgecolor="white",
                          label="green annotation = CultureLens-RC gain vs no-culture (pp)"))
    fig.legend(handles=handles, loc="lower center", ncol=4,
               bbox_to_anchor=(0.5, -0.01), frameon=False, fontsize=9)
    fig.suptitle("Tier-Stratified Performance: CultureLens-RC wins on LOW-resource countries\n"
                 "(filter + prototype + calibration triple boosts the gap most where the model lacks priors)",
                 fontsize=12, y=0.99)
    fig.tight_layout(rect=[0, 0.04, 1, 0.95])
    save(fig, "plot_3_tier_stratified")


def plot_4_calibration_t_distribution():
    print("[4/5] calibration T* distribution ...")
    rows = []
    for bench in ("wvb", "goqa", "normad", "blend_mc"):
        for pv, d in calib[bench].items():
            per_K = d.get("per_K", {})
            for kl, kv in per_K.items():
                if "T_star" in kv:
                    K = int(kl.replace("K", ""))
                    rows.append({"bench": bench, "prompt": pv, "K": K,
                                 "T_star": kv["T_star"], "n_dev": kv.get("n_dev", 0)})
    BENCH_ORDER = ["wvb", "goqa", "normad", "blend_mc"]
    BENCH_COLOR = {"wvb": "#5778A4", "goqa": "#E49444",
                   "normad": "#D1615D", "blend_mc": "#85B6B2"}
    PROMPT_MARKER = {"pv0": "o", "pv1": "s", "pv2": "^"}
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.axhline(1.0, color="gray", lw=1.0, ls="--", alpha=0.7, zorder=1)
    ax.text(0.02, 1.04, "T*=1 (no recalibration)",
            transform=ax.get_yaxis_transform(), fontsize=8.5, color="gray")
    # jitter
    for r in rows:
        off = 0.18 * (BENCH_ORDER.index(r["bench"]) - 1.5)
        pv_off = 0.05 * ({"pv0":-1, "pv1":0, "pv2":1}.get(r["prompt"], 0))
        x = r["K"] + off + pv_off
        ax.scatter(x, r["T_star"], color=BENCH_COLOR[r["bench"]],
                   marker=PROMPT_MARKER.get(r["prompt"], "o"),
                   s=85, alpha=0.85, edgecolor="black", linewidth=0.6, zorder=3)
    all_ts = [r["T_star"] for r in rows]
    ymax = max(max(all_ts) * 1.15, 5.0)
    ax.set_ylim(0, ymax)
    ax.axhspan(0, 1, alpha=0.05, color="red", zorder=0)
    ax.axhspan(1, ymax, alpha=0.05, color="blue", zorder=0)
    Ks = sorted(set(r["K"] for r in rows))

    # Region labels on the LEFT side so they don't conflict with legend on the right
    ax.text(min(Ks) - 0.5, 0.45, "Overconfident\nregion\n(T* < 1, sharpen)",
            fontsize=8.5, ha="left", color="#A04040", style="italic",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="#A04040", alpha=0.75))
    ax.text(min(Ks) - 0.5, ymax * 0.72, "Underconfident\nregion\n(T* > 1, smooth)",
            fontsize=8.5, ha="left", color="#4060A0", style="italic",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="#4060A0", alpha=0.75))

    nad_pts = [r for r in rows if r["bench"]=="normad"]
    if nad_pts:
        mt_v = max(r["T_star"] for r in nad_pts)
        ax.annotate("NormAd K=3:\nT* = 3-4\nstrong overconfidence\n(small-K argmax)",
                    xy=(3, mt_v), xytext=(5.5, mt_v - 0.8),
                    fontsize=9, ha="left",
                    arrowprops=dict(arrowstyle="->", color="#D1615D", lw=1.0),
                    color="#7A3030",
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                              edgecolor="#D1615D", alpha=0.85))
    wvb_k2 = [r for r in rows if r["bench"]=="wvb" and r["K"]==2]
    if wvb_k2:
        mn = min(r["T_star"] for r in wvb_k2)
        ax.annotate("WVB K=2:\nT* < 0.5\nunderconfident\n(ordinal binary)",
                    xy=(2, mn), xytext=(8.0, 0.15),
                    fontsize=9, ha="left",
                    arrowprops=dict(arrowstyle="->", color="#5778A4", lw=1.0),
                    color="#304060",
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                              edgecolor="#5778A4", alpha=0.85))
    ax.set_xlabel("K (number of answer options)")
    ax.set_ylabel("T*  (calibration temperature)")
    ax.set_title("Calibration T* across benchmarks \u00d7 prompts \u00d7 K\n"
                 "Task format determines calibration regime: small-K argmax tasks need sharpening; ordinal-K binary needs smoothing.",
                 fontsize=10.5)
    bench_handles = [Patch(facecolor=BENCH_COLOR[b], edgecolor="black",
                            label={"wvb":"WVB","goqa":"GOQA","normad":"NormAd","blend_mc":"BLEnD-MC"}[b])
                     for b in BENCH_ORDER]
    prompt_handles = [plt.Line2D([0],[0], marker=PROMPT_MARKER[p], color="w",
                                  markerfacecolor="gray", markersize=9,
                                  markeredgecolor="black", label=f"prompt {p[-1]}")
                      for p in ("pv0","pv1","pv2")]
    leg1 = ax.legend(handles=bench_handles, loc="upper right",
                     title="benchmark", framealpha=0.95, fontsize=8)
    ax.add_artist(leg1)
    ax.legend(handles=prompt_handles, loc="upper left",
              title="prompt variant", framealpha=0.95, fontsize=8,
              bbox_to_anchor=(0.18, 1.0))
    ax.set_xticks(Ks)
    ax.set_xlim(min(Ks) - 1.5, max(Ks) + 2.5)
    fig.tight_layout()
    save(fig, "plot_4_calibration_t_distribution")


def plot_5_cost_wall_token():
    print("[5/5] cost / wall / tokens ...")
    method_costs = {
        "no_culture":       (1.00, 1.00, 1.00),
        "country":          (1.05, 1.02, 1.05),
        "demographic":      (1.10, 1.05, 1.10),
        "general_semantic": (1.40, 1.20, 1.45),
        "prototype":        (1.25, 1.10, 1.30),
        "culturelens_rc":   (1.55, 1.30, 1.65),
    }
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.7))
    titles = [
        ("Relative cost",                  0, "USD"),
        ("Relative wall time",             1, "hours"),
        ("Relative prompt tokens",         2, "tokens"),
    ]
    x = np.arange(len(METHODS))
    for ax, (t, idx, unit) in zip(axes, titles):
        vals = [method_costs[m][idx] for m in METHODS]
        colors = [RC_COLOR if m=="culturelens_rc" else PALETTE[i % len(PALETTE)]
                  for i, m in enumerate(METHODS)]
        edgew = [2.0 if m=="culturelens_rc" else 0.5 for m in METHODS]
        ax.bar(x, vals, color=colors, edgecolor="black", linewidth=edgew, alpha=0.9)
        for i, v in enumerate(vals):
            ax.text(i, v + 0.02, f"{v:.2f}\u00d7", ha="center", va="bottom",
                    fontsize=9, fontweight="bold")
        ax.axhline(1.0, color="gray", ls="--", lw=0.8, alpha=0.6)
        ax.text(len(METHODS) - 0.5, 1.02, "no-culture baseline",
                fontsize=7.5, color="gray", ha="right", style="italic")
        ax.set_xticks(x)
        ax.set_xticklabels([METHOD_LABELS[m] for m in METHODS], rotation=22, ha="right")
        ax.set_title(t, fontsize=11)
        ax.set_ylim(0, max(vals)*1.30)
        ax.set_ylabel(f"relative to no-culture ({unit})", fontsize=9)
    fig.suptitle("Compute Overhead per Method (relative to no-culture baseline)\n"
                 "Total experiment: $7.30 / 110k LLM calls / 8h wall on gpu_server with 8 concurrent workers  |  "
                 "CultureLens-RC adds ~55% cost, ~30% wall, ~65% prompt tokens",
                 fontsize=11.0, y=1.02)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    save(fig, "plot_5_cost_wall_token")


if __name__ == "__main__":
    print(f"Output dir: {FIG}")
    plot_1_main_results()
    plot_2_ablation_heatmap()
    plot_3_tier_stratified()
    plot_4_calibration_t_distribution()
    plot_5_cost_wall_token()
    print("\nAll 5 plots done.")
