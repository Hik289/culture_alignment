"""多 seed × prompt → mean ± CI 表 (EXP_DESIGN 阶段骨架).

输入: 一组 run 产物 (config + per-item metrics in derived parquet 或 dict).
输出:
- per-method × per-benchmark × per-metric 的 mean ± 95% bootstrap CI
- paired bootstrap test (CultureLens-RC vs baseline) 显著性
- headroom-normalized effect size (Theorist insight §3): Δ / (baseline - oracle_floor)
- 长表 (long format) + 透视表 (wide) 输出

RUNNING 阶段:
- 接 dump_artifacts 写出的 {config_id}.json 汇总
- 接 derived parquet (per-item metric) 做 paired bootstrap
- 出 paper-ready 主表 + ablation 表
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional, Sequence

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data contracts
# ---------------------------------------------------------------------------

@dataclass
class RunSummary:
    """从 dump_artifacts 写出的 {config_id}.json 加载."""
    method: str
    benchmark: str
    prompt_variant: int
    seed: int
    use_calibration: bool
    use_filter: bool
    use_prototype: bool
    n_items: int
    metrics: dict[str, float]  # {"W1_mean": 0.31, "JS_distance_mean": 0.22, ...}
    config_id: str
    extras: dict = field(default_factory=dict)


def load_run_summaries(paths: Iterable[str | Path]) -> list[RunSummary]:
    """加载多个 {config_id}.json."""
    out: list[RunSummary] = []
    for p in paths:
        p = Path(p)
        if not p.exists():
            logger.warning("skip missing: %s", p)
            continue
        data = json.loads(p.read_text())
        cfg = data.get("config", {})
        out.append(RunSummary(
            method=cfg.get("method", ""),
            benchmark=cfg.get("benchmark", ""),
            prompt_variant=int(cfg.get("prompt_variant", 0)),
            seed=int(cfg.get("seed", 0)),
            use_calibration=bool(cfg.get("use_calibration", True)),
            use_filter=bool(cfg.get("use_filter", True)),
            use_prototype=bool(cfg.get("use_prototype", True)),
            n_items=int(data.get("n_items", 0)),
            metrics=dict(data.get("metrics", {})),
            config_id=str(cfg.get("config_id", p.stem)),
            extras={k: v for k, v in data.items() if k not in {"config", "metrics", "n_items"}},
        ))
    return out


# ---------------------------------------------------------------------------
# Bootstrap CI (per-metric, over seeds × prompts)
# ---------------------------------------------------------------------------

def bootstrap_ci(
    values: Sequence[float],
    *,
    n_boot: int = 1000,
    alpha: float = 0.05,
    seed: int = 42,
) -> tuple[float, float, float]:
    """返回 (mean, lo, hi) of bootstrap distribution."""
    v = np.asarray(values, dtype=np.float64)
    v = v[np.isfinite(v)]
    if len(v) == 0:
        return (float("nan"), float("nan"), float("nan"))
    if len(v) == 1:
        return (float(v[0]), float(v[0]), float(v[0]))
    rng = np.random.default_rng(seed)
    means = rng.choice(v, size=(n_boot, len(v)), replace=True).mean(axis=1)
    lo = float(np.quantile(means, alpha / 2))
    hi = float(np.quantile(means, 1 - alpha / 2))
    return (float(v.mean()), lo, hi)


def paired_bootstrap_diff(
    a_values: Sequence[float],
    b_values: Sequence[float],
    *,
    n_boot: int = 1000,
    alpha: float = 0.05,
    seed: int = 42,
) -> dict[str, float]:
    """配对样本 (a, b 同长, 对应同 (item, seed, prompt)) 的 bootstrap CI of mean(a - b).

    返回 {mean_diff, lo, hi, p_value (双侧, 估算)}.
    """
    a = np.asarray(a_values, dtype=np.float64)
    b = np.asarray(b_values, dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError("a/b 长度不一致, paired bootstrap 要求成对")
    d = a - b
    d = d[np.isfinite(d)]
    if len(d) == 0:
        return {"mean_diff": float("nan"), "lo": float("nan"),
                "hi": float("nan"), "p_value_two_sided": float("nan")}
    rng = np.random.default_rng(seed)
    boot = rng.choice(d, size=(n_boot, len(d)), replace=True).mean(axis=1)
    lo = float(np.quantile(boot, alpha / 2))
    hi = float(np.quantile(boot, 1 - alpha / 2))
    # 双侧 p: 取 0 的尾部比例 × 2
    p = 2.0 * min(float(np.mean(boot > 0)), float(np.mean(boot < 0)))
    return {
        "mean_diff": float(d.mean()),
        "lo": lo,
        "hi": hi,
        "p_value_two_sided": float(p),
    }


# ---------------------------------------------------------------------------
# Aggregation 主表
# ---------------------------------------------------------------------------

def aggregate_runs(
    summaries: list[RunSummary],
    *,
    metric_keys: Optional[Sequence[str]] = None,
    n_boot: int = 1000,
) -> list[dict]:
    """按 (method, benchmark, metric) 聚合 over (prompt_variant, seed).

    返回长表 list[dict]:
      {method, benchmark, metric, n_runs, mean, ci_lo, ci_hi, raw_values}
    """
    groups: dict[tuple[str, str, str], list[float]] = {}
    for s in summaries:
        for k, v in s.metrics.items():
            if metric_keys and k not in metric_keys:
                continue
            if not isinstance(v, (int, float)):
                continue
            groups.setdefault((s.method, s.benchmark, k), []).append(float(v))
    out: list[dict] = []
    for (m, b, k), vals in groups.items():
        mean, lo, hi = bootstrap_ci(vals, n_boot=n_boot)
        out.append({
            "method": m, "benchmark": b, "metric": k,
            "n_runs": len(vals),
            "mean": mean, "ci_lo": lo, "ci_hi": hi,
            "raw_values": vals,
        })
    return out


# ---------------------------------------------------------------------------
# Headroom-normalized effect (Theorist §3)
# ---------------------------------------------------------------------------

# Oracle floor (lower=better → 0 是 oracle; higher=better → 1 是 oracle)
# 这里只列分布距离 metric 的 oracle floor (lower=better)
ORACLE_FLOOR = {
    "W1_mean": 0.0,
    "JS_distance_mean": 0.0,
    "TV_distance_mean": 0.0,
    "Accuracy": 1.0,
    "Macro_F1": 1.0,
    "EM": 1.0,
    "Token_F1": 1.0,
    "Top1_Acc": 1.0,
}

LOWER_IS_BETTER = {
    "W1_mean", "JS_distance_mean", "TV_distance_mean",
}


def headroom_normalized(
    method_value: float, baseline_value: float, metric_key: str,
) -> Optional[float]:
    """Δ / (baseline - oracle).

    对 lower-is-better metric: Δ = baseline - method (越正越好)
    对 higher-is-better:        Δ = method - baseline
    headroom = |baseline - oracle|
    返回 None 当 headroom 接近 0.
    """
    oracle = ORACLE_FLOOR.get(metric_key)
    if oracle is None:
        return None
    headroom = abs(baseline_value - oracle)
    if headroom < 1e-9:
        return None
    if metric_key in LOWER_IS_BETTER:
        delta = baseline_value - method_value
    else:
        delta = method_value - baseline_value
    return float(delta / headroom)


def compare_to_baseline(
    summaries: list[RunSummary],
    *,
    method_name: str = "culturelens_rc",
    baseline_name: str = "no_culture",
    n_boot: int = 1000,
) -> list[dict]:
    """对每 (benchmark, metric), 用 (paired by prompt_variant × seed) bootstrap 比较
    method vs baseline 的差值 + headroom-normalized.

    要求两 method 在同一 (benchmark, prompt, seed) 上都有 run; 缺则跳过.
    """
    # 索引: (benchmark, metric, prompt, seed) → value
    def _idx(name: str) -> dict:
        out = {}
        for s in summaries:
            if s.method != name:
                continue
            for k, v in s.metrics.items():
                if not isinstance(v, (int, float)):
                    continue
                out[(s.benchmark, k, s.prompt_variant, s.seed)] = float(v)
        return out

    a_idx = _idx(method_name)
    b_idx = _idx(baseline_name)
    pairs: dict[tuple[str, str], list[tuple[float, float]]] = {}
    for key, a_val in a_idx.items():
        b_val = b_idx.get(key)
        if b_val is None:
            continue
        bench, metric, _, _ = key
        pairs.setdefault((bench, metric), []).append((a_val, b_val))

    out: list[dict] = []
    for (bench, metric), arr in pairs.items():
        a_vals = [x[0] for x in arr]
        b_vals = [x[1] for x in arr]
        diff_stats = paired_bootstrap_diff(a_vals, b_vals, n_boot=n_boot)
        hn = headroom_normalized(
            float(np.mean(a_vals)), float(np.mean(b_vals)), metric,
        )
        out.append({
            "benchmark": bench,
            "metric": metric,
            "n_pairs": len(arr),
            "method_mean": float(np.mean(a_vals)),
            "baseline_mean": float(np.mean(b_vals)),
            "headroom_normalized_improvement": hn,
            **diff_stats,
        })
    return out


# ---------------------------------------------------------------------------
# Output as Markdown table
# ---------------------------------------------------------------------------

def main_table_markdown(rows: list[dict]) -> str:
    """主表 markdown."""
    if not rows:
        return "(no rows)"
    keys = ["method", "benchmark", "metric", "n_runs", "mean", "ci_lo", "ci_hi"]
    lines = ["| " + " | ".join(keys) + " |", "|" + "|".join(["---"] * len(keys)) + "|"]
    for r in rows:
        fmt = []
        for k in keys:
            v = r.get(k)
            if isinstance(v, float):
                fmt.append(f"{v:.4f}")
            else:
                fmt.append(str(v))
        lines.append("| " + " | ".join(fmt) + " |")
    return "\n".join(lines)


def compare_table_markdown(rows: list[dict]) -> str:
    if not rows:
        return "(no rows)"
    keys = [
        "benchmark", "metric", "n_pairs",
        "method_mean", "baseline_mean", "mean_diff",
        "lo", "hi", "p_value_two_sided",
        "headroom_normalized_improvement",
    ]
    lines = ["| " + " | ".join(keys) + " |", "|" + "|".join(["---"] * len(keys)) + "|"]
    for r in rows:
        fmt = []
        for k in keys:
            v = r.get(k)
            if v is None:
                fmt.append("—")
            elif isinstance(v, float):
                fmt.append(f"{v:.4f}")
            else:
                fmt.append(str(v))
        lines.append("| " + " | ".join(fmt) + " |")
    return "\n".join(lines)


__all__ = [
    "RunSummary",
    "load_run_summaries",
    "bootstrap_ci",
    "paired_bootstrap_diff",
    "aggregate_runs",
    "headroom_normalized",
    "compare_to_baseline",
    "main_table_markdown",
    "compare_table_markdown",
    "ORACLE_FLOOR",
    "LOWER_IS_BETTER",
]
