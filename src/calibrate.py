"""温度校准 T-scaling (readme §12, H0.method_calibration).

T-scaling 公式:
    p_calibrated(c) ∝ p_raw(c)^(1/T)
其中 T > 0; T=1 等同 raw, T>1 平滑分布, T<1 锐化.

设计:
- 单参数 T 通过最小化 dev split NLL 拟合 (闭式不可解, 一维线搜或 scipy.minimize)
- 每 benchmark 独立选 T (theorist insight: T* 在 4 基准上漂移大)
- prompt variant 也分别拟合 T* (calibration 对 prompt 措辞敏感)
- 不改 argmax, 因此 Accuracy / EM 不变 (除非 ties)

EXP_DESIGN 阶段: 接口 + dummy fit (假数据测试). RUNNING 阶段填实际数据流.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 基本 ops
# ---------------------------------------------------------------------------

def apply_temperature(probs: np.ndarray, T: float, *, eps: float = 1e-12) -> np.ndarray:
    """T-scaling 应用. probs shape (..., K) → 同 shape.

    支持 1-D (单样本) 和 2-D (batch). 不改 argmax.
    """
    if T <= 0:
        raise ValueError(f"T must be > 0, got {T}")
    p = np.asarray(probs, dtype=np.float64)
    if T == 1.0:
        return p / (p.sum(axis=-1, keepdims=True) + eps)
    # p^(1/T) 然后重新归一. p 是概率, log 安全负无穷处理
    log_p = np.log(np.clip(p, eps, None))
    scaled_logits = log_p / T
    # softmax-stable
    m = scaled_logits.max(axis=-1, keepdims=True)
    exp = np.exp(scaled_logits - m)
    return exp / (exp.sum(axis=-1, keepdims=True) + eps)


def negative_log_likelihood_from_probs(
    probs: np.ndarray, y_true: Sequence[int], *, eps: float = 1e-12
) -> float:
    """probs (N, K), y_true (N,) → mean NLL."""
    p = np.asarray(probs, dtype=np.float64)
    if p.ndim != 2:
        raise ValueError("probs must be (N, K)")
    y = np.asarray(list(y_true), dtype=int)
    if p.shape[0] != y.shape[0]:
        raise ValueError("probs / y_true 长度不一致")
    chosen = p[np.arange(p.shape[0]), y]
    chosen = np.clip(chosen, eps, 1.0)
    return float(-np.mean(np.log(chosen)))


# ---------------------------------------------------------------------------
# Fit T on dev set
# ---------------------------------------------------------------------------

@dataclass
class CalibrationResult:
    T_star: float
    nll_before: float
    nll_after: float
    bench: Optional[str] = None
    prompt_variant: Optional[int] = None
    n_dev_samples: int = 0
    search_grid: list[float] = field(default_factory=list)
    search_nll: list[float] = field(default_factory=list)


def fit_temperature(
    probs_dev: np.ndarray,
    y_dev: Sequence[int],
    *,
    grid: Optional[Sequence[float]] = None,
    bench: Optional[str] = None,
    prompt_variant: Optional[int] = None,
    refine: bool = True,
) -> CalibrationResult:
    """在 dev set 上线搜 T*. 默认 grid 在 [0.1, 5.0].

    流程:
      1. coarse grid (默认 25 点 log 间隔)
      2. refine: 在最优点附近 ±25% 做 fine search (默认 50 点)
    NLL 是凸函数 (about T 在 log space), 但保险起见两步搜.
    """
    if probs_dev.ndim != 2:
        raise ValueError("probs_dev must be (N, K)")
    if grid is None:
        grid = np.logspace(np.log10(0.1), np.log10(5.0), 25).tolist()

    grid = list(grid)
    nlls = []
    for T in grid:
        p_t = apply_temperature(probs_dev, T)
        nlls.append(negative_log_likelihood_from_probs(p_t, y_dev))
    best_i = int(np.argmin(nlls))
    T_star = float(grid[best_i])
    nll_star = float(nlls[best_i])

    nll_before = float(negative_log_likelihood_from_probs(probs_dev, y_dev))

    full_grid = list(grid)
    full_nll = list(nlls)

    if refine:
        lo = T_star * 0.75
        hi = T_star * 1.25
        fine = np.linspace(lo, hi, 50).tolist()
        fine_nlls = []
        for T in fine:
            p_t = apply_temperature(probs_dev, T)
            fine_nlls.append(negative_log_likelihood_from_probs(p_t, y_dev))
        best_j = int(np.argmin(fine_nlls))
        if fine_nlls[best_j] < nll_star:
            T_star = float(fine[best_j])
            nll_star = float(fine_nlls[best_j])
        full_grid.extend(fine)
        full_nll.extend(fine_nlls)

    return CalibrationResult(
        T_star=T_star,
        nll_before=nll_before,
        nll_after=nll_star,
        bench=bench,
        prompt_variant=prompt_variant,
        n_dev_samples=int(probs_dev.shape[0]),
        search_grid=full_grid,
        search_nll=full_nll,
    )


# ---------------------------------------------------------------------------
# Per-benchmark / per-prompt 独立 T (Theorist insight §3 强调)
# ---------------------------------------------------------------------------

def fit_per_benchmark(
    dev_data: dict[str, tuple[np.ndarray, Sequence[int]]],
) -> dict[str, CalibrationResult]:
    """dev_data: {bench: (probs_dev, y_dev)} → {bench: CalibrationResult}."""
    out: dict[str, CalibrationResult] = {}
    for bench, (probs, y) in dev_data.items():
        out[bench] = fit_temperature(probs, y, bench=bench)
        logger.info("bench %s T*=%.3f, NLL %.4f → %.4f (n=%d)",
                    bench, out[bench].T_star, out[bench].nll_before,
                    out[bench].nll_after, out[bench].n_dev_samples)
    return out


def apply_per_benchmark(
    probs_test: dict[str, np.ndarray],
    calibrations: dict[str, CalibrationResult],
) -> dict[str, np.ndarray]:
    """{bench: probs_test (N,K)} + calibrations → {bench: probs_calibrated (N,K)}."""
    out: dict[str, np.ndarray] = {}
    for bench, p in probs_test.items():
        cal = calibrations.get(bench)
        if cal is None:
            logger.warning("no calibration for %s; pass through", bench)
            out[bench] = p
            continue
        out[bench] = apply_temperature(p, cal.T_star)
    return out


__all__ = [
    "apply_temperature",
    "negative_log_likelihood_from_probs",
    "CalibrationResult",
    "fit_temperature",
    "fit_per_benchmark",
    "apply_per_benchmark",
]
