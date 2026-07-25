r"""Temperature scaling used by the calibration ablation (README §12).

For $T > 0$, calibration applies
$p_{cal}(c) \propto p_{raw}(c)^{1/T}$. Each benchmark and prompt variant
receives its own development-set estimate because their optimal temperatures
can differ. The transformation preserves the argmax except at ties.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np

logger = logging.getLogger(__name__)


def apply_temperature(probs: np.ndarray, T: float, *, eps: float = 1e-12) -> np.ndarray:
    """Apply temperature scaling to an array whose final axis is class."""
    if T <= 0:
        raise ValueError(f"T must be > 0, got {T}")
    p = np.asarray(probs, dtype=np.float64)
    if T == 1.0:
        return p / (p.sum(axis=-1, keepdims=True) + eps)
    # Work in log space to avoid underflow for small probabilities.
    log_p = np.log(np.clip(p, eps, None))
    scaled_logits = log_p / T
    m = scaled_logits.max(axis=-1, keepdims=True)
    exp = np.exp(scaled_logits - m)
    return exp / (exp.sum(axis=-1, keepdims=True) + eps)


def negative_log_likelihood_from_probs(
    probs: np.ndarray, y_true: Sequence[int], *, eps: float = 1e-12
) -> float:
    """Compute mean negative log likelihood for an ``(N, K)`` array."""
    p = np.asarray(probs, dtype=np.float64)
    if p.ndim != 2:
        raise ValueError("probs must be (N, K)")
    y = np.asarray(list(y_true), dtype=int)
    if p.shape[0] != y.shape[0]:
        raise ValueError("probs and y_true must contain the same number of rows")
    chosen = p[np.arange(p.shape[0]), y]
    chosen = np.clip(chosen, eps, 1.0)
    return float(-np.mean(np.log(chosen)))


@dataclass
class CalibrationResult:
    T_star: float
    nll_before: float
    nll_after: float
    bench: str | None = None
    prompt_variant: int | None = None
    n_dev_samples: int = 0
    search_grid: list[float] = field(default_factory=list)
    search_nll: list[float] = field(default_factory=list)


def fit_temperature(
    probs_dev: np.ndarray,
    y_dev: Sequence[int],
    *,
    grid: Sequence[float] | None = None,
    bench: str | None = None,
    prompt_variant: int | None = None,
    refine: bool = True,
) -> CalibrationResult:
    """Fit $T^*$ on the development set with coarse and local grid searches."""
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


def fit_per_benchmark(
    dev_data: dict[str, tuple[np.ndarray, Sequence[int]]],
) -> dict[str, CalibrationResult]:
    """Fit an independent calibration for each benchmark."""
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
    """Apply benchmark-specific calibrations, passing unknown benchmarks through."""
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
    "CalibrationResult",
    "apply_per_benchmark",
    "apply_temperature",
    "fit_per_benchmark",
    "fit_temperature",
    "negative_log_likelihood_from_probs",
]
