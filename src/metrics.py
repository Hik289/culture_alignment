"""评估指标实现 (CultureLens-RC 项目).

对应 readme §16:
- 分布预测: W1 (有序), JS-D, KL-D, TV-D, Top-1 Accuracy
- 分类: Accuracy, Macro-F1, NLL, Brier Score, ECE
- 短答案: EM, Token F1

所有指标接受 numpy 数组形式的概率分布 / 标签 / 文本, 返回 float.
设计原则:
- 数值稳定 (KL 用 epsilon 平滑)
- 输入校验 (shape / sum / non-negative)
- 解析解可对齐 (单元测试要求 ± 1e-6)
"""

from __future__ import annotations

import math
import re
import string
from collections import Counter
from collections.abc import Iterable, Sequence
from typing import Union

import numpy as np

ArrayLike = Union[Sequence[float], np.ndarray]

_EPS = 1e-12


# ---------------------------------------------------------------------------
# 辅助
# ---------------------------------------------------------------------------

def _as_prob(p: ArrayLike, *, axis_check: bool = True, eps: float = _EPS) -> np.ndarray:
    """转 numpy float, 校验非负 + sum≈1 (允许 ±1e-6 误差)."""
    arr = np.asarray(p, dtype=np.float64)
    if arr.ndim != 1:
        raise ValueError(f"概率分布必须是 1-D, got shape {arr.shape}")
    if np.any(arr < -1e-9):
        raise ValueError(f"概率分布出现负数: min={arr.min()}")
    arr = np.clip(arr, 0.0, None)
    s = arr.sum()
    if axis_check and not math.isclose(s, 1.0, abs_tol=1e-3):
        raise ValueError(f"概率分布未归一化, sum={s}")
    if s > 0:
        arr = arr / s
    return arr


def _smooth(p: np.ndarray, eps: float) -> np.ndarray:
    """epsilon 平滑后重新归一."""
    p = p + eps
    return p / p.sum()


# ---------------------------------------------------------------------------
# 16.1 分布预测指标
# ---------------------------------------------------------------------------

def wasserstein_1(p: ArrayLike, q: ArrayLike, support: ArrayLike | None = None) -> float:
    """有序选项的 W1 距离.

    1-D 情况下 W1(p,q) = sum_i |F_p(x_i) - F_q(x_i)| * (x_{i+1} - x_i).
    若 support 未给出, 默认整数支撑 {0,1,...,K-1}, 相邻间距 1.

    例: p=[0.5,0.5], q=[0.0,1.0], support={1,2} → W1 = 0.5.
    """
    p = _as_prob(p)
    q = _as_prob(q)
    if p.shape != q.shape:
        raise ValueError(f"p,q shape 不一致: {p.shape} vs {q.shape}")
    K = p.shape[0]
    if support is None:
        support = np.arange(K, dtype=np.float64)
    support = np.asarray(support, dtype=np.float64)
    if support.shape != (K,):
        raise ValueError("support 长度须等于分布长度")
    if not np.all(np.diff(support) > 0):
        raise ValueError("support 必须严格递增")

    cdf_p = np.cumsum(p)
    cdf_q = np.cumsum(q)
    # W1 = integral |F_p - F_q| dx
    # 离散支撑上, 每段宽度 = support[i+1] - support[i], 高度 = |cdf_p[i] - cdf_q[i]|
    widths = np.diff(support)  # 长度 K-1
    diffs = np.abs(cdf_p[:-1] - cdf_q[:-1])  # 累计差 K-1 个区段
    return float(np.sum(diffs * widths))


def js_divergence(p: ArrayLike, q: ArrayLike, *, base: float = 2.0, eps: float = _EPS) -> float:
    """Jensen-Shannon divergence (返回 散度, 单位由 base 决定; base=2 → [0,1]).

    若用户在 §16 中写"JS-D 越低越好"既包括距离也包括散度, 这里返回散度. 距离=sqrt(散度)."""
    p = _as_prob(p)
    q = _as_prob(q)
    p = _smooth(p, eps)
    q = _smooth(q, eps)
    m = 0.5 * (p + q)

    def _kl(a, b):
        return float(np.sum(a * (np.log(a) - np.log(b))))

    raw = 0.5 * _kl(p, m) + 0.5 * _kl(q, m)
    if base == 2.0:
        return raw / math.log(2.0)
    if base == math.e:
        return raw
    return raw / math.log(base)


def js_distance(p: ArrayLike, q: ArrayLike, *, base: float = 2.0, eps: float = _EPS) -> float:
    """JS 距离 = sqrt(JS 散度), base=2 时 ∈ [0,1]."""
    div = js_divergence(p, q, base=base, eps=eps)
    return float(math.sqrt(max(div, 0.0)))


def kl_divergence(p: ArrayLike, q: ArrayLike, *, base: float = math.e, eps: float = _EPS) -> float:
    """KL(p || q), epsilon 平滑."""
    p = _as_prob(p)
    q = _as_prob(q)
    p = _smooth(p, eps)
    q = _smooth(q, eps)
    raw = float(np.sum(p * (np.log(p) - np.log(q))))
    if base == math.e:
        return raw
    return raw / math.log(base)


def tv_distance(p: ArrayLike, q: ArrayLike) -> float:
    """Total Variation = 0.5 * sum |p_i - q_i|."""
    p = _as_prob(p)
    q = _as_prob(q)
    return float(0.5 * np.sum(np.abs(p - q)))


def top1_accuracy(p_pred: ArrayLike, p_true: ArrayLike) -> float:
    """两个分布最高概率选项是否一致. 返回 1.0 或 0.0."""
    p = _as_prob(p_pred)
    q = _as_prob(p_true)
    return float(int(np.argmax(p) == np.argmax(q)))


# ---------------------------------------------------------------------------
# 16.2 分类指标
# ---------------------------------------------------------------------------

def accuracy(y_true: Iterable[int], y_pred: Iterable[int]) -> float:
    y_true = np.asarray(list(y_true))
    y_pred = np.asarray(list(y_pred))
    if y_true.shape != y_pred.shape:
        raise ValueError("y_true / y_pred 长度不一致")
    if y_true.size == 0:
        return 0.0
    return float(np.mean(y_true == y_pred))


def macro_f1(y_true: Iterable[int], y_pred: Iterable[int], labels: Sequence[int] | None = None) -> float:
    y_true = np.asarray(list(y_true))
    y_pred = np.asarray(list(y_pred))
    if labels is None:
        labels = sorted(set(y_true.tolist()) | set(y_pred.tolist()))
    f1s = []
    for c in labels:
        tp = int(np.sum((y_pred == c) & (y_true == c)))
        fp = int(np.sum((y_pred == c) & (y_true != c)))
        fn = int(np.sum((y_pred != c) & (y_true == c)))
        if tp + fp == 0:
            precision = 0.0
        else:
            precision = tp / (tp + fp)
        if tp + fn == 0:
            recall = 0.0
        else:
            recall = tp / (tp + fn)
        if precision + recall == 0:
            f1s.append(0.0)
        else:
            f1s.append(2 * precision * recall / (precision + recall))
    return float(np.mean(f1s)) if f1s else 0.0


def negative_log_likelihood(probs: np.ndarray, y_true: Iterable[int], *, eps: float = _EPS) -> float:
    """probs shape (N, C), y_true shape (N,). 返回 mean NLL."""
    probs = np.asarray(probs, dtype=np.float64)
    if probs.ndim != 2:
        raise ValueError("probs 必须是 (N, C)")
    y = np.asarray(list(y_true), dtype=int)
    if probs.shape[0] != y.shape[0]:
        raise ValueError("probs 与 y_true 行数不一致")
    chosen = probs[np.arange(probs.shape[0]), y]
    chosen = np.clip(chosen, eps, 1.0)
    return float(-np.mean(np.log(chosen)))


def brier_score(probs: np.ndarray, y_true: Iterable[int]) -> float:
    """多类 Brier = mean_n sum_c (probs[n,c] - 1{y_n==c})^2."""
    probs = np.asarray(probs, dtype=np.float64)
    if probs.ndim != 2:
        raise ValueError("probs 必须是 (N, C)")
    y = np.asarray(list(y_true), dtype=int)
    N, C = probs.shape
    onehot = np.zeros_like(probs)
    onehot[np.arange(N), y] = 1.0
    return float(np.mean(np.sum((probs - onehot) ** 2, axis=1)))


def expected_calibration_error(
    probs: np.ndarray,
    y_true: Iterable[int],
    *,
    n_bins: int = 10,
) -> float:
    """ECE (top-label, equal-width bins).

    对每个样本取 max prob 作为 confidence, 取 argmax 作为预测.
    将 [0,1] 平均分 n_bins 段, 每段计算 |acc - conf|, 用样本占比加权.
    """
    probs = np.asarray(probs, dtype=np.float64)
    if probs.ndim != 2:
        raise ValueError("probs 必须是 (N, C)")
    y = np.asarray(list(y_true), dtype=int)
    N = probs.shape[0]
    if N == 0:
        return 0.0
    confidences = probs.max(axis=1)
    predictions = probs.argmax(axis=1)
    correct = (predictions == y).astype(np.float64)

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        # 最后一桶闭区间, 其他左闭右开
        if i == n_bins - 1:
            mask = (confidences >= lo) & (confidences <= hi)
        else:
            mask = (confidences >= lo) & (confidences < hi)
        n_in_bin = mask.sum()
        if n_in_bin == 0:
            continue
        bin_acc = float(correct[mask].mean())
        bin_conf = float(confidences[mask].mean())
        ece += (n_in_bin / N) * abs(bin_acc - bin_conf)
    return float(ece)


# ---------------------------------------------------------------------------
# 16.3 短答案指标
# ---------------------------------------------------------------------------

_ARTICLES_RE = re.compile(r"\b(a|an|the)\b", re.IGNORECASE)
_WS_RE = re.compile(r"\s+")
_PUNCT_TBL = str.maketrans("", "", string.punctuation)


def _normalize_answer(s: str) -> str:
    """SQuAD-style 归一化: 去冠词, 去标点, 小写, 折叠空白."""
    s = s.lower()
    s = s.translate(_PUNCT_TBL)
    s = _ARTICLES_RE.sub(" ", s)
    s = _WS_RE.sub(" ", s).strip()
    return s


def exact_match(pred: str, gold: str) -> float:
    return float(_normalize_answer(pred) == _normalize_answer(gold))


def token_f1(pred: str, gold: str) -> float:
    p_tokens = _normalize_answer(pred).split()
    g_tokens = _normalize_answer(gold).split()
    if len(p_tokens) == 0 and len(g_tokens) == 0:
        return 1.0
    if len(p_tokens) == 0 or len(g_tokens) == 0:
        return 0.0
    common = Counter(p_tokens) & Counter(g_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(p_tokens)
    recall = num_same / len(g_tokens)
    return float(2 * precision * recall / (precision + recall))


def exact_match_corpus(preds: Sequence[str], golds: Sequence[Sequence[str] | str]) -> float:
    """支持多 gold (列表), 取 max."""
    if len(preds) != len(golds):
        raise ValueError("preds/golds 长度不一致")
    if not preds:
        return 0.0
    scores = []
    for p, g in zip(preds, golds):
        if isinstance(g, str):
            scores.append(exact_match(p, g))
        else:
            scores.append(max((exact_match(p, gi) for gi in g), default=0.0))
    return float(np.mean(scores))


def token_f1_corpus(preds: Sequence[str], golds: Sequence[Sequence[str] | str]) -> float:
    if len(preds) != len(golds):
        raise ValueError("preds/golds 长度不一致")
    if not preds:
        return 0.0
    scores = []
    for p, g in zip(preds, golds):
        if isinstance(g, str):
            scores.append(token_f1(p, g))
        else:
            scores.append(max((token_f1(p, gi) for gi in g), default=0.0))
    return float(np.mean(scores))


__all__ = [
    "accuracy",
    "brier_score",
    "exact_match",
    "exact_match_corpus",
    "expected_calibration_error",
    "js_distance",
    "js_divergence",
    "kl_divergence",
    "macro_f1",
    "negative_log_likelihood",
    "token_f1",
    "token_f1_corpus",
    "top1_accuracy",
    "tv_distance",
    "wasserstein_1",
]
