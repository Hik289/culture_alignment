"""检索接口骨架 (STANDBY 阶段, 不做实质实验).

readme §10 分层检索 + §14.5 通用语义检索 的两条路径接口:

  general_semantic: 单一向量索引, 只按问题文本 top-k. 用作 baseline (§14.5).
  hierarchical    : 国家过滤 → 主题过滤 → 时间过滤 → 向量重排 (§10).

设计原则:
- 接口稳定, 后端可换 (FAISS / cuVS / 简单 numpy)
- 真实索引构建/查询留给 EXP_DESIGN 后实现
- 当前仅: 抽象基类 + 数据契约 + 简单 numpy 后端的占位 (供单元测试)
- 不下载/不调用 sentence-transformers (那是 RUNNING 阶段事)

下游消费者:
- src/prompts.render_survey_distribution(retrieved_evidence=...)
- src/prompts.render_norm_judgment(retrieved_evidence=...)
- src/prompts.render_daily_knowledge(retrieved_evidence=...)
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 数据契约
# ---------------------------------------------------------------------------

@dataclass
class EvidenceItem:
    """证据条目 (readme §9.2 统一证据格式).

    Fields (subset of §9.2; 全字段在 EXP_DESIGN 后补齐):
      id           : 全局唯一字符串
      text         : 证据文本 (英语 normalized)
      country      : 来源国家 / 地区 (大小写不规范, 用 ISO-3166 normalize at index time)
      topic        : 主题 tag (family / religion / politics / etiquette / food / ...)
      year         : 发布或数据采集年 (int, optional)
      source       : 来源标识 (wvs / oecd / blend_train / wikipedia / ...)
      lang         : 主语言 (ISO-639-1, 默认 'en')
      score        : 排序分 (检索时填, 索引内为 None)
      meta         : 其他自由字段 (split / response_count / etc.)
    """
    id: str
    text: str
    country: str | None = None
    topic: str | None = None
    year: int | None = None
    source: str | None = None
    lang: str = "en"
    score: float | None = None
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class RetrievalQuery:
    """检索查询.

    text         : 查询文本 (问题 / 场景 / 短答案问句)
    country      : 目标国家 / 地区, 用于 hierarchical 过滤
    topic        : 任务主题, 用于 hierarchical 过滤
    lang         : 语言偏好
    top_k        : 返回条数
    year_window  : 时间窗 [start, end], 用于 hierarchical 过滤
    exclude_split: 排除某些 split 标记 (避免 test 答案泄漏 → 见 leakage_check)
    """
    text: str
    country: str | None = None
    topic: str | None = None
    lang: str = "en"
    top_k: int = 8
    year_window: tuple[int, int] | None = None
    exclude_split: Sequence[str] = ()


# ---------------------------------------------------------------------------
# Embedder Protocol (后端可换)
# ---------------------------------------------------------------------------

class Embedder(Protocol):
    """文本嵌入 callable: encode(list[str]) -> np.ndarray of shape (N, D)."""
    def encode(self, texts: list[str]) -> np.ndarray: ...


@dataclass
class IdentityEmbedder:
    """占位 embedder: 用 hash 把文本映射为定长向量, 供单元测试 (不需要真实模型)."""
    dim: int = 32

    def encode(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, t in enumerate(texts):
            # 简单 hash → bytes → float, deterministic 用于 test
            h = abs(hash(t))
            rng = np.random.default_rng(h)
            v = rng.standard_normal(self.dim).astype(np.float32)
            v /= (np.linalg.norm(v) + 1e-9)
            out[i] = v
        return out


# ---------------------------------------------------------------------------
# 检索基类
# ---------------------------------------------------------------------------

class Retriever(ABC):
    """检索器抽象基类.

    生命周期:
        retriever = SomeRetriever(embedder)
        retriever.build(items)              # 一次性建索引
        results = retriever.search(query)   # 重复查询
    """

    def __init__(self, embedder: Embedder):
        self.embedder = embedder
        self._items: list[EvidenceItem] = []
        self._embeddings: np.ndarray | None = None
        self._built = False

    @abstractmethod
    def search(self, query: RetrievalQuery) -> list[EvidenceItem]:
        """返回 top_k 条按 score 降序排列的 evidence (含 score 字段)."""

    def build(self, items: Iterable[EvidenceItem]) -> None:
        self._items = list(items)
        if not self._items:
            self._embeddings = np.zeros((0, 0), dtype=np.float32)
            self._built = True
            return
        texts = [it.text for it in self._items]
        self._embeddings = self.embedder.encode(texts)
        self._built = True
        logger.info("built %s with %d items", type(self).__name__, len(self._items))

    @property
    def n_items(self) -> int:
        return len(self._items)

    def _check_built(self):
        if not self._built:
            raise RuntimeError("调用 search 前必须先 build()")


# ---------------------------------------------------------------------------
# §14.5 General semantic (单一索引, 仅按文本相似度)
# ---------------------------------------------------------------------------

class GeneralSemanticRetriever(Retriever):
    """单向量索引, 只按 query embedding 与 item embedding 的内积排序.

    无任何过滤 (国家 / 主题 / 时间), 用作 §14.5 baseline.
    """

    def search(self, query: RetrievalQuery) -> list[EvidenceItem]:
        self._check_built()
        if self.n_items == 0:
            return []
        q = self.embedder.encode([query.text])  # (1, D)
        if q.shape[1] != self._embeddings.shape[1]:
            raise ValueError(
                f"query dim {q.shape[1]} != index dim {self._embeddings.shape[1]}"
            )
        # cosine similarity (embeddings 已归一化时 = 内积)
        sims = (self._embeddings @ q.T).squeeze(-1)  # (N,)
        # 排除 split (e.g. test 自身)
        if query.exclude_split:
            mask = np.array([it.meta.get("split") not in query.exclude_split for it in self._items])
            sims = np.where(mask, sims, -np.inf)
        top_idx = np.argsort(-sims)[: query.top_k]
        out: list[EvidenceItem] = []
        for i in top_idx:
            if not np.isfinite(sims[i]):
                continue
            it = self._items[i]
            out.append(EvidenceItem(
                id=it.id, text=it.text, country=it.country, topic=it.topic,
                year=it.year, source=it.source, lang=it.lang,
                score=float(sims[i]), meta=dict(it.meta),
            ))
        return out


# ---------------------------------------------------------------------------
# §10 Hierarchical (country → topic → year → vector rerank)
# ---------------------------------------------------------------------------

class HierarchicalRetriever(Retriever):
    """分层检索:
        1. country 过滤 (若 query.country 给出, 仅保留同国家 + (optional) 同语系 / 同 cluster fallback)
        2. topic 过滤 (若 query.topic 给出)
        3. year_window 过滤
        4. exclude_split 过滤
        5. 剩余候选用向量内积 rerank, 返回 top_k

    EXP_DESIGN 后会:
      - 加 country fallback chain (target → region → continent → global)
      - 加 topic embedding-based proxy
      - 加 score 权重组合 (vector_sim * w1 + recency * w2 + ...)
    现在仅实现核心顺序, 留 hook.
    """

    # hook: 子类/EXP_DESIGN 后覆盖
    def score_combine(self, vec_sim: float, item: EvidenceItem, query: RetrievalQuery) -> float:
        """默认 = 纯向量相似度. EXP_DESIGN 后可加 recency / source 权重."""
        return float(vec_sim)

    def search(self, query: RetrievalQuery) -> list[EvidenceItem]:
        self._check_built()
        if self.n_items == 0:
            return []

        # 1-4: 过滤
        cand_idx: list[int] = []
        for i, it in enumerate(self._items):
            if query.country and (it.country or "").lower() != query.country.lower():
                continue
            if query.topic and (it.topic or "").lower() != query.topic.lower():
                continue
            if query.year_window and it.year is not None:
                lo, hi = query.year_window
                if not (lo <= it.year <= hi):
                    continue
            if query.exclude_split and it.meta.get("split") in query.exclude_split:
                continue
            cand_idx.append(i)

        if not cand_idx:
            return []

        # 5: 向量 rerank
        q = self.embedder.encode([query.text])
        cand_emb = self._embeddings[cand_idx]
        sims = (cand_emb @ q.T).squeeze(-1)  # (n_cand,)
        scored = [
            (i, self.score_combine(float(sims[k]), self._items[i], query))
            for k, i in enumerate(cand_idx)
        ]
        scored.sort(key=lambda x: -x[1])

        out: list[EvidenceItem] = []
        for i, sc in scored[: query.top_k]:
            it = self._items[i]
            out.append(EvidenceItem(
                id=it.id, text=it.text, country=it.country, topic=it.topic,
                year=it.year, source=it.source, lang=it.lang,
                score=sc, meta=dict(it.meta),
            ))
        return out


# ---------------------------------------------------------------------------
# 便利构造
# ---------------------------------------------------------------------------

def build_retriever(
    kind: str,
    embedder: Embedder | None = None,
    items: Iterable[EvidenceItem] | None = None,
) -> Retriever:
    """工厂. kind ∈ {"general", "hierarchical"}."""
    emb = embedder or IdentityEmbedder()
    if kind == "general":
        r = GeneralSemanticRetriever(emb)
    elif kind == "hierarchical":
        r = HierarchicalRetriever(emb)
    else:
        raise ValueError(f"未知 kind: {kind}; 可选 general / hierarchical")
    if items is not None:
        r.build(items)
    return r


__all__ = [
    "Embedder",
    "EvidenceItem",
    "GeneralSemanticRetriever",
    "HierarchicalRetriever",
    "IdentityEmbedder",
    "RetrievalQuery",
    "Retriever",
    "build_retriever",
]
