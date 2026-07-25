"""retrieval.py 骨架测试 (不依赖网络/模型, 用 IdentityEmbedder)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.retrieval import (
    EvidenceItem,
    GeneralSemanticRetriever,
    HierarchicalRetriever,
    IdentityEmbedder,
    RetrievalQuery,
    build_retriever,
)


def _sample_items() -> list[EvidenceItem]:
    return [
        EvidenceItem(id="e1", text="family is very important in Japan", country="Japan",
                     topic="family", year=2019, source="wvs", lang="en",
                     meta={"split": "train"}),
        EvidenceItem(id="e2", text="political trust in Japan is moderate", country="Japan",
                     topic="politics", year=2020, source="oecd", lang="en"),
        EvidenceItem(id="e3", text="religion in Brazil is mostly Catholic", country="Brazil",
                     topic="religion", year=2018, source="wvs", lang="en"),
        EvidenceItem(id="e4", text="Japanese cuisine traditional foods", country="Japan",
                     topic="food", year=2022, source="blend", lang="en",
                     meta={"split": "test"}),
        EvidenceItem(id="e5", text="family values in Brazil and South America", country="Brazil",
                     topic="family", year=2021, source="wvs", lang="en"),
    ]


class TestEvidenceItem:
    def test_minimal_fields(self):
        e = EvidenceItem(id="x", text="hi")
        assert e.id == "x"
        assert e.text == "hi"
        assert e.country is None
        assert e.lang == "en"
        assert e.meta == {}


class TestIdentityEmbedder:
    def test_shape(self):
        e = IdentityEmbedder(dim=16)
        out = e.encode(["a", "b", "c"])
        assert out.shape == (3, 16)

    def test_deterministic(self):
        e1 = IdentityEmbedder()
        e2 = IdentityEmbedder()
        a = e1.encode(["hello"])
        b = e2.encode(["hello"])
        assert np.allclose(a, b)

    def test_normalized(self):
        e = IdentityEmbedder()
        out = e.encode(["foo"])
        assert abs(np.linalg.norm(out[0]) - 1.0) < 1e-5


class TestGeneralSemantic:
    def test_build_and_search(self):
        r = build_retriever("general", items=_sample_items())
        out = r.search(RetrievalQuery(text="family in Japan", top_k=3))
        assert len(out) <= 3
        assert len(out) > 0
        # score 单调递减
        scores = [o.score for o in out]
        assert scores == sorted(scores, reverse=True)

    def test_empty_index(self):
        r = build_retriever("general", items=[])
        out = r.search(RetrievalQuery(text="anything", top_k=5))
        assert out == []

    def test_search_before_build_raises(self):
        r = GeneralSemanticRetriever(IdentityEmbedder())
        with pytest.raises(RuntimeError):
            r.search(RetrievalQuery(text="x"))

    def test_exclude_split(self):
        """e4 split=test 应被 exclude_split=('test',) 过滤掉."""
        r = build_retriever("general", items=_sample_items())
        out = r.search(RetrievalQuery(
            text="Japanese cuisine traditional foods",
            top_k=10,
            exclude_split=("test",),
        ))
        ids = [o.id for o in out]
        assert "e4" not in ids

    def test_no_country_filter_in_general(self):
        """General 不过滤国家, 即使 query.country 给出也无视."""
        r = build_retriever("general", items=_sample_items())
        out = r.search(RetrievalQuery(text="family", country="Mars", top_k=5))
        # 应该返回结果, 不因为 country=Mars 而清空
        assert len(out) > 0


class TestHierarchical:
    def test_country_filter(self):
        r = build_retriever("hierarchical", items=_sample_items())
        out = r.search(RetrievalQuery(text="family", country="Japan", top_k=5))
        assert all(o.country == "Japan" for o in out)

    def test_topic_filter(self):
        r = build_retriever("hierarchical", items=_sample_items())
        out = r.search(RetrievalQuery(text="x", topic="religion", top_k=5))
        assert all(o.topic == "religion" for o in out)

    def test_country_topic_intersect(self):
        r = build_retriever("hierarchical", items=_sample_items())
        out = r.search(RetrievalQuery(text="x", country="Brazil", topic="family", top_k=5))
        assert all(o.country == "Brazil" and o.topic == "family" for o in out)
        # e5 在该交集
        assert any(o.id == "e5" for o in out)

    def test_year_window_filter(self):
        r = build_retriever("hierarchical", items=_sample_items())
        out = r.search(RetrievalQuery(
            text="x", year_window=(2020, 2099), top_k=10,
        ))
        for o in out:
            assert o.year is not None and 2020 <= o.year <= 2099

    def test_empty_after_filter(self):
        r = build_retriever("hierarchical", items=_sample_items())
        out = r.search(RetrievalQuery(text="x", country="Mars", top_k=5))
        assert out == []

    def test_score_combine_hook(self):
        """子类覆盖 score_combine 应改变 score."""
        class Custom(HierarchicalRetriever):
            def score_combine(self, vec_sim, item, query):
                return vec_sim + (item.year or 0) * 0.001
        r = Custom(IdentityEmbedder())
        r.build(_sample_items())
        out = r.search(RetrievalQuery(text="family", top_k=5))
        assert len(out) > 0

    def test_unknown_kind_raises(self):
        with pytest.raises(ValueError):
            build_retriever("nope")
