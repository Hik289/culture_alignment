"""prototypes.py 骨架测试 (不依赖网络/LLM)."""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.prototypes import (  # noqa: E402
    DEFAULT_WARNING,
    LLMSummarizerProtoBuilder,
    PrototypeCard,
    ProtoBuilder,
    RuleBasedProtoBuilder,
    group_evidence_by_country,
)


def _sample_evidence() -> list[dict]:
    return [
        {"id": "ev1", "text": "Family is central to Japanese society.",
         "country": "Japan", "topic": "family", "year": 2019, "split": "train"},
        {"id": "ev2", "text": "Japan has moderate institutional trust.",
         "country": "Japan", "topic": "politics", "year": 2020, "split": "train"},
        {"id": "ev3", "text": "Japanese cuisine traditional foods include sushi and miso.",
         "country": "Japan", "topic": "food", "year": 2018, "split": "train"},
        {"id": "ev4", "text": "Brazil has strong Catholic religious tradition.",
         "country": "Brazil", "topic": "religion", "year": 2018, "split": "train"},
        {"id": "ev5", "text": "Brazil family ties tend to be tight in many regions.",
         "country": "Brazil", "topic": "family", "year": 2021, "split": "train"},
        # test split → 应被默认排除
        {"id": "ev6", "text": "Test answer leak should be filtered.",
         "country": "Japan", "topic": "family", "year": 2022, "split": "test"},
    ]


class TestPrototypeCard:
    def test_minimal(self):
        c = PrototypeCard(country="Japan", summary="X")
        assert c.country == "Japan"
        assert c.summary == "X"
        assert c.warning == DEFAULT_WARNING  # 必填默认
        assert c.n_evidence == 0
        assert c.key_values == {}

    def test_to_prompt_dict(self):
        c = PrototypeCard(
            country="Japan",
            summary="Strong family values, high uncertainty avoidance.",
            key_values={"family": "high", "trust": "moderate"},
        )
        d = c.to_prompt_dict()
        assert d["summary"].startswith("Strong family")
        assert d["family"] == "high"
        assert d["trust"] == "moderate"
        assert d["warning"] == DEFAULT_WARNING

    def test_json_roundtrip(self):
        c = PrototypeCard(country="X", summary="Y", n_evidence=3,
                          key_values={"a": "b"}, sources=["s1", "s2"])
        s = c.to_json()
        d = json.loads(s)
        assert d["country"] == "X"
        assert d["key_values"]["a"] == "b"
        assert d["warning"] == DEFAULT_WARNING


class TestGroupEvidence:
    def test_group_and_exclude_test(self):
        groups = group_evidence_by_country(_sample_evidence())
        assert "Japan" in groups
        assert "Brazil" in groups
        # ev6 (Japan, split=test) 应被排除
        japan_ids = [e["id"] for e in groups["Japan"]]
        assert "ev6" not in japan_ids
        assert len(groups["Japan"]) == 3  # ev1, ev2, ev3
        assert len(groups["Brazil"]) == 2

    def test_custom_exclude(self):
        groups = group_evidence_by_country(
            _sample_evidence(), exclude_split=()
        )
        japan_ids = [e["id"] for e in groups["Japan"]]
        assert "ev6" in japan_ids  # 未排除

    def test_missing_country_skipped(self):
        rows = [{"id": "x", "text": "y"}, {"id": "z", "text": "w", "country": "X"}]
        groups = group_evidence_by_country(rows)
        assert "X" in groups
        assert len(groups["X"]) == 1
        # 没 country 的不进 group
        assert sum(len(v) for v in groups.values()) == 1


class TestRuleBasedProto:
    def test_build_one(self):
        b = RuleBasedProtoBuilder()
        groups = group_evidence_by_country(_sample_evidence())
        card = b.build_one("Japan", groups["Japan"])
        assert card.country == "Japan"
        assert card.n_evidence == 3
        assert card.summary  # 非空
        assert card.warning == DEFAULT_WARNING
        # topic 出现都被记录
        keys = set(card.key_values.keys())
        assert {"family", "politics", "food"}.issubset(keys)

    def test_empty_evidence(self):
        b = RuleBasedProtoBuilder()
        card = b.build_one("XX", [])
        assert card.country == "XX"
        assert card.n_evidence == 0
        assert "no evidence" in card.summary.lower()
        assert card.warning == DEFAULT_WARNING  # warning 仍必填

    def test_build_many(self):
        b = RuleBasedProtoBuilder()
        groups = group_evidence_by_country(_sample_evidence())
        cards = b.build_many(groups)
        assert set(cards.keys()) == {"Japan", "Brazil"}
        for c in cards.values():
            assert c.warning  # 全部必有 warning

    def test_summary_truncation(self):
        b = RuleBasedProtoBuilder()
        long_text = "x" * 500
        card = b.build_one("Y", [{"id": "1", "text": long_text, "topic": "t"}])
        assert len(card.summary) <= 240
        assert card.summary.endswith("...")

    def test_dump_load_roundtrip(self):
        b = RuleBasedProtoBuilder()
        groups = group_evidence_by_country(_sample_evidence())
        cards = b.build_many(groups)
        with tempfile.TemporaryResearchery() as td:
            p = Path(td) / "proto.json"
            b.dump(cards, p)
            loaded = ProtoBuilder.load(p)
        assert set(loaded.keys()) == set(cards.keys())
        for k in cards:
            assert loaded[k].summary == cards[k].summary
            assert loaded[k].warning == cards[k].warning


class TestLLMSummarizerPlaceholder:
    def test_falls_back_to_rule_based(self):
        """当前实现是占位, 直接 fallback. 不应抛错."""
        b = LLMSummarizerProtoBuilder()
        groups = group_evidence_by_country(_sample_evidence())
        card = b.build_one("Japan", groups["Japan"])
        assert card.country == "Japan"
        assert card.n_evidence == 3
        assert card.meta.get("builder", "").startswith("llm_placeholder")
        assert card.warning == DEFAULT_WARNING


class TestWarningEnforcement:
    """idea.md §Risks 要求 warning 字段必有. 验证所有路径都生成 warning."""

    def test_rule_based_always_has_warning(self):
        b = RuleBasedProtoBuilder()
        for ev in [[], [{"id": "1", "text": "x"}], _sample_evidence()]:
            c = b.build_one("X", ev)
            assert c.warning and len(c.warning) > 0

    def test_placeholder_always_has_warning(self):
        b = LLMSummarizerProtoBuilder()
        c = b.build_one("X", [{"id": "1", "text": "y"}])
        assert c.warning and len(c.warning) > 0

    def test_prompt_dict_carries_warning(self):
        c = PrototypeCard(country="X", summary="Y")
        d = c.to_prompt_dict()
        assert "warning" in d
        assert d["warning"]
