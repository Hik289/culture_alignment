"""prompts.py 模板测试 — 不依赖 LLM, 验证 placeholder 全部能 fill.

覆盖:
- §13.2 survey distribution
- §13.3 norm judgment
- §13.4 daily knowledge (MC + short)
- §14.1-14.4 baselines
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.prompts import (  # noqa: E402
    SYSTEM_PROMPT,
    render_baseline,
    render_daily_knowledge,
    render_norm_judgment,
    render_survey_distribution,
    to_chat_messages,
)


REAL_EVIDENCE = [
    {
        "text": "In WVS Wave 7, 73% of Japanese respondents marked family as 'very important'.",
        "source": "WVS Wave 7",
        "year": 2019,
        "country": "Japan",
    },
    {
        "text": "OECD Better Life Index ranks Japan's life satisfaction below OECD average.",
        "source": "OECD BLI",
        "year": 2022,
    },
]

REAL_PROTO = {
    "individualism_low": True,
    "long_term_orientation": "high",
    "uncertainty_avoidance": "high",
}

REAL_DEMO = {"age_bucket": "30-39", "gender": "female", "education": "tertiary"}


# ---------------------------------------------------------------------------
# §13.2 调查分布
# ---------------------------------------------------------------------------

class TestSurveyDistribution:
    def test_basic_fill(self):
        r = render_survey_distribution(
            item_id="wvb_q1",
            country_or_region="Japan",
            question_text="How important is family in your life?",
            answer_options=["not important", "rather not", "rather important", "very important"],
            demographic_attributes=REAL_DEMO,
            cultural_prototype=REAL_PROTO,
            retrieved_evidence=REAL_EVIDENCE,
        )
        assert r["task_type"] == "survey_distribution"
        u = r["user"]
        # 关键字段都进了
        assert "Japan" in u
        assert "wvb_q1" in u
        assert "How important is family" in u
        assert "very important" in u
        assert "age_bucket=30-39" in u
        assert "individualism_low" in u
        assert "WVS Wave 7" in u
        # placeholder 应全部填完, 不剩 {.*}
        assert "{country_or_region}" not in u
        assert "{question_text}" not in u
        assert "{retrieved_evidence}" not in u
        assert "{prob_schema}" not in u

    def test_empty_evidence(self):
        r = render_survey_distribution(
            item_id="x",
            country_or_region="US",
            question_text="Q?",
            answer_options=["a", "b"],
        )
        assert "(无)" in r["user"]
        assert "{retrieved_evidence}" not in r["user"]

    def test_empty_options_raises(self):
        with pytest.raises(ValueError):
            render_survey_distribution(
                item_id="x",
                country_or_region="US",
                question_text="Q?",
                answer_options=[],
            )

    def test_to_chat_messages(self):
        r = render_survey_distribution(
            item_id="x",
            country_or_region="US",
            question_text="Q?",
            answer_options=["a", "b"],
        )
        msgs = to_chat_messages(r)
        assert len(msgs) == 2
        assert msgs[0]["role"] == "system"
        assert msgs[0]["content"] == SYSTEM_PROMPT
        assert msgs[1]["role"] == "user"


# ---------------------------------------------------------------------------
# §13.3 规范判断
# ---------------------------------------------------------------------------

class TestNormJudgment:
    def test_basic_fill(self):
        r = render_norm_judgment(
            item_id="normad_42",
            country_or_region="India",
            scenario="A guest refuses food offered by the host.",
            answer_options=["acceptable", "not_acceptable"],
            provided_value_or_norm="Hospitality is a core value.",
            retrieved_evidence=REAL_EVIDENCE,
        )
        u = r["user"]
        assert "India" in u
        assert "normad_42" in u
        assert "refuses food" in u
        assert "Hospitality is a core value" in u
        assert "acceptable" in u
        assert "{country_or_region}" not in u
        assert "{prob_schema}" not in u

    def test_no_value_or_evidence(self):
        r = render_norm_judgment(
            item_id="x",
            country_or_region="X",
            scenario="S",
            answer_options=["a", "b"],
        )
        # 两个 (无) 应出现
        assert r["user"].count("(无)") >= 2


# ---------------------------------------------------------------------------
# §13.4 日常文化知识
# ---------------------------------------------------------------------------

class TestDailyKnowledge:
    def test_mc_mode(self):
        r = render_daily_knowledge(
            item_id="blend_001",
            country_or_region="South Korea",
            language="ko",
            question_text="Which food is most commonly eaten for breakfast?",
            answer_format="MC",
            answer_options=["rice and soup", "bread", "cereal", "noodles"],
            retrieved_evidence=REAL_EVIDENCE,
        )
        assert r["task_type"] == "daily_knowledge_mc"
        u = r["user"]
        assert "South Korea" in u
        assert "ko" in u
        assert "blend_001" in u
        assert "rice and soup" in u
        assert '"probabilities"' in u  # MC mode has probs schema
        assert "{prob_schema}" not in u

    def test_short_mode(self):
        r = render_daily_knowledge(
            item_id="blend_short_1",
            country_or_region="Japan",
            language="ja",
            question_text="Name a traditional fermented food.",
            answer_format="short answer (string)",
        )
        assert r["task_type"] == "daily_knowledge_short"
        u = r["user"]
        assert "blend_short_1" in u
        assert '"answer": "<short string>"' in u
        assert '"probabilities"' not in u


# ---------------------------------------------------------------------------
# §14 Baselines
# ---------------------------------------------------------------------------

class TestBaselines:
    def test_no_culture(self):
        r = render_baseline(
            "no_culture",
            item_id="x",
            question_text="What is 2+2?",
            answer_options=["3", "4", "5"],
        )
        assert r["task_type"] == "baseline_no_culture"
        u = r["user"]
        assert "What is 2+2" in u
        # 不该出现国家/原型/人口字段
        assert "国家或地区" not in u
        assert "文化原型" not in u
        assert "人口属性" not in u

    def test_country(self):
        r = render_baseline(
            "country",
            item_id="x",
            question_text="Q",
            answer_options=["a", "b"],
            country_or_region="Brazil",
        )
        u = r["user"]
        assert "Brazil" in u
        assert "国家或地区" in u
        assert "文化原型" not in u

    def test_country_missing_raises(self):
        with pytest.raises(ValueError):
            render_baseline("country", item_id="x", question_text="Q", answer_options=["a"])

    def test_demographic(self):
        r = render_baseline(
            "demographic",
            item_id="x",
            question_text="Q",
            answer_options=["a", "b"],
            country_or_region="US",
            demographic_attributes=REAL_DEMO,
        )
        u = r["user"]
        assert "US" in u
        assert "30-39" in u
        assert "人口属性" in u

    def test_demographic_missing_raises(self):
        with pytest.raises(ValueError):
            render_baseline(
                "demographic",
                item_id="x",
                question_text="Q",
                answer_options=["a"],
                country_or_region="US",
            )

    def test_prototype(self):
        r = render_baseline(
            "prototype",
            item_id="x",
            question_text="Q",
            answer_options=["a", "b"],
            country_or_region="Japan",
            cultural_prototype=REAL_PROTO,
        )
        u = r["user"]
        assert "Japan" in u
        assert "individualism_low" in u

    def test_prototype_missing_raises(self):
        with pytest.raises(ValueError):
            render_baseline(
                "prototype",
                item_id="x",
                question_text="Q",
                answer_options=["a"],
                country_or_region="Japan",
            )

    def test_unknown_kind_raises(self):
        with pytest.raises(ValueError):
            render_baseline(
                "wat",
                item_id="x",
                question_text="Q",
                answer_options=["a"],
            )

    def test_short_answer_baseline(self):
        r = render_baseline(
            "country",
            item_id="x",
            question_text="What is the capital?",
            answer_options=None,
            country_or_region="Japan",
        )
        u = r["user"]
        assert '"answer": "<short string>"' in u
        assert '"probabilities"' not in u


# ---------------------------------------------------------------------------
# 全 placeholder 扫描 — 任一模板渲染后, 不应残留 {xxx} 占位符
# ---------------------------------------------------------------------------

import re  # noqa: E402

_PH_RE = re.compile(r"\{[a-z_][a-z0-9_]*\}")


def _has_placeholder(s: str) -> bool:
    """检查是否还有未填充的 {placeholder} (但 JSON 的 {...} 不算, 这里只匹配 {snake_case})."""
    return bool(_PH_RE.search(s))


class TestNoLeftoverPlaceholders:
    @pytest.mark.parametrize("kw", [
        dict(item_id="i", country_or_region="X", question_text="Q",
             answer_options=["a", "b"]),
        dict(item_id="i", country_or_region="X", question_text="Q",
             answer_options=["a"], demographic_attributes={"age": "20"},
             cultural_prototype=["x"], retrieved_evidence=REAL_EVIDENCE),
    ])
    def test_survey_no_placeholder(self, kw):
        r = render_survey_distribution(**kw)
        assert not _has_placeholder(r["user"]), r["user"]

    def test_norm_no_placeholder(self):
        r = render_norm_judgment(
            item_id="i", country_or_region="X", scenario="S",
            answer_options=["a", "b"], retrieved_evidence=REAL_EVIDENCE,
        )
        assert not _has_placeholder(r["user"]), r["user"]

    def test_daily_mc_no_placeholder(self):
        r = render_daily_knowledge(
            item_id="i", country_or_region="X", question_text="Q",
            answer_format="MC", answer_options=["a", "b"],
        )
        assert not _has_placeholder(r["user"])

    def test_daily_short_no_placeholder(self):
        r = render_daily_knowledge(
            item_id="i", country_or_region="X", question_text="Q",
            answer_format="short",
        )
        assert not _has_placeholder(r["user"])

    @pytest.mark.parametrize("kind,extra", [
        ("no_culture", {}),
        ("country", {"country_or_region": "X"}),
        ("demographic", {"country_or_region": "X", "demographic_attributes": REAL_DEMO}),
        ("prototype", {"country_or_region": "X", "cultural_prototype": REAL_PROTO}),
    ])
    def test_baselines_no_placeholder(self, kind, extra):
        r = render_baseline(
            kind, item_id="i", question_text="Q",
            answer_options=["a", "b"], **extra,
        )
        assert not _has_placeholder(r["user"]), (kind, r["user"])


# ---------------------------------------------------------------------------
# Researcher 派单: 4 类输出 schema 验证
#
# 每个模板的 user 文本里嵌入了一个 JSON schema 块, LLM 必须按该 schema 返回.
# 这里通过模拟 LLM 返回符合 schema 的合规 JSON, 验证下游 parser (azure_client._try_parse_json)
# 能正确解析这个 JSON, 且符合 prompts.py 在文本中声明的字段约束.
# ---------------------------------------------------------------------------

import json  # noqa: E402

from src.azure_client import _try_parse_json  # noqa: E402


def _extract_schema_block(user_text: str) -> str | None:
    """从 user prompt 中抓出最后一个 ```{ ... }``` 风格块或 '只输出 JSON: { ... }' 块."""
    import re
    # 抓 '只输出 JSON' 或 '输出 JSON' 后的最大花括号块
    m = re.search(r"(只输出|输出).{0,10}JSON.{0,50}?(\{.*?\})\s*$", user_text, re.DOTALL)
    if m:
        return m.group(2)
    # fallback: 最后一个 {...}
    rev = user_text.rfind("{")
    if rev >= 0:
        return user_text[rev:].split("\n\n")[0]
    return None


class TestEmbeddedSchemaIsParseable:
    """验证 prompt 中嵌入的 schema 文本本身可被下游 parser 处理 (虽然含 placeholder).

    严格 JSON 解析会失败 (有 <one of options> 等 placeholder), 但 _try_parse_json 应能
    抓出 {...} 平衡子串, 即结构 well-formed.
    """

    def test_survey_dist_schema_balanced(self):
        r = render_survey_distribution(
            item_id="i", country_or_region="X", question_text="Q",
            answer_options=["a", "b", "c"],
        )
        sch = _extract_schema_block(r["user"])
        assert sch is not None
        # 大括号配平
        assert sch.count("{") == sch.count("}")
        # 含关键字段
        assert "item_id" in sch
        assert "probabilities" in sch
        assert '"a"' in sch and '"c"' in sch

    def test_norm_judgment_schema_balanced(self):
        r = render_norm_judgment(
            item_id="i", country_or_region="X", scenario="S",
            answer_options=["yes", "no"],
        )
        sch = _extract_schema_block(r["user"])
        assert sch is not None
        assert sch.count("{") == sch.count("}")
        assert "answer" in sch
        assert "probabilities" in sch

    def test_daily_mc_schema_balanced(self):
        r = render_daily_knowledge(
            item_id="i", country_or_region="X", question_text="Q",
            answer_format="MC", answer_options=["a", "b"],
        )
        sch = _extract_schema_block(r["user"])
        assert sch is not None
        assert sch.count("{") == sch.count("}")

    def test_daily_short_schema_balanced(self):
        r = render_daily_knowledge(
            item_id="i", country_or_region="X", question_text="Q",
            answer_format="short",
        )
        sch = _extract_schema_block(r["user"])
        assert sch is not None
        assert sch.count("{") == sch.count("}")
        assert "answer" in sch
        # short answer 不应有 probabilities
        assert "probabilities" not in sch


class TestSimulatedLLMResponseParses:
    """模拟 LLM 返回符合 schema 的 JSON, 验证 _try_parse_json 解析成功且字段满足约束."""

    def test_survey_dist_response(self):
        # 模拟 LLM 按 survey 模板返回
        response = json.dumps({
            "item_id": "wvb_q1",
            "country_or_region": "Japan",
            "probabilities": {"1": 0.05, "2": 0.15, "3": 0.30, "4": 0.50},
        })
        parsed, fb, err = _try_parse_json(response)
        assert err is None
        assert parsed["item_id"] == "wvb_q1"
        # 概率非负且 sum ≈ 1
        probs = list(parsed["probabilities"].values())
        assert all(p >= 0 for p in probs)
        assert abs(sum(probs) - 1.0) < 1e-9

    def test_survey_dist_response_with_fence(self):
        """LLM 加 markdown fence 时也能被解析."""
        response = "```json\n" + json.dumps({
            "item_id": "i", "probabilities": {"a": 0.5, "b": 0.5}
        }) + "\n```"
        parsed, fb, err = _try_parse_json(response)
        assert err is None
        assert fb is True
        assert parsed["probabilities"]["a"] == 0.5

    def test_norm_judgment_response(self):
        response = json.dumps({
            "item_id": "n1", "answer": "no",
            "probabilities": {"yes": 0.1, "no": 0.8, "neutral": 0.1},
        })
        parsed, _, err = _try_parse_json(response)
        assert err is None
        assert parsed["answer"] == "no"
        # answer 在 probabilities keys 内
        assert parsed["answer"] in parsed["probabilities"]
        assert abs(sum(parsed["probabilities"].values()) - 1.0) < 1e-9

    def test_daily_mc_response(self):
        response = json.dumps({
            "item_id": "b1", "answer": "rice and miso soup",
            "probabilities": {"rice and miso soup": 0.9, "bread": 0.05, "noodles": 0.05},
        })
        parsed, _, err = _try_parse_json(response)
        assert err is None
        # answer 必须在 options
        assert parsed["answer"] in parsed["probabilities"]

    def test_daily_short_response(self):
        response = json.dumps({"item_id": "s1", "answer": "Canberra"})
        parsed, _, err = _try_parse_json(response)
        assert err is None
        assert isinstance(parsed["answer"], str)
        assert len(parsed["answer"]) > 0


class TestSchemaConstraintViolations:
    """模拟 LLM 给出违反约束的 JSON, 验证我们能识别 (这些通常在 run.py 里检查, 这里只看
    parser 不崩 + 通过 manual check 能发现问题)."""

    def test_prob_sum_violation(self):
        bad = json.dumps({
            "item_id": "x", "probabilities": {"a": 0.3, "b": 0.3, "c": 0.2}  # sum=0.8
        })
        parsed, _, err = _try_parse_json(bad)
        assert err is None
        s = sum(parsed["probabilities"].values())
        # 我们的检查应能发现 sum 偏离 1
        assert abs(s - 1.0) > 1e-3

    def test_negative_prob(self):
        bad = json.dumps({"item_id": "x", "probabilities": {"a": -0.1, "b": 1.1}})
        parsed, _, err = _try_parse_json(bad)
        assert err is None
        assert any(p < 0 for p in parsed["probabilities"].values())

    def test_missing_required_field(self):
        bad = json.dumps({"probabilities": {"a": 0.5, "b": 0.5}})  # 少 item_id
        parsed, _, err = _try_parse_json(bad)
        assert err is None
        assert "item_id" not in parsed

    def test_answer_not_in_options(self):
        bad = json.dumps({
            "item_id": "x", "answer": "wrong_label",
            "probabilities": {"yes": 0.5, "no": 0.5},
        })
        parsed, _, err = _try_parse_json(bad)
        assert err is None
        assert parsed["answer"] not in parsed["probabilities"]
