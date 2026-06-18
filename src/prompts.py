"""提示模板 (readme §13 + §14).

设计:
- 所有模板用 str.format(**fields) 填充, 不用 f-string (允许在 dict 中动态拼)
- 每个模板暴露:
    - SYSTEM_PROMPT: 系统提示 (readme §13.1)
    - render_xxx(fields) -> {"system": ..., "user": ...}: 渲染为可直接喂给 OpenAI chat.completions.create 的 messages
- placeholder 缺失时抛 KeyError, 不静默
- 适配 4 类任务:
    1. survey distribution (WVB / GOQA ordinal)  — §13.2
    2. norm judgment (NormAd)                    — §13.3
    3. daily culture knowledge (BLEnD)           — §13.4
    4. baselines: no-culture / country / demo / prototype — §14.1-14.4
- evidence / prototype 可为 None → 渲染时输出 "(无)" 或省略 (传 strict=False 时, 默认 True)
"""

from __future__ import annotations

import json
from typing import Any, Mapping, Sequence

# ---------------------------------------------------------------------------
# 系统提示
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "你需要根据给定的调查证据、文化背景和问题, 预测群体层面的回答分布或完成文化判断。\n"
    "证据只表示聚合数据或特定资料, 不表示该群体中的每个人。\n"
    "不要使用绝对化文化刻板印象。\n"
    "严格按照给定 JSON 格式输出。"
)


# ---------------------------------------------------------------------------
# 工具
# ---------------------------------------------------------------------------

def _fmt_options(options: Sequence[str]) -> str:
    """将 options 列表渲染成多行编号格式."""
    return "\n".join(f"  {i+1}. {o}" for i, o in enumerate(options))


def _fmt_evidence(evidence: Any) -> str:
    """evidence 可以是 None / str / list[str] / list[dict].

    list[dict] 时假设每条有 'text' 或 'content' 或 'value' 字段, 取首个可用.
    None / 空 → '(无)'.
    """
    if evidence is None:
        return "(无)"
    if isinstance(evidence, str):
        return evidence.strip() or "(无)"
    if isinstance(evidence, list):
        if not evidence:
            return "(无)"
        lines = []
        for i, e in enumerate(evidence, 1):
            if isinstance(e, str):
                lines.append(f"[{i}] {e}")
            elif isinstance(e, dict):
                txt = e.get("text") or e.get("content") or e.get("value") or json.dumps(e, ensure_ascii=False)
                meta = []
                if "source" in e:
                    meta.append(f"source={e['source']}")
                if "year" in e:
                    meta.append(f"year={e['year']}")
                if "country" in e:
                    meta.append(f"country={e['country']}")
                meta_s = f" ({', '.join(meta)})" if meta else ""
                lines.append(f"[{i}] {txt}{meta_s}")
            else:
                lines.append(f"[{i}] {e}")
        return "\n".join(lines)
    return str(evidence)


def _fmt_demographics(demo: Any) -> str:
    """demo dict {age: 30-39, gender: female, ...} → 'age=30-39; gender=female'."""
    if demo is None:
        return "(无)"
    if isinstance(demo, dict):
        if not demo:
            return "(无)"
        return "; ".join(f"{k}={v}" for k, v in demo.items())
    return str(demo)


def _fmt_prototype(proto: Any) -> str:
    if proto is None:
        return "(无)"
    if isinstance(proto, dict):
        return "\n".join(f"- {k}: {v}" for k, v in proto.items()) or "(无)"
    if isinstance(proto, list):
        return "\n".join(f"- {x}" for x in proto) or "(无)"
    return str(proto)


# ---------------------------------------------------------------------------
# §13.2 调查分布提示 (WVB / GOQA)
# ---------------------------------------------------------------------------

SURVEY_DIST_USER_TEMPLATE = """任务：
估计指定国家、地区或人口群体对目标调查题的回答分布。

国家或地区：
{country_or_region}

人口属性：
{demographic_attributes}

训练数据文化原型：
{cultural_prototype}

检索到的训练证据：
{retrieved_evidence}

目标问题：
{question_text}

回答选项：
{answer_options}

只输出 JSON：
{{
  "item_id": "{item_id}",
  "country_or_region": "{country_or_region}",
  "probabilities": {{
{prob_schema}
  }}
}}"""


def render_survey_distribution(
    *,
    item_id: str,
    country_or_region: str,
    question_text: str,
    answer_options: Sequence[str],
    demographic_attributes: Any = None,
    cultural_prototype: Any = None,
    retrieved_evidence: Any = None,
) -> dict:
    """渲染调查分布提示 (§13.2)."""
    if not answer_options:
        raise ValueError("answer_options 不能为空")
    prob_lines = ",\n".join(f'    "{o}": probability' for o in answer_options)
    user = SURVEY_DIST_USER_TEMPLATE.format(
        item_id=item_id,
        country_or_region=country_or_region,
        demographic_attributes=_fmt_demographics(demographic_attributes),
        cultural_prototype=_fmt_prototype(cultural_prototype),
        retrieved_evidence=_fmt_evidence(retrieved_evidence),
        question_text=question_text,
        answer_options=_fmt_options(answer_options),
        prob_schema=prob_lines,
    )
    return {"system": SYSTEM_PROMPT, "user": user, "task_type": "survey_distribution"}


# ---------------------------------------------------------------------------
# §13.3 规范判断提示 (NormAd)
# ---------------------------------------------------------------------------

NORM_JUDGMENT_USER_TEMPLATE = """任务：
根据给定国家或地区的文化背景、规范证据和场景, 判断该行为在当前场景中是否合适。

国家或地区：
{country_or_region}

抽象价值或明确规范：
{provided_value_or_norm}

检索证据：
{retrieved_evidence}

场景：
{scenario}

回答选项：
{answer_options}

输出每个选项的概率和最终答案, 只输出 JSON:
{{
  "item_id": "{item_id}",
  "answer": "<one of options>",
  "probabilities": {{
{prob_schema}
  }}
}}"""


def render_norm_judgment(
    *,
    item_id: str,
    country_or_region: str,
    scenario: str,
    answer_options: Sequence[str],
    provided_value_or_norm: Any = None,
    retrieved_evidence: Any = None,
) -> dict:
    """渲染规范判断提示 (§13.3)."""
    if not answer_options:
        raise ValueError("answer_options 不能为空")
    prob_lines = ",\n".join(f'    "{o}": probability' for o in answer_options)
    val_str = provided_value_or_norm if provided_value_or_norm else "(无)"
    user = NORM_JUDGMENT_USER_TEMPLATE.format(
        item_id=item_id,
        country_or_region=country_or_region,
        provided_value_or_norm=val_str,
        retrieved_evidence=_fmt_evidence(retrieved_evidence),
        scenario=scenario,
        answer_options=_fmt_options(answer_options),
        prob_schema=prob_lines,
    )
    return {"system": SYSTEM_PROMPT, "user": user, "task_type": "norm_judgment"}


# ---------------------------------------------------------------------------
# §13.4 日常文化知识提示 (BLEnD)
# ---------------------------------------------------------------------------

DAILY_KNOW_USER_TEMPLATE = """任务：
回答与指定国家、地区或语言相关的日常文化问题。

国家或地区：
{country_or_region}

语言：
{language}

检索证据：
{retrieved_evidence}

问题：
{question_text}

回答选项或回答格式：
{answer_format}

只输出 JSON:
{schema_hint}"""

DAILY_KNOW_MC_SCHEMA = """{
  "item_id": "{item_id}",
  "answer": "<one of options>",
  "probabilities": {
{prob_schema}
  }
}"""

DAILY_KNOW_SHORT_SCHEMA = """{
  "item_id": "{item_id}",
  "answer": "<short string>"
}"""


def render_daily_knowledge(
    *,
    item_id: str,
    country_or_region: str,
    question_text: str,
    answer_format: str,
    language: str = "english",
    retrieved_evidence: Any = None,
    answer_options: Sequence[str] | None = None,
) -> dict:
    """渲染日常文化知识提示 (§13.4).

    answer_options 给出 → MC 模式 (schema 含 probabilities)
    否则 → short answer 模式
    """
    if answer_options:
        prob_lines = ",\n".join(f'    "{o}": probability' for o in answer_options)
        schema_hint = DAILY_KNOW_MC_SCHEMA.replace("{item_id}", item_id).replace(
            "{prob_schema}", prob_lines
        )
    else:
        schema_hint = DAILY_KNOW_SHORT_SCHEMA.replace("{item_id}", item_id)

    user = DAILY_KNOW_USER_TEMPLATE.format(
        country_or_region=country_or_region,
        language=language or "(unspecified)",
        retrieved_evidence=_fmt_evidence(retrieved_evidence),
        question_text=question_text,
        answer_format=answer_format,
        schema_hint=schema_hint,
    )
    task_type = "daily_knowledge_mc" if answer_options else "daily_knowledge_short"
    return {"system": SYSTEM_PROMPT, "user": user, "task_type": task_type}


# ---------------------------------------------------------------------------
# §14 Baselines
# ---------------------------------------------------------------------------

BASELINE_NO_CULTURE = """任务：
回答下面的问题, 给出对回答选项的概率分布 (若有) 或简短答案。

问题：
{question_text}

回答选项：
{answer_options_or_format}

{schema_hint}"""

BASELINE_COUNTRY = """任务：
针对指定国家或地区的群体, 回答下面的问题。

国家或地区：
{country_or_region}

问题：
{question_text}

回答选项：
{answer_options_or_format}

{schema_hint}"""

BASELINE_DEMO = """任务：
针对指定国家+人口属性的群体, 回答下面的问题。

国家或地区：
{country_or_region}

人口属性：
{demographic_attributes}

问题：
{question_text}

回答选项：
{answer_options_or_format}

{schema_hint}"""

BASELINE_PROTOTYPE = """任务：
针对指定国家或地区, 在文化原型背景下回答下面的问题。

国家或地区：
{country_or_region}

文化原型：
{cultural_prototype}

问题：
{question_text}

回答选项：
{answer_options_or_format}

{schema_hint}"""


def _baseline_schema_hint(
    item_id: str, answer_options: Sequence[str] | None
) -> str:
    if answer_options:
        prob_lines = ",\n".join(f'    "{o}": probability' for o in answer_options)
        return (
            "只输出 JSON:\n{\n"
            f'  "item_id": "{item_id}",\n'
            f'  "answer": "<one of options>",\n'
            f'  "probabilities": {{\n{prob_lines}\n  }}\n'
            "}"
        )
    return (
        "只输出 JSON:\n{\n"
        f'  "item_id": "{item_id}",\n'
        f'  "answer": "<short string>"\n'
        "}"
    )


def render_baseline(
    kind: str,
    *,
    item_id: str,
    question_text: str,
    answer_options: Sequence[str] | None = None,
    country_or_region: str | None = None,
    demographic_attributes: Any = None,
    cultural_prototype: Any = None,
) -> dict:
    """渲染 §14 基线.

    kind ∈ {"no_culture", "country", "demographic", "prototype"}.
    """
    options_str = _fmt_options(answer_options) if answer_options else "(短答案)"
    schema_hint = _baseline_schema_hint(item_id, answer_options)

    if kind == "no_culture":
        user = BASELINE_NO_CULTURE.format(
            question_text=question_text,
            answer_options_or_format=options_str,
            schema_hint=schema_hint,
        )
    elif kind == "country":
        if country_or_region is None:
            raise ValueError("country 基线需要 country_or_region")
        user = BASELINE_COUNTRY.format(
            country_or_region=country_or_region,
            question_text=question_text,
            answer_options_or_format=options_str,
            schema_hint=schema_hint,
        )
    elif kind == "demographic":
        if country_or_region is None or demographic_attributes is None:
            raise ValueError("demographic 基线需要 country_or_region + demographic_attributes")
        user = BASELINE_DEMO.format(
            country_or_region=country_or_region,
            demographic_attributes=_fmt_demographics(demographic_attributes),
            question_text=question_text,
            answer_options_or_format=options_str,
            schema_hint=schema_hint,
        )
    elif kind == "prototype":
        if country_or_region is None or cultural_prototype is None:
            raise ValueError("prototype 基线需要 country_or_region + cultural_prototype")
        user = BASELINE_PROTOTYPE.format(
            country_or_region=country_or_region,
            cultural_prototype=_fmt_prototype(cultural_prototype),
            question_text=question_text,
            answer_options_or_format=options_str,
            schema_hint=schema_hint,
        )
    else:
        raise ValueError(f"未知 baseline kind: {kind}")

    return {"system": SYSTEM_PROMPT, "user": user, "task_type": f"baseline_{kind}"}


# ---------------------------------------------------------------------------
# 统一打包成 OpenAI messages
# ---------------------------------------------------------------------------

def to_chat_messages(rendered: Mapping[str, Any]) -> list[dict]:
    """{system, user, task_type} → OpenAI messages list."""
    return [
        {"role": "system", "content": rendered["system"]},
        {"role": "user", "content": rendered["user"]},
    ]


__all__ = [
    "SYSTEM_PROMPT",
    "render_survey_distribution",
    "render_norm_judgment",
    "render_daily_knowledge",
    "render_baseline",
    "to_chat_messages",
]
