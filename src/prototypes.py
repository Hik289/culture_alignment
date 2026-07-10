"""文化原型生成接口骨架 (STANDBY 阶段, 不跑).

readme §9.3 文化原型 + §13.2 训练数据文化原型字段.

原型 = 从 train 集 evidence 聚合得到的一个国家 / 地区的简短特征描述.
不暴露任何 test 答案 (由 leakage_check 保证).

设计:
- 抽象 ProtoBuilder + 几个常用实现 (rule-based aggregator / LLM-summarizer hook)
- 数据契约 PrototypeCard
- 序列化 / 反序列化为 JSON (供 prompts.render_baseline("prototype") 直接喂)
- warning 字段强制存在 (idea.md §Risks: "文化刻板印象 → 所有输出 ... 添加 warning 字段")

EXP_DESIGN 后:
- 增加 LLM-as-summarizer 后端 (基于 train evidence 用 gpt-5.4-mini 写 1-2 句原型)
- 增加质量评估 (perplexity / coverage)
"""

from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence

logger = logging.getLogger(__name__)

# 默认 warning 文本 (idea.md §Risks)
DEFAULT_WARNING = (
    "本原型仅反映训练数据中的聚合分布, 不代表该国家 / 地区中的每个个体, "
    "不应被当作绝对化的文化刻板印象."
)


# ---------------------------------------------------------------------------
# 数据契约
# ---------------------------------------------------------------------------

@dataclass
class PrototypeCard:
    """单一国家 / 地区的文化原型.

    Fields:
      country         : 国家 / 地区 (规范化, 如 ISO-3166 短名)
      summary         : 1-2 句概括 (人类可读)
      key_values      : 短字典 (维度 → 强度), e.g. {"family": "high", "individualism": "low"}
      sources         : 该原型聚合时用到的 evidence id 列表
      n_evidence      : 聚合用到的证据条数
      warning         : 必填. 默认 DEFAULT_WARNING
      meta            : 自由字段 (生成方法 / 时间 / 模型 / ...)
    """
    country: str
    summary: str
    key_values: dict[str, str] = field(default_factory=dict)
    sources: list[str] = field(default_factory=list)
    n_evidence: int = 0
    warning: str = DEFAULT_WARNING
    meta: dict[str, Any] = field(default_factory=dict)

    def to_prompt_dict(self) -> dict:
        """转成 prompts.render_baseline / render_survey_distribution 接受的 dict.

        prompts._fmt_prototype 接受 dict[str, Any] → 渲染为
            - key: value
            - key: value
        """
        out: dict[str, Any] = {"summary": self.summary}
        out.update(self.key_values)
        out["warning"] = self.warning
        return out

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False, indent=2)


# ---------------------------------------------------------------------------
# 基类
# ---------------------------------------------------------------------------

class ProtoBuilder(ABC):
    """从 evidence 聚合到 PrototypeCard."""

    @abstractmethod
    def build_one(self, country: str, evidence: Sequence[dict]) -> PrototypeCard:
        """单个国家 → 一张卡片. evidence dict 至少含 text/source/year."""

    def build_many(
        self, evidence_by_country: dict[str, Sequence[dict]]
    ) -> dict[str, PrototypeCard]:
        """批量构建. 默认顺序循环, 子类可重写为并行."""
        out: dict[str, PrototypeCard] = {}
        for country, ev in evidence_by_country.items():
            out[country] = self.build_one(country, ev)
            logger.debug("built prototype for %s (n_ev=%d)", country, len(ev))
        return out

    def dump(self, cards: dict[str, PrototypeCard], path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {k: asdict(v) for k, v in cards.items()}
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2))
        logger.info("dumped %d prototype cards → %s", len(cards), path)

    @classmethod
    def load(cls, path: str | Path) -> dict[str, PrototypeCard]:
        path = Path(path)
        raw = json.loads(path.read_text())
        out: dict[str, PrototypeCard] = {}
        for k, v in raw.items():
            out[k] = PrototypeCard(**v)
        return out


# ---------------------------------------------------------------------------
# 规则式聚合 (无需 LLM, 单元测试可跑)
# ---------------------------------------------------------------------------

class RuleBasedProtoBuilder(ProtoBuilder):
    """简单聚合: source 计数 + topic 出现频率 + 用最常出现的 evidence text 作 summary.

    用于:
      (a) 单元测试 / sanity check
      (b) baseline 对比 (vs LLM summarizer)
    """

    def __init__(self, max_sources: int = 10):
        self.max_sources = max_sources

    def build_one(self, country: str, evidence: Sequence[dict]) -> PrototypeCard:
        if not evidence:
            return PrototypeCard(
                country=country,
                summary="(no evidence in train split)",
                n_evidence=0,
                meta={"builder": "rule_based"},
            )

        # 计 topic 频率
        topic_counts = Counter(
            (e.get("topic") or "unknown").lower() for e in evidence
        )
        # 用最长的 evidence text 做 summary (信息量近似)
        texts = [(e.get("text") or "").strip() for e in evidence if e.get("text")]
        summary = max(texts, key=len, default="(no text)")
        # 截断, 避免过长
        if len(summary) > 240:
            summary = summary[:237] + "..."

        # key_values: topic → high/low 启发式 (count > median 算 high)
        if topic_counts:
            counts = sorted(topic_counts.values())
            median = counts[len(counts) // 2]
            key_values = {
                t: ("high" if c > median else "moderate")
                for t, c in topic_counts.items()
            }
        else:
            key_values = {}

        # 取前 N 个 source id
        sources = [str(e.get("id") or e.get("source") or "") for e in evidence]
        sources = [s for s in sources if s][: self.max_sources]

        return PrototypeCard(
            country=country,
            summary=summary,
            key_values=key_values,
            sources=sources,
            n_evidence=len(evidence),
            warning=DEFAULT_WARNING,
            meta={"builder": "rule_based"},
        )


# ---------------------------------------------------------------------------
# LLM Summarizer 占位 (EXP_DESIGN 后接 src.model_client.chat_json)
# ---------------------------------------------------------------------------

class LLMSummarizerProtoBuilder(ProtoBuilder):
    """用 LLM 对一组 evidence 写 1-2 句原型.

    当前是占位 (raise NotImplementedError on real call), 接口稳定后 EXP_DESIGN 后实现:
        - 调 src.model_client.chat_json(...) with json_object response_format
        - prompt 强制要求 warning 字段非空
        - 失败 → fallback 到 RuleBasedProtoBuilder.build_one
    """

    def __init__(self, *, max_evidence: int = 20, fallback: Optional[ProtoBuilder] = None):
        self.max_evidence = max_evidence
        self.fallback = fallback or RuleBasedProtoBuilder()

    def build_one(self, country: str, evidence: Sequence[dict]) -> PrototypeCard:
        # EXP_DESIGN 后实现; 当前只回退 rule-based 让 pipeline 不破
        logger.warning("LLMSummarizerProtoBuilder.build_one 占位实现, 回退 rule-based (待 EXP_DESIGN 后接 LLM)")
        card = self.fallback.build_one(country, evidence)
        card.meta = {**card.meta, "builder": "llm_placeholder_fallback_to_rule_based"}
        return card


# ---------------------------------------------------------------------------
# 便利: 按 evidence rows 列分组
# ---------------------------------------------------------------------------

def group_evidence_by_country(
    rows: Iterable[dict],
    *,
    country_field: str = "country",
    exclude_split: Sequence[str] = ("test", "probe"),
) -> dict[str, list[dict]]:
    """从 evidence rows (e.g. parquet to_dict('records')) 按国家分组, 排除测试集.

    Args:
        rows         : 每行 dict, 必须含 country_field 和 'split' (optional)
        country_field: 国家字段名
        exclude_split: 排除哪些 split (默认排除 test/probe 防泄漏)
    """
    out: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        c = r.get(country_field)
        if not c:
            continue
        s = r.get("split")
        if s in exclude_split:
            continue
        out[str(c)].append(r)
    return dict(out)


__all__ = [
    "DEFAULT_WARNING",
    "PrototypeCard",
    "ProtoBuilder",
    "RuleBasedProtoBuilder",
    "LLMSummarizerProtoBuilder",
    "group_evidence_by_country",
]
