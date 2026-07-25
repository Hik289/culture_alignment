"""
test_leakage_check.py — H_anchor_5 ASSERT (readme §10.4)

Invariant:
  - Inject N=10 fake leakage pairs (evidence directly derived from test question)
    → recall = 1.0 (every one detected)
  - Inject N=10 clean evidence pairs (different question, different country/topic,
    different text, low embedding similarity)
    → false_positive_rate < 0.1

Each fake leak exercises a different rule (R1–R4) so the test simultaneously
validates rule coverage and overall detector behaviour.
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from leakage_check import (
    EvidenceRecord,
    LeakageChecker,
    TargetRecord,
    normalize_text,
    text_hash,
)

random.seed(20260615)


# ── helper: build a small canonical test-question pool ────────────────────
TEST_QUESTIONS = [
    {
        "id": "Q1",
        "benchmark": "worldvaluesbench",
        "text": "On a scale of 1 to 4, how important is family in your life?",
        "country": "Japan",
    },
    {
        "id": "goqa_abc123def456",
        "benchmark": "globalopinionqa",
        "text": "When it comes to immigration, do you think there are too many immigrants in your country?",
        "country": "France",
    },
    {
        "id": "nad_789ffe0011aa",
        "benchmark": "normad",
        "text": "Is it considered rude to keep your shoes on when entering someone's home in Japan?",
        "country": "Japan",
    },
    {
        "id": "Al-en-01",
        "benchmark": "blend",
        "text": "What is a common snack for preschool kids in the US?",
        "country": "US",
    },
    {
        "id": "Q46",
        "benchmark": "worldvaluesbench",
        "text": "How would you describe your political views on a scale of 1 to 10?",
        "country": "Germany",
    },
]

# ── 10 LEAK pairs, distributed across rules R1–R4 ─────────────────────────
def build_leak_pairs():
    pairs = []
    # R1 trigger ×2: evidence.split == "test"
    pairs.append((
        EvidenceRecord(evidence_id="lk_R1_a", source="worldvaluesbench",
                       split="test", source_question_id="Q1",
                       text="On a scale of 1 to 4, how important is family in your life?"),
        TargetRecord(target_benchmark="worldvaluesbench", target_split="test",
                     target_item_id="Q1",
                     question_text=TEST_QUESTIONS[0]["text"]),
        True))
    pairs.append((
        EvidenceRecord(evidence_id="lk_R1_b", source="globalopinionqa",
                       split="test", source_question_id="goqa_unrelated_xyz",
                       text="A totally unrelated question about EU policy."),
        TargetRecord(target_benchmark="globalopinionqa", target_split="test",
                     target_item_id="goqa_abc123def456",
                     question_text=TEST_QUESTIONS[1]["text"]),
        True))
    # R2 trigger ×2: same source_question_id == target_item_id, different split
    pairs.append((
        EvidenceRecord(evidence_id="lk_R2_a", source="worldvaluesbench",
                       split="train", source_question_id="Q1",
                       text="Country: Brazil. Distribution across the 4 options is broadly U-shaped."),
        TargetRecord(target_benchmark="worldvaluesbench", target_split="test",
                     target_item_id="Q1",
                     question_text=TEST_QUESTIONS[0]["text"]),
        True))
    pairs.append((
        EvidenceRecord(evidence_id="lk_R2_b", source="blend",
                       split="train", source_question_id="Al-en-01",
                       text="In the United States preschool kids snack on goldfish crackers and fruit."),
        TargetRecord(target_benchmark="blend", target_split="test",
                     target_item_id="Al-en-01",
                     question_text=TEST_QUESTIONS[3]["text"]),
        True))
    # R3 trigger ×3: normalized-text-hash match (paraphrase that normalizes to same)
    pairs.append((
        EvidenceRecord(evidence_id="lk_R3_a", source="normad",
                       split="train", source_question_id="nad_other_id",
                       # Same as target but with extra punctuation/case
                       text="IS IT considered RUDE to keep your shoes on when entering, someone's home in Japan???"),
        TargetRecord(target_benchmark="normad", target_split="test",
                     target_item_id="nad_789ffe0011aa",
                     question_text=TEST_QUESTIONS[2]["text"]),
        True))
    pairs.append((
        EvidenceRecord(evidence_id="lk_R3_b", source="globalopinionqa",
                       split="valid", source_question_id="goqa_different",
                       text="When it comes to immigration do you think there are too many immigrants in your country"),  # missing punct
        TargetRecord(target_benchmark="globalopinionqa", target_split="test",
                     target_item_id="goqa_abc123def456",
                     question_text=TEST_QUESTIONS[1]["text"]),
        True))
    pairs.append((
        EvidenceRecord(evidence_id="lk_R3_c", source="blend",
                       split="train", source_question_id="another_id",
                       text="what is a common snack for preschool kids in the us"),  # lowercased exact
        TargetRecord(target_benchmark="blend", target_split="test",
                     target_item_id="Al-en-99",  # different qid to avoid R2
                     question_text=TEST_QUESTIONS[3]["text"]),
        True))
    # R4 trigger ×3: near-duplicate paraphrase (same meaning, surface tweaks)
    # Each chosen to score >= 0.97 cosine similarity under all-MiniLM-L6-v2
    pairs.append((
        EvidenceRecord(evidence_id="lk_R4_a", source="worldvaluesbench",
                       split="train", source_question_id="Q1_other",
                       text="On a scale of 1-4, how important is family in your life?"),
        TargetRecord(target_benchmark="worldvaluesbench", target_split="test",
                     target_item_id="Q1_target",
                     question_text=TEST_QUESTIONS[0]["text"]),
        True))
    pairs.append((
        EvidenceRecord(evidence_id="lk_R4_b", source="normad",
                       split="train", source_question_id="nad_other",
                       text="Is it rude to keep your shoes on when entering someone's home in Japan?"),
        TargetRecord(target_benchmark="normad", target_split="test",
                     target_item_id="nad_target",
                     question_text=TEST_QUESTIONS[2]["text"]),
        True))
    pairs.append((
        EvidenceRecord(evidence_id="lk_R4_c", source="worldvaluesbench",
                       split="train", source_question_id="Q46_other",
                       text="On a scale of 1 to 10, how would you describe your political views?"),
        TargetRecord(target_benchmark="worldvaluesbench", target_split="test",
                     target_item_id="Q46_target",
                     question_text=TEST_QUESTIONS[4]["text"]),
        True))
    return pairs


# ── 10 CLEAN pairs (must NOT trigger any rule) ────────────────────────────
def build_clean_pairs():
    pairs = []
    # different question, different topic, different country
    pairs.append((
        EvidenceRecord(evidence_id="cl_01", source="worldvaluesbench",
                       split="train", source_question_id="Q2",
                       text="How important is religion in your life on a 1 to 4 scale?"),
        TargetRecord(target_benchmark="worldvaluesbench", target_split="test",
                     target_item_id="Q1",
                     question_text=TEST_QUESTIONS[0]["text"]),
        False))
    pairs.append((
        EvidenceRecord(evidence_id="cl_02", source="globalopinionqa",
                       split="train", source_question_id="goqa_otherid",
                       text="Should foreign aid be increased by the government?"),
        TargetRecord(target_benchmark="globalopinionqa", target_split="test",
                     target_item_id="goqa_abc123def456",
                     question_text=TEST_QUESTIONS[1]["text"]),
        False))
    pairs.append((
        EvidenceRecord(evidence_id="cl_03", source="normad",
                       split="train", source_question_id="nad_other_qid",
                       text="Bowing is the standard greeting in Japan, and the depth of the bow indicates respect."),
        TargetRecord(target_benchmark="normad", target_split="test",
                     target_item_id="nad_789ffe0011aa",
                     question_text=TEST_QUESTIONS[2]["text"]),
        False))
    pairs.append((
        EvidenceRecord(evidence_id="cl_04", source="blend",
                       split="train", source_question_id="Al-en-02",
                       text="What is a popular food to go with beer in the US?"),
        TargetRecord(target_benchmark="blend", target_split="test",
                     target_item_id="Al-en-01",
                     question_text=TEST_QUESTIONS[3]["text"]),
        False))
    pairs.append((
        EvidenceRecord(evidence_id="cl_05", source="worldvaluesbench",
                       split="valid", source_question_id="Q47",
                       text="How would you rate your overall life satisfaction on a 1 to 10 scale?"),
        TargetRecord(target_benchmark="worldvaluesbench", target_split="test",
                     target_item_id="Q46",
                     question_text=TEST_QUESTIONS[4]["text"]),
        False))
    pairs.append((
        EvidenceRecord(evidence_id="cl_06", source="culturebank",
                       split="train", source_question_id=None,
                       text="In Japan many households remove their shoes at the genkan, an entryway recess."),
        TargetRecord(target_benchmark="normad", target_split="test",
                     target_item_id="nad_789ffe0011aa",
                     question_text=TEST_QUESTIONS[2]["text"]),
        False))
    pairs.append((
        EvidenceRecord(evidence_id="cl_07", source="globalopinionqa",
                       split="train", source_question_id="goqa_diff_topic",
                       text="Do you support investment in renewable energy infrastructure?"),
        TargetRecord(target_benchmark="globalopinionqa", target_split="test",
                     target_item_id="goqa_abc123def456",
                     question_text=TEST_QUESTIONS[1]["text"]),
        False))
    pairs.append((
        EvidenceRecord(evidence_id="cl_08", source="worldvaluesbench",
                       split="train", source_question_id="Q3",
                       text="How important is leisure time on a 1 to 4 scale?"),
        TargetRecord(target_benchmark="worldvaluesbench", target_split="test",
                     target_item_id="Q1",
                     question_text=TEST_QUESTIONS[0]["text"]),
        False))
    pairs.append((
        EvidenceRecord(evidence_id="cl_09", source="blend",
                       split="train", source_question_id="Br-en-03",
                       text="What is the most popular sport for adults in Brazil?"),
        TargetRecord(target_benchmark="blend", target_split="test",
                     target_item_id="Al-en-01",
                     question_text=TEST_QUESTIONS[3]["text"]),
        False))
    pairs.append((
        EvidenceRecord(evidence_id="cl_10", source="normad",
                       split="train", source_question_id="nad_diff_country",
                       text="In Germany direct eye contact during conversation signals attentiveness."),
        TargetRecord(target_benchmark="normad", target_split="test",
                     target_item_id="nad_789ffe0011aa",
                     question_text=TEST_QUESTIONS[2]["text"]),
        False))
    return pairs


# ── tests ─────────────────────────────────────────────────────────────────
def test_normalize_text_is_robust():
    assert normalize_text("ABC, def?") == normalize_text("abc def")
    assert normalize_text("家庭，重要吗？") == normalize_text("家庭 重要吗")
    assert text_hash("Hello, world!") == text_hash("hello world")


@pytest.fixture(scope="module")
def checker():
    return LeakageChecker(similarity_threshold=0.95)


@pytest.fixture(scope="module")
def all_pairs():
    return build_leak_pairs() + build_clean_pairs()


def test_leakage_check_recall_and_fpr(checker, all_pairs):
    report = checker.evaluate(all_pairs)

    # Persist for analysis/anchor_5_assertions.json
    out_path = Path(__file__).resolve().parents[1] / "analysis" / "_pytest_anchor_5_raw.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        # strip embeddings, keep small
        rep_small = {k: v for k, v in report.items() if k != "per_pair"}
        rep_small["per_pair"] = report["per_pair"]
        json.dump(rep_small, f, indent=2, ensure_ascii=False)

    assert report["recall"] == 1.0, (
        f"recall={report['recall']} < 1.0; missed leaks: "
        + str([p for p in report["per_pair"]
               if p["gold_label"] and not p["predicted_leak"]])
    )
    assert report["false_positive_rate"] < 0.1, (
        f"FPR={report['false_positive_rate']} >= 0.1; false positives: "
        + str([p for p in report["per_pair"]
               if (not p["gold_label"]) and p["predicted_leak"]])
    )


def test_each_rule_fires_at_least_once(checker, all_pairs):
    report = checker.evaluate(all_pairs)
    hits = report["rule_hits"]
    assert hits["R1"] >= 1, hits
    assert hits["R2"] >= 1, hits
    assert hits["R3"] >= 1, hits
    assert hits["R4"] >= 1, hits


def test_clean_pair_with_same_country_does_not_leak(checker):
    """A non-leak record about the same country/topic must NOT be flagged.

    This guards against an over-aggressive R4 threshold.
    """
    ev = EvidenceRecord(evidence_id="cl_topic_match", source="normad",
                        split="train", source_question_id="nad_x",
                        text="Bowing is the standard greeting in Japan, and the depth of the bow indicates respect.")
    tg = TargetRecord(target_benchmark="normad", target_split="test",
                      target_item_id="nad_target",
                      question_text="Is it considered rude to keep your shoes on when entering someone's home in Japan?")
    v = checker.is_leak(ev, tg)
    assert not v.is_leak, f"clean topical evidence wrongly flagged: {v.reasons}, sim={v.similarity}"
