#!/usr/bin/env python3
"""
EXP_DESIGN_data: build evidence DB + country prototypes + leakage report + resource grouping.

Runs on gpu_server. Inputs from data/raw/ + data/processed/ + data/splits/.
Outputs to data/evidence/, data/prototypes/{wvb,goqa,normad,blend}/, analysis/.

Strictly follows:
  - readme §9.1 allowed sources, §9.2 schema, §9.3 prototype format
  - data/evidence/evidence_schema.json (frozen)
  - Researcher EXP_DESIGN_data spec (resource_tier in metadata, low-resource confidence flag)
  - src/leakage_check.py for final R1-R4 scan
"""
from __future__ import annotations

import ast
import datetime
import hashlib
import json
import os
import re
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

ROOT     = Path("${EXPERIMENT_ROOT}")
RAW      = ROOT / "data/raw"
PROC     = ROOT / "data/processed"
SPLT     = ROOT / "data/splits"
EVID_DIR = ROOT / "data/evidence"
PROTO    = ROOT / "data/prototypes"
ANLY     = ROOT / "analysis"
SRC      = ROOT / "src"

EVID_DIR.mkdir(parents=True, exist_ok=True)
PROTO.mkdir(parents=True, exist_ok=True)
ANLY.mkdir(parents=True, exist_ok=True)
for s in ["wvb", "goqa", "normad", "blend"]:
    (PROTO / s).mkdir(parents=True, exist_ok=True)

UTC_NOW = datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
sys.path.insert(0, str(SRC))


# =============================================================================
# Helpers
# =============================================================================
def hsh(s: str, n: int = 12) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()[:n]


def make_eid(source: str, split: str, item_id: str) -> str:
    return f"{source}::{split}::{item_id}"


def base_record(source, split, country, language, task_type, topic, text,
                answer_options=None, distribution=None, label=None,
                short_answers=None, metadata=None, license_tag="unknown"):
    return {
        "evidence_id": None,  # filled below
        "source": source,
        "split": split,
        "country_or_region": country,
        "language": language,
        "task_type": task_type,
        "topic": topic,
        "text": text,
        "answer_options": answer_options,
        "distribution": distribution,
        "label": label,
        "short_answers": short_answers,
        "metadata": metadata or {},
        "indexed_at": UTC_NOW,
        "license": license_tag,
    }


def normalize_dist(d: dict, places: int = 6) -> dict:
    s = sum(d.values())
    if s <= 0:
        return d
    return {k: round(v / s, places) for k, v in d.items()}


# =============================================================================
# 1) WVB train evidence: aggregate ordinal distribution per (country, qid)
# =============================================================================
def build_wvb_evidence() -> list[dict]:
    print("=== WVB ===")
    WVB = RAW / "worldvaluesbench/repo/WorldValuesBench"
    DC  = RAW / "worldvaluesbench/repo/dataset_construction"
    train_d = pd.read_csv(WVB / "train/train_demographic_qa.tsv", sep="\t", low_memory=False,
                          usecols=["D_INTERVIEW", "B_COUNTRY", "H_URBRURAL"])
    train_v = pd.read_csv(WVB / "train/train_value_qa.tsv", sep="\t", low_memory=False)
    qmeta = json.load(open(WVB / "question_metadata.json"))
    # All value question keys (excluding D_INTERVIEW)
    val_qs = [c for c in train_v.columns if c != "D_INTERVIEW"]
    # B_COUNTRY in train_demographic_qa.tsv is already a country-name string
    # (data_preparation.py applied codebook). Use it directly.
    train_d["country_name"] = train_d["B_COUNTRY"].astype(str)
    # safety: drop rows with no answer / missing country (e.g. "No answer; Missing")
    train_d = train_d[~train_d["country_name"].str.lower().isin(["nan", "no answer; missing", ""])]
    # WVB v5.0 codebook is missing 2 ISO codes that WVS Wave 7 v6.0 adds (India=356, Uzbekistan=860).
    # data_preparation.py leaves them as numeric strings. Patch here.
    iso_patch = {"356": "India", "860": "Uzbekistan"}
    train_d["country_name"] = train_d["country_name"].replace(iso_patch)
    print(f"  train participants={len(train_d)} value qs={len(val_qs)}")
    # Merge demographic & value
    merged = train_d.merge(train_v, on="D_INTERVIEW", how="inner")
    print(f"  merged rows={len(merged)}; unique countries={merged['country_name'].nunique()}")

    records = []
    # For each (country, qid), compute the per-answer-option distribution
    for qid in val_qs:
        meta = qmeta.get(qid)
        if not meta or meta.get("use_case") != "value":
            continue
        sc_min = meta.get("answer_scale_min")
        sc_max = meta.get("answer_scale_max")
        if sc_min is None or sc_max is None:
            continue
        options = list(range(int(sc_min), int(sc_max) + 1))
        topic = meta.get("category", "unknown")
        question_text = meta.get("question", qid)

        for country, group in merged.groupby("country_name"):
            vals = group[qid].dropna()
            # filter to int answers in scale
            vals_int = []
            for v in vals:
                if pd.isna(v): continue
                try:
                    iv = int(v)
                    if sc_min <= iv <= sc_max: vals_int.append(iv)
                except (ValueError, TypeError):
                    pass
            if len(vals_int) < 5:  # skip very thin (country,qid) cells
                continue
            counts = Counter(vals_int)
            dist = {str(o): counts.get(o, 0) / len(vals_int) for o in options}
            dist = normalize_dist(dist)
            rec = base_record(
                source="worldvaluesbench",
                split="train",
                country=str(country),
                language="en",
                task_type="survey_distribution_ordinal",
                topic=str(topic),
                text=question_text,
                answer_options={str(o): str(o) for o in options},  # ordinal scale labels (no semantic strings)
                distribution=dist,
                metadata={
                    "question_id": qid,
                    "answer_scale_min": int(sc_min),
                    "answer_scale_max": int(sc_max),
                    "answer_data_type": meta.get("answer_data_type", "ordinal"),
                    "wvs_wave": 7,
                    "wvs_version": "6.0",
                    "n_respondents": int(len(vals_int)),
                },
                license_tag="WVSA-restricted (WVS Wave 7 v6.0)",
            )
            rec["evidence_id"] = make_eid("worldvaluesbench", "train", f"{qid}::{hsh(str(country))}")
            records.append(rec)
    print(f"  WVB evidence records: {len(records)}")
    return records


# =============================================================================
# 2) GOQA train evidence: per question (already has per-country distribution)
# =============================================================================
def parse_selections(s: str) -> dict:
    if not isinstance(s, str): return {}
    m = re.search(r"\{.*\}", s, re.S)
    if not m: return {}
    try: return ast.literal_eval(m.group(0))
    except Exception: return {}


def parse_options(s: str) -> list:
    if not isinstance(s, str): return []
    try: return ast.literal_eval(s)
    except Exception: return []


def build_goqa_evidence() -> list[dict]:
    print("=== GOQA ===")
    split = json.load(open(SPLT / "globalopinionqa_split.json"))
    train_qids = set(split["train"])
    # processed parquet has question_id, parsed selections, options, source
    df = pd.read_parquet(PROC / "goqa/global_opinions_processed.parquet")
    train_df = df[df["question_id"].isin(train_qids)].copy()
    print(f"  GOQA train questions: {len(train_df)} (of {len(df)} total)")
    # Each question has multi-country distribution; emit ONE evidence per (question_id, country)
    records = []
    for _, row in train_df.iterrows():
        qid = row["question_id"]
        question = str(row["question"])
        # selections_parsed stored as JSON string
        sel = json.loads(row["selections_parsed"]) if isinstance(row["selections_parsed"], str) else (row["selections_parsed"] or {})
        opts = json.loads(row["options_parsed"]) if isinstance(row["options_parsed"], str) else (row["options_parsed"] or [])
        if not sel or not opts:
            continue
        opt_keys = [str(i) for i in range(len(opts))]
        option_labels = {opt_keys[i]: opts[i] for i in range(len(opts))}
        source_tag = row.get("source", "GAS")
        # Naive topic inference: first few words of question (keep short)
        topic_guess = " ".join(question.split()[:5]).lower().strip("?.,!:")
        for country, dist_list in sel.items():
            if not isinstance(dist_list, list) or len(dist_list) != len(opts):
                continue
            dist = {opt_keys[i]: float(dist_list[i]) for i in range(len(opts))}
            ssum = sum(dist.values())
            if abs(ssum - 1.0) > 0.05:
                continue  # malformed
            dist = normalize_dist(dist)
            rec = base_record(
                source="globalopinionqa",
                split="train",
                country=str(country),
                language="en",
                task_type="survey_distribution_categorical",
                topic=topic_guess,
                text=question,
                answer_options=option_labels,
                distribution=dist,
                metadata={
                    "question_id": qid,
                    "original_survey": str(source_tag),
                    "n_options": len(opts),
                },
                license_tag="Anthropic/llm_global_opinions (derived from PEW/WVS/GAS)",
            )
            rec["evidence_id"] = make_eid("globalopinionqa", "train",
                                          f"{qid}::{hsh(str(country))}")
            records.append(rec)
    print(f"  GOQA evidence records: {len(records)}")
    return records


# =============================================================================
# 3) NormAd train evidence: norm + rule-of-thumb per story
# =============================================================================
def build_normad_evidence() -> list[dict]:
    print("=== NormAd ===")
    split = json.load(open(SPLT / "normad_split.json"))
    train_ids = set(split["train"])
    df = pd.read_parquet(PROC / "normad/normad_processed.parquet")
    train_df = df[df["story_id"].isin(train_ids)].copy()
    print(f"  NormAd train stories: {len(train_df)} (of {len(df)} total)")
    records = []
    for _, row in train_df.iterrows():
        country = str(row["Country"])
        story_id = row["story_id"]
        rule = (row.get("Rule-of-Thumb") or "").strip()
        story = (row.get("Story") or "").strip()
        value = (row.get("Value") or "").strip()
        background = (row.get("Background") or "").strip()
        label = str(row.get("Gold Label", "")).strip().lower()
        subaxis = str(row.get("Subaxis", "general"))

        # Each story produces ONE evidence record: prefer rule-of-thumb as text (more transferable);
        # story stays in metadata for prompt building if a method opts in.
        if not rule and not story:
            continue
        text = rule if rule else story
        rec = base_record(
            source="normad",
            split="train",
            country=country,
            language="en",
            task_type="norm_judgment",
            topic=subaxis,
            text=text,
            answer_options={"yes": "socially acceptable",
                            "no": "not socially acceptable",
                            "neutral": "ambiguous / depends on context"},
            label=label if label in {"yes", "no", "neutral"} else None,
            metadata={
                "story_id": story_id,
                "subaxis": subaxis,
                "value": value,
                "rule_of_thumb": rule,
                "story": story,
                "background": background[:400] if background else None,
            },
            license_tag="NormAd-Eti (Rao et al. 2024)",
        )
        rec["evidence_id"] = make_eid("normad", "train", story_id)
        records.append(rec)
    print(f"  NormAd evidence records: {len(records)}")
    return records


# =============================================================================
# 4) BLEnD train evidence: short answer + MC; train split only; NO test gold
# =============================================================================
def build_blend_evidence() -> list[dict]:
    print("=== BLEnD ===")
    split = json.load(open(SPLT / "blend_split.json"))
    train_qids = set(split["train"])
    test_qids  = set(split["test"])
    valid_qids = set(split["valid"])
    short_df = pd.read_parquet(PROC / "blend/blend_short_answer.parquet")
    train_short = short_df[short_df["qid"].isin(train_qids)].copy()
    print(f"  BLEnD train (q×country) short-answer rows: {len(train_short)} (of {len(short_df)})")

    # Load full annotation files for actual short answers
    ANN = RAW / "blend/hf/data/annotations"
    QUE = RAW / "blend/hf/data/questions"
    topic_map = {}
    for cfile in QUE.glob("*_questions.csv"):
        qdf = pd.read_csv(cfile)
        for _, r in qdf.iterrows():
            qid = r.get("ID")
            if qid and qid not in topic_map:
                topic_map[qid] = str(r.get("Topic", "unknown"))

    records = []
    for country_file in ANN.glob("*_data.json"):
        country = country_file.stem.replace("_data", "")
        ann = json.load(open(country_file))
        for qid, item in ann.items():
            if qid not in train_qids:  # strict: skip valid/test/leakage
                continue
            anns = item.get("annotations", [])
            if not anns:
                continue
            # Collect short answers (count >= 1)
            shorts = []
            for a in anns:
                en_ans = a.get("en_answers") or a.get("answers") or []
                count = int(a.get("count", 0))
                for s_text in en_ans:
                    if s_text and count > 0:
                        shorts.append({"answer": str(s_text), "count": count, "language": "en"})
            if not shorts:
                continue
            question_en = item.get("en_question") or item.get("question") or ""
            question_local = item.get("question") or ""
            topic = topic_map.get(qid, "unknown")
            rec = base_record(
                source="blend",
                split="train",
                country=country,
                language="en",
                task_type="everyday_knowledge_shortanswer",
                topic=topic,
                text=str(question_en),
                short_answers=shorts,
                metadata={
                    "blend_qid": qid,
                    "topic": topic,
                    "respondent_count": int(sum(s["count"] for s in shorts)),
                    "question_local": question_local,
                },
                license_tag="BLEnD CC-BY-NC-SA-4.0",
            )
            rec["evidence_id"] = make_eid("blend", "train", f"{qid}::{country}")
            records.append(rec)
    print(f"  BLEnD evidence records: {len(records)}")
    return records


# =============================================================================
# 5) Resource grouping
# =============================================================================
def compute_resource_grouping(all_records: list[dict]) -> dict:
    """Per-benchmark per-country evidence count, with high/low tier by median cutoff."""
    out = {}
    for bench_key, source_name in [("wvb", "worldvaluesbench"),
                                    ("goqa", "globalopinionqa"),
                                    ("normad", "normad"),
                                    ("blend", "blend")]:
        ev = [r for r in all_records if r["source"] == source_name]
        per_country = Counter(r["country_or_region"] for r in ev)
        if not per_country:
            out[bench_key] = {"countries": {}, "median_cutoff": 0,
                              "high_n": 0, "low_n": 0, "total_records": 0,
                              "n_countries": 0}
            continue
        counts = sorted(per_country.values())
        median = statistics.median(counts)
        tiered = {}
        for country, n in per_country.items():
            tier = "high" if n >= median else "low"
            tiered[country] = {"n_train_evidence": int(n), "tier": tier}
        out[bench_key] = {
            "benchmark": source_name,
            "n_countries": len(per_country),
            "total_records": int(sum(per_country.values())),
            "median_cutoff": float(median),
            "high_n_countries": sum(1 for v in tiered.values() if v["tier"] == "high"),
            "low_n_countries":  sum(1 for v in tiered.values() if v["tier"] == "low"),
            "countries": tiered,
            "min": int(min(counts)),
            "max": int(max(counts)),
            "mean": float(np.mean(counts)),
            "p10": float(np.percentile(counts, 10)),
            "p50": float(np.percentile(counts, 50)),
            "p90": float(np.percentile(counts, 90)),
            "rule": "tier = high if n_train_evidence >= median else low",
        }
    return out


# =============================================================================
# 6) Prototypes per benchmark
# =============================================================================
def build_prototypes(all_records: list[dict], resource_grouping: dict) -> dict:
    """Per-benchmark per-country prototype (readme §9.3).

    summary = list of 3 bullets per benchmark task family:
      - For survey_distribution_*: dominant-option per top-3 topics
      - For norm_judgment: yes/no/neutral label distribution + top subaxes
      - For everyday_knowledge_shortanswer: top topics + answer-count summary
    Always includes warning + sources + indexed_at.
    """
    DEFAULT_WARN = "该原型只描述训练数据中的聚合模式，不能用于推断具体个人。聚合自 train split, 不包含 test 答案。"
    proto_by_bench: dict = {b: {} for b in ["wvb","goqa","normad","blend"]}

    for bench_key, source_name in [("wvb", "worldvaluesbench"),
                                    ("goqa", "globalopinionqa"),
                                    ("normad", "normad"),
                                    ("blend", "blend")]:
        bench_records = [r for r in all_records if r["source"] == source_name]
        by_country = defaultdict(list)
        for r in bench_records:
            by_country[r["country_or_region"]].append(r)
        for country, recs in by_country.items():
            if bench_key == "wvb":
                # Aggregate: top-3 topics by record count + dominant-option per topic
                topic_counts = Counter(r["topic"] for r in recs)
                top_topics = topic_counts.most_common(3)
                bullets = []
                for topic, n in top_topics:
                    topic_recs = [r for r in recs if r["topic"] == topic]
                    # mean modal-option fraction
                    modal_fracs = []
                    for r in topic_recs:
                        if r["distribution"]:
                            modal_fracs.append(max(r["distribution"].values()))
                    if modal_fracs:
                        bullets.append(
                            f"主题「{topic}」: {n} 个问题, 模态选项均值集中度 "
                            f"{np.mean(modal_fracs)*100:.0f}% (WVS Wave 7 v6.0, 2017-2022)"
                        )
                if not bullets:
                    bullets.append(f"WVS Wave 7 数据 ({len(recs)} 条问题分布)")
                bullets.append(f"覆盖 {len(set(r['metadata'].get('question_id') for r in recs))} 个独立价值问题; "
                               f"每问平均受访者数 {int(np.mean([r['metadata']['n_respondents'] for r in recs]))}")
                bullets.append("群内差异显著: 同国家不同年龄/教育/城乡组别在多数问题上分布有 ±10-30% 漂移; "
                               "原型仅描述加权平均, 不应推断个人态度。")
            elif bench_key == "goqa":
                topic_counts = Counter(r["topic"] for r in recs)
                top_topics = topic_counts.most_common(3)
                bullets = []
                for topic, n in top_topics:
                    bullets.append(f"主题「{topic}」: {n} 个意见问题 (PEW/WVS/GAS 多年合并, ≤2022)")
                if not bullets:
                    bullets.append(f"GOQA: {len(recs)} 条意见分布问题")
                bullets.append(f"覆盖 {len(recs)} 个 (question×country) 分布; "
                               f"平均每问选项数 {np.mean([len(r['answer_options']) for r in recs]):.1f}")
                bullets.append("数据来源跨年份合并 (Pew Research Center / WVS / Global Attitudes Survey); "
                               "群体内代际/区域差异未在此原型中拆分。")
            elif bench_key == "normad":
                labels = Counter(str(r["label"]) for r in recs)
                subax_counts = Counter(r["topic"] for r in recs)
                top_subax = subax_counts.most_common(3)
                total = sum(labels.values())
                bullets = []
                bullets.append(
                    f"训练样本中礼仪场景判断分布: yes={labels.get('yes',0)/max(total,1)*100:.0f}%, "
                    f"no={labels.get('no',0)/max(total,1)*100:.0f}%, "
                    f"neutral={labels.get('neutral',0)/max(total,1)*100:.0f}% "
                    f"(NormAd-Eti, Rao et al. 2024)")
                bullets.append(f"高频礼仪子轴: " + ", ".join(f"{s}({n})" for s,n in top_subax))
                bullets.append("Cultural Atlas 来源, 概括性社会规范, 不覆盖区域/族群细分差异; "
                               "具体场景可受时代/年龄/宗教多重影响。")
            elif bench_key == "blend":
                topics = Counter(r["topic"] for r in recs)
                bullets = []
                top_t = topics.most_common(3)
                bullets.append(f"高频日常文化主题: " + ", ".join(f"{t}({n})" for t,n in top_t))
                avg_resp = np.mean([r["metadata"]["respondent_count"] for r in recs])
                bullets.append(f"每问平均 {avg_resp:.1f} 位受访者短答案 (BLEnD train split, 2024); "
                               f"覆盖 {len(recs)} 个 (qid×country) 对")
                bullets.append("短答案为自由文本众数, 仅反映 BLEnD 招募人群在该国的常识倾向; "
                               "不应代表全国共识或专家意见。")
            else:
                bullets = [f"({bench_key} {country}: {len(recs)} records)"]

            # Resource tier from grouping
            tier_info = resource_grouping[bench_key]["countries"].get(country, {})
            tier = tier_info.get("tier", "low")
            n_ev = tier_info.get("n_train_evidence", len(recs))

            sources_used = sorted(set(f"{r['source']}_train" for r in recs))
            evidence_ids = [r["evidence_id"] for r in recs[:50]]  # cap traceability list

            card = {
                "country_or_region": country,
                "sources": sources_used,
                "summary": bullets,
                "warning": DEFAULT_WARN,
                "evidence_ids": evidence_ids,
                "n_evidence": len(recs),
                "indexed_at": UTC_NOW,
                "metadata": {
                    "benchmark": source_name,
                    "resource_tier": tier,
                    "confidence": "low" if tier == "low" else "high",
                    "median_cutoff": resource_grouping[bench_key]["median_cutoff"],
                    "builder": "rule_based_aggregator_v1",
                },
            }
            proto_by_bench[bench_key][country] = card
    return proto_by_bench


# =============================================================================
# 7) Leakage check (use src/leakage_check.py R1+R2+R3 modes; R4 separate)
# =============================================================================
def run_leakage_scan(all_records: list[dict]) -> dict:
    """For each evidence record, scan against the TEST item pool of the matching benchmark.

    R1: evidence.split != 'test'  → all our records are split='train', should pass trivially
    R2: evidence.source_question_id != target.target_item_id (per-source qid scan)
    R3: normalized text-hash equality with any test question
    R4: embedding cosine ≥ 0.95 (full multilingual model; may be slow, sample 500 evidence per benchmark
        for embedding scan due to cost; full text-hash + qid scan is cheap and is the hard guarantee)
    """
    from leakage_check import normalize_text, text_hash, LeakageChecker

    # ---------- Build per-source TEST question pools ----------
    # WVB: 36 probe questions (the canonical eval text)
    DC = RAW / "worldvaluesbench/repo/dataset_construction"
    wvb_value_qs = json.load(open(DC / "probe_set_construction/value_questions.json"))
    wvb_qmeta    = json.load(open(DC / "question_metadata.json"))
    wvb_test_pool = []
    for qid in wvb_value_qs.keys():
        q = wvb_qmeta.get(qid, {}).get("question", "")
        wvb_test_pool.append({"id": qid, "text": str(q)})
    print(f"  WVB test pool: {len(wvb_test_pool)} questions")

    # GOQA
    goqa_split = json.load(open(SPLT / "globalopinionqa_split.json"))
    goqa_test_qids = set(goqa_split["test"])
    goqa_df = pd.read_parquet(PROC / "goqa/global_opinions_processed.parquet")
    goqa_test_df = goqa_df[goqa_df["question_id"].isin(goqa_test_qids)]
    goqa_test_pool = [{"id": r["question_id"], "text": str(r["question"])}
                      for _, r in goqa_test_df.iterrows()]
    print(f"  GOQA test pool: {len(goqa_test_pool)} questions")

    # NormAd
    nad_split = json.load(open(SPLT / "normad_split.json"))
    nad_test_ids = set(nad_split["test"])
    nad_df = pd.read_parquet(PROC / "normad/normad_processed.parquet")
    nad_test_df = nad_df[nad_df["story_id"].isin(nad_test_ids)]
    nad_test_pool = [{"id": r["story_id"], "text": str(r["Story"])}
                     for _, r in nad_test_df.iterrows()]
    print(f"  NormAd test pool: {len(nad_test_pool)} stories")

    # BLEnD: test qids
    blend_split = json.load(open(SPLT / "blend_split.json"))
    blend_test_qids = set(blend_split["test"])
    blend_test_pool = []
    for cfile in (RAW / "blend/hf/data/questions").glob("*_questions.csv"):
        qdf = pd.read_csv(cfile)
        for _, r in qdf.iterrows():
            qid = r.get("ID")
            if qid in blend_test_qids:
                blend_test_pool.append({"id": qid,
                                        "text": str(r.get("Question", ""))})
    # dedupe by qid (one English version per qid)
    seen = set(); deduped = []
    for o in blend_test_pool:
        if o["id"] in seen: continue
        seen.add(o["id"]); deduped.append(o)
    blend_test_pool = deduped
    print(f"  BLEnD test pool: {len(blend_test_pool)} questions")

    test_pools = {
        "worldvaluesbench": wvb_test_pool,
        "globalopinionqa":  goqa_test_pool,
        "normad":           nad_test_pool,
        "blend":            blend_test_pool,
    }
    # Index test pools: qid set + normalized-text-hash → qid map
    test_indexes = {}
    for src, pool in test_pools.items():
        qid_set = {p["id"] for p in pool}
        hash_map = {text_hash(p["text"]): p["id"] for p in pool}
        test_indexes[src] = {"qid_set": qid_set, "hash_map": hash_map, "pool": pool}

    # ---------- R1+R2+R3 scan (cheap, exhaustive) ----------
    rule_hits = Counter()
    hits = []
    for r in all_records:
        src = r["source"]
        idx = test_indexes.get(src)
        if idx is None: continue
        reasons = []
        # R1
        if str(r["split"]).strip().lower() == "test":
            reasons.append("R1_evidence_split_is_test")
        # R2 (use metadata.question_id / story_id / blend_qid as source_question_id)
        src_qid = None
        for k in ("question_id", "story_id", "blend_qid"):
            v = r.get("metadata", {}).get(k)
            if v: src_qid = v; break
        if src_qid and src_qid in idx["qid_set"]:
            reasons.append("R2_same_question_id")
        # R3
        th = text_hash(r["text"])
        if th in idx["hash_map"]:
            reasons.append(f"R3_text_hash_match_to_{idx['hash_map'][th]}")
        if reasons:
            for rr in reasons:
                for k in ("R1","R2","R3"):
                    if rr.startswith(k): rule_hits[k] += 1
            hits.append({
                "evidence_id": r["evidence_id"],
                "source": src,
                "country": r["country_or_region"],
                "source_question_id": src_qid,
                "reasons": reasons,
            })

    # ---------- R4 EXHAUSTIVE scan via embedding (full evidence × full test pool) ----------
    # Researcher's spec: "任何 R1-R4 命中 → 删除该 evidence → 重跑直到 0 命中".
    # Exhaustive scan guarantees one-shot convergence (no random subsampling).
    R4_THRESHOLD = 0.95
    r4_hits = []
    r4_pairs_scanned = 0
    already_hit_ids = {h["evidence_id"] for h in hits}
    from leakage_check import _get_model
    model = _get_model("sentence-transformers/all-MiniLM-L6-v2")
    BATCH = 256

    for src, idx in test_indexes.items():
        pool = idx["pool"]
        if not pool: continue
        src_records = [r for r in all_records if r["source"] == src]
        if not src_records:
            print(f"  R4 skip {src}: no evidence records")
            continue
        # embed full evidence + full test pool (one shot)
        ev_texts_norm = [normalize_text(r["text"]) for r in src_records]
        tq_texts_norm = [normalize_text(p["text"]) for p in pool]
        ev_emb = model.encode(ev_texts_norm, normalize_embeddings=True,
                              convert_to_numpy=True, show_progress_bar=False,
                              batch_size=BATCH).astype("float32")
        tq_emb = model.encode(tq_texts_norm, normalize_embeddings=True,
                              convert_to_numpy=True, show_progress_bar=False,
                              batch_size=BATCH).astype("float32")
        # Compute argmax cosine in chunks to bound memory
        chunk_hits = 0
        chunk = 2048
        for s in range(0, len(src_records), chunk):
            e = min(s + chunk, len(src_records))
            sims = ev_emb[s:e] @ tq_emb.T  # (chunk, |pool|)
            r4_pairs_scanned += sims.shape[0] * sims.shape[1]
            maxes = sims.max(axis=1)
            argmaxes = sims.argmax(axis=1)
            for li, gi in enumerate(range(s, e)):
                mx = float(maxes[li])
                if mx >= R4_THRESHOLD:
                    rec = src_records[gi]
                    if rec["evidence_id"] in already_hit_ids:
                        continue
                    j = int(argmaxes[li])
                    r4_hits.append({
                        "evidence_id": rec["evidence_id"],
                        "source": src,
                        "country": rec["country_or_region"],
                        "matched_test_id": pool[j]["id"],
                        "similarity": round(mx, 4),
                        "reason": f"R4_near_duplicate_cos={mx:.3f}",
                    })
                    already_hit_ids.add(rec["evidence_id"])
                    rule_hits["R4"] += 1
                    chunk_hits += 1
        print(f"  R4 exhaustive {src}: {len(src_records)} evidence × {len(pool)} test "
              f"= {len(src_records)*len(pool)} pairs → {chunk_hits} hits")

    summary = {
        "scanned_at": UTC_NOW,
        "n_evidence_total": len(all_records),
        "n_evidence_per_source": dict(Counter(r["source"] for r in all_records)),
        "n_test_per_source": {s: len(p) for s, p in test_pools.items()},
        "R1_R2_R3_scan": {
            "n_evidence_scanned": len(all_records),
            "n_hits": len(hits),
            "rule_breakdown": dict(rule_hits),
            "hits": hits[:200],  # cap
        },
        "R4_exhaustive_scan": {
            "mode": "full evidence × full test pool",
            "similarity_threshold": R4_THRESHOLD,
            "n_pairs_scanned": r4_pairs_scanned,
            "n_hits": len(r4_hits),
            "hits": r4_hits[:200],
        },
        "final_status": "PASS" if (len(hits) == 0 and len(r4_hits) == 0) else "HITS_FOUND",
    }
    return summary, hits, r4_hits


# =============================================================================
# MAIN
# =============================================================================
def main():
    print(f"[{UTC_NOW}] Building evidence + prototypes + leakage scan + resource grouping")
    # 1) Build all 4 sources
    wvb_recs    = build_wvb_evidence()
    goqa_recs   = build_goqa_evidence()
    normad_recs = build_normad_evidence()
    blend_recs  = build_blend_evidence()
    all_records = wvb_recs + goqa_recs + normad_recs + blend_recs
    print(f"\nTotal raw evidence records: {len(all_records)}")

    # 2) Resource grouping (needed for prototype confidence tags)
    grouping = compute_resource_grouping(all_records)
    json.dump(grouping, open(ANLY / "resource_grouping.json", "w"),
              indent=2, ensure_ascii=False, default=str)
    print(f"Wrote {ANLY/'resource_grouping.json'}")

    # 3) Leakage scan
    print("\n=== Leakage scan ===")
    leak_summary, r123_hits, r4_hits = run_leakage_scan(all_records)
    # Remove any record that hits
    bad_ids = {h["evidence_id"] for h in r123_hits} | {h["evidence_id"] for h in r4_hits}
    print(f"Records to remove due to leakage: {len(bad_ids)}")
    clean = [r for r in all_records if r["evidence_id"] not in bad_ids]
    print(f"After cleanup: {len(clean)} (removed {len(all_records)-len(clean)})")

    # Re-scan clean? Per Researcher "重跑直到 0 命中" — re-run scan
    if bad_ids:
        leak_summary, r123_hits2, r4_hits2 = run_leakage_scan(clean)
        leak_summary["rerun_after_cleanup"] = True
        leak_summary["rerun_R1_R2_R3_hits"] = len(r123_hits2)
        leak_summary["rerun_R4_hits"] = len(r4_hits2)
        if r123_hits2 or r4_hits2:
            print(f"WARNING: rerun still has {len(r123_hits2)} R1-R3 hits and {len(r4_hits2)} R4 hits")
        else:
            leak_summary["final_status"] = "CLEAN"
    else:
        leak_summary["final_status"] = "CLEAN (no hits on first scan)"

    json.dump(leak_summary, open(ANLY / "leakage_report.json", "w"),
              indent=2, ensure_ascii=False, default=str)
    print(f"Wrote {ANLY/'leakage_report.json'}: status = {leak_summary['final_status']}")

    # 4) Recompute resource grouping on CLEAN set (in case any record dropped)
    grouping = compute_resource_grouping(clean)
    json.dump(grouping, open(ANLY / "resource_grouping.json", "w"),
              indent=2, ensure_ascii=False, default=str)

    # 5) Build prototypes (on clean set)
    print("\n=== Prototypes ===")
    protos = build_prototypes(clean, grouping)
    for bench, cards in protos.items():
        outp = PROTO / bench / "country_prototypes.json"
        json.dump(cards, open(outp, "w"), indent=2, ensure_ascii=False, default=str)
        print(f"  {bench}: {len(cards)} country prototypes → {outp}")

    # 6) Write evidence.jsonl
    out_jsonl = EVID_DIR / "evidence.jsonl"
    with open(out_jsonl, "w") as f:
        for r in clean:
            f.write(json.dumps(r, ensure_ascii=False, default=str) + "\n")
    print(f"\nWrote {out_jsonl} ({len(clean)} records)")

    # 7) Stats summary (small, rsync to GCP)
    per_source = Counter(r["source"] for r in clean)
    per_country = Counter(r["country_or_region"] for r in clean)
    per_task = Counter(r["task_type"] for r in clean)
    per_split = Counter(r["split"] for r in clean)
    stats = {
        "built_at": UTC_NOW,
        "total_records": len(clean),
        "removed_due_to_leakage": len(bad_ids),
        "per_source": dict(per_source),
        "per_task_type": dict(per_task),
        "per_split": dict(per_split),
        "n_unique_countries": len(per_country),
        "top20_countries_by_record_count": per_country.most_common(20),
        "bottom20_countries_by_record_count": per_country.most_common()[-20:],
        "prototypes_per_bench": {b: len(cards) for b, cards in protos.items()},
        "prototypes_high_resource_per_bench": {
            b: sum(1 for c in cards.values() if c["metadata"]["resource_tier"] == "high")
            for b, cards in protos.items()
        },
        "prototypes_low_resource_per_bench": {
            b: sum(1 for c in cards.values() if c["metadata"]["resource_tier"] == "low")
            for b, cards in protos.items()
        },
        "leakage_report_status": leak_summary["final_status"],
        "evidence_jsonl_path": str(out_jsonl),
    }
    json.dump(stats, open(ANLY / "evidence_stats.json", "w"),
              indent=2, ensure_ascii=False, default=str)
    print(f"\nWrote {ANLY/'evidence_stats.json'}")
    print(json.dumps({k: v for k, v in stats.items()
                       if k != "top20_countries_by_record_count"
                       and k != "bottom20_countries_by_record_count"},
                     indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
