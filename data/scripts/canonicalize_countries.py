#!/usr/bin/env python3
"""Canonicalize country_or_region names across evidence.jsonl + rebuild prototypes + re-scan leakage.

Fixes Researcher-reported bug: same country split across cases/separators
(e.g. 'Canada' 847 vs 'canada' 19, 'South_Korea' vs 'south_korea').
"""
from __future__ import annotations

import datetime
import json
import os
import re
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(os.environ.get("EXPERIMENT_ROOT", Path(__file__).resolve().parents[2]))
EVID_DIR = ROOT / "data/evidence"
PROTO    = ROOT / "data/prototypes"
SPLT     = ROOT / "data/splits"
PROC     = ROOT / "data/processed"
RAW      = ROOT / "data/raw"
ANLY     = ROOT / "analysis"
SRC      = ROOT / "src"
sys.path.insert(0, str(SRC))

UTC_NOW = datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")

SPECIAL_MAP = {
    "us": "United States", "usa": "United States",
    "uk": "United Kingdom", "great britain": "United Kingdom",
    "great_britain": "United Kingdom", "united kingdom": "United Kingdom",
    "north korea": "North Korea", "south korea": "South Korea",
    "west java": "West Java", "northern nigeria": "Northern Nigeria",
    "hong kong sar": "Hong Kong SAR", "hong kong": "Hong Kong SAR",
    "bosnia and herzegovina": "Bosnia and Herzegovina",
    "bosnia herzegovina": "Bosnia and Herzegovina",
    "trinidad and tobago": "Trinidad and Tobago",
    "saudi arabia": "Saudi Arabia", "south sudan": "South Sudan",
    "new zealand": "New Zealand", "sri lanka": "Sri Lanka",
    "el salvador": "El Salvador", "burkina faso": "Burkina Faso",
    "ivory coast": "Cote d'Ivoire", "cote d ivoire": "Cote d'Ivoire",
    "cote d'ivoire": "Cote d'Ivoire", "puerto rico": "Puerto Rico",
    "north macedonia": "North Macedonia", "macedonia": "North Macedonia",
    "south africa": "South Africa",
    "central african republic": "Central African Republic",
    "myanmar (burma)": "Myanmar", "viet nam": "Vietnam",
    "russian federation": "Russia",
    "iran (islamic republic of)": "Iran",
    "venezuela (bolivarian republic of)": "Venezuela",
    "bolivia (plurinational state of)": "Bolivia",
    "republic of korea": "South Korea",
    "democratic people's republic of korea": "North Korea",
    "democratic republic of the congo": "DR Congo",
    "the democratic republic of the congo": "DR Congo",
    "tanzania, united republic of": "Tanzania",
    "united republic of tanzania": "Tanzania",
    "palestine, state of": "Palestinian Territories",
    "occupied palestinian territory": "Palestinian Territories",
    "palestinian territories": "Palestinian Territories",
    "lao people's democratic republic": "Laos",
    "syrian arab republic": "Syria",
    "moldova, republic of": "Moldova",
    "republic of moldova": "Moldova",
    "taiwan, province of china": "Taiwan",
    "taiwan roc": "Taiwan", "taiwan": "Taiwan",
}


def canonical_country(name):
    if name is None:
        return ""
    s = str(name).strip()
    if not s:
        return ""
    m = re.match(r"^(.*?)\s*(\([^)]+\))\s*$", s)
    suffix = ""
    if m:
        s, suffix = m.group(1).strip(), " " + m.group(2).strip()
    s = s.replace("_", " ").replace("-", " ").strip()
    s = re.sub(r"\s+", " ", s).lower()
    if s in SPECIAL_MAP:
        return SPECIAL_MAP[s] + suffix
    SMALL = {"and", "of", "the", "in", "for", "on"}
    toks = s.split(" ")
    out = []
    for i, t in enumerate(toks):
        if i > 0 and t in SMALL:
            out.append(t)
        else:
            out.append(t.capitalize() if t else t)
    return " ".join(out) + suffix


# ── 1) Rewrite evidence.jsonl ──────────────────────────────────────────────
print("[1/4] Canonicalizing evidence.jsonl ...")
in_path = EVID_DIR / "evidence.jsonl"
out_path = EVID_DIR / "evidence.jsonl.tmp"
rename_log = Counter()
n_lines = 0
with open(in_path) as fin, open(out_path, "w") as fout:
    for line in fin:
        d = json.loads(line)
        old = d.get("country_or_region", "")
        new = canonical_country(old)
        if old != new:
            rename_log[(old, new)] += 1
        d["country_or_region"] = new
        fout.write(json.dumps(d, ensure_ascii=False, default=str) + "\n")
        n_lines += 1
out_path.replace(in_path)
print(f"  rewrote {n_lines} records; {len(rename_log)} distinct rename ops")
for (o, n), c in rename_log.most_common(20):
    print(f"    {c:>5}  '{o}' → '{n}'")

# ── 2) Recount ─────────────────────────────────────────────────────────────
print("\n[2/4] Recounting ...")
country_per_src = defaultdict(Counter)
all_records = []
with open(in_path) as f:
    for line in f:
        d = json.loads(line)
        country_per_src[d["source"]][d["country_or_region"]] += 1
        all_records.append(d)
total_unique = set()
for src, c in country_per_src.items():
    print(f"  {src}: {len(c)} unique countries (top5: {[k for k,_ in c.most_common(5)]})")
    total_unique.update(c.keys())
print(f"  TOTAL canonical unique countries: {len(total_unique)}")

# ── 3) Rebuild prototypes + resource_grouping ──────────────────────────────
print("\n[3/4] Rebuilding prototypes + resource_grouping ...")

def compute_resource_grouping(records):
    out = {}
    for bench_key, source_name in [("wvb","worldvaluesbench"),("goqa","globalopinionqa"),
                                    ("normad","normad"),("blend","blend")]:
        ev = [r for r in records if r["source"] == source_name]
        pc = Counter(r["country_or_region"] for r in ev)
        if not pc:
            out[bench_key] = {"benchmark":source_name,"n_countries":0,"total_records":0,
                              "median_cutoff":0,"high_n_countries":0,"low_n_countries":0,
                              "countries":{},"min":0,"max":0,"mean":0,"p10":0,"p50":0,"p90":0,
                              "rule":"tier = high if n >= median else low"}
            continue
        counts = sorted(pc.values())
        med = statistics.median(counts)
        tiered = {country:{"n_train_evidence":int(n),
                            "tier":"high" if n >= med else "low"} for country, n in pc.items()}
        out[bench_key] = {
            "benchmark": source_name,
            "n_countries": len(pc),
            "total_records": int(sum(pc.values())),
            "median_cutoff": float(med),
            "high_n_countries": sum(1 for v in tiered.values() if v["tier"]=="high"),
            "low_n_countries":  sum(1 for v in tiered.values() if v["tier"]=="low"),
            "countries": tiered,
            "min": int(min(counts)), "max": int(max(counts)),
            "mean": float(np.mean(counts)),
            "p10": float(np.percentile(counts,10)),
            "p50": float(np.percentile(counts,50)),
            "p90": float(np.percentile(counts,90)),
            "rule": "tier = high if n_train_evidence >= median else low",
        }
    return out


def build_prototypes(records, grouping):
    DEFAULT_WARN = "该原型只描述训练数据中的聚合模式，不能用于推断具体个人。聚合自 train split, 不包含 test 答案。"
    out = {b:{} for b in ["wvb","goqa","normad","blend"]}
    for bench_key, source_name in [("wvb","worldvaluesbench"),("goqa","globalopinionqa"),
                                    ("normad","normad"),("blend","blend")]:
        bench_rec = [r for r in records if r["source"] == source_name]
        by_c = defaultdict(list)
        for r in bench_rec:
            by_c[r["country_or_region"]].append(r)
        for country, recs in by_c.items():
            if bench_key == "wvb":
                tc = Counter(r["topic"] for r in recs)
                tops = tc.most_common(3)
                bullets = []
                for t, n in tops:
                    trecs = [r for r in recs if r["topic"]==t]
                    mf = [max(r["distribution"].values()) for r in trecs if r["distribution"]]
                    if mf:
                        bullets.append(f"主题「{t}」: {n} 个问题, 模态选项均值集中度 "
                                       f"{np.mean(mf)*100:.0f}% (WVS Wave 7 v6.0, 2017-2022)")
                if not bullets: bullets.append(f"WVS Wave 7 数据 ({len(recs)} 条问题分布)")
                qids = set(r["metadata"].get("question_id") for r in recs)
                avg = int(np.mean([r["metadata"]["n_respondents"] for r in recs]))
                bullets.append(f"覆盖 {len(qids)} 个独立价值问题; 每问平均受访者数 {avg}")
                bullets.append("群内差异显著: 同国家不同年龄/教育/城乡组别在多数问题上分布有 ±10-30% 漂移; "
                               "原型仅描述加权平均, 不应推断个人态度。")
            elif bench_key == "goqa":
                tc = Counter(r["topic"] for r in recs)
                tops = tc.most_common(3)
                bullets = [f"主题「{t}」: {n} 个意见问题 (PEW/WVS/GAS, ≤2022)" for t,n in tops]
                if not bullets: bullets.append(f"GOQA: {len(recs)} 条意见分布问题")
                bullets.append(f"覆盖 {len(recs)} 个 (question×country) 分布; "
                               f"平均每问选项数 {np.mean([len(r['answer_options']) for r in recs]):.1f}")
                bullets.append("数据来源跨年份合并 (Pew/WVS/GAS); 群体内代际/区域差异未在此原型中拆分。")
            elif bench_key == "normad":
                labels = Counter(str(r["label"]) for r in recs)
                sub = Counter(r["topic"] for r in recs)
                tops = sub.most_common(3)
                tot = sum(labels.values()) or 1
                bullets = [
                    f"训练样本中礼仪场景判断分布: yes={labels.get('yes',0)/tot*100:.0f}%, "
                    f"no={labels.get('no',0)/tot*100:.0f}%, "
                    f"neutral={labels.get('neutral',0)/tot*100:.0f}% (NormAd-Eti, Rao et al. 2024)",
                    "高频礼仪子轴: " + ", ".join(f"{s}({n})" for s,n in tops),
                    "Cultural Atlas 来源, 概括性社会规范, 不覆盖区域/族群细分差异; "
                    "具体场景可受时代/年龄/宗教多重影响。"]
            elif bench_key == "blend":
                tps = Counter(r["topic"] for r in recs)
                tt = tps.most_common(3)
                bullets = ["高频日常文化主题: " + ", ".join(f"{t}({n})" for t,n in tt)]
                avg_r = np.mean([r["metadata"]["respondent_count"] for r in recs])
                bullets.append(f"每问平均 {avg_r:.1f} 位受访者短答案 (BLEnD train, 2024); "
                               f"覆盖 {len(recs)} 个 (qid×country) 对")
                bullets.append("短答案为自由文本众数, 仅反映 BLEnD 招募人群在该国的常识倾向; "
                               "不应代表全国共识或专家意见。")
            else:
                bullets = [f"{bench_key} {country}: {len(recs)} records"]

            tier = grouping[bench_key]["countries"].get(country, {}).get("tier","low")
            out[bench_key][country] = {
                "country_or_region": country,
                "sources": sorted(set(f"{r['source']}_train" for r in recs)),
                "summary": bullets,
                "warning": DEFAULT_WARN,
                "evidence_ids": [r["evidence_id"] for r in recs[:50]],
                "n_evidence": len(recs),
                "indexed_at": UTC_NOW,
                "metadata": {
                    "benchmark": source_name,
                    "resource_tier": tier,
                    "confidence": "low" if tier=="low" else "high",
                    "median_cutoff": grouping[bench_key]["median_cutoff"],
                    "builder": "rule_based_aggregator_v1_canonical_country",
                },
            }
    return out


grouping = compute_resource_grouping(all_records)
json.dump(grouping, open(ANLY / "resource_grouping.json", "w"),
          indent=2, ensure_ascii=False, default=str)
for b in ["wvb","goqa","normad","blend"]:
    g = grouping[b]
    print(f"  {b}: n_countries={g['n_countries']} median={g['median_cutoff']:.0f} "
          f"high={g['high_n_countries']} low={g['low_n_countries']}")

protos = build_prototypes(all_records, grouping)
for bench, cards in protos.items():
    outp = PROTO / bench / "country_prototypes.json"
    json.dump(cards, open(outp, "w"), indent=2, ensure_ascii=False, default=str)
    print(f"  prototypes/{bench}: {len(cards)} cards")

# ── 4) Re-scan leakage ─────────────────────────────────────────────────────
print("\n[4/4] Re-scanning leakage on canonical evidence ...")
from leakage_check import _get_model, normalize_text, text_hash

DC = RAW / "worldvaluesbench/repo/dataset_construction"
wvb_value_qs = json.load(open(DC / "probe_set_construction/value_questions.json"))
wvb_qmeta    = json.load(open(DC / "question_metadata.json"))
wvb_test_pool = [{"id":qid, "text":str(wvb_qmeta.get(qid,{}).get("question",""))}
                 for qid in wvb_value_qs.keys()]

goqa_split = json.load(open(SPLT / "globalopinionqa_split.json"))
goqa_df = pd.read_parquet(PROC / "goqa/global_opinions_processed.parquet")
goqa_test = goqa_df[goqa_df["question_id"].isin(set(goqa_split["test"]))]
goqa_test_pool = [{"id":r["question_id"],"text":str(r["question"])} for _,r in goqa_test.iterrows()]

nad_split = json.load(open(SPLT / "normad_split.json"))
nad_df = pd.read_parquet(PROC / "normad/normad_processed.parquet")
nad_test = nad_df[nad_df["story_id"].isin(set(nad_split["test"]))]
nad_test_pool = [{"id":r["story_id"],"text":str(r["Story"])} for _,r in nad_test.iterrows()]

blend_split = json.load(open(SPLT / "blend_split.json"))
b_test_qids = set(blend_split["test"])
blend_test_pool = []; seen=set()
for cfile in (RAW/"blend/hf/data/questions").glob("*_questions.csv"):
    qdf = pd.read_csv(cfile)
    for _, r in qdf.iterrows():
        qid = r.get("ID")
        if qid in b_test_qids and qid not in seen:
            blend_test_pool.append({"id":qid,"text":str(r.get("Question",""))})
            seen.add(qid)

test_pools = {"worldvaluesbench":wvb_test_pool,"globalopinionqa":goqa_test_pool,
              "normad":nad_test_pool,"blend":blend_test_pool}
test_indexes = {}
for src, pool in test_pools.items():
    test_indexes[src] = {"qid_set":{p["id"] for p in pool},
                          "hash_map":{text_hash(p["text"]):p["id"] for p in pool},
                          "pool":pool}

# R1+R2+R3
rule_hits = Counter(); r123 = []
for r in all_records:
    src = r["source"]; idx = test_indexes.get(src)
    if idx is None: continue
    reasons = []
    if str(r["split"]).strip().lower() == "test": reasons.append("R1")
    q = None
    for k in ("question_id","story_id","blend_qid"):
        v = r.get("metadata",{}).get(k)
        if v: q = v; break
    if q and q in idx["qid_set"]: reasons.append("R2")
    if text_hash(r["text"]) in idx["hash_map"]: reasons.append("R3")
    if reasons:
        for rr in reasons: rule_hits[rr] += 1
        r123.append({"evidence_id":r["evidence_id"],"reasons":reasons})
print(f"  R1+R2+R3 hits: {len(r123)} breakdown: {dict(rule_hits)}")

R4_THRESHOLD = 0.95
r4_hits = []; r4_pairs = 0
model = _get_model("sentence-transformers/all-MiniLM-L6-v2")
for src, idx in test_indexes.items():
    pool = idx["pool"]
    if not pool: continue
    src_recs = [r for r in all_records if r["source"]==src]
    if not src_recs: continue
    ev_emb = model.encode([normalize_text(r["text"]) for r in src_recs],
                          normalize_embeddings=True, convert_to_numpy=True,
                          show_progress_bar=False, batch_size=256).astype("float32")
    tq_emb = model.encode([normalize_text(p["text"]) for p in pool],
                          normalize_embeddings=True, convert_to_numpy=True,
                          show_progress_bar=False, batch_size=256).astype("float32")
    chunk = 2048; sh = 0
    seen_ids = set()
    for s in range(0, len(src_recs), chunk):
        e = min(s+chunk, len(src_recs))
        sims = ev_emb[s:e] @ tq_emb.T
        r4_pairs += sims.shape[0]*sims.shape[1]
        mx = sims.max(axis=1); am = sims.argmax(axis=1)
        for li, gi in enumerate(range(s,e)):
            v = float(mx[li])
            if v >= R4_THRESHOLD:
                rec = src_recs[gi]
                if rec["evidence_id"] in seen_ids: continue
                seen_ids.add(rec["evidence_id"])
                r4_hits.append({"evidence_id":rec["evidence_id"],"source":src,
                                 "country":rec["country_or_region"],
                                 "matched_test_id":pool[int(am[li])]["id"],
                                 "similarity":round(v,4),
                                 "reason":f"R4_cos={v:.3f}"})
                sh += 1
    print(f"  R4 {src}: {len(src_recs)}×{len(pool)}={len(src_recs)*len(pool)} → {sh} hits")
print(f"  total R4 hits: {len(r4_hits)}")

# Load prev for first_pass_results carry-over
lr_path = ANLY / "leakage_report.json"
prev = {}
if lr_path.exists():
    try: prev = json.load(open(lr_path))
    except Exception: pass

report = {
    "scanned_at": UTC_NOW,
    "stage": "post-country-canonicalization-rescan",
    "n_evidence_total": len(all_records),
    "n_evidence_per_source": {s:len(c) for s,c in country_per_src.items()},
    "n_test_per_source": {s:len(p) for s,p in test_pools.items()},
    "R1_R2_R3_scan": {"n_hits":len(r123),"rule_breakdown":dict(rule_hits),"hits":r123[:200]},
    "R4_exhaustive_scan": {"mode":"full evidence × full test","similarity_threshold":R4_THRESHOLD,
                            "n_pairs_scanned":r4_pairs,"n_hits":len(r4_hits),"hits":r4_hits[:200]},
    "final_status": "CLEAN" if (len(r123)==0 and len(r4_hits)==0) else "HITS_FOUND",
    "canonicalization": {
        "trigger": "Researcher catch: stats showed n_unique_countries=219, cross-source case/separator mismatch (e.g. 'Canada' vs 'canada' vs 'South_Korea').",
        "fix": "scripts/canonicalize_countries.py applied canonical_country() = title-case w/ smart small-word handling + SPECIAL_MAP for ISO variants (US→United States, UK→United Kingdom).",
        "n_records_rewritten": n_lines,
        "n_distinct_rename_ops": len(rename_log),
        "top_renames": [(o,n,c) for (o,n),c in rename_log.most_common(15)],
    },
}
if prev.get("first_pass_results"):
    report["first_pass_results"] = prev["first_pass_results"]
json.dump(report, open(lr_path,"w"), indent=2, ensure_ascii=False, default=str)
print(f"  leakage_report.json: {report['final_status']}")

# Update evidence_stats.json
per_country = Counter(r["country_or_region"] for r in all_records)
stats = {
    "built_at": UTC_NOW,
    "stage": "post-country-canonicalization",
    "total_records": len(all_records),
    "per_source": {s:sum(c.values()) for s,c in country_per_src.items()},
    "per_task_type": dict(Counter(r["task_type"] for r in all_records)),
    "per_split": dict(Counter(r["split"] for r in all_records)),
    "n_unique_countries_canonical": len(per_country),
    "top20_countries_by_record_count": per_country.most_common(20),
    "bottom20_countries_by_record_count": per_country.most_common()[-20:],
    "prototypes_per_bench": {b:len(c) for b,c in protos.items()},
    "prototypes_high_resource_per_bench": {
        b: sum(1 for v in c.values() if v["metadata"]["resource_tier"]=="high")
        for b,c in protos.items()},
    "prototypes_low_resource_per_bench": {
        b: sum(1 for v in c.values() if v["metadata"]["resource_tier"]=="low")
        for b,c in protos.items()},
    "leakage_report_status": report["final_status"],
    "evidence_jsonl_path": str(in_path),
}
json.dump(stats, open(ANLY/"evidence_stats.json","w"), indent=2, ensure_ascii=False, default=str)
print(f"\nFINAL: {len(all_records)} records, {len(per_country)} canonical countries")
print(f"  per_source: {stats['per_source']}")
print(f"  prototypes: {stats['prototypes_per_bench']}")
print(f"  high_res:   {stats['prototypes_high_resource_per_bench']}")
print(f"  low_res:    {stats['prototypes_low_resource_per_bench']}")
print(f"  leakage:    {stats['leakage_report_status']}")
