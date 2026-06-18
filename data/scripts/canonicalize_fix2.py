#!/usr/bin/env python3
"""Second-pass canonicalization fix:
- 'Macao Sar' / 'Macau Sar' → 'Macao SAR'
- 'S. Korea' → 'South Korea'
- 'S. Africa' → 'South Africa'
- 'Czech Rep.' → 'Czech Republic'
- 'Palest. Ter.' → 'Palestinian Territories'

Also re-run prototype + resource_grouping rebuild on the updated names.
No need to re-scan leakage (we are only consolidating equivalent names; no new evidence).
"""
import json
import datetime
import statistics
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np

ROOT = Path("${EXPERIMENT_ROOT}")
EVID = ROOT / "data/evidence/evidence.jsonl"
PROTO = ROOT / "data/prototypes"
ANLY = ROOT / "analysis"
UTC_NOW = datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")

FIXES = {
    "Macao Sar": "Macao SAR",
    "Macau Sar": "Macao SAR",
    "S. Korea": "South Korea",
    "S. Africa": "South Africa",
    "S. Africa (Non-national sample)": "South Africa (Non-national sample)",
    "Czech Rep.": "Czech Republic",
    "Palest. Ter.": "Palestinian Territories",
}

rename_log = Counter()
tmp = EVID.with_suffix(".jsonl.tmp")
n = 0
with open(EVID) as fin, open(tmp, "w") as fout:
    for line in fin:
        d = json.loads(line)
        c = d.get("country_or_region", "")
        if c in FIXES:
            rename_log[(c, FIXES[c])] += 1
            d["country_or_region"] = FIXES[c]
        fout.write(json.dumps(d, ensure_ascii=False, default=str) + "\n")
        n += 1
tmp.replace(EVID)
print(f"rewrote {n} lines; {len(rename_log)} distinct rename ops")
for (o, nn), c in rename_log.most_common():
    print(f"  {c:>5}  {o!r} → {nn!r}")

# Recount
all_recs = []
with open(EVID) as f:
    for line in f:
        all_recs.append(json.loads(line))

print(f"\nUnique canonical countries: {len({r['country_or_region'] for r in all_recs})}")

# Rebuild resource_grouping
def compute_grouping(records):
    out = {}
    for bench_key, source_name in [("wvb","worldvaluesbench"),("goqa","globalopinionqa"),
                                    ("normad","normad"),("blend","blend")]:
        ev = [r for r in records if r["source"] == source_name]
        pc = Counter(r["country_or_region"] for r in ev)
        if not pc: continue
        counts = sorted(pc.values())
        med = statistics.median(counts)
        tiered = {c:{"n_train_evidence":int(n),"tier":"high" if n>=med else "low"} for c,n in pc.items()}
        out[bench_key] = {
            "benchmark":source_name,"n_countries":len(pc),
            "total_records":int(sum(pc.values())),"median_cutoff":float(med),
            "high_n_countries":sum(1 for v in tiered.values() if v["tier"]=="high"),
            "low_n_countries":sum(1 for v in tiered.values() if v["tier"]=="low"),
            "countries":tiered,"min":int(min(counts)),"max":int(max(counts)),
            "mean":float(np.mean(counts)),
            "p10":float(np.percentile(counts,10)),
            "p50":float(np.percentile(counts,50)),
            "p90":float(np.percentile(counts,90)),
            "rule":"tier = high if n_train_evidence >= median else low",
        }
    return out

grouping = compute_grouping(all_recs)
json.dump(grouping, open(ANLY/"resource_grouping.json","w"), indent=2, ensure_ascii=False, default=str)

# Rebuild prototypes
DEFAULT_WARN = "该原型只描述训练数据中的聚合模式，不能用于推断具体个人。聚合自 train split, 不包含 test 答案。"
proto = {b:{} for b in ["wvb","goqa","normad","blend"]}
for bench_key, source_name in [("wvb","worldvaluesbench"),("goqa","globalopinionqa"),
                                ("normad","normad"),("blend","blend")]:
    bench_rec = [r for r in all_recs if r["source"] == source_name]
    byC = defaultdict(list)
    for r in bench_rec: byC[r["country_or_region"]].append(r)
    for country, recs in byC.items():
        if bench_key == "wvb":
            tc = Counter(r["topic"] for r in recs).most_common(3)
            bullets = []
            for t, n in tc:
                mf = [max(r["distribution"].values()) for r in recs if r["topic"]==t and r["distribution"]]
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
            tc = Counter(r["topic"] for r in recs).most_common(3)
            bullets = [f"主题「{t}」: {n} 个意见问题 (PEW/WVS/GAS, ≤2022)" for t,n in tc]
            if not bullets: bullets.append(f"GOQA: {len(recs)} 条意见分布问题")
            bullets.append(f"覆盖 {len(recs)} 个 (question×country) 分布; "
                           f"平均每问选项数 {np.mean([len(r['answer_options']) for r in recs]):.1f}")
            bullets.append("数据来源跨年份合并 (Pew/WVS/GAS); 群体内代际/区域差异未在此原型中拆分。")
        elif bench_key == "normad":
            labels = Counter(str(r["label"]) for r in recs)
            sub = Counter(r["topic"] for r in recs).most_common(3)
            tot = sum(labels.values()) or 1
            bullets = [
                f"训练样本中礼仪场景判断分布: yes={labels.get('yes',0)/tot*100:.0f}%, "
                f"no={labels.get('no',0)/tot*100:.0f}%, "
                f"neutral={labels.get('neutral',0)/tot*100:.0f}% (NormAd-Eti, Rao et al. 2024)",
                "高频礼仪子轴: " + ", ".join(f"{s}({n})" for s,n in sub),
                "Cultural Atlas 来源, 概括性社会规范, 不覆盖区域/族群细分差异; "
                "具体场景可受时代/年龄/宗教多重影响。"]
        elif bench_key == "blend":
            tps = Counter(r["topic"] for r in recs).most_common(3)
            bullets = ["高频日常文化主题: " + ", ".join(f"{t}({n})" for t,n in tps)]
            avg_r = np.mean([r["metadata"]["respondent_count"] for r in recs])
            bullets.append(f"每问平均 {avg_r:.1f} 位受访者短答案 (BLEnD train, 2024); "
                           f"覆盖 {len(recs)} 个 (qid×country) 对")
            bullets.append("短答案为自由文本众数, 仅反映 BLEnD 招募人群在该国的常识倾向; "
                           "不应代表全国共识或专家意见。")
        else: bullets = [f"{bench_key} {country}: {len(recs)}"]
        tier = grouping[bench_key]["countries"].get(country,{}).get("tier","low")
        proto[bench_key][country] = {
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
                "builder": "rule_based_aggregator_v1_canonical_country_fix2",
            },
        }

for bench, cards in proto.items():
    outp = PROTO / bench / "country_prototypes.json"
    json.dump(cards, open(outp,"w"), indent=2, ensure_ascii=False, default=str)
    print(f"prototypes/{bench}: {len(cards)} cards")

# Update evidence_stats.json
per_country = Counter(r["country_or_region"] for r in all_recs)
stats = json.load(open(ANLY/"evidence_stats.json"))
stats["built_at"] = UTC_NOW
stats["stage"] = "post-country-canonicalization-fix2"
stats["n_unique_countries_canonical"] = len(per_country)
stats["top20_countries_by_record_count"] = per_country.most_common(20)
stats["bottom20_countries_by_record_count"] = per_country.most_common()[-20:]
stats["prototypes_per_bench"] = {b:len(c) for b,c in proto.items()}
stats["prototypes_high_resource_per_bench"] = {
    b: sum(1 for v in c.values() if v["metadata"]["resource_tier"]=="high")
    for b,c in proto.items()}
stats["prototypes_low_resource_per_bench"] = {
    b: sum(1 for v in c.values() if v["metadata"]["resource_tier"]=="low")
    for b,c in proto.items()}
json.dump(stats, open(ANLY/"evidence_stats.json","w"), indent=2, ensure_ascii=False, default=str)

# Update leakage_report.json canonicalization note (no need to re-scan: only renames within already-clean set)
lr = json.load(open(ANLY/"leakage_report.json"))
lr["canonicalization_fix2"] = {
    "applied_at": UTC_NOW,
    "additional_renames": [(o,n,c) for (o,n),c in rename_log.most_common()],
    "note": "Second-pass fix for acronym handling (SAR) + abbreviated forms (S. Korea → South Korea, Czech Rep. → Czech Republic, Palest. Ter. → Palestinian Territories). No re-scan needed: only consolidating equivalent country names within already-clean evidence set; text/distribution/labels unchanged.",
}
json.dump(lr, open(ANLY/"leakage_report.json","w"), indent=2, ensure_ascii=False, default=str)

print(f"\nFINAL: {len(all_recs)} records, {len(per_country)} canonical countries")
print(f"  prototypes: {stats['prototypes_per_bench']}")
print(f"  high_res:   {stats['prototypes_high_resource_per_bench']}")
print(f"  low_res:    {stats['prototypes_low_resource_per_bench']}")
