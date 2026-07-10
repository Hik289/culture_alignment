"""Step 3: 主表 6 methods × 4 bench × 3 prompts × 3 seeds.

6 methods (Researcher 确认):
- no_culture
- country
- demographic (WVB/GOQA 有, NormAd/BLEnD 退化 country)
- prototype (data_sci card)
- general_semantic (FAISS top-5 无 filter, 无 proto, 无 calib)
- culturelens_rc (country×topic filter + hier top-5 + proto + per-K T*)
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.model_client import chat_json, MODEL_NAME  # noqa: E402
from src.calibrate import apply_temperature  # noqa: E402
from src.io import load_bench  # noqa: E402
from src.metrics import (  # noqa: E402
    accuracy, exact_match, js_distance, macro_f1,
    token_f1, top1_accuracy, tv_distance, wasserstein_1,
)
from src.prompts import render_baseline, to_chat_messages  # noqa: E402
from src.retrieval import EvidenceItem  # noqa: E402
from src.run_predictions import dispatch_items  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)
logger = logging.getLogger("step3")

JST = timezone(timedelta(hours=9))

CA_ROOT = Path("${EXPERIMENT_ROOT}")
if not CA_ROOT.exists():
    CA_ROOT = ROOT
PROTOTYPES_DIR = CA_ROOT / "data" / "prototypes"
SPLITS_DIR = CA_ROOT / "data" / "splits"
EMBEDDINGS_DIR = Path("/tmp/culturelens_rc/embeddings")
CALIBRATIONS_PATH = CA_ROOT / "analysis" / "step_2_calibrations.json"

_CACHE: dict = {}


def get_prototypes(bench: str) -> dict:
    key = f"protos_{bench}"
    if key not in _CACHE:
        bench_key = "blend" if bench.startswith("blend") else bench
        p = PROTOTYPES_DIR / bench_key / "country_prototypes.json"
        _CACHE[key] = json.loads(p.read_text()) if p.exists() else {}
    return _CACHE[key]


def get_calibrations() -> dict:
    if "cal" not in _CACHE:
        _CACHE["cal"] = json.loads(CALIBRATIONS_PATH.read_text()) if CALIBRATIONS_PATH.exists() else {}
    return _CACHE["cal"]


def get_retrievers():
    if "ret" in _CACHE:
        return _CACHE["ret"]
    import faiss
    idx_path = EMBEDDINGS_DIR / "general.faiss"
    meta_path = EMBEDDINGS_DIR / "items_meta.jsonl"
    if not idx_path.exists():
        raise FileNotFoundError(f"FAISS index missing: {idx_path}")
    index = faiss.read_index(str(idx_path))
    items = []
    with meta_path.open() as f:
        for line in f:
            d = json.loads(line)
            items.append(EvidenceItem(
                id=d["id"], text=d["text"],
                country=d.get("country"), topic=d.get("topic"),
                year=d.get("year"), source=d.get("source"),
                lang=d.get("lang", "en"), meta=d.get("meta", {}),
            ))
    _CACHE["ret"] = {"faiss": index, "items": items, "n": index.ntotal}
    return _CACHE["ret"]


def get_embedder():
    if "emb" not in _CACHE:
        from sentence_transformers import SentenceTransformer
        _CACHE["emb"] = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    return _CACHE["emb"]


def retrieve_general(query: str, top_k: int = 5) -> list[EvidenceItem]:
    R = get_retrievers()
    q_emb = get_embedder().encode([query], normalize_embeddings=True, convert_to_numpy=True).astype("float32")
    D, I = R["faiss"].search(q_emb, top_k)
    out = []
    for rank, ii in enumerate(I[0]):
        if ii < 0:
            continue
        it = R["items"][ii]
        out.append(EvidenceItem(
            id=it.id, text=it.text, country=it.country, topic=it.topic,
            year=it.year, source=it.source, lang=it.lang,
            score=float(D[0][rank]), meta=dict(it.meta),
        ))
    return out


def retrieve_hierarchical(query: str, country: str, topic: Optional[str] = None,
                          top_k: int = 5) -> list[EvidenceItem]:
    R = get_retrievers()
    items = R["items"]
    exclude = {"test", "probe"}
    country = _normalize_country(country)
    country_lc = (country or "").lower()
    cand = []
    for i, it in enumerate(items):
        if country and (it.country or "").lower() != country_lc:
            continue
        if topic and (it.topic or "").lower() != topic.lower():
            continue
        if it.meta.get("split") in exclude:
            continue
        cand.append(i)
    if not cand:
        # fallback country only
        for i, it in enumerate(items):
            if country and (it.country or "").lower() != country_lc:
                continue
            if it.meta.get("split") in exclude:
                continue
            cand.append(i)
    if not cand:
        return []
    if "all_emb" not in _CACHE:
        _CACHE["all_emb"] = np.load(EMBEDDINGS_DIR / "embeddings.npy")
    all_emb = _CACHE["all_emb"]
    q_emb = get_embedder().encode([query], normalize_embeddings=True, convert_to_numpy=True).astype("float32")
    sims = (all_emb[cand] @ q_emb.T).squeeze(-1)
    order = np.argsort(-sims)[:top_k]
    out = []
    for rank in order:
        gi = cand[rank]
        it = items[gi]
        out.append(EvidenceItem(
            id=it.id, text=it.text, country=it.country, topic=it.topic,
            year=it.year, source=it.source, lang=it.lang,
            score=float(sims[rank]), meta=dict(it.meta),
        ))
    return out


# 国家名 alias (跨 bench 命名差异)
COUNTRY_ALIAS = {
    "UK": "United Kingdom",
    "United_Kingdom": "United Kingdom",
    "Britain": "United Kingdom",
    "US": "United States",
    "USA": "United States",
    "United_States": "United States",
    "Northern_Nigeria": "Northern Nigeria",
    "North_Korea": "North Korea",
    "South_Korea": "South Korea",
    "West_Java": "West Java",
    "S. Korea": "South Korea",
    "N. Korea": "North Korea",
}


def _normalize_country(country: str) -> str:
    if not country:
        return country
    c = country.strip()
    return COUNTRY_ALIAS.get(c, c)


def get_prototype_card(bench: str, country: str) -> Optional[dict]:
    protos = get_prototypes(bench)
    if not protos:
        return None
    c = _normalize_country(country)
    if c in protos:
        card = protos[c]
    else:
        lookup = {k.lower(): v for k, v in protos.items()}
        if c.lower() in lookup:
            card = lookup[c.lower()]
        else:
            return None
    summary = card.get("summary")
    summary_str = " | ".join(summary) if isinstance(summary, list) else str(summary or "")
    return {
        "summary": summary_str[:500],
        "warning": card.get("warning", "Aggregate-level; do not stereotype."),
        "n_evidence": card.get("n_evidence", 0),
    }


def get_T_star(bench: str, prompt_variant: int, K: int) -> float:
    cals = get_calibrations()
    pv_data = cals.get(bench, {}).get(f"pv{prompt_variant}", {})
    per_K = pv_data.get("per_K", {})
    if f"K{K}" in per_K:
        return float(per_K[f"K{K}"]["T_star"])
    avg = pv_data.get("T_star_weighted_avg")
    return float(avg) if avg is not None else 1.0


def apply_prompt_variant(rendered: dict, variant: int) -> dict:
    if variant == 0:
        return rendered
    user = rendered["user"]
    if variant == 1:
        user += "\n\nThink briefly about the relevant cultural context before answering."
    elif variant == 2:
        user += "\n\nOutput ONLY the JSON. No prose, no markdown fences."
    return {**rendered, "user": user}


def _fmt_evidence(items: list[EvidenceItem]) -> str:
    if not items:
        return "(none)"
    lines = []
    for i, it in enumerate(items, 1):
        meta = f"[{it.country or '?'} | {it.topic or '?'}]"
        lines.append(f"{i}. {meta} {it.text[:200]}")
    return "\n".join(lines)


def render_for_method(method, *, bench, item_id, country, question_text,
                       answer_options, demographic=None, topic=None, prompt_variant=0):
    if method == "no_culture":
        r = render_baseline("no_culture", item_id=item_id,
                            question_text=question_text, answer_options=answer_options)
    elif method == "country":
        r = render_baseline("country", item_id=item_id,
                            question_text=question_text, answer_options=answer_options,
                            country_or_region=country or "(unspecified)")
    elif method == "demographic":
        if demographic and country:
            r = render_baseline("demographic", item_id=item_id,
                                question_text=question_text, answer_options=answer_options,
                                country_or_region=country, demographic_attributes=demographic)
        else:
            r = render_baseline("country", item_id=item_id,
                                question_text=question_text, answer_options=answer_options,
                                country_or_region=country or "(unspecified)")
    elif method == "prototype":
        if not country:
            r = render_baseline("no_culture", item_id=item_id,
                                question_text=question_text, answer_options=answer_options)
        else:
            proto = get_prototype_card(bench, country)
            if proto is None:
                r = render_baseline("country", item_id=item_id,
                                    question_text=question_text, answer_options=answer_options,
                                    country_or_region=country)
            else:
                r = render_baseline("prototype", item_id=item_id,
                                    question_text=question_text, answer_options=answer_options,
                                    country_or_region=country, cultural_prototype=proto)
    elif method == "general_semantic":
        retrieved = retrieve_general(question_text, top_k=5)
        ev_text = _fmt_evidence(retrieved)
        r = render_baseline("country", item_id=item_id,
                            question_text=question_text + f"\n\nRetrieved evidence (general semantic):\n{ev_text}",
                            answer_options=answer_options,
                            country_or_region=country or "(unspecified)")
    elif method == "culturelens_rc":
        if not country:
            country = "(unspecified)"
        retrieved = retrieve_hierarchical(question_text, country=country, topic=topic, top_k=5)
        ev_text = _fmt_evidence(retrieved)
        proto = get_prototype_card(bench, country) or {"summary": "(no prototype)", "warning": ""}
        r = render_baseline("prototype", item_id=item_id,
                            question_text=question_text + f"\n\nRetrieved evidence (country x topic):\n{ev_text}",
                            answer_options=answer_options,
                            country_or_region=country, cultural_prototype=proto)
    else:
        raise ValueError(f"unknown method {method!r}")
    return apply_prompt_variant(r, prompt_variant)


def _safe_extract_probs(parsed, options):
    if not isinstance(parsed, dict):
        return None
    d = None
    for k in ("probabilities", "probs", "distribution"):
        if k in parsed and isinstance(parsed[k], (dict, list)):
            d = parsed[k]
            break
    if d is None:
        return None
    if isinstance(d, list):
        if len(d) != len(options):
            return None
        try:
            arr = np.array([float(x) for x in d], dtype=np.float64)
        except Exception:
            return None
    else:
        try:
            arr = np.array([float(d.get(str(o), d.get(o, 0.0))) for o in options], dtype=np.float64)
        except Exception:
            return None
    if np.any(arr < -1e-9):
        return None
    s = arr.sum()
    if s <= 0:
        return None
    return arr / s


def _safe_extract_answer(parsed):
    if not isinstance(parsed, dict):
        return None
    a = parsed.get("answer")
    if isinstance(a, str):
        return a.strip()
    if isinstance(a, list) and a:
        return str(a[0]).strip()
    return None


NORMAD_OPTIONS = ["yes", "no", "neutral"]
METHODS_ALL = ("no_culture", "country", "demographic", "prototype", "general_semantic", "culturelens_rc")


def _load_wvb_items(seed, n):
    df = load_bench("wvb.probe_distributions")
    sp = json.loads((SPLITS_DIR / "wvb_split.json").read_text())
    df = df[df["qid"].isin(set(sp["test"]))].reset_index(drop=True)
    if n and len(df) > n:
        df = df.sample(n=n, random_state=seed).reset_index(drop=True)
    qmeta = json.loads((CA_ROOT / "data/processed/wvb/probe_question_metadata.json").read_text())
    items = []
    for i, row in df.iterrows():
        qid = row["qid"]
        opts_raw = row["answer_options"]
        if isinstance(opts_raw, str):
            opts_raw = json.loads(opts_raw)
        opts = [str(o) for o in opts_raw]
        gold_raw = row["distribution"]
        if isinstance(gold_raw, str):
            gold_raw = json.loads(gold_raw)
        gold = np.array([float(gold_raw[o]) for o in opts], dtype=np.float64)
        gold = gold / gold.sum()
        q_text = qmeta.get(qid, {}).get("question", f"WVS {qid}")
        topic = qmeta.get(qid, {}).get("category")
        items.append({
            "item_id": f"wvb_{qid}_{i}",
            "bench": "wvb",
            "country": row.get("continent"),
            "topic": topic,
            "demographic": {"urb_rur": row.get("urb_rur"), "edu": row.get("edu")},
            "question_text": q_text,
            "answer_options": opts,
            "support": [float(o) for o in opts_raw],
            "gold_dist": gold.tolist(),
            "K": len(opts),
        })
    return items


def _load_goqa_items(seed, n):
    df = load_bench("goqa")
    sp = json.loads((SPLITS_DIR / "goqa_split.json").read_text())
    df = df[df["question_id"].isin(set(sp["test"]))].reset_index(drop=True)
    if n and len(df) > n:
        df = df.sample(n=n, random_state=seed).reset_index(drop=True)
    items = []
    for i, row in df.iterrows():
        qid = row["question_id"]
        opts = row["options_parsed"]
        if isinstance(opts, str):
            opts = json.loads(opts)
        sels = row["selections_parsed"]
        if isinstance(sels, str):
            sels = json.loads(sels)
        if not opts or not sels:
            continue
        country = next(iter(sels.keys()))
        gold = np.array([float(x) for x in sels[country]], dtype=np.float64)
        if gold.sum() <= 0:
            continue
        gold /= gold.sum()
        items.append({
            "item_id": f"goqa_{qid}_{i}",
            "bench": "goqa",
            "country": country,
            "topic": row.get("source"),
            "demographic": None,
            "question_text": row["question"],
            "answer_options": opts,
            "support": None,
            "gold_dist": gold.tolist(),
            "K": len(opts),
        })
    return items


def _load_normad_items(seed, n):
    df = load_bench("normad")
    sp = json.loads((SPLITS_DIR / "normad_split.json").read_text())
    df = df[df["story_id"].isin(set(sp["test"]))].reset_index(drop=True)
    if n and len(df) > n:
        df = df.sample(n=n, random_state=seed).reset_index(drop=True)
    items = []
    for i, row in df.iterrows():
        sid = row["story_id"]
        country = row["Country"]
        question = (f"Below is a social scenario. Decide whether the described behavior is "
                    f"socially acceptable in the cultural context of {country}.\n\n"
                    f"Scenario:\n{row['Story']}\n\n"
                    f"Choose exactly one of: yes (acceptable), no (not acceptable), neutral (depends).")
        items.append({
            "item_id": f"normad_{sid}",
            "bench": "normad",
            "country": country,
            "topic": row.get("Axis"),
            "demographic": None,
            "question_text": question,
            "answer_options": NORMAD_OPTIONS,
            "support": None,
            "gold_label": str(row["Gold Label"]).strip().lower(),
            "K": 3,
        })
    return items


def _load_blend_mc_items(seed, n):
    df = load_bench("blend.mc")
    sp = json.loads((SPLITS_DIR / "blend_split.json").read_text())
    df = df[df["ID"].isin(set(sp["test"]))].reset_index(drop=True)
    df = df.groupby("ID", as_index=False).first().reset_index(drop=True)
    if n and len(df) > n:
        df = df.sample(n=n, random_state=seed).reset_index(drop=True)
    items = []
    for i, row in df.iterrows():
        mcqid = row["MCQID"]
        choices_raw = row["choices"]
        if isinstance(choices_raw, str):
            choices_raw = json.loads(choices_raw)
        letters = sorted(choices_raw.keys())
        options = [choices_raw[L] for L in letters]
        items.append({
            "item_id": f"blend_mc_{mcqid}",
            "bench": "blend_mc",
            "country": row["country"],
            "topic": "Food",
            "demographic": None,
            "question_text": row["prompt"],
            "answer_options": options,
            "support": None,
            "gold_letter": row["answer_idx"],
            "K": len(letters),
            "letters": letters,
        })
    return items


def _load_blend_saq_items(seed, n):
    raw_path = CA_ROOT / "data/raw/blend/hf/data/annotations_hf/US_data.json"
    if not raw_path.exists():
        return []
    annot = json.loads(raw_path.read_text())
    valid = [x for x in annot if x.get("annotations") and any(a.get("en_answers") for a in x["annotations"])]
    import random
    rng = random.Random(seed)
    rng.shuffle(valid)
    chosen = valid[:n] if n else valid
    items = []
    for x in chosen:
        sid = x["ID"]
        q = x.get("en_question") or x.get("question")
        golds = []
        for a in x["annotations"]:
            for g in (a.get("en_answers") or []):
                if isinstance(g, str) and g.strip():
                    golds.append(g.strip().lower())
        if not golds:
            continue
        items.append({
            "item_id": f"blend_saq_{sid}",
            "bench": "blend_saq",
            "country": "US",
            "topic": "Food",
            "demographic": None,
            "question_text": q,
            "answer_options": None,
            "support": None,
            "golds": golds,
            "K": None,
        })
    return items


BENCH_LOADERS = {
    "wvb": _load_wvb_items,
    "goqa": _load_goqa_items,
    "normad": _load_normad_items,
    "blend_mc": _load_blend_mc_items,
    "blend_saq": _load_blend_saq_items,
}


def run_cell(method, bench, prompt_variant, seed, n_samples=None, max_workers=8):
    logger.info("=== %s × %s × pv%d × seed%d (n=%s) ===", method, bench, prompt_variant, seed, n_samples)
    items = BENCH_LOADERS[bench](seed, n_samples)
    if not items:
        return {"error": "no items"}
    task_items = []
    for it in items:
        rendered = render_for_method(method, bench=bench, item_id=it["item_id"],
                                      country=it.get("country"), question_text=it["question_text"],
                                      answer_options=it.get("answer_options"),
                                      demographic=it.get("demographic"), topic=it.get("topic"),
                                      prompt_variant=prompt_variant)
        task_items.append({
            "item_id": it["item_id"],
            "messages": to_chat_messages(rendered),
            "item": it,
        })

    def handler(t):
        res = chat_json(messages=t["messages"], max_retries=2)
        return {
            "item_id": t["item_id"], "ok": bool(res.ok), "parsed": res.parsed,
            "prompt_tokens": res.prompt_tokens, "completion_tokens": res.completion_tokens,
            "latency_seconds": res.latency_seconds, "n_attempts": res.n_attempts,
        }

    results = dispatch_items(task_items, handler, max_workers=max_workers, on_error="record")
    by_item = {r["item_id"]: r for r in results}
    return _compute_metrics(bench, method, prompt_variant, items, by_item)


def _compute_metrics(bench, method, prompt_variant, items, by_item):
    sanity, derived = [], []
    n_ok = n_fail = 0
    if bench in ("wvb", "goqa"):
        w1, jsd, tvd, top1 = [], [], [], []
        for it in items:
            r = by_item.get(it["item_id"])
            if not r or not r.get("ok"):
                n_fail += 1; continue
            pred = _safe_extract_probs(r["parsed"], it["answer_options"])
            if pred is None:
                n_fail += 1; continue
            gold = np.array(it["gold_dist"], dtype=np.float64)
            if method == "culturelens_rc":
                T = get_T_star(bench, prompt_variant, it["K"])
                pred = apply_temperature(pred, T)
            jsd.append(js_distance(pred, gold))
            tvd.append(tv_distance(pred, gold))
            top1.append(top1_accuracy(pred, gold))
            if bench == "wvb" and it.get("support") is not None:
                w1.append(wasserstein_1(pred, gold, support=it["support"]))
            n_ok += 1
            d = {"item_id": it["item_id"], "country": it.get("country"),
                 "jsd": jsd[-1], "tvd": tvd[-1], "top1": int(top1[-1])}
            if w1 and bench == "wvb":
                d["w1"] = w1[-1]
            derived.append(d)
            if len(sanity) < 100:
                sanity.append({**d, "pred_probs": pred.tolist(), "gold_probs": gold.tolist(),
                              "prompt_tokens": r.get("prompt_tokens"),
                              "completion_tokens": r.get("completion_tokens")})
        out = {
            "bench": bench, "method": method, "prompt_variant": prompt_variant,
            "n_items": len(items), "n_ok": n_ok, "n_fail": n_fail,
            "metrics": {
                "JS_distance_mean": float(np.mean(jsd)) if jsd else None,
                "TV_distance_mean": float(np.mean(tvd)) if tvd else None,
                "Top1_Acc": float(np.mean(top1)) if top1 else None,
            },
        }
        if w1:
            out["metrics"]["W1_mean"] = float(np.mean(w1))
            out["metrics"]["W1_std"] = float(np.std(w1))
        return {"summary": out, "sanity": sanity, "derived": derived}

    if bench == "normad":
        y_true, y_pred = [], []
        for it in items:
            r = by_item.get(it["item_id"])
            if not r or not r.get("ok"):
                n_fail += 1; continue
            ans = _safe_extract_answer(r["parsed"])
            if not ans:
                n_fail += 1; continue
            ans_n = ans.strip().lower()
            if ans_n not in NORMAD_OPTIONS:
                if ans_n.startswith("yes"):
                    ans_n = "yes"
                elif ans_n.startswith("no"):
                    ans_n = "no"
                elif "neutral" in ans_n or "depend" in ans_n:
                    ans_n = "neutral"
                else:
                    n_fail += 1; continue
            m = {"yes": 0, "no": 1, "neutral": 2}
            y_true.append(m[it["gold_label"]])
            y_pred.append(m[ans_n])
            n_ok += 1
            d = {"item_id": it["item_id"], "country": it.get("country"),
                 "gold": it["gold_label"], "pred": ans_n,
                 "correct": int(it["gold_label"] == ans_n)}
            derived.append(d)
            if len(sanity) < 100:
                sanity.append({**d, "prompt_tokens": r.get("prompt_tokens"),
                              "completion_tokens": r.get("completion_tokens")})
        out = {
            "bench": bench, "method": method, "prompt_variant": prompt_variant,
            "n_items": len(items), "n_ok": n_ok, "n_fail": n_fail,
            "metrics": {
                "Accuracy": accuracy(y_true, y_pred) if y_true else None,
                "Macro_F1": macro_f1(y_true, y_pred, labels=[0, 1, 2]) if y_true else None,
            },
        }
        return {"summary": out, "sanity": sanity, "derived": derived}

    if bench == "blend_mc":
        y_true, y_pred = [], []
        for it in items:
            r = by_item.get(it["item_id"])
            if not r or not r.get("ok"):
                n_fail += 1; continue
            ans = _safe_extract_answer(r["parsed"])
            if not ans:
                n_fail += 1; continue
            letters = it["letters"]
            choices = {L: it["answer_options"][i] for i, L in enumerate(letters)}
            pred_letter = None
            ans_lc = ans.strip().lower()
            for L, txt in choices.items():
                if ans_lc == (txt or "").strip().lower():
                    pred_letter = L; break
            if pred_letter is None:
                ah = ans.strip().upper()
                if ah in choices:
                    pred_letter = ah
            if pred_letter is None:
                n_fail += 1; continue
            L2I = {L: i for i, L in enumerate(letters)}
            y_true.append(L2I[it["gold_letter"]])
            y_pred.append(L2I[pred_letter])
            n_ok += 1
            d = {"item_id": it["item_id"], "country": it.get("country"),
                 "gold_letter": it["gold_letter"], "pred_letter": pred_letter,
                 "correct": int(it["gold_letter"] == pred_letter)}
            derived.append(d)
            if len(sanity) < 100:
                sanity.append({**d, "prompt_tokens": r.get("prompt_tokens"),
                              "completion_tokens": r.get("completion_tokens")})
        out = {
            "bench": bench, "method": method, "prompt_variant": prompt_variant,
            "n_items": len(items), "n_ok": n_ok, "n_fail": n_fail,
            "metrics": {"Accuracy": accuracy(y_true, y_pred) if y_true else None},
        }
        return {"summary": out, "sanity": sanity, "derived": derived}

    if bench == "blend_saq":
        em, f1 = [], []
        for it in items:
            r = by_item.get(it["item_id"])
            if not r or not r.get("ok"):
                n_fail += 1; continue
            ans = _safe_extract_answer(r["parsed"])
            if not ans:
                n_fail += 1; continue
            em.append(max(exact_match(ans, g) for g in it["golds"]))
            f1.append(max(token_f1(ans, g) for g in it["golds"]))
            n_ok += 1
            d = {"item_id": it["item_id"], "country": it.get("country"),
                 "pred": ans, "golds": it["golds"][:3], "em": em[-1], "f1": f1[-1]}
            derived.append(d)
            if len(sanity) < 100:
                sanity.append({**d, "prompt_tokens": r.get("prompt_tokens"),
                              "completion_tokens": r.get("completion_tokens")})
        out = {
            "bench": bench, "method": method, "prompt_variant": prompt_variant,
            "n_items": len(items), "n_ok": n_ok, "n_fail": n_fail,
            "metrics": {
                "EM": float(np.mean(em)) if em else None,
                "Token_F1": float(np.mean(f1)) if f1 else None,
            },
        }
        return {"summary": out, "sanity": sanity, "derived": derived}

    return {"summary": {"error": f"unknown bench {bench}"}}


def main_smoke():
    """1 sample × 4 main bench × 6 methods = 24 调用."""
    out_dir = ROOT / "experiments" / "step_3_main" / "smoke"
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    results = {}
    n_ok = n_fail = 0
    for bench in ("wvb", "goqa", "normad", "blend_mc"):
        results[bench] = {}
        for method in METHODS_ALL:
            try:
                cell = run_cell(method, bench, prompt_variant=0, seed=42, n_samples=1, max_workers=4)
            except Exception as e:
                logger.exception("smoke %s × %s failed", method, bench)
                results[bench][method] = {"error": f"{type(e).__name__}: {e}"}
                n_fail += 1
                continue
            s = cell.get("summary", cell)
            results[bench][method] = s
            if isinstance(s, dict) and s.get("n_fail", 0) == 0 and s.get("n_ok", 0) > 0:
                n_ok += 1
            else:
                n_fail += 1
    elapsed = time.time() - t0
    log = {
        "step": "step_3_smoke",
        "timestamp_jst": datetime.now(JST).isoformat(),
        "n_cells_total": 4 * len(METHODS_ALL),
        "n_cells_ok": n_ok,
        "n_cells_failed": n_fail,
        "elapsed_seconds": elapsed,
        "results": results,
    }
    (out_dir / "smoke_results.json").write_text(json.dumps(log, ensure_ascii=False, indent=2))
    logger.info("=== SMOKE DONE === %d ok / %d total in %.1fs", n_ok, 4 * len(METHODS_ALL), elapsed)
    print(json.dumps(log, ensure_ascii=False, indent=2))


# ---------------------------------------------------------------------------
# Main table driver (cell-by-cell with checkpoint)
# ---------------------------------------------------------------------------

SEEDS_ALL = [42, 123, 456]
PROMPTS_ALL = [0, 1, 2]

# 每 bench 子样规模 (Researcher exp_design)
BENCH_N_SAMPLES = {
    "wvb": 1656,      # 全 test
    "goqa": 511,      # 全 test
    "normad": 400,
    "blend_mc": 400,
    "blend_saq": 200,
}


def main_table():
    """主表: 6 method × 3 prompt × 3 seed × 5 bench = 270 cells.

    Checkpoint every cell. 失败 cell 加 error_log, 不阻塞.
    """
    out_dir = ROOT / "experiments" / "step_3_main"
    runs_dir = out_dir / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    progress_path = out_dir / "progress.json"

    # 加载已有进度 (resume 友好)
    completed = set()
    if progress_path.exists():
        prog = json.loads(progress_path.read_text())
        completed = set(prog.get("completed", []))
        logger.info("RESUME: %d cells already completed", len(completed))
    else:
        prog = {
            "start_ts_jst": datetime.now(JST).isoformat(),
            "completed": [],
            "errors": [],
            "n_calls_total": 0,
            "cost_usd_estimate": 0.0,
        }

    cells_total = []
    for method in METHODS_ALL:
        for pv in PROMPTS_ALL:
            for seed in SEEDS_ALL:
                for bench in ("wvb", "goqa", "normad", "blend_mc", "blend_saq"):
                    cell_id = f"{method}_{bench}_pv{pv}_seed{seed}"
                    cells_total.append((method, bench, pv, seed, cell_id))

    logger.info("=== Main table: %d cells total ===", len(cells_total))
    t_start = time.time()
    last_ckpt_ts = t_start

    for idx, (method, bench, pv, seed, cell_id) in enumerate(cells_total):
        if cell_id in completed:
            continue
        cell_path = runs_dir / f"{cell_id}.json"
        if cell_path.exists():
            completed.add(cell_id)
            prog["completed"] = list(completed)
            continue

        logger.info("[%d/%d] %s", idx + 1, len(cells_total), cell_id)
        n_samples = BENCH_N_SAMPLES[bench]
        try:
            cell = run_cell(method, bench, pv, seed, n_samples=n_samples, max_workers=8)
        except Exception as e:
            logger.exception("cell %s failed", cell_id)
            prog["errors"].append({
                "cell": cell_id, "ts": datetime.now(JST).isoformat(),
                "error": f"{type(e).__name__}: {e}",
            })
            cell = {"summary": {"error": str(e)}, "sanity": [], "derived": []}

        # 落盘 (sanity + derived 都在; raw responses 不留)
        cell_path.write_text(json.dumps(cell, ensure_ascii=False, indent=2, default=str))
        completed.add(cell_id)
        prog["completed"] = list(completed)
        # 累计 token / cost
        summary = cell.get("summary", {})
        n_items = summary.get("n_items", 0)
        prog["n_calls_total"] += n_items
        prog["cost_usd_estimate"] = prog["n_calls_total"] * 0.000066

        # Checkpoint every 6h
        now = time.time()
        if now - last_ckpt_ts > 6 * 3600 or idx == len(cells_total) - 1:
            prog["last_ckpt_jst"] = datetime.now(JST).isoformat()
            prog["elapsed_seconds"] = now - t_start
            progress_path.write_text(json.dumps(prog, ensure_ascii=False, indent=2))
            last_ckpt_ts = now
            logger.info("CKPT: %d/%d cells, $%.4f", len(completed), len(cells_total),
                        prog["cost_usd_estimate"])

        # 在每 cell 完成后也写一次小 ckpt (resume 友好)
        progress_path.write_text(json.dumps(prog, ensure_ascii=False, indent=2))

    elapsed = time.time() - t_start
    prog["done_ts_jst"] = datetime.now(JST).isoformat()
    prog["total_elapsed_seconds"] = elapsed
    progress_path.write_text(json.dumps(prog, ensure_ascii=False, indent=2))
    logger.info("=== MAIN TABLE DONE === %d cells, $%.4f, %.1fs", 
                len(completed), prog["cost_usd_estimate"], elapsed)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--main", action="store_true")
    args = parser.parse_args()
    if args.smoke:
        main_smoke()
    elif args.main:
        main_table()
    else:
        print("usage: --smoke | --main")
        sys.exit(1)
