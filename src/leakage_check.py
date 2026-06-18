"""
leakage_check.py — CultureLens-RC evidence leakage detector
============================================================

Implements readme §10.4: every retrieved evidence record MUST satisfy:

  (R1) evidence.split != "test"                                  ── split barrier
  (R2) evidence.source_question_id != target.question_id          ── same-question leak
  (R3) normalized_text_hash(evidence.text) !=
       normalized_text_hash(target.question_text)                 ── exact-text leak
  (R4) cosine_similarity(embedding(evidence.text),
                         embedding(target.question_text)) < tau   ── near-dup leak

A check FLAGS the (evidence, target) pair as a "leakage hit" if ANY of the
above rules fires. Recall = fraction of injected leakage pairs flagged;
False-Positive-Rate = fraction of clean pairs that are wrongly flagged.

The module exposes:

  - `normalize_text(s) -> str`            (lower, collapse whitespace, strip punct)
  - `text_hash(s) -> str`                 (sha256 of normalized text)
  - `LeakageChecker.is_leak(e, t) -> dict` (per-pair verdict + reasons)
  - `LeakageChecker.evaluate(records) -> dict` (recall / FPR / per-rule hits)

Embedding model: defaults to `sentence-transformers/all-MiniLM-L6-v2` (cheap,
sufficient for sanity check). For real evidence-bank scans use a multilingual
model (e.g. `paraphrase-multilingual-MiniLM-L12-v2`).
"""
from __future__ import annotations

import hashlib
import re
import unicodedata
from dataclasses import dataclass, field
from typing import Iterable, Sequence

import numpy as np

# ── lazy embedding model: importing sentence_transformers is heavy ─────────
_EMBED_MODEL = None
_DEFAULT_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def _get_model(name: str = _DEFAULT_MODEL_NAME):
    global _EMBED_MODEL
    if _EMBED_MODEL is None:
        from sentence_transformers import SentenceTransformer  # local import

        _EMBED_MODEL = SentenceTransformer(name)
    return _EMBED_MODEL


# ── text normalization ────────────────────────────────────────────────────
_PUNCT_RE = re.compile(r"[\s\p{P}\p{S}]+", re.UNICODE) if False else re.compile(
    r"[\s\.,;:!\?\"'\(\)\[\]\{\}\-—–_/\\\*\&\^\%\$\#\@\~\`\<\>\|\+\=]+"
)


def normalize_text(s: str) -> str:
    """Lowercase, NFKC unicode normalize, collapse punctuation+whitespace to single space."""
    if s is None:
        return ""
    s = unicodedata.normalize("NFKC", str(s)).lower()
    s = _PUNCT_RE.sub(" ", s)
    return s.strip()


def text_hash(s: str) -> str:
    return hashlib.sha256(normalize_text(s).encode("utf-8")).hexdigest()


# ── data records ──────────────────────────────────────────────────────────
@dataclass
class EvidenceRecord:
    evidence_id: str
    source: str  # 'worldvaluesbench' | 'globalopinionqa' | ...
    split: str   # 'train' | 'valid' | 'dev' | (must not be 'test')
    source_question_id: str | None  # the question_id this evidence was derived from
    text: str    # the standardized text the retriever would put in the prompt


@dataclass
class TargetRecord:
    target_benchmark: str       # 'worldvaluesbench' | ...
    target_split: str           # typically 'test' (the question we are about to answer)
    target_item_id: str         # question_id / story_id of the target
    question_text: str


@dataclass
class LeakVerdict:
    is_leak: bool
    reasons: list[str] = field(default_factory=list)
    similarity: float = 0.0


# ── checker ───────────────────────────────────────────────────────────────
class LeakageChecker:
    """
    Implements readme §10.4 rules R1-R4.

    Parameters
    ----------
    similarity_threshold : float
        Cosine similarity (cosine of normalized embeddings) at or above which
        an evidence is treated as a near-duplicate of the target question.
        Default 0.95 per Researcher task spec.
    model_name : str | None
        Sentence-transformer model name. If None, embedding check (R4) is
        skipped (useful for unit tests without HF model download).
    """

    def __init__(self, similarity_threshold: float = 0.95,
                 model_name: str | None = _DEFAULT_MODEL_NAME):
        self.tau = float(similarity_threshold)
        self.model_name = model_name
        self._embeddings_cache: dict[str, np.ndarray] = {}

    # -- embedding utility -------------------------------------------------
    def _embed(self, texts: Sequence[str]) -> np.ndarray:
        if self.model_name is None:
            # zero embeddings; cosine similarity will be 0
            return np.zeros((len(texts), 1), dtype=np.float32)
        model = _get_model(self.model_name)
        normed = [normalize_text(t) for t in texts]
        # check cache
        out = np.zeros((len(texts), model.get_sentence_embedding_dimension()),
                       dtype=np.float32)
        to_encode_idx: list[int] = []
        to_encode_txt: list[str] = []
        for i, t in enumerate(normed):
            if t in self._embeddings_cache:
                out[i] = self._embeddings_cache[t]
            else:
                to_encode_idx.append(i)
                to_encode_txt.append(t)
        if to_encode_txt:
            embs = model.encode(to_encode_txt, normalize_embeddings=True,
                                show_progress_bar=False, convert_to_numpy=True)
            for j, i in enumerate(to_encode_idx):
                out[i] = embs[j]
                self._embeddings_cache[normed[i]] = embs[j]
        return out

    # -- single pair -------------------------------------------------------
    def is_leak(self, evidence: EvidenceRecord, target: TargetRecord) -> LeakVerdict:
        reasons: list[str] = []
        # R1: split barrier
        if str(evidence.split).strip().lower() == "test":
            reasons.append("R1_evidence_split_is_test")
        # R2: same question_id
        if (evidence.source_question_id is not None
                and evidence.source == target.target_benchmark
                and str(evidence.source_question_id).strip()
                == str(target.target_item_id).strip()):
            reasons.append("R2_same_question_id")
        # R3: exact normalized text hash
        if text_hash(evidence.text) == text_hash(target.question_text):
            reasons.append("R3_exact_text_hash_match")
        # R4: near-duplicate embedding cosine
        sim = 0.0
        if self.model_name is not None:
            embs = self._embed([evidence.text, target.question_text])
            sim = float(np.clip(np.dot(embs[0], embs[1]), -1.0, 1.0))
            if sim >= self.tau:
                reasons.append(f"R4_near_duplicate_cos={sim:.3f}")
        return LeakVerdict(is_leak=bool(reasons), reasons=reasons, similarity=sim)

    # -- batch evaluation --------------------------------------------------
    def evaluate(self, records: Iterable[tuple[EvidenceRecord, TargetRecord, bool]]
                 ) -> dict:
        """
        records: iterable of (evidence, target, is_truly_leak_label).
        Returns recall / FPR / per-rule hit counts.
        """
        records = list(records)
        tp = fp = tn = fn = 0
        rule_hits = {"R1": 0, "R2": 0, "R3": 0, "R4": 0}
        per_pair = []
        for ev, tg, gold in records:
            v = self.is_leak(ev, tg)
            for r in v.reasons:
                for k in rule_hits:
                    if r.startswith(k):
                        rule_hits[k] += 1
            if gold and v.is_leak:
                tp += 1
            elif gold and not v.is_leak:
                fn += 1
            elif (not gold) and v.is_leak:
                fp += 1
            else:
                tn += 1
            per_pair.append({
                "evidence_id": ev.evidence_id,
                "target_id": tg.target_item_id,
                "gold_label": bool(gold),
                "predicted_leak": v.is_leak,
                "reasons": v.reasons,
                "similarity": v.similarity,
            })

        recall = tp / max(tp + fn, 1)
        fpr = fp / max(fp + tn, 1)
        return {
            "tp": tp, "fp": fp, "tn": tn, "fn": fn,
            "recall": recall, "false_positive_rate": fpr,
            "rule_hits": rule_hits,
            "similarity_threshold": self.tau,
            "model_name": self.model_name,
            "n_records": len(records),
            "per_pair": per_pair,
        }


# ── CLI entry (matches readme §22 invocation) ────────────────────────────
if __name__ == "__main__":
    import argparse, json, sys

    p = argparse.ArgumentParser(description="CultureLens-RC leakage check (readme §10.4)")
    p.add_argument("--retrieval-log", required=True,
                   help="JSONL of {evidence: {...}, target: {...}, gold?: bool}")
    p.add_argument("--output", required=True, help="output JSON report path")
    p.add_argument("--similarity-threshold", type=float, default=0.95)
    p.add_argument("--model-name", default=_DEFAULT_MODEL_NAME)
    p.add_argument("--no-embedding", action="store_true",
                   help="Skip R4 (embedding) check; useful for fast smoke tests.")
    args = p.parse_args()

    checker = LeakageChecker(similarity_threshold=args.similarity_threshold,
                             model_name=None if args.no_embedding else args.model_name)
    rows = []
    with open(args.retrieval_log) as f:
        for line in f:
            line = line.strip()
            if not line: continue
            o = json.loads(line)
            ev = EvidenceRecord(**o["evidence"])
            tg = TargetRecord(**o["target"])
            rows.append((ev, tg, bool(o.get("gold", False))))
    report = checker.evaluate(rows)
    with open(args.output, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(json.dumps({k: v for k, v in report.items() if k != "per_pair"}, indent=2))
    if report["recall"] < 1.0 or report["false_positive_rate"] >= 0.1:
        sys.exit(2)
