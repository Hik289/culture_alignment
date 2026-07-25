"""Build retrieval indexes from a validated evidence JSONL file.

The identity embedder provides a deterministic offline test path. Production
runs may select a sentence-transformer; FAISS is used when installed, with a
NumPy index fallback otherwise.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

# Support direct execution from the repository root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.retrieval import (
    Embedder,
    EvidenceItem,
    GeneralSemanticRetriever,
    HierarchicalRetriever,
    IdentityEmbedder,
)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class BuildConfig:
    evidence_path: str = "data/evidence/evidence.jsonl"
    out_dir: str = "experiments/retrieval"
    embedder_kind: str = "identity"  # "identity" / "sentence_transformers"
    embedder_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedder_dim: int = 384
    batch_size: int = 64
    index_kinds: tuple[str, ...] = ("general", "hierarchical")
    cache_embeddings: bool = True
    persistent_only_final: bool = True

    def validate(self) -> None:
        if self.embedder_kind not in ("identity", "sentence_transformers"):
            raise ValueError(f"unknown embedder_kind {self.embedder_kind!r}")
        for k in self.index_kinds:
            if k not in ("general", "hierarchical"):
                raise ValueError(f"unknown index kind {k!r}")
        if self.batch_size <= 0:
            raise ValueError("batch_size > 0")


@dataclass
class BuildResult:
    config: BuildConfig
    n_evidence: int = 0
    embedding_cache_path: str | None = None
    index_paths: dict[str, str] = field(default_factory=dict)
    elapsed_seconds: float = 0.0

    def to_dict(self) -> dict:
        d = asdict(self)
        return d


# ---------------------------------------------------------------------------
# Evidence loader
# ---------------------------------------------------------------------------

def load_evidence_jsonl(path: str | os.PathLike) -> list[EvidenceItem]:
    """Load and validate evidence records.

    支持两种 schema:

    (A) Materialized schema:
        evidence_id / text / country_or_region / topic / source / language / task_type / ...

    (B) Legacy compact schema:
        id / text / country / topic / year / source / lang

    Both require ``text`` and either ``evidence_id`` or ``id``. Unrecognized
    fields are retained in ``EvidenceItem.meta``.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"evidence file not found: {p}")
    items: list[EvidenceItem] = []

    KNOWN_DS = {
        "evidence_id", "text", "country_or_region", "topic",
        "source", "language", "score",
        "split", "task_type", "answer_options", "distribution",
        "label", "short_answers", "metadata", "indexed_at", "license",
    }
    KNOWN_SIMPLE = {"id", "text", "country", "topic", "year", "source", "lang", "score"}

    with p.open() as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"{p}:{line_no} invalid JSON: {e}")

            if "evidence_id" in d:
                if "text" not in d:
                    raise ValueError(f"{p}:{line_no} missing text")
                year = None
                metadata = d.get("metadata") or {}
                if isinstance(metadata, dict):
                    for k in ("year", "wvs_wave", "publication_year"):
                        if k in metadata:
                            try:
                                year = int(metadata[k])
                                break
                            except (ValueError, TypeError):
                                pass
                meta = {k: v for k, v in d.items() if k not in KNOWN_DS}
                # Preserve fields used by filtering and downstream analysis.
                for k in ("split", "task_type", "answer_options", "distribution", "label"):
                    if k in d:
                        meta[k] = d[k]
                items.append(EvidenceItem(
                    id=str(d["evidence_id"]),
                    text=str(d["text"]),
                    country=d.get("country_or_region"),
                    topic=d.get("topic"),
                    year=year,
                    source=d.get("source"),
                    lang=d.get("language", "en"),
                    meta=meta,
                ))
            elif "id" in d:
                if "text" not in d:
                    raise ValueError(f"{p}:{line_no} missing text")
                meta = {k: v for k, v in d.items() if k not in KNOWN_SIMPLE}
                items.append(EvidenceItem(
                    id=str(d["id"]),
                    text=str(d["text"]),
                    country=d.get("country"),
                    topic=d.get("topic"),
                    year=d.get("year"),
                    source=d.get("source"),
                    lang=d.get("lang", "en"),
                    meta=meta,
                ))
            else:
                raise ValueError(
                    f"{p}:{line_no} missing required field 'evidence_id' or 'id'"
                )
    return items


# ---------------------------------------------------------------------------
# Embedder factory
# ---------------------------------------------------------------------------

class SentenceTransformerEmbedder:
    """Thin sentence-transformers adapter returning normalized float32 arrays."""

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
                 batch_size: int = 64, device: str | None = None,
                 normalize: bool = True):
        # Import lazily so offline identity-embedder tests need no model package.
        from sentence_transformers import SentenceTransformer
        self.model_name = model_name
        self.batch_size = batch_size
        self.normalize = normalize
        self.model = SentenceTransformer(model_name, device=device)
        self.dim = self.model.get_sentence_embedding_dimension()

    def encode(self, texts: list[str]):
        import numpy as np
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        emb = self.model.encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=self.normalize,
            convert_to_numpy=True,
            show_progress_bar=len(texts) > 1000,
        )
        return emb.astype("float32")


def make_embedder(config: BuildConfig) -> Embedder:
    """Construct the deterministic test or sentence-transformer embedder."""
    if config.embedder_kind == "identity":
        return IdentityEmbedder(dim=config.embedder_dim)
    if config.embedder_kind == "sentence_transformers":
        return SentenceTransformerEmbedder(
            model_name=config.embedder_model,
            batch_size=config.batch_size,
        )
    raise ValueError(f"unknown embedder kind: {config.embedder_kind}")


# ---------------------------------------------------------------------------
# Build pipeline
# ---------------------------------------------------------------------------

def build_index(config: BuildConfig) -> BuildResult:
    """Build shared embeddings, search index files, and item metadata."""
    import time
    config.validate()
    t0 = time.time()

    items = load_evidence_jsonl(config.evidence_path)
    logger.info("loaded %d evidence items from %s", len(items), config.evidence_path)

    embedder = make_embedder(config)
    out_dir = Path(config.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    result = BuildResult(config=config, n_evidence=len(items))

    # Build embeddings once and share them across retrieval strategies.
    r_general = None
    embeddings = None
    if "general" in config.index_kinds or config.cache_embeddings:
        r_general = GeneralSemanticRetriever(embedder)
        r_general.build(items)
        embeddings = r_general._embeddings  # numpy float32 (N, D)
        if config.cache_embeddings and embeddings is not None and embeddings.size > 0:
            import numpy as np
            cache_path = out_dir / "embeddings.npy"
            np.save(cache_path, embeddings)
            result.embedding_cache_path = str(cache_path)
            logger.info("cached embeddings %s shape=%s", cache_path, embeddings.shape)

    # Normalized embeddings make inner product equivalent to cosine similarity.
    if "general" in config.index_kinds and embeddings is not None and embeddings.size > 0:
        path = _persist_faiss_index(embeddings, out_dir / "general.faiss")
        result.index_paths["general"] = str(path)

    # Hierarchical filtering happens at query time, so it shares the same index.
    if "hierarchical" in config.index_kinds:
        r_hier = HierarchicalRetriever(embedder)
        if r_general is not None and embeddings is not None:
            r_hier._items = list(items)
            r_hier._embeddings = embeddings
            r_hier._built = True
        else:
            r_hier.build(items)
        if "general" in result.index_paths:
            result.index_paths["hierarchical_shares_with_general"] = result.index_paths["general"]
        elif embeddings is not None and embeddings.size > 0:
            path = _persist_faiss_index(embeddings, out_dir / "hierarchical.faiss")
            result.index_paths["hierarchical"] = str(path)

    # Preserve the row-to-record mapping required when decoding search results.
    if items:
        meta_path = out_dir / "items_meta.jsonl"
        with meta_path.open("w") as f:
            for it in items:
                f.write(json.dumps(asdict(it), ensure_ascii=False) + "\n")
        result.index_paths["items_meta"] = str(meta_path)

    result.elapsed_seconds = time.time() - t0
    return result


def _persist_faiss_index(embeddings, out_path: Path) -> Path:
    """Persist an inner-product FAISS index, or a NumPy fallback."""
    out_path = Path(out_path)
    try:
        import faiss  # type: ignore
        import numpy as np
        index = faiss.IndexFlatIP(embeddings.shape[1])
        index.add(np.ascontiguousarray(embeddings, dtype="float32"))
        faiss.write_index(index, str(out_path))
        logger.info("wrote FAISS index %s (n=%d, d=%d)",
                    out_path, index.ntotal, embeddings.shape[1])
        return out_path
    except ImportError:
        import numpy as np
        fallback = out_path.with_suffix(".npy")
        np.save(fallback, embeddings)
        logger.warning("faiss not installed, fallback to npy: %s", fallback)
        return fallback


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv: list[str] | None = None) -> BuildConfig:
    p = argparse.ArgumentParser(description="culturealignment retrieval index builder")
    p.add_argument("--evidence", default="data/evidence/evidence.jsonl")
    p.add_argument("--out-dir", default="experiments/retrieval")
    p.add_argument("--embedder", default="identity",
                   choices=["identity", "sentence_transformers"])
    p.add_argument("--embedder-model", default="sentence-transformers/all-MiniLM-L6-v2")
    p.add_argument("--embedder-dim", type=int, default=384)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--no-general", action="store_true")
    p.add_argument("--no-hierarchical", action="store_true")
    p.add_argument("--no-cache-embeddings", action="store_true")
    a = p.parse_args(argv)
    kinds: list[str] = []
    if not a.no_general:
        kinds.append("general")
    if not a.no_hierarchical:
        kinds.append("hierarchical")
    return BuildConfig(
        evidence_path=a.evidence,
        out_dir=a.out_dir,
        embedder_kind=a.embedder,
        embedder_model=a.embedder_model,
        embedder_dim=a.embedder_dim,
        batch_size=a.batch_size,
        index_kinds=tuple(kinds),
        cache_embeddings=not a.no_cache_embeddings,
    )


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s | %(levelname)s | %(message)s")
    config = parse_args(argv)
    result = build_index(config)
    out_p = Path(config.out_dir) / "build_result.json"
    out_p.write_text(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))
    logger.info("wrote %s, n_evidence=%d, elapsed=%.2fs",
                out_p, result.n_evidence, result.elapsed_seconds)


__all__ = [
    "BuildConfig",
    "BuildResult",
    "build_index",
    "load_evidence_jsonl",
    "main",
    "make_embedder",
    "parse_args",
]


if __name__ == "__main__":
    main(sys.argv[1:])
