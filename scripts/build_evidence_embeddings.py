"""跑真 sentence-transformers + FAISS 索引 (RUNNING 启动用).

执行机器: gpu_server
路径策略 (依 disk_budget.md Strategy B):
  - embeddings.npy + .faiss → /tmp/culturelens_rc/embeddings/  (重启丢失 OK)
  - items_meta.jsonl → 同上 (不大, 1-5 MB)
  - 仅最终 CultureLens-RC 配置的 FAISS index 落到 persistent (实施 RUNNING 阶段再决定)

用法:
  # dry run with identity embedder (sanity, < 5 sec)
  python -m scripts.build_evidence_embeddings --dry-run

  # 真跑 (RUNNING 阶段, Researcher 启动信号后)
  python -m scripts.build_evidence_embeddings \\
    --evidence ${EXPERIMENT_ROOT}/data/evidence/evidence.jsonl \\
    --out-dir /tmp/culturelens_rc/embeddings \\
    --embedder sentence_transformers \\
    --model sentence-transformers/all-MiniLM-L6-v2 \\
    --batch-size 128

依赖:
  pip install sentence-transformers faiss-cpu

faiss-cpu 不在 culturelens_rc_env 默认装中, 跑前提示是否安装.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.build_retrieval_index import BuildConfig, build_index  # noqa: E402


def check_deps(embedder_kind: str) -> dict[str, bool]:
    """检查依赖, 不安装. 返回 {package: installed}."""
    status = {}
    try:
        import sentence_transformers  # noqa: F401
        status["sentence_transformers"] = True
    except ImportError:
        status["sentence_transformers"] = False
    try:
        import faiss  # noqa: F401
        status["faiss"] = True
    except ImportError:
        status["faiss"] = False
    return status


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="build evidence embeddings + FAISS index")
    p.add_argument("--evidence",
                   default="${EXPERIMENT_ROOT}/data/evidence/evidence.jsonl",
                   help="data_scientist evidence jsonl path")
    p.add_argument("--out-dir", default="/tmp/culturelens_rc/embeddings",
                   help="Strategy B: /tmp 重启丢失 OK")
    p.add_argument("--embedder", default="sentence_transformers",
                   choices=["identity", "sentence_transformers"])
    p.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2",
                   help="anchor_5 已用此模型, 保持一致")
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--dry-run", action="store_true",
                   help="用 identity embedder 走通流程不下载真模型")
    a = p.parse_args(argv)

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s | %(levelname)s | %(message)s")

    # Dry-run 等同 --embedder identity
    embedder_kind = "identity" if a.dry_run else a.embedder

    # 检查依赖
    deps = check_deps(embedder_kind)
    logger.info("deps status: %s", deps)
    if embedder_kind == "sentence_transformers" and not deps["sentence_transformers"]:
        logger.error("sentence-transformers not installed; pip install sentence-transformers")
        sys.exit(1)
    if not deps["faiss"]:
        logger.warning("faiss not installed; will fallback to npy index. "
                       "Install with: pip install faiss-cpu")

    config = BuildConfig(
        evidence_path=a.evidence,
        out_dir=a.out_dir,
        embedder_kind=embedder_kind,
        embedder_model=a.model,
        batch_size=a.batch_size,
        index_kinds=("general", "hierarchical"),
        cache_embeddings=True,
    )

    logger.info("starting build_index with config: %s", config)
    result = build_index(config)

    # 写 build_result.json
    out_p = Path(a.out_dir) / "build_result.json"
    out_p.parent.mkdir(parents=True, exist_ok=True)
    out_p.write_text(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))

    logger.info("=== DONE ===")
    logger.info("n_evidence: %d", result.n_evidence)
    logger.info("embedding_cache: %s", result.embedding_cache_path)
    logger.info("index_paths: %s", result.index_paths)
    logger.info("elapsed: %.2f s", result.elapsed_seconds)
    logger.info("result manifest: %s", out_p)


if __name__ == "__main__":
    main(sys.argv[1:])
