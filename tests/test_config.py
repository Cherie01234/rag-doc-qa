"""Sanity tests for shared config (no heavy deps -> CI-safe)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config  # noqa: E402


def test_chunk_params_are_sane():
    assert isinstance(config.CHUNK_SIZE, int) and config.CHUNK_SIZE > 0
    assert isinstance(config.CHUNK_OVERLAP, int) and config.CHUNK_OVERLAP >= 0
    # Overlap must be smaller than the chunk, otherwise splitting never advances.
    assert config.CHUNK_OVERLAP < config.CHUNK_SIZE


def test_paths_and_model_set():
    assert config.DOCS_DIR and isinstance(config.DOCS_DIR, str)
    assert config.INDEX_DIR and isinstance(config.INDEX_DIR, str)
    assert "/" in config.EMBEDDING_MODEL  # e.g. "sentence-transformers/..."
