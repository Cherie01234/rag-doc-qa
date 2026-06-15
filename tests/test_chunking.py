"""Test the text-splitting behavior used by ingest (no embedding model needed)."""
import sys
from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config  # noqa: E402


def test_long_text_is_chunked_within_size():
    text = "これはテスト用の文章です。RAGのチャンク分割を検証します。" * 200
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
    )
    chunks = splitter.split_text(text)

    assert len(chunks) > 1                                  # long text -> many chunks
    assert all(len(c) <= config.CHUNK_SIZE for c in chunks)  # respects chunk_size


def test_short_text_is_single_chunk():
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
    )
    assert splitter.split_text("短い文。") == ["短い文。"]
