"""Tests for the retrieval evaluation set and metrics (no embedding model needed -> CI-safe)."""
import json
import sys
from pathlib import Path

import pytest
from langchain_core.documents import Document

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "eval"))
import config  # noqa: E402
import run_eval  # noqa: E402

DOCS = {d.metadata["source"]: d.page_content for d in run_eval.load_corpus()}
QUESTIONS = run_eval.load_questions()


def test_questions_are_well_formed():
    assert len(QUESTIONS) >= 30
    ids = [q["id"] for q in QUESTIONS]
    assert len(ids) == len(set(ids))
    for q in QUESTIONS:
        assert q["question"].strip() and q["evidence"].strip()


def test_evidence_appears_exactly_once_in_its_document():
    for q in QUESTIONS:
        assert q["doc"] in DOCS, q["id"]
        assert DOCS[q["doc"]].count(q["evidence"]) == 1, q["id"]


@pytest.mark.parametrize("chunk_size", run_eval.CHUNK_SIZES)
def test_every_evidence_fits_in_some_chunk(chunk_size):
    # If a split cut through an evidence span, that question could never be a hit.
    chunks = run_eval.split_documents(run_eval.load_corpus(), chunk_size, config.CHUNK_OVERLAP)
    assert len({c.metadata["chunk_id"] for c in chunks}) == len(chunks)
    for q in QUESTIONS:
        assert any(run_eval.is_relevant(c, q) for c in chunks), (q["id"], chunk_size)


def test_rank_and_metrics():
    q = {"doc": "a.txt", "evidence": "正解"}
    retrieved = [
        Document(page_content="正解を含むが別の文書", metadata={"source": "b.txt"}),
        Document(page_content="無関係", metadata={"source": "a.txt"}),
        Document(page_content="ここに正解がある", metadata={"source": "a.txt"}),
    ]
    assert run_eval.first_relevant_rank(retrieved, q) == 3
    assert run_eval.first_relevant_rank(retrieved[:2], q) is None

    ranks = [1, 3, None, 2]
    assert run_eval.hit_at_k(ranks, 1) == 0.25
    assert run_eval.hit_at_k(ranks, 3) == 0.75
    assert run_eval.mrr(ranks) == pytest.approx((1 + 1 / 3 + 0 + 1 / 2) / 4)
    assert run_eval.random_hit_at_k(10, 1, 3) == pytest.approx(0.3)
    assert run_eval.random_hit_at_k(5, 1, 10) == 1.0


def test_results_file_matches_current_dataset_and_config():
    # Guards against a stale results.json after editing questions or config.py.
    results = json.loads(run_eval.RESULTS_PATH.read_text(encoding="utf-8"))
    assert results["dataset"]["n_questions"] == len(QUESTIONS)
    assert results["dataset"]["n_documents"] == len(DOCS)

    app_runs = [r for r in results["runs"] if r["is_app_setting"]]
    assert len(app_runs) == 1
    assert app_runs[0]["embedding_model"] == config.EMBEDDING_MODEL
    assert app_runs[0]["chunk_size"] == config.CHUNK_SIZE
    assert app_runs[0]["chunk_overlap"] == config.CHUNK_OVERLAP

    question_ids = {q["id"] for q in QUESTIONS}
    for run in results["runs"]:
        assert set(run["ranks"]) == question_ids
        hits = [run["metrics"][f"hit@{k}"] for k in results["settings"]["k_values"]]
        assert hits == sorted(hits)  # Hit@k never decreases with k
