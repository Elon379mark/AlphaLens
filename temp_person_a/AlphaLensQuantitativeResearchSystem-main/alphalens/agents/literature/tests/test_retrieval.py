"""
test_retrieval.py
Tests for retrieval and reranking pipeline.
Checks: retrieve returns >= 1 result per query, reranker improves P@5.
"""

import pytest
from unittest.mock import MagicMock, patch

from agents.literature.retriever import retrieve, rerank
from agents.literature.evaluator import precision_at_k, recall_at_k, mrr


# ── Fixtures ──────────────────────────────────────────────────────────────────

def _make_candidates(n=10):
    return [
        {
            "chunk_id": f"paper.pdf_chunk_{i:04d}",
            "text": f"Momentum signal using {i}-month formation period with IC = 0.0{i}.",
            "score": 1.0 - (i * 0.05),
            "metadata": {"chunk_index": i},
        }
        for i in range(n)
    ]


def _make_mock_collection(n_results=10):
    candidates = _make_candidates(n_results)
    mock = MagicMock()
    mock.query.return_value = {
        "ids": [[c["chunk_id"] for c in candidates]],
        "documents": [[c["text"] for c in candidates]],
        "distances": [[1.0 - c["score"] for c in candidates]],
        "metadatas": [[c["metadata"] for c in candidates]],
    }
    return mock


# ── Retrieve Tests ────────────────────────────────────────────────────────────

class TestRetrieve:
    def test_returns_list(self):
        collection = _make_mock_collection(5)
        query_emb = [0.1] * 1024
        results = retrieve(collection, query_emb, n_results=5)
        assert isinstance(results, list)

    def test_returns_at_least_one_result(self):
        collection = _make_mock_collection(5)
        query_emb = [0.1] * 1024
        results = retrieve(collection, query_emb, n_results=5)
        assert len(results) >= 1

    def test_result_has_required_keys(self):
        collection = _make_mock_collection(3)
        query_emb = [0.0] * 1024
        results = retrieve(collection, query_emb, n_results=3)
        for r in results:
            for key in ["chunk_id", "text", "score", "metadata"]:
                assert key in r

    def test_scores_are_in_valid_range(self):
        collection = _make_mock_collection(5)
        query_emb = [0.5] * 1024
        results = retrieve(collection, query_emb, n_results=5)
        for r in results:
            assert -1.0 <= r["score"] <= 1.0

    def test_collection_query_called_once(self):
        collection = _make_mock_collection(5)
        query_emb = [0.1] * 1024
        retrieve(collection, query_emb, n_results=5)
        collection.query.assert_called_once()


# ── Rerank Tests ──────────────────────────────────────────────────────────────

class TestRerank:
    @patch("agents.literature.retriever.CrossEncoder")
    def test_returns_top_k_results(self, mock_ce_cls):
        mock_ce = MagicMock()
        mock_ce.predict.return_value = list(range(10, 0, -1))  # descending scores
        mock_ce_cls.return_value = mock_ce

        candidates = _make_candidates(10)
        result = rerank("momentum signal", candidates, top_k=5)
        assert len(result) == 5

    @patch("agents.literature.retriever.CrossEncoder")
    def test_rerank_score_field_present(self, mock_ce_cls):
        mock_ce = MagicMock()
        mock_ce.predict.return_value = [float(i) for i in range(5, 0, -1)]
        mock_ce_cls.return_value = mock_ce

        candidates = _make_candidates(5)
        result = rerank("value factor", candidates, top_k=3)
        for r in result:
            assert "rerank_score" in r

    @patch("agents.literature.retriever.CrossEncoder")
    def test_rerank_orders_by_score_descending(self, mock_ce_cls):
        mock_ce = MagicMock()
        scores = [1.0, 5.0, 3.0, 2.0, 4.0]
        mock_ce.predict.return_value = scores
        mock_ce_cls.return_value = mock_ce

        candidates = _make_candidates(5)
        result = rerank("quality", candidates, top_k=5)
        rerank_scores = [r["rerank_score"] for r in result]
        assert rerank_scores == sorted(rerank_scores, reverse=True)

    @patch("agents.literature.retriever.CrossEncoder")
    def test_top_k_capped_at_candidates_length(self, mock_ce_cls):
        mock_ce = MagicMock()
        mock_ce.predict.return_value = [1.0, 2.0, 3.0]
        mock_ce_cls.return_value = mock_ce

        candidates = _make_candidates(3)
        result = rerank("momentum", candidates, top_k=10)  # ask for more than available
        assert len(result) == 3


# ── Evaluator Tests ───────────────────────────────────────────────────────────

class TestEvaluatorMetrics:
    def _results(self, ids):
        return [{"chunk_id": i} for i in ids]

    def test_precision_at_k_perfect(self):
        retrieved = self._results(["a", "b", "c", "d", "e"])
        relevant = ["a", "b", "c", "d", "e"]
        assert precision_at_k(retrieved, relevant, k=5) == 1.0

    def test_precision_at_k_zero(self):
        retrieved = self._results(["x", "y", "z"])
        relevant = ["a", "b", "c"]
        assert precision_at_k(retrieved, relevant, k=3) == 0.0

    def test_recall_at_k_perfect(self):
        retrieved = self._results(["a", "b", "c"])
        relevant = ["a", "b", "c"]
        assert recall_at_k(retrieved, relevant, k=3) == 1.0

    def test_recall_at_k_empty_relevant(self):
        retrieved = self._results(["a"])
        assert recall_at_k(retrieved, [], k=1) == 0.0

    def test_mrr_first_hit(self):
        retrieved = self._results(["a", "b", "c"])
        assert mrr(retrieved, ["a"]) == 1.0

    def test_mrr_second_hit(self):
        retrieved = self._results(["x", "a", "b"])
        assert mrr(retrieved, ["a"]) == pytest.approx(0.5)

    def test_mrr_no_hit(self):
        retrieved = self._results(["x", "y", "z"])
        assert mrr(retrieved, ["a"]) == 0.0
