"""
Unit tests for the RAG engine (no API calls required).

Run:
    pytest tests/
"""

import math
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from rag_engine import VectorStore


# ---------------------------------------------------------------------------
# VectorStore unit tests (all local, no network)
# ---------------------------------------------------------------------------

@pytest.fixture()
def store():
    """Return a VectorStore populated with 3 toy entries."""
    vs = VectorStore()
    # Patch the embedder so tests don't download the model
    with patch("rag_engine._get_embedder") as mock_embed:
        # Each call to .encode() returns a simple deterministic float list
        call_count = [0]
        def fake_encode(text, **kwargs):
            """
            Returns a unit vector that encodes the *position* of the call.
            Entry 0 → [1, 0, 0], entry 1 → [0, 1, 0], entry 2 → [0, 0, 1].
            """
            import numpy as np
            idx = call_count[0] % 3
            call_count[0] += 1
            v = [0.0, 0.0, 0.0]
            v[idx] = 1.0
            return np.array(v)

        mock_embed.return_value.encode = fake_encode

        vs.add("IVF question", "IVF answer")
        vs.add("sperm question", "sperm answer")
        vs.add("diet question", "diet answer")

    return vs, mock_embed


class TestCosine:
    def test_identical_vectors(self):
        assert VectorStore._cosine_similarity([1, 0, 0], [1, 0, 0]) == pytest.approx(1.0)

    def test_orthogonal_vectors(self):
        assert VectorStore._cosine_similarity([1, 0, 0], [0, 1, 0]) == pytest.approx(0.0)

    def test_opposite_vectors(self):
        assert VectorStore._cosine_similarity([1, 0, 0], [-1, 0, 0]) == pytest.approx(-1.0)

    def test_zero_vector_returns_zero(self):
        assert VectorStore._cosine_similarity([0, 0, 0], [1, 2, 3]) == 0.0

    def test_normalized(self):
        # [3, 4, 0] and [6, 8, 0] point in the same direction
        assert VectorStore._cosine_similarity([3, 4, 0], [6, 8, 0]) == pytest.approx(1.0)


class TestVectorStoreSearch:
    def test_top_k_respected(self, store):
        vs, mock_embed = store
        import numpy as np

        with patch("rag_engine._get_embedder") as m:
            m.return_value.encode.return_value = np.array([1.0, 0.0, 0.0])
            results = vs.search("some query", top_k=2)

        assert len(results) == 2

    def test_best_match_is_first(self, store):
        vs, mock_embed = store
        import numpy as np

        # Query vector aligned with entry 1 ([0, 1, 0])
        with patch("rag_engine._get_embedder") as m:
            m.return_value.encode.return_value = np.array([0.0, 1.0, 0.0])
            results = vs.search("some query", top_k=3)

        assert results[0]["question"] == "sperm question"
        assert results[0]["score"] == pytest.approx(1.0)

    def test_scores_descending(self, store):
        vs, mock_embed = store
        import numpy as np

        with patch("rag_engine._get_embedder") as m:
            m.return_value.encode.return_value = np.array([0.9, 0.3, 0.1])
            results = vs.search("mixed query", top_k=3)

        scores = [r["score"] for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_empty_store_raises(self):
        vs = VectorStore()
        with pytest.raises(RuntimeError, match="empty"):
            vs.search("anything")

    def test_chunk_keys_present(self, store):
        vs, mock_embed = store
        import numpy as np

        with patch("rag_engine._get_embedder") as m:
            m.return_value.encode.return_value = np.array([1.0, 0.0, 0.0])
            results = vs.search("query", top_k=1)

        assert "question" in results[0]
        assert "answer" in results[0]
        assert "score" in results[0]


class TestBuildFromFile:
    def test_loads_correct_count(self, tmp_path):
        import json
        import numpy as np

        faqs = [
            {"question": f"Q{i}", "answer": f"A{i}"} for i in range(5)
        ]
        path = tmp_path / "faqs.json"
        path.write_text(json.dumps(faqs))

        vs = VectorStore()
        call_count = [0]

        with patch("rag_engine._get_embedder") as m:
            def fake_encode(text, **kwargs):
                v = [0.0] * 8
                v[call_count[0] % 8] = 1.0
                call_count[0] += 1
                return np.array(v)

            m.return_value.encode = fake_encode
            vs.build_from_file(path)

        assert len(vs.chunks) == 5
        assert len(vs.embeddings) == 5
