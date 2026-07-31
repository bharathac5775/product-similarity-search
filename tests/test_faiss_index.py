"""Tests for the FAISS HNSW fast path (Part 3).

The approximate (FAISS) ranking should closely agree with the exact
brute-force cosine ranking.
"""
from product_similarity.config import Settings
from product_similarity.engine import SimilarityEngine


def test_faiss_agrees_with_exact(sample_ldjson_path):
    exact = SimilarityEngine.from_path(sample_ldjson_path, Settings(use_faiss=False))
    approx = SimilarityEngine.from_path(sample_ldjson_path, Settings(use_faiss=True))

    a = set(exact.find_similar_products("p1", 3))
    b = set(approx.find_similar_products("p1", 3))
    # On this tiny set, allow a small approximation slack.
    assert len(a & b) >= 2
