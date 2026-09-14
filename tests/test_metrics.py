from __future__ import annotations

from eval.metrics import ndcg_at_k, recall_at_k


def test_recall_perfect_and_miss():
    assert recall_at_k(["a", "b", "c"], {"a"}, k=10) == 1.0
    assert recall_at_k(["b", "c"], {"a"}, k=10) == 0.0
    assert recall_at_k(["a", "b"], {"a", "c"}, k=1) == 0.5


def test_ndcg_discount():
    perfect = ndcg_at_k(["a", "b"], {"a"}, k=10)
    later = ndcg_at_k(["b", "a"], {"a"}, k=10)
    miss = ndcg_at_k(["b", "c"], {"a"}, k=10)
    assert perfect == 1.0
    assert later < perfect
    assert miss == 0.0
