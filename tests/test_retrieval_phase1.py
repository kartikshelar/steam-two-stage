from __future__ import annotations

import numpy as np
import torch

from retrieval.index import FaissHnswIndex, overlap_at_k
from retrieval.two_tower import TwoTower


def test_id_only_two_tower_forward():
    m = TwoTower(5, 7, dim=8, use_content=False)
    u = torch.tensor([1, 2])
    i = torch.tensor([3, 4])
    loss = m.inbatch_loss(u, i, log_q=None, temperature=0.07)
    assert torch.isfinite(loss)


def test_content_two_tower_shapes():
    m = TwoTower(5, 7, dim=8, use_content=True, n_genres=4, n_tags=4)
    u = torch.tensor([1, 2])
    i = torch.tensor([3, 4])
    g = torch.tensor([[1, 0], [2, 0]])
    t = torch.tensor([[1, 0], [0, 0]])
    n = torch.zeros(2, 4)
    z = m.encode_items(i, genre_ids=g, tag_ids=t, numeric=n)
    assert z.shape == (2, 8)
    loss = m.inbatch_loss(u, i, log_q=None, temperature=0.07, genre_ids=g, tag_ids=t, numeric=n, id_dropout=0.5)
    assert torch.isfinite(loss)


def test_faiss_hnsw_overlap_on_normalized_vectors():
    rng = np.random.RandomState(0)
    vecs = rng.randn(64, 8).astype(np.float32)
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-8
    vecs[0] = 0
    index = FaissHnswIndex(m=8, ef_construction=50, ef_search=32)
    index.build(vecs)
    q = vecs[1:6]
    scores = q @ vecs.T
    scores[:, 0] = -1e9
    exact = np.argsort(-scores, axis=1)[:, :10]
    _, ann = index.search(q, 10)
    ov = overlap_at_k(exact, ann, 10)
    assert ov >= 0.5
