from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import torch

try:
    import faiss
except ImportError:  # pragma: no cover
    faiss = None


@torch.no_grad()
def retrieve_for_users(
    user_vectors: torch.Tensor,
    item_vectors: torch.Tensor,
    idx_to_item: list[str],
    seen_idx: list[set[int]],
    k: int,
) -> list[list[str]]:
    """Exact search over the full catalog. No sampled negatives.

    user_vectors: [U, D], item_vectors: [N, D] including pad row 0.
    """
    scores = user_vectors @ item_vectors.t()
    scores[:, item_vectors.norm(dim=-1) < 1e-8] = -1e9
    scores[:, 0] = -1e9
    for row, banned in enumerate(seen_idx):
        if banned:
            idx = torch.tensor(list(banned), device=scores.device, dtype=torch.long)
            scores[row, idx] = -1e9
    topk = torch.topk(scores, k=min(k, scores.size(1) - 1), dim=1).indices.cpu().numpy()
    out: list[list[str]] = []
    for row in topk:
        out.append([idx_to_item[i] for i in row if i != 0])
    return out


@torch.no_grad()
def retrieve_scored(
    user_vectors: torch.Tensor,
    item_vectors: torch.Tensor,
    idx_to_item: list[str],
    seen_idx: list[set[int]],
    k: int,
) -> list[list[tuple[str, float]]]:
    """Exact search returning (item_id, inner_product) pairs, best first."""
    scores = user_vectors @ item_vectors.t()
    scores[:, item_vectors.norm(dim=-1) < 1e-8] = -1e9
    scores[:, 0] = -1e9
    for row, banned in enumerate(seen_idx):
        if banned:
            idx = torch.tensor(list(banned), device=scores.device, dtype=torch.long)
            scores[row, idx] = -1e9
    k_eff = min(k, scores.size(1) - 1)
    topk = torch.topk(scores, k=k_eff, dim=1)
    inds = topk.indices.cpu().numpy()
    vals = topk.values.cpu().numpy()
    out: list[list[tuple[str, float]]] = []
    for row in range(inds.shape[0]):
        recs: list[tuple[str, float]] = []
        for i, v in zip(inds[row], vals[row]):
            if int(i) == 0:
                continue
            recs.append((idx_to_item[int(i)], float(v)))
        out.append(recs)
    return out


@dataclass
class AnnBuildResult:
    backend: str
    n_vectors: int
    dim: int
    build_ms: float
    extra: dict


class FaissHnswIndex:
    """HNSW over L2-normalized item vectors, inner-product metric. Pad row 0 is excluded."""

    def __init__(self, m: int = 32, ef_construction: int = 200, ef_search: int = 64):
        if faiss is None:
            raise RuntimeError("faiss is not installed")
        self.m = m
        self.ef_construction = ef_construction
        self.ef_search = ef_search
        self.index = None
        self.row_to_item_idx: np.ndarray | None = None

    def build(self, item_vectors: np.ndarray, skip_pad: bool = True) -> AnnBuildResult:
        vecs = np.ascontiguousarray(item_vectors.astype(np.float32))
        if skip_pad:
            self.row_to_item_idx = np.arange(1, vecs.shape[0], dtype=np.int64)
            vecs = vecs[1:]
        else:
            self.row_to_item_idx = np.arange(vecs.shape[0], dtype=np.int64)
        dim = int(vecs.shape[1])
        index = faiss.IndexHNSWFlat(dim, self.m, faiss.METRIC_INNER_PRODUCT)
        index.hnsw.efConstruction = self.ef_construction
        t0 = time.perf_counter()
        index.add(vecs)
        build_ms = (time.perf_counter() - t0) * 1000.0
        index.hnsw.efSearch = self.ef_search
        self.index = index
        return AnnBuildResult(
            backend="faiss.IndexHNSWFlat",
            n_vectors=int(vecs.shape[0]),
            dim=dim,
            build_ms=float(build_ms),
            extra={"m": self.m, "ef_construction": self.ef_construction, "ef_search": self.ef_search},
        )

    def search(self, queries: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        if self.index is None or self.row_to_item_idx is None:
            raise RuntimeError("index is not built")
        q = np.ascontiguousarray(queries.astype(np.float32))
        scores, rows = self.index.search(q, k)
        item_idx = self.row_to_item_idx[np.clip(rows, 0, len(self.row_to_item_idx) - 1)]
        item_idx = np.where(rows < 0, 0, item_idx)
        return scores, item_idx


def overlap_at_k(exact_idx: np.ndarray, ann_idx: np.ndarray, k: int) -> float:
    """Mean |exact[:k] ∩ ann[:k]| / k over rows."""
    hits = []
    for a, b in zip(exact_idx, ann_idx):
        sa = set(int(x) for x in a[:k] if int(x) != 0)
        sb = set(int(x) for x in b[:k] if int(x) != 0)
        if k == 0:
            continue
        hits.append(len(sa & sb) / float(k))
    return float(np.mean(hits)) if hits else float("nan")
