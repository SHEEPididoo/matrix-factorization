from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class ItemIndex:
    item_ids: list
    texts: list[str]
    embeddings: np.ndarray  # shape: (n_items, dim) normalized


def load_item_index(path: Path) -> ItemIndex:
    with path.open("rb") as f:
        obj = pickle.load(f)
    return ItemIndex(
        item_ids=obj["item_ids"],
        texts=obj["texts"],
        embeddings=np.asarray(obj["embeddings"], dtype=np.float32),
    )


def _safe_import_sentence_transformers():
    try:
        from sentence_transformers import SentenceTransformer  # type: ignore
    except Exception as e:  # pragma: no cover
        raise ImportError(
            "未安装 sentence-transformers。请运行：pip install -U sentence-transformers"
        ) from e
    return SentenceTransformer


def encode_query(text: str, model_name: str) -> np.ndarray:
    SentenceTransformer = _safe_import_sentence_transformers()
    model = SentenceTransformer(model_name)
    q = model.encode([text], convert_to_numpy=True, normalize_embeddings=True)
    return np.asarray(q[0], dtype=np.float32)


def search_topk(index: ItemIndex, query_vec: np.ndarray, k: int) -> list[tuple[object, float]]:
    # embeddings 已归一化，cosine = dot
    sims = index.embeddings @ query_vec
    if k <= 0:
        return []
    k = min(k, sims.shape[0])
    top_idx = np.argpartition(-sims, kth=k - 1)[:k]
    top_sorted = top_idx[np.argsort(-sims[top_idx])]
    return [(index.item_ids[i], float(sims[i])) for i in top_sorted]

