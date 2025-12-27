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


@dataclass(frozen=True)
class TfidfItemIndex:
    item_ids: list
    texts: list[str]
    vectorizer: object
    matrix: object  # scipy.sparse.csr_matrix (pickled)


def load_item_index(path: Path) -> ItemIndex:
    with path.open("rb") as f:
        obj = pickle.load(f)
    index_type = obj.get("index_type", "dense")
    if index_type == "tfidf":
        return TfidfItemIndex(
            item_ids=obj["item_ids"],
            texts=obj["texts"],
            vectorizer=obj["vectorizer"],
            matrix=obj["matrix"],
        )
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


def encode_query_for_index(index: ItemIndex | TfidfItemIndex, text: str, model_name: str | None) -> object:
    if isinstance(index, TfidfItemIndex):
        return index.vectorizer.transform([text])
    if model_name is None:
        raise ValueError("dense index 需要提供 model_name（sentence-transformers）")
    return encode_query(text, model_name=model_name)


def _dot_sims(index: ItemIndex | TfidfItemIndex, query_vec: object) -> np.ndarray:
    if isinstance(index, TfidfItemIndex):
        sims = index.matrix @ query_vec.T
        if hasattr(sims, "toarray"):
            sims = sims.toarray()
        return np.asarray(sims, dtype=np.float32).ravel()
    # embeddings 已归一化，cosine = dot
    return np.asarray(index.embeddings @ query_vec, dtype=np.float32).ravel()


def search_topk(index: ItemIndex | TfidfItemIndex, query_vec: object, k: int) -> list[tuple[object, float]]:
    sims = _dot_sims(index, query_vec)
    if k <= 0:
        return []
    k = min(k, sims.shape[0])
    top_idx = np.argpartition(-sims, kth=k - 1)[:k]
    top_sorted = top_idx[np.argsort(-sims[top_idx])]
    return [(index.item_ids[i], float(sims[i])) for i in top_sorted]


def search_topk_pos(
    index: ItemIndex | TfidfItemIndex, query_vec: object, k: int
) -> list[tuple[int, object, float]]:
    """
    与 search_topk 类似，但同时返回命中的位置 pos，便于在 UI 层读取对应 text。
    返回: [(pos, item_id, sim), ...]
    """
    sims = _dot_sims(index, query_vec)
    if k <= 0:
        return []
    k = min(k, sims.shape[0])
    top_idx = np.argpartition(-sims, kth=k - 1)[:k]
    top_sorted = top_idx[np.argsort(-sims[top_idx])]
    return [(int(i), index.item_ids[int(i)], float(sims[int(i)])) for i in top_sorted]
