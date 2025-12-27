from __future__ import annotations

"""
Week6：Hybrid 离线评估（Embedding 召回 + 模型 rerank/混合）

简化实现（教学/轻量数据可用）：
- 对每个用户：从其历史中切出 test_items（relevant），其余为 train_items（items_known）
- 用户画像 embedding：train_items 的 item embedding 均值（仅使用在索引中存在的 items）
- 候选召回：对所有 item 计算 cosine（已归一化 => dot），取 candidate_k
- 排序：对候选 item 用模型预测打分（或与 embedding 分数混合），取 Top-K
- 评估：Precision/Recall/NDCG@K

输入：
- project_template/data/ratings.parquet
- project_template/artifacts/model.pkl
- project_template/artifacts/item_index.pkl
"""

import argparse
import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .common import Paths, repo_root_from_this_file, require_columns


@dataclass(frozen=True)
class TopKResult:
    precision: float
    recall: float
    ndcg: float


def _ndcg_at_k(relevance: np.ndarray) -> float:
    if relevance.size == 0:
        return 0.0
    gains = (2 ** relevance - 1) / np.log2(np.arange(2, relevance.size + 2))
    dcg = float(np.sum(gains))
    ideal = np.sort(relevance)[::-1]
    ideal_gains = (2 ** ideal - 1) / np.log2(np.arange(2, ideal.size + 2))
    idcg = float(np.sum(ideal_gains))
    return dcg / idcg if idcg > 0 else 0.0


def _pick_test_items(
    user_hist: pd.DataFrame,
    n_test: int,
    positive_threshold: float,
    rng: np.random.RandomState,
) -> tuple[list[str], list[str]]:
    if user_hist.shape[0] <= n_test:
        return [], []

    pos = user_hist[user_hist["rating"] >= positive_threshold]
    if pos.shape[0] >= n_test:
        test = pos.sample(n=n_test, random_state=rng)
    else:
        test = user_hist.sort_values("rating", ascending=False).head(n_test)

    test_items = test["item_id"].astype(str).tolist()
    train_items = (
        user_hist.loc[~user_hist["item_id"].astype(str).isin(test_items), "item_id"]
        .astype(str)
        .tolist()
    )
    return train_items, test_items


def _minmax(x: np.ndarray) -> np.ndarray:
    if x.size == 0:
        return x
    lo = float(np.min(x))
    hi = float(np.max(x))
    if hi - lo < 1e-8:
        return np.zeros_like(x, dtype=np.float32)
    return ((x - lo) / (hi - lo)).astype(np.float32)


def evaluate_hybrid(
    ratings: pd.DataFrame,
    model,
    item_ids: list[str],
    item_emb: np.ndarray,
    k: int,
    candidate_k: int,
    alpha: float,
    positive_threshold: float,
    n_test: int,
    seed: int,
    min_profile_items: int,
) -> TopKResult:
    require_columns(ratings, ["user_id", "item_id", "rating"], "ratings")
    rng = np.random.RandomState(seed)

    # index: item_id -> row
    idx = {it: j for j, it in enumerate(item_ids)}
    n_items, dim = item_emb.shape

    precisions: list[float] = []
    recalls: list[float] = []
    ndcgs: list[float] = []

    users = ratings["user_id"].astype(str).unique()
    for u in users:
        user_hist = ratings[ratings["user_id"].astype(str) == u].copy()
        user_hist["item_id"] = user_hist["item_id"].astype(str)
        user_hist["rating"] = pd.to_numeric(user_hist["rating"], errors="coerce").fillna(0.0)

        train_items, test_items = _pick_test_items(
            user_hist=user_hist,
            n_test=n_test,
            positive_threshold=positive_threshold,
            rng=rng,
        )
        if not train_items or not test_items:
            continue

        # 构造用户画像 embedding（均值）
        prof_rows = [idx[it] for it in train_items if it in idx]
        if len(prof_rows) < min_profile_items:
            continue
        prof = item_emb[np.array(prof_rows)].mean(axis=0)
        # 归一化，确保 dot=cosine
        norm = float(np.linalg.norm(prof))
        if norm > 0:
            prof = prof / norm
        prof = prof.astype(np.float32)

        # 召回候选（包含所有 item，再过滤 train_items）
        sims = item_emb @ prof  # (n_items,)
        cand_k = min(candidate_k, n_items)
        top_idx = np.argpartition(-sims, kth=cand_k - 1)[:cand_k]
        top_idx = top_idx[np.argsort(-sims[top_idx])]
        cand_item_ids = [item_ids[int(j)] for j in top_idx if item_ids[int(j)] not in set(train_items)]
        cand_sims = np.asarray([float(sims[int(j)]) for j in top_idx if item_ids[int(j)] not in set(train_items)], dtype=np.float32)

        if len(cand_item_ids) == 0:
            continue

        # 模型打分（若 model 为空则退化为纯 embedding）
        if model is None:
            model_scores = np.zeros(len(cand_item_ids), dtype=np.float32)
        else:
            X = pd.DataFrame({"user_id": [u] * len(cand_item_ids), "item_id": cand_item_ids})
            preds = model.predict(X, bound_ratings=False)
            model_scores = np.asarray(preds, dtype=np.float32)

        score = alpha * _minmax(model_scores) + (1.0 - alpha) * _minmax(cand_sims)
        topk_idx = np.argsort(-score)[: min(k, len(cand_item_ids))]
        rec_items = [cand_item_ids[int(j)] for j in topk_idx]

        relevant = set(test_items)
        hit = np.array([1 if it in relevant else 0 for it in rec_items], dtype=np.int32)
        precisions.append(float(hit.mean()) if hit.size else 0.0)
        recalls.append(float(hit.sum() / max(1, len(relevant))))
        ndcgs.append(_ndcg_at_k(hit))

    if not precisions:
        return TopKResult(precision=0.0, recall=0.0, ndcg=0.0)
    return TopKResult(
        precision=float(np.mean(precisions)),
        recall=float(np.mean(recalls)),
        ndcg=float(np.mean(ndcgs)),
    )


def main() -> None:
    paths = Paths.from_repo(repo_root_from_this_file())
    parser = argparse.ArgumentParser(description="Hybrid 离线评估：embedding 召回 + 模型混合排序")
    parser.add_argument("--ratings", type=str, default=str(paths.data_dir / "ratings.parquet"))
    parser.add_argument("--model", type=str, default=str(paths.artifacts_dir / "model.pkl"))
    parser.add_argument("--item-index", type=str, default=str(paths.artifacts_dir / "item_index.pkl"))
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--candidate-k", type=int, default=200)
    parser.add_argument("--alpha", type=float, default=0.7)
    parser.add_argument("--positive-threshold", type=float, default=2.0)
    parser.add_argument("--n-test", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-profile-items", type=int, default=3)
    args = parser.parse_args()

    ratings = pd.read_parquet(Path(args.ratings))

    model = None
    model_path = Path(args.model)
    if model_path.exists():
        with model_path.open("rb") as f:
            model = pickle.load(f)

    with Path(args.item_index).open("rb") as f:
        obj = pickle.load(f)
    item_ids = [str(x) for x in obj["item_ids"]]
    emb = np.asarray(obj["embeddings"], dtype=np.float32)

    res = evaluate_hybrid(
        ratings=ratings,
        model=model,
        item_ids=item_ids,
        item_emb=emb,
        k=args.k,
        candidate_k=args.candidate_k,
        alpha=float(args.alpha),
        positive_threshold=float(args.positive_threshold),
        n_test=int(args.n_test),
        seed=int(args.seed),
        min_profile_items=int(args.min_profile_items),
    )
    print(f"Hybrid Precision@{args.k}: {res.precision:.4f}")
    print(f"Hybrid Recall@{args.k}:    {res.recall:.4f}")
    print(f"Hybrid NDCG@{args.k}:      {res.ndcg:.4f}")


if __name__ == "__main__":
    main()

