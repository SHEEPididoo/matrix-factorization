from __future__ import annotations

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


def _ndcg_at_k(relevance: np.ndarray, k: int) -> float:
    # relevance: 1/0 vector of length k
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
) -> tuple[list, list]:
    """
    将用户历史拆成 train/test：
    - test: 从 rating>=threshold 的正样本里抽 n_test 个（若不足则退化为取 rating 最高的 n_test 个）
    - train: 其余 item

    返回：(train_items, test_items)
    """
    if user_hist.shape[0] <= n_test:
        # 历史太少：无法切分
        return [], []

    pos = user_hist[user_hist["rating"] >= positive_threshold]
    if pos.shape[0] >= n_test:
        test = pos.sample(n=n_test, random_state=rng)
    else:
        test = user_hist.sort_values("rating", ascending=False).head(n_test)

    test_items = test["item_id"].tolist()
    train_items = user_hist.loc[~user_hist["item_id"].isin(test_items), "item_id"].tolist()
    return train_items, test_items


def evaluate_topk(
    ratings: pd.DataFrame,
    model,
    k: int,
    positive_threshold: float,
    n_test: int,
    seed: int,
) -> TopKResult:
    # 简化版 Top-K 评估（留一法/小切分）：
    # 对每个用户：把其历史拆成 train/test
    # - train items 作为 items_known（推荐时排除）
    # - test items 作为 relevant（评估命中）
    require_columns(ratings, ["user_id", "item_id", "rating"], "ratings")

    users = ratings["user_id"].unique()
    rng = np.random.RandomState(seed)

    precisions = []
    recalls = []
    ndcgs = []

    for u in users:
        user_hist = ratings[ratings["user_id"] == u]
        train_items, test_items = _pick_test_items(
            user_hist=user_hist,
            n_test=n_test,
            positive_threshold=positive_threshold,
            rng=rng,
        )
        if not train_items or not test_items:
            continue

        relevant = set(test_items)

        # recommend 会用模型内部 item_id_map 的 keys 作为候选（更贴近真实服务）
        rec = model.recommend(user=u, amount=k, items_known=train_items, include_user=False)
        rec_items = rec["item_id"].tolist()

        hit = np.array([1 if i in relevant else 0 for i in rec_items], dtype=np.int32)
        precisions.append(float(hit.mean()) if hit.size > 0 else 0.0)
        recalls.append(float(hit.sum() / max(1, len(relevant))))
        ndcgs.append(_ndcg_at_k(hit, k=k))

    if len(precisions) == 0:
        return TopKResult(precision=0.0, recall=0.0, ndcg=0.0)

    return TopKResult(
        precision=float(np.mean(precisions)),
        recall=float(np.mean(recalls)),
        ndcg=float(np.mean(ndcgs)),
    )


def main() -> None:
    paths = Paths.from_repo(repo_root_from_this_file())

    parser = argparse.ArgumentParser(description="离线 Top-K 评估（简化版）")
    parser.add_argument(
        "--ratings",
        type=str,
        default=str(paths.data_dir / "ratings.parquet"),
        help="ratings.parquet 路径（需要 user_id,item_id,rating）",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=str(paths.artifacts_dir / "model.pkl"),
        help="训练好的模型 pickle 路径",
    )
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument(
        "--positive-threshold",
        type=float,
        default=1.0,
        help="rating >= threshold 视为 relevant",
    )
    parser.add_argument(
        "--n-test",
        type=int,
        default=1,
        help="每个用户留出的 test item 数（默认 1）",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    ratings = pd.read_parquet(Path(args.ratings))
    with Path(args.model).open("rb") as f:
        model = pickle.load(f)

    res = evaluate_topk(
        ratings=ratings,
        model=model,
        k=args.k,
        positive_threshold=args.positive_threshold,
        n_test=args.n_test,
        seed=args.seed,
    )
    print(f"Precision@{args.k}: {res.precision:.4f}")
    print(f"Recall@{args.k}:    {res.recall:.4f}")
    print(f"NDCG@{args.k}:      {res.ndcg:.4f}")


if __name__ == "__main__":
    main()

