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


def evaluate_topk(
    ratings: pd.DataFrame,
    model,
    k: int,
    positive_threshold: float,
) -> TopKResult:
    # 简化版 Top-K 评估：
    # 对每个用户：用其已交互物品中“正样本”当作 relevant 集合
    # 候选集：全部 item（这在大数据集上会慢，但教学/中小数据足够）
    require_columns(ratings, ["user_id", "item_id", "rating"], "ratings")

    users = ratings["user_id"].unique()
    all_items = ratings["item_id"].unique().tolist()

    precisions = []
    recalls = []
    ndcgs = []

    for u in users:
        user_hist = ratings[ratings["user_id"] == u]
        relevant = set(user_hist.loc[user_hist["rating"] >= positive_threshold, "item_id"].tolist())
        if len(relevant) == 0:
            continue

        # recommend 会用模型内部 item_id_map 的 keys 作为候选
        # 为避免“模型没见过某些 item”导致缺失，优先走 recommend（更贴近真实服务）
        rec = model.recommend(user=u, amount=k, items_known=user_hist["item_id"].tolist(), include_user=False)
        rec_items = rec["item_id"].tolist()

        hit = np.array([1 if i in relevant else 0 for i in rec_items], dtype=np.int32)
        precisions.append(float(hit.mean()) if hit.size > 0 else 0.0)
        recalls.append(float(hit.sum() / len(relevant)))
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
        default=4.0,
        help="rating >= threshold 视为 relevant",
    )
    args = parser.parse_args()

    ratings = pd.read_parquet(Path(args.ratings))
    with Path(args.model).open("rb") as f:
        model = pickle.load(f)

    res = evaluate_topk(ratings=ratings, model=model, k=args.k, positive_threshold=args.positive_threshold)
    print(f"Precision@{args.k}: {res.precision:.4f}")
    print(f"Recall@{args.k}:    {res.recall:.4f}")
    print(f"NDCG@{args.k}:      {res.ndcg:.4f}")


if __name__ == "__main__":
    main()

