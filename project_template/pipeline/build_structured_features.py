from __future__ import annotations

"""
Week5：结构化特征工程（轻量版）

输入：
- project_template/data/ratings.parquet  (user_id,item_id,rating)
- （可选）project_template/data/items.parquet

输出（缓存到 features/）：
- project_template/features/user_features.parquet
- project_template/features/item_features.parquet

这些特征可用于：
- 简单 rerank（热门/活跃度/均值）
- 解释（“因为你偏好高时长/你常玩某类型”）
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from .common import Paths, repo_root_from_this_file, require_columns


def build_features(ratings_path: Path, out_dir: Path) -> tuple[Path, Path]:
    ratings = pd.read_parquet(ratings_path)
    require_columns(ratings, ["user_id", "item_id", "rating"], "ratings")

    # 统一类型，避免 join/序列化问题
    ratings = ratings.copy()
    ratings["user_id"] = ratings["user_id"].astype(str)
    ratings["item_id"] = ratings["item_id"].astype(str)
    ratings["rating"] = pd.to_numeric(ratings["rating"], errors="coerce").fillna(0.0).astype(np.float32)

    # item 侧：流行度、均值、方差
    item_agg = ratings.groupby("item_id")["rating"].agg(
        item_interactions="count",
        item_rating_mean="mean",
        item_rating_std="std",
        item_rating_min="min",
        item_rating_max="max",
    ).reset_index()
    item_agg["item_rating_std"] = item_agg["item_rating_std"].fillna(0.0).astype(np.float32)

    # user 侧：活跃度、均值、方差
    user_agg = ratings.groupby("user_id")["rating"].agg(
        user_interactions="count",
        user_rating_mean="mean",
        user_rating_std="std",
        user_rating_min="min",
        user_rating_max="max",
    ).reset_index()
    user_agg["user_rating_std"] = user_agg["user_rating_std"].fillna(0.0).astype(np.float32)

    # 全局统计（可用于归一化/解释）
    global_mean = float(ratings["rating"].mean())
    global_std = float(ratings["rating"].std()) if float(ratings["rating"].std()) == float(ratings["rating"].std()) else 0.0
    user_agg["global_rating_mean"] = global_mean
    user_agg["global_rating_std"] = global_std
    item_agg["global_rating_mean"] = global_mean
    item_agg["global_rating_std"] = global_std

    out_dir.mkdir(parents=True, exist_ok=True)
    out_user = out_dir / "user_features.parquet"
    out_item = out_dir / "item_features.parquet"
    user_agg.to_parquet(out_user, index=False)
    item_agg.to_parquet(out_item, index=False)
    return out_user, out_item


def main() -> None:
    paths = Paths.from_repo(repo_root_from_this_file())
    parser = argparse.ArgumentParser(description="生成结构化 user/item 特征（缓存到 features/）")
    parser.add_argument(
        "--ratings",
        type=str,
        default=str(paths.data_dir / "ratings.parquet"),
        help="ratings.parquet 路径",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=str(paths.features_dir),
        help="输出目录（默认 project_template/features）",
    )
    args = parser.parse_args()

    out_user, out_item = build_features(Path(args.ratings), Path(args.out_dir))
    print(f"[OK] user features saved to: {out_user}")
    print(f"[OK] item features saved to: {out_item}")


if __name__ == "__main__":
    main()

