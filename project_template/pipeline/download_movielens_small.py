from __future__ import annotations

"""
下载并准备 MovieLens 小数据集（ml-latest-small）。

产出：
- project_template/data/ratings.parquet  (user_id,item_id,rating,timestamp)
- project_template/data/items.parquet    (item_id,text)

说明：
- 仅用于教学与示例跑通，不建议作为最终项目唯一数据来源。
"""

import argparse
import io
import zipfile
from pathlib import Path
from urllib.request import urlopen

import pandas as pd

from .common import Paths, repo_root_from_this_file


MOVIELENS_SMALL_ZIP_URL = "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip"


def _download_zip_bytes(url: str) -> bytes:
    with urlopen(url) as r:  # nosec - 教学用途下载公开数据集
        return r.read()


def prepare_movielens_small(
    out_ratings: Path,
    out_items: Path,
    sample_users: int | None,
    min_interactions: int,
) -> None:
    data = _download_zip_bytes(MOVIELENS_SMALL_ZIP_URL)
    zf = zipfile.ZipFile(io.BytesIO(data))

    # zip 内路径形如：ml-latest-small/ratings.csv
    with zf.open("ml-latest-small/ratings.csv") as f:
        ratings = pd.read_csv(f)
    with zf.open("ml-latest-small/movies.csv") as f:
        movies = pd.read_csv(f)

    # 规范化列名到模板契约
    ratings = ratings.rename(
        columns={
            "userId": "user_id",
            "movieId": "item_id",
            "rating": "rating",
            "timestamp": "timestamp",
        }
    )
    movies = movies.rename(columns={"movieId": "item_id"})

    # 构造 text（用于 embedding/LLM）
    # 这里用 title + genres 拼接，足够演示
    movies["text"] = (
        movies["title"].fillna("").astype(str)
        + " | genres: "
        + movies["genres"].fillna("").astype(str).str.replace("|", ", ", regex=False)
    )

    # 过滤低交互用户（可选）
    if min_interactions > 1:
        cnt = ratings.groupby("user_id")["item_id"].count()
        keep_users = cnt[cnt >= min_interactions].index
        ratings = ratings[ratings["user_id"].isin(keep_users)].copy()

    # 抽样用户（用于更快跑通）
    if sample_users is not None:
        users = ratings["user_id"].drop_duplicates().sample(
            n=min(sample_users, ratings["user_id"].nunique()), random_state=42
        )
        ratings = ratings[ratings["user_id"].isin(users)].copy()

    # 只保留出现过的 item
    keep_items = set(ratings["item_id"].unique().tolist())
    movies = movies[movies["item_id"].isin(keep_items)].copy()

    out_ratings.parent.mkdir(parents=True, exist_ok=True)
    out_items.parent.mkdir(parents=True, exist_ok=True)

    ratings[["user_id", "item_id", "rating", "timestamp"]].to_parquet(out_ratings, index=False)
    movies[["item_id", "text"]].to_parquet(out_items, index=False)


def main() -> None:
    paths = Paths.from_repo(repo_root_from_this_file())

    parser = argparse.ArgumentParser(description="下载并生成 MovieLens 小样本数据（parquet）")
    parser.add_argument(
        "--out-ratings",
        type=str,
        default=str(paths.data_dir / "ratings.parquet"),
        help="输出 ratings.parquet 路径",
    )
    parser.add_argument(
        "--out-items",
        type=str,
        default=str(paths.data_dir / "items.parquet"),
        help="输出 items.parquet 路径",
    )
    parser.add_argument(
        "--sample-users",
        type=int,
        default=500,
        help="抽样用户数（None 表示不抽样，使用全量）",
    )
    parser.add_argument(
        "--min-interactions",
        type=int,
        default=10,
        help="每个用户最少交互数（过滤过稀疏用户）",
    )
    args = parser.parse_args()

    prepare_movielens_small(
        out_ratings=Path(args.out_ratings),
        out_items=Path(args.out_items),
        sample_users=None if args.sample_users <= 0 else args.sample_users,
        min_interactions=args.min_interactions,
    )
    print(f"[OK] ratings saved to: {args.out_ratings}")
    print(f"[OK] items saved to:   {args.out_items}")


if __name__ == "__main__":
    main()

