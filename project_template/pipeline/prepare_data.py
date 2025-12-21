from __future__ import annotations

"""
prepare_data.py

这是一个“占位但可扩展”的数据准备脚本：
- 教学上不同组会用不同数据源，因此这里不强绑定具体下载逻辑
- 你可以把原始数据路径传进来，最终统一写出 ratings.parquet/items.parquet

最小目标：产出满足 README 数据契约的两张表。
"""

import argparse
from pathlib import Path

import pandas as pd

from .common import Paths, repo_root_from_this_file, require_columns


def prepare_from_csv(
    ratings_csv: Path,
    items_csv: Path,
    out_ratings: Path,
    out_items: Path,
    user_col: str,
    item_col: str,
    rating_col: str,
    text_col: str,
) -> None:
    ratings = pd.read_csv(ratings_csv)
    items = pd.read_csv(items_csv)

    require_columns(ratings, [user_col, item_col, rating_col], "ratings_csv")
    require_columns(items, [item_col, text_col], "items_csv")

    out_r = ratings[[user_col, item_col, rating_col]].rename(
        columns={user_col: "user_id", item_col: "item_id", rating_col: "rating"}
    )
    out_i = items[[item_col, text_col]].rename(columns={item_col: "item_id", text_col: "text"})

    out_ratings.parent.mkdir(parents=True, exist_ok=True)
    out_items.parent.mkdir(parents=True, exist_ok=True)
    out_r.to_parquet(out_ratings, index=False)
    out_i.to_parquet(out_items, index=False)


def main() -> None:
    paths = Paths.from_repo(repo_root_from_this_file())

    parser = argparse.ArgumentParser(description="将 CSV 数据规范化为模板所需 parquet")
    parser.add_argument("--ratings-csv", type=str, required=True)
    parser.add_argument("--items-csv", type=str, required=True)
    parser.add_argument("--user-col", type=str, default="user_id")
    parser.add_argument("--item-col", type=str, default="item_id")
    parser.add_argument("--rating-col", type=str, default="rating")
    parser.add_argument("--text-col", type=str, default="text")
    parser.add_argument("--out-ratings", type=str, default=str(paths.data_dir / "ratings.parquet"))
    parser.add_argument("--out-items", type=str, default=str(paths.data_dir / "items.parquet"))
    args = parser.parse_args()

    prepare_from_csv(
        ratings_csv=Path(args.ratings_csv),
        items_csv=Path(args.items_csv),
        out_ratings=Path(args.out_ratings),
        out_items=Path(args.out_items),
        user_col=args.user_col,
        item_col=args.item_col,
        rating_col=args.rating_col,
        text_col=args.text_col,
    )

    print(f"[OK] ratings saved to: {args.out_ratings}")
    print(f"[OK] items saved to:   {args.out_items}")


if __name__ == "__main__":
    main()

