from __future__ import annotations

import argparse
import pickle
from pathlib import Path
from typing import Optional

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from .common import Paths, repo_root_from_this_file, require_columns


def build_tfidf_index(
    items_path: Path,
    out_index_path: Path,
    max_rows: Optional[int],
    max_features: int,
    min_df: int,
    ngram_max: int,
    stop_words: Optional[str],
) -> None:
    items = pd.read_parquet(items_path)
    require_columns(items, ["item_id", "text"], "items")

    if max_rows is not None:
        items = items.head(max_rows)

    item_ids = items["item_id"].tolist()
    texts = items["text"].fillna("").astype(str).tolist()

    vectorizer = TfidfVectorizer(
        max_features=max_features if max_features > 0 else None,
        min_df=min_df,
        ngram_range=(1, max(1, ngram_max)),
        stop_words=stop_words,
        norm="l2",
    )
    matrix = vectorizer.fit_transform(texts)

    out_index_path.parent.mkdir(parents=True, exist_ok=True)
    with out_index_path.open("wb") as f:
        pickle.dump(
            {
                "index_type": "tfidf",
                "item_ids": item_ids,
                "texts": texts,
                "vectorizer": vectorizer,
                "matrix": matrix,
            },
            f,
        )


def main() -> None:
    paths = Paths.from_repo(repo_root_from_this_file())

    parser = argparse.ArgumentParser(description="构建一个 TF-IDF 检索索引（无需 sentence-transformers）")
    parser.add_argument(
        "--items",
        type=str,
        default=str(paths.data_dir / "items.parquet"),
        help="items.parquet 路径（需要 item_id,text）",
    )
    parser.add_argument(
        "--out-index",
        type=str,
        default=str(paths.artifacts_dir / "item_index.pkl"),
        help="输出 item_index.pkl 路径（默认覆盖 artifacts/item_index.pkl）",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=100_000,
        help="最多使用多少条 items（默认 100000；<=0 表示全量）",
    )
    parser.add_argument("--max-features", type=int, default=50_000, help="TF-IDF 词表大小上限（默认 50000）")
    parser.add_argument("--min-df", type=int, default=2, help="忽略出现次数太少的 token（默认 2）")
    parser.add_argument("--ngram-max", type=int, default=1, help="ngram 最大长度（默认 1）")
    parser.add_argument(
        "--stop-words",
        type=str,
        default="english",
        help="停用词：'english' 或 'none'（默认 english）",
    )
    args = parser.parse_args()

    max_rows = args.max_rows if args.max_rows and args.max_rows > 0 else None
    stop_words = None if args.stop_words.lower() == "none" else args.stop_words

    build_tfidf_index(
        items_path=Path(args.items),
        out_index_path=Path(args.out_index),
        max_rows=max_rows,
        max_features=int(args.max_features),
        min_df=int(args.min_df),
        ngram_max=int(args.ngram_max),
        stop_words=stop_words,
    )
    print(f"[OK] tfidf index saved to: {args.out_index}")


if __name__ == "__main__":
    main()
