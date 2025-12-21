from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from .common import Paths, repo_root_from_this_file, require_columns


def export_item_index(items_path: Path, emb_path: Path, out_path: Path) -> None:
    items = pd.read_parquet(items_path)
    require_columns(items, ["item_id", "text"], "items")

    emb_df = pd.read_parquet(emb_path)
    require_columns(emb_df, ["item_id", "embedding"], "items_emb")

    merged = items[["item_id", "text"]].merge(emb_df, on="item_id", how="inner")
    if merged.shape[0] == 0:
        raise ValueError("items 与 items_emb 无交集 item_id，无法导出索引")

    item_ids = merged["item_id"].tolist()
    texts = merged["text"].fillna("").astype(str).tolist()
    matrix = np.vstack([np.asarray(v, dtype=np.float32) for v in merged["embedding"].tolist()])

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("wb") as f:
        pickle.dump(
            {"item_ids": item_ids, "texts": texts, "embeddings": matrix},
            f,
        )


def main() -> None:
    paths = Paths.from_repo(repo_root_from_this_file())

    parser = argparse.ArgumentParser(description="导出 demo 所需 artifacts（embedding 索引等）")
    parser.add_argument(
        "--items",
        type=str,
        default=str(paths.data_dir / "items.parquet"),
        help="items.parquet 路径",
    )
    parser.add_argument(
        "--items-emb",
        type=str,
        default=str(paths.features_dir / "items_emb.parquet"),
        help="items_emb.parquet 路径",
    )
    parser.add_argument(
        "--out-index",
        type=str,
        default=str(paths.artifacts_dir / "item_index.pkl"),
        help="输出索引 pickle 路径",
    )
    args = parser.parse_args()

    export_item_index(
        items_path=Path(args.items),
        emb_path=Path(args.items_emb),
        out_path=Path(args.out_index),
    )
    print(f"[OK] item index saved to: {args.out_index}")


if __name__ == "__main__":
    main()

