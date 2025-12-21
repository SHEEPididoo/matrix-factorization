from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import pandas as pd

from matrix_factorization import BaselineModel, KernelMF, ItemItemCF, UserUserCF

from .common import Paths, repo_root_from_this_file, require_columns


def train_model(
    model_name: str,
    kernel: str,
    ratings_path: Path,
    out_path: Path,
) -> None:
    ratings = pd.read_parquet(ratings_path)
    require_columns(ratings, ["user_id", "item_id", "rating"], "ratings")

    X = ratings[["user_id", "item_id"]]
    y = ratings["rating"]

    if model_name == "baseline":
        model = BaselineModel(method="sgd", n_epochs=30, lr=0.01, reg=0.02, verbose=1)
    elif model_name == "kernel_mf":
        model = KernelMF(
            n_factors=50,
            n_epochs=30,
            kernel=kernel,
            lr=0.01,
            reg=0.02,
            verbose=1,
        )
    elif model_name == "item_cf":
        model = ItemItemCF(n_neighbors=50, similarity_metric="cosine", verbose=0)
    elif model_name == "user_cf":
        model = UserUserCF(n_neighbors=50, similarity_metric="cosine", verbose=0)
    else:
        raise ValueError("model 必须是: baseline | kernel_mf | item_cf | user_cf")

    model.fit(X, y)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("wb") as f:
        pickle.dump(model, f)


def main() -> None:
    paths = Paths.from_repo(repo_root_from_this_file())

    parser = argparse.ArgumentParser(description="训练一个推荐模型并保存到 artifacts/")
    parser.add_argument(
        "--ratings",
        type=str,
        default=str(paths.data_dir / "ratings.parquet"),
        help="ratings.parquet 路径（需要 user_id,item_id,rating）",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="kernel_mf",
        help="baseline | kernel_mf | item_cf | user_cf",
    )
    parser.add_argument(
        "--kernel",
        type=str,
        default="linear",
        help="KernelMF 的 kernel：linear | sigmoid | rbf",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=str(paths.artifacts_dir / "model.pkl"),
        help="输出模型文件路径（pickle）",
    )
    args = parser.parse_args()

    train_model(
        model_name=args.model,
        kernel=args.kernel,
        ratings_path=Path(args.ratings),
        out_path=Path(args.out),
    )
    print(f"[OK] model saved to: {args.out}")


if __name__ == "__main__":
    main()

