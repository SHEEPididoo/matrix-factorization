from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from .common import Paths, repo_root_from_this_file, require_columns


def _safe_import_sentence_transformers():
    try:
        from sentence_transformers import SentenceTransformer  # type: ignore
    except Exception as e:  # pragma: no cover
        raise ImportError(
            "未安装 sentence-transformers。请运行：pip install -U sentence-transformers"
        ) from e
    return SentenceTransformer


def build_embeddings(
    items_path: Path,
    out_path: Path,
    model_name: str,
    batch_size: int,
) -> None:
    items = pd.read_parquet(items_path)
    require_columns(items, ["item_id", "text"], "items")

    SentenceTransformer = _safe_import_sentence_transformers()
    model = SentenceTransformer(model_name)

    texts = items["text"].fillna("").astype(str).tolist()
    emb = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    if not isinstance(emb, np.ndarray):
        emb = np.asarray(emb)

    out = pd.DataFrame({"item_id": items["item_id"].values})
    out["embedding"] = [row.astype(np.float32) for row in emb]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(out_path, index=False)


def main() -> None:
    paths = Paths.from_repo(repo_root_from_this_file())

    parser = argparse.ArgumentParser(description="为 items.text 生成 embedding 并缓存到 features/")
    parser.add_argument(
        "--items",
        type=str,
        default=str(paths.data_dir / "items.parquet"),
        help="items.parquet 路径（需要 item_id,text）",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=str(paths.features_dir / "items_emb.parquet"),
        help="输出 embedding parquet 路径",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="sentence-transformers 模型名",
    )
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    build_embeddings(
        items_path=Path(args.items),
        out_path=Path(args.out),
        model_name=args.model,
        batch_size=args.batch_size,
    )

    print(f"[OK] embeddings saved to: {args.out}")


if __name__ == "__main__":
    main()

