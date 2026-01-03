from __future__ import annotations

"""
从 items.parquet 生成“食谱结构化特征”，用于：
- 硬过滤（忌口/过敏源）
- 更稳定的离线评估口径（避免 LLM 主观判“奶酪算不算 milk”）
- 训练弱监督 reranker 的特征输入

输入：
- project_template/data/items.parquet (item_id, text)

输出：
- project_template/features/recipe_meta.parquet
  列示例：
  - item_id
  - time_min (nullable int)
  - has_dairy/has_peanut/has_tree_nuts/...
  - ingredients_count
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from ..app.recipe_features import (
    compute_flags,
    compute_low_calorie_score,
    compute_protein_score,
    extract_ingredient_phrases,
    extract_directions_text,
    extract_time_minutes,
)
from .common import Paths, repo_root_from_this_file, require_columns


def build_recipe_metadata(items_path: Path, out_path: Path, max_rows: int | None) -> int:
    items = pd.read_parquet(items_path)
    require_columns(items, ["item_id", "text"], "items")
    if max_rows is not None:
        items = items.head(int(max_rows))

    item_ids = items["item_id"].astype(str).tolist()
    texts = items["text"].fillna("").astype(str).tolist()

    time_min: list[int | None] = []
    ingredients_count: list[int] = []
    protein_score: list[float] = []
    low_calorie_score: list[float] = []
    high_calorie_penalty: list[float] = []
    flags_rows = []

    for t in texts:
        ing = extract_ingredient_phrases(t)
        ingredients_count.append(int(len(ing)))
        time_min.append(extract_time_minutes(t))
        protein_score.append(compute_protein_score(ing))
        low_s, hi_p = compute_low_calorie_score(ingredients=ing, directions=extract_directions_text(t))
        low_calorie_score.append(low_s)
        high_calorie_penalty.append(hi_p)
        flags = compute_flags(ing)
        flags_rows.append(flags)

    out = pd.DataFrame(
        {
            "item_id": item_ids,
            "time_min": [x if x is not None else np.nan for x in time_min],
            "ingredients_count": ingredients_count,
            "protein_score": protein_score,
            "low_calorie_score": low_calorie_score,
            "high_calorie_penalty": high_calorie_penalty,
            "has_peanut": [f.has_peanut for f in flags_rows],
            "has_tree_nuts": [f.has_tree_nuts for f in flags_rows],
            "has_dairy": [f.has_dairy for f in flags_rows],
            "has_egg": [f.has_egg for f in flags_rows],
            "has_wheat": [f.has_wheat for f in flags_rows],
            "has_soy": [f.has_soy for f in flags_rows],
            "has_fish": [f.has_fish for f in flags_rows],
            "has_shellfish": [f.has_shellfish for f in flags_rows],
        }
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(out_path, index=False)
    return int(out.shape[0])


def main() -> None:
    paths = Paths.from_repo(repo_root_from_this_file())
    parser = argparse.ArgumentParser(description="生成食谱结构化特征（recipe_meta.parquet）")
    parser.add_argument("--items", type=str, default=str(paths.data_dir / "items.parquet"))
    parser.add_argument("--out", type=str, default=str(paths.features_dir / "recipe_meta.parquet"))
    parser.add_argument("--max-rows", type=int, default=0, help="<=0 表示全量")
    args = parser.parse_args()

    max_rows = int(args.max_rows)
    max_rows_opt = None if max_rows <= 0 else max_rows
    n = build_recipe_metadata(Path(args.items), Path(args.out), max_rows=max_rows_opt)
    print(f"[OK] wrote {n} rows -> {args.out}")


if __name__ == "__main__":
    main()


