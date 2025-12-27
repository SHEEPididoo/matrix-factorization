from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from .common import Paths, repo_root_from_this_file


def _parse_json_list(raw: object) -> list[str]:
    if raw is None:
        return []
    if not isinstance(raw, str):
        return []
    s = raw.strip()
    if not s:
        return []
    try:
        value = json.loads(s)
    except Exception:
        return []
    if not isinstance(value, list):
        return []
    out: list[str] = []
    for item in value:
        if item is None:
            continue
        item_str = str(item).strip()
        if item_str:
            out.append(item_str)
    return out


def _build_text(
    *,
    title: str,
    ner: list[str],
    ingredients: list[str],
    directions: list[str],
    include_directions: bool,
    max_ingredients: int,
    max_steps: int,
) -> str:
    parts: list[str] = []
    title_clean = (title or "").strip()
    if title_clean:
        parts.append(f"title: {title_clean}")

    tokens = ner if ner else ingredients
    if max_ingredients > 0:
        tokens = tokens[:max_ingredients]
    if tokens:
        parts.append("ingredients: " + "; ".join(tokens))

    if include_directions:
        steps = directions
        if max_steps > 0:
            steps = steps[:max_steps]
        steps = [s.strip() for s in steps if s and str(s).strip()]
        if steps:
            parts.append("directions: " + " ".join(steps))

    return "\n".join(parts).strip()


def prepare_items_parquet(
    csv_path: Path,
    out_items: Path,
    max_rows: int | None,
    chunksize: int,
    include_directions: bool,
    max_ingredients: int,
    max_steps: int,
) -> int:
    usecols = ["title", "ingredients", "directions", "link", "source", "NER"]

    out_items.parent.mkdir(parents=True, exist_ok=True)
    writer: pq.ParquetWriter | None = None
    rows_written = 0

    try:
        for chunk in pd.read_csv(csv_path, usecols=usecols, chunksize=chunksize):
            if max_rows is not None and rows_written >= max_rows:
                break

            if max_rows is not None:
                remaining = max_rows - rows_written
                if remaining <= 0:
                    break
                chunk = chunk.head(remaining)

            titles = chunk["title"].fillna("").astype(str).tolist()
            links = chunk["link"].fillna("").astype(str).tolist()
            ners_raw = chunk["NER"].tolist()
            ingredients_raw = chunk["ingredients"].tolist()
            directions_raw = chunk["directions"].tolist()

            item_ids: list[str] = []
            texts: list[str] = []
            for title, link, ner_raw, ing_raw, dir_raw in zip(
                titles, links, ners_raw, ingredients_raw, directions_raw
            ):
                item_id = link.strip() if isinstance(link, str) else ""
                if not item_id:
                    item_id = str(rows_written + len(item_ids))

                ner = _parse_json_list(ner_raw)
                ingredients = _parse_json_list(ing_raw)
                directions = _parse_json_list(dir_raw)

                text = _build_text(
                    title=title,
                    ner=ner,
                    ingredients=ingredients,
                    directions=directions,
                    include_directions=include_directions,
                    max_ingredients=max_ingredients,
                    max_steps=max_steps,
                )
                if not text:
                    continue

                item_ids.append(item_id)
                texts.append(text)

            out_df = pd.DataFrame({"item_id": item_ids, "text": texts})
            table = pa.Table.from_pandas(out_df, preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(out_items, table.schema, compression="snappy")
            writer.write_table(table)
            rows_written += len(out_df)
    finally:
        if writer is not None:
            writer.close()

    return rows_written


def main() -> None:
    paths = Paths.from_repo(repo_root_from_this_file())
    repo_root = repo_root_from_this_file()

    parser = argparse.ArgumentParser(description="将 full_dataset.csv 处理为 items.parquet（recipe 文本）")
    parser.add_argument(
        "--csv",
        type=str,
        default=str(repo_root / "data" / "full_dataset.csv"),
        help="输入 CSV 路径（默认: data/full_dataset.csv）",
    )
    parser.add_argument(
        "--out-items",
        type=str,
        default=str(paths.data_dir / "items.parquet"),
        help="输出 items.parquet 路径（默认: project_template/data/items.parquet）",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=100_000,
        help="最多处理多少行（默认 100000；<=0 表示全量，注意很大）",
    )
    parser.add_argument("--chunksize", type=int, default=50_000)
    parser.add_argument("--include-directions", action="store_true", help="把 directions（步骤）也写入 text")
    parser.add_argument(
        "--max-ingredients",
        type=int,
        default=30,
        help="最多保留多少个 ingredients/NER token（默认 30；<=0 表示不截断）",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=3,
        help="include-directions 时最多保留多少个步骤（默认 3；<=0 表示不截断）",
    )
    args = parser.parse_args()

    max_rows = args.max_rows if args.max_rows and args.max_rows > 0 else None

    n = prepare_items_parquet(
        csv_path=Path(args.csv),
        out_items=Path(args.out_items),
        max_rows=max_rows,
        chunksize=int(args.chunksize),
        include_directions=bool(args.include_directions),
        max_ingredients=int(args.max_ingredients),
        max_steps=int(args.max_steps),
    )
    print(f"[OK] wrote {n} rows -> {args.out_items}")


if __name__ == "__main__":
    main()
