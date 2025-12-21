from __future__ import annotations

"""
prepare_steam_light.py

将 Kaggle 下载到本地的 Steam 数据集转换为项目模板需要的：
- project_template/data/ratings.parquet  (user_id,item_id,rating)
- project_template/data/items.parquet    (item_id,text)

支持的数据源（建议）：
1) tamber/steam-video-games
   - 通常文件名为 steam-200k.csv 或 Steam-200k.csv（无表头，5列）
2) nikdavis/steam-store-games
   - 通常包含 steam.csv + steam_description_data.csv 等（有表头，多表）

设计目标：轻量、鲁棒、可抽样、MacBook 可跑。
"""

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

from .common import Paths, repo_root_from_this_file


def _find_first(paths: list[Path]) -> Path | None:
    for p in paths:
        if p.exists():
            return p
    return None


def _normalize_name(x: str) -> str:
    s = str(x).lower().strip()
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"[^a-z0-9 ]+", "", s)  # 保守：去掉标点
    return s


def _auto_find_steam_200k_csv(data_dir: Path) -> Path | None:
    candidates = sorted(
        list(data_dir.glob("**/*steam*200k*.csv"))
        + list(data_dir.glob("**/*Steam*200k*.csv"))
        + list(data_dir.glob("**/*steam*video*games*.csv"))
    )
    return candidates[0] if candidates else None


def _auto_find_store_files(data_dir: Path) -> tuple[Path | None, Path | None]:
    steam_csv = _find_first(
        [
            data_dir / "steam.csv",
            data_dir / "Steam.csv",
            # 有些解压后在子目录
            *_sorted_first(data_dir, "**/steam.csv"),
            *_sorted_first(data_dir, "**/Steam.csv"),
        ]
    )
    desc_csv = _find_first(
        [
            data_dir / "steam_description_data.csv",
            data_dir / "Steam_description_data.csv",
            *_sorted_first(data_dir, "**/steam_description_data.csv"),
            *_sorted_first(data_dir, "**/Steam_description_data.csv"),
        ]
    )
    return steam_csv, desc_csv


def _sorted_first(root: Path, pattern: str) -> list[Path]:
    xs = sorted(list(root.glob(pattern)))
    return xs[:3]  # 最多取前几个，避免候选太多


def load_steam_200k(path: Path) -> pd.DataFrame:
    # steam-200k.csv 通常为 5 列：user_id, game_title, behavior_name, value, 0
    df = pd.read_csv(path, header=None)

    # 若意外带表头，尝试识别
    if df.shape[1] >= 4 and str(df.iloc[0, 2]).lower() in {"behavior_name", "behavior"}:
        df = pd.read_csv(path)  # 用默认表头
        # 宽容处理：尽量映射到标准列名
        cols = [c.lower() for c in df.columns]
        rename = {}
        for c in df.columns:
            lc = c.lower()
            if "user" in lc:
                rename[c] = "user_id"
            elif "game" in lc or "title" in lc or "item" in lc:
                rename[c] = "game_title"
            elif "behavior" in lc:
                rename[c] = "behavior_name"
            elif lc in {"value", "hours", "playtime"}:
                rename[c] = "value"
        df = df.rename(columns=rename)
        return df[["user_id", "game_title", "behavior_name", "value"]].copy()

    # 无表头：强制列名
    df = df.rename(
        columns={
            0: "user_id",
            1: "game_title",
            2: "behavior_name",
            3: "value",
        }
    )
    df = df[["user_id", "game_title", "behavior_name", "value"]].copy()
    return df


def build_ratings(
    interactions: pd.DataFrame,
    mode: str,
    transform: str,
    min_interactions: int,
    sample_users: int | None,
) -> pd.DataFrame:
    interactions = interactions.copy()
    interactions["behavior_name"] = interactions["behavior_name"].astype(str).str.lower().str.strip()

    if mode == "play_only":
        interactions = interactions[interactions["behavior_name"] == "play"].copy()
    elif mode == "purchase_only":
        interactions = interactions[interactions["behavior_name"] == "purchase"].copy()
    elif mode == "both":
        interactions = interactions[interactions["behavior_name"].isin(["purchase", "play"])].copy()
    else:
        raise ValueError("mode 必须是: play_only | purchase_only | both")

    # item_id 统一使用游戏标题（轻量项目最稳；不用解决 appid 映射）
    interactions["item_id"] = interactions["game_title"].astype(str)
    interactions["user_id"] = interactions["user_id"].astype(str)

    # rating：play 用小时数；purchase 用 1
    interactions["value"] = pd.to_numeric(interactions["value"], errors="coerce").fillna(0.0)
    is_purchase = interactions["behavior_name"] == "purchase"
    rating = interactions["value"].astype(float)
    rating.loc[is_purchase] = 1.0

    if transform == "raw":
        pass
    elif transform == "log1p":
        rating = np.log1p(rating)
    else:
        raise ValueError("transform 必须是: raw | log1p")

    ratings = pd.DataFrame(
        {
            "user_id": interactions["user_id"].values,
            "item_id": interactions["item_id"].values,
            "rating": rating.values.astype(np.float32),
        }
    )

    # 去重：同一 user-item 多条时取最大 rating（避免 RecommenderBase 抛 duplicate error）
    ratings = ratings.groupby(["user_id", "item_id"], as_index=False)["rating"].max()

    # 过滤过稀疏用户
    if min_interactions > 1:
        cnt = ratings.groupby("user_id")["item_id"].count()
        keep_users = cnt[cnt >= min_interactions].index
        ratings = ratings[ratings["user_id"].isin(keep_users)].copy()

    # 抽样用户以控制规模
    if sample_users is not None:
        users = ratings["user_id"].drop_duplicates().sample(
            n=min(sample_users, ratings["user_id"].nunique()),
            random_state=42,
        )
        ratings = ratings[ratings["user_id"].isin(users)].copy()

    return ratings


def build_items(
    item_ids: pd.Series,
    store_steam_csv: Path | None,
    store_desc_csv: Path | None,
) -> pd.DataFrame:
    items = pd.DataFrame({"item_id": item_ids.astype(str).values})
    items["text"] = items["item_id"].astype(str)

    # 尝试用 Steam Store Games 元数据丰富 text（按 name 近似匹配）
    if store_steam_csv is None:
        return items

    try:
        steam = pd.read_csv(store_steam_csv)
    except Exception:
        return items

    # 典型列：appid, name, genres, categories, steamspy_tags, developer, publisher
    if "name" not in steam.columns:
        return items

    steam = steam.copy()
    steam["name_norm"] = steam["name"].astype(str).map(_normalize_name)

    desc = None
    if store_desc_csv is not None and store_desc_csv.exists():
        try:
            desc = pd.read_csv(store_desc_csv)
        except Exception:
            desc = None

    if desc is not None and "steam_appid" in desc.columns:
        # desc 的 key 是 steam_appid；steam 的 key 通常是 appid
        if "appid" in steam.columns:
            desc = desc.rename(columns={"steam_appid": "appid"})
            steam = steam.merge(desc, on="appid", how="left")

    # 构造 metadata_text（尽量用常见列，缺失则忽略）
    cols_pref = [
        "name",
        "genres",
        "categories",
        "steamspy_tags",
        "developer",
        "publisher",
        "short_description",
        "about_the_game",
        "detailed_description",
    ]
    available = [c for c in cols_pref if c in steam.columns]
    if not available:
        return items

    def join_text(row) -> str:
        parts = []
        for c in available:
            v = row.get(c)
            if pd.isna(v):
                continue
            s = str(v).strip()
            if not s:
                continue
            parts.append(f"{c}: {s}")
        return " | ".join(parts) if parts else ""

    # 预先生成文本（避免逐条 merge 后再 apply）
    steam = steam.drop_duplicates(subset=["name_norm"]).copy()
    steam["meta_text"] = steam.apply(join_text, axis=1)

    items["name_norm"] = items["item_id"].map(_normalize_name)
    merged = items.merge(
        steam[["name_norm", "meta_text"]],
        on="name_norm",
        how="left",
    )

    # 用 enriched text 覆盖（若匹配失败则保持 title）
    merged["text"] = np.where(
        merged["meta_text"].notna() & (merged["meta_text"].astype(str).str.len() > 0),
        merged["meta_text"].astype(str),
        merged["text"].astype(str),
    )
    merged = merged.drop(columns=["name_norm", "meta_text"])
    return merged


def main() -> None:
    paths = Paths.from_repo(repo_root_from_this_file())

    parser = argparse.ArgumentParser(description="将 Steam Kaggle 数据转换为模板 parquet")
    parser.add_argument(
        "--data-dir",
        type=str,
        default=str(paths.data_dir),
        help="包含 Kaggle 解压文件的目录（默认 project_template/data）",
    )
    parser.add_argument(
        "--steam-200k",
        type=str,
        default="",
        help="steam-200k.csv 路径（留空则自动在 data-dir 下搜索）",
    )
    parser.add_argument(
        "--store-steam-csv",
        type=str,
        default="",
        help="Steam Store Games 的 steam.csv 路径（留空则自动搜索）",
    )
    parser.add_argument(
        "--store-desc-csv",
        type=str,
        default="",
        help="Steam Store Games 的 steam_description_data.csv 路径（留空则自动搜索）",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="play_only",
        help="play_only | purchase_only | both",
    )
    parser.add_argument(
        "--transform",
        type=str,
        default="log1p",
        help="raw | log1p（推荐 log1p，减小 playtime 尺度差异）",
    )
    parser.add_argument("--min-interactions", type=int, default=10)
    parser.add_argument("--sample-users", type=int, default=500, help="<=0 表示不抽样")
    parser.add_argument(
        "--out-ratings",
        type=str,
        default=str(paths.data_dir / "ratings.parquet"),
    )
    parser.add_argument(
        "--out-items",
        type=str,
        default=str(paths.data_dir / "items.parquet"),
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)

    steam_200k = Path(args.steam_200k) if args.steam_200k else _auto_find_steam_200k_csv(data_dir)
    if steam_200k is None or not steam_200k.exists():
        raise FileNotFoundError(
            "找不到 Steam Video Games CSV（例如 steam-200k.csv）。\n"
            "请确认已下载并解压到 project_template/data，或使用 --steam-200k 指定路径。"
        )

    store_steam_csv = Path(args.store_steam_csv) if args.store_steam_csv else None
    store_desc_csv = Path(args.store_desc_csv) if args.store_desc_csv else None
    if store_steam_csv is None and store_desc_csv is None:
        auto_s, auto_d = _auto_find_store_files(data_dir)
        store_steam_csv = auto_s
        store_desc_csv = auto_d

    interactions = load_steam_200k(steam_200k)
    ratings = build_ratings(
        interactions=interactions,
        mode=args.mode,
        transform=args.transform,
        min_interactions=args.min_interactions,
        sample_users=None if args.sample_users <= 0 else args.sample_users,
    )
    items = build_items(
        item_ids=pd.Series(ratings["item_id"].unique()),
        store_steam_csv=store_steam_csv,
        store_desc_csv=store_desc_csv,
    )

    out_ratings = Path(args.out_ratings)
    out_items = Path(args.out_items)
    out_ratings.parent.mkdir(parents=True, exist_ok=True)
    out_items.parent.mkdir(parents=True, exist_ok=True)
    ratings.to_parquet(out_ratings, index=False)
    items.to_parquet(out_items, index=False)

    # 简短报告
    enriched_ratio = float((items["text"] != items["item_id"]).mean()) if items.shape[0] else 0.0
    print("[OK] saved:")
    print(f" - ratings: {out_ratings}  rows={ratings.shape[0]} users={ratings['user_id'].nunique()} items={ratings['item_id'].nunique()}")
    print(f" - items:   {out_items}    rows={items.shape[0]} enriched_text_ratio={enriched_ratio:.2%}")
    print("[INFO] sources:")
    print(f" - interactions_csv: {steam_200k}")
    if store_steam_csv is not None:
        print(f" - store_steam_csv:  {store_steam_csv}")
    if store_desc_csv is not None:
        print(f" - store_desc_csv:   {store_desc_csv}")


if __name__ == "__main__":
    main()

