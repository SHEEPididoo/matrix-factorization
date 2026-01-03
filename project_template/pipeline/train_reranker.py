from __future__ import annotations

"""
弱监督 reranker 训练（无用户历史）：

思路：
- 从 items.text/recipe_meta 自动生成一些“可命中”的合成 query（英文为主，匹配语料）
- 用当前检索（TF-IDF 或 dense）召回候选
- 将“生成该 query 的原 item”视为正样本，候选中的其他项视为负样本
- 训练一个轻量模型（LogisticRegression）学习把相似度 + 结构化特征融合成更好的排序分数

输出：
- project_template/artifacts/reranker.pkl
  包含：model, feature_names, config
"""

import argparse
import pickle
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from ..app.recipe_features import (
    compute_flags,
    compute_low_calorie_score,
    compute_protein_score,
    extract_ingredient_phrases,
    extract_directions_text,
    extract_time_minutes,
    parse_query_intent,
)
from ..app.retrieval import TfidfItemIndex, encode_query_for_index, load_item_index, search_topk_pos
from .common import Paths, repo_root_from_this_file, require_columns


def _parse_time_constraint_from_query(q: str) -> int | None:
    ql = (q or "").lower()
    m = re.search(r"under\s+(\d+)\s+minutes?", ql)
    if m:
        return int(m.group(1))
    m = re.search(r"ready\s+in\s+(\d+)\s+minutes?", ql)
    if m:
        return int(m.group(1))
    return None


def _keyword_overlap(query: str, text: str) -> float:
    q_terms = [t for t in re.split(r"\W+", (query or "").lower()) if t]
    if not q_terms:
        return 0.0
    hay = (text or "").lower()
    hit = sum(1 for t in q_terms[:20] if t and t in hay)
    return float(hit / max(1, min(20, len(q_terms))))


@dataclass(frozen=True)
class FeatRow:
    sim: float
    kw_overlap: float
    has_dairy: float
    has_peanut: float
    has_tree_nuts: float
    has_egg: float
    has_wheat: float
    has_soy: float
    has_fish: float
    has_shellfish: float
    time_min: float
    time_ok: float
    protein_score: float
    protein_ok: float
    low_calorie_score: float
    low_calorie_ok: float
    ingredients_count: float


FEATURE_NAMES = [
    "sim",
    "kw_overlap",
    "has_dairy",
    "has_peanut",
    "has_tree_nuts",
    "has_egg",
    "has_wheat",
    "has_soy",
    "has_fish",
    "has_shellfish",
    "time_min",
    "time_ok",
    "protein_score",
    "protein_ok",
    "low_calorie_score",
    "low_calorie_ok",
    "ingredients_count",
]


def _featurize(query: str, item_text: str, sim: float, meta_row: Optional[pd.Series]) -> np.ndarray:
    ing = extract_ingredient_phrases(item_text)
    flags = compute_flags(ing)
    tmin = extract_time_minutes(item_text)
    pscore = compute_protein_score(ing)
    low_score, hi_pen = compute_low_calorie_score(ingredients=ing, directions=extract_directions_text(item_text))
    if meta_row is not None:
        # 若 meta 提供则优先使用（更稳定）
        if "time_min" in meta_row and pd.notna(meta_row["time_min"]):
            try:
                tmin = int(meta_row["time_min"])
            except Exception:
                pass
        if "protein_score" in meta_row and pd.notna(meta_row["protein_score"]):
            try:
                pscore = float(meta_row["protein_score"])
            except Exception:
                pass
        if "low_calorie_score" in meta_row and pd.notna(meta_row["low_calorie_score"]):
            try:
                low_score = float(meta_row["low_calorie_score"])
            except Exception:
                pass
    q_t = _parse_time_constraint_from_query(query)
    time_ok = 1.0
    if q_t is not None and tmin is not None:
        time_ok = 1.0 if int(tmin) <= int(q_t) else 0.0

    intent = parse_query_intent(query)
    protein_ok = 1.0
    if intent.get("want_high_protein", False):
        # 经验阈值：>=2 表示命中至少两个蛋白源（更稳）
        protein_ok = 1.0 if float(pscore) >= 2.0 else 0.0

    low_cal_ok = 1.0
    if intent.get("want_low_calorie", False):
        # 经验阈值：>=1.0 表示有低卡倾向（更宽松）
        low_cal_ok = 1.0 if float(low_score) >= 1.0 else 0.0

    kw = _keyword_overlap(query, item_text)
    ing_cnt = float(len(ing))
    return np.asarray(
        [
            float(sim),
            float(kw),
            1.0 if flags.has_dairy else 0.0,
            1.0 if flags.has_peanut else 0.0,
            1.0 if flags.has_tree_nuts else 0.0,
            1.0 if flags.has_egg else 0.0,
            1.0 if flags.has_wheat else 0.0,
            1.0 if flags.has_soy else 0.0,
            1.0 if flags.has_fish else 0.0,
            1.0 if flags.has_shellfish else 0.0,
            float(tmin) if tmin is not None else -1.0,
            float(time_ok),
            float(pscore),
            float(protein_ok),
            float(low_score),
            float(low_cal_ok),
            float(ing_cnt),
        ],
        dtype=np.float32,
    )


def _generate_queries(item_text: str, rng: random.Random) -> list[str]:
    ing = list(extract_ingredient_phrases(item_text))
    flags = compute_flags(ing)
    tmin = extract_time_minutes(item_text)

    # 抽几个“代表性 token”作为 query 关键词
    tokens = []
    for x in ing[:]:
        # 取短 token 作为关键词（避免整句）
        w = str(x).split(",")[0].strip()
        if 2 <= len(w) <= 24:
            tokens.append(w)
    rng.shuffle(tokens)
    tokens = tokens[:3]

    meals = ["breakfast", "lunch", "dinner"]
    goals = ["high protein", "low carb", "low sodium", "low calorie"]
    out: list[str] = []

    meal = rng.choice(meals)
    goal = rng.choice(goals)

    # 时间约束（可选）
    if tmin is None:
        t = rng.choice([20, 30, 45, 60])
    else:
        # 往上取整到常见桶
        t = 30 if tmin <= 30 else 45 if tmin <= 45 else 60

    base = f"{goal} {meal}"
    if tokens:
        base = base + " with " + " and ".join(tokens[:2])

    out.append(base)
    out.append(base + f" under {t} minutes")

    # 加一些典型忌口（让模型学会在排序时避开）
    if flags.has_dairy:
        out.append(base + " no dairy")
    if flags.has_tree_nuts or flags.has_peanut:
        out.append(base + " no nuts")

    # 去重
    uniq = []
    seen = set()
    for q in out:
        q2 = re.sub(r"\s+", " ", q.strip().lower())
        if q2 and q2 not in seen:
            seen.add(q2)
            uniq.append(q.strip())
    return uniq[:4]


def train_reranker(
    *,
    items_path: Path,
    index_path: Path,
    meta_path: Optional[Path],
    out_path: Path,
    n_seed_items: int,
    candidate_k: int,
    n_neg: int,
    seed: int,
    embedding_model: str,
    log_every: int,
) -> dict[str, Any]:
    items = pd.read_parquet(items_path)
    require_columns(items, ["item_id", "text"], "items")
    items = items.copy()
    items["item_id"] = items["item_id"].astype(str)
    items["text"] = items["text"].fillna("").astype(str)

    if n_seed_items > 0:
        items = items.head(int(n_seed_items))

    index = load_item_index(index_path)
    index_type = "tfidf" if isinstance(index, TfidfItemIndex) else "dense"
    emb_model_opt: Optional[str] = None if isinstance(index, TfidfItemIndex) else embedding_model

    meta_map: dict[str, pd.Series] = {}
    if meta_path is not None and meta_path.exists():
        meta = pd.read_parquet(meta_path)
        if "item_id" in meta.columns:
            meta["item_id"] = meta["item_id"].astype(str)
            meta_map = {r["item_id"]: r for _, r in meta.iterrows()}

    rng = random.Random(int(seed))

    X_rows: list[np.ndarray] = []
    y_rows: list[int] = []

    # 为了让 query 能检索到 seed item，我们直接用 seed 的 text 生成 query
    total = int(items.shape[0])
    for i_row, (_, row) in enumerate(items.iterrows(), start=1):
        seed_id = str(row["item_id"])
        seed_text = str(row["text"])
        queries = _generate_queries(seed_text, rng=rng)
        if log_every > 0 and (i_row == 1 or i_row % log_every == 0):
            print(
                f"[progress] seed_items={i_row}/{total}  samples={len(y_rows)}  index_type={index_type}"
            )
        for q in queries:
            qvec = encode_query_for_index(index, q, model_name=emb_model_opt)
            cand = search_topk_pos(index, qvec, k=int(candidate_k))
            if not cand:
                continue
            # 查找 seed 是否在候选
            pos_i = None
            for (pos, item_id, sim) in cand:
                if str(item_id) == seed_id:
                    pos_i = (pos, item_id, sim)
                    break
            if pos_i is None:
                continue

            # 正样本
            p_pos, p_id, p_sim = pos_i
            p_text = index.texts[int(p_pos)]
            X_rows.append(_featurize(q, p_text, float(p_sim), meta_map.get(seed_id)))
            y_rows.append(1)

            # 负样本：从 top 里采样（hard negatives）
            neg_pool = [(p, iid, s) for (p, iid, s) in cand if str(iid) != seed_id]
            rng.shuffle(neg_pool)
            for (n_pos, n_id, n_sim) in neg_pool[: int(n_neg)]:
                n_text = index.texts[int(n_pos)]
                X_rows.append(_featurize(q, n_text, float(n_sim), meta_map.get(str(n_id))))
                y_rows.append(0)

    if not X_rows:
        raise RuntimeError("训练样本为空：可能 query 无法召回到 seed item（检查 index_type/语料语言/candidate_k）。")

    X = np.vstack(X_rows).astype(np.float32)
    y = np.asarray(y_rows, dtype=np.int32)

    # 轻量模型：可解释、训练快
    clf = LogisticRegression(max_iter=200, solver="liblinear", class_weight="balanced")
    clf.fit(X, y)

    payload = {
        "model_type": "logreg",
        "index_type": index_type,
        "feature_names": FEATURE_NAMES,
        "model": clf,
        "config": {
            "n_seed_items": int(n_seed_items),
            "candidate_k": int(candidate_k),
            "n_neg": int(n_neg),
            "seed": int(seed),
            "embedding_model": emb_model_opt,
            "meta_used": bool(meta_map),
        },
        "train_stats": {
            "n_samples": int(X.shape[0]),
            "pos_rate": float(y.mean()),
        },
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("wb") as f:
        pickle.dump(payload, f)
    return payload


def main() -> None:
    paths = Paths.from_repo(repo_root_from_this_file())
    parser = argparse.ArgumentParser(description="训练弱监督 reranker（无用户历史）")
    parser.add_argument("--items", type=str, default=str(paths.data_dir / "items.parquet"))
    parser.add_argument("--index", type=str, default=str(paths.artifacts_dir / "item_index.pkl"))
    parser.add_argument("--meta", type=str, default=str(paths.features_dir / "recipe_meta.parquet"))
    parser.add_argument("--out", type=str, default=str(paths.artifacts_dir / "reranker.pkl"))
    parser.add_argument("--n-seed-items", type=int, default=2000, help="用于生成弱监督样本的 seed item 数（越大越慢）")
    parser.add_argument("--candidate-k", type=int, default=80)
    parser.add_argument("--n-neg", type=int, default=10, help="每个正样本采样多少负样本")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-every", type=int, default=100, help="每处理多少个 seed item 打印一次进度；<=0 关闭")
    parser.add_argument(
        "--embedding-model",
        type=str,
        default="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        help="dense index 时用于编码 query 的模型名（TF-IDF 时忽略）",
    )
    args = parser.parse_args()

    meta_path = Path(args.meta) if str(args.meta).strip() else None
    payload = train_reranker(
        items_path=Path(args.items),
        index_path=Path(args.index),
        meta_path=meta_path,
        out_path=Path(args.out),
        n_seed_items=int(args.n_seed_items),
        candidate_k=int(args.candidate_k),
        n_neg=int(args.n_neg),
        seed=int(args.seed),
        embedding_model=str(args.embedding_model),
        log_every=int(args.log_every),
    )
    print(f"[OK] reranker saved to: {args.out}")
    print(f"[INFO] train_stats: {payload['train_stats']}")


if __name__ == "__main__":
    main()


