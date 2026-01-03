from __future__ import annotations

import re
import sys
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

# streamlit run 会以“脚本方式”执行该文件，默认没有包上下文，导致相对导入失败。
# 这里把仓库根目录加入 sys.path，确保可用绝对导入 project_template.*
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from project_template.pipeline.common import Paths, repo_root_from_this_file  # noqa: E402
from project_template.app.retrieval import (  # noqa: E402
    load_item_index,
    TfidfItemIndex,
    encode_query_for_index,
    search_topk_pos,
)
from project_template.app.recipe_features import (  # noqa: E402
    contains_any,
    compute_low_calorie_score,
    compute_protein_score,
    expand_avoid_terms,
    extract_ingredient_phrases,
    extract_time_minutes,
    jaccard,
    parse_query_intent,
    split_terms,
)


@st.cache_resource
def _load_model(path: Path):
    if not path.exists():
        return None
    with path.open("rb") as f:
        return pickle.load(f)


@st.cache_resource
def _load_index(path: Path):
    if not path.exists():
        return None
    return load_item_index(path)


def _model_score(model, user_id, item_ids: list) -> np.ndarray:
    if model is None or user_id is None or user_id == "":
        return np.zeros(len(item_ids), dtype=np.float32)
    X = pd.DataFrame({"user_id": [user_id] * len(item_ids), "item_id": item_ids})
    try:
        preds = model.predict(X, bound_ratings=False)
    except Exception:
        return np.zeros(len(item_ids), dtype=np.float32)
    return np.asarray(preds, dtype=np.float32)


def _minmax(x: np.ndarray) -> np.ndarray:
    if x.size == 0:
        return x
    lo, hi = float(x.min()), float(x.max())
    if hi - lo < 1e-8:
        return np.zeros_like(x)
    return (x - lo) / (hi - lo)


def _select_diverse_greedy(
    *,
    base_score: np.ndarray,
    cand_sets: list[set[str]],
    n: int,
    diversity_penalty: float,
    already_selected_sets: list[set[str]] | None = None,
) -> tuple[list[int], list[set[str]]]:
    """
    贪心多样性选择：
    每次选择 argmax_i (base_score[i] - diversity_penalty * max_jaccard(i, selected))
    """
    n = int(n)
    if n <= 0 or base_score.size == 0:
        return [], list(already_selected_sets or [])

    selected: list[int] = []
    selected_sets: list[set[str]] = list(already_selected_sets or [])
    remaining = set(range(int(base_score.size)))

    while remaining and len(selected) < n:
        best_i = None
        best_s = -1e9
        for i in remaining:
            overlap = 0.0
            if selected_sets:
                # 只取最大重叠作为惩罚（比累加更稳）
                overlap = max(jaccard(cand_sets[i], s) for s in selected_sets) if cand_sets[i] else 0.0
            s = float(base_score[i]) - float(diversity_penalty) * float(overlap)
            if s > best_s:
                best_s = s
                best_i = i
        if best_i is None:
            break
        selected.append(int(best_i))
        selected_sets.append(cand_sets[int(best_i)])
        remaining.remove(best_i)

    return selected, selected_sets


def _parse_time_constraint_from_query(q: str) -> int | None:
    ql = (q or "").lower()
    m = re.search(r"under\s+(\d+)\s+minutes?", ql)
    if m:
        return int(m.group(1))
    m = re.search(r"ready\s+in\s+(\d+)\s+minutes?", ql)
    if m:
        return int(m.group(1))
    return None


def _build_query_from_needs(
    *,
    goal: str,
    meal_type: str,
    cuisines: list[str],
    dietary: list[str],
    must_include: list[str],
    avoid: list[str],
    max_time_min: int,
    extra: str,
) -> str:
    goal_map = {
        "随便推荐": "",
        "减脂/低卡": "low calorie",
        "增肌/高蛋白": "high protein",
        "控糖/低碳": "low carb",
        "清淡/低盐": "low sodium",
    }
    meal_map = {"不限": "", "早餐": "breakfast", "午餐": "lunch", "晚餐": "dinner", "加餐/零食": "snack"}
    cuisine_map = {
        "家常": "home cooking",
        "中式": "chinese",
        "川菜": "sichuan",
        "粤菜": "cantonese",
        "日式": "japanese",
        "韩式": "korean",
        "泰式": "thai",
        "西式": "western",
        "地中海": "mediterranean",
        "墨西哥": "mexican",
        "印度": "indian",
    }
    dietary_map = {
        "素食": "vegetarian",
        "纯素": "vegan",
        "无麸质": "gluten-free",
        "无乳糖": "lactose-free",
        "不吃猪肉": "no pork",
        "不吃牛肉": "no beef",
        "清真": "halal",
    }

    parts: list[str] = []

    goal_en = goal_map.get(goal, goal)
    if goal_en:
        parts.append(goal_en)

    meal_en = meal_map.get(meal_type, meal_type)
    if meal_en:
        parts.append(meal_en)

    if cuisines:
        cuisines_en = [cuisine_map.get(c, c) for c in cuisines]
        parts.append("cuisine: " + ", ".join(cuisines_en))
    if dietary:
        dietary_en = [dietary_map.get(d, d) for d in dietary]
        parts.append("dietary: " + ", ".join(dietary_en))
    if max_time_min:
        parts.append(f"ready in {max_time_min} minutes")
    if must_include:
        parts.append("include: " + ", ".join(must_include))
    if avoid:
        parts.append("avoid: " + ", ".join(avoid))
    if extra and extra.strip():
        parts.append(extra.strip())

    return "; ".join(parts) if parts else "random"


def main() -> None:
    st.set_page_config(page_title="Recommender Demo", layout="wide")
    st.title("饮食推荐 Demo（Streamlit）")

    paths = Paths.from_repo(repo_root_from_this_file())
    model_path = paths.artifacts_dir / "model.pkl"
    index_path = paths.artifacts_dir / "item_index.pkl"
    reranker_path = paths.artifacts_dir / "reranker.pkl"
    meta_path = paths.features_dir / "recipe_meta.parquet"

    model = _load_model(model_path)
    index = _load_index(index_path)
    # 可选：弱监督 reranker 与结构化特征
    reranker = None
    if reranker_path.exists():
        with reranker_path.open("rb") as f:
            reranker = pickle.load(f)
    meta = None
    if meta_path.exists():
        try:
            meta_df = pd.read_parquet(meta_path)
            if "item_id" in meta_df.columns:
                meta_df = meta_df.copy()
                meta_df["item_id"] = meta_df["item_id"].astype(str)
                meta = meta_df.set_index("item_id")
        except Exception:
            meta = None

    index_type = "missing"
    if index is not None:
        index_type = "tfidf" if isinstance(index, TfidfItemIndex) else "dense"

    with st.sidebar:
        st.header("设置")
        st.caption(f"index_type: {index_type}")

        embedding_model = None
        if index_type == "dense":
            embedding_model = st.text_input(
                "Embedding 模型（sentence-transformers）",
                value="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
            )
        else:
            st.text_input(
                "Embedding 模型（sentence-transformers）",
                value="（TF-IDF index 不需要）",
                disabled=True,
            )
        # 本项目目标：无用户历史的问卷推荐（个性化可留作扩展）
        with st.expander("高级设置（可选）", expanded=False):
            user_id = st.text_input("user_id（可选，用于个性化扩展）", value="")
        k = st.slider("返回数量 K", min_value=1, max_value=50, value=10, step=1)
        candidate_k = st.slider("召回候选数", min_value=10, max_value=500, value=50, step=10)
        alpha = st.slider("混合权重 alpha（模型 vs 相似度）", min_value=0.0, max_value=1.0, value=0.7, step=0.05)
        use_reranker = st.checkbox("启用 reranker（若存在 artifacts/reranker.pkl）", value=reranker is not None)
        rerank_weight = st.slider(
            "reranker 权重（越大越依赖 reranker）",
            min_value=0.0,
            max_value=1.0,
            value=0.6,
            step=0.05,
            disabled=not (use_reranker and reranker is not None),
        )
        strict_time = st.checkbox("严格按时间过滤（需要 recipe_meta.time_min）", value=True, disabled=meta is None)
        strict_high_protein = st.checkbox("高蛋白目标时启用硬过滤（protein_score）", value=False, disabled=meta is None)
        strict_low_calorie = st.checkbox("低卡目标时启用硬过滤（low_calorie_score）", value=False, disabled=meta is None)
        diversity_penalty = st.slider(
            "多样性惩罚（越大越不重复食材）",
            min_value=0.0,
            max_value=1.0,
            value=0.25,
            step=0.05,
        )
        expand_avoid = st.checkbox("扩展忌口同义词（milk→dairy，nuts→tree nuts）", value=True)

        st.divider()
        st.caption("期望存在的 artifacts：")
        st.code(str(model_path))
        st.code(str(index_path))
        st.code(str(reranker_path))
        st.code(str(meta_path))

    if index is None:
        st.error(
            "缺少 `item_index.pkl`。\n\n"
            "如果你要用食谱数据 `data/full_dataset.csv`：\n"
            "1) python -m project_template.pipeline.prepare_recipes_full_dataset\n"
            "2) python -m project_template.pipeline.build_tfidf_index  （快速/无网络）\n\n"
            "或使用 embedding 路线：\n"
            "2) python -m project_template.pipeline.build_item_embeddings\n"
            "3) python -m project_template.pipeline.export_artifacts\n"
        )
        st.stop()

    tabs = st.tabs(["需求选择（问卷）", "自由文本"])

    with tabs[0]:
        st.subheader("选择你的需求")

        allergen_options = {
            "花生 peanut": "peanut",
            "坚果 nuts": "nuts",
            "牛奶 milk": "milk",
            "鸡蛋 egg": "egg",
            "小麦/麸质 wheat": "wheat",
            "大豆 soy": "soy",
            "鱼 fish": "fish",
            "虾/贝类 shrimp/shellfish": "shrimp",
        }

        with st.form("needs_form", clear_on_submit=False):
            daily_plan = st.checkbox("生成一日三餐（早餐/午餐/晚餐）", value=True)
            include_snack = st.checkbox("额外生成加餐/零食", value=False, disabled=not daily_plan)
            per_meal_k = st.slider("每餐返回数量", min_value=1, max_value=5, value=1, step=1, disabled=not daily_plan)
            goal = st.selectbox("目标", ["随便推荐", "减脂/低卡", "增肌/高蛋白", "控糖/低碳", "清淡/低盐"])
            meal_type = st.selectbox("餐次", ["不限", "早餐", "午餐", "晚餐", "加餐/零食"], disabled=daily_plan)
            max_time_min = st.slider("可接受的制作时间（分钟）", min_value=5, max_value=180, value=30, step=5)
            cuisines = st.multiselect(
                "偏好菜系（可多选）",
                ["家常", "中式", "川菜", "粤菜", "日式", "韩式", "泰式", "西式", "地中海", "墨西哥", "印度"],
                default=["家常"],
            )
            dietary = st.multiselect(
                "饮食偏好/限制（可多选）",
                ["素食", "纯素", "无麸质", "无乳糖", "不吃猪肉", "不吃牛肉", "清真"],
                default=[],
            )
            selected_allergens = st.multiselect(
                "过敏/忌口（严格排除）",
                list(allergen_options.keys()),
                default=[],
            )
            must_include_raw = st.text_input("想包含的食材（逗号分隔，可选）", value="")
            avoid_raw = st.text_input("其他不想出现的食材（逗号分隔，可选）", value="")
            extra = st.text_area("补充说明（可选）", value="", height=60)

            strict_exclude = st.checkbox("按关键词严格排除忌口/过敏", value=True)
            strict_must_include = st.checkbox("按关键词要求至少包含一个“想包含食材”", value=False)

            submitted = st.form_submit_button("生成推荐", type="primary")

        if submitted:
            must_include = split_terms(must_include_raw)
            avoid = split_terms(avoid_raw)
            avoid_terms_raw = [allergen_options[k] for k in selected_allergens] + avoid
            avoid_terms = expand_avoid_terms(avoid_terms_raw, enable=bool(expand_avoid))
            meal_list: list[str]
            if daily_plan:
                meal_list = ["早餐", "午餐", "晚餐"]
                if include_snack:
                    meal_list.append("加餐/零食")
            else:
                meal_list = [meal_type]

            # 跨餐次多样性：把已选过的食材集合带到下一餐
            chosen_sets: list[set[str]] = []

            st.subheader("推荐结果")
            for mt in meal_list:
            query = _build_query_from_needs(
                goal=goal,
                    meal_type=mt,
                cuisines=cuisines,
                dietary=dietary,
                must_include=must_include,
                avoid=avoid_terms,
                max_time_min=max_time_min,
                extra=extra,
            )

                with st.expander(f"查看生成的查询描述（{mt}）", expanded=False):
                st.code(query)

            try:
                qvec = encode_query_for_index(index, query, model_name=embedding_model)
            except Exception as e:
                st.error(str(e))
                st.stop()
            candidates = search_topk_pos(index, qvec, k=candidate_k)

            if strict_exclude and avoid_terms:
                candidates = [
                    (pos, item_id, sim)
                    for (pos, item_id, sim) in candidates
                        if not contains_any(index.texts[pos], avoid_terms)
                ]
            if strict_must_include and must_include:
                candidates = [
                    (pos, item_id, sim)
                    for (pos, item_id, sim) in candidates
                        if contains_any(index.texts[pos], must_include)
                    ]

                # 结构化时间过滤（可选）：缺失 time_min 的条目默认放行（不然很容易过滤光）
                if strict_time and meta is not None:
                    filtered = []
                    for (pos, item_id, sim) in candidates:
                        item_id_s = str(item_id)
                        if item_id_s not in meta.index:
                            filtered.append((pos, item_id, sim))
                            continue
                        row = meta.loc[item_id_s]
                        tmin = None
                        if "time_min" in row and pd.notna(row["time_min"]):
                            try:
                                tmin = float(row["time_min"])
                            except Exception:
                                tmin = None
                        if tmin is None or tmin <= float(max_time_min):
                            filtered.append((pos, item_id, sim))
                    candidates = filtered

                # 高蛋白硬过滤（可选）：只在“增肌/高蛋白”目标启用
                if strict_high_protein and meta is not None and goal == "增肌/高蛋白":
                    filtered = []
                    for (pos, item_id, sim) in candidates:
                        item_id_s = str(item_id)
                        if item_id_s not in meta.index:
                            filtered.append((pos, item_id, sim))
                            continue
                        row = meta.loc[item_id_s]
                        ps = None
                        if "protein_score" in row and pd.notna(row["protein_score"]):
                            try:
                                ps = float(row["protein_score"])
                            except Exception:
                                ps = None
                        # 阈值经验值：>=2 更像“高蛋白”
                        if ps is None or ps >= 2.0:
                            filtered.append((pos, item_id, sim))
                    candidates = filtered

                # 低卡硬过滤（可选）：只在“减脂/低卡”目标启用
                if strict_low_calorie and meta is not None and goal == "减脂/低卡":
                    filtered = []
                    for (pos, item_id, sim) in candidates:
                        item_id_s = str(item_id)
                        if item_id_s not in meta.index:
                            filtered.append((pos, item_id, sim))
                            continue
                        row = meta.loc[item_id_s]
                        ls = None
                        if "low_calorie_score" in row and pd.notna(row["low_calorie_score"]):
                            try:
                                ls = float(row["low_calorie_score"])
                            except Exception:
                                ls = None
                        # 阈值经验值：>=1.0 表示有低卡倾向（更宽松）
                        if ls is None or ls >= 1.0:
                            filtered.append((pos, item_id, sim))
                    candidates = filtered
            if len(candidates) == 0:
                    st.warning(f"{mt}: 筛选后没有候选结果。可以关掉严格筛选，或增大候选数 candidate_k。")
                    continue

            cand_pos = [pos for (pos, _, _) in candidates]
            cand_item_ids = [item_id for (_, item_id, _) in candidates]
            cand_sims = np.asarray([sim for (_, _, sim) in candidates], dtype=np.float32)

                # 本项目默认无 user_id：ms 通常为 0，score 主要由相似度决定
            ms = _model_score(model, user_id if user_id != "" else None, cand_item_ids)
                base = alpha * _minmax(ms) + (1.0 - alpha) * _minmax(cand_sims)

                # 可选：reranker（弱监督训练）
                if use_reranker and reranker is not None:
                    # 从 query/问卷提取“意图”，并计算 time_ok/protein_ok 等可控特征
                    q_time = _parse_time_constraint_from_query(query)
                    if q_time is None:
                        q_time = int(max_time_min) if max_time_min else None
                    intent = parse_query_intent(query)
                    q_low = (query or "").lower()
                    q_terms = [t for t in re.split(r"\W+", q_low) if t][:20]
                    feats = []
                    for (pos, item_id, sim) in candidates:
                        text = index.texts[int(pos)]
                        item_id_s = str(item_id)
                        row = meta.loc[item_id_s] if meta is not None and item_id_s in meta.index else None

                        hay = (text or "").lower()
                        hit = sum(1 for t in q_terms if t in hay) if q_terms else 0
                        kw_overlap = float(hit / max(1, len(q_terms))) if q_terms else 0.0

                        ing = extract_ingredient_phrases(text)
                        has_dairy = float(row["has_dairy"]) if row is not None and "has_dairy" in row else (1.0 if ("cheese" in hay or "milk" in hay or "cream" in hay) else 0.0)
                        has_peanut = float(row["has_peanut"]) if row is not None and "has_peanut" in row else (1.0 if "peanut" in hay else 0.0)
                        has_tree_nuts = float(row["has_tree_nuts"]) if row is not None and "has_tree_nuts" in row else (1.0 if "nuts" in hay else 0.0)
                        has_egg = float(row["has_egg"]) if row is not None and "has_egg" in row else (1.0 if "egg" in hay else 0.0)
                        has_wheat = float(row["has_wheat"]) if row is not None and "has_wheat" in row else 0.0
                        has_soy = float(row["has_soy"]) if row is not None and "has_soy" in row else 0.0
                        has_fish = float(row["has_fish"]) if row is not None and "has_fish" in row else 0.0
                        has_shellfish = float(row["has_shellfish"]) if row is not None and "has_shellfish" in row else 0.0
                        time_min = float(row["time_min"]) if row is not None and "time_min" in row and pd.notna(row["time_min"]) else -1.0
                        time_ok = 1.0
                        if q_time is not None and time_min >= 0:
                            time_ok = 1.0 if float(time_min) <= float(q_time) else 0.0
                        # protein_score/protein_ok：meta 优先，否则回退 text 解析
                        protein_score = float(row["protein_score"]) if row is not None and "protein_score" in row and pd.notna(row["protein_score"]) else float(compute_protein_score(ing))
                        protein_ok = 1.0
                        if intent.get("want_high_protein", False):
                            protein_ok = 1.0 if float(protein_score) >= 2.0 else 0.0
                        # low_calorie_score/ok：meta 优先，否则回退 text 解析
                        if row is not None and "low_calorie_score" in row and pd.notna(row["low_calorie_score"]):
                            low_calorie_score = float(row["low_calorie_score"])
                        else:
                            low_calorie_score, _hi_pen = compute_low_calorie_score(
                                ingredients=ing, directions=(text or "")
                            )
                        low_calorie_ok = 1.0
                        if intent.get("want_low_calorie", False):
                            low_calorie_ok = 1.0 if float(low_calorie_score) >= 1.0 else 0.0
                        ingredients_count = float(row["ingredients_count"]) if row is not None and "ingredients_count" in row else float(len(ing))

                        feats.append(
                            [
                                float(sim),
                                kw_overlap,
                                has_dairy,
                                has_peanut,
                                has_tree_nuts,
                                has_egg,
                                has_wheat,
                                has_soy,
                                has_fish,
                                has_shellfish,
                                time_min,
                                time_ok,
                                protein_score,
                                protein_ok,
                                float(low_calorie_score),
                                float(low_calorie_ok),
                                ingredients_count,
                            ]
                        )
                    feats_np = np.asarray(feats, dtype=np.float32)
                    try:
                        rr = np.asarray(reranker["model"].predict_proba(feats_np)[:, 1], dtype=np.float32)
                    except Exception:
                        rr = np.zeros(feats_np.shape[0], dtype=np.float32)
                    base = (1.0 - float(rerank_weight)) * _minmax(base) + float(rerank_weight) * _minmax(rr)

                cand_sets = [extract_ingredient_phrases(index.texts[p]) for p in cand_pos]

                take_n = int(per_meal_k) if daily_plan else int(k)
                top_idx, chosen_sets = _select_diverse_greedy(
                    base_score=base,
                    cand_sets=cand_sets,
                    n=take_n,
                    diversity_penalty=float(diversity_penalty),
                    already_selected_sets=chosen_sets,
                )

            rows = []
            for rank, i in enumerate(top_idx, start=1):
                pos = cand_pos[int(i)]
                item_id = cand_item_ids[int(i)]
                text = index.texts[pos]
                preview = (text or "").replace("\n", " ").strip()
                if len(preview) > 140:
                    preview = preview[:140] + "…"
                    # 展示时间/蛋白提示（若 meta 存在）
                    item_id_s = str(item_id)
                    tmin_disp = None
                    ps_disp = None
                    if meta is not None and item_id_s in meta.index:
                        row_meta = meta.loc[item_id_s]
                        if "time_min" in row_meta and pd.notna(row_meta["time_min"]):
                            try:
                                tmin_disp = float(row_meta["time_min"])
                            except Exception:
                                tmin_disp = None
                        if "protein_score" in row_meta and pd.notna(row_meta["protein_score"]):
                            try:
                                ps_disp = float(row_meta["protein_score"])
                            except Exception:
                                ps_disp = None
                    low_disp = None
                    if meta is not None and item_id_s in meta.index:
                        row_meta = meta.loc[item_id_s]
                        if "low_calorie_score" in row_meta and pd.notna(row_meta["low_calorie_score"]):
                            try:
                                low_disp = float(row_meta["low_calorie_score"])
                            except Exception:
                                low_disp = None
                rows.append(
                    {
                            "meal": mt,
                        "rank": rank,
                        "item_id": item_id,
                            "score": float(base[int(i)]),
                        "sim": float(cand_sims[int(i)]),
                            "time_min": tmin_disp,
                            "protein_score": ps_disp,
                            "low_calorie_score": low_disp,
                        "preview": preview,
                    }
                )

                st.markdown(f"**{mt}**")
            st.dataframe(pd.DataFrame(rows), use_container_width=True)

            st.subheader("解释（教学用）")
            st.write(
                "当前链路是：**问卷生成 query → 检索召回候选 → 关键词过滤 → 多样性重排（食材 overlap 惩罚）**。"
            )

    with tabs[1]:
        st.subheader("自由文本")
        query = st.text_area(
            "输入你的需求（自由文本）",
            value="I want a quick high protein dinner under 30 minutes, no peanut, no milk.",
            height=80,
        )
        run = st.button("生成推荐", type="primary")

        if run:
            try:
                qvec = encode_query_for_index(index, query, model_name=embedding_model)
            except Exception as e:
                st.error(str(e))
                st.stop()
            candidates = search_topk_pos(index, qvec, k=candidate_k)
            cand_pos = [pos for (pos, _, _) in candidates]
            cand_item_ids = [item_id for (_, item_id, _) in candidates]
            cand_sims = np.asarray([sim for (_, _, sim) in candidates], dtype=np.float32)

            ms = _model_score(model, user_id if user_id != "" else None, cand_item_ids)
            score = alpha * _minmax(ms) + (1.0 - alpha) * _minmax(cand_sims)

            top_idx = np.argsort(-score)[:k]

            rows = []
            for rank, i in enumerate(top_idx, start=1):
                pos = cand_pos[int(i)]
                item_id = cand_item_ids[int(i)]
                text = index.texts[pos]
                preview = (text or "").replace("\n", " ").strip()
                if len(preview) > 140:
                    preview = preview[:140] + "…"
                rows.append(
                    {
                        "rank": rank,
                        "item_id": item_id,
                        "score": float(score[int(i)]),
                        "sim": float(cand_sims[int(i)]),
                        "model_score": float(ms[int(i)]) if ms.size > 0 else 0.0,
                        "preview": preview,
                    }
                )

            st.subheader("推荐结果")
            st.dataframe(pd.DataFrame(rows), use_container_width=True)

            st.subheader("解释（教学用）")
            if model is None or user_id == "":
                st.write("当前是 **纯相似度召回**（未加载模型或未提供 user_id）。")
            else:
                st.write(f"当前是 **混合排序**：{alpha:.2f}*模型预测 + {1-alpha:.2f}*文本相似度（均做 min-max 归一化）。")


if __name__ == "__main__":
    main()
