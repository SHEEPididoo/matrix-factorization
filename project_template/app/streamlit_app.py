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


def _split_terms(raw: str) -> list[str]:
    if raw is None:
        return []
    parts = re.split(r"[，,;；\n]+", raw)
    return [p.strip() for p in parts if p.strip()]


def _contains_any(text: str, terms: list[str]) -> bool:
    if not terms:
        return False
    haystack = (text or "").lower()
    return any(t.lower() in haystack for t in terms)


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

    model = _load_model(model_path)
    index = _load_index(index_path)

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
                value="sentence-transformers/all-MiniLM-L6-v2",
            )
        else:
            st.text_input(
                "Embedding 模型（sentence-transformers）",
                value="（TF-IDF index 不需要）",
                disabled=True,
            )
        user_id = st.text_input("user_id（可选，用于个性化）", value="")
        k = st.slider("返回数量 K", min_value=1, max_value=50, value=10, step=1)
        candidate_k = st.slider("召回候选数", min_value=10, max_value=500, value=50, step=10)
        alpha = st.slider("混合权重 alpha（模型 vs 相似度）", min_value=0.0, max_value=1.0, value=0.7, step=0.05)

        st.divider()
        st.caption("期望存在的 artifacts：")
        st.code(str(model_path))
        st.code(str(index_path))

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
            goal = st.selectbox("目标", ["随便推荐", "减脂/低卡", "增肌/高蛋白", "控糖/低碳", "清淡/低盐"])
            meal_type = st.selectbox("餐次", ["不限", "早餐", "午餐", "晚餐", "加餐/零食"])
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
            must_include = _split_terms(must_include_raw)
            avoid = _split_terms(avoid_raw)
            avoid_terms = [allergen_options[k] for k in selected_allergens] + avoid

            query = _build_query_from_needs(
                goal=goal,
                meal_type=meal_type,
                cuisines=cuisines,
                dietary=dietary,
                must_include=must_include,
                avoid=avoid_terms,
                max_time_min=max_time_min,
                extra=extra,
            )

            with st.expander("查看生成的查询描述", expanded=False):
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
                    if not _contains_any(index.texts[pos], avoid_terms)
                ]

            if strict_must_include and must_include:
                candidates = [
                    (pos, item_id, sim)
                    for (pos, item_id, sim) in candidates
                    if _contains_any(index.texts[pos], must_include)
                ]

            if len(candidates) == 0:
                st.warning("筛选后没有候选结果：可以关掉严格筛选，或增大候选数 candidate_k。")
                st.stop()

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
