from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from ..pipeline.common import Paths, repo_root_from_this_file
from .retrieval import load_item_index, encode_query, search_topk


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
    preds = model.predict(X, bound_ratings=False)
    return np.asarray(preds, dtype=np.float32)


def _minmax(x: np.ndarray) -> np.ndarray:
    if x.size == 0:
        return x
    lo, hi = float(x.min()), float(x.max())
    if hi - lo < 1e-8:
        return np.zeros_like(x)
    return (x - lo) / (hi - lo)


def main() -> None:
    st.set_page_config(page_title="Recommender Demo", layout="wide")
    st.title("推荐系统课堂 Demo（Streamlit）")

    paths = Paths.from_repo(repo_root_from_this_file())
    model_path = paths.artifacts_dir / "model.pkl"
    index_path = paths.artifacts_dir / "item_index.pkl"

    with st.sidebar:
        st.header("设置")
        embedding_model = st.text_input(
            "Embedding 模型（sentence-transformers）",
            value="sentence-transformers/all-MiniLM-L6-v2",
        )
        user_id = st.text_input("user_id（可选，用于个性化）", value="")
        k = st.slider("返回数量 K", min_value=1, max_value=50, value=10, step=1)
        candidate_k = st.slider("召回候选数", min_value=10, max_value=500, value=50, step=10)
        alpha = st.slider("混合权重 alpha（模型 vs 相似度）", min_value=0.0, max_value=1.0, value=0.7, step=0.05)

        st.divider()
        st.caption("期望存在的 artifacts：")
        st.code(str(model_path))
        st.code(str(index_path))

    model = _load_model(model_path)
    index = _load_index(index_path)

    if index is None:
        st.error(
            "缺少 `item_index.pkl`。请先按模板运行：\n"
            "1) 生成数据（可选 MovieLens）\n"
            "2) build_item_embeddings\n"
            "3) export_artifacts\n"
        )
        st.stop()

    query = st.text_area("输入你的需求（自由文本）", value="我想看轻松搞笑、适合周末的电影", height=80)
    run = st.button("生成推荐", type="primary")

    if run:
        qvec = encode_query(query, model_name=embedding_model)
        candidates = search_topk(index, qvec, k=candidate_k)
        cand_item_ids = [item_id for (item_id, _) in candidates]
        cand_sims = np.asarray([sim for (_, sim) in candidates], dtype=np.float32)

        ms = _model_score(model, user_id if user_id != "" else None, cand_item_ids)
        score = alpha * _minmax(ms) + (1.0 - alpha) * _minmax(cand_sims)

        top_idx = np.argsort(-score)[:k]

        rows = []
        for rank, i in enumerate(top_idx, start=1):
            item_id = cand_item_ids[int(i)]
            rows.append(
                {
                    "rank": rank,
                    "item_id": item_id,
                    "score": float(score[int(i)]),
                    "sim": float(cand_sims[int(i)]),
                    "model_score": float(ms[int(i)]) if ms.size > 0 else 0.0,
                }
            )

        st.subheader("推荐结果")
        st.dataframe(pd.DataFrame(rows), use_container_width=True)

        st.subheader("解释（教学用）")
        if model is None or user_id == "":
            st.write("当前是 **纯 embedding 相似度召回**（未加载模型或未提供 user_id）。")
        else:
            st.write(f"当前是 **混合排序**：{alpha:.2f}*模型预测 + {1-alpha:.2f}*文本相似度（均做 min-max 归一化）。")


if __name__ == "__main__":
    main()

