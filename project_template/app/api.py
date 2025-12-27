from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field

from ..pipeline.common import Paths, repo_root_from_this_file
from .retrieval import TfidfItemIndex, load_item_index, encode_query_for_index, search_topk


class RecommendRequest(BaseModel):
    user_id: Any | None = Field(default=None, description="可选：若提供则会使用协同过滤/MF 的个性化推荐")
    query: str = Field(..., description="自由文本查询，例如：'轻松搞笑、适合周末看的电影'")
    k: int = Field(default=10, ge=1, le=50)
    candidate_k: int = Field(default=50, ge=1, le=500, description="embedding 召回候选数")
    alpha: float = Field(default=0.7, ge=0.0, le=1.0, description="混合权重：alpha*模型分数 + (1-alpha)*embedding相似度")
    embedding_model: str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2",
        description="用于编码 query 的 sentence-transformers 模型名（仅 dense index 需要）",
    )


class RecommendItem(BaseModel):
    item_id: Any
    score: float
    reason: str


class RecommendResponse(BaseModel):
    results: list[RecommendItem]


def _load_pickle(path: Path):
    with path.open("rb") as f:
        return pickle.load(f)


def _model_score(model, user_id: Any | None, item_ids: list[Any]) -> np.ndarray:
    if user_id is None:
        # 没有 user_id 时：退化成“非个性化”分数 0
        return np.zeros(len(item_ids), dtype=np.float32)
    X = pd.DataFrame({"user_id": [user_id] * len(item_ids), "item_id": item_ids})
    try:
        preds = model.predict(X, bound_ratings=False)
    except Exception:
        return np.zeros(len(item_ids), dtype=np.float32)
    return np.asarray(preds, dtype=np.float32)


def create_app() -> FastAPI:
    paths = Paths.from_repo(repo_root_from_this_file())

    model_path = paths.artifacts_dir / "model.pkl"
    index_path = paths.artifacts_dir / "item_index.pkl"

    app = FastAPI(title="Recommendation Demo API", version="0.1.0")

    state = {}
    if model_path.exists():
        state["model"] = _load_pickle(model_path)
    else:
        state["model"] = None

    if index_path.exists():
        state["index"] = load_item_index(index_path)
    else:
        state["index"] = None

    @app.get("/health")
    def health():
        index_type = None
        if state["index"] is not None:
            index_type = "tfidf" if isinstance(state["index"], TfidfItemIndex) else "dense"
        return {
            "ok": True,
            "has_model": state["model"] is not None,
            "has_item_index": state["index"] is not None,
            "index_type": index_type,
            "expected_artifacts": {
                "model": str(model_path),
                "item_index": str(index_path),
            },
        }

    @app.post("/recommend", response_model=RecommendResponse)
    def recommend(req: RecommendRequest) -> RecommendResponse:
        if state["index"] is None:
            raise RuntimeError(
                "缺少 item_index.pkl。你可以：\n"
                "A) TF-IDF（无需 sentence-transformers）：python -m project_template.pipeline.build_tfidf_index\n"
                "B) Embedding：python -m project_template.pipeline.build_item_embeddings 然后运行："
                "python -m project_template.pipeline.export_artifacts"
            )

        index = state["index"]
        qvec = encode_query_for_index(index, req.query, model_name=req.embedding_model)
        candidates = search_topk(index, qvec, k=req.candidate_k)
        cand_item_ids = [item_id for (item_id, _) in candidates]
        cand_sims = np.asarray([sim for (_, sim) in candidates], dtype=np.float32)

        model = state["model"]
        if model is None:
            model_scores = np.zeros_like(cand_sims)
        else:
            model_scores = _model_score(model, req.user_id, cand_item_ids)

        # 统一到可比较尺度（min-max），避免不同模型分数范围差异太大
        def minmax(x: np.ndarray) -> np.ndarray:
            if x.size == 0:
                return x
            lo, hi = float(x.min()), float(x.max())
            if hi - lo < 1e-8:
                return np.zeros_like(x)
            return (x - lo) / (hi - lo)

        s_m = minmax(model_scores)
        s_e = minmax(cand_sims)
        score = req.alpha * s_m + (1.0 - req.alpha) * s_e

        top_idx = np.argsort(-score)[: req.k]
        results: list[RecommendItem] = []
        for i in top_idx:
            item_id = cand_item_ids[int(i)]
            reason = "embedding 相似度召回"
            if req.user_id is not None and model is not None:
                reason = f"混合：{req.alpha:.2f}*模型预测 + {1-req.alpha:.2f}*文本相似度"
            results.append(RecommendItem(item_id=item_id, score=float(score[int(i)]), reason=reason))

        return RecommendResponse(results=results)

    return app


def main() -> None:
    import uvicorn

    app = create_app()
    uvicorn.run(app, host="127.0.0.1", port=8000)


if __name__ == "__main__":
    main()
