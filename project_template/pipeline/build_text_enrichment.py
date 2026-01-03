from __future__ import annotations

"""
Week5：文本/LLM 增广（统一只从 OPENAI_API_KEY 环境变量读取）

输入：
- project_template/data/items.parquet（item_id,text）

输出（缓存到 features/）：
- project_template/features/items_text_enriched.parquet
  列：item_id, tags, summary

provider：
- keywords（默认，最轻量）：TF-IDF 抽关键词作为 tags，summary 取截断文本
- openai（可选）：用 OpenAI 生成 tags/summary（需要 OPENAI_API_KEY 环境变量）

注意：
- 本脚本不会接收/保存任何 key。
- 如你使用 .env：请在仓库根目录创建 `.env`（可由 `cp env.example .env` 得到），并安装 python-dotenv。
"""

import argparse
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from .common import Paths, repo_root_from_this_file, require_columns


def _maybe_load_dotenv() -> None:
    try:
        from dotenv import load_dotenv  # type: ignore
    except Exception:
        return
    load_dotenv()


def _keywords_enrich(texts: list[str], top_k: int) -> tuple[list[list[str]], list[str]]:
    # TF-IDF：对每条文本取 top_k 关键词；summary 用截断文本
    vec = TfidfVectorizer(
        max_features=5000,
        stop_words="english",
        ngram_range=(1, 2),
        min_df=2,
    )
    X = vec.fit_transform(texts)
    vocab = np.array(vec.get_feature_names_out())

    tags: list[list[str]] = []
    summaries: list[str] = []
    for i in range(X.shape[0]):
        row = X.getrow(i)
        if row.nnz == 0:
            tags.append([])
        else:
            idx = row.indices[np.argsort(row.data)[-top_k:]][::-1]
            tags.append(vocab[idx].tolist())
        t = texts[i].strip()
        summaries.append(t[:240] + ("..." if len(t) > 240 else ""))
    return tags, summaries


def _safe_import_openai():
    try:
        from openai import OpenAI  # type: ignore
    except Exception as e:  # pragma: no cover
        raise ImportError("未安装 openai。请安装 project_template/requirements-optional.txt") from e
    return OpenAI


def _openai_enrich(items: pd.DataFrame, model: str, top_k: int) -> tuple[list[list[str]], list[str]]:
    _maybe_load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("未检测到 OPENAI_API_KEY 环境变量。请先设置后再运行 provider=openai。")

    OpenAI = _safe_import_openai()
    client = OpenAI(api_key=api_key)

    tags: list[list[str]] = []
    summaries: list[str] = []

    # 轻量：逐条调用（教学规模 + 抽样数据可接受）。需要更快可以后续做批量/并发。
    for _, row in items.iterrows():
        text = str(row["text"] or "").strip()
        prompt = (
            "你是推荐系统的特征工程助手。请对给定的食谱/菜谱文本做两件事：\n"
            f"1) 输出不超过 {top_k} 个标签（tags），使用英文短词/短语；\n"
            "2) 输出一段不超过 60 字的中文简介（summary）。\n"
            "请用严格 JSON 格式返回：{\"tags\": [...], \"summary\": \"...\"}\n\n"
            f"文本：{text[:2000]}"
        )
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
        )
        content = resp.choices[0].message.content or "{}"
        # 极简 JSON 解析（避免引入额外依赖）；失败则回退 keywords
        parsed: dict[str, Any] = {}
        try:
            import json

            parsed = json.loads(content)
        except Exception:
            parsed = {}
        t = parsed.get("tags") if isinstance(parsed.get("tags"), list) else []
        s = parsed.get("summary") if isinstance(parsed.get("summary"), str) else ""
        tags.append([str(x) for x in t][:top_k])
        summaries.append(str(s)[:240])

    return tags, summaries


def build_text_enrichment(
    items_path: Path,
    out_path: Path,
    provider: str,
    top_k: int,
    openai_model: str,
) -> None:
    items = pd.read_parquet(items_path)
    require_columns(items, ["item_id", "text"], "items")

    items = items.copy()
    items["item_id"] = items["item_id"].astype(str)
    items["text"] = items["text"].fillna("").astype(str)

    if provider == "keywords":
        tags, summaries = _keywords_enrich(items["text"].tolist(), top_k=top_k)
    elif provider == "openai":
        tags, summaries = _openai_enrich(items, model=openai_model, top_k=top_k)
    else:
        raise ValueError("provider 必须是: keywords | openai")

    out = pd.DataFrame(
        {
            "item_id": items["item_id"].values,
            "tags": tags,
            "summary": summaries,
        }
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(out_path, index=False)


def main() -> None:
    paths = Paths.from_repo(repo_root_from_this_file())
    parser = argparse.ArgumentParser(description="对 items.text 做关键词/LLM 增广并落盘缓存")
    parser.add_argument(
        "--items",
        type=str,
        default=str(paths.data_dir / "items.parquet"),
        help="items.parquet 路径（item_id,text）",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=str(paths.features_dir / "items_text_enriched.parquet"),
        help="输出 parquet 路径",
    )
    parser.add_argument(
        "--provider",
        type=str,
        default="keywords",
        help="keywords | openai",
    )
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument(
        "--openai-model",
        type=str,
        default="gpt-4o-mini",
        help="provider=openai 时使用的模型名",
    )
    args = parser.parse_args()

    build_text_enrichment(
        items_path=Path(args.items),
        out_path=Path(args.out),
        provider=args.provider,
        top_k=args.top_k,
        openai_model=args.openai_model,
    )
    print(f"[OK] items text enriched saved to: {args.out}")


if __name__ == "__main__":
    main()

