from __future__ import annotations

"""
LLM 评估（无用户历史场景）：

目标：对“问卷/自由文本 → 推荐结果”做自动化评估，重点覆盖：
- 相关性：是否符合 query（目标/餐次/时间/菜系/偏好）
- 约束满足：must_include / avoid 是否被满足（可作为 hard constraints）
- 多样性：推荐列表内食材重复程度（越不重复越好）

输入：
- project_template/artifacts/item_index.pkl（TF-IDF 或 dense embedding）

使用方式（keywords 无网版本）：
- python -m project_template.pipeline.evaluate_llm --provider keywords --query "high protein dinner" --avoid "peanut,milk"

使用方式（OpenAI LLM 版本，需要 OPENAI_API_KEY）：
- python -m project_template.pipeline.evaluate_llm --provider openai --openai-model gpt-4o-mini --query "low calorie breakfast" --avoid "peanut"
"""

import argparse
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np

from ..app.retrieval import (
    TfidfItemIndex,
    encode_query_for_index,
    load_item_index,
    search_topk_pos,
)
from .common import Paths, repo_root_from_this_file
from ..app.recipe_features import expand_avoid_terms, split_terms, contains_any


def _maybe_load_dotenv() -> None:
    try:
        from dotenv import load_dotenv  # type: ignore
    except Exception:
        return
    load_dotenv()


def _safe_import_openai():
    try:
        from openai import OpenAI  # type: ignore
    except Exception as e:  # pragma: no cover
        raise ImportError("未安装 openai。请安装 project_template/requirements-optional.txt") from e
    return OpenAI


def _split_terms(raw: str) -> list[str]:
    # 兼容旧接口：复用 app 的 split_terms
    return split_terms(raw)


def _extract_ingredient_phrases(text: str) -> set[str]:
    if not text:
        return set()
    lines = [ln.strip() for ln in str(text).splitlines() if ln.strip()]
    ing_line = ""
    for ln in lines:
        if ln.lower().startswith("ingredients:"):
            ing_line = ln.split(":", 1)[1]
            break
    if not ing_line:
        return set()
    parts = re.split(r"[;；,，]+", ing_line)
    out = set()
    for p in parts:
        s = p.strip().lower()
        if not s:
            continue
        s = re.sub(r"\s+", " ", s)
        out.add(s)
    return out


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b)
    if inter == 0:
        return 0.0
    union = len(a | b)
    return float(inter / union) if union > 0 else 0.0


def _diversity_score(texts: list[str]) -> float:
    """
    返回 [0,1]，越大越多样：
    diversity = 1 - mean_pairwise_jaccard(ingredient_sets)
    """
    sets = [_extract_ingredient_phrases(t) for t in texts]
    if len(sets) <= 1:
        return 1.0
    js: list[float] = []
    for i in range(len(sets)):
        for j in range(i + 1, len(sets)):
            js.append(_jaccard(sets[i], sets[j]))
    mean_j = float(np.mean(js)) if js else 0.0
    return float(max(0.0, min(1.0, 1.0 - mean_j)))


@dataclass(frozen=True)
class ItemEval:
    item_id: Any
    relevance: float  # 0..5
    violations: list[str]
    reason: str


@dataclass(frozen=True)
class CaseReport:
    query: str
    k: int
    must_include: list[str]
    avoid: list[str]
    index_type: str
    n_retrieved: int
    n_after_filter: int
    diversity: float
    avg_relevance: float
    violation_rate: float
    items: list[ItemEval]

def _report_to_jsonable(rep: CaseReport) -> dict[str, Any]:
    """
    将 dataclass 报告转换为可 JSON 序列化的 dict（避免 ItemEval 无法 dumps）。
    """
    return {
        "query": rep.query,
        "k": int(rep.k),
        "must_include": list(rep.must_include),
        "avoid": list(rep.avoid),
        "index_type": rep.index_type,
        "n_retrieved": int(rep.n_retrieved),
        "n_after_filter": int(rep.n_after_filter),
        "diversity": float(rep.diversity),
        "avg_relevance": float(rep.avg_relevance),
        "violation_rate": float(rep.violation_rate),
        "items": [
            {
                "item_id": str(it.item_id),
                "relevance": float(it.relevance),
                "violations": list(it.violations),
                "reason": str(it.reason),
            }
            for it in rep.items
        ],
    }


def _keywords_judge(
    *,
    query: str,
    item_id: Any,
    text: str,
    must_include: list[str],
    avoid: list[str],
) -> ItemEval:
    violations: list[str] = []
    if avoid and contains_any(text, avoid):
        violations.append("contains_avoid_term")
    if must_include and not contains_any(text, must_include):
        violations.append("missing_must_include")

    # 相关性：轻量 heuristics（仅用于无网 fallback）
    # - 如果命中 must_include，加 2 分
    # - 如果命中 query 的任意词，加 1 分（非常粗糙）
    q_terms = [t for t in re.split(r"\W+", (query or "").lower()) if t]
    hit_q = any(t in (text or "").lower() for t in q_terms[:10]) if q_terms else False
    rel = 1.0
    if must_include and _contains_any(text, must_include):
        rel += 2.0
    if hit_q:
        rel += 1.0
    if violations:
        rel = max(0.0, rel - 2.0)
    rel = float(max(0.0, min(5.0, rel)))
    reason = "keywords_fallback"
    return ItemEval(item_id=item_id, relevance=rel, violations=violations, reason=reason)


def _openai_judge(
    *,
    query: str,
    items: list[tuple[Any, str]],
    must_include: list[str],
    avoid: list[str],
    openai_model: str,
) -> list[ItemEval]:
    _maybe_load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("未检测到 OPENAI_API_KEY 环境变量。请先设置后再运行 provider=openai。")

    OpenAI = _safe_import_openai()
    client = OpenAI(api_key=api_key)

    # 控制 prompt 大小：每条只放前 500 字
    payload = [
        {"item_id": str(item_id), "text": (text or "")[:500]} for (item_id, text) in items
    ]

    prompt = (
        "你是一个饮食/食谱推荐系统的评估器。给你用户需求与推荐结果，请逐条评估。\n"
        "请输出严格 JSON 数组，每个元素对应一条推荐，字段：\n"
        "- item_id: string\n"
        "- relevance: number（0-5，越高越符合需求）\n"
        "- violations: string[]（违反的约束，如 'contains_avoid_term' / 'missing_must_include' / 'not_meal_type' 等；没有就 []）\n"
        "- reason: string（不超过 30 字中文）\n\n"
        f"用户需求(query)：{query}\n"
        f"must_include（至少包含其一，可为空）：{must_include}\n"
        f"avoid（必须排除，可为空）：{avoid}\n\n"
        f"推荐列表(items)：{json.dumps(payload, ensure_ascii=False)}\n"
    )

    resp = client.chat.completions.create(
        model=openai_model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1,
    )
    content = resp.choices[0].message.content or "[]"

    parsed: list[dict[str, Any]] = []
    try:
        parsed = json.loads(content)
    except Exception:
        # 极简兜底：提取第一个 [...] 段
        m = re.search(r"\[[\s\S]*\]", content)
        if m:
            parsed = json.loads(m.group(0))
        else:
            parsed = []

    out: list[ItemEval] = []
    by_id = {str(item_id): item_id for (item_id, _) in items}
    for row in parsed:
        if not isinstance(row, dict):
            continue
        item_id_str = str(row.get("item_id", ""))
        raw_rel = row.get("relevance", 0.0)
        try:
            rel = float(raw_rel)
        except Exception:
            rel = 0.0
        rel = float(max(0.0, min(5.0, rel)))
        viol = row.get("violations") if isinstance(row.get("violations"), list) else []
        viol = [str(x) for x in viol][:20]
        reason = str(row.get("reason", ""))[:120]
        if item_id_str in by_id:
            out.append(ItemEval(item_id=by_id[item_id_str], relevance=rel, violations=viol, reason=reason))

    # 若 LLM 返回不全，补齐缺失项（避免崩）
    seen = set(str(x.item_id) for x in out)
    for item_id, text in items:
        if str(item_id) not in seen:
            out.append(
                _keywords_judge(
                    query=query, item_id=item_id, text=text, must_include=must_include, avoid=avoid
                )
            )
    return out


def _sanitize_llm_violations(
    judged: list[ItemEval], *, must_include: list[str], avoid: list[str]
) -> list[ItemEval]:
    """
    LLM 可能会在 must_include/avoid 为空时仍输出对应违规标签，导致 violation_rate 虚高。
    这里做“硬口径”校正：
    - must_include 为空：移除 missing_must_include
    - avoid 为空：移除 contains_avoid_term
    """
    mi_empty = len(must_include or []) == 0
    av_empty = len(avoid or []) == 0
    if not (mi_empty or av_empty):
        return judged
    out: list[ItemEval] = []
    for it in judged:
        viol = list(it.violations or [])
        if mi_empty:
            viol = [v for v in viol if v != "missing_must_include"]
        if av_empty:
            viol = [v for v in viol if v != "contains_avoid_term"]
        out.append(ItemEval(item_id=it.item_id, relevance=it.relevance, violations=viol, reason=it.reason))
    return out


def evaluate_case(
    *,
    index_path: Path,
    query: str,
    k: int,
    candidate_k: int,
    embedding_model: Optional[str],
    provider: str,
    must_include: list[str],
    avoid: list[str],
    openai_model: str,
    apply_filters: bool,
) -> CaseReport:
    index = load_item_index(index_path)
    index_type = "tfidf" if isinstance(index, TfidfItemIndex) else "dense"

    qvec = encode_query_for_index(index, query, model_name=embedding_model)
    candidates = search_topk_pos(index, qvec, k=candidate_k)
    n_retrieved = int(len(candidates))

    # 评估前过滤（严格）：先扩展 avoid（同义词/近义词），再排除 avoid，最后要求 must_include（至少命中其一）
    avoid_expanded = expand_avoid_terms(avoid, enable=True) if avoid else []
    if apply_filters:
        if avoid_expanded:
            candidates = [
                (pos, item_id, sim)
                for (pos, item_id, sim) in candidates
                if not contains_any(index.texts[pos], avoid_expanded)
            ]
        if must_include:
            candidates = [
                (pos, item_id, sim)
                for (pos, item_id, sim) in candidates
                if contains_any(index.texts[pos], must_include)
            ]

    candidates = candidates[: max(1, int(k))]
    n_after_filter = int(len(candidates))

    item_ids = [item_id for (_, item_id, _) in candidates]
    texts = [index.texts[pos] for (pos, _, _) in candidates]

    if provider == "keywords":
        judged = [
            _keywords_judge(
                query=query,
                item_id=item_id,
                text=text,
                must_include=must_include,
                avoid=avoid_expanded,
            )
            for item_id, text in zip(item_ids, texts)
        ]
    elif provider == "openai":
        judged = _openai_judge(
            query=query,
            items=list(zip(item_ids, texts)),
            must_include=must_include,
            avoid=avoid_expanded,
            openai_model=openai_model,
        )
    else:
        raise ValueError("provider 必须是: keywords | openai")

    # 校正 LLM 的 violations：确保与“硬约束输入是否为空”一致
    judged = _sanitize_llm_violations(judged, must_include=must_include, avoid=avoid_expanded)

    avg_rel = float(np.mean([x.relevance for x in judged])) if judged else 0.0
    violation_rate = float(np.mean([1.0 if x.violations else 0.0 for x in judged])) if judged else 0.0
    div = _diversity_score(texts[: int(k)])

    return CaseReport(
        query=query,
        k=int(k),
        must_include=must_include,
        avoid=avoid,
        index_type=index_type,
        n_retrieved=n_retrieved,
        n_after_filter=n_after_filter,
        diversity=div,
        avg_relevance=avg_rel,
        violation_rate=violation_rate,
        items=judged,
    )


def main() -> None:
    paths = Paths.from_repo(repo_root_from_this_file())
    parser = argparse.ArgumentParser(description="LLM/keywords 评估（无用户历史推荐）")
    parser.add_argument("--index", type=str, default=str(paths.artifacts_dir / "item_index.pkl"))
    parser.add_argument("--provider", type=str, default="keywords", help="keywords | openai")
    parser.add_argument("--openai-model", type=str, default="gpt-4o-mini")
    parser.add_argument(
        "--embedding-model",
        type=str,
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="dense index 编码 query 用的模型名（TF-IDF 时会忽略）",
    )
    parser.add_argument("--query", type=str, default="")
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--candidate-k", type=int, default=50)
    parser.add_argument("--must-include", type=str, default="", help="逗号分隔，可空")
    parser.add_argument("--avoid", type=str, default="", help="逗号分隔，可空")
    parser.add_argument(
        "--apply-filters",
        action="store_true",
        help="评估前先按 avoid/must_include 做严格过滤（推荐开启，避免把应被过滤掉的结果也算作违规）",
    )
    parser.add_argument("--cases", type=str, default="", help="可选：JSON 文件，格式为 [{query,must_include,avoid,k}, ...]")
    parser.add_argument("--out", type=str, default=str(paths.artifacts_dir / "llm_eval_report.json"))
    args = parser.parse_args()

    index_path = Path(args.index)
    if not index_path.exists():
        raise FileNotFoundError(f"找不到 index：{index_path}")

    # TF-IDF index 不需要 embedding_model
    index = load_item_index(index_path)
    embedding_model: Optional[str] = None if isinstance(index, TfidfItemIndex) else str(args.embedding_model)

    reports: list[dict[str, Any]] = []
    if args.cases:
        cases = json.loads(Path(args.cases).read_text(encoding="utf-8"))
        for c in cases:
            q = str(c.get("query", ""))
            if not q.strip():
                continue
            rep = evaluate_case(
                index_path=index_path,
                query=q,
                k=int(c.get("k", args.k)),
                candidate_k=int(c.get("candidate_k", args.candidate_k)),
                embedding_model=embedding_model,
                provider=str(args.provider),
                must_include=_split_terms(c.get("must_include", "")),
                avoid=_split_terms(c.get("avoid", "")),
                openai_model=str(args.openai_model),
                apply_filters=bool(c.get("apply_filters", args.apply_filters)),
            )
            reports.append(_report_to_jsonable(rep))
    else:
        if not str(args.query).strip():
            raise ValueError("请提供 --query，或使用 --cases 指定评估用例 JSON。")
        rep = evaluate_case(
            index_path=index_path,
            query=str(args.query),
            k=int(args.k),
            candidate_k=int(args.candidate_k),
            embedding_model=embedding_model,
            provider=str(args.provider),
            must_include=_split_terms(args.must_include),
            avoid=_split_terms(args.avoid),
            openai_model=str(args.openai_model),
            apply_filters=bool(args.apply_filters),
        )
        reports.append(_report_to_jsonable(rep))

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(reports, ensure_ascii=False, indent=2), encoding="utf-8")

    # 控制台摘要
    for i, r in enumerate(reports, start=1):
        print(f"[{i}] avg_relevance={r['avg_relevance']:.3f}  violation_rate={r['violation_rate']:.3f}  diversity={r['diversity']:.3f}")
        print(f"    retrieved={r['n_retrieved']}  after_filter={r['n_after_filter']}  apply_filters={bool(args.apply_filters)}")
        print(f"    query={r['query']}")
    print(f"[OK] report saved to: {out_path}")


if __name__ == "__main__":
    main()


