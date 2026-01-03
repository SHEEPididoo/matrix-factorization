from __future__ import annotations

"""
食谱文本解析与结构化特征（无用户历史推荐场景）：
- 从 items.text（title/ingredients/directions）中解析食材列表
- 抽取粗略的制作时间（minutes）
- 过敏源/忌口 flags（dairy/nuts/peanut 等）
- 忌口同义词扩展（例如 milk -> dairy 相关词）

设计目标：轻量、可解释、可用于过滤/多样性/弱监督训练。
"""

import re
from dataclasses import dataclass
from typing import Iterable


def split_terms(raw: str) -> list[str]:
    if raw is None:
        return []
    parts = re.split(r"[，,;；\n]+", str(raw))
    return [p.strip() for p in parts if p.strip()]


def contains_any(text: str, terms: list[str]) -> bool:
    if not terms:
        return False
    haystack = (text or "").lower()
    return any(str(t).lower() in haystack for t in terms)


def extract_ingredient_phrases(text: str) -> set[str]:
    """
    从 items.text 里提取 ingredients 行里的短语集合。

    期望格式（prepare_recipes_full_dataset.py 生成）：
    - title: ...
    - ingredients: a; b; c
    - directions: ...
    """
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


def extract_directions_text(text: str) -> str:
    if not text:
        return ""
    lines = [ln.strip() for ln in str(text).splitlines() if ln.strip()]
    for ln in lines:
        if ln.lower().startswith("directions:"):
            return ln.split(":", 1)[1].strip()
    return ""


def extract_time_minutes(text: str) -> int | None:
    """
    从 directions 文本里抽取大致时间（分钟）。
    规则非常粗糙，但在大多数菜谱里能抓到 “Bake 30 minutes / cook 1 hour” 这类表达。
    """
    d = extract_directions_text(text).lower()
    if not d:
        return None

    # 处理区间：10-15 minutes / 10 to 15 minutes
    m = re.search(r"(\d+)\s*(?:-|to)\s*(\d+)\s*(?:minutes?|mins?)", d)
    if m:
        hi = int(m.group(2))
        return hi

    mins = re.findall(r"(\d+)\s*(?:minutes?|mins?)", d)
    hrs = re.findall(r"(\d+)\s*(?:hours?|hrs?|hr)\b", d)
    total = 0
    if hrs:
        total += 60 * int(hrs[0])
    if mins:
        total += int(mins[0])
    return total if total > 0 else None


@dataclass(frozen=True)
class RecipeFlags:
    has_peanut: bool
    has_tree_nuts: bool
    has_dairy: bool
    has_egg: bool
    has_wheat: bool
    has_soy: bool
    has_fish: bool
    has_shellfish: bool


_TREE_NUT_WORDS = [
    "almond",
    "walnut",
    "pecan",
    "cashew",
    "pistachio",
    "hazelnut",
    "macadamia",
    "pine nut",
    "brazil nut",
]
_DAIRY_WORDS = [
    "milk",
    "cheese",
    "cream",
    "butter",
    "yogurt",
    "sour cream",
    "buttermilk",
    "whey",
    "half and half",
    "condensed milk",
    "evaporated milk",
]
_EGG_WORDS = ["egg", "eggs"]
_WHEAT_WORDS = ["wheat", "flour", "bread", "pasta", "noodle", "breadcrumbs", "cracker"]
_SOY_WORDS = ["soy", "soy sauce", "tofu", "edamame"]
_FISH_WORDS = ["fish", "salmon", "tuna", "cod", "tilapia", "trout", "sardine"]
_SHELLFISH_WORDS = ["shrimp", "prawn", "crab", "lobster", "clam", "mussel", "oyster", "scallop"]


def compute_flags(ingredients: Iterable[str]) -> RecipeFlags:
    ing = " ; ".join([str(x).lower() for x in ingredients if x])
    has_peanut = "peanut" in ing
    has_tree_nuts = ("nuts" in ing) or any(w in ing for w in _TREE_NUT_WORDS)
    has_dairy = ("dairy" in ing) or any(w in ing for w in _DAIRY_WORDS)
    has_egg = any(w in ing for w in _EGG_WORDS)
    has_wheat = any(w in ing for w in _WHEAT_WORDS)
    has_soy = any(w in ing for w in _SOY_WORDS)
    has_fish = any(w in ing for w in _FISH_WORDS)
    has_shellfish = any(w in ing for w in _SHELLFISH_WORDS)
    return RecipeFlags(
        has_peanut=has_peanut,
        has_tree_nuts=has_tree_nuts,
        has_dairy=has_dairy,
        has_egg=has_egg,
        has_wheat=has_wheat,
        has_soy=has_soy,
        has_fish=has_fish,
        has_shellfish=has_shellfish,
    )


# ---- protein heuristics (weak but stable) ----
_PROTEIN_HIGH_WORDS = [
    "chicken breast",
    "chicken",
    "turkey",
    "tuna",
    "salmon",
    "shrimp",
    "prawn",
    "lean beef",
    "beef",
    "pork loin",
    "pork",
    "fish",
    "tofu",
    "tempeh",
    "lentil",
    "lentils",
    "beans",
    "black beans",
    "kidney beans",
    "chickpea",
    "chickpeas",
    "garbanzo",
    "edamame",
    "egg",
    "eggs",
    "greek yogurt",
    "cottage cheese",
]


def compute_protein_score(ingredients: Iterable[str]) -> float:
    """
    从 ingredients 短语集合估计“高蛋白倾向”分数（越大越可能高蛋白）。
    注意：这是启发式，用于排序/过滤的可控信号，并非营养学精确计算。
    """
    ing = " ; ".join([str(x).lower() for x in ingredients if x])
    if not ing:
        return 0.0
    score = 0.0
    # 命中一个核心蛋白源 +1；命中多个可叠加
    for w in _PROTEIN_HIGH_WORDS:
        if w in ing:
            score += 1.0
    # 轻微加分：如果出现 “protein” 字样
    if "protein" in ing:
        score += 0.5
    return float(score)


def parse_query_intent(query: str) -> dict[str, bool]:
    """
    从 query 文本解析“意图标签”（用于 time_ok/protein_ok）。
    """
    q = (query or "").lower()
    return {
        "want_high_protein": ("high protein" in q) or ("protein" in q and "low protein" not in q),
        "want_low_calorie": ("low calorie" in q) or ("low-calorie" in q),
    }


# ---- low calorie heuristics (weak but stable) ----
_HIGH_CAL_WORDS = [
    # fats/oils
    "oil",
    "olive oil",
    "vegetable oil",
    "canola oil",
    "shortening",
    "lard",
    "butter",
    "margarine",
    # sugar / sweets
    "sugar",
    "brown sugar",
    "powdered sugar",
    "confectioners sugar",
    "corn syrup",
    "honey",
    "maple syrup",
    "chocolate",
    "chips",
    # dairy rich
    "cream",
    "heavy cream",
    "whipping cream",
    "sour cream",
    "cheese",
    "cream cheese",
    "condensed milk",
    # processed / fatty meats
    "bacon",
    "sausage",
    "pepperoni",
    "mayonnaise",
]
_LOW_CAL_HINT_WORDS = [
    "low calorie",
    "low-calorie",
    "low fat",
    "low-fat",
    "fat free",
    "fat-free",
    "light",
    "lite",
    "sugar-free",
    "sugar free",
    "skinless",
    "lean",
]
_FRY_WORDS = ["fry", "fried", "deep fry", "deep-fry", "pan-fry"]
_LIGHT_COOK_WORDS = ["steam", "steamed", "grill", "grilled", "bake", "baked", "roast", "roasted"]


def compute_low_calorie_score(*, ingredients: Iterable[str], directions: str) -> tuple[float, float]:
    """
    返回：(low_calorie_score, high_calorie_penalty)
    - low_calorie_score 越大：越可能低卡（启发式）
    - high_calorie_penalty 越大：越可能高热量（糖/油/奶油/油炸等）
    """
    ing = " ; ".join([str(x).lower() for x in ingredients if x])
    d = (directions or "").lower()
    score = 0.0
    penalty = 0.0

    # 惩罚：高热量原料命中次数（上限避免极端）
    for w in _HIGH_CAL_WORDS:
        if w in ing:
            penalty += 1.0
    penalty = min(penalty, 8.0)

    # 惩罚：油炸
    if any(w in d for w in _FRY_WORDS):
        penalty += 2.0

    # 加分：低脂/无糖等显式提示（更可信）
    for w in _LOW_CAL_HINT_WORDS:
        if w in ing or w in d:
            score += 1.0
    score = min(score, 4.0)

    # 加分：更“清淡”的烹饪方式（弱信号）
    if any(w in d for w in _LIGHT_COOK_WORDS):
        score += 0.5

    # 综合：低卡倾向 = score - 0.5*penalty（再 clip 到 [0, 5]）
    low_cal = max(0.0, min(5.0, score - 0.5 * penalty))
    return float(low_cal), float(penalty)


def expand_avoid_terms(terms: list[str], *, enable: bool) -> list[str]:
    """
    把用户的 avoid 词扩展成更严格的同义词/近义词集合。
    例如：milk -> dairy 相关词；nuts -> tree nuts 相关词。

    注意：这是“硬过滤”，宁可多过滤一点。
    """
    base = [str(t).strip().lower() for t in (terms or []) if str(t).strip()]
    if not enable:
        return base

    out: list[str] = []
    out.extend(base)

    def add_many(xs: list[str]) -> None:
        for x in xs:
            xl = str(x).strip().lower()
            if xl and xl not in out:
                out.append(xl)

    if "milk" in base or "dairy" in base:
        add_many(_DAIRY_WORDS)
        add_many(["dairy"])
    if "nuts" in base or "nut" in base:
        add_many(_TREE_NUT_WORDS)
        add_many(["nuts", "nut"])
    if "peanut" in base:
        add_many(["peanut", "peanuts"])
        # 很多菜谱写 “nuts” 但不区分花生/坚果，这里不强行扩展到所有 nuts，避免过度过滤
    return out


def jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b)
    if inter == 0:
        return 0.0
    union = len(a | b)
    return float(inter / union) if union > 0 else 0.0


