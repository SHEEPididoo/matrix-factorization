## 每日饮食食谱推荐系统（无用户历史）

本仓库包含两部分：

- **`project_template/`**：当前项目的主实现，一个“问卷/文本输入 → 召回 → 硬过滤 → rerank → 多样性重排 → 每日餐单”的推荐系统 Demo（支持 TF‑IDF 或多语 Embedding；支持 LLM 评估）。
- **`matrix_factorization/`**：课程 Week3 的传统推荐算法实现（MF/CF 等）。本项目场景为“无用户历史”，因此 MF/CF 不作为主链路依赖，但保留在仓库中。

### 你能得到什么
- **问卷式推荐**：目标（低卡/高蛋白/低碳…）+ 餐次 + 菜系 + 饮食限制 + 忌口/过敏 + 时间约束
- **每日餐单**：早餐/午餐/晚餐（可选加餐），跨餐次多样性（食材 overlap 惩罚）
- **弱监督 reranker**：无需用户行为，也能训练一个轻量模型改进排序（`artifacts/reranker.pkl`）
- **LLM 评估（可选）**：对推荐结果做自动评分与违规分析，且评估口径与线上硬过滤保持一致

---

## 目录结构（与你当前项目相关的部分）

```
project_template/
  data/                 # 原始/清洗后的数据（例如 full_dataset.csv、items.parquet）
  features/             # 缓存特征（items_emb.parquet、recipe_meta.parquet 等）
  artifacts/            # 索引/模型产物（item_index.pkl、reranker.pkl、llm_eval_report.json）
  pipeline/             # 数据/特征/训练/评估脚本（python -m 运行）
  app/                  # Demo（Streamlit / FastAPI）与通用工具
```

---

## 环境安装

### 方式 A：conda（推荐）

```bash
conda env create -f environment.yml
conda activate recsys-week3
```

### 方式 B：pip（仅基础依赖）

```bash
pip install -r requirements.txt
pip install -r project_template/requirements-optional.txt
```

---

## 数据准备（你的食谱数据集）

你当前使用的原始数据：`project_template/data/full_dataset.csv`（字段见 EDA：title/ingredients/directions/link/source/NER）。

### 1) 生成 `items.parquet`（必须）

```bash
python -m project_template.pipeline.prepare_recipes_full_dataset \
  --csv "project_template/data/full_dataset.csv" \
  --max-rows 100000
```

输出：
- `project_template/data/items.parquet`（`item_id,text`；`item_id` 默认使用 `link`）

---

## 构建召回索引（两条路线二选一）

### 路线 1：TF‑IDF（最快、无网）

```bash
python -m project_template.pipeline.build_tfidf_index \
  --items "project_template/data/items.parquet" \
  --max-rows 100000
```

输出：
- `project_template/artifacts/item_index.pkl`（TF‑IDF）

### 路线 2：多语 Embedding（中文输入更强）

```bash
python -m project_template.pipeline.build_item_embeddings \
  --items "project_template/data/items.parquet" \
  --model "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

python -m project_template.pipeline.export_artifacts \
  --items "project_template/data/items.parquet" \
  --items-emb "project_template/features/items_emb.parquet"
```

输出：
- `project_template/features/items_emb.parquet`
- `project_template/artifacts/item_index.pkl`（dense embedding）

---

## 生成结构化特征（用于硬过滤 / reranker / 评估一致性）

```bash
python -m project_template.pipeline.build_recipe_metadata \
  --items "project_template/data/items.parquet"
```

输出：
- `project_template/features/recipe_meta.parquet`（time_min、dairy/nuts/peanut 等 flags、ingredients_count…）

---

## 训练弱监督 reranker（无用户历史也能训）

```bash
python -m project_template.pipeline.train_reranker \
  --items "project_template/data/items.parquet" \
  --index "project_template/artifacts/item_index.pkl" \
  --meta "project_template/features/recipe_meta.parquet" \
  --n-seed-items 2000 \
  --candidate-k 80 \
  --n-neg 10
```

输出：
- `project_template/artifacts/reranker.pkl`

---

## 启动 Demo（推荐 Streamlit）

```bash
streamlit run project_template/app/streamlit_app.py
```

在 UI 里你可以：
- 勾选 **生成一日三餐**
- 开启 **扩展忌口同义词**（例如 `milk → dairy` 相关词）
- 如果你训练了 reranker：勾选 **启用 reranker** 并调节权重

---

## 评估（LLM 可选）

### 1) 配置 OpenAI Key（仅当你使用 provider=openai）

在仓库根目录创建 `.env`：

```bash
cp env.template .env
# 编辑 .env：OPENAI_API_KEY=...
```

### 2) 运行评估

无网（keywords fallback）：

```bash
python -m project_template.pipeline.evaluate_llm \
  --provider keywords \
  --apply-filters \
  --query "low calorie high protein dinner under 30 minutes" \
  --avoid "peanut,milk"
```

OpenAI（LLM 评分）：

```bash
python -m project_template.pipeline.evaluate_llm \
  --provider openai \
  --openai-model gpt-4o-mini \
  --apply-filters \
  --query "low calorie high protein dinner under 30 minutes" \
  --avoid "peanut,milk"
```

输出：
- `project_template/artifacts/llm_eval_report.json`

说明：
- `--apply-filters` 会先按线上同样的“硬过滤”过滤候选（包含 `avoid` 的同义词扩展），再做评估，确保口径一致。

---

## 常见问题（FAQ）

### 1) 为什么我中文输入效果差？
- 你用的是 TF‑IDF 或英文 embedding 模型时，中文 query 召回会很弱。请切到“多语 Embedding”路线，并使用：  
  `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`

### 2) 为什么过滤后还有 contains_avoid_term？
- 如果你没开“扩展忌口同义词”，`milk` 不会自动扩展到 `cheese/cream/butter/...`。  
- 现在 UI 与 `evaluate_llm.py` 都支持同义词扩展，建议开启。

### 3) 数据太大跑不动怎么办？
- 先用 `--max-rows 100000` 跑通全链路，再逐步放大。

---

## 许可证

本项目采用 MIT 许可证，详见 `LICENSE`。
