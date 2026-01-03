## Final Demo 报告模板（可直接复制填写）

> 建议文件：`project_template/docs/final.md`（把本模板复制过去填写）

---

### 1. 项目概述（Problem & Goal）

- **推荐对象**：推荐什么（例如：食谱/菜谱/电影/商品…）
- **用户目标**：用户要解决什么问题（找相似/找新/节省时间/探索…）
- **约束**：数据规模、离线为主、MacBook 本地跑等

---

### 2. 数据（Data）

- **数据集来源与链接**：
- **数据契约**：
  - `ratings.parquet`: `user_id,item_id,rating`
  - `items.parquet`: `item_id,text`
- **统计与现象**（放 1–3 张图更好）：
  - #users/#items/#interactions
  - 稀疏度、长尾分布
  - 冷启动比例（交互少于 N 的用户/物品占比）
- **切分策略**：随机/留一/时间切分（为什么）

---

### 3. 特征工程（Week5）

#### 3.1 结构化特征

- 你做了哪些（如流行度、均分、活跃度等）？
- 产物文件：`project_template/features/user_features.parquet`、`item_features.parquet`

#### 3.2 文本/Embedding/LLM 增广

- Embedding 模型与输入文本是什么？
- LLM（可选）做了什么抽取/增强？
- 产物文件：
  - `project_template/features/items_emb.parquet`
  - `project_template/features/items_text_enriched.parquet`（可选）

---

### 4. 模型（Week6）

#### 4.1 Baselines（至少两个）

- Baseline A：
- Baseline B：

#### 4.2 增强/Hybrid（至少一个）

- Hybrid 方案：embedding 召回 + 模型 rerank / alpha 混合
- 为什么这样设计？

---

### 5. 评估（Evaluation）

#### 5.1 指标与设置

- Top-K：Precision@K / Recall@K / NDCG@K
- 切分：留一法（n_test=…）或其他
- 阈值：positive_threshold=…

#### 5.2 结果表（务必给对比表）

| Model | Precision@10 | Recall@10 | NDCG@10 | Notes |
|------|--------------:|----------:|--------:|------|
| baseline |  |  |  |  |
| item_cf / user_cf |  |  |  |  |
| kernel_mf |  |  |  |  |
| hybrid |  |  |  |  |

#### 5.3 Ablation（增强开/关）

- 无增强 vs 有增强 的差异与解释：

#### 5.4 失败案例（至少 2 个）

- Case 1：推荐不准/过于热门/冷启动 → 你认为原因？
- Case 2：…

---

### 6. Demo（Week7）

- **启动方式**：
  - Streamlit：`streamlit run project_template/app/streamlit_app.py`
  - FastAPI：`python -m project_template.app.api`
- **交互能力**：自由文本输入、可选 user_id 个性化
- **解释**：展示 1–2 条“为什么推荐它”

---

### 7. 复现（Conda）

- `conda env create -f environment.yml`
- `conda activate recsys-week3`
- 跑通命令清单（数据→特征→训练→评估→demo）：

---

### 8. 总结与未来工作

- 优势：
- 限制：
- 下一步：

