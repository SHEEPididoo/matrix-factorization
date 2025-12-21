## 项目模板：LLM-Enhanced Recommendation System（Week4–Week8）

本模板基于你在 Week3 已经完成的 `matrix_factorization/` 代码（统一 `fit/predict/recommend/update_users` 接口与多种推荐算法），目标是帮助学员用**同一套工程骨架**完成从数据 → 特征/LLM增广 → 多模型对比 → Demo 部署 → 最终展示的端到端项目。

---

## 目录结构（建议保持不变）

```
project_template/
  data/                 # 原始/清洗后的数据（建议用 .parquet）
  features/             # 缓存的特征（embedding、标签、统计特征等）
  artifacts/            # 训练好的模型与索引（可直接用于 demo）
  pipeline/             # 数据/特征/训练/评估脚本
  app/                  # Demo（API 或 UI）
  docs/                 # 周计划、check-in、rubric、说明文档
```

---

## 数据契约（所有脚本默认遵守）

### 1) 交互表（ratings）

文件：`project_template/data/ratings.parquet`

必须包含列：
- `user_id`：用户外部 ID（string/int 均可）
- `item_id`：物品外部 ID（string/int 均可）
- `rating`：显式评分（float/int）；如果是隐式反馈请先转成数值强度（0/1 或点击次数等）

可选列：
- `timestamp`

### 2) 物品表（items）

文件：`project_template/data/items.parquet`

必须包含列：
- `item_id`
- `text`：用于 embedding/LLM 的文本字段（标题+简介/评论拼接也可以）

可选列：
- 结构化字段（category/brand/price/...）

---

## 运行方式（最小闭环）

### （可选）快速生成示例数据：MovieLens 小样本

不想自己找数据时，可以用 MovieLens `ml-latest-small` 一键生成模板所需 parquet：

```bash
python -m project_template.pipeline.download_movielens_small --sample-users 500 --min-interactions 10
```

会生成：
- `project_template/data/ratings.parquet`
- `project_template/data/items.parquet`

### 0) 准备数据

把清洗后的 `ratings.parquet` 与 `items.parquet` 放到 `project_template/data/`。

### 1) 生成 item embeddings（Week5）

```bash
python -m project_template.pipeline.build_item_embeddings
```

### 2) 训练模型并导出 artifacts（Week6）

```bash
python -m project_template.pipeline.train --model kernel_mf --kernel linear
python -m project_template.pipeline.export_artifacts
```

### 3) 离线评估（Week6）

```bash
python -m project_template.pipeline.evaluate --k 10
```

### 4) 启动 Demo API（Week7）

```bash
python -m project_template.app.api
```

然后访问：
- `http://127.0.0.1:8000/docs`（Swagger UI）

### 5) 启动课堂展示 UI（Streamlit，推荐）

```bash
streamlit run project_template/app/streamlit_app.py
```

---

## 周计划（与课程 Week4–Week8 对齐）

详见：
- `project_template/docs/WEEK_PLAN.md`
- `project_template/docs/CHECKINS.md`
- `project_template/docs/RUBRIC.md`

---

## 你应该做到的最低交付（强制）

- **可复现**：从新环境到跑出 demo，有清晰步骤
- **可对比**：至少 2 个模型（经典 + LLM/Embedding增强）同一切分同一指标
- **可解释**：输出推荐理由（相似度/同类用户/文本主题等至少一种）
- **可展示**：交互入口（API 或 UI）可用

