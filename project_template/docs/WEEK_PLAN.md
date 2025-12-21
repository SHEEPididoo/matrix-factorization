## Week4–Week8 项目推进计划（建议版本）

本计划默认你已经完成 Week3 并具备：`matrix_factorization` 的多模型实现与统一接口。

---

## Week 4 — Project Kickoff：选域 & 数据集探索

### 目标
- 选定推荐场景与用户目标（“推荐什么给谁，为了解决什么问题”）
- 找到可用数据集（公开或自建）并完成基础清洗
- 明确评估方式与数据切分策略

### 最低要求（强制）
- 数据包含 **交互表**（`user_id,item_id,rating/implicit`）与 **物品表**（至少有 `item_id,text`）
- 产出可复现的数据准备脚本（或 notebook），可一键生成 `ratings.parquet/items.parquet`

### Check-in
- 提交 `project_template/docs/checkin_week4.md`（模板见 `CHECKINS.md`）
- 提交 `project_template/data/` 下样例数据（可脱敏/抽样）或数据获取说明

---

## Week 5 — Feature Engineering & LLM Integration

### 目标
- 结构化特征工程（流行度、均分、活跃度、类别偏好等）
- 文本特征工程（embedding 或 LLM 信息抽取）
- 形成“可缓存、可复用”的特征产物

### 推荐落地方式（优先顺序）
1. **Embedding（推荐）**：对 item `text` 生成向量并缓存（`features/items_emb.parquet`）  
2. **LLM 抽取（可选）**：从文本抽取主题/标签/情绪并写回结构化列（`features/items_tags.parquet`）

### Check-in
- 提交特征列表与生成方式（`features/README.md` 或简短文档）
- 提交 embedding/LLM 结果缓存文件的生成脚本（不要把大文件直接提交到 git）

---

## Week 6 — Model Implementation：经典 + LLM/Embedding 增强

### 目标
- 至少实现并对比 2 种推荐算法（经典）+ 1 种增强（embedding/LLM）
- 完成离线评估与 ablation（有/无增强对比）

### 建议组合（易讲、易对比）
- 经典：`KernelMF`（或 `BaselineModel`） + `ItemItemCF`（或 `UserUserCF`）
- 增强：Embedding 候选召回 + MF rerank 或混合打分

### Check-in
- 提交评估结果表格（Top-K 指标 +（可选）RMSE）
- 提交 2–3 个失败案例分析（推荐不准/冷启动/长尾等）

---

## Week 7 — Packaging & Deployment

### 目标
- 把训练好的模型与特征导出为可服务的 artifacts
- 做一个可交互 demo（API 或 UI），支持自由文本输入并返回推荐

### Check-in
- GitHub 仓库可运行：新环境按 README 能启动 demo
- Demo 具备至少一种解释（相似原因/偏好原因/同类用户原因）

---

## Week 8 — Final Demo & Showcase

### 目标
- 完整演示 + 清晰讲述设计选择
- 说明优势/限制/下一步改进

### 展示建议结构
- Problem → Data → Baselines → Enhancement → Evaluation → Demo → Limitations → Next steps

