## Check-in 提交清单（Week4–Week8）

你可以要求学员每周提交一个 markdown（或 notebook）+ 必要脚本。下面给出建议模板，便于统一批改。

---

## Week 4 Check-in：数据集与探索

建议文件：`project_template/docs/checkin_week4.md`

最低内容：
- **Domain**：你做什么推荐（电影/书籍/商品/音乐…），用户目标是什么？
- **Dataset**：数据来源、链接/获取方式、许可证（license）
- **字段字典**：ratings/items 两张表字段说明
- **数据统计**：
  - #users/#items/#interactions
  - 稀疏度、长尾分布（至少 1 张图）
  - 冷启动比例（交互 < N 的用户/物品占比）
- **切分策略**：随机/时间/留一法，为什么
- **清洗规则**：去重、异常值处理、缺失处理
- **可复现命令**：如何一键生成 `ratings.parquet/items.parquet`

---

## Week 5 Check-in：特征与 LLM/Embedding

建议文件：`project_template/docs/checkin_week5.md`

最低内容：
- **特征列表**：结构化特征、文本特征分别有哪些
- **Embedding 模型**：用的哪种模型（或 API），输入是什么文本，向量维度
- **缓存策略**：特征写到哪里、如何复用、如何避免重复调用
- **样例展示**：给 3 个 item 展示文本、抽取标签/embedding 相似结果

---

## Week 6 Check-in：模型对比与评估

建议文件：`project_template/docs/checkin_week6.md`

最低内容：
- **实现的模型**：至少 2 个经典模型 + 1 个增强（或混合）
- **离线评估指标**：
  - Top-K：Precision@K / Recall@K / NDCG@K（至少两个）
  - （可选）RMSE（显式评分才建议）
- **对比表格**：同一切分、同一 K，输出结果表
- **Ablation**：增强模块 on/off 对比
- **失败案例**：至少 2 个案例 + 你认为的原因

---

## Week 7 Check-in：可运行 Demo

建议文件：`project_template/docs/checkin_week7.md`

最低内容：
- **Demo 入口**：API 或 UI，如何启动
- **交互能力**：自由文本输入 → 推荐结果（Top-N）
- **解释**：至少一种解释（相似度/偏好/同类用户）
- **工程说明**：模型与特征从哪里加载（artifacts/features），启动耗时与缓存策略

---

## Week 8 Final：展示与复盘

建议文件：`project_template/docs/final.md`

最低内容：
- **设计选择**：为什么选这个模型/特征/混合策略
- **优势与限制**：长尾/冷启动/多样性/偏差等
- **未来扩展**：在线学习、A/B 测试、反馈闭环、召回-排序-重排等

