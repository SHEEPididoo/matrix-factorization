## matrix_factorization 代码通读指南（从接口到实现流程）

这份文档的目标是：带着你把 `matrix_factorization/` 目录从头到尾“按调用链”读一遍，搞清楚每个模型如何 **fit → predict → recommend →（可选）update_users**，以及数据在各层如何流动。

---

## 1. 目录速览：每个文件负责什么

- **`__init__.py`**：对外导出统一 API（你在 notebook / 示例里 `from matrix_factorization import ...` 基本都从这里来）。
- **`recommender_base.py`**：所有推荐模型的抽象基类（接口 + 通用能力）。
  - 统一做：ID 映射、数据预处理、Top-N 推荐 `recommend()`。
- **`baseline_model.py`**：最小可运行的显式评分模型（global mean + user bias + item bias）。
  - 适合用来理解：SGD/ALS 训练循环、冷启动回退、`predict()` 的数据流。
- **`kernel_matrix_factorization.py`**：真正的矩阵分解（KernelMF）。
  - 通过 SGD 学习：bias + latent factors（P/Q），并支持 linear/sigmoid/rbf 三种 kernel。
- **`kernels.py`**：KernelMF 的“数学核心”与 SGD 单步更新。
  - `kernel_*`：给定 user/item 特征向量计算预测值。
  - `kernel_*_sgd_update`：对单个 (u,i,r) 做一次梯度更新。
- **`collaborative_filtering.py`**：User-User CF 和 Item-Item CF（仍遵守同一接口）。
- **`content_based.py`**：基于内容的推荐（同样遵守接口，但其 `fit()` 还可额外接收 `item_features`）。
- **`utils.py`**：与“更新/冷启动评估”相关的数据切分工具。

---

## 2. 总体心智模型：所有模型都遵循同一条主线

你可以把所有模型的生命周期统一理解为：

1. **训练（fit）**：
   - 输入：`X = DataFrame[user_id, item_id]`，`y = Series[rating]`
   - 先做：`_preprocess_data(type="fit")` → 产生内部整数 ID（0..n-1）
   - 再做：模型自己的参数初始化与训练（SGD / ALS / 相似度矩阵等）

2. **预测（predict）**：
   - 输入：`X = DataFrame[user_id, item_id]`
   - 先做：`_preprocess_data(type="predict")` → 未见过的 user/item 会被映射为 `-1`
   - 再做：模型自己的预测逻辑（并可选择 clip 到 `[min_rating,max_rating]`）

3. **推荐（recommend）**：
   - 输入：原始 `user`（外部 ID）、可选 `items_known`
   - 内部：拼一个“该 user × 候选 items”的表，调用 `predict()`，按 `rating_pred` 排序取 Top-N

4. **增量更新（update_users，可选）**：
   - 主要面向“新用户冷启动/增量”场景：只更新 user 侧参数，通常锁住 item 侧参数。

---

## 3. 第一站：统一骨架 `RecommenderBase`（必须先读懂）

### 3.1 `_preprocess_data()`：贯穿 fit/predict/update 的“数据入口”

它负责把外部 ID 转为内部 ID，并根据模式处理未知 ID：

- **`type="fit"`**：
  - 检查重复 (user,item)
  - 建立 `user_id_map` / `item_id_map`
  - 记录 `n_users/n_items`

- **`type="predict"`**：
  - 用已有映射去 map
  - **未知 user 或 item 会变成 `NaN`，再被填为 `-1`**（这就是冷启动回退的触发条件）

- **`type="update"`**：
  - 只保留“已知 item”（新用户更新时，通常只允许对已知物品做更新）
  - 新用户会被加入 `user_id_map`，并返回 `known_users/new_users` 方便模型扩容参数

通读时你要抓住一个点：**所有模型的 fit/predict/update 都是先调用它**。

### 3.2 `recommend()`：Top-N 推荐的统一实现

`recommend()` 不关心模型内部细节，它只需要模型实现了 `predict()`：

- 构造候选物品集合
- 对每个候选物品预测评分
- 排序取 Top-N

因此：**只要你实现了 `predict()`，就天然获得 Top-N 推荐能力**。

---

## 4. 第二站：用 `BaselineModel` 快速建立“闭环感”

### 4.1 模型形式

\[
\hat r_{ui} = \mu + b_u + b_i
\]

- `global_mean = μ`
- `user_biases[u] = b_u`
- `item_biases[i] = b_i`

### 4.2 `fit()` 的流程

1. `X = _preprocess_data(type="fit")`
2. `global_mean = mean(ratings)`
3. 初始化参数为 0
4. 进入参数估计：
   - `method="sgd"`：逐条样本更新 bias
   - `method="als"`：交替闭式解 user_bias 与 item_bias

### 4.3 `predict()` 的冷启动回退

预测前会 `type="predict"`，因此可能出现：

- user 未见过 → `user_id = -1`
- item 未见过 → `item_id = -1`

Baseline 的预测逻辑是：

- 先从 `global_mean` 起步
- user_known 才加 `user_bias`
- item_known 才加 `item_bias`

这就是你在所有模型里都能看到的“冷启动最小回退”：**全局均值 + 能用的偏置/特征**。

---

## 5. 第三站：`KernelMF`（矩阵分解）是怎么训练与预测的

### 5.1 参数与表示

KernelMF 同时学习：

- `user_biases` / `item_biases`
- `user_features = P`（形状：`(n_users, n_factors)`）
- `item_features = Q`（形状：`(n_items, n_factors)`）

### 5.2 `fit()` 主线

1. `X = _preprocess_data(type="fit")`
2. `global_mean = mean(ratings)`
3. 初始化：
   - bias 为 0
   - P/Q 为正态随机初始化（`init_mean/init_sd`）
4. 进入 `_sgd()`：按 epoch 循环，shuffle 数据，逐条样本更新参数

### 5.3 kernel 的作用：把“打分函数”抽象出来

KernelMF 支持三种 kernel：

- **linear**：
  - 其实就是标准 MF：\(\mu + b_u + b_i + P_u \cdot Q_i\)
- **sigmoid**：
  - 先算线性和，再 sigmoid，再用 `a=min_rating`、`c=max-min` 缩放到评分区间
- **rbf**：
  - 基于 \(\|P_u - Q_i\|\) 的 RBF 形式（也用 `a/c` 缩放）

这些都在 `kernels.py` 的 `kernel_linear/kernel_sigmoid/kernel_rbf` 里。

### 5.4 SGD 单步更新在哪里？

核心点：`kernel_matrix_factorization.py` 的 `_sgd()` 并不自己写梯度细节，而是 **把“单步更新”委派给 `kernels.py`**：

- `kernel_linear_sgd_update()`
- `kernel_sigmoid_sgd_update()`
- `kernel_rbf_sgd_update()`

你通读时建议：

- 先完整读懂 `kernel_linear_sgd_update()`（它最像标准 MF）
- 再快速扫另外两个（主要差异是预测公式与导数链）

### 5.5 `predict()` 如何处理未知 user/item

同样来自 `type="predict"` 的 `-1`：

- user 未知 → user_bias=0，P_u=0 向量
- item 未知 → item_bias=0，Q_i=0 向量
- 因此最小回退就是 `global_mean`（以及你能用到的那部分）

---

## 6. 第四站：传统方法也“接入同一接口”

### 6.1 `UserUserCF` / `ItemItemCF`

- `fit()`：先 `_preprocess_data(type="fit")`，再 pivot 成矩阵（缺失填 0），再算相似度矩阵
- `predict()`：逐行 iterrows，对每个 (u,i) 用相似用户/相似物品做加权回归
- 冷启动：user 或 item 为 `-1` → 回退 global_mean

这些模型没有 numba/SGD，但 **接口一致**，因此仍可直接用 `recommend()`。

### 6.2 `ContentBasedRecommender`

它的 `fit()` 允许额外传入 `item_features`：

- 传了 `item_features`：构建 item 相似度矩阵，再根据用户评分历史构建用户 profile
- 没传 `item_features`：退化为“用户平均评分档案”（更像 baseline 的简化版）

同样：`predict()` 接口一致，`recommend()` 可直接复用。

---

## 7. 冷启动/更新评估：`utils.train_update_test_split`

`train_update_test_split(X, frac_new_users)` 专门用来构造“新用户更新”实验：

- 先随机挑一部分用户作为 **新用户集合**
- 其他用户全部进 `train_initial`
- 新用户的交互再一分为二：
  - `train_update`：用于 `model.update_users(...)`
  - `test_update`：用于更新后评估（RMSE 等）

典型用法顺序：

1. `model.fit(X_train_initial, y_train_initial)`
2. `model.update_users(X_train_update, y_train_update)`
3. `pred = model.predict(X_test_update)` 与 `y_test_update` 比较

---

## 8. 端到端“最小调用链”示例（读代码时用来对照）

下面示例展示完整主线：训练 → 预测 → 推荐 → 新用户更新评估。

```python
import pandas as pd
from matrix_factorization import KernelMF, train_update_test_split

# ratings: 必须至少包含三列：user_id, item_id, rating
# ratings = pd.read_csv(...)

X = ratings[["user_id", "item_id"]]
y = ratings["rating"]

# 1) 冷启动/更新实验切分
(
    X_train_initial,
    y_train_initial,
    X_train_update,
    y_train_update,
    X_test_update,
    y_test_update,
) = train_update_test_split(ratings, frac_new_users=0.2)

# 2) 训练初始模型
model = KernelMF(n_factors=50, n_epochs=20, kernel="linear", lr=0.01, reg=0.01, verbose=1)
model.fit(X_train_initial, y_train_initial)

# 3) 对新用户做增量更新（只更新 user 侧）
model.update_users(X_train_update, y_train_update, lr=0.01, n_epochs=10)

# 4) 评估：对新用户未见过的一半交互做预测
pred = model.predict(X_test_update)

# 5) 推荐：给某个用户做 Top-N
user = ratings["user_id"].iloc[0]
rec_df = model.recommend(user=user, amount=10)
print(rec_df.head())
```

读代码时你可以把它映射到：

- `fit()` → `RecommenderBase._preprocess_data(type="fit")` → 模型训练循环
- `update_users()` → `RecommenderBase._preprocess_data(type="update")` → 参数扩容 → SGD（锁 item）
- `predict()` → `RecommenderBase._preprocess_data(type="predict")` → 模型 `_predict`
- `recommend()` → 构造候选表 → `predict()` → 排序

---

## 9. 建议的“通读节奏”（一次读懂不迷路）

- **第 1 轮（只看调用链）**：
  1) `recommender_base.py`（理解 `_preprocess_data` 与 `recommend`）
  2) `baseline_model.py`（理解 SGD/ALS 与冷启动）

- **第 2 轮（看矩阵分解细节）**：
  3) `kernel_matrix_factorization.py`（看 `fit/predict/update_users` 如何组织）
  4) `kernels.py`（只要吃透 linear 的单步更新，另外两个快速对照）

- **第 3 轮（扩展：传统方法接入统一接口）**：
  5) `collaborative_filtering.py`、`content_based.py`

- **第 4 轮（评估/冷启动实验）**：
  6) `utils.py` + `examples/recommender-evaluation.ipynb`

---

## 10. 你读完应该能回答的 6 个问题（自检）

1. 为什么所有模型都先调用 `_preprocess_data`？它在 fit/predict/update 三种模式分别干什么？
2. `recommend()` 为什么无需知道模型内部实现？它依赖的最小接口是什么？
3. BaselineModel 的冷启动回退逻辑是什么？`predictions_possible` 用来干什么？
4. KernelMF 的训练 loop 在哪里？为什么 `_sgd()` 里不直接写梯度？
5. 三种 kernel 的差异本质是什么（打分函数不同 + 导数不同）？
6. `train_update_test_split` 为什么要把新用户的交互一分为二？更新时为什么通常锁 item 参数？

---

## 附：关键定位（你在 IDE 里跳转用）

- 基类入口：`recommender_base.py` → `_preprocess_data()`、`recommend()`
- 最小闭环：`baseline_model.py` → `fit()`、`predict()`、`_sgd()`、`_predict()`
- MF 主线：`kernel_matrix_factorization.py` → `KernelMF.fit/predict/update_users`、`_sgd()`、`_predict()`
- kernel 与单步更新：`kernels.py` → `kernel_*`、`kernel_*_sgd_update`
- 冷启动切分：`utils.py` → `train_update_test_split()`
