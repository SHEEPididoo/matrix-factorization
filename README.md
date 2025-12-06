# 推荐系统实现（Recommendation Systems）

本课程项目实现了多种推荐系统算法，包括基于内容的推荐、协同过滤（用户-用户、物品-物品）和基于矩阵分解的模型。

## 📚 课程内容

**Build a basic recommender using content-based filtering and implement user-user and item-item collaborative filtering.**

本课程涵盖以下推荐系统方法：

1. **基于内容的推荐（Content-Based Filtering）** - 📘 [学习顺序：第1个](#-第一阶段基础推荐系统方法)
2. **协同过滤（Collaborative Filtering）** - 📘 [学习顺序：第2-3个](#-第一阶段基础推荐系统方法)
   - 用户-用户协同过滤（User-User CF）
   - 物品-物品协同过滤（Item-Item CF）
3. **基于模型的协同过滤（Model-Based CF）** - 📘 [学习顺序：第4个](#-第二阶段高级模型方法)
   - 矩阵分解（Matrix Factorization）
   - 核矩阵分解（Kernel Matrix Factorization）
4. **推荐系统评估与实际实现** - 📘 [学习顺序：第5个](#-第三阶段评估与实践)
   - 评估指标、相似度度量、离线/在线评估、冷启动问题

---

## 🎯 推荐系统方法详解

### 1. 基于内容的推荐（Content-Based Filtering）

**核心思想：**
- 通过分析物品的特征（如电影的类型、演员、导演等）来推荐与用户历史偏好相似的物品
- 如果用户喜欢某个物品，那么具有相似特征的其他物品也应该被推荐

**工作原理：**
1. 提取物品特征（如电影类型、演员、导演等）
2. 构建用户偏好档案（基于用户历史评分）
3. 计算物品之间的相似度（基于特征）
4. 推荐与用户偏好最相似的物品

**优点：**
- ✅ **可解释性强**：可以解释为什么推荐某个物品（基于特征相似度）
- ✅ **冷启动友好**：新用户只要有历史偏好就能推荐，新物品只要有特征就能被推荐
- ✅ **不受流行度影响**：不会因为物品流行就推荐，可以推荐小众但符合用户偏好的物品

**缺点：**
- ❌ **特征工程复杂**：需要提取和选择好的特征
- ❌ **推荐多样性有限**：可能过度推荐相似物品
- ❌ **无法发现新兴趣**：只能基于已有偏好推荐

**使用示例：**
```python
from matrix_factorization import ContentBasedRecommender
import pandas as pd

# 加载数据和物品特征
movie_data = pd.read_csv('data/ml-100k/u.data', ...)
item_features = pd.read_csv('data/ml-100k/u.item', ...)  # 包含电影类型等特征

# 训练模型
model = ContentBasedRecommender(min_rating=1, max_rating=5)
model.fit(X_train, y_train, item_features=item_features)

# 生成推荐
recommendations = model.recommend(user=200, amount=10)
```

---

### 2. 用户-用户协同过滤（User-User Collaborative Filtering）

**核心思想：**
- 通过找到与目标用户相似的其他用户，然后推荐这些相似用户喜欢的物品
- 相似的用户会有相似的偏好

**工作原理：**
1. 构建用户-物品评分矩阵
2. 计算用户之间的相似度（余弦相似度或皮尔逊相关系数）
3. 找到与目标用户最相似的K个用户
4. 基于相似用户的评分，加权预测目标用户对物品的评分

**预测公式：**
```
pred(u, i) = mean_u + Σ(sim(u, v) × (rating(v, i) - mean_v)) / Σ|sim(u, v)|
```
其中：
- `u` 是目标用户
- `i` 是目标物品
- `v` 是相似用户
- `sim(u, v)` 是用户相似度
- `mean_u` 是用户u的平均评分

**优点：**
- ✅ **能够发现用户的潜在兴趣**：通过相似用户发现新物品
- ✅ **推荐多样性好**：可以推荐不同类型的物品
- ✅ **不需要物品特征**：只需要用户-物品评分矩阵

**缺点：**
- ❌ **冷启动问题**：新用户没有足够评分，难以找到相似用户
- ❌ **稀疏性问题**：用户-物品矩阵通常很稀疏
- ❌ **计算复杂度高**：需要计算所有用户之间的相似度
- ❌ **可扩展性差**：用户数量增长时，相似度矩阵会变得非常大

**使用示例：**
```python
from matrix_factorization import UserUserCF

# 训练模型
model = UserUserCF(
    min_rating=1, 
    max_rating=5, 
    n_neighbors=50,  # 使用50个最相似的用户
    similarity_metric='cosine'
)
model.fit(X_train, y_train)

# 生成推荐
recommendations = model.recommend(user=200, amount=10)
```

---

### 3. 物品-物品协同过滤（Item-Item Collaborative Filtering）

**核心思想：**
- 通过找到与目标物品相似的其他物品，然后基于用户对这些相似物品的评分来预测
- 相似的物品会收到相似的评分

**工作原理：**
1. 构建用户-物品评分矩阵
2. 计算物品之间的相似度（余弦相似度或皮尔逊相关系数）
3. 找到与目标物品最相似的K个物品
4. 基于用户对相似物品的评分，加权预测用户对目标物品的评分

**预测公式：**
```
pred(u, i) = mean_i + Σ(sim(i, j) × (rating(u, j) - mean_j)) / Σ|sim(i, j)|
```
其中：
- `u` 是目标用户
- `i` 是目标物品
- `j` 是相似物品
- `sim(i, j)` 是物品相似度
- `mean_i` 是物品i的平均评分

**优点：**
- ✅ **物品相似度更稳定**：物品特征变化较慢，相似度矩阵可以预先计算和缓存
- ✅ **可扩展性更好**：物品数量通常远少于用户数量，相似度矩阵更小
- ✅ **推荐解释性强**：可以解释为"因为您喜欢X，而Y与X相似，所以推荐Y"
- ✅ **实时推荐效率高**：可以快速为新用户生成推荐

**缺点：**
- ❌ **冷启动问题**：新物品没有足够评分，难以找到相似物品
- ❌ **稀疏性问题**：用户-物品矩阵通常很稀疏

**为什么物品-物品通常优于用户-用户？**
- 物品数量通常远少于用户数量（如Netflix有数亿用户但只有数万电影）
- 物品相似度更稳定，可以预先计算
- 推荐解释更直观（"因为您喜欢X，推荐相似的Y"）

**使用示例：**
```python
from matrix_factorization import ItemItemCF

# 训练模型
model = ItemItemCF(
    min_rating=1, 
    max_rating=5, 
    n_neighbors=50,  # 使用50个最相似的物品
    similarity_metric='cosine'
)
model.fit(X_train, y_train)

# 生成推荐
recommendations = model.recommend(user=200, amount=10)
```

---

### 4. 基于模型的协同过滤（Model-Based CF）

#### 4.1 基线模型（Baseline Model）

简单的偏差模型，将用户-物品评分建模为：
```
r_ui = μ + bias_u + bias_i
```
其中：
- `μ` 是全局平均评分
- `bias_u` 是用户偏差
- `bias_i` 是物品偏差

**训练方法：**
- **SGD（随机梯度下降）**：迭代优化用户和物品偏差
- **ALS（交替最小二乘法）**：交替优化用户偏差和物品偏差

#### 4.2 矩阵分解（Matrix Factorization）

将用户-物品评分矩阵分解为两个低维矩阵：
```
R ≈ P × Q^T
```
其中：
- `P` 是用户特征矩阵 (n_users × n_factors)
- `Q` 是物品特征矩阵 (n_items × n_factors)
- `n_factors` 是潜在因子数量

**预测公式：**
```
r_ui = μ + bias_u + bias_i + P_u · Q_i^T
```

**核函数：**
- **线性核（Linear）**：标准点积
- **Sigmoid核**：使用sigmoid函数进行非线性变换
- **RBF核（径向基函数）**：基于高斯核的相似度

**优点：**
- ✅ **处理稀疏数据**：通过潜在因子捕获用户和物品的隐含特征
- ✅ **可扩展性好**：可以处理大规模数据
- ✅ **在线更新**：支持新用户的在线学习

**使用示例：**
```python
from matrix_factorization import KernelMF

# 训练模型
model = KernelMF(
    n_epochs=20,
    n_factors=100,
    kernel='linear',  # 或 'sigmoid', 'rbf'
    lr=0.001,
    reg=0.005
)
model.fit(X_train, y_train)

# 在线更新新用户
model.update_users(X_new_users, y_new_users, n_epochs=20)

# 生成推荐
recommendations = model.recommend(user=200, amount=10)
```

---

## 📁 项目结构

```
week3/
├── data/
│   └── ml-100k/                      # MovieLens 100K数据集
├── examples/
│   ├── [0] example.py                        # 快速示例（可选）
│   ├── [1] content-based-filtering.ipynb     # 基于内容推荐（第1个学习）
│   ├── [2] user-user-cf.ipynb                # 用户-用户协同过滤（第2个学习）
│   ├── [3] item-item-cf.ipynb                # 物品-物品协同过滤（第3个学习）
│   ├── [4] recommender-system.ipynb          # 矩阵分解完整示例（第4个学习）
│   └── [5] recommender-evaluation.ipynb      # 评估与实际实现（第5个学习）
├── matrix_factorization/
│   ├── __init__.py                           # 模块导出
│   ├── recommender_base.py                   # 推荐系统基类
│   ├── baseline_model.py                     # 基线模型
│   ├── kernel_matrix_factorization.py        # 核矩阵分解
│   ├── content_based.py                      # 基于内容推荐
│   ├── collaborative_filtering.py            # 协同过滤（用户-用户、物品-物品）
│   ├── kernels.py                            # 核函数实现
│   └── utils.py                              # 工具函数
├── README.md                                  # 本文件
└── requirements.txt                           # 依赖包
```

**注意：** 文件名前的 `[数字]` 表示推荐的学习顺序

---

## 🚀 快速开始

### 安装依赖

```bash
pip install -r requirements.txt
```

---

## 📖 学习顺序（推荐阅读路径）

本课程建议按照以下顺序学习，从基础到进阶，循序渐进：

### 第一阶段：基础推荐系统方法

#### 📘 1. 基于内容的推荐（Content-Based Filtering）
**文件：** `examples/content-based-filtering.ipynb`

**学习目标：**
- 理解基于内容的推荐原理
- 学习如何使用物品特征进行推荐
- 掌握特征工程的基本方法

**为什么先学这个？**
- 最直观易懂，不需要其他用户数据
- 可以处理冷启动问题
- 为后续方法打下基础

---

#### 📘 2. 用户-用户协同过滤（User-User Collaborative Filtering）
**文件：** `examples/user-user-cf.ipynb`

**学习目标：**
- 理解协同过滤的核心思想
- 学习如何计算用户相似度
- 掌握基于相似用户的推荐方法

**为什么学这个？**
- 理解"物以类聚，人以群分"的推荐思想
- 学习相似度度量的应用
- 为理解物品-物品协同过滤做准备

---

#### 📘 3. 物品-物品协同过滤（Item-Item Collaborative Filtering）
**文件：** `examples/item-item-cf.ipynb`

**学习目标：**
- 理解物品相似度的概念
- 学习最常用的协同过滤方法
- 掌握为什么物品-物品通常优于用户-用户

**为什么学这个？**
- 这是实际应用中最常用的方法
- 理解可扩展性的重要性
- 学习推荐系统的工程实践

---

### 第二阶段：高级模型方法

#### 📘 4. 矩阵分解（Matrix Factorization）
**文件：** `examples/recommender-system.ipynb`

**学习目标：**
- 理解矩阵分解的原理
- 学习潜在因子模型
- 掌握不同核函数的使用
- 学习在线更新机制

**为什么学这个？**
- 理解现代推荐系统的核心算法
- 学习如何处理大规模稀疏数据
- 掌握模型训练和调优方法

**包含内容：**
- 基线模型（Baseline Model）
- 矩阵分解（线性核、Sigmoid核、RBF核）
- 在线学习（Online Learning）
- Scikit-learn兼容性

---

### 第三阶段：评估与实践

#### 📘 5. 推荐系统评估与实际实现
**文件：** `examples/recommender-evaluation.ipynb`

**学习目标：**
- 掌握推荐系统的评估指标
- 理解相似度度量的原理
- 学习离线评估和在线A/B测试
- 解决冷启动问题
- 构建完整的评估管道

**为什么最后学这个？**
- 需要先理解各种推荐方法
- 评估方法适用于所有模型
- 这是实际项目中的关键技能

**包含内容：**
- 评估指标：RMSE, Precision, Recall, F1-score
- 相似度度量：余弦相似度、欧氏距离、皮尔逊相关系数
- 离线评估 vs 在线A/B测试
- 冷启动问题及处理策略
- 模型性能比较

---

### 快速参考

#### 📘 0. 快速示例（可选）
**文件：** `examples/example.py`

**用途：**
- 快速了解基本用法
- 代码示例参考
- 不需要深入学习，可作为快速参考

---

## 📚 详细学习路径

### 初学者路径（推荐）

1. **第一步**：阅读 `content-based-filtering.ipynb`
   - 理解推荐系统的基本概念
   - 学习最简单的推荐方法

2. **第二步**：阅读 `user-user-cf.ipynb`
   - 理解协同过滤思想
   - 学习相似度计算

3. **第三步**：阅读 `item-item-cf.ipynb`
   - 学习最实用的协同过滤方法
   - 理解为什么物品-物品更常用

4. **第四步**：阅读 `recommender-system.ipynb`
   - 深入学习矩阵分解
   - 理解现代推荐系统算法

5. **第五步**：阅读 `recommender-evaluation.ipynb`
   - 学习如何评估推荐系统
   - 掌握实际项目技能

### 进阶路径

如果你已经熟悉基础概念，可以：
- 直接学习 `recommender-system.ipynb` 和 `recommender-evaluation.ipynb`
- 重点关注模型调优和评估方法

### 实践路径

如果你想快速上手：
1. 先看 `example.py` 了解基本用法
2. 选择一种方法深入学习（推荐 `item-item-cf.ipynb`）
3. 学习 `recommender-evaluation.ipynb` 进行评估

---

## 🎯 学习建议

1. **循序渐进**：按照推荐顺序学习，不要跳跃
2. **动手实践**：每个notebook都要运行代码，理解结果
3. **对比学习**：学完所有方法后，对比它们的优缺点
4. **项目实践**：尝试在自己的数据集上应用这些方法
5. **深入理解**：不仅要会用，还要理解原理

---

## 📁 文件说明

### Examples目录结构

```
examples/
├── [0] example.py                    # 快速示例（可选）
├── [1] content-based-filtering.ipynb # 基于内容推荐
├── [2] user-user-cf.ipynb            # 用户-用户协同过滤
├── [3] item-item-cf.ipynb            # 物品-物品协同过滤
├── [4] recommender-system.ipynb      # 矩阵分解（完整示例）
└── [5] recommender-evaluation.ipynb  # 评估与实际实现
```

**注意：** 文件名前的 `[数字]` 表示推荐的学习顺序

---

## 📊 方法对比

| 方法 | 优点 | 缺点 | 适用场景 |
|------|------|------|----------|
| **基于内容** | 可解释性强、冷启动友好 | 特征工程复杂、推荐多样性有限 | 有丰富物品特征的场景 |
| **用户-用户CF** | 发现潜在兴趣、推荐多样 | 计算复杂度高、可扩展性差 | 用户数量较少的场景 |
| **物品-物品CF** | 可扩展性好、解释性强 | 冷启动问题 | **最常用**，大多数推荐系统 |
| **矩阵分解** | 处理稀疏数据、可扩展 | 可解释性较弱 | 大规模数据、需要在线更新 |

---

## 🔧 API 参考

### ContentBasedRecommender

```python
ContentBasedRecommender(min_rating=0, max_rating=5, verbose=0)
```

**方法：**
- `fit(X, y, item_features=None)`: 训练模型
- `predict(X, bound_ratings=True)`: 预测评分
- `recommend(user, amount=10, items_known=None)`: 生成推荐

### UserUserCF

```python
UserUserCF(min_rating=0, max_rating=5, n_neighbors=50, similarity_metric='cosine', verbose=0)
```

**方法：**
- `fit(X, y)`: 训练模型
- `predict(X, bound_ratings=True)`: 预测评分
- `recommend(user, amount=10, items_known=None)`: 生成推荐

### ItemItemCF

```python
ItemItemCF(min_rating=0, max_rating=5, n_neighbors=50, similarity_metric='cosine', verbose=0)
```

**方法：**
- `fit(X, y)`: 训练模型
- `predict(X, bound_ratings=True)`: 预测评分
- `recommend(user, amount=10, items_known=None)`: 生成推荐

### KernelMF

```python
KernelMF(n_factors=100, n_epochs=100, kernel='linear', lr=0.01, reg=1, verbose=1)
```

**方法：**
- `fit(X, y)`: 训练模型
- `predict(X, bound_ratings=True)`: 预测评分
- `update_users(X, y, lr=0.01, n_epochs=20)`: 在线更新新用户
- `recommend(user, amount=10, items_known=None)`: 生成推荐

---

## 🎓 完整学习路径总结

```
开始学习
    ↓
[1] 基于内容的推荐
    ├─ 理解推荐系统基础概念
    ├─ 学习特征工程
    └─ 掌握最简单的推荐方法
    ↓
[2] 用户-用户协同过滤
    ├─ 理解协同过滤思想
    ├─ 学习相似度计算
    └─ 掌握基于用户的推荐
    ↓
[3] 物品-物品协同过滤
    ├─ 学习最实用的协同过滤方法
    ├─ 理解可扩展性
    └─ 掌握工程实践
    ↓
[4] 矩阵分解
    ├─ 深入学习现代推荐算法
    ├─ 学习潜在因子模型
    ├─ 掌握不同核函数
    └─ 学习在线更新机制
    ↓
[5] 评估与实际实现
    ├─ 掌握评估指标
    ├─ 理解相似度度量
    ├─ 学习离线/在线评估
    ├─ 解决冷启动问题
    └─ 构建评估管道
    ↓
完成学习，可以开始实际项目！
```

---

## 📖 参考资料

- Steffen Rendle, Lars Schmidt-Thieme. [Online-updating regularized kernel matrix factorization models for large-scale recommender systems](https://dl.acm.org/doi/10.1145/1454008.1454047)
- [MovieLens Dataset](https://grouplens.org/datasets/movielens/)

---

## 📝 许可证

本项目采用 MIT 许可证。

---

## 🙏 致谢

感谢 MovieLens 提供的数据集。
