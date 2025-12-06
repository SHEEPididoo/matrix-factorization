"""
[0] 快速示例 - 推荐系统基础用法

这是一个快速入门示例，展示如何使用矩阵分解模型进行推荐。
适合快速了解基本用法，不需要深入学习。

学习顺序：这是可选文件，可以在任何时间查看作为快速参考。
"""

from matrix_factorization import BaselineModel, KernelMF, train_update_test_split

import pandas as pd
from sklearn.metrics import mean_squared_error

# Movie data found here https://grouplens.org/datasets/movielens/
cols = ["user_id", "item_id", "rating", "timestamp"]
movie_data = pd.read_csv(
    "../data/ml-100k/u.data", names=cols, sep="\t", usecols=[0, 1, 2], engine="python"
)

X = movie_data[["user_id", "item_id"]]
y = movie_data["rating"]

# Prepare data for online learning
(
    X_train_initial,
    y_train_initial,
    X_train_update,
    y_train_update,
    X_test_update,
    y_test_update,
) = train_update_test_split(movie_data, frac_new_users=0.2)

# Initial training
matrix_fact = KernelMF(n_epochs=20, n_factors=100, verbose=1, lr=0.001, reg=0.005)
matrix_fact.fit(X_train_initial, y_train_initial)

# Update model with new users
matrix_fact.update_users(
    X_train_update, y_train_update, lr=0.001, n_epochs=20, verbose=1
)
pred = matrix_fact.predict(X_test_update)
mse = mean_squared_error(y_test_update, pred)
rmse = mse ** 0.5
print(f"\nTest RMSE: {rmse:.4f}")

# Get recommendations
user = 200
items_known = X_train_initial.query("user_id == @user")["item_id"]
matrix_fact.recommend(user=user, items_known=items_known)
