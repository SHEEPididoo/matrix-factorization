"""
推荐系统基础模块
提供所有推荐系统模型的抽象基类
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin

from abc import ABCMeta, abstractmethod
from typing import Any, Tuple, Union


class RecommenderBase(BaseEstimator, RegressorMixin, metaclass=ABCMeta):
    """
    所有推荐系统模型的抽象基类
    所有子类必须实现 fit() 和 predict() 方法
    
    这是推荐系统的核心抽象类，定义了推荐系统的基本接口和通用功能。
    它继承自 scikit-learn 的 BaseEstimator 和 RegressorMixin，使其能够与
    scikit-learn 的工具（如 GridSearchCV）兼容。

    参数:
        min_rating (int): 最小评分值，默认为 0
        max_rating (int): 最大评分值，默认为 5
        verbose (int): 训练时的详细输出级别。0 表示不输出，1 表示输出训练过程，默认为 1

    属性:
        n_users (int): 用户数量
        n_items (int): 物品数量
        global_mean (float): 所有评分的全局均值
        user_id_map (dict): 用户ID到内部整数ID的映射字典
        item_id_map (dict): 物品ID到内部整数ID的映射字典
        known_users (set): 已知用户ID的集合
        known_items (set): 已知物品ID的集合
    """

    @abstractmethod
    def __init__(self, min_rating: float = 0, max_rating: float = 5, verbose: int = 0):
        """
        初始化推荐系统基类
        
        参数:
            min_rating (float): 最小评分值，默认为 0
            max_rating (float): 最大评分值，默认为 5
            verbose (int): 详细输出级别，默认为 0
        """
        self.min_rating = min_rating
        self.max_rating = max_rating
        self.verbose = verbose
        return

    @property
    def known_users(self):
        """
        获取所有已知用户ID的集合
        
        返回:
            set: 已知用户ID的集合
        """
        return set(self.user_id_map.keys())

    @property
    def known_items(self):
        """
        获取所有已知物品ID的集合
        
        返回:
            set: 已知物品ID的集合
        """
        return set(self.item_id_map.keys())

    def contains_user(self, user_id: Any) -> bool:
        """
        检查模型是否在包含给定用户ID的数据上训练过
        
        参数:
            user_id (Any): 用户ID
            
        返回:
            bool: 如果用户ID已知则返回 True，否则返回 False
        """
        return user_id in self.known_users

    def contains_item(self, item_id: Any) -> bool:
        """
        检查模型是否在包含给定物品ID的数据上训练过
        
        参数:
            item_id (Any): 物品ID
            
        返回:
            bool: 如果物品ID已知则返回 True，否则返回 False
        """
        return item_id in self.known_items

    def _preprocess_data(
        self, X: pd.DataFrame, y: pd.Series = None, type: str = "fit"
    ) -> Union[pd.DataFrame, Tuple[pd.DataFrame, list, list]]:
        """
        在执行 fit、update 或 predict 之前的数据预处理步骤
        
        这个函数负责：
        1. 提取必要的列（user_id, item_id）
        2. 检查重复的用户-物品评分
        3. 创建或更新用户ID和物品ID的映射
        4. 将原始ID转换为内部整数ID
        5. 处理新用户（在update模式下）
        
        参数:
            X (pd.DataFrame): 包含 user_id 和 item_id 列的数据框
            y (pd.Series, optional): 包含评分的序列，在 predict 模式下可以为 None
            type (str): 预处理类型。可选值：'fit'（训练）、'predict'（预测）、'update'（更新），默认为 'fit'
        
        返回:
            X (pd.DataFrame): 处理后的数据框，包含 user_id, item_id 和 rating（如果提供）
            known_users (list, 仅update模式): 在 X 中已经存在的用户列表，仅在 type='update' 时返回
            new_users (list, 仅update模式): 在 X 中的新用户列表，仅在 type='update' 时返回
        """
        X = X.loc[:, ["user_id", "item_id"]]

        if type != "predict":
            X["rating"] = y

        if type in ("fit", "update"):
            # Check for duplicate user-item ratings
            if X.duplicated(subset=["user_id", "item_id"]).sum() != 0:
                raise ValueError("Duplicate user-item ratings in matrix")

            # Shuffle rows
            X = X.sample(frac=1, replace=False)

        if type == "fit":
            # Create mapping of user_id and item_id to assigned integer ids
            user_ids = X["user_id"].unique()
            item_ids = X["item_id"].unique()
            self.user_id_map = {user_id: i for (i, user_id) in enumerate(user_ids)}
            self.item_id_map = {item_id: i for (i, item_id) in enumerate(item_ids)}
            self.n_users = len(user_ids)
            self.n_items = len(item_ids)

        elif type == "update":
            # Keep only item ratings for which the item is already known
            items = self.item_id_map.keys()
            X = X.query("item_id in @items").copy()

            # Add information on new users
            new_users, known_users = [], []
            users = X["user_id"].unique()
            new_user_id = max(self.user_id_map.values()) + 1

            for user in users:
                if user in self.user_id_map.keys():
                    known_users.append(user)
                    continue

                # Add to user id mapping
                new_users.append(user)
                self.user_id_map[user] = new_user_id
                new_user_id += 1

        # Remap user id and item id to assigned integer ids
        X.loc[:, "user_id"] = X["user_id"].map(self.user_id_map)
        X.loc[:, "item_id"] = X["item_id"].map(self.item_id_map)

        if type == "predict":
            # Replace missing mappings with -1
            X.fillna(-1, inplace=True)

        if type == "update":
            return X, known_users, new_users
        else:
            return X

    @abstractmethod
    def fit(self, X: pd.DataFrame, y: pd.Series):
        """
        在给定数据上训练模型
        
        这是一个抽象方法，必须在子类中实现。它应该：
        1. 预处理数据
        2. 初始化模型参数
        3. 执行训练算法（如SGD、ALS等）
        4. 保存训练好的参数
        
        参数:
            X (pd.DataFrame): 包含 user_id 和 item_id 列的数据框
            y (pd.Series): 包含评分的序列
            
        返回:
            self: 返回自身以支持链式调用
        """
        return self

    @abstractmethod
    def predict(self, X: pd.DataFrame, bound_ratings: bool = True) -> list:
        """
        预测给定用户和物品的评分
        
        这是一个抽象方法，必须在子类中实现。它应该：
        1. 预处理输入数据
        2. 使用训练好的模型参数计算预测评分
        3. 可选择地将评分限制在 [min_rating, max_rating] 范围内
        
        参数:
            X (pd.DataFrame): 包含 user_id 和 item_id 列的数据框
            bound_ratings (bool): 是否将评分限制在 [min_rating, max_rating] 范围内，默认为 True
        
        返回:
            list: 包含所有用户-物品对预测评分的列表，顺序与输入 X 相同
        """
        return []

    def recommend(
        self,
        user: Any,
        amount: int = 10,
        items_known: list = None,
        include_user: bool = True,
        bound_ratings: bool = True,
    ) -> pd.DataFrame:
        """
        为给定用户返回推荐物品的DataFrame，按评分从高到低排序
        
        这个方法会：
        1. 获取所有可推荐的物品（排除用户已知的物品）
        2. 为每个物品预测评分
        3. 按预测评分降序排序
        4. 返回前 N 个推荐物品
        
        参数:
            user (Any): 要为其生成推荐的用户ID（使用原始用户ID，不是内部映射的ID）
            amount (int): 要返回的推荐物品数量，默认为 10
            items_known (list, optional): 用户已知的物品列表，这些物品将不会出现在推荐中。默认为 None
            include_user (bool, optional): 是否在输出DataFrame中包含user_id列，默认为 True
            bound_ratings (bool): 是否将评分限制在 [min_rating, max_rating] 范围内，默认为 True
        
        返回:
            pd.DataFrame: 包含推荐物品的DataFrame，列包括：
                - user_id (可选): 用户ID
                - item_id: 物品ID
                - rating_pred: 预测评分
                按评分从高到低排序
        """
        items = list(self.item_id_map.keys())

        # If items_known is provided then filter by items that the user does not know
        if items_known is not None:
            items_known = list(items_known)
            items = [item for item in items if item not in items_known]

        # Get rating predictions for given user and all unknown items
        items_recommend = pd.DataFrame({"user_id": user, "item_id": items})
        items_recommend["rating_pred"] = self.predict(
            X=items_recommend, bound_ratings=False
        )

        # Sort and keep top n items
        items_recommend.sort_values(by="rating_pred", ascending=False, inplace=True)
        items_recommend = items_recommend.head(amount)

        # Bound ratings
        if bound_ratings:
            items_recommend["rating_pred"] = items_recommend["rating_pred"].clip(
                lower=self.min_rating, upper=self.max_rating
            )

        if not include_user:
            items_recommend.drop(["user_id"], axis="columns", inplace=True)

        return items_recommend

