"""
协同过滤推荐系统（Collaborative Filtering）
包括用户-用户协同过滤和物品-物品协同过滤
"""

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from typing import Optional

from .recommender_base import RecommenderBase


class UserUserCF(RecommenderBase):
    """
    用户-用户协同过滤（User-User Collaborative Filtering）
    
    该方法通过找到与目标用户相似的其他用户，然后推荐这些相似用户喜欢的物品。
    核心思想是：相似的用户会有相似的偏好。
    
    参数:
        min_rating (float): 最小评分值，默认为 0
        max_rating (float): 最大评分值，默认为 5
        n_neighbors (int): 用于预测的相似用户数量，默认为 50
        similarity_metric (str): 相似度计算方法，'cosine' 或 'pearson'，默认为 'cosine'
        verbose (int): 详细输出级别，默认为 0
    
    属性:
        user_item_matrix (pd.DataFrame): 用户-物品评分矩阵
        user_similarity_matrix (np.ndarray): 用户相似度矩阵
        user_mean_ratings (pd.Series): 每个用户的平均评分
    """
    
    def __init__(
        self, 
        min_rating: float = 0, 
        max_rating: float = 5, 
        n_neighbors: int = 50,
        similarity_metric: str = 'cosine',
        verbose: int = 0
    ):
        super().__init__(min_rating=min_rating, max_rating=max_rating, verbose=verbose)
        self.n_neighbors = n_neighbors
        self.similarity_metric = similarity_metric
        self.user_item_matrix = None
        self.user_similarity_matrix = None
        self.user_mean_ratings = None
        
    def fit(self, X: pd.DataFrame, y: pd.Series):
        """
        训练用户-用户协同过滤模型
        
        参数:
            X (pd.DataFrame): 包含 user_id 和 item_id 列的训练数据
            y (pd.Series): 用户对物品的评分
        """
        # 预处理数据
        X = self._preprocess_data(X=X, y=y, type="fit")
        X["rating"] = y.values
        
        # 构建用户-物品评分矩阵
        self.user_item_matrix = X.pivot_table(
            index='user_id', 
            columns='item_id', 
            values='rating', 
            fill_value=0
        )
        
        # 计算全局平均评分
        self.global_mean = X["rating"].mean()
        
        # 计算每个用户的平均评分（用于中心化）
        self.user_mean_ratings = self.user_item_matrix.mean(axis=1)
        
        # 计算用户相似度矩阵
        self._compute_user_similarity()
        
        return self
    
    def _compute_user_similarity(self):
        """
        计算用户之间的相似度矩阵
        """
        if self.similarity_metric == 'cosine':
            # 使用余弦相似度
            # 先中心化（减去用户平均评分）
            centered_matrix = self.user_item_matrix.sub(self.user_mean_ratings, axis=0)
            # 计算余弦相似度
            self.user_similarity_matrix = cosine_similarity(centered_matrix.values)
        elif self.similarity_metric == 'pearson':
            # 使用皮尔逊相关系数
            centered_matrix = self.user_item_matrix.sub(self.user_mean_ratings, axis=0)
            # 皮尔逊相关系数可以通过余弦相似度计算（在中心化后）
            self.user_similarity_matrix = cosine_similarity(centered_matrix.values)
        else:
            raise ValueError(f"Unknown similarity metric: {self.similarity_metric}")
    
    def predict(self, X: pd.DataFrame, bound_ratings: bool = True) -> list:
        """
        预测用户对物品的评分
        
        参数:
            X (pd.DataFrame): 包含 user_id 和 item_id 的数据框
            bound_ratings (bool): 是否将评分限制在范围内，默认为 True
            
        返回:
            list: 预测评分列表
        """
        if X.shape[0] == 0:
            return []
            
        X = self._preprocess_data(X=X, type="predict")
        predictions = []
        
        for _, row in X.iterrows():
            user_id = int(row['user_id'])
            item_id = int(row['item_id'])
            
            # 如果用户或物品未知，返回全局均值
            if user_id == -1 or item_id == -1:
                pred = self.global_mean if hasattr(self, 'global_mean') else (self.min_rating + self.max_rating) / 2
            else:
                pred = self._predict_rating(user_id, item_id)
            
            # 限制评分范围
            if bound_ratings:
                pred = max(self.min_rating, min(self.max_rating, pred))
                
            predictions.append(pred)
            
        return predictions
    
    def _predict_rating(self, user_id: int, item_id: int) -> float:
        """
        使用相似用户预测评分
        
        参数:
            user_id (int): 用户内部ID
            item_id (int): 物品内部ID
            
        返回:
            float: 预测评分
        """
        # 检查用户和物品是否在矩阵中
        if user_id >= len(self.user_similarity_matrix) or item_id >= self.user_item_matrix.shape[1]:
            return self.user_mean_ratings.iloc[user_id] if user_id < len(self.user_mean_ratings) else self.global_mean
        
        # 获取该用户与其他用户的相似度
        user_similarities = self.user_similarity_matrix[user_id]
        
        # 获取所有用户对该物品的评分
        item_ratings = self.user_item_matrix.iloc[:, item_id].values
        
        # 找到对该物品有评分的用户
        rated_mask = item_ratings > 0
        
        if not rated_mask.any():
            # 如果没有人评分过这个物品，返回用户平均评分
            return self.user_mean_ratings.iloc[user_id] if user_id < len(self.user_mean_ratings) else self.global_mean
        
        # 获取相似度和对应的评分
        similarities = user_similarities[rated_mask]
        ratings = item_ratings[rated_mask]
        user_means = self.user_mean_ratings.values[rated_mask]
        
        # 排除自己（相似度为1）
        if user_id < len(similarities):
            mask = np.arange(len(similarities)) != np.where(rated_mask)[0][np.where(rated_mask)[0] == user_id][0] if user_id in np.where(rated_mask)[0] else np.ones(len(similarities), dtype=bool)
            similarities = similarities[mask] if mask.any() else similarities
            ratings = ratings[mask] if mask.any() else ratings
            user_means = user_means[mask] if mask.any() else user_means
        
        # 选择最相似的k个用户
        if len(similarities) > self.n_neighbors:
            top_k_indices = np.argsort(similarities)[-self.n_neighbors:]
            similarities = similarities[top_k_indices]
            ratings = ratings[top_k_indices]
            user_means = user_means[top_k_indices]
        
        # 加权平均预测
        # 公式: pred = user_mean + sum(sim * (rating - other_user_mean)) / sum(|sim|)
        numerator = np.sum(similarities * (ratings - user_means))
        denominator = np.sum(np.abs(similarities))
        
        if denominator == 0:
            pred = self.user_mean_ratings.iloc[user_id] if user_id < len(self.user_mean_ratings) else self.global_mean
        else:
            pred = self.user_mean_ratings.iloc[user_id] + numerator / denominator
        
        return pred


class ItemItemCF(RecommenderBase):
    """
    物品-物品协同过滤（Item-Item Collaborative Filtering）
    
    该方法通过找到与目标物品相似的其他物品，然后基于用户对这些相似物品的评分来预测。
    核心思想是：相似的物品会收到相似的评分。
    
    参数:
        min_rating (float): 最小评分值，默认为 0
        max_rating (float): 最大评分值，默认为 5
        n_neighbors (int): 用于预测的相似物品数量，默认为 50
        similarity_metric (str): 相似度计算方法，'cosine' 或 'pearson'，默认为 'cosine'
        verbose (int): 详细输出级别，默认为 0
    
    属性:
        user_item_matrix (pd.DataFrame): 用户-物品评分矩阵
        item_similarity_matrix (np.ndarray): 物品相似度矩阵
        item_mean_ratings (pd.Series): 每个物品的平均评分
    """
    
    def __init__(
        self, 
        min_rating: float = 0, 
        max_rating: float = 5, 
        n_neighbors: int = 50,
        similarity_metric: str = 'cosine',
        verbose: int = 0
    ):
        super().__init__(min_rating=min_rating, max_rating=max_rating, verbose=verbose)
        self.n_neighbors = n_neighbors
        self.similarity_metric = similarity_metric
        self.user_item_matrix = None
        self.item_similarity_matrix = None
        self.item_mean_ratings = None
        
    def fit(self, X: pd.DataFrame, y: pd.Series):
        """
        训练物品-物品协同过滤模型
        
        参数:
            X (pd.DataFrame): 包含 user_id 和 item_id 列的训练数据
            y (pd.Series): 用户对物品的评分
        """
        # 预处理数据
        X = self._preprocess_data(X=X, y=y, type="fit")
        X["rating"] = y.values
        
        # 构建用户-物品评分矩阵（转置后是物品-用户矩阵）
        self.user_item_matrix = X.pivot_table(
            index='user_id', 
            columns='item_id', 
            values='rating', 
            fill_value=0
        )
        
        # 计算全局平均评分
        self.global_mean = X["rating"].mean()
        
        # 计算每个物品的平均评分（用于中心化）
        self.item_mean_ratings = self.user_item_matrix.mean(axis=0)
        
        # 计算物品相似度矩阵
        self._compute_item_similarity()
        
        return self
    
    def _compute_item_similarity(self):
        """
        计算物品之间的相似度矩阵
        """
        if self.similarity_metric == 'cosine':
            # 使用余弦相似度
            # 先中心化（减去物品平均评分）
            centered_matrix = self.user_item_matrix.sub(self.item_mean_ratings, axis=1)
            # 转置后计算余弦相似度（物品之间的相似度）
            self.item_similarity_matrix = cosine_similarity(centered_matrix.values.T)
        elif self.similarity_metric == 'pearson':
            # 使用皮尔逊相关系数
            centered_matrix = self.user_item_matrix.sub(self.item_mean_ratings, axis=1)
            self.item_similarity_matrix = cosine_similarity(centered_matrix.values.T)
        else:
            raise ValueError(f"Unknown similarity metric: {self.similarity_metric}")
    
    def predict(self, X: pd.DataFrame, bound_ratings: bool = True) -> list:
        """
        预测用户对物品的评分
        
        参数:
            X (pd.DataFrame): 包含 user_id 和 item_id 的数据框
            bound_ratings (bool): 是否将评分限制在范围内，默认为 True
            
        返回:
            list: 预测评分列表
        """
        if X.shape[0] == 0:
            return []
            
        X = self._preprocess_data(X=X, type="predict")
        predictions = []
        
        for _, row in X.iterrows():
            user_id = int(row['user_id'])
            item_id = int(row['item_id'])
            
            # 如果用户或物品未知，返回全局均值
            if user_id == -1 or item_id == -1:
                pred = self.global_mean if hasattr(self, 'global_mean') else (self.min_rating + self.max_rating) / 2
            else:
                pred = self._predict_rating(user_id, item_id)
            
            # 限制评分范围
            if bound_ratings:
                pred = max(self.min_rating, min(self.max_rating, pred))
                
            predictions.append(pred)
            
        return predictions
    
    def _predict_rating(self, user_id: int, item_id: int) -> float:
        """
        使用相似物品预测评分
        
        参数:
            user_id (int): 用户内部ID
            item_id (int): 物品内部ID
            
        返回:
            float: 预测评分
        """
        # 检查用户和物品是否在矩阵中
        if user_id >= self.user_item_matrix.shape[0] or item_id >= len(self.item_similarity_matrix):
            return self.item_mean_ratings.iloc[item_id] if item_id < len(self.item_mean_ratings) else self.global_mean
        
        # 获取该物品与其他物品的相似度
        item_similarities = self.item_similarity_matrix[item_id]
        
        # 获取该用户对所有物品的评分
        user_ratings = self.user_item_matrix.iloc[user_id, :].values
        
        # 找到该用户评分过的物品
        rated_mask = user_ratings > 0
        
        if not rated_mask.any():
            # 如果用户没有评分过任何物品，返回物品平均评分
            return self.item_mean_ratings.iloc[item_id] if item_id < len(self.item_mean_ratings) else self.global_mean
        
        # 获取相似度和对应的评分
        similarities = item_similarities[rated_mask]
        ratings = user_ratings[rated_mask]
        item_means = self.item_mean_ratings.values[rated_mask]
        
        # 排除自己（相似度为1）
        if item_id < len(similarities):
            mask = np.arange(len(similarities)) != np.where(rated_mask)[0][np.where(rated_mask)[0] == item_id][0] if item_id in np.where(rated_mask)[0] else np.ones(len(similarities), dtype=bool)
            similarities = similarities[mask] if mask.any() else similarities
            ratings = ratings[mask] if mask.any() else ratings
            item_means = item_means[mask] if mask.any() else item_means
        
        # 选择最相似的k个物品
        if len(similarities) > self.n_neighbors:
            top_k_indices = np.argsort(similarities)[-self.n_neighbors:]
            similarities = similarities[top_k_indices]
            ratings = ratings[top_k_indices]
            item_means = item_means[top_k_indices]
        
        # 加权平均预测
        # 公式: pred = item_mean + sum(sim * (rating - other_item_mean)) / sum(|sim|)
        numerator = np.sum(similarities * (ratings - item_means))
        denominator = np.sum(np.abs(similarities))
        
        if denominator == 0:
            pred = self.item_mean_ratings.iloc[item_id] if item_id < len(self.item_mean_ratings) else self.global_mean
        else:
            pred = self.item_mean_ratings.iloc[item_id] + numerator / denominator
        
        return pred

