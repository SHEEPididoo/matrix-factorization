"""
基于内容的推荐系统（Content-Based Filtering）
根据物品的特征和用户的历史偏好来推荐相似物品
"""

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Optional

from .recommender_base import RecommenderBase


class ContentBasedRecommender(RecommenderBase):
    """
    基于内容的推荐系统
    
    该方法通过分析物品的特征（如电影的类型、演员、导演等）来推荐与用户历史偏好相似的物品。
    核心思想是：如果用户喜欢某个物品，那么具有相似特征的其他物品也应该被推荐。
    
    参数:
        min_rating (float): 最小评分值，默认为 0
        max_rating (float): 最大评分值，默认为 5
        verbose (int): 详细输出级别，默认为 0
    
    属性:
        item_features (pd.DataFrame): 物品特征矩阵
        user_profiles (dict): 用户偏好档案，键为用户ID，值为特征向量
        item_similarity_matrix (np.ndarray): 物品相似度矩阵
    """
    
    def __init__(self, min_rating: float = 0, max_rating: float = 5, verbose: int = 0):
        super().__init__(min_rating=min_rating, max_rating=max_rating, verbose=verbose)
        self.item_features = None
        self.user_profiles = {}
        self.item_similarity_matrix = None
        
    def fit(self, X: pd.DataFrame, y: pd.Series, item_features: pd.DataFrame = None):
        """
        训练基于内容的推荐模型
        
        参数:
            X (pd.DataFrame): 包含 user_id 和 item_id 列的训练数据
            y (pd.Series): 用户对物品的评分
            item_features (pd.DataFrame, optional): 物品特征数据框，包含 item_id 和特征列
                                                    如果为None，将从训练数据中构建用户偏好档案
        """
        # 预处理数据
        X = self._preprocess_data(X=X, y=y, type="fit")
        X["rating"] = y.values
        
        # 计算全局平均评分
        self.global_mean = X["rating"].mean()
        
        # 如果没有提供物品特征，则基于用户评分构建用户偏好档案
        if item_features is None:
            self._build_user_profiles(X)
        else:
            # 使用物品特征构建推荐系统
            self.item_features = item_features.copy()
            # 确保item_features中的item_id映射到内部ID
            if 'item_id' in self.item_features.columns:
                self.item_features['item_id'] = self.item_features['item_id'].map(self.item_id_map)
                self.item_features = self.item_features.dropna(subset=['item_id'])
                self.item_features['item_id'] = self.item_features['item_id'].astype(int)
            self._build_item_similarity()
            self._build_user_profiles_from_features(X)
            
        return self
    
    def _build_user_profiles(self, X: pd.DataFrame):
        """
        基于用户评分历史构建用户偏好档案
        
        参数:
            X (pd.DataFrame): 包含 user_id, item_id, rating 的数据框
        """
        # 计算每个用户的平均评分作为用户偏好
        user_ratings = X.groupby('user_id')['rating'].mean()
        
        for user_id in self.user_id_map.keys():
            internal_id = self.user_id_map[user_id]
            if user_id in user_ratings.index:
                self.user_profiles[internal_id] = user_ratings[user_id]
            else:
                self.user_profiles[internal_id] = self.global_mean
                
    def _build_user_profiles_from_features(self, X: pd.DataFrame):
        """
        基于物品特征和用户评分构建用户偏好档案
        
        参数:
            X (pd.DataFrame): 包含 user_id, item_id, rating 的数据框
        """
        # 确保item_features包含item_id列
        if 'item_id' not in self.item_features.columns:
            raise ValueError("item_features must contain 'item_id' column")
            
        # 获取特征列（排除item_id）
        feature_cols = [col for col in self.item_features.columns if col != 'item_id']
        
        # 为每个用户构建偏好向量
        for user_id in self.user_id_map.keys():
            internal_id = self.user_id_map[user_id]
            
            # 获取该用户评分过的物品
            user_items = X[X['user_id'] == internal_id]
            
            if len(user_items) == 0:
                # 如果没有评分，使用零向量
                self.user_profiles[internal_id] = np.zeros(len(feature_cols))
            else:
                # 计算加权平均特征向量
                profile = np.zeros(len(feature_cols))
                total_weight = 0
                
                for _, row in user_items.iterrows():
                    item_id = int(row['item_id'])
                    rating = row['rating']
                    
                    # 获取物品特征
                    item_row = self.item_features[self.item_features['item_id'] == item_id]
                    if len(item_row) > 0:
                        item_features = item_row[feature_cols].values[0]
                        # 使用评分作为权重
                        weight = rating - self.min_rating  # 归一化评分
                        profile += item_features * weight
                        total_weight += weight
                
                if total_weight > 0:
                    profile /= total_weight
                    
                self.user_profiles[internal_id] = profile
                
    def _build_item_similarity(self):
        """
        构建物品相似度矩阵（基于物品特征）
        """
        if self.item_features is None:
            return
            
        # 获取特征列
        feature_cols = [col for col in self.item_features.columns if col != 'item_id']
        feature_matrix = self.item_features[feature_cols].values
        
        # 计算余弦相似度
        self.item_similarity_matrix = cosine_similarity(feature_matrix)
        
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
                # 基于用户偏好和物品特征计算预测
                if self.item_features is not None and self.item_similarity_matrix is not None:
                    # 使用物品相似度
                    pred = self._predict_with_similarity(user_id, item_id)
                else:
                    # 使用用户偏好档案
                    pred = self.user_profiles.get(user_id, self.global_mean)
            
            # 限制评分范围
            if bound_ratings:
                pred = max(self.min_rating, min(self.max_rating, pred))
                
            predictions.append(pred)
            
        return predictions
    
    def _predict_with_similarity(self, user_id: int, item_id: int) -> float:
        """
        使用物品相似度矩阵进行预测
        
        参数:
            user_id (int): 用户内部ID
            item_id (int): 物品内部ID
            
        返回:
            float: 预测评分
        """
        # 获取用户评分过的物品
        user_rated_items = []
        user_ratings = []
        
        # 这里需要访问训练数据，简化处理：使用用户偏好档案
        if user_id in self.user_profiles:
            # 如果物品在相似度矩阵中
            if item_id < len(self.item_similarity_matrix):
                # 找到最相似的物品（简化版本）
                similarities = self.item_similarity_matrix[item_id]
                # 使用用户偏好档案的平均值
                pred = self.user_profiles[user_id]
                if isinstance(pred, np.ndarray):
                    pred = np.mean(pred) if len(pred) > 0 else self.global_mean
            else:
                pred = self.user_profiles[user_id]
                if isinstance(pred, np.ndarray):
                    pred = np.mean(pred) if len(pred) > 0 else self.global_mean
        else:
            pred = self.global_mean
            
        return pred

