"""
A股ETF预测系统 - ETF预测模型模块
使用LightGBM进行ETF涨跌预测
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Tuple
import logging
import pickle
import json
from pathlib import Path

import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, ndcg_score

from config.settings import MODEL_CONFIG, MODEL_DIR
from data.etf_data_processor import ETFFeatureEngineer

logger = logging.getLogger(__name__)


class ETFPredictModel:
    """ETF预测模型"""
    
    def __init__(self):
        self.model = None
        self.feature_engineer = ETFFeatureEngineer()
        self.feature_columns = self.feature_engineer.get_feature_columns()
        self.model_config = MODEL_CONFIG
        self.model_path = MODEL_DIR / "etf_predict_model.pkl"
        self.feature_importance_path = MODEL_DIR / "etf_feature_importance.csv"
    
    def train(self, df: pd.DataFrame, 
              train_start: str = None,
              train_end: str = None,
              valid_ratio: float = 0.2) -> Dict:
        """
        训练ETF预测模型
        """
        logger.info("开始ETF模型训练...")
        
        # 数据过滤
        if train_start:
            df = df[df["date"] >= train_start]
        if train_end:
            df = df[df["date"] <= train_end]
        
        # 移除标签为空的行
        df = df.dropna(subset=["label_score"])
        
        if len(df) < 50:
            logger.error(f"ETF训练数据不足: {len(df)} 行")
            return {"status": "error", "message": "训练数据不足"}
        
        # 准备特征和标签
        available_features = [c for c in self.feature_columns if c in df.columns]
        X = df[available_features].fillna(0)
        y = df["label_score"]
        
        # 划分训练集和验证集
        split_idx = int(len(X) * (1 - valid_ratio))
        X_train, X_valid = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_valid = y.iloc[:split_idx], y.iloc[split_idx:]
        
        logger.info(f"ETF训练集: {len(X_train)} 样本, 验证集: {len(X_valid)} 样本")
        logger.info(f"特征数: {len(available_features)}")
        
        # 创建LightGBM数据集
        train_data = lgb.Dataset(X_train, label=y_train)
        valid_data = lgb.Dataset(X_valid, label=y_valid, reference=train_data)
        
        # 训练参数
        params = {
            "objective": "regression",
            "metric": "mse",
            "boosting_type": "gbdt",
            "num_leaves": 31,
            "learning_rate": 0.05,
            "feature_fraction": 0.8,
            "bagging_fraction": 0.8,
            "bagging_freq": 5,
            "verbose": -1,
            "n_estimators": 300,
            "early_stopping_rounds": 30,
        }
        
        self.model = lgb.train(
            params,
            train_data,
            valid_sets=[train_data, valid_data],
            valid_names=["train", "valid"],
            num_boost_round=params.get("n_estimators", 300),
            callbacks=[
                lgb.early_stopping(params.get("early_stopping_rounds", 30)),
                lgb.log_evaluation(period=50)
            ]
        )
        
        # 计算验证集指标
        y_pred = self.model.predict(X_valid)
        mse = mean_squared_error(y_valid, y_pred)
        
        # 计算NDCG
        df_valid = df.iloc[split_idx:].copy()
        df_valid["pred_score"] = y_pred
        
        ndcg_scores = []
        for date in df_valid["date"].unique():
            date_df = df_valid[df_valid["date"] == date]
            if len(date_df) < 2:
                continue
            true_relevance = date_df["label_score"].values.reshape(1, -1)
            pred_scores = date_df["pred_score"].values.reshape(1, -1)
            try:
                ndcg = ndcg_score(true_relevance, pred_scores, k=5)
                ndcg_scores.append(ndcg)
            except:
                pass
        
        avg_ndcg = np.mean(ndcg_scores) if ndcg_scores else 0
        
        # 保存模型
        self.save_model()
        
        # 保存特征重要性
        self._save_feature_importance(available_features)
        
        result = {
            "status": "success",
            "train_samples": len(X_train),
            "valid_samples": len(X_valid),
            "mse": mse,
            "ndcg@5": avg_ndcg,
            "best_iteration": self.model.best_iteration,
        }
        
        logger.info(f"ETF模型训练完成: MSE={mse:.4f}, NDCG@5={avg_ndcg:.4f}")
        
        return result
    
    def predict(self, df: pd.DataFrame, top_k: int = 5) -> pd.DataFrame:
        """
        预测ETF得分
        """
        if self.model is None:
            self.load_model()
            if self.model is None:
                logger.error("ETF模型未训练或加载失败")
                return pd.DataFrame()
        
        # 准备特征
        available_features = [c for c in self.feature_columns if c in df.columns]
        X = df[available_features].fillna(0)
        
        # 预测
        scores = self.model.predict(X)
        
        # 构建结果
        result = df[["date", "etf_code", "etf_name"]].copy()
        result["pred_score"] = scores
        
        # 添加预测理由
        result = self._add_prediction_reasons(result, df)
        
        # 排序并取Top-K
        result = result.sort_values("pred_score", ascending=False)
        result["rank"] = range(1, len(result) + 1)
        
        top_result = result.head(top_k)
        
        logger.info(f"ETF预测完成，Top-{top_k}: {top_result['etf_name'].tolist()}")
        
        return top_result
    
    def _add_prediction_reasons(self, result: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
        """添加ETF预测理由"""
        reasons = []
        
        for idx in result.index:
            reason_list = []
            
            # 检查动量
            if "return_5d" in df.columns:
                ret = df.loc[idx, "return_5d"]
                if ret > 2:
                    reason_list.append(f"5日涨幅{ret:.1f}%，趋势向上")
                elif ret < -2:
                    reason_list.append(f"5日跌幅{abs(ret):.1f}%，超跌反弹预期")
            
            # 检查资金流
            if "main_net_inflow" in df.columns:
                inflow = df.loc[idx, "main_net_inflow"]
                if inflow is not None and inflow > 0:
                    reason_list.append(f"资金净流入")
            
            # 检查量能
            if "volume_ratio" in df.columns:
                vol_ratio = df.loc[idx, "volume_ratio"]
                if vol_ratio > 1.5:
                    reason_list.append("放量突破")
                elif vol_ratio < 0.6:
                    reason_list.append("缩量蓄势")
            
            # 检查RSI
            if "rsi_14" in df.columns:
                rsi = df.loc[idx, "rsi_14"]
                if rsi < 30:
                    reason_list.append("RSI超卖")
                elif rsi > 70:
                    reason_list.append("RSI强势")
            
            # 检查价格位置
            if "price_position" in df.columns:
                pos = df.loc[idx, "price_position"]
                if pos < 0.3:
                    reason_list.append("价格低位")
                elif pos > 0.8:
                    reason_list.append("突破高位")
            
            reasons.append("; ".join(reason_list) if reason_list else "综合因子评分较高")
        
        result["prediction_reason"] = reasons
        return result
    
    def save_model(self):
        """保存模型"""
        if self.model is not None:
            MODEL_DIR.mkdir(parents=True, exist_ok=True)
            with open(self.model_path, "wb") as f:
                pickle.dump(self.model, f)
            logger.info(f"ETF模型已保存: {self.model_path}")
    
    def load_model(self) -> bool:
        """加载模型"""
        if self.model_path.exists():
            with open(self.model_path, "rb") as f:
                self.model = pickle.load(f)
            logger.info(f"ETF模型已加载: {self.model_path}")
            return True
        logger.warning("ETF模型文件不存在")
        return False
    
    def _save_feature_importance(self, feature_names: List[str]):
        """保存特征重要性"""
        importance = self.model.feature_importance(importance_type="gain")
        
        df_importance = pd.DataFrame({
            "feature": feature_names,
            "importance": importance
        }).sort_values("importance", ascending=False)
        
        df_importance.to_csv(self.feature_importance_path, index=False)
        logger.info(f"ETF特征重要性已保存: {self.feature_importance_path}")


class ETFRollingTrainer:
    """ETF滚动训练器"""
    
    def __init__(self, model: ETFPredictModel):
        self.model = model
        self.train_interval_days = 7  # 每周重训练
    
    def should_retrain(self, last_train_date: str = None) -> bool:
        """判断是否需要重新训练"""
        if last_train_date is None:
            return True
        
        try:
            last_date = datetime.strptime(last_train_date, "%Y-%m-%d")
            days_since = (datetime.now() - last_date).days
            return days_since >= self.train_interval_days
        except:
            return True
    
    def rolling_train(self, df: pd.DataFrame, train_window_months: int = 6) -> Dict:
        """滚动训练"""
        if df.empty:
            return {"status": "error", "message": "无训练数据"}
        
        # 计算训练窗口
        end_date = df["date"].max()
        start_date = (datetime.strptime(end_date, "%Y-%m-%d") - 
                      timedelta(days=train_window_months * 30)).strftime("%Y-%m-%d")
        
        logger.info(f"ETF滚动训练: {start_date} ~ {end_date}")
        
        return self.model.train(df, train_start=start_date, train_end=end_date)
