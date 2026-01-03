"""
A股板块涨停预测系统 - 模型训练模块
使用LightGBM进行板块涨停预测
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

from config.settings import MODEL_CONFIG, MODEL_DIR, RISK_CONFIG
from data.data_processor import FeatureEngineer

logger = logging.getLogger(__name__)


class SectorPredictModel:
    """板块涨停预测模型"""
    
    def __init__(self):
        self.model = None
        self.feature_engineer = FeatureEngineer()
        self.feature_columns = self.feature_engineer.get_feature_columns()
        self.model_config = MODEL_CONFIG
        self.model_path = MODEL_DIR / "sector_predict_model.pkl"
        self.feature_importance_path = MODEL_DIR / "feature_importance.csv"
    
    def train(self, df: pd.DataFrame, 
              train_start: str = None, 
              train_end: str = None,
              valid_ratio: float = 0.2) -> Dict:
        """
        训练模型
        
        Args:
            df: 包含特征和标签的数据
            train_start: 训练开始日期
            train_end: 训练结束日期
            valid_ratio: 验证集比例
            
        Returns:
            训练结果统计
        """
        logger.info("开始模型训练...")
        
        # 数据过滤
        if train_start:
            df = df[df["date"] >= train_start]
        if train_end:
            df = df[df["date"] <= train_end]
        
        # 移除标签为空的行
        df = df.dropna(subset=["label_score"])
        
        if len(df) < 100:
            logger.error(f"训练数据不足: {len(df)} 行")
            return {"status": "error", "message": "训练数据不足"}
        
        # 准备特征和标签
        available_features = [c for c in self.feature_columns if c in df.columns]
        X = df[available_features].fillna(0)
        y = df["label_score"]
        
        # 划分训练集和验证集（按时间顺序）
        split_idx = int(len(X) * (1 - valid_ratio))
        X_train, X_valid = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_valid = y.iloc[:split_idx], y.iloc[split_idx:]
        
        logger.info(f"训练集: {len(X_train)} 样本, 验证集: {len(X_valid)} 样本")
        logger.info(f"特征数: {len(available_features)}")
        
        # 创建LightGBM数据集
        train_data = lgb.Dataset(X_train, label=y_train)
        valid_data = lgb.Dataset(X_valid, label=y_valid, reference=train_data)
        
        # 训练模型
        params = self.model_config["lgbm_params"].copy()
        
        self.model = lgb.train(
            params,
            train_data,
            valid_sets=[train_data, valid_data],
            valid_names=["train", "valid"],
            num_boost_round=params.get("n_estimators", 500),
            callbacks=[
                lgb.early_stopping(params.get("early_stopping_rounds", 50)),
                lgb.log_evaluation(period=100)
            ]
        )
        
        # 计算验证集指标
        y_pred = self.model.predict(X_valid)
        mse = mean_squared_error(y_valid, y_pred)
        
        # 计算排名指标（NDCG）
        # 按日期分组计算NDCG
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
        
        logger.info(f"模型训练完成: MSE={mse:.4f}, NDCG@5={avg_ndcg:.4f}")
        
        return result
    
    def predict(self, df: pd.DataFrame, top_k: int = None) -> pd.DataFrame:
        """
        预测板块得分
        
        Args:
            df: 包含特征的数据（最新一天）
            top_k: 返回Top-K板块
            
        Returns:
            预测结果DataFrame
        """
        if self.model is None:
            self.load_model()
            if self.model is None:
                logger.error("模型未训练或加载失败")
                return pd.DataFrame()
        
        if top_k is None:
            top_k = self.model_config.get("top_k", 5)
        
        # 准备特征
        available_features = [c for c in self.feature_columns if c in df.columns]
        X = df[available_features].fillna(0)
        
        # 预测
        scores = self.model.predict(X)
        
        # 构建结果
        result = df[["date", "sector_id", "sector_name"]].copy()
        result["pred_score"] = scores
        
        # 添加预测理由
        result = self._add_prediction_reasons(result, df)
        
        # 应用风控过滤
        result = self._apply_risk_filter(result, df)
        
        # 排序并取Top-K
        result = result.sort_values("pred_score", ascending=False)
        result["rank"] = range(1, len(result) + 1)
        
        top_result = result.head(top_k)
        
        logger.info(f"预测完成，Top-{top_k} 板块: {top_result['sector_name'].tolist()}")
        
        return top_result
    
    def _add_prediction_reasons(self, result: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
        """添加预测理由"""
        reasons = []
        
        for idx in result.index:
            reason_list = []
            
            # 检查资金流入
            if "main_net_inflow" in df.columns:
                inflow = df.loc[idx, "main_net_inflow"]
                if inflow > 0:
                    reason_list.append(f"资金净流入{inflow/1e8:.2f}亿")
            
            # 检查量价背离
            if "divergence_factor" in df.columns:
                div = df.loc[idx, "divergence_factor"]
                if div > 0.3:
                    reason_list.append("量价背离(吸筹信号)")
            
            # 检查涨停惯性
            if "limit_up_ma_3" in df.columns:
                lu_ma = df.loc[idx, "limit_up_ma_3"]
                if lu_ma > 1:
                    reason_list.append(f"近3日均涨停{lu_ma:.1f}家")
            
            # 检查资金蓄力
            if "money_accumulation_5" in df.columns:
                acc = df.loc[idx, "money_accumulation_5"]
                if acc > 2:
                    reason_list.append("资金持续蓄力")
            
            reasons.append("; ".join(reason_list) if reason_list else "综合因子评分较高")
        
        result["prediction_reason"] = reasons
        return result
    
    def _apply_risk_filter(self, result: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
        """应用风控过滤"""
        risk_config = RISK_CONFIG
        filtered_out = []
        
        for idx in result.index:
            sector_name = result.loc[idx, "sector_name"]
            
            # 高位避险过滤
            if f"price_momentum_{risk_config['high_position_days']}" in df.columns:
                momentum = df.loc[idx, f"price_momentum_{risk_config['high_position_days']}"]
                if momentum > risk_config["high_position_threshold"] * 100:
                    # 检查今日是否巨阴线
                    if "change_pct" in df.columns:
                        change = df.loc[idx, "change_pct"]
                        if change < risk_config["big_drop_threshold"] * 100:
                            filtered_out.append(idx)
                            logger.warning(f"风控过滤: {sector_name} (高位巨阴线)")
        
        # 移除被过滤的板块
        result = result.drop(filtered_out)
        
        return result
    
    def _save_feature_importance(self, feature_names: List[str]):
        """保存特征重要性"""
        importance = self.model.feature_importance(importance_type="gain")
        
        df_importance = pd.DataFrame({
            "feature": feature_names,
            "importance": importance
        }).sort_values("importance", ascending=False)
        
        df_importance.to_csv(self.feature_importance_path, index=False)
        logger.info(f"特征重要性已保存到 {self.feature_importance_path}")
        
        # 打印Top-10特征
        logger.info("Top-10 重要特征:")
        for _, row in df_importance.head(10).iterrows():
            logger.info(f"  {row['feature']}: {row['importance']:.2f}")
    
    def save_model(self):
        """保存模型"""
        if self.model is not None:
            with open(self.model_path, "wb") as f:
                pickle.dump(self.model, f)
            logger.info(f"模型已保存到 {self.model_path}")
    
    def load_model(self):
        """加载模型"""
        if self.model_path.exists():
            with open(self.model_path, "rb") as f:
                self.model = pickle.load(f)
            logger.info(f"模型已从 {self.model_path} 加载")
        else:
            logger.warning(f"模型文件不存在: {self.model_path}")


class RollingTrainer:
    """滚动训练器"""
    
    def __init__(self, model: SectorPredictModel):
        self.model = model
        self.config = MODEL_CONFIG
    
    def should_retrain(self, last_train_date: str = None) -> bool:
        """判断是否需要重新训练"""
        if last_train_date is None:
            return True
        
        last_date = datetime.strptime(last_train_date, "%Y-%m-%d")
        days_since_train = (datetime.now() - last_date).days
        
        return days_since_train >= self.config.get("rolling_step_days", 30)
    
    def rolling_train(self, df: pd.DataFrame, 
                      train_window_months: int = 24) -> Dict:
        """
        滚动训练
        
        Args:
            df: 完整历史数据
            train_window_months: 训练窗口（月）
            
        Returns:
            训练结果
        """
        if df.empty or "date" not in df.columns:
            logger.warning("数据为空或缺少date列，跳过训练")
            return {"status": "skipped", "reason": "数据不足"}
        
        # 计算训练窗口
        end_date = df["date"].max()
        start_date = (
            datetime.strptime(end_date, "%Y-%m-%d") - 
            timedelta(days=train_window_months * 30)
        ).strftime("%Y-%m-%d")
        
        logger.info(f"滚动训练: {start_date} -> {end_date}")
        
        return self.model.train(df, train_start=start_date, train_end=end_date)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # 测试模型训练
    model = SectorPredictModel()
    
    print("模型模块测试完成")
