"""
A股ETF预测系统 - ETF预测模型模块
使用深度神经网络(Deep Neural Network)进行ETF涨跌预测
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Tuple
import logging
import pickle
import json
from pathlib import Path
import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, ndcg_score

from config.settings import MODEL_CONFIG, MODEL_DIR
from data.etf_data_processor import ETFFeatureEngineer

logger = logging.getLogger(__name__)

# 设置随机种子以保证可复现性
def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

set_seed(42)

class ETFNet(nn.Module):
    """ETF预测深度神经网络模型"""
    def __init__(self, input_dim, hidden_dim1=64, hidden_dim2=32, dropout_rate=0.2):
        super(ETFNet, self).__init__()
        self.layer1 = nn.Linear(input_dim, hidden_dim1)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout_rate)
        
        self.layer2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout_rate)
        
        self.output = nn.Linear(hidden_dim2, 1)
        
    def forward(self, x):
        x = self.layer1(x)
        x = self.relu1(x)
        x = self.dropout1(x)
        
        x = self.layer2(x)
        x = self.relu2(x)
        x = self.dropout2(x)
        
        x = self.output(x)
        return x

class ETFPredictModel:
    """ETF预测模型 (基于深度神经网络)"""
    
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.feature_engineer = ETFFeatureEngineer()
        self.feature_columns = self.feature_engineer.get_feature_columns()
        self.model_config = MODEL_CONFIG
        self.model_path = MODEL_DIR / "etf_predict_model.pth"
        self.scaler_path = MODEL_DIR / "etf_scaler.pkl"
        self.feature_importance_path = MODEL_DIR / "etf_feature_importance.csv"
        
        # 检查是否有GPU
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"使用计算设备: {self.device}")
    
    def train(self, df: pd.DataFrame, 
              train_start: str = None,
              train_end: str = None,
              valid_ratio: float = 0.2,
              epochs: int = 100,
              batch_size: int = 32,
              learning_rate: float = 0.001) -> Dict:
        """
        训练ETF预测模型
        """
        logger.info("开始ETF深度神经网络模型训练...")
        
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
        X = df[available_features].fillna(0).values
        y = df["label_score"].values
        
        logger.info(f"特征数: {len(available_features)}")
        
        # 划分训练集和验证集
        split_idx = int(len(X) * (1 - valid_ratio))
        X_train_raw, X_valid_raw = X[:split_idx], X[split_idx:]
        y_train, y_valid = y[:split_idx], y[split_idx:]
        
        # 数据标准化
        X_train = self.scaler.fit_transform(X_train_raw)
        X_valid = self.scaler.transform(X_valid_raw)
        
        # 转换为Tensor
        X_train_tensor = torch.FloatTensor(X_train).to(self.device)
        y_train_tensor = torch.FloatTensor(y_train).view(-1, 1).to(self.device)
        X_valid_tensor = torch.FloatTensor(X_valid).to(self.device)
        y_valid_tensor = torch.FloatTensor(y_valid).view(-1, 1).to(self.device)
        
        # 创建DataLoader
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        # 初始化模型
        input_dim = len(available_features)
        self.model = ETFNet(input_dim).to(self.device)
        
        # 定义损失函数和优化器
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        
        logger.info(f"ETF训练集: {len(X_train)} 样本, 验证集: {len(X_valid)} 样本")
        
        # 训练循环
        best_valid_loss = float('inf')
        patience = 10
        patience_counter = 0
        best_model_state = None
        
        for epoch in range(epochs):
            self.model.train()
            train_loss = 0.0
            
            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                train_loss += loss.item() * batch_X.size(0)
            
            train_loss /= len(train_loader.dataset)
            
            # 验证
            self.model.eval()
            with torch.no_grad():
                valid_outputs = self.model(X_valid_tensor)
                valid_loss = criterion(valid_outputs, y_valid_tensor).item()
            
            if (epoch + 1) % 10 == 0:
                logger.info(f"Epoch [{epoch+1}/{epochs}], Train Loss: {train_loss:.4f}, Valid Loss: {valid_loss:.4f}")
            
            # 早停机制
            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                best_model_state = self.model.state_dict()
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    logger.info(f"Early stopping at epoch {epoch+1}")
                    break
        
        # 加载最佳模型
        if best_model_state:
            self.model.load_state_dict(best_model_state)
        
        # 计算最终验证集指标
        self.model.eval()
        with torch.no_grad():
            y_pred_tensor = self.model(X_valid_tensor)
            y_pred = y_pred_tensor.cpu().numpy().flatten()
            
        mse = mean_squared_error(y_valid, y_pred)
        
        # 计算NDCG (复用原有逻辑)
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
        
        # 保存模型和Scaler
        self.save_model()
        
        # 特征重要性 (深度学习模型不如树模型直观，这里使用简单的梯度或权重分析替代，或者暂时略过)
        # 简单的权重绝对值平均作为重要性近似
        try:
            importances = np.abs(self.model.layer1.weight.detach().cpu().numpy()).mean(axis=0)
            self._save_feature_importance(available_features, importances)
        except:
            pass
        
        result = {
            "status": "success",
            "train_samples": len(X_train),
            "valid_samples": len(X_valid),
            "mse": mse,
            "ndcg@5": avg_ndcg,
            "best_valid_loss": best_valid_loss,
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
        # 确保特征维度匹配 (如果在训练时只有部分特征，这里可能会有问题，需假设特征一致)
        # 为了健壮性，我们可以检查输入特征数量和Scaler的特征数量
        if hasattr(self.scaler, 'n_features_in_'):
             if len(available_features) != self.scaler.n_features_in_:
                 logger.warning(f"预测特征数量({len(available_features)})与训练时({self.scaler.n_features_in_})不一致，可能导致错误")

        X = df[available_features].fillna(0).values
        
        # 预测
        self.model.eval()
        with torch.no_grad():
            X_scaled = self.scaler.transform(X)
            X_tensor = torch.FloatTensor(X_scaled).to(self.device)
            scores_tensor = self.model(X_tensor)
            scores = scores_tensor.cpu().numpy().flatten()
        
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
            # 保存PyTorch模型
            torch.save(self.model.state_dict(), self.model_path)
            # 保存Scaler
            with open(self.scaler_path, "wb") as f:
                pickle.dump(self.scaler, f)
            logger.info(f"ETF模型已保存: {self.model_path}")
    
    def load_model(self) -> bool:
        """加载模型"""
        if self.model_path.exists() and self.scaler_path.exists():
            try:
                # 加载Scaler
                with open(self.scaler_path, "rb") as f:
                    self.scaler = pickle.load(f)
                
                # 初始化并加载模型
                # 注意：这里需要知道输入维度，暂时假设在predict时可以推断或之前已初始化
                # 为了解决这个问题，我们可以在加载时先不初始化model，在predict时根据scaler的feature数初始化
                n_features = self.scaler.n_features_in_
                self.model = ETFNet(n_features).to(self.device)
                self.model.load_state_dict(torch.load(self.model_path, map_location=self.device))
                self.model.eval()
                
                logger.info(f"ETF模型已加载: {self.model_path}")
                return True
            except Exception as e:
                logger.error(f"模型加载失败: {e}")
                return False
        
        logger.warning("ETF模型文件不存在")
        return False
    
    def _save_feature_importance(self, feature_names: List[str], importances: np.ndarray = None):
        """保存特征重要性"""
        if importances is None:
            return
            
        df_importance = pd.DataFrame({
            "feature": feature_names,
            "importance": importances
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
