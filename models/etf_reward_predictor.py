"""
A股ETF预测系统 - 带奖惩机制的深度神经网络模型
使用强化学习思想：基于实际收益进行奖励/惩罚训练
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Tuple
import logging
import pickle
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, ndcg_score

from config.settings import MODEL_CONFIG, MODEL_DIR

logger = logging.getLogger(__name__)

# 设置随机种子
def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(42)


class RewardETFNet(nn.Module):
    """带奖惩机制的ETF预测深度神经网络"""
    
    def __init__(self, input_dim, hidden_dims=[128, 64, 32], dropout_rate=0.3):
        super(RewardETFNet, self).__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            prev_dim = hidden_dim
        
        self.feature_layers = nn.Sequential(*layers)
        self.output = nn.Linear(prev_dim, 1)
        
        # 初始化权重
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x):
        features = self.feature_layers(x)
        output = self.output(features)
        return output


class RewardLoss(nn.Module):
    """
    自定义奖惩损失函数 - 轻惩罚重奖励版本
    
    核心洞察：
    1. 预测 Top 3，实际下跌 → 普通惩罚
    2. 预测 Top 3，实际大涨 → 加重奖励（这是我们追求的目标）
    
    关键参数：
    - top_k_strict=3: 严格的 Top 3 判断
    - big_gain_threshold=2.0: 大涨阈值 (2%)
    - big_gain_reward_multiplier=2.0: 大涨奖励倍数
    """
    
    def __init__(self, reward_weight=1.0, penalty_weight=0.5, top_k=5, 
                 top_k_strict=3, big_gain_threshold=2.0,
                 big_gain_reward_multiplier=2.0):
        super(RewardLoss, self).__init__()
        self.mse = nn.MSELoss(reduction='none')
        self.reward_weight = reward_weight  # 基础奖励权重
        self.penalty_weight = penalty_weight  # 基础惩罚权重
        self.top_k = top_k
        self.top_k_strict = top_k_strict  # 严格的 Top K (用于加重奖励判断)
        self.big_gain_threshold = big_gain_threshold  # 大涨阈值 (%)
        self.big_gain_reward_multiplier = big_gain_reward_multiplier  # 大涨奖励倍数
    
    def forward(self, pred_scores, true_scores, actual_returns=None):
        """
        计算奖惩损失
        
        Args:
            pred_scores: 预测分数 (batch_size, 1)
            true_scores: 真实排名分数 (batch_size, 1)
            actual_returns: 实际收益率 (batch_size, 1)，用于奖惩
        """
        # 基础MSE损失
        base_loss = self.mse(pred_scores, true_scores)
        
        if actual_returns is None:
            return base_loss.mean()
        
        # 计算预测排名
        pred_ranks = self._get_ranks(pred_scores)
        
        # 奖惩调整
        batch_size = pred_scores.size(0)
        adjustments = torch.zeros_like(base_loss)
        
        for i in range(batch_size):
            pred_rank = pred_ranks[i].item()
            actual_ret = actual_returns[i].item()
            
            # === 核心逻辑：轻惩罚、重奖励 ===
            
            if pred_rank <= self.top_k_strict:
                # 预测为 Top 3 (严格判断)
                if actual_ret < 0:
                    # 预测 Top 3 但实际下跌 → 普通惩罚
                    penalty = self.penalty_weight * abs(actual_ret) / 100
                    adjustments[i] = penalty
                elif actual_ret >= self.big_gain_threshold:
                    # 🟢 完美预测：预测 Top 3 且大涨 → 加重奖励
                    reward = self.reward_weight * self.big_gain_reward_multiplier * actual_ret / 100
                    adjustments[i] = -reward
                elif actual_ret > 0:
                    # 预测 Top 3 且小涨 → 普通奖励
                    reward = self.reward_weight * actual_ret / 100
                    adjustments[i] = -reward
                    
            elif pred_rank <= self.top_k:
                # 预测为 Top 4-5
                if actual_ret < 0:
                    # 预测靠前但下跌 → 轻惩罚
                    penalty = self.penalty_weight * 0.5 * abs(actual_ret) / 100
                    adjustments[i] = penalty
                elif actual_ret >= self.big_gain_threshold:
                    # 预测靠前且大涨 → 奖励
                    reward = self.reward_weight * actual_ret / 100
                    adjustments[i] = -reward
                elif actual_ret > 0:
                    # 小涨 → 小奖励
                    reward = self.reward_weight * 0.5 * actual_ret / 100
                    adjustments[i] = -reward
            else:
                # 预测为非 Top-K
                if actual_ret >= self.big_gain_threshold:
                    # 错过大涨 → 轻惩罚
                    penalty = self.penalty_weight * 0.3 * abs(actual_ret) / 100
                    adjustments[i] = penalty
                # 预测非 Top-K 且确实没涨 → 正确，无调整
        
        # 组合损失
        total_loss = base_loss + adjustments
        return total_loss.mean()
    
    def _get_ranks(self, scores):
        """获取排名（1为最高）"""
        sorted_indices = torch.argsort(scores.squeeze(), descending=True)
        ranks = torch.zeros_like(sorted_indices)
        ranks[sorted_indices] = torch.arange(1, len(scores) + 1, device=scores.device)
        return ranks


class ETFRewardModel:
    """带奖惩机制的ETF预测模型"""
    
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 模型保存路径
        self.model_path = MODEL_DIR / "etf_reward_model.pth"
        self.scaler_path = MODEL_DIR / "etf_reward_scaler.pkl"
        self.history_path = MODEL_DIR / "etf_reward_training_history.json"
        
        # 特征列
        self.feature_columns = self._get_feature_columns()
        
        logger.info(f"奖惩模型使用设备: {self.device}")
    
    def _get_feature_columns(self) -> List[str]:
        """获取特征列名"""
        features = []
        
        # 动量特征
        for window in [3, 5, 10, 20]:
            features.extend([f"return_{window}d", f"volatility_{window}d"])
        
        # 成交量特征
        features.extend(["volume_ratio", "turnover_ratio"])
        
        # 价格位置
        features.append("price_position")
        
        # 均线偏离
        features.extend(["ma5_bias", "ma10_bias", "ma20_bias"])
        
        # 资金流
        features.extend(["money_flow_ma3", "money_flow_ma5", "money_flow_momentum"])
        
        # 技术指标
        features.extend(["rsi_14", "atr_14"])
        
        return features
    
    def train_with_reward(self, df: pd.DataFrame,
                          train_start: str = None,
                          train_end: str = None,
                          epochs: int = 150,
                          batch_size: int = 64,
                          learning_rate: float = 0.001,
                          reward_weight: float = 1.0,
                          penalty_weight: float = 0.5,
                          top_k_strict: int = 3,
                          big_gain_threshold: float = 2.0,
                          big_gain_reward_multiplier: float = 2.0) -> Dict:
        """
        使用奖惩机制训练模型 - 轻惩罚重奖励版本
        
        核心策略：
        - 预测 Top 3 且大涨 → 加重奖励（重点鼓励）
        - 预测 Top 3 但下跌 → 普通惩罚
        
        Args:
            df: 包含特征和标签的数据
            train_start/train_end: 训练数据范围
            epochs: 训练轮数
            batch_size: 批大小
            learning_rate: 学习率
            reward_weight: 基础奖励权重 (默认1.0)
            penalty_weight: 基础惩罚权重 (默认0.5)
            top_k_strict: 严格 Top-K 判断阈值 (默认3)
            big_gain_threshold: 大涨阈值百分比 (默认2.0%)
            big_gain_reward_multiplier: 大涨加重奖励倍数 (默认2.0)
        """
        logger.info("=" * 60)
        logger.info("🎯 开始【轻惩罚重奖励】深度学习训练...")
        logger.info(f"   基础奖励权重: {reward_weight}, 基础惩罚权重: {penalty_weight}")
        logger.info(f"   严格 Top-K: {top_k_strict}, 大涨阈值: {big_gain_threshold}%")
        logger.info(f"   大涨加重奖励倍数: {big_gain_reward_multiplier}x")
        
        # 数据过滤
        if train_start:
            df = df[df["date"] >= train_start]
        if train_end:
            df = df[df["date"] <= train_end]
        
        # 移除无效数据
        df = df.dropna(subset=["label_score"])
        
        if len(df) < 100:
            logger.error(f"训练数据不足: {len(df)} 行")
            return {"status": "error", "message": "训练数据不足"}
        
        logger.info(f"训练数据范围: {df['date'].min()} ~ {df['date'].max()}")
        logger.info(f"训练样本数: {len(df)}")
        
        # 准备特征
        available_features = [c for c in self.feature_columns if c in df.columns]
        X = df[available_features].fillna(0).values
        y = df["label_score"].values
        
        # 获取实际收益（用于奖惩）
        actual_returns = df["label_next_return"].fillna(0).values if "label_next_return" in df.columns else None
        
        # 划分训练集和验证集 (时间序列划分)
        split_idx = int(len(X) * 0.8)
        X_train_raw, X_valid_raw = X[:split_idx], X[split_idx:]
        y_train, y_valid = y[:split_idx], y[split_idx:]
        
        if actual_returns is not None:
            returns_train, returns_valid = actual_returns[:split_idx], actual_returns[split_idx:]
        else:
            returns_train, returns_valid = None, None
        
        # 标准化
        X_train = self.scaler.fit_transform(X_train_raw)
        X_valid = self.scaler.transform(X_valid_raw)
        
        # 转为Tensor
        X_train_t = torch.FloatTensor(X_train).to(self.device)
        y_train_t = torch.FloatTensor(y_train).view(-1, 1).to(self.device)
        X_valid_t = torch.FloatTensor(X_valid).to(self.device)
        y_valid_t = torch.FloatTensor(y_valid).view(-1, 1).to(self.device)
        
        if returns_train is not None:
            returns_train_t = torch.FloatTensor(returns_train).view(-1, 1).to(self.device)
            returns_valid_t = torch.FloatTensor(returns_valid).view(-1, 1).to(self.device)
        else:
            returns_train_t, returns_valid_t = None, None
        
        # 创建DataLoader
        if returns_train_t is not None:
            train_dataset = TensorDataset(X_train_t, y_train_t, returns_train_t)
        else:
            train_dataset = TensorDataset(X_train_t, y_train_t)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        # 初始化模型
        input_dim = len(available_features)
        self.model = RewardETFNet(input_dim, hidden_dims=[128, 64, 32]).to(self.device)
        
        # 奖惩损失函数 - 轻惩罚重奖励
        criterion = RewardLoss(
            reward_weight=reward_weight, 
            penalty_weight=penalty_weight,
            top_k_strict=top_k_strict,
            big_gain_threshold=big_gain_threshold,
            big_gain_reward_multiplier=big_gain_reward_multiplier
        )
        optimizer = optim.AdamW(self.model.parameters(), lr=learning_rate, weight_decay=0.01)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
        
        logger.info(f"模型参数量: {sum(p.numel() for p in self.model.parameters())}")
        logger.info(f"训练集: {len(X_train)} 样本, 验证集: {len(X_valid)} 样本")
        
        # 训练循环
        best_valid_loss = float('inf')
        best_model_state = None
        patience = 15
        patience_counter = 0
        history = {"train_loss": [], "valid_loss": [], "ndcg": []}
        
        for epoch in range(epochs):
            # 训练阶段
            self.model.train()
            train_loss = 0.0
            
            for batch in train_loader:
                if len(batch) == 3:
                    batch_X, batch_y, batch_returns = batch
                else:
                    batch_X, batch_y = batch
                    batch_returns = None
                
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y, batch_returns)
                loss.backward()
                
                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                optimizer.step()
                train_loss += loss.item() * batch_X.size(0)
            
            train_loss /= len(train_loader.dataset)
            
            # 验证阶段
            self.model.eval()
            with torch.no_grad():
                valid_outputs = self.model(X_valid_t)
                valid_loss = criterion(valid_outputs, y_valid_t, returns_valid_t).item()
            
            # 学习率调整
            scheduler.step(valid_loss)
            
            # 记录历史
            history["train_loss"].append(train_loss)
            history["valid_loss"].append(valid_loss)
            
            if (epoch + 1) % 10 == 0:
                logger.info(f"Epoch [{epoch+1}/{epochs}] | Train Loss: {train_loss:.4f} | Valid Loss: {valid_loss:.4f}")
            
            # 早停
            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                best_model_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    logger.info(f"⏹️ Early stopping at epoch {epoch+1}")
                    break
        
        # 加载最佳模型
        if best_model_state:
            self.model.load_state_dict(best_model_state)
            self.model.to(self.device)
        
        # 计算最终指标
        self.model.eval()
        with torch.no_grad():
            y_pred = self.model(X_valid_t).cpu().numpy().flatten()
        
        mse = mean_squared_error(y_valid, y_pred)
        
        # 计算NDCG
        df_valid = df.iloc[split_idx:].copy()
        df_valid["pred_score"] = y_pred
        
        ndcg_scores = []
        for date in df_valid["date"].unique():
            date_df = df_valid[df_valid["date"] == date]
            if len(date_df) < 2:
                continue
            try:
                ndcg = ndcg_score(
                    date_df["label_score"].values.reshape(1, -1),
                    date_df["pred_score"].values.reshape(1, -1),
                    k=5
                )
                ndcg_scores.append(ndcg)
            except:
                pass
        
        avg_ndcg = np.mean(ndcg_scores) if ndcg_scores else 0
        
        # 保存模型
        self.save_model()
        
        result = {
            "status": "success",
            "train_samples": len(X_train),
            "valid_samples": len(X_valid),
            "mse": mse,
            "ndcg@5": avg_ndcg,
            "best_valid_loss": best_valid_loss,
            "epochs_trained": epoch + 1,
        }
        
        logger.info("=" * 60)
        logger.info(f"✅ 奖惩模型训练完成!")
        logger.info(f"   MSE: {mse:.4f} | NDCG@5: {avg_ndcg:.4f}")
        logger.info("=" * 60)
        
        return result
    
    def predict(self, df: pd.DataFrame, top_k: int = 5) -> pd.DataFrame:
        """预测ETF得分"""
        if self.model is None:
            self.load_model()
            if self.model is None:
                logger.error("模型未加载")
                return pd.DataFrame()
        
        available_features = [c for c in self.feature_columns if c in df.columns]
        X = df[available_features].fillna(0).values
        
        self.model.eval()
        with torch.no_grad():
            X_scaled = self.scaler.transform(X)
            X_tensor = torch.FloatTensor(X_scaled).to(self.device)
            scores = self.model(X_tensor).cpu().numpy().flatten()
        
        result = df[["date", "etf_code", "etf_name"]].copy()
        result["pred_score"] = scores
        result = result.sort_values("pred_score", ascending=False)
        result["rank"] = range(1, len(result) + 1)
        
        return result.head(top_k)
    
    def save_model(self):
        """保存模型"""
        if self.model is not None:
            MODEL_DIR.mkdir(parents=True, exist_ok=True)
            torch.save(self.model.state_dict(), self.model_path)
            with open(self.scaler_path, "wb") as f:
                pickle.dump(self.scaler, f)
            logger.info(f"奖惩模型已保存: {self.model_path}")
    
    def load_model(self) -> bool:
        """加载模型"""
        if self.model_path.exists() and self.scaler_path.exists():
            try:
                with open(self.scaler_path, "rb") as f:
                    self.scaler = pickle.load(f)
                
                n_features = self.scaler.n_features_in_
                self.model = RewardETFNet(n_features, hidden_dims=[128, 64, 32]).to(self.device)
                self.model.load_state_dict(torch.load(self.model_path, map_location=self.device))
                self.model.eval()
                
                logger.info(f"奖惩模型已加载")
                return True
            except Exception as e:
                logger.error(f"模型加载失败: {e}")
                return False
        return False


class RewardRollingBacktest:
    """
    奖惩机制滚动回测引擎
    
    策略：
    1. 使用2022-2023年数据按月滚动训练
    2. 每月重新训练，纳入新数据
    3. 2024全年作为回测期
    """
    
    def __init__(self, 
                 train_window_months: int = 12,
                 retrain_interval_months: int = 1,
                 top_k: int = 5,
                 reward_weight: float = 0.5,
                 penalty_weight: float = 0.8):
        """
        Args:
            train_window_months: 训练窗口（月）
            retrain_interval_months: 重训练间隔（月）
            top_k: 预测Top-K
            reward_weight: 奖励权重
            penalty_weight: 惩罚权重
        """
        self.train_window_months = train_window_months
        self.retrain_interval_months = retrain_interval_months
        self.top_k = top_k
        self.reward_weight = reward_weight
        self.penalty_weight = penalty_weight
        
        self.model = ETFRewardModel()
        self.results = []
    
    def run_backtest(self, df: pd.DataFrame,
                     train_start: str = "2022-01-01",
                     train_end: str = "2023-12-31",
                     test_start: str = "2024-01-01",
                     test_end: str = "2024-12-31") -> Dict:
        """
        执行滚动回测
        
        Args:
            df: 完整历史数据（含特征和标签）
            train_start/train_end: 初始训练期
            test_start/test_end: 回测期
        """
        logger.info("=" * 70)
        logger.info("🚀 开始奖惩机制滚动回测")
        logger.info(f"   训练期: {train_start} ~ {train_end}")
        logger.info(f"   回测期: {test_start} ~ {test_end}")
        logger.info(f"   训练窗口: {self.train_window_months}个月")
        logger.info(f"   重训练间隔: {self.retrain_interval_months}个月")
        logger.info("=" * 70)
        
        # 确保日期格式
        df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
        df = df.sort_values(["date", "etf_code"]).reset_index(drop=True)
        
        # 获取回测期的所有交易日
        test_df = df[(df["date"] >= test_start) & (df["date"] <= test_end)]
        test_dates = sorted(test_df["date"].unique())
        
        if len(test_dates) == 0:
            logger.error("回测期无数据!")
            return {"status": "error", "message": "回测期无数据"}
        
        logger.info(f"回测交易日数: {len(test_dates)}")
        
        # 初始化
        all_predictions = []
        monthly_returns = []
        current_train_end = train_end
        last_retrain_month = None
        
        # 初始训练
        logger.info("\n📚 初始模型训练...")
        train_data = df[(df["date"] >= train_start) & (df["date"] <= train_end)]
        self.model.train_with_reward(
            train_data,
            epochs=100,
            reward_weight=self.reward_weight,
            penalty_weight=self.penalty_weight
        )
        
        # 逐日回测
        for i, test_date in enumerate(test_dates):
            test_month = test_date[:7]  # YYYY-MM
            
            # 检查是否需要重新训练（每月初）
            if last_retrain_month is None or test_month != last_retrain_month:
                if last_retrain_month is not None:
                    # 更新训练窗口
                    new_train_end = (datetime.strptime(test_date, "%Y-%m-%d") - timedelta(days=1)).strftime("%Y-%m-%d")
                    new_train_start = (datetime.strptime(new_train_end, "%Y-%m-%d") - 
                                      timedelta(days=self.train_window_months * 30)).strftime("%Y-%m-%d")
                    
                    logger.info(f"\n🔄 [{test_month}] 重新训练模型...")
                    logger.info(f"   新训练期: {new_train_start} ~ {new_train_end}")
                    
                    retrain_data = df[(df["date"] >= new_train_start) & (df["date"] <= new_train_end)]
                    if len(retrain_data) > 100:
                        self.model.train_with_reward(
                            retrain_data,
                            epochs=80,
                            reward_weight=self.reward_weight,
                            penalty_weight=self.penalty_weight
                        )
                
                last_retrain_month = test_month
            
            # 获取当日数据进行预测
            day_data = df[df["date"] == test_date].copy()
            
            if day_data.empty:
                continue
            
            # 预测
            predictions = self.model.predict(day_data, top_k=self.top_k)
            
            if predictions.empty:
                continue
            
            # 获取实际收益
            for _, row in predictions.iterrows():
                etf_code = row["etf_code"]
                pred_rank = row["rank"]
                pred_score = row["pred_score"]
                
                # 查找实际次日收益
                actual_return = day_data[day_data["etf_code"] == etf_code]["label_next_return"].values
                actual_return = actual_return[0] if len(actual_return) > 0 else 0
                
                all_predictions.append({
                    "date": test_date,
                    "etf_code": etf_code,
                    "etf_name": row["etf_name"],
                    "pred_rank": pred_rank,
                    "pred_score": pred_score,
                    "actual_return": actual_return,
                    "is_positive": actual_return > 0,
                })
            
            if (i + 1) % 20 == 0:
                logger.info(f"   已完成 {i+1}/{len(test_dates)} 个交易日")
        
        # 生成回测报告
        results_df = pd.DataFrame(all_predictions)
        report = self._generate_report(results_df)
        
        # 保存结果
        self._save_results(results_df, report)
        
        return report
    
    def _generate_report(self, results_df: pd.DataFrame) -> Dict:
        """生成回测报告"""
        if results_df.empty:
            return {"status": "error", "message": "无预测结果"}
        
        # 初始本金
        INITIAL_CAPITAL = 10000.0
        
        # 基础统计
        total_predictions = len(results_df)
        total_positive = results_df["is_positive"].sum()
        hit_rate = total_positive / total_predictions if total_predictions > 0 else 0
        
        # 按排名统计
        top1_results = results_df[results_df["pred_rank"] == 1]
        top3_results = results_df[results_df["pred_rank"] <= 3]
        
        top1_hit_rate = top1_results["is_positive"].mean() if len(top1_results) > 0 else 0
        top3_hit_rate = top3_results["is_positive"].mean() if len(top3_results) > 0 else 0
        
        # 收益统计
        avg_return = results_df["actual_return"].mean()
        
        # Top-1策略收益（每天买入Top-1，用本金复利计算）
        top1_daily_returns = top1_results.groupby("date")["actual_return"].mean()
        top1_capital = INITIAL_CAPITAL
        for ret in top1_daily_returns:
            top1_capital *= (1 + ret / 100)
        top1_final_capital = top1_capital
        top1_total_return = ((top1_final_capital - INITIAL_CAPITAL) / INITIAL_CAPITAL) * 100
        top1_sharpe = (top1_daily_returns.mean() / top1_daily_returns.std() * np.sqrt(252)) if top1_daily_returns.std() > 0 else 0
        
        # Top-5等权策略（用本金复利计算）
        top5_daily_returns = results_df.groupby("date")["actual_return"].mean()
        top5_capital = INITIAL_CAPITAL
        for ret in top5_daily_returns:
            top5_capital *= (1 + ret / 100)
        top5_final_capital = top5_capital
        top5_total_return = ((top5_final_capital - INITIAL_CAPITAL) / INITIAL_CAPITAL) * 100
        top5_sharpe = (top5_daily_returns.mean() / top5_daily_returns.std() * np.sqrt(252)) if top5_daily_returns.std() > 0 else 0
        
        # 最大回撤
        cumulative = (1 + top5_daily_returns / 100).cumprod()
        rolling_max = cumulative.expanding().max()
        drawdown = (cumulative - rolling_max) / rolling_max * 100
        max_drawdown = drawdown.min()
        
        # 胜率
        win_days = (top5_daily_returns > 0).sum()
        total_days = len(top5_daily_returns)
        win_rate = win_days / total_days if total_days > 0 else 0
        
        # 月度统计
        results_df["month"] = results_df["date"].str[:7]
        monthly_stats = results_df.groupby("month").agg({
            "actual_return": "mean",
            "is_positive": "mean"
        }).rename(columns={"actual_return": "avg_return", "is_positive": "hit_rate"})
        
        report = {
            "status": "success",
            "period": f"{results_df['date'].min()} ~ {results_df['date'].max()}",
            "total_days": total_days,
            "total_predictions": total_predictions,
            
            # 本金信息
            "initial_capital": INITIAL_CAPITAL,
            "top1_final_capital": top1_final_capital,
            "top5_final_capital": top5_final_capital,
            
            # 命中率
            "overall_hit_rate": hit_rate,
            "top1_hit_rate": top1_hit_rate,
            "top3_hit_rate": top3_hit_rate,
            
            # 收益（年化）
            "avg_daily_return": avg_return,
            "top1_total_return": top1_total_return,
            "top5_total_return": top5_total_return,
            
            # 风险指标
            "top1_sharpe": top1_sharpe,
            "top5_sharpe": top5_sharpe,
            "max_drawdown": max_drawdown,
            "win_rate": win_rate,
            
            # 月度数据
            "monthly_stats": monthly_stats.to_dict(),
        }
        
        return report
    
    def _save_results(self, results_df: pd.DataFrame, report: Dict):
        """保存回测结果"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 保存详细预测
        detail_path = BACKTEST_RESULTS_DIR / f"reward_backtest_detail_{timestamp}.csv"
        results_df.to_csv(detail_path, index=False, encoding="utf-8-sig")
        
        # 保存报告
        report_path = BACKTEST_RESULTS_DIR / f"reward_backtest_report_{timestamp}.md"
        
        report_content = f"""# 🎯 奖惩机制深度学习回测报告

**生成时间**: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

---

## 📊 回测概览

| 指标 | 数值 |
|------|------|
| 回测期间 | {report.get('period', 'N/A')} |
| 交易天数 | {report.get('total_days', 0)} |
| 总预测次数 | {report.get('total_predictions', 0)} |
| 初始本金 | ¥{report.get('initial_capital', 10000):.2f} |

---

## 🎯 命中率统计

| 排名 | 命中率 |
|------|--------|
| Top-1 | {report.get('top1_hit_rate', 0):.2%} |
| Top-3 | {report.get('top3_hit_rate', 0):.2%} |
| 整体 | {report.get('overall_hit_rate', 0):.2%} |

---

## 💰 收益统计（本金 ¥10,000）

| 策略 | 最终资金 | 年化收益率 | 夏普比率 |
|------|----------|------------|----------|
| Top-1策略 | ¥{report.get('top1_final_capital', 10000):.2f} | {report.get('top1_total_return', 0):.2f}% | {report.get('top1_sharpe', 0):.2f} |
| Top-5等权 | ¥{report.get('top5_final_capital', 10000):.2f} | {report.get('top5_total_return', 0):.2f}% | {report.get('top5_sharpe', 0):.2f} |

---

## 📉 风险指标

| 指标 | 数值 |
|------|------|
| 最大回撤 | {report.get('max_drawdown', 0):.2f}% |
| 胜率 | {report.get('win_rate', 0):.2%} |
| 平均日收益 | {report.get('avg_daily_return', 0):.4f}% |

---

## 📅 月度表现

"""
        
        monthly_stats = report.get('monthly_stats', {})
        if monthly_stats:
            report_content += "| 月份 | 平均收益 | 命中率 |\n|------|----------|--------|\n"
            avg_returns = monthly_stats.get('avg_return', {})
            hit_rates = monthly_stats.get('hit_rate', {})
            for month in sorted(avg_returns.keys()):
                ret = avg_returns.get(month, 0)
                hit = hit_rates.get(month, 0)
                report_content += f"| {month} | {ret:.2f}% | {hit:.2%} |\n"
        
        report_content += """
---

## ⚠️ 免责声明

本回测结果仅供研究参考，不构成投资建议。历史表现不代表未来收益。

*报告由奖惩机制深度学习系统自动生成*
"""
        
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(report_content)
        
        logger.info(f"📄 回测详情已保存: {detail_path}")
        logger.info(f"📄 回测报告已保存: {report_path}")
    
    def print_report(self, report: Dict):
        """打印回测报告"""
        print("\n" + "=" * 70)
        print("🎯 奖惩机制深度学习回测报告")
        print("=" * 70)
        
        print(f"\n📊 回测期间: {report.get('period', 'N/A')}")
        print(f"   交易天数: {report.get('total_days', 0)}")
        print(f"   总预测数: {report.get('total_predictions', 0)}")
        print(f"   初始本金: ¥{report.get('initial_capital', 10000):.2f}")
        
        print(f"\n🎯 命中率:")
        print(f"   Top-1 命中率: {report.get('top1_hit_rate', 0):.2%}")
        print(f"   Top-3 命中率: {report.get('top3_hit_rate', 0):.2%}")
        print(f"   整体命中率:   {report.get('overall_hit_rate', 0):.2%}")
        
        print(f"\n💰 收益统计（本金 ¥10,000）:")
        print(f"   Top-1策略最终资金: ¥{report.get('top1_final_capital', 10000):.2f}")
        print(f"   Top-1策略年化收益: {report.get('top1_total_return', 0):.2f}%")
        print(f"   Top-5等权最终资金: ¥{report.get('top5_final_capital', 10000):.2f}")
        print(f"   Top-5等权年化收益: {report.get('top5_total_return', 0):.2f}%")
        print(f"   平均日收益: {report.get('avg_daily_return', 0):.4f}%")
        
        print(f"\n📈 风险指标:")
        print(f"   Top-1 夏普比率: {report.get('top1_sharpe', 0):.2f}")
        print(f"   Top-5 夏普比率: {report.get('top5_sharpe', 0):.2f}")
        print(f"   最大回撤: {report.get('max_drawdown', 0):.2f}%")
        print(f"   胜率: {report.get('win_rate', 0):.2%}")
        
        print("\n" + "=" * 70)


# 导入常量
BACKTEST_RESULTS_DIR = Path(__file__).parent.parent / "data" / "backtest_results"
BACKTEST_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
