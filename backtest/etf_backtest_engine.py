"""
ETF预测系统 - ETF滚动回测引擎
真正的历史回测验证：用过去数据训练，预测未来数据
支持3年历史数据训练
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Tuple
import logging
from pathlib import Path
import pickle
import json
import warnings
import time

import akshare as ak
import lightgbm as lgb
from sklearn.metrics import ndcg_score, mean_squared_error

warnings.filterwarnings('ignore')

PROJECT_ROOT = Path(__file__).parent.parent
BACKTEST_RESULTS_DIR = PROJECT_ROOT / "data" / "backtest_results"
BACKTEST_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

logger = logging.getLogger(__name__)


# 完整ETF池
ETF_POOL = {
    # 宽基ETF
    "510300": "沪深300ETF",
    "510500": "中证500ETF",
    "159915": "创业板ETF",
    "510050": "上证50ETF",
    "588000": "科创50ETF",
    "159901": "深证100ETF",
    "512100": "中证1000ETF",
    # 行业ETF
    "512480": "半导体ETF",
    "515030": "新能源车ETF",
    "512690": "白酒ETF",
    "512660": "军工ETF",
    "512010": "医药ETF",
    "515790": "光伏ETF",
    "512800": "银行ETF",
    "512880": "证券ETF",
    "515050": "5GETF",
    "512400": "有色金属ETF",
    "159825": "农业ETF",
    "512200": "房地产ETF",
    "516150": "稀土ETF",
    "515220": "煤炭ETF",
    "159766": "旅游ETF",
    "512580": "环保ETF",
    "512760": "芯片ETF",
    "512720": "计算机ETF",
    "159928": "消费ETF",
    "512170": "医疗ETF",
    "159869": "游戏ETF",
    "515000": "科技ETF",
    "512980": "传媒ETF",
    "159740": "人工智能ETF",
    "512670": "国防ETF",
    "516510": "云计算ETF",
    "159607": "储能ETF",
    "159611": "电力ETF",
}


class ETFHistoricalDataFetcher:
    """ETF历史数据获取器 - 支持3年数据"""
    
    def __init__(self):
        self.etf_pool = ETF_POOL
        self.retry_times = 3
        self.retry_delay = 2
    
    def _retry_fetch(self, func, *args, **kwargs):
        """带重试的数据获取"""
        for i in range(self.retry_times):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                logger.warning(f"数据获取失败 (尝试 {i+1}/{self.retry_times}): {e}")
                if i < self.retry_times - 1:
                    time.sleep(self.retry_delay)
        return None
    
    def fetch_etf_history(self, etf_code: str, start_date: str = None, 
                          end_date: str = None) -> Optional[pd.DataFrame]:
        """
        获取单只ETF完整历史数据
        
        Args:
            etf_code: ETF代码
            start_date: 开始日期 (YYYY-MM-DD)
            end_date: 结束日期 (YYYY-MM-DD)
        """
        try:
            df = self._retry_fetch(
                ak.fund_etf_hist_em,
                symbol=etf_code,
                period="daily",
                adjust="qfq"
            )
            
            if df is not None and not df.empty:
                df = df.rename(columns={
                    "日期": "date",
                    "开盘": "open",
                    "收盘": "close",
                    "最高": "high",
                    "最低": "low",
                    "成交量": "volume",
                    "成交额": "turnover",
                    "振幅": "amplitude",
                    "涨跌幅": "change_pct",
                    "涨跌额": "change_amount",
                    "换手率": "turnover_rate",
                })
                
                df["etf_code"] = etf_code
                df["etf_name"] = self.etf_pool.get(etf_code, etf_code)
                df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
                
                # 日期过滤
                if start_date:
                    df = df[df["date"] >= start_date]
                if end_date:
                    df = df[df["date"] <= end_date]
                
                return df
        except Exception as e:
            logger.warning(f"获取ETF {etf_code} 历史数据失败: {e}")
        
        return None
    
    def fetch_all_etf_history(self, years: int = 3) -> pd.DataFrame:
        """
        获取所有ETF池3年历史数据
        
        Args:
            years: 获取年数，默认3年
        """
        end_date = datetime.now().strftime("%Y-%m-%d")
        start_date = (datetime.now() - timedelta(days=years * 365)).strftime("%Y-%m-%d")
        
        logger.info(f"获取ETF历史数据: {start_date} ~ {end_date} ({years}年)")
        
        all_data = []
        total = len(self.etf_pool)
        
        for i, (etf_code, etf_name) in enumerate(self.etf_pool.items(), 1):
            logger.info(f"[{i}/{total}] 获取 {etf_name} ({etf_code})...")
            
            df = self.fetch_etf_history(etf_code, start_date, end_date)
            if df is not None and not df.empty:
                all_data.append(df)
                logger.info(f"   获取 {len(df)} 条记录")
            
            time.sleep(0.5)  # 避免请求过快
        
        if all_data:
            result = pd.concat(all_data, ignore_index=True)
            logger.info(f"ETF历史数据获取完成: {len(result)} 条, {len(self.etf_pool)} 只ETF")
            return result
        
        return pd.DataFrame()
    
    def save_history_data(self, df: pd.DataFrame, filename: str = "etf_history_3y.csv"):
        """保存历史数据"""
        save_path = PROJECT_ROOT / "data" / "historical" / filename
        save_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(save_path, index=False, encoding="utf-8-sig")
        logger.info(f"ETF历史数据已保存: {save_path}")
        return str(save_path)
    
    def load_history_data(self, filename: str = "etf_history_3y.csv") -> pd.DataFrame:
        """加载历史数据"""
        load_path = PROJECT_ROOT / "data" / "historical" / filename
        if load_path.exists():
            df = pd.read_csv(load_path)
            logger.info(f"ETF历史数据已加载: {len(df)} 条")
            return df
        return pd.DataFrame()


class ETFFeatureEngineer:
    """ETF特征工程 - 增强版"""
    
    def __init__(self):
        self.feature_windows = [3, 5, 10, 20, 60]
        self.feature_columns = self.get_feature_columns()
    
    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        创建ETF特征数据集
        """
        if df.empty or len(df) < 100:
            logger.warning("ETF数据不足，无法创建特征")
            return pd.DataFrame()
        
        # 确保日期格式
        df = df.copy()
        df['date'] = pd.to_datetime(df['date']).dt.strftime('%Y-%m-%d')
        df = df.sort_values(['etf_code', 'date']).reset_index(drop=True)
        
        all_features = []
        
        for etf_code in df['etf_code'].unique():
            etf_df = df[df['etf_code'] == etf_code].copy()
            
            if len(etf_df) < 60:  # 需要足够历史数据
                continue
            
            features = self._compute_single_etf_features(etf_df)
            all_features.append(features)
        
        if not all_features:
            return pd.DataFrame()
        
        result = pd.concat(all_features, ignore_index=True)
        
        # 创建标签（T+1日涨跌幅）
        result = self._create_labels(result)
        
        # 删除缺失值
        result = result.dropna(subset=['label_score'])
        
        logger.info(f"ETF特征工程完成: {len(result)} 条数据, {result['etf_code'].nunique()} 只ETF")
        return result
    
    def _compute_single_etf_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算单只ETF的特征"""
        result = df[['date', 'etf_code', 'etf_name', 'open', 'high', 'low', 
                     'close', 'volume', 'turnover', 'change_pct', 
                     'turnover_rate', 'amplitude']].copy()
        
        # 1. 收益率特征
        for window in self.feature_windows:
            result[f'return_{window}d'] = df['close'].pct_change(window) * 100
            result[f'volatility_{window}d'] = df['close'].pct_change().rolling(window).std() * 100
        
        # 2. 均线特征
        for window in [5, 10, 20, 60]:
            result[f'ma{window}'] = df['close'].rolling(window).mean()
            result[f'ma{window}_bias'] = (df['close'] - result[f'ma{window}']) / result[f'ma{window}'] * 100
        
        # 3. 均线多头/空头排列
        result['ma_bull'] = ((result['ma5'] > result['ma10']) & 
                            (result['ma10'] > result['ma20'])).astype(int)
        result['ma_bear'] = ((result['ma5'] < result['ma10']) & 
                            (result['ma10'] < result['ma20'])).astype(int)
        
        # 4. 成交量特征
        for window in [5, 10, 20]:
            result[f'volume_ma{window}'] = df['volume'].rolling(window).mean()
        result['volume_ratio'] = df['volume'] / result['volume_ma5']
        result['volume_trend'] = (result['volume_ma5'] / result['volume_ma20']).fillna(1)
        
        # 5. 价格位置特征
        result['high_20d'] = df['high'].rolling(20).max()
        result['low_20d'] = df['low'].rolling(20).min()
        result['price_position_20d'] = (df['close'] - result['low_20d']) / (result['high_20d'] - result['low_20d'] + 1e-8)
        
        result['high_60d'] = df['high'].rolling(60).max()
        result['low_60d'] = df['low'].rolling(60).min()
        result['price_position_60d'] = (df['close'] - result['low_60d']) / (result['high_60d'] - result['low_60d'] + 1e-8)
        
        # 6. 技术指标
        result['rsi_6'] = self._compute_rsi(df['close'], 6)
        result['rsi_12'] = self._compute_rsi(df['close'], 12)
        result['rsi_24'] = self._compute_rsi(df['close'], 24)
        
        # 7. MACD
        ema12 = df['close'].ewm(span=12, adjust=False).mean()
        ema26 = df['close'].ewm(span=26, adjust=False).mean()
        result['macd'] = ema12 - ema26
        result['macd_signal'] = result['macd'].ewm(span=9, adjust=False).mean()
        result['macd_hist'] = result['macd'] - result['macd_signal']
        
        # 8. 布林带
        result['bb_mid'] = df['close'].rolling(20).mean()
        bb_std = df['close'].rolling(20).std()
        result['bb_upper'] = result['bb_mid'] + 2 * bb_std
        result['bb_lower'] = result['bb_mid'] - 2 * bb_std
        result['bb_width'] = (result['bb_upper'] - result['bb_lower']) / result['bb_mid']
        result['bb_position'] = (df['close'] - result['bb_lower']) / (result['bb_upper'] - result['bb_lower'] + 1e-8)
        
        # 9. ATR (平均真实波幅)
        result['atr_14'] = self._compute_atr(df, 14)
        result['atr_ratio'] = result['atr_14'] / df['close'] * 100
        
        # 10. 动量指标
        result['momentum_10'] = df['close'] - df['close'].shift(10)
        result['momentum_20'] = df['close'] - df['close'].shift(20)
        result['roc_10'] = df['close'].pct_change(10) * 100
        result['roc_20'] = df['close'].pct_change(20) * 100
        
        # 11. 量价背离特征
        result['volume_rank'] = df['volume'].rolling(20).apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1])
        result['price_rank'] = df['close'].pct_change().rolling(20).apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1])
        result['divergence_score'] = result['volume_rank'] - result['price_rank']
        
        # 12. 趋势强度
        result['adx'] = self._compute_adx(df, 14)
        
        # 13. 日内特征
        result['upper_shadow'] = (df['high'] - df[['open', 'close']].max(axis=1)) / df['close'] * 100
        result['lower_shadow'] = (df[['open', 'close']].min(axis=1) - df['low']) / df['close'] * 100
        result['body_size'] = abs(df['close'] - df['open']) / df['close'] * 100
        
        # 14. 连续涨跌特征
        result['up_streak'] = self._compute_streak(df['change_pct'], True)
        result['down_streak'] = self._compute_streak(df['change_pct'], False)
        
        # 15. 反转特征（超跌反弹）
        result['oversold_signal'] = ((result['rsi_6'] < 30) & (result['price_position_20d'] < 0.2)).astype(int)
        result['overbought_signal'] = ((result['rsi_6'] > 70) & (result['price_position_20d'] > 0.8)).astype(int)
        
        # 16. 动量反转
        result['momentum_reversal_5d'] = np.where(
            (result['return_5d'] < -3) & (df['change_pct'] > 0), 1,
            np.where((result['return_5d'] > 3) & (df['change_pct'] < 0), -1, 0)
        )
        
        # 17. 成交量异常
        result['volume_spike'] = (result['volume_ratio'] > 2).astype(int)
        result['volume_dry'] = (result['volume_ratio'] < 0.5).astype(int)
        
        # 18. 波动率变化
        result['volatility_change'] = result['volatility_5d'] / (result['volatility_20d'] + 1e-8)
        
        # 19. 趋势与动量组合
        result['trend_momentum'] = result['ma5_bias'] * result['momentum_10']
        
        # 20. 价格效率
        result['price_efficiency'] = abs(df['close'] - df['close'].shift(10)) / (
            (df['high'].rolling(10).max() - df['low'].rolling(10).min()) + 1e-8)
        
        return result
    
    def _compute_streak(self, change_pct: pd.Series, is_up: bool) -> pd.Series:
        """计算连续涨跌天数"""
        if is_up:
            condition = change_pct > 0
        else:
            condition = change_pct < 0
        
        streak = pd.Series(0, index=change_pct.index)
        current_streak = 0
        
        for i in range(len(change_pct)):
            if condition.iloc[i]:
                current_streak += 1
            else:
                current_streak = 0
            streak.iloc[i] = current_streak
        
        return streak
    
    def _compute_rsi(self, prices: pd.Series, window: int) -> pd.Series:
        """计算RSI"""
        delta = prices.diff()
        gain = delta.where(delta > 0, 0).rolling(window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window).mean()
        rs = gain / (loss + 1e-8)
        return 100 - (100 / (1 + rs))
    
    def _compute_atr(self, df: pd.DataFrame, window: int) -> pd.Series:
        """计算ATR"""
        high = df['high']
        low = df['low']
        close = df['close']
        
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        return tr.rolling(window).mean()
    
    def _compute_adx(self, df: pd.DataFrame, window: int = 14) -> pd.Series:
        """计算ADX趋势强度指标"""
        high = df['high']
        low = df['low']
        close = df['close']
        
        plus_dm = high.diff()
        minus_dm = -low.diff()
        
        plus_dm[plus_dm < 0] = 0
        minus_dm[minus_dm < 0] = 0
        
        tr = self._compute_atr(df, 1) * window
        
        plus_di = 100 * (plus_dm.rolling(window).mean() / (tr + 1e-8))
        minus_di = 100 * (minus_dm.rolling(window).mean() / (tr + 1e-8))
        
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di + 1e-8)
        adx = dx.rolling(window).mean()
        
        return adx
    
    def _create_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        """创建标签"""
        df = df.sort_values(['etf_code', 'date']).reset_index(drop=True)
        
        # T+1日涨跌幅作为标签
        df['label_next_return'] = df.groupby('etf_code')['change_pct'].shift(-1)
        
        # 将涨跌幅转换为排名分数（0-1）
        df['label_score'] = df.groupby('date')['label_next_return'].transform(
            lambda x: (x.rank() - 1) / (len(x) - 1) if len(x) > 1 else 0.5
        )
        
        return df
    
    def get_feature_columns(self) -> List[str]:
        """获取所有特征列名"""
        features = []
        
        # 收益率和波动率特征
        for window in [3, 5, 10, 20, 60]:
            features.extend([f'return_{window}d', f'volatility_{window}d'])
        
        # 均线偏离
        for window in [5, 10, 20, 60]:
            features.append(f'ma{window}_bias')
        
        features.extend([
            'ma_bull', 'ma_bear',
            'volume_ratio', 'volume_trend',
            'price_position_20d', 'price_position_60d',
            'rsi_6', 'rsi_12', 'rsi_24',
            'macd', 'macd_signal', 'macd_hist',
            'bb_width', 'bb_position',
            'atr_ratio',
            'momentum_10', 'momentum_20', 'roc_10', 'roc_20',
            'divergence_score',
            'adx',
            'upper_shadow', 'lower_shadow', 'body_size',
            'change_pct', 'turnover_rate', 'amplitude',
            # 新增特征
            'up_streak', 'down_streak',
            'oversold_signal', 'overbought_signal',
            'momentum_reversal_5d',
            'volume_spike', 'volume_dry',
            'volatility_change',
            'trend_momentum',
            'price_efficiency',
        ])
        
        return features


class ETFRollingBacktestEngine:
    """ETF滚动回测引擎 - 3年训练"""
    
    def __init__(self, 
                 train_window_months: int = 24,  # 2年训练
                 valid_window_months: int = 6,   # 6个月验证
                 step_months: int = 1):          # 每月滚动
        """
        Args:
            train_window_months: 训练窗口（月）
            valid_window_months: 验证窗口（月）
            step_months: 滚动步长（月）
        """
        self.train_window_months = train_window_months
        self.valid_window_months = valid_window_months
        self.step_months = step_months
        
        self.feature_engineer = ETFFeatureEngineer()
        self.results = []
        
        # 优化的LightGBM参数
        self.lgbm_params = {
            "objective": "lambdarank",  # 使用排序目标
            "metric": "ndcg",
            "ndcg_eval_at": [5],
            "boosting_type": "gbdt",
            "num_leaves": 63,
            "learning_rate": 0.03,
            "feature_fraction": 0.7,
            "bagging_fraction": 0.7,
            "bagging_freq": 5,
            "lambda_l1": 0.1,
            "lambda_l2": 0.5,
            "min_data_in_leaf": 30,
            "max_depth": 8,
            "verbose": -1,
            "seed": 42,
            "n_estimators": 500,
            "early_stopping_rounds": 50,
        }
    
    def _get_feature_columns(self, df: pd.DataFrame) -> List[str]:
        """获取可用的特征列"""
        exclude_cols = ['date', 'etf_code', 'etf_name', 'open', 'close', 
                        'high', 'low', 'volume', 'turnover',
                        'ma5', 'ma10', 'ma20', 'ma60',
                        'volume_ma5', 'volume_ma10', 'volume_ma20',
                        'high_20d', 'low_20d', 'high_60d', 'low_60d',
                        'bb_mid', 'bb_upper', 'bb_lower',
                        'volume_rank', 'price_rank',
                        'label_next_return', 'label_score']
        
        return [c for c in df.columns 
                if c not in exclude_cols 
                and df[c].dtype in ['float64', 'int64', 'float32', 'int32']]
    
    def run_backtest(self, df_history: pd.DataFrame,
                     test_start_date: str,
                     test_end_date: str) -> pd.DataFrame:
        """
        运行ETF滚动回测
        
        Args:
            df_history: ETF历史数据
            test_start_date: 测试开始日期
            test_end_date: 测试结束日期
        """
        logger.info(f"开始ETF滚动回测: {test_start_date} -> {test_end_date}")
        logger.info(f"训练窗口: {self.train_window_months}个月")
        
        # 创建特征
        logger.info("创建ETF特征数据集...")
        df = self.feature_engineer.create_features(df_history)
        
        if df.empty:
            logger.error("ETF特征数据为空!")
            return pd.DataFrame()
        
        # 获取特征列
        feature_cols = self._get_feature_columns(df)
        logger.info(f"使用特征: {len(feature_cols)} 个")
        
        # 获取测试日期
        test_dates = df[(df['date'] >= test_start_date) & 
                        (df['date'] <= test_end_date)]['date'].unique()
        test_dates = sorted(test_dates)
        
        logger.info(f"测试日期: {len(test_dates)} 天")
        
        all_predictions = []
        current_month = None
        model = None
        
        for test_date in test_dates:
            test_month = test_date[:7]
            
            # 每月重新训练
            if test_month != current_month:
                current_month = test_month
                
                # 计算训练窗口
                train_end = datetime.strptime(test_date, '%Y-%m-%d') - timedelta(days=1)
                train_start = train_end - timedelta(days=self.train_window_months * 30)
                
                train_end_str = train_end.strftime('%Y-%m-%d')
                train_start_str = train_start.strftime('%Y-%m-%d')
                
                # 获取训练数据
                train_df = df[(df['date'] >= train_start_str) & 
                              (df['date'] <= train_end_str)]
                
                if len(train_df) < 500:
                    logger.warning(f"训练数据不足: {len(train_df)}, 跳过 {test_month}")
                    continue
                
                # 训练模型
                model = self._train_model(train_df, feature_cols)
                logger.info(f"[{test_month}] 训练完成, 样本: {len(train_df)}")
            
            if model is None:
                continue
            
            # 预测当天
            test_df = df[df['date'] == test_date]
            if test_df.empty:
                continue
            
            predictions = self._predict(model, test_df, feature_cols)
            all_predictions.append(predictions)
        
        if not all_predictions:
            return pd.DataFrame()
        
        results = pd.concat(all_predictions, ignore_index=True)
        
        # 评估结果
        self._evaluate_results(results)
        
        # 保存结果
        self._save_results(results)
        
        return results
    
    def _train_model(self, train_df: pd.DataFrame, 
                     feature_cols: List[str]) -> lgb.Booster:
        """训练LightGBM模型 - 优化版V2
        
        核心改进：
        1. 使用更多数据训练（取消过早early stopping）
        2. 添加分类概念（涨/跌）
        3. 更合理的验证集比例
        """
        # 使用原始收益率作为标签
        X = train_df[feature_cols].fillna(0)
        y = train_df['label_next_return'].fillna(0)
        
        # 按时间分割（90%训练，10%验证）
        dates = sorted(train_df['date'].unique())
        split_date = dates[int(len(dates) * 0.9)]
        
        train_mask = train_df['date'] < split_date
        valid_mask = train_df['date'] >= split_date
        
        X_train = X[train_mask]
        y_train = y[train_mask]
        X_valid = X[valid_mask]
        y_valid = y[valid_mask]
        
        # 创建数据集
        train_data = lgb.Dataset(X_train, label=y_train)
        valid_data = lgb.Dataset(X_valid, label=y_valid, reference=train_data)
        
        # 更强的训练参数
        params = {
            "objective": "regression",
            "metric": "mse",
            "boosting_type": "gbdt",
            "num_leaves": 127,           # 增加复杂度
            "learning_rate": 0.05,       # 适中学习率
            "feature_fraction": 0.8,     # 特征采样
            "bagging_fraction": 0.8,     # 数据采样
            "bagging_freq": 3,
            "lambda_l1": 0.05,           # 减小正则化
            "lambda_l2": 0.2,            # 减小正则化
            "min_data_in_leaf": 20,      # 减小叶子节点最小样本
            "max_depth": 10,             # 增加深度
            "min_gain_to_split": 0.001,
            "verbose": -1,
            "seed": 42,
            "num_threads": 4,
            "force_col_wise": True,
        }
        
        model = lgb.train(
            params,
            train_data,
            valid_sets=[train_data, valid_data],
            valid_names=['train', 'valid'],
            num_boost_round=300,         # 固定轮数
            callbacks=[
                lgb.early_stopping(200),  # 放宽early stopping
                lgb.log_evaluation(period=0)
            ]
        )
        
        return model
    
    def _predict(self, model: lgb.Booster, test_df: pd.DataFrame,
                 feature_cols: List[str]) -> pd.DataFrame:
        """预测 - 纯动量策略 + 模型辅助
        
        核心思路（基于动量效应）：
        1. 中期动量（10-20日）是核心
        2. 短期动量（3-5日）确认
        3. 模型排除极端风险
        4. 成交量确认趋势有效性
        """
        X = test_df[feature_cols].fillna(0)
        model_scores = model.predict(X)
        
        result = test_df[['date', 'etf_code', 'etf_name', 'label_next_return']].copy()
        result['model_score'] = model_scores
        
        # 1. 中期动量 (10日涨幅) - 核心因子
        if 'return_10d' in test_df.columns:
            return_10d = test_df['return_10d'].values
            result['momentum_10d'] = pd.Series(return_10d).rank(pct=True).values
        else:
            result['momentum_10d'] = 0.5
        
        # 2. 短期动量 (5日涨幅) - 确认因子
        if 'return_5d' in test_df.columns:
            return_5d = test_df['return_5d'].values
            result['momentum_5d'] = pd.Series(return_5d).rank(pct=True).values
        else:
            result['momentum_5d'] = 0.5
        
        # 3. 成交量趋势 - 量价配合
        if 'volume_trend' in test_df.columns:
            vol_trend = test_df['volume_trend'].values
            result['vol_score'] = pd.Series(vol_trend).rank(pct=True).values
        else:
            result['vol_score'] = 0.5
        
        # 4. 模型分数归一化 - 过滤用
        min_score = result['model_score'].min()
        max_score = result['model_score'].max()
        if max_score > min_score:
            result['model_norm'] = (result['model_score'] - min_score) / (max_score - min_score)
        else:
            result['model_norm'] = 0.5
        
        # 综合分数：动量主导
        # 10日动量 45% + 5日动量 30% + 模型 15% + 成交量 10%
        result['pred_score'] = (
            0.45 * result['momentum_10d'] +
            0.30 * result['momentum_5d'] +
            0.15 * result['model_norm'] +
            0.10 * result['vol_score']
        )
        
        result['pred_rank'] = result['pred_score'].rank(ascending=False)
        result['actual_rank'] = result['label_next_return'].rank(ascending=False)
        
        return result
    
    def _evaluate_results(self, results: pd.DataFrame):
        """评估回测结果"""
        logger.info("\n" + "=" * 60)
        logger.info("📊 ETF回测结果评估")
        logger.info("=" * 60)
        
        # 按日期分组评估
        daily_stats = []
        
        for date in results['date'].unique():
            day_df = results[results['date'] == date]
            
            # Top5预测
            top5_pred = day_df.nsmallest(5, 'pred_rank')
            top5_actual_return = top5_pred['label_next_return'].mean()
            
            # 基准收益
            benchmark_return = day_df['label_next_return'].mean()
            
            # 命中率（预测Top5是否在实际Top10）
            actual_top10 = day_df.nsmallest(10, 'actual_rank')['etf_code'].tolist()
            hit_count = sum(1 for etf in top5_pred['etf_code'] if etf in actual_top10)
            
            daily_stats.append({
                'date': date,
                'top5_return': top5_actual_return,
                'benchmark_return': benchmark_return,
                'excess_return': top5_actual_return - benchmark_return,
                'hit_count': hit_count,
                'hit_rate': hit_count / 5,
            })
        
        stats_df = pd.DataFrame(daily_stats)
        
        # 汇总统计
        logger.info(f"\n📅 统计周期: {stats_df['date'].min()} ~ {stats_df['date'].max()}")
        logger.info(f"📈 总交易天数: {len(stats_df)}")
        logger.info(f"🎯 平均命中率: {stats_df['hit_rate'].mean():.2%}")
        logger.info(f"💰 Top5平均日收益: {stats_df['top5_return'].mean():.3f}%")
        logger.info(f"📊 基准平均日收益: {stats_df['benchmark_return'].mean():.3f}%")
        logger.info(f"💎 平均超额收益: {stats_df['excess_return'].mean():.3f}%")
        logger.info(f"📈 累计超额收益: {stats_df['excess_return'].sum():.2f}%")
        
        # 胜率
        win_days = (stats_df['excess_return'] > 0).sum()
        logger.info(f"🏆 超额收益胜率: {win_days}/{len(stats_df)} ({win_days/len(stats_df):.2%})")
        
        # 最大回撤
        cumulative = stats_df['excess_return'].cumsum()
        max_drawdown = (cumulative.cummax() - cumulative).max()
        logger.info(f"📉 最大回撤: {max_drawdown:.2f}%")
        
        # 夏普比率
        daily_std = stats_df['excess_return'].std()
        if daily_std > 0:
            sharpe = stats_df['excess_return'].mean() / daily_std * np.sqrt(252)
            logger.info(f"📊 夏普比率: {sharpe:.2f}")
        
        logger.info("=" * 60)
        
        self.backtest_stats = stats_df
    
    def _save_results(self, results: pd.DataFrame):
        """保存回测结果"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # 保存详细结果
        result_path = BACKTEST_RESULTS_DIR / f"etf_backtest_detail_{timestamp}.csv"
        results.to_csv(result_path, index=False, encoding='utf-8-sig')
        logger.info(f"详细结果已保存: {result_path}")
        
        # 保存汇总统计
        if hasattr(self, 'backtest_stats'):
            stats_path = BACKTEST_RESULTS_DIR / f"etf_backtest_stats_{timestamp}.csv"
            self.backtest_stats.to_csv(stats_path, index=False, encoding='utf-8-sig')
            logger.info(f"统计结果已保存: {stats_path}")
        
        # 生成报告
        self._generate_report(results, timestamp)
    
    def _generate_report(self, results: pd.DataFrame, timestamp: str):
        """生成回测报告"""
        stats = self.backtest_stats if hasattr(self, 'backtest_stats') else None
        
        lines = [
            "# 📊 ETF预测系统回测报告",
            "",
            f"**回测时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"**训练窗口**: {self.train_window_months} 个月",
            "",
            "---",
            "",
            "## 📈 综合绩效",
            "",
        ]
        
        if stats is not None:
            lines.extend([
                f"| 指标 | 数值 |",
                f"|------|------|",
                f"| 统计周期 | {stats['date'].min()} ~ {stats['date'].max()} |",
                f"| 交易天数 | {len(stats)} |",
                f"| 平均命中率 | {stats['hit_rate'].mean():.2%} |",
                f"| Top5平均日收益 | {stats['top5_return'].mean():.3f}% |",
                f"| 基准平均日收益 | {stats['benchmark_return'].mean():.3f}% |",
                f"| 平均超额收益 | {stats['excess_return'].mean():.3f}% |",
                f"| 累计超额收益 | {stats['excess_return'].sum():.2f}% |",
                f"| 超额收益胜率 | {(stats['excess_return'] > 0).mean():.2%} |",
                "",
            ])
        
        lines.extend([
            "---",
            "",
            "## 🔧 模型参数",
            "",
            "```",
            f"训练窗口: {self.train_window_months} 个月",
            f"滚动步长: {self.step_months} 个月",
            f"模型: LightGBM",
            f"特征数: {len(self.feature_engineer.get_feature_columns())}",
            "```",
            "",
            "---",
            "",
            "*报告由 ETF预测系统 自动生成*",
        ])
        
        report_content = "\n".join(lines)
        report_path = BACKTEST_RESULTS_DIR / f"etf_backtest_report_{timestamp}.md"
        
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(report_content)
        
        logger.info(f"回测报告已生成: {report_path}")
