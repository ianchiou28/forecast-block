"""
A股板块涨停预测系统 - 滚动回测引擎
真正的历史回测验证：用过去数据训练，预测未来数据
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

import lightgbm as lgb
from sklearn.metrics import ndcg_score, mean_squared_error

warnings.filterwarnings('ignore')

PROJECT_ROOT = Path(__file__).parent.parent
HISTORICAL_DATA_DIR = PROJECT_ROOT / "data" / "historical"
BACKTEST_RESULTS_DIR = PROJECT_ROOT / "data" / "backtest_results"
BACKTEST_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

logger = logging.getLogger(__name__)


class HistoricalFeatureEngineer:
    """历史数据特征工程"""
    
    def __init__(self):
        self.feature_columns = [
            # 基础行情特征
            "change_pct",           # 当日涨跌幅
            "turnover_rate",        # 换手率
            "amplitude",            # 振幅
            
            # 时间序列特征
            "change_pct_ma3",       # 3日涨幅均线
            "change_pct_ma5",       # 5日涨幅均线
            "change_pct_ma10",      # 10日涨幅均线
            "change_pct_std5",      # 5日涨幅波动率
            "change_pct_std10",     # 10日涨幅波动率
            
            # 动量特征
            "momentum_3d",          # 3日动量
            "momentum_5d",          # 5日动量
            "momentum_10d",         # 10日动量
            "momentum_20d",         # 20日动量
            
            # 涨停相关特征
            "limit_up_count",       # 当日涨停家数
            "limit_up_count_ma3",   # 3日涨停均线
            "limit_up_count_ma5",   # 5日涨停均线
            "limit_up_rank",        # 涨停家数排名
            
            # 成交额特征
            "turnover_ma5",         # 5日成交额均线
            "turnover_ma10",        # 10日成交额均线
            "volume_ratio",         # 量比
            
            # 趋势特征
            "above_ma5",            # 价格在5日均线上方
            "above_ma10",           # 价格在10日均线上方
            "above_ma20",           # 价格在20日均线上方
            
            # 量价背离特征（核心）
            "divergence_score",     # 量价背离评分
        ]
    
    def create_features(self, sector_history: pd.DataFrame, 
                        limit_up_agg: pd.DataFrame) -> pd.DataFrame:
        """
        创建特征数据集
        
        Args:
            sector_history: 板块历史行情数据
            limit_up_agg: 按板块聚合的涨停数据
            
        Returns:
            包含特征和标签的DataFrame
        """
        if sector_history.empty:
            return pd.DataFrame()
        
        # 确保日期格式一致
        sector_history = sector_history.copy()
        sector_history['date'] = pd.to_datetime(sector_history['date']).dt.strftime('%Y-%m-%d')
        
        # 按板块分组处理
        all_features = []
        
        for sector_name in sector_history['sector_name'].unique():
            sector_df = sector_history[sector_history['sector_name'] == sector_name].copy()
            sector_df = sector_df.sort_values('date').reset_index(drop=True)
            
            if len(sector_df) < 30:  # 需要足够的历史数据
                continue
            
            # 基础特征
            features = sector_df[['date', 'sector_name', 'sector_type', 
                                   'open', 'close', 'high', 'low', 
                                   'volume', 'turnover', 'change_pct', 
                                   'turnover_rate', 'amplitude']].copy()
            
            # 时间序列特征
            features['change_pct_ma3'] = sector_df['change_pct'].rolling(3).mean()
            features['change_pct_ma5'] = sector_df['change_pct'].rolling(5).mean()
            features['change_pct_ma10'] = sector_df['change_pct'].rolling(10).mean()
            features['change_pct_std5'] = sector_df['change_pct'].rolling(5).std()
            features['change_pct_std10'] = sector_df['change_pct'].rolling(10).std()
            
            # 动量特征
            features['momentum_3d'] = sector_df['close'].pct_change(3) * 100
            features['momentum_5d'] = sector_df['close'].pct_change(5) * 100
            features['momentum_10d'] = sector_df['close'].pct_change(10) * 100
            features['momentum_20d'] = sector_df['close'].pct_change(20) * 100
            
            # 成交额特征
            features['turnover_ma5'] = sector_df['turnover'].rolling(5).mean()
            features['turnover_ma10'] = sector_df['turnover'].rolling(10).mean()
            features['volume_ratio'] = sector_df['turnover'] / features['turnover_ma5']
            
            # 均线位置特征
            features['ma5'] = sector_df['close'].rolling(5).mean()
            features['ma10'] = sector_df['close'].rolling(10).mean()
            features['ma20'] = sector_df['close'].rolling(20).mean()
            features['above_ma5'] = (sector_df['close'] > features['ma5']).astype(int)
            features['above_ma10'] = (sector_df['close'] > features['ma10']).astype(int)
            features['above_ma20'] = (sector_df['close'] > features['ma20']).astype(int)
            
            # 量价背离评分（核心特征）
            # 资金流入排名高但涨幅排名低 = 吸筹信号
            features['turnover_rank'] = features['turnover'].rank(pct=True)
            features['change_rank'] = features['change_pct'].rank(pct=True)
            features['divergence_score'] = features['turnover_rank'] - features['change_rank']
            
            all_features.append(features)
        
        if not all_features:
            return pd.DataFrame()
        
        result = pd.concat(all_features, ignore_index=True)
        
        # 合并涨停数据
        if not limit_up_agg.empty:
            limit_up_agg = limit_up_agg.copy()
            limit_up_agg['date'] = pd.to_datetime(limit_up_agg['date'].astype(str)).dt.strftime('%Y-%m-%d')
            result = result.merge(
                limit_up_agg[['date', 'sector_name', 'limit_up_count', 
                              'total_seal_amount', 'max_continuous_limit_up']],
                on=['date', 'sector_name'],
                how='left'
            )
            result['limit_up_count'] = result['limit_up_count'].fillna(0)
        else:
            result['limit_up_count'] = 0
            result['total_seal_amount'] = 0
            result['max_continuous_limit_up'] = 0
        
        # 涨停家数时序特征
        for sector_name in result['sector_name'].unique():
            mask = result['sector_name'] == sector_name
            result.loc[mask, 'limit_up_count_ma3'] = result.loc[mask, 'limit_up_count'].rolling(3).mean()
            result.loc[mask, 'limit_up_count_ma5'] = result.loc[mask, 'limit_up_count'].rolling(5).mean()
        
        # 每日涨停排名
        for date in result['date'].unique():
            mask = result['date'] == date
            result.loc[mask, 'limit_up_rank'] = result.loc[mask, 'limit_up_count'].rank(ascending=False)
        
        # 创建标签（T+1日涨幅和涨停数）
        result = result.sort_values(['sector_name', 'date'])
        result['label_change_pct'] = result.groupby('sector_name')['change_pct'].shift(-1)
        result['label_limit_up_count'] = result.groupby('sector_name')['limit_up_count'].shift(-1)
        
        # 综合标签：涨幅 + 涨停奖励
        result['label_score'] = result['label_change_pct'] + result['label_limit_up_count'] * 2
        
        # 删除缺失值
        result = result.dropna(subset=['label_score'])
        
        logger.info(f"特征工程完成: {len(result)} 条数据, {result['sector_name'].nunique()} 个板块")
        return result


class RollingBacktestEngine:
    """滚动回测引擎"""
    
    def __init__(self, train_window_months: int = 24, 
                 valid_window_months: int = 3,
                 step_months: int = 1):
        """
        Args:
            train_window_months: 训练窗口（月）
            valid_window_months: 验证窗口（月）
            step_months: 滚动步长（月）
        """
        self.train_window_months = train_window_months
        self.valid_window_months = valid_window_months
        self.step_months = step_months
        
        self.feature_engineer = HistoricalFeatureEngineer()
        self.results = []
        
        # LightGBM参数
        self.lgbm_params = {
            "objective": "regression",
            "metric": "mse",
            "boosting_type": "gbdt",
            "num_leaves": 31,
            "learning_rate": 0.05,
            "feature_fraction": 0.8,
            "bagging_fraction": 0.8,
            "bagging_freq": 5,
            "lambda_l1": 0.1,
            "lambda_l2": 0.1,
            "min_data_in_leaf": 20,
            "verbose": -1,
            "seed": 42,
        }
    
    def _get_feature_columns(self, df: pd.DataFrame) -> List[str]:
        """获取可用的特征列"""
        exclude_cols = ['date', 'sector_name', 'sector_type', 'open', 'close', 
                        'high', 'low', 'volume', 'turnover', 'ma5', 'ma10', 'ma20',
                        'turnover_rank', 'change_rank',
                        'label_change_pct', 'label_limit_up_count', 'label_score']
        
        return [c for c in df.columns if c not in exclude_cols and df[c].dtype in ['float64', 'int64', 'float32', 'int32']]
    
    def run_backtest(self, sector_history: pd.DataFrame, 
                     limit_up_history: pd.DataFrame,
                     test_start_date: str,
                     test_end_date: str) -> pd.DataFrame:
        """
        运行滚动回测
        
        Args:
            sector_history: 板块历史行情
            limit_up_history: 涨停池历史
            test_start_date: 测试开始日期 (YYYY-MM-DD)
            test_end_date: 测试结束日期 (YYYY-MM-DD)
            
        Returns:
            回测结果DataFrame
        """
        logger.info(f"开始滚动回测: {test_start_date} -> {test_end_date}")
        logger.info(f"训练窗口: {self.train_window_months}个月, 步长: {self.step_months}个月")
        
        # 聚合涨停数据
        from .historical_data import HistoricalDataFetcher
        fetcher = HistoricalDataFetcher()
        limit_up_agg = fetcher.aggregate_limit_up_by_sector(limit_up_history)
        
        # 创建特征
        logger.info("创建特征数据集...")
        df = self.feature_engineer.create_features(sector_history, limit_up_agg)
        
        if df.empty:
            logger.error("特征数据为空!")
            return pd.DataFrame()
        
        # 获取特征列
        feature_cols = self._get_feature_columns(df)
        logger.info(f"使用特征: {len(feature_cols)} 个")
        
        # 获取测试期内的所有日期
        test_dates = df[(df['date'] >= test_start_date) & 
                        (df['date'] <= test_end_date)]['date'].unique()
        test_dates = sorted(test_dates)
        
        logger.info(f"测试日期: {len(test_dates)} 天")
        
        all_predictions = []
        
        # 按月滚动
        current_month = None
        model = None
        
        for i, test_date in enumerate(test_dates):
            test_month = test_date[:7]  # YYYY-MM
            
            # 每月重新训练模型
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
                
                if len(train_df) < 100:
                    logger.warning(f"训练数据不足: {len(train_df)} 条, 跳过 {test_month}")
                    continue
                
                # 训练模型
                X_train = train_df[feature_cols].fillna(0)
                y_train = train_df['label_score']
                
                train_data = lgb.Dataset(X_train, label=y_train)
                
                model = lgb.train(
                    self.lgbm_params,
                    train_data,
                    num_boost_round=300,
                    callbacks=[lgb.log_evaluation(period=0)]
                )
                
                logger.info(f"模型训练完成 ({test_month}): {len(train_df)} 样本, "
                           f"{train_start_str} -> {train_end_str}")
            
            if model is None:
                continue
            
            # 获取测试日的数据
            test_df = df[df['date'] == test_date].copy()
            
            if test_df.empty:
                continue
            
            # 预测
            X_test = test_df[feature_cols].fillna(0)
            test_df['pred_score'] = model.predict(X_test)
            
            # 按预测分数排名
            test_df['pred_rank'] = test_df['pred_score'].rank(ascending=False)
            
            # 保存结果
            result = test_df[['date', 'sector_name', 'sector_type',
                              'change_pct', 'limit_up_count',
                              'label_change_pct', 'label_limit_up_count', 'label_score',
                              'pred_score', 'pred_rank']].copy()
            all_predictions.append(result)
            
            if (i + 1) % 50 == 0:
                logger.info(f"回测进度: {i+1}/{len(test_dates)} ({(i+1)/len(test_dates)*100:.1f}%)")
        
        if not all_predictions:
            logger.error("没有生成任何预测结果!")
            return pd.DataFrame()
        
        results = pd.concat(all_predictions, ignore_index=True)
        self.results = results
        
        logger.info(f"回测完成: {len(results)} 条预测, {results['date'].nunique()} 个交易日")
        
        return results
    
    def save_results(self, results: pd.DataFrame, name: str = "backtest"):
        """保存回测结果"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        path = BACKTEST_RESULTS_DIR / f"{name}_{timestamp}.parquet"
        results.to_parquet(path, index=False)
        logger.info(f"回测结果已保存: {path}")
        return path


class BacktestEvaluator:
    """回测结果评估器"""
    
    def __init__(self, results: pd.DataFrame):
        """
        Args:
            results: 回测结果DataFrame
        """
        self.results = results
        self.daily_metrics = None
        self.overall_metrics = None
    
    def calculate_daily_metrics(self, top_k: int = 5) -> pd.DataFrame:
        """
        计算每日评估指标
        
        Args:
            top_k: 选取Top-K个预测
            
        Returns:
            每日指标DataFrame
        """
        daily_metrics = []
        
        for date in self.results['date'].unique():
            day_df = self.results[self.results['date'] == date].copy()
            
            if len(day_df) < top_k:
                continue
            
            # 选取预测排名前K的板块
            top_k_df = day_df.nsmallest(top_k, 'pred_rank')
            
            # 计算指标
            top_k_avg_return = top_k_df['label_change_pct'].mean()
            market_avg_return = day_df['label_change_pct'].mean()
            excess_return = top_k_avg_return - market_avg_return
            
            # 计算预测板块在全市场的排名表现
            # 如果预测的Top-K板块实际涨幅排在前20%，算命中
            day_df_sorted = day_df.sort_values('label_change_pct', ascending=False)
            top_20pct_threshold = day_df_sorted['label_change_pct'].quantile(0.8)
            top_10pct_threshold = day_df_sorted['label_change_pct'].quantile(0.9)
            
            metrics = {
                'date': date,
                'total_sectors': len(day_df),
                
                # 收益指标
                'top_k_avg_return': top_k_avg_return,
                'top_k_total_return': top_k_df['label_change_pct'].sum(),
                'market_avg_return': market_avg_return,
                'excess_return': excess_return,
                
                # 命中率指标（基于板块涨幅）
                # 涨幅>1%命中率
                'hit_rate_1pct': (top_k_df['label_change_pct'] > 1).mean(),
                # 涨幅>2%命中率
                'hit_rate_2pct': (top_k_df['label_change_pct'] > 2).mean(),
                # 涨幅>3%命中率（强势板块）
                'hit_rate_3pct': (top_k_df['label_change_pct'] > 3).mean(),
                
                # 排名命中率
                # 预测板块进入当日涨幅前20%
                'rank_hit_top20': (top_k_df['label_change_pct'] >= top_20pct_threshold).mean(),
                # 预测板块进入当日涨幅前10%
                'rank_hit_top10': (top_k_df['label_change_pct'] >= top_10pct_threshold).mean(),
                
                # 超额收益命中
                # 预测板块涨幅超过市场平均
                'beat_market_rate': (top_k_df['label_change_pct'] > market_avg_return).mean(),
            }
            
            # 计算NDCG@K
            if len(day_df) >= top_k:
                try:
                    true_relevance = day_df['label_score'].values.reshape(1, -1)
                    pred_scores = day_df['pred_score'].values.reshape(1, -1)
                    metrics['ndcg'] = ndcg_score(true_relevance, pred_scores, k=top_k)
                except:
                    metrics['ndcg'] = np.nan
            else:
                metrics['ndcg'] = np.nan
            
            daily_metrics.append(metrics)
        
        self.daily_metrics = pd.DataFrame(daily_metrics)
        return self.daily_metrics
    
    def calculate_overall_metrics(self) -> Dict:
        """计算整体评估指标"""
        if self.daily_metrics is None:
            self.calculate_daily_metrics()
        
        dm = self.daily_metrics
        
        # 基础统计
        total_days = len(dm)
        
        # 收益统计
        avg_daily_return = dm['top_k_avg_return'].mean()
        total_return = dm['top_k_avg_return'].sum()
        avg_excess_return = dm['excess_return'].mean()
        total_excess_return = dm['excess_return'].sum()
        
        # 胜率统计（预测板块收益>0）
        win_days = (dm['top_k_avg_return'] > 0).sum()
        win_rate = win_days / total_days if total_days > 0 else 0
        
        # 超额胜率（预测板块跑赢大盘的天数）
        beat_market_days = (dm['excess_return'] > 0).sum()
        beat_market_rate = beat_market_days / total_days if total_days > 0 else 0
        
        # 命中率统计（基于涨幅阈值）
        avg_hit_rate_1pct = dm['hit_rate_1pct'].mean()
        avg_hit_rate_2pct = dm['hit_rate_2pct'].mean()
        avg_hit_rate_3pct = dm['hit_rate_3pct'].mean()
        
        # 排名命中率
        avg_rank_hit_top20 = dm['rank_hit_top20'].mean()
        avg_rank_hit_top10 = dm['rank_hit_top10'].mean()
        avg_beat_market = dm['beat_market_rate'].mean()
        
        # NDCG统计
        avg_ndcg = dm['ndcg'].mean()
        
        # 风险指标
        daily_returns = dm['top_k_avg_return']
        sharpe_ratio = (daily_returns.mean() / daily_returns.std() * np.sqrt(252)) if daily_returns.std() > 0 else 0
        
        # 最大回撤
        cumulative_return = (1 + daily_returns / 100).cumprod()
        rolling_max = cumulative_return.expanding().max()
        drawdown = (cumulative_return - rolling_max) / rolling_max
        max_drawdown = drawdown.min() * 100
        
        # 按成功/失败天数统计（以跑赢市场为标准）
        success_days = dm[dm['excess_return'] > 0]
        fail_days = dm[dm['excess_return'] <= 0]
        
        avg_success_return = success_days['top_k_avg_return'].mean() if len(success_days) > 0 else 0
        avg_fail_return = fail_days['top_k_avg_return'].mean() if len(fail_days) > 0 else 0
        
        self.overall_metrics = {
            # 基础统计
            'total_trading_days': total_days,
            'win_days': win_days,
            'beat_market_days': beat_market_days,
            'success_days': len(success_days),
            'fail_days': len(fail_days),
            'win_rate': win_rate,
            'beat_market_rate': beat_market_rate,
            
            # 收益指标
            'avg_daily_return': avg_daily_return,
            'total_return': total_return,
            'avg_excess_return': avg_excess_return,
            'total_excess_return': total_excess_return,
            'avg_success_return': avg_success_return,
            'avg_fail_return': avg_fail_return,
            
            # 命中率指标（板块涨幅阈值）
            'hit_rate_1pct': avg_hit_rate_1pct,
            'hit_rate_2pct': avg_hit_rate_2pct,
            'hit_rate_3pct': avg_hit_rate_3pct,
            
            # 排名命中率
            'rank_hit_top20': avg_rank_hit_top20,
            'rank_hit_top10': avg_rank_hit_top10,
            'beat_market_avg': avg_beat_market,
            
            # 排序指标
            'avg_ndcg': avg_ndcg,
            
            # 风险指标
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'return_std': daily_returns.std(),
        }
        
        return self.overall_metrics
    
    def generate_report(self, output_path: str = None) -> str:
        """生成回测报告"""
        if self.overall_metrics is None:
            self.calculate_overall_metrics()
        
        om = self.overall_metrics
        dm = self.daily_metrics
        
        # 月度统计
        dm['month'] = pd.to_datetime(dm['date']).dt.strftime('%Y-%m')
        monthly = dm.groupby('month').agg({
            'top_k_avg_return': 'sum',
            'excess_return': 'sum',
            'rank_hit_top20': 'mean',
            'ndcg': 'mean',
        }).reset_index()
        
        report_lines = [
            "# 📊 A股板块预测系统 - 回测验证报告",
            "",
            f"**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "---",
            "",
            "## 📈 整体绩效统计",
            "",
            "### 基础统计",
            "",
            f"| 指标 | 数值 | 说明 |",
            f"|------|------|------|",
            f"| 总交易天数 | {om['total_trading_days']} | 回测覆盖的交易日 |",
            f"| 盈利天数 | {om['win_days']} | 预测板块收益>0的天数 |",
            f"| 跑赢市场天数 | {om['beat_market_days']} | 预测板块跑赢大盘的天数 |",
            f"| **胜率** | **{om['win_rate']:.2%}** | 盈利天数/总天数 |",
            f"| **超额胜率** | **{om['beat_market_rate']:.2%}** | 跑赢市场天数/总天数 |",
            "",
            "### 收益指标",
            "",
            f"| 指标 | 数值 | 说明 |",
            f"|------|------|------|",
            f"| 日均收益 | {om['avg_daily_return']:.2f}% | 预测板块的平均日收益 |",
            f"| **累计收益** | **{om['total_return']:.2f}%** | 全周期累计收益 |",
            f"| 日均超额收益 | {om['avg_excess_return']:.2f}% | 超过市场平均的收益 |",
            f"| **累计超额收益** | **{om['total_excess_return']:.2f}%** | 全周期累计超额 |",
            f"| 成功日平均涨幅 | {om['avg_success_return']:.2f}% | 跑赢市场时的平均收益 |",
            f"| 失败日平均收益 | {om['avg_fail_return']:.2f}% | 跑输市场时的平均收益 |",
            "",
            "### 命中率指标（板块涨幅）",
            "",
            f"| 指标 | 数值 | 说明 |",
            f"|------|------|------|",
            f"| 涨幅>1%命中率 | {om['hit_rate_1pct']:.2%} | 预测板块涨幅超1%的比例 |",
            f"| 涨幅>2%命中率 | {om['hit_rate_2pct']:.2%} | 预测板块涨幅超2%的比例 |",
            f"| **涨幅>3%命中率** | **{om['hit_rate_3pct']:.2%}** | 预测板块涨幅超3%的比例 |",
            "",
            "### 排名命中率",
            "",
            f"| 指标 | 数值 | 说明 |",
            f"|------|------|------|",
            f"| **进入Top20%** | **{om['rank_hit_top20']:.2%}** | 预测板块实际涨幅排前20% |",
            f"| 进入Top10% | {om['rank_hit_top10']:.2%} | 预测板块实际涨幅排前10% |",
            f"| 跑赢市场均值 | {om['beat_market_avg']:.2%} | 预测板块涨幅超过市场均值 |",
            "",
            "### 排序质量与风险指标",
            "",
            f"| 指标 | 数值 | 说明 |",
            f"|------|------|------|",
            f"| **NDCG@5** | **{om['avg_ndcg']:.4f}** | 排序质量评分(0-1) |",
            f"| **夏普比率** | **{om['sharpe_ratio']:.2f}** | 风险调整后收益 |",
            f"| 最大回撤 | {om['max_drawdown']:.2f}% | 最大亏损幅度 |",
            f"| 日收益标准差 | {om['return_std']:.2f}% | 收益波动率 |",
            "",
            "---",
            "",
            "## 📅 月度绩效",
            "",
            "| 月份 | 累计收益 | 超额收益 | Top20%命中 | NDCG |",
            "|------|---------|---------|------------|------|",
        ]
        
        for _, row in monthly.iterrows():
            report_lines.append(
                f"| {row['month']} | {row['top_k_avg_return']:.2f}% | "
                f"{row['excess_return']:.2f}% | {row['rank_hit_top20']:.2%} | "
                f"{row['ndcg']:.4f} |"
            )
        
        report_lines.extend([
            "",
            "---",
            "",
            "## 💡 结论与建议",
            "",
        ])
        
        # 根据结果给出评价
        if om['avg_ndcg'] > 0.65 and om['beat_market_rate'] > 0.55:
            report_lines.append("✅ **模型表现优秀**: NDCG和超额胜率均达到较高水平，可考虑实盘测试。")
        elif om['avg_ndcg'] > 0.55 and om['beat_market_rate'] > 0.50:
            report_lines.append("🟡 **模型表现良好**: 有一定预测能力，建议继续优化特征和参数。")
        else:
            report_lines.append("🔴 **模型需要改进**: 预测效果不理想，需要重新审视特征工程和模型选择。")
        
        report_lines.extend([
            "",
            "---",
            "",
            "*报告由 A股板块涨停预测系统 自动生成*",
        ])
        
        report_content = "\n".join(report_lines)
        
        if output_path is None:
            output_path = BACKTEST_RESULTS_DIR / f"backtest_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(report_content)
        
        logger.info(f"回测报告已生成: {output_path}")
        return str(output_path)
    
    def print_summary(self):
        """打印回测摘要"""
        if self.overall_metrics is None:
            self.calculate_overall_metrics()
        
        om = self.overall_metrics
        
        print("\n" + "=" * 60)
        print("📊 回测绩效摘要")
        print("=" * 60)
        
        print(f"\n📅 交易统计:")
        print(f"   总交易天数: {om['total_trading_days']}")
        print(f"   盈利天数: {om['win_days']} ({om['win_rate']*100:.1f}%)")
        print(f"   跑赢市场天数: {om['beat_market_days']} ({om['beat_market_rate']*100:.1f}%)")
        
        print(f"\n💰 收益统计:")
        print(f"   日均收益: {om['avg_daily_return']:.2f}%")
        print(f"   累计收益: {om['total_return']:.2f}%")
        print(f"   日均超额: {om['avg_excess_return']:.2f}%")
        print(f"   累计超额: {om['total_excess_return']:.2f}%")
        
        print(f"\n🎯 命中率（板块涨幅）:")
        print(f"   涨幅>1%命中率: {om['hit_rate_1pct']:.2%}")
        print(f"   涨幅>2%命中率: {om['hit_rate_2pct']:.2%}")
        print(f"   涨幅>3%命中率: {om['hit_rate_3pct']:.2%}")
        
        print(f"\n📊 排名命中率:")
        print(f"   进入Top20%: {om['rank_hit_top20']:.2%}")
        print(f"   进入Top10%: {om['rank_hit_top10']:.2%}")
        print(f"   跑赢市场均值: {om['beat_market_avg']:.2%}")
        
        print(f"\n📈 排序质量:")
        print(f"   NDCG@5: {om['avg_ndcg']:.4f}")
        
        print(f"\n⚠️ 风险指标:")
        print(f"   夏普比率: {om['sharpe_ratio']:.2f}")
        print(f"   最大回撤: {om['max_drawdown']:.2f}%")
        
        print("\n" + "=" * 60)


def run_full_backtest(train_years: Tuple[int, int] = (2022, 2023),
                      test_year: int = 2024):
    """
    运行完整回测流程
    
    Args:
        train_years: 训练年份范围
        test_year: 测试年份
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )
    
    print(f"\n{'='*60}")
    print(f"🚀 开始历史回测验证")
    print(f"   训练数据: {train_years[0]} - {train_years[1]}")
    print(f"   测试数据: {test_year}")
    print(f"{'='*60}\n")
    
    # 1. 加载历史数据
    from .historical_data import HistoricalDataFetcher
    fetcher = HistoricalDataFetcher()
    
    print("📂 加载历史数据...")
    sector_history, limit_up_history = fetcher.load_historical_data()
    
    if sector_history.empty:
        print("❌ 历史数据为空! 请先运行:")
        print("   python -m backtest.historical_data")
        return
    
    print(f"   ✓ 板块数据: {len(sector_history)} 条")
    print(f"   ✓ 涨停数据: {len(limit_up_history)} 条\n")
    
    # 2. 运行回测
    print("🔄 运行滚动回测...")
    engine = RollingBacktestEngine(
        train_window_months=24,
        valid_window_months=3,
        step_months=1
    )
    
    results = engine.run_backtest(
        sector_history=sector_history,
        limit_up_history=limit_up_history,
        test_start_date=f"{test_year}-01-01",
        test_end_date=f"{test_year}-12-31"
    )
    
    if results.empty:
        print("❌ 回测没有生成结果!")
        return
    
    # 3. 保存结果
    engine.save_results(results, f"backtest_{train_years[0]}-{train_years[1]}_test_{test_year}")
    
    # 4. 评估结果
    print("\n📊 评估回测结果...")
    evaluator = BacktestEvaluator(results)
    evaluator.calculate_daily_metrics(top_k=5)
    evaluator.calculate_overall_metrics()
    
    # 5. 生成报告
    report_path = evaluator.generate_report()
    
    # 6. 打印摘要
    evaluator.print_summary()
    
    print(f"\n📄 详细报告: {report_path}")
    print(f"{'='*60}\n")
    
    return evaluator


if __name__ == "__main__":
    # 运行 2022-2023 训练 -> 2024 测试 的回测
    run_full_backtest(train_years=(2022, 2023), test_year=2024)
