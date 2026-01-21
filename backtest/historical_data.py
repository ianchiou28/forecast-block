"""
A股板块涨停预测系统 - 历史数据获取模块
获取2022-2025年的历史数据用于回测验证
"""
import akshare as ak
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Tuple
import logging
import time
import os
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import warnings

warnings.filterwarnings('ignore')

# 获取项目根目录
PROJECT_ROOT = Path(__file__).parent.parent
HISTORICAL_DATA_DIR = PROJECT_ROOT / "data" / "historical"

logger = logging.getLogger(__name__)


class HistoricalDataFetcher:
    """历史数据获取器 - 用于获取2022-2025年回测数据"""
    
    def __init__(self):
        self.retry_times = 3
        self.retry_delay = 2
        HISTORICAL_DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    def _retry_fetch(self, func, *args, **kwargs):
        """带重试的数据获取"""
        for i in range(self.retry_times):
            try:
                result = func(*args, **kwargs)
                return result
            except Exception as e:
                logger.warning(f"数据获取失败 (尝试 {i+1}/{self.retry_times}): {e}")
                if i < self.retry_times - 1:
                    time.sleep(self.retry_delay * (i + 1))
        return None
    
    def get_trading_dates(self, start_date: str, end_date: str) -> List[str]:
        """
        获取指定范围内的交易日期
        
        Args:
            start_date: 开始日期 YYYY-MM-DD
            end_date: 结束日期 YYYY-MM-DD
            
        Returns:
            交易日期列表 ['20220104', '20220105', ...]
        """
        logger.info(f"获取交易日历: {start_date} -> {end_date}")
        
        try:
            # 使用上证指数历史数据获取交易日
            df = ak.stock_zh_index_daily(symbol="sh000001")
            df['date'] = pd.to_datetime(df['date'])
            
            start = pd.to_datetime(start_date)
            end = pd.to_datetime(end_date)
            
            trading_dates = df[(df['date'] >= start) & (df['date'] <= end)]['date']
            trading_dates = [d.strftime('%Y%m%d') for d in trading_dates]
            
            logger.info(f"获取到 {len(trading_dates)} 个交易日")
            return trading_dates
        except Exception as e:
            logger.error(f"获取交易日历失败: {e}")
            return []
    
    def fetch_limit_up_history(self, date: str) -> Optional[pd.DataFrame]:
        """
        获取指定日期的涨停池数据
        
        Args:
            date: 日期 YYYYMMDD格式
            
        Returns:
            涨停池DataFrame
        """
        try:
            df = ak.stock_zt_pool_em(date=date)
            
            if df is not None and not df.empty:
                df = df.rename(columns={
                    "序号": "rank",
                    "代码": "stock_code",
                    "名称": "stock_name",
                    "涨跌幅": "change_pct",
                    "最新价": "close_price",
                    "成交额": "turnover",
                    "流通市值": "float_market_cap",
                    "总市值": "total_market_cap",
                    "换手率": "turnover_rate",
                    "封板资金": "seal_amount",
                    "首次封板时间": "first_seal_time",
                    "最后封板时间": "last_seal_time",
                    "炸板次数": "open_count",
                    "涨停统计": "limit_up_stats",
                    "连板数": "continuous_limit_up",
                    "所属行业": "industry",
                })
                df["date"] = date
                return df
        except Exception as e:
            logger.debug(f"获取 {date} 涨停池失败: {e}")
        return None
    
    def fetch_concept_board_list(self) -> List[str]:
        """获取所有概念板块列表"""
        try:
            df = ak.stock_board_concept_name_em()
            if df is not None:
                return df['板块名称'].tolist()
        except Exception as e:
            logger.error(f"获取概念板块列表失败: {e}")
        return []
    
    def fetch_concept_history(self, concept_name: str, 
                               start_date: str = None, 
                               end_date: str = None) -> Optional[pd.DataFrame]:
        """
        获取单个概念板块的历史行情
        
        Args:
            concept_name: 概念板块名称
            start_date: 开始日期
            end_date: 结束日期
            
        Returns:
            历史行情DataFrame
        """
        try:
            df = ak.stock_board_concept_hist_em(
                symbol=concept_name,
                period="daily",
                start_date=start_date,
                end_date=end_date,
                adjust=""
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
                df["sector_name"] = concept_name
                df["sector_type"] = "concept"
                return df
        except Exception as e:
            logger.debug(f"获取概念 {concept_name} 历史行情失败: {e}")
        return None
    
    def fetch_industry_board_list(self) -> List[str]:
        """获取所有行业板块列表"""
        try:
            df = ak.stock_board_industry_name_em()
            if df is not None:
                return df['板块名称'].tolist()
        except Exception as e:
            logger.error(f"获取行业板块列表失败: {e}")
        return []
    
    def fetch_industry_history(self, industry_name: str,
                                start_date: str = None,
                                end_date: str = None) -> Optional[pd.DataFrame]:
        """
        获取单个行业板块的历史行情
        """
        try:
            df = ak.stock_board_industry_hist_em(
                symbol=industry_name,
                period="daily",
                start_date=start_date,
                end_date=end_date,
                adjust=""
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
                df["sector_name"] = industry_name
                df["sector_type"] = "industry"
                return df
        except Exception as e:
            logger.debug(f"获取行业 {industry_name} 历史行情失败: {e}")
        return None
    
    def fetch_all_sector_history(self, start_date: str, end_date: str,
                                  sector_type: str = "both") -> pd.DataFrame:
        """
        获取所有板块的历史行情数据
        
        Args:
            start_date: 开始日期 YYYYMMDD
            end_date: 结束日期 YYYYMMDD
            sector_type: "concept", "industry", or "both"
            
        Returns:
            合并后的历史行情DataFrame
        """
        all_data = []
        
        # 获取概念板块
        if sector_type in ["concept", "both"]:
            concepts = self.fetch_concept_board_list()
            logger.info(f"开始获取 {len(concepts)} 个概念板块历史数据...")
            
            for i, concept in enumerate(concepts):
                if i % 20 == 0:
                    logger.info(f"概念板块进度: {i}/{len(concepts)}")
                
                df = self._retry_fetch(
                    self.fetch_concept_history, 
                    concept, start_date, end_date
                )
                if df is not None and not df.empty:
                    all_data.append(df)
                time.sleep(0.3)  # 避免请求过快
        
        # 获取行业板块
        if sector_type in ["industry", "both"]:
            industries = self.fetch_industry_board_list()
            logger.info(f"开始获取 {len(industries)} 个行业板块历史数据...")
            
            for i, industry in enumerate(industries):
                if i % 20 == 0:
                    logger.info(f"行业板块进度: {i}/{len(industries)}")
                
                df = self._retry_fetch(
                    self.fetch_industry_history,
                    industry, start_date, end_date
                )
                if df is not None and not df.empty:
                    all_data.append(df)
                time.sleep(0.3)
        
        if all_data:
            result = pd.concat(all_data, ignore_index=True)
            logger.info(f"共获取 {len(result)} 条板块历史行情数据")
            return result
        
        return pd.DataFrame()
    
    def fetch_all_limit_up_history(self, start_date: str, end_date: str) -> pd.DataFrame:
        """
        获取日期范围内所有涨停池数据
        
        Args:
            start_date: 开始日期 YYYY-MM-DD
            end_date: 结束日期 YYYY-MM-DD
            
        Returns:
            合并后的涨停池DataFrame
        """
        trading_dates = self.get_trading_dates(start_date, end_date)
        
        all_data = []
        logger.info(f"开始获取 {len(trading_dates)} 个交易日的涨停池数据...")
        
        for i, date in enumerate(trading_dates):
            if i % 50 == 0:
                logger.info(f"涨停池进度: {i}/{len(trading_dates)}")
            
            df = self._retry_fetch(self.fetch_limit_up_history, date)
            if df is not None and not df.empty:
                all_data.append(df)
            time.sleep(0.2)
        
        if all_data:
            result = pd.concat(all_data, ignore_index=True)
            logger.info(f"共获取 {len(result)} 条涨停池数据")
            return result
        
        return pd.DataFrame()
    
    def aggregate_limit_up_by_sector(self, df_limit_up: pd.DataFrame) -> pd.DataFrame:
        """
        按板块和日期聚合涨停数据
        
        Returns:
            每日每板块的涨停统计
        """
        if df_limit_up is None or df_limit_up.empty:
            return pd.DataFrame()
        
        # 按日期和行业聚合
        agg_result = df_limit_up.groupby(["date", "industry"]).agg({
            "stock_code": "count",           # 涨停家数
            "turnover": "sum",               # 总成交额
            "seal_amount": "sum",            # 总封板资金
            "continuous_limit_up": "max",    # 最大连板数
        }).reset_index()
        
        agg_result.columns = [
            "date", "sector_name", "limit_up_count", "total_turnover",
            "total_seal_amount", "max_continuous_limit_up"
        ]
        
        return agg_result
    
    def save_historical_data(self, sector_history: pd.DataFrame,
                              limit_up_history: pd.DataFrame,
                              start_date: str, end_date: str):
        """
        保存历史数据到本地
        """
        prefix = f"{start_date}_{end_date}"
        
        if not sector_history.empty:
            path = HISTORICAL_DATA_DIR / f"sector_history_{prefix}.parquet"
            sector_history.to_parquet(path, index=False)
            logger.info(f"板块历史数据已保存: {path}")
        
        if not limit_up_history.empty:
            path = HISTORICAL_DATA_DIR / f"limit_up_history_{prefix}.parquet"
            limit_up_history.to_parquet(path, index=False)
            logger.info(f"涨停池历史数据已保存: {path}")
    
    def load_historical_data(self, start_date: str = None, end_date: str = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        加载本地历史数据
        
        Returns:
            (sector_history, limit_up_history)
        """
        sector_files = list(HISTORICAL_DATA_DIR.glob("sector_history_*.parquet"))
        limit_up_files = list(HISTORICAL_DATA_DIR.glob("limit_up_history_*.parquet"))
        
        sector_data = []
        for f in sector_files:
            df = pd.read_parquet(f)
            sector_data.append(df)
        
        limit_up_data = []
        for f in limit_up_files:
            df = pd.read_parquet(f)
            limit_up_data.append(df)
        
        sector_history = pd.concat(sector_data, ignore_index=True) if sector_data else pd.DataFrame()
        limit_up_history = pd.concat(limit_up_data, ignore_index=True) if limit_up_data else pd.DataFrame()
        
        # 按日期过滤
        if start_date and not sector_history.empty:
            sector_history = sector_history[sector_history['date'] >= start_date]
        if end_date and not sector_history.empty:
            sector_history = sector_history[sector_history['date'] <= end_date]
        
        if start_date and not limit_up_history.empty:
            limit_up_history = limit_up_history[limit_up_history['date'] >= start_date.replace('-', '')]
        if end_date and not limit_up_history.empty:
            limit_up_history = limit_up_history[limit_up_history['date'] <= end_date.replace('-', '')]
        
        logger.info(f"加载板块数据: {len(sector_history)} 条, 涨停数据: {len(limit_up_history)} 条")
        return sector_history, limit_up_history


def download_historical_data(start_year: int = 2022, end_year: int = 2025):
    """
    下载指定年份范围的历史数据
    
    Usage:
        python -m backtest.historical_data
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )
    
    fetcher = HistoricalDataFetcher()
    
    start_date = f"{start_year}0101"
    end_date = f"{end_year}1231"
    
    print(f"\n{'='*60}")
    print(f"开始下载历史数据: {start_year} - {end_year}")
    print(f"{'='*60}\n")
    
    # 1. 获取板块历史行情
    print("📊 步骤1: 获取板块历史行情...")
    sector_history = fetcher.fetch_all_sector_history(start_date, end_date)
    print(f"   ✓ 获取到 {len(sector_history)} 条板块行情数据\n")
    
    # 2. 获取涨停池历史
    print("🔥 步骤2: 获取涨停池历史数据...")
    limit_up_history = fetcher.fetch_all_limit_up_history(
        f"{start_year}-01-01", f"{end_year}-12-31"
    )
    print(f"   ✓ 获取到 {len(limit_up_history)} 条涨停池数据\n")
    
    # 3. 保存数据
    print("💾 步骤3: 保存数据...")
    fetcher.save_historical_data(
        sector_history, limit_up_history, 
        str(start_year), str(end_year)
    )
    
    print(f"\n{'='*60}")
    print("✅ 历史数据下载完成!")
    print(f"{'='*60}")
    
    # 显示数据摘要
    if not sector_history.empty:
        print(f"\n📈 板块行情数据摘要:")
        print(f"   - 日期范围: {sector_history['date'].min()} ~ {sector_history['date'].max()}")
        print(f"   - 板块数量: {sector_history['sector_name'].nunique()}")
        print(f"   - 数据条数: {len(sector_history)}")
    
    if not limit_up_history.empty:
        print(f"\n🔥 涨停池数据摘要:")
        print(f"   - 日期范围: {limit_up_history['date'].min()} ~ {limit_up_history['date'].max()}")
        print(f"   - 涨停股票数: {limit_up_history['stock_code'].nunique()}")
        print(f"   - 数据条数: {len(limit_up_history)}")


if __name__ == "__main__":
    download_historical_data(2022, 2025)
