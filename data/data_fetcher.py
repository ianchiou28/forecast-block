"""
A股板块涨停预测系统 - 数据获取模块
使用AkShare获取资金流向、涨停池、北向资金等数据
"""
import akshare as ak
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional, Dict, List
import logging
import time

from config.settings import RAW_DATA_DIR, DATA_CONFIG

logger = logging.getLogger(__name__)


class SectorDataFetcher:
    """板块数据获取器"""
    
    def __init__(self):
        self.retry_times = 3
        self.retry_delay = 2  # 秒
    
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
    
    def fetch_concept_money_flow(self) -> Optional[pd.DataFrame]:
        """
        获取概念板块资金流向（实时）
        核心特征：衡量主力资金意图
        """
        logger.info("获取概念板块资金流向...")
        df = self._retry_fetch(ak.stock_fund_flow_concept, symbol="即时")
        
        if df is not None:
            # AkShare返回的列名可能有变化，需要灵活处理
            rename_map = {
                "序号": "rank",
                "行业": "sector_name",
                "公司家数": "stock_count",
                "今日涨跌幅": "change_pct",
                "行业-涨跌幅": "change_pct",
                "净额": "main_net_inflow",  # 简化版列名
                "流入资金": "inflow",
                "流出资金": "outflow",
                "今日主力净流入-净额": "main_net_inflow",
                "今日主力净流入-净占比": "main_net_inflow_pct",
                "今日超大单净流入-净额": "super_large_net_inflow",
                "今日超大单净流入-净占比": "super_large_net_inflow_pct",
                "今日大单净流入-净额": "large_net_inflow",
                "今日大单净流入-净占比": "large_net_inflow_pct",
                "今日中单净流入-净额": "medium_net_inflow",
                "今日中单净流入-净占比": "medium_net_inflow_pct",
                "今日小单净流入-净额": "small_net_inflow",
                "今日小单净流入-净占比": "small_net_inflow_pct",
            }
            # 只重命名存在的列
            existing_rename = {k: v for k, v in rename_map.items() if k in df.columns}
            df = df.rename(columns=existing_rename)
            
            # 如果没有main_net_inflow但有净额，使用净额
            if "main_net_inflow" not in df.columns and "净额" in df.columns:
                df["main_net_inflow"] = df["净额"]
            
            # 转换资金单位（亿元 -> 元，方便后续处理）
            if "main_net_inflow" in df.columns:
                df["main_net_inflow"] = df["main_net_inflow"] * 1e8  # 亿转元
            
            df["fetch_date"] = datetime.now().strftime("%Y-%m-%d")
            df["fetch_time"] = datetime.now().strftime("%H:%M:%S")
            logger.info(f"成功获取 {len(df)} 个概念板块资金流向数据")
        return df
    
    def fetch_industry_money_flow(self) -> Optional[pd.DataFrame]:
        """
        获取行业板块资金流向（实时）
        """
        logger.info("获取行业板块资金流向...")
        df = self._retry_fetch(ak.stock_fund_flow_industry, symbol="即时")
        
        if df is not None:
            df = df.rename(columns={
                "序号": "rank",
                "行业": "sector_name",
                "公司家数": "stock_count",
                "今日涨跌幅": "change_pct",
                "今日主力净流入-净额": "main_net_inflow",
                "今日主力净流入-净占比": "main_net_inflow_pct",
                "今日超大单净流入-净额": "super_large_net_inflow",
                "今日超大单净流入-净占比": "super_large_net_inflow_pct",
                "今日大单净流入-净额": "large_net_inflow",
                "今日大单净流入-净占比": "large_net_inflow_pct",
                "今日中单净流入-净额": "medium_net_inflow",
                "今日中单净流入-净占比": "medium_net_inflow_pct",
                "今日小单净流入-净额": "small_net_inflow",
                "今日小单净流入-净占比": "small_net_inflow_pct",
            })
            df["sector_type"] = "industry"
            df["fetch_date"] = datetime.now().strftime("%Y-%m-%d")
            logger.info(f"成功获取 {len(df)} 个行业板块资金流向数据")
        return df
    
    def fetch_limit_up_pool(self, date: Optional[str] = None) -> Optional[pd.DataFrame]:
        """
        获取涨停池数据
        核心标签：计算板块内的涨停密度
        """
        if date is None:
            date = datetime.now().strftime("%Y%m%d")
        
        logger.info(f"获取 {date} 涨停池数据...")
        df = self._retry_fetch(ak.stock_zt_pool_em, date=date)
        
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
            df["fetch_date"] = date
            logger.info(f"成功获取 {len(df)} 只涨停股票数据")
        return df
    
    def fetch_northbound_flow(self) -> Optional[pd.DataFrame]:
        """
        获取北向资金板块排行
        辅助特征：捕捉外资动向
        """
        logger.info("获取北向资金板块排行...")
        df = self._retry_fetch(ak.stock_hsgt_board_rank_em, symbol="北向资金增持行业板块排行")
        
        if df is not None:
            df["fetch_date"] = datetime.now().strftime("%Y-%m-%d")
            df["direction"] = "northbound"
            logger.info(f"成功获取 {len(df)} 个北向资金板块数据")
        return df
    
    def fetch_sector_history(self, sector_name: str, period: str = "daily",
                             start_date: str = None, end_date: str = None) -> Optional[pd.DataFrame]:
        """
        获取板块历史行情
        基础特征：计算板块动量与趋势
        """
        logger.info(f"获取板块 {sector_name} 历史行情...")
        
        try:
            df = ak.stock_board_concept_hist_em(
                symbol=sector_name,
                period=period,
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
                df["sector_name"] = sector_name
                logger.info(f"成功获取 {sector_name} 板块 {len(df)} 条历史数据")
            return df
        except Exception as e:
            logger.warning(f"获取板块 {sector_name} 历史行情失败: {e}")
            return None
    
    def fetch_dragon_tiger_list(self, date: Optional[str] = None) -> Optional[pd.DataFrame]:
        """
        获取龙虎榜数据
        透视游资动向的关键数据
        """
        if date is None:
            date = datetime.now().strftime("%Y%m%d")
        
        logger.info(f"获取 {date} 龙虎榜数据...")
        df = self._retry_fetch(ak.stock_lhb_detail_em, date=date)
        
        if df is not None and not df.empty:
            df["fetch_date"] = date
            logger.info(f"成功获取 {len(df)} 条龙虎榜数据")
        return df
    
    def fetch_all_daily_data(self, date: Optional[str] = None) -> Dict[str, pd.DataFrame]:
        """
        获取所有每日数据
        """
        result = {}
        
        # 概念资金流
        result["concept_money_flow"] = self.fetch_concept_money_flow()
        
        # 行业资金流
        result["industry_money_flow"] = self.fetch_industry_money_flow()
        
        # 涨停池
        result["limit_up_pool"] = self.fetch_limit_up_pool(date)
        
        # 北向资金
        result["northbound_flow"] = self.fetch_northbound_flow()
        
        # 龙虎榜
        result["dragon_tiger"] = self.fetch_dragon_tiger_list(date)
        
        return result
    
    def save_daily_data(self, data: Dict[str, pd.DataFrame], date: str = None):
        """
        保存每日数据到CSV
        """
        if date is None:
            date = datetime.now().strftime("%Y%m%d")
        
        save_dir = RAW_DATA_DIR / date
        save_dir.mkdir(parents=True, exist_ok=True)
        
        for name, df in data.items():
            if df is not None and not df.empty:
                file_path = save_dir / f"{name}.csv"
                df.to_csv(file_path, index=False, encoding="utf-8-sig")
                logger.info(f"保存 {name} 数据到 {file_path}")


def aggregate_limit_up_by_sector(df_limit_up: pd.DataFrame) -> pd.DataFrame:
    """
    按板块聚合涨停数据
    """
    if df_limit_up is None or df_limit_up.empty:
        return pd.DataFrame()
    
    # 按行业聚合
    agg_result = df_limit_up.groupby("industry").agg({
        "stock_code": "count",  # 涨停家数
        "turnover": "sum",       # 总成交额
        "seal_amount": "sum",    # 总封板资金
        "continuous_limit_up": "max",  # 最大连板数
    }).reset_index()
    
    agg_result.columns = [
        "sector_name", "limit_up_count", "total_turnover",
        "total_seal_amount", "max_continuous_limit_up"
    ]
    
    return agg_result


if __name__ == "__main__":
    # 测试数据获取
    logging.basicConfig(level=logging.INFO)
    fetcher = SectorDataFetcher()
    
    # 获取所有数据
    data = fetcher.fetch_all_daily_data()
    
    # 打印结果
    for name, df in data.items():
        if df is not None:
            print(f"\n{name}: {len(df)} rows")
            print(df.head())
