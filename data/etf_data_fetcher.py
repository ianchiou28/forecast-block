"""
A股ETF预测系统 - ETF数据获取模块
使用AkShare获取ETF行情、资金流向、持仓等数据
"""
import akshare as ak
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional, Dict, List
import logging
import time
from pathlib import Path

from config.settings import RAW_DATA_DIR, DATA_CONFIG

logger = logging.getLogger(__name__)


# ETF基础池 - 热门宽基+行业ETF
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


class ETFDataFetcher:
    """ETF数据获取器"""
    
    def __init__(self):
        self.retry_times = 3
        self.retry_delay = 2
        self.etf_pool = ETF_POOL
    
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
    
    def fetch_etf_realtime(self) -> Optional[pd.DataFrame]:
        """
        获取ETF实时行情数据
        """
        logger.info("获取ETF实时行情...")
        
        all_data = []
        for etf_code, etf_name in self.etf_pool.items():
            try:
                # 尝试获取实时行情
                df = self._retry_fetch(ak.fund_etf_spot_em)
                if df is not None:
                    # 筛选指定ETF
                    df_etf = df[df["代码"] == etf_code]
                    if not df_etf.empty:
                        all_data.append(df_etf)
            except Exception as e:
                logger.warning(f"获取 {etf_name} 实时行情失败: {e}")
        
        if all_data:
            result = pd.concat(all_data, ignore_index=True)
            result = self._process_realtime_data(result)
            logger.info(f"成功获取 {len(result)} 只ETF实时数据")
            return result
        
        # 如果实时行情获取失败，尝试获取所有ETF行情
        logger.info("尝试获取ETF全市场行情...")
        df = self._retry_fetch(ak.fund_etf_spot_em)
        if df is not None:
            # 筛选池内ETF
            df = df[df["代码"].isin(self.etf_pool.keys())]
            df = self._process_realtime_data(df)
            logger.info(f"成功获取 {len(df)} 只ETF实时数据")
            return df
        
        return None
    
    def _process_realtime_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """处理实时行情数据"""
        rename_map = {
            "代码": "etf_code",
            "名称": "etf_name",
            "最新价": "close",
            "涨跌幅": "change_pct",
            "涨跌额": "change_amount",
            "成交量": "volume",
            "成交额": "turnover",
            "开盘价": "open",
            "最高价": "high",
            "最低价": "low",
            "昨收": "pre_close",
            "换手率": "turnover_rate",
        }
        existing_rename = {k: v for k, v in rename_map.items() if k in df.columns}
        df = df.rename(columns=existing_rename)
        
        df["fetch_date"] = datetime.now().strftime("%Y-%m-%d")
        df["fetch_time"] = datetime.now().strftime("%H:%M:%S")
        
        return df
    
    def fetch_etf_history(self, etf_code: str, days: int = 60) -> Optional[pd.DataFrame]:
        """
        获取单只ETF历史数据
        """
        try:
            # 获取历史行情
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
                df = df.tail(days)
                return df
        except Exception as e:
            logger.warning(f"获取ETF {etf_code} 历史数据失败: {e}")
        
        return None
    
    def fetch_all_etf_history(self, days: int = 60) -> pd.DataFrame:
        """
        获取所有ETF池历史数据
        """
        logger.info(f"获取ETF池历史数据 (近{days}天)...")
        
        all_data = []
        for etf_code in self.etf_pool.keys():
            df = self.fetch_etf_history(etf_code, days)
            if df is not None:
                all_data.append(df)
            time.sleep(0.3)  # 避免请求过快
        
        if all_data:
            result = pd.concat(all_data, ignore_index=True)
            logger.info(f"成功获取 {len(self.etf_pool)} 只ETF的历史数据")
            return result
        
        return pd.DataFrame()
    
    def fetch_etf_fund_flow(self) -> Optional[pd.DataFrame]:
        """
        获取ETF资金流向数据
        """
        logger.info("获取ETF资金流向...")
        
        try:
            # 尝试获取ETF资金流向
            df = self._retry_fetch(ak.fund_etf_fund_flow_em)
            
            if df is not None:
                # 筛选池内ETF
                if "代码" in df.columns:
                    df = df[df["代码"].isin(self.etf_pool.keys())]
                elif "基金代码" in df.columns:
                    df = df[df["基金代码"].isin(self.etf_pool.keys())]
                
                df = self._process_fund_flow(df)
                logger.info(f"成功获取 {len(df)} 只ETF资金流向数据")
                return df
        except Exception as e:
            logger.warning(f"获取ETF资金流向失败: {e}")
        
        return None
    
    def _process_fund_flow(self, df: pd.DataFrame) -> pd.DataFrame:
        """处理资金流向数据"""
        rename_map = {
            "代码": "etf_code",
            "基金代码": "etf_code",
            "名称": "etf_name",
            "基金简称": "etf_name",
            "今日主力净流入-净额": "main_net_inflow",
            "今日主力净流入-净占比": "main_net_inflow_pct",
            "今日超大单净流入-净额": "super_large_net_inflow",
            "今日大单净流入-净额": "large_net_inflow",
            "今日中单净流入-净额": "medium_net_inflow",
            "今日小单净流入-净额": "small_net_inflow",
        }
        existing_rename = {k: v for k, v in rename_map.items() if k in df.columns}
        df = df.rename(columns=existing_rename)
        
        df["fetch_date"] = datetime.now().strftime("%Y-%m-%d")
        return df
    
    def fetch_all_daily_data(self) -> Dict[str, pd.DataFrame]:
        """
        获取所有ETF每日数据
        """
        data = {}
        
        # 实时行情
        realtime = self.fetch_etf_realtime()
        if realtime is not None:
            data["etf_realtime"] = realtime
        
        # 资金流向
        fund_flow = self.fetch_etf_fund_flow()
        if fund_flow is not None:
            data["etf_fund_flow"] = fund_flow
        
        return data
    
    def save_daily_data(self, data: Dict[str, pd.DataFrame], date: str = None):
        """保存每日ETF数据"""
        if date is None:
            date = datetime.now().strftime("%Y%m%d")
        
        save_dir = RAW_DATA_DIR / date
        save_dir.mkdir(parents=True, exist_ok=True)
        
        for name, df in data.items():
            if df is not None and not df.empty:
                path = save_dir / f"{name}.csv"
                df.to_csv(path, index=False, encoding="utf-8-sig")
                logger.info(f"保存ETF数据: {path}")
