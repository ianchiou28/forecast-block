"""
A股ETF预测系统 - ETF数据处理与特征工程模块
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Tuple
import logging
import sqlite3
from pathlib import Path

from config.settings import DATABASE_CONFIG, PROCESSED_DATA_DIR

logger = logging.getLogger(__name__)


class ETFDataProcessor:
    """ETF数据处理器"""
    
    def __init__(self):
        self.db_path = DATABASE_CONFIG["sqlite_path"]
        self._init_etf_tables()
    
    def _init_etf_tables(self):
        """初始化ETF相关数据库表"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # ETF每日数据表
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS daily_etf_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                date TEXT,
                etf_code TEXT,
                etf_name TEXT,
                open REAL,
                high REAL,
                low REAL,
                close REAL,
                volume REAL,
                turnover REAL,
                change_pct REAL,
                turnover_rate REAL,
                main_net_inflow REAL,
                main_net_inflow_pct REAL,
                created_at TEXT,
                UNIQUE(date, etf_code)
            )
        """)
        
        # ETF预测记录表
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS etf_predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                predict_date TEXT,
                etf_code TEXT,
                etf_name TEXT,
                pred_score REAL,
                rank INTEGER,
                prediction_reason TEXT,
                actual_change_pct REAL,
                validated INTEGER DEFAULT 0,
                created_at TEXT
            )
        """)
        
        conn.commit()
        conn.close()
        logger.info("ETF数据库表初始化完成")
    
    def process_daily_data(self, data: Dict[str, pd.DataFrame], date: str = None) -> pd.DataFrame:
        """
        处理每日ETF数据
        """
        if date is None:
            date = datetime.now().strftime("%Y-%m-%d")
        
        df_realtime = data.get("etf_realtime")
        df_fund_flow = data.get("etf_fund_flow")
        
        if df_realtime is None or df_realtime.empty:
            logger.warning("ETF实时数据为空")
            return pd.DataFrame()
        
        result = df_realtime.copy()
        
        # 合并资金流向数据
        if df_fund_flow is not None and not df_fund_flow.empty:
            merge_cols = ["main_net_inflow", "main_net_inflow_pct"]
            existing_cols = [c for c in merge_cols if c in df_fund_flow.columns]
            if existing_cols:
                result = result.merge(
                    df_fund_flow[["etf_code"] + existing_cols],
                    on="etf_code",
                    how="left"
                )
        
        result["date"] = date
        result["created_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        return result
    
    def save_to_database(self, df: pd.DataFrame):
        """保存ETF数据到数据库"""
        if df.empty:
            return
        
        conn = sqlite3.connect(self.db_path)
        
        # 准备数据
        columns = [
            "date", "etf_code", "etf_name", "open", "high", "low", "close",
            "volume", "turnover", "change_pct", "turnover_rate",
            "main_net_inflow", "main_net_inflow_pct", "created_at"
        ]
        
        existing_cols = [c for c in columns if c in df.columns]
        df_save = df[existing_cols].copy()
        
        # 插入或更新
        for _, row in df_save.iterrows():
            placeholders = ", ".join(["?"] * len(existing_cols))
            cols_str = ", ".join(existing_cols)
            
            conn.execute(
                f"INSERT OR REPLACE INTO daily_etf_data ({cols_str}) VALUES ({placeholders})",
                tuple(row[existing_cols])
            )
        
        conn.commit()
        conn.close()
        logger.info(f"保存 {len(df)} 条ETF数据到数据库")
    
    def load_history_data(self, days: int = 60) -> pd.DataFrame:
        """加载历史ETF数据"""
        conn = sqlite3.connect(self.db_path)
        
        query = f"""
            SELECT * FROM daily_etf_data
            WHERE date >= date('now', '-{days} days')
            ORDER BY date, etf_code
        """
        
        df = pd.read_sql(query, conn)
        conn.close()
        
        if df.empty:
            logger.warning("无历史ETF数据")
        else:
            logger.info(f"加载 {len(df)} 条历史ETF数据")
        
        return df


class ETFFeatureEngineer:
    """ETF特征工程"""
    
    def __init__(self):
        self.feature_windows = [3, 5, 10, 20]
    
    def compute_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        计算ETF特征
        """
        if df.empty or len(df) < 5:
            logger.warning("ETF数据不足，无法计算特征")
            return pd.DataFrame()
        
        # 确保日期排序
        df = df.sort_values(["etf_code", "date"]).reset_index(drop=True)
        
        features_list = []
        
        for etf_code in df["etf_code"].unique():
            etf_df = df[df["etf_code"] == etf_code].copy()
            
            if len(etf_df) < 5:
                continue
            
            # 计算各种特征
            features = self._compute_single_etf_features(etf_df)
            features_list.append(features)
        
        if not features_list:
            return pd.DataFrame()
        
        result = pd.concat(features_list, ignore_index=True)
        
        # 计算标签（次日涨跌幅）
        result = self._compute_labels(result)
        
        logger.info(f"计算完成 {len(result)} 条ETF特征数据")
        return result
    
    def _compute_single_etf_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算单只ETF的特征"""
        result = df.copy()
        
        # 1. 价格动量特征
        for window in self.feature_windows:
            result[f"return_{window}d"] = result["close"].pct_change(window) * 100
            result[f"volatility_{window}d"] = result["close"].pct_change().rolling(window).std() * 100
        
        # 2. 成交量特征
        result["volume_ma5"] = result["volume"].rolling(5).mean()
        result["volume_ratio"] = result["volume"] / result["volume_ma5"]
        
        result["turnover_ma5"] = result["turnover"].rolling(5).mean()
        result["turnover_ratio"] = result["turnover"] / result["turnover_ma5"]
        
        # 3. 价格位置特征
        result["high_20d"] = result["high"].rolling(20).max()
        result["low_20d"] = result["low"].rolling(20).min()
        result["price_position"] = (result["close"] - result["low_20d"]) / (result["high_20d"] - result["low_20d"] + 1e-8)
        
        # 4. 均线特征
        result["ma5"] = result["close"].rolling(5).mean()
        result["ma10"] = result["close"].rolling(10).mean()
        result["ma20"] = result["close"].rolling(20).mean()
        
        result["ma5_bias"] = (result["close"] - result["ma5"]) / result["ma5"] * 100
        result["ma10_bias"] = (result["close"] - result["ma10"]) / result["ma10"] * 100
        result["ma20_bias"] = (result["close"] - result["ma20"]) / result["ma20"] * 100
        
        # 5. 资金流特征
        if "main_net_inflow" in result.columns:
            result["money_flow_ma3"] = result["main_net_inflow"].rolling(3).mean()
            result["money_flow_ma5"] = result["main_net_inflow"].rolling(5).mean()
            result["money_flow_momentum"] = result["main_net_inflow"] - result["money_flow_ma5"]
        
        # 6. 技术指标
        result["rsi_14"] = self._compute_rsi(result["close"], 14)
        
        # 7. 波动率特征
        result["atr_14"] = self._compute_atr(result, 14)
        
        return result
    
    def _compute_rsi(self, prices: pd.Series, window: int = 14) -> pd.Series:
        """计算RSI"""
        delta = prices.diff()
        gain = delta.where(delta > 0, 0).rolling(window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window).mean()
        rs = gain / (loss + 1e-8)
        return 100 - (100 / (1 + rs))
    
    def _compute_atr(self, df: pd.DataFrame, window: int = 14) -> pd.Series:
        """计算ATR"""
        high = df["high"]
        low = df["low"]
        close = df["close"]
        
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        return tr.rolling(window).mean()
    
    def _compute_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算标签（次日涨跌幅）"""
        df = df.sort_values(["etf_code", "date"]).reset_index(drop=True)
        
        # 计算次日涨跌幅作为标签
        df["label_next_return"] = df.groupby("etf_code")["change_pct"].shift(-1)
        
        # 计算标签分数（用于排名学习）
        # 将涨跌幅转换为0-1分数
        df["label_score"] = df.groupby("date")["label_next_return"].transform(
            lambda x: (x.rank() - 1) / (len(x) - 1) if len(x) > 1 else 0.5
        )
        
        return df
    
    def get_feature_columns(self) -> List[str]:
        """获取特征列名"""
        features = []
        
        # 动量特征
        for window in self.feature_windows:
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
