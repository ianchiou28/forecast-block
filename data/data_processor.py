"""
A股板块涨停预测系统 - 数据处理与特征工程模块
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Tuple
import logging
import sqlite3
from pathlib import Path

from config.settings import (
    RAW_DATA_DIR, PROCESSED_DATA_DIR, DATABASE_CONFIG,
    FACTOR_CONFIG, DATA_CONFIG
)

logger = logging.getLogger(__name__)


class SectorDataProcessor:
    """板块数据处理器"""
    
    def __init__(self):
        self.db_path = DATABASE_CONFIG["sqlite_path"]
        self._init_database()
        self.sector_mapping = self._load_or_create_mapping()
    
    def _init_database(self):
        """初始化数据库表结构"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # 板块映射表
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS sector_mapping (
                sector_id TEXT PRIMARY KEY,
                sector_name TEXT UNIQUE,
                sector_type TEXT,
                created_date TEXT
            )
        """)
        
        # 每日板块数据表
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS daily_sector_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                date TEXT,
                sector_id TEXT,
                sector_name TEXT,
                -- 基础行情数据
                open REAL,
                high REAL,
                low REAL,
                close REAL,
                volume REAL,
                turnover REAL,
                change_pct REAL,
                -- 资金流数据
                main_net_inflow REAL,
                main_net_inflow_pct REAL,
                super_large_net_inflow REAL,
                large_net_inflow REAL,
                medium_net_inflow REAL,
                small_net_inflow REAL,
                -- 涨停数据
                limit_up_count INTEGER,
                total_seal_amount REAL,
                max_continuous_limit_up INTEGER,
                -- 北向资金数据
                northbound_net_inflow REAL,
                -- 时间戳
                created_at TEXT,
                UNIQUE(date, sector_id)
            )
        """)
        
        # 特征表（用于模型训练）
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS sector_features (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                date TEXT,
                sector_id TEXT,
                sector_name TEXT,
                -- 特征列（动态生成）
                features TEXT,  -- JSON格式存储
                -- 标签
                label_limit_up_count INTEGER,
                label_change_pct REAL,
                -- 综合标签分数
                label_score REAL,
                created_at TEXT,
                UNIQUE(date, sector_id)
            )
        """)
        
        conn.commit()
        conn.close()
        logger.info("数据库初始化完成")
    
    def _load_or_create_mapping(self) -> Dict[str, str]:
        """加载或创建板块ID映射"""
        conn = sqlite3.connect(self.db_path)
        df = pd.read_sql("SELECT * FROM sector_mapping", conn)
        conn.close()
        
        if df.empty:
            return {}
        return dict(zip(df["sector_name"], df["sector_id"]))
    
    def _get_or_create_sector_id(self, sector_name: str, sector_type: str = "concept") -> str:
        """获取或创建板块ID"""
        if sector_name in self.sector_mapping:
            return self.sector_mapping[sector_name]
        
        # 创建新ID
        sector_id = f"BK{len(self.sector_mapping) + 1:04d}"
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute(
            "INSERT OR IGNORE INTO sector_mapping (sector_id, sector_name, sector_type, created_date) VALUES (?, ?, ?, ?)",
            (sector_id, sector_name, sector_type, datetime.now().strftime("%Y-%m-%d"))
        )
        conn.commit()
        conn.close()
        
        self.sector_mapping[sector_name] = sector_id
        logger.info(f"创建新板块映射: {sector_name} -> {sector_id}")
        return sector_id
    
    def process_daily_data(self, data: Dict[str, pd.DataFrame], date: str = None) -> pd.DataFrame:
        """
        处理每日数据，合并各数据源
        """
        if date is None:
            date = datetime.now().strftime("%Y-%m-%d")
        
        # 1. 处理概念资金流数据
        df_flow = data.get("concept_money_flow")
        if df_flow is None or df_flow.empty:
            logger.warning("概念资金流数据为空")
            return pd.DataFrame()
        
        # 2. 处理涨停池数据
        df_limit_up = data.get("limit_up_pool")
        limit_up_agg = self._aggregate_limit_up(df_limit_up)
        
        # 3. 合并数据
        result = df_flow.copy()
        
        # 为每个板块分配ID
        result["sector_id"] = result["sector_name"].apply(
            lambda x: self._get_or_create_sector_id(x)
        )
        
        # 合并涨停数据
        if not limit_up_agg.empty:
            result = result.merge(
                limit_up_agg,
                on="sector_name",
                how="left"
            )
        else:
            result["limit_up_count"] = 0
            result["total_seal_amount"] = 0
            result["max_continuous_limit_up"] = 0
        
        # 填充空值
        result = result.fillna(0)
        result["date"] = date
        result["created_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        return result
    
    def _aggregate_limit_up(self, df_limit_up: pd.DataFrame) -> pd.DataFrame:
        """按板块聚合涨停数据"""
        if df_limit_up is None or df_limit_up.empty:
            return pd.DataFrame()
        
        # 处理可能的列名问题
        if "industry" not in df_limit_up.columns:
            if "所属行业" in df_limit_up.columns:
                df_limit_up = df_limit_up.rename(columns={"所属行业": "industry"})
            else:
                return pd.DataFrame()
        
        agg_result = df_limit_up.groupby("industry").agg({
            "stock_code": "count" if "stock_code" in df_limit_up.columns else "size",
        }).reset_index()
        
        agg_result.columns = ["sector_name", "limit_up_count"]
        
        # 添加更多聚合指标
        if "seal_amount" in df_limit_up.columns:
            seal_agg = df_limit_up.groupby("industry")["seal_amount"].sum().reset_index()
            seal_agg.columns = ["sector_name", "total_seal_amount"]
            agg_result = agg_result.merge(seal_agg, on="sector_name", how="left")
        else:
            agg_result["total_seal_amount"] = 0
        
        if "continuous_limit_up" in df_limit_up.columns:
            cont_agg = df_limit_up.groupby("industry")["continuous_limit_up"].max().reset_index()
            cont_agg.columns = ["sector_name", "max_continuous_limit_up"]
            agg_result = agg_result.merge(cont_agg, on="sector_name", how="left")
        else:
            agg_result["max_continuous_limit_up"] = 0
        
        return agg_result
    
    def save_to_database(self, df: pd.DataFrame):
        """保存处理后的数据到数据库"""
        if df.empty:
            return
        
        conn = sqlite3.connect(self.db_path)
        
        # 选择需要保存的列
        columns = [
            "date", "sector_id", "sector_name", "change_pct",
            "main_net_inflow", "main_net_inflow_pct",
            "super_large_net_inflow", "large_net_inflow",
            "medium_net_inflow", "small_net_inflow",
            "limit_up_count", "total_seal_amount", "max_continuous_limit_up",
            "created_at"
        ]
        
        # 确保所有列都存在
        for col in columns:
            if col not in df.columns:
                df[col] = 0
        
        df_save = df[columns].copy()
        
        # 使用 INSERT OR REPLACE 避免重复
        for _, row in df_save.iterrows():
            try:
                conn.execute("""
                    INSERT OR REPLACE INTO daily_sector_data 
                    (date, sector_id, sector_name, change_pct,
                     main_net_inflow, main_net_inflow_pct,
                     super_large_net_inflow, large_net_inflow,
                     medium_net_inflow, small_net_inflow,
                     limit_up_count, total_seal_amount, max_continuous_limit_up,
                     created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, tuple(row))
            except Exception as e:
                logger.error(f"保存数据失败: {e}")
        
        conn.commit()
        conn.close()
        logger.info(f"成功保存 {len(df_save)} 条板块数据到数据库")
    
    def load_history_data(self, days: int = 60) -> pd.DataFrame:
        """加载历史数据"""
        conn = sqlite3.connect(self.db_path)
        
        end_date = datetime.now().strftime("%Y-%m-%d")
        start_date = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
        
        query = f"""
            SELECT * FROM daily_sector_data 
            WHERE date >= '{start_date}' AND date <= '{end_date}'
            ORDER BY date, sector_id
        """
        
        df = pd.read_sql(query, conn)
        conn.close()
        
        return df


class FeatureEngineer:
    """特征工程类"""
    
    def __init__(self):
        self.factor_config = FACTOR_CONFIG
    
    def compute_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        计算所有特征
        """
        if df.empty:
            return df
        
        # 确保按板块和日期排序
        df = df.sort_values(["sector_id", "date"]).reset_index(drop=True)
        
        features_list = []
        
        for sector_id in df["sector_id"].unique():
            sector_df = df[df["sector_id"] == sector_id].copy()
            
            if len(sector_df) < 5:  # 数据不足
                continue
            
            # 计算各类因子
            sector_df = self._compute_money_flow_factors(sector_df)
            sector_df = self._compute_limit_up_factors(sector_df)
            sector_df = self._compute_momentum_factors(sector_df)
            sector_df = self._compute_divergence_factors(sector_df)
            
            features_list.append(sector_df)
        
        if not features_list:
            return pd.DataFrame()
        
        result = pd.concat(features_list, ignore_index=True)
        
        # 计算标签（T+1涨停数）
        result = self._compute_labels(result)
        
        return result
    
    def _compute_money_flow_factors(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算资金流因子"""
        windows = self.factor_config["money_flow_windows"]
        
        for w in windows:
            # 资金流均值
            df[f"money_flow_ma_{w}"] = df["main_net_inflow"].rolling(w).mean()
            
            # 资金流标准差
            df[f"money_flow_std_{w}"] = df["main_net_inflow"].rolling(w).std()
            
            # 资金流动量（当前/过去均值）
            df[f"money_flow_momentum_{w}"] = df["main_net_inflow"] / (df[f"money_flow_ma_{w}"] + 1e-8)
            
            # 资金流累积
            df[f"money_flow_sum_{w}"] = df["main_net_inflow"].rolling(w).sum()
            
            # 资金蓄力因子（均值/标准差）
            df[f"money_accumulation_{w}"] = df[f"money_flow_ma_{w}"] / (df[f"money_flow_std_{w}"] + 1e-8)
        
        return df
    
    def _compute_limit_up_factors(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算涨停相关因子"""
        windows = self.factor_config["limit_up_windows"]
        
        for w in windows:
            # 涨停数均值（涨停惯性）
            df[f"limit_up_ma_{w}"] = df["limit_up_count"].rolling(w).mean()
            
            # 涨停数变化率
            df[f"limit_up_change_{w}"] = df["limit_up_count"].diff(w)
            
            # 涨停数最大值
            df[f"limit_up_max_{w}"] = df["limit_up_count"].rolling(w).max()
        
        # 是否有涨停
        df["has_limit_up"] = (df["limit_up_count"] > 0).astype(int)
        
        # 连续涨停天数
        df["consecutive_limit_up_days"] = df["has_limit_up"].rolling(5).sum()
        
        return df
    
    def _compute_momentum_factors(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算动量因子"""
        windows = self.factor_config["price_momentum_windows"]
        
        for w in windows:
            # 价格动量（涨跌幅累积）
            df[f"price_momentum_{w}"] = df["change_pct"].rolling(w).sum()
            
            # 涨跌幅均值
            df[f"price_change_ma_{w}"] = df["change_pct"].rolling(w).mean()
            
            # 波动率
            df[f"volatility_{w}"] = df["change_pct"].rolling(w).std()
        
        return df
    
    def _compute_divergence_factors(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算量价背离因子（核心因子）"""
        # 资金流排名
        df["money_flow_rank"] = df.groupby("date")["main_net_inflow"].rank(pct=True)
        
        # 涨幅排名
        df["change_rank"] = df.groupby("date")["change_pct"].rank(pct=True)
        
        # 量价背离因子：资金流入排名高但涨幅排名低
        df["divergence_factor"] = df["money_flow_rank"] - df["change_rank"]
        
        # 资金流入但价格未涨（吸筹信号）
        df["accumulation_signal"] = (
            (df["main_net_inflow"] > 0) & 
            (df["change_pct"] < df["change_pct"].rolling(5).mean())
        ).astype(int)
        
        # 5日量价相关性
        df["money_price_corr_5"] = df["main_net_inflow"].rolling(5).corr(df["change_pct"])
        
        return df
    
    def _compute_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算预测标签（T+1涨停数）"""
        df = df.sort_values(["sector_id", "date"]).reset_index(drop=True)
        
        # T+1涨停数
        df["label_limit_up_count"] = df.groupby("sector_id")["limit_up_count"].shift(-1)
        
        # T+1涨跌幅
        df["label_change_pct"] = df.groupby("sector_id")["change_pct"].shift(-1)
        
        # 综合标签分数（排名加权）
        df["label_limit_up_rank"] = df.groupby("date")["label_limit_up_count"].rank(pct=True)
        df["label_change_rank"] = df.groupby("date")["label_change_pct"].rank(pct=True)
        
        # 综合分数 = 0.7 * 涨幅排名 + 0.3 * 涨停排名
        df["label_score"] = 0.7 * df["label_change_rank"] + 0.3 * df["label_limit_up_rank"]
        
        return df
    
    def get_feature_columns(self) -> List[str]:
        """获取所有特征列名"""
        feature_cols = []
        
        # 资金流因子
        for w in self.factor_config["money_flow_windows"]:
            feature_cols.extend([
                f"money_flow_ma_{w}",
                f"money_flow_std_{w}",
                f"money_flow_momentum_{w}",
                f"money_flow_sum_{w}",
                f"money_accumulation_{w}",
            ])
        
        # 涨停因子
        for w in self.factor_config["limit_up_windows"]:
            feature_cols.extend([
                f"limit_up_ma_{w}",
                f"limit_up_change_{w}",
                f"limit_up_max_{w}",
            ])
        
        feature_cols.extend([
            "has_limit_up",
            "consecutive_limit_up_days",
        ])
        
        # 动量因子
        for w in self.factor_config["price_momentum_windows"]:
            feature_cols.extend([
                f"price_momentum_{w}",
                f"price_change_ma_{w}",
                f"volatility_{w}",
            ])
        
        # 背离因子
        feature_cols.extend([
            "money_flow_rank",
            "change_rank",
            "divergence_factor",
            "accumulation_signal",
            "money_price_corr_5",
        ])
        
        # 原始特征
        feature_cols.extend([
            "main_net_inflow",
            "main_net_inflow_pct",
            "change_pct",
            "limit_up_count",
        ])
        
        return feature_cols


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # 测试数据处理
    processor = SectorDataProcessor()
    
    # 加载历史数据
    df = processor.load_history_data(days=30)
    print(f"加载历史数据: {len(df)} 行")
    
    # 计算特征
    engineer = FeatureEngineer()
    df_features = engineer.compute_features(df)
    print(f"特征数据: {len(df_features)} 行")
    print(f"特征列: {engineer.get_feature_columns()}")
