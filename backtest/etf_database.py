"""
ETF预测系统 - ETF回测数据库模块
记录ETF预测结果与实际表现，用于评估模型准确性
"""
import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Tuple
import logging
from pathlib import Path
import json

from config.settings import DATA_DIR, REPORT_DIR

logger = logging.getLogger(__name__)


class ETFBacktestDatabase:
    """ETF回测数据库管理"""
    
    def __init__(self):
        self.db_path = DATA_DIR / "etf_backtest.db"
        self._init_database()
    
    def _init_database(self):
        """初始化ETF回测数据库表结构"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # ETF每日预测记录表
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS etf_daily_predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                predict_date TEXT NOT NULL,          -- 预测日期（T日发出预测）
                target_date TEXT NOT NULL,           -- 目标日期（T+1日实际验证）
                etf_code TEXT NOT NULL,
                etf_name TEXT NOT NULL,
                predict_rank INTEGER,                -- 预测排名
                predict_score REAL,                  -- 预测得分
                predict_reason TEXT,                 -- 预测理由
                -- 实际结果（次日收盘后填充）
                actual_change_pct REAL,              -- 实际涨跌幅
                actual_rank INTEGER,                 -- 实际涨幅排名
                is_hit INTEGER DEFAULT 0,            -- 是否命中（涨幅>0）
                is_top5 INTEGER DEFAULT 0,           -- 是否在实际Top5
                -- 元数据
                created_at TEXT,
                updated_at TEXT,
                UNIQUE(predict_date, etf_code)
            )
        """)
        
        # ETF每日汇总表
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS etf_daily_summary (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                predict_date TEXT UNIQUE NOT NULL,   -- 预测日期
                target_date TEXT,                    -- 目标日期
                -- 预测统计
                total_predictions INTEGER,           -- 预测数量
                -- 实际结果统计
                hit_count INTEGER DEFAULT 0,         -- 命中数（涨幅>0）
                hit_rate REAL DEFAULT 0,             -- 命中率
                top5_hit_count INTEGER DEFAULT 0,    -- Top5命中数
                top5_hit_rate REAL DEFAULT 0,        -- Top5命中率
                avg_return REAL DEFAULT 0,           -- 平均收益
                max_return REAL DEFAULT 0,           -- 最大收益
                min_return REAL DEFAULT 0,           -- 最小收益
                -- 基准对比
                benchmark_return REAL DEFAULT 0,     -- 基准收益（ETF池均值）
                excess_return REAL DEFAULT 0,        -- 超额收益
                -- 状态
                status TEXT DEFAULT 'pending',       -- pending/validated
                created_at TEXT,
                validated_at TEXT
            )
        """)
        
        # ETF累计绩效表
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS etf_performance_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                metric_date TEXT UNIQUE NOT NULL,
                -- 累计指标
                total_days INTEGER,                  -- 总交易天数
                cumulative_return REAL,              -- 累计收益率
                cumulative_hit_rate REAL,            -- 累计命中率
                sharpe_ratio REAL,                   -- 夏普比率
                max_drawdown REAL,                   -- 最大回撤
                win_rate REAL,                       -- 胜率
                -- 滚动指标
                rolling_7d_return REAL,              -- 7日滚动收益
                rolling_30d_return REAL,             -- 30日滚动收益
                rolling_7d_hit_rate REAL,            -- 7日命中率
                -- 更新时间
                updated_at TEXT
            )
        """)
        
        # ETF模型版本记录表
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS etf_model_versions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                version_date TEXT NOT NULL,
                model_type TEXT,
                train_samples INTEGER,
                valid_ndcg REAL,
                valid_mse REAL,
                train_start TEXT,
                train_end TEXT,
                features_used TEXT,                  -- JSON格式
                params TEXT,                         -- JSON格式
                notes TEXT,
                created_at TEXT
            )
        """)
        
        conn.commit()
        conn.close()
        logger.info(f"ETF回测数据库初始化完成: {self.db_path}")
    
    def record_predictions(self, predictions: pd.DataFrame, predict_date: str = None):
        """
        记录ETF每日预测结果
        """
        if predictions.empty:
            logger.warning("ETF预测结果为空，跳过记录")
            return
        
        if predict_date is None:
            predict_date = datetime.now().strftime("%Y-%m-%d")
        
        # 计算目标日期（下一个交易日）
        target_date = self._get_next_trading_day(predict_date)
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # 插入预测记录
        for _, row in predictions.iterrows():
            try:
                cursor.execute("""
                    INSERT OR REPLACE INTO etf_daily_predictions 
                    (predict_date, target_date, etf_code, etf_name, 
                     predict_rank, predict_score, predict_reason, created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    predict_date,
                    target_date,
                    row.get('etf_code', ''),
                    row.get('etf_name', ''),
                    row.get('rank', 0),
                    row.get('pred_score', 0),
                    row.get('prediction_reason', ''),
                    now,
                    now
                ))
            except Exception as e:
                logger.error(f"记录ETF预测失败: {e}")
        
        # 创建每日汇总记录
        cursor.execute("""
            INSERT OR REPLACE INTO etf_daily_summary 
            (predict_date, target_date, total_predictions, status, created_at)
            VALUES (?, ?, ?, 'pending', ?)
        """, (predict_date, target_date, len(predictions), now))
        
        conn.commit()
        conn.close()
        
        logger.info(f"已记录 {len(predictions)} 条ETF预测到回测数据库")
    
    def _get_next_trading_day(self, date_str: str) -> str:
        """获取下一个交易日（简单实现：跳过周末）"""
        date = datetime.strptime(date_str, "%Y-%m-%d")
        next_day = date + timedelta(days=1)
        
        # 跳过周末
        while next_day.weekday() >= 5:
            next_day += timedelta(days=1)
        
        return next_day.strftime("%Y-%m-%d")
    
    def validate_predictions(self, target_date: str, actual_data: pd.DataFrame):
        """
        验证ETF预测结果
        
        Args:
            target_date: 目标日期
            actual_data: 实际数据DataFrame，需包含 etf_code, change_pct
        """
        if actual_data.empty:
            logger.warning("ETF实际数据为空，跳过验证")
            return
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # 获取该日期的预测记录
        cursor.execute("""
            SELECT id, etf_code FROM etf_daily_predictions 
            WHERE target_date = ?
        """, (target_date,))
        predictions = cursor.fetchall()
        
        if not predictions:
            logger.warning(f"未找到 {target_date} 的ETF预测记录")
            conn.close()
            return
        
        # 计算实际排名
        actual_data = actual_data.copy()
        actual_data['actual_rank'] = actual_data['change_pct'].rank(ascending=False)
        
        # 更新每条预测的实际结果
        hit_count = 0
        top5_hit_count = 0
        returns = []
        
        for pred_id, etf_code in predictions:
            actual_row = actual_data[actual_data['etf_code'] == etf_code]
            
            if actual_row.empty:
                continue
            
            actual_change = actual_row['change_pct'].values[0]
            actual_rank = actual_row['actual_rank'].values[0]
            
            # 判断是否命中（涨幅>0）
            is_hit = 1 if actual_change > 0 else 0
            is_top5 = 1 if actual_rank <= 5 else 0
            
            cursor.execute("""
                UPDATE etf_daily_predictions 
                SET actual_change_pct = ?, actual_rank = ?, 
                    is_hit = ?, is_top5 = ?, updated_at = ?
                WHERE id = ?
            """, (actual_change, actual_rank, is_hit, is_top5, now, pred_id))
            
            hit_count += is_hit
            top5_hit_count += is_top5
            returns.append(actual_change)
        
        # 计算基准收益（ETF池平均）
        benchmark_return = actual_data['change_pct'].mean()
        
        # 更新每日汇总
        avg_return = np.mean(returns) if returns else 0
        max_return = np.max(returns) if returns else 0
        min_return = np.min(returns) if returns else 0
        hit_rate = hit_count / len(predictions) if predictions else 0
        top5_hit_rate = top5_hit_count / len(predictions) if predictions else 0
        excess_return = avg_return - benchmark_return
        
        # 找到对应的predict_date
        cursor.execute("""
            SELECT predict_date FROM etf_daily_predictions 
            WHERE target_date = ? LIMIT 1
        """, (target_date,))
        result = cursor.fetchone()
        predict_date = result[0] if result else None
        
        if predict_date:
            cursor.execute("""
                UPDATE etf_daily_summary 
                SET hit_count = ?, hit_rate = ?, 
                    top5_hit_count = ?, top5_hit_rate = ?,
                    avg_return = ?, max_return = ?, min_return = ?,
                    benchmark_return = ?, excess_return = ?,
                    status = 'validated', validated_at = ?
                WHERE predict_date = ?
            """, (hit_count, hit_rate, top5_hit_count, top5_hit_rate,
                  avg_return, max_return, min_return,
                  benchmark_return, excess_return, now, predict_date))
        
        conn.commit()
        conn.close()
        
        logger.info(f"已验证 {target_date} ETF预测: 命中率={hit_rate:.2%}, 平均收益={avg_return:.2f}%, 超额={excess_return:.2f}%")
    
    def get_performance_report(self, days: int = 30) -> Dict:
        """
        获取ETF绩效报告
        """
        conn = sqlite3.connect(self.db_path)
        
        # 获取已验证的汇总数据
        query = f"""
            SELECT * FROM etf_daily_summary 
            WHERE status = 'validated'
            AND predict_date >= date('now', '-{days} days')
            ORDER BY predict_date DESC
        """
        df = pd.read_sql(query, conn)
        conn.close()
        
        if df.empty:
            return {"status": "no_data", "message": "暂无已验证的ETF预测数据"}
        
        # 计算综合指标
        total_predictions = df['total_predictions'].sum()
        total_hits = df['hit_count'].sum()
        overall_hit_rate = total_hits / total_predictions if total_predictions > 0 else 0
        
        avg_daily_return = df['avg_return'].mean()
        total_return = df['avg_return'].sum()
        avg_excess_return = df['excess_return'].mean()
        
        # 计算分排名命中率
        total_top5_hits = df['top5_hit_count'].sum()
        top5_hit_rate = total_top5_hits / total_predictions if total_predictions > 0 else 0
        
        # 计算胜率（日收益>0的天数）
        win_days = (df['avg_return'] > 0).sum()
        win_rate = win_days / len(df) if len(df) > 0 else 0
        
        # 计算最大回撤
        cumulative = df['avg_return'].cumsum()
        max_drawdown = (cumulative.cummax() - cumulative).max()
        
        # 计算夏普比率（假设无风险收益率为3%年化）
        daily_std = df['avg_return'].std()
        if daily_std > 0:
            sharpe_ratio = (avg_daily_return - 0.03/252) / daily_std * np.sqrt(252)
        else:
            sharpe_ratio = 0
        
        return {
            "status": "success",
            "period": f"{df['predict_date'].min()} ~ {df['predict_date'].max()}",
            "total_days": len(df),
            "total_predictions": total_predictions,
            "total_hits": total_hits,
            "overall_hit_rate": overall_hit_rate,
            "top5_hit_rate": top5_hit_rate,
            "avg_daily_return": avg_daily_return,
            "total_return": total_return,
            "avg_excess_return": avg_excess_return,
            "win_rate": win_rate,
            "max_drawdown": max_drawdown,
            "sharpe_ratio": sharpe_ratio,
        }
    
    def get_prediction_history(self, days: int = 7) -> pd.DataFrame:
        """获取近期ETF预测历史"""
        conn = sqlite3.connect(self.db_path)
        
        query = f"""
            SELECT predict_date, etf_code, etf_name, predict_rank, 
                   predict_score, actual_change_pct, actual_rank, is_hit
            FROM etf_daily_predictions 
            WHERE predict_date >= date('now', '-{days} days')
            ORDER BY predict_date DESC, predict_rank
        """
        
        df = pd.read_sql(query, conn)
        conn.close()
        
        return df
    
    def export_report(self, days: int = 30) -> str:
        """导出ETF回测报告"""
        report = self.get_performance_report(days)
        history = self.get_prediction_history(days)
        
        # 生成Markdown报告
        lines = [
            "# 📊 ETF预测回测报告",
            "",
            f"**统计周期**: {report.get('period', 'N/A')}",
            f"**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "---",
            "",
            "## 📈 综合绩效指标",
            "",
            f"| 指标 | 数值 |",
            f"|------|------|",
            f"| 总交易天数 | {report.get('total_days', 0)} |",
            f"| 总预测次数 | {report.get('total_predictions', 0)} |",
            f"| 总命中次数 | {report.get('total_hits', 0)} |",
            f"| 整体命中率 | {report.get('overall_hit_rate', 0):.2%} |",
            f"| Top5命中率 | {report.get('top5_hit_rate', 0):.2%} |",
            f"| 平均日收益 | {report.get('avg_daily_return', 0):.2f}% |",
            f"| 累计收益 | {report.get('total_return', 0):.2f}% |",
            f"| 平均超额收益 | {report.get('avg_excess_return', 0):.2f}% |",
            f"| 胜率 | {report.get('win_rate', 0):.2%} |",
            f"| 最大回撤 | {report.get('max_drawdown', 0):.2f}% |",
            f"| 夏普比率 | {report.get('sharpe_ratio', 0):.2f} |",
            "",
        ]
        
        if not history.empty:
            lines.extend([
                "---",
                "",
                "## 📋 近期预测明细",
                "",
                "| 日期 | ETF代码 | ETF名称 | 预测排名 | 实际涨跌 | 命中 |",
                "|------|---------|---------|----------|----------|------|",
            ])
            
            for _, row in history.head(30).iterrows():
                hit_mark = "✅" if row.get('is_hit') == 1 else "❌"
                actual_change = row.get('actual_change_pct')
                actual_str = f"{actual_change:.2f}%" if pd.notna(actual_change) else "待验证"
                lines.append(
                    f"| {row['predict_date']} | {row['etf_code']} | {row['etf_name']} | "
                    f"{row['predict_rank']} | {actual_str} | {hit_mark} |"
                )
        
        lines.extend([
            "",
            "---",
            "",
            "*报告由 ETF预测系统 自动生成*",
        ])
        
        report_content = "\n".join(lines)
        
        # 保存报告
        report_path = REPORT_DIR / f"etf_backtest_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(report_content)
        
        logger.info(f"ETF回测报告已生成: {report_path}")
        return str(report_path)
    
    def record_model_version(self, train_info: Dict):
        """记录模型版本信息"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO etf_model_versions 
            (version_date, model_type, train_samples, valid_ndcg, valid_mse,
             train_start, train_end, features_used, params, notes, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            datetime.now().strftime("%Y-%m-%d"),
            train_info.get('model_type', 'LightGBM'),
            train_info.get('train_samples', 0),
            train_info.get('ndcg@5', 0),
            train_info.get('mse', 0),
            train_info.get('train_start', ''),
            train_info.get('train_end', ''),
            json.dumps(train_info.get('features', [])),
            json.dumps(train_info.get('params', {})),
            train_info.get('notes', ''),
            datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        ))
        
        conn.commit()
        conn.close()
        logger.info("ETF模型版本信息已记录")
