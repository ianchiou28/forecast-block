"""
A股板块涨停预测系统 - 回测数据库模块
记录每日预测结果与实际表现，用于评估模型准确性
"""
import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Tuple
import logging
from pathlib import Path
import json

from config.settings import DATA_DIR, DATABASE_CONFIG

logger = logging.getLogger(__name__)


class BacktestDatabase:
    """回测数据库管理"""
    
    def __init__(self):
        self.db_path = DATA_DIR / "backtest.db"
        self._init_database()
    
    def _init_database(self):
        """初始化回测数据库表结构"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # 每日预测记录表
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS daily_predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                predict_date TEXT NOT NULL,          -- 预测日期（T日发出预测）
                target_date TEXT NOT NULL,           -- 目标日期（T+1日实际验证）
                sector_id TEXT NOT NULL,
                sector_name TEXT NOT NULL,
                predict_rank INTEGER,                -- 预测排名
                predict_score REAL,                  -- 预测得分
                predict_reason TEXT,                 -- 预测理由
                -- 实际结果（次日收盘后填充）
                actual_change_pct REAL,              -- 实际涨跌幅
                actual_limit_up_count INTEGER,       -- 实际涨停家数
                actual_rank INTEGER,                 -- 实际涨幅排名
                is_hit INTEGER DEFAULT 0,            -- 是否命中（涨幅>3%或有涨停）
                -- 元数据
                created_at TEXT,
                updated_at TEXT,
                UNIQUE(predict_date, sector_id)
            )
        """)
        
        # 每日汇总表
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS daily_summary (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                predict_date TEXT UNIQUE NOT NULL,   -- 预测日期
                target_date TEXT,                    -- 目标日期
                -- 预测统计
                total_predictions INTEGER,           -- 预测数量
                -- 实际结果统计
                hit_count INTEGER DEFAULT 0,         -- 命中数
                hit_rate REAL DEFAULT 0,             -- 命中率
                avg_return REAL DEFAULT 0,           -- 平均收益
                max_return REAL DEFAULT 0,           -- 最大收益
                min_return REAL DEFAULT 0,           -- 最小收益
                total_limit_up INTEGER DEFAULT 0,    -- 总涨停数
                -- 基准对比
                benchmark_return REAL DEFAULT 0,     -- 基准收益（全市场均值）
                excess_return REAL DEFAULT 0,        -- 超额收益
                -- 状态
                status TEXT DEFAULT 'pending',       -- pending/validated
                created_at TEXT,
                validated_at TEXT
            )
        """)
        
        # 累计绩效表
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS performance_metrics (
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
        
        # 模型版本记录表
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS model_versions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                version_date TEXT NOT NULL,
                model_type TEXT,
                train_samples INTEGER,
                valid_ndcg REAL,
                features_used TEXT,                  -- JSON格式
                notes TEXT,
                created_at TEXT
            )
        """)
        
        conn.commit()
        conn.close()
        logger.info(f"回测数据库初始化完成: {self.db_path}")
    
    def record_predictions(self, predictions: pd.DataFrame, predict_date: str = None):
        """
        记录每日预测结果
        
        Args:
            predictions: 预测结果DataFrame
            predict_date: 预测日期（默认今天）
        """
        if predictions.empty:
            logger.warning("预测结果为空，跳过记录")
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
                    INSERT OR REPLACE INTO daily_predictions 
                    (predict_date, target_date, sector_id, sector_name, 
                     predict_rank, predict_score, predict_reason, created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    predict_date,
                    target_date,
                    row.get('sector_id', ''),
                    row.get('sector_name', ''),
                    row.get('rank', 0),
                    row.get('pred_score', 0),
                    row.get('prediction_reason', ''),
                    now,
                    now
                ))
            except Exception as e:
                logger.error(f"记录预测失败: {e}")
        
        # 创建每日汇总记录
        cursor.execute("""
            INSERT OR REPLACE INTO daily_summary 
            (predict_date, target_date, total_predictions, status, created_at)
            VALUES (?, ?, ?, 'pending', ?)
        """, (predict_date, target_date, len(predictions), now))
        
        conn.commit()
        conn.close()
        
        logger.info(f"已记录 {len(predictions)} 条预测到回测数据库")
    
    def validate_predictions(self, target_date: str, actual_data: pd.DataFrame):
        """
        验证预测结果（用实际数据更新）
        
        Args:
            target_date: 目标日期
            actual_data: 实际数据DataFrame，需包含 sector_name, change_pct, limit_up_count
        """
        if actual_data.empty:
            logger.warning("实际数据为空，跳过验证")
            return
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # 获取该日期的预测记录
        cursor.execute("""
            SELECT id, sector_name FROM daily_predictions 
            WHERE target_date = ?
        """, (target_date,))
        predictions = cursor.fetchall()
        
        if not predictions:
            logger.warning(f"未找到 {target_date} 的预测记录")
            conn.close()
            return
        
        # 计算实际排名
        actual_data = actual_data.copy()
        actual_data['actual_rank'] = actual_data['change_pct'].rank(ascending=False)
        
        # 更新每条预测的实际结果
        hit_count = 0
        returns = []
        total_limit_up = 0
        
        for pred_id, sector_name in predictions:
            # 查找对应的实际数据
            actual_row = actual_data[actual_data['sector_name'] == sector_name]
            
            if actual_row.empty:
                continue
            
            actual_change = actual_row['change_pct'].values[0]
            actual_limit_up = actual_row.get('limit_up_count', pd.Series([0])).values[0]
            actual_rank = actual_row['actual_rank'].values[0]
            
            # 判断是否命中（涨幅>3% 或 有涨停）
            is_hit = 1 if (actual_change > 3 or actual_limit_up > 0) else 0
            
            cursor.execute("""
                UPDATE daily_predictions 
                SET actual_change_pct = ?, actual_limit_up_count = ?, 
                    actual_rank = ?, is_hit = ?, updated_at = ?
                WHERE id = ?
            """, (actual_change, actual_limit_up, actual_rank, is_hit, now, pred_id))
            
            hit_count += is_hit
            returns.append(actual_change)
            total_limit_up += actual_limit_up
        
        # 计算基准收益（全市场平均）
        benchmark_return = actual_data['change_pct'].mean()
        
        # 更新每日汇总
        avg_return = np.mean(returns) if returns else 0
        max_return = np.max(returns) if returns else 0
        min_return = np.min(returns) if returns else 0
        hit_rate = hit_count / len(predictions) if predictions else 0
        excess_return = avg_return - benchmark_return
        
        # 找到对应的predict_date
        cursor.execute("""
            SELECT predict_date FROM daily_predictions 
            WHERE target_date = ? LIMIT 1
        """, (target_date,))
        result = cursor.fetchone()
        predict_date = result[0] if result else None
        
        if predict_date:
            cursor.execute("""
                UPDATE daily_summary 
                SET hit_count = ?, hit_rate = ?, avg_return = ?, 
                    max_return = ?, min_return = ?, total_limit_up = ?,
                    benchmark_return = ?, excess_return = ?,
                    status = 'validated', validated_at = ?
                WHERE predict_date = ?
            """, (hit_count, hit_rate, avg_return, max_return, min_return,
                  total_limit_up, benchmark_return, excess_return, now, predict_date))
        
        conn.commit()
        conn.close()
        
        logger.info(f"已验证 {target_date} 预测: 命中率={hit_rate:.2%}, 平均收益={avg_return:.2f}%, 超额={excess_return:.2f}%")
    
    def auto_validate_yesterday(self, processor):
        """
        自动验证昨日预测（使用最新数据）
        
        Args:
            processor: SectorDataProcessor实例
        """
        # 获取待验证的记录
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT DISTINCT target_date FROM daily_summary 
            WHERE status = 'pending' AND target_date <= date('now')
        """)
        pending_dates = [row[0] for row in cursor.fetchall()]
        conn.close()
        
        if not pending_dates:
            logger.info("没有待验证的预测记录")
            return
        
        # 加载实际数据
        df = processor.load_history_data(days=30)
        
        for target_date in pending_dates:
            actual_data = df[df['date'] == target_date]
            if not actual_data.empty:
                self.validate_predictions(target_date, actual_data)
    
    def get_performance_report(self, days: int = 30) -> Dict:
        """
        获取绩效报告
        
        Args:
            days: 统计天数
            
        Returns:
            绩效指标字典
        """
        conn = sqlite3.connect(self.db_path)
        
        # 获取汇总数据
        query = f"""
            SELECT * FROM daily_summary 
            WHERE status = 'validated'
            ORDER BY predict_date DESC
            LIMIT {days}
        """
        df_summary = pd.read_sql(query, conn)
        
        # 获取详细预测数据
        query_detail = f"""
            SELECT * FROM daily_predictions 
            WHERE actual_change_pct IS NOT NULL
            ORDER BY predict_date DESC
        """
        df_detail = pd.read_sql(query_detail, conn)
        conn.close()
        
        if df_summary.empty:
            return {"status": "no_data", "message": "暂无已验证的预测数据"}
        
        # 计算各项指标
        report = {
            "status": "success",
            "period": f"最近{len(df_summary)}天",
            "total_predictions": int(df_summary['total_predictions'].sum()),
            
            # 命中率
            "overall_hit_rate": df_summary['hit_rate'].mean(),
            "total_hits": int(df_summary['hit_count'].sum()),
            
            # 收益统计
            "avg_daily_return": df_summary['avg_return'].mean(),
            "total_return": df_summary['avg_return'].sum(),
            "max_single_return": df_summary['max_return'].max(),
            "min_single_return": df_summary['min_return'].min(),
            
            # 超额收益
            "avg_excess_return": df_summary['excess_return'].mean(),
            "total_excess_return": df_summary['excess_return'].sum(),
            
            # 涨停统计
            "total_limit_up_captured": int(df_summary['total_limit_up'].sum()),
            
            # 按排名统计
            "top1_hit_rate": self._calc_rank_hit_rate(df_detail, 1),
            "top3_hit_rate": self._calc_rank_hit_rate(df_detail, 3),
            "top5_hit_rate": self._calc_rank_hit_rate(df_detail, 5),
            
            # 最近表现
            "recent_7d_hit_rate": df_summary.head(7)['hit_rate'].mean() if len(df_summary) >= 7 else None,
            "recent_7d_return": df_summary.head(7)['avg_return'].mean() if len(df_summary) >= 7 else None,
        }
        
        return report
    
    def _calc_rank_hit_rate(self, df: pd.DataFrame, rank: int) -> float:
        """计算特定排名的命中率"""
        rank_data = df[df['predict_rank'] <= rank]
        if rank_data.empty:
            return 0
        return rank_data['is_hit'].mean()
    
    def _get_next_trading_day(self, date_str: str) -> str:
        """获取下一个交易日（简单实现，跳过周末）"""
        date = datetime.strptime(date_str, "%Y-%m-%d")
        next_day = date + timedelta(days=1)
        
        # 跳过周末
        while next_day.weekday() >= 5:  # 5=周六, 6=周日
            next_day += timedelta(days=1)
        
        return next_day.strftime("%Y-%m-%d")
    
    def get_prediction_history(self, days: int = 7) -> pd.DataFrame:
        """获取预测历史记录"""
        conn = sqlite3.connect(self.db_path)
        query = f"""
            SELECT 
                p.predict_date,
                p.target_date,
                p.sector_name,
                p.predict_rank,
                p.predict_score,
                p.actual_change_pct,
                p.actual_limit_up_count,
                p.is_hit,
                s.hit_rate as daily_hit_rate,
                s.avg_return as daily_avg_return
            FROM daily_predictions p
            LEFT JOIN daily_summary s ON p.predict_date = s.predict_date
            ORDER BY p.predict_date DESC, p.predict_rank ASC
            LIMIT {days * 10}
        """
        df = pd.read_sql(query, conn)
        conn.close()
        return df
    
    def export_report(self, output_path: str = None) -> str:
        """导出回测报告"""
        if output_path is None:
            output_path = DATA_DIR / f"backtest_report_{datetime.now().strftime('%Y%m%d')}.md"
        
        report = self.get_performance_report(days=30)
        history = self.get_prediction_history(days=7)
        
        lines = [
            "# 📊 A股板块涨停预测系统 - 回测报告",
            "",
            f"**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "---",
            "",
            "## 📈 整体绩效",
            "",
            f"| 指标 | 数值 |",
            f"|------|------|",
            f"| 统计周期 | {report.get('period', 'N/A')} |",
            f"| 总预测次数 | {report.get('total_predictions', 0)} |",
            f"| 总命中次数 | {report.get('total_hits', 0)} |",
            f"| **整体命中率** | **{report.get('overall_hit_rate', 0):.2%}** |",
            f"| 平均日收益 | {report.get('avg_daily_return', 0):.2f}% |",
            f"| 累计收益 | {report.get('total_return', 0):.2f}% |",
            f"| **平均超额收益** | **{report.get('avg_excess_return', 0):.2f}%** |",
            f"| 捕获涨停数 | {report.get('total_limit_up_captured', 0)} |",
            "",
            "## 🎯 分排名命中率",
            "",
            f"| 排名 | 命中率 |",
            f"|------|--------|",
            f"| Top-1 | {report.get('top1_hit_rate', 0):.2%} |",
            f"| Top-3 | {report.get('top3_hit_rate', 0):.2%} |",
            f"| Top-5 | {report.get('top5_hit_rate', 0):.2%} |",
            "",
        ]
        
        if not history.empty:
            lines.extend([
                "## 📋 近期预测记录",
                "",
                "| 预测日期 | 板块 | 排名 | 实际涨幅 | 涨停数 | 命中 |",
                "|----------|------|------|----------|--------|------|",
            ])
            
            for _, row in history.head(20).iterrows():
                hit_mark = "✅" if row['is_hit'] == 1 else "❌"
                change = f"{row['actual_change_pct']:.2f}%" if pd.notna(row['actual_change_pct']) else "待验证"
                limit_up = int(row['actual_limit_up_count']) if pd.notna(row['actual_limit_up_count']) else "-"
                lines.append(
                    f"| {row['predict_date']} | {row['sector_name']} | {int(row['predict_rank'])} | {change} | {limit_up} | {hit_mark} |"
                )
        
        lines.extend([
            "",
            "---",
            "",
            "*报告由 A股板块涨停预测系统 自动生成*",
        ])
        
        content = "\n".join(lines)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        logger.info(f"回测报告已导出: {output_path}")
        return str(output_path)


class BacktestAnalyzer:
    """回测分析器"""
    
    def __init__(self, db: BacktestDatabase):
        self.db = db
    
    def analyze_by_sector(self) -> pd.DataFrame:
        """按板块分析命中率"""
        conn = sqlite3.connect(self.db.db_path)
        query = """
            SELECT 
                sector_name,
                COUNT(*) as predict_count,
                SUM(is_hit) as hit_count,
                AVG(is_hit) as hit_rate,
                AVG(actual_change_pct) as avg_return,
                SUM(actual_limit_up_count) as total_limit_up
            FROM daily_predictions
            WHERE actual_change_pct IS NOT NULL
            GROUP BY sector_name
            HAVING predict_count >= 3
            ORDER BY hit_rate DESC
        """
        df = pd.read_sql(query, conn)
        conn.close()
        return df
    
    def analyze_by_weekday(self) -> pd.DataFrame:
        """按星期几分析"""
        conn = sqlite3.connect(self.db.db_path)
        query = """
            SELECT 
                strftime('%w', target_date) as weekday,
                COUNT(*) as predict_count,
                AVG(is_hit) as hit_rate,
                AVG(actual_change_pct) as avg_return
            FROM daily_predictions
            WHERE actual_change_pct IS NOT NULL
            GROUP BY weekday
            ORDER BY weekday
        """
        df = pd.read_sql(query, conn)
        conn.close()
        
        # 转换星期
        weekday_map = {'0': '周日', '1': '周一', '2': '周二', '3': '周三', 
                       '4': '周四', '5': '周五', '6': '周六'}
        df['weekday'] = df['weekday'].map(weekday_map)
        
        return df
    
    def get_best_performing_sectors(self, top_n: int = 10) -> pd.DataFrame:
        """获取表现最好的板块"""
        df = self.analyze_by_sector()
        return df.head(top_n)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # 测试回测数据库
    db = BacktestDatabase()
    
    # 模拟预测数据
    test_predictions = pd.DataFrame({
        "sector_id": ["BK0001", "BK0002", "BK0003"],
        "sector_name": ["AI智能体", "AIGC概念", "军民融合"],
        "rank": [1, 2, 3],
        "pred_score": [0.95, 0.90, 0.85],
        "prediction_reason": ["资金流入大", "动量强", "北向增持"]
    })
    
    # 记录预测
    db.record_predictions(test_predictions)
    
    # 获取绩效报告
    report = db.get_performance_report()
    print("绩效报告:", report)
    
    # 导出报告
    db.export_report()
