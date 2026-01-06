"""
A股板块涨停预测系统 - 主入口
每日早上8点预测当日涨停板块
"""
import sys
import os
import logging
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent))

from config.settings import LOG_DIR, DATA_CONFIG
from data.data_fetcher import SectorDataFetcher
from data.data_processor import SectorDataProcessor, FeatureEngineer
from data.etf_data_fetcher import ETFDataFetcher
from data.etf_data_processor import ETFDataProcessor, ETFFeatureEngineer
from models.predictor import SectorPredictModel, RollingTrainer
from models.etf_predictor import ETFPredictModel, ETFRollingTrainer
from utils.report_generator import ReportGenerator, NotificationSender
from scheduler.task_scheduler import TaskScheduler, is_trading_day
from backtest.database import BacktestDatabase, BacktestAnalyzer
from backtest.etf_database import ETFBacktestDatabase
from backtest.etf_backtest_engine import (
    ETFHistoricalDataFetcher, 
    ETFFeatureEngineer as ETFBacktestFeatureEngineer,
    ETFRollingBacktestEngine
)

# 配置日志
def setup_logging():
    """配置日志系统"""
    log_file = LOG_DIR / f"system_{datetime.now().strftime('%Y%m%d')}.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file, encoding="utf-8"),
            logging.StreamHandler()
        ]
    )

logger = logging.getLogger(__name__)


class SectorPredictSystem:
    """板块涨停预测系统"""
    
    def __init__(self):
        self.fetcher = SectorDataFetcher()
        self.processor = SectorDataProcessor()
        self.feature_engineer = FeatureEngineer()
        self.model = SectorPredictModel()
        self.report_generator = ReportGenerator()
        self.notifier = NotificationSender()
        self.scheduler = TaskScheduler()
        self.backtest_db = BacktestDatabase()  # 回测数据库
        
        self.last_train_date = None
        self.model_info = {}
    
    def fetch_daily_data(self):
        """
        任务1: 获取每日数据（收盘后执行）
        """
        logger.info("=" * 50)
        logger.info("开始获取每日数据...")
        
        # 获取所有数据
        data = self.fetcher.fetch_all_daily_data()
        
        # 保存原始数据
        self.fetcher.save_daily_data(data)
        
        # 处理数据
        df_processed = self.processor.process_daily_data(data)
        
        # 保存到数据库
        self.processor.save_to_database(df_processed)
        
        # 自动验证昨日预测
        self.backtest_db.auto_validate_yesterday(self.processor)
        
        logger.info("每日数据获取完成")
        return df_processed
    
    def train_model(self, force: bool = False):
        """
        任务2: 训练/更新模型
        
        Args:
            force: 是否强制重新训练
        """
        logger.info("=" * 50)
        logger.info("检查模型训练状态...")
        
        trainer = RollingTrainer(self.model)
        
        # 判断是否需要重新训练
        if not force and not trainer.should_retrain(self.last_train_date):
            logger.info("模型无需重新训练")
            return
        
        # 加载历史数据
        df = self.processor.load_history_data(days=DATA_CONFIG["history_days"])
        
        if df.empty:
            logger.warning("历史数据为空，无法训练模型")
            return
        
        # 计算特征
        df_features = self.feature_engineer.compute_features(df)
        
        if df_features.empty:
            logger.warning("特征数据为空，跳过训练（需要至少5天数据积累）")
            return
        
        # 滚动训练
        self.model_info = trainer.rolling_train(
            df_features,
            train_window_months=DATA_CONFIG["train_window_months"]
        )
        
        if self.model_info.get("status") != "skipped":
            self.last_train_date = datetime.now().strftime("%Y-%m-%d")
        
        logger.info("模型训练完成")
    
    def predict_today(self) -> dict:
        """
        任务3: 生成今日预测（早上8点执行）
        
        Returns:
            预测结果字典
        """
        logger.info("=" * 50)
        logger.info("开始生成今日预测...")
        
        # 检查是否交易日
        if not is_trading_day():
            logger.info("今日非交易日，跳过预测")
            return {"status": "skipped", "reason": "非交易日"}
        
        # 加载最新数据（使用昨日收盘数据）
        df = self.processor.load_history_data(days=DATA_CONFIG["history_days"])
        
        if df.empty:
            logger.warning("无可用数据，无法预测")
            return {"status": "error", "reason": "数据不足"}
        
        # 计算特征
        df_features = self.feature_engineer.compute_features(df)
        
        if df_features.empty:
            logger.warning("特征数据为空，使用简单排名预测")
            # 使用简单的资金流排名作为预测
            df_latest = df[df["date"] == df["date"].max()].copy()
            df_latest["pred_score"] = df_latest["main_net_inflow"].rank(pct=True)
            df_latest["prediction_reason"] = "基于资金净流入排名（模型训练中）"
            df_latest = df_latest.sort_values("pred_score", ascending=False)
            df_latest["rank"] = range(1, len(df_latest) + 1)
            predictions = df_latest.head(5)
        else:
            # 获取最新一天的数据用于预测
            latest_date = df_features["date"].max()
            df_latest = df_features[df_features["date"] == latest_date]
            
            logger.info(f"使用 {latest_date} 数据进行预测")
            
            # 检查模型是否存在
            if self.model.model is None:
                self.model.load_model()
            
            if self.model.model is None:
                logger.warning("模型未训练，使用简单资金流排名")
                df_latest = df_latest.copy()
                df_latest["pred_score"] = df_latest["main_net_inflow"].rank(pct=True)
                df_latest["prediction_reason"] = "基于资金净流入排名（模型训练中）"
                df_latest = df_latest.sort_values("pred_score", ascending=False)
                df_latest["rank"] = range(1, len(df_latest) + 1)
                predictions = df_latest.head(5)
            else:
                # 执行模型预测
                predictions = self.model.predict(df_latest, top_k=5)
        
        if predictions.empty:
            logger.warning("预测结果为空")
            return {"status": "error", "reason": "预测失败"}
        
        # 记录预测到回测数据库
        self.backtest_db.record_predictions(predictions)
        logger.info("预测已记录到回测数据库")
        
        # 生成报告
        report_path = self.report_generator.generate_daily_report(
            predictions, self.model_info
        )
        
        # 生成HTML报告
        html_path = self.report_generator.generate_html_report(predictions)
        
        # 生成文本摘要
        summary = self.report_generator.generate_text_summary(predictions)
        
        # 发送通知
        self.notifier.send_all(summary)
        
        # 打印预测结果
        print("\n" + "=" * 60)
        print(summary)
        print("=" * 60 + "\n")
        
        return {
            "status": "success",
            "predictions": predictions.to_dict(orient="records"),
            "report_path": report_path,
            "html_path": html_path,
        }
    
    def run_full_pipeline(self):
        """运行完整流程（用于测试或手动执行）"""
        logger.info("运行完整预测流程...")
        
        # 1. 获取数据
        self.fetch_daily_data()
        
        # 2. 训练模型
        self.train_model()
        
        # 3. 预测
        result = self.predict_today()
        
        return result
    
    def start_scheduler(self):
        """启动定时调度"""
        logger.info("配置定时任务...")
        
        # 早上8点: 执行预测
        self.scheduler.add_daily_task(
            DATA_CONFIG["predict_time"],
            self.predict_today,
            "早盘涨停预测"
        )
        
        # 下午3:05: 获取数据
        self.scheduler.add_daily_task(
            DATA_CONFIG["fetch_time"],
            self.fetch_daily_data,
            "收盘数据更新"
        )
        
        # 下午3:30: 模型更新检查
        self.scheduler.add_daily_task(
            "15:30",
            self.train_model,
            "模型更新检查"
        )
        
        # 启动调度器
        self.scheduler.start()
        
        logger.info(f"调度器已启动，下次执行: {self.scheduler.get_next_run_time()}")
        
        return self.scheduler


class ETFPredictSystem:
    """ETF预测系统"""
    
    def __init__(self):
        self.fetcher = ETFDataFetcher()
        self.processor = ETFDataProcessor()
        self.feature_engineer = ETFFeatureEngineer()
        self.model = ETFPredictModel()
        self.report_generator = ReportGenerator()
        self.notifier = NotificationSender()
        
        self.last_train_date = None
        self.model_info = {}
    
    def fetch_daily_data(self):
        """获取每日ETF数据"""
        logger.info("=" * 50)
        logger.info("开始获取ETF每日数据...")
        
        # 获取所有ETF数据
        data = self.fetcher.fetch_all_daily_data()
        
        # 保存原始数据
        self.fetcher.save_daily_data(data)
        
        # 处理数据
        df_processed = self.processor.process_daily_data(data)
        
        # 保存到数据库
        self.processor.save_to_database(df_processed)
        
        logger.info("ETF每日数据获取完成")
        return df_processed
    
    def fetch_history_data(self, days: int = 60):
        """获取ETF历史数据（用于首次训练）"""
        logger.info("=" * 50)
        logger.info(f"开始获取ETF历史数据 (近{days}天)...")
        
        # 获取历史数据
        df_history = self.fetcher.fetch_all_etf_history(days=days)
        
        if df_history.empty:
            logger.warning("ETF历史数据获取失败")
            return pd.DataFrame()
        
        # 保存到数据库
        conn = __import__('sqlite3').connect(self.processor.db_path)
        
        for _, row in df_history.iterrows():
            try:
                conn.execute("""
                    INSERT OR REPLACE INTO daily_etf_data 
                    (date, etf_code, etf_name, open, high, low, close, 
                     volume, turnover, change_pct, turnover_rate, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    str(row.get("date", ""))[:10],
                    row.get("etf_code"),
                    row.get("etf_name"),
                    row.get("open"),
                    row.get("high"),
                    row.get("low"),
                    row.get("close"),
                    row.get("volume"),
                    row.get("turnover"),
                    row.get("change_pct"),
                    row.get("turnover_rate"),
                    datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                ))
            except Exception as e:
                pass
        
        conn.commit()
        conn.close()
        
        logger.info(f"ETF历史数据获取完成，共 {len(df_history)} 条记录")
        return df_history
    
    def train_model(self, force: bool = False):
        """训练ETF预测模型"""
        logger.info("=" * 50)
        logger.info("检查ETF模型训练状态...")
        
        trainer = ETFRollingTrainer(self.model)
        
        if not force and not trainer.should_retrain(self.last_train_date):
            logger.info("ETF模型无需重新训练")
            return
        
        # 加载历史数据
        df = self.processor.load_history_data(days=DATA_CONFIG.get("history_days", 60))
        
        if df.empty:
            logger.warning("ETF历史数据为空，尝试获取历史数据...")
            self.fetch_history_data(days=60)
            df = self.processor.load_history_data(days=60)
        
        if df.empty:
            logger.warning("无法获取ETF历史数据，跳过训练")
            return
        
        # 计算特征
        df_features = self.feature_engineer.compute_features(df)
        
        if df_features.empty:
            logger.warning("ETF特征数据为空，跳过训练（需要更多数据积累）")
            return
        
        # 滚动训练
        self.model_info = trainer.rolling_train(
            df_features,
            train_window_months=DATA_CONFIG.get("train_window_months", 6)
        )
        
        if self.model_info.get("status") == "success":
            self.last_train_date = datetime.now().strftime("%Y-%m-%d")
            logger.info("ETF模型训练完成")
        else:
            logger.warning(f"ETF模型训练失败: {self.model_info.get('message', 'unknown')}")
    
    def predict_today(self) -> dict:
        """生成今日ETF预测"""
        logger.info("=" * 50)
        logger.info("开始生成今日ETF预测...")
        
        # 加载历史数据
        df = self.processor.load_history_data(days=DATA_CONFIG.get("history_days", 60))
        
        if df.empty:
            logger.warning("无ETF历史数据，尝试获取...")
            self.fetch_history_data(days=60)
            df = self.processor.load_history_data(days=60)
        
        if df.empty:
            logger.warning("无可用ETF数据")
            return {"status": "error", "reason": "数据不足"}
        
        # 计算特征
        df_features = self.feature_engineer.compute_features(df)
        
        if df_features.empty:
            logger.warning("ETF特征数据为空，使用简单排名预测")
            # 使用简单的涨幅排名作为预测
            df_latest = df[df["date"] == df["date"].max()].copy()
            if df_latest.empty:
                return {"status": "error", "reason": "无最新数据"}
            
            # 根据近期涨跌幅和成交量综合排名
            df_latest["pred_score"] = df_latest["change_pct"].rank(pct=True)
            df_latest["prediction_reason"] = "基于近期表现排名（模型训练中）"
            df_latest = df_latest.sort_values("pred_score", ascending=False)
            df_latest["rank"] = range(1, len(df_latest) + 1)
            predictions = df_latest.head(5)
        else:
            # 获取最新一天的数据
            latest_date = df_features["date"].max()
            df_latest = df_features[df_features["date"] == latest_date]
            
            logger.info(f"使用 {latest_date} ETF数据进行预测")
            
            # 检查模型
            if self.model.model is None:
                self.model.load_model()
            
            if self.model.model is None:
                logger.warning("ETF模型未训练，使用简单排名")
                df_latest = df_latest.copy()
                df_latest["pred_score"] = df_latest.get("return_5d", df_latest.get("change_pct", 0)).rank(pct=True)
                df_latest["prediction_reason"] = "基于近期表现排名（模型训练中）"
                df_latest = df_latest.sort_values("pred_score", ascending=False)
                df_latest["rank"] = range(1, len(df_latest) + 1)
                predictions = df_latest.head(5)
            else:
                # 执行模型预测
                predictions = self.model.predict(df_latest, top_k=5)
        
        if predictions.empty:
            return {"status": "error", "reason": "预测失败"}
        
        # 记录预测
        self._record_predictions(predictions)
        
        # 生成报告
        report_path = self._generate_etf_report(predictions)
        
        # 生成文本摘要
        summary = self._generate_etf_summary(predictions)
        
        # 打印预测结果
        print("\n" + "=" * 60)
        print(summary)
        print("=" * 60 + "\n")
        
        return {
            "status": "success",
            "predictions": predictions.to_dict(orient="records"),
            "report_path": report_path,
        }
    
    def _record_predictions(self, predictions: pd.DataFrame):
        """记录预测到数据库"""
        conn = __import__('sqlite3').connect(self.processor.db_path)
        
        today = datetime.now().strftime("%Y-%m-%d")
        
        for _, row in predictions.iterrows():
            conn.execute("""
                INSERT INTO etf_predictions 
                (predict_date, etf_code, etf_name, pred_score, rank, prediction_reason, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                today,
                row.get("etf_code"),
                row.get("etf_name"),
                row.get("pred_score"),
                row.get("rank"),
                row.get("prediction_reason"),
                datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            ))
        
        conn.commit()
        conn.close()
        logger.info(f"已记录 {len(predictions)} 条ETF预测")
    
    def _generate_etf_report(self, predictions: pd.DataFrame) -> str:
        """生成ETF预测报告"""
        today = datetime.now().strftime("%Y-%m-%d")
        predict_date = datetime.now().strftime("%Y年%m月%d日")
        
        report_lines = [
            f"# 📊 ETF涨幅预测报告",
            f"",
            f"**预测日期**: {predict_date}",
            f"**生成时间**: {datetime.now().strftime('%H:%M:%S')}",
            f"",
            f"---",
            f"",
            f"## 🎯 今日预测ETF Top-5",
            f"",
            f"| 排名 | ETF代码 | ETF名称 | 预测得分 | 预测理由 |",
            f"|------|---------|---------|----------|----------|",
        ]
        
        for _, row in predictions.head(5).iterrows():
            rank = row.get("rank", "-")
            code = row.get("etf_code", "-")
            name = row.get("etf_name", "-")
            score = row.get("pred_score", 0)
            reason = row.get("prediction_reason", "-")
            report_lines.append(f"| {rank} | {code} | **{name}** | {score:.4f} | {reason} |")
        
        report_lines.extend([
            f"",
            f"---",
            f"",
            f"## ⚠️ 风险提示",
            f"",
            f"1. 本预测仅供参考，不构成投资建议",
            f"2. ETF投资有风险，请谨慎决策",
            f"",
            f"*报告由 ETF预测系统 自动生成*",
        ])
        
        report_content = "\n".join(report_lines)
        
        # 保存报告
        from config.settings import REPORT_DIR
        report_path = REPORT_DIR / f"etf_prediction_report_{today}.md"
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(report_content)
        
        logger.info(f"ETF预测报告已生成: {report_path}")
        return str(report_path)
    
    def _generate_etf_summary(self, predictions: pd.DataFrame) -> str:
        """生成ETF预测摘要"""
        today = datetime.now().strftime("%Y年%m月%d日")
        
        lines = [
            f"📊 【ETF涨幅预测】{today}",
            f"",
            f"🎯 今日预测ETF:",
        ]
        
        for i, (_, row) in enumerate(predictions.head(5).iterrows(), 1):
            code = row.get("etf_code", "-")
            name = row.get("etf_name", "-")
            score = row.get("pred_score", 0)
            reason = row.get("prediction_reason", "")
            lines.append(f"{i}. {name}({code}) (得分:{score:.2f})")
            if reason:
                lines.append(f"   └─ {reason}")
        
        lines.extend([
            f"",
            f"⚠️ 仅供参考，不构成投资建议",
        ])
        
        return "\n".join(lines)


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="A股板块涨停预测系统 & ETF预测系统")
    parser.add_argument(
        "--mode", 
        choices=["predict", "train", "fetch", "full", "daemon", "backtest", "report",
                 "etf-predict", "etf-train", "etf-fetch", "etf-full", "etf-backtest", 
                 "etf-backtest-run", "etf-report", "all-predict"],
        default="predict",
        help="运行模式: predict(板块预测), train(板块训练), fetch(获取板块数据), full(板块完整流程), "
             "daemon(守护进程), backtest(回测统计), report(生成回测报告), "
             "etf-predict(ETF预测), etf-train(ETF训练), etf-fetch(获取ETF数据), etf-full(ETF完整流程), "
             "etf-backtest(ETF回测统计), etf-backtest-run(运行3年ETF滚动回测), etf-report(ETF回测报告), "
             "all-predict(同时预测板块和ETF)"
    )
    parser.add_argument(
        "--force-train",
        action="store_true",
        help="强制重新训练模型"
    )
    parser.add_argument(
        "--days",
        type=int,
        default=30,
        help="回测统计天数 / ETF历史数据天数"
    )
    parser.add_argument(
        "--years",
        type=int,
        default=3,
        help="ETF历史数据年数（默认3年）"
    )
    parser.add_argument(
        "--train-months",
        type=int,
        default=24,
        help="ETF回测训练窗口（月），默认24个月"
    )
    
    args = parser.parse_args()
    
    # 初始化日志
    setup_logging()
    
    logger.info("=" * 60)
    logger.info("A股预测系统启动")
    logger.info(f"运行模式: {args.mode}")
    logger.info("=" * 60)
    
    # 根据模式选择系统
    if args.mode.startswith("etf"):
        
        # ETF回测相关命令
        if args.mode == "etf-backtest-run":
            # 运行3年ETF滚动回测
            logger.info("=" * 60)
            logger.info("📊 开始ETF滚动回测（3年数据）...")
            logger.info("=" * 60)
            
            # 获取历史数据
            fetcher = ETFHistoricalDataFetcher()
            
            # 尝试加载本地缓存
            df_history = fetcher.load_history_data()
            
            if df_history.empty:
                logger.info(f"本地无缓存，获取{args.years}年ETF历史数据...")
                df_history = fetcher.fetch_all_etf_history(years=args.years)
                if not df_history.empty:
                    fetcher.save_history_data(df_history)
            else:
                logger.info(f"使用本地缓存数据: {len(df_history)} 条")
            
            if df_history.empty:
                logger.error("无法获取ETF历史数据!")
                return
            
            # 计算回测时间范围
            end_date = datetime.now().strftime("%Y-%m-%d")
            start_date = (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")  # 最近1年回测
            
            # 运行滚动回测
            engine = ETFRollingBacktestEngine(
                train_window_months=args.train_months,
                step_months=1
            )
            
            results = engine.run_backtest(df_history, start_date, end_date)
            
            if not results.empty:
                print(f"\n✅ ETF回测完成! 共 {len(results)} 条预测记录")
            else:
                print("\n❌ ETF回测失败!")
                
        elif args.mode == "etf-backtest":
            # ETF回测统计
            etf_backtest_db = ETFBacktestDatabase()
            report = etf_backtest_db.get_performance_report(days=args.days)
            
            print("\n" + "=" * 60)
            print("📊 ETF回测绩效统计")
            print("=" * 60)
            
            if report.get("status") == "no_data":
                print("暂无已验证的ETF预测数据，请先运行回测或几天预测后再查看")
            else:
                print(f"📅 统计周期: {report.get('period', 'N/A')}")
                print(f"📈 总交易天数: {report.get('total_days', 0)}")
                print(f"📈 总预测次数: {report.get('total_predictions', 0)}")
                print(f"✅ 总命中次数: {report.get('total_hits', 0)}")
                print(f"🎯 整体命中率: {report.get('overall_hit_rate', 0):.2%}")
                print(f"🎯 Top5命中率: {report.get('top5_hit_rate', 0):.2%}")
                print(f"💰 平均日收益: {report.get('avg_daily_return', 0):.2f}%")
                print(f"📊 累计收益: {report.get('total_return', 0):.2f}%")
                print(f"💎 平均超额收益: {report.get('avg_excess_return', 0):.2f}%")
                print(f"🏆 胜率: {report.get('win_rate', 0):.2%}")
                print(f"📉 最大回撤: {report.get('max_drawdown', 0):.2f}%")
                print(f"📊 夏普比率: {report.get('sharpe_ratio', 0):.2f}")
            
            print("=" * 60)
            
        elif args.mode == "etf-report":
            # 生成ETF回测报告
            etf_backtest_db = ETFBacktestDatabase()
            report_path = etf_backtest_db.export_report(days=args.days)
            print(f"\nETF回测报告已生成: {report_path}")
            
            # 显示近期预测
            history = etf_backtest_db.get_prediction_history(days=7)
            if not history.empty:
                print("\n📋 近期ETF预测记录:")
                print(history[['predict_date', 'etf_code', 'etf_name', 
                              'predict_rank', 'actual_change_pct', 'is_hit']].to_string(index=False))
        
        else:
            # ETF预测系统
            system = ETFPredictSystem()
            
            if args.mode == "etf-predict":
                result = system.predict_today()
                print(f"\nETF预测结果: {result['status']}")
                
            elif args.mode == "etf-train":
                system.train_model(force=args.force_train)
            
            elif args.mode == "etf-fetch":
                # 获取历史数据
                system.fetch_history_data(days=args.days)
                # 获取最新数据
                system.fetch_daily_data()
                
            elif args.mode == "etf-full":
                # 完整流程：获取数据 -> 训练 -> 预测
                logger.info("执行ETF完整流程...")
                system.fetch_history_data(days=args.days)
                system.train_model(force=True)
                result = system.predict_today()
                print(f"\nETF执行结果: {result['status']}")
    
    elif args.mode == "all-predict":
        # 同时预测板块和ETF
        logger.info("=" * 60)
        logger.info("📈 开始板块预测...")
        logger.info("=" * 60)
        
        sector_system = SectorPredictSystem()
        sector_result = sector_system.predict_today()
        
        logger.info("=" * 60)
        logger.info("📊 开始ETF预测...")
        logger.info("=" * 60)
        
        etf_system = ETFPredictSystem()
        etf_result = etf_system.predict_today()
        
        print(f"\n板块预测结果: {sector_result['status']}")
        print(f"ETF预测结果: {etf_result['status']}")
        
    else:
        # 板块预测系统
        system = SectorPredictSystem()
        
        if args.mode == "predict":
            result = system.predict_today()
            print(f"\n预测结果: {result['status']}")
            
        elif args.mode == "train":
            system.train_model(force=args.force_train)
            
        elif args.mode == "fetch":
            system.fetch_daily_data()
            
        elif args.mode == "full":
            result = system.run_full_pipeline()
            print(f"\n执行结果: {result['status']}")
            
        elif args.mode == "daemon":
            scheduler = system.start_scheduler()
            
            print("\n系统已进入守护模式，按 Ctrl+C 退出")
            print(f"下次预测时间: {scheduler.get_next_run_time()}")
            
            try:
                while True:
                    import time
                    time.sleep(60)
            except KeyboardInterrupt:
                logger.info("收到退出信号，正在关闭...")
                scheduler.stop()
        
        elif args.mode == "backtest":
            report = system.backtest_db.get_performance_report(days=args.days)
            
            print("\n" + "=" * 60)
            print("📊 回测绩效统计")
            print("=" * 60)
            
            if report.get("status") == "no_data":
                print("暂无已验证的预测数据，请先运行几天预测后再查看")
            else:
                print(f"📅 统计周期: {report.get('period', 'N/A')}")
                print(f"📈 总预测次数: {report.get('total_predictions', 0)}")
                print(f"✅ 总命中次数: {report.get('total_hits', 0)}")
                print(f"🎯 整体命中率: {report.get('overall_hit_rate', 0):.2%}")
                print(f"💰 平均日收益: {report.get('avg_daily_return', 0):.2f}%")
                print(f"📊 累计收益: {report.get('total_return', 0):.2f}%")
                print(f"💎 平均超额收益: {report.get('avg_excess_return', 0):.2f}%")
                print(f"\n🎯 分排名命中率:")
                print(f"   Top-1: {report.get('top1_hit_rate', 0):.2%}")
                print(f"   Top-3: {report.get('top3_hit_rate', 0):.2%}")
                print(f"   Top-5: {report.get('top5_hit_rate', 0):.2%}")
            
            print("=" * 60)
            
        elif args.mode == "report":
            report_path = system.backtest_db.export_report()
            print(f"\n回测报告已生成: {report_path}")
            
            history = system.backtest_db.get_prediction_history(days=7)
            if not history.empty:
                print("\n📋 近期预测记录:")
                print(history[['predict_date', 'sector_name', 'predict_rank', 
                              'actual_change_pct', 'is_hit']].to_string(index=False))
    
    logger.info("系统退出")


if __name__ == "__main__":
    main()
