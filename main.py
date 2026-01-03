"""
A股板块涨停预测系统 - 主入口
每日早上8点预测当日涨停板块
"""
import sys
import os
import logging
from datetime import datetime
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent))

from config.settings import LOG_DIR, DATA_CONFIG
from data.data_fetcher import SectorDataFetcher
from data.data_processor import SectorDataProcessor, FeatureEngineer
from models.predictor import SectorPredictModel, RollingTrainer
from utils.report_generator import ReportGenerator, NotificationSender
from scheduler.task_scheduler import TaskScheduler, is_trading_day
from backtest.database import BacktestDatabase, BacktestAnalyzer

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


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="A股板块涨停预测系统")
    parser.add_argument(
        "--mode", 
        choices=["predict", "train", "fetch", "full", "daemon", "backtest", "report"],
        default="predict",
        help="运行模式: predict(预测), train(训练), fetch(获取数据), full(完整流程), daemon(守护进程), backtest(回测统计), report(生成回测报告)"
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
        help="回测统计天数"
    )
    
    args = parser.parse_args()
    
    # 初始化日志
    setup_logging()
    
    logger.info("=" * 60)
    logger.info("A股板块涨停预测系统启动")
    logger.info(f"运行模式: {args.mode}")
    logger.info("=" * 60)
    
    # 初始化系统
    system = SectorPredictSystem()
    
    if args.mode == "predict":
        # 仅执行预测
        result = system.predict_today()
        print(f"\n预测结果: {result['status']}")
        
    elif args.mode == "train":
        # 训练模型
        system.train_model(force=args.force_train)
        
    elif args.mode == "fetch":
        # 获取数据
        system.fetch_daily_data()
        
    elif args.mode == "full":
        # 完整流程
        result = system.run_full_pipeline()
        print(f"\n执行结果: {result['status']}")
        
    elif args.mode == "daemon":
        # 守护进程模式
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
        # 回测统计
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
        # 生成回测报告
        report_path = system.backtest_db.export_report()
        print(f"\n回测报告已生成: {report_path}")
        
        # 查看历史预测
        history = system.backtest_db.get_prediction_history(days=7)
        if not history.empty:
            print("\n📋 近期预测记录:")
            print(history[['predict_date', 'sector_name', 'predict_rank', 
                          'actual_change_pct', 'is_hit']].to_string(index=False))
    
    logger.info("系统退出")


if __name__ == "__main__":
    main()
