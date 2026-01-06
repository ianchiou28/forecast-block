"""
ETF奖惩机制深度学习回测脚本

使用方法:
    python run_reward_backtest.py

回测策略:
    1. 使用2022-2023年数据进行滚动训练
    2. 每月重新训练模型（纳入最新数据）
    3. 2024全年作为回测期验证收益
"""
import sys
import logging
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent))

from config.settings import LOG_DIR
from backtest.etf_backtest_engine import ETFHistoricalDataFetcher, ETFFeatureEngineer
from models.etf_reward_predictor import RewardRollingBacktest

# 配置日志
def setup_logging():
    log_file = LOG_DIR / f"reward_backtest_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file, encoding="utf-8"),
            logging.StreamHandler()
        ]
    )

logger = logging.getLogger(__name__)


def load_or_fetch_data(years: int = 4) -> pd.DataFrame:
    """加载或获取历史数据"""
    
    fetcher = ETFHistoricalDataFetcher()
    
    # 尝试加载本地缓存
    cache_path = Path(__file__).parent / "data" / "historical" / f"etf_history_{years}y.csv"
    
    if cache_path.exists():
        logger.info(f"📂 加载本地缓存: {cache_path}")
        df = pd.read_csv(cache_path)
        logger.info(f"   数据量: {len(df)} 条")
        return df
    
    # 获取数据
    logger.info(f"🌐 获取{years}年ETF历史数据...")
    df = fetcher.fetch_all_etf_history(years=years)
    
    if not df.empty:
        # 保存缓存
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(cache_path, index=False, encoding="utf-8-sig")
        logger.info(f"💾 数据已缓存: {cache_path}")
    
    return df


def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """计算特征"""
    logger.info("🔧 计算技术特征...")
    
    feature_engineer = ETFFeatureEngineer()
    df_features = feature_engineer.compute_features(df)
    
    logger.info(f"   特征计算完成: {len(df_features)} 条")
    return df_features


def run_backtest():
    """执行回测"""
    
    print("\n" + "=" * 70)
    print("🚀 ETF奖惩机制深度学习回测系统")
    print("=" * 70)
    print("\n📋 回测配置:")
    print("   • 训练数据: 2022-01-01 ~ 2023-12-31 (滚动窗口)")
    print("   • 回测数据: 2024-01-01 ~ 2024-12-31")
    print("   • 训练窗口: 12个月")
    print("   • 重训练间隔: 每月初")
    print("   • 预测Top-K: 5")
    print("   • 奖励权重: 0.5")
    print("   • 惩罚权重: 0.8")
    print("=" * 70 + "\n")
    
    # 1. 加载数据
    df_raw = load_or_fetch_data(years=4)  # 2021-2024，4年数据
    
    if df_raw.empty:
        logger.error("❌ 无法获取历史数据!")
        return
    
    # 2. 计算特征
    df_features = compute_features(df_raw)
    
    if df_features.empty:
        logger.error("❌ 特征计算失败!")
        return
    
    # 检查数据范围
    date_range = df_features["date"].agg(["min", "max"])
    logger.info(f"📅 数据范围: {date_range['min']} ~ {date_range['max']}")
    
    # 3. 执行回测
    backtest = RewardRollingBacktest(
        train_window_months=12,     # 12个月训练窗口
        retrain_interval_months=1,  # 每月重训练
        top_k=5,
        reward_weight=0.5,
        penalty_weight=0.8
    )
    
    report = backtest.run_backtest(
        df_features,
        train_start="2022-01-01",
        train_end="2023-12-31",
        test_start="2024-01-01",
        test_end="2024-12-31"
    )
    
    # 4. 打印报告
    if report.get("status") == "success":
        backtest.print_report(report)
        print("\n✅ 回测完成! 详细报告已保存至 data/backtest_results/ 目录")
    else:
        print(f"\n❌ 回测失败: {report.get('message', 'unknown error')}")


def run_parameter_search():
    """参数搜索（可选）"""
    
    print("\n🔍 开始参数搜索...")
    
    # 加载数据
    df_raw = load_or_fetch_data(years=4)
    if df_raw.empty:
        return
    
    df_features = compute_features(df_raw)
    if df_features.empty:
        return
    
    # 参数网格
    reward_weights = [0.3, 0.5, 0.7]
    penalty_weights = [0.5, 0.8, 1.0]
    train_windows = [6, 12, 18]
    
    results = []
    
    for rw in reward_weights:
        for pw in penalty_weights:
            for tw in train_windows:
                logger.info(f"\n📊 测试参数: reward={rw}, penalty={pw}, window={tw}个月")
                
                backtest = RewardRollingBacktest(
                    train_window_months=tw,
                    retrain_interval_months=1,
                    top_k=5,
                    reward_weight=rw,
                    penalty_weight=pw
                )
                
                report = backtest.run_backtest(
                    df_features,
                    train_start="2022-01-01",
                    train_end="2023-12-31",
                    test_start="2024-01-01",
                    test_end="2024-12-31"
                )
                
                if report.get("status") == "success":
                    results.append({
                        "reward_weight": rw,
                        "penalty_weight": pw,
                        "train_window": tw,
                        "top1_return": report.get("top1_total_return", 0),
                        "top5_return": report.get("top5_total_return", 0),
                        "sharpe": report.get("top5_sharpe", 0),
                        "hit_rate": report.get("overall_hit_rate", 0),
                        "max_drawdown": report.get("max_drawdown", 0),
                    })
    
    # 输出最优参数
    if results:
        results_df = pd.DataFrame(results)
        results_df = results_df.sort_values("top5_return", ascending=False)
        
        print("\n" + "=" * 70)
        print("🏆 参数搜索结果 (按Top-5收益排序)")
        print("=" * 70)
        print(results_df.to_string(index=False))
        
        # 保存结果
        save_path = Path(__file__).parent / "data" / "backtest_results" / "param_search_results.csv"
        results_df.to_csv(save_path, index=False)
        print(f"\n📄 结果已保存: {save_path}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="ETF奖惩机制深度学习回测")
    parser.add_argument("--mode", choices=["backtest", "search"], default="backtest",
                       help="运行模式: backtest(单次回测) 或 search(参数搜索)")
    
    args = parser.parse_args()
    
    setup_logging()
    
    if args.mode == "backtest":
        run_backtest()
    elif args.mode == "search":
        run_parameter_search()
