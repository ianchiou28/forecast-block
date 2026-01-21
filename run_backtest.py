#!/usr/bin/env python
"""
A股板块涨停预测系统 - 历史回测工具
真正的回测验证：用过去数据训练，预测未来数据

使用方法:
    # 步骤1: 下载历史数据 (首次运行需要，耗时约30-60分钟)
    python run_backtest.py download --start 2022 --end 2025
    
    # 步骤2: 运行回测 (用2022-2023训练，测试2024)
    python run_backtest.py run --train-start 2022 --train-end 2023 --test 2024
    
    # 快速测试 (用最近数据)
    python run_backtest.py quick
"""
import sys
import argparse
import logging
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent))


def setup_logging(verbose: bool = False):
    """配置日志"""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )


def cmd_download(args):
    """下载历史数据"""
    from backtest.historical_data import download_historical_data
    download_historical_data(args.start, args.end)


def cmd_run(args):
    """运行回测"""
    from backtest.backtest_engine import run_full_backtest
    run_full_backtest(
        train_years=(args.train_start, args.train_end),
        test_year=args.test
    )


def cmd_quick(args):
    """快速测试"""
    from backtest.historical_data import HistoricalDataFetcher
    from backtest.backtest_engine import RollingBacktestEngine, BacktestEvaluator
    
    print("\n" + "=" * 60)
    print("🚀 快速回测测试")
    print("=" * 60)
    
    # 检查数据
    fetcher = HistoricalDataFetcher()
    sector_history, limit_up_history = fetcher.load_historical_data()
    
    if sector_history.empty:
        print("\n❌ 没有找到历史数据!")
        print("\n请先下载历史数据:")
        print("  python run_backtest.py download --start 2022 --end 2025")
        return
    
    # 获取数据范围
    min_date = sector_history['date'].min()
    max_date = sector_history['date'].max()
    
    print(f"\n📂 已有数据范围: {min_date} ~ {max_date}")
    print(f"   板块数据: {len(sector_history)} 条")
    print(f"   涨停数据: {len(limit_up_history)} 条")
    
    # 自动确定测试范围（最后3个月）
    test_end = max_date
    test_start = str(int(test_end[:4]) - 1) + test_end[4:]  # 前一年
    
    print(f"\n🔄 运行回测: {test_start} -> {test_end}")
    
    engine = RollingBacktestEngine(train_window_months=12, step_months=1)
    results = engine.run_backtest(
        sector_history, limit_up_history,
        test_start, test_end
    )
    
    if results.empty:
        print("❌ 回测失败，没有生成结果")
        return
    
    # 评估
    evaluator = BacktestEvaluator(results)
    evaluator.print_summary()
    
    # 生成报告
    report_path = evaluator.generate_report()
    print(f"\n📄 详细报告: {report_path}")


def cmd_report(args):
    """查看最新报告"""
    import os
    
    results_dir = Path(__file__).parent / "data" / "backtest_results"
    
    if not results_dir.exists():
        print("❌ 没有找到回测结果目录")
        return
    
    # 找到最新的报告
    reports = list(results_dir.glob("*.md"))
    
    if not reports:
        print("❌ 没有找到回测报告")
        return
    
    latest_report = max(reports, key=os.path.getctime)
    
    print(f"\n📄 最新报告: {latest_report}\n")
    print("-" * 60)
    
    with open(latest_report, 'r', encoding='utf-8') as f:
        print(f.read())


def cmd_status(args):
    """查看数据状态"""
    from backtest.historical_data import HistoricalDataFetcher, HISTORICAL_DATA_DIR
    
    print("\n" + "=" * 60)
    print("📊 历史数据状态")
    print("=" * 60)
    
    if not HISTORICAL_DATA_DIR.exists():
        print("\n❌ 数据目录不存在")
        print("\n请先下载历史数据:")
        print("  python run_backtest.py download --start 2022 --end 2025")
        return
    
    # 列出数据文件
    print(f"\n📂 数据目录: {HISTORICAL_DATA_DIR}")
    
    sector_files = list(HISTORICAL_DATA_DIR.glob("sector_history_*.parquet"))
    limit_up_files = list(HISTORICAL_DATA_DIR.glob("limit_up_history_*.parquet"))
    
    if sector_files:
        print(f"\n📈 板块历史数据:")
        for f in sector_files:
            size_mb = f.stat().st_size / 1024 / 1024
            print(f"   - {f.name} ({size_mb:.1f} MB)")
    else:
        print("\n⚠️ 没有找到板块历史数据")
    
    if limit_up_files:
        print(f"\n🔥 涨停池历史数据:")
        for f in limit_up_files:
            size_mb = f.stat().st_size / 1024 / 1024
            print(f"   - {f.name} ({size_mb:.1f} MB)")
    else:
        print("\n⚠️ 没有找到涨停池历史数据")
    
    # 加载并显示详细信息
    fetcher = HistoricalDataFetcher()
    sector_history, limit_up_history = fetcher.load_historical_data()
    
    if not sector_history.empty:
        print(f"\n📊 数据摘要:")
        print(f"   日期范围: {sector_history['date'].min()} ~ {sector_history['date'].max()}")
        print(f"   板块数量: {sector_history['sector_name'].nunique()}")
        print(f"   总记录数: {len(sector_history)}")
    
    if not limit_up_history.empty:
        print(f"\n🔥 涨停数据摘要:")
        print(f"   日期范围: {limit_up_history['date'].min()} ~ {limit_up_history['date'].max()}")
        print(f"   涨停股票: {limit_up_history['stock_code'].nunique()}")
        print(f"   总记录数: {len(limit_up_history)}")
    
    print("\n" + "=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="A股板块涨停预测系统 - 历史回测工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  # 1. 下载历史数据（首次运行）
  python run_backtest.py download --start 2022 --end 2025
  
  # 2. 查看数据状态
  python run_backtest.py status
  
  # 3. 运行回测（2022-2023训练，2024测试）
  python run_backtest.py run --train-start 2022 --train-end 2023 --test 2024
  
  # 4. 查看报告
  python run_backtest.py report
        """
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="详细输出")
    
    subparsers = parser.add_subparsers(dest="command", help="命令")
    
    # download 命令
    p_download = subparsers.add_parser("download", help="下载历史数据")
    p_download.add_argument("--start", type=int, default=2022, help="开始年份 (默认: 2022)")
    p_download.add_argument("--end", type=int, default=2025, help="结束年份 (默认: 2025)")
    
    # run 命令
    p_run = subparsers.add_parser("run", help="运行回测")
    p_run.add_argument("--train-start", type=int, default=2022, help="训练开始年份")
    p_run.add_argument("--train-end", type=int, default=2023, help="训练结束年份")
    p_run.add_argument("--test", type=int, default=2024, help="测试年份")
    
    # quick 命令
    p_quick = subparsers.add_parser("quick", help="快速测试")
    
    # report 命令
    p_report = subparsers.add_parser("report", help="查看最新报告")
    
    # status 命令
    p_status = subparsers.add_parser("status", help="查看数据状态")
    
    args = parser.parse_args()
    
    setup_logging(args.verbose)
    
    if args.command == "download":
        cmd_download(args)
    elif args.command == "run":
        cmd_run(args)
    elif args.command == "quick":
        cmd_quick(args)
    elif args.command == "report":
        cmd_report(args)
    elif args.command == "status":
        cmd_status(args)
    else:
        parser.print_help()
        print("\n💡 快速开始:")
        print("  1. python run_backtest.py download  # 下载数据")
        print("  2. python run_backtest.py run       # 运行回测")


if __name__ == "__main__":
    main()
