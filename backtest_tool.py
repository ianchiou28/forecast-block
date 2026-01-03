"""
回测验证脚本 - 用于手动验证历史预测
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd
from datetime import datetime, timedelta
from backtest.database import BacktestDatabase, BacktestAnalyzer
from data.data_processor import SectorDataProcessor


def validate_with_actual_data():
    """使用实际数据验证预测"""
    db = BacktestDatabase()
    processor = SectorDataProcessor()
    
    print("=" * 60)
    print("回测验证工具")
    print("=" * 60)
    
    # 自动验证待验证的记录
    db.auto_validate_yesterday(processor)
    
    print("\n✅ 验证完成!")


def show_performance_summary():
    """显示绩效汇总"""
    db = BacktestDatabase()
    
    report = db.get_performance_report(days=30)
    
    print("\n" + "=" * 60)
    print("📊 回测绩效汇总")
    print("=" * 60)
    
    if report.get("status") == "no_data":
        print("\n暂无已验证的预测数据")
        print("请先运行几天预测，然后每天收盘后运行 'python main.py --mode fetch' 获取数据并自动验证")
        return
    
    print(f"\n📅 统计周期: {report.get('period', 'N/A')}")
    print(f"📈 总预测次数: {report.get('total_predictions', 0)}")
    print(f"✅ 总命中次数: {report.get('total_hits', 0)}")
    print(f"🎯 整体命中率: {report.get('overall_hit_rate', 0):.2%}")
    print(f"💰 平均日收益: {report.get('avg_daily_return', 0):.2f}%")
    print(f"📊 累计收益: {report.get('total_return', 0):.2f}%")
    print(f"💎 平均超额收益: {report.get('avg_excess_return', 0):.2f}%")
    print(f"🔥 捕获涨停数: {report.get('total_limit_up_captured', 0)}")
    
    print(f"\n🎯 分排名命中率:")
    print(f"   Top-1: {report.get('top1_hit_rate', 0):.2%}")
    print(f"   Top-3: {report.get('top3_hit_rate', 0):.2%}")
    print(f"   Top-5: {report.get('top5_hit_rate', 0):.2%}")
    
    if report.get('recent_7d_hit_rate'):
        print(f"\n📆 近7日表现:")
        print(f"   命中率: {report.get('recent_7d_hit_rate', 0):.2%}")
        print(f"   平均收益: {report.get('recent_7d_return', 0):.2f}%")


def show_prediction_history(days: int = 7):
    """显示预测历史"""
    db = BacktestDatabase()
    
    history = db.get_prediction_history(days=days)
    
    print("\n" + "=" * 60)
    print(f"📋 近{days}天预测记录")
    print("=" * 60)
    
    if history.empty:
        print("\n暂无预测记录")
        return
    
    # 按日期分组显示
    for date in history['predict_date'].unique():
        day_data = history[history['predict_date'] == date]
        print(f"\n📅 预测日期: {date}")
        print("-" * 50)
        
        for _, row in day_data.iterrows():
            hit_mark = "✅" if row['is_hit'] == 1 else "❌" if pd.notna(row['is_hit']) else "⏳"
            change = f"{row['actual_change_pct']:.2f}%" if pd.notna(row['actual_change_pct']) else "待验证"
            limit_up = int(row['actual_limit_up_count']) if pd.notna(row['actual_limit_up_count']) else "-"
            
            print(f"  {int(row['predict_rank'])}. {row['sector_name']:<12} | 涨幅: {change:>8} | 涨停: {limit_up:>3} | {hit_mark}")


def analyze_by_sector():
    """按板块分析"""
    db = BacktestDatabase()
    analyzer = BacktestAnalyzer(db)
    
    df = analyzer.analyze_by_sector()
    
    print("\n" + "=" * 60)
    print("📈 板块命中率分析")
    print("=" * 60)
    
    if df.empty:
        print("\n暂无足够数据进行分析")
        return
    
    print("\n命中率最高的板块 (至少预测3次):")
    print("-" * 60)
    
    for _, row in df.head(10).iterrows():
        print(f"  {row['sector_name']:<15} | 预测次数: {int(row['predict_count']):>3} | "
              f"命中率: {row['hit_rate']:.2%} | 平均收益: {row['avg_return']:.2f}%")


def export_report():
    """导出完整报告"""
    db = BacktestDatabase()
    
    path = db.export_report()
    print(f"\n📄 报告已导出: {path}")


def main():
    """主菜单"""
    import argparse
    
    parser = argparse.ArgumentParser(description="回测验证工具")
    parser.add_argument("action", choices=["validate", "summary", "history", "analyze", "export"],
                       help="操作: validate(验证), summary(汇总), history(历史), analyze(分析), export(导出)")
    parser.add_argument("--days", type=int, default=7, help="历史天数")
    
    args = parser.parse_args()
    
    if args.action == "validate":
        validate_with_actual_data()
    elif args.action == "summary":
        show_performance_summary()
    elif args.action == "history":
        show_prediction_history(args.days)
    elif args.action == "analyze":
        analyze_by_sector()
    elif args.action == "export":
        export_report()


if __name__ == "__main__":
    if len(sys.argv) == 1:
        # 无参数时显示所有信息
        show_performance_summary()
        show_prediction_history(7)
    else:
        main()
