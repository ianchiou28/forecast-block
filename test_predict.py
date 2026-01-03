"""
快速测试预测
"""
import sys
sys.path.insert(0, '.')
from main import SectorPredictSystem, setup_logging
from backtest.database import BacktestDatabase
setup_logging()

system = SectorPredictSystem()
backtest_db = BacktestDatabase()

# 加载数据
df = system.processor.load_history_data(days=60)

if not df.empty:
    latest_date = df['date'].max()
    df_latest = df[df['date'] == latest_date].copy()
    
    # 按资金净流入排序（金额越大排名越高）
    df_latest = df_latest.sort_values('main_net_inflow', ascending=False)
    df_latest['rank'] = range(1, len(df_latest) + 1)
    df_latest['pred_score'] = 1 - (df_latest['rank'] / len(df_latest))
    
    # 添加预测理由
    def get_reason(row):
        inflow = row['main_net_inflow']
        if inflow > 1e9:
            return f"主力资金净流入{inflow/1e8:.1f}亿元"
        elif inflow > 1e8:
            return f"主力资金净流入{inflow/1e8:.2f}亿元"
        elif inflow > 0:
            return f"主力资金净流入{inflow/1e7:.1f}千万"
        else:
            return f"主力资金净流出{abs(inflow)/1e8:.2f}亿元"
    
    df_latest['prediction_reason'] = df_latest.apply(get_reason, axis=1)
    
    predictions = df_latest.head(5)
    
    print('=' * 60)
    print('📈 【A股板块涨停预测】测试预测')
    print('=' * 60)
    print()
    print('🎯 今日资金流入板块 Top-5 (明日预测涨停):')
    print()
    for _, row in predictions.iterrows():
        print(f"{row['rank']}. {row['sector_name']}")
        print(f"   得分: {row['pred_score']:.4f}")
        print(f"   理由: {row['prediction_reason']}")
        if row['limit_up_count'] > 0:
            print(f"   今日涨停: {int(row['limit_up_count'])}家")
        print()
    
    print('=' * 60)
    print('⚠️ 仅供参考，不构成投资建议')
    print()
    
    # 记录到回测数据库
    backtest_db.record_predictions(predictions)
    print('✅ 预测已记录到回测数据库')
    print()
    
    # 生成报告
    report_path = system.report_generator.generate_daily_report(predictions, {})
    html_path = system.report_generator.generate_html_report(predictions)
    print(f"📄 Markdown报告: {report_path}")
    print(f"🌐 HTML报告: {html_path}")
