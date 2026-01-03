import os
import sys
from pathlib import Path
from datetime import datetime
import json
import re
import sqlite3

from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
import uvicorn
import pandas as pd

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from config.settings import DATA_DIR, REPORT_DIR as REPORTS_DIR

app = FastAPI()

# Mount static files
app.mount("/static", StaticFiles(directory="web/static"), name="static")

# Templates
templates = Jinja2Templates(directory="web/templates")

def get_latest_report_date():
    """Find the latest report date from reports directory"""
    report_files = list(REPORTS_DIR.glob("prediction_report_*.md"))
    if not report_files:
        return datetime.now().strftime("%Y-%m-%d")
    
    # Sort by date in filename
    dates = []
    for f in report_files:
        match = re.search(r"prediction_report_(\d{4}-\d{2}-\d{2})\.md", f.name)
        if match:
            dates.append(match.group(1))
    
    if dates:
        return sorted(dates)[-1]
    return datetime.now().strftime("%Y-%m-%d")

def get_dashboard_data(date_str):
    """
    Gather data for the dashboard. 
    Tries to read from real data files if available, otherwise returns structure for UI.
    """
    data = {
        "date": date_str,
        "summary": {
            "report_count": 1,
            "major_events": 0,
            "positive_events": 0,
            "negative_events": 0
        },
        "sentiment": {
            "positive": 0,
            "neutral": 0,
            "negative": 0
        },
        "market_segmentation": {
            "cn": {"up": 0, "down": 0},
            "us": {"up": 0, "down": 0}
        },
        "events": [],
        "predictions": []
    }

    # 1. Try to read predictions from Markdown report
    report_path = REPORTS_DIR / f"prediction_report_{date_str}.md"
    if report_path.exists():
        with open(report_path, "r", encoding="utf-8") as f:
            content = f.read()
            
        # Parse Top 5 table
        # | 1 | **AI智能体** | 0.9974 | 主力资金净流入107.8亿元 |
        matches = re.findall(r"\|\s*\d+\s*\|\s*\*\*(.*?)\*\*\s*\|\s*([\d\.]+)\s*\|\s*(.*?)\s*\|", content)
        for m in matches:
            data["predictions"].append({
                "name": m[0],
                "score": float(m[1]),
                "reason": m[2]
            })
            
        data["summary"]["report_count"] = 1
        data["summary"]["major_events"] = len(data["predictions"])
        data["summary"]["positive_events"] = len(data["predictions"]) # Assuming predictions are positive opportunities

    # 2. Try to read raw data for sentiment/market stats
    # Path: data/raw/YYYYMMDD/
    raw_date = date_str.replace("-", "")
    raw_dir = DATA_DIR / "raw" / raw_date
    
    if raw_dir.exists():
        # Concept Money Flow
        concept_file = raw_dir / "concept_money_flow.csv"
        if concept_file.exists():
            try:
                df = pd.read_csv(concept_file)
                # Assuming columns like '净流入' or similar exist. 
                # If we don't know exact columns, we'll just count rows for now or guess.
                # Let's assume if we have data, we can calculate some stats.
                # For now, let's just use row counts as a proxy for "events" if we can't parse perfectly without checking CSV structure.
                pass
            except Exception as e:
                print(f"Error reading concept file: {e}")

        # Limit Up Pool (Sentiment)
        limit_up_file = raw_dir / "limit_up_pool.csv"
        if limit_up_file.exists():
            try:
                df_limit = pd.read_csv(limit_up_file)
                limit_count = len(df_limit)
                data["sentiment"]["positive"] = limit_count
                # We don't have limit down data in the file list, so we'll mock neutral/negative relative to positive
                data["sentiment"]["neutral"] = int(limit_count * 0.5)
                data["sentiment"]["negative"] = int(limit_count * 0.2)
                
                data["market_segmentation"]["cn"]["up"] = limit_count
                data["market_segmentation"]["cn"]["down"] = int(limit_count * 0.3) # Mock
            except:
                pass

    return data


def get_backtest_data(days: int = 30):
    """获取回测数据"""
    db_path = DATA_DIR / "backtest.db"
    
    if not db_path.exists():
        return {
            "status": "no_data",
            "message": "回测数据库不存在",
            "performance": {},
            "history": []
        }
    
    try:
        conn = sqlite3.connect(db_path)
        
        # 获取绩效汇总
        query_summary = """
            SELECT * FROM daily_summary 
            WHERE status = 'validated'
            ORDER BY predict_date DESC
            LIMIT ?
        """
        df_summary = pd.read_sql(query_summary, conn, params=(days,))
        
        # 获取预测历史详情
        query_history = """
            SELECT 
                p.predict_date,
                p.target_date,
                p.sector_name,
                p.predict_rank,
                p.predict_score,
                p.actual_change_pct,
                p.actual_limit_up_count,
                p.is_hit
            FROM daily_predictions p
            WHERE p.actual_change_pct IS NOT NULL
            ORDER BY p.predict_date DESC, p.predict_rank ASC
            LIMIT 100
        """
        df_history = pd.read_sql(query_history, conn)
        conn.close()
        
        if df_summary.empty:
            return {
                "status": "no_validated_data",
                "message": "暂无已验证的预测数据",
                "performance": {},
                "history": []
            }
        
        # 计算绩效指标
        total_predictions = int(df_summary['total_predictions'].sum())
        total_hits = int(df_summary['hit_count'].sum())
        overall_hit_rate = total_hits / total_predictions if total_predictions > 0 else 0
        
        performance = {
            "total_days": len(df_summary),
            "total_predictions": total_predictions,
            "total_hits": total_hits,
            "overall_hit_rate": round(overall_hit_rate * 100, 2),
            "avg_daily_return": round(df_summary['avg_return'].mean(), 2),
            "total_return": round(df_summary['avg_return'].sum(), 2),
            "avg_excess_return": round(df_summary['excess_return'].mean(), 2),
            "total_limit_up": int(df_summary['total_limit_up'].sum()),
        }
        
        # 计算分排名命中率
        if not df_history.empty:
            top1 = df_history[df_history['predict_rank'] == 1]
            top3 = df_history[df_history['predict_rank'] <= 3]
            top5 = df_history[df_history['predict_rank'] <= 5]
            
            performance["top1_hit_rate"] = round(top1['is_hit'].mean() * 100, 2) if not top1.empty else 0
            performance["top3_hit_rate"] = round(top3['is_hit'].mean() * 100, 2) if not top3.empty else 0
            performance["top5_hit_rate"] = round(top5['is_hit'].mean() * 100, 2) if not top5.empty else 0
        
        # 转换历史数据为列表
        history = []
        for _, row in df_history.iterrows():
            history.append({
                "predict_date": row['predict_date'],
                "target_date": row['target_date'],
                "sector_name": row['sector_name'],
                "predict_rank": int(row['predict_rank']),
                "predict_score": round(row['predict_score'], 4) if pd.notna(row['predict_score']) else 0,
                "actual_change_pct": round(row['actual_change_pct'], 2) if pd.notna(row['actual_change_pct']) else None,
                "actual_limit_up": int(row['actual_limit_up_count']) if pd.notna(row['actual_limit_up_count']) else 0,
                "is_hit": bool(row['is_hit']) if pd.notna(row['is_hit']) else None
            })
        
        return {
            "status": "success",
            "performance": performance,
            "history": history
        }
        
    except Exception as e:
        return {
            "status": "error",
            "message": str(e),
            "performance": {},
            "history": []
        }


@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/api/data")
async def get_data(date: str = None):
    if not date:
        date = get_latest_report_date()
    return get_dashboard_data(date)

@app.get("/api/backtest")
async def get_backtest(days: int = 30):
    return get_backtest_data(days)

if __name__ == "__main__":
    uvicorn.run("web_server:app", host="127.0.0.1", port=8000, reload=True)
