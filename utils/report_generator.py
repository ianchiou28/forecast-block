"""
A股板块涨停预测系统 - 报告生成模块
生成预测报告并支持多渠道推送
"""
import pandas as pd
from datetime import datetime
from typing import Optional, Dict, List
import logging
import json
import requests
from pathlib import Path

from config.settings import REPORT_DIR, NOTIFICATION_CONFIG

logger = logging.getLogger(__name__)


class ReportGenerator:
    """报告生成器"""
    
    def __init__(self):
        self.report_dir = REPORT_DIR
    
    def generate_daily_report(self, predictions: pd.DataFrame, 
                              model_info: Dict = None) -> str:
        """
        生成每日预测报告
        
        Args:
            predictions: 预测结果DataFrame
            model_info: 模型信息
            
        Returns:
            报告文件路径
        """
        today = datetime.now().strftime("%Y-%m-%d")
        predict_date = datetime.now().strftime("%Y年%m月%d日")
        
        # 构建Markdown报告
        report_lines = [
            f"# 📈 A股板块涨停预测报告",
            f"",
            f"**预测日期**: {predict_date}",
            f"**生成时间**: {datetime.now().strftime('%H:%M:%S')}",
            f"",
            f"---",
            f"",
            f"## 🎯 今日预测涨停板块 Top-5",
            f"",
        ]
        
        if predictions.empty:
            report_lines.append("⚠️ 今日无有效预测数据")
        else:
            # 预测结果表格
            report_lines.append("| 排名 | 板块名称 | 预测得分 | 预测理由 |")
            report_lines.append("|------|----------|----------|----------|")
            
            for _, row in predictions.head(5).iterrows():
                rank = row.get("rank", "-")
                sector = row.get("sector_name", "-")
                score = row.get("pred_score", 0)
                reason = row.get("prediction_reason", "-")
                report_lines.append(f"| {rank} | **{sector}** | {score:.4f} | {reason} |")
        
        report_lines.extend([
            f"",
            f"---",
            f"",
            f"## 📊 预测依据说明",
            f"",
            f"### 核心因子权重",
            f"",
            f"1. **资金流向因子** (40%): 主力资金净流入、超大单净流入等",
            f"2. **涨停动量因子** (25%): 近期涨停家数、连板数等",
            f"3. **量价背离因子** (20%): 资金流入但价格未涨的吸筹信号",
            f"4. **趋势动量因子** (15%): 价格动量、波动率等",
            f"",
            f"### 风控过滤规则",
            f"",
            f"- ❌ 过去20日涨幅超30%且今日大跌的板块（高位避险）",
            f"- ❌ 资金持续流出超过3日的板块",
            f"",
        ])
        
        # 添加模型信息
        if model_info:
            report_lines.extend([
                f"---",
                f"",
                f"## 🤖 模型信息",
                f"",
                f"- **模型类型**: LightGBM",
                f"- **训练样本**: {model_info.get('train_samples', 'N/A')}",
                f"- **验证NDCG@5**: {model_info.get('ndcg@5', 'N/A'):.4f}" if isinstance(model_info.get('ndcg@5'), float) else f"- **验证NDCG@5**: N/A",
                f"",
            ])
        
        report_lines.extend([
            f"---",
            f"",
            f"## ⚠️ 风险提示",
            f"",
            f"1. 本预测仅供参考，不构成投资建议",
            f"2. 股市有风险，入市需谨慎",
            f"3. 历史表现不代表未来收益",
            f"4. 建议结合基本面和消息面综合判断",
            f"",
            f"---",
            f"",
            f"*报告由 A股板块涨停预测系统 自动生成*",
        ])
        
        report_content = "\n".join(report_lines)
        
        # 保存报告
        report_path = self.report_dir / f"prediction_report_{today}.md"
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(report_content)
        
        logger.info(f"预测报告已生成: {report_path}")
        
        return str(report_path)
    
    def generate_text_summary(self, predictions: pd.DataFrame) -> str:
        """
        生成简短文本摘要（用于消息推送）
        """
        today = datetime.now().strftime("%Y年%m月%d日")
        
        lines = [
            f"📈 【A股板块涨停预测】{today}",
            f"",
            f"🎯 今日预测涨停板块:",
        ]
        
        if predictions.empty:
            lines.append("⚠️ 今日无有效预测")
        else:
            for i, (_, row) in enumerate(predictions.head(5).iterrows(), 1):
                sector = row.get("sector_name", "-")
                score = row.get("pred_score", 0)
                reason = row.get("prediction_reason", "")
                lines.append(f"{i}. {sector} (得分:{score:.2f})")
                if reason:
                    lines.append(f"   └─ {reason}")
        
        lines.extend([
            f"",
            f"⚠️ 仅供参考，不构成投资建议",
        ])
        
        return "\n".join(lines)
    
    def generate_html_report(self, predictions: pd.DataFrame) -> str:
        """生成HTML格式报告"""
        today = datetime.now().strftime("%Y-%m-%d")
        predict_date = datetime.now().strftime("%Y年%m月%d日")
        
        html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>A股板块涨停预测报告 - {predict_date}</title>
    <style>
        body {{
            font-family: 'Microsoft YaHei', Arial, sans-serif;
            max-width: 800px;
            margin: 0 auto;
            padding: 20px;
            background: #f5f5f5;
        }}
        .container {{
            background: white;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        h1 {{
            color: #e74c3c;
            text-align: center;
            border-bottom: 2px solid #e74c3c;
            padding-bottom: 15px;
        }}
        .date-info {{
            text-align: center;
            color: #666;
            margin-bottom: 30px;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }}
        th, td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }}
        th {{
            background: #e74c3c;
            color: white;
        }}
        tr:hover {{
            background: #f5f5f5;
        }}
        .rank-1 {{ color: #e74c3c; font-weight: bold; }}
        .rank-2 {{ color: #f39c12; font-weight: bold; }}
        .rank-3 {{ color: #27ae60; font-weight: bold; }}
        .score {{
            background: #27ae60;
            color: white;
            padding: 3px 8px;
            border-radius: 4px;
        }}
        .warning {{
            background: #fff3cd;
            border: 1px solid #ffc107;
            padding: 15px;
            border-radius: 5px;
            margin-top: 30px;
        }}
        .footer {{
            text-align: center;
            color: #999;
            margin-top: 30px;
            font-size: 12px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>📈 A股板块涨停预测报告</h1>
        <div class="date-info">
            预测日期: {predict_date} | 生成时间: {datetime.now().strftime('%H:%M:%S')}
        </div>
        
        <h2>🎯 今日预测涨停板块</h2>
        <table>
            <tr>
                <th>排名</th>
                <th>板块名称</th>
                <th>预测得分</th>
                <th>预测理由</th>
            </tr>
"""
        
        if not predictions.empty:
            for i, (_, row) in enumerate(predictions.head(5).iterrows(), 1):
                rank_class = f"rank-{i}" if i <= 3 else ""
                sector = row.get("sector_name", "-")
                score = row.get("pred_score", 0)
                reason = row.get("prediction_reason", "-")
                
                html_content += f"""
            <tr>
                <td class="{rank_class}">{i}</td>
                <td><strong>{sector}</strong></td>
                <td><span class="score">{score:.4f}</span></td>
                <td>{reason}</td>
            </tr>
"""
        else:
            html_content += """
            <tr>
                <td colspan="4" style="text-align:center;">⚠️ 今日无有效预测数据</td>
            </tr>
"""
        
        html_content += """
        </table>
        
        <div class="warning">
            <strong>⚠️ 风险提示:</strong>
            本预测仅供参考，不构成投资建议。股市有风险，入市需谨慎。
        </div>
        
        <div class="footer">
            报告由 A股板块涨停预测系统 自动生成
        </div>
    </div>
</body>
</html>
"""
        
        # 保存HTML报告
        report_path = self.report_dir / f"prediction_report_{today}.html"
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(html_content)
        
        logger.info(f"HTML报告已生成: {report_path}")
        
        return str(report_path)


class NotificationSender:
    """通知推送器"""
    
    def __init__(self):
        self.config = NOTIFICATION_CONFIG
    
    def send_dingtalk(self, message: str) -> bool:
        """发送钉钉通知"""
        if not self.config.get("enable_dingtalk"):
            return False
        
        webhook = self.config.get("dingtalk_webhook")
        if not webhook:
            logger.warning("钉钉Webhook未配置")
            return False
        
        try:
            headers = {"Content-Type": "application/json"}
            data = {
                "msgtype": "text",
                "text": {"content": message}
            }
            response = requests.post(webhook, headers=headers, json=data, timeout=10)
            
            if response.status_code == 200:
                logger.info("钉钉通知发送成功")
                return True
            else:
                logger.error(f"钉钉通知发送失败: {response.text}")
                return False
        except Exception as e:
            logger.error(f"钉钉通知发送异常: {e}")
            return False
    
    def send_wechat(self, message: str) -> bool:
        """发送企业微信通知（需要配置企业微信机器人）"""
        if not self.config.get("enable_wechat"):
            return False
        
        # 企业微信机器人实现（预留）
        logger.warning("企业微信通知未实现")
        return False
    
    def send_email(self, subject: str, body: str, html_body: str = None) -> bool:
        """发送邮件通知"""
        if not self.config.get("enable_email"):
            return False
        
        # 邮件发送实现（预留）
        logger.warning("邮件通知未实现")
        return False
    
    def send_all(self, message: str, subject: str = "A股板块涨停预测"):
        """发送所有启用的通知渠道"""
        results = {
            "dingtalk": self.send_dingtalk(message),
            "wechat": self.send_wechat(message),
        }
        return results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # 测试报告生成
    generator = ReportGenerator()
    
    # 模拟预测数据
    test_predictions = pd.DataFrame({
        "rank": [1, 2, 3, 4, 5],
        "sector_name": ["固态电池", "人形机器人", "低空经济", "算力概念", "合成生物"],
        "pred_score": [0.92, 0.88, 0.85, 0.82, 0.79],
        "prediction_reason": [
            "资金连续3日流入，量价背离",
            "昨日涨停家数激增，动量效应",
            "北向资金增持，资金蓄力",
            "主力资金净流入居前",
            "近期涨停惯性较强"
        ]
    })
    
    # 生成报告
    report_path = generator.generate_daily_report(test_predictions)
    print(f"Markdown报告: {report_path}")
    
    html_path = generator.generate_html_report(test_predictions)
    print(f"HTML报告: {html_path}")
    
    summary = generator.generate_text_summary(test_predictions)
    print(f"\n文本摘要:\n{summary}")
