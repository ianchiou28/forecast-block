"""
A股板块涨停预测系统 - 配置文件
"""
import os
from pathlib import Path

# ==================== 路径配置 ====================
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODEL_DIR = BASE_DIR / "models" / "saved"
LOG_DIR = BASE_DIR / "logs"
REPORT_DIR = BASE_DIR / "reports"

# 确保目录存在
for dir_path in [RAW_DATA_DIR, PROCESSED_DATA_DIR, MODEL_DIR, LOG_DIR, REPORT_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# ==================== 数据获取配置 ====================
DATA_CONFIG = {
    # 数据获取时间（收盘后）
    "fetch_time": "15:05",
    # 预测时间（早上8点）
    "predict_time": "08:00",
    # 历史数据天数（用于特征计算）
    "history_days": 60,
    # 训练数据窗口（月）
    "train_window_months": 24,
    # 验证数据窗口（月）
    "valid_window_months": 3,
}

# ==================== 资金流因子配置 ====================
FACTOR_CONFIG = {
    # 资金流动量窗口
    "money_flow_windows": [3, 5, 10, 20],
    # 涨停动量窗口
    "limit_up_windows": [3, 5],
    # 价格动量窗口
    "price_momentum_windows": [5, 10, 20],
    # 波动率窗口
    "volatility_windows": [5, 10, 20],
}

# ==================== 模型配置 ====================
MODEL_CONFIG = {
    "lgbm_params": {
        "objective": "regression",
        "metric": "mse",
        "boosting_type": "gbdt",
        "num_leaves": 210,
        "max_depth": 8,
        "learning_rate": 0.01,
        "feature_fraction": 0.88,
        "bagging_fraction": 0.87,
        "bagging_freq": 5,
        "lambda_l1": 205.69,
        "verbose": -1,
        "n_estimators": 500,
        "early_stopping_rounds": 50,
    },
    # 滚动训练步长（天）
    "rolling_step_days": 30,
    # 预测输出Top-K板块
    "top_k": 5,
}

# ==================== 风控配置 ====================
RISK_CONFIG = {
    # 高位避险：过去N天涨幅阈值
    "high_position_days": 20,
    "high_position_threshold": 0.30,  # 30%
    # 巨阴线阈值
    "big_drop_threshold": -0.03,  # -3%
}

# ==================== 通知配置 ====================
NOTIFICATION_CONFIG = {
    "enable_wechat": False,
    "enable_email": False,
    "enable_dingtalk": False,
    # 钉钉机器人Webhook（需自行配置）
    "dingtalk_webhook": "",
    # 邮件配置
    "email_smtp_server": "",
    "email_from": "",
    "email_to": [],
}

# ==================== 数据库配置 ====================
DATABASE_CONFIG = {
    "type": "sqlite",  # sqlite / mysql
    "sqlite_path": DATA_DIR / "sector_data.db",
    # MySQL配置（可选）
    "mysql_host": "localhost",
    "mysql_port": 3306,
    "mysql_user": "root",
    "mysql_password": "",
    "mysql_database": "sector_predict",
}
