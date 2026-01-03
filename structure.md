# A股板块涨停预测系统

> 基于资金流向与机器学习的A股板块涨停预测系统，每日早上8点自动预测当日涨停板块。

## 📊 系统架构

```
forecast-block/
├── config/              # 配置模块
│   ├── __init__.py
│   └── settings.py      # 系统配置
├── data/                # 数据模块
│   ├── __init__.py
│   ├── data_fetcher.py  # 数据获取（AkShare）
│   ├── data_processor.py # 数据处理与特征工程
│   ├── sector_data.db   # 板块数据SQLite
│   ├── backtest.db      # 回测数据SQLite
│   ├── raw/             # 原始数据
│   └── processed/       # 处理后数据
├── models/              # 模型模块
│   ├── __init__.py
│   ├── predictor.py     # LightGBM预测模型
│   └── saved/           # 保存的模型
├── backtest/            # 回测模块
│   ├── __init__.py
│   └── database.py      # 回测数据库
├── utils/               # 工具模块
│   ├── __init__.py
│   └── report_generator.py # 报告生成
├── scheduler/           # 调度模块
│   ├── __init__.py
│   └── task_scheduler.py # 定时任务
├── logs/                # 日志目录
├── reports/             # 报告输出
├── main.py              # 主入口
├── test_predict.py      # 快速测试
├── backtest_tool.py     # 回测工具
├── environment.yml      # Conda环境配置
├── install.bat          # Windows安装脚本
├── install.sh           # Linux/Mac安装脚本
├── run.bat              # Windows启动脚本
├── run.sh               # Linux/Mac启动脚本
└── requirements.txt     # 依赖包
```

## 🚀 快速开始

### 1. 环境准备 (推荐 Conda)

**方式一：一键安装 (推荐)**

```bash
# Windows - 双击 install.bat 或运行:
install.bat

# Linux/Mac:
chmod +x install.sh
./install.sh
```

**方式二：手动安装**

```bash
# 使用 conda 创建环境
conda env create -f environment.yml

# 激活环境
conda activate forecast
```

**方式三：使用 pip**

```bash
# 安装Python 3.9+
pip install -r requirements.txt
```

### 2. 运行系统

```bash
# 激活 conda 环境
conda activate forecast

# 首次运行 - 获取数据
python main.py --mode fetch

# 运行预测
python main.py --mode predict
```

**快捷启动脚本:**

**Windows:**
```batch
# 双击 run.bat 或命令行执行
run.bat
```

**Linux/Mac:**
```bash
chmod +x run.sh
./run.sh
```

### 3. 运行模式

| 模式 | 命令 | 说明 |
|------|------|------|
| 预测 | `python main.py --mode predict` | 立即执行一次预测 |
| 获取数据 | `python main.py --mode fetch` | 获取最新市场数据 |
| 训练模型 | `python main.py --mode train` | 训练/更新模型 |
| 完整流程 | `python main.py --mode full` | 数据→训练→预测 |
| 守护模式 | `python main.py --mode daemon` | 定时自动执行 |

## 📈 功能特性

### 核心功能

1. **数据获取** - 自动获取以下数据：
   - 概念板块资金流向
   - 行业板块资金流向
   - 涨停池数据
   - 北向资金流向
   - 龙虎榜数据

2. **特征工程** - 构建多维度因子：
   - 资金流因子（流入/流出、蓄力信号）
   - 涨停动量因子（惯性、连板）
   - 量价背离因子（吸筹信号）
   - 趋势动量因子（波动率）

3. **模型预测** - LightGBM排序学习：
   - 滚动训练机制（每月更新）
   - 综合评分排名
   - 特征重要性分析

4. **报告生成** - 多格式输出：
   - Markdown报告
   - HTML可视化报告
   - 文本摘要（用于推送）

### 定时任务

| 时间 | 任务 |
|------|------|
| 08:00 | 执行当日涨停预测 |
| 15:05 | 获取收盘数据 + 自动验证预测 |
| 15:30 | 模型更新检查 |

## 📊 回测系统

系统内置回测数据库，自动记录每日预测并评估准确性。

### 回测功能

| 命令 | 说明 |
|------|------|
| `python main.py --mode backtest` | 查看回测绩效统计 |
| `python main.py --mode report` | 生成回测报告 |
| `python backtest_tool.py` | 交互式回测工具 |
| `python backtest_tool.py summary` | 绩效汇总 |
| `python backtest_tool.py history --days 7` | 查看预测历史 |
| `python backtest_tool.py analyze` | 板块命中分析 |
| `python backtest_tool.py export` | 导出报告 |

### 回测指标

- **命中率**: 预测板块次日涨幅>3%或出现涨停的比例
- **超额收益**: 预测板块平均收益 vs 全市场平均收益
- **分排名统计**: Top-1/Top-3/Top-5命中率
- **涨停捕获数**: 成功预测到涨停的累计数量

### 数据流程

```
每日早8点 → 发出预测 → 记录到回测数据库
                          ↓
每日15:05 → 获取实际数据 → 自动验证昨日预测
                          ↓
               更新命中率、收益等指标
```

## 🔧 配置说明

编辑 `config/settings.py` 自定义配置：

```python
# 预测时间
DATA_CONFIG = {
    "predict_time": "08:00",  # 早盘预测时间
    "fetch_time": "15:05",    # 数据获取时间
}

# 模型配置
MODEL_CONFIG = {
    "top_k": 5,  # 输出Top-K板块
    "rolling_step_days": 30,  # 滚动训练步长
}

# 通知配置（可选）
NOTIFICATION_CONFIG = {
    "enable_dingtalk": True,
    "dingtalk_webhook": "https://...",
}
```

## 📊 预测输出示例

```
📈 【A股板块涨停预测】2026年01月03日

🎯 今日预测涨停板块:
1. 固态电池 (得分:0.92)
   └─ 资金连续3日流入，量价背离
2. 人形机器人 (得分:0.88)
   └─ 昨日涨停家数激增，动量效应
3. 低空经济 (得分:0.85)
   └─ 北向资金增持，资金蓄力
4. 算力概念 (得分:0.82)
   └─ 主力资金净流入居前
5. 合成生物 (得分:0.79)
   └─ 近期涨停惯性较强

⚠️ 仅供参考，不构成投资建议
```

## 🛡️ 风控机制

- **高位避险**: 过去20日涨幅>30%且今日大跌的板块自动过滤
- **资金背离检测**: 识别主力出货信号
- **模型回测验证**: NDCG@5指标监控

## 📝 注意事项

1. **数据延迟**: 资金流向数据为收盘后更新，早盘预测使用前一日数据
2. **首次运行**: 需先积累至少5天数据才能进行有效预测
3. **模型冷启动**: 建议先运行 `--mode full` 完成初始化
4. **非交易日**: 系统自动跳过周末预测

## ⚠️ 风险提示

- 本系统仅供学习研究使用
- 预测结果不构成投资建议
- 股市有风险，入市需谨慎
- 历史表现不代表未来收益

## 📚 技术参考

- [AkShare](https://akshare.akfamily.xyz/) - 金融数据接口
- [LightGBM](https://lightgbm.readthedocs.io/) - 梯度提升框架
- [Microsoft Qlib](https://github.com/microsoft/qlib) - 量化投资平台

## 📄 License

MIT License
