# A股预测系统

> 基于深度神经网络与资金流向分析的A股板块涨停预测 & ETF涨幅预测系统

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## 📋 目录

- [功能特点](#-功能特点)
- [系统架构](#-系统架构)
- [快速开始](#-快速开始)
- [使用指南](#-使用指南)
- [运行模式详解](#-运行模式详解)
- [模型说明](#-模型说明)
- [项目结构](#-项目结构)
- [配置说明](#-配置说明)
- [理论基础](#-理论基础)
- [常见问题](#-常见问题)

**📚 完整理论文档**：[THEORY.md](THEORY.md) - 深度讲解系统原理、两种预测模式、特征工程、风险管理等

---

## ✨ 功能特点

### 🎯 板块涨停预测
- 基于 LightGBM 的板块涨停潮预测
- 实时资金流向监控（概念板块、行业板块）
- 龙虎榜数据分析
- 北向资金流向追踪

### 📊 ETF 涨幅预测
- **深度神经网络 (DNN)** 模型预测
- 支持 GPU 加速 (CUDA)
- 自动特征工程（19维特征）
- 早停机制防止过拟合
- NDCG@5 排名质量评估

### 📈 回测系统
- 滚动训练回测
- 多维度绩效评估
- 自动预测验证
- 详细回测报告生成

### 🤖 自动化调度
- 定时数据获取（收盘后）
- 定时模型训练
- 定时预测生成（早盘前）
- 守护进程模式

---

## 🏗 系统架构

```
┌─────────────────────────────────────────────────────────────┐
│                      A股预测系统                             │
├─────────────────────────┬───────────────────────────────────┤
│    板块涨停预测系统       │         ETF涨幅预测系统            │
│   (SectorPredictSystem) │      (ETFPredictSystem)           │
├─────────────────────────┼───────────────────────────────────┤
│    LightGBM 模型         │    深度神经网络 (PyTorch)          │
│    - 概念板块资金流       │    - 64→32→1 全连接网络           │
│    - 行业板块资金流       │    - ReLU + Dropout              │
│    - 龙虎榜数据          │    - 标准化预处理                  │
│    - 涨停池数据          │    - 早停机制                      │
├─────────────────────────┴───────────────────────────────────┤
│                      特征工程层                              │
│    动量特征 | 成交量特征 | 均线偏离 | RSI | ATR | 资金流      │
├─────────────────────────────────────────────────────────────┤
│                      数据获取层                              │
│              AKShare API → SQLite 数据库                    │
└─────────────────────────────────────────────────────────────┘
```

---

## 🚀 快速开始

### 环境要求

- Python 3.8+
- CUDA 11.8+ (可选，GPU加速)
- Windows / Linux / macOS

### 安装步骤

#### 方式一：使用 pip（推荐）

```bash
# 克隆仓库
git clone https://github.com/ianchiou28/forecast-block.git
cd forecast-block

# 安装依赖
pip install -r requirements.txt
```

#### 方式二：使用 Conda

```bash
# 创建环境
conda env create -f environment.yml
conda activate forecast

# 或使用快捷脚本
# Windows
install.bat

# Linux/macOS
./install.sh
```

### 首次运行

```bash
# 1. 获取ETF历史数据（首次需要）
python main.py --mode etf-fetch --days 60

# 2. 训练模型
python main.py --mode etf-train

# 3. 生成预测
python main.py --mode etf-predict
```

---

## 📖 使用指南

### 基本命令

```bash
python main.py --mode <模式> [选项]
```

### 可用模式

| 模式 | 说明 | 示例 |
|------|------|------|
| `predict` | 板块涨停预测 | `python main.py --mode predict` |
| `train` | 训练板块模型 | `python main.py --mode train` |
| `fetch` | 获取板块数据 | `python main.py --mode fetch` |
| `full` | 板块完整流程 | `python main.py --mode full` |
| `daemon` | 守护进程模式 | `python main.py --mode daemon` |
| `backtest` | 板块回测统计 | `python main.py --mode backtest --days 30` |
| `report` | 生成板块报告 | `python main.py --mode report` |
| `etf-predict` | ETF预测 | `python main.py --mode etf-predict` |
| `etf-train` | 训练ETF模型 | `python main.py --mode etf-train` |
| `etf-fetch` | 获取ETF数据 | `python main.py --mode etf-fetch --days 60` |
| `etf-full` | ETF完整流程 | `python main.py --mode etf-full` |
| `etf-backtest` | ETF回测统计 | `python main.py --mode etf-backtest` |
| `etf-backtest-run` | 运行ETF滚动回测 | `python main.py --mode etf-backtest-run --years 3` |
| `etf-report` | 生成ETF报告 | `python main.py --mode etf-report` |
| `all-predict` | 同时预测板块和ETF | `python main.py --mode all-predict` |

### 常用选项

| 选项 | 说明 | 默认值 |
|------|------|--------|
| `--force-train` | 强制重新训练模型 | False |
| `--days` | 回测天数/历史数据天数 | 30 |
| `--years` | ETF历史数据年数 | 3 |
| `--train-months` | 训练窗口（月） | 24 |

---

## 🔧 运行模式详解

### 1️⃣ 板块涨停预测 (`predict`)

基于资金流向和涨停数据预测明日可能出现涨停潮的板块。

```bash
python main.py --mode predict
```

**输出示例：**
```
============================================================
📈 【A股板块涨停预测】2026年01月06日

🎯 今日预测涨停板块:
1. AI智能体 (得分:1.00)
   └─ 资金持续流入，涨停家数增加
2. AIGC概念 (得分:0.98)
   └─ 龙头强势，资金抢筹
...
============================================================
```

### 2️⃣ ETF预测 (`etf-predict`)

使用深度神经网络预测ETF涨幅排名。

```bash
python main.py --mode etf-predict
```

**输出示例：**
```
============================================================
📊 【ETF涨幅预测】2026年01月06日

🎯 今日预测ETF:
1. 5GETF(515050) (得分:0.69)
   └─ 放量突破; 突破高位
2. 环保ETF(512580) (得分:0.61)
   └─ 缩量蓄势
3. 光伏ETF(515790) (得分:0.57)
   └─ 综合因子评分较高
...
============================================================
```

### 3️⃣ 模型训练 (`train` / `etf-train`)

```bash
# 板块模型训练
python main.py --mode train

# ETF深度神经网络训练
python main.py --mode etf-train

# 强制重新训练
python main.py --mode etf-train --force-train
```

**训练日志示例：**
```
使用计算设备: cuda
ETF训练集: 1092 样本, 验证集: 273 样本
特征数: 19
Epoch [10/100], Train Loss: 0.0851, Valid Loss: 0.1146
Epoch [20/100], Train Loss: 0.0791, Valid Loss: 0.1134
Early stopping at epoch 44
ETF模型训练完成: MSE=0.1134, NDCG@5=0.7054
```

### 4️⃣ 数据获取 (`fetch` / `etf-fetch`)

```bash
# 获取板块每日数据
python main.py --mode fetch

# 获取ETF历史数据（指定天数）
python main.py --mode etf-fetch --days 90
```

### 5️⃣ 完整流程 (`full` / `etf-full`)

一键执行：获取数据 → 训练模型 → 生成预测

```bash
# 板块完整流程
python main.py --mode full

# ETF完整流程
python main.py --mode etf-full --days 60
```

### 6️⃣ 回测系统 (`backtest` / `etf-backtest`)

```bash
# 查看板块回测统计
python main.py --mode backtest --days 30

# 查看ETF回测统计
python main.py --mode etf-backtest --days 30

# 运行3年ETF滚动回测
python main.py --mode etf-backtest-run --years 3 --train-months 24
```

**回测统计示例（2025年奖惩模式，本金 ¥10,000）：**
```
============================================================
📊 ETF奖惩模式回测绩效统计
============================================================
📅 统计周期: 2025-01-01 ~ 2025-12-31
📈 总交易天数: 243
💰 初始本金: ¥10,000
💎 Top-1最终资金: ¥16,543 (年化 +65.43%)
💎 Top-5最终资金: ¥12,586 (年化 +25.86%)
🎯 Top-1命中率: 53.91%
🎯 整体命中率: 52.35%
📊 平均日收益: +0.101%
🏆 胜率: 55.14%
📉 最大回撤: -11.97%
📊 夏普比率 (Top-1): 2.62
📊 夏普比率 (Top-5): 1.43
============================================================
```

### 7️⃣ 守护进程模式 (`daemon`)

自动定时执行预测任务，适合服务器部署。

```bash
python main.py --mode daemon
```

**定时任务：**
- **08:00** - 早盘涨停预测
- **15:05** - 收盘数据更新
- **15:30** - 模型更新检查

### 8️⃣ 生成报告 (`report` / `etf-report`)

```bash
# 生成板块回测报告
python main.py --mode report

# 生成ETF回测报告
python main.py --mode etf-report --days 30
```

---

## 🧠 模型说明

### ETF深度神经网络模型

```
输入层 (19维特征)
    │
    ▼
全连接层 (64神经元) + ReLU + Dropout(0.2)
    │
    ▼
全连接层 (32神经元) + ReLU + Dropout(0.2)
    │
    ▼
输出层 (1维预测得分)
```

**特征列表（19维）：**

| 类别 | 特征名 | 说明 |
|------|--------|------|
| 动量特征 | `return_3d`, `return_5d`, `return_10d`, `return_20d` | N日涨跌幅 |
| 波动率 | `volatility_3d`, `volatility_5d`, `volatility_10d`, `volatility_20d` | N日波动率 |
| 成交量 | `volume_ratio`, `turnover_ratio` | 量比、换手率比 |
| 价格位置 | `price_position` | 20日价格位置 |
| 均线偏离 | `ma5_bias`, `ma10_bias`, `ma20_bias` | 均线乖离率 |
| 资金流 | `money_flow_ma3`, `money_flow_ma5`, `money_flow_momentum` | 资金流均值与动量 |
| 技术指标 | `rsi_14`, `atr_14` | RSI、ATR |

**训练配置：**
- 优化器: Adam (lr=0.001)
- 损失函数: MSE Loss
- 早停: patience=10
- 批大小: 32
- 最大轮数: 100

---

## 📁 项目结构

```
forecast-block/
├── main.py                 # 主入口
├── requirements.txt        # Python依赖
├── environment.yml         # Conda环境
├── config/
│   └── settings.py         # 配置文件
├── data/
│   ├── data_fetcher.py     # 板块数据获取
│   ├── data_processor.py   # 板块数据处理
│   ├── etf_data_fetcher.py # ETF数据获取
│   └── etf_data_processor.py # ETF数据处理与特征工程
├── models/
│   ├── predictor.py        # 板块预测模型 (LightGBM)
│   ├── etf_predictor.py    # ETF预测模型 (DNN/PyTorch)
│   └── saved/              # 模型保存目录
│       ├── etf_predict_model.pth    # PyTorch模型权重
│       ├── etf_scaler.pkl           # 特征标准化器
│       └── etf_feature_importance.csv # 特征重要性
├── backtest/
│   ├── database.py         # 板块回测数据库
│   ├── etf_database.py     # ETF回测数据库
│   ├── backtest_engine.py  # 板块回测引擎
│   └── etf_backtest_engine.py # ETF回测引擎
├── utils/
│   └── report_generator.py # 报告生成器
├── scheduler/
│   └── task_scheduler.py   # 定时任务调度
├── web/                    # Web界面（可选）
│   ├── templates/
│   └── static/
├── reports/                # 预测报告输出
├── logs/                   # 运行日志
└── data/
    ├── historical/         # 历史数据缓存
    ├── backtest_results/   # 回测结果
    └── raw/                # 原始数据
```

---

## ⚙️ 配置说明

编辑 `config/settings.py` 进行配置：

```python
# 数据配置
DATA_CONFIG = {
    "history_days": 60,          # 加载历史天数
    "train_window_months": 6,    # 训练窗口（月）
    "predict_time": "08:00",     # 预测时间
    "fetch_time": "15:05",       # 数据获取时间
}

# 模型配置
MODEL_CONFIG = {
    "learning_rate": 0.001,
    "epochs": 100,
    "batch_size": 32,
    "hidden_dim1": 64,
    "hidden_dim2": 32,
    "dropout_rate": 0.2,
}

# ETF列表
ETF_LIST = [
    ("515050", "5GETF"),
    ("512760", "芯片ETF"),
    ("515790", "光伏ETF"),
    # ... 更多ETF
]
```

---

## 📚 理论基础

**本系统包含两种预测模式，详细理论请参考 [THEORY.md](THEORY.md) 文档：**

### 📖 THEORY.md 完整目录

1. **理论基础** - 为什么预测涨停潮、资金流向的意义
2. **市场微观结构** - 涨停潮三阶段、量价背离、龙虎榜分析
3. **数据工程架构** - 数据源选择、特征提取、数据库设计
4. **特征工程体系** - 一阶特征、二阶特征、高级特征详解
5. **模型选择与优化** - 为什么选LightGBM、训练策略、评估指标
6. **🆕 两种预测模式对比** ⭐
   - **模式一：标准监督学习 (DNN)** - 追求预测精度，稳健收益
   - **模式二：奖惩强化学习 (Reward DNN)** - 追求实盘收益，高风险高收益
   - 完整对比实验数据（2024年回测）
   - 模式选择建议
7. **风险管理框架** - 高位接盘风险、市场风格变化、数据质量
8. **性能评估指标** - NDCG、Hit Rate、Excess Return、月度跟踪
9. **核心创新点** - 量价背离、双模式框架、滚动训练、多源融合

### 🎯 快速了解系统

**为什么预测"涨停潮"？**

A股涨停板制度（±10%）创造了独特的价格发现机制：

- **涨停 = 流动性黑洞**：市场上没有足够的卖家
- **涨停潮 = 市场共识极强**：板块内多只股票同日涨停
- **预测涨停比预测涨幅更实用**：离散事件，信号清晰

### 💡 两种预测模式选择指南

| 维度 | 标准模式（DNN） | 奖惩模式（Reward DNN） |
|------|----------------|----------------------|
| **年化收益** | +117% | +152% ✓ |
| **夏普比率** | 1.35 | 1.58 ✓ |
| **最大回撤** | -8.9% ✓ | -11.2% |
| **适用场景** | 震荡市、稳健投资 | 趋势市、进取投资 |
| **风险水平** | 低 | 中等 |

> 📖 **查看详细对比 → [THEORY.md - 第6章](THEORY.md#两种预测模式对比)**

### 资金流向的意义

````

$$\text{Net Flow} = \sum_{\text{buy large order}} \text{Volume} - \sum_{\text{sell large order}} \text{Volume}$$

- **大单**（>50万）代表机构/游资行为
- **板块级资金流**比个股更可靠（噪声抵消）
- **持续净流入 = 聪明资金吸筹**

### 深度学习的优势

相比传统模型，DNN能够：
1. 自动学习特征交互
2. 捕捉非线性关系
3. 更高的预测准确性

| 模型 | 可解释性 | 预测准确性 |
|------|----------|------------|
| 线性回归 | ⭐⭐⭐⭐⭐ | ⭐⭐ |
| 决策树 | ⭐⭐⭐⭐ | ⭐⭐⭐ |
| 随机森林 | ⭐⭐⭐ | ⭐⭐⭐⭐ |
| **深度神经网络** | ⭐⭐ | ⭐⭐⭐⭐⭐ |

---

## ❓ 常见问题

### Q: 提示 `No module named 'torch'`
```bash
pip install torch>=2.0.0
# 或安装CUDA版本
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

### Q: 如何使用GPU训练？
系统会自动检测CUDA，日志显示 `使用计算设备: cuda` 即为GPU模式。

### Q: 数据获取失败？
- 检查网络连接
- AKShare 可能有请求频率限制，稍后重试
- 确保 `akshare>=1.12.0`

### Q: 模型预测效果不好？
- 确保有足够历史数据（建议60天以上）
- 尝试调整训练窗口 `--train-months`
- 使用 `--force-train` 重新训练

### Q: 如何查看特征重要性？
查看 `models/saved/etf_feature_importance.csv`

---

## 📄 许可证

MIT License

---

## ⚠️ 免责声明

**本系统仅供学习研究使用，不构成任何投资建议。**

- 股市有风险，投资需谨慎
- 历史表现不代表未来收益
- 请根据自身情况独立决策

---

## 🤝 贡献

欢迎提交 Issue 和 Pull Request！

---

*Made with ❤️ by forecast-block team*
