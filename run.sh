#!/bin/bash
# A股板块涨停预测系统 - Linux/Mac启动脚本

echo "============================================"
echo "   A股板块涨停预测系统 - 快速启动"
echo "============================================"
echo ""

# 检查Python
if ! command -v python3 &> /dev/null; then
    echo "[错误] 未找到Python，请先安装Python 3.9+"
    exit 1
fi

# 检查依赖
echo "[1/3] 检查依赖包..."
if ! python3 -c "import akshare" 2>/dev/null; then
    echo "正在安装依赖包..."
    pip3 install -r requirements.txt
fi

echo ""
echo "[2/3] 选择运行模式:"
echo "  1. 立即预测 (predict)"
echo "  2. 获取数据 (fetch)"
echo "  3. 训练模型 (train)"
echo "  4. 完整流程 (full)"
echo "  5. 守护模式 (daemon) - 自动定时执行"
echo ""

read -p "请输入选项 (1-5): " choice

case $choice in
    1)
        echo ""
        echo "[3/3] 执行预测..."
        python3 main.py --mode predict
        ;;
    2)
        echo ""
        echo "[3/3] 获取数据..."
        python3 main.py --mode fetch
        ;;
    3)
        echo ""
        echo "[3/3] 训练模型..."
        python3 main.py --mode train --force-train
        ;;
    4)
        echo ""
        echo "[3/3] 执行完整流程..."
        python3 main.py --mode full
        ;;
    5)
        echo ""
        echo "[3/3] 启动守护模式..."
        echo "系统将在每日8:00自动执行预测"
        python3 main.py --mode daemon
        ;;
    *)
        echo "无效选项"
        ;;
esac
