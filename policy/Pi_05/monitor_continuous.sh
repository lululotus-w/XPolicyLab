#!/bin/bash
# 持续监控训练进度

LOG_FILE="/home/weihonglian/robodojo/RoboDojo/XPolicyLab/policy/Pi_05/train_20260911_190820.log"
MONITOR_INTERVAL=30  # 每30秒检查一次

echo "========================================"
echo "训练监控启动 - $(date)"
echo "日志文件: $LOG_FILE"
echo "监控间隔: ${MONITOR_INTERVAL}秒"
echo "========================================"
echo ""

while true; do
    clear
    echo "========================================"
    echo "训练状态更新 - $(date)"
    echo "========================================"
    echo ""

    # 检查进程
    TRAIN_PID=$(ps aux | grep "train.py pi05_base" | grep -v grep | awk '{print $2}' | head -1)
    if [ -z "$TRAIN_PID" ]; then
        echo "⚠️  训练进程未运行"
        echo ""
        echo "最后 20 行日志:"
        tail -20 "$LOG_FILE"
        break
    else
        echo "✓ 训练进程运行中 (PID: $TRAIN_PID)"
        ps aux | grep $TRAIN_PID | grep -v grep | awk '{print "  CPU:", $3"%", " MEM:", $4"%", " 运行时间:", $10}'
    fi
    echo ""

    # GPU 使用
    GPU_INFO=$(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader 2>/dev/null | grep $TRAIN_PID)
    if [ -n "$GPU_INFO" ]; then
        echo "✓ GPU 使用: $GPU_INFO"
    else
        echo "⏳ GPU 尚未激活（可能在初始化）"
    fi
    echo ""

    # 日志统计
    if [ -f "$LOG_FILE" ]; then
        LINE_COUNT=$(wc -l < "$LOG_FILE")
        FILE_SIZE=$(du -h "$LOG_FILE" | awk '{print $1}')
        echo "📄 日志: $LINE_COUNT 行, $FILE_SIZE"
    fi
    echo ""

    # 训练进度
    STEP_INFO=$(grep -E "step [0-9]+" "$LOG_FILE" | tail -1)
    if [ -n "$STEP_INFO" ]; then
        echo "🚀 训练进度:"
        echo "  $STEP_INFO"
    else
        echo "⏳ 初始化中（尚未开始训练步骤）"
        echo ""
        echo "最新日志（最后 5 行）:"
        tail -5 "$LOG_FILE" | sed 's/^/  /'
    fi
    echo ""

    # 检查错误
    if grep -q "Error\|Traceback\|Exception" "$LOG_FILE"; then
        echo "❌ 发现错误，查看日志:"
        grep -A 3 "Error\|Traceback" "$LOG_FILE" | tail -10
        break
    fi

    echo "========================================"
    echo "下次更新: $MONITOR_INTERVAL 秒后"
    echo "按 Ctrl+C 停止监控"
    echo "========================================"

    sleep $MONITOR_INTERVAL
done
