#!/bin/bash
# 训练监控脚本

LOG_FILE="/home/weihonglian/robodojo/RoboDojo/XPolicyLab/policy/Pi_05/train_20260911_184725.log"

echo "=================================="
echo "训练监控 - $(date)"
echo "=================================="
echo ""

# 检查进程
echo "1. 训练进程状态:"
ps aux | grep "train.py pi05_base" | grep -v grep | awk '{print "  PID:", $2, "  CPU:", $3"%", "  MEM:", $4"%", "  运行时间:", $10}'
echo ""

# GPU 使用
echo "2. GPU 使用情况:"
nvidia-smi --query-compute-apps=pid,used_memory,gpu_name --format=csv,noheader | grep 459834 || echo "  未检测到 GPU 使用"
echo ""

# 日志大小和最新行数
echo "3. 日志信息:"
echo "  文件大小: $(du -h $LOG_FILE | awk '{print $1}')"
echo "  总行数: $(wc -l < $LOG_FILE)"
echo ""

# 最新日志（过滤关键信息）
echo "4. 最新进度（最后 20 行）:"
tail -20 $LOG_FILE | grep -E "step|loss|epoch|Downloading|Loading|Error|Traceback" || tail -5 $LOG_FILE
echo ""

# Wandb 链接
echo "5. 可视化链接:"
grep "View run at" $LOG_FILE | tail -1
echo ""

echo "=================================="
echo "提示: 运行 'bash monitor_training.sh' 查看实时状态"
echo "      或访问上面的 wandb 链接查看训练曲线"
echo "=================================="
