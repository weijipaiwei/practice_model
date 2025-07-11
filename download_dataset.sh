#!/bin/bash

# HuggingFace数据集自动下载脚本
# 支持断点续传和网络错误自动重试

set -e  # 遇到错误时退出

# 配置参数
DATASET_NAME="YashJain/UI-Elements-Detection-Dataset"
LOCAL_DIR="/slurm/home/yrd/kanlab/zhangchenfeng/program/practice_model/und_sft/dataset"
MAX_RETRIES=1000
RETRY_DELAY=3

echo "🚀 HuggingFace数据集自动下载脚本"
echo "=================================================="
echo "数据集: $DATASET_NAME"
echo "下载目录: $LOCAL_DIR"
echo "最大重试次数: $MAX_RETRIES"
echo "重试间隔: ${RETRY_DELAY}秒"
echo "=================================================="
export HF_ENDPOINT=https://hf-mirror.com

# 检查huggingface-cli是否可用
if ! command -v huggingface-cli &> /dev/null; then
    echo "❌ 错误: 未找到 huggingface-cli 命令"
    echo "请先安装: pip install huggingface_hub"
    exit 1
fi

# 创建下载目录
mkdir -p "$LOCAL_DIR"

# 下载函数
download_dataset() {
    local attempt=$1
    echo ""
    echo "第 $attempt 次尝试下载..."
    
    # 运行下载命令
    if huggingface-cli download \
        --repo-type dataset \
        --resume-download \
        "$DATASET_NAME" \
        --local-dir "$LOCAL_DIR"; then
        echo "✅ 下载成功完成！"
        return 0
    else
        echo "❌ 下载失败 (返回码: $?)"
        return 1
    fi
}

# 主下载循环
for ((attempt=1; attempt<=MAX_RETRIES; attempt++)); do
    if download_dataset $attempt; then
        echo ""
        echo "🎉 数据集下载完成！"
        exit 0
    fi
    
    # 如果不是最后一次尝试，等待后重试
    if [ $attempt -lt $MAX_RETRIES ]; then
        echo "⏳ 等待 ${RETRY_DELAY} 秒后重试..."
        sleep $RETRY_DELAY
    else
        echo "❌ 已达到最大重试次数 ($MAX_RETRIES)，下载失败"
        echo ""
        echo "💥 数据集下载失败，请检查网络连接或稍后重试"
        exit 1
    fi
done 