#!/bin/bash

# 设置 PYTHONPATH，将当前目录添加到 Python 模块搜索路径
module load gcc/11.3.0
export PYTHONPATH=$(pwd):$PYTHONPATH

# 使用 accelerate 启动训练脚本
accelerate launch --config_file train_flowllm_f8d16/config/accelerate_onegpu.yaml \
    train_flowllm_f8d16/inference_model.py --config train_flowllm_f8d16/config/inference_default.yaml

#  >> /slurm/home/yrd/kanlab/zhangchenfeng/program/diff_gen/train_flowllm/test/output.log 2>&1