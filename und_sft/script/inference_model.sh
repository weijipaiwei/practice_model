#!/bin/bash

# 设置 PYTHONPATH，将当前目录添加到 Python 模块搜索路径
module load gcc/11.3.0
export PYTHONPATH=$(pwd):$PYTHONPATH

python und_sft/inference.py