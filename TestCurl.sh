#!/bin/bash

# 定义模型和对应的主机地址
models=("qwen2_72b_instruct_q5_K_M" "llama3_70b_instruct_q5_K_M" "yi_34b_v1_5" "gemma2_27b" "command_r_plus" "deepseek_v2_236b")
urls=("http://13.65.249.11:8880" "http://13.65.249.11:6665" "http://13.65.249.11:8885" "http://13.65.249.11:8889" "http://13.65.249.11:8887" "http://13.65.249.11:6662")

# 遍历所有模型并发起请求
for i in "${!models[@]}"; do
    model="${models[$i]}"
    url="${urls[$i]}"
    echo "Checking status for model $model at $url"

    # 发起curl请求并打印结果
    response=$(curl -s "$url")
    echo "Response from $model:"
    echo "$response"
    echo "----------------------------------------"
done
