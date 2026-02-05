#!/bin/bash

# 遍历所有 png 文件
for png_file in dataset/keyshot/*.png; do
    # 提取文件名（不带路径和扩展名）
    base=$(basename "$png_file" .png)
    
    # 构造对应的 json 路径
    json_file="dataset/keyjson/${base}.json"
    
    # 检查 json 是否存在
    if [ -f "$json_file" ]; then
        echo "处理: $base"
        python inference.py \
            --model ckpt/screenbert \
            --png "$png_file" \
            --dom "$json_file"
        echo "-------------------"
    else
        echo "跳过: $base (找不到对应的 json: $json_file)"
    fi
done