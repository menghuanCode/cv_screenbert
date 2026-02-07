#!/bin/bash

echo "开始批量测试..."
count=0

for png in dataset/keyshot/*.png; do
    base=$(basename "$png" .png)
    json="dataset/keyjson/${base}.json"
    
    # 检查 json 是否存在
    if [ ! -f "$json" ]; then
        echo "跳过: $base (无对应 json)"
        continue
    fi
    
    echo "=== 测试 $((++count)): $base ==="
    
    python inference.py \
        --model ckpt/screenbert \
        --png "$png" \
        --dom "$json"
    
    echo "-------------------"
    echo ""
done

echo "完成，共测试 $count 个样本"