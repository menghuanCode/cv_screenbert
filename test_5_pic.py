for png in $(ls dataset/keyshot/*.png | head -5); do
    base=$(basename $png .png)
    echo "测试: $base"
    python inference.py \
        --model ckpt/screenbert \
        --png "dataset/keyshot/${base}.png" \
        --dom "dataset/keyjson/${base}.json"
    echo "-------------------"
done