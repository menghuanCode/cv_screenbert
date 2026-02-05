screenbert/
├─ data/arrow/          ← 步骤 1 生成的 Arrow 数据(含 png, dom, label, target_idx）
├─ config/
│   ├─ screenbert_base/ ← 模型配置（空文件夹即可，代码会自动初始化）
├─ build_arrow.py       ← 已给
├─ train.py             ← **全量训练代码（下面）**
├─ inference.py         ← **全量推理代码（下面）**
├─ model.py             ← ScreenBERT 双塔模型（下面）
├─ data_collator.py     ← 批处理 collator（下面）
└─ requirements.txt     ← 依赖清单（下面）

### 训练命令
```bash
export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME="/root/autodl-tmp/hf_cache"  # 指定缓存目录

python train.py --data data/arrow --output ckpt/screenbert --epoch 3 --batch 8 --fp16
```

### 推理命令
```bash
python inference.py \
        --model ckpt/screenbert \
        --png "dataset/keyshot/${base}.png" \
        --dom "dataset/keyjson/${base}.json"
    echo "-------------------"
```
