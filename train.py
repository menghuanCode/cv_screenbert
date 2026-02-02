import os, datetime, argparse
from datasets import load_from_disk
from transformers import (
    TrainingArguments,
    Trainer,
    set_send
)
from model import ScreenBERT
# 批处理 collator
from data_collator import ScreenCollator

def main():
    # 解析者
    parser = argparse.ArgumentParset()
    # 数据目录
    parser.add_argument("--data", type=str, default="data/arrow")
    # 输出目录
    parser.add_argument("--output", type=str, default="ckpt/screenbert")
    # 训练轮数（epoch）和学习率（lr）是深度学习训练中最关键的超参数，它们的选择依据主要来自模型稳定性、数据特性和资源约束：
    # 训练轮数
    parser.add_argument("--epoch", type=int, default=3)
    # 学习率
    # 量级经验法则（不同场景的典型值）
    # 预训练（Pre-training）: 1e-4  ~ 1e-3
    # 微调（Fine-tuning BERT）: 1e-5 ~ 5e-5  (最常用 2e-5, 3e-5)
    # 训练 from scratch: 1e-3 ~ 1e-2
    # BERT 类模型必须用小学习率（< 5e-5），因为预训练权重已经很好，大 lr 会破坏特征。
    parser.add_argument("--lr", type=float, default=5e-5)
    # 开关型参数，命令行写了 --fp16，值就是True，不写就是False
    parser.add_argument("--fp16", action="store_true", default=False)
    # 真正解析命令行输入、所有参数值都存在 args 这个对象里
    # 方式 1：全用默认值（训练 3 轮，batch 16，lr 5e-5）
    # python train.py

    # 方式 2：自定义参数（训练 10 轮，batch 32，用 fp16）
    # python train.py --epoch 10 --batch 32 --fp16

    # 方式 3：指定数据和输出路径
    # python train.py --data /root/my_data --output /root/ckpt/my_model
    args = parser.parse_args()

    