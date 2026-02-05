
import argparse
import torch
from model import ScreenBERT, ScreenBERTConfig
# from data_collator import ScreenCollator
from datasets import load_from_disk
from transformers import Trainer, TrainingArguments
from dataset import ScreenDataset

import os
from transformers import DefaultDataCollator

from collections import Counter



os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HOME"] = "/root/.cache/huggingface"

# export HF_ENDPOINT=https://hf-mirror.com
# export HF_HOME="/root/autodl-tmp/hf_cache"  # 指定缓存目录



# ========== 自定义 Trainer（类名改为 ScreenBERTTrainer，避免冲突）==========
class ScreenBERTTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **keyargs):
        # 安全弹出标签
        action_labels = inputs.pop("action_labels", None)
        target_labels = inputs.pop("target_labels", None)

        # 调试：看看标签是否存在
        if action_labels is None or target_labels is None:
            print(f"警告：action_labels={action_labels}, target_labels={target_labels}")
            # 临时用 0 填充，避免崩溃
            # batch_size = inputs["pixel_values"].shape[0]
            # action_labels = torch.zeros(batch_size, dtype=torch.long, device=inputs["pixel_values"].device)
            # target_labels = torch.zeros(batch_size, dtype=torch.long, device=inputs["pixel_values"].device)

        # 向前传播
        outputs = model(**inputs, action_labels=action_labels, target_labels=target_labels)
        loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]
        return (loss, outputs) if return_outputs else loss

def main():

    # 参数解析
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="data/arrow")
    parser.add_argument("--output", type=str, default="ckpt/screenbert")
    parser.add_argument("--epoch", type=int, default=3)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--fp16", action="store_true")
    args = parser.parse_args()


    # ========== 初始化模型（这里 ScreenBERT 是模型，不是 Trainer）==========
    print("初始化模型...")
    config =ScreenBERTConfig()
    model = ScreenBERT(config)


    # 2. 用 ScreenDataset（它内部调用 model.prepare_inputs）
    print(f"\n加载数据集:{args.data}")
    train_ds = ScreenDataset("data/arrow", split="train", model=model)
    val_ds = ScreenDataset("data/arrow", split="validation", model=model)

    print(f"\n训练集：{len(train_ds)} 条")
    print(f"验证集：{len(val_ds)} 条")
    
    # 加载数据集后，检查标签范围
    print("\n===== 检查标签范围 =====")
    actions = []
    targets = []
    for i in range(min(100, len(train_ds))):  # 抽查100条
        sample = train_ds[i]
        actions.append(sample['action_labels'].item())
        targets.append(sample['target_labels'].item())

    print(f"Action 范围: {min(actions)} ~ {max(actions)}")
    print(f"Target 范围: {min(targets)} ~ {max(targets)}")
    print(f"Action 唯一值: {sorted(set(actions))}")
    print(f"Target 唯一值: {sorted(set(targets))[:20]}...")  # 前20个
    print("========================\n")
    
    
    # 测试一条样本
    print("\n===== 测试 Dataset =====")
    sample = train_ds[0]
    print(f"Sample keys: {sample.keys()}")
    for k, v in sample.items():
        
        if isinstance(v, torch.Tensor):
            print(f"  {k}: shape={v.shape}, dtype={v.dtype}")
        else:
            print(f"  {k}: {type(v)} = {v}")
    print("===== 测试通过 =====\n")


    training_args = TrainingArguments(
        # 模型检查点和日志的保存路径。建议每个实验用独立目录，方便管理和对比
        output_dir=args.output,
        # 完整遍历训练数据的轮数。BERT 类模型通常 2-4 个 epoch 就够了，太多容易过拟合
        num_train_epochs=args.epoch,
        # 每张 GPU 的训练 batch size。结合你的显存大小调整（FP16 约需 ~8-10GB）
        per_device_train_batch_size=args.batch,
        # 验证时的 batch size，通常可以设大一点（验证时不需要梯度，省显存）
        # 显存估算：Batch Size 8 + FP16 + BERT-base，单卡大约需要 6-8GB 显存
        per_device_eval_batch_size=args.batch,

        # 学习率。BERT 微调的经典值，范围通常在 1e-5 到 5e-5 之间。太小收敛慢，太大不稳定
        learning_rate=args.lr,
        # 开启混合精度训练。节省约 40% 显存，速度提升 1.5-2 倍，几乎不损失精度
        fp16=args.fp16,

        # 每 10 步打印一次日志。设太小日志太频繁，设太大看不到训练进展
        logging_steps=10,

        #  注意：evaluation_strategy 和 save_strategy 通常设为相同值，确保保存的模型都有评估分数
        # 每个 epoch 结束时在验证集上评估。可选 "steps"（按步数）或 "no"（不评估）
        evaluation_strategy="epoch" if val_ds else "no",
        # 每个 epoch 结束时在验证集上评估。可选 "steps"（按步数）或 "no"（不评估）
        save_strategy="epoch",

        # 训练结束后自动加载验证集上表现最好的模型
        load_best_model_at_end=bool(val_ds),
        # 用于判断"最佳模型"的指标（这里用 eval_loss，越小越好）
        metric_for_best_model="eval_loss" if val_ds else None,
        # 重要：保留 action_labels/target_labels
        remove_unused_columns=False
    )

    # ========== 训练（使用自定义 Trainer）==========
    trainer = ScreenBERTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        # data_collator=ScreenCollator()
        data_collator=None,      # 因为 Dataset 已经返回 tensor，用默认 collator
    )

    print("开始训练...")
    trainer.train()
    trainer.save_model(args.output)
    print(f"模型已保存：{args.output}")

if __name__ == "__main__":
    main()