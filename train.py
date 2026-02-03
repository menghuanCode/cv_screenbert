
import argparse
from model import ScreenBERT, ScreenBERTConfig
# from data_collator import ScreenCollator
from datasets import load_from_disk
from transformers import Trainer, TrainingArguments
from dataset import ScreenDataset

from transformers import DefaultDataCollator



# ========== 自定义 Trainer（类名改为 ScreenBERTTrainer，避免冲突）==========
class ScreenBERTTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **keyargs):
        # 安全弹出标签
        action_labels = inputs.pop("action_labels", None)
        target_labels = inputs.pop("target_labels", None)

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
    config =ScreenBERTConfig()
    model = ScreenBERT(config)


    # 2. 用 ScreenDataset（它内部调用 model.prepare_inputs）
    train_ds = ScreenDataset("data/arrow", split="train", model=model)
    val_ds = ScreenDataset("data/arrow", split="validation", model=model)

    
    print(f"Available splits: {list(dataset_dict.keys())}")
    for key in dataset_dict.keys():
        print(f"  {key}: {len(dataset_dict[key])} samples")

    # 数据集
    train_ds = dataset_dict["train"]
    val_ds = dataset_dict.get("validation") or dataset_dict.get("test") # 可选

    if len(train_ds) == 0:
        raise ValueError("训练集为空！")


    print("\n===== 测试 DataCollator =====")
    collator = ScreenCollator()

    # 取前 2 条样本测试
    sample_batch = [train_ds[i] for i in range(min(2, len(train_ds)))]
    print(f"原始样本数: {len(sample_batch)}")
    print(f"原始样本 keys: {sample_batch[0].keys() if sample_batch else '空'}")

    # 测试 collator
    processed = collator(sample_batch)
    print(f"处理后类型: {type(processed)}")
    print(f"处理后 keys: {processed.keys() if isinstance(processed, dict) else 'N/A'}")
    print(f"处理后 batch size: {len(processed.get('input_ids', [])) if isinstance(processed, dict) else 'N/A'}")
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