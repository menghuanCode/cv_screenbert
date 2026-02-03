
from model import ScreenBERT, ScreenBERTConfig
from data_collator import ScreenCollator
from datasets import load_from_disk
from transformers import Trainer, TrainingArguments


# 初始化
config =ScreenBERTConfig()
model = ScreenBERT(config)

# 数据集
train_ds = load_from_disk("data/arrow/train")
val_ds = load_from_disk("data/arrow/validation")

training_args = TrainingArguments(
    # 模型检查点和日志的保存路径。建议每个实验用独立目录，方便管理和对比
    output_dir="ckpt/screenbert",
    # 完整遍历训练数据的轮数。BERT 类模型通常 2-4 个 epoch 就够了，太多容易过拟合
    num_train_epochs=3,
    # 每张 GPU 的训练 batch size。结合你的显存大小调整（FP16 约需 ~8-10GB）
    per_device_train_batch_size=8,
    # 验证时的 batch size，通常可以设大一点（验证时不需要梯度，省显存）
    # 显存估算：Batch Size 8 + FP16 + BERT-base，单卡大约需要 6-8GB 显存
    per_device_eval_batch_size=8,

    # 学习率。BERT 微调的经典值，范围通常在 1e-5 到 5e-5 之间。太小收敛慢，太大不稳定
    learning_rate=2e-5,
    # 开启混合精度训练。节省约 40% 显存，速度提升 1.5-2 倍，几乎不损失精度
    fp16=True,

    # 每 50 步打印一次日志。设太小日志太频繁，设太大看不到训练进展
    logging_steps=50,

    #  注意：evaluation_strategy 和 save_strategy 通常设为相同值，确保保存的模型都有评估分数
    # 每个 epoch 结束时在验证集上评估。可选 "steps"（按步数）或 "no"（不评估）
    evaluation_strategy="epoch",
    # 每个 epoch 结束时在验证集上评估。可选 "steps"（按步数）或 "no"（不评估）
    save_strategy="epoch",

    # 训练结束后自动加载验证集上表现最好的模型
    load_best_model_at_end=True,
    # 用于判断"最佳模型"的指标（这里用 eval_loss，越小越好）
    metric_for_best_model="eval_loss"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    data_collator=ScreenCollator()
)

trainer.train()