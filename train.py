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

    # 固定随机性，保证可复现
    set_send(42)
    # 加载处理好的数据集（Arrow格式，快）
    train_ds = load_from_disk(os.path.join(args.data, 'train'))         # 训练数据
    val_ds = load_from_disk(os.path.join(args.data, 'validation'))      # 验证数据

    # 3. 初始化模型（从头训练或加载预训练权重）
    model = ScreenBERT.from_pretrained("config/screenbert_base") # 空文件夹即可自动 init
#    ↳ 如果有权重文件：加载预训练参数（微调）
#    ↳ 如果只有config：随机初始化（预训练/从头训练）

    # 显卡优化 + 自动监控
    training_args = TrainingArguments(
        
        # 模型保存路径（如 ckpt/screenbert）
        output_dir=args.output,
        # 训练 3 轮
        num_train_epochs=args.epoch,
        # 每张卡 batch=16
        per_device_train_batch_size=args.batch,
        # 验证时也用 batch=16
        per_device_eval_batch_size=args.batch,
        # 学习效率
        learning_rate=args.lr, # 5e-5


        # 显存优化三板斧（重点）
        # 显存计算逻辑：
        # 名义 Batch：16（单卡能放下的大小）
        # 实际有效 Batch：16 × 4（累积）= 64
        # 等效显存占用：如果你直接设 batch=64 会 OOM，但通过累积实现同样效果
        # 梯度累积：每 4 步更新一次参数
        gradient_accumulation_steps=4,
        # 混合精度：FP16 训练，省 40% 显存
        fp16=args.fp16,
        # 梯度检查点：省 30% 显存，训练慢 20%
        gradient_checkpointing=True,


        # 评估与保存策略
        evaluation_strategy="epoch",            # 每轮结束后评估验证集
        save_strategy="epoch",                  # 每轮结束后保存模型
        load_best_model_at_end=True,            # 训练完自动加载验证 loss 最低的模型
        metric_for_best_model="eval_loss"       # 以验证集 loss 作为选最优标准
        # 自动早停逻辑：配合 early_stopping_patience 可实现早停（需额外配置 callback）
    

        # 日志与监控
        logging_steps=50,                       # 每 50 步（iteration）打印一次 loss                       
        report_to="tensorboard",                # 自动写入 tensorboard 日志
    )

    trainer = Trainer(
        model=model,                            # ScreenBERT 模型实例
        args=training_args,                     # 刚才配置的 TrainingArguments（超参数）
        train_dataset=train_ds,                 # 训练集（ArrowDataset）
        eval_dataset=val_ds,                    # 训练集（ArrowDataset）
        
        # ScreenCollator —— 多模态数据的关键
        # 为什么必须自定义 data_collator 而不是用默认的？
        # 因为 ScreenBERT 是双模态输入（截图 + 文本），默认 collator 只能处理纯文本：
        data_collator=ScreenCollator()          # 关键：自定义数据整理器
    )

    # 开始训练 
    trainer.train()
    trainer.save_model(args.output)
    print("✅ 训练完成，模型已保存到", args.output)
    

    if __name__ == "__main__":
        main()