# 4. 数据批处理
import torch

class ScreenCollator:
    """简单组装已经预处理号的 tensor"""
    def __call__(self, features):
        return {
            "pixel_values": torch.stack([f["pixel_values"] for f in features]),
            "input_ids": torch.stack([f["input_ids"] for f in features]),
            "attention_mask": torch.stack([f["attention_mask"] for f in features]),
            "labels": torch.tensor([f["labels"] for f in features], dtype=torch.long),
            "target_idx": torch.tensor([f["target_idx"] for f in features], dtype=torch.long)
        }