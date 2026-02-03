# dataset.py
import torch
from torch.utils.data import Dataset

class ScreenDataset(Dataset):
    def __init__(self, jsonl_path, model):
        """
          Args:
              jsonl_path: 每行是 {"png_path": "...", "dom": [...], "action": 0, "target": 5}
              model: ScreenBERT 实例，用于调用 prepare_inputs
        """

        import json
        with open(jsonl_path, "r", encoding="utf-8") as f:
            self.data = [json.loads(line) for line in f]
        self.model = model

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        # 加载图片 bytes
        with open(item["png_path"], "rb") as f:
            png_bytes = f.read()

        # 调用model 的预处理（单条）
        inputs = self.model.prepare_inputs(png_bytes, item["dom"])

        return {
            "pixel_values":inputs["pixel_values"].squeeze(0),             # [3, 224, 224]
            "input_ids":inputs["input_ids"].squeeze(0),                   # [512]
            "attention_mask":inputs["attention_mask"].squeeze(0),         # [512]
            "labels":item["action"],                                      # int 0-4
            "target_idx":item["target"]                                   # int 0-255
        }
