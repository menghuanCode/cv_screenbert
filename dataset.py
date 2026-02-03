"""
ScreenBERT 数据集处理
负责从 Arrow 格式加载数据，并将原始数据转换为模型输入格式
"""
import os
import json
import io
from typing import Dict, List, Optional, Union

import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
from datasets import load_from_disk, DatasetDict
from transformers import AutoTokenizer

class ScreenDataset(Dataset):
    def __init__(self, jsonl_path, model):
        """
          Args:
              jsonl_path: 每行是 {"png_path": "...", "dom": [...], "action": 0, "target": 5}
              model: ScreenBERT 实例，用于调用 prepare_inputs
        """

        self.data = []
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    self.data.append(json.loads(line))

        self.model = model
        print(f"加载了 {len(self.data)} 条样本 from {jsonl_path}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        # 加载图片 bytes
        with open(item["png_path"], "rb") as f:
            png_bytes = f.read()

        # 调用model 的预处理（单条）
        # 假设 prepare_inputs 返回的是 dict，每个 value 是 [1, ...] 的 tensor
        inputs = self.model.prepare_inputs(png_bytes, item["dom"])

        return {
            "pixel_values":inputs["pixel_values"].squeeze(0),             # [3, 224, 224]
            "input_ids":inputs["input_ids"].squeeze(0),                   # [512]
            "attention_mask":inputs["attention_mask"].squeeze(0),         # [512]
            "labels":item["action"],                                      # int 0-4
            "target_idx":item["target"]                                   # int 0-255
        }
