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
    def __init__(self, data_path, split, model):
        """
        Args:
            data_path: Arrow 数据集路径（包含 train/ validation 目录）
            split: "train" 或 "validation"
            model: ScreenBERT 实例，用于调用 prepare_inputs
        """

        self.model = model

        # 从 Arrow 加载
        dataset_dict = load_from_disk(data_path)
        if split not in dataset_dict:
            available = list(dataset_dict.keys())
            raise ValueError(f"Split '{split}' 不存在，可用：{available}")

        self.dataset = dataset_dict[split]
        print(f"[{split}] 加载了 {len(self.dataset)} 条样本")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
        item = self.dataset[idx]

        # ===== 调试：看看 dom 的结构 =====
        # print(f"\n--- 样本 {idx} ---")
        # print(f"dom type: {type(item['dom'])}")
        # print(f"dom content: {item['dom'][:200] if isinstance(item['dom'], str) else item['dom']}")
        # =================================

        png = item["png"]
        dom_data = item["dom"]

        # 解包 dom 结构
        if isinstance(dom_data, dict) and 'dom' in dom_data:
            dom_list = dom_data["dom"]  # 真正的元素列表在这里
        elif isinstance(dom_data, list):
            dom_list = dom_data
        else:
            return ValueError(f"未知的 dom 格式：{type(dom_data)}")

        # 调用model 的预处理（单条）
        # 假设 prepare_inputs 返回的是 dict，每个 value 是 [1, ...] 的 tensor
        inputs = self.model.prepare_inputs(item["png"], dom_list)

        # 组装输出
        result = {
            "pixel_values":inputs["pixel_values"].squeeze(0),             # [3, 224, 224]
            "input_ids":inputs["input_ids"].squeeze(0),                   # [512]
            "attention_mask":inputs["attention_mask"].squeeze(0),         # [512]
        }

        # 标签
        if "action" in item:
            result["action_labels"] = torch.tensor(item["action"], dtype=torch.long)
        if "target" in item:
            result["target_labels"] = torch.tensor(item["target"], dtype=torch.long)

        return result
