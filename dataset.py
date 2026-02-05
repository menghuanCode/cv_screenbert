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
    def __init__(
        self,
        data_path: str,
        split: str,
        model=None,
        image_size: int = 224,
        max_length: int = 512,
    ):
        """
        Args:
            data_path: Arrow 数据集路径
            split: "train" 或 "validation"
            model: ScreenBERT 实例，用于调用 prepare_inputs
            image_size: 图片缩放尺寸
            max_length: 最大序列长度
        """
        self.model = model
        self.image_size = image_size
        self.max_length = max_length
        
         # 调用 _load_arrow 进行过滤加载
        self._load_arrow(data_path, split)

    def __len__(self):
        return len(self.data)

    def _load_arrow(self, data_path: str, split: str):
        dataset_dict = load_from_disk(data_path)
        raw_dataset = dataset_dict[split]
        
        # 过滤 action=-1，并统计分布
        self.data = []
        action_counts = {}
        for item in raw_dataset:
            action = item.get("action", -1)
            if action != -1:
                self.data.append(item)
                action_counts[action] = action_counts.get(action, 0) + 1
        
        print(f"[{split}] 过滤后 {len(self.data)}/{len(raw_dataset)} 条")
        print(f"动作分布: {action_counts}")

    
    def _load_image(self, png_bytes: bytes) -> torch.Tensor:
        """
        从 bytes 加载图片并转换为 tensor
        """
        # Bytes -> PIL Image
        image = Image.open(io.BytesIO(png_bytes)).convert("RGB")
        
        # Resize
        image = image.resize((self.image_size, self.image_size))
        
        # 转换为 tensor [C, H, W]
        img_array = np.array(image).astype(np.float32) / 255.0
        img_tensor = torch.from_numpy(img_array).permute(2, 0, 1)
        
        return img_tensor

    def _process_dom(self, dom_data: Union[str, dict, list]) -> List[dict]:
            """处理 DOM 数据，统一返回列表格式"""
            if isinstance(dom_data, str):
                dom_data = json.loads(dom_data)
            
            if isinstance(dom_data, dict):
                dom_list = dom_data.get("dom", dom_data)
            elif isinstance(dom_data, list):
                dom_list = dom_data
            else:
                raise ValueError(f"未知的 dom 格式: {type(dom_data)}")
            
            return dom_list

    def _find_target_index(self, dom_list, target_id):
        """根据元素 ID 或 selector 找到索引"""
        if not target_id:
            return 0
        
        # 尝试匹配 id
        for i, elem in enumerate(dom_list):
            if elem.get("id") == target_id:
                return i
        
        # 尝试匹配 selector（去掉 #）
        clean_target = target_id.lstrip("#.")
        for i, elem in enumerate(dom_list):
            elem_id = elem.get("id", "")
            elem_txt = elem.get("txt", "")
            if elem_id == clean_target or clean_target in elem_txt:
                return i
        
        return 0  # 默认第一个


    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
        """获取单条样本"""
        item = self.data[idx]
        
        # 获取 png bytes（Arrow 中直接是 bytes）
        png_bytes = item["png"]
        
        # 加载图片
        try:
            pixel_values = self._load_image(png_bytes)
        except Exception as e:
            print(f"警告: 加载图片失败 idx={idx}, 错误: {e}")
            pixel_values = torch.zeros(3, self.image_size, self.image_size)
        
        # 处理 DOM
        dom_list = self._process_dom(item["dom"])
        
        # 使用 model.prepare_inputs 处理文本
        if self.model is not None and hasattr(self.model, 'prepare_inputs'):
            # prepare_inputs 可能需要调整，先手动处理
            inputs = self._prepare_text_inputs(dom_list)
        else:
            inputs = self._simple_tokenize(dom_list)
        
        # 组装结果
        result = {
            "pixel_values": pixel_values,
            "input_ids": inputs["input_ids"].squeeze(0),
            "attention_mask": inputs["attention_mask"].squeeze(0),
        }
        
        # 添加标签
        action = item.get("action", -1)
        target = item.get("target", "")
        
        # target 转为索引
        if isinstance(target, str) and target:
            target_idx = self._find_target_index(dom_list, target)
        elif isinstance(target, int):
            target_idx = target
        else:
            target_idx = 0
        
        result["action_labels"] = torch.tensor(int(action), dtype=torch.long)
        result["target_labels"] = torch.tensor(target_idx, dtype=torch.long)
        
        return result

    def _prepare_text_inputs(self, dom_list: List[dict]) -> Dict[str, torch.Tensor]:
        """使用 tokenizer 处理 DOM"""
        # 将 DOM 转为文本
        text_parts = []
        for elem in dom_list[:256]:  # 截断
            desc = f"{elem.get('t', '')} {elem.get('id', '')} {elem.get('txt', '')[:20]}"
            text_parts.append(desc.strip())
        
        full_text = " | ".join(text_parts)
        
        # 获取 tokenizer
        if self.model is not None and hasattr(self.model, 'tokenizer'):
            tokenizer = self.model.tokenizer
        else:
            tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese")
        
        encoded = tokenizer(
            full_text,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        
        return {
            "input_ids": encoded["input_ids"],
            "attention_mask": encoded["attention_mask"],
        }


    def _simple_tokenize(self, dom_list: List[dict]) -> Dict[str, torch.Tensor]:
            """简单的 tokenize"""
            text = json.dumps(dom_list, ensure_ascii=False)
            
            tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese")
            encoded = tokenizer(
                text,
                padding="max_length",
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt"
            )
            
            return {
                "input_ids": encoded["input_ids"],
                "attention_mask": encoded["attention_mask"],
            }