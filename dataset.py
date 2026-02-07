"""
ScreenBERT 数据集处理
负责从 Arrow 格式加载数据，并将原始数据转换为模型输入格式
"""
import os
import json
from typing import Dict, List

import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
from datasets import load_from_disk
from transformers import AutoTokenizer
from collections import Counter
import pyarrow.ipc as ipc  # 添加到文件顶部

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
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese")

        
        # 加载数据
        self._load_arrow(data_path, split)
        self.statistics()
        self.validate_targets()

    def _load_arrow(self, data_path: str, split: str):
        """加载 Arrow 数据"""
        # 直接读取 Arrow 文件，而不是目录
        arrow_file = os.path.join(data_path, f"{split}.arrow")
        
        if not os.path.exists(arrow_file):
            raise FileNotFoundError(f"找不到文件: {arrow_file}")
        
        # 使用 pyarrow 读取
        table = ipc.open_file(arrow_file).read_all()
        
        # 转换为字典列表
        self.data = []
        for i in range(len(table)):
            row = {
                'screenshot': table.column('screenshot')[i].as_py(),
                'dom_elements': table.column('dom_elements')[i].as_py(),
                'target': table.column('target')[i].as_py(),
                'action': table.column('action')[i].as_py(),
                'instruction': table.column('instruction')[i].as_py(),
                'url': table.column('url')[i].as_py(),
            }
            self.data.append(row)
        
        print(f"[{split}] 加载 {len(self.data)} 条")

    def statistics(self):
        """统计基本信息"""
        if len(self.data) == 0:
            print("警告: 没有数据")
            return

        # 统计 action
        actions = [item.get("action", -1) for item in self.data]
        action_counter = Counter(actions)
        print(f"动作分布: {dict(action_counter)}")

        # 统计 target
        targets = []
        for item in self.data:
            t = item.get("target", 0)
            try:
                targets.append(int(t))
            except:
                targets.append(0)

        target_counter = Counter(targets)
        print(f"\nTarget 分布:")
        for t, c in sorted(target_counter.items()):
            print(f"  {t}: {c} ({c/len(targets)*100:.1f}%)")

        # 检查 DOM 长度
        dom_lens = []
        for item in self.data:
            dom = self._parse_dom(item.get("dom_elements", "[]"))
            dom_lens.append(len(dom))

        print(f"\nDOM 长度: min={min(dom_lens)}, max={max(dom_lens)}, avg={sum(dom_lens)/len(dom_lens):.0f}")

        # 显示第一个样本
        sample = self.data[0]
        dom = self._parse_dom(sample.get("dom_elements", "[]"))
        target = int(sample.get("target", 0))

        print(f"\n样本示例:")
        print(f"  instruction: {sample.get('instruction', '')}")
        print(f"  target: {target}")
        print(f"  DOM长度: {len(dom)}")

        if 0 <= target < len(dom):
            elem = dom[target]
            print(f"  target元素: [{elem.get('t')}] {elem.get('selector', '')[:40]}")
        else:
            print(f"  ⚠️ target={target} 超出范围!")


    def validate_targets(self):
        """验证所有 target 有效性"""
        invalid = 0
        
        for i, item in enumerate(self.data):
            dom = self._parse_dom(item.get("dom_elements", "[]"))
            target = int(item.get("target", 0))
            
            if target < 0 or target >= len(dom):
                print(f"样本 {i}: target={target} 超出 DOM 范围 {len(dom)}")
                invalid += 1
            else:
                elem = dom[target]
                # 检查元素是否有效
                if not elem.get('visible', True) or elem.get('t') in ['SCRIPT', 'STYLE']:
                    print(f"样本 {i}: target={target} -> 不可见元素 [{elem.get('t')}]")
                    invalid += 1
        
        print(f"\n无效 target: {invalid}/{len(self.data)} ({invalid/len(self.data)*100:.1f}%)")
        return invalid

    def _parse_dom(self, dom_data) -> List[dict]:
        """解析 DOM 数据"""
        if isinstance(dom_data, str):
            return json.loads(dom_data)
        elif isinstance(dom_data, list):
            return dom_data
        elif isinstance(dom_data, dict):
            return dom_data.get("dom", [])
        return []

    def _load_image(self, image_path: str) -> torch.Tensor:
        """从文件路径加载图片"""
        try:
            # 你的 screenshot 字段保存的是路径
            image = Image.open(image_path).convert("RGB")
            image = image.resize((self.image_size, self.image_size))
            
            # [H, W, C] -> [C, H, W]
            img_array = np.array(image).astype(np.float32) / 255.0
            img_tensor = torch.from_numpy(img_array).permute(2, 0, 1)
            
            return img_tensor
        except Exception as e:
            print(f"加载图片失败 {image_path}: {e}")
            return torch.zeros(3, self.image_size, self.image_size)

    def _prepare_text(self, dom_list: List[dict], instruction: str) -> Dict[str, torch.Tensor]:
        """准备文本输入"""
        # 简化 DOM 描述
        dom_desc = []
        for elem in dom_list[:128]:  # 只取前128个元素
            tag = elem.get('t', '')
            selector = elem.get('selector', '')[:30]
            txt = elem.get('txt', '')[:20]
            dom_desc.append(f"{tag}:{selector}:{txt}")
        
        dom_text = " | ".join(dom_desc)
        
        # 组合 instruction 和 DOM
        full_text = f"任务: {instruction} 页面元素: {dom_text}"
        
        # Tokenize
        encoded = self.tokenizer(
            full_text,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        
        return {
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
        }

    def __len__(self):
        return len(self.data)


    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
        """获取单条样本"""
        item = self.data[idx]
        
        # 1. 加载图片（从路径）
        image_path = item.get("screenshot", "")
        pixel_values = self._load_image(image_path)
          
        # 2. 解析 DOM
        dom_list = self._parse_dom(item.get("dom_elements", "[]"))
        
        # 3. 准备文本
        instruction = item.get("instruction", "")
        text_inputs = self._prepare_text(dom_list, instruction)
        
        # 4. 获取标签
        action = int(item.get("action", 0))
        target = int(item.get("target", 0))
  
        if target >= len(dom_list):
            print(f"警告: 样本 {idx} target={target} >= DOM长度={len(dom_list)}, 设为0")
            target = 0
        
        return {
            "pixel_values": pixel_values,
            "input_ids": text_inputs["input_ids"],
            "attention_mask": text_inputs["attention_mask"],
            "action_labels": torch.tensor(action, dtype=torch.long),
            "target_labels": torch.tensor(target, dtype=torch.long),
        }


if __name__ == "__main__":
    dataset = ScreenDataset("data/arrow", "train")
    print(f"\n数据集大小: {len(dataset)}")
    
    # 测试取一个样本
    sample = dataset[0]
    print(f"\n样本张量形状:")
    for k, v in sample.items():
        print(f"  {k}: {v.shape if hasattr(v, 'shape') else v}")