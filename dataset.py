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
from collections import Counter

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
        self.statistics()           # 初始化时自动统计
        self.validate_targets()     # 快速检查 target 有效性

    def validate_targets(self):
        """验证所有 target 是否指向有效元素"""
        invalid_count = 0
        
        for i, item in enumerate(self.data):
            target_str = item.get("target", "0")
            try:
                original_idx = int(str(target_str).lstrip('#'))
            except:
                original_idx = 0
            
            dom_list = self._process_dom(item["dom"])
            
            # 找匹配的元素
            matched = None
            for j, elem in enumerate(dom_list):
                if elem.get('idx') == original_idx:
                    matched = elem
                    new_idx = j
                    break
            
            if matched is None:
                print(f"样本 {i}: target={original_idx} 未找到")
                invalid_count += 1
            else:
                # 检查元素质量
                elem_selector  = matched.get('selector', '')
                elem_tag = matched.get('t', '')
                elem_txt = matched.get('txt', '')[:30]
                
                if elem_selector == '' and elem_tag in ['SCRIPT', 'STYLE', '#text']:
                    print(f"样本 {i}: target={original_idx} -> 无效元素 [{elem_tag}] id='{elem_selector}' txt='{elem_txt}...'")
                    invalid_count += 1
                else:
                    print(f"样本 {i}: target={original_idx} -> 新索引 {new_idx} [{elem_tag}] id='{elem_selector}'")
        
        print(f"\n无效 target 比例: {invalid_count}/{len(self.data)} ({invalid_count/len(self.data)*100:.1f}%)")
        return invalid_count

    def statistics(self):
        """统计 target_id 分布"""
        target_ids = []
        missing_target = 0
        
        for sample in self.data:
            target = sample.get('target', '')
            
            # 处理不同格式
            if isinstance(target, str):
                # 去掉 # 前缀，转整数
                clean_target = target.lstrip('#')
                try:
                    target_id = int(clean_target) if clean_target else 0
                except ValueError:
                    target_id = 0  # 转换失败默认0
            elif isinstance(target, int):
                target_id = target
            else:
                target_id = 0
                
            target_ids.append(target_id)
            
            if target_id == 0:
                missing_target += 1
        
        counter = Counter(target_ids)
        
        print("=" * 50)
        print("Target ID 分布统计:")
        print(f"  缺失/无效 target: {missing_target} ({missing_target/len(target_ids)*100:.1f}%)")
        print("-" * 50)
        for target_id, count in sorted(counter.items()):
            percentage = count / len(target_ids) * 100
            marker = " ⚠️" if percentage > 30 and target_id == 0 else ""
            print(f"  ID {target_id:3d}: {count:6d} ({percentage:5.2f}%){marker}")
        print(f"  总计: {len(target_ids)} 个样本")
        print("=" * 50)
        
        # 警告
        if counter[0] / len(target_ids) > 0.5:
            print("\n⚠️ 警告: target=0 占比超过 50%！")
        if len(counter) <= 2:
            print("\n❌ 警告: target 种类太少，模型可能无法有效学习！")
        
        # 检查第一个样本的 dom 结构
        sample = self.data[0]
        dom_list = self._process_dom(sample["dom"])
        print(f"\nDOM 列表长度: {len(dom_list)}")
        print(f"前3个元素: {dom_list[:3]}")
        print(f"target 值: '{sample.get('target')}'")
        print(f"target 类型: {type(sample.get('target'))}")

        # 验证 target 到底是什么
        sample = self.data[0]
        target = sample.get("target")
        dom_list = self._process_dom(sample["dom"])

        print(f"\n验证 Target:")
        print(f"  target 值: {target}")
        print(f"  DOM 长度: {len(dom_list)}")

        # 尝试作为索引
        try:
            idx = int(str(target).lstrip('#'))
            if idx < len(dom_list):
                elem = dom_list[idx]
                print(f"  作为索引 {idx}: 元素 selector={elem.get('selector')}, txt={elem.get('txt', '')[:30]}...")
            else:
                print(f"  索引 {idx} 超出范围")
        except Exception as e:
            print(f"  作为索引失败: {e}")

        # 尝试作为 id 匹配
        matches = [i for i, elem in enumerate(dom_list) if str(elem.get('selector')) == str(target).lstrip('#')]
        print(f"  作为 selector 匹配: 找到 {len(matches)} 个")        
        
        return counter
        
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
            if action == -1:
                continue

            # 复制少数类
            repeat = 10 if action in [0, 1] else 1  # click, type 复制10倍

            for _ in range(repeat):
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

    def _find_target_index(self, dom_list, target):
        """根据元素 selector 找到索引"""
        if not target:
            return 0
        
        # 统一转为字符串比较
        target_str = str(target).lstrip('#.')
        
        # 尝试匹配 id（都转为字符串）
        for i, elem in enumerate(dom_list):
            elem_id = str(elem.get("selector", ""))
            if elem_id == target_str:
                return i
        
        # 尝试匹配 txt 内容
        for i, elem in enumerate(dom_list):
            elem_txt = elem.get("txt", "")
            elem_selector = elem.get("selector", "")
            if target_str in str(elem_txt) or target_str in str(elem_selector):
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

  
        # 只保留有 id 或可见的元素，重新编号
        valid_elements = [(i, e) for i, e in enumerate(dom_list) 
                        if e.get('selector') or e.get('visible')]
        
        # 创建映射：原始 idx -> 新索引
        idx_map = {orig_idx: new_idx for new_idx, (orig_idx, _) in enumerate(valid_elements)}
        
        # 获取原始 target
        raw_target = item.get('target', 0)
        try:
            original_idx = int(str(raw_target).lstrip('#'))
        except:
            original_idx = 0
        
        # 映射到新索引（如果在有效列表中）
        target_idx = idx_map.get(original_idx, 0)
        
        # 截断到 max_dom_len，但用 valid_elements
        elements = [e for _, e in valid_elements[:1280]]
        
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
        target = item.get("target", "")  # 这是字符串 "4"
        
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
            desc = f"{elem.get('t', '')} {elem.get('selector', '')} {elem.get('txt', '')[:20]}"
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