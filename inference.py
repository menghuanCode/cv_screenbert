
"""
ScreenBERT 推理代码
加载训练好的模型，对单张截图进行预测
"""

import argparse
import json
import os
import torch
from model import ScreenBERT, ScreenBERTConfig
from safetensors.torch import load_file  

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="ckpt/screenbert", help="模型路径")
    parser.add_argument("--png", type=str, required=True, help="截图路径")
    parser.add_argument("--dom", type=str, default=None, help="DOM json 路径（默认同名）")
    args = parser.parse_args()

    # 这段代码展示了 ScreenBERT 的完整单条推理流程，特点是端到端的数据预处理（prepare_inputs）和多模态输入（图片 + DOM 结构）：



    # 1. 加载模型
    print(f"加载模型: {args.model}")
    config = ScreenBERTConfig()
    # 关键：设置 max_dom_len 与训练时一致（1280）
    config.max_dom_len = 1280
    model = ScreenBERT(config)

    
    # 加载权重
    model_path = os.path.join(args.model, "model.safetensors")
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"找不到模型权重: {model_path}")

    print(f"加载权重: {model_path}")

    # safetensors 格式用专门的 loader
    state_dict = load_file(model_path, device="cpu")

    model.load_state_dict(state_dict)
    model.eval()
    print("模型加载完成")

    # 2. 加载图片
    if not os.path.exists(args.png):
        raise FileNotFoundError(f"截图不存在: {args.png}")
    
    with open(args.png, "rb") as f:
        png_bytes = f.read()
    print(f"加载截图: {args.png}")

    
    # 3. 加载 DOM
    dom_path = args.dom or args.png.replace(".png", ".json")
    if not os.path.exists(dom_path):
        raise FileNotFoundError(f"DOM 文件不存在: {dom_path}")
    
    with open(dom_path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)


    # 处理不同格式
    if isinstance(raw_data, list):
        dom_data = raw_data
    elif isinstance(raw_data, dict):
        dom_data = raw_data.get("dom", [])  # 从 "dom" 字段提取
    else:
        dom_data = []
        
    print(f"加载 DOM: {len(dom_data)} 个元素")

    
    # 4. 预处理（使用模型的 prepare_inputs）
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)  # 移到设备
    print(f"模型加载完成，使用设备: {device}")

    inputs = model.prepare_inputs(png_bytes, dom_data)

    
    # 获取截断后的元素列表（用于后续映射）
    # 注意：prepare_inputs 需要返回 elements
    if 'elements' not in inputs:
        # 如果 prepare_inputs 没返回 elements，手动截断
        max_len = config.max_dom_len
        inputs['elements'] = dom_data[:max_len]
    
    
    # 将输入移到相同设备
    inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
              for k, v in inputs.items()}

  # 5. 推理
    with torch.no_grad():
        outputs = model(
            pixel_values=inputs['pixel_values'],
            input_ids=inputs['input_ids'],
            attention_mask=inputs['attention_mask']
        )

    # 6. 解码结果
    logits_action = outputs.logits_action
    logits_target = outputs.logits_target
    
    action_id = logits_action.argmax(-1).item()
    target_id = logits_target.argmax(-1).item()
    
    # 获取置信度
    action_probs = torch.softmax(logits_action, dim=-1)
    target_probs = torch.softmax(logits_target, dim=-1)
    
    action_conf = action_probs.max().item()
    target_conf = target_probs.max().item()

    # 动作映射
    action_map = {0: "click", 1: "type", 2: "scroll", 3: "wait", 4: "download"}
    
    # 获取目标元素（从截断后的列表中）
    elements = inputs.get('elements', dom_data[:config.max_dom_len])
    target_elem = elements[target_id] if 0 <= target_id < len(elements) else None
    
    # 映射回原始索引
    original_idx = target_elem.get('idx', target_id) if target_elem else target_id
    
    result = {
        "action": action_map.get(action_id, f"unknown({action_id})"),
        "action_id": action_id,
        "action_confidence": round(action_conf, 4),
        "target_id": target_id,  # 截断后列表中的位置
        "original_target_idx": original_idx,  # 原始 DOM 中的 idx
        "target_confidence": round(target_conf, 4),
        "target_element": {
            "id": target_elem.get("id", "") if target_elem else "",
            "tag": target_elem.get("t", "") if target_elem else "",
            "text": target_elem.get("txt", "")[:100] if target_elem else "",
            "bbox": {
                "x": target_elem.get("x", 0) if target_elem else 0,
                "y": target_elem.get("y", 0) if target_elem else 0,
                "w": target_elem.get("w", 0) if target_elem else 0,
                "h": target_elem.get("h", 0) if target_elem else 0,
            }
        } if target_elem else None
    }

    print(f"\n预测 target 位置: {target_id} -> 原始 idx: {original_idx}")
    print("\n预测结果:")
    print(json.dumps(result, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()





