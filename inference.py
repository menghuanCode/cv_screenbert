# 推理代码
# 这是推理/预测阶段的代码，用于加载训练好的模型并对单张截图进行预测。相比训练代码，这里更轻量，核心是加载权重 + 单条推理：
import torch
import argparse, json, hashlib
from PIL import Image
from model import ScreenBERT
from transformers import AutoTokenizer

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="ckpt/screenbert")
    parser.add_argument("--png", type=str, required=True, help="单张截图路径")
    args = parser.parse_args()

    # 这段代码展示了 ScreenBERT 的完整单条推理流程，特点是端到端的数据预处理（prepare_inputs）和多模态输入（图片 + DOM 结构）：

    model = ScreenBERT.from_pretrained(args.model)                          
    tokenizer = AutoTokenizer.from_pretrained(args.model)                   

    png_bytes = open(args.png, "rb").read()                            # 二进制图片流
    dom_json = json.load(open(args.png.replace(".png", ".json")))      # 通目录同名json

    # 自定义预处理（ScreenBERT 专属）
    inputs = model.prepare_inputs(png_bytes=png_bytes, dom_json=dom_json)


    with torch.no_grad():                           # 禁用梯度计算，省显存
        outputs = model(**inputs)
        logits_action = outputs.logits_action
        logits_target = outputs.logits_target


    action_id = logits_action.argmax(-1).item()
    target_id = logits_target.argmax(-1).item()


    # 这段代码将动作ID解码为可读动作，并尝试定位操作目标元素

    action_map = { 0: "click", 1: "type", 2: "scroll", 3: "wait", 4: "download" }
    
    result = {
        "action": action_map[action_id],
        "target": dom_json.get("dom", [])[target_id].get("id", "") if 0 <= target_id < len(dom_json.get("dom", [])) else "",
    }

    print(json.dumps(result, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()





