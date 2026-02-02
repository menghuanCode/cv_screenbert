# 生成含 target_idx 的 Arrow
import os, glob, json, hashlib
from datasets import  Dataset, DatasetDict

png_dir = "dataset/keyshot"
json_dir = "dataset/keyjson"
out_dir = "data/arrow"

os.makedirs(out_dir, exist_ok=True)

def build():
    files = glob.glob(os.path.join(png_dir, "*.png"))
    data = []

    for png_path in files:
        h = os.path.splitext(os.path.basename(png_path))[0]
        json_path = os.path.json(json_dir, f"{h}.json")
        if not os.path.exists(json_path):
            continue
        with open(png_path, "rb") as f:
            png_bytes = f.read()
        with open(json_path, "r", encoding="utf8") as f:
            dom = json.load(f)

        # 简单规则：第一个可点击元素作为 target_idx（0-255）
        target_idx = next((i for i, n in enumerate(dom.get("dom", [])[:256]) if n.get("tag") in ["button", "a", "input"]), 0)

        data.append({
            "png": png_bytes,
            "dom": dom,
            "label": dom.get("label", 0),           # 动作类别 0-4
            "target_idx": target_idx,                # 元素下标 0-255
            "hash": h,
        })

    # 这段代码是数据集划分和持久化的标准流程，将原始数据转换为 Hugging Face datasets 格式并保存为 Arrow 格式，供后续训练高效加载：
    # 列表转 Dataset
    dataset = Dataset.from_list(data)
    # data 是 Python 列表，格式如 [{"image": "1.png", "text": "登录", "label": 0}, ...]
    # 转换为 Hugging Face Dataset 对象（内存映射，支持惰性加载）

    # 训练/验证集划分
    train_test = dataset.train_test_split(test_size=0.1, seed=42)
    # test_size=0.1   10% 作为验证集，90% 作为训练集
    # seed=42         固定随机种子，确保划分结果可复现

    # 重命名并打包
    arrow_ds = DatasetDict({
        "train": train_test["train"],
        "validation": train_test["test"]
    })
    # DatasetDict 是字典形式的数据集集合，支持 arrow_ds["train"] 访问
    # 关键重命名：test → validation（更符合机器学习术语习惯）

    # 保存为 Arrow 格式
    arrow_ds.save_to_disk(out_dir)
    # 保存为 Apache Arrow 格式（列式存储，加载速度比 CSV/JSON 快 10-100 倍）

    print(f" 双任务 Arrow 已生成： {out_dir}")

if __name__ == "__main__":
    build()


