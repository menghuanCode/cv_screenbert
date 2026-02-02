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

    dataset = Dataset.from_list(data)
    train_test = dataset.train_test_split(test_size=0.1, seed=42)
    arrow_ds = DatasetDict({
        "train": train_test["train"],
        "validation": train_test["test"]
    })
    arrow_ds.save_to_disk(out_dir)
    print(f" 双任务 Arrow 已生成： {out_dir}")

if __name__ == "__main__":
    build()


