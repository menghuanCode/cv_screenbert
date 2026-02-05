from datasets import load_from_disk

dataset_dict = load_from_disk('data/arrow')
print(f'Splits: {list(dataset_dict.keys())}')

for split_name, dataset in dataset_dict.items():
    print(f'{split_name}: {len(dataset)} 条样本')
    print(f' 特征： {dataset.features}')
    if len(dataset) > 0:
        print(f' 第一条： {dataset[0].keys()}')     # 只看有哪些字段

# # 查看有哪些 split
# print(f"可用的 split: {list(dataset_dict.keys())}")

# # 访问具体的 split
# train_dataset = dataset_dict['train']

# print(f"训练集大小：{len(train_dataset)}")
# print(f"特征：{train_dataset.features}")
# print(f"第一条样本：{train_dataset[0]}")


