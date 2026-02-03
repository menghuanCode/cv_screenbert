# 双头模型
from typing import Optional
import torch, json, io, numpy as np
from PIL import Image


from torchvision import transforms
from transformers import PreTrainedModel, PretrainedConfig, AutoModel, AutoTokenizer
from torchvision.models import resnet50

from dataclasses import dataclass

from transformers.utils import ModelOutput


@dataclass
class ScreenBERTOutput(ModelOutput):
    """自定义双任务输出"""
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None                # 主 logits（动作）
    logits_action: torch.FloatTensor = None         # 给你推理用
    logits_target: torch.FloatTensor = None

# todo 模型配置类
# 继承 PretrainedConfig：获得 .from_pretrained()、.save_pretrained() 等标准方法
class ScreenBERTConfig(PretrainedConfig):
    # 注册模型类型：HuggingFace Hub 识别用，也用于 AutoConfig 自动映射
    model_type = "screenbert"

    def __init__(self, **kwargs):
        # 调用父类初始化，处理通用参数（如 vocab_size、hidden_act 等）
        super().__init__(**kwargs)
        self.hidden_size = 768              # 隐藏层维度（BERT-base 标准）
        self.num_labels = 5                 # 分类头输出维度（5 个动作类别）
        self.max_dom_len = 256              # DOM 元素最大序列长度
        self.dropout = 0.1                  # Dropout 比率
        self.loss_alpha = 0.7               # 动作权重
        self.loss_beta = 0.3                # 目标权重

# todo 这段代码是 ScreenBERT 的模型架构定义，实现了双编码器融合（视觉 + DOM 文本）
class ScreenBERT(PreTrainedModel):
    config_class = ScreenBERTConfig

    def __init__(self, config):
        super().__init__(config)

        # 1. 视觉编码器，ResNet50 -> 768维
        self.vision = resnet50(pretrained=True)
        self.vision.fc = torch.nn.Linear(self.vision.fc.in_features, config.hidden_size)

        # 2. 文本编码器：BERT-base-chinese
        self.dom_encoder = AutoModel.from_pretrained("bert-base-chinese")
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese")

        # 3. 融合和分类
        self.dropout = torch.nn.Dropout(config.dropout)
        self.classifier = torch.nn.Linear(config.hidden_size * 2, config.num_labels)        # 动作头
        self.target_head = torch.nn.Linear(config.hidden_size * 2, config.max_dom_len)      # 元素索引头

        self.post_init()        # ✅ 添加标准初始化

    # todo 从原始数据到模型数据的转换
    def prepare_inputs(self, png_bytes, dom_json):
        """
           Args:
               png_bytes: PNG 图片的二进制数据
               dom_json: 列表，每个元素是 {"txt": "文本", "x": 0, "y": 0, ...}
           Returns:
               dict: 匹配 ScreenBERT.forward 的输入格式
           """
        # ========== 1. 视觉预处理（标准化） ==========
        # 使用 ResNet 标准预处理
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),            # 自动 [0,255] → [0,1] 并调整维度为 [C,H,W]
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],   # ImageNet 均值
                std=[0.229, 0.224, 0.225]     # ImageNet 标准差
            )
        ])

        img = Image.open(io.BytesIO(png_bytes)).convert("RGB")
        pixel_values = transform(img).unsqueeze(0) # [1, 3, 224, 224]

        # ========== 2. DOM 文本预处理 ==========
        # 改进：保留更多结构化信息
        elements = dom_json[:self.config.max_dom_len]        # 截断元素数量

        # 构造带位置标记的文本（示例：<button>登录</button> [0,0,100,50]）
        dom_text = ' '.join([f"<{n.get('tag', 'div')}>{n.get('txt', '')}</>" for n in elements])

        # Tokenize （使用实例的 tokenizer）
        dom_inputs = self.tokenizer(
            dom_text,
            return_tensors="pt",
            max_length=512,             # BERT 标准长度，或 self.config.max_dom_len
            truncation=True,
            padding="max_length"        # 填充到固定长度，方便 batch 处理
        )

        # ============== 3. 返回格式修正（关键！） ========================

        return {
            "pixel_values": pixel_values,
            "input_ids": dom_inputs["input_ids"],
            "attention_mask": dom_inputs["attention_mask"],
            # 可选：返回 bbox 信息(如果模型支持)
            # "bbox": self._process_bbox(elements)
        }

    # todo 实现了 视觉-文本特征融合分类
    def forward(self, pixel_values, input_ids, attention_mask, action_labels=None, target_labels=None):
        """
           Args:
               pixel_values: [B, 3, 224, 224]
               input_ids: [B, seq_len]
               attention_mask: [B, seq_len]
               action_labels: [B] 动作标签（可选）
               target_labels: [B] 目标标签（可选）
           Returns:
               SequenceClassifierOutput: HuggingFace 标准输出格式
           """
        # 视觉特征 [B, 768]
        vis_out = self.vision(pixel_values)

        # 文本特征 [B, 768]
        dom_out = self.dom_encoder(input_ids=input_ids, attention_mask=attention_mask).pooler_output

        # 特征融合 [B, 1536]
        fused = torch.cat([vis_out, dom_out], dim=-1)
        fused = self.dropout(fused)

        # 分类 [B, num_labels]
        logits_action = self.classifier(fused)
        logits_target = self.target_head(fused)


        # 双任务损失
        total_loss = None
        if action_labels is not None and target_labels is not None:
            # ✅ 添加范围保护
            # target_idx = torch.clamp(target_idx, 0, self.config.max_dom_len - 1)
            loss_fct = torch.nn.CrossEntropyLoss()

            loss_action = loss_fct(logits_action, action_labels)
            loss_target = loss_fct(logits_target, target_labels)

            # total_loss = loss_action + loss_target      # 可加权：0.7 * loss_action + 0.3 * loss_target
            # ✅ 配置化权重
            total_loss = self.config.loss_alpha * loss_action + self.config.loss_beta * loss_target
        # 计算损失
        return ScreenBERTOutput(
            loss=total_loss,
            logits=logits_action,           # 主 logits，避免 Trainer 警告
            logits_action=logits_action,
            logits_target=logits_target
        )
