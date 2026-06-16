"""
demo.py
功能描述：索引构建-多模态嵌入
作者：chb
创建时间：2026/6/12
"""
import os
# 设置Hugging Face的镜像源为国内镜像
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
# 导入pytorch深度学习框架
import torch

# 从visual_bge包中导入多模态模型Visualized_BGE
# visual_bge是一个第三方库，用于图文联合嵌入
from visual_bge.visual_bge.modeling import Visualized_BGE


# 实例化模型
# model_name_bge: 指定底层的BERT-like文本编码器，这里使用BAAI发布的英文版bge-base模型
# model_weight: 指定预训练好的多模态权重文件路径（.pth文件），该文件包含了将图像编码器与文本编码器对齐的参数
model = Visualized_BGE(model_name_bge="BAAI/bge-base-en-v1.5", model_weight="../../models/bge/Visualized_base_en_v1.5"
                                                                            ".pth")

# 将模型设置评估模式，会关闭Dropout，BatchNorm等训练专用层
model.eval()

# 使用torch.no_grad()上下文管理器，表示在此范围内的计算不会记录梯度，节省显存并加速推理
with torch.no_grad():
    text_emb = model.encode(text="blue whale")
    img_emb_1 = model.encode(image="../../data/C3/imgs/datawhale01.png")
    multi_emb_1 = model.encode(image="../../data/C3/imgs/datawhale01.png", text="blue whale")
    img_emb_2 = model.encode(image="../../data/C3/imgs/datawhale02.png")
    multi_emb_2 = model.encode(image="../../data/C3/imgs/datawhale02.png", text="blue whale")

# 计算相似度
sim_1 = img_emb_1 @ img_emb_2.T
sim_2 = img_emb_1 @ multi_emb_1.T
sim_3 = text_emb @ multi_emb_1.T
sim_4 = multi_emb_1 @ multi_emb_2.T
sim_5 = text_emb @ img_emb_1.T

print("=== 相似度计算结果 ===")
print(f"纯图像 vs 纯图像: {sim_1}")
print(f"图文结合1 vs 纯图像: {sim_2}")
print(f"图文结合1 vs 纯文本: {sim_3}")
print(f"图文结合1 vs 图文结合2: {sim_4}")
print(f"纯文本 vs 纯图像: {sim_5}")


















