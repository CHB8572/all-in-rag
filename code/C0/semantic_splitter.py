"""
semantic_splitter.py
功能描述：语义分块SemanticChunker
作者：chb
创建时间：2026/6/11
"""
import os

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from langchain_experimental.text_splitter import SemanticChunker
from langchain_community.document_loaders import TextLoader
from langchain_community.embeddings import HuggingFaceEmbeddings

# HuggingFaceEmbeddings 嵌入封装类，用于加载hugging Face的各种嵌入模型，并把文本转为向量（embedding）


embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-zh-v1.5",
    model_kwargs={'device': 'cpu'},
    # encode_kwargs={'normalize': True}
)

text_splitter = SemanticChunker(
    embeddings,
    breakpoint_threshold_type='percentile'  # 选中断点识别方法  percentile(百分位法) standard_deviation(标准差法)
    # interquartile(四分位矩法) gradient(梯度法)
)

loader = TextLoader("../../file/C0/蜂医.txt", encoding='utf-8')
docs = loader.load()  # 获取Document对象

chunks = text_splitter.split_documents(docs)
print(f"文本被分为{len(chunks)}个分块")
for i, chunk in enumerate(chunks[:5]):
    print("=" * 50)
    # chunk是一个Document对象，需要访问它的.page_content属性来获取文本
    print(f'块{i + 1}(长度:{len(chunk.page_content)}):"{chunk.page_content}"')
