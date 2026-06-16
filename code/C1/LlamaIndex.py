"""
LlamaIndex.py
功能描述：索引构建-向量数据库-LlamaIndex示例
作者：chb
创建时间：2026/6/13
"""
from llama_index.core import VectorStoreIndex, Document, Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# 1.配置全局嵌入模型

Settings.embed_model = HuggingFaceEmbedding("BAAI/bge-small-zh-v1.5")


# 2.创建示例文档
texts = [
    "张三是法外狂徒",
    "FAISS是一个用于高效相似性搜索和密集向量聚类的库。",
    "LangChain是一个用于开发由语言模型驱动的应用程序的框架。",
    "石宇航是diao毛"
]
docs = [Document(text=t) for t in texts]


# 3.创建索引并持久化到本地
index = VectorStoreIndex.from_documents(docs)
persist_path = "llamaindex_index_store"
index.storage_context.persist(persist_dir=persist_path)
print(f"LlamaIndex 索引已保存至: {persist_path}")





