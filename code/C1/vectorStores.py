"""
vectorStores.py
功能描述：索引构建-向量数据库
作者：chb
创建时间：2026/6/12
"""


import os
# 在所有 import 之前添加这两行
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "600"  # 增加超时时间，防止大文件下载中断[reference:3]
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document

# 1.示例文本和嵌入模型
texts = [
    "张三是法外狂徒",
    "FAISS是一个用于高效相似性搜索和密集向量聚类的库。",
    "LangChain是一个用于开发由语言模型驱动的应用程序的框架。",
    "石宇航是diao毛"
]
docs = [Document(page_content=t) for t in texts]
embedding = HuggingFaceEmbeddings(model_name="BAAI/bge-small-zh-v1.5")

# 2.创建向量存储并保存到本地
vectorstore = FAISS.from_documents(docs, embedding)

local_faiss_path = './faiss_index_store'
vectorstore.save_local(local_faiss_path)

print(f"FAISS index has been saved to {local_faiss_path}")

# 3.加载索引并执行查询
# 加载时需指定相同的嵌入模型，并允许反序列化
loaded_vectorstore = FAISS.load_local(
    local_faiss_path,
    embedding,
    allow_dangerous_deserialization=True
)

# 相似性搜索
query = "diao毛是谁"
results = loaded_vectorstore.similarity_search(query, k=1)

print(f"\n查询：{query}")
print("相似度最高的文档：")
for doc in results:
    print(f"-{doc.page_content}")
































