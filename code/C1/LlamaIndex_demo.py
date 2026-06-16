"""
LlamaIndex_demo.py
功能描述：实现对LlamaIndex存储数据的加载和相似性搜索
作者：chb
创建时间：2026/6/13
"""
from llama_index.core import StorageContext, load_index_from_storage
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings
# Setting是llamaIndex的全局配置对象

# 1. 设置和创建索引时相同的 embedding 模型
Settings.embed_model = HuggingFaceEmbedding("BAAI/bge-small-zh-v1.5")

# 2. 加载之前保存的 LlamaIndex 索引
persist_path = "llamaindex_index_store"
storage_context = StorageContext.from_defaults(persist_dir=persist_path)

index = load_index_from_storage(storage_context)  # 把persist_path保存的索引真正加载出来。

# 3. 创建检索器：只做相似性搜索，不调用大模型
retriever = index.as_retriever(similarity_top_k=1)

# 4.执行相似性搜索
query = "请介绍一下谁是diao毛"
nodes = retriever.retrieve(query)

# 5. 输出检索结果
print("相似性搜索结果：")
for i, node in enumerate(nodes, start=1):
    print(f"\n===== 第 {i} 条结果 =====")
    print(f"相似度分数：{node.score}")
    print(node.node.get_content())






