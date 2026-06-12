from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader

loader = TextLoader("../../file/C0/蜂医.txt", encoding="utf-8")
docs = loader.load()  # 提取txt的文本内容，包装为langchain的documents文档对象列表，其中 page_content保存正文内容，metadata保存文档来源，路径等信息

text_splitter = CharacterTextSplitter(
    chunk_size=200,  # 每一块目标大小为100个字符
    chunk_overlap=10  # 每一块之间重叠10个字符，用于缓解语义割裂
)

chunks = text_splitter.split_documents(docs)

print(f"文本被切分为{len(chunks)}个块")
print("---前5个块内容示例---")
for i, chunk in enumerate(chunks[:5]):
    print("=" * 50)
    # chunk是一个Document对象，需要访问它的.page_content属性来获取文本
    print(f'块{i + 1}(长度:{len(chunk.page_content)}):"{chunk.page_content}"')
