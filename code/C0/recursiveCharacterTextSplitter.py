from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader

loader = TextLoader("../../file/C0/蜂医.txt", encoding="utf-8")
docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=200,
    chunk_overlap=10,
    separators=["\n\n", "\n", "。", "!", "?", "，", ""]
)
chunks = text_splitter.split_documents(docs)

print(f"文本被切分为{len(chunks)}个块")
print("---前5个块内容示例---")
for i, chunk in enumerate(chunks[:5]):
    print("=" * 50)
    # chunk是一个Document对象，需要访问它的.page_content属性来获取文本
    print(f'块{i + 1}(长度:{len(chunk.page_content)}):"{chunk.page_content}"')
