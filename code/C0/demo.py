from unstructured.partition.auto import partition
from unstructured.partition.pdf import partition_pdf

# PDF文件路径
pdf_path = "../../file/C0/sample1.pdf"

# 使用Unstructured加载并解析文档
elements = partition_pdf(
    filename=pdf_path,
    # content_type="application/pdf",
    # strategy="hi_res",  # high resolution 高分辨率布局分析
    strategy="ocr_only"  # ocr only 纯光学字符识别
)

# 打印解析结果
print(f"解析完成：{len(elements)}个元素, {sum(len(str(e)) for e in elements)} 字符")

# 统计元素类型

from collections import Counter

types = Counter(e.category for e in elements)
print(f"元素类型：{dict(types)}")

# 显示所有元素
print("\n所有元素：")
for i, elements in enumerate(elements, 1):
    print(f"第{i}个元素  元素类型：{elements.category}, 元素：{elements}")
    print("=" * 60)

#  对比实验结果证明hi_res对于包含文字的pdf文件识别更优秀，能成功保留文档的结构，而orc_only得出的结果会有乱码以及段落错乱等现象
