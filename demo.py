from sentence_transformers import SentenceTransformer
from elasticsearch import Elasticsearch
from langchain_text_splitters import MarkdownHeaderTextSplitter
from elasticsearch.helpers import bulk
from sentence_transformers import CrossEncoder
import torch

# ==========================================
# 【自动选择】有GPU用GPU，没有用CPU
# ==========================================
if torch.cuda.is_available():
    DEVICE = "cuda"
    print("✅ 使用 GPU (CUDA)")
else:
    DEVICE = "cpu"
    print("⚠️  GPU 不可用，使用 CPU 加速模式")

# 你的本地模型路径
embedding_model_path = r"D:\ai_models\bge-small-zh-v1.5"
rerank_model_path = r"D:\ai_models\BAAI\bge-reranker-base"

# 加载模型（自动最优速度）
embedding_model = SentenceTransformer(
    embedding_model_path,
    device=DEVICE
)
VECTOR_DIMS = embedding_model.get_sentence_embedding_dimension()

# ES 连接
es = Elasticsearch("http://localhost:9200")
INDEX_NAME = "pipeline_test"

# 分块器
headers_to_split_on = [("#", "Chapter"), ("##", "Article")]
markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)

# Rerank 模型
rerank_model = CrossEncoder(
    rerank_model_path,
    device=DEVICE,
    max_length=512
)


def process_pipeline(text_content: str):
    print("🚀 开始执行数据处理流水线...")

    # 1. 分块
    print("  - 步骤1: 智能分块...")
    langchain_chunks = markdown_splitter.split_text(text_content)
    print(f"    分块完成，得到 {len(langchain_chunks)} 个块。")

    # 2. 向量化（自动最优速度）
    print("  - 步骤2: 文本向量化...")
    chunks_to_embed = [chunk.page_content for chunk in langchain_chunks]

    vectors = embedding_model.encode(
        chunks_to_embed,
        normalize_embeddings=True,
        batch_size=64,
        convert_to_numpy=True,
        show_progress_bar=False
    )
    print("    向量化完成。")

    # 3. 写入 ES（只建一次索引）
    print("  - 步骤3: 写入 Elasticsearch...")
    if not es.indices.exists(index=INDEX_NAME):
        mapping = {
            "properties": {
                "content": {"type": "text"},
                "vector": {"type": "dense_vector", "dims": VECTOR_DIMS}
            }
        }
        es.indices.create(index=INDEX_NAME, mappings=mapping)

    # 批量写入
    actions = []
    for i, chunk in enumerate(langchain_chunks):
        action = {
            "_index": INDEX_NAME,
            "_source": {
                "content": chunk.page_content,
                "vector": vectors[i]
            }
        }
        actions.append(action)

    success, _ = bulk(es, actions)
    print(f"    成功写入 {success} 个文档！")
    print("✅ 流水线执行完毕！")


if __name__ == "__main__":
    sample_text = """
# 导言
欢迎使用本系统。
# 第一章 用户指南
## 第一节 注册
用户需要使用邮箱进行注册。
## 第二节 登录
注册后即可登录。
"""
    process_pipeline(sample_text)