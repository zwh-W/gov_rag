# app/core/es_client.py
"""
Elasticsearch 客户端

设计原则：
1. 单例：整个进程共用一个 es 连接
2. 启动时检查连通性，索引不存在则自动创建
3. 索引 mapping 在这里统一定义，es_api / rag_api 不重复写
"""
import traceback
from elasticsearch import Elasticsearch

from app.core.config import settings
from app.core.logger import get_logger

logger = get_logger(__name__)

# ==========================================
# 建立连接
# ==========================================
def _build_es_client() -> Elasticsearch:
    cfg = settings.es
    node = {"host": cfg.host, "port": cfg.port, "scheme": cfg.scheme}

    if cfg.username and cfg.password:
        client = Elasticsearch([node], basic_auth=(cfg.username, cfg.password))
    else:
        client = Elasticsearch([node])

    return client


es = _build_es_client()


# ==========================================
# 索引 Mapping 定义
# ==========================================
def _document_meta_mapping() -> dict:
    """
    document_meta：存文档级别的元信息
    file_name / abstract 用 ik_max_word 分词，支持中文全文检索
    """
    return {
        "mappings": {
            "properties": {
                "document_id":   {"type": "integer"},
                "knowledge_id":  {"type": "integer"},
                "document_name": {
                    "type": "text",
                    "analyzer": "ik_max_word",
                    "search_analyzer": "ik_max_word"
                },
                "file_path":     {"type": "keyword"},
                "abstract": {
                    "type": "text",
                    "analyzer": "ik_max_word",
                    "search_analyzer": "ik_max_word"
                },
            }
        }
    }


def _chunk_info_mapping() -> dict:
    """
    chunk_info：存每个 chunk 的内容和向量
    chunk_content 支持 BM25 全文检索
    embedding_vector 支持 kNN 向量检索
    """
    dims = settings.embedding_model_params[settings.rag.embedding_model]["dims"]
    return {
        "mappings": {
            "properties": {
                "document_id":  {"type": "integer"},
                "knowledge_id": {"type": "integer"},
                "page_number":  {"type": "integer"},
                "chunk_id":     {"type": "integer"},
                # 面包屑路径：记录 chunk 来自哪个章节，如"第二章>第三条"
                # 这是你要补充的分块策略里需要用到的字段
                "breadcrumb":   {"type": "keyword"},
                "chunk_content": {
                    "type": "text",
                    "analyzer": "ik_max_word",
                    "search_analyzer": "ik_max_word"
                },
                "embedding_vector": {
                    "type": "dense_vector",
                    "element_type": "float",
                    "dims": dims,
                    "index": True,
                    "index_options": {"type": "int8_hnsw"}
                }
            }
        }
    }


def init_es() -> bool:
    """
    检查 ES 连通性，创建索引（如果不存在）
    返回 True 表示初始化成功
    """
    if not es.ping():
        logger.error("无法连接到 Elasticsearch，请检查服务是否启动")
        return False

    logger.info("Elasticsearch 连接成功")

    index_configs = [
        (settings.es.index_document_meta, _document_meta_mapping()),
        (settings.es.index_chunk_info,    _chunk_info_mapping()),
    ]

    for index_name, mapping in index_configs:
        try:
            if not es.indices.exists(index=index_name):
                es.indices.create(index=index_name, body=mapping)
                logger.info(f"创建索引：{index_name}")
            else:
                logger.info(f"索引已存在，跳过创建：{index_name}")
        except Exception:
            logger.error(f"创建索引 {index_name} 失败：\n{traceback.format_exc()}")
            return False

    return True