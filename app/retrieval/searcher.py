# app/retrieval/searcher.py

from typing import List, Dict, Any
from app.core.es_client import es
from app.core.config import settings
from app.core.logger import get_logger
from app.models.model_manager import get_embedding, get_rerank_scores

logger = get_logger(__name__)


def reciprocal_rank_fusion(search_results_list: List[List[Dict]], k: int = 60) -> List[Dict]:
    """
    RRF 融合算法
    :param search_results_list: 多个检索渠道的结果列表
    :return: 融合后重新排序的结果列表
    """
    fused_scores = {}
    doc_map = {}  # 用于保存文档的具体内容，方便后续提取

    for results in search_results_list:
        for rank, hit in enumerate(results):
            doc_id = hit["_id"]
            if doc_id not in fused_scores:
                fused_scores[doc_id] = 0
                doc_map[doc_id] = hit["_source"]

            fused_scores[doc_id] += 1 / (k + rank + 1)

    # 排序
    reranked_results = sorted(fused_scores.items(), key=lambda item: item[1], reverse=True)

    # 组装返回格式
    final_docs = []
    for doc_id, score in reranked_results:
        doc_data = doc_map[doc_id]
        doc_data["_id"] = doc_id
        doc_data["rrf_score"] = score
        final_docs.append(doc_data)

    return final_docs


def hybrid_search(query: str, knowledge_id: int) -> List[Dict[str, Any]]:
    """
    执行 ES 混合检索 (BM25 + 向量) -> RRF 融合 -> Rerank 精排
    """
    logger.info(f"开始混合检索，查询: '{query}', 知识库ID: {knowledge_id}")
    index_name = settings.es.index_chunk_info

    # 1. BM25 全文检索 (加上 knowledge_id 过滤)
    bm25_query = {
        "bool": {
            "must": [{"match": {"chunk_content": query}}],
            "filter": [{"term": {"knowledge_id": knowledge_id}}]
        }
    }
    try:
        bm25_res = es.search(index=index_name, query=bm25_query, size=settings.rag.bm25_top_k)
        bm25_hits = bm25_res["hits"]["hits"]
    except Exception as e:
        logger.error(f"BM25 检索失败: {e}")
        bm25_hits = []

    # 2. 向量检索 (调用 model_manager 获取向量)
    try:
        query_vector = get_embedding(query)[0].tolist()  # 获取单句向量并转为 list
        knn_query = {
            "field": "embedding_vector",
            "query_vector": query_vector,
            "k": settings.rag.vector_top_k,
            "num_candidates": 50,
            "filter": {"term": {"knowledge_id": knowledge_id}}
        }
        knn_res = es.search(index=index_name, knn=knn_query, size=settings.rag.vector_top_k)
        knn_hits = knn_res["hits"]["hits"]
    except Exception as e:
        logger.error(f"向量检索失败: {e}")
        knn_hits = []

    # 3. RRF 融合
    fused_docs = reciprocal_rank_fusion([bm25_hits, knn_hits], k=settings.rag.rrf_k)
    logger.info(f"RRF 融合完成，共召回 {len(fused_docs)} 个文档块")

    # 如果配置中不使用 Rerank，直接返回
    if not settings.rag.use_rerank or not fused_docs:
        return fused_docs[:settings.rag.rerank_top_k]

    # 4. Rerank 精排
    logger.info("开始 Rerank 精排...")
    pairs = [[query, doc["chunk_content"]] for doc in fused_docs]

    # 调用 model_manager 进行打分
    scores = get_rerank_scores(pairs)

    # 将分数更新回文档，并按精排分数重新排序
    for i, doc in enumerate(fused_docs):
        doc["rerank_score"] = float(scores[i])

    reranked_docs = sorted(fused_docs, key=lambda x: x["rerank_score"], reverse=True)

    # 取 Top K 并过滤掉分数低于阈值的辣鸡文档
    final_results = []
    for doc in reranked_docs[:settings.rag.rerank_top_k]:
        # threshold 可以在 config.yaml 里配，比如 0.0，低于此分数说明完全不相关
        if doc["rerank_score"] > settings.rag.confidence_threshold:
            final_results.append(doc)

    logger.info(f"Rerank 完成，最终保留 {len(final_results)} 个高质量文档块")
    return final_results