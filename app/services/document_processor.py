# app/services/document_processor.py
from app.db.session import get_session
from app.db.models import Document
from app.utils.parser import extract_text_from_pdf, chunk_text_by_headers
from app.core.es_client import es
from app.core.config import settings
from app.core.logger import get_logger
from app.models.model_manager import get_embedding
from elasticsearch.helpers import bulk
import traceback

logger = get_logger(__name__)


def process_document_background(document_id: int):
    """
    后台文档处理流水线：解析全文 -> 智能分块 -> 向量化 -> 存入 ES
    """
    with get_session() as session:
        doc = session.query(Document).filter(Document.document_id == document_id).first()
        if not doc:
            logger.error(f"找不到 document_id={document_id}")
            return

        try:
            doc.process_status = "processing"
            session.commit()
            logger.info(f"🚀 开始处理文档: {doc.title} (ID: {doc.document_id})")

            # 1. 提取文本
            full_text = extract_text_from_pdf(doc.file_path)
            if not full_text.strip():
                raise ValueError("提取的文本为空，可能是空白文档或解析失败。")

            # 2. 智能分块
            chunks = chunk_text_by_headers(full_text)

            # 3. 准备写入 ES 的数据 (包含文本和向量)
            actions = []
            for i, chunk in enumerate(chunks):
                # 🌟 调用模型，把文字变成向量！
                vector = get_embedding(chunk["content"])[0].tolist()

                chunk_doc = {
                    "_index": settings.es.index_chunk_info,
                    "_source": {
                        "document_id": doc.document_id,
                        "knowledge_id": doc.knowledge_id,
                        "page_number": 1,  # 这里简化处理为1
                        "chunk_id": i,
                        "breadcrumb": chunk["breadcrumb"],
                        "chunk_content": chunk["content"],
                        "embedding_vector": vector  # 存入向量字段！
                    }
                }
                actions.append(chunk_doc)

            # 4. 批量写入 Elasticsearch
            success, failed = bulk(es, actions)
            logger.info(f"✅ 成功向 ES 写入 {success} 个分块。")

            # 5. 更新状态
            doc.process_status = "completed"
            logger.info(f"🎉 文档处理成功: {doc.title}")

        except Exception as e:
            logger.error(f"❌ 处理文档失败: {traceback.format_exc()}")
            doc.process_status = "failed"
            doc.error_msg = str(e)
        finally:
            session.commit()