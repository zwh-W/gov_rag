# app/services/document_processor.py
import traceback
from elasticsearch.helpers import bulk

from app.db.session import get_session
from app.db.models import Document
from app.utils.parser import extract_text_from_pdf, chunk_text_by_headers
from app.core.es_client import es
from app.core.config import settings
from app.core.logger import get_logger
from app.models.model_manager import get_embedding

logger = get_logger(__name__)


def process_document_background(document_id: int):
    """
    后台文档处理流水线：解析全文 -> 智能分块 -> 向量化 -> 存入 ES

    【流水线各步骤说明】
    Step1: 从数据库读取文档信息，更新状态为 processing
    Step2: 解析 PDF，提取纯文本（含 OCR 兜底）
    Step3: 按政务文档结构（章/节/条）智能分块
    Step4: 批量向量化所有 chunk
    Step5: 批量写入 Elasticsearch
    Step6: 更新数据库状态为 completed / failed
    """

    # ============================================================
    # 【P0修改1】把"读取文档信息"和"耗时的处理工作"拆成两个独立的 session
    #
    # 原来的问题：
    #   整个函数只用了一个 with get_session() as session，
    #   也就是说从"开始解析PDF"到"写入ES完成"，这个 session（数据库连接）
    #   一直被占用，可能持续几十秒。
    #
    #   SQLite 在写操作时会锁表，这几十秒里，
    #   其他用户的请求（比如创建知识库）全部阻塞，服务看起来卡死。
    #
    # 解决：
    #   Session1（很短）：只做"读信息 + 改状态为processing"，立刻关闭释放锁。
    #   中间阶段（可能很长）：解析、分块、向量化，不占用数据库连接。
    #   Session2（很短）：只做"改状态为completed/failed"，立刻关闭。
    # ============================================================

    # ── Session 1：读取文档信息，更新状态为 processing ──────────────
    file_path = None
    knowledge_id = None
    doc_title = None

    try:
        with get_session() as session:
            doc = session.query(Document).filter(
                Document.document_id == document_id
            ).first()

            if not doc:
                logger.error(f"找不到 document_id={document_id}，任务终止")
                return

            # 把需要的信息取出来，存到局部变量
            # 原因：session 关闭后，doc 对象会变成"游离态"，
            #       访问它的属性会报 DetachedInstanceError
            file_path = doc.file_path
            knowledge_id = doc.knowledge_id
            doc_title = doc.title

            doc.process_status = "processing"
            # session.__exit__ 时自动 commit，不需要手动调用
        # ← with 块结束，session 关闭，数据库锁释放

        logger.info(f"开始处理文档: {doc_title} (ID: {document_id})")

    except Exception as e:
        logger.error(f"读取文档信息失败: {traceback.format_exc()}")
        return

    # ── 主处理阶段：耗时操作，不持有 session ─────────────────────────
    try:
        # Step1：解析 PDF
        logger.info(f"Step1/4 解析 PDF: {file_path}")
        full_text = extract_text_from_pdf(file_path)

        if not full_text.strip():
            raise ValueError("提取的文本为空，可能是空白文档或扫描件 OCR 失败")

        # Step2：智能分块
        logger.info("Step2/4 智能分块...")
        chunks = chunk_text_by_headers(full_text)

        if not chunks:
            raise ValueError("分块结果为空，请检查文档格式")

        # Step3：批量向量化
        # ============================================================
        # 【P0修改2】改为批量向量化，而不是逐条调用 get_embedding
        #
        # 原来的写法：
        #   for chunk in chunks:
        #       vector = get_embedding(chunk["content"])[0]   ← 每个 chunk 单独调一次
        #
        # 问题：
        #   如果有100个 chunk，就调100次模型推理。
        #   批量推理（一次传入100条文本）比单条推理快3-5倍，
        #   因为 GPU/CPU 可以并行计算矩阵，而单条推理有大量固定开销。
        #
        # 解决：把所有 chunk 的文本收集成列表，一次性传给 get_embedding。
        # ============================================================
        logger.info(f"Step3/4 批量向量化 {len(chunks)} 个分块...")
        all_texts = [chunk["content"] for chunk in chunks]
        all_vectors = get_embedding(all_texts)  # shape: (n_chunks, dims)

        # Step4：准备 ES 批量写入数据
        logger.info("Step4/4 写入 Elasticsearch...")
        actions = []
        for i, (chunk, vector) in enumerate(zip(chunks, all_vectors)):
            actions.append({
                "_index": settings.es.index_chunk_info,
                "_source": {
                    "document_id": document_id,
                    "knowledge_id": knowledge_id,
                    # ============================================================
                    # 【P0修改3】page_number 从 chunk 里读取，不再硬编码为 1
                    #
                    # 原来的问题：
                    #   "page_number": 1,  # 这里简化处理为1
                    #   所有 chunk 的溯源都指向第1页，答案溯源功能完全失效。
                    #
                    # 解决：
                    #   parser.py 的 chunk_text_by_headers 已经在每个 chunk 里
                    #   记录了正确的 page_number，这里直接读取即可。
                    # ============================================================
                    "page_number": chunk.get("page_number", 1),
                    "chunk_id": i,
                    "breadcrumb": chunk["breadcrumb"],
                    "chunk_content": chunk["content"],
                    "embedding_vector": vector.tolist(),
                }
            })

        # 批量写入，比逐条写入快很多
        success_count, failed_items = bulk(es, actions, raise_on_error=False)
        logger.info(f"ES 写入完成：成功 {success_count} 条，失败 {len(failed_items)} 条")

        if failed_items:
            logger.warning(f"部分写入失败：{failed_items[:3]}")  # 只打印前3条，避免日志爆炸

        # ── Session 2：更新状态为 completed ──────────────────────────
        with get_session() as session:
            doc = session.query(Document).filter(
                Document.document_id == document_id
            ).first()
            if doc:
                doc.process_status = "completed"
        logger.info(f"文档处理成功: {doc_title} (ID: {document_id})")

    except Exception as e:
        logger.error(f"处理文档失败: {traceback.format_exc()}")

        # ── Session 2（失败分支）：更新状态为 failed ─────────────────
        try:
            with get_session() as session:
                doc = session.query(Document).filter(
                    Document.document_id == document_id
                ).first()
                if doc:
                    doc.process_status = "failed"
                    # error_msg 截断到500字，避免超出数据库字段长度
                    doc.error_msg = str(e)[:500]
        except Exception as db_err:
            # 更新状态也失败了，只能记日志，不能让异常往外抛
            logger.error(f"更新失败状态时也报错了: {db_err}")