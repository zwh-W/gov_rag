# main.py
import time
import os
import uuid
import uvicorn
from fastapi import FastAPI, UploadFile, File, Form, BackgroundTasks, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from typing import List

from app.core.config import settings
from app.core.logger import get_logger
from app.db.session import get_session
from app.db.models import KnowledgeBase, Document
from app.api.schemas import (
    KnowledgeBaseCreateRequest, KnowledgeBaseResponse,
    DocumentResponse, RAGRequest, RAGResponse, ChatMessage
)
from app.services.document_processor import process_document_background
from app.services.qa_service import chat_with_knowledge_base, stream_chat_with_knowledge_base

logger = get_logger(__name__)

app = FastAPI(title="政务/企业级 RAG 问答系统", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = settings.base_dir / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

ALLOWED_CONTENT_TYPES = {
    "application/pdf",
    "application/msword",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
}
MAX_FILE_SIZE_BYTES = 50 * 1024 * 1024  # 50MB


# ==========================================
# 全局异常处理（P0已有，不变）
# ==========================================
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    request_id = str(uuid.uuid4())
    logger.error(
        f"未捕获的异常 | request_id={request_id} | "
        f"path={request.url.path} | error={exc}",
        exc_info=True
    )
    return JSONResponse(
        status_code=500,
        content={
            "request_id": request_id,
            "response_code": 500,
            "response_msg": "服务内部错误，请稍后重试",
            "processing_time": 0.0,
        }
    )


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "request_id": str(uuid.uuid4()),
            "response_code": exc.status_code,
            "response_msg": exc.detail,
            "processing_time": 0.0,
        }
    )


# ==========================================
# 1. 知识库管理接口（不变）
# ==========================================
@app.post("/v1/knowledge_base", response_model=KnowledgeBaseResponse, summary="创建知识库")
def create_knowledge_base(req: KnowledgeBaseCreateRequest):
    start_time = time.time()
    with get_session() as session:
        kb = KnowledgeBase(title=req.title, category=req.category)
        session.add(kb)
        session.flush()
        return KnowledgeBaseResponse(
            response_code=200,
            response_msg="知识库创建成功",
            processing_time=time.time() - start_time,
            knowledge_id=kb.knowledge_id,
            title=kb.title,
            category=kb.category
        )


# ==========================================
# 2. 文档上传（不变）
# ==========================================
@app.post("/v1/document", response_model=DocumentResponse, summary="上传文档并后台解析")
def upload_document(
        background_tasks: BackgroundTasks,
        knowledge_id: int = Form(..., description="所属知识库ID"),
        title: str = Form(..., description="文档标题"),
        category: str = Form("default", description="文档分类"),
        file: UploadFile = File(...),
):
    start_time = time.time()

    if file.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"不支持的文件类型：{file.content_type}。只支持 PDF 和 Word 文档。"
        )

    file_content = file.file.read()
    if len(file_content) > MAX_FILE_SIZE_BYTES:
        raise HTTPException(
            status_code=400,
            detail=f"文件过大：{len(file_content) / 1024 / 1024:.1f}MB，最大支持 50MB。"
        )

    with get_session() as session:
        kb = session.query(KnowledgeBase).filter(
            KnowledgeBase.knowledge_id == knowledge_id
        ).first()
        if not kb:
            raise HTTPException(status_code=404, detail="知识库不存在，请先创建知识库")

        doc = Document(
            knowledge_id=knowledge_id,
            title=title,
            category=category,
            file_type=file.content_type,
            process_status="pending"
        )
        session.add(doc)
        session.flush()

        file_extension = os.path.splitext(file.filename)[1]
        file_path = UPLOAD_DIR / f"doc_{doc.document_id}{file_extension}"

        with open(file_path, "wb") as buffer:
            buffer.write(file_content)

        doc.file_path = str(file_path)
        session.commit()

        background_tasks.add_task(process_document_background, doc.document_id)

        return DocumentResponse(
            response_code=200,
            response_msg="文件上传成功，正在后台解析中...",
            processing_time=time.time() - start_time,
            document_id=doc.document_id,
            knowledge_id=knowledge_id,
            title=title,
            category=category,
            file_type=file.content_type,
            process_status="pending"
        )


# ==========================================
# 3. 文档状态查询（不变）
# ==========================================
@app.get("/v1/document/{document_id}", response_model=DocumentResponse, summary="查询文档处理状态")
def get_document_status(document_id: int):
    start_time = time.time()
    with get_session() as session:
        doc = session.query(Document).filter(
            Document.document_id == document_id
        ).first()
        if not doc:
            raise HTTPException(status_code=404, detail="文档不存在")

        return DocumentResponse(
            response_code=200,
            response_msg=f"文档状态：{doc.process_status}",
            processing_time=time.time() - start_time,
            document_id=doc.document_id,
            knowledge_id=doc.knowledge_id,
            title=doc.title,
            category=doc.category or "",
            file_type=doc.file_type or "",
            process_status=doc.process_status,
        )


# ==========================================
# 【新增】4. 文档列表接口
#
# 原来的问题：
#   用户无法查看某个知识库里有哪些文档，只能靠记忆。
#   这是基础功能缺失，没有这个接口用户不知道知识库里有什么。
#
# 实现：
#   查询指定知识库下的所有文档，按创建时间倒序排列（最新的在前）。
#   同时返回每个文档的处理状态，用户可以看到哪些文档可用。
# ==========================================
@app.get("/v1/knowledge_base/{knowledge_id}/documents",
         response_model=List[DocumentResponse],
         summary="查询知识库下的所有文档")
def list_documents(knowledge_id: int):
    """
    返回指定知识库下的所有文档列表。
    前端可以用这个接口展示知识库的文档目录。
    """
    with get_session() as session:
        # 先验证知识库存在
        kb = session.query(KnowledgeBase).filter(
            KnowledgeBase.knowledge_id == knowledge_id
        ).first()
        if not kb:
            raise HTTPException(status_code=404, detail="知识库不存在")

        # 查所有文档，按创建时间倒序（最新上传的在最前面）
        docs = session.query(Document).filter(
            Document.knowledge_id == knowledge_id
        ).order_by(Document.create_dt.desc()).all()

        # 组装返回列表
        return [
            DocumentResponse(
                response_code=200,
                response_msg="ok",
                processing_time=0.0,
                document_id=doc.document_id,
                knowledge_id=doc.knowledge_id,
                title=doc.title,
                category=doc.category or "",
                file_type=doc.file_type or "",
                process_status=doc.process_status,
            )
            for doc in docs
        ]


# ==========================================
# 【新增】5. 文档删除接口（同步清理ES数据）
#
# 原来的问题：
#   没有删除接口。即使手动删了数据库记录，
#   ES 里的向量数据还在，会污染所有后续检索结果。
#   用户问问题时会检索到已删除文档的内容，且无法溯源到文档。
#
# 解决：
#   删除时做两件事：
#   1. 删除数据库里的 Document 记录
#   2. 删除 ES 里该文档对应的所有 chunk（按 document_id 过滤删除）
#   3. 删除服务器上的原始文件
#
# 为什么先删ES再删数据库？
#   如果先删数据库记录，ES删除中途失败，
#   数据库里没有记录但ES里还有数据，且无法再次触发清理（没有记录了）。
#   先删ES，失败了数据库记录还在，可以重试。
# ==========================================
@app.delete("/v1/document/{document_id}", response_model=DocumentResponse, summary="删除文档并清理ES数据")
def delete_document(document_id: int):
    """
    删除文档，同时清理 ES 里的所有相关 chunk 向量数据。
    这是保证数据一致性的关键操作。
    """
    start_time = time.time()

    # ── Step1：从数据库查出文档信息（删除前先拿到信息备用）──────────
    with get_session() as session:
        doc = session.query(Document).filter(
            Document.document_id == document_id
        ).first()
        if not doc:
            raise HTTPException(status_code=404, detail="文档不存在")

        # 保存到局部变量，session关闭后doc对象会失效
        doc_title    = doc.title
        doc_category = doc.category or ""
        doc_kt_id    = doc.knowledge_id
        doc_file_type = doc.file_type or ""
        doc_file_path = doc.file_path
        doc_status   = doc.process_status

    # ── Step2：删除 ES 里该文档的所有 chunk ─────────────────────────
    # ============================================================
    # 为什么用 delete_by_query 而不是逐条删？
    #   一个文档可能有几十上百个 chunk，逐条删要发几百次请求。
    #   delete_by_query 一次请求删除所有匹配的文档，效率高得多。
    # ============================================================
    try:
        from app.core.es_client import es
        es_response = es.delete_by_query(
            index=settings.es.index_chunk_info,
            body={
                "query": {
                    "term": {"document_id": document_id}
                }
            },
            # refresh=True 确保删除立即生效，后续检索不会再检索到这些数据
            refresh=True
        )
        deleted_count = es_response.get("deleted", 0)
        logger.info(f"ES 清理完成：文档 {document_id} 共删除 {deleted_count} 个 chunk")
    except Exception as e:
        # ES删除失败，不继续删数据库，记录日志方便排查
        logger.error(f"ES 删除失败，终止删除操作: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"ES 数据清理失败，文档未删除：{str(e)}"
        )

    # ── Step3：删除数据库记录 ────────────────────────────────────────
    with get_session() as session:
        doc = session.query(Document).filter(
            Document.document_id == document_id
        ).first()
        if doc:
            session.delete(doc)
        # session.__exit__ 自动 commit

    # ── Step4：删除服务器上的原始文件（可选，失败不影响主流程）───────
    if doc_file_path and os.path.exists(doc_file_path):
        try:
            os.remove(doc_file_path)
            logger.info(f"原始文件已删除：{doc_file_path}")
        except Exception as e:
            # 文件删除失败不是致命错误，记录日志即可
            logger.warning(f"原始文件删除失败（不影响功能）：{e}")

    logger.info(f"文档删除成功：{doc_title} (ID: {document_id})")

    return DocumentResponse(
        response_code=200,
        response_msg=f"文档删除成功，已清理 ES 向量数据",
        processing_time=time.time() - start_time,
        document_id=document_id,
        knowledge_id=doc_kt_id,
        title=doc_title,
        category=doc_category,
        file_type=doc_file_type,
        process_status=doc_status,
    )


# ==========================================
# 6. RAG 问答接口（不变）
# ==========================================
@app.post("/chat", response_model=RAGResponse, summary="智能知识库问答")
def chat(req: RAGRequest):
    start_time = time.time()

    if not req.messages:
        raise HTTPException(status_code=400, detail="对话历史不能为空")

    user_query = req.messages[-1].content
    if not user_query.strip():
        raise HTTPException(status_code=400, detail="问题内容不能为空")

    answer, sources = chat_with_knowledge_base(req.knowledge_id, user_query, req.messages)
    new_messages = req.messages + [ChatMessage(role="assistant", content=answer)]

    return RAGResponse(
        response_code=200,
        response_msg="回答生成成功",
        processing_time=time.time() - start_time,
        answer=answer,
        sources=sources,
        messages=new_messages
    )


# ==========================================
# 7. 流式问答接口（不变）
# ==========================================
@app.post("/chat/stream", summary="智能知识库问答（流式打字机效果）")
def chat_stream(req: RAGRequest):
    if not req.messages:
        raise HTTPException(status_code=400, detail="对话历史不能为空")

    user_query = req.messages[-1].content
    if not user_query.strip():
        raise HTTPException(status_code=400, detail="问题内容不能为空")

    generator = stream_chat_with_knowledge_base(req.knowledge_id, user_query, req.messages)
    return StreamingResponse(generator, media_type="text/event-stream")


# ==========================================
# 8. 健康检查
# ==========================================
@app.get("/health", summary="健康检查")
def health_check():
    return {"status": "ok", "version": "1.0.0"}


if __name__ == "__main__":
    logger.info("启动政务/企业级 RAG 问答系统...")
    from app.core.es_client import init_es
    init_es()
    uvicorn.run(app, host=settings.app.host, port=settings.app.port)