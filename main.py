# main.py
import time
import os
import uuid
import shutil
import uvicorn
from fastapi import FastAPI, UploadFile, File, Form, BackgroundTasks, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.core.config import settings
from app.core.logger import get_logger
from app.db.session import get_session
from app.db.models import KnowledgeBase, Document
from app.api.schemas import (
    KnowledgeBaseCreateRequest, KnowledgeBaseResponse,
    DocumentResponse, RAGRequest, RAGResponse, ChatMessage
)
from app.services.document_processor import process_document_background
from app.services.qa_service import chat_with_knowledge_base

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

# ============================================================
# 【P0修改1】新增：允许上传的文件类型白名单 + 文件大小限制
#
# 原来的问题：
#   接口没有任何文件校验，用户可以上传任意类型、任意大小的文件。
#   上传一个 500MB 的视频，系统会傻乎乎地存下来然后尝试解析，
#   最终解析失败，浪费了服务器存储和处理资源。
#
# 解决：
#   1. 只允许 PDF 和 Word 文档（政务场景的主要文档格式）
#   2. 文件大小限制 50MB，超过直接拒绝
# ============================================================
ALLOWED_CONTENT_TYPES = {
    "application/pdf",
    "application/msword",                                                    # .doc
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document"  # .docx
}
MAX_FILE_SIZE_BYTES = 50 * 1024 * 1024  # 50MB


# ============================================================
# 【P0修改2】新增：全局异常处理器
#
# 原来的问题：
#   任何未被捕获的异常，FastAPI 会返回默认的 500 响应：
#   {"detail": "Internal Server Error"}
#   这个格式和我们正常接口的返回格式（response_code, response_msg）
#   完全不一样，前端需要写两套错误处理逻辑，非常麻烦。
#   而且没有 request_id，出了问题很难定位是哪个请求。
#
# 解决：
#   注册全局异常处理器，把所有未捕获异常都转换成统一格式，
#   同时记录日志，方便排查问题。
# ============================================================
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    request_id = str(uuid.uuid4())
    logger.error(
        f"未捕获的异常 | request_id={request_id} | "
        f"path={request.url.path} | error={exc}",
        exc_info=True  # 打印完整堆栈
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


# ============================================================
# 【P0修改3】新增：HTTPException 也统一格式
#
# 原因：HTTPException（如 404 知识库不存在）的默认响应格式是
#   {"detail": "知识库不存在"}
#   同样和我们的统一格式不一致，一并处理。
# ============================================================
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
# 1. 知识库管理接口（无修改，原逻辑正确）
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
# 2. 文档上传与解析接口
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

    # ============================================================
    # 【P0修改4】新增文件类型校验
    #
    # 原来的问题：没有任何校验，任意文件都能上传。
    # 解决：检查 content_type 是否在白名单里，不在直接返回 400。
    #
    # 为什么用 content_type 而不是文件扩展名？
    # 扩展名可以伪造（把 exe 改成 pdf），content_type 由浏览器/客户端
    # 根据文件内容判断，相对更可靠。但也不是100%安全，
    # 生产环境还应该用 python-magic 读取文件头来做二次验证。
    # ============================================================
    if file.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"不支持的文件类型：{file.content_type}。只支持 PDF 和 Word 文档。"
        )

    # ============================================================
    # 【P0修改5】新增文件大小校验
    #
    # 原来的问题：没有大小限制，大文件会撑爆服务器存储和内存。
    #
    # 实现方式：先读到内存，检查大小，再写到磁盘。
    # 注意：file.read() 会把指针移到末尾，写文件前要 seek(0) 重置。
    # ============================================================
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

        # 保存文件
        file_extension = os.path.splitext(file.filename)[1]
        file_path = UPLOAD_DIR / f"doc_{doc.document_id}{file_extension}"

        # ============================================================
        # 【P0修改6】用已读取的 file_content 写文件，不再用 copyfileobj
        #
        # 原因：上面已经 file.file.read() 读取了全部内容来检查大小，
        #       文件指针现在在末尾，copyfileobj 会写入空内容。
        #       直接用读取到的 bytes 写入文件。
        # ============================================================
        with open(file_path, "wb") as buffer:
            buffer.write(file_content)

        doc.file_path = str(file_path)
        session.commit()

        background_tasks.add_task(process_document_background, doc.document_id)

        return DocumentResponse(
            response_code=200,
            response_msg="文件上传成功，正在后台解析中，请通过文档查询接口轮询处理状态...",
            processing_time=time.time() - start_time,
            document_id=doc.document_id,
            knowledge_id=knowledge_id,
            title=title,
            category=category,
            file_type=file.content_type,
            process_status="pending"
        )


# ============================================================
# 【P0修改7】新增：文档状态查询接口
#
# 原来的问题：
#   用户上传文档后，返回 process_status="pending"，
#   但没有提供查询状态的接口，用户无法知道文档什么时候解析完成，
#   也不知道有没有失败。
#
# 解决：
#   新增 GET /v1/document/{document_id} 接口，
#   前端可以每隔几秒轮询一次，直到看到 completed 才开始提问。
# ============================================================
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
# 3. RAG 问答接口（逻辑不变，qa_service 已修复）
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
# 4. 健康检查（运维用）
# ==========================================
@app.get("/health", summary="健康检查")
def health_check():
    return {"status": "ok", "version": "1.0.0"}


if __name__ == "__main__":
    logger.info("启动政务/企业级 RAG 问答系统...")
    uvicorn.run(app, host=settings.app.host, port=settings.app.port)