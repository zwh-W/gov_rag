# main.py
import time
import os
import shutil
import uvicorn
from fastapi import FastAPI, UploadFile, File, Form, BackgroundTasks, HTTPException
from fastapi.middleware.cors import CORSMiddleware

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

# 允许跨域，方便前端页面调用
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# 确保上传目录存在
UPLOAD_DIR = settings.base_dir / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


# ==========================================
# 1. 知识库管理接口
# ==========================================
@app.post("/v1/knowledge_base", response_model=KnowledgeBaseResponse, summary="创建知识库")
def create_knowledge_base(req: KnowledgeBaseCreateRequest):
    start_time = time.time()
    with get_session() as session:
        kb = KnowledgeBase(title=req.title, category=req.category)
        session.add(kb)
        session.flush()  # 获取 ID

        return KnowledgeBaseResponse(
            response_code=200,
            response_msg="知识库创建成功",
            processing_time=time.time() - start_time,
            knowledge_id=kb.knowledge_id,
            title=kb.title,
            category=kb.category
        )


# ==========================================
# 2. 文档上传与解析接口 (异步流水线)
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
    with get_session() as session:
        # 验证知识库是否存在
        kb = session.query(KnowledgeBase).filter(KnowledgeBase.knowledge_id == knowledge_id).first()
        if not kb:
            raise HTTPException(status_code=404, detail="知识库不存在")

        # 记录到数据库
        doc = Document(
            knowledge_id=knowledge_id,
            title=title,
            category=category,
            file_type=file.content_type,
            process_status="pending"
        )
        session.add(doc)
        session.flush()  # 获取文档 ID

        # 保存文件到本地
        file_extension = os.path.splitext(file.filename)[1]
        file_path = UPLOAD_DIR / f"doc_{doc.document_id}{file_extension}"
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        doc.file_path = str(file_path)
        session.commit()  # 提交所有更改

        # 🚀 触发后台解析流水线 (PDF解析 -> 智能分块 -> BGE向量化 -> ES存入)
        background_tasks.add_task(process_document_background, doc.document_id)

        return DocumentResponse(
            response_code=200,
            response_msg="文件上传成功，正在后台解析...",
            processing_time=time.time() - start_time,
            document_id=doc.document_id,
            knowledge_id=knowledge_id,
            title=title,
            category=category,
            file_type=file.content_type,
            process_status="pending"
        )


# ==========================================
# 3. 终极问答接口 (RAG Chat)
# ==========================================
@app.post("/chat", response_model=RAGResponse, summary="智能知识库问答")
def chat(req: RAGRequest):
    start_time = time.time()

    # 获取用户最新提问
    if not req.messages:
        raise HTTPException(status_code=400, detail="对话历史不能为空")

    user_query = req.messages[-1].content

    # 🚀 核心调用：检索 + LLM 生成
    answer, sources = chat_with_knowledge_base(req.knowledge_id, user_query, req.messages)

    # 将新的回答加入历史
    new_messages = req.messages + [ChatMessage(role="assistant", content=answer)]

    return RAGResponse(
        response_code=200,
        response_msg="回答生成成功",
        processing_time=time.time() - start_time,
        answer=answer,
        sources=sources,
        messages=new_messages
    )


if __name__ == "__main__":
    logger.info("🚀 启动政务/企业级 RAG 问答系统服务...")
    # 运行 FastAPI 服务
    uvicorn.run(app, host=settings.app.host, port=settings.app.port)