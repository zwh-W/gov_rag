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


# main.py (在 upload_document 函数下面添加)

@app.delete("/v1/document/{document_id}", summary="删除指定文档及其所有数据")
def delete_document(document_id: int):
    start_time = time.time()

    with get_session() as session:
        # 1. 从数据库中找到该文档记录
        doc_to_delete = session.query(Document).filter(Document.document_id == document_id).first()

        if not doc_to_delete:
            raise HTTPException(status_code=404, detail=f"文档 ID:{document_id} 不存在")

        # 在删除数据库记录前，先把需要用的信息取出来
        file_path = doc_to_delete.file_path

        # 2. 从 Elasticsearch 中删除所有相关的 chunk
        logger.info(f"正在从 Elasticsearch 删除 document_id={document_id} 的所有分块...")
        try:
            es.delete_by_query(
                index=settings.es.index_chunk_info,
                query={"term": {"document_id": document_id}}
            )
        except Exception as e:
            # 即便 ES 删除失败，也应继续尝试删除文件和数据库记录，并记录错误
            logger.error(f"从 ES 删除 document_id={document_id} 的分块失败: {e}")

        # 3. 从文件系统中删除物理文件
        logger.info(f"正在从文件系统删除物理文件: {file_path}")
        if file_path and os.path.exists(file_path):
            try:
                os.remove(file_path)
            except OSError as e:
                logger.error(f"删除物理文件 {file_path} 失败: {e}")

        # 4. 从数据库中删除文档记录
        logger.info(f"正在从数据库删除 document_id={document_id} 的元数据记录...")
        session.delete(doc_to_delete)
        session.commit()

        return {
            "response_code": 200,
            "response_msg": f"文档 ID:{document_id} 已被彻底删除",
            "processing_time": time.time() - start_time
        }


# main.py (在 create_knowledge_base 函数下面添加)

@app.delete("/v1/knowledge_base/{knowledge_id}", summary="删除指定知识库及其所有文档")
def delete_knowledge_base(knowledge_id: int):
    start_time = time.time()

    with get_session() as session:
        # 1. 找到知识库记录
        kb_to_delete = session.query(KnowledgeBase).filter(KnowledgeBase.knowledge_id == knowledge_id).first()

        if not kb_to_delete:
            raise HTTPException(status_code=404, detail=f"知识库 ID:{knowledge_id} 不存在")

        # 2. 在删除数据库记录前，先收集所有待删除的物理文件路径
        #    因为一旦数据库记录被删，就找不到这些路径了！
        file_paths_to_delete = [doc.file_path for doc in kb_to_delete.documents if doc.file_path]

        # 3. 从 Elasticsearch 中删除该知识库的所有 chunk
        logger.info(f"正在从 Elasticsearch 删除 knowledge_id={knowledge_id} 的所有分块...")
        try:
            es.delete_by_query(
                index=settings.es.index_chunk_info,
                query={"term": {"knowledge_id": knowledge_id}}
            )
        except Exception as e:
            logger.error(f"从 ES 删除 knowledge_id={knowledge_id} 的分块失败: {e}")

        # 4. 循环删除所有物理文件
        logger.info(f"正在删除知识库 {knowledge_id} 下的所有物理文件...")
        for file_path in file_paths_to_delete:
            if os.path.exists(file_path):
                try:
                    os.remove(file_path)
                except OSError as e:
                    logger.error(f"删除物理文件 {file_path} 失败: {e}")

        # 5. 从数据库中删除知识库记录
        #    SQLAlchemy 的 cascade="all, delete-orphan" 会自动删除所有关联的 document 记录！
        logger.info(f"正在从数据库删除 knowledge_id={knowledge_id} 的元数据记录...")
        session.delete(kb_to_delete)
        session.commit()

        return {
            "response_code": 200,
            "response_msg": f"知识库 ID:{knowledge_id} 及其所有内容已被彻底删除",
            "processing_time": time.time() - start_time
        }

if __name__ == "__main__":
    logger.info("🚀 启动政务/企业级 RAG 问答系统服务...")
    # 运行 FastAPI 服务
    uvicorn.run(app, host=settings.app.host, port=settings.app.port)