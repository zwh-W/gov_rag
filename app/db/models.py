# app/db/models.py
"""
数据库 ORM 模型

存什么：知识库和文档的元信息（名称、路径、状态等）
不存什么：文档内容和向量——这些在 ES 里

process_status 字段是关键设计：
  pending   → 文档刚上传，还没解析
  processing → 后台正在解析
  completed  → 解析完成，可以检索
  failed     → 解析失败
这样前端可以轮询文档状态，用户知道文档是否可用
"""
import datetime
from sqlalchemy import Column, Integer, String, DateTime, ForeignKey
from sqlalchemy.orm import declarative_base, relationship

Base = declarative_base()


class KnowledgeBase(Base):
    """知识库表：一个知识库包含多个文档"""
    __tablename__ = "knowledge_base"

    knowledge_id = Column(Integer, primary_key=True, autoincrement=True)
    title = Column(String(255), nullable=False, comment="知识库名称")
    category = Column(String(100), nullable=False, comment="知识库类型，如：法规/政策/通知")
    create_dt = Column(DateTime, default=datetime.datetime.now)
    update_dt = Column(DateTime, default=datetime.datetime.now, onupdate=datetime.datetime.now)

    documents = relationship("Document", back_populates="knowledge_base", cascade="all, delete-orphan")

    def __repr__(self):
        return f"<KnowledgeBase id={self.knowledge_id} title={self.title}>"


class Document(Base):
    """文档表：一个文档属于一个知识库"""
    __tablename__ = "document"

    document_id = Column(Integer, primary_key=True, autoincrement=True)
    knowledge_id = Column(Integer, ForeignKey("knowledge_base.knowledge_id"), nullable=False)
    title = Column(String(255), nullable=False, comment="文档标题")
    category = Column(String(100), comment="文档分类")
    file_path = Column(String(500), comment="文件在服务器上的存储路径")
    file_type = Column(String(50), comment="文件 MIME 类型，如 application/pdf")
    # 文档解析是异步的，这个字段记录当前状态
    process_status = Column(String(20), default="pending",
                            comment="pending / processing / completed / failed")
    error_msg = Column(String(500), comment="解析失败时的错误信息")
    create_dt = Column(DateTime, default=datetime.datetime.now)
    update_dt = Column(DateTime, default=datetime.datetime.now, onupdate=datetime.datetime.now)

    knowledge_base = relationship("KnowledgeBase", back_populates="documents")

    def __repr__(self):
        return f"<Document id={self.document_id} title={self.title} status={self.process_status}>"
