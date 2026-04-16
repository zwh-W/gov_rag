# app/api/schemas.py
"""
API 请求/响应数据结构

设计原则：
1. 所有接口统一返回格式：request_id + response_code + response_msg + processing_time
2. 业务字段放在各自的 Response 里
3. process_status 表示文档处理状态，前端可以用来轮询
"""
import uuid
from enum import Enum
from typing import Union, List, Optional, Tuple

from pydantic import BaseModel, Field, field_validator


# ==========================================
# 通用基类
# ==========================================
class BaseResponse(BaseModel):
    request_id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="请求唯一ID")
    response_code: int = Field(description="200=成功，4xx=客户端错误，5xx=服务端错误")
    response_msg: str = Field(description="响应描述")
    processing_time: float = Field(description="接口耗时（秒）")


# ==========================================
# 知识库
# ==========================================
class KnowledgeBaseCreateRequest(BaseModel):
    title: str = Field(description="知识库名称")
    category: str = Field(description="知识库类型，如：法规/政策/通知")


class KnowledgeBaseResponse(BaseResponse):
    knowledge_id: int = Field(description="知识库ID")
    title: str
    category: str


# ==========================================
# 文档
# ==========================================
class DocumentResponse(BaseResponse):
    document_id: int = Field(description="文档ID")
    knowledge_id: int
    title: str
    category: str
    file_type: str
    process_status: str = Field(description="文档解析状态: pending/processing/completed/failed")


# ==========================================
# Embedding / Rerank
# ==========================================
class EmbeddingRequest(BaseModel):
    text: Union[str, List[str]]
    model: Optional[str] = None


class EmbeddingResponse(BaseResponse):
    vector: List[List[float]]


class RerankRequest(BaseModel):
    text_pair: List[Tuple[str, str]] = Field(description="[(query, chunk), ...]")
    model: Optional[str] = None


class RerankResponse(BaseResponse):
    scores: List[float]


# ==========================================
# RAG 问答
# ==========================================

# ============================================================
# 【P1修改1】新增 MessageRole 枚举，约束 role 字段的合法值
#
# 原来的问题：
#   role: str  ← 可以填任意字符串，没有任何约束
#   课程原代码里把 LLM 的回答存成 role="system"，这是错的。
#   "system" 是给系统提示词用的，LLM 的回答必须是 "assistant"。
#   如果 role 填错，部分大模型 API 会直接报错或行为异常。
#
# 解决：
#   定义枚举类，role 只能填 "user" / "assistant" / "system" 三种。
#   填了别的值，Pydantic 在接收请求时就会报 422 参数校验错误，
#   不会等到调用 LLM 时才出问题。
#
# 三种 role 的含义：
#   user      → 用户发的消息
#   assistant → LLM 回复的消息（注意：不是 system！）
#   system    → 系统提示词，通常只有第一条消息用
# ============================================================
class MessageRole(str, Enum):
    user = "user"
    assistant = "assistant"
    system = "system"


class ChatMessage(BaseModel):
    # ============================================================
    # 【P1修改2】role 字段类型从 str 改为 MessageRole 枚举
    #
    # 改之前：role: str
    # 改之后：role: MessageRole
    #
    # 实际影响：
    # 1. 前端传 role="system" 给 LLM 的回答 → 422 错误，立刻发现问题
    # 2. IDE 写代码时有自动补全提示（MessageRole.assistant）
    # 3. 面试时能说出"我用枚举约束了 role 字段的合法值，
    #    防止课程代码里 role 写错导致 LLM API 行为异常的问题"
    # ============================================================
    role: MessageRole = Field(description="消息角色：user / assistant / system")
    content: str = Field(description="消息内容")

    # ============================================================
    # 【P1修改3】新增 content 非空校验
    #
    # 原因：空消息传给 LLM 会浪费 token，甚至导致 LLM 行为异常。
    #       在数据结构层就拦截，不让空消息进入业务逻辑。
    # ============================================================
    @field_validator("content")
    @classmethod
    def content_must_not_be_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("消息内容不能为空")
        return v.strip()  # 顺便去掉首尾空白


class RAGRequest(BaseModel):
    knowledge_id: int = Field(description="在哪个知识库里检索")
    messages: List[ChatMessage] = Field(
        description="对话历史，最后一条必须是 role=user 的消息"
    )

    # ============================================================
    # 【P1修改4】新增消息列表校验：最后一条必须是用户消息
    #
    # 原因：RAG 问答流程依赖 messages[-1] 是用户的提问。
    #       如果最后一条是 assistant 的回复（客户端 bug 导致的），
    #       直接拿它去检索会得到错误结果，而且难以排查。
    #       在入参校验层提前拦截，报错信息更清晰。
    # ============================================================
    @field_validator("messages")
    @classmethod
    def last_message_must_be_user(cls, v: list) -> list:
        if not v:
            raise ValueError("对话历史不能为空")
        if v[-1].role != MessageRole.user:
            raise ValueError("对话历史的最后一条消息必须是用户消息（role=user）")
        return v


class RAGSource(BaseModel):
    """答案溯源：告诉用户这段回答来自哪个文档的哪一页"""
    document_id: int
    document_name: str = Field(description="文档标题，从数据库反查的真实名称")
    page_number: int
    chunk_content: str = Field(description="原文片段，用户可自行核验")


class RAGResponse(BaseResponse):
    answer: str = Field(description="LLM 生成的回答")
    sources: List[RAGSource] = Field(description="答案来源列表，支持溯源核验")
    messages: List[ChatMessage] = Field(description="包含本轮回答的完整对话历史")
