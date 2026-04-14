# app/services/qa_service.py
import traceback
from typing import List, Dict, Tuple

from app.core.config import settings
from app.core.logger import get_logger
from app.retrieval.searcher import hybrid_search
from app.api.schemas import RAGSource, ChatMessage

logger = get_logger(__name__)


# ============================================================
# 【P0修改1】LLM 客户端改为懒加载，不在模块级直接初始化
#
# 原来的问题：
#   模块顶部直接执行 OpenAI(...)，意味着：
#   1. 只要 import 这个文件，就会尝试初始化客户端
#   2. 如果 API Key 没配置，import 就会失败或产生警告
#   3. 进而导致 main.py 启动失败，连知识库管理接口都用不了
#
# 解决：
#   用 _get_llm_client() 函数封装，第一次真正需要调用 LLM 时才初始化。
#   初始化失败只影响问答接口，不影响其他接口。
# ============================================================
_llm_client = None


def _get_llm_client():
    """懒加载 LLM 客户端，第一次调用时初始化，之后复用"""
    global _llm_client
    if _llm_client is not None:
        return _llm_client

    if not settings.rag.llm_api_key:
        raise ValueError(
            "LLM API Key 未配置！\n"
            "请设置环境变量：export DASHSCOPE_API_KEY='your-key'\n"
            "Windows PowerShell：$env:DASHSCOPE_API_KEY='your-key'"
        )

    from openai import OpenAI
    _llm_client = OpenAI(
        api_key=settings.rag.llm_api_key,
        base_url=settings.rag.llm_base_url,
    )
    logger.info("LLM 客户端初始化成功")
    return _llm_client


# ============================================================
# 政务场景的 System Prompt
#
# 设计原则：
# 1. 明确身份：告诉 LLM 它是知识库助手，不是通用聊天机器人
# 2. 硬约束：只能用参考资料回答，不能用内部知识
# 3. 拒答指令：资料不够时明确拒答，而不是强行编造
# 4. 格式要求：条理清晰，方便用户阅读
# ============================================================
SYSTEM_PROMPT = """你是一个专业的政务与企业知识库助手。
请严格遵循以下规则回答用户的问题：
1. 你必须、且只能基于提供给你的 [参考资料] 来生成答案。
2. 如果 [参考资料] 中没有提及相关信息，或者提供的资料不足以回答问题，请直接回答："抱歉，在知识库的参考资料中未找到相关内容，无法准确回答。"
3. 绝不允许编造、推测或使用你的内部知识库来回答！
4. 回答要条理清晰，简明扼要，直接切中要害。
"""


def build_prompt(query: str, retrieved_docs: List[Dict]) -> Tuple[str, List[RAGSource]]:
    """
    将检索到的文档块拼接成 Prompt，同时提取溯源信息。

    返回：
    - user_prompt: 发给 LLM 的完整用户消息
    - sources:     溯源列表，返回给前端展示
    """
    if not retrieved_docs:
        return "", []

    context_str = ""
    sources = []

    for i, doc in enumerate(retrieved_docs):
        content = doc.get("chunk_content", "")
        breadcrumb = doc.get("breadcrumb", "未知章节")
        doc_id = doc.get("document_id", 0)
        doc_name = doc.get("document_name", f"文档_{doc_id}")
        page_number = doc.get("page_number", 1)

        # 拼接给 LLM 看的上下文，带章节路径帮助 LLM 理解来源
        context_str += f"--- 资料 [{i + 1}] ---\n"
        context_str += f"来源：{doc_name}｜章节：{breadcrumb}｜第{page_number}页\n"
        context_str += f"内容：{content}\n\n"

        sources.append(RAGSource(
            document_id=doc_id,
            document_name=doc_name,
            page_number=page_number,
            chunk_content=content,
        ))

    # ============================================================
    # 【P0修改2】删除 Prompt 中的垃圾文字 "chat_with_knowledge_base"
    #
    # 原来的问题：
    #   user_prompt 字符串里有一行孤零零的 "chat_with_knowledge_base"，
    #   这是复制粘贴时不小心带进来的函数名。
    #   它会原封不动地发给千问大模型，LLM 看到这个莫名其妙的字符串，
    #   会被干扰，回答质量下降，甚至可能把它当成问题的一部分来理解。
    #
    # 解决：直接删掉那行，Prompt 格式干净清晰。
    # ============================================================
    user_prompt = f"""请基于以下 [参考资料] 回答问题。

[参考资料]:
{context_str}
[用户问题]:
{query}
"""
    return user_prompt, sources


def chat_with_knowledge_base(
        knowledge_id: int,
        query: str,
        history: List[ChatMessage]
) -> Tuple[str, List[RAGSource]]:
    """
    核心 QA 流程：混合检索 -> 组装 Prompt -> 调用千问 -> 返回答案+溯源
    """
    # ── 1. 初始化 LLM 客户端（懒加载，失败时直接返回错误信息）──────
    try:
        client = _get_llm_client()
    except ValueError as e:
        logger.error(str(e))
        return "系统未配置大模型 API Key，无法生成回答。请联系管理员。", []

    # ── 2. 混合检索 + Rerank ────────────────────────────────────────
    logger.info(f"接收到用户提问: {query}")
    retrieved_docs = hybrid_search(query, knowledge_id)

    if not retrieved_docs:
        return "抱歉，在知识库中没有检索到与您问题相关的内容。", []

    # ── 3. 组装 Prompt 和溯源列表 ───────────────────────────────────
    user_prompt, sources = build_prompt(query, retrieved_docs)
    logger.debug(f"组装的 User Prompt:\n{user_prompt}")

    # ── 4. 构造发给 LLM 的消息列表 ─────────────────────────────────
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]
    # 注意：多轮对话的 history 处理在 P1 阶段实现（Query Rewrite）
    # 现在先保持单轮，确保基础功能正确

    # 4. 调用通义千问大模型
    logger.info("正在调用大模型生成回答...")
    try:
        response = client.chat.completions.create(
            model=settings.rag.llm_model,
            messages=messages,
            temperature=settings.rag.llm_temperature,
            top_p=settings.rag.llm_top_p,
            max_tokens=settings.rag.llm_max_tokens,
        )
        answer = response.choices[0].message.content
        logger.info("大模型回答生成完毕")
        return answer, sources

    except Exception as e:
        logger.error(f"调用大模型报错:\n{traceback.format_exc()}")
        return f"生成回答时发生系统错误，请稍后重试。错误信息：{str(e)}", sources