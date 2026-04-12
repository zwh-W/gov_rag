# app/services/qa_service.py

from typing import List, Dict, Tuple
from openai import OpenAI
import traceback

from app.core.config import settings
from app.core.logger import get_logger
from app.retrieval.searcher import hybrid_search
from app.api.schemas import RAGSource, ChatMessage

logger = get_logger(__name__)

# 1. 初始化大模型客户端 (自动读取你的 config.yaml)
# 注意：API Key 已经在 config.py 里通过环境变量加载到了 settings.rag.llm_api_key
try:
    llm_client = OpenAI(
        api_key=settings.rag.llm_api_key,
        base_url=settings.rag.llm_base_url
    )
except Exception as e:
    logger.error(f"大模型客户端初始化失败: {e}")
    llm_client = None

# 2. 定义政务场景的终极“紧箍咒” (System Prompt)
SYSTEM_PROMPT = """你是一个专业的政务与企业知识库助手。
请严格遵循以下规则回答用户的问题：
1. 你必须、且只能基于提供给你的 [参考资料] 来生成答案。
2. 如果 [参考资料] 中没有提及相关信息，或者提供的资料不足以回答问题，请直接回答：“抱歉，在知识库的参考资料中未找到相关内容，无法准确回答。”
3. 绝不允许编造、推测或使用你的内部知识库来回答！
4. 回答要条理清晰，简明扼要，直接切中要害。
"""


def build_prompt(query: str, retrieved_docs: List[Dict]) -> Tuple[str, List[RAGSource]]:
    """
    将检索到的文档块拼接成字符串，同时提取出溯源信息
    """
    if not retrieved_docs:
        return "", []

    context_str = ""
    sources = []

    for i, doc in enumerate(retrieved_docs):
        # 提取各个字段，做好防空处理
        content = doc.get("chunk_content", "")
        breadcrumb = doc.get("breadcrumb", "未知章节")
        doc_id = doc.get("document_id", 0)
        # TODO: 从数据库反查 document_name，这里先简化处理
        doc_name = f"文档_{doc_id}"

        # 拼接给 LLM 看的上下文
        context_str += f"--- 资料 [{i + 1}] ---\n"
        context_str += f"来源章节：{breadcrumb}\n"
        context_str += f"内容：{content}\n\n"

        # 收集溯源信息，准备返回给前端展示
        sources.append(
            RAGSource(
                document_id=doc_id,
                document_name=doc_name,
                page_number=doc.get("page_number", 1),
                chunk_content=content
            )
        )

    user_prompt = f"""请基于以下 [参考资料] 回答问题。

[参考资料]:
{context_str}
chat_with_knowledge_base
[用户问题]:
{query}
"""
    return user_prompt, sources


def chat_with_knowledge_base(knowledge_id: int, query: str, history: List[ChatMessage]) -> Tuple[str, List[RAGSource]]:
    """
    核心 QA 流程：检索 -> 组装 Prompt -> 调用大模型 -> 答案溯源
    """
    if not llm_client:
        return "系统未配置大模型 API Key，无法生成回答。", []

    # 1. 执行混合检索 + Rerank
    logger.info(f"接收到用户提问: {query}")
    retrieved_docs = hybrid_search(query, knowledge_id)

    if not retrieved_docs:
        return "抱歉，在知识库中没有检索到与您问题相关的内容。", []

    # 2. 组装 Prompt 和 溯源列表
    user_prompt, sources = build_prompt(query, retrieved_docs)
    logger.debug(f"组装的 User Prompt:\n{user_prompt}")

    # 3. 构造完整的对话消息体
    # 这里我们简化处理，真实多轮对话需要将 history 也拼进去
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt}
    ]

    # 4. 调用通义千问大模型
    logger.info("正在调用大模型生成回答...")
    try:
        response = llm_client.chat.completions.create(
            model=settings.rag.llm_model,
            messages=messages,
            temperature=settings.rag.llm_temperature,
            top_p=settings.rag.llm_top_p,
            max_tokens=settings.rag.llm_max_tokens
        )
        answer = response.choices[0].message.content
        logger.info("大模型回答生成完毕。")
        return answer, sources

    except Exception as e:
        logger.error(f"调用大模型报错:\n{traceback.format_exc()}")
        return f"生成回答时发生系统错误: {str(e)}", sources