# app/retrieval/query_rewriter.py
"""
Query Rewrite 模块

解决的问题：
  多轮对话中，用户的追问往往依赖上下文，无法独立理解。
  例：
    第一轮：用户问"第三条的规定是什么"
    第二轮：用户问"那例外情况呢"  ← 直接拿这句话去ES检索，什么都找不到

  Query Rewrite 用 LLM 把"那例外情况呢"结合对话历史，
  改写成"第三条规定的例外情况是什么"，再去 ES 检索。

设计原则：
  1. 第一轮对话不需要改写（没有上下文依赖）
  2. 改写失败时降级使用原始 query，不影响主流程
  3. 改写用的 LLM 调用 temperature=0.0，保证稳定性
  4. 只保留最近 4 轮历史（太长超 token，太短没上下文）
"""
import traceback
from typing import List

from app.core.config import settings
from app.core.logger import get_logger

logger = get_logger(__name__)

# ============================================================
# Query Rewrite 的 Prompt
#
# 设计要点：
# 1. 明确任务：改写成独立问题，不依赖上下文
# 2. 给出示例：让 LLM 理解"独立"的含义
# 3. 要求只输出改写结果：防止 LLM 加解释文字
# 4. 如果不需要改写：原样返回，不强行改写
# ============================================================
_REWRITE_PROMPT_TEMPLATE = """你是一个问题改写助手。
你的任务是：根据对话历史，将用户的最新提问改写成一个完整、独立的问题，使其不依赖上下文也能被理解。

改写规则：
1. 如果最新提问本身已经完整清晰，直接原样返回，不要改写。
2. 如果最新提问依赖上下文（如"那这个呢"、"例外情况呢"、"还有吗"），
   请结合对话历史补全指代词和缺失信息。
3. 只输出改写后的问题，不要任何解释、标点符号以外的内容。

对话历史（最近几轮）：
{history_str}

用户最新提问：{query}

改写后的完整问题："""


# ============================================================
# 懒加载 LLM 客户端
# 原因：和 qa_service.py 一样，不在模块级初始化，
#       避免 import 时因 Key 未配置导致服务启动失败
# ============================================================
_rewrite_client = None


def _get_rewrite_client():
    global _rewrite_client
    if _rewrite_client is not None:
        return _rewrite_client

    if not settings.rag.llm_api_key:
        raise ValueError("LLM API Key 未配置，无法执行 Query Rewrite")

    from openai import OpenAI
    _rewrite_client = OpenAI(
        api_key=settings.rag.llm_api_key,
        base_url=settings.rag.llm_base_url,
    )
    return _rewrite_client


def rewrite_query(query: str, history: list) -> str:
    """
    将多轮对话中的模糊提问改写成完整独立的问题。

    :param query:   用户最新提问
    :param history: ChatMessage 列表（完整对话历史，包含最新提问）
    :return:        改写后的问题。改写失败时返回原始 query。

    使用示例：
        from app.retrieval.query_rewriter import rewrite_query
        rewritten = rewrite_query("那例外情况呢", history)
        # rewritten → "第三条规定的例外情况是什么"
    """

    # ============================================================
    # 第一轮不需要改写
    #
    # 判断依据：history 里只有当前这一条 user 消息（长度<=1），
    #           说明是第一轮，没有上下文可以参考，直接返回原始 query。
    # ============================================================
    # history 包含了当前这轮的 user 消息，所以 <= 1 表示第一轮
    user_messages = [m for m in history if m.role == "user"]
    if len(user_messages) <= 1:
        logger.debug("第一轮对话，跳过 Query Rewrite")
        return query

    # ============================================================
    # 构造历史对话字符串
    #
    # 只取最近 4 轮（8条消息：4 user + 4 assistant），原因：
    # 1. 太长会超出 LLM 的 token 限制
    # 2. 太早的对话和当前问题关联性很低，反而引入噪音
    # 3. 4 轮通常足够理解当前追问的上下文
    # ============================================================
    recent_history = history[-8:]  # 最近8条（4轮对话）

    history_lines = []
    for msg in recent_history:
        # 跳过当前最新的 user 消息（它就是 query，不需要放进历史）
        if msg.role == "user" and msg.content == query:
            continue
        role_label = "用户" if msg.role == "user" else "助手"
        history_lines.append(f"{role_label}：{msg.content}")

    history_str = "\n".join(history_lines)

    # 如果历史为空（过滤后没有内容），不需要改写
    if not history_str.strip():
        return query

    # ============================================================
    # 调用 LLM 改写
    # ============================================================
    prompt = _REWRITE_PROMPT_TEMPLATE.format(
        history_str=history_str,
        query=query,
    )

    try:
        client = _get_rewrite_client()
        response = client.chat.completions.create(
            model=settings.rag.llm_model,
            messages=[{"role": "user", "content": prompt}],
            # temperature=0.0：改写任务要求稳定，不需要创造性
            temperature=0.0,
            # max_tokens 设小一点：改写结果应该很短，一句话就够
            max_tokens=150,
        )
        rewritten = response.choices[0].message.content.strip()

        # 防止 LLM 返回空字符串
        if not rewritten:
            logger.warning("Query Rewrite 返回空字符串，使用原始 query")
            return query

        logger.info(f"Query Rewrite: '{query}' → '{rewritten}'")
        return rewritten

    except Exception:
        # ============================================================
        # 改写失败时降级：直接用原始 query
        #
        # 原因：Query Rewrite 是优化项，不是必需项。
        #       改写失败不应该导致整个问答流程失败。
        #       降级用原始 query 还能给用户一个（质量可能差一点的）回答，
        #       比直接报错体验好得多。
        # ============================================================
        logger.warning(f"Query Rewrite 失败，降级使用原始 query: {traceback.format_exc()}")
        return query


# ============================================================
# 本地测试：直接运行这个文件验证改写效果
#
# 运行方式（在项目根目录）：
#   python -m app.retrieval.query_rewriter
# ============================================================
if __name__ == "__main__":
    from app.api.schemas import ChatMessage

    # 模拟一段多轮对话历史
    test_history = [
        ChatMessage(role="user",      content="第三条的规定是什么？"),
        ChatMessage(role="assistant", content="第三条规定：申请人须提交身份证明材料。"),
        ChatMessage(role="user",      content="那例外情况呢？"),
    ]

    test_cases = [
        # (query, history, 期望结果描述)
        ("那例外情况呢？",     test_history,  "应改写为含'第三条'的完整问题"),
        ("第五条是什么内容？", test_history[:1], "第一轮，应原样返回"),
        ("还有其他要求吗？",   test_history,  "应补全主语和上下文"),
    ]

    print("=" * 60)
    print("Query Rewrite 测试")
    print("=" * 60)
    for query, history, expected in test_cases:
        result = rewrite_query(query, history)
        print(f"\n原始 query : {query}")
        print(f"期望       : {expected}")
        print(f"改写结果   : {result}")
    print("=" * 60)