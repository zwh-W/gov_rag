# tests/test_parser.py

import pytest
from app.utils.parser import chunk_text_by_headers


def test_normal_gov_document():
    """测试用例 1：标准的政务文档结构，能否正确提取面包屑？"""
    text = """
第一章 总则
第一条 为了规范政务处分，制定本法。
第二条 本法适用于监察机关。
第二章 处罚种类
第三条 种类包括警告、记过。
    """

    chunks = chunk_text_by_headers(text)

    # 我们期望它被切成 3 块（第一条、第二条、第三条）
    assert len(chunks) == 3

    # 验证第一块的面包屑和内容
    assert chunks[0]["breadcrumb"] == "第一章 总则 > 第一条 为了规范政务处分，制定本法。"
    assert chunks[0]["content"] == "第一条 为了规范政务处分，制定本法。"

    # 验证第三块跨章后的面包屑
    assert chunks[2]["breadcrumb"] == "第二章 处罚种类 > 第三条 种类包括警告、记过。"


def test_page_number_extraction():
    """测试用例 2：测试提取 PDF 埋点的页码标记 <<PAGE:X>> 是否生效"""
    text = """
<<PAGE:5>>
第一章 总则
第一条 这一条在第五页。
<<PAGE:6>>
第二条 这一条在第六页。
    """
    chunks = chunk_text_by_headers(text)

    assert len(chunks) == 2
    assert chunks[0]["page_number"] == 5
    assert chunks[1]["page_number"] == 6


def test_overlength_chunk_fallback():
    """测试用例 3：测试超长文本是否会触发二次切分（兜底策略）"""
    # 构造一个 800 字的超长正文
    long_text = "啊" * 800
    text = f"""
第一章 总则
第一条 这是一个极其冗长的条款。
{long_text}
    """

    chunks = chunk_text_by_headers(text)

    # 因为 max_chunk_size 默认是 500，800 多字的文本必然被切成了多块
    assert len(chunks) >= 2

    # 验证二次切分的面包屑是否带了 "(片段X)" 后缀
    assert "(片段1)" in chunks[0]["breadcrumb"]
    assert "(片段2)" in chunks[1]["breadcrumb"]


def test_plain_text_without_headers():
    """测试用例 4：测试完全没有“章/条”的普通小说文本（鲁棒性测试）"""
    # 一段没有章、没有条的普通文本
    text = """
    这是小明写的一篇日记。
    今天天气很好，我去公园玩了。
    """

    chunks = chunk_text_by_headers(text)

    # 我们期望它不会报错，而是把整段话当成一个块（或者超长时被二次切分）
    assert len(chunks) == 1
    assert chunks[0]["breadcrumb"] == "正文"  # 没有标题时，默认面包屑是"正文"