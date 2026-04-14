import pytest
import numpy as np
from unittest.mock import patch, MagicMock
from contextlib import contextmanager

from app.services.document_processor import process_document_background
from app.db.models import Document


@pytest.fixture
def mock_doc(db_session):
    doc = Document(
        document_id=999,
        title="优化测试文档",
        file_path="/fake/path.pdf",
        knowledge_id=101,
        process_status="pending"
    )
    db_session.add(doc)
    db_session.commit()
    return doc


# 我们拦截外部调用，但核心逻辑（你的那三个优化点）完全走真实代码
@patch("app.services.document_processor.get_session")
@patch("app.services.document_processor.bulk")
@patch("app.services.document_processor.get_embedding")
@patch("app.services.document_processor.chunk_text_by_headers")
@patch("app.services.document_processor.extract_text_from_pdf")
def test_watch_my_optimizations(
        mock_extract,
        mock_chunk,
        mock_embedding,
        mock_bulk,
        mock_get_session,
        db_session,
        mock_doc
):
    # 1. 模拟数据库上下文
    @contextmanager
    def mock_session_scope():
        yield db_session
        db_session.commit()

    mock_get_session.side_effect = mock_session_scope

    # 2. 模拟解析器返回数据（专门造几个不同页码的 chunk，用来测试 P0修改3）
    mock_extract.return_value = "假装这是提取出来的超长文本"
    mock_chunk.return_value = [
        {"content": "第一页的内容", "page_number": 1, "breadcrumb": "第一章"},
        {"content": "第二页的内容", "page_number": 2, "breadcrumb": "第二章"},
        {"content": "第五页的内容", "page_number": 5, "breadcrumb": "第三章"}
    ]

    # 3. 模拟向量模型返回 3 条向量，每条维度 512
    mock_embedding.return_value = np.random.rand(3, 512)

    # 4. 模拟 ES 成功写入 3 条
    mock_bulk.return_value = (3, [])

    # 执行主程序！(打断点准备迎接它)
    process_document_background(mock_doc.document_id)