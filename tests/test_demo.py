import pytest
from unittest.mock import patch, MagicMock

# 导入你刚刚优化的模块
from app.services import qa_service
from app.services.qa_service import _get_llm_client, build_prompt


@pytest.fixture(autouse=True)
def reset_global_state():
    """
    【关键机制】：因为 _llm_client 是全局变量，
    为了防止测试用例之间互相污染，每次执行测试前后都要强制清空它。
    """
    qa_service._llm_client = None
    yield
    qa_service._llm_client = None


class TestQAServiceOptimizations:

    # ==========================================
    # 测试 P0修改1：LLM 客户端的懒加载与单例模式
    # ==========================================

    @patch("app.services.qa_service.settings")
    def test_get_llm_client_missing_key(self, mock_settings):
        """测试：当没有配置 API Key 时，首次调用应该抛出 ValueError"""
        # 模拟环境变量里没有 Key 的情况
        mock_settings.rag.llm_api_key = None

        # 断言是否正确抛出了异常
        with pytest.raises(ValueError, match="LLM API Key 未配置"):
            _get_llm_client()

    @patch("app.services.qa_service.settings")
    @patch("openai.OpenAI")  # <--- 这里改成了拦截官方包
    def test_get_llm_client_lazy_and_singleton(self, mock_openai_class, mock_settings):
        """测试：配置了 Key 时，应该正常初始化，并且多次调用返回同一个实例"""
        # 模拟环境变量配置正确
        mock_settings.rag.llm_api_key = "test-sk-12345"
        mock_settings.rag.llm_base_url = "https://test.api.com"

        # 模拟 OpenAI 返回一个假想的客户端对象
        mock_fake_client = MagicMock()
        mock_openai_class.return_value = mock_fake_client

        # 第一次调用：应该触发真实初始化
        client1 = _get_llm_client()

        # 第二次调用：应该直接返回已有的，不再去初始化
        client2 = _get_llm_client()

        # 断言客户端是否成功获取，且是同一个内存对象
        assert client1 is not None
        assert client1 is client2

        # 验证 OpenAI 这个类在全过程中只被实例化了一次
        mock_openai_class.assert_called_once_with(
            api_key="test-sk-12345",
            base_url="https://test.api.com"
        )

    # ==========================================
    # 测试 P0修改2：Prompt 格式组装优化
    # ==========================================

    def test_build_prompt_clean_format(self):
        """测试：组装的 Prompt 是否干净，溯源信息是否正确提取"""

        # 构造模拟的检索结果 List[Dict]
        mock_retrieved_docs = [
            {
                "document_id": 99,
                "document_name": "测试安全规范.pdf",
                "breadcrumb": "第二章 > 密码管理",
                "page_number": 5,
                "chunk_content": "员工必须每90天更换一次密码。"
            }
        ]
        user_query = "密码多久换一次？"

        # 执行你优化的函数
        user_prompt, sources = build_prompt(user_query, mock_retrieved_docs)

        # 【断言 1】检查垃圾文字是否真的被删除了
        assert "chat_with_knowledge_base" not in user_prompt, "致命错误：Prompt 中依然包含垃圾函数名！"

        # 【断言 2】检查必要的信息是否成功拼接入了 Prompt
        assert "测试安全规范.pdf" in user_prompt
        assert "第二章 > 密码管理" in user_prompt
        assert "第5页" in user_prompt
        assert "员工必须每90天更换一次密码" in user_prompt
        assert user_query in user_prompt

        # 【断言 3】检查溯源对象 (sources) 列表生成是否正确
        assert len(sources) == 1
        assert sources[0].document_id == 99
        assert sources[0].page_number == 5

    def test_build_prompt_empty_docs(self):
        """测试：边缘场景，如果没有检索到任何资料怎么处理"""
        user_prompt, sources = build_prompt("今天天气如何？", [])
        assert user_prompt == ""
        assert sources == []