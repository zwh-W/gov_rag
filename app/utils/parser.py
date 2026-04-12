# app/utils/parser.py
import pdfplumber
import pytesseract
from typing import List, Dict, Any
# 新版本正确导入路径
from langchain_text_splitters import MarkdownHeaderTextSplitter
from app.core.logger import get_logger

logger = get_logger(__name__)


def extract_text_from_pdf(file_path: str) -> str:
    """从 PDF 中提取文本（包含扫描件 OCR 兜底）"""
    full_text = ""
    logger.info(f"开始解析PDF文件: {file_path}")
    try:
        with pdfplumber.open(file_path) as pdf:
            for i, page in enumerate(pdf.pages):
                page_text = page.extract_text()
                # 如果字数太少，怀疑是扫描件，启动 OCR
                if page_text and len(page_text.strip()) > 20:
                    full_text += page_text + "\n\n"
                else:
                    logger.info(f"第 {i + 1} 页文本量少，尝试 OCR...")
                    img = page.to_image(resolution=300).original
                    try:
                        ocr_text = pytesseract.image_to_string(img, lang='chi_sim+eng')
                        full_text += ocr_text + "\n\n"
                    except Exception as e:
                        logger.warning(f"OCR 失败: {e}")
    except Exception as e:
        logger.error(f"解析 PDF 失败: {e}")
        raise e
    return full_text


def chunk_text_by_headers(text: str) -> List[Dict[str, Any]]:
    """使用 LangChain 按标题进行智能分块"""
    logger.info("开始智能分块...")
    headers_to_split_on = [
        ("#", "一级标题"),
        ("##", "二级标题"),
        ("###", "三级标题"),
    ]
    markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
    chunks = markdown_splitter.split_text(text)

    result_chunks = []
    for chunk in chunks:
        # 拼接面包屑
        breadcrumb = " > ".join(chunk.metadata.values()) if chunk.metadata else "正文"
        result_chunks.append({
            "content": chunk.page_content,
            "breadcrumb": breadcrumb
        })
    logger.info(f"分块完成，共生成 {len(result_chunks)} 个块。")
    return result_chunks