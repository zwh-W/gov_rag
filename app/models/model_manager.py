# app/models/model_manager.py
"""
模型管理器

设计原则：
1. 懒加载：服务启动时不强制加载模型，第一次调用时才加载
   → 这样即使模型文件不存在，其他接口（知识库管理、文档上传）也能正常工作
2. 加载后常驻内存，不重复加载
3. 设备自动识别：有 GPU 用 GPU，没有用 CPU
"""
import numpy as np
import torch
from typing import List, Union

from app.core.config import settings
from app.core.logger import get_logger

logger = get_logger(__name__)

# 全局模型存储，懒加载后放这里复用
_embedding_model = None
_rerank_model = None
_rerank_tokenizer = None


# ==========================================
# Embedding 模型
# ==========================================
def _load_embedding_model():
    global _embedding_model
    if _embedding_model is not None:
        return

    model_name = settings.rag.embedding_model
    model_path = settings.embedding_model_params[model_name]["local_path"]

    logger.info(f"加载 Embedding 模型：{model_name}，路径：{model_path}")

    from sentence_transformers import SentenceTransformer
    _embedding_model = SentenceTransformer(model_path)
    logger.info("Embedding 模型加载完成")


def get_embedding(text: Union[str, List[str]]) -> np.ndarray:
    """
    对文本编码，返回归一化后的向量
    :param text: 单条文本或文本列表
    :return: shape=(n, dims) 的 numpy 数组
    """
    _load_embedding_model()

    if isinstance(text, str):
        text = [text]

    vectors = _embedding_model.encode(text, normalize_embeddings=True)
    return vectors


# ==========================================
# Rerank 模型
# ==========================================
def _load_rerank_model():
    global _rerank_model, _rerank_tokenizer
    if _rerank_model is not None:
        return

    model_name = settings.rag.rerank_model
    model_path = settings.rerank_model_params[model_name]["local_path"]

    logger.info(f"加载 Rerank 模型：{model_name}，路径：{model_path}")

    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    _rerank_tokenizer = AutoTokenizer.from_pretrained(model_path)
    _rerank_model = AutoModelForSequenceClassification.from_pretrained(model_path)
    _rerank_model.eval()
    _rerank_model.to(settings.device)

    logger.info("Rerank 模型加载完成")


def get_rerank_scores(text_pairs: List[List[str]]) -> np.ndarray:
    """
    对 [(query, chunk), ...] 打分，分数越高越相关
    :param text_pairs: [(query, chunk1), (query, chunk2), ...]
    :return: shape=(n,) 的 numpy 数组，每个 pair 的相关性分数
    """
    _load_rerank_model()

    with torch.no_grad():
        inputs = _rerank_tokenizer(
            text_pairs,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        )
        inputs = {k: v.to(settings.device) for k, v in inputs.items()}
        scores = _rerank_model(**inputs, return_dict=True).logits.view(-1).float()
        return scores.cpu().numpy()
