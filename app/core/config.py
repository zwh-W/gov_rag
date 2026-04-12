# app/core/config.py
"""
配置加载模块

设计原则：
1. 所有配置从 config.yaml 读取，敏感信息（API Key）从环境变量读取
2. 用 dataclass 做类型约束，避免到处用 config["xxx"]["xxx"] 裸字符串
3. 单例模式：整个进程只加载一次配置
"""
import os
import yaml
import warnings
from dataclasses import dataclass, field
from typing import Dict, Optional
from pathlib import Path
from dotenv import load_dotenv

# 加载项目根目录的 .env 文件（必须在读取环境变量之前！）
load_dotenv()
# 项目根目录：这个文件在 app/core/config.py，所以往上两层是根目录
BASE_DIR = Path(__file__).parent.parent.parent


def _load_yaml() -> dict:
    config_path = BASE_DIR / "config.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"找不到配置文件：{config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# ==========================================
# 用 dataclass 定义各模块的配置结构
# 好处：有类型提示、IDE 能自动补全、少写很多 config["xxx"]
# ==========================================

@dataclass
class AppConfig:
    host: str
    port: int
    debug: bool


@dataclass
class ESConfig:
    host: str
    port: int
    scheme: str
    username: str
    password: str
    index_document_meta: str
    index_chunk_info: str


@dataclass
class DatabaseConfig:
    engine: str
    path: str
    host: str
    port: int
    username: str
    password: str
    db_name: str


@dataclass
class RAGConfig:
    embedding_model: str
    use_rerank: bool
    rerank_model: str
    chunk_size: int
    chunk_overlap: int
    bm25_top_k: int
    vector_top_k: int
    rrf_k: int
    rerank_top_k: int
    confidence_threshold: float
    llm_base_url: str
    llm_model: str
    llm_temperature: float
    llm_top_p: float
    llm_max_tokens: int
    # API Key 从环境变量读取，不写进 yaml
    llm_api_key: str = field(default="")


@dataclass
class LoggingConfig:
    level: str
    file: str


@dataclass
class Settings:
    """
    全局配置入口，使用方式：
        from app.core.config import settings
        settings.rag.chunk_size
        settings.es.host
    """
    app: AppConfig
    es: ESConfig
    db: DatabaseConfig
    rag: RAGConfig
    logging: LoggingConfig
    device: str
    base_dir: Path
    # embedding 和 rerank 模型的详细参数（路径、维度等）
    embedding_model_params: Dict
    rerank_model_params: Dict


def _auto_detect_device() -> str:
    """
    【优化1】自动检测设备：有GPU用cuda，没有用cpu
    不用手动改 config.yaml 里的 device 了
    """
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
        # 有Apple Silicon的话也可以用mps
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
    except ImportError:
        pass
    return "cpu"


def _validate_model_paths(settings: Settings) -> None:
    """
    【优化2】启动时检查模型路径是否存在
    如果模型路径不对，提前报错，不用等到运行时才发现
    """
    # 检查Embedding模型路径
    embed_model = settings.rag.embedding_model
    embed_path = settings.embedding_model_params[embed_model]["local_path"]
    if not Path(embed_path).exists():
        warnings.warn(
            f"Embedding模型路径不存在：{embed_path}\n"
            f"请检查 config.yaml 里的 models.embedding.{embed_model}.local_path",
            stacklevel=2
        )

    # 检查Rerank模型路径（如果启用了Rerank）
    if settings.rag.use_rerank:
        rerank_model = settings.rag.rerank_model
        rerank_path = settings.rerank_model_params[rerank_model]["local_path"]
        if not Path(rerank_path).exists():
            warnings.warn(
                f"Rerank模型路径不存在：{rerank_path}\n"
                f"请检查 config.yaml 里的 models.rerank.{rerank_model}.local_path",
                stacklevel=2
            )


def _build_settings() -> Settings:
    raw = _load_yaml()

    rag_raw = raw["rag"]

    # 【优化3】Windows友好的环境变量读取：同时支持 DASHSCOPE_API_KEY 和 LLM_API_KEY
    llm_api_key = os.getenv("LLM_API_KEY", os.getenv("LLM_API_KEY", ""))
    if not llm_api_key:
        warnings.warn(
            "环境变量 DASHSCOPE_API_KEY 或 LLM_API_KEY 未设置，LLM 相关接口将无法使用。\n"
            "Windows PowerShell 设置方式：$env:DASHSCOPE_API_KEY='your-key'\n"
            "Windows CMD 设置方式：set DASHSCOPE_API_KEY=your-key",
            stacklevel=2
        )

    # 自动检测设备
    device = raw.get("device", "auto")
    if device == "auto":
        device = _auto_detect_device()

    settings = Settings(
        app=AppConfig(**raw["app"]),
        es=ESConfig(**raw["elasticsearch"]),
        db=DatabaseConfig(**raw["database"]),
        rag=RAGConfig(
            embedding_model=rag_raw["embedding_model"],
            use_rerank=rag_raw["use_rerank"],
            rerank_model=rag_raw["rerank_model"],
            chunk_size=rag_raw["chunk_size"],
            chunk_overlap=rag_raw["chunk_overlap"],
            bm25_top_k=rag_raw["bm25_top_k"],
            vector_top_k=rag_raw["vector_top_k"],
            rrf_k=rag_raw["rrf_k"],
            rerank_top_k=rag_raw["rerank_top_k"],
            confidence_threshold=rag_raw["confidence_threshold"],
            llm_base_url=rag_raw["llm_base_url"],
            llm_model=rag_raw["llm_model"],
            llm_temperature=rag_raw["llm_temperature"],
            llm_top_p=rag_raw["llm_top_p"],
            llm_max_tokens=rag_raw["llm_max_tokens"],
            llm_api_key=llm_api_key,
        ),
        logging=LoggingConfig(**raw["logging"]),
        device=device,
        base_dir=BASE_DIR,
        embedding_model_params=raw["models"]["embedding"],
        rerank_model_params=raw["models"]["rerank"],
    )

    # 启动时验证模型路径
    _validate_model_paths(settings)

    return settings


# 单例：模块级别只初始化一次
settings = _build_settings()