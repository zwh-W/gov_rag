
# app/core/logger.py
"""
日志模块

设计原则：
1. 同时输出到文件和控制台
2. 文件按天滚动，不会无限膨胀
3. 统一格式：时间 | 级别 | 模块名 | 消息
4. 其他模块统一用 get_logger(__name__) 获取，不要直接用 print
"""
import logging
import sys
from logging.handlers import TimedRotatingFileHandler
from pathlib import Path

from app.core.config import settings

# 日志目录不存在则自动创建
log_file = settings.base_dir / settings.logging.file
log_file.parent.mkdir(parents=True, exist_ok=True)

_LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# 根 logger 只配置一次
_root_configured = False


def _configure_root_logger() -> None:
    global _root_configured
    if _root_configured:
        return

    level = getattr(logging, settings.logging.level.upper(), logging.INFO)
    root = logging.getLogger()
    root.setLevel(level)

    formatter = logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT)

    # 1. 控制台 handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    root.addHandler(console_handler)

    # 2. 文件 handler（按天滚动，保留最近 30 天）
    file_handler = TimedRotatingFileHandler(
        filename=log_file,
        when="midnight",
        interval=1,
        backupCount=30,
        encoding="utf-8",
    )
    file_handler.setFormatter(formatter)
    root.addHandler(file_handler)

    _root_configured = True


def get_logger(name: str) -> logging.Logger:
    """
    使用方式：
        from app.core.logger import get_logger
        logger = get_logger(__name__)
        logger.info("something happened")
    """
    _configure_root_logger()
    return logging.getLogger(name)