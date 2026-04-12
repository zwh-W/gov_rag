# app/db/session.py
"""
数据库 Session 管理

设计原则：
1. 用 context manager 管理 session 生命周期，防止连接泄漏
2. 支持 sqlite（开发）和 mysql（生产）无缝切换
3. 只在这里创建 engine，其他地方用 get_session()
"""
from contextlib import contextmanager
from typing import Generator

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session

from app.core.config import settings
from app.core.logger import get_logger
from app.db.models import Base

logger = get_logger(__name__)


def _build_engine():
    db = settings.db

    if db.engine == "sqlite":
        url = f"sqlite:///{settings.base_dir / db.path}"
        # sqlite 不支持多线程并发写，check_same_thread=False 是必须的
        return create_engine(url, connect_args={"check_same_thread": False}, echo=False)

    elif db.engine in ("mysql", "mysql+pymysql"):
        url = (
            f"mysql+pymysql://{db.username}:{db.password}"
            f"@{db.host}:{db.port}/{db.db_name}?charset=utf8mb4"
        )
        return create_engine(url, pool_size=10, max_overflow=20, echo=False)

    else:
        raise ValueError(f"不支持的数据库类型：{db.engine}，请使用 sqlite 或 mysql")


_engine = _build_engine()

# 启动时自动创建所有表（如果不存在）
Base.metadata.create_all(_engine)
logger.info(f"数据库初始化完成，引擎：{settings.db.engine}")

_SessionFactory = sessionmaker(bind=_engine, autocommit=False, autoflush=False)


@contextmanager
def get_session() -> Generator[Session, None, None]:
    """
    使用方式：
        from app.db.session import get_session
        with get_session() as session:
            record = session.query(KnowledgeBase).first()

    出了 with 块自动提交或回滚，不需要手动管理
    """
    session = _SessionFactory()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()