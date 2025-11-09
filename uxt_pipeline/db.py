from __future__ import annotations
from datetime import datetime
from typing import Optional

from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime, ForeignKey
from sqlalchemy.orm import declarative_base, relationship, sessionmaker

DB_PATH = "uxt_pipeline/uxt.db"
engine = create_engine(f"sqlite:///{DB_PATH}", echo=False, future=True)
SessionLocal = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False, future=True)
Base = declarative_base()

class Run(Base):
    __tablename__ = "runs"
    id = Column(Integer, primary_key=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    chunk_size = Column(Integer, nullable=False)
    overlap = Column(Integer, nullable=False)
    comment = Column(String(255), default="")
    chunks = relationship("ChunkRow", back_populates="run", cascade="all, delete-orphan")

class ChunkRow(Base):
    __tablename__ = "chunks"
    id = Column(Integer, primary_key=True)
    run_id = Column(Integer, ForeignKey("runs.id"), nullable=False, index=True)
    source = Column(String(255), nullable=False)
    text = Column(Text, nullable=False)
    len_words = Column(Integer, nullable=False)
    run = relationship("Run", back_populates="chunks")

def init_db() -> None:
    Base.metadata.create_all(engine)
    _init_fts()

def _init_fts() -> None:
    """Створюємо FTS5-таблицю без явної колонки rowid + тригери синхронізації."""
    with engine.begin() as conn:
        conn.exec_driver_sql("""
        CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts
        USING fts5(text, source, content='');
        """)
        conn.exec_driver_sql("""
        CREATE TRIGGER IF NOT EXISTS chunks_ai AFTER INSERT ON chunks BEGIN
          INSERT INTO chunks_fts(rowid, text, source) VALUES (new.id, new.text, new.source);
        END;""")
        conn.exec_driver_sql("""
        CREATE TRIGGER IF NOT EXISTS chunks_ad AFTER DELETE ON chunks BEGIN
          DELETE FROM chunks_fts WHERE rowid = old.id;
        END;""")
        conn.exec_driver_sql("""
        CREATE TRIGGER IF NOT EXISTS chunks_au AFTER UPDATE ON chunks BEGIN
          UPDATE chunks_fts SET text=new.text, source=new.source WHERE rowid = old.id;
        END;""")

def rebuild_fts() -> None:
    """Ручна перебудова FTS із наявних даних."""
    with engine.begin() as conn:
        conn.exec_driver_sql("DELETE FROM chunks_fts;")
        conn.exec_driver_sql("""
            INSERT INTO chunks_fts(rowid, text, source)
            SELECT id, text, source FROM chunks;
        """)
