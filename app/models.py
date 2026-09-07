import uuid
from enum import Enum
from datetime import datetime
from sqlalchemy import Integer, DateTime, Text, UUID, func, ForeignKey
from sqlalchemy.orm import Mapped, mapped_column, DeclarativeBase
from pgvector.sqlalchemy import Vector


class FileStatus(str, Enum):
    """Status enum for tracking document processing pipeline."""
    PENDING = "PENDING"
    PROCESSING = "PROCESSING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"

class Base(DeclarativeBase):
    pass

class Document(Base):
    """SQLAlchemy model representing an uploaded document and its processing metadata."""
    __tablename__ = "documents"

    id: Mapped[uuid.UUID] = mapped_column(UUID, primary_key=True, default=uuid.uuid4)
    status: Mapped[FileStatus] = mapped_column(Text, nullable=False, default=FileStatus.PENDING)
    filename: Mapped[str] = mapped_column(Text, nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now(), onupdate=func.now())
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())

class Chunk(Base):
    __tablename__ = "chunks"

    id: Mapped[uuid.UUID] = mapped_column(UUID, primary_key=True, default=uuid.uuid4)
    document_id: Mapped[uuid.UUID] = mapped_column(UUID, ForeignKey("documents.id"), nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    embedding: Mapped[list[float]] = mapped_column(Vector(768), nullable=False)
    chunk_index: Mapped[int] = mapped_column(Integer, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())

