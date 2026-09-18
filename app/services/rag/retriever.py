import uuid
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from app.services.rag.embedder import embed_text
from app.models import Chunk

async def retrieve_vec(question: str, document_id: uuid.UUID | None, top_k: int, db: AsyncSession):
    """Retrieve top-k relevant document chunks using vector cosine similarity.

    Args:
        question (str): User query string.
        document_id (uuid.UUID | None): Optional document ID filter. If None, searches across all chunks.
        top_k (int): Number of top matching chunks to retrieve.
        db (AsyncSession): Database session.

    Returns:
        Sequence[Chunk]: List/Sequence of matching Chunk ORM objects ordered by similarity.
    """
    embed_vec = await embed_text(question)
    selected_obj = select(Chunk)
    if document_id:
        selected_obj = selected_obj.where(Chunk.document_id == document_id)

    chunks = (
        await db.scalars(
            selected_obj
            .order_by(
                Chunk.embedding.cosine_distance(embed_vec)
            )
            .limit(top_k)
        )
    ).all()
    return chunks
