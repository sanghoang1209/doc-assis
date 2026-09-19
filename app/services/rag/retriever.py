import uuid
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from app.models import Chunk
from app.config import THRESHOLD
from app.services.rag.embedder import embed_text

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
    distance = Chunk.embedding.cosine_distance(embed_vec).label("distance")
    selected_obj = select(Chunk, distance)

    if document_id:
        selected_obj = selected_obj.where(Chunk.document_id == document_id)    

    rows = (
        await db.execute(
            selected_obj
            .where(distance < THRESHOLD)
            .order_by(distance)
            .limit(top_k)
        )
    ).all()

    results: list[tuple[Chunk, float]] = []
    for row in rows:
        chunk = row.Chunk
        distance_val = row.distance
        results.append((chunk, distance_val))

    return results


async def main():
    from app.database import SessionLocal
    async with SessionLocal() as session:
        results = await retrieve_vec("Hello", None, 5, session)

    for result in results:
        print(result[0])
        print(result[1])
        print()


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())