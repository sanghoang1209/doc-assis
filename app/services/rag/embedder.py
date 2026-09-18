import asyncio
from app.services.client import EMBED_MODEL_NAME, ollama_client as client

# add retry logic when ollama fail
# batch embedding
async def embed_text(texts: list[str]) -> list[list[float]]:
    """Embedd user query using nomic-embed-text via Ollama

    Args:
        text (str): user query input

    Raises:
        e: detail error

    Returns:
        list[float]: text input represented in vector
    """
    try:
        batch = await client.embed(model=EMBED_MODEL_NAME, input=texts)
        return batch['embeddings']
    except Exception as e:
        print(f"Error in embedding with {EMBED_MODEL_NAME}: {e}")
        raise e

async def main():
    embeddings = await embed_text([
        "Hello",
        "Hahahaah"
    ])

    print(len(embeddings))

if __name__ == "__main__":
    asyncio.run(main())