def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> list[str]:
    """
    Split text into fixed-size character chunks.

    Args:
        text (str): Input text.
        chunk_size (int): Character length of each chunk.
        overlap (int): Number of overlapping characters between adjacent chunks.

    Returns:
        list[str]: List of text chunks.
    """
    if not text:
        return []

    chunks = []
    start = 0
    len_text = len(text)
    while start <= len_text:
        end = start + chunk_size
        chunks.append(text[start:end])
        if end >= len_text:
            break
        start += chunk_size - overlap
    return chunks