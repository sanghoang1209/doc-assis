import re

def _clean_text(text: str) -> str:
    """
    Clean raw extracted text before chunking.

    Removes common PDF extraction artifacts:
    - Mid-word hyphenated line breaks (e.g. "exam-\nple" → "example")
    - Standalone page numbers on their own line
    - Excessive blank lines collapsed to a single newline
    - Leading/trailing whitespace per line

    Args:
        text (str): Raw extracted text.

    Returns:
        str: Cleaned text.
    """
    # Rejoin hyphenated line breaks (e.g. "exam-\nple")
    text = re.sub(r'-\n(\S)', r'\1', text)

    # Remove lines that are only a number (page numbers)
    text = re.sub(r'^\s*\d+\s*$', '', text, flags=re.MULTILINE)

    # Collapse 3+ consecutive newlines into two
    text = re.sub(r'\n{3,}', '\n\n', text)

    # Strip trailing whitespace from each line
    text = '\n'.join(line.rstrip() for line in text.splitlines())

    return text.strip()


def split_into_sentences(text: str) -> list[str]:
    """
    Split text into sentences for mixed Vietnamese/English content.

    Splits at sentence boundary punctuation (.!?) while preventing false
    splits on decimals, abbreviations, and ellipses.

    Args:
        text (str): Input text.

    Returns:
        list[str]: List of extracted sentences.
    """
    text = re.sub(r'(\d)\.(\d)', r'\1<DECIMAL>\2', text)
    text = re.sub(r'\.{2,}', '<ELLIPSIS>', text)

    abbreviations = ['Mr', 'Mrs', 'Ms', 'Dr', 'Prof', 'Sr', 'Jr',
                     'vs', 'etc', 'e.g', 'i.e', 'Fig', 'fig',
                     'No', 'Vol', 'Jan', 'Feb', 'Mar', 'Apr',
                     'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    for abbr in abbreviations:
        text = re.sub(
            rf'\b{abbr}\.',
            f'{abbr}<ABBR>',
            text,
            flags=re.IGNORECASE
        )

    sentences = re.split(r'([.!?])\s+', text)

    result = []
    i = 0
    while i < len(sentences):
        if i + 1 < len(sentences) and sentences[i + 1] in '.!?':
            merged = sentences[i] + sentences[i + 1]
            result.append(merged.strip())
            i += 2
        else:
            if sentences[i].strip():
                result.append(sentences[i].strip())
            i += 1

    restored = []
    for s in result:
        s = s.replace('<DECIMAL>', '.')
        s = s.replace('<ELLIPSIS>', '...')
        s = s.replace('<ABBR>', '.')
        restored.append(s)

    return [s for s in restored if s.strip()]


def chunk_by_sentences(
    text: str,
    max_chars: int = 500,
    overlap_sentences: int = 1,
    min_chars: int = 80,
) -> list[str]:
    """
    Chunk text along sentence boundaries.

    Short sentences (below min_chars) are merged into the next sentence
    rather than being emitted as standalone chunks. This avoids wasting
    embedding slots on fragments like "Ok." or a lone heading.

    Args:
        text (str): Input text.
        max_chars (int): Soft character limit for each chunk.
        overlap_sentences (int): Number of overlapping sentences between chunks.
        min_chars (int): Minimum character count for a standalone sentence;
                         shorter sentences are merged forward.

    Returns:
        list[str]: List of sentence-aligned text chunks.
    """
    if not text:
        return []

    text = _clean_text(text)
    sentences = split_into_sentences(text)

    if not sentences:
        return []

    # Merge sentences that are too short into their successor
    merged_sentences: list[str] = []
    buffer = ""
    for sentence in sentences:
        if buffer:
            sentence = buffer + " " + sentence
            buffer = ""
        if len(sentence) < min_chars:
            buffer = sentence
        else:
            merged_sentences.append(sentence)
    if buffer:
        # Last sentence was short — append to previous or keep as-is
        if merged_sentences:
            merged_sentences[-1] = merged_sentences[-1] + " " + buffer
        else:
            merged_sentences.append(buffer)

    chunks = []
    current_sentences: list[str] = []
    current_chars = 0

    for sentence in merged_sentences:
        sentence_chars = len(sentence)

        if not current_sentences:
            current_sentences.append(sentence)
            current_chars += sentence_chars
            continue

        if current_chars + sentence_chars > max_chars:
            chunks.append(" ".join(current_sentences))

            num_overlap = min(overlap_sentences, max(0, len(current_sentences) - 1))
            if num_overlap > 0:
                current_sentences = current_sentences[-num_overlap:]
                current_chars = sum(len(s) for s in current_sentences)
            else:
                current_sentences = []
                current_chars = 0

        current_sentences.append(sentence)
        current_chars += sentence_chars

    if current_sentences:
        chunks.append(" ".join(current_sentences))

    return chunks