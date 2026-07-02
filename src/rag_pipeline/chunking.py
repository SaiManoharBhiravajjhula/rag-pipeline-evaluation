from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DocumentChunk:
    chunk_id: str
    source: str
    text: str


def load_documents(directory: str | Path) -> list[tuple[str, str]]:
    path = Path(directory)
    return [(file.name, file.read_text(encoding="utf-8")) for file in sorted(path.glob("*.txt"))]


def chunk_text(source: str, text: str, max_words: int = 70, overlap: int = 12) -> list[DocumentChunk]:
    """Split text into overlapping chunks so context is not lost at boundaries."""
    words = text.split()
    chunks: list[DocumentChunk] = []
    start = 0
    index = 0

    while start < len(words):
        end = min(start + max_words, len(words))
        chunks.append(DocumentChunk(chunk_id=f"{source}:{index}", source=source, text=" ".join(words[start:end])))
        if end == len(words):
            break
        start = max(0, end - overlap)
        index += 1

    return chunks


def build_chunks(directory: str | Path) -> list[DocumentChunk]:
    chunks: list[DocumentChunk] = []
    for source, text in load_documents(directory):
        chunks.extend(chunk_text(source, text))
    return chunks
