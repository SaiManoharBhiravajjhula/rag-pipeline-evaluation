from __future__ import annotations

from dataclasses import dataclass

from rag_pipeline.chunking import DocumentChunk
from rag_pipeline.embeddings import LocalEmbeddingModel, cosine_similarity


@dataclass(frozen=True)
class RetrievalResult:
    chunk: DocumentChunk
    score: float


class LocalVectorIndex:
    def __init__(self, embedding_model: LocalEmbeddingModel | None = None) -> None:
        self.embedding_model = embedding_model or LocalEmbeddingModel()
        self._rows: list[tuple[DocumentChunk, dict[str, float]]] = []

    def add(self, chunks: list[DocumentChunk]) -> None:
        for chunk in chunks:
            self._rows.append((chunk, self.embedding_model.embed(chunk.text)))

    def search(self, query: str, top_k: int = 3) -> list[RetrievalResult]:
        query_embedding = self.embedding_model.embed(query)
        scored = [RetrievalResult(chunk=chunk, score=cosine_similarity(query_embedding, embedding)) for chunk, embedding in self._rows]
        return sorted(scored, key=lambda result: result.score, reverse=True)[:top_k]
