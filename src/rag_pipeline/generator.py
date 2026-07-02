from __future__ import annotations

from rag_pipeline.index import RetrievalResult


def generate_answer(question: str, results: list[RetrievalResult]) -> str:
    """Create a grounded answer from retrieved context."""
    useful_results = [result for result in results if result.score > 0]

    if not useful_results:
        return "I could not find enough information in the indexed documents."

    best = useful_results[0].chunk
    sources = ", ".join(sorted({result.chunk.source for result in useful_results}))

    return f"Answer based on {sources}: {best.text}\n\nQuestion handled: {question}"