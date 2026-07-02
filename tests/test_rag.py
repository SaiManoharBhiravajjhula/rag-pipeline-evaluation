from rag_pipeline.chunking import chunk_text
from rag_pipeline.index import LocalVectorIndex


def test_chunking_creates_chunks():
    chunks = chunk_text("sample.txt", "one two three four five", max_words=3, overlap=1)
    assert len(chunks) == 2
    assert chunks[0].source == "sample.txt"


def test_retrieval_finds_relevant_chunk():
    chunks = chunk_text("incident.txt", "priority incidents require escalation and customer updates")
    index = LocalVectorIndex()
    index.add(chunks)
    results = index.search("incident escalation")
    assert results[0].chunk.source == "incident.txt"
    assert results[0].score > 0
