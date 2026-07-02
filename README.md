# RAG Pipeline with Evaluation Framework

This project demonstrates an end-to-end Retrieval-Augmented Generation pipeline for enterprise documents. It includes document loading, chunking, indexing, semantic retrieval, answer generation, and a small evaluation framework.

The demo runs locally with a lightweight vector index. The design is compatible with LlamaIndex, Pinecone, and OpenAI embeddings for production use.

## What This Project Shows

- Document chunking and metadata tracking
- Embedding-style semantic search
- Retrieval-augmented answer generation
- Evaluation metrics for retrieval accuracy, answer relevance, and latency
- Caching hooks for repeated queries

## Architecture

```text
Documents -> Chunker -> Embedding Model -> Vector Index -> Retriever -> Answer Generator -> Evaluation Report
```

## Quick Start

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
$env:PYTHONPATH="src"
python -m rag_pipeline.main --question "What is the escalation process for priority incidents?"
python -m rag_pipeline.evaluate
pytest
```

## Production Integration Notes

- Use LlamaIndex document loaders and node parsers for richer source formats.
- Use OpenAI embeddings instead of the local token embedding model.
- Store vectors in Pinecone instead of the local in-memory index.
- Replace simple keyword evaluation with RAGAS, TruLens, or LLM-as-judge scoring.

## Evaluation Metrics

- Retrieval accuracy: whether expected source documents appear in top results
- Answer relevance: keyword overlap between expected answer and generated answer
- Latency: elapsed time for retrieval plus generation
