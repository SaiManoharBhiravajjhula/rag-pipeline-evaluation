from __future__ import annotations

import argparse
from pathlib import Path

from rag_pipeline.chunking import build_chunks
from rag_pipeline.generator import generate_answer
from rag_pipeline.index import LocalVectorIndex


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the RAG pipeline demo.")
    parser.add_argument("--question", required=True)
    args = parser.parse_args()

    doc_dir = Path(__file__).resolve().parents[2] / "data" / "documents"
    index = LocalVectorIndex()
    index.add(build_chunks(doc_dir))
    results = index.search(args.question)

    print(generate_answer(args.question, results))
    print("\nRetrieved chunks:")
    for result in results:
        print(f"- {result.chunk.chunk_id} score={result.score:.2f}")


if __name__ == "__main__":
    main()