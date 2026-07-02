from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

from rag_pipeline.chunking import build_chunks
from rag_pipeline.embeddings import tokenize
from rag_pipeline.generator import generate_answer
from rag_pipeline.index import LocalVectorIndex


@dataclass(frozen=True)
class EvaluationCase:
    question: str
    expected_source: str
    expected_terms: set[str]


@dataclass(frozen=True)
class EvaluationResult:
    question: str
    retrieval_hit: bool
    answer_relevance: float
    latency_ms: float


CASES = [
    EvaluationCase(
        "What is the escalation process for priority incidents?",
        "incident_response.txt",
        {"escalation", "incident", "owner", "updates"},
    ),
    EvaluationCase(
        "How is production customer data protected?",
        "security_policy.txt",
        {"encrypted", "approval", "access", "review"},
    ),
]


def evaluate_case(case: EvaluationCase, index: LocalVectorIndex) -> EvaluationResult:
    start = time.perf_counter()
    results = index.search(case.question, top_k=2)
    answer = generate_answer(case.question, results)
    latency_ms = (time.perf_counter() - start) * 1000

    retrieved_sources = {result.chunk.source for result in results}
    answer_terms = set(tokenize(answer))
    relevance = len(case.expected_terms & answer_terms) / len(case.expected_terms)

    return EvaluationResult(
        question=case.question,
        retrieval_hit=case.expected_source in retrieved_sources,
        answer_relevance=relevance,
        latency_ms=latency_ms,
    )


def run_evaluation(doc_dir: str | Path | None = None) -> list[EvaluationResult]:
    directory = Path(doc_dir) if doc_dir else Path(__file__).resolve().parents[2] / "data" / "documents"
    index = LocalVectorIndex()
    index.add(build_chunks(directory))
    return [evaluate_case(case, index) for case in CASES]


def main() -> None:
    for result in run_evaluation():
        print(
            f"{result.question}\n"
            f"  retrieval_hit={result.retrieval_hit}\n"
            f"  answer_relevance={result.answer_relevance:.2f}\n"
            f"  latency_ms={result.latency_ms:.2f}"
        )


if __name__ == "__main__":
    main()