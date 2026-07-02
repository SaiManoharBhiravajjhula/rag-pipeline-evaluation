from __future__ import annotations

import math
from collections import Counter


def tokenize(text: str) -> list[str]:
    return [word.strip(".,!?;:()[]").lower() for word in text.split() if word.strip()]


class LocalEmbeddingModel:
    """Small local embedding substitute; swap for OpenAI embeddings in production."""

    def embed(self, text: str) -> dict[str, float]:
        counts = Counter(tokenize(text))
        norm = math.sqrt(sum(value * value for value in counts.values())) or 1.0
        return {token: value / norm for token, value in counts.items()}


def cosine_similarity(left: dict[str, float], right: dict[str, float]) -> float:
    return sum(left.get(token, 0.0) * right.get(token, 0.0) for token in set(left) | set(right))
