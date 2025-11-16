from __future__ import annotations

from collections import Counter
from typing import Tuple

from ava.core.interfaces import Budget, ModelProvider


def self_consistency(
    prompt: str,
    model: ModelProvider,
    budget: Budget,
    *,
    k: int = 5,
    temperature: float = 0.8,
) -> Tuple[str, Counter]:
    """Sample k completions and majority-vote the final short answer.

    Returns (final_answer, histogram)
    """
    from ava.core.interfaces import Generation
    
    generations = []
    for _ in range(k):
        # Estimate tokens needed (conservative)
        estimated_tokens = 100
        if not budget.can_use_tokens(estimated_tokens):
            break
        g = model.generate(prompt, temperature=temperature, max_tokens=512)
        generations.append(g)
        # Use actual token count from API response
        tokens_used = g.metadata.get("tokens", estimated_tokens)  # type: ignore[arg-type]
        try:
            budget.consume_tokens(tokens_used)
        except RuntimeError:
            break
    votes = Counter(g.text.strip() for g in generations)
    final = votes.most_common(1)[0][0] if votes else ""
    return final, votes



