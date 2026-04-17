from __future__ import annotations

import re
from collections import Counter
from typing import List, Tuple

from ava.core.interfaces import Budget, Generation, ModelProvider


def _extract_answer_signature(text: str) -> str:
    """Extract a compact answer signature from verbose model output.

    Shared logic with self_consistency._extract_answer_signature.
    Works for both numeric (GSM8K) and text (HotpotQA) answers.
    """
    from ava.baselines.self_consistency import _extract_answer_signature as _sc_extract
    return _sc_extract(text)


def confidence_early_exit(
    prompt: str,
    model: ModelProvider,
    budget: Budget,
    *,
    threshold: float = 0.8,
    batch_size: int = 2,
    max_rounds: int = 5,
    temperature: float = 0.8,
) -> Tuple[str, float]:
    """Confidence-threshold early-exit baseline.

    Generates answers in batches, computes self-consistency confidence
    after each batch, and stops as soon as confidence >= threshold.

    Votes on extracted answer signatures (final numbers / short answers)
    rather than full text, so that different verbose explanations leading
    to the same answer are counted as agreement.

    This is the simplest adaptive compute baseline: it spends more tokens
    only when the model is uncertain (low agreement among samples).

    Args:
        prompt: Input prompt.
        model: Model provider.
        budget: Token budget.
        threshold: Confidence threshold to stop (fraction of majority votes).
        batch_size: Number of samples per round.
        max_rounds: Maximum sampling rounds before forced stop.
        temperature: Sampling temperature.

    Returns:
        (best_answer, confidence) tuple. best_answer is the full text of
        the most recent generation whose signature matches the majority.
    """
    # Track votes on signatures, and store full text for each signature
    signature_votes: Counter = Counter()
    signature_to_full_text: dict[str, str] = {}

    for _round in range(max_rounds):
        # Generate a batch of samples
        for _ in range(batch_size):
            if not budget.can_use_tokens(50):
                break
            g = model.generate(
                prompt,
                temperature=temperature,
                max_tokens=min(512, budget.token_limit - budget.tokens_used),
            )
            tokens_used = g.metadata.get("tokens", 50)  # type: ignore[arg-type]
            try:
                budget.consume_tokens(tokens_used)
            except RuntimeError:
                break
            if g.text and g.text.strip():
                sig = _extract_answer_signature(g.text)
                signature_votes[sig] += 1
                signature_to_full_text[sig] = g.text.strip()

        if not signature_votes:
            continue

        # Compute confidence = fraction of votes for majority signature
        total_votes = sum(signature_votes.values())
        majority_sig, majority_count = signature_votes.most_common(1)[0]
        confidence = majority_count / total_votes

        # Early exit if confident enough
        if confidence >= threshold:
            return signature_to_full_text[majority_sig], confidence

        # Check if budget allows another round
        if not budget.can_use_tokens(50):
            break

    # Budget exhausted or max rounds reached: return best answer
    if signature_votes:
        majority_sig = signature_votes.most_common(1)[0][0]
        total_votes = sum(signature_votes.values())
        confidence = signature_votes[majority_sig] / total_votes
        return signature_to_full_text[majority_sig], confidence

    return "", 0.0
