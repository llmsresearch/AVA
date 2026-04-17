from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

from ava.core.interfaces import Budget
from ava.verification.base import Verifier, HeuristicVerifier


@dataclass
class VerifierLevel:
    """A single verifier level in a cascade."""

    verifier: Verifier
    cost: float  # Cost per call (relative)
    threshold: float  # Confidence threshold to pass this level


class VerifierCascade:
    """
    Cascaded verification with early exits.

    Verifies with cheap heuristics first, only calling expensive verifiers
    when uncertainty is high or cheap verifiers fail.
    """

    def __init__(self, levels: List[VerifierLevel]) -> None:
        self.levels = levels

    def verify(
        self,
        input_text: str,
        output_text: str,
        budget: Budget,
        *,
        early_exit_threshold: float = 0.95,
    ) -> Tuple[bool, float, int]:
        """
        Run cascade verification with early exit.

        Args:
            input_text: Input prompt
            output_text: Generated output
            budget: Compute budget
            early_exit_threshold: If confidence >= this, stop and return

        Returns:
            (is_valid, final_confidence, level_reached)
        """
        confidence = 0.0
        level_reached = 0

        for i, level in enumerate(self.levels):
            if not budget.can_call_verifier():
                break

            is_valid, score = level.verifier.verify(input_text, output_text)
            budget.consume_verify_call()

            # Update confidence (take max or weighted average)
            confidence = max(confidence, score if is_valid else 1.0 - score)

            level_reached = i + 1

            # Early exit if confident enough
            if confidence >= early_exit_threshold:
                break

            # Early exit if cheap verifier clearly rejects
            if not is_valid and score < 0.3:
                break

        is_valid = confidence >= 0.5
        return is_valid, confidence, level_reached


def create_default_cascade() -> VerifierCascade:
    """Create a default 3-level cascade with heuristic verifiers.
    
    Note: For production use, replace with domain-specific verifiers:
    - Level 1: Fast format/length checks
    - Level 2: Medium-cost semantic checks (e.g., LLM-as-judge, partial execution)
    - Level 3: Expensive full validation (e.g., complete execution, formal verification)
    """
    # Level 1: Very cheap heuristic (length, format checks)
    class CheapHeuristic(Verifier):
        def verify(self, input_text: str, output_text: str) -> Tuple[bool, float]:
            is_valid = len(output_text.strip()) > 5
            return is_valid, 0.7 if is_valid else 0.3

    # Level 2: Medium-cost heuristic verifier
    # Level 3: Higher-cost heuristic verifier (for now; replace with executors in production)
    levels = [
        VerifierLevel(verifier=CheapHeuristic(), cost=0.1, threshold=0.6),
        VerifierLevel(verifier=HeuristicVerifier(), cost=1.0, threshold=0.8),
        VerifierLevel(verifier=HeuristicVerifier(), cost=5.0, threshold=0.95),
    ]
    return VerifierCascade(levels)


