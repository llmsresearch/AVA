from __future__ import annotations

from typing import Protocol, Tuple


class Verifier(Protocol):
    """Protocol for verifiers that check output validity/quality."""

    def verify(self, input_text: str, output_text: str) -> Tuple[bool, float]:
        """
        Verify output correctness and return score.

        Args:
            input_text: Input prompt/context
            output_text: Generated output to verify

        Returns:
            (is_valid: bool, score: float in [0,1])
        """
        ...


class HeuristicVerifier:
    """Simple heuristic verifier using basic checks."""
    
    def verify(self, input_text: str, output_text: str) -> Tuple[bool, float]:
        """
        Verify output using heuristics (length, format, basic structure).
        
        This is a lightweight verifier. For production use, implement
        domain-specific verifiers (executors, parsers, etc.).
        """
        output = output_text.strip()
        if not output:
            return False, 0.0
        
        # Basic heuristics
        has_content = len(output) > 3
        has_structure = any(char in output for char in [' ', '\n', '\t']) or len(output) < 50
        
        score = 0.5
        if has_content:
            score += 0.2
        if has_structure:
            score += 0.3
        
        is_valid = score >= 0.5
        return is_valid, min(1.0, score)


