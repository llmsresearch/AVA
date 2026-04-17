from __future__ import annotations

from typing import Optional, Tuple

from ava.baselines.self_consistency import _extract_answer_signature
from ava.core.interfaces import Budget, ModelProvider


class LLMVerifier:
    """Re-solve-and-vote verifier.

    Instead of asking the model "is this correct?" (which suffers from
    confirmation bias and doesn't work with temperature-locked models like
    GPT-5), this verifier re-solves the problem independently and compares
    the answer to the proposed one.

    This provides:
    1. A verification signal (agree/disagree with proposed answer)
    2. An additional answer that can be added to the vote pool
    """

    def __init__(self, model: ModelProvider) -> None:
        self.model = model

    def verify(
        self,
        input_text: str,
        output_text: str,
        budget: Budget,
    ) -> Tuple[bool, float]:
        """Verify by re-solving and comparing answers.

        Returns:
            (agrees_with_proposed, confidence) tuple.
        """
        if not budget.can_use_tokens(10):
            return True, 0.5

        proposed_sig = _extract_answer_signature(output_text)

        verify_prompt = (
            f"{input_text}\n\n"
            f"Double-check your work carefully. What is the final answer?"
        )

        try:
            response = self.model.generate(
                verify_prompt, max_tokens=512
            )
            tokens_used = response.metadata.get("tokens", 50)
            try:
                budget.consume_tokens(tokens_used)
            except RuntimeError:
                budget.tokens_used = budget.token_limit

            if not response.text or not response.text.strip():
                return True, 0.5

            verify_sig = _extract_answer_signature(response.text)

            if verify_sig == proposed_sig:
                return True, 0.9
            else:
                return False, 0.3
        except Exception:
            return True, 0.5

    def verify_and_vote(
        self,
        input_text: str,
        output_text: str,
        budget: Budget,
    ) -> Tuple[bool, float, Optional[str], Optional[str]]:
        """Re-solve, compare, and return the new answer for the vote pool.

        Returns:
            (agrees, confidence, new_answer_text, new_answer_signature)
        """
        if not budget.can_use_tokens(10):
            return True, 0.5, None, None

        proposed_sig = _extract_answer_signature(output_text)

        verify_prompt = (
            f"{input_text}\n\n"
            f"Double-check your work carefully. What is the final answer?"
        )

        try:
            response = self.model.generate(
                verify_prompt, max_tokens=512
            )
            tokens_used = response.metadata.get("tokens", 50)
            try:
                budget.consume_tokens(tokens_used)
            except RuntimeError:
                budget.tokens_used = budget.token_limit

            if not response.text or not response.text.strip():
                return True, 0.5, None, None

            verify_sig = _extract_answer_signature(response.text)
            agrees = verify_sig == proposed_sig
            confidence = 0.9 if agrees else 0.3

            return agrees, confidence, response.text.strip(), verify_sig
        except Exception:
            return True, 0.5, None, None
