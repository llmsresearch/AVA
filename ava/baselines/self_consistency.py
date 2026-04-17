from __future__ import annotations

import re
from collections import Counter
from typing import Tuple

from ava.core.interfaces import Budget, ModelProvider


def _normalize_number(s: str) -> str:
    """Normalize a numeric string: '18.00' → '18', '70,000' → '70000'."""
    s = s.replace(",", "").strip("$%() ")
    try:
        val = float(s)
        if val == int(val):
            return str(int(val))
        return str(val)
    except ValueError:
        return s


def _extract_answer_signature(text: str) -> str:
    """Extract a compact answer signature from verbose model output.

    Works for both numeric (GSM8K) and text (HotpotQA) answers.
    Extracts the core answer for voting so that different verbose
    explanations leading to the same answer count as agreement.
    """
    if not text:
        return ""
    text = text.strip()

    # GSM8K-style #### delimiter
    if "####" in text:
        raw = text.split("####")[-1].strip()
        nums = re.findall(r'-?\d+(?:,\d{3})*(?:\.\d+)?', raw)
        if nums:
            return _normalize_number(nums[-1])
        return raw.lower()

    lower = text.lower()

    # "ANSWER: X" structured format — return the FULL expression, not just
    # the first number. MATH answers can be fractions, coordinates,
    # expressions like "(3, π/2)" or "\\frac{14}{3}".
    for pattern in ["answer:", "the answer is"]:
        if pattern in lower:
            idx = lower.rfind(pattern)
            after = text[idx + len(pattern):].strip()
            answer = after.split("\n")[0].strip()
            answer = answer.strip(".,;:!?\"'")
            # Normalize whitespace and common LaTeX variations
            answer = answer.replace("\\dfrac", "\\frac").replace("\\tfrac", "\\frac")
            answer = answer.replace("\\left", "").replace("\\right", "")
            answer = re.sub(r'\s+', ' ', answer).strip()
            # If answer is PURELY numeric, normalize it
            try:
                val = float(answer.replace(",", "").strip())
                return _normalize_number(answer)
            except (ValueError, TypeError):
                pass
            if answer:
                return answer.lower()

    # Yes/No detection (common in QA)
    first_word = text.split()[0].lower().strip(".,;:!?") if text.split() else ""
    if first_word in ("yes", "no"):
        return first_word

    # Last number in text (math problems)
    numbers = re.findall(r'-?\d+(?:,\d{3})*(?:\.\d+)?', text)
    if numbers:
        return _normalize_number(numbers[-1])

    # Fallback: last non-empty line (normalized)
    lines = [l.strip() for l in text.strip().split("\n") if l.strip()]
    if lines:
        return lines[-1][:100].lower()

    return text[:100].lower()


def self_consistency(
    prompt: str,
    model: ModelProvider,
    budget: Budget,
    *,
    k: int = 5,
    temperature: float = 0.8,
) -> Tuple[str, Counter]:
    """Sample k completions and majority-vote on extracted answer signatures.

    Votes on compact answer signatures (final numbers) rather than full text,
    so that different verbose explanations leading to the same answer are
    counted as agreement.

    Returns (final_answer_full_text, signature_histogram)
    """
    generations = []
    for _ in range(k):
        estimated_tokens = 100
        if not budget.can_use_tokens(estimated_tokens):
            break
        g = model.generate(prompt, temperature=temperature, max_tokens=512)
        generations.append(g)
        tokens_used = g.metadata.get("tokens", estimated_tokens)  # type: ignore[arg-type]
        try:
            budget.consume_tokens(tokens_used)
        except RuntimeError:
            break

    if not generations:
        return "", Counter()

    # Vote on extracted signatures, not full text
    sig_to_full_text: dict[str, str] = {}
    signature_votes: Counter = Counter()
    for g in generations:
        if g.text and g.text.strip():
            sig = _extract_answer_signature(g.text)
            signature_votes[sig] += 1
            sig_to_full_text[sig] = g.text.strip()

    if not signature_votes:
        return "", Counter()

    majority_sig = signature_votes.most_common(1)[0][0]
    return sig_to_full_text[majority_sig], signature_votes
