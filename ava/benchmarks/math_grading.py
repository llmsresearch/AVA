"""MATH-specific answer extraction and equivalence checking.

Approach follows Hendrycks et al. MATH benchmark conventions:
1. Extract answer: prefer \\boxed{...}, fall back to "Answer:" patterns
2. Normalize LaTeX: strip whitespace, unify fraction forms, remove text wrappers
3. Equivalence check: string equality → numeric → SymPy symbolic
"""

from __future__ import annotations

import re
from typing import Optional

try:
    import sympy
    from sympy.parsing.latex import parse_latex
    SYMPY_AVAILABLE = True
except ImportError:
    SYMPY_AVAILABLE = False


def _find_matching_brace(text: str, start: int) -> int:
    """Find index of matching closing brace for '{' at position start-1."""
    depth = 1
    i = start
    while i < len(text):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return i
        i += 1
    return -1


def extract_boxed(text: str) -> Optional[str]:
    """Extract content of the LAST \\boxed{...} in text, handling nested braces."""
    last_idx = text.rfind("\\boxed{")
    if last_idx < 0:
        return None
    start = last_idx + len("\\boxed{")
    end = _find_matching_brace(text, start)
    if end < 0:
        return None
    return text[start:end].strip()


def extract_answer(text: str) -> str:
    """Extract the model's final answer from a MATH response.

    Priority:
    1. Last \\boxed{...}
    2. After "Answer:" / "The answer is" / "Final answer"
    3. Last numeric/expression token
    """
    if not text:
        return ""

    # Priority 1: \\boxed{} (the MATH standard)
    boxed = extract_boxed(text)
    if boxed is not None:
        return boxed

    # Priority 2: Answer keywords — use rfind so we get the LAST occurrence
    # (the actual final answer, not any "Answer:" in the prompt repeat)
    lower = text.lower()
    # Find the last occurrence of any answer pattern
    best_idx = -1
    best_pat = None
    for pattern in ["final answer:", "final answer is", "the answer is", "answer:", "answer is"]:
        idx = lower.rfind(pattern)
        if idx > best_idx:
            best_idx = idx
            best_pat = pattern
    if best_idx >= 0:
        after = text[best_idx + len(best_pat):].strip()
        # If there's ANOTHER "Answer:" right after (nested pattern), recurse
        if re.match(r'(final\s+)?answer[\s:]*is|(final\s+)?answer\s*:', after.lower()):
            # Re-extract from remainder
            nested = extract_answer(after)
            if nested:
                return nested
        # Strip common LaTeX wrappers
        after = re.sub(r'^\\?\[|\\?\]$', '', after).strip()
        after = re.sub(r'^\\?\(|\\?\)$', '', after).strip()
        after = re.sub(r'^\\text\{[^}]*\}\s*', '', after).strip()
        # Strip leading "}" left over from "\text{Answer: " pattern
        # where the closing brace follows after the colon
        after = re.sub(r'^\}\s*', '', after)
        after = re.sub(r'^[\s:]+', '', after)
        # Take until blank line / sentence break
        after = re.split(r'\n\s*\n|\. [A-Z]', after)[0]
        # Remove trailing period/comma
        after = after.rstrip('.,;').strip()
        # Strip outer quote/bold markers
        after = after.strip('*').strip()
        # If the answer is wrapped in \\boxed{} (nested case), unwrap
        nested_boxed = extract_boxed(after)
        if nested_boxed is not None:
            return nested_boxed
        # If extracted is long prose (>80 chars), try to find a numeric/latex tail
        if len(after) > 80:
            # Try last number or boxed expression
            nums = re.findall(r'-?\d+(?:\.\d+)?(?:/\d+)?', after)
            if nums:
                return nums[-1]
        if after:
            return after

    # Priority 3: Fall back to last line that looks like an answer
    lines = [l.strip() for l in text.strip().split('\n') if l.strip()]
    if lines:
        return lines[-1]
    return text.strip()


def _strip_string(s: str) -> str:
    """Normalize LaTeX string for comparison.

    Based on the reference implementation from the Hendrycks MATH paper.
    """
    # Linebreaks
    s = s.replace("\n", "")
    # Remove inverse spaces
    s = s.replace("\\!", "")
    # Replace \\ with \
    s = s.replace("\\\\", "\\")
    # Replace tfrac and dfrac with frac
    s = s.replace("tfrac", "frac")
    s = s.replace("dfrac", "frac")
    # Remove \left and \right
    s = s.replace("\\left", "")
    s = s.replace("\\right", "")
    # Remove degree marker
    s = s.replace("^{\\circ}", "")
    s = s.replace("^\\circ", "")
    # Remove dollar signs
    s = s.replace("\\$", "")
    s = s.replace("$", "")
    # Remove \text{...} wrappers
    s = re.sub(r'\\text\s*\{\s*([^}]*?)\s*\}', r'\1', s)
    s = re.sub(r'\\mbox\s*\{\s*([^}]*?)\s*\}', r'\1', s)
    s = re.sub(r'\\mathrm\s*\{\s*([^}]*?)\s*\}', r'\1', s)
    s = re.sub(r'\\textbf\s*\{\s*([^}]*?)\s*\}', r'\1', s)
    # Normalize \\frac{a}{b} variations
    s = re.sub(r'\\frac\s+(\w)\s+(\w)', r'\\frac{\1}{\2}', s)
    # Remove spaces
    s = s.replace(" ", "")
    # Remove trailing period
    s = s.rstrip('.')
    # Strip outer braces
    if s.startswith("{") and s.endswith("}"):
        s = s[1:-1]
    # Balance parentheses: drop trailing unmatched close, or add missing close
    open_paren = s.count("(")
    close_paren = s.count(")")
    if open_paren > close_paren:
        s = s + ")" * (open_paren - close_paren)
    elif close_paren > open_paren:
        s = "(" * (close_paren - open_paren) + s
    return s


def _numeric_equiv(a: str, b: str, tol: float = 1e-6) -> bool:
    """Try numeric equivalence."""
    try:
        va = float(a.replace(",", "").replace(" ", ""))
        vb = float(b.replace(",", "").replace(" ", ""))
        return abs(va - vb) < tol
    except (ValueError, TypeError):
        return False


def _sympy_equiv(a: str, b: str) -> bool:
    """Try symbolic equivalence via SymPy."""
    if not SYMPY_AVAILABLE:
        return False
    # Clean for parse_latex
    def prep(s):
        s = s.replace("\\\\", "\\")
        # parse_latex doesn't like \left/\right
        s = s.replace("\\left", "").replace("\\right", "")
        # Remove text wrappers
        s = re.sub(r'\\text\s*\{[^}]*\}', '', s)
        return s.strip()
    try:
        ea = parse_latex(prep(a))
        eb = parse_latex(prep(b))
        if ea is None or eb is None:
            return False
        diff = sympy.simplify(ea - eb)
        return diff == 0
    except Exception:
        return False


def is_equivalent(predicted: str, expected: str) -> bool:
    """Check if predicted math answer is equivalent to expected.

    Cascade: normalize+string eq → numeric → SymPy symbolic.
    """
    if not predicted or not expected:
        return False

    # Normalize both
    p = _strip_string(predicted)
    e = _strip_string(expected)

    # Direct string equality after normalization
    if p == e:
        return True

    # Numeric equivalence (for plain numbers)
    if _numeric_equiv(p, e):
        return True

    # SymPy symbolic equivalence (for expressions)
    if _sympy_equiv(p, e):
        return True

    return False


def check_correct_math_v2(predicted_text: str, expected: str) -> bool:
    """Full MATH evaluator: extract answer from model output, check equivalence."""
    extracted = extract_answer(predicted_text)
    return is_equivalent(extracted, expected)
