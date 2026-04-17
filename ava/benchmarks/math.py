"""
MATH Dataset Loader

The MATH dataset contains challenging competition mathematics problems
organized by difficulty level (1-5) and subject area (algebra, geometry, etc.).

Used for out-of-distribution testing of calibration transfer from GSM8K.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class MATHExample:
    """A single MATH problem with metadata."""

    problem: str
    solution: str
    answer: str  # Final boxed answer
    level: int  # Difficulty 1-5
    subject: str  # e.g., "algebra", "geometry", "number_theory"


def extract_boxed_answer(solution: str) -> str:
    """
    Extract the final answer from a MATH solution string.

    MATH solutions contain the answer in \\boxed{...} format.
    """
    # Find the last \boxed{...} in the solution
    boxed_start = solution.rfind("\\boxed{")
    if boxed_start == -1:
        # No boxed answer, try to extract last number or expression
        return ""

    # Find matching closing brace
    brace_count = 0
    start_idx = boxed_start + len("\\boxed{")
    for i, char in enumerate(solution[start_idx:], start=start_idx):
        if char == "{":
            brace_count += 1
        elif char == "}":
            if brace_count == 0:
                return solution[start_idx:i].strip()
            brace_count -= 1

    return ""


def load_from_json(path: str) -> List[MATHExample]:
    """
    Load MATH examples from a JSON file.

    Expected format: List of dicts with 'problem', 'solution', 'level', 'type' keys.
    """
    if not os.path.exists(path):
        return []

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    examples = []
    for item in data:
        if not isinstance(item, dict):
            continue

        problem = item.get("problem", "")
        solution = item.get("solution", "")
        level_str = item.get("level", "Level 1")
        subject = item.get("type", "unknown")

        # Parse level number from string like "Level 3"
        try:
            level = int(level_str.replace("Level ", ""))
        except (ValueError, AttributeError):
            level = 3  # Default to medium difficulty

        # Extract boxed answer
        answer = extract_boxed_answer(solution)

        examples.append(
            MATHExample(
                problem=problem,
                solution=solution,
                answer=answer,
                level=level,
                subject=subject,
            )
        )

    return examples


def load_from_jsonl(path: str) -> List[MATHExample]:
    """Load MATH examples from JSONL format (one JSON object per line)."""
    examples = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
                problem = item.get("problem", "")
                solution = item.get("solution", "")
                level_str = item.get("level", "Level 3")
                subject = item.get("type", "unknown")

                try:
                    level = int(level_str.replace("Level ", ""))
                except (ValueError, AttributeError):
                    level = 3

                answer = extract_boxed_answer(solution)
                examples.append(
                    MATHExample(
                        problem=problem,
                        solution=solution,
                        answer=answer,
                        level=level,
                        subject=subject,
                    )
                )
            except json.JSONDecodeError:
                continue

    return examples


def load_math(
    split: str = "test",
    data_dir: Optional[str] = None,
    difficulty: Optional[int] = None,
    subject: Optional[str] = None,
) -> List[MATHExample]:
    """
    Load MATH dataset.

    Args:
        split: "train" or "test"
        data_dir: Directory containing MATH data. If None, checks MATH_DATA_DIR env var.
        difficulty: Filter to specific difficulty level (1-5). None for all.
        subject: Filter to specific subject (e.g., "algebra"). None for all.

    Returns:
        List of MATHExample objects

    Raises:
        ValueError: If data directory not found or no examples loaded
    """
    if data_dir is None:
        data_dir = os.getenv("MATH_DATA_DIR", "")

    if not data_dir:
        raise ValueError(
            "MATH_DATA_DIR not set. Please set it in .env file or pass data_dir parameter.\n"
            "Example: MATH_DATA_DIR=data/MATH"
        )

    if not os.path.exists(data_dir):
        raise ValueError(
            f"MATH data directory not found: {data_dir}\nPlease check MATH_DATA_DIR in .env file."
        )

    # Try different file patterns
    patterns = [
        f"math_{split}.jsonl",
        f"MATH_{split}.jsonl",
        f"{split}.jsonl",
        f"math_{split}.json",
        f"MATH_{split}.json",
        f"{split}.json",
    ]

    examples = []
    for pattern in patterns:
        path = os.path.join(data_dir, pattern)
        if os.path.exists(path):
            if pattern.endswith(".jsonl"):
                examples = load_from_jsonl(path)
            else:
                examples = load_from_json(path)
            if examples:
                break

    if not examples:
        raise ValueError(
            f"No MATH {split} data found in {data_dir}.\n"
            f"Tried patterns: {patterns}"
        )

    # Apply filters
    if difficulty is not None:
        examples = [ex for ex in examples if ex.level == difficulty]

    if subject is not None:
        examples = [ex for ex in examples if ex.subject.lower() == subject.lower()]

    return examples


def prompts_from_examples(examples: List[MATHExample]) -> List[str]:
    """Format MATH examples as prompts for model input."""
    return [
        f"Problem: {ex.problem}\n\nSolution:"
        for ex in examples
    ]


def get_difficulty_distribution(examples: List[MATHExample]) -> dict:
    """Get distribution of difficulty levels in the dataset."""
    counts = {i: 0 for i in range(1, 6)}
    for ex in examples:
        if 1 <= ex.level <= 5:
            counts[ex.level] += 1
    return counts


def get_subject_distribution(examples: List[MATHExample]) -> dict:
    """Get distribution of subjects in the dataset."""
    counts = {}
    for ex in examples:
        counts[ex.subject] = counts.get(ex.subject, 0) + 1
    return counts
