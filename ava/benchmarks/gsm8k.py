from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Iterable, List, Optional


@dataclass
class QAExample:
    question: str
    answer: str


def load_from_json(path: str) -> List[QAExample]:
    """
    Load GSM8K examples from JSON file.

    Expected format: List of dicts with 'question' and 'answer' keys.
    """
    if not os.path.exists(path):
        return []

    with open(path, "r") as f:
        data = json.load(f)

    examples = []
    for item in data:
        if isinstance(item, dict):
            question = item.get("question", "")
            answer = item.get("answer", "")
            # Extract final answer number if answer is formatted
            if answer and "####" in answer:
                answer = answer.split("####")[-1].strip()
            examples.append(QAExample(question=question, answer=answer))
        elif isinstance(item, str):
            # Fallback: treat as question
            examples.append(QAExample(question=item, answer=""))

    return examples


def prompts_from_examples(examples: Iterable[QAExample]) -> List[str]:
    """Format examples as prompts for model input."""
    return [f"Question: {ex.question}\nAnswer:" for ex in examples]


def load_gsm8k(
    split: str = "test", data_dir: Optional[str] = None
) -> List[QAExample]:
    """
    Load GSM8K dataset.

    Args:
        split: "train" or "test"
        data_dir: Directory containing GSM8K JSON files. If None, checks
                  GSM8K_DATA_DIR env var.

    Returns:
        List of QAExample objects
        
    Raises:
        ValueError: If data directory not found or no examples loaded
    """
    if data_dir is None:
        data_dir = os.getenv("GSM8K_DATA_DIR", "")

    if not data_dir:
        raise ValueError(
            "GSM8K_DATA_DIR not set. Please set it in .env file or pass data_dir parameter.\n"
            "Example: GSM8K_DATA_DIR=data/gsm8k"
        )

    if not os.path.exists(data_dir):
        raise ValueError(f"GSM8K data directory not found: {data_dir}\nPlease check GSM8K_DATA_DIR in .env file.")

    # Try JSONL first (common format)
    jsonl_path = os.path.join(data_dir, f"gsm8k_{split}.jsonl")
    if os.path.exists(jsonl_path):
        examples = load_from_jsonl(jsonl_path)
        if examples:
            return examples
    
    json_path = os.path.join(data_dir, f"gsm8k_{split}.json")
    if os.path.exists(json_path):
        examples = load_from_json(json_path)
        if examples:
            return examples

    # Try alternative naming
    alt_path = os.path.join(data_dir, f"{split}.jsonl")
    if os.path.exists(alt_path):
        examples = load_from_jsonl(alt_path)
        if examples:
            return examples

    raise ValueError(
        f"No GSM8K {split} data found in {data_dir}.\n"
        f"Expected files: gsm8k_{split}.jsonl, gsm8k_{split}.json, or {split}.jsonl"
    )


def load_from_jsonl(path: str) -> List[QAExample]:
    """Load from JSONL format (one JSON object per line)."""
    examples = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
                question = item.get("question", "")
                answer = item.get("answer", "")
                # Extract final answer number from GSM8K format
                if "####" in answer:
                    answer = answer.split("####")[-1].strip()
                elif isinstance(answer, str) and answer.strip():
                    # Sometimes answer is just the number
                    answer = answer.strip()
                examples.append(QAExample(question=question, answer=answer))
            except (json.JSONDecodeError, KeyError) as e:
                continue
    return examples
