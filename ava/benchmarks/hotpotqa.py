from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Iterable, List, Optional


@dataclass
class QAExample:
    question: str
    answer: str
    context: Optional[str] = None  # For multi-hop, may include supporting facts


def load_from_json(path: str) -> List[QAExample]:
    """
    Load HotpotQA examples from JSON file.

    Expected format: List of dicts with 'question', 'answer', and optionally 'context'/'supporting_facts'.
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
            # Try multiple keys for context
            context = (
                item.get("context")
                or item.get("supporting_facts")
                or item.get("contexts")
            )
            if isinstance(context, list):
                context = " ".join(str(c) for c in context)
            examples.append(QAExample(question=question, answer=answer, context=context))

    return examples


def load_from_jsonl(path: str) -> List[QAExample]:
    """Load from JSONL format."""
    examples = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
                question = item.get("question", "")
                answer = item.get("answer", "")
                context = item.get("context") or item.get("supporting_facts")
                if isinstance(context, list):
                    context = " ".join(str(c) for c in context)
                examples.append(QAExample(question=question, answer=answer, context=context))
            except json.JSONDecodeError:
                continue
    return examples


def load_from_path(path: str) -> List[QAExample]:
    """Load HotpotQA examples from JSON or JSONL file."""
    if not os.path.exists(path):
        return []
    if path.endswith(".jsonl"):
        return load_from_jsonl(path)
    return load_from_json(path)


def load_hotpotqa(
    split: str = "dev", data_dir: Optional[str] = None
) -> List[QAExample]:
    """
    Load HotpotQA dataset.

    Args:
        split: "train", "dev", or "test"
        data_dir: Directory containing HotpotQA JSON files. If None, checks
                  HOTPOTQA_DATA_DIR env var.

    Returns:
        List of QAExample objects
        
    Raises:
        ValueError: If data directory not found or no examples loaded
    """
    if data_dir is None:
        data_dir = os.getenv("HOTPOTQA_DATA_DIR", "")

    if not data_dir:
        raise ValueError(
            "HOTPOTQA_DATA_DIR not set. Please set it in .env file or pass data_dir parameter.\n"
            "Example: HOTPOTQA_DATA_DIR=data/hotpotqa"
        )
    
    if not os.path.exists(data_dir):
        raise ValueError(f"HotpotQA data directory not found: {data_dir}\nPlease check HOTPOTQA_DATA_DIR in .env file.")

    json_path = os.path.join(data_dir, f"hotpotqa_{split}.json")
    if os.path.exists(json_path):
        examples = load_from_json(json_path)
        if examples:
            return examples

    alt_path = os.path.join(data_dir, f"{split}.jsonl")
    if os.path.exists(alt_path):
        examples = load_from_jsonl(alt_path)
        if examples:
            return examples

    raise ValueError(
        f"No HotpotQA {split} data found in {data_dir}.\n"
        f"Expected files: hotpotqa_{split}.json or {split}.jsonl"
    )


def prompts_from_examples(examples: Iterable[QAExample]) -> List[str]:
    """Format examples as prompts for model input."""
    prompts = []
    for ex in examples:
        if ex.context:
            prompt = f"Context: {ex.context}\nQuestion: {ex.question}\nAnswer:"
        else:
            prompt = f"Question: {ex.question}\nAnswer:"
        prompts.append(prompt)
    return prompts
