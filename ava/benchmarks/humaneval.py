from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Iterable, List, Optional


@dataclass
class CodeExample:
    prompt: str  # Function signature + docstring
    test_code: str  # Test cases
    solution: str  # Reference solution (optional, for evaluation)


def load_from_json(path: str) -> List[CodeExample]:
    """
    Load HumanEval examples from JSON file.

    Expected format: List of dicts with 'prompt', 'test', and 'canonical_solution' keys.
    """
    if not os.path.exists(path):
        return []

    with open(path, "r") as f:
        data = json.load(f)

    examples = []
    for item in data:
        if isinstance(item, dict):
            prompt = item.get("prompt", "")
            test_code = item.get("test", "")
            solution = item.get("canonical_solution", "")
            examples.append(
                CodeExample(prompt=prompt, test_code=test_code, solution=solution)
            )

    return examples


def load_from_path(path: str) -> List[CodeExample]:
    """
    Load HumanEval examples from JSON file.

    Returns empty list if file not found.
    """
    if not os.path.exists(path):
        return []
    return load_from_json(path)


def load_humaneval(
    split: str = "test", data_dir: Optional[str] = None
) -> List[CodeExample]:
    """
    Load HumanEval dataset.

    Args:
        split: "train" or "test" (HumanEval typically only has test)
        data_dir: Directory containing HumanEval JSON. If None, checks
                  HUMANEVAL_DATA_DIR env var.

    Returns:
        List of CodeExample objects
        
    Raises:
        ValueError: If data directory not found or no examples loaded
    """
    if data_dir is None:
        data_dir = os.getenv("HUMANEVAL_DATA_DIR", "")

    if not data_dir:
        raise ValueError(
            "HUMANEVAL_DATA_DIR not set. Please set it in .env file or pass data_dir parameter.\n"
            "Example: HUMANEVAL_DATA_DIR=data/humaneval"
        )
    
    if not os.path.exists(data_dir):
        raise ValueError(f"HumanEval data directory not found: {data_dir}\nPlease check HUMANEVAL_DATA_DIR in .env file.")

    json_path = os.path.join(data_dir, "HumanEval.json")
    if os.path.exists(json_path):
        examples = load_from_json(json_path)
        if examples:
            return examples
    
    # Try alternative naming
    jsonl_path = os.path.join(data_dir, "HumanEval.jsonl")
    if os.path.exists(jsonl_path):
        # JSONL format - one example per line
        examples = []
        with open(jsonl_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                    prompt = item.get("prompt", "")
                    test_code = item.get("test", "")
                    solution = item.get("canonical_solution", "")
                    examples.append(CodeExample(prompt=prompt, test_code=test_code, solution=solution))
                except json.JSONDecodeError:
                    continue
        if examples:
            return examples

    raise ValueError(
        f"No HumanEval data found in {data_dir}.\n"
        f"Expected file: HumanEval.json or HumanEval.jsonl"
    )


def format_prompt(ex: CodeExample, include_tests: bool = False) -> str:
    """Format code example as prompt."""
    prompt = ex.prompt
    if include_tests:
        prompt += f"\n\n# Tests:\n{ex.test_code}\n\n# Solution:"
    else:
        prompt += "\n\n# Solution:"
    return prompt
