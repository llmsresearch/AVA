"""
Full Evaluation Runner for AVA

Runs AVA and all baselines on specified benchmarks with configurable model backend.
Produces JSON results and LaTeX tables for paper.

Usage:
    python experiments/run_full_evaluation.py --model ollama --benchmarks gsm8k,hotpotqa,humaneval
    python experiments/run_full_evaluation.py --model ollama --benchmarks gsm8k --methods ava,difficulty_bin --budgets 400,600,800
    python experiments/run_full_evaluation.py --model ollama --benchmarks gsm8k --limit 50
"""

import argparse
import json
import os
import re
import sys
import time
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from dotenv import load_dotenv

# Load .env from project root
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ava.agents.ava_agent import AVAAgent
from ava.baselines.confidence_early_exit import confidence_early_exit
from ava.baselines.difficulty_bin import difficulty_bin_solve
from ava.baselines.fixed_depth_search import FixedDepthTreeSearch
from ava.baselines.self_consistency import self_consistency
from ava.core.interfaces import Budget, Generation, ModelProvider
from ava.verification.cascade import create_default_cascade


@dataclass
class EvalResult:
    """Result from evaluating a single example."""

    method: str
    dataset: str
    budget_limit: int
    correct: bool
    tokens_used: int
    answer: str
    expected: str
    prompt_length: int


def create_model(model_type: str, model_name: Optional[str] = None) -> ModelProvider:
    """Create model provider based on type."""
    if model_type == "ollama":
        from ava.models.ollama_model import OllamaModel
        name = model_name or "qwen2.5:14b"
        return OllamaModel(model_name=name)
    elif model_type == "azure":
        from ava.models.azure_openai import AzureOpenAIModel
        return AzureOpenAIModel()
    else:
        raise ValueError(f"Unknown model type: {model_type}. Use 'ollama' or 'azure'.")


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


def extract_answer_gsm8k(text: str) -> str:
    """Extract numeric answer from GSM8K response."""
    if not text:
        return ""
    # Look for #### delimiter
    if "####" in text:
        raw = text.split("####")[-1].strip()
        nums = re.findall(r'-?\d+(?:,\d{3})*(?:\.\d+)?', raw)
        if nums:
            return _normalize_number(nums[-1])
        return raw
    # Look for "the answer is" / "answer:" pattern
    lower = text.lower()
    for pattern in ["the answer is", "answer:", "= "]:
        if pattern in lower:
            idx = lower.rfind(pattern)
            after = text[idx + len(pattern):].strip()
            nums = re.findall(r'-?\d+(?:,\d{3})*(?:\.\d+)?', after.split("\n")[0])
            if nums:
                return _normalize_number(nums[0])
    # Last resort: find the last number in the text
    numbers = re.findall(r'-?\d+(?:,\d{3})*(?:\.\d+)?', text)
    if numbers:
        return _normalize_number(numbers[-1])
    return text.strip()


def extract_answer_qa(text: str) -> str:
    """Extract answer from QA response (HotpotQA).

    Looks for 'Answer:' anywhere in text (model puts reasoning first),
    then falls back to start-of-text extraction.
    """
    if not text:
        return ""
    text = text.strip()

    # Look for "Answer:" anywhere in the text (GPT-5 puts reasoning first)
    lower = text.lower()
    for pattern in ["answer:", "the answer is"]:
        if pattern in lower:
            idx = lower.rfind(pattern)
            after = text[idx + len(pattern):].strip()
            # Take until end of line or period
            answer = re.split(r'[\n]', after)[0].strip()
            answer = answer.strip(".,;:!?\"'")
            if answer:
                return answer

    # Yes/No at start of response
    first_word = text.split()[0].lower().strip(".,;:!?") if text.split() else ""
    if first_word in ("yes", "no"):
        return first_word

    # Fallback: first sentence
    for sep in [".", "\n"]:
        if sep in text:
            return text[:text.index(sep)].strip()

    return text.strip()


def extract_answer_code(text: str) -> str:
    """Extract code from HumanEval response."""
    if not text:
        return ""
    # Extract code block if present
    if "```python" in text:
        start = text.index("```python") + len("```python")
        end = text.index("```", start) if "```" in text[start:] else len(text)
        return text[start:end].strip()
    if "```" in text:
        start = text.index("```") + 3
        end = text.index("```", start) if "```" in text[start:] else len(text)
        return text[start:end].strip()
    return text.strip()


def normalize_answer(answer: str) -> str:
    """Normalize answer for comparison."""
    answer = answer.strip().lower()
    # Remove common prefixes/suffixes
    answer = answer.strip(".,;:!?\"'")
    # Normalize whitespace
    answer = " ".join(answer.split())
    return answer


def normalize_math_answer(s: str) -> str:
    """Normalize a MATH answer for comparison.

    Handles LaTeX formatting, fractions, etc.
    """
    s = s.strip()
    # Remove \boxed{} wrapper
    if "\\boxed{" in s:
        start = s.index("\\boxed{") + len("\\boxed{")
        depth = 1
        for i in range(start, len(s)):
            if s[i] == "{": depth += 1
            elif s[i] == "}": depth -= 1
            if depth == 0:
                s = s[start:i]
                break
    # Remove dollar signs and whitespace
    s = s.replace("$", "").replace(" ", "").strip()
    # Normalize common LaTeX
    s = s.replace("\\left(", "(").replace("\\right)", ")")
    s = s.replace("\\left[", "[").replace("\\right]", "]")
    s = s.replace("\\%", "%").replace("\\$", "$")
    s = s.replace("\\text{", "").replace("\\mathrm{", "").replace("\\textbf{", "")
    s = s.replace("\\dfrac", "\\frac")
    s = s.replace("\\tfrac", "\\frac")
    # Try numeric conversion
    try:
        val = float(s.replace(",", ""))
        if val == int(val):
            return str(int(val))
        return str(val)
    except ValueError:
        pass
    return s.lower()


def check_correct_math(predicted: str, expected: str) -> bool:
    """Check if MATH answer is correct."""
    pred_raw = extract_answer_gsm8k(predicted)  # Extract answer from model output
    pred = normalize_math_answer(pred_raw)
    exp = normalize_math_answer(expected)
    if pred == exp:
        return True
    # Try numeric comparison
    try:
        return abs(float(pred) - float(exp)) < 1e-5
    except (ValueError, TypeError):
        pass
    # Containment check for short answers
    if len(exp) <= 10 and exp in pred:
        return True
    return False


def check_correct_gsm8k(predicted: str, expected: str) -> bool:
    """Check if GSM8K answer is correct (numeric comparison)."""
    pred = extract_answer_gsm8k(predicted)
    exp = expected.strip()
    # Normalize both to numbers
    try:
        pred_num = float(pred.replace(",", ""))
        exp_num = float(exp.replace(",", ""))
        return abs(pred_num - exp_num) < 1e-5
    except (ValueError, TypeError):
        return normalize_answer(pred) == normalize_answer(exp)


def check_correct_qa(predicted: str, expected: str) -> bool:
    """Check if QA answer is correct (containment match after normalization)."""
    pred = normalize_answer(extract_answer_qa(predicted))
    exp = normalize_answer(expected)
    if not pred or not exp:
        return False
    # Exact match
    if pred == exp:
        return True
    # Containment: predicted contains expected or vice versa
    if exp in pred or pred in exp:
        return True
    return False


def check_correct_code(predicted: str, expected: str, test_code: str) -> bool:
    """Check if code is correct (simplified: exact match of canonical solution)."""
    # In a full evaluation we would execute the code with test cases.
    # For this evaluation, we compare normalized outputs.
    pred = extract_answer_code(predicted)
    exp = expected.strip()
    return normalize_answer(pred) == normalize_answer(exp)


def run_method(
    method: str,
    prompt: str,
    model: ModelProvider,
    budget_limit: int,
) -> Tuple[str, int]:
    """
    Run a single method on a single prompt.

    Returns (answer_text, tokens_used).
    """
    budget = Budget(
        token_limit=budget_limit,
        tool_calls_limit=20,
        verify_calls_limit=20,
    )

    if method == "ava":
        agent = AVAAgent(model, target_reliability=0.9, max_search_depth=2, max_samples=10)
        result = agent.solve(prompt, budget, max_iterations=15)
        return result.text, budget.tokens_used

    elif method == "ava_no_feedback":
        agent = AVAAgent(model, target_reliability=0.9, use_feedback=False, max_search_depth=2, max_samples=10)
        result = agent.solve(prompt, budget, max_iterations=15)
        return result.text, budget.tokens_used

    elif method == "self_consistency":
        answer, votes = self_consistency(prompt, model, budget, k=5)
        return answer, budget.tokens_used

    elif method == "fixed_depth":
        search = FixedDepthTreeSearch(max_depth=3, branching_factor=3)
        result = search.run(prompt, model, budget)
        return result.text, budget.tokens_used

    elif method == "always_verify":
        # Generate + full verification
        if budget.can_use_tokens(50):
            gen = model.generate(
                prompt, max_tokens=min(200, budget.token_limit - budget.tokens_used)
            )
            tokens_used = gen.metadata.get("tokens", 50)  # type: ignore[arg-type]
            try:
                budget.consume_tokens(tokens_used)
            except RuntimeError:
                # Budget overshoot — cap at limit (fair cost reporting)
                budget.tokens_used = budget.token_limit
            # Apply full cascade
            cascade = create_default_cascade()
            if gen.text and gen.text.strip() and budget.can_call_verifier():
                cascade.verify(prompt, gen.text, budget)
            return gen.text, budget.tokens_used
        return "", 0

    elif method == "confidence_exit":
        answer, conf = confidence_early_exit(
            prompt, model, budget, threshold=0.8, batch_size=2, max_rounds=5
        )
        return answer, budget.tokens_used

    elif method == "difficulty_bin":
        result = difficulty_bin_solve(prompt, model, budget)
        return result.text, budget.tokens_used

    else:
        raise ValueError(f"Unknown method: {method}")


def load_dataset(
    dataset_name: str, data_dir: Optional[str] = None, limit: Optional[int] = None
) -> List[Dict[str, str]]:
    """
    Load dataset examples.

    Returns list of dicts with 'prompt', 'answer', and optionally 'test_code'.
    """
    examples = []

    if dataset_name == "gsm8k":
        from ava.benchmarks.gsm8k import load_gsm8k, prompts_from_examples
        dir_path = data_dir or os.getenv("GSM8K_DATA_DIR", "data/gsm8k")
        raw = load_gsm8k(split="test", data_dir=dir_path)
        for ex in raw:
            examples.append({
                "prompt": f"Question: {ex.question}\nAnswer:",
                "answer": ex.answer,
            })

    elif dataset_name == "hotpotqa":
        from ava.benchmarks.hotpotqa import load_hotpotqa
        dir_path = data_dir or os.getenv("HOTPOTQA_DATA_DIR", "data/hotpotqa")
        raw = load_hotpotqa(split="dev", data_dir=dir_path)
        for ex in raw:
            if ex.context:
                prompt = (
                    f"Context: {ex.context}\n"
                    f"Question: {ex.question}\n"
                    f"Think step by step, then give your final answer on a new line starting with 'Answer:'."
                )
            else:
                prompt = (
                    f"Question: {ex.question}\n"
                    f"Think step by step, then give your final answer on a new line starting with 'Answer:'."
                )
            examples.append({
                "prompt": prompt,
                "answer": ex.answer,
            })

    elif dataset_name == "math":
        from ava.benchmarks.math import load_from_json
        dir_path = data_dir or os.getenv("MATH_DATA_DIR", "data/math")
        # Use Level 3-4 (hard but feasible) if available, else full test set
        hard_path = os.path.join(dir_path, "test_hard.json")
        full_path = os.path.join(dir_path, "test.json")
        raw = load_from_json(hard_path) if os.path.exists(hard_path) else load_from_json(full_path)
        for ex in raw:
            examples.append({
                "prompt": (
                    f"Problem: {ex.problem}\n\n"
                    f"Solve this step by step. Put your final answer on a new line starting with 'Answer:'."
                ),
                "answer": ex.answer,
            })

    elif dataset_name == "humaneval":
        from ava.benchmarks.humaneval import load_humaneval, format_prompt
        dir_path = data_dir or os.getenv("HUMANEVAL_DATA_DIR", "data/humaneval")
        raw = load_humaneval(split="test", data_dir=dir_path)
        for ex in raw:
            examples.append({
                "prompt": format_prompt(ex),
                "answer": ex.solution,
                "test_code": ex.test_code,
            })

    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    if limit:
        examples = examples[:limit]

    return examples


def evaluate_dataset(
    dataset_name: str,
    method: str,
    model: ModelProvider,
    budget_limit: int,
    examples: List[Dict[str, str]],
) -> List[EvalResult]:
    """Evaluate a method on a dataset at a given budget."""
    results = []
    total = len(examples)

    for i, ex in enumerate(examples):
        prompt = ex["prompt"]
        expected = ex["answer"]

        try:
            answer, tokens_used = run_method(method, prompt, model, budget_limit)
        except Exception as e:
            print(f"  Error on example {i+1}: {e}")
            answer = ""
            tokens_used = 0

        # Check correctness
        if dataset_name == "gsm8k":
            correct = check_correct_gsm8k(answer, expected)
        elif dataset_name == "math":
            correct = check_correct_math(answer, expected)
        elif dataset_name == "hotpotqa":
            correct = check_correct_qa(answer, expected)
        elif dataset_name == "humaneval":
            test_code = ex.get("test_code", "")
            correct = check_correct_code(answer, expected, test_code)
        else:
            correct = normalize_answer(answer) == normalize_answer(expected)

        results.append(EvalResult(
            method=method,
            dataset=dataset_name,
            budget_limit=budget_limit,
            correct=correct,
            tokens_used=tokens_used,
            answer=answer[:200],  # Truncate for storage
            expected=expected[:200],
            prompt_length=len(prompt),
        ))

        if (i + 1) % 10 == 0 or (i + 1) == total:
            acc = sum(r.correct for r in results) / len(results)
            avg_tokens = sum(r.tokens_used for r in results) / len(results)
            print(f"  [{method}][{dataset_name}][B={budget_limit}] {i+1}/{total} acc={acc:.1%} avg_tok={avg_tokens:.0f}")

    return results


def compute_metrics(results: List[EvalResult]) -> Dict[str, Any]:
    """Compute aggregate metrics from evaluation results."""
    if not results:
        return {}

    correct = sum(r.correct for r in results)
    total = len(results)
    tokens = [r.tokens_used for r in results]

    return {
        "accuracy": correct / total,
        "n_correct": correct,
        "n_total": total,
        "avg_tokens": np.mean(tokens),
        "median_tokens": np.median(tokens),
        "method": results[0].method,
        "dataset": results[0].dataset,
        "budget_limit": results[0].budget_limit,
    }


def generate_main_results_table(all_metrics: List[Dict[str, Any]]) -> str:
    """Generate LaTeX table from metrics."""
    # Group by dataset and budget
    from collections import defaultdict
    grouped = defaultdict(dict)
    for m in all_metrics:
        key = (m["dataset"], m["budget_limit"])
        grouped[key][m["method"]] = m["accuracy"]

    methods = ["self_consistency", "fixed_depth", "always_verify", "difficulty_bin", "confidence_exit", "ava"]
    method_headers = ["Self-Cons.", "Fixed-Depth", "Always-Ver.", "Diff-Bin", "Conf-Exit", "AVA"]

    lines = [
        r"\begin{table}[htbp]",
        r"  \centering",
        r"  \footnotesize",
        r"  \setlength{\tabcolsep}{3pt}",
        r"  \begin{tabular}{@{}lc" + "c" * len(methods) + r"@{}}",
        r"    \toprule",
        r"    \textbf{Dataset} & \textbf{Budget} & " + " & ".join(f"\\textbf{{{h}}}" for h in method_headers) + r" \\",
        r"    \midrule",
    ]

    datasets = sorted(set(m["dataset"] for m in all_metrics))
    for ds in datasets:
        budgets = sorted(set(m["budget_limit"] for m in all_metrics if m["dataset"] == ds))
        for j, b in enumerate(budgets):
            key = (ds, b)
            accs = []
            for method in methods:
                acc = grouped[key].get(method)
                if acc is not None:
                    accs.append(f"{acc:.1%}")
                else:
                    accs.append("--")

            ds_label = f"\\multirow{{{len(budgets)}}}{{*}}{{{ds.upper()}}}" if j == 0 else ""
            lines.append(f"    {ds_label} & {b} & " + " & ".join(accs) + r" \\")
        lines.append(r"    \midrule")

    # Remove last midrule and add bottomrule
    lines[-1] = r"    \bottomrule"
    lines.extend([
        r"  \end{tabular}",
        r"  \caption{Accuracy at fixed token budgets across all methods.}",
        r"  \label{tab:cross_model_results}",
        r"\end{table}",
    ])

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="AVA Full Evaluation Runner")
    parser.add_argument(
        "--model", type=str, default="ollama",
        help="Model type: 'ollama' or 'azure' (default: ollama)"
    )
    parser.add_argument(
        "--model-name", type=str, default=None,
        help="Model name for Ollama (default: qwen2.5:7b)"
    )
    parser.add_argument(
        "--benchmarks", type=str, default="gsm8k",
        help="Comma-separated benchmarks: gsm8k,hotpotqa,humaneval (default: gsm8k)"
    )
    parser.add_argument(
        "--methods", type=str, default="ava,self_consistency,fixed_depth,always_verify,difficulty_bin",
        help="Comma-separated methods (default: all)"
    )
    parser.add_argument(
        "--budgets", type=str, default="400,600,800,1000",
        help="Comma-separated token budgets (default: 400,600,800,1000)"
    )
    parser.add_argument(
        "--output", type=str, default="results/evaluation",
        help="Output directory (default: results/evaluation)"
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Limit number of examples per dataset (for testing)"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed (default: 42)"
    )
    args = parser.parse_args()

    np.random.seed(args.seed)

    benchmarks = [b.strip() for b in args.benchmarks.split(",")]
    methods = [m.strip() for m in args.methods.split(",")]
    budgets = [int(b.strip()) for b in args.budgets.split(",")]

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"AVA Full Evaluation")
    print(f"=" * 60)
    print(f"Model: {args.model} ({args.model_name or 'default'})")
    print(f"Benchmarks: {benchmarks}")
    print(f"Methods: {methods}")
    print(f"Budgets: {budgets}")
    print(f"Limit: {args.limit or 'full'}")
    print(f"Output: {output_dir}")
    print(f"=" * 60)

    # Create model
    print(f"\nInitializing model...")
    model = create_model(args.model, args.model_name)
    print(f"Model ready.")

    all_results: List[EvalResult] = []
    all_metrics: List[Dict[str, Any]] = []

    for dataset in benchmarks:
        print(f"\n{'=' * 40}")
        print(f"Dataset: {dataset.upper()}")
        print(f"{'=' * 40}")

        try:
            examples = load_dataset(dataset, limit=args.limit)
            print(f"Loaded {len(examples)} examples")
        except (ValueError, FileNotFoundError) as e:
            print(f"Skipping {dataset}: {e}")
            continue

        for budget_limit in budgets:
            for method in methods:
                print(f"\n--- {method} | budget={budget_limit} ---")
                start_time = time.time()

                results = evaluate_dataset(
                    dataset, method, model, budget_limit, examples
                )
                all_results.extend(results)

                metrics = compute_metrics(results)
                all_metrics.append(metrics)

                elapsed = time.time() - start_time
                print(f"  Done: acc={metrics['accuracy']:.1%} avg_tok={metrics['avg_tokens']:.0f} time={elapsed:.1f}s")

    # Save results
    results_data = [asdict(r) for r in all_results]
    results_path = output_dir / "eval_results.json"
    with open(results_path, "w") as f:
        json.dump(results_data, f, indent=2, default=str)
    print(f"\nResults saved to {results_path}")

    # Save metrics
    metrics_path = output_dir / "eval_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(all_metrics, f, indent=2, default=str)
    print(f"Metrics saved to {metrics_path}")

    # Generate LaTeX table
    if all_metrics:
        latex_table = generate_main_results_table(all_metrics)
        latex_path = output_dir / "results_table.tex"
        with open(latex_path, "w") as f:
            f.write(latex_table)
        print(f"LaTeX table saved to {latex_path}")

    # Print summary
    print(f"\n{'=' * 60}")
    print("SUMMARY")
    print(f"{'=' * 60}")
    for m in all_metrics:
        print(f"  {m['dataset']:12s} | {m['method']:20s} | B={m['budget_limit']:5d} | acc={m['accuracy']:.1%} | tok={m['avg_tokens']:.0f}")


if __name__ == "__main__":
    main()
