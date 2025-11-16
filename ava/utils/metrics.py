from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
from sklearn.calibration import calibration_curve


@dataclass
class ReliabilityMetrics:
    """Core metrics for reliability-under-budget evaluation."""

    accuracy: float  # 0-1, exact match or task success rate
    tokens_used: int
    tool_calls_used: int
    verify_calls_used: int
    total_cost: float  # Can be weighted sum of tokens/tools/verifies

    # Calibration metrics (optional)
    expected_calibration_error: Optional[float] = None
    brier_score: Optional[float] = None

    def to_dict(self) -> Dict[str, float]:
        d = {
            "accuracy": self.accuracy,
            "tokens_used": self.tokens_used,
            "tool_calls_used": self.tool_calls_used,
            "verify_calls_used": self.verify_calls_used,
            "total_cost": self.total_cost,
        }
        if self.expected_calibration_error is not None:
            d["ece"] = self.expected_calibration_error
        if self.brier_score is not None:
            d["brier"] = self.brier_score
        return d


def compute_reliability_at_budget(
    results: List[Dict[str, any]],
    *,
    budget_tokens: Optional[int] = None,
    budget_cost: Optional[float] = None,
) -> float:
    """
    Compute accuracy/reliability for results that stayed within budget.

    Args:
        results: List of dicts with keys: 'correct' (bool), 'tokens_used', 'total_cost'
        budget_tokens: Optional token limit
        budget_cost: Optional total cost limit (weighted)

    Returns:
        Accuracy (0-1) for results within budget
    """
    filtered = results
    if budget_tokens is not None:
        filtered = [r for r in filtered if r.get("tokens_used", 0) <= budget_tokens]
    if budget_cost is not None:
        filtered = [r for r in filtered if r.get("total_cost", 0) <= budget_cost]

    if not filtered:
        return 0.0
    correct = sum(1 for r in filtered if r.get("correct", False))
    return correct / len(filtered)


def compute_cost_at_target_reliability(
    results: List[Dict[str, any]],
    target_reliability: float,
    *,
    cost_key: str = "total_cost",
    reliability_key: str = "correct",
) -> Optional[float]:
    """
    Find minimum cost needed to achieve target reliability threshold.

    Args:
        results: List of result dicts
        target_reliability: Target accuracy (0-1)
        cost_key: Key in result dict for cost
        reliability_key: Key in result dict for correctness (bool or 0/1)

    Returns:
        Minimum cost, or None if target not achievable
    """
    sorted_results = sorted(results, key=lambda r: r.get(cost_key, float("inf")))
    correct_count = 0
    total_count = 0

    for r in sorted_results:
        total_count += 1
        correct = r.get(reliability_key, False)
        if isinstance(correct, bool):
            correct_count += int(correct)
        else:
            correct_count += correct
        current_reliability = correct_count / total_count if total_count > 0 else 0.0
        if current_reliability >= target_reliability:
            return r.get(cost_key)

    return None


def compute_expected_calibration_error(
    predicted_probs: List[float], actual_labels: List[bool], n_bins: int = 10
) -> float:
    """
    Compute Expected Calibration Error (ECE).

    Args:
        predicted_probs: List of predicted probabilities [0,1]
        actual_labels: List of true labels (bool)
        n_bins: Number of bins for calibration curve

    Returns:
        ECE score (lower is better)
    """
    if len(predicted_probs) == 0:
        return 0.0

    y_true = np.array([float(x) for x in actual_labels])
    y_pred = np.array(predicted_probs)

    fraction_of_positives, mean_predicted_value = calibration_curve(
        y_true, y_pred, n_bins=n_bins, strategy="uniform"
    )

    bin_counts = np.histogram(y_pred, bins=n_bins)[0]
    weights = bin_counts / len(y_pred)

    ece = np.sum(weights * np.abs(fraction_of_positives - mean_predicted_value))
    return float(ece)


def compute_brier_score(predicted_probs: List[float], actual_labels: List[bool]) -> float:
    """
    Compute Brier score (mean squared error of probabilities).

    Args:
        predicted_probs: List of predicted probabilities [0,1]
        actual_labels: List of true labels (bool)

    Returns:
        Brier score (lower is better)
    """
    if len(predicted_probs) == 0:
        return 0.0

    y_true = np.array([float(x) for x in actual_labels])
    y_pred = np.array(predicted_probs)

    return float(np.mean((y_pred - y_true) ** 2))


def compute_expected_utility(
    results: List[Dict[str, any]],
    *,
    value_per_success: float = 1.0,
    cost_per_token: float = 0.001,
    cost_per_tool: float = 0.1,
    cost_per_verify: float = 0.05,
) -> float:
    """
    Compute expected utility: task value minus compute cost.

    Args:
        results: List of result dicts with keys: 'correct', 'tokens_used', etc.
        value_per_success: Value of a successful task
        cost_per_token: Cost per token used
        cost_per_tool: Cost per tool call
        cost_per_verify: Cost per verifier call

    Returns:
        Expected utility (higher is better)
    """
    if not results:
        return 0.0

    total_value = sum(
        value_per_success if r.get("correct", False) else 0.0 for r in results
    )
    total_cost = sum(
        r.get("tokens_used", 0) * cost_per_token
        + r.get("tool_calls_used", 0) * cost_per_tool
        + r.get("verify_calls_used", 0) * cost_per_verify
        for r in results
    )

    return (total_value - total_cost) / len(results)


