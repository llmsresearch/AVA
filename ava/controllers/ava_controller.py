from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

from ava.core.interfaces import Budget, Controller
from ava.uncertainty.estimators import UncertaintyEstimate


@dataclass
class ControllerState:
    """State passed to controller for decision-making."""

    uncertainty: UncertaintyEstimate
    budget_remaining: float  # Fraction 0-1
    depth_reached: int
    nodes_expanded: int
    task_complexity: Optional[float] = None  # Estimated from input features


class AVAController(Controller):
    """
    Anytime Verified Agent Controller.

    Adaptively allocates compute resources (search depth/breadth, verification intensity,
    tool calls) based on uncertainty estimates and budget constraints.
    """

    def __init__(
        self,
        *,
        target_reliability: float = 0.9,
        cost_per_token: float = 0.001,
        cost_per_tool: float = 0.1,
        cost_per_verify: float = 0.05,
    ) -> None:
        self.target_reliability = target_reliability
        self.cost_per_token = cost_per_token
        self.cost_per_tool = cost_per_tool
        self.cost_per_verify = cost_per_verify

    def decide(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Decide next action parameters based on state.

        Args:
            state: Dict with keys:
                - uncertainty: UncertaintyEstimate
                - budget_remaining: float
                - depth_reached: int
                - nodes_expanded: int
                - task_complexity: Optional[float]

        Returns:
            Dict with action parameters:
                - samples: int (for self-consistency)
                - search_depth: int
                - search_breadth: int
                - verifier_level: int (0 = skip, 1-N = cascade level)
                - should_call_tool: bool
        """
        uncertainty = state.get("uncertainty")
        budget_remaining = state.get("budget_remaining", 1.0)
        depth_reached = state.get("depth_reached", 0)
        task_complexity = state.get("task_complexity", 0.5)

        if not isinstance(uncertainty, UncertaintyEstimate):
            # Fallback if uncertainty not provided
            confidence = 0.5
        else:
            confidence = uncertainty.confidence

        # Compute confidence gap
        confidence_gap = self.target_reliability - confidence

        # Decision logic: allocate resources based on gap and budget
        decisions: Dict[str, Any] = {}

        # Samples for self-consistency
        if confidence_gap > 0.3 and budget_remaining > 0.3:
            decisions["samples"] = 10
        elif confidence_gap > 0.1:
            decisions["samples"] = 5
        else:
            decisions["samples"] = 1

        # Search parameters
        if confidence_gap > 0.4 and budget_remaining > 0.5:
            decisions["search_depth"] = min(5, depth_reached + 3)
            decisions["search_breadth"] = 4
        elif confidence_gap > 0.2:
            decisions["search_depth"] = min(3, depth_reached + 2)
            decisions["search_breadth"] = 3
        else:
            decisions["search_depth"] = 1
            decisions["search_breadth"] = 2

        # Verification level (0 = skip, 1 = cheap, 2 = medium, 3 = expensive)
        if confidence_gap > 0.5 or confidence < 0.3:
            decisions["verifier_level"] = 3  # Use full cascade
        elif confidence_gap > 0.2:
            decisions["verifier_level"] = 2  # Use medium verifier
        elif confidence_gap > 0.05:
            decisions["verifier_level"] = 1  # Use cheap verifier
        else:
            decisions["verifier_level"] = 0  # Skip verification

        # Tool calls (only if uncertainty is high and budget allows)
        decisions["should_call_tool"] = (
            confidence < 0.4
            and budget_remaining > 0.4
            and task_complexity > 0.6
        )

        # Early stopping: if we've reached target reliability, minimize further compute
        if confidence >= self.target_reliability:
            decisions["samples"] = 1
            decisions["search_depth"] = 0
            decisions["search_breadth"] = 1
            decisions["verifier_level"] = 0
            decisions["should_call_tool"] = False

        return decisions

    def should_stop(self, state: Dict[str, Any]) -> bool:
        """
        Determine if we should stop and return current best answer.

        Args:
            state: Controller state dict

        Returns:
            True if should stop early
        """
        uncertainty = state.get("uncertainty")
        if not isinstance(uncertainty, UncertaintyEstimate):
            return False

        confidence = uncertainty.confidence
        budget_remaining = state.get("budget_remaining", 1.0)

        # Stop if we've reached target reliability
        if confidence >= self.target_reliability:
            return True

        # Stop if budget is exhausted
        if budget_remaining < 0.05:
            return True

        # Stop if confidence is high enough and budget is low
        if confidence >= 0.85 and budget_remaining < 0.2:
            return True

        return False


