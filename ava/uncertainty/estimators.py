from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np


@dataclass
class UncertaintyEstimate:
    """Uncertainty estimate with optional calibration."""

    confidence: float  # 0-1, higher = more confident
    entropy: Optional[float] = None  # Token-level entropy if available
    consistency_score: Optional[float] = None  # From self-consistency votes
    verifier_score: Optional[float] = None  # From verifier
    calibrated_confidence: Optional[float] = None  # After calibration

    @property
    def uncertainty(self) -> float:
        """Return uncertainty as 1 - confidence."""
        return 1.0 - self.confidence


class TokenEntropyEstimator:
    """Estimate uncertainty from token-level log probabilities."""

    def estimate(self, logprobs: List[float]) -> float:
        """
        Compute entropy from log probabilities.

        Args:
            logprobs: List of log probabilities per token

        Returns:
            Entropy in nats (higher = more uncertain)
        """
        if not logprobs:
            return 1.0

        probs = np.exp(logprobs)
        probs = probs / np.sum(probs)  # Normalize
        entropy = -np.sum(probs * np.log(probs + 1e-10))
        return float(entropy)


class ConsistencyEstimator:
    """Estimate uncertainty from self-consistency vote distribution."""

    def estimate(self, vote_counts: dict[str, int]) -> float:
        """
        Compute uncertainty from vote distribution (entropy of votes).

        Args:
            vote_counts: Dict mapping answer -> count

        Returns:
            Normalized entropy [0,1], where 1 = maximum uncertainty
        """
        if not vote_counts:
            return 1.0

        total = sum(vote_counts.values())
        probs = [count / total for count in vote_counts.values()]
        entropy = -np.sum(p * np.log(p + 1e-10) for p in probs)
        max_entropy = np.log(len(vote_counts)) if len(vote_counts) > 1 else 1.0
        normalized = entropy / max_entropy if max_entropy > 0 else 0.0
        return min(1.0, float(normalized))


class TrajectoryFeatureEstimator:
    """Estimate uncertainty from trajectory features (search depth, expansion patterns, etc.)."""

    def estimate(
        self,
        depth_reached: int,
        nodes_expanded: int,
        budget_remaining: float,
        *,
        max_depth: int = 10,
    ) -> float:
        """
        Heuristic uncertainty: higher if search was shallow or budget exhausted early.

        Args:
            depth_reached: Maximum depth reached in search
            nodes_expanded: Number of nodes expanded
            budget_remaining: Fraction of budget remaining (0-1)
            max_depth: Maximum possible depth

        Returns:
            Uncertainty score [0,1]
        """
        depth_uncertainty = 1.0 - (depth_reached / max_depth)
        budget_uncertainty = 1.0 - budget_remaining
        # Combine: if we didn't explore deeply and budget is low, high uncertainty
        combined = 0.6 * depth_uncertainty + 0.4 * budget_uncertainty
        return min(1.0, max(0.0, combined))


def aggregate_uncertainty(
    entropy: Optional[float] = None,
    consistency: Optional[float] = None,
    verifier: Optional[float] = None,
    trajectory: Optional[float] = None,
    *,
    weights: Optional[dict[str, float]] = None,
) -> UncertaintyEstimate:
    """
    Aggregate multiple uncertainty signals into a single estimate.

    Args:
        entropy: Token entropy (higher = more uncertain)
        consistency: Consistency entropy (higher = more uncertain)
        verifier: Verifier score inverted (1 - score if score is confidence)
        trajectory: Trajectory-based uncertainty
        weights: Optional weights for each signal

    Returns:
        UncertaintyEstimate with aggregated confidence
    """
    if weights is None:
        weights = {"entropy": 0.3, "consistency": 0.4, "verifier": 0.2, "trajectory": 0.1}

    signals = []
    weights_list = []

    if entropy is not None:
        # Convert entropy to confidence (normalize and invert)
        conf = 1.0 - min(1.0, entropy / 10.0)  # Rough normalization
        signals.append(conf)
        weights_list.append(weights.get("entropy", 0.0))

    if consistency is not None:
        conf = 1.0 - consistency  # consistency is already [0,1] uncertainty
        signals.append(conf)
        weights_list.append(weights.get("consistency", 0.0))

    if verifier is not None:
        # Assume verifier gives confidence directly
        signals.append(verifier)
        weights_list.append(weights.get("verifier", 0.0))

    if trajectory is not None:
        conf = 1.0 - trajectory  # trajectory is [0,1] uncertainty
        signals.append(conf)
        weights_list.append(weights.get("trajectory", 0.0))

    if not signals:
        return UncertaintyEstimate(confidence=0.5)

    # Weighted average
    total_weight = sum(weights_list)
    if total_weight == 0:
        confidence = np.mean(signals)
    else:
        confidence = sum(s * w for s, w in zip(signals, weights_list)) / total_weight

    return UncertaintyEstimate(
        confidence=float(confidence),
        entropy=entropy,
        consistency_score=consistency,
        verifier_score=verifier,
    )

