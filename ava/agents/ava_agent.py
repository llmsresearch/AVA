from __future__ import annotations

from typing import Optional

from ava.controllers.ava_controller import AVAController, ControllerState
from ava.core.interfaces import Budget, Generation, ModelProvider
from ava.search.adaptive import AdaptiveTreeSearch
from ava.uncertainty.estimators import (
    UncertaintyEstimate,
    aggregate_uncertainty,
    ConsistencyEstimator,
    TokenEntropyEstimator,
    TrajectoryFeatureEstimator,
)
from ava.verification.cascade import VerifierCascade, create_default_cascade


class AVAAgent:
    """
    End-to-end Anytime Verified Agent.

    Integrates:
    - AVA Controller for adaptive compute allocation
    - Adaptive search with value-of-information
    - Verifier cascade with early exits
    - Uncertainty estimation and calibration
    """

    def __init__(
        self,
        model: ModelProvider,
        *,
        controller: Optional[AVAController] = None,
        verifier_cascade: Optional[VerifierCascade] = None,
        target_reliability: float = 0.9,
    ) -> None:
        self.model = model
        self.controller = controller or AVAController(target_reliability=target_reliability)
        self.verifier_cascade = verifier_cascade or create_default_cascade()
        self.target_reliability = target_reliability

        # Uncertainty estimators
        self.token_entropy_estimator = TokenEntropyEstimator()
        self.consistency_estimator = ConsistencyEstimator()
        self.trajectory_estimator = TrajectoryFeatureEstimator()

    def solve(
        self,
        prompt: str,
        budget: Budget,
        *,
        max_iterations: int = 10,
    ) -> Generation:
        """
        Solve a task with adaptive compute allocation.

        Args:
            prompt: Input prompt
            budget: Compute budget
            max_iterations: Maximum controller decision iterations

        Returns:
            Best generation found
        """
        best_generation: Optional[Generation] = None
        best_confidence = 0.0
        depth_reached = 0
        nodes_expanded = 0

        # Step 1: Bootstrap - generate initial answer if we don't have one
        if best_generation is None and budget.can_use_tokens(50):
            try:
                remaining_budget = budget.token_limit - budget.tokens_used
                if remaining_budget >= 50:
                    initial_gen = self.model.generate(
                        prompt, max_tokens=min(200, remaining_budget)
                    )
                    if initial_gen.text and initial_gen.text.strip():
                        tokens_used = initial_gen.metadata.get("tokens", 50)  # type: ignore[arg-type]
                        try:
                            budget.consume_tokens(tokens_used)
                            best_generation = initial_gen
                        except RuntimeError:
                            # Budget exceeded, but use the generation anyway
                            best_generation = initial_gen
            except Exception:
                pass  # Will retry in iterations

        # Step 2: Iterative refinement loop
        for iteration in range(max_iterations):
            if not budget.can_use_tokens(50):
                break

            # Calculate budget remaining correctly
            budget_remaining = (budget.token_limit - budget.tokens_used) / max(1, budget.token_limit)

            # Estimate current uncertainty from best generation
            uncertainty_est = self._estimate_uncertainty(
                prompt=prompt,
                generation=best_generation,
                depth_reached=depth_reached,
                nodes_expanded=nodes_expanded,
                budget_remaining=budget_remaining,
            )

            # Check if we should stop
            state = {
                "uncertainty": uncertainty_est,
                "budget_remaining": budget_remaining,
                "depth_reached": depth_reached,
                "nodes_expanded": nodes_expanded,
            }

            if self.controller.should_stop(state):
                break

            # Get controller decision
            decision = self.controller.decide(state)

            # Execute decision: adaptive search
            search_depth = decision.get("search_depth", 0)
            if search_depth > 0:
                search = AdaptiveTreeSearch(
                    max_depth=search_depth,
                    initial_branching=decision.get("search_breadth", 3),
                )

                def get_unc(node):
                    est = self._estimate_uncertainty(
                        prompt=node.prompt,
                        generation=node.generation,
                        depth_reached=node.depth,
                        nodes_expanded=nodes_expanded + 1,
                        budget_remaining=(budget.token_limit - budget.tokens_used) / max(1, budget.token_limit),
                    )
                    return 1.0 - est.confidence  # Convert confidence to uncertainty

                result = search.run(
                    prompt,
                    self.model,
                    budget,
                    get_uncertainty=get_unc,
                )

                # Update best generation if we got a valid result with text
                if result and result.text and result.text.strip():
                    best_generation = result
                    depth_reached = max(depth_reached, search_depth)
                    nodes_expanded += decision.get("search_breadth", 1)

            # Execute decision: verification if we have a generation
            verifier_level = decision.get("verifier_level", 0)
            if verifier_level > 0 and best_generation and best_generation.text and best_generation.text.strip():
                is_valid, confidence, level = self.verifier_cascade.verify(
                    prompt,
                    best_generation.text,
                    budget,
                    early_exit_threshold=self.target_reliability,
                )
                if confidence > best_confidence:
                    best_confidence = confidence

            # Early exit if target reliability reached
            if best_confidence >= self.target_reliability:
                break

        # Step 3: Final fallback - ensure we always return a generation with text
        if best_generation is None or not best_generation.text or not best_generation.text.strip():
            remaining = budget.token_limit - budget.tokens_used
            if remaining >= 50:
                try:
                    final_gen = self.model.generate(
                        prompt, max_tokens=min(200, remaining)
                    )
                    if final_gen.text and final_gen.text.strip():
                        best_generation = final_gen
                        # Try to consume tokens, but don't fail if budget exceeded
                        try:
                            tokens_used = final_gen.metadata.get("tokens", 50)  # type: ignore[arg-type]
                            budget.consume_tokens(tokens_used)
                        except RuntimeError:
                            pass  # Budget exceeded, but we have the generation
                except Exception:
                    pass  # Generation failed

        # Ultimate fallback: if still no valid generation, generate minimal one
        if best_generation is None or not best_generation.text or not best_generation.text.strip():
            remaining = budget.token_limit - budget.tokens_used
            if remaining > 0:
                try:
                    fallback_gen = self.model.generate(prompt, max_tokens=min(100, remaining))
                    if fallback_gen.text and fallback_gen.text.strip():
                        best_generation = fallback_gen
                except Exception:
                    pass

        # Final safety: ensure we never return None
        if best_generation is None:
            best_generation = Generation(text="", logprobs=None, metadata={"tokens": 0})

        return best_generation

    def _estimate_uncertainty(
        self,
        prompt: str,
        generation: Optional[Generation],
        depth_reached: int,
        nodes_expanded: int,
        budget_remaining: float,
    ) -> UncertaintyEstimate:
        """Estimate uncertainty from multiple signals."""
        entropy = None
        consistency = None
        verifier_score = None

        if generation and generation.logprobs:
            entropy = self.token_entropy_estimator.estimate(generation.logprobs)

        trajectory_unc = self.trajectory_estimator.estimate(
            depth_reached, nodes_expanded, budget_remaining
        )

        return aggregate_uncertainty(
            entropy=entropy,
            consistency=consistency,
            verifier=verifier_score,
            trajectory=trajectory_unc,
        )

