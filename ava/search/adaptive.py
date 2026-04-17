from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque, List, Optional, Tuple

from ava.core.interfaces import Budget, Generation, ModelProvider
from ava.uncertainty.estimators import UncertaintyEstimate


@dataclass
class SearchNode:
    """Node in adaptive search tree."""

    prompt: str
    generation: Optional[Generation]
    depth: int
    uncertainty: float
    value_estimate: float  # Estimated value-of-information


class AdaptiveTreeSearch:
    """
    Adaptive tree search that expands based on value-of-information.

    Expands nodes with highest expected marginal improvement per unit cost.
    """

    def __init__(
        self,
        max_depth: int = 5,
        initial_branching: int = 2,
        uncertainty_threshold: float = 0.3,
    ) -> None:
        self.max_depth = max_depth
        self.initial_branching = initial_branching
        self.uncertainty_threshold = uncertainty_threshold

    def run(
        self,
        prompt: str,
        model: ModelProvider,
        budget: Budget,
        *,
        get_uncertainty,
    ) -> Generation:
        """
        Run adaptive search.

        Args:
            prompt: Initial prompt
            model: Model provider
            budget: Compute budget
            get_uncertainty: Function to estimate uncertainty from node state

        Returns:
            Best generation found
        """
        # Priority queue: prioritize by value-of-information
        # For now, simple implementation expands uncertain nodes first
        queue: Deque[SearchNode] = deque(
            [
                SearchNode(
                    prompt=prompt,
                    generation=None,
                    depth=0,
                    uncertainty=1.0,
                    value_estimate=1.0,
                )
            ]
        )
        best_leaf: Optional[Generation] = None
        best_confidence = 0.0
        iterations = 0
        max_iterations = 100  # Prevent infinite loops - max 100 search iterations

        while queue and budget.can_use_tokens(50) and iterations < max_iterations:
            iterations += 1
            node = queue.popleft()

            if node.depth >= self.max_depth:
                # Leaf: ALWAYS generate final answer at leaf nodes
                if budget.can_use_tokens(50):
                    try:
                        gen = model.generate(
                            node.prompt, 
                            max_tokens=min(100, budget.token_limit - budget.tokens_used)
                        )
                        tokens_used = gen.metadata.get("tokens", 50)  # type: ignore[arg-type]
                        budget.consume_tokens(tokens_used)
                        # Only use if it has actual text
                        if gen.text and gen.text.strip():
                            node.generation = gen
                            # Estimate confidence
                            unc = get_uncertainty(node) if callable(get_uncertainty) else 0.5
                            conf = 1.0 - unc
                            if conf > best_confidence:
                                best_leaf = gen
                                best_confidence = conf
                    except RuntimeError:
                        # Budget exhausted, keep what we have
                        pass
                continue

            # Expand: decide branching factor based on uncertainty
            branching = self._adaptive_branching(node.uncertainty)

            children: List[SearchNode] = []
            for _ in range(branching):
                if not budget.can_use_tokens(50):
                    break

                child_gen = model.generate(node.prompt, max_tokens=50)
                # Consume tokens, but handle budget exceeded gracefully
                tokens_used = child_gen.metadata.get("tokens", 50)  # type: ignore[arg-type]
                try:
                    budget.consume_tokens(tokens_used)
                except RuntimeError:
                    # Budget exceeded, break out of expansion loop
                    break

                # Only create child node if generation has actual text
                if child_gen.text and child_gen.text.strip():
                    new_prompt = f"{node.prompt}\n{child_gen.text}"
                    child_uncertainty = get_uncertainty(
                        SearchNode(
                            prompt=new_prompt,
                            generation=child_gen,
                            depth=node.depth + 1,
                            uncertainty=node.uncertainty * 0.9,  # Decrease uncertainty as we go deeper
                            value_estimate=0.0,
                        )
                    ) if callable(get_uncertainty) else node.uncertainty * 0.9

                    child = SearchNode(
                        prompt=new_prompt,
                        generation=child_gen,
                        depth=node.depth + 1,
                        uncertainty=child_uncertainty,
                        value_estimate=self._value_of_information(child_uncertainty, node.depth),
                    )
                    children.append(child)

            # Add children to queue, sorted by value-of-information
            # Limit queue size to prevent memory issues and infinite loops
            sorted_children = sorted(children, key=lambda n: n.value_estimate, reverse=True)
            max_queue_size = 50  # Limit queue to prevent exponential growth
            if len(queue) + len(sorted_children) > max_queue_size:
                # Keep only top-N nodes by value
                remaining_slots = max_queue_size - len(queue)
                queue.extend(sorted_children[:remaining_slots])
            else:
                queue.extend(sorted_children)
            
            # Early stop if we've found a good enough solution
            if best_confidence > 0.9:
                break

        # Return best leaf - should always have one from leaf node generation
        # Final safety: if somehow no leaf found or empty, generate one
        if best_leaf is None or not best_leaf.text or not best_leaf.text.strip():
            remaining_budget = budget.token_limit - budget.tokens_used
            if remaining_budget >= 50:
                try:
                    final_gen = model.generate(
                        prompt, max_tokens=min(200, remaining_budget)
                    )
                    # Always use the generation if it has text
                    if final_gen.text and final_gen.text.strip():
                        best_leaf = final_gen
                        # Try to consume tokens, but don't fail if budget exceeded
                        try:
                            tokens_used = final_gen.metadata.get("tokens", 50)  # type: ignore[arg-type]
                            budget.consume_tokens(tokens_used)
                        except RuntimeError:
                            pass  # Budget exceeded, but we have the generation
                except Exception:
                    pass  # Generation failed, will use fallback below

        # Ultimate fallback: if still empty, generate minimal response
        if best_leaf is None or not best_leaf.text or not best_leaf.text.strip():
            remaining = budget.token_limit - budget.tokens_used
            if remaining > 0:
                try:
                    fallback_gen = model.generate(prompt, max_tokens=min(100, remaining))
                    if fallback_gen.text and fallback_gen.text.strip():
                        best_leaf = fallback_gen
                except Exception:
                    pass
            
            # If still empty, return empty (better than None)
            if best_leaf is None or not best_leaf.text:
                best_leaf = Generation(text="", logprobs=None, metadata={"tokens": 0})

        return best_leaf

    def _adaptive_branching(self, uncertainty: float) -> int:
        """Decide branching factor based on uncertainty."""
        # More uncertain -> more expansion
        if uncertainty > 0.7:
            return 4
        elif uncertainty > 0.4:
            return 3
        else:
            return 2

    def _value_of_information(self, uncertainty: float, depth: int) -> float:
        """
        Estimate value-of-information for expanding a node.

        Higher uncertainty and shallower depth -> higher value.
        """
        depth_discount = 1.0 / (1.0 + depth * 0.2)
        return uncertainty * depth_discount


