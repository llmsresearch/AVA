from __future__ import annotations

from collections import deque
from typing import List, Optional, Tuple

from ava.core.interfaces import Budget, Generation, ModelProvider, SearchStrategy


class FixedDepthTreeSearch(SearchStrategy):
    """
    Fixed-depth tree search baseline (Tree-of-Thoughts style).

    Expands a fixed number of levels with fixed branching factor per node.
    Does not adapt depth or breadth based on uncertainty or budget.
    """

    def __init__(
        self,
        max_depth: int = 3,
        branching_factor: int = 3,
        temperature: float = 0.8,
    ) -> None:
        self.max_depth = max_depth
        self.branching_factor = branching_factor
        self.temperature = temperature

    def run(self, prompt: str, model: ModelProvider, budget: Budget) -> Generation:
        """
        Run fixed-depth tree search.

        Returns the best leaf node according to a simple scoring heuristic
        (simple heuristic: first generation that fits; could use verifier score in production).
        """
        # BFS up to max_depth
        queue: deque[Tuple[str, int, Optional[Generation]]] = deque([(prompt, 0, None)])
        best_leaf: Optional[Generation] = None

        while queue and budget.can_use_tokens(100):  # Rough estimate per node
            current_prompt, depth, parent_gen = queue.popleft()

            if depth >= self.max_depth:
                # Leaf node - generate final answer from this node
                if budget.can_use_tokens(50):
                    # Always generate a final answer at leaf nodes
                    try:
                        g = model.generate(
                            current_prompt, max_tokens=min(100, budget.token_limit - budget.tokens_used)
                        )
                        # Only use if generation has actual text
                        if g.text and g.text.strip():
                            tokens_used = g.metadata.get("tokens", 50)  # type: ignore[arg-type]
                            try:
                                budget.consume_tokens(tokens_used)
                                # Only update best_leaf if this is better or we don't have one
                                if best_leaf is None or self._score(g.text) > self._score(best_leaf.text if best_leaf and best_leaf.text else ""):
                                    best_leaf = g
                            except RuntimeError:
                                # Budget exceeded, but use the generation anyway
                                if best_leaf is None or self._score(g.text) > self._score(best_leaf.text if best_leaf and best_leaf.text else ""):
                                    best_leaf = g
                    except (RuntimeError, Exception):
                        # Generation failed, continue to next node
                        pass
                continue

            # Expand: generate k children
            children: List[Generation] = []
            for _ in range(self.branching_factor):
                # Check budget before generation (use estimate)
                estimated_tokens = 100  # Conservative estimate
                if not budget.can_use_tokens(estimated_tokens):
                    break
                child_gen = model.generate(
                    current_prompt, max_tokens=50
                )
                children.append(child_gen)
                # Use actual token count from API response
                tokens_used = child_gen.metadata.get("tokens", estimated_tokens)  # type: ignore[arg-type]
                try:
                    budget.consume_tokens(tokens_used)
                except RuntimeError:
                    # Budget exceeded, stop expanding
                    break

            # Enqueue children with updated prompts
            for child in children:
                # Only enqueue if child has actual text
                if child.text and child.text.strip():
                    # Simple continuation: append child text to prompt
                    new_prompt = f"{current_prompt}\n{child.text}"
                    queue.append((new_prompt, depth + 1, child))

        # Return best leaf - should always have one from leaf node generation
        # Final safety check: if somehow we have no leaf or empty text, generate one
        if best_leaf is None or not best_leaf.text or not best_leaf.text.strip():
            # Try to generate a final answer
            if budget.can_use_tokens(50):
                try:
                    final_gen = model.generate(
                        prompt, max_tokens=min(200, budget.token_limit - budget.tokens_used)
                    )
                    if final_gen.text and final_gen.text.strip():
                        tokens_used = final_gen.metadata.get("tokens", 50)  # type: ignore[arg-type]
                        try:
                            budget.consume_tokens(tokens_used)
                            best_leaf = final_gen
                        except RuntimeError:
                            # Budget exceeded, but use the generation anyway
                            best_leaf = final_gen
                except (RuntimeError, Exception) as e:
                    # If generation fails, ensure we return something
                    if best_leaf is None:
                        best_leaf = Generation(text="", logprobs=None, metadata={"tokens": 0})

        # Ultimate fallback: if still empty, return minimal response
        if best_leaf is None or not best_leaf.text or not best_leaf.text.strip():
            best_leaf = Generation(text="", logprobs=None, metadata={"tokens": 0})

        return best_leaf

    def _score(self, text: str) -> float:
        """Simple scoring heuristic based on output quality."""
        # Prefer longer, more structured text as proxy for quality
        return len(text.split())


def fixed_depth_search(
    prompt: str,
    model: ModelProvider,
    budget: Budget,
    *,
    max_depth: int = 3,
    branching_factor: int = 3,
) -> Generation:
    """
    Convenience function for fixed-depth tree search.

    Args:
        prompt: Input prompt
        model: Model provider
        budget: Compute budget
        max_depth: Maximum tree depth
        branching_factor: Children per node

    Returns:
        Best generation found
    """
    strategy = FixedDepthTreeSearch(
        max_depth=max_depth, branching_factor=branching_factor
    )
    return strategy.run(prompt, model, budget)

