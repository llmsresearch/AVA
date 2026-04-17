from __future__ import annotations

from collections import Counter
from typing import Optional, Tuple

from ava.baselines.self_consistency import _extract_answer_signature
from ava.core.interfaces import Budget, Generation, ModelProvider
from ava.baselines.fixed_depth_search import FixedDepthTreeSearch
from ava.verification.cascade import VerifierCascade, create_default_cascade


# Difficulty tier thresholds (prompt character count)
EASY_THRESHOLD = 200
HARD_THRESHOLD = 500

# Fixed compute profiles per tier
TIER_PROFILES = {
    "easy": {"samples": 1, "search_depth": 1, "branching": 2, "verify_level": 0},
    "medium": {"samples": 5, "search_depth": 2, "branching": 3, "verify_level": 1},
    "hard": {"samples": 10, "search_depth": 3, "branching": 3, "verify_level": 3},
}


def classify_difficulty(prompt: str) -> str:
    """
    Classify a problem into Easy/Medium/Hard based on prompt length.

    Uses character count as a proxy for problem complexity:
    - Short prompts (< 200 chars): likely simple, single-step problems
    - Medium prompts (200-500 chars): moderate complexity
    - Long prompts (> 500 chars): multi-step or context-heavy problems

    Args:
        prompt: Input prompt text

    Returns:
        Difficulty tier: "easy", "medium", or "hard"
    """
    length = len(prompt.strip())
    if length < EASY_THRESHOLD:
        return "easy"
    elif length < HARD_THRESHOLD:
        return "medium"
    else:
        return "hard"


def difficulty_bin_solve(
    prompt: str,
    model: ModelProvider,
    budget: Budget,
    *,
    cascade: Optional[VerifierCascade] = None,
) -> Generation:
    """
    Solve a task using fixed compute allocation per difficulty tier.

    Classifies the input into Easy/Medium/Hard and applies a pre-defined
    compute profile. This baseline tests whether a simple heuristic
    allocation can match adaptive allocation.

    Args:
        prompt: Input prompt
        model: Model provider
        budget: Compute budget
        cascade: Optional verifier cascade for verification tiers

    Returns:
        Best generation found
    """
    tier = classify_difficulty(prompt)
    profile = TIER_PROFILES[tier]

    n_samples = profile["samples"]
    search_depth = profile["search_depth"]
    branching = profile["branching"]
    verify_level = profile["verify_level"]

    best_generation: Optional[Generation] = None
    best_confidence = 0.0

    # Step 1: Self-consistency sampling (vote on extracted answer signatures)
    if n_samples > 1:
        sig_votes: Counter = Counter()
        sig_to_gen: dict[str, Generation] = {}
        for _ in range(n_samples):
            if not budget.can_use_tokens(50):
                break
            g = model.generate(
                prompt, max_tokens=min(512, budget.token_limit - budget.tokens_used)
            )
            tokens_used = g.metadata.get("tokens", 50)  # type: ignore[arg-type]
            try:
                budget.consume_tokens(tokens_used)
            except RuntimeError:
                break
            if g.text and g.text.strip():
                sig = _extract_answer_signature(g.text)
                sig_votes[sig] += 1
                sig_to_gen[sig] = g

        if sig_votes:
            majority_sig = sig_votes.most_common(1)[0][0]
            best_generation = sig_to_gen[majority_sig]
    else:
        # Single sample
        if budget.can_use_tokens(50):
            g = model.generate(
                prompt, max_tokens=min(512, budget.token_limit - budget.tokens_used)
            )
            tokens_used = g.metadata.get("tokens", 50)  # type: ignore[arg-type]
            try:
                budget.consume_tokens(tokens_used)
            except RuntimeError:
                pass
            if g.text and g.text.strip():
                best_generation = g

    # Step 2: Fixed-depth search (for medium and hard tiers)
    if search_depth > 0 and budget.can_use_tokens(100):
        search = FixedDepthTreeSearch(
            max_depth=search_depth, branching_factor=branching
        )
        search_result = search.run(prompt, model, budget)
        if search_result and search_result.text and search_result.text.strip():
            # Use search result if we don't have a generation yet
            if best_generation is None:
                best_generation = search_result

    # Step 3: Verification (for medium and hard tiers)
    if verify_level > 0 and best_generation and best_generation.text:
        if cascade is None:
            cascade = create_default_cascade()
        if budget.can_call_verifier():
            is_valid, confidence, level = cascade.verify(
                prompt, best_generation.text, budget
            )
            best_confidence = confidence

    # Ensure we return something
    if best_generation is None:
        best_generation = Generation(text="", logprobs=None, metadata={"tokens": 0})

    return best_generation
