from __future__ import annotations

from collections import Counter
from typing import Optional

from ava.baselines.self_consistency import _extract_answer_signature
from ava.controllers.ava_controller import AVAController, ControllerState
from ava.core.interfaces import Budget, Generation, ModelProvider
from ava.uncertainty.estimators import (
    UncertaintyEstimate,
    aggregate_uncertainty,
    ConsistencyEstimator,
    TokenEntropyEstimator,
    TrajectoryFeatureEstimator,
)
from ava.verification.llm_verifier import LLMVerifier


class AVAAgent:
    """
    Anytime Verified Agent — adaptive compute allocation for LLM reasoning.

    Strategy:
    1. Bootstrap: generate 3 samples, quick-exit if all agree
    2. Adaptive sampling: generate more samples based on controller, stop when confident
    3. Verify-and-vote: on close votes only, re-solve the problem independently
       and add the result to the vote pool (verification as additional evidence,
       not as a veto)
    """

    def __init__(
        self,
        model: ModelProvider,
        *,
        controller: Optional[AVAController] = None,
        target_reliability: float = 0.9,
        use_feedback: bool = True,
        max_samples: Optional[int] = None,
        max_search_depth: Optional[int] = None,
    ) -> None:
        self.model = model
        self.controller = controller or AVAController(target_reliability=target_reliability)
        self.target_reliability = target_reliability
        self.use_feedback = use_feedback
        self.max_samples = max_samples
        self.verifier = LLMVerifier(model)
        self.consistency_estimator = ConsistencyEstimator()
        self.token_entropy_estimator = TokenEntropyEstimator()
        self.trajectory_estimator = TrajectoryFeatureEstimator()

    def _add_to_pool(
        self,
        text: str,
        global_sig_votes: Counter,
        sig_to_text: dict[str, str],
    ) -> str:
        sig = _extract_answer_signature(text)
        global_sig_votes[sig] += 1
        sig_to_text[sig] = text.strip()
        return sig

    def _pool_confidence(self, global_sig_votes: Counter) -> float:
        if not global_sig_votes:
            return 0.0
        total = sum(global_sig_votes.values())
        _, majority_count = global_sig_votes.most_common(1)[0]
        return majority_count / total

    def _best_from_pool(
        self,
        global_sig_votes: Counter,
        sig_to_text: dict[str, str],
    ) -> Optional[Generation]:
        if not global_sig_votes:
            return None
        majority_sig = global_sig_votes.most_common(1)[0][0]
        text = sig_to_text.get(majority_sig, "")
        if text:
            return Generation(text=text, logprobs=None, metadata={"tokens": 0})
        return None

    def _generate_sample(
        self, prompt: str, budget: Budget, global_sig_votes: Counter,
        sig_to_text: dict[str, str],
    ) -> bool:
        if not budget.can_use_tokens(10):
            return False
        try:
            g = self.model.generate(prompt, max_tokens=512)
            tokens_used = g.metadata.get("tokens", 50)
            if g.text and g.text.strip():
                try:
                    budget.consume_tokens(tokens_used)
                except RuntimeError:
                    budget.tokens_used = budget.token_limit
                self._add_to_pool(g.text, global_sig_votes, sig_to_text)
                return True
            else:
                # Empty response (reasoning model exhausted token budget).
                # Charge a heavy penalty — the API still billed for reasoning
                # tokens even though we got no visible output.
                try:
                    budget.consume_tokens(max(tokens_used, 200))
                except RuntimeError:
                    budget.tokens_used = budget.token_limit
                return False
        except Exception:
            pass
        return False

    def solve(
        self,
        prompt: str,
        budget: Budget,
        *,
        max_iterations: int = 10,
    ) -> Generation:
        """
        Solve with adaptive compute allocation.

        Phase 1 — Bootstrap (3 samples):
          Quick-exit if all 3 agree (easy problem, high confidence).

        Phase 2 — Adaptive sampling:
          Generate more samples guided by controller until confident
          (pool_conf >= 0.8 with >= 5 votes) or budget exhausted.

        Phase 3 — Verify-and-vote (close calls only):
          If the majority has < 75% of votes, re-solve the problem
          independently. The re-solve answer is added to the vote pool
          as additional evidence (not a veto). This gives the majority
          a chance to strengthen or the minority a chance to overtake.
        """
        global_sig_votes: Counter = Counter()
        sig_to_text: dict[str, str] = {}

        self._last_consistency_score: Optional[float] = None
        self._last_verifier_score: Optional[float] = None

        # ── Phase 1: Lean bootstrap (2 samples, permissive quick-exit) ──
        # Start with 2 samples rather than 3. If they agree, the problem is
        # likely easy for the model — exit immediately without paying for more
        # compute. This dramatically reduces token usage on easy problems where
        # adaptive machinery adds cost without benefit.
        for _ in range(2):
            self._generate_sample(prompt, budget, global_sig_votes, sig_to_text)

        total_votes = sum(global_sig_votes.values())
        pool_conf = self._pool_confidence(global_sig_votes)

        # Permissive quick-exit: if we got at least 2 samples and they all agree,
        # return immediately regardless of budget. Multiple samples agreeing is
        # strong evidence the answer is correct, and extra compute provides
        # diminishing returns.
        if total_votes >= 2 and pool_conf >= 1.0:
            result = self._best_from_pool(global_sig_votes, sig_to_text)
            if result:
                return result

        # Legacy budget-pressure quick-exit (kept for safety)
        remaining_frac = (budget.token_limit - budget.tokens_used) / max(1, budget.token_limit)
        if remaining_frac < 0.3 and pool_conf >= 0.66 and total_votes >= 2:
            result = self._best_from_pool(global_sig_votes, sig_to_text)
            if result:
                return result

        # ── Phase 2: Adaptive refinement ──
        depth_reached = 0
        nodes_expanded = 0
        consecutive_empty = 0

        for iteration in range(max_iterations):
            if not budget.can_use_tokens(10):
                break

            # If the model keeps returning empty, stop — it can't solve this one
            if consecutive_empty >= 3:
                break

            pool_conf = self._pool_confidence(global_sig_votes)
            total_votes = sum(global_sig_votes.values())

            # Strong agreement with enough evidence — stop
            if pool_conf >= 0.8 and total_votes >= 5:
                break

            # Diminishing returns: extra samples on "model doesn't know"
            # problems just add noise and burn budget. Stop earlier on low
            # confidence — these are systematic failures, not noisy failures.
            if total_votes >= 5 and pool_conf < 0.5:
                break
            if total_votes >= 7 and pool_conf < 0.6:
                break

            budget_remaining = (budget.token_limit - budget.tokens_used) / max(1, budget.token_limit)

            best_gen = self._best_from_pool(global_sig_votes, sig_to_text)
            uncertainty_est = self._estimate_uncertainty(
                prompt=prompt,
                generation=best_gen,
                depth_reached=depth_reached,
                nodes_expanded=nodes_expanded,
                budget_remaining=budget_remaining,
            )

            state = {
                "uncertainty": uncertainty_est,
                "budget_remaining": budget_remaining,
                "depth_reached": depth_reached,
                "nodes_expanded": nodes_expanded,
            }

            if self.controller.should_stop(state):
                break

            decision = self.controller.decide(state)

            n_samples = max(decision.get("samples", 2), 2)
            if self.max_samples is not None:
                n_samples = min(n_samples, self.max_samples)

            got_content = False
            for _ in range(n_samples):
                if self._generate_sample(prompt, budget, global_sig_votes, sig_to_text):
                    got_content = True
                    consecutive_empty = 0
                elif not budget.can_use_tokens(10):
                    break

                # Check stop conditions after each sample to avoid
                # over-sampling on problems where more samples don't help.
                _curr_votes = sum(global_sig_votes.values())
                _curr_conf = self._pool_confidence(global_sig_votes)
                # Strong agreement — stop immediately
                if _curr_conf >= 0.8 and _curr_votes >= 3:
                    break
                # Systematic failure pattern — stop, more samples won't help
                if _curr_votes >= 5 and _curr_conf < 0.5:
                    break
                if _curr_votes >= 7 and _curr_conf < 0.6:
                    break

            if not got_content:
                consecutive_empty += 1

            nodes_expanded += n_samples
            depth_reached += 1

            if global_sig_votes and self.use_feedback:
                self._last_consistency_score = self.consistency_estimator.estimate(
                    dict(global_sig_votes)
                )

        # ── Phase 3: Verify-and-vote on close calls ──
        best_generation = self._best_from_pool(global_sig_votes, sig_to_text)

        if best_generation and best_generation.text and best_generation.text.strip():
            pool_conf = self._pool_confidence(global_sig_votes)
            total_votes = sum(global_sig_votes.values())

            # Only verify on CLOSE votes (majority < 75%) with enough budget
            # Skip if already too many samples — verification won't help noisy pools
            if pool_conf < 0.75 and total_votes >= 3 and total_votes <= 9 and budget.can_use_tokens(10):
                agrees, conf, new_text, new_sig = self.verifier.verify_and_vote(
                    prompt, best_generation.text, budget
                )
                # Add the re-solve answer to the vote pool (not a veto!)
                if new_sig and new_text:
                    global_sig_votes[new_sig] += 1
                    sig_to_text[new_sig] = new_text

                    # If still close after adding vote, try one more re-solve
                    new_conf = self._pool_confidence(global_sig_votes)
                    if new_conf < 0.6 and budget.can_use_tokens(10):
                        _, _, new_text2, new_sig2 = self.verifier.verify_and_vote(
                            prompt, best_generation.text, budget
                        )
                        if new_sig2 and new_text2:
                            global_sig_votes[new_sig2] += 1
                            sig_to_text[new_sig2] = new_text2

                # Update best from the expanded pool
                best_generation = self._best_from_pool(global_sig_votes, sig_to_text)

        # Fallback: generate one more if we have nothing
        if best_generation is None or not best_generation.text or not best_generation.text.strip():
            remaining = budget.token_limit - budget.tokens_used
            if remaining >= 50:
                try:
                    fallback = self.model.generate(prompt, max_tokens=min(512, remaining))
                    if fallback.text and fallback.text.strip():
                        best_generation = fallback
                        try:
                            budget.consume_tokens(fallback.metadata.get("tokens", 50))
                        except RuntimeError:
                            pass
                except Exception:
                    pass

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
        entropy = None
        if generation and generation.logprobs:
            entropy = self.token_entropy_estimator.estimate(generation.logprobs)

        consistency = getattr(self, "_last_consistency_score", None)
        verifier_score = getattr(self, "_last_verifier_score", None)

        trajectory_unc = self.trajectory_estimator.estimate(
            depth_reached, nodes_expanded, budget_remaining
        )

        return aggregate_uncertainty(
            entropy=entropy,
            consistency=consistency,
            verifier=verifier_score,
            trajectory=trajectory_unc,
        )
