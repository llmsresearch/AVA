from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Protocol, Tuple


@dataclass
class Generation:
    text: str
    logprobs: Optional[List[float]] = None
    metadata: Optional[Dict[str, Any]] = None


class ModelProvider(Protocol):
    def generate(self, prompt: str, *, temperature: float = 0.7, max_tokens: int = 512) -> Generation: ...

    def batch_generate(self, prompts: List[str], *, temperature: float = 0.7, max_tokens: int = 512) -> List[Generation]: ...


class Verifier(Protocol):
    def verify(self, input_text: str, output_text: str) -> Tuple[bool, float]:
        """
        Returns (is_valid, score in [0,1]). Implementations may execute code,
        parse structure, or call heuristics.
        """


class Tool(Protocol):
    name: str

    def call(self, *args: Any, **kwargs: Any) -> Any: ...


class SearchStrategy(Protocol):
    def run(self, prompt: str, model: ModelProvider, budget: "Budget") -> Generation: ...


class Controller(Protocol):
    def decide(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Return next action parameters: samples, depth, verifier level, etc."""


@dataclass
class Budget:
    token_limit: int
    tool_calls_limit: int = 0
    verify_calls_limit: int = 0

    tokens_used: int = 0
    tool_calls_used: int = 0
    verify_calls_used: int = 0

    def can_use_tokens(self, n: int) -> bool:
        return (self.tokens_used + n) <= self.token_limit

    def consume_tokens(self, n: int) -> None:
        if not self.can_use_tokens(n):
            raise RuntimeError("Token budget exceeded")
        self.tokens_used += n

    def can_call_tool(self) -> bool:
        return (self.tool_calls_used + 1) <= self.tool_calls_limit

    def consume_tool_call(self) -> None:
        if not self.can_call_tool():
            raise RuntimeError("Tool-call budget exceeded")
        self.tool_calls_used += 1

    def can_call_verifier(self) -> bool:
        return (self.verify_calls_used + 1) <= self.verify_calls_limit

    def consume_verify_call(self) -> None:
        if not self.can_call_verifier():
            raise RuntimeError("Verifier-call budget exceeded")
        self.verify_calls_used += 1



