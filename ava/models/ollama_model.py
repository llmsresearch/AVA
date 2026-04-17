from __future__ import annotations

import time
from typing import List, Optional

import requests

from ava.core.interfaces import Generation, ModelProvider


class OllamaModel(ModelProvider):
    """
    Ollama model provider implementing ModelProvider protocol.

    Connects to a local Ollama instance for inference. Supports any model
    available in the local Ollama registry (e.g., qwen2.5:7b, gemma3:27b).

    Args:
        model_name: Ollama model identifier (default: "qwen2.5:7b")
        base_url: Ollama server URL (default: "http://localhost:11434")
    """

    def __init__(
        self,
        model_name: str = "qwen2.5:7b",
        base_url: str = "http://localhost:11434",
    ) -> None:
        self.model_name = model_name
        self.base_url = base_url.rstrip("/")

        # Verify Ollama is reachable
        try:
            resp = requests.get(f"{self.base_url}/api/tags", timeout=5)
            resp.raise_for_status()
            models = [m["name"] for m in resp.json().get("models", [])]
            if model_name not in models:
                available = ", ".join(models) if models else "none"
                raise ValueError(
                    f"Model '{model_name}' not found in Ollama. "
                    f"Available models: {available}. "
                    f"Pull it with: ollama pull {model_name}"
                )
        except requests.ConnectionError as e:
            raise RuntimeError(
                f"Cannot connect to Ollama at {self.base_url}. "
                f"Start it with: ollama serve"
            ) from e

    def generate(
        self, prompt: str, *, temperature: float = 0.7, max_tokens: int = 512
    ) -> Generation:
        """
        Generate a single completion via Ollama chat API.

        Args:
            prompt: Input prompt
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate

        Returns:
            Generation object
        """
        url = f"{self.base_url}/api/chat"

        payload = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            },
        }

        max_retries = 5
        for attempt in range(max_retries):
            try:
                response = requests.post(url, json=payload, timeout=120)

                if response.status_code >= 500 and attempt < max_retries - 1:
                    wait_time = min(2 ** attempt, 30)
                    time.sleep(wait_time)
                    continue

                if not response.ok:
                    error_detail = response.text[:500]
                    raise RuntimeError(
                        f"Ollama API error {response.status_code}: {error_detail}\n"
                        f"Model: {self.model_name}"
                    )

                data = response.json()
                break

            except requests.RequestException as e:
                if attempt < max_retries - 1:
                    wait_time = min(2 ** attempt, 30)
                    time.sleep(wait_time)
                    continue
                raise RuntimeError(
                    f"Ollama API error after {max_retries} retries: {e}"
                ) from e

        # Extract content from chat response
        message = data.get("message", {})
        content = message.get("content", "")

        # Token counts from Ollama response
        prompt_tokens = data.get("prompt_eval_count", 0)
        completion_tokens = data.get("eval_count", 0)

        # Budget tracks OUTPUT tokens only (not prompt).
        # Prompt cost is fixed and identical across methods.
        return Generation(
            text=content,
            logprobs=None,
            metadata={
                "tokens": completion_tokens,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "model": self.model_name,
            },
        )

    def batch_generate(
        self, prompts: List[str], *, temperature: float = 0.7, max_tokens: int = 512
    ) -> List[Generation]:
        """
        Generate completions for multiple prompts (sequential).

        Args:
            prompts: List of input prompts
            temperature: Sampling temperature
            max_tokens: Maximum tokens per generation

        Returns:
            List of Generation objects
        """
        return [
            self.generate(p, temperature=temperature, max_tokens=max_tokens)
            for p in prompts
        ]
