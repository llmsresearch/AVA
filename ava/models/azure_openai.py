from __future__ import annotations

import os
import time
from typing import List, Optional

import requests

from ava.core.interfaces import Generation, ModelProvider
from ava.models.rate_limiter import get_rate_limiter


class AzureOpenAIModel(ModelProvider):
    """
    Azure OpenAI model provider implementing ModelProvider protocol.

    Uses environment variables:
    - MODEL_ENDPOINT: Azure endpoint URL
    - MODEL_KEY: Azure API key
    - MODEL_DEPLOYMENT: Deployment name (e.g., "gpt-4")
    - MODEL_API_VERSION: API version (e.g., "2024-12-01-preview")
    """

    def __init__(
        self,
        endpoint: Optional[str] = None,
        api_key: Optional[str] = None,
        deployment: Optional[str] = None,
        api_version: Optional[str] = None,
    ) -> None:
        self.endpoint = endpoint or os.getenv("AZURE_MODEL_ENDPOINT", os.getenv("MODEL_ENDPOINT", ""))
        self.api_key = api_key or os.getenv("AZURE_MODEL_KEY", os.getenv("MODEL_KEY", ""))
        self.deployment = deployment or os.getenv("AZURE_MODEL_DEPLOYMENT", os.getenv("MODEL_DEPLOYMENT", "gpt-4"))
        self.api_version = api_version or os.getenv("AZURE_MODEL_API_VERSION", os.getenv("MODEL_API_VERSION", "2024-12-01-preview"))

        if not self.endpoint or not self.api_key:
            raise ValueError(
                "Azure OpenAI credentials not provided. Set MODEL_ENDPOINT and MODEL_KEY "
                "environment variables or pass them as arguments."
            )

        # Ensure endpoint doesn't end with /
        self.endpoint = self.endpoint.rstrip("/")
        self.base_url = f"{self.endpoint}/openai/deployments/{self.deployment}"

    def generate(
        self, prompt: str, *, temperature: float = 0.7, max_tokens: int = 512
    ) -> Generation:
        """
        Generate a single completion.

        Args:
            prompt: Input prompt
            temperature: Sampling temperature (ignored for this model)
            max_tokens: Maximum tokens to generate

        Returns:
            Generation object
        """
        url = f"{self.base_url}/chat/completions?api-version={self.api_version}"

        headers = {
            "Content-Type": "application/json",
            "api-key": self.api_key,
        }

        payload = {
            "messages": [{"role": "user", "content": prompt}],
            "max_completion_tokens": max_tokens,  # Use max_completion_tokens for Azure OpenAI
        }
        # Only include temperature if model supports it (this model only supports 1.0)
        # For models that support variable temperature, uncomment:
        # if temperature != 1.0:
        #     payload["temperature"] = temperature

        # Rate limiting: prevent hitting limits proactively
        rate_limiter = get_rate_limiter()
        rate_limiter.wait_if_needed(estimated_tokens=max_tokens)
        
        # Retry logic for transient errors and rate limits
        max_retries = 10  # Increased for rate limit handling
        increased_tokens_attempts = 0
        for attempt in range(max_retries):
            try:
                response = requests.post(url, headers=headers, json=payload, timeout=60)
                
                # Handle rate limiting (429 Too Many Requests)
                if response.status_code == 429:
                    time.sleep(60)
                    continue
                
                if response.status_code >= 500 and attempt < max_retries - 1:
                    wait_time = min(2 ** attempt, 30)
                    time.sleep(wait_time)
                    continue
                
                if not response.ok:
                    error_detail = response.text[:500]
                    if "rate limit" in error_detail.lower() or "throttle" in error_detail.lower():
                        time.sleep(60)
                        continue

                    # Handle insufficient max tokens (Azure returns 400)
                    if (
                        response.status_code == 400
                        and (
                            "max_tokens" in error_detail.lower()
                            or "max completion tokens" in error_detail.lower()
                            or "model output limit was reached" in error_detail.lower()
                        )
                        and increased_tokens_attempts < 3
                    ):
                        # Increase max completion tokens and retry
                        new_max = min(int(payload.get("max_completion_tokens", max_tokens)) * 2, 2048)
                        if new_max > payload.get("max_completion_tokens", max_tokens):
                            increased_tokens_attempts += 1
                            payload["max_completion_tokens"] = new_max
                            print(
                                f"[Max tokens increase] 400 error: increasing max_completion_tokens to {new_max} and retrying..."
                            )
                            # brief backoff
                            time.sleep(1)
                            continue
                    
                    raise RuntimeError(
                        f"Azure OpenAI API error {response.status_code}: {error_detail}\n"
                        f"URL: {url}\n"
                        f"Check: deployment name '{self.deployment}' and API version '{self.api_version}'"
                    )
                
                response.raise_for_status()
                data = response.json()
                break
                
            except requests.RequestException as e:
                error_str = str(e).lower()
                if "429" in error_str or "rate limit" in error_str or "throttle" in error_str:
                    time.sleep(60)
                    if attempt < max_retries - 1:
                        continue
                
                if attempt < max_retries - 1:
                    wait_time = min(2 ** attempt, 30)
                    time.sleep(wait_time)
                    continue
                    
                raise RuntimeError(f"Azure OpenAI API error after {max_retries} retries: {e}") from e

        # Retry wrapper: if content is empty, retry up to 2 more times
        for _empty_retry in range(3):
            choices = data.get("choices", [])
            if not choices:
                raise ValueError("No choices in response")

            choice = choices[0]
            message = choice.get("message", {})
            content = message.get("content") or ""

            if not content:
                content = choice.get("text") or ""
            if not content:
                content = (message.get("delta") or {}).get("content") or ""

            # If content is non-empty, break out of retry loop
            if content and content.strip():
                break

            # Empty content — retry if we have attempts left
            if _empty_retry < 2:
                time.sleep(0.5)
                try:
                    response = requests.post(url, headers=headers, json=payload, timeout=60)
                    if response.ok:
                        data = response.json()
                        continue
                except Exception:
                    pass
                break

        usage = data.get("usage", {})
        prompt_tokens = usage.get("prompt_tokens", 0)
        completion_tokens = usage.get("completion_tokens", 0)
        total_tokens = usage.get("total_tokens", 0)

        # For reasoning models (GPT-5/o1/o3), exclude hidden reasoning tokens
        # from the budget — they are model-internal and not controllable by the method.
        completion_details = usage.get("completion_tokens_details") or {}
        reasoning_tokens = completion_details.get("reasoning_tokens") or 0
        output_tokens = completion_tokens - reasoning_tokens

        # Record TOTAL tokens for rate limiting (API bills for all tokens)
        rate_limiter = get_rate_limiter()
        rate_limiter.record_tokens(total_tokens)

        # Budget tracks OUTPUT tokens only (not prompt, not reasoning).
        # Prompt cost is fixed and identical across methods; reasoning is
        # model-internal and uncontrollable.  Output tokens are what the
        # method actually decides to generate.
        return Generation(
            text=content,
            logprobs=None,
            metadata={
                "tokens": output_tokens,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": output_tokens,
                "reasoning_tokens": reasoning_tokens,
                "total_tokens_actual": total_tokens,
            },
        )

    def batch_generate(
        self, prompts: List[str], *, temperature: float = 0.7, max_tokens: int = 512
    ) -> List[Generation]:
        """
        Generate completions for multiple prompts (sequential for now).

        Args:
            prompts: List of input prompts
            temperature: Sampling temperature
            max_tokens: Maximum tokens per generation

        Returns:
            List of Generation objects
        """
        return [self.generate(p, temperature=temperature, max_tokens=max_tokens) for p in prompts]
