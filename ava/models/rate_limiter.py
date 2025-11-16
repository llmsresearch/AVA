"""Simple rate limiter to prevent hitting API rate limits."""

from __future__ import annotations

import time
from collections import deque
from threading import Lock


class RateLimiter:
    """
    Simple rate limiter to track and enforce rate limits.
    
    For GPT-5:
    - 10,000,000 tokens per minute
    - 100,000 requests per minute
    """
    
    def __init__(
        self,
        requests_per_minute: int = 100000,
        tokens_per_minute: int = 10000000,
        safety_margin: float = 0.9,  # Use 90% of limit to be safe
    ) -> None:
        self.requests_per_minute = int(requests_per_minute * safety_margin)
        self.tokens_per_minute = int(tokens_per_minute * safety_margin)
        self.request_times: deque[float] = deque()
        self.token_counts: deque[tuple[float, int]] = deque()  # (timestamp, tokens)
        self.lock = Lock()
    
    def wait_if_needed(self, estimated_tokens: int = 0) -> None:
        """
        Wait if we're approaching rate limits.
        
        Args:
            estimated_tokens: Estimated tokens for next request
        """
        with self.lock:
            now = time.time()
            
            # Clean old entries (> 1 minute old)
            while self.request_times and now - self.request_times[0] > 60:
                self.request_times.popleft()
            
            while self.token_counts and now - self.token_counts[0][0] > 60:
                self.token_counts.popleft()
            
            # Check request rate
            if len(self.request_times) >= self.requests_per_minute:
                oldest = self.request_times[0]
                wait_time = 60 - (now - oldest) + 1  # Add 1s buffer
                if wait_time > 0:
                    print(f"[Rate Limiter] Request limit: waiting {wait_time:.1f}s")
                    time.sleep(wait_time)
                    now = time.time()
                    # Clean again after wait
                    while self.request_times and now - self.request_times[0] > 60:
                        self.request_times.popleft()
            
            # Check token rate
            total_tokens = sum(tokens for _, tokens in self.token_counts)
            if total_tokens + estimated_tokens > self.tokens_per_minute:
                oldest = self.token_counts[0][0] if self.token_counts else now
                wait_time = 60 - (now - oldest) + 1
                if wait_time > 0:
                    print(f"[Rate Limiter] Token limit: waiting {wait_time:.1f}s")
                    time.sleep(wait_time)
                    now = time.time()
                    # Clean again after wait
                    while self.token_counts and now - self.token_counts[0][0] > 60:
                        self.token_counts.popleft()
            
            # Record this request
            self.request_times.append(now)
            if estimated_tokens > 0:
                self.token_counts.append((now, estimated_tokens))
    
    def record_tokens(self, actual_tokens: int) -> None:
        """Record actual tokens used for accurate tracking."""
        with self.lock:
            now = time.time()
            self.token_counts.append((now, actual_tokens))
            
            # Clean old entries
            while self.token_counts and now - self.token_counts[0][0] > 60:
                self.token_counts.popleft()


# Global rate limiter instance
_global_limiter: RateLimiter | None = None


def get_rate_limiter() -> RateLimiter:
    """Get or create global rate limiter."""
    global _global_limiter
    if _global_limiter is None:
        _global_limiter = RateLimiter()
    return _global_limiter


