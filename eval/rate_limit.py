"""Shared OpenAI rate limiter for eval runs.

The agent's model and the judge model both call gpt-4o under the same org,
sharing the same tokens-per-minute cap. A single InMemoryRateLimiter instance
passed to both keeps combined throughput under that cap instead of each
client bursting independently.
"""

from langchain_core.rate_limiters import InMemoryRateLimiter

shared_rate_limiter = InMemoryRateLimiter(
    requests_per_second=0.1,  # ~1 request per 10s
    check_every_n_seconds=0.5,
    max_bucket_size=1,  # no burst allowance
)
