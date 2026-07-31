"""Shared OpenAI rate limiter for eval runs.

The agent's model and the judge model both call gpt-4o under the same org and
share one tokens-per-minute cap (30k TPM). A single limiter instance passed to
both keeps their combined throughput under it, instead of each client bursting
independently and getting 429s.

Without this, the *judge* exhausts its retries and returns score=None, so
examples run but never get graded.

NOTE: this limiter was once suspected of causing eval runs to hang and was
removed. It was not the cause — the hang was a GIL/SQLite-mutex deadlock from
the strip_accents() UDF on agent.py's shared connection (see the _db_lock
comment there). Removing this limiter only costs you unscored examples.

Pacing: ~0.35 req/s ≈ 21 requests/min. Agent calls run ~1.3k tokens each, so
this lands near ~21k TPM — comfortably under the 30k cap, with the models'
own max_retries absorbing any spillover.
"""

from langchain_core.rate_limiters import InMemoryRateLimiter

shared_rate_limiter = InMemoryRateLimiter(
    requests_per_second=0.35,
    check_every_n_seconds=0.5,
    max_bucket_size=1,  # no burst allowance
)
