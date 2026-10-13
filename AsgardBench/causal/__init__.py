"""Causal-invariant next-action diagnostics for AsgardBench."""

from dotenv import find_dotenv, load_dotenv

# Local credentials are explicitly gitignored. Existing process variables win,
# so production/CI secret injection is never overwritten.
local_env = find_dotenv(".env.causal.local", usecwd=True)
if local_env:
    load_dotenv(local_env, override=False)

from AsgardBench.causal.metrics import compute_metrics

__all__ = ["compute_metrics"]
