"""Settings of the API, read from the environment when the app is created (no heavy imports here: demo mode needs none)."""
import os
from dataclasses import dataclass
from pathlib import Path

DESIGN_VERSION = "agent-v3.1"  # the git tag of the agent code this service is built on
MODES = ("demo", "live")


@dataclass
class Settings:
    mode: str = "demo"
    bundle_path: Path = Path("demo/memos.json")
    rate_limit_per_hour: int = 6  # live runs per client IP per hour
    daily_llm_budget: int = 400  # LLM calls per day across all clients (the free tier allows 500 per model per day)
    request_timeout_s: float = 240.0
    max_concurrent_runs: int = 1  # free-tier throttle: one agent run at a time

    @classmethod
    def from_env(cls, **overrides):
        env = os.environ
        mode = (overrides.pop("mode", None) or env.get("TRACEGUARD_MODE") or "demo").strip().lower()
        if mode not in MODES:
            raise ValueError(f"TRACEGUARD_MODE must be one of {MODES}, got {mode!r}")
        s = cls(
            mode=mode,
            bundle_path=Path(env.get("TRACEGUARD_BUNDLE", "demo/memos.json")),
            rate_limit_per_hour=int(env.get("TRACEGUARD_RATE_LIMIT_PER_HOUR", cls.rate_limit_per_hour)),
            daily_llm_budget=int(env.get("TRACEGUARD_DAILY_LLM_BUDGET", cls.daily_llm_budget)),
            request_timeout_s=float(env.get("TRACEGUARD_REQUEST_TIMEOUT_S", cls.request_timeout_s)),
        )
        for k, v in overrides.items():
            setattr(s, k, v)
        return s
