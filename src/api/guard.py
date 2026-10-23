"""Live-mode guardrails: a per-client sliding-window rate limit and a global daily LLM-call budget. In memory, per process."""
import math
import threading
import time
from collections import defaultdict, deque
from datetime import datetime, timedelta, timezone


def _quota_tz():
    """The free-tier daily quota resets at midnight Pacific time; fall back to a fixed UTC-8 where tzdata is missing."""
    try:
        from zoneinfo import ZoneInfo

        return ZoneInfo("America/Los_Angeles")
    except Exception:
        return timezone(timedelta(hours=-8))


def quota_day(now=None):
    return (now or datetime.now(timezone.utc)).astimezone(_quota_tz()).date()


def seconds_until_quota_reset(now=None):
    now = (now or datetime.now(timezone.utc)).astimezone(_quota_tz())
    nxt = (now + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
    return max(1, math.ceil((nxt - now).total_seconds()))


class RateLimiter:
    """At most `limit` accepted requests per `window_s` seconds for each key (the client IP)."""

    def __init__(self, limit, window_s=3600.0, clock=time.monotonic):
        self.limit, self.window_s, self.clock = limit, window_s, clock
        self._hits = defaultdict(deque)
        self._lock = threading.Lock()

    def check(self, key):
        """(allowed, retry_after_seconds). An allowed call is recorded."""
        with self._lock:
            now, hits = self.clock(), self._hits[key]
            while hits and now - hits[0] >= self.window_s:
                hits.popleft()
            if len(hits) >= self.limit:
                return False, max(1, math.ceil(self.window_s - (now - hits[0])))
            hits.append(now)
            return True, 0


class DailyBudget:
    """Counts LLM calls per quota day across all clients. A run that is already under way is not interrupted, so the
    budget can be overshot by at most one run (an agent run is capped at 30 LLM calls)."""

    def __init__(self, limit, today=quota_day):
        self.limit, self._today = limit, today
        self._day, self._used = today(), 0
        self._lock = threading.Lock()

    def _roll(self):
        d = self._today()
        if d != self._day:
            self._day, self._used = d, 0

    def used(self):
        with self._lock:
            self._roll()
            return self._used

    def remaining(self):
        return max(0, self.limit - self.used())

    def add(self, n):
        with self._lock:
            self._roll()
            self._used += max(0, int(n))
