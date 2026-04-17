"""
cache_manager.py
Provides intelligent caching and rate-limiting for API calls to Wikimedia Commons.

Reduces redundant API calls, improves performance, and helps respect Wikimedia's
rate limits. Implements TTL-based caching with fallback strategies.
"""

import logging
import pickle
import time
from datetime import datetime, timedelta
from pathlib import Path
from threading import Lock
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class CacheManager:
    """Thread-safe cache manager for API responses with TTL and persistence."""

    def __init__(self, cache_dir: str = "./cache", ttl_hours: int = 24):
        """Initialize cache manager.
        
        Args:
            cache_dir: Directory to store cache files
            ttl_hours: Time-to-live for cached entries in hours
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.ttl_seconds = ttl_hours * 3600
        self._memory_cache: Dict[str, tuple[Any, float]] = {}
        self._lock = Lock()
        logger.info(f"Cache initialized at {self.cache_dir} (TTL: {ttl_hours}h)")

    def _get_cache_path(self, key: str) -> Path:
        """Generate cache file path for a key."""
        safe_key = "".join(c if c.isalnum() or c in "-_" else "_" for c in key)
        return self.cache_dir / f"{safe_key}.pkl"

    def _is_expired(self, timestamp: float) -> bool:
        """Check if a cached entry has expired."""
        return (time.time() - timestamp) > self.ttl_seconds

    def get(self, key: str) -> Optional[Any]:
        """Retrieve a cached value if it exists and hasn't expired."""
        with self._lock:
            # Check memory cache first (fastest)
            if key in self._memory_cache:
                value, timestamp = self._memory_cache[key]
                if not self._is_expired(timestamp):
                    logger.debug(f"Cache hit (memory): {key}")
                    return value
                else:
                    del self._memory_cache[key]

            # Check disk cache
            cache_path = self._get_cache_path(key)
            if cache_path.exists():
                try:
                    with open(cache_path, "rb") as f:
                        value, timestamp = pickle.load(f)
                    if not self._is_expired(timestamp):
                        self._memory_cache[key] = (value, timestamp)
                        logger.debug(f"Cache hit (disk): {key}")
                        return value
                    else:
                        cache_path.unlink()  # Remove expired cache
                except Exception as e:
                    logger.warning(f"Failed to load cache for {key}: {e}")

        return None

    def set(self, key: str, value: Any) -> None:
        """Store a value in cache with current timestamp."""
        with self._lock:
            timestamp = time.time()
            self._memory_cache[key] = (value, timestamp)

            # Persist to disk
            cache_path = self._get_cache_path(key)
            try:
                with open(cache_path, "wb") as f:
                    pickle.dump((value, timestamp), f)
                logger.debug(f"Cached: {key}")
            except Exception as e:
                logger.error(f"Failed to cache {key}: {e}")

    def clear(self) -> None:
        """Clear all cached entries."""
        with self._lock:
            self._memory_cache.clear()
            for cache_file in self.cache_dir.glob("*.pkl"):
                cache_file.unlink()
            logger.info("Cache cleared")

    def stats(self) -> Dict[str, int]:
        """Return cache statistics."""
        with self._lock:
            memory_count = len(self._memory_cache)
            disk_count = len(list(self.cache_dir.glob("*.pkl")))
            return {"memory": memory_count, "disk": disk_count}


class RateLimiter:
    """Token-bucket rate limiter for API calls."""

    def __init__(self, calls_per_second: float = 1.0, burst_size: int = 5):
        """Initialize rate limiter.
        
        Args:
            calls_per_second: Allowed rate (e.g., 0.25 = 1 call per 4 seconds)
            burst_size: Maximum burst capacity
        """
        self.rate = calls_per_second
        self.burst_size = burst_size
        self.tokens = float(burst_size)
        self.last_update = time.time()
        self._lock = Lock()

    def wait(self) -> None:
        """Block until a token is available, then consume it."""
        with self._lock:
            now = time.time()
            elapsed = now - self.last_update
            self.tokens = min(self.burst_size, self.tokens + elapsed * self.rate)
            self.last_update = now

            if self.tokens < 1.0:
                sleep_time = (1.0 - self.tokens) / self.rate
                logger.debug(f"Rate limit: sleeping {sleep_time:.2f}s")
                time.sleep(sleep_time)
                self.tokens = 0.0
            else:
                self.tokens -= 1.0
