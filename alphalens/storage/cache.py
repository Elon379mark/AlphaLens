import json
import logging
import threading
import time
from typing import Any, Dict, Optional

try:
    import redis
except ImportError:
    redis = None

logger = logging.getLogger(__name__)

class CacheManager:
    """
    Manages Redis caching of active cross-sectional factor matrices for sub-millisecond layer delivery.
    Falls back to a thread-safe in-memory cache for local development/testing.
    """
    def __init__(
        self,
        host: Optional[str] = None,
        port: Optional[int] = None,
        db: Optional[int] = None,
        password: Optional[str] = None
    ):
        import os
        self.host = host or os.environ.get("REDIS_HOST", "localhost")
        try:
            self.port = int(port or os.environ.get("REDIS_PORT", 6379))
        except (ValueError, TypeError):
            self.port = 6379
        try:
            self.db = int(db or os.environ.get("REDIS_DB", 0))
        except (ValueError, TypeError):
            self.db = 0
        self.password = password or os.environ.get("REDIS_PASSWORD", None)
        self.client = None
        self.is_redis = False
        
        # In-memory fallback: stores {key: (serialized_data, expiry_timestamp)}
        self._local_cache: Dict[str, tuple] = {}
        self._lock = threading.Lock()
        
        self._init_redis()

    def _init_redis(self):
        if redis is not None:
            try:
                self.client = redis.Redis(
                    host=self.host,
                    port=self.port,
                    db=self.db,
                    password=self.password,
                    socket_connect_timeout=2.0
                )
                # Ping to check connectivity
                self.client.ping()
                self.is_redis = True
                logger.info("Connected to Redis server.")
                return
            except Exception as e:
                logger.warning(f"Failed to connect to Redis: {e}. Falling back to in-memory cache.")
        else:
            logger.info("Redis package not installed. Using in-memory cache.")
        
        self.is_redis = False
        self.client = None

    def set_matrix(self, key: str, matrix_data: Dict[str, Any], expire_seconds: int = 3600):
        """
        Saves a cross-sectional factor matrix as serialized JSON.
        """
        serialized = json.dumps(matrix_data)
        if self.is_redis and self.client:
            try:
                self.client.setex(key, expire_seconds, serialized)
                return
            except Exception as e:
                logger.error(f"Redis set error: {e}. Writing to in-memory fallback.")
        
        with self._lock:
            self._local_cache[key] = (serialized, time.time() + expire_seconds)

    def get_matrix(self, key: str) -> Optional[Dict[str, Any]]:
        """
        Retrieves a cross-sectional factor matrix and deserializes it.
        """
        serialized = None
        redis_failed = False
        if self.is_redis and self.client:
            try:
                val = self.client.get(key)
                if val:
                    serialized = val.decode("utf-8")
            except Exception as e:
                logger.error(f"Redis get error: {e}. Reading from in-memory fallback.")
                redis_failed = True

        # Fall through to in-memory fallback ONLY if Redis is not configured or failed
        if not serialized and (not self.is_redis or redis_failed):
            with self._lock:
                entry = self._local_cache.get(key)
                if entry:
                    val, expiry = entry
                    if time.time() < expiry:
                        serialized = val
                    else:
                        del self._local_cache[key]

        if serialized:
            try:
                return json.loads(serialized)
            except json.JSONDecodeError:
                logger.error(f"Failed to deserialize matrix from key: {key}")
                return None
        return None

    def delete(self, key: str):
        if self.is_redis and self.client:
            try:
                self.client.delete(key)
            except Exception as e:
                logger.error(f"Redis delete error: {e}.")
        
        with self._lock:
            if key in self._local_cache:
                del self._local_cache[key]
