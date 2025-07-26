"""
Rate Limiting and Caching module for API optimization
Author: kg290
Date: 2025-07-26
"""

import time
import asyncio
import logging
from typing import Dict, Any, Optional, Callable
from dataclasses import dataclass, field
from collections import defaultdict
import hashlib
import json
from enum import Enum

logger = logging.getLogger(__name__)


class RateLimitStatus(Enum):
    SUCCESS = "success"
    RATE_LIMITED = "rate_limited"
    ERROR = "error"


@dataclass
class RateLimitConfig:
    requests_per_minute: int = 30  # Conservative rate limit for Groq API
    requests_per_hour: int = 1000
    max_retries: int = 3
    base_delay: float = 1.0  # Base delay for exponential backoff
    max_delay: float = 60.0  # Max delay between retries
    cache_ttl: int = 3600  # Cache TTL in seconds (1 hour)
    enable_caching: bool = True


@dataclass
class CacheEntry:
    data: Any
    timestamp: float
    ttl: int
    access_count: int = 0
    last_accessed: float = field(default_factory=time.time)


@dataclass
class RateLimitResult:
    status: RateLimitStatus
    data: Any = None
    cached: bool = False
    retry_after: Optional[float] = None
    attempts: int = 1
    error_message: Optional[str] = None


class ResponseCache:
    """Thread-safe response cache with TTL support"""
    
    def __init__(self, max_size: int = 1000):
        self.cache: Dict[str, CacheEntry] = {}
        self.max_size = max_size
        self._lock = asyncio.Lock()
    
    def _generate_key(self, query: str, context_hash: str = "") -> str:
        """Generate cache key from query and context"""
        combined = f"{query}:{context_hash}"
        return hashlib.md5(combined.encode()).hexdigest()
    
    async def get(self, key: str) -> Optional[Any]:
        """Get cached response if valid"""
        async with self._lock:
            if key not in self.cache:
                return None
            
            entry = self.cache[key]
            current_time = time.time()
            
            # Check if entry has expired
            if current_time - entry.timestamp > entry.ttl:
                del self.cache[key]
                logger.info(f"[kg290] Cache entry expired: {key[:8]}...")
                return None
            
            # Update access stats
            entry.access_count += 1
            entry.last_accessed = current_time
            
            logger.info(f"[kg290] Cache hit for key: {key[:8]}... (accessed {entry.access_count} times)")
            return entry.data
    
    async def set(self, key: str, data: Any, ttl: int = 3600):
        """Set cached response with TTL"""
        async with self._lock:
            # Implement LRU eviction if cache is full
            if len(self.cache) >= self.max_size:
                await self._evict_lru()
            
            self.cache[key] = CacheEntry(
                data=data,
                timestamp=time.time(),
                ttl=ttl
            )
            logger.info(f"[kg290] Cached response for key: {key[:8]}... (TTL: {ttl}s)")
    
    async def _evict_lru(self):
        """Evict least recently used entry"""
        if not self.cache:
            return
        
        lru_key = min(self.cache.keys(), 
                      key=lambda k: self.cache[k].last_accessed)
        del self.cache[lru_key]
        logger.info(f"[kg290] Evicted LRU cache entry: {lru_key[:8]}...")
    
    async def clear(self):
        """Clear all cache entries"""
        async with self._lock:
            self.cache.clear()
            logger.info("[kg290] Cache cleared")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        current_time = time.time()
        valid_entries = sum(1 for entry in self.cache.values() 
                           if current_time - entry.timestamp <= entry.ttl)
        
        return {
            "total_entries": len(self.cache),
            "valid_entries": valid_entries,
            "expired_entries": len(self.cache) - valid_entries,
            "max_size": self.max_size,
            "hit_rate": sum(entry.access_count for entry in self.cache.values()),
            "user": "kg290",
            "timestamp": current_time
        }


class RateLimiter:
    """Advanced rate limiter with exponential backoff and caching"""
    
    def __init__(self, config: RateLimitConfig = None):
        self.config = config or RateLimitConfig()
        self.request_times: Dict[str, list] = defaultdict(list)
        self.cache = ResponseCache() if self.config.enable_caching else None
        self._lock = asyncio.Lock()
        
        logger.info(f"[kg290] RateLimiter initialized with {self.config.requests_per_minute} RPM limit")
    
    async def execute_with_rate_limit(self, 
                                     func: Callable,
                                     cache_key: str,
                                     *args,
                                     **kwargs) -> RateLimitResult:
        """Execute function with rate limiting and caching"""
        
        # Check cache first
        if self.cache:
            cached_data = await self.cache.get(cache_key)
            if cached_data is not None:
                return RateLimitResult(
                    status=RateLimitStatus.SUCCESS,
                    data=cached_data,
                    cached=True
                )
        
        # Execute with rate limiting
        for attempt in range(1, self.config.max_retries + 1):
            # Check rate limits
            delay = await self._check_rate_limits()
            if delay > 0:
                logger.warning(f"[kg290] Rate limit reached, waiting {delay:.2f}s (attempt {attempt})")
                await asyncio.sleep(delay)
            
            try:
                # Execute the function
                result = await self._execute_function(func, *args, **kwargs)
                
                # Update request tracking
                await self._track_request()
                
                # Cache the result
                if self.cache:
                    await self.cache.set(cache_key, result, self.config.cache_ttl)
                
                return RateLimitResult(
                    status=RateLimitStatus.SUCCESS,
                    data=result,
                    cached=False,
                    attempts=attempt
                )
                
            except Exception as e:
                error_msg = str(e)
                
                # Check if it's a rate limit error
                if "429" in error_msg or "rate limit" in error_msg.lower():
                    delay = await self._calculate_backoff_delay(attempt)
                    logger.warning(f"[kg290] Rate limit error (attempt {attempt}/{self.config.max_retries}): {error_msg}")
                    
                    if attempt < self.config.max_retries:
                        await asyncio.sleep(delay)
                        continue
                    
                    return RateLimitResult(
                        status=RateLimitStatus.RATE_LIMITED,
                        error_message=error_msg,
                        retry_after=delay,
                        attempts=attempt
                    )
                else:
                    # Non-rate-limit error
                    logger.error(f"[kg290] Function execution error: {error_msg}")
                    return RateLimitResult(
                        status=RateLimitStatus.ERROR,
                        error_message=error_msg,
                        attempts=attempt
                    )
        
        # Max retries exceeded
        return RateLimitResult(
            status=RateLimitStatus.RATE_LIMITED,
            error_message="Maximum retries exceeded",
            attempts=self.config.max_retries
        )
    
    async def _execute_function(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function, handling both sync and async"""
        if asyncio.iscoroutinefunction(func):
            return await func(*args, **kwargs)
        else:
            return func(*args, **kwargs)
    
    async def _check_rate_limits(self) -> float:
        """Check if rate limits are exceeded and return delay needed"""
        async with self._lock:
            current_time = time.time()
            minute_key = "per_minute"
            hour_key = "per_hour"
            
            # Clean old requests
            self._clean_old_requests(minute_key, current_time - 60)
            self._clean_old_requests(hour_key, current_time - 3600)
            
            # Check per-minute limit
            if len(self.request_times[minute_key]) >= self.config.requests_per_minute:
                oldest_request = self.request_times[minute_key][0]
                return 60 - (current_time - oldest_request)
            
            # Check per-hour limit
            if len(self.request_times[hour_key]) >= self.config.requests_per_hour:
                oldest_request = self.request_times[hour_key][0]
                return 3600 - (current_time - oldest_request)
            
            return 0
    
    async def _track_request(self):
        """Track a successful request"""
        async with self._lock:
            current_time = time.time()
            self.request_times["per_minute"].append(current_time)
            self.request_times["per_hour"].append(current_time)
    
    def _clean_old_requests(self, key: str, cutoff_time: float):
        """Remove requests older than cutoff time"""
        self.request_times[key] = [
            req_time for req_time in self.request_times[key] 
            if req_time > cutoff_time
        ]
    
    async def _calculate_backoff_delay(self, attempt: int) -> float:
        """Calculate exponential backoff delay"""
        delay = self.config.base_delay * (2 ** (attempt - 1))
        return min(delay, self.config.max_delay)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get rate limiter statistics"""
        current_time = time.time()
        
        # Clean old requests for accurate stats
        self._clean_old_requests("per_minute", current_time - 60)
        self._clean_old_requests("per_hour", current_time - 3600)
        
        stats = {
            "requests_last_minute": len(self.request_times["per_minute"]),
            "requests_last_hour": len(self.request_times["per_hour"]),
            "minute_limit": self.config.requests_per_minute,
            "hour_limit": self.config.requests_per_hour,
            "cache_enabled": self.config.enable_caching,
            "user": "kg290",
            "timestamp": current_time
        }
        
        if self.cache:
            stats["cache_stats"] = self.cache.get_stats()
        
        return stats


# Factory function
def create_rate_limiter(config: RateLimitConfig = None) -> RateLimiter:
    """Factory function to create rate limiter instance"""
    logger.info("[kg290] Creating RateLimiter instance")
    return RateLimiter(config)


# Utility function to generate context hash for caching
def generate_context_hash(chunks: list, metadata: list = None) -> str:
    """Generate hash from retrieval context for cache key"""
    context_str = "".join(chunks[:3])  # Use top 3 chunks for hash
    if metadata:
        context_str += json.dumps(metadata[:3], sort_keys=True)
    return hashlib.md5(context_str.encode()).hexdigest()[:16]