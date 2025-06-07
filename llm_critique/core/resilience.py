"""
Production resilience patterns for LLM Critique.

This module provides retry logic, circuit breakers, timeouts, and graceful degradation
for robust production operation.
"""

import asyncio
import time
import logging
from typing import Any, Callable, Optional, Union, TypeVar, Awaitable
from functools import wraps
from dataclasses import dataclass
from enum import Enum
import traceback

logger = logging.getLogger(__name__)

T = TypeVar('T')

class CircuitBreakerState(Enum):
    CLOSED = "closed"
    OPEN = "open" 
    HALF_OPEN = "half_open"

@dataclass
class RetryConfig:
    """Configuration for retry behavior."""
    max_attempts: int = 3
    initial_delay: float = 1.0
    max_delay: float = 60.0
    exponential_base: float = 2.0
    jitter: bool = True

@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker."""
    failure_threshold: int = 5
    recovery_timeout: float = 60.0
    expected_exceptions: tuple = (Exception,)

class CircuitBreakerError(Exception):
    """Raised when circuit breaker is open."""
    pass

class TimeoutError(Exception):
    """Raised when operation times out."""
    pass

class CircuitBreaker:
    """Circuit breaker implementation for fault tolerance."""
    
    def __init__(self, config: CircuitBreakerConfig):
        self.config = config
        self.state = CircuitBreakerState.CLOSED
        self.failure_count = 0
        self.last_failure_time: Optional[float] = None
        self.success_count = 0
        
    def can_execute(self) -> bool:
        """Check if circuit breaker allows execution."""
        if self.state == CircuitBreakerState.CLOSED:
            return True
        elif self.state == CircuitBreakerState.OPEN:
            if (time.time() - (self.last_failure_time or 0)) > self.config.recovery_timeout:
                self.state = CircuitBreakerState.HALF_OPEN
                self.success_count = 0
                logger.info("Circuit breaker transitioning to half-open")
                return True
            return False
        else:  # HALF_OPEN
            return True
    
    def record_success(self):
        """Record a successful operation."""
        if self.state == CircuitBreakerState.HALF_OPEN:
            self.success_count += 1
            if self.success_count >= 3:  # Require 3 successes to close
                self.state = CircuitBreakerState.CLOSED
                self.failure_count = 0
                logger.info("Circuit breaker closed after successful recovery")
        elif self.state == CircuitBreakerState.CLOSED:
            self.failure_count = max(0, self.failure_count - 1)
    
    def record_failure(self, exception: Exception):
        """Record a failed operation."""
        if isinstance(exception, self.config.expected_exceptions):
            self.failure_count += 1
            self.last_failure_time = time.time()
            
            if self.failure_count >= self.config.failure_threshold:
                self.state = CircuitBreakerState.OPEN
                logger.warning(f"Circuit breaker opened after {self.failure_count} failures")
        
def with_retry(config: RetryConfig = None):
    """Decorator to add retry logic to functions."""
    if config is None:
        config = RetryConfig()
    
    def decorator(func: Callable[..., Union[T, Awaitable[T]]]) -> Callable[..., Union[T, Awaitable[T]]]:
        if asyncio.iscoroutinefunction(func):
            @wraps(func)
            async def async_wrapper(*args, **kwargs) -> T:
                last_exception = None
                
                for attempt in range(config.max_attempts):
                    try:
                        result = await func(*args, **kwargs)
                        if attempt > 0:
                            logger.info(f"Operation succeeded on attempt {attempt + 1}")
                        return result
                    except Exception as e:
                        last_exception = e
                        
                        if attempt == config.max_attempts - 1:
                            logger.error(f"Operation failed after {config.max_attempts} attempts: {e}")
                            break
                        
                        delay = min(
                            config.initial_delay * (config.exponential_base ** attempt),
                            config.max_delay
                        )
                        
                        if config.jitter:
                            import random
                            delay *= (0.5 + random.random() * 0.5)
                        
                        logger.warning(f"Attempt {attempt + 1} failed: {e}. Retrying in {delay:.2f}s")
                        await asyncio.sleep(delay)
                
                raise last_exception
            return async_wrapper
        else:
            @wraps(func)
            def sync_wrapper(*args, **kwargs) -> T:
                last_exception = None
                
                for attempt in range(config.max_attempts):
                    try:
                        result = func(*args, **kwargs)
                        if attempt > 0:
                            logger.info(f"Operation succeeded on attempt {attempt + 1}")
                        return result
                    except Exception as e:
                        last_exception = e
                        
                        if attempt == config.max_attempts - 1:
                            logger.error(f"Operation failed after {config.max_attempts} attempts: {e}")
                            break
                        
                        delay = min(
                            config.initial_delay * (config.exponential_base ** attempt),
                            config.max_delay
                        )
                        
                        if config.jitter:
                            import random
                            delay *= (0.5 + random.random() * 0.5)
                        
                        logger.warning(f"Attempt {attempt + 1} failed: {e}. Retrying in {delay:.2f}s")
                        time.sleep(delay)
                
                raise last_exception
            return sync_wrapper
    
    return decorator

def with_timeout(timeout_seconds: float):
    """Decorator to add timeout to async functions."""
    def decorator(func: Callable[..., Awaitable[T]]) -> Callable[..., Awaitable[T]]:
        @wraps(func)
        async def wrapper(*args, **kwargs) -> T:
            try:
                return await asyncio.wait_for(func(*args, **kwargs), timeout=timeout_seconds)
            except asyncio.TimeoutError:
                logger.error(f"Operation timed out after {timeout_seconds}s")
                raise TimeoutError(f"Operation timed out after {timeout_seconds}s")
        return wrapper
    return decorator

def with_circuit_breaker(circuit_breaker: CircuitBreaker):
    """Decorator to add circuit breaker protection."""
    def decorator(func: Callable[..., Union[T, Awaitable[T]]]) -> Callable[..., Union[T, Awaitable[T]]]:
        if asyncio.iscoroutinefunction(func):
            @wraps(func)
            async def async_wrapper(*args, **kwargs) -> T:
                if not circuit_breaker.can_execute():
                    raise CircuitBreakerError("Circuit breaker is open")
                
                try:
                    result = await func(*args, **kwargs)
                    circuit_breaker.record_success()
                    return result
                except Exception as e:
                    circuit_breaker.record_failure(e)
                    raise
            return async_wrapper
        else:
            @wraps(func)
            def sync_wrapper(*args, **kwargs) -> T:
                if not circuit_breaker.can_execute():
                    raise CircuitBreakerError("Circuit breaker is open")
                
                try:
                    result = func(*args, **kwargs)
                    circuit_breaker.record_success()
                    return result
                except Exception as e:
                    circuit_breaker.record_failure(e)
                    raise
            return sync_wrapper
    
    return decorator

class GracefulErrorHandler:
    """Handles errors gracefully with fallback responses."""
    
    @staticmethod
    def handle_api_error(e: Exception, context: str = "") -> dict:
        """Handle API errors with structured response."""
        error_id = f"err_{int(time.time())}"
        
        error_response = {
            "success": False,
            "error_id": error_id,
            "error_type": type(e).__name__,
            "message": "An error occurred during processing",
            "context": context,
            "timestamp": time.time(),
            "recoverable": True
        }
        
        # Log detailed error for debugging
        logger.error(f"Error {error_id}: {str(e)}", extra={
            "error_id": error_id,
            "exception_type": type(e).__name__,
            "context": context,
            "traceback": traceback.format_exc()
        })
        
        # Categorize error types
        if "timeout" in str(e).lower():
            error_response["message"] = "Request timed out. Please try again."
            error_response["error_category"] = "timeout"
        elif "rate limit" in str(e).lower():
            error_response["message"] = "Rate limit exceeded. Please wait before retrying."
            error_response["error_category"] = "rate_limit"
            error_response["recoverable"] = True
        elif "authentication" in str(e).lower() or "api key" in str(e).lower():
            error_response["message"] = "Authentication failed. Please check API configuration."
            error_response["error_category"] = "auth"
            error_response["recoverable"] = False
        elif isinstance(e, CircuitBreakerError):
            error_response["message"] = "Service temporarily unavailable. Please try again later."
            error_response["error_category"] = "circuit_breaker"
            error_response["recoverable"] = True
        else:
            error_response["error_category"] = "unknown"
        
        return error_response

# Global circuit breakers for different services
api_circuit_breaker = CircuitBreaker(CircuitBreakerConfig(
    failure_threshold=5,
    recovery_timeout=60.0,
    expected_exceptions=(ConnectionError, TimeoutError, Exception)
))

file_circuit_breaker = CircuitBreaker(CircuitBreakerConfig(
    failure_threshold=3,
    recovery_timeout=30.0,
    expected_exceptions=(IOError, OSError, MemoryError)
)) 