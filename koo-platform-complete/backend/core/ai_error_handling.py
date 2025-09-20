"""
Enhanced AI Service Error Handling for KOO Platform
Comprehensive error handling with circuit breakers, retry logic, rate limiting,
and service health monitoring for AI services
"""

import asyncio
import time
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, Callable, List, Union
from dataclasses import dataclass, field
from enum import Enum
import logging
from collections import defaultdict, deque
import json
from pathlib import Path

from .exceptions import ExternalServiceError, APIKeyError, RateLimitError
from .config import settings

logger = logging.getLogger(__name__)

class ServiceState(Enum):
    """AI Service states"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    CIRCUIT_OPEN = "circuit_open"
    RATE_LIMITED = "rate_limited"
    MAINTENANCE = "maintenance"

class ErrorType(Enum):
    """Types of AI service errors"""
    API_KEY_ERROR = "api_key_error"
    RATE_LIMIT_ERROR = "rate_limit_error"
    TIMEOUT_ERROR = "timeout_error"
    CONNECTION_ERROR = "connection_error"
    QUOTA_EXCEEDED = "quota_exceeded"
    SERVICE_UNAVAILABLE = "service_unavailable"
    INVALID_REQUEST = "invalid_request"
    UNKNOWN_ERROR = "unknown_error"

@dataclass
class ErrorMetrics:
    """Track error metrics for AI services"""
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    error_counts: Dict[ErrorType, int] = field(default_factory=lambda: defaultdict(int))
    last_error_time: Optional[datetime] = None
    last_success_time: Optional[datetime] = None
    average_response_time: float = 0.0
    response_times: deque = field(default_factory=lambda: deque(maxlen=100))

@dataclass
class CircuitBreakerConfig:
    """Circuit breaker configuration for AI services"""
    failure_threshold: int = 5
    recovery_timeout: int = 60  # seconds
    half_open_max_calls: int = 3
    success_threshold: int = 2  # successes needed to close circuit

@dataclass
class RateLimitConfig:
    """Rate limiting configuration"""
    requests_per_minute: int = 60
    requests_per_hour: int = 1000
    requests_per_day: int = 10000
    cost_per_request: float = 0.001
    daily_budget: float = 10.0

@dataclass
class ServiceCircuitBreaker:
    """Circuit breaker state for AI services"""
    state: ServiceState = ServiceState.HEALTHY
    failure_count: int = 0
    success_count: int = 0
    last_failure_time: Optional[datetime] = None
    last_success_time: Optional[datetime] = None
    next_attempt_time: Optional[datetime] = None
    config: CircuitBreakerConfig = field(default_factory=CircuitBreakerConfig)

@dataclass
class RateLimiter:
    """Rate limiter for AI services"""
    requests_minute: deque = field(default_factory=lambda: deque(maxlen=1000))
    requests_hour: deque = field(default_factory=lambda: deque(maxlen=10000))
    requests_day: deque = field(default_factory=lambda: deque(maxlen=100000))
    daily_cost: float = 0.0
    config: RateLimitConfig = field(default_factory=RateLimitConfig)
    last_reset: datetime = field(default_factory=datetime.now)

class AIServiceErrorHandler:
    """
    Comprehensive error handler for AI services with circuit breakers,
    retry logic, rate limiting, and health monitoring
    """

    def __init__(self):
        self.services: Dict[str, Dict[str, Any]] = {}
        self.error_metrics: Dict[str, ErrorMetrics] = {}
        self.circuit_breakers: Dict[str, ServiceCircuitBreaker] = {}
        self.rate_limiters: Dict[str, RateLimiter] = {}
        
        # Retry configuration from settings
        self.max_retries = settings.AI_MAX_RETRIES
        self.base_delay = settings.AI_RETRY_BASE_DELAY
        self.max_delay = settings.AI_RETRY_MAX_DELAY
        self.backoff_multiplier = settings.AI_RETRY_BACKOFF_MULTIPLIER

        # Health check configuration from settings
        self.health_check_interval = settings.AI_HEALTH_CHECK_INTERVAL
        self.last_health_checks: Dict[str, datetime] = {}
        
        # Load persistent state
        self._load_state()

    def register_service(self, service_name: str, 
                        circuit_breaker_config: Optional[CircuitBreakerConfig] = None,
                        rate_limit_config: Optional[RateLimitConfig] = None) -> None:
        """Register an AI service for monitoring"""
        self.services[service_name] = {
            "registered_at": datetime.now(),
            "last_used": None
        }
        
        self.error_metrics[service_name] = ErrorMetrics()
        
        # Use provided config or create default from settings
        default_circuit_config = CircuitBreakerConfig(
            failure_threshold=settings.AI_CIRCUIT_BREAKER_FAILURE_THRESHOLD,
            recovery_timeout=settings.AI_CIRCUIT_BREAKER_RECOVERY_TIMEOUT,
            success_threshold=settings.AI_CIRCUIT_BREAKER_SUCCESS_THRESHOLD
        )

        default_rate_config = RateLimitConfig(
            requests_per_minute=settings.AI_DEFAULT_REQUESTS_PER_MINUTE,
            requests_per_hour=settings.AI_DEFAULT_REQUESTS_PER_HOUR,
            requests_per_day=settings.AI_DEFAULT_REQUESTS_PER_DAY,
            daily_budget=settings.AI_DEFAULT_DAILY_BUDGET
        )

        self.circuit_breakers[service_name] = ServiceCircuitBreaker(
            config=circuit_breaker_config or default_circuit_config
        )

        self.rate_limiters[service_name] = RateLimiter(
            config=rate_limit_config or default_rate_config
        )
        
        logger.info(f"Registered AI service: {service_name}")

    def _classify_error(self, error: Exception) -> ErrorType:
        """Classify error type for proper handling"""
        error_str = str(error).lower()
        
        if isinstance(error, APIKeyError) or "api key" in error_str or "unauthorized" in error_str:
            return ErrorType.API_KEY_ERROR
        elif isinstance(error, RateLimitError) or "rate limit" in error_str or "too many requests" in error_str:
            return ErrorType.RATE_LIMIT_ERROR
        elif "timeout" in error_str or "timed out" in error_str:
            return ErrorType.TIMEOUT_ERROR
        elif "connection" in error_str or "network" in error_str:
            return ErrorType.CONNECTION_ERROR
        elif "quota" in error_str or "exceeded" in error_str:
            return ErrorType.QUOTA_EXCEEDED
        elif "unavailable" in error_str or "service" in error_str:
            return ErrorType.SERVICE_UNAVAILABLE
        elif "invalid" in error_str or "bad request" in error_str:
            return ErrorType.INVALID_REQUEST
        else:
            return ErrorType.UNKNOWN_ERROR

    def _should_retry(self, error_type: ErrorType, attempt: int) -> bool:
        """Determine if error should be retried"""
        # Don't retry certain error types
        non_retryable = {
            ErrorType.API_KEY_ERROR,
            ErrorType.INVALID_REQUEST,
            ErrorType.QUOTA_EXCEEDED
        }
        
        if error_type in non_retryable:
            return False
        
        # Don't retry if we've exceeded max attempts
        if attempt >= self.max_retries:
            return False
        
        return True

    def _calculate_delay(self, attempt: int, error_type: ErrorType) -> float:
        """Calculate delay for retry with jitter"""
        base_delay = self.base_delay
        
        # Longer delays for rate limiting
        if error_type == ErrorType.RATE_LIMIT_ERROR:
            base_delay = 60.0  # Start with 1 minute for rate limits
        
        delay = min(
            base_delay * (self.backoff_multiplier ** attempt),
            self.max_delay
        )
        
        # Add jitter (±20%)
        import random
        jitter = delay * 0.2 * (random.random() - 0.5)
        return max(0.1, delay + jitter)

    def _check_circuit_breaker(self, service_name: str) -> bool:
        """Check if circuit breaker allows requests"""
        if service_name not in self.circuit_breakers:
            return True
        
        breaker = self.circuit_breakers[service_name]
        now = datetime.now()
        
        if breaker.state == ServiceState.HEALTHY:
            return True
        
        elif breaker.state == ServiceState.CIRCUIT_OPEN:
            # Check if we should attempt recovery
            if (breaker.next_attempt_time and now >= breaker.next_attempt_time):
                breaker.state = ServiceState.DEGRADED
                logger.info(f"Circuit breaker for {service_name} transitioning to DEGRADED for recovery")
                return True
            return False
        
        elif breaker.state == ServiceState.DEGRADED:
            # Allow limited requests for recovery testing
            return True
        
        elif breaker.state == ServiceState.UNHEALTHY:
            # Check if enough time has passed for recovery
            if (breaker.last_failure_time and
                now - breaker.last_failure_time > timedelta(seconds=breaker.config.recovery_timeout)):
                breaker.state = ServiceState.DEGRADED
                logger.info(f"Circuit breaker for {service_name} transitioning to DEGRADED after timeout")
                return True
            return False
        
        return False

    def _check_rate_limit(self, service_name: str) -> bool:
        """Check if request is within rate limits"""
        if service_name not in self.rate_limiters:
            return True
        
        limiter = self.rate_limiters[service_name]
        now = datetime.now()
        current_time = time.time()
        
        # Clean old requests
        minute_ago = current_time - 60
        hour_ago = current_time - 3600
        day_ago = current_time - 86400
        
        # Remove old requests
        while limiter.requests_minute and limiter.requests_minute[0] < minute_ago:
            limiter.requests_minute.popleft()
        while limiter.requests_hour and limiter.requests_hour[0] < hour_ago:
            limiter.requests_hour.popleft()
        while limiter.requests_day and limiter.requests_day[0] < day_ago:
            limiter.requests_day.popleft()
        
        # Reset daily cost if new day
        if now.date() > limiter.last_reset.date():
            limiter.daily_cost = 0.0
            limiter.last_reset = now
        
        # Check limits
        if len(limiter.requests_minute) >= limiter.config.requests_per_minute:
            return False
        if len(limiter.requests_hour) >= limiter.config.requests_per_hour:
            return False
        if len(limiter.requests_day) >= limiter.config.requests_per_day:
            return False
        if limiter.daily_cost >= limiter.config.daily_budget:
            return False
        
        return True

    def _record_request(self, service_name: str, success: bool, 
                       response_time: float, error: Optional[Exception] = None) -> None:
        """Record request metrics"""
        now = datetime.now()
        current_time = time.time()
        
        # Update error metrics
        if service_name in self.error_metrics:
            metrics = self.error_metrics[service_name]
            metrics.total_requests += 1
            metrics.response_times.append(response_time)
            
            if success:
                metrics.successful_requests += 1
                metrics.last_success_time = now
                
                # Update average response time
                if metrics.response_times:
                    metrics.average_response_time = sum(metrics.response_times) / len(metrics.response_times)
            else:
                metrics.failed_requests += 1
                metrics.last_error_time = now
                
                if error:
                    error_type = self._classify_error(error)
                    metrics.error_counts[error_type] += 1
        
        # Update circuit breaker
        if service_name in self.circuit_breakers:
            breaker = self.circuit_breakers[service_name]
            
            if success:
                breaker.success_count += 1
                breaker.failure_count = 0
                breaker.last_success_time = now
                
                # Close circuit if enough successes
                if (breaker.state == ServiceState.DEGRADED and 
                    breaker.success_count >= breaker.config.success_threshold):
                    breaker.state = ServiceState.HEALTHY
                    logger.info(f"Circuit breaker for {service_name} closed - service recovered")
            else:
                breaker.failure_count += 1
                breaker.success_count = 0
                breaker.last_failure_time = now
                
                # Open circuit if too many failures
                if breaker.failure_count >= breaker.config.failure_threshold:
                    breaker.state = ServiceState.CIRCUIT_OPEN
                    breaker.next_attempt_time = now + timedelta(seconds=breaker.config.recovery_timeout)
                    logger.error(f"Circuit breaker for {service_name} opened after {breaker.failure_count} failures")
                else:
                    breaker.state = ServiceState.UNHEALTHY
        
        # Update rate limiter
        if service_name in self.rate_limiters:
            limiter = self.rate_limiters[service_name]
            limiter.requests_minute.append(current_time)
            limiter.requests_hour.append(current_time)
            limiter.requests_day.append(current_time)
            limiter.daily_cost += limiter.config.cost_per_request
        
        # Update service last used
        if service_name in self.services:
            self.services[service_name]["last_used"] = now
        
        # Save state periodically
        self._save_state()

    async def execute_with_error_handling(self, service_name: str,
                                        operation: Callable,
                                        *args, **kwargs) -> Any:
        """
        Execute AI service operation with comprehensive error handling
        """
        if service_name not in self.services:
            raise ExternalServiceError(service_name, "Service not registered")

        # Check circuit breaker
        if not self._check_circuit_breaker(service_name):
            raise ExternalServiceError(service_name, "Service circuit breaker is open")

        # Check rate limits
        if not self._check_rate_limit(service_name):
            raise RateLimitError("Rate limit exceeded for service")

        last_exception = None
        start_time = time.time()

        for attempt in range(self.max_retries + 1):
            try:
                # Execute the operation
                result = await operation(*args, **kwargs)

                # Record success
                response_time = time.time() - start_time
                self._record_request(service_name, True, response_time)

                return result

            except Exception as e:
                last_exception = e
                response_time = time.time() - start_time
                error_type = self._classify_error(e)

                # Record failure
                self._record_request(service_name, False, response_time, e)

                # Check if we should retry
                if not self._should_retry(error_type, attempt):
                    logger.error(f"Non-retryable error for {service_name}: {e}")
                    raise ExternalServiceError(service_name, str(e))

                if attempt < self.max_retries:
                    delay = self._calculate_delay(attempt, error_type)
                    logger.warning(f"AI service {service_name} failed (attempt {attempt + 1}/{self.max_retries + 1}), "
                                 f"retrying in {delay:.2f}s: {e}")
                    await asyncio.sleep(delay)
                    start_time = time.time()  # Reset timer for next attempt
                else:
                    logger.error(f"AI service {service_name} failed after {self.max_retries + 1} attempts")
                    raise ExternalServiceError(service_name, f"Service failed after all retries: {str(e)}")

        # This should never be reached
        if last_exception:
            raise ExternalServiceError(service_name, str(last_exception))

    def get_service_status(self, service_name: str) -> Dict[str, Any]:
        """Get comprehensive status for a service"""
        if service_name not in self.services:
            return {"error": "Service not registered"}

        metrics = self.error_metrics.get(service_name, ErrorMetrics())
        breaker = self.circuit_breakers.get(service_name, ServiceCircuitBreaker())
        limiter = self.rate_limiters.get(service_name, RateLimiter())

        # Calculate success rate
        success_rate = 0.0
        if metrics.total_requests > 0:
            success_rate = metrics.successful_requests / metrics.total_requests

        # Get current rate limit status
        now = time.time()
        minute_requests = sum(1 for t in limiter.requests_minute if t > now - 60)
        hour_requests = sum(1 for t in limiter.requests_hour if t > now - 3600)
        day_requests = sum(1 for t in limiter.requests_day if t > now - 86400)

        return {
            "service_name": service_name,
            "state": breaker.state.value,
            "health": {
                "circuit_breaker_state": breaker.state.value,
                "failure_count": breaker.failure_count,
                "success_count": breaker.success_count,
                "last_failure": breaker.last_failure_time.isoformat() if breaker.last_failure_time else None,
                "last_success": breaker.last_success_time.isoformat() if breaker.last_success_time else None,
                "next_attempt": breaker.next_attempt_time.isoformat() if breaker.next_attempt_time else None,
            },
            "metrics": {
                "total_requests": metrics.total_requests,
                "successful_requests": metrics.successful_requests,
                "failed_requests": metrics.failed_requests,
                "success_rate": round(success_rate, 4),
                "average_response_time": round(metrics.average_response_time, 4),
                "error_counts": {error_type.value: count for error_type, count in metrics.error_counts.items()},
                "last_error": metrics.last_error_time.isoformat() if metrics.last_error_time else None,
                "last_success": metrics.last_success_time.isoformat() if metrics.last_success_time else None,
            },
            "rate_limits": {
                "requests_per_minute": f"{minute_requests}/{limiter.config.requests_per_minute}",
                "requests_per_hour": f"{hour_requests}/{limiter.config.requests_per_hour}",
                "requests_per_day": f"{day_requests}/{limiter.config.requests_per_day}",
                "daily_cost": f"${limiter.daily_cost:.4f}/${limiter.config.daily_budget}",
                "cost_per_request": limiter.config.cost_per_request,
            },
            "configuration": {
                "circuit_breaker": {
                    "failure_threshold": breaker.config.failure_threshold,
                    "recovery_timeout": breaker.config.recovery_timeout,
                    "success_threshold": breaker.config.success_threshold,
                },
                "rate_limits": {
                    "requests_per_minute": limiter.config.requests_per_minute,
                    "requests_per_hour": limiter.config.requests_per_hour,
                    "requests_per_day": limiter.config.requests_per_day,
                    "daily_budget": limiter.config.daily_budget,
                }
            },
            "last_updated": datetime.now().isoformat()
        }

    def get_all_services_status(self) -> Dict[str, Any]:
        """Get status for all registered services"""
        return {
            service_name: self.get_service_status(service_name)
            for service_name in self.services.keys()
        }

    def reset_circuit_breaker(self, service_name: str) -> bool:
        """Manually reset circuit breaker for a service"""
        if service_name not in self.circuit_breakers:
            return False

        breaker = self.circuit_breakers[service_name]
        breaker.state = ServiceState.HEALTHY
        breaker.failure_count = 0
        breaker.success_count = 0
        breaker.last_failure_time = None
        breaker.next_attempt_time = None

        logger.info(f"Circuit breaker for {service_name} manually reset")
        self._save_state()
        return True

    def reset_rate_limiter(self, service_name: str) -> bool:
        """Manually reset rate limiter for a service"""
        if service_name not in self.rate_limiters:
            return False

        limiter = self.rate_limiters[service_name]
        limiter.requests_minute.clear()
        limiter.requests_hour.clear()
        limiter.requests_day.clear()
        limiter.daily_cost = 0.0
        limiter.last_reset = datetime.now()

        logger.info(f"Rate limiter for {service_name} manually reset")
        self._save_state()
        return True

    def _load_state(self) -> None:
        """Load persistent state from storage"""
        try:
            state_file = Path("data/ai_error_handler_state.json")
            if state_file.exists():
                with open(state_file, 'r') as f:
                    data = json.load(f)

                # Load error metrics
                for service_name, metrics_data in data.get("error_metrics", {}).items():
                    metrics = ErrorMetrics()
                    metrics.total_requests = metrics_data.get("total_requests", 0)
                    metrics.successful_requests = metrics_data.get("successful_requests", 0)
                    metrics.failed_requests = metrics_data.get("failed_requests", 0)
                    metrics.average_response_time = metrics_data.get("average_response_time", 0.0)

                    # Load error counts
                    for error_type_str, count in metrics_data.get("error_counts", {}).items():
                        try:
                            error_type = ErrorType(error_type_str)
                            metrics.error_counts[error_type] = count
                        except ValueError:
                            pass  # Skip unknown error types

                    # Load timestamps
                    if metrics_data.get("last_error_time"):
                        metrics.last_error_time = datetime.fromisoformat(metrics_data["last_error_time"])
                    if metrics_data.get("last_success_time"):
                        metrics.last_success_time = datetime.fromisoformat(metrics_data["last_success_time"])

                    self.error_metrics[service_name] = metrics

                # Load circuit breaker states
                for service_name, breaker_data in data.get("circuit_breakers", {}).items():
                    breaker = ServiceCircuitBreaker()
                    try:
                        breaker.state = ServiceState(breaker_data.get("state", "healthy"))
                    except ValueError:
                        breaker.state = ServiceState.HEALTHY

                    breaker.failure_count = breaker_data.get("failure_count", 0)
                    breaker.success_count = breaker_data.get("success_count", 0)

                    if breaker_data.get("last_failure_time"):
                        breaker.last_failure_time = datetime.fromisoformat(breaker_data["last_failure_time"])
                    if breaker_data.get("last_success_time"):
                        breaker.last_success_time = datetime.fromisoformat(breaker_data["last_success_time"])
                    if breaker_data.get("next_attempt_time"):
                        breaker.next_attempt_time = datetime.fromisoformat(breaker_data["next_attempt_time"])

                    self.circuit_breakers[service_name] = breaker

                logger.info("AI error handler state loaded successfully")

        except Exception as e:
            logger.warning(f"Failed to load AI error handler state: {e}")

    def _save_state(self) -> None:
        """Save persistent state to storage"""
        try:
            Path("data").mkdir(exist_ok=True)
            state_file = Path("data/ai_error_handler_state.json")

            data = {
                "error_metrics": {},
                "circuit_breakers": {},
                "last_saved": datetime.now().isoformat()
            }

            # Save error metrics
            for service_name, metrics in self.error_metrics.items():
                data["error_metrics"][service_name] = {
                    "total_requests": metrics.total_requests,
                    "successful_requests": metrics.successful_requests,
                    "failed_requests": metrics.failed_requests,
                    "average_response_time": metrics.average_response_time,
                    "error_counts": {error_type.value: count for error_type, count in metrics.error_counts.items()},
                    "last_error_time": metrics.last_error_time.isoformat() if metrics.last_error_time else None,
                    "last_success_time": metrics.last_success_time.isoformat() if metrics.last_success_time else None,
                }

            # Save circuit breaker states
            for service_name, breaker in self.circuit_breakers.items():
                data["circuit_breakers"][service_name] = {
                    "state": breaker.state.value,
                    "failure_count": breaker.failure_count,
                    "success_count": breaker.success_count,
                    "last_failure_time": breaker.last_failure_time.isoformat() if breaker.last_failure_time else None,
                    "last_success_time": breaker.last_success_time.isoformat() if breaker.last_success_time else None,
                    "next_attempt_time": breaker.next_attempt_time.isoformat() if breaker.next_attempt_time else None,
                }

            with open(state_file, 'w') as f:
                json.dump(data, f, indent=2)

        except Exception as e:
            logger.error(f"Failed to save AI error handler state: {e}")

# Global AI error handler instance
ai_error_handler = AIServiceErrorHandler()

# Convenience functions
async def execute_ai_operation(service_name: str, operation: Callable, *args, **kwargs) -> Any:
    """Execute AI operation with comprehensive error handling"""
    return await ai_error_handler.execute_with_error_handling(service_name, operation, *args, **kwargs)

def get_ai_service_status(service_name: str) -> Dict[str, Any]:
    """Get status for specific AI service"""
    return ai_error_handler.get_service_status(service_name)

def get_all_ai_services_status() -> Dict[str, Any]:
    """Get status for all AI services"""
    return ai_error_handler.get_all_services_status()
