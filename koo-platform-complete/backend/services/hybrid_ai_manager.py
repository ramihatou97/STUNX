"""
KOO Platform Hybrid AI Service Manager
Seamlessly switches between API calls and browser automation
Optimizes for cost, availability, and performance
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union, Literal
from dataclasses import dataclass, asdict
from enum import Enum
import aiohttp
from playwright.async_api import async_playwright, Browser, Page, Playwright
import os
from pathlib import Path

from ..core.config import settings
from ..core.exceptions import ExternalServiceError, APIKeyError
from ..core.api_key_manager import api_key_manager, APIProvider
from ..core.ai_error_handling import (
    ai_error_handler,
    execute_ai_operation,
    CircuitBreakerConfig,
    RateLimitConfig
)

logger = logging.getLogger(__name__)

class AccessMethod(Enum):
    """Access method for AI services"""
    API = "api"
    WEB = "web"
    HYBRID = "hybrid"

class Priority(Enum):
    """Service priority levels"""
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"

@dataclass
class UsageStats:
    """Track usage statistics for cost optimization"""
    api_calls: int = 0
    web_calls: int = 0
    api_cost: float = 0.0
    tokens_used: int = 0
    last_reset: datetime = None
    daily_budget_used: float = 0.0

@dataclass
class ServiceConfig:
    """Configuration for each AI service"""
    provider: APIProvider
    access_method: AccessMethod
    api_available: bool = False
    web_available: bool = False
    daily_budget: float = 10.0  # USD
    cost_per_1k_tokens: float = 0.001
    max_tokens_per_request: int = 4000
    rate_limit_per_minute: int = 60
    web_login_required: bool = True
    web_base_url: str = ""
    web_selectors: Dict[str, str] = None

class HybridAIManager:
    """Manages hybrid access to AI services"""

    def __init__(self):
        self.playwright: Optional[Playwright] = None
        self.browser: Optional[Browser] = None
        self.pages: Dict[str, Page] = {}
        self.usage_stats: Dict[str, UsageStats] = {}
        self.service_configs: Dict[str, ServiceConfig] = {}
        self.session_active = False

        # Initialize service configurations
        self._initialize_service_configs()

        # Load usage stats
        self._load_usage_stats()

        # Register services with error handler
        self._register_services_with_error_handler()

    def _initialize_service_configs(self):
        """Initialize configurations for all AI services"""

        self.service_configs = {
            "gemini": ServiceConfig(
                provider=APIProvider.GEMINI,
                access_method=AccessMethod.HYBRID,
                daily_budget=15.0,
                cost_per_1k_tokens=0.001,
                max_tokens_per_request=8000,
                web_base_url="https://gemini.google.com/app",
                web_selectors={
                    "input": 'div[contenteditable="true"]',
                    "send_button": 'button[aria-label*="Send"]',
                    "response": 'div[data-test-id="conversation-turn-2"] div[class*="markdown"]',
                    "new_chat": 'button[aria-label*="New chat"]'
                }
            ),

            "claude": ServiceConfig(
                provider=APIProvider.CLAUDE,
                access_method=AccessMethod.HYBRID,
                daily_budget=20.0,
                cost_per_1k_tokens=0.015,
                max_tokens_per_request=4000,
                web_base_url="https://claude.ai/chats",
                web_selectors={
                    "input": 'div[contenteditable="true"]',
                    "send_button": 'button[aria-label="Send Message"]',
                    "response": 'div[class*="font-claude-message"]',
                    "new_chat": 'button:has-text("Start new chat")'
                }
            ),

            "perplexity": ServiceConfig(
                provider=APIProvider.PERPLEXITY,
                access_method=AccessMethod.API,
                daily_budget=10.0,
                cost_per_1k_tokens=0.001,
                web_base_url="https://www.perplexity.ai",
                web_selectors={
                    "input": 'textarea[placeholder*="Ask anything"]',
                    "send_button": 'button[aria-label="Submit"]',
                    "response": 'div[class*="prose"]'
                }
            )
        }

    def _register_services_with_error_handler(self):
        """Register AI services with the error handler"""
        for service_name, config in self.service_configs.items():
            # Configure circuit breaker based on service characteristics
            circuit_config = CircuitBreakerConfig(
                failure_threshold=5,
                recovery_timeout=60,
                success_threshold=2
            )

            # Configure rate limiting based on service config
            rate_config = RateLimitConfig(
                requests_per_minute=config.rate_limit_per_minute,
                requests_per_hour=config.rate_limit_per_minute * 60,
                requests_per_day=config.rate_limit_per_minute * 60 * 24,
                cost_per_request=config.cost_per_1k_tokens,
                daily_budget=config.daily_budget
            )

            ai_error_handler.register_service(
                service_name=service_name,
                circuit_breaker_config=circuit_config,
                rate_limit_config=rate_config
            )

            logger.info(f"Registered {service_name} with AI error handler")

    def _load_usage_stats(self):
        """Load usage statistics from storage"""
        stats_file = Path("data/usage_stats.json")

        if stats_file.exists():
            try:
                with open(stats_file, 'r') as f:
                    data = json.load(f)
                    for service, stats_data in data.items():
                        self.usage_stats[service] = UsageStats(**stats_data)
            except Exception as e:
                logger.error(f"Failed to load usage stats: {e}")

        # Initialize missing stats
        for service in self.service_configs.keys():
            if service not in self.usage_stats:
                self.usage_stats[service] = UsageStats(last_reset=datetime.now())

    def _save_usage_stats(self):
        """Save usage statistics to storage"""
        try:
            os.makedirs("data", exist_ok=True)
            stats_file = Path("data/usage_stats.json")

            data = {}
            for service, stats in self.usage_stats.items():
                data[service] = asdict(stats)
                # Convert datetime to string
                if stats.last_reset:
                    data[service]['last_reset'] = stats.last_reset.isoformat()

            with open(stats_file, 'w') as f:
                json.dump(data, f, indent=2)

        except Exception as e:
            logger.error(f"Failed to save usage stats: {e}")

    async def initialize_browser(self):
        """Initialize Playwright browser for web automation"""
        if not self.session_active:
            try:
                self.playwright = await async_playwright().start()
                self.browser = await self.playwright.chromium.launch(
                    headless=True,  # Set to False for debugging
                    args=[
                        '--no-sandbox',
                        '--disable-blink-features=AutomationControlled',
                        '--disable-extensions'
                    ]
                )
                self.session_active = True
                logger.info("Browser session initialized successfully")

            except Exception as e:
                logger.error(f"Failed to initialize browser: {e}")
                raise ExternalServiceError("browser", "Failed to initialize browser session")

    async def close_browser(self):
        """Close browser session and cleanup"""
        try:
            if self.pages:
                for page in self.pages.values():
                    await page.close()
                self.pages.clear()

            if self.browser:
                await self.browser.close()
                self.browser = None

            if self.playwright:
                await self.playwright.stop()
                self.playwright = None

            self.session_active = False
            logger.info("Browser session closed")

        except Exception as e:
            logger.error(f"Error closing browser: {e}")

    async def get_page(self, service: str) -> Page:
        """Get or create a page for the service"""
        if service not in self.pages:
            if not self.browser:
                await self.initialize_browser()

            page = await self.browser.new_page()

            # Set user agent to avoid detection
            await page.set_extra_http_headers({
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            })

            self.pages[service] = page

        return self.pages[service]

    def _should_use_api(self, service: str, estimated_tokens: int = 1000) -> bool:
        """Determine if API should be used based on cost and availability"""
        config = self.service_configs[service]
        stats = self.usage_stats[service]

        # Check if API is available
        if not config.api_available:
            return False

        # Check daily budget
        estimated_cost = (estimated_tokens / 1000) * config.cost_per_1k_tokens
        if stats.daily_budget_used + estimated_cost > config.daily_budget:
            logger.info(f"{service}: Daily budget exceeded, using web interface")
            return False

        # Use API if within budget and available
        return True

    async def _call_api(self, service: str, prompt: str, **kwargs) -> str:
        """Make API call to service"""
        config = self.service_configs[service]

        try:
            if service == "gemini":
                return await self._call_gemini_api(prompt, **kwargs)
            elif service == "claude":
                return await self._call_claude_api(prompt, **kwargs)
            elif service == "perplexity":
                return await self._call_perplexity_api(prompt, **kwargs)
            else:
                raise ExternalServiceError(service, "API not implemented")

        except Exception as e:
            logger.error(f"API call failed for {service}: {e}")
            # Fallback to web if API fails
            if config.web_available:
                return await self._call_web(service, prompt, **kwargs)
            raise

    async def _call_gemini_api(self, prompt: str, **kwargs) -> str:
        """Call Gemini API"""
        api_key = api_key_manager.get_api_key(APIProvider.GEMINI)
        if not api_key:
            raise APIKeyError("gemini", "No API key configured")

        headers = {
            'Content-Type': 'application/json',
            'x-goog-api-key': api_key
        }

        data = {
            "contents": [{
                "parts": [{"text": prompt}]
            }],
            "generationConfig": {
                "maxOutputTokens": kwargs.get("max_tokens", 2000),
                "temperature": kwargs.get("temperature", 0.7)
            }
        }

        async with aiohttp.ClientSession() as session:
            async with session.post(
                "https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent",
                headers=headers,
                json=data
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    return result['candidates'][0]['content']['parts'][0]['text']
                else:
                    error_text = await response.text()
                    raise ExternalServiceError("gemini", f"API error: {error_text}")

    async def _call_claude_api(self, prompt: str, **kwargs) -> str:
        """Call Claude API"""
        api_key = api_key_manager.get_api_key(APIProvider.CLAUDE)
        if not api_key:
            raise APIKeyError("claude", "No API key configured")

        headers = {
            'Content-Type': 'application/json',
            'x-api-key': api_key,
            'anthropic-version': '2023-06-01'
        }

        data = {
            "model": "claude-3-opus-20240229",
            "max_tokens": kwargs.get("max_tokens", 2000),
            "temperature": kwargs.get("temperature", 0.7),
            "messages": [
                {"role": "user", "content": prompt}
            ]
        }

        async with aiohttp.ClientSession() as session:
            async with session.post(
                "https://api.anthropic.com/v1/messages",
                headers=headers,
                json=data
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    return result['content'][0]['text']
                else:
                    error_text = await response.text()
                    raise ExternalServiceError("claude", f"API error: {error_text}")

    async def _call_perplexity_api(self, prompt: str, **kwargs) -> str:
        """Call Perplexity API"""
        api_key = api_key_manager.get_api_key(APIProvider.PERPLEXITY)
        if not api_key:
            raise APIKeyError("perplexity", "No API key configured")

        headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {api_key}'
        }

        data = {
            "model": "llama-3.1-sonar-large-128k-online",
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "max_tokens": kwargs.get("max_tokens", 2000),
            "temperature": kwargs.get("temperature", 0.7)
        }

        async with aiohttp.ClientSession() as session:
            async with session.post(
                "https://api.perplexity.ai/chat/completions",
                headers=headers,
                json=data
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    return result['choices'][0]['message']['content']
                else:
                    error_text = await response.text()
                    raise ExternalServiceError("perplexity", f"API error: {error_text}")

    async def _call_web(self, service: str, prompt: str, **kwargs) -> str:
        """Make web interface call"""
        config = self.service_configs[service]

        try:
            if service == "gemini":
                return await self._call_gemini_web(prompt, **kwargs)
            elif service == "claude":
                return await self._call_claude_web(prompt, **kwargs)
            elif service == "perplexity":
                return await self._call_perplexity_web(prompt, **kwargs)
            else:
                raise ExternalServiceError(service, "Web interface not implemented")

        except Exception as e:
            logger.error(f"Web call failed for {service}: {e}")
            raise ExternalServiceError(service, f"Web interface error: {str(e)}")

    async def _call_gemini_web(self, prompt: str, **kwargs) -> str:
        """Call Gemini via web interface"""
        page = await self.get_page("gemini")
        config = self.service_configs["gemini"]

        try:
            # Navigate to Gemini
            await page.goto(config.web_base_url, wait_until="networkidle")

            # Wait for input field and enter prompt
            await page.wait_for_selector(config.web_selectors["input"], timeout=10000)
            await page.fill(config.web_selectors["input"], prompt)

            # Click send button
            await page.click(config.web_selectors["send_button"])

            # Wait for response
            await page.wait_for_selector(config.web_selectors["response"], timeout=30000)

            # Extract response text
            response_element = await page.query_selector(config.web_selectors["response"])
            response_text = await response_element.inner_text()

            # Update usage stats
            self._update_usage_stats("gemini", web_call=True)

            return response_text.strip()

        except Exception as e:
            logger.error(f"Gemini web automation failed: {e}")
            raise ExternalServiceError("gemini", f"Web automation failed: {str(e)}")

    async def _call_claude_web(self, prompt: str, **kwargs) -> str:
        """Call Claude via web interface"""
        page = await self.get_page("claude")
        config = self.service_configs["claude"]

        try:
            # Navigate to Claude
            await page.goto(config.web_base_url, wait_until="networkidle")

            # Wait for input field and enter prompt
            await page.wait_for_selector(config.web_selectors["input"], timeout=10000)
            await page.fill(config.web_selectors["input"], prompt)

            # Click send button
            await page.click(config.web_selectors["send_button"])

            # Wait for response
            await page.wait_for_selector(config.web_selectors["response"], timeout=30000)

            # Extract response text
            response_element = await page.query_selector(config.web_selectors["response"])
            response_text = await response_element.inner_text()

            # Update usage stats
            self._update_usage_stats("claude", web_call=True)

            return response_text.strip()

        except Exception as e:
            logger.error(f"Claude web automation failed: {e}")
            raise ExternalServiceError("claude", f"Web automation failed: {str(e)}")

    async def _call_perplexity_web(self, prompt: str, **kwargs) -> str:
        """Call Perplexity via web interface"""
        page = await self.get_page("perplexity")
        config = self.service_configs["perplexity"]

        try:
            # Navigate to Perplexity
            await page.goto(config.web_base_url, wait_until="networkidle")

            # Wait for input field and enter prompt
            await page.wait_for_selector(config.web_selectors["input"], timeout=10000)
            await page.fill(config.web_selectors["input"], prompt)

            # Press Enter or click send
            await page.press(config.web_selectors["input"], "Enter")

            # Wait for response
            await page.wait_for_selector(config.web_selectors["response"], timeout=30000)

            # Extract response text
            response_element = await page.query_selector(config.web_selectors["response"])
            response_text = await response_element.inner_text()

            # Update usage stats
            self._update_usage_stats("perplexity", web_call=True)

            return response_text.strip()

        except Exception as e:
            logger.error(f"Perplexity web automation failed: {e}")
            raise ExternalServiceError("perplexity", f"Web automation failed: {str(e)}")

    def _update_usage_stats(self, service: str, api_call: bool = False, web_call: bool = False,
                           tokens: int = 0, cost: float = 0.0):
        """Update usage statistics"""
        stats = self.usage_stats[service]

        if api_call:
            stats.api_calls += 1
            stats.tokens_used += tokens
            stats.api_cost += cost
            stats.daily_budget_used += cost

        if web_call:
            stats.web_calls += 1

        self._save_usage_stats()

    async def query(self, service: str, prompt: str, **kwargs) -> str:
        """Enhanced main query method with comprehensive error handling"""

        # Check if service is configured
        if service not in self.service_configs:
            raise ExternalServiceError(service, "Service not configured")

        config = self.service_configs[service]
        estimated_tokens = len(prompt.split()) * 1.3  # Rough estimation

        # Determine access method
        use_api = self._should_use_api(service, estimated_tokens)

        # Define the operation to execute with error handling
        async def execute_query():
            if use_api and config.api_available:
                logger.info(f"Using API for {service}")
                response = await self._call_api(service, prompt, **kwargs)

                # Update usage stats
                cost = (estimated_tokens / 1000) * config.cost_per_1k_tokens
                self._update_usage_stats(service, api_call=True, tokens=int(estimated_tokens), cost=cost)

                return response

            elif config.web_available:
                logger.info(f"Using web interface for {service}")
                response = await self._call_web(service, prompt, **kwargs)

                # Update usage stats for web calls
                self._update_usage_stats(service, web_call=True)

                return response

            else:
                raise ExternalServiceError(service, "No access method available")

        # Execute with comprehensive error handling
        try:
            return await execute_ai_operation(service, execute_query)
        except Exception as e:
            logger.error(f"Query failed for {service} after error handling: {e}")
            raise

    async def batch_query(self, queries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Process multiple queries efficiently"""
        results = []

        # Group queries by service for optimization
        service_groups = {}
        for i, query in enumerate(queries):
            service = query.get("service")
            if service not in service_groups:
                service_groups[service] = []
            service_groups[service].append((i, query))

        # Process each service group
        for service, service_queries in service_groups.items():
            for i, query in service_queries:
                try:
                    response = await self.query(
                        service=query["service"],
                        prompt=query["prompt"],
                        **query.get("kwargs", {})
                    )

                    results.append({
                        "index": i,
                        "service": service,
                        "response": response,
                        "success": True
                    })

                except Exception as e:
                    results.append({
                        "index": i,
                        "service": service,
                        "error": str(e),
                        "success": False
                    })

        # Sort results by original index
        results.sort(key=lambda x: x["index"])
        return results

    def get_usage_stats(self) -> Dict[str, Any]:
        """Get current usage statistics"""
        return {
            service: {
                "api_calls": stats.api_calls,
                "web_calls": stats.web_calls,
                "api_cost": stats.api_cost,
                "tokens_used": stats.tokens_used,
                "daily_budget_used": stats.daily_budget_used,
                "daily_budget": self.service_configs[service].daily_budget
            }
            for service, stats in self.usage_stats.items()
        }

    def get_enhanced_service_status(self, service: str) -> Dict[str, Any]:
        """Get comprehensive service status including error handling metrics"""
        if service not in self.service_configs:
            return {"error": "Service not configured"}

        # Get basic usage stats
        basic_stats = self.get_usage_stats().get(service, {})

        # Get error handler status
        error_handler_status = ai_error_handler.get_service_status(service)

        # Get service configuration
        config = self.service_configs[service]

        # Combine all information
        return {
            "service_name": service,
            "configuration": {
                "provider": config.provider.value,
                "access_method": config.access_method.value,
                "api_available": config.api_available,
                "web_available": config.web_available,
                "daily_budget": config.daily_budget,
                "cost_per_1k_tokens": config.cost_per_1k_tokens,
                "max_tokens_per_request": config.max_tokens_per_request,
                "rate_limit_per_minute": config.rate_limit_per_minute,
            },
            "usage_statistics": basic_stats,
            "error_handling": error_handler_status,
            "browser_session": {
                "active": self.session_active,
                "page_available": service in self.pages
            },
            "last_updated": datetime.now().isoformat()
        }

    def get_all_services_status(self) -> Dict[str, Any]:
        """Get comprehensive status for all services"""
        services_status = {}

        for service_name in self.service_configs.keys():
            services_status[service_name] = self.get_enhanced_service_status(service_name)

        # Add overall system status
        overall_status = {
            "total_services": len(self.service_configs),
            "browser_session_active": self.session_active,
            "services": services_status,
            "system_health": self._get_system_health(),
            "timestamp": datetime.now().isoformat()
        }

        return overall_status

    def _get_system_health(self) -> Dict[str, Any]:
        """Get overall system health status"""
        all_ai_status = ai_error_handler.get_all_services_status()

        healthy_services = 0
        total_services = len(self.service_configs)
        circuit_breakers_open = 0
        rate_limited_services = 0

        for service_name in self.service_configs.keys():
            if service_name in all_ai_status:
                status = all_ai_status[service_name]
                state = status.get("state", "unknown")

                if state == "healthy":
                    healthy_services += 1
                elif state == "circuit_open":
                    circuit_breakers_open += 1
                elif state == "rate_limited":
                    rate_limited_services += 1

        health_percentage = (healthy_services / total_services * 100) if total_services > 0 else 0

        return {
            "overall_health": "healthy" if health_percentage >= 80 else "degraded" if health_percentage >= 50 else "unhealthy",
            "health_percentage": round(health_percentage, 2),
            "healthy_services": healthy_services,
            "total_services": total_services,
            "circuit_breakers_open": circuit_breakers_open,
            "rate_limited_services": rate_limited_services,
            "browser_session_active": self.session_active
        }

    def reset_daily_stats(self):
        """Reset daily usage statistics"""
        for stats in self.usage_stats.values():
            stats.daily_budget_used = 0.0
            stats.last_reset = datetime.now()
        self._save_usage_stats()

    def reset_service_errors(self, service: str) -> bool:
        """Reset error handling state for a specific service"""
        if service not in self.service_configs:
            return False

        # Reset circuit breaker
        circuit_reset = ai_error_handler.reset_circuit_breaker(service)

        # Reset rate limiter
        rate_reset = ai_error_handler.reset_rate_limiter(service)

        # Reset usage stats
        if service in self.usage_stats:
            self.usage_stats[service] = UsageStats(last_reset=datetime.now())
            self._save_usage_stats()

        logger.info(f"Reset error handling state for {service}")
        return circuit_reset and rate_reset

    def reset_all_services_errors(self) -> Dict[str, bool]:
        """Reset error handling state for all services"""
        results = {}
        for service in self.service_configs.keys():
            results[service] = self.reset_service_errors(service)
        return results

    async def health_check_service(self, service: str) -> Dict[str, Any]:
        """Perform health check for a specific service"""
        if service not in self.service_configs:
            return {"error": "Service not configured"}

        config = self.service_configs[service]
        health_result = {
            "service": service,
            "timestamp": datetime.now().isoformat(),
            "api_health": False,
            "web_health": False,
            "overall_health": False
        }

        # Test API if available
        if config.api_available:
            try:
                test_prompt = "Hello, this is a health check."
                await self._call_api(service, test_prompt, max_tokens=10)
                health_result["api_health"] = True
                logger.info(f"API health check passed for {service}")
            except Exception as e:
                health_result["api_error"] = str(e)
                logger.warning(f"API health check failed for {service}: {e}")

        # Test web interface if available
        if config.web_available and self.session_active:
            try:
                # Simple test - just check if we can get a page
                page = await self.get_page(service)
                if page:
                    health_result["web_health"] = True
                    logger.info(f"Web health check passed for {service}")
            except Exception as e:
                health_result["web_error"] = str(e)
                logger.warning(f"Web health check failed for {service}: {e}")

        # Overall health
        health_result["overall_health"] = health_result["api_health"] or health_result["web_health"]

        return health_result

    async def health_check_all_services(self) -> Dict[str, Any]:
        """Perform health check for all services"""
        results = {}
        for service in self.service_configs.keys():
            results[service] = await self.health_check_service(service)

        # Calculate overall system health
        healthy_count = sum(1 for result in results.values() if result.get("overall_health", False))
        total_count = len(results)

        return {
            "services": results,
            "summary": {
                "healthy_services": healthy_count,
                "total_services": total_count,
                "health_percentage": (healthy_count / total_count * 100) if total_count > 0 else 0,
                "overall_healthy": healthy_count >= (total_count * 0.5)  # At least 50% healthy
            },
            "timestamp": datetime.now().isoformat()
        }

# Global instance
hybrid_ai_manager = HybridAIManager()

# Enhanced Convenience Functions
async def query_ai(service: str, prompt: str, **kwargs) -> str:
    """Query AI service with enhanced error handling and monitoring"""
    return await hybrid_ai_manager.query(service, prompt, **kwargs)

async def query_multiple_ai(queries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Query multiple AI services with enhanced error handling"""
    return await hybrid_ai_manager.batch_query(queries)

def get_ai_service_health(service: str) -> Dict[str, Any]:
    """Get comprehensive health status for AI service"""
    return hybrid_ai_manager.get_enhanced_service_status(service)

def get_all_ai_services_health() -> Dict[str, Any]:
    """Get comprehensive health status for all AI services"""
    return hybrid_ai_manager.get_all_services_status()

async def perform_ai_health_check(service: Optional[str] = None) -> Dict[str, Any]:
    """Perform active health check for AI services"""
    if service:
        return await hybrid_ai_manager.health_check_service(service)
    else:
        return await hybrid_ai_manager.health_check_all_services()

def reset_ai_service_errors(service: str) -> bool:
    """Reset error state for specific AI service"""
    return hybrid_ai_manager.reset_service_errors(service)

def reset_all_ai_services_errors() -> Dict[str, bool]:
    """Reset error state for all AI services"""
    return hybrid_ai_manager.reset_all_services_errors()