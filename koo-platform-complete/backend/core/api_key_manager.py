"""
KOO Platform API Key Integration Manager
Secure API key management, validation, and health checking
"""

import os
import asyncio
import aiohttp
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum
import logging
from cryptography.fernet import Fernet
import base64
import hashlib

from .exceptions import APIKeyError, APIKeyValidationError, APIKeyConfigurationError, ExternalServiceError

logger = logging.getLogger(__name__)

class APIProvider(Enum):
    """Supported API providers"""
    GEMINI = "gemini"
    CLAUDE = "claude"
    PUBMED = "pubmed"
    PERPLEXITY = "perplexity"
    SEMANTIC_SCHOLAR = "semantic_scholar"
    ELSEVIER = "elsevier"
    BIODIGITAL = "biodigital"

@dataclass
class APIKeyConfig:
    """API key configuration with validation"""
    provider: APIProvider
    key: str
    encrypted: bool = False
    last_validated: Optional[datetime] = None
    is_valid: bool = False
    usage_count: int = 0
    rate_limit_remaining: Optional[int] = None
    rate_limit_reset: Optional[datetime] = None
    endpoint_url: Optional[str] = None
    headers: Optional[Dict[str, str]] = None

class APIKeyManager:
    """Comprehensive API key management system"""

    def __init__(self, secret_key: Optional[str] = None):
        self.secret_key = secret_key or os.getenv("SECRET_KEY", "default-key-change-this")
        self.cipher_suite = self._create_cipher()
        self.api_configs: Dict[APIProvider, APIKeyConfig] = {}
        self.validation_cache = {}
        self.health_check_interval = timedelta(hours=1)

        # Initialize from environment
        self._load_from_environment()

    def _create_cipher(self) -> Fernet:
        """Create encryption cipher from secret key"""
        key = base64.urlsafe_b64encode(
            hashlib.sha256(self.secret_key.encode()).digest()
        )
        return Fernet(key)

    def _load_from_environment(self) -> None:
        """Load API keys from environment variables"""
        env_mappings = {
            APIProvider.GEMINI: ("GEMINI_API_KEY", "https://generativelanguage.googleapis.com/v1/models"),
            APIProvider.CLAUDE: ("CLAUDE_API_KEY", "https://api.anthropic.com/v1/messages"),
            APIProvider.PUBMED: ("PUBMED_API_KEY", "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"),
            APIProvider.PERPLEXITY: ("PERPLEXITY_API_KEY", "https://api.perplexity.ai/chat/completions"),
            APIProvider.SEMANTIC_SCHOLAR: ("SEMANTIC_SCHOLAR_API_KEY", "https://api.semanticscholar.org/v1"),
            APIProvider.ELSEVIER: ("ELSEVIER_API_KEY", "https://api.elsevier.com/content"),
            APIProvider.BIODIGITAL: ("BIODIGITAL_API_KEY", "https://api.biodigital.com/v1")
        }

        for provider, (env_var, endpoint) in env_mappings.items():
            api_key = os.getenv(env_var)
            if api_key:
                self.add_api_key(provider, api_key, endpoint)

    def add_api_key(self, provider: APIProvider, key: str, endpoint_url: Optional[str] = None,
                   headers: Optional[Dict[str, str]] = None, encrypt: bool = True) -> bool:
        """Add or update an API key"""
        try:
            processed_key = self._encrypt_key(key) if encrypt else key

            config = APIKeyConfig(
                provider=provider,
                key=processed_key,
                encrypted=encrypt,
                endpoint_url=endpoint_url,
                headers=headers or {}
            )

            self.api_configs[provider] = config
            logger.info(f"Added API key for {provider.value}")
            return True

        except Exception as e:
            logger.error(f"Failed to add API key for {provider.value}: {e}")
            return False

    def _encrypt_key(self, key: str) -> str:
        """Encrypt API key"""
        return self.cipher_suite.encrypt(key.encode()).decode()

    def _decrypt_key(self, encrypted_key: str) -> str:
        """Decrypt API key"""
        return self.cipher_suite.decrypt(encrypted_key.encode()).decode()

    def get_api_key(self, provider: APIProvider) -> Optional[str]:
        """Get decrypted API key for provider"""
        config = self.api_configs.get(provider)
        if not config:
            return None

        try:
            if config.encrypted:
                return self._decrypt_key(config.key)
            return config.key
        except Exception as e:
            logger.error(f"Failed to decrypt key for {provider.value}: {e}")
            return None

    def remove_api_key(self, provider: APIProvider) -> bool:
        """Remove API key for provider"""
        if provider in self.api_configs:
            del self.api_configs[provider]
            logger.info(f"Removed API key for {provider.value}")
            return True
        return False

    def get_provider_headers(self, provider: APIProvider) -> Dict[str, str]:
        """Get headers for API provider including authentication"""
        config = self.api_configs.get(provider)
        if not config:
            return {}

        api_key = self.get_api_key(provider)
        if not api_key:
            return {}

        # Provider-specific header formats
        auth_headers = {
            APIProvider.GEMINI: {"x-goog-api-key": api_key},
            APIProvider.CLAUDE: {"x-api-key": api_key, "anthropic-version": "2023-06-01"},
            APIProvider.PUBMED: {"api_key": api_key},
            APIProvider.PERPLEXITY: {"Authorization": f"Bearer {api_key}"},
            APIProvider.SEMANTIC_SCHOLAR: {"x-api-key": api_key},
            APIProvider.ELSEVIER: {"X-ELS-APIKey": api_key},
            APIProvider.BIODIGITAL: {"Authorization": f"Bearer {api_key}"}
        }

        headers = auth_headers.get(provider, {"Authorization": f"Bearer {api_key}"})
        headers.update(config.headers or {})

        return headers

    async def validate_api_key(self, provider: APIProvider) -> Tuple[bool, str]:
        """Validate API key by making a test request"""
        config = self.api_configs.get(provider)
        if not config:
            return False, f"No API key configured for {provider.value}"

        api_key = self.get_api_key(provider)
        if not api_key:
            return False, f"Failed to retrieve API key for {provider.value}"

        # Check cache first
        cache_key = f"{provider.value}_{api_key[:10]}"
        if cache_key in self.validation_cache:
            cached_result, cached_time = self.validation_cache[cache_key]
            if datetime.now() - cached_time < timedelta(minutes=30):
                return cached_result

        try:
            headers = self.get_provider_headers(provider)
            validation_result = await self._perform_validation_request(provider, headers)

            # Update configuration
            config.last_validated = datetime.now()
            config.is_valid = validation_result[0]

            # Cache result
            self.validation_cache[cache_key] = (validation_result, datetime.now())

            return validation_result

        except Exception as e:
            logger.error(f"API key validation failed for {provider.value}: {e}")
            config.is_valid = False
            return False, str(e)

    async def _perform_validation_request(self, provider: APIProvider, headers: Dict[str, str]) -> Tuple[bool, str]:
        """Perform actual validation request for each provider"""
        config = self.api_configs[provider]

        validation_endpoints = {
            APIProvider.GEMINI: (f"{config.endpoint_url}", "GET"),
            APIProvider.CLAUDE: ("https://api.anthropic.com/v1/messages", "POST"),
            APIProvider.PUBMED: (f"{config.endpoint_url}/esummary.fcgi?db=pubmed&id=1&retmode=json", "GET"),
            APIProvider.PERPLEXITY: (f"{config.endpoint_url}", "POST"),
            APIProvider.SEMANTIC_SCHOLAR: (f"{config.endpoint_url}/paper/search?query=test&limit=1", "GET"),
            APIProvider.ELSEVIER: (f"{config.endpoint_url}/search/sciencedirect?query=test&count=1", "GET"),
            APIProvider.BIODIGITAL: (f"{config.endpoint_url}/models", "GET")
        }

        if provider not in validation_endpoints:
            return False, f"Validation not implemented for {provider.value}"

        endpoint, method = validation_endpoints[provider]

        async with aiohttp.ClientSession() as session:
            try:
                if method == "GET":
                    async with session.get(endpoint, headers=headers, timeout=10) as response:
                        if response.status == 200:
                            return True, "API key is valid"
                        elif response.status == 401:
                            return False, "Invalid API key"
                        elif response.status == 403:
                            return False, "API key lacks permissions"
                        else:
                            return False, f"Unexpected response: {response.status}"

                elif method == "POST":
                    # Minimal test payload for POST endpoints
                    test_data = self._get_test_payload(provider)
                    async with session.post(endpoint, headers=headers, json=test_data, timeout=10) as response:
                        if response.status in [200, 201]:
                            return True, "API key is valid"
                        elif response.status == 401:
                            return False, "Invalid API key"
                        elif response.status == 403:
                            return False, "API key lacks permissions"
                        else:
                            return False, f"Unexpected response: {response.status}"

            except asyncio.TimeoutError:
                return False, "Request timeout"
            except Exception as e:
                return False, f"Request failed: {str(e)}"

    def _get_test_payload(self, provider: APIProvider) -> Dict[str, Any]:
        """Get minimal test payload for POST validation"""
        test_payloads = {
            APIProvider.CLAUDE: {
                "model": "claude-3-haiku-20240307",
                "max_tokens": 10,
                "messages": [{"role": "user", "content": "test"}]
            },
            APIProvider.PERPLEXITY: {
                "model": "llama-3.1-sonar-small-128k-online",
                "messages": [{"role": "user", "content": "test"}],
                "max_tokens": 10
            }
        }
        return test_payloads.get(provider, {})

    async def validate_all_keys(self) -> Dict[APIProvider, Tuple[bool, str]]:
        """Validate all configured API keys"""
        results = {}
        tasks = []

        for provider in self.api_configs.keys():
            tasks.append(self.validate_api_key(provider))

        if tasks:
            validation_results = await asyncio.gather(*tasks, return_exceptions=True)

            for i, provider in enumerate(self.api_configs.keys()):
                result = validation_results[i]
                if isinstance(result, Exception):
                    results[provider] = (False, str(result))
                else:
                    results[provider] = result

        return results

    async def health_check(self) -> Dict[str, Any]:
        """Comprehensive health check of all API services"""
        validation_results = await self.validate_all_keys()

        health_status = {
            "timestamp": datetime.now().isoformat(),
            "total_providers": len(self.api_configs),
            "valid_providers": sum(1 for result in validation_results.values() if result[0]),
            "invalid_providers": sum(1 for result in validation_results.values() if not result[0]),
            "providers": {}
        }

        for provider, (is_valid, message) in validation_results.items():
            config = self.api_configs[provider]
            health_status["providers"][provider.value] = {
                "is_valid": is_valid,
                "message": message,
                "last_validated": config.last_validated.isoformat() if config.last_validated else None,
                "usage_count": config.usage_count,
                "rate_limit_remaining": config.rate_limit_remaining,
                "rate_limit_reset": config.rate_limit_reset.isoformat() if config.rate_limit_reset else None
            }

        return health_status

    def get_available_providers(self) -> List[APIProvider]:
        """Get list of providers with valid API keys"""
        return [provider for provider, config in self.api_configs.items() if config.is_valid]

    def increment_usage(self, provider: APIProvider) -> None:
        """Increment usage counter for provider"""
        config = self.api_configs.get(provider)
        if config:
            config.usage_count += 1

    def update_rate_limit_info(self, provider: APIProvider, remaining: int, reset_time: datetime) -> None:
        """Update rate limit information for provider"""
        config = self.api_configs.get(provider)
        if config:
            config.rate_limit_remaining = remaining
            config.rate_limit_reset = reset_time

    def export_configuration(self, include_keys: bool = False) -> Dict[str, Any]:
        """Export configuration (optionally including keys for backup)"""
        export_data = {
            "timestamp": datetime.now().isoformat(),
            "providers": {}
        }

        for provider, config in self.api_configs.items():
            provider_data = {
                "provider": provider.value,
                "encrypted": config.encrypted,
                "last_validated": config.last_validated.isoformat() if config.last_validated else None,
                "is_valid": config.is_valid,
                "usage_count": config.usage_count,
                "endpoint_url": config.endpoint_url,
                "headers": config.headers
            }

            if include_keys:
                provider_data["key"] = config.key  # Encrypted if encryption was used

            export_data["providers"][provider.value] = provider_data

        return export_data

# Global instance
api_key_manager = APIKeyManager()

# Convenience functions
async def validate_provider_key(provider: APIProvider) -> Tuple[bool, str]:
    """Validate a specific provider's API key"""
    return await api_key_manager.validate_api_key(provider)

async def get_api_headers(provider: APIProvider) -> Dict[str, str]:
    """Get headers for API requests"""
    return api_key_manager.get_provider_headers(provider)

async def check_all_api_health() -> Dict[str, Any]:
    """Check health of all API services"""
    return await api_key_manager.health_check()

def get_valid_providers() -> List[APIProvider]:
    """Get list of providers with valid keys"""
    return api_key_manager.get_available_providers()