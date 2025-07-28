import logging
from typing import Optional, Dict, Any, List
import anthropic
import os
import json
import hashlib
from functools import lru_cache
import time

logger = logging.getLogger(__name__)

class LLMClient:
    def __init__(self, provider: str = "anthropic", api_key: Optional[str] = None, use_mock: bool = False, 
                 enable_caching: bool = True, cache_size: int = 1000):
        self.provider = provider
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        self.use_mock = use_mock
        self.enable_caching = enable_caching
        self.cache_size = cache_size
        self.request_count = 0
        self.total_cost = 0.0

        if self.provider == "anthropic" and not self.use_mock:
            if not self.api_key:
                raise ValueError("Anthropic API key required for live calls")
            self.client = anthropic.Anthropic(api_key=self.api_key)
        
        # Initialize cache if enabled
        if self.enable_caching:
            self._vendor_cache = {}
            self._category_cache = {}

    def _generate_cache_key(self, *args) -> str:
        """Generate a cache key from input arguments"""
        content = "|".join(str(arg) for arg in args)
        return hashlib.md5(content.encode()).hexdigest()

    def _get_cached_response(self, cache_dict: dict, cache_key: str) -> Optional[Any]:
        """Get cached response if available"""
        if not self.enable_caching:
            return None
        return cache_dict.get(cache_key)

    def _set_cached_response(self, cache_dict: dict, cache_key: str, response: Any):
        """Set cached response"""
        if not self.enable_caching:
            return
        if len(cache_dict) >= self.cache_size:
            # Remove oldest entry (simple FIFO)
            oldest_key = next(iter(cache_dict))
            del cache_dict[oldest_key]
        cache_dict[cache_key] = response

    def resolve_vendor(self, merchant: str, description: str = "", memo: str = "") -> str:
        """Suggest a normalized vendor name with caching and enhanced error handling"""
        if self.use_mock:
            return self._mock_llm_response("vendor", merchant)

        # Generate cache key
        cache_key = self._generate_cache_key("vendor", merchant, description, memo)
        
        # Check cache first
        cached_result = self._get_cached_response(self._vendor_cache, cache_key)
        if cached_result:
            logger.debug(f"Vendor cache hit for: {merchant}")
            return cached_result

        prompt = f"""
You are a financial data standardizer specializing in business vendor normalization.
Given the raw vendor details below, return the most likely normalized vendor name.

IMPORTANT: Return ONLY the normalized vendor name, nothing else.
Examples of good responses: 'Meta', 'Amazon', 'Google Workspace', 'DigitalOcean', 'Stripe'

Merchant: {merchant}
Description: {description}
Memo: {memo}

Normalized vendor name:
"""
        
        try:
            start_time = time.time()
            response = self.client.messages.create(
                model="claude-3.5-sonnet-20240620",
                max_tokens=50,
                temperature=0.1,  # Lower temperature for more consistent results
                messages=[{"role": "user", "content": prompt}]
            )
            
            result = response.content[0].text.strip()
            processing_time = time.time() - start_time
            
            # Update metrics
            self.request_count += 1
            self.total_cost += 0.01  # Approximate cost per request
            
            logger.info(f"Vendor resolution: {merchant} -> {result} (took {processing_time:.2f}s)")
            
            # Cache the result
            self._set_cached_response(self._vendor_cache, cache_key, result)
            
            return result
            
        except Exception as e:
            logger.error(f"Anthropic vendor resolution failed for '{merchant}': {e}")
            return "Unknown Vendor"

    def suggest_category(self, merchant: str, description: str = "", amount: float = 0.0, memo: str = "") -> Dict[str, Any]:
        """Suggest a category and confidence with enhanced business categories"""
        if self.use_mock:
            return {
                "category": self._mock_llm_response("category", merchant),
                "confidence": 0.85
            }

        # Generate cache key
        cache_key = self._generate_cache_key("category", merchant, description, amount, memo)
        
        # Check cache first
        cached_result = self._get_cached_response(self._category_cache, cache_key)
        if cached_result:
            logger.debug(f"Category cache hit for: {merchant}")
            return cached_result

        # Enhanced category list for business intelligence
        categories = [
            'Marketing & Advertising',
            'Software & Technology', 
            'Office Supplies & Equipment',
            'Travel & Transportation',
            'Meals & Entertainment',
            'Professional Services',
            'Insurance & Legal',
            'Utilities & Rent',
            'Employee Benefits',
            'Banking & Finance',
            'Other'
        ]

        prompt = f"""
You are a business expense classifier for financial data analysis.
Given the transaction info below, return the most likely expense category.

Available categories: {categories}

Merchant: {merchant}
Description: {description}
Amount: ${amount:,.2f}
Memo: {memo}

Consider the amount, merchant type, and context to make the best classification.
Respond in JSON format like this:
{{ "category": "Software & Technology", "confidence": 0.88 }}

JSON response:
"""
        
        try:
            start_time = time.time()
            response = self.client.messages.create(
                model="claude-3.5-sonnet-20240620",
                max_tokens=150,
                temperature=0.2,
                messages=[{"role": "user", "content": prompt}]
            )
            
            # Extract JSON safely
            content = response.content[0].text.strip()
            
            # Try to extract JSON from response
            try:
                # Find JSON in response (in case there's extra text)
                start_idx = content.find('{')
                end_idx = content.rfind('}') + 1
                if start_idx != -1 and end_idx > start_idx:
                    json_content = content[start_idx:end_idx]
                    parsed = json.loads(json_content)
                else:
                    parsed = json.loads(content)
                    
                result = {
                    "category": parsed.get("category", "Other"),
                    "confidence": float(parsed.get("confidence", 0.0))
                }
                
            except json.JSONDecodeError:
                logger.warning(f"Failed to parse JSON response: {content}")
                result = {"category": "Other", "confidence": 0.0}
            
            processing_time = time.time() - start_time
            
            # Update metrics
            self.request_count += 1
            self.total_cost += 0.01  # Approximate cost per request
            
            logger.info(f"Category suggestion: {merchant} -> {result['category']} (confidence: {result['confidence']:.2f}, took {processing_time:.2f}s)")
            
            # Cache the result
            self._set_cached_response(self._category_cache, cache_key, result)
            
            return result
            
        except Exception as e:
            logger.error(f"Anthropic category suggestion failed for '{merchant}': {e}")
            return {"category": "Other", "confidence": 0.0}

    def analyze_transaction(self, merchant: str, description: str = "", amount: float = 0.0, memo: str = "") -> Dict[str, Any]:
        """Comprehensive transaction analysis for business intelligence"""
        if self.use_mock:
            return {
                "vendor": self._mock_llm_response("vendor", merchant),
                "category": self._mock_llm_response("category", merchant),
                "confidence": 0.85,
                "business_impact": "Standard operational expense",
                "risk_level": "Low"
            }

        prompt = f"""
You are a business intelligence analyst for financial data.
Analyze this transaction and provide comprehensive insights.

Merchant: {merchant}
Description: {description}
Amount: ${amount:,.2f}
Memo: {memo}

Provide analysis in JSON format:
{{
    "vendor": "Normalized vendor name",
    "category": "Expense category",
    "confidence": 0.85,
    "business_impact": "High/Medium/Low impact description",
    "risk_level": "High/Medium/Low",
    "recommendation": "Optional cost optimization suggestion"
}}

JSON response:
"""
        
        try:
            response = self.client.messages.create(
                model="claude-3.5-sonnet-20240620",
                max_tokens=200,
                temperature=0.3,
                messages=[{"role": "user", "content": prompt}]
            )
            
            content = response.content[0].text.strip()
            
            try:
                start_idx = content.find('{')
                end_idx = content.rfind('}') + 1
                if start_idx != -1 and end_idx > start_idx:
                    json_content = content[start_idx:end_idx]
                    parsed = json.loads(json_content)
                else:
                    parsed = json.loads(content)
                    
                result = {
                    "vendor": parsed.get("vendor", "Unknown Vendor"),
                    "category": parsed.get("category", "Other"),
                    "confidence": float(parsed.get("confidence", 0.0)),
                    "business_impact": parsed.get("business_impact", "Standard expense"),
                    "risk_level": parsed.get("risk_level", "Low"),
                    "recommendation": parsed.get("recommendation", "")
                }
                
            except json.JSONDecodeError:
                logger.warning(f"Failed to parse JSON response: {content}")
                result = {
                    "vendor": "Unknown Vendor",
                    "category": "Other",
                    "confidence": 0.0,
                    "business_impact": "Standard expense",
                    "risk_level": "Low",
                    "recommendation": ""
                }
            
            # Update metrics
            self.request_count += 1
            self.total_cost += 0.02  # Higher cost for comprehensive analysis
            
            return result
            
        except Exception as e:
            logger.error(f"Transaction analysis failed for '{merchant}': {e}")
            return {
                "vendor": "Unknown Vendor",
                "category": "Other",
                "confidence": 0.0,
                "business_impact": "Standard expense",
                "risk_level": "Low",
                "recommendation": ""
            }

    def get_usage_stats(self) -> Dict[str, Any]:
        """Get usage statistics for monitoring and billing"""
        return {
            "request_count": self.request_count,
            "total_cost": self.total_cost,
            "cache_hit_rate": self._calculate_cache_hit_rate(),
            "average_cost_per_request": self.total_cost / max(self.request_count, 1)
        }

    def _calculate_cache_hit_rate(self) -> float:
        """Calculate cache hit rate"""
        if not self.enable_caching:
            return 0.0
        
        total_cache_entries = len(self._vendor_cache) + len(self._category_cache)
        # This is a simplified calculation - in production you'd track actual hits
        return min(0.3, total_cache_entries / max(self.request_count, 1))

    def _mock_llm_response(self, mode: str, content: str) -> str:
        """Mock fallback for local dev/testing"""
        if mode == "vendor":
            content_lower = content.lower()
            if "google" in content_lower:
                return "Google Workspace"
            elif "meta" in content_lower or "facebook" in content_lower:
                return "Meta"
            elif "amazon" in content_lower:
                return "Amazon"
            elif "digitalocean" in content_lower:
                return "DigitalOcean"
            elif "stripe" in content_lower:
                return "Stripe"
            else:
                return "Unknown Vendor"
        elif mode == "category":
            content_lower = content.lower()
            if "meta" in content_lower or "facebook" in content_lower:
                return "Marketing & Advertising"
            elif "notion" in content_lower or "zoom" in content_lower or "slack" in content_lower:
                return "Software & Technology"
            elif "amazon" in content_lower:
                return "Office Supplies & Equipment"
            else:
                return "Other"
        return "Unknown"

    def reset_metrics(self):
        """Reset usage metrics"""
        self.request_count = 0
        self.total_cost = 0.0

    def clear_cache(self):
        """Clear all caches"""
        if hasattr(self, '_vendor_cache'):
            self._vendor_cache.clear()
        if hasattr(self, '_category_cache'):
            self._category_cache.clear() 