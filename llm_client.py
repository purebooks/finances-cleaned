import logging
from typing import Optional, Dict, Any, List
from category_rules import ALLOWED_CATEGORIES as RULE_CATEGORIES
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
Given the raw vendor details below, return the most likely normalized vendor NAME.

CRITICAL RULES:
- Never answer with generic product or plan words (e.g., Subscription, Team Plan, Annual Plan, Software, Pro License, Plan, License, Licence).
- Prefer the company/brand (e.g., DigitalOcean, Amazon Web Services, Airtable, Stripe, PayPal, LinkedIn, Spotify).
- If unclear or generic, respond exactly with: Unknown Vendor

Return ONLY the normalized vendor name, nothing else.
Examples of good responses: 'Meta', 'Amazon', 'Google Workspace', 'DigitalOcean', 'Stripe'

Merchant: {merchant}
Description: {description}
Memo: {memo}

Normalized vendor name:
"""
        
        try:
            start_time = time.time()
            response = self.client.messages.create(
                model="claude-3-5-sonnet-20240620",
                max_tokens=50,
                temperature=0.1,  # Lower temperature for more consistent results
                messages=[{"role": "user", "content": prompt}]
            )
            
            result = response.content[0].text.strip()
            # Post-filter: coerce generic terms to Unknown Vendor
            low = (result or '').strip().lower()
            generic_terms = {
                'subscription','team plan','annual plan','software','pro licence',
                'pro license','plan','license','licence','description','memo'
            }
            if (not low) or (low in generic_terms):
                result = "Unknown Vendor"
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

        # Build allowed categories from shared rules
        categories = sorted(list(RULE_CATEGORIES))

        allowed_str = "\n".join(f"- {c}" for c in categories)
        prompt = f"""
You are a business expense classifier.
Choose exactly ONE category from the allowed list below. Do not invent new labels.

Allowed categories (choose exactly one):
{allowed_str}

Merchant: {merchant}
Description: {description}
Amount: ${amount:,.2f}
Memo: {memo}

Rules:
- Respond ONLY in JSON like: {{"category": "<one of allowed>", "confidence": 0.0-1.0}}
- If uncertain, pick the best fit from the allowed list (do not return synonyms).
- Do not include any explanation or extra text.

JSON response:
"""
        
        try:
            start_time = time.time()
            response = self.client.messages.create(
                model="claude-3-5-sonnet-20240620",
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
                    
                raw_category = str(parsed.get("category", "")).strip()
                confidence_val = float(parsed.get("confidence", 0.0))
                coerced = self._coerce_category_to_allowed(raw_category)
                # If model gave an off-list label, coerce to allowed or Other
                result = {
                    "category": coerced,
                    "confidence": max(0.0, min(1.0, confidence_val))
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
                model="claude-3-5-sonnet-20240620",
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
        """Enhanced mock fallback for comprehensive testing"""
        if mode == "vendor":
            content_lower = content.lower()
            
            # Technology companies
            if "google" in content_lower:
                return "Google"
            elif "meta" in content_lower or "facebook" in content_lower:
                return "Meta"
            elif "amazon" in content_lower:
                return "Amazon"
            elif "digitalocean" in content_lower:
                return "DigitalOcean"
            elif "stripe" in content_lower:
                return "Stripe"
            elif "netflix" in content_lower:
                return "Netflix"
            elif "spotify" in content_lower:
                return "Spotify"
            elif "apple" in content_lower:
                return "Apple"
            elif "microsoft" in content_lower:
                return "Microsoft"
            elif "adobe" in content_lower:
                return "Adobe"
            elif "salesforce" in content_lower:
                return "Salesforce"
            elif "dropbox" in content_lower:
                return "Dropbox"
            
            # Food & Coffee
            elif "starbucks" in content_lower:
                return "Starbucks"
            elif "mcdonald" in content_lower:
                return "McDonald's"
            elif "chipotle" in content_lower:
                return "Chipotle"
            elif "subway" in content_lower:
                return "Subway"
            elif "pizza hut" in content_lower:
                return "Pizza Hut"
            elif "domino" in content_lower:
                return "Domino's"
            elif "papa john" in content_lower:
                return "Papa John's"
            
            # Transportation
            elif "uber" in content_lower:
                return "Uber"
            elif "lyft" in content_lower:
                return "Lyft"
            elif "shell" in content_lower:
                return "Shell"
            elif "chevron" in content_lower:
                return "Chevron"
            elif "delta" in content_lower:
                return "Delta Airlines"
            elif "united" in content_lower:
                return "United Airlines"
            elif "southwest" in content_lower:
                return "Southwest Airlines"
            elif "enterprise" in content_lower:
                return "Enterprise Rent-A-Car"
            elif "hertz" in content_lower:
                return "Hertz"
            elif "budget" in content_lower:
                return "Budget"
            
            # Retail
            elif "target" in content_lower:
                return "Target"
            elif "walmart" in content_lower:
                return "Walmart"
            elif "costco" in content_lower:
                return "Costco"
            elif "home depot" in content_lower:
                return "Home Depot"
            elif "best buy" in content_lower:
                return "Best Buy"
            elif "whole foods" in content_lower:
                return "Whole Foods"
            elif "safeway" in content_lower:
                return "Safeway"
            elif "kroger" in content_lower:
                return "Kroger"
            
            # Healthcare & Pharmacy
            elif "cvs" in content_lower:
                return "CVS Pharmacy"
            elif "walgreens" in content_lower:
                return "Walgreens"
            
            # Banking & Finance
            elif "bank of america" in content_lower:
                return "Bank of America"
            elif "wells fargo" in content_lower:
                return "Wells Fargo"
            elif "chase" in content_lower:
                return "Chase Bank"
            elif "american express" in content_lower:
                return "American Express"
            
            # Utilities & Telecom
            elif "at&t" in content_lower or "att" in content_lower:
                return "AT&T"
            elif "verizon" in content_lower:
                return "Verizon"
            elif "comcast" in content_lower:
                return "Comcast"
            
            # Hotels & Travel
            elif "hilton" in content_lower:
                return "Hilton Hotels"
            elif "marriott" in content_lower:
                return "Marriott"
            elif "airbnb" in content_lower:
                return "Airbnb"
            
            else:
                # Extract clean name from messy formats
                clean_name = content.replace("PAYPAL*", "").replace("SQ *", "").replace("AUTO PAY", "").replace("*STORE", "").replace("#", "").replace("*", "").strip()
                return clean_name if clean_name else "Unknown Vendor"
                
        elif mode == "category":
            content_lower = content.lower()
            
            # Technology & Software
            if any(tech in content_lower for tech in ["google", "microsoft", "apple", "adobe", "salesforce", "dropbox", "netflix", "spotify"]):
                return "Software & Technology"
            
            # Food & Entertainment  
            elif any(food in content_lower for food in ["starbucks", "mcdonald", "chipotle", "subway", "pizza", "domino", "papa john", "whole foods", "safeway", "kroger"]):
                return "Meals & Entertainment"
            
            # Transportation
            elif any(transport in content_lower for transport in ["uber", "lyft", "shell", "chevron", "delta", "united", "southwest", "enterprise", "hertz", "budget"]):
                return "Travel & Transportation"
            
            # Retail & Office Supplies
            elif any(retail in content_lower for retail in ["amazon", "target", "walmart", "costco", "home depot", "best buy"]):
                return "Office Supplies & Equipment"
            
            # Healthcare & Professional Services
            elif any(health in content_lower for health in ["cvs", "walgreens"]):
                return "Professional Services"
            
            # Banking & Finance
            elif any(bank in content_lower for bank in ["bank", "chase", "wells fargo", "american express"]):
                return "Banking & Finance"
            
            # Utilities
            elif any(utility in content_lower for utility in ["at&t", "verizon", "comcast", "electric", "gas", "water"]):
                return "Utilities & Rent"
            
            # Hotels & Travel
            elif any(travel in content_lower for travel in ["hilton", "marriott", "airbnb"]):
                return "Travel & Transportation"
            
            # Marketing & Advertising
            elif any(marketing in content_lower for marketing in ["meta", "facebook"]):
                return "Marketing & Advertising"
            
            else:
                return "Other"
        return "Unknown"

    def _coerce_category_to_allowed(self, raw_category: str) -> str:
        """Map arbitrary model output to one of the allowed categories."""
        if not raw_category:
            return "Other"
        normalized = raw_category.strip().lower()

        # Direct case-insensitive match to allowed set
        for allowed in RULE_CATEGORIES:
            if normalized == allowed.lower():
                return allowed

        # Simple synonym funnel
        synonym_map = {
            "software": "Software & Technology",
            "saas": "Software & Technology",
            "technology": "Software & Technology",
            "it": "Software & Technology",
            "ads": "Marketing & Advertising",
            "advertising": "Marketing & Advertising",
            "marketing": "Marketing & Advertising",
            "food": "Meals & Entertainment",
            "dining": "Meals & Entertainment",
            "entertainment": "Meals & Entertainment",
            "travel": "Travel & Transportation",
            "transport": "Travel & Transportation",
            "transportation": "Travel & Transportation",
            "airfare": "Travel & Transportation",
            "hotel": "Travel & Transportation",
            "parking": "Travel & Transportation",
            "fuel": "Travel & Transportation",
            "gas": "Travel & Transportation",
            "office": "Office Supplies & Equipment",
            "supplies": "Office Supplies & Equipment",
            "equipment": "Office Supplies & Equipment",
            "hardware": "Office Supplies & Equipment",
            "furniture": "Office Supplies & Equipment",
            "legal": "Insurance & Legal",
            "insurance": "Insurance & Legal",
            "bank": "Banking & Finance",
            "finance": "Banking & Finance",
            "fees": "Banking & Finance",
            "interest": "Banking & Finance",
            "utilities": "Utilities & Rent",
            "internet": "Utilities & Rent",
            "phone": "Utilities & Rent",
            "telecom": "Utilities & Rent",
            "rent": "Utilities & Rent",
            "benefits": "Employee Benefits",
            "payroll": "Employee Benefits",
            "consulting": "Professional Services",
            "services": "Professional Services",
        }

        mapped = synonym_map.get(normalized)
        if mapped and mapped in RULE_CATEGORIES:
            return mapped

        # Last resort
        return "Other"

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