"""
Advanced LLM Flow Components
Intelligent processing with caching, tracking, and source attribution
"""

import hashlib
import json
import time
import logging
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass
from enum import Enum
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

class ProcessingSource(Enum):
    """Sources of processing decisions"""
    RULE_BASED = "rule_based"
    CACHE = "cache"
    LLM = "llm"

@dataclass
class ProcessingResult:
    """Result of any processing operation with source tracking"""
    value: Any
    source: ProcessingSource
    confidence: float
    explanation: str
    processing_time: float
    cost: float = 0.0

class IntelligentCache:
    """Enhanced caching system with smart key generation and size management"""
    
    def __init__(self, max_cache_size: int = 1000):
        self.vendor_cache = {}
        self.category_cache = {}
        self.intelligence_cache = {}
        self.max_cache_size = max_cache_size
        self.cache_stats = {
            'vendor_hits': 0,
            'vendor_misses': 0,
            'category_hits': 0,
            'category_misses': 0,
            'intelligence_hits': 0,
            'intelligence_misses': 0,
            'cache_evictions': 0
        }
    
    def _normalize_vendor_name(self, vendor: str) -> str:
        """Normalize vendor names for better cache hits"""
        if not vendor:
            return ""
        
        vendor_clean = vendor.upper().strip()
        
        # Remove common variations that don't affect categorization
        import re
        
        # Remove store numbers: "STARBUCKS STORE #1234" → "STARBUCKS STORE"
        vendor_clean = re.sub(r'\s*#\d+', '', vendor_clean)
        vendor_clean = re.sub(r'\s*STORE\s*\d+', ' STORE', vendor_clean)
        
        # Remove payment processor prefixes: "SQ *CAFE NAME" → "CAFE NAME"
        prefixes = ['SQ *', 'PAYPAL *', 'TST*', 'PP*']
        for prefix in prefixes:
            if vendor_clean.startswith(prefix):
                vendor_clean = vendor_clean[len(prefix):].strip()
                break
        
        # Remove trailing location codes: "AMAZON.COM AMZN.MK" → "AMAZON.COM"
        vendor_clean = re.sub(r'\s+[A-Z]{2,}\.[A-Z]{2,}$', '', vendor_clean)
        
        return vendor_clean.strip()
    
    def _get_amount_range(self, amount: float) -> str:
        """Convert exact amounts to ranges for better cache hits"""
        abs_amount = abs(amount)
        
        if abs_amount == 0:
            return "zero"
        elif abs_amount < 5:
            return "micro"  # $0-5
        elif abs_amount < 15:
            return "small"  # $5-15
        elif abs_amount < 50:
            return "medium" # $15-50
        elif abs_amount < 200:
            return "large"  # $50-200
        elif abs_amount < 1000:
            return "xlarge" # $200-1000
        else:
            return "huge"   # $1000+
    
    def _generate_cache_key(self, data: Dict[str, Any]) -> str:
        """Generate smart cache keys with normalization"""
        processed_data = {}
        
        for key, value in data.items():
            if key == 'vendor' and isinstance(value, str):
                processed_data[key] = self._normalize_vendor_name(value)
            elif key == 'amount' and isinstance(value, (int, float)):
                processed_data['amount_range'] = self._get_amount_range(value)
            else:
                processed_data[key] = value
        
        sorted_data = json.dumps(processed_data, sort_keys=True)
        return hashlib.md5(sorted_data.encode()).hexdigest()
    
    def _evict_if_needed(self, cache_dict: dict):
        """Evict oldest entries if cache is too large"""
        if len(cache_dict) >= self.max_cache_size:
            # Remove 20% of entries (simple LRU approximation)
            items_to_remove = max(1, len(cache_dict) // 5)
            keys_to_remove = list(cache_dict.keys())[:items_to_remove]
            for key in keys_to_remove:
                del cache_dict[key]
            self.cache_stats['cache_evictions'] += items_to_remove
    
    def get_vendor_cache(self, vendor: str) -> Optional[Dict[str, Any]]:
        """Get vendor standardization from cache with enhanced key matching"""
        key = self._generate_cache_key({'vendor': vendor})
        if key in self.vendor_cache:
            self.cache_stats['vendor_hits'] += 1
            logger.debug(f"Cache HIT for vendor: {vendor} → {self._normalize_vendor_name(vendor)}")
            return self.vendor_cache[key]
        self.cache_stats['vendor_misses'] += 1
        logger.debug(f"Cache MISS for vendor: {vendor} → {self._normalize_vendor_name(vendor)}")
        return None
    
    def set_vendor_cache(self, vendor: str, result: Dict[str, Any]):
        """Cache vendor standardization result with eviction management"""
        self._evict_if_needed(self.vendor_cache)
        key = self._generate_cache_key({'vendor': vendor})
        self.vendor_cache[key] = result
        logger.debug(f"Cached vendor: {vendor} → {self._normalize_vendor_name(vendor)}")
    
    def get_category_cache(self, vendor: str, amount: float) -> Optional[Dict[str, Any]]:
        """Get category classification from cache with amount range matching"""
        key = self._generate_cache_key({'vendor': vendor, 'amount': amount})
        if key in self.category_cache:
            self.cache_stats['category_hits'] += 1
            logger.debug(f"Cache HIT for category: {self._normalize_vendor_name(vendor)} @ {self._get_amount_range(amount)}")
            return self.category_cache[key]
        self.cache_stats['category_misses'] += 1
        logger.debug(f"Cache MISS for category: {self._normalize_vendor_name(vendor)} @ {self._get_amount_range(amount)}")
        return None
    
    def set_category_cache(self, vendor: str, amount: float, result: Dict[str, Any]):
        """Cache category classification result with amount range optimization"""
        self._evict_if_needed(self.category_cache)
        key = self._generate_cache_key({'vendor': vendor, 'amount': amount})
        self.category_cache[key] = result
        logger.debug(f"Cached category: {self._normalize_vendor_name(vendor)} @ {self._get_amount_range(amount)} → {result.get('category', 'Unknown')}")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get comprehensive enhanced cache statistics"""
        total_vendor = self.cache_stats['vendor_hits'] + self.cache_stats['vendor_misses']
        total_category = self.cache_stats['category_hits'] + self.cache_stats['category_misses']
        total_requests = total_vendor + total_category
        total_hits = self.cache_stats['vendor_hits'] + self.cache_stats['category_hits']
        
        return {
            'vendor_hit_rate': self.cache_stats['vendor_hits'] / total_vendor if total_vendor > 0 else 0,
            'category_hit_rate': self.cache_stats['category_hits'] / total_category if total_category > 0 else 0,
            'overall_hit_rate': total_hits / total_requests if total_requests > 0 else 0,
            'total_vendor_requests': total_vendor,
            'total_category_requests': total_category,
            'total_requests': total_requests,
            'cache_size': len(self.vendor_cache) + len(self.category_cache),
            'vendor_cache_size': len(self.vendor_cache),
            'category_cache_size': len(self.category_cache),
            'cache_evictions': self.cache_stats['cache_evictions'],
            'max_cache_size': self.max_cache_size,
            'cache_efficiency': {
                'vendor_hits': self.cache_stats['vendor_hits'],
                'category_hits': self.cache_stats['category_hits'],
                'vendor_misses': self.cache_stats['vendor_misses'],
                'category_misses': self.cache_stats['category_misses']
            }
        }

class LLMTracker:
    """Comprehensive LLM cost, time, and performance tracking"""
    
    def __init__(self):
        self.cost_tracker = {
            'total_cost': 0.0,
            'vendor_standardization_cost': 0.0,
            'category_classification_cost': 0.0,
            'transaction_intelligence_cost': 0.0
        }
        self.time_tracker = {
            'total_time': 0.0,
            'vendor_standardization_time': 0.0,
            'category_classification_time': 0.0,
            'transaction_intelligence_time': 0.0
        }
        self.performance_tracker = {
            'total_llm_calls': 0,
            'successful_llm_calls': 0,
            'failed_llm_calls': 0,
            'average_response_time': 0.0
        }
    
    def track_llm_call(self, operation: str, cost: float, time: float, success: bool = True):
        """Track LLM call metrics"""
        self.cost_tracker[f'{operation}_cost'] += cost
        self.cost_tracker['total_cost'] += cost
        self.time_tracker[f'{operation}_time'] += time
        self.time_tracker['total_time'] += time
        self.performance_tracker['total_llm_calls'] += 1
        
        if success:
            self.performance_tracker['successful_llm_calls'] += 1
        else:
            self.performance_tracker['failed_llm_calls'] += 1
        
        # Update average response time
        total_calls = self.performance_tracker['successful_llm_calls'] + self.performance_tracker['failed_llm_calls']
        if total_calls > 0:
            self.performance_tracker['average_response_time'] = self.time_tracker['total_time'] / total_calls
    
    def get_tracking_summary(self) -> Dict[str, Any]:
        """Get comprehensive tracking summary"""
        return {
            'cost_analysis': self.cost_tracker,
            'performance_metrics': self.time_tracker,
            'llm_performance': self.performance_tracker,
            'cost_per_call': self.cost_tracker['total_cost'] / self.performance_tracker['total_llm_calls'] if self.performance_tracker['total_llm_calls'] > 0 else 0
        }

class AdvancedLLMProcessor:
    """Advanced LLM processor with intelligent flow"""
    
    def __init__(self, llm_client, config: Dict[str, Any]):
        self.llm_client = llm_client
        self.config = config
        self.cache = IntelligentCache()
        self.tracker = LLMTracker()
        self.processing_history = []
    
    def process_vendor_standardization(self, vendor: str, row: pd.Series) -> ProcessingResult:
        """Intelligent vendor standardization with source tracking"""
        start_time = time.time()
        
        # Step 1: Rule-based matching
        rule_result = self._apply_vendor_rules(vendor)
        if rule_result['matched']:
            return ProcessingResult(
                value=rule_result['standardized'],
                source=ProcessingSource.RULE_BASED,
                confidence=rule_result['confidence'],
                explanation=rule_result['explanation'],
                processing_time=time.time() - start_time
            )
        
        # Step 2: Cache lookup
        cache_result = self.cache.get_vendor_cache(vendor)
        if cache_result:
            return ProcessingResult(
                value=cache_result['standardized'],
                source=ProcessingSource.CACHE,
                confidence=cache_result['confidence'],
                explanation=cache_result['explanation'],
                processing_time=time.time() - start_time
            )
        
        # Step 3: LLM processing
        try:
            llm_start_time = time.time()
            llm_result = self.llm_client.resolve_vendor(vendor, "", "")
            llm_time = time.time() - llm_start_time
            
            # Cache the result
            self.cache.set_vendor_cache(vendor, {
                'standardized': llm_result,
                'confidence': 0.8,  # Default confidence for LLM
                'explanation': f"AI-standardized vendor name"
            })
            
            # Track LLM call
            self.tracker.track_llm_call('vendor_standardization', 0.01, llm_time, True)
            
            return ProcessingResult(
                value=llm_result,
                source=ProcessingSource.LLM,
                confidence=0.8,
                explanation="AI-powered vendor standardization",
                processing_time=time.time() - start_time,
                cost=0.01
            )
            
        except Exception as e:
            logger.warning(f"LLM vendor standardization failed for {vendor}: {e}")
            self.tracker.track_llm_call('vendor_standardization', 0.0, time.time() - start_time, False)
            
            return ProcessingResult(
                value=vendor,  # Return original if LLM fails
                source=ProcessingSource.RULE_BASED,
                confidence=0.5,
                explanation=f"LLM failed, using original vendor name",
                processing_time=time.time() - start_time
            )
    
    def process_category_classification(self, vendor: str, amount: float, row: pd.Series) -> ProcessingResult:
        """Intelligent category classification with source tracking"""
        start_time = time.time()
        
        # Step 1: High-value transaction check - always use AI for validation if >$500
        if amount > 500:
            logger.info(f"High-value transaction (${amount}) - using AI validation for {vendor}")
            try:
                llm_start_time = time.time()
                llm_result = self.llm_client.suggest_category(vendor, "", amount)
                llm_time = time.time() - llm_start_time
                
                # Cache the result
                self.cache.set_category_cache(vendor, amount, {
                    'category': llm_result['category'],
                    'confidence': llm_result['confidence'],
                    'explanation': f"AI high-value classification"
                })
                
                # Track LLM call
                self.tracker.track_llm_call('category_classification', 0.015, llm_time, True)
                
                return ProcessingResult(
                    value=llm_result['category'],
                    source=ProcessingSource.LLM,
                    confidence=llm_result['confidence'],
                    explanation=f"AI-powered high-value transaction classification (${amount})",
                    processing_time=time.time() - start_time,
                    cost=0.015
                )
            except Exception as e:
                logger.warning(f"LLM high-value classification failed for {vendor}: {e}")
                # Fall through to rule-based
        
        # Step 2: Rule-based classification
        rule_result = self._apply_category_rules(vendor, amount)
        if rule_result['matched']:
            return ProcessingResult(
                value=rule_result['category'],
                source=ProcessingSource.RULE_BASED,
                confidence=rule_result['confidence'],
                explanation=rule_result['explanation'],
                processing_time=time.time() - start_time
            )
        
        # Step 2: Cache lookup
        cache_result = self.cache.get_category_cache(vendor, amount)
        if cache_result:
            return ProcessingResult(
                value=cache_result['category'],
                source=ProcessingSource.CACHE,
                confidence=cache_result['confidence'],
                explanation=cache_result['explanation'],
                processing_time=time.time() - start_time
            )
        
        # Step 3: LLM processing
        try:
            llm_start_time = time.time()
            llm_result = self.llm_client.suggest_category(vendor, "", amount)
            llm_time = time.time() - llm_start_time
            
            # Cache the result
            self.cache.set_category_cache(vendor, amount, {
                'category': llm_result['category'],
                'confidence': llm_result['confidence'],
                'explanation': f"AI-classified category"
            })
            
            # Track LLM call
            self.tracker.track_llm_call('category_classification', 0.01, llm_time, True)
            
            return ProcessingResult(
                value=llm_result['category'],
                source=ProcessingSource.LLM,
                confidence=llm_result['confidence'],
                explanation="AI-powered category classification",
                processing_time=time.time() - start_time,
                cost=0.01
            )
            
        except Exception as e:
            logger.warning(f"LLM category classification failed for {vendor}: {e}")
            self.tracker.track_llm_call('category_classification', 0.0, time.time() - start_time, False)
            
            return ProcessingResult(
                value="Uncategorized",
                source=ProcessingSource.RULE_BASED,
                confidence=0.3,
                explanation=f"LLM failed, using default category",
                processing_time=time.time() - start_time
            )
    
    def process_transaction_intelligence(self, row: pd.Series) -> Dict[str, Any]:
        """Generate transaction intelligence (separate from CSV)"""
        intelligence = {
            'tags': self._generate_transaction_tags(row),
            'insights': self._generate_transaction_insights(row),
            'explainability': self._generate_transaction_explanation(row),
            'risk_score': self._calculate_transaction_risk(row),
            'anomaly_detection': self._detect_transaction_anomalies(row)
        }
        
        return intelligence
    
    def _apply_vendor_rules(self, vendor: str) -> Dict[str, Any]:
        """Apply rule-based vendor standardization with payment processor prefix handling"""
        vendor_lower = vendor.lower()
        
        # CRITICAL: Check for payment processor prefixes FIRST and extract the real vendor
        if any(vendor.upper().startswith(prefix) for prefix in ["PAYPAL *", "PAYPAL*", "SQ *", "TST*", "AUTO PAY "]):
            cleaned_vendor = self._clean_vendor_name_for_rules(vendor)
            cleaned_lower = cleaned_vendor.lower()
            
            # Now look up the cleaned vendor (the real business)
            if cleaned_vendor and len(cleaned_vendor) > 1:
                # Try to match the extracted vendor
                real_vendor_result = self._match_vendor_name(cleaned_vendor)
                if real_vendor_result['matched']:
                    return {
                        'matched': True,
                        'standardized': real_vendor_result['standardized'],
                        'confidence': 0.78,
                        'explanation': f"Payment processor prefix removed: {vendor} → {real_vendor_result['standardized']}"
                    }
                else:
                    # Return the cleaned name even if no exact match
                    return {
                        'matched': True,
                        'standardized': cleaned_vendor,
                        'confidence': 0.72,
                        'explanation': f"Payment processor prefix removed: {vendor} → {cleaned_vendor}"
                    }
        
        # For non-prefixed vendors, use normal matching
        return self._match_vendor_name(vendor)
    
    def _match_vendor_name(self, vendor: str) -> Dict[str, Any]:
        """Match a vendor name against known mappings"""
        vendor_lower = vendor.lower()
        
        # Common vendor mappings
        vendor_mappings = {
            'google': 'Google',
            'meta': 'Meta',
            'facebook': 'Meta',
            'amazon': 'Amazon',
            'aws': 'Amazon Web Services',
            'digitalocean': 'DigitalOcean',
            'stripe': 'Stripe',
            'netflix': 'Netflix',
            'spotify': 'Spotify',
            'uber': 'Uber',
            'lyft': 'Lyft',
            'apple': 'Apple',
            'microsoft': 'Microsoft',
            'adobe': 'Adobe',
            'salesforce': 'Salesforce',
            'dropbox': 'Dropbox',
            'starbucks': 'Starbucks',
            'mcdonald': 'McDonald\'s',
            'chipotle': 'Chipotle',
            'subway': 'Subway',
            'domino': 'Domino\'s',
            'pizza hut': 'Pizza Hut',
            'papa john': 'Papa John\'s',
            'southwest airlines': 'Southwest Airlines',
            'delta airlines': 'Delta Airlines',
            'united airlines': 'United Airlines',
            'american airlines': 'American Airlines',
            'hilton': 'Hilton Hotels',
            'marriott': 'Marriott',
            'airbnb': 'Airbnb',
            'budget': 'Budget',
            'hertz': 'Hertz',
            'enterprise': 'Enterprise',
            'target': 'Target',
            'walmart': 'Walmart',
            'costco': 'Costco',
            'home depot': 'Home Depot',
            'best buy': 'Best Buy',
            'cvs': 'CVS Pharmacy',
            'walgreens': 'Walgreens',
            'chase bank': 'Chase Bank',
            'wells fargo': 'Wells Fargo',
            'bank of america': 'Bank of America',
            'american express': 'American Express',
            'comcast': 'Comcast',
            'verizon': 'Verizon',
            'at&t': 'AT&T'
        }
        
        # Try exact matches first
        if vendor_lower in vendor_mappings:
            return {
                'matched': True,
                'standardized': vendor_mappings[vendor_lower],
                'confidence': 0.82,
                'explanation': f"Rule-based exact match: {vendor} → {vendor_mappings[vendor_lower]}"
            }
        
        # Try partial matches
        for pattern, standardized in vendor_mappings.items():
            if pattern in vendor_lower:
                return {
                    'matched': True,
                    'standardized': standardized,
                    'confidence': 0.90,
                    'explanation': f"Rule-based partial match: {pattern} found in {vendor} → {standardized}"
                }
        
        return {'matched': False}
    
    def _clean_vendor_name_for_rules(self, vendor: str) -> str:
        """Clean vendor name for rule-based processing"""
        if not vendor:
            return ""
        
        cleaned = vendor.strip()
        
        # Remove payment processor prefixes (order matters - longer first)
        prefixes = [
            "PAYPAL *", "PAYPAL*", "SQ *", "TST* ", "TST*", "AUTO PAY ",
            "AMZ*", "AMZ *", "AMAZON*", "AMAZON *"
        ]
        
        for prefix in prefixes:
            if cleaned.upper().startswith(prefix.upper()):
                cleaned = cleaned[len(prefix):].strip()
                break  # Only remove one prefix
        
        # Remove common suffixes
        suffixes = [
            " INC", " LLC", " CORP", " ONLINE", " .COM", ".COM",
            "*STORE 001", "*STORE", " #123456", "#123456", " STORE",
            " 001", "#001"
        ]
        
        for suffix in suffixes:
            if cleaned.upper().endswith(suffix.upper()):
                cleaned = cleaned[:-len(suffix)].strip()
        
        # Remove extra characters
        cleaned = cleaned.replace("*", "").replace("#", "").strip()
        
        # Handle specific cases
        if cleaned.lower() == "mcdonalds":
            cleaned = "McDonald's"
        elif "chase bank" in cleaned.lower():
            cleaned = "Chase Bank"
        elif "bank of america" in cleaned.lower():
            cleaned = "Bank of America"
        
        return cleaned
    
    def _apply_category_rules(self, vendor: str, amount: float) -> Dict[str, Any]:
        """Delegate to shared category rules with optional custom overrides."""
        try:
            from category_rules import apply_category_rules
        except Exception:
            return {'matched': False}
        try:
            categorization = self.config.get('categorization', {}) if isinstance(self.config, dict) else {}
            custom = categorization.get('custom_category_rules', {})
        except Exception:
            custom = {}
        return apply_category_rules(vendor, amount, custom)
    
    def _generate_transaction_tags(self, row: pd.Series) -> List[str]:
        """Generate transaction tags"""
        tags = []
        
        # Amount-based tags
        amount = float(row.get('amount', 0))
        if amount > 1000:
            tags.append('high_value')
        elif amount < 10:
            tags.append('low_value')
        
        # Vendor-based tags
        vendor = str(row.get('merchant', '')).lower()
        if any(keyword in vendor for keyword in ['subscription', 'monthly', 'annual']):
            tags.append('subscription')
        if any(keyword in vendor for keyword in ['food', 'restaurant', 'cafe']):
            tags.append('food_dining')
        
        return tags
    
    def _generate_transaction_insights(self, row: pd.Series) -> List[str]:
        """Generate transaction insights"""
        insights = []
        
        amount = float(row.get('amount', 0))
        vendor = str(row.get('merchant', ''))
        
        if amount > 500:
            insights.append(f"High-value transaction: ${amount:.2f} at {vendor}")
        
        if 'subscription' in vendor.lower():
            insights.append("Recurring subscription payment detected")
        
        return insights
    
    def _generate_transaction_explanation(self, row: pd.Series) -> str:
        """Generate transaction explainability"""
        amount = float(row.get('amount', 0))
        vendor = str(row.get('merchant', ''))
        
        if amount > 1000:
            return f"High-value transaction requiring attention"
        elif amount < 10:
            return f"Low-value transaction, likely incidental"
        else:
            return f"Standard transaction amount"
    
    def _calculate_transaction_risk(self, row: pd.Series) -> float:
        """Calculate transaction risk score (0-1)"""
        risk_score = 0.0
        
        amount = float(row.get('amount', 0))
        if amount > 1000:
            risk_score += 0.3
        if amount > 5000:
            risk_score += 0.4
        
        vendor = str(row.get('merchant', '')).lower()
        if any(keyword in vendor for keyword in ['unknown', 'unrecognized']):
            risk_score += 0.2
        
        return min(1.0, risk_score)
    
    def _detect_transaction_anomalies(self, row: pd.Series) -> Dict[str, Any]:
        """Detect transaction anomalies"""
        anomalies = {}
        
        amount = float(row.get('amount', 0))
        vendor = str(row.get('merchant', ''))
        
        # Amount anomalies
        if amount > 10000:
            anomalies['high_amount'] = True
        if amount < 0:
            anomalies['negative_amount'] = True
        
        # Vendor anomalies
        if 'unknown' in vendor.lower() or vendor.strip() == '':
            anomalies['unknown_vendor'] = True
        
        return anomalies
    
    def get_processing_summary(self) -> Dict[str, Any]:
        """Get comprehensive processing summary"""
        return {
            'cache_stats': self.cache.get_cache_stats(),
            'tracking_summary': self.tracker.get_tracking_summary(),
            'processing_history': self.processing_history
        } 