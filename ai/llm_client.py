#!/usr/bin/env python3

import os
import json
import time
import hashlib
import logging
from typing import Optional, Dict, Any, List

try:
	from cachetools import LRUCache
except Exception:  # fallback minimal cache
	LRUCache = dict  # type: ignore

logger = logging.getLogger(__name__)


class LLMClient:
	"""Unified LLM client with caching and optional Anthropic provider.

	Methods used by the app:
	- resolve_vendor(merchant, description='', memo='') -> str
	- suggest_category(merchant, description='', amount=0.0, memo='') -> Dict[str, Any]
	- analyze_transaction(...): Optional extended analysis
	- get_usage_stats(), reset_metrics(), clear_cache()

	When use_mock=True or API keys are missing, returns heuristic mock outputs.
	"""

	def __init__(
		self,
		api_key: Optional[str] = None,
		use_mock: bool = False,
		enable_caching: bool = True,
		cache_size: int = 1000,
		provider: str = "anthropic",
	):
		self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
		self.use_mock = use_mock or not bool(self.api_key)
		self.enable_caching = enable_caching
		self.cache_size = max(1, cache_size)
		self.provider = provider
		self.request_count = 0
		self.total_cost = 0.0

		# Client init
		self.client = None
		if not self.use_mock and self.provider == "anthropic":
			try:
				import anthropic  # lazy import
				self.client = anthropic.Anthropic(api_key=self.api_key)
			except Exception as e:
				logger.warning(f"Anthropic client init failed, using mock: {e}")
				self.use_mock = True

		# Caches
		if self.enable_caching:
			try:
				self._vendor_cache = LRUCache(maxsize=self.cache_size)  # type: ignore[arg-type]
				self._category_cache = LRUCache(maxsize=self.cache_size)  # type: ignore[arg-type]
			except TypeError:  # dict fallback
				self._vendor_cache = {}
				self._category_cache = {}
		else:
			self._vendor_cache = {}
			self._category_cache = {}

	def _cache_get(self, cache: dict, key: str):
		if not self.enable_caching:
			return None
		return cache.get(key)

	def _cache_set(self, cache: dict, key: str, value: Any):
		if not self.enable_caching:
			return
		try:
			cache[key] = value
		except Exception:
			# dict fallback uses simple eviction by clearing oldest key
			if len(cache) >= self.cache_size:
				oldest = next(iter(cache))
				del cache[oldest]
			cache[key] = value

	def _mock_vendor(self, text: str) -> str:
		low = (text or '').lower()
		mapping = {
			'google': 'Google', 'meta': 'Meta', 'facebook': 'Meta', 'amazon': 'Amazon',
			'digitalocean': 'DigitalOcean', 'stripe': 'Stripe', 'netflix': 'Netflix',
			'spotify': 'Spotify', 'apple': 'Apple', 'microsoft': 'Microsoft', 'adobe': 'Adobe',
			'salesforce': 'Salesforce', 'dropbox': 'Dropbox', 'uber': 'Uber', 'lyft': 'Lyft',
			'starbucks': 'Starbucks', 'chipotle': 'Chipotle', 'target': 'Target', 'walmart': 'Walmart',
			'cvs': 'CVS Pharmacy', 'walgreens': 'Walgreens', 'airbnb': 'Airbnb',
		}
		for k, v in mapping.items():
			if k in low:
				return v
		# strip payment processor prefixes
		clean = (text or '').replace('PAYPAL*', '').replace('SQ *', '').replace('AUTO PAY', '').strip()
		return clean or 'Unknown Vendor'

	def _mock_category(self, text: str) -> Dict[str, Any]:
		low = (text or '').lower()
		if any(k in low for k in ['google', 'microsoft', 'adobe', 'aws', 'digitalocean', 'github', 'slack', 'zoom']):
			return {"category": "Software & Technology", "confidence": 0.9}
		if any(k in low for k in ['starbucks', 'restaurant', 'coffee', 'chipotle']):
			return {"category": "Meals & Entertainment", "confidence": 0.9}
		if any(k in low for k in ['uber', 'lyft', 'shell', 'chevron', 'delta']):
			return {"category": "Travel & Transportation", "confidence": 0.9}
		return {"category": "Other", "confidence": 0.5}

	def _anthropic_call(self, prompt: str, max_tokens: int = 256, temperature: float = 0.2) -> str:
		if self.use_mock or not self.client:
			return ''
		start = time.time()
		resp = self.client.messages.create(
			model="claude-3-5-sonnet-20240620",
			max_tokens=max_tokens,
			temperature=temperature,
			messages=[{"role": "user", "content": prompt}],
		)
		self.request_count += 1
		self.total_cost += 0.01
		logger.debug(f"LLM call took {time.time()-start:.2f}s")
		try:
			return (resp.content[0].text or '').strip()
		except Exception:
			return ''

	def resolve_vendor(self, merchant: str, description: str = "", memo: str = "") -> str:
		if self.use_mock:
			return self._mock_vendor(merchant)
		key = hashlib.md5(f"vendor|{merchant}|{description}|{memo}".encode()).hexdigest()
		cached = self._cache_get(self._vendor_cache, key)
		if cached:
			return cached
		prompt = (
			"You normalize business vendor names. Return ONLY the vendor name.\n"
			"If unclear, return 'Unknown Vendor'.\n\n"
			f"Merchant: {merchant}\nDescription: {description}\nMemo: {memo}\n\nVendor:"
		)
		result = self._anthropic_call(prompt, max_tokens=50, temperature=0.1) or 'Unknown Vendor'
		self._cache_set(self._vendor_cache, key, result)
		return result

	def suggest_category(self, merchant: str, description: str = "", amount: float = 0.0, memo: str = "") -> Dict[str, Any]:
		if self.use_mock:
			return self._mock_category(merchant)
		key = hashlib.md5(f"category|{merchant}|{description}|{amount}|{memo}".encode()).hexdigest()
		cached = self._cache_get(self._category_cache, key)
		if cached:
			return cached
		prompt = (
			"Classify expense into one allowed category. Respond ONLY JSON: {\"category\":...,\"confidence\":0..1}.\n"
			f"Merchant: {merchant}\nDescription: {description}\nAmount: ${amount:.2f}\nMemo: {memo}\n"
		)
		text = self._anthropic_call(prompt, max_tokens=120, temperature=0.2)
		try:
			start = text.find('{'); end = text.rfind('}') + 1
			parsed = json.loads(text[start:end] if start != -1 and end > start else text)
			res = {"category": parsed.get("category", "Other"), "confidence": float(parsed.get("confidence", 0.0))}
		except Exception:
			res = {"category": "Other", "confidence": 0.0}
		self._cache_set(self._category_cache, key, res)
		return res

	def analyze_transaction(self, merchant: str, description: str = "", amount: float = 0.0, memo: str = "") -> Dict[str, Any]:
		if self.use_mock:
			cat = self._mock_category(merchant)
			return {"vendor": self._mock_vendor(merchant), "category": cat["category"], "confidence": cat["confidence"], "business_impact": "Standard expense", "risk_level": "Low", "recommendation": ""}
		prompt = (
			"Analyze transaction, return JSON with vendor, category, confidence, business_impact, risk_level, recommendation.\n"
			f"Merchant:{merchant}\nDescription:{description}\nAmount:${amount:.2f}\nMemo:{memo}\nJSON:"
		)
		text = self._anthropic_call(prompt, max_tokens=200, temperature=0.3)
		try:
			start = text.find('{'); end = text.rfind('}') + 1
			parsed = json.loads(text[start:end] if start != -1 and end > start else text)
			return {
				"vendor": parsed.get("vendor", "Unknown Vendor"),
				"category": parsed.get("category", "Other"),
				"confidence": float(parsed.get("confidence", 0.0)),
				"business_impact": parsed.get("business_impact", "Standard expense"),
				"risk_level": parsed.get("risk_level", "Low"),
				"recommendation": parsed.get("recommendation", ""),
			}
		except Exception:
			return {"vendor": "Unknown Vendor", "category": "Other", "confidence": 0.0, "business_impact": "Standard expense", "risk_level": "Low", "recommendation": ""}

	def get_usage_stats(self) -> Dict[str, Any]:
		avg = self.total_cost / max(self.request_count, 1)
		return {"request_count": self.request_count, "total_cost": self.total_cost, "average_cost_per_request": avg}

	def reset_metrics(self):
		self.request_count = 0
		self.total_cost = 0.0

	def clear_cache(self):
		try:
			self._vendor_cache.clear()
			self._category_cache.clear()
		except Exception:
			pass

