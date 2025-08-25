import os
import json
import time
import logging
from typing import Any, Dict, List, Optional

try:
    import anthropic
except Exception:
    anthropic = None  # Optional

try:
    from openai import OpenAI
except Exception:
    OpenAI = None  # Optional


logger = logging.getLogger(__name__)


class _AnthropicProvider:
    def __init__(self, api_key: Optional[str]):
        if not anthropic or not api_key:
            self.client = None
        else:
            self.client = anthropic.Anthropic(api_key=api_key)

    def available(self) -> bool:
        return self.client is not None

    def call(self, prompt: str, model: str = "claude-3-5-sonnet-20240620", max_tokens: int = 512, temperature: float = 0.2) -> str:
        if not self.client:
            raise RuntimeError("Anthropic provider not available")
        resp = self.client.messages.create(
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=[{"role": "user", "content": prompt}]
        )
        try:
            return resp.content[0].text
        except Exception:
            return ""


class _OpenAIProvider:
    def __init__(self, api_key: Optional[str]):
        if not OpenAI or not api_key:
            self.client = None
        else:
            # OpenAI client picks up api key automatically, but pass explicitly for clarity
            self.client = OpenAI(api_key=api_key)

    def available(self) -> bool:
        return self.client is not None

    def call(self, prompt: str, model: str = "gpt-4o-mini", max_tokens: int = 512, temperature: float = 0.2) -> str:
        if not self.client:
            raise RuntimeError("OpenAI provider not available")
        resp = self.client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        try:
            return resp.choices[0].message.content or ""
        except Exception:
            return ""


class MultiLLMOrchestrator:
    """Multi-LLM router with task-aware provider selection and fallbacks.

    Drop-in compatible with existing LLM client usage in this codebase. Implements:
    - resolve_vendor(merchant, description="", memo="") -> str
    - suggest_category(merchant, description="", amount=0.0, memo="") -> {category, confidence}
    - process_transaction_batch(rows) -> list of {standardized_vendor, category, confidence, ...}
    - suggest_vendors_from_description(description, amount) -> list[str]
    - _make_llm_call(prompt) -> str (generic)

    Exposes:
    - use_mock: bool
    - get_usage_stats() / reset_metrics()
    """

    def __init__(
        self,
        use_mock: bool = False,
        enable_caching: bool = True,
        cache_size: int = 2048,
        primary_vendor_provider: Optional[str] = None,
        primary_category_provider: Optional[str] = None,
    ):
        self.use_mock = use_mock
        self.enable_caching = enable_caching
        self.cache_size = max(1, cache_size)
        self._generic_cache: Dict[str, str] = {}
        self._vendor_cache: Dict[str, str] = {}
        self._category_cache: Dict[str, Dict[str, Any]] = {}
        self._cache_fifo: List[str] = []

        # Providers
        anthropic_key = os.getenv("ANTHROPIC_API_KEY")
        openai_key = os.getenv("OPENAI_API_KEY")
        self._anthropic = _AnthropicProvider(anthropic_key)
        self._openai = _OpenAIProvider(openai_key)

        # Defaults and routing preferences
        self.primary_vendor_provider = (primary_vendor_provider or os.getenv("PRIMARY_VENDOR_PROVIDER") or "anthropic").lower()
        self.primary_category_provider = (primary_category_provider or os.getenv("PRIMARY_CATEGORY_PROVIDER") or "anthropic").lower()

        # Metrics
        self._total_calls = 0
        self._total_cost_estimate = 0.0
        self._total_time = 0.0

    # ---------- Public API (compatibility) ----------
    def resolve_vendor(self, merchant: str, description: str = "", memo: str = "") -> str:
        if not merchant:
            return "Unknown Vendor"

        cleaned = self._clean_vendor_name(merchant)
        if cleaned and cleaned.lower() not in {"subscription", "plan", "license", "licence", "software"}:
            # Quick rule-based normalization first
            if self._looks_like_payment_processor_prefix(merchant):
                return cleaned
            # Cache hit
            cached = self._vendor_cache.get(cleaned)
            if cached:
                return cached

        if self.use_mock or not self._any_provider_available():
            result = cleaned or "Unknown Vendor"
            self._vendor_cache[merchant] = result
            return result

        prompt = self._build_vendor_prompt(merchant, description, memo)
        text = self._route_call(prompt, task="vendor")
        result = self._parse_single_line(text) or cleaned or "Unknown Vendor"
        # Guard against generic answers
        low = result.strip().lower()
        if low in {"subscription", "team plan", "annual plan", "software", "pro license", "pro licence", "plan", "license", "licence", "description", "memo"}:
            result = "Unknown Vendor"
        self._vendor_cache[merchant] = result
        return result

    def suggest_category(self, merchant: str, description: str = "", amount: float = 0.0, memo: str = "") -> Dict[str, Any]:
        # Heuristic fast-path to avoid LLM when obvious
        fast_category = self._fast_category(merchant)
        if fast_category:
            return {"category": fast_category, "confidence": 0.9}

        cache_key = f"{merchant}|{self._amount_bucket(amount)}"
        cached = self._category_cache.get(cache_key)
        if cached:
            return cached

        if self.use_mock or not self._any_provider_available():
            return {"category": fast_category or "Other", "confidence": 0.6 if fast_category else 0.5}

        prompt = self._build_category_prompt(merchant, description, amount, memo)
        text = self._route_call(prompt, task="category")
        parsed = self._parse_category_json(text)
        if not parsed:
            parsed = {"category": fast_category or "Other", "confidence": 0.5}
        self._category_cache[cache_key] = parsed
        return parsed

    def process_transaction_batch(self, rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not rows:
            return []

        # Try to avoid LLM for each row by using heuristics first
        provisional: List[Dict[str, Any]] = []
        need_llm_indices: List[int] = []
        for i, row in enumerate(rows):
            merchant = str(row.get('merchant', '') or '')
            standardized = self._clean_vendor_name(merchant)
            category = self._fast_category(standardized)
            confidence = 0.85 if category else 0.6
            result = {
                "original_merchant": merchant,
                "standardized_vendor": standardized or "Unknown Vendor",
                "category": category or "Other",
                "confidence": confidence,
                "processing_source": "rules"
            }
            
            if (not category) or (confidence < 0.7) or (result["standardized_vendor"] == "Unknown Vendor"):
                need_llm_indices.append(i)
            provisional.append(result)

        if self.use_mock or not self._any_provider_available() or not need_llm_indices:
            return provisional

        # Batch prompt for LLM enhancement
        merchant_lines = []
        for idx in need_llm_indices:
            merchant_lines.append(f"{len(merchant_lines)+1}. {provisional[idx]['original_merchant']}")
        prompt = self._build_batch_prompt(merchant_lines)
        text = self._route_call(prompt, task="batch")
        try:
            batch_results = json.loads(text)
        except Exception:
            # If parsing fails, return provisional
            return provisional

        for j, idx in enumerate(need_llm_indices):
            if j < len(batch_results):
                br = batch_results[j]
                provisional[idx].update({
                    "standardized_vendor": br.get("vendor", provisional[idx]["standardized_vendor"]),
                    "category": br.get("category", provisional[idx]["category"]),
                    "confidence": br.get("confidence", 0.9),
                    "processing_source": "llm_batch"
                })
        return provisional

    def suggest_vendors_from_description(self, description: str, amount: float) -> List[str]:
        if not description:
            return []
        if self.use_mock or not self._any_provider_available():
            # Lightweight heuristic suggestions
            base = description.strip()[:20]
            return [f"Mock Vendor for '{base}'", "Google", "Amazon"]

        prompt = self._build_vendor_suggestion_prompt(description, amount)
        text = self._route_call(prompt, task="vendor_suggestions")
        try:
            start = text.find('[')
            end = text.rfind(']') + 1
            if start != -1 and end > start:
                suggestions = json.loads(text[start:end])
                if isinstance(suggestions, list) and all(isinstance(s, str) for s in suggestions):
                    return suggestions[:3]
        except Exception:
            pass
        return []

    def _make_llm_call(self, prompt: str) -> str:
        # Generic call used by category batch enhancement in v5 cleaner
        if self.use_mock or not self._any_provider_available():
            return "[]"
        return self._route_call(prompt, task="generic")

    def get_usage_stats(self) -> Dict[str, Any]:
        return {
            "total_calls": self._total_calls,
            "total_cost_estimate": self._total_cost_estimate,
            "average_response_time": (self._total_time / self._total_calls) if self._total_calls else 0.0,
        }

    def reset_metrics(self) -> None:
        self._total_calls = 0
        self._total_cost_estimate = 0.0
        self._total_time = 0.0

    # ---------- Routing and helpers ----------
    def _route_call(self, prompt: str, task: str) -> str:
        # Cache generic prompts when enabled
        if self.enable_caching:
            cached = self._generic_cache.get(prompt)
            if cached is not None:
                return cached

        start = time.time()

        providers_order = self._provider_order_for_task(task)
        text = ""
        for provider_name in providers_order:
            try:
                if provider_name == "anthropic" and self._anthropic.available():
                    text = self._anthropic.call(prompt)
                elif provider_name == "openai" and self._openai.available():
                    text = self._openai.call(prompt)
                if text:
                    break
            except Exception as e:
                logger.warning(f"Provider {provider_name} failed for task {task}: {e}")
                continue

        elapsed = time.time() - start
        self._total_calls += 1
        # Rough, conservative cost estimates per call
        self._total_cost_estimate += 0.005
        self._total_time += elapsed

        if self.enable_caching and text:
            self._put_cache(prompt, text)
        return text

    def _provider_order_for_task(self, task: str) -> List[str]:
        if task == "vendor":
            primary = self.primary_vendor_provider
        elif task in {"category", "batch", "vendor_suggestions", "generic"}:
            primary = self.primary_category_provider
        else:
            primary = "anthropic"
        fallbacks = [p for p in ["anthropic", "openai"] if p != primary]
        return [primary] + fallbacks

    def _put_cache(self, key: str, value: str) -> None:
        self._generic_cache[key] = value
        self._cache_fifo.append(key)
        if len(self._generic_cache) > self.cache_size:
            oldest = self._cache_fifo.pop(0)
            try:
                del self._generic_cache[oldest]
            except Exception:
                pass

    # ---------- Prompt builders ----------
    def _build_vendor_prompt(self, merchant: str, description: str, memo: str) -> str:
        return f"""
You are a financial data standardizer specializing in business vendor normalization.
Given the raw vendor details below, return the most likely normalized vendor NAME.

CRITICAL RULES:
- Never answer with generic product or plan words (e.g., Subscription, Team Plan, Annual Plan, Software, Pro License, Plan, License, Licence).
- Prefer the company/brand (e.g., DigitalOcean, Amazon Web Services, Airtable, Stripe, PayPal, LinkedIn, Spotify).
- If unclear or generic, respond exactly with: Unknown Vendor

Return ONLY the normalized vendor name, nothing else.

Merchant: {merchant}
Description: {description}
Memo: {memo}

Normalized vendor name:
"""

    def _build_category_prompt(self, merchant: str, description: str, amount: float, memo: str) -> str:
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
        return f"""
You are a business expense classifier for financial data analysis.
Given the transaction info below, return the most likely expense category.

Available categories: {categories}

Merchant: {merchant}
Description: {description}
Amount: ${amount:,.2f}
Memo: {memo}

Respond in JSON format like this:
{{ "category": "Software & Technology", "confidence": 0.88 }}

JSON response:
"""

    def _build_batch_prompt(self, merchant_lines: List[str]) -> str:
        return f"""
You are a financial transaction classifier. Analyze these merchant names and provide standardized vendor names and categories.

MERCHANTS TO ANALYZE:
{os.linesep.join(merchant_lines)}

Return JSON array format:
[
  {{"vendor": "standardized_name_1", "category": "category_1", "confidence": 0.95}},
  {{"vendor": "standardized_name_2", "category": "category_2", "confidence": 0.90}}
]

Categories: Software & Technology, Meals & Entertainment, Travel & Transportation, 
Office Supplies & Equipment, Professional Services, Banking & Finance, Utilities & Rent, Other
"""

    def _build_vendor_suggestion_prompt(self, description: str, amount: float) -> str:
        return f"""
You are a financial data expert specializing in inferring missing information. A transaction has a missing vendor. Your task is to predict the 3 most likely vendors based on the provided context.

Analyze the following transaction:
- Description: "{description}"
- Amount: ${amount:.2f}

Return ONLY a JSON array of the three predicted vendor names, ranked from most to least likely.
"""

    # ---------- Parsing helpers ----------
    def _parse_single_line(self, text: str) -> str:
        if not text:
            return ""
        line = text.strip().splitlines()[0].strip().strip('"')
        return line

    def _parse_category_json(self, text: str) -> Optional[Dict[str, Any]]:
        try:
            start = text.find('{')
            end = text.rfind('}') + 1
            if start != -1 and end > start:
                parsed = json.loads(text[start:end])
            else:
                parsed = json.loads(text)
            cat = parsed.get("category", "Other")
            conf_raw = parsed.get("confidence", 0.0)
            try:
                conf = float(conf_raw)
            except Exception:
                conf = 0.0
            return {"category": cat, "confidence": conf}
        except Exception:
            return None

    # ---------- Heuristics and cleaning ----------
    def _looks_like_payment_processor_prefix(self, vendor: str) -> bool:
        if not vendor:
            return False
        v = vendor.upper().strip()
        return any(v.startswith(p) for p in ["PAYPAL *", "PAYPAL*", "SQ *", "TST*", "AUTO PAY ", "AMZ*", "AMAZON*"])

    def _clean_vendor_name(self, input_text: str) -> str:
        if not input_text:
            return ""
        cleaned = str(input_text).strip()
        prefixes = [
            "PAYPAL *", "PAYPAL*", "SQ *", "TST* ", "TST*", "AUTO PAY ",
            "AMZ*", "AMZ *", "AMAZON*", "AMAZON *"
        ]
        for prefix in prefixes:
            if cleaned.upper().startswith(prefix.upper()):
                cleaned = cleaned[len(prefix):].strip()
                break
        suffixes = [
            " INC", " LLC", " CORP", " ONLINE", " .COM", ".COM",
            "*STORE 001", "*STORE", " #123456", "#123456", " STORE",
            " 001", "#001"
        ]
        for suffix in suffixes:
            if cleaned.upper().endswith(suffix.upper()):
                cleaned = cleaned[:-len(suffix)].strip()
        cleaned = cleaned.replace("*", "").replace("#", "").strip()
        if cleaned.lower() == "mcdonalds":
            cleaned = "McDonald's"
        return cleaned

    def _fast_category(self, vendor_text: str) -> Optional[str]:
        if not vendor_text:
            return None
        v = vendor_text.lower()
        tech = ["google", "microsoft", "apple", "adobe", "salesforce", "dropbox", "netflix", "spotify", "github", "slack", "zoom", "aws", "digitalocean", "stripe"]
        if any(t in v for t in tech):
            return "Software & Technology"
        food = ["starbucks", "mcdonald", "chipotle", "subway", "pizza", "domino", "papa john", "whole foods", "safeway", "kroger"]
        if any(f in v for f in food):
            return "Meals & Entertainment"
        if "square" in v:
            return "Meals & Entertainment"
        transport = ["uber", "lyft", "shell", "chevron", "delta", "united", "southwest", "enterprise", "hertz", "budget", "avis", "airlines", "airport"]
        if any(t in v for t in transport):
            return "Travel & Transportation"
        retail = ["amazon", "target", "walmart", "costco", "home depot", "best buy", "office depot", "staples", "amazon web services"]
        if any(r in v for r in retail):
            return "Office Supplies & Equipment"
        health = ["cvs", "walgreens", "pharmacy", "medical", "dental", "doctor", "clinic"]
        if any(h in v for h in health):
            return "Professional Services"
        finance = ["bank", "chase", "wells fargo", "american express", "visa", "mastercard", "paypal", "venmo", "credit union"]
        if any(b in v for b in finance):
            return "Banking & Finance"
        utilities = ["at&t", "att", "verizon", "comcast", "electric", "gas", "water", "internet", "cable", "phone"]
        if any(u in v for u in utilities):
            return "Utilities & Rent"
        travel = ["hilton", "marriott", "airbnb", "hotel", "resort", "booking", "expedia"]
        if any(t in v for t in travel):
            return "Travel & Transportation"
        marketing = ["meta", "facebook", "google ads", "linkedin", "twitter", "x.com", "x "]
        if any(m in v for m in marketing):
            return "Marketing & Advertising"
        return None

    def _amount_bucket(self, amount: float) -> str:
        try:
            a = abs(float(amount))
        except Exception:
            a = 0.0
        # simple bucketing to stabilize cache
        if a < 10:
            return "<10"
        if a < 50:
            return "10-49"
        if a < 100:
            return "50-99"
        if a < 250:
            return "100-249"
        if a < 500:
            return "250-499"
        return ">=500"


