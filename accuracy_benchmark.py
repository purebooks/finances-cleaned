import json
import os
import time
import re
from typing import List, Dict, Any, Tuple

import pandas as pd
import requests


ALLOWED_CATEGORIES = set([
    'Software & Technology', 'Meals & Entertainment', 'Travel & Transportation',
    'Office Supplies & Equipment', 'Professional Services', 'Banking & Finance',
    'Utilities & Rent', 'Marketing & Advertising', 'Employee Benefits', 'Insurance & Legal', 'Other'
])


def normalize_vendor(s: str) -> str:
    if s is None:
        return ''
    s = s.lower()
    s = re.sub(r"[^a-z0-9]+", "", s)
    return s


def post_to_api(rows: List[Dict[str, Any]], config: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], Dict[str, Any], float]:
    payload = {
        'user_intent': 'comprehensive cleaning and standardization',
        'data': rows,
        'config': config,
    }
    start = time.time()
    base_url = os.getenv('API_BASE_URL', 'http://localhost:8080')
    resp = requests.post(f'{base_url}/process', json=payload, timeout=1800)
    elapsed = time.time() - start
    resp.raise_for_status()
    js = resp.json()
    return js.get('cleaned_data', []), js.get('insights', {}), elapsed


def compute_metrics(cleaned: List[Dict[str, Any]], expected: List[Dict[str, Any]], original_rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    total = len(cleaned)
    none_count = 0
    other_count = 0
    invalid_count = 0
    cat_matches = 0
    vend_matches = 0

    # Vendor normalization rate vs original merchant
    normalized_changes = 0
    comparable = min(len(cleaned), len(expected))

    def _row_category(r: Dict[str, Any]) -> str:
        return (r.get('category') or r.get('Category') or '').strip()

    def _row_vendor(r: Dict[str, Any]) -> str:
        return (r.get('standardized_vendor') or r.get('Clean Vendor') or r.get('Vendor') or r.get('merchant') or r.get('Merchant') or '').strip()

    for i, row in enumerate(cleaned):
        cat = _row_category(row)
        if not cat or cat.lower() == 'none':
            none_count += 1
        if cat == 'Other':
            other_count += 1
        if cat and cat not in ALLOWED_CATEGORIES:
            invalid_count += 1

        if i < comparable:
            exp = expected[i]
            # Expected schema uses 'expected_clean_merchant' and 'expected_category'
            if normalize_vendor(_row_vendor(row)) == normalize_vendor(exp.get('expected_clean_merchant')):
                vend_matches += 1
            if _row_category(row) == (exp.get('expected_category') or '').strip():
                cat_matches += 1

        # normalization vs original vendor text
        try:
            orig = original_rows[i]
            orig_merchant = str(orig.get('Merchant', '') or orig.get('merchant', '') or '')
            std_ven = _row_vendor(row)
            if std_ven and std_ven != orig_merchant.strip():
                normalized_changes += 1
        except Exception:
            pass

    metrics = {
        'rows': total,
        'none_percent': round((none_count / total) * 100, 2) if total else 0.0,
        'other_percent': round((other_count / total) * 100, 2) if total else 0.0,
        'invalid_percent': round((invalid_count / total) * 100, 2) if total else 0.0,
    }

    if comparable:
        metrics.update({
            'vendor_accuracy': round((vend_matches / comparable) * 100, 2),
            'category_accuracy': round((cat_matches / comparable) * 100, 2),
        })
    metrics['vendor_normalization_rate'] = round((normalized_changes / total) * 100, 2) if total else 0.0
    return metrics


def run_benchmark(dataset_path: str, expected_path: str = '', take: int = 0, config: Dict[str, Any] = None) -> Dict[str, Any]:
    if dataset_path.endswith('.csv'):
        df = pd.read_csv(dataset_path).fillna('')
        df['Amount'] = pd.to_numeric(df.get('Amount', 0), errors='coerce').fillna(0)
        rows = df.to_dict('records')
    else:
        with open(dataset_path, 'r') as f:
            obj = json.load(f)
        rows = obj['data'] if isinstance(obj, dict) and 'data' in obj else obj

    if take and len(rows) > take:
        rows = rows[:take]

    expected = []
    if expected_path:
        with open(expected_path, 'r') as f:
            expected = json.load(f)
        if take and len(expected) > take:
            expected = expected[:take]

    cfg = config or {
        # App flags that actually control AI usage
        'enable_ai': True,
        'ai_vendor_enabled': True,
        'ai_category_enabled': True,
        'preserve_schema': False,

        # Performance and behavior
        'enable_parallel_processing': True,
        'max_workers': 8,
        'enable_transaction_intelligence': False,

        # Accuracy-first knobs used by our cleaner
        'accuracy_first': True,
        'quality_slo_proper_min': 0.97,
        'quality_slo_none_max': 0.0,
        'llm_trigger_amount': 0.0,
        'llm_parallel_conf_threshold': 0.90,
        'llm_route_unknown': True,
        'llm_minimum_fraction': 0.0,
        'llm_max_enhancements': 180,
        'llm_batch_size': 10,
        'llm_cost_cap_per_request': 1.5,
        'emergency_llm_cap': 0.0
    }

    cleaned, insights, elapsed = post_to_api(rows, cfg)
    metrics = compute_metrics(cleaned, expected, rows)
    # add timing/cost
    metrics.update({
        'ai_requests': insights.get('ai_requests', 0),
        'ai_cost_total': insights.get('ai_cost', 0.0),
        'ai_cost_per_row': round((insights.get('ai_cost', 0.0) or 0.0) / max(1, len(cleaned)), 6),
        'time_s': round(elapsed, 2),
        'time_per_row_s': round(elapsed / max(1, len(cleaned)), 4)
    })
    return metrics


if __name__ == '__main__':
    # Example benchmark on 500-row golden set
    result = run_benchmark('test_500_rows.csv', expected_path='test_500_rows_expected.json', take=500)
    print(json.dumps(result, indent=2))

