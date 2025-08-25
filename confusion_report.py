#!/usr/bin/env python3
import os
import csv
import json
import re
import requests
from collections import Counter, defaultdict

API_BASE = os.getenv('API_BASE_URL', 'http://localhost:8102')
TAKE = int(os.getenv('TAKE_ROWS', '100'))

ALLOWED = {
    'Software & Technology','Meals & Entertainment','Travel & Transportation',
    'Office Supplies & Equipment','Professional Services','Banking & Finance',
    'Utilities & Rent','Marketing & Advertising','Employee Benefits','Insurance & Legal','Other'
}

def norm_vendor(s: str) -> str:
    if not s:
        return ''
    s = s.lower().strip()
    s = re.sub(r"[^a-z0-9 &]+","", s)
    s = re.sub(r"\s+"," ", s)
    return s

def load_rows(csv_path: str, take: int):
    rows = []
    with open(csv_path, 'r') as f:
        r = csv.DictReader(f)
        for i, row in enumerate(r):
            try:
                row['Amount'] = float(row.get('Amount') or 0)
            except Exception:
                row['Amount'] = 0.0
            rows.append(row)
            if take and len(rows) >= take:
                break
    return rows

def main():
    rows = load_rows('test_500_rows.csv', TAKE)
    expected = json.load(open('test_500_rows_expected.json'))[:len(rows)]

    payload = {
        'user_intent': 'comprehensive cleaning and standardization',
        'data': rows,
        'config': {
            'enable_ai': True,
            'ai_vendor_enabled': True,
            'ai_category_enabled': True,
            'preserve_schema': False,
            'enable_parallel_processing': True
        }
    }
    resp = requests.post(f'{API_BASE}/process', json=payload, timeout=1800)
    resp.raise_for_status()
    cleaned = resp.json().get('cleaned_data', [])

    disagreements = []
    vendor_suggestions = defaultdict(Counter)
    for i, row in enumerate(cleaned):
        got_vendor = (row.get('standardized_vendor') or row.get('Clean Vendor') or row.get('Vendor') or row.get('merchant') or row.get('Merchant') or '').strip()
        got_cat = (row.get('category') or row.get('Category') or '').strip()
        exp = expected[i]
        exp_vendor = exp.get('expected_clean_merchant','').strip()
        exp_cat = exp.get('expected_category','').strip()
        if got_cat != exp_cat:
            disagreements.append((got_vendor, got_cat, exp_vendor, exp_cat))
        # Count expected category for the normalized vendor to propose mapping
        nv = norm_vendor(exp_vendor)
        if nv and exp_cat:
            vendor_suggestions[nv][exp_cat] += 1

    # Build suggestion map: pick most common expected category per vendor
    suggestions = {}
    for v, cnt in vendor_suggestions.items():
        cat, _ = cnt.most_common(1)[0]
        suggestions[v] = cat

    # Save suggestions
    out_path = 'gold_vendor_suggestions.json'
    with open(out_path, 'w') as f:
        json.dump(suggestions, f, indent=2)

    # Brief report
    print(json.dumps({
        'rows': len(cleaned),
        'disagreements': len(disagreements),
        'unique_vendors_suggested': len(suggestions),
        'suggestions_file': out_path
    }, indent=2))

if __name__ == '__main__':
    main()


