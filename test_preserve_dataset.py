#!/usr/bin/env python3
import os
import json
from typing import List, Dict

import pandas as pd
import requests


def load_dataset(path: str) -> List[Dict]:
    if path.lower().endswith('.csv'):
        df = pd.read_csv(path)
        df = df.fillna("")
        return df.to_dict('records')
    with open(path, 'r') as f:
        data = json.load(f)
    # If file is an object with key 'data', unwrap
    if isinstance(data, dict) and 'data' in data:
        return data['data']
    if isinstance(data, list):
        return data
    return []


def run(dataset_path: str):
    rows = load_dataset(dataset_path)
    print(f"📄 Loaded {len(rows)} rows from {dataset_path}")

    payload = {
        'user_intent': 'standard clean',
        'data': rows,
        'config': {
            'preserve_schema': True,
            'enable_ai': True,
            'ai_vendor_enabled': True,
            'ai_category_enabled': True,
            'ai_preserve_schema_apply': True,
            'ai_mode': 'apply',
            'ai_confidence_threshold': 0.6
        }
    }

    base_url = os.getenv('API_BASE_URL', 'http://127.0.0.1:8081')
    r = requests.post(f"{base_url}/process", json=payload, timeout=180)
    print(f"HTTP {r.status_code}")
    if r.status_code != 200:
        print(r.text)
        return False

    resp = r.json()
    cleaned = resp.get('cleaned_data', [])
    print(f"✅ Returned {len(cleaned)} rows")

    # Print a quick sample of key fields without assuming exact column names
    def get_first(row: Dict, keys):
        for k in keys:
            if k in row:
                return row[k]
        return ''

    print("\nPreview (up to 10 rows):")
    for i, row in enumerate(cleaned[:10], 1):
        date = get_first(row, ['Date', 'Transaction Date', 'posted', 'date'])
        amount = get_first(row, ['Amount', 'amount'])
        vendor = get_first(row, ['standardized_vendor', 'Clean Vendor', 'Merchant', 'merchant'])
        category = get_first(row, ['Category', 'category'])
        print(json.dumps({'#': i, 'Date': date, 'Amount': amount, 'Vendor': vendor, 'Category': category}))
    return True


if __name__ == '__main__':
    dataset = os.getenv('DATASET') or 'test_financial_data.csv'
    ok = run(dataset)
    raise SystemExit(0 if ok else 1)


