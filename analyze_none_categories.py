#!/usr/bin/env python3
"""
Analyze None categories to identify where we need LLM calls
"""

import os
import pandas as pd
import requests
import json

def analyze_none_categories():
    print('🔍 ANALYZING NONE CATEGORIES...')
    print('=' * 50)
    
    # Process 50 rows to identify None patterns
    df = pd.read_csv('test_500_rows.csv').head(50).fillna('')
    df['Amount'] = pd.to_numeric(df['Amount'], errors='coerce').fillna(0) 
    test_data = df.to_dict('records')

    payload = {
        'user_intent': 'comprehensive cleaning and standardization',
        'data': test_data,
        'config': {
            'preserve_schema': False,
            'enable_ai': True,
            'ai_vendor_enabled': True,
            'ai_category_enabled': True,
            'ai_confidence_threshold': 0.7
        }
    }
    base_url = os.getenv('API_BASE_URL') or f"http://localhost:{os.getenv('PORT', '8080')}"
    response = requests.post(f"{base_url}/process", json=payload, timeout=60)

    if response.status_code == 200:
        result = response.json()
        cleaned_data = result.get('cleaned_data') or result.get('clean_data') or []
        
        none_cases = []
        for i, row in enumerate(cleaned_data):
            category = row.get('category', '')
            if not category or category == 'None' or category == 'Other':
                original_merchant = test_data[i].get('Merchant', '')
                vendor = row.get('standardized_vendor', '')
                amount = row.get('Amount', 0)
                vendor_confidence = row.get('vendor_confidence', 0)
                none_cases.append({
                    'row': i,
                    'original': original_merchant,
                    'vendor': vendor,
                    'amount': amount,
                    'category': category,
                    'vendor_confidence': vendor_confidence
                })
        
        print(f'Found {len(none_cases)} None/Other categories:')
        print()
        
        for case in none_cases[:10]:  # Show first 10
            row_num = case['row'] + 1
            orig = str(case['original'])[:25]
            vendor = str(case['vendor'])[:20]
            cat = str(case['category']) if case['category'] else 'None'
            amt = case['amount'] if case['amount'] else 0
            conf = case['vendor_confidence'] if case['vendor_confidence'] else 0
            print(f'{row_num:2d}. {orig:25s} → {vendor:20s} | {cat:10s} | ${amt:6.0f} | conf:{conf:.2f}')
        
        if len(none_cases) > 10:
            print(f'    ... and {len(none_cases) - 10} more')
        
        total_rows = len(cleaned_data)
        if total_rows == 0:
            print('\nNone rate: 0/0 (N/A) — no rows returned by API')
            return []
        none_rate = (len(none_cases) / total_rows) * 100
        print(f'\nNone rate: {len(none_cases)}/{total_rows} ({none_rate:.1f}%)')
        
        # Analyze patterns
        print('\n🎯 VENDORS NEEDING LLM CALLS:')
        unique_vendors = {}
        for case in none_cases:
            vendor = case['vendor']
            if vendor not in unique_vendors:
                unique_vendors[vendor] = []
            unique_vendors[vendor].append(case)
        
        for vendor, cases in unique_vendors.items():
            count = len(cases)
            avg_conf = sum(c['vendor_confidence'] for c in cases) / count
            print(f'   {vendor} ({count} times, avg conf: {avg_conf:.2f})')
        
        print(f'\n📊 SUMMARY:')
        print(f'   Total None categories: {len(none_cases)}')
        print(f'   Unique vendors needing help: {len(unique_vendors)}')
        print(f'   Target for LLM calls: {len(none_cases)} transactions')
        
        return none_cases

    else:
        print(f'Error: {response.status_code}')
        return []

if __name__ == "__main__":
    analyze_none_categories()