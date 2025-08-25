#!/usr/bin/env python3
"""
Find the 7 "Other" category cases from the stress test
"""

import pandas as pd
import requests
import json

def find_other_categories():
    print('🕵️ HUNTING DOWN THE 7 "OTHER" CATEGORIES...')
    print('What vendors are still challenging our system?')
    print()

    # Quick test to find the Others
    df = pd.read_csv('test_500_rows.csv').head(500).fillna('')
    df['Amount'] = pd.to_numeric(df['Amount'], errors='coerce').fillna(0)
    test_data = df.to_dict('records')

    payload = {'user_intent': 'comprehensive cleaning and standardization', 'data': test_data}
    response = requests.post('http://localhost:8080/process', json=payload, timeout=60)

    if response.status_code == 200:
        result = response.json()
        cleaned_data = result.get('cleaned_data', [])
        
        other_cases = []
        for i, row in enumerate(cleaned_data):
            if row.get('category') == 'Other':
                original_merchant = test_data[i].get('Merchant', '')
                vendor = row.get('standardized_vendor', '')
                amount = test_data[i].get('Amount', 0)
                other_cases.append({
                    'row': i + 1,
                    'original': original_merchant,
                    'vendor': vendor,
                    'category': row.get('category', ''),
                    'amount': amount
                })
        
        print(f'🎯 FOUND {len(other_cases)} "OTHER" CATEGORY CASES:')
        print('These could be improved with targeted LLM calls or better rules:')
        print()
        
        for case in other_cases:
            row_num = case['row']
            original = case['original']
            vendor = case['vendor']
            amount = case['amount']
            print(f'Row {row_num:3d}: {original:30s} → {vendor:20s} | ${amount}')
        
        print(f'\n📈 IMPROVEMENT OPPORTUNITY:')
        print(f'   Current: 98.6% proper categories')
        print(f'   Potential: 100% with {len(other_cases)} targeted LLM calls')
        print(f'   Cost: ~${len(other_cases) * 0.01:.2f} (1¢ per LLM call)')
        print(f'   Speed impact: Minimal (only {len(other_cases)} out of 500)')
        
        if len(other_cases) <= 10:
            print(f'\n✨ These edge cases are PERFECT for selective LLM enhancement!')
            print(f'   Very manageable number of unknown vendors')
            
        # Show the most common patterns
        vendors = [case['vendor'] for case in other_cases]
        print(f'\n🔍 VENDOR PATTERNS:')
        for vendor in set(vendors):
            count = vendors.count(vendor)
            print(f'   {vendor} ({count} times)')
            
        return other_cases
    else:
        print('Error fetching data')
        return []

if __name__ == "__main__":
    find_other_categories()