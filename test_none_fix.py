#!/usr/bin/env python3
"""
Test the None category fix
"""

import requests
import json
import time

def test_none_fix():
    print('🧪 TESTING NONE CATEGORY FIXES...')
    print('=' * 50)
    
    # Test with the specific vendors that were showing None
    test_data = [
        {'Merchant': 'Budget INC', 'Amount': 89.99, 'Date': '2024-01-06', 'Notes': ''},
        {'Merchant': 'Apple', 'Amount': 4.99, 'Date': '2024-01-05', 'Notes': ''},
        {'Merchant': 'Google', 'Amount': 15.99, 'Date': '2024-01-02', 'Notes': ''},
        {'Merchant': 'SQ *Netflix', 'Amount': 15.99, 'Date': '2024-01-02', 'Notes': ''}
    ]

    payload = {'user_intent': 'comprehensive cleaning and standardization', 'data': test_data}

    start_time = time.time()
    response = requests.post('http://localhost:8080/process', json=payload, timeout=60)
    total_time = time.time() - start_time

    if response.status_code == 200:
        result = response.json()
        cleaned_data = result.get('cleaned_data', [])
        insights = result.get('insights', {})
        
        print(f'⏱️  Processing time: {total_time:.2f}s')
        print(f'🤖 AI requests: {insights.get("ai_requests", 0)}')
        print(f'💰 AI cost: ${insights.get("ai_cost", 0):.4f}')
        print()
        
        print('RESULTS:')
        for i, row in enumerate(cleaned_data):
            original = test_data[i]['Merchant']
            vendor = row.get('standardized_vendor', 'N/A')
            category = row.get('category', 'N/A')
            source = row.get('category_source', 'N/A')
            print(f'{i+1}. {original:20s} → {vendor:15s} | {category:20s} | {source}')
        
        # Check results
        none_count = sum(1 for row in cleaned_data if not row.get('category') or row.get('category') == 'None')
        print(f'\\nNone categories remaining: {none_count}/{len(cleaned_data)}')
        
        if none_count == 0:
            print('🎉 SUCCESS! All None categories eliminated!')
        elif insights.get('ai_requests', 0) > 0:
            print('✅ LLM calls triggered, but some still None')
        else:
            print('❌ No LLM calls - need to debug rule-based system')
        
        return none_count

    else:
        print(f'Error: {response.status_code} - {response.text}')
        return -1

if __name__ == "__main__":
    test_none_fix()