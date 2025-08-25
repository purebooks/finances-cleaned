#!/usr/bin/env python3
"""
Final comprehensive test of speed, accuracy, and None elimination
"""

import pandas as pd
import requests
import json
import time

def run_final_test():
    print('🎯 COMPREHENSIVE FINAL TEST: Speed + Accuracy + None elimination')
    print('=' * 70)

    # Load 100 rows for comprehensive test
    df = pd.read_csv('test_500_rows.csv').head(100).fillna('')
    df['Amount'] = pd.to_numeric(df['Amount'], errors='coerce').fillna(0)
    test_data = df.to_dict('records')

    # Load expected results
    with open('test_500_rows_expected.json', 'r') as f:
        expected = json.load(f)[:100]

    payload = {
        'user_intent': 'comprehensive cleaning and standardization',
        'data': test_data,
        'config': {'use_real_llm': True}
    }

    start_time = time.time()
    response = requests.post('http://localhost:8080/process', json=payload, timeout=120)
    total_time = time.time() - start_time

    if response.status_code == 200:
        result = response.json()
        cleaned_data = result.get('cleaned_data', [])
        insights = result.get('insights', {})
        
        print(f'✅ PROCESSING COMPLETE')
        print(f'⏱️  Time: {total_time:.2f}s (target: <25s)')
        print(f'🤖 AI requests: {insights.get("ai_requests", 0)}')
        print(f'📊 Rows processed: {len(cleaned_data)}/100')
        
        # Analyze None categories
        none_count = 0
        other_count = 0
        proper_categories = 0
        
        none_examples = []
        
        for i, row in enumerate(cleaned_data):
            category = row.get('category', '')
            if not category or category == 'None':
                none_count += 1
                if len(none_examples) < 5:  # Collect first 5 examples
                    none_examples.append({
                        'original': test_data[i].get('Merchant', ''),
                        'vendor': row.get('standardized_vendor', ''),
                        'row': i + 1
                    })
            elif category == 'Other':
                other_count += 1
            else:
                proper_categories += 1
        
        print(f'\n📊 CATEGORY BREAKDOWN:')
        print(f'   Proper categories: {proper_categories}/100 ({proper_categories}%)')
        print(f'   None categories: {none_count}/100 ({none_count}%)')
        print(f'   Other categories: {other_count}/100 ({other_count}%)')
        
        if none_count > 0:
            print(f'\n❌ REMAINING NONE CATEGORIES:')
            for ex in none_examples:
                print(f'   Row {ex["row"]}: {ex["original"]} → {ex["vendor"]}')
        else:
            print('\n🎉 NO NONE CATEGORIES! Perfect category assignment!')
        
        # Calculate rough accuracy
        vendor_correct = 0
        category_correct = 0
        
        for i in range(min(len(cleaned_data), len(expected))):
            # Vendor check
            cleaned_merchant = cleaned_data[i].get('standardized_vendor', '').strip()
            expected_merchant = expected[i]['expected_clean_merchant'].strip()
            
            if (cleaned_merchant.lower() in expected_merchant.lower() or 
                expected_merchant.lower() in cleaned_merchant.lower()):
                vendor_correct += 1
            
            # Category check (non-None only)
            cleaned_category = cleaned_data[i].get('category', '').strip()
            if cleaned_category and cleaned_category != 'None':
                category_correct += 1
        
        vendor_accuracy = (vendor_correct / 100) * 100
        category_assignment_rate = (category_correct / 100) * 100
        
        print(f'\n📈 PERFORMANCE METRICS:')
        print(f'   Vendor Accuracy: {vendor_accuracy:.1f}%')
        print(f'   Category Assignment Rate: {category_assignment_rate:.1f}%')
        print(f'   Speed: {100/total_time:.1f} rows/sec')
        
        # Production readiness assessment
        speed_ok = total_time <= 25
        categories_ok = none_count <= 5  # ≤5% None acceptable
        vendor_ok = vendor_accuracy >= 95
        
        print(f'\n🎯 PRODUCTION READINESS:')
        print(f'   Speed ≤25s: {"✅" if speed_ok else "❌"} ({total_time:.1f}s)')
        print(f'   None ≤5%: {"✅" if categories_ok else "❌"} ({none_count}%)')
        print(f'   Vendor ≥95%: {"✅" if vendor_ok else "❌"} ({vendor_accuracy:.1f}%)')
        
        if speed_ok and categories_ok and vendor_ok:
            print(f'\n🎉 PRODUCTION READY FOR $80/MONTH!')
            print(f'   All targets hit - ready to deploy!')
        else:
            print(f'\n📊 Close to production ready!')
            needed = []
            if not speed_ok: needed.append('speed')
            if not categories_ok: needed.append('categories') 
            if not vendor_ok: needed.append('vendor accuracy')
            print(f'   Still need: {", ".join(needed)}')
        
        return {
            'speed': total_time,
            'none_count': none_count,
            'vendor_accuracy': vendor_accuracy,
            'category_rate': category_assignment_rate,
            'production_ready': speed_ok and categories_ok and vendor_ok
        }

    else:
        print(f'❌ Error: {response.status_code}')
        return None

if __name__ == "__main__":
    run_final_test()