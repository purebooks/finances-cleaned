#!/usr/bin/env python3
"""
Quick analysis of category classification failures
"""

import json
import pandas as pd

def analyze_failures():
    # Load expected data 
    with open('test_500_rows_expected.json', 'r') as f:
        expected = json.load(f)
    
    # Load results 
    with open('test_500_rows_results.json', 'r') as f:
        results = json.load(f)
    
    # Find category failures
    failures = []
    for i, result in enumerate(results['detailed_results']):
        if not result['category_match']:
            exp = expected[i]
            failures.append({
                'original_merchant': exp['original_merchant'],
                'expected_clean_merchant': exp['expected_clean_merchant'],
                'expected_category': exp['expected_category'],
                'row_id': i
            })
    
    print(f"🔍 Found {len(failures)} category classification failures:")
    print("=" * 60)
    
    # Group by expected category
    by_category = {}
    for f in failures:
        cat = f['expected_category']
        if cat not in by_category:
            by_category[cat] = []
        by_category[cat].append(f)
    
    for category, fails in by_category.items():
        print(f"\n📂 {category}: {len(fails)} failures")
        for f in fails[:5]:  # Show first 5
            print(f"   - {f['original_merchant']} → {f['expected_clean_merchant']}")
        if len(fails) > 5:
            print(f"   ... and {len(fails) - 5} more")
    
    print(f"\n📊 Total failures: {len(failures)}")
    print(f"📊 Success rate: {((500 - len(failures)) / 500) * 100:.1f}%")

if __name__ == "__main__":
    analyze_failures()