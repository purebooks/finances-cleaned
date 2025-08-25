#!/usr/bin/env python3
"""
Ultimate 500-row stress test simulating real accountant workload
"""

import pandas as pd
import requests
import json
import time

def run_stress_test():
    print('🔥 ULTIMATE 500-ROW STRESS TEST: Real Accountant Scenario')
    print('=' * 70)
    print('📊 Simulating: Messy client export with every possible edge case...')
    print()

    # Load full 500 rows - the real challenge
    df = pd.read_csv('test_500_rows.csv').fillna('')
    print(f'📂 Loaded {len(df)} rows of challenging financial data')

    # Show some sample complexity
    print('\n🎯 DATA COMPLEXITY PREVIEW:')
    sample_merchants = df['Merchant'].head(10).tolist()
    for i, merchant in enumerate(sample_merchants):
        print(f'{i+1:2d}. {merchant}')

    print('\n⚡ PROCESSING 500 ROWS...')
    print('   (This is the real test - enterprise-scale, messy data)')

    # Convert to API format
    df['Amount'] = pd.to_numeric(df['Amount'], errors='coerce').fillna(0)
    test_data = df.to_dict('records')

    payload = {
        'user_intent': 'comprehensive cleaning and standardization',
        'data': test_data
    }

    # THE BIG TEST
    start_time = time.time()
    print(f'🚀 Started processing at {time.strftime("%H:%M:%S")}...')

    try:
        response = requests.post(
            'http://localhost:8080/process', 
            json=payload, 
            timeout=180  # 3 minutes max
        )
        total_time = time.time() - start_time
        
        print(f'✅ Processing completed in {total_time:.2f}s')
        print(f'⏱️  Speed: {len(test_data)/total_time:.1f} rows/second')
        
        if response.status_code == 200:
            result = response.json()
            cleaned_data = result.get('cleaned_data', [])
            insights = result.get('insights', {})
            
            print(f'\n📊 PROCESSING RESULTS:')
            print(f'   Input rows: {len(test_data)}')
            print(f'   Output rows: {len(cleaned_data)}')
            print(f'   AI requests: {insights.get("ai_requests", 0)}')
            print(f'   AI cost: ${insights.get("ai_cost", 0):.4f}')
            print(f'   Processing time: {total_time:.2f}s')
            
            # Analyze quality
            none_categories = sum(1 for row in cleaned_data if not row.get('category') or row.get('category') == 'None')
            other_categories = sum(1 for row in cleaned_data if row.get('category') == 'Other')
            proper_categories = len(cleaned_data) - none_categories - other_categories
            
            print(f'\n🎯 QUALITY ANALYSIS:')
            print(f'   Proper categories: {proper_categories}/{len(cleaned_data)} ({proper_categories/len(cleaned_data)*100:.1f}%)')
            print(f'   None categories: {none_categories}/{len(cleaned_data)} ({none_categories/len(cleaned_data)*100:.1f}%)')
            print(f'   Other categories: {other_categories}/{len(cleaned_data)} ({other_categories/len(cleaned_data)*100:.1f}%)')
            
            # Show examples of remaining None categories (if any)
            if none_categories > 0:
                print(f'\n❌ REMAINING NONE CATEGORIES (need LLM calls):')
                none_examples = []
                for i, row in enumerate(cleaned_data):
                    if not row.get('category') or row.get('category') == 'None':
                        none_examples.append({
                            'row': i + 1,
                            'original': test_data[i].get('Merchant', ''),
                            'vendor': row.get('standardized_vendor', ''),
                            'amount': row.get('amount', 0)
                        })
                        
                for ex in none_examples[:5]:  # Show first 5
                    print(f'   Row {ex["row"]}: {ex["original"]} → {ex["vendor"]} (${ex["amount"]})')
                    
                if len(none_examples) > 5:
                    print(f'   ... and {len(none_examples) - 5} more')
            
            # Success criteria for production
            speed_ok = total_time <= 30  # Under 30 seconds
            quality_ok = none_categories <= 25  # ≤5% None
            cost_ok = insights.get('ai_cost', 0) <= 2.0  # Under $2
            
            print(f'\n🚀 PRODUCTION READINESS CHECK:')
            print(f'   Speed ≤30s: {"✅" if speed_ok else "❌"} ({total_time:.1f}s)')
            print(f'   None ≤5%: {"✅" if quality_ok else "❌"} ({none_categories} = {none_categories/len(cleaned_data)*100:.1f}%)')
            print(f'   Cost ≤$2: {"✅" if cost_ok else "❌"} (${insights.get("ai_cost", 0):.4f})')
            
            if speed_ok and quality_ok and cost_ok:
                print(f'\n🎉 PRODUCTION READY! All stress tests passed!')
                print(f'   Ready to handle real accountant workloads!')
                print(f'   Can confidently charge $80/month for this quality!')
            else:
                print(f'\n⚠️  Need optimization before production deployment')
                needed = []
                if not speed_ok: needed.append('speed optimization')
                if not quality_ok: needed.append('category classification')
                if not cost_ok: needed.append('cost reduction')
                print(f'   Focus areas: {", ".join(needed)}')
                
            # Save results for analysis
            with open('stress_test_results.json', 'w') as f:
                json.dump({
                    'input_rows': len(test_data),
                    'output_rows': len(cleaned_data),
                    'processing_time': total_time,
                    'rows_per_second': len(test_data)/total_time,
                    'ai_cost': insights.get('ai_cost', 0),
                    'ai_requests': insights.get('ai_requests', 0),
                    'none_categories': none_categories,
                    'other_categories': other_categories,
                    'proper_categories': proper_categories,
                    'none_percentage': none_categories/len(cleaned_data)*100,
                    'quality_percentage': proper_categories/len(cleaned_data)*100,
                    'speed_ok': speed_ok,
                    'quality_ok': quality_ok,
                    'cost_ok': cost_ok,
                    'production_ready': speed_ok and quality_ok and cost_ok,
                    'sample_cleaned_data': cleaned_data[:10]  # Sample for inspection
                }, f, indent=2)
            
            print(f'\n💾 Full results saved to stress_test_results.json')
            
            return {
                'success': True,
                'production_ready': speed_ok and quality_ok and cost_ok,
                'metrics': {
                    'speed': total_time,
                    'quality': proper_categories/len(cleaned_data)*100,
                    'cost': insights.get('ai_cost', 0)
                }
            }
            
        else:
            print(f'❌ ERROR: {response.status_code}')
            print(f'Response: {response.text}')
            return {'success': False, 'error': f'HTTP {response.status_code}'}
            
    except requests.exceptions.Timeout:
        print(f'❌ TIMEOUT: Processing took longer than 3 minutes')
        print(f'   This would fail in production - need optimization')
        return {'success': False, 'error': 'timeout'}
        
    except Exception as e:
        print(f'❌ CRITICAL ERROR: {str(e)}')
        print(f'   System failed under stress - not production ready')
        return {'success': False, 'error': str(e)}

if __name__ == "__main__":
    run_stress_test()