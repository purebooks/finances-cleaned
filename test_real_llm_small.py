#!/usr/bin/env python3
"""
Test real LLM performance on a small dataset (50 rows)
"""

import requests
import json
import pandas as pd
import time

def test_real_llm_small():
    """Test real LLM on 50 rows to check performance and cost"""
    
    # Load test data 
    df = pd.read_csv('test_500_rows.csv')
    
    # Take first 50 rows
    small_df = df.head(50).fillna("")
    
    print("🧪 Testing Real LLM on 50 rows")
    print("=" * 50)
    print(f"📊 Sample size: {len(small_df)} rows")
    
    # Convert to API format
    data = []
    for _, row in small_df.iterrows():
        # Handle amount formatting
        amount_str = str(row['Amount']).replace('$', '').replace(',', '') if pd.notna(row['Amount']) else '0'
        try:
            amount = float(amount_str)
        except (ValueError, TypeError):
            amount = 0.0
            
        data.append({
            "Date": str(row['Date']),
            "Merchant": str(row['Merchant']),
            "Amount": amount,
            "Notes": str(row['Notes'])
        })
    
    # Test with real LLM
    payload = {
        "data": data,
        "config": {
            "use_real_llm": True  # Force real LLM
        }
    }
    
    print("🚀 Sending to API with REAL LLM enabled...")
    start_time = time.time()
    
    try:
        response = requests.post(
            'http://localhost:8080/process',
            json=payload,
            timeout=120  # 2 minute timeout for real LLM
        )
        
        processing_time = time.time() - start_time
        
        if response.status_code == 200:
            result = response.json()
            cleaned_data = result.get('cleaned_data', [])
            insights = result.get('insights', {})
            
            print(f"✅ Success! Processed in {processing_time:.2f}s")
            print(f"📊 Rows processed: {len(cleaned_data)}")
            print(f"💰 AI requests: {insights.get('ai_requests', 0)}")
            print(f"💰 Estimated cost: ${insights.get('ai_cost', 0.0):.4f}")
            print(f"⚡ Rows per second: {len(cleaned_data) / processing_time:.1f}")
            
            # Show sample results
            print("\n📋 Sample Results:")
            for i, row in enumerate(cleaned_data[:5]):
                vendor = row.get('standardized_vendor', 'N/A')
                category = row.get('category', 'N/A')
                original = row.get('Merchant', 'N/A')
                print(f"  {i+1}. {original} → {vendor} | {category}")
            
            # Check for categories
            with_categories = sum(1 for row in cleaned_data if row.get('category') and row.get('category') != 'Other')
            category_rate = (with_categories / len(cleaned_data)) * 100
            print(f"\n📈 Categories assigned: {with_categories}/{len(cleaned_data)} ({category_rate:.1f}%)")
            
            # Save results
            with open('real_llm_50_results.json', 'w') as f:
                json.dump(result, f, indent=2)
            print("💾 Results saved to real_llm_50_results.json")
            
            # Ask if we should continue
            print(f"\n🎯 Cost estimate for 500 rows: ${insights.get('ai_cost', 0.0) * 10:.2f}")
            print(f"🎯 Time estimate for 500 rows: {processing_time * 10:.1f}s")
            
            return True
            
        else:
            print(f"❌ API Error: {response.status_code} - {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Request failed: {e}")
        return False

if __name__ == "__main__":
    test_real_llm_small()