#!/usr/bin/env python3
"""
Test hybrid approach with merchants that should definitely trigger real LLM
"""

import requests
import json

def test_with_failure_cases():
    """Test with merchants that should trigger real LLM due to low confidence"""
    
    # These should definitely NOT be in our mock logic patterns
    definitely_unknown = [
        {"Date": "2023-01-01", "Merchant": "XKCD RANDOM CORP 9872", "Amount": 25.0, "Notes": "Unknown"},
        {"Date": "2023-01-02", "Merchant": "QWERTY ZXCVBNM LTD", "Amount": 50.0, "Notes": "Gibberish name"},
        {"Date": "2023-01-03", "Merchant": "AAAAAA BBBBBB CCCCCC", "Amount": 75.0, "Notes": "Nonsense vendor"},
        {"Date": "2023-01-04", "Merchant": "123456789 NUMERIC CORP", "Amount": 100.0, "Notes": "Numbers"},
        {"Date": "2023-01-05", "Merchant": "ZZZZZ END OF ALPHABET INC", "Amount": 30.0, "Notes": "Weird name"}
    ]
    
    print("🧪 Testing with DEFINITELY UNKNOWN merchants")
    print("=" * 60)
    
    payload = {
        "data": definitely_unknown,
        "config": {"use_real_llm": True}
    }
    
    response = requests.post('http://localhost:8080/process', json=payload, timeout=30)
    
    if response.status_code == 200:
        result = response.json()
        insights = result.get('insights', {})
        cleaned_data = result.get('cleaned_data', [])
        
        print(f"📊 Results:")
        print(f"   AI requests: {insights.get('ai_requests', 0)}")
        print(f"   AI cost: ${insights.get('ai_cost', 0.0):.4f}")
        
        print(f"\n📋 Vendor Processing:")
        for i, row in enumerate(cleaned_data):
            original = definitely_unknown[i]['Merchant']
            vendor = row.get('standardized_vendor', 'N/A')
            vendor_source = row.get('vendor_source', 'unknown')
            vendor_conf = row.get('vendor_confidence', 0)
            category = row.get('category', 'N/A')
            category_source = row.get('category_source', 'unknown')
            
            emoji = "🤖" if vendor_source == "llm" else "🔧"
            print(f"   {emoji} {original[:30]:<30} → {vendor} ({vendor_conf:.1%}) [{vendor_source}]")
            print(f"      Category: {category} [{category_source}]")
        
        # Count processing sources
        vendor_sources = {}
        category_sources = {}
        
        for row in cleaned_data:
            v_source = row.get('vendor_source', 'unknown')
            c_source = row.get('category_source', 'unknown')
            vendor_sources[v_source] = vendor_sources.get(v_source, 0) + 1
            category_sources[c_source] = category_sources.get(c_source, 0) + 1
        
        print(f"\n📈 Processing Summary:")
        print(f"   Vendor sources: {vendor_sources}")
        print(f"   Category sources: {category_sources}")
        
        with open('failure_case_results.json', 'w') as f:
            json.dump(result, f, indent=2)
        print(f"\n💾 Results saved to failure_case_results.json")
        
    else:
        print(f"❌ Error: {response.status_code} - {response.text}")

if __name__ == "__main__":
    test_with_failure_cases()