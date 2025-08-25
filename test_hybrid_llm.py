#!/usr/bin/env python3
"""
Test the new hybrid LLM approach:
- Python rules for high confidence cases
- Real LLM for low confidence/unknown cases
"""

import requests
import json
import time

def test_hybrid_approach():
    """Test hybrid Python + Real LLM processing"""
    
    # Mix of known vendors (high confidence) and unknown vendors (low confidence)
    test_data = [
        # HIGH CONFIDENCE - Should use Python rules
        {"Date": "2023-01-01", "Merchant": "Starbucks", "Amount": 5.75, "Notes": "Coffee"},
        {"Date": "2023-01-02", "Merchant": "Netflix", "Amount": 15.99, "Notes": "Streaming"},
        {"Date": "2023-01-03", "Merchant": "Amazon", "Amount": 29.99, "Notes": "Books"},
        
        # LOW CONFIDENCE - Should trigger real LLM
        {"Date": "2023-01-04", "Merchant": "Obscure Local Bakery XYZ", "Amount": 12.50, "Notes": "Pastries"},
        {"Date": "2023-01-05", "Merchant": "Mystery Tech Consulting LLC", "Amount": 500.0, "Notes": "IT Services"},
        {"Date": "2023-01-06", "Merchant": "Random Food Truck #47", "Amount": 8.99, "Notes": "Lunch"},
    ]
    
    print("🧪 Testing Hybrid LLM Approach")
    print("=" * 60)
    print("📋 Test Data:")
    for item in test_data:
        print(f"   - {item['Merchant']}")
    
    payload = {
        "data": test_data,
        "config": {"use_real_llm": True}  # Enable hybrid mode
    }
    
    print(f"\n🚀 Processing {len(test_data)} transactions...")
    start_time = time.time()
    
    try:
        response = requests.post(
            'http://localhost:8080/process',
            json=payload,
            timeout=60
        )
        
        processing_time = time.time() - start_time
        
        if response.status_code == 200:
            result = response.json()
            cleaned_data = result.get('cleaned_data', [])
            insights = result.get('insights', {})
            
            print(f"✅ Success! Processed in {processing_time:.2f}s")
            print(f"💰 AI requests: {insights.get('ai_requests', 0)}")
            print(f"💰 AI cost: ${insights.get('ai_cost', 0.0):.4f}")
            
            print(f"\n📊 Processing Sources:")
            sources = {}
            for row in cleaned_data:
                source = row.get('processing_source', 'unknown')
                sources[source] = sources.get(source, 0) + 1
            
            for source, count in sources.items():
                print(f"   {source}: {count} rows")
            
            print(f"\n📋 Detailed Results:")
            for i, row in enumerate(cleaned_data):
                original = test_data[i]['Merchant']
                vendor = row.get('standardized_vendor', 'N/A')
                category = row.get('category', 'N/A')
                confidence = row.get('confidence', 0)
                source = row.get('processing_source', 'unknown')
                
                emoji = "🤖" if "llm" in source else "🔧" 
                print(f"   {emoji} {original[:25]:<25} → {vendor} | {category} ({confidence:.1%}) [{source}]")
            
            # Save detailed results
            with open('hybrid_test_results.json', 'w') as f:
                json.dump(result, f, indent=2)
            print(f"\n💾 Results saved to hybrid_test_results.json")
            
            return True
            
        else:
            print(f"❌ API Error: {response.status_code} - {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Request failed: {e}")
        return False

if __name__ == "__main__":
    test_hybrid_approach()