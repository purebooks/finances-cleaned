#!/usr/bin/env python3
"""
Test with completely unknown vendors to force real LLM usage
"""

import requests
import json

def test_unknown_vendors():
    """Test with vendors not in our rule-based system"""
    
    # Vendors NOT in our mock logic
    unknown_data = [
        {"Date": "2023-01-01", "Merchant": "Obscure Local Restaurant XYZ", "Amount": 25.0, "Notes": "Lunch"},
        {"Date": "2023-01-02", "Merchant": "Random Software Startup Inc", "Amount": 99.0, "Notes": "SaaS subscription"},
        {"Date": "2023-01-03", "Merchant": "Joe's Unknown Hardware Store", "Amount": 45.0, "Notes": "Tools"},
        {"Date": "2023-01-04", "Merchant": "Mystery Consulting LLC", "Amount": 500.0, "Notes": "Professional services"},
        {"Date": "2023-01-05", "Merchant": "Weird Food Truck Name", "Amount": 12.0, "Notes": "Street food"}
    ]
    
    print("🧪 Testing with UNKNOWN vendors to force real LLM")
    print("=" * 60)
    
    for vendor in unknown_data:
        print(f"   - {vendor['Merchant']}")
    
    payload = {
        "data": unknown_data,
        "config": {"use_real_llm": True}
    }
    
    response = requests.post('http://localhost:8080/process', json=payload, timeout=60)
    
    if response.status_code == 200:
        result = response.json()
        insights = result.get('insights', {})
        
        print(f"\n📊 Results:")
        print(f"   AI requests: {insights.get('ai_requests', 0)}")
        print(f"   AI cost: ${insights.get('ai_cost', 0.0):.4f}")
        print(f"   Processing time: {insights.get('processing_time', 0):.2f}s")
        
        # Show vendor classifications
        cleaned_data = result.get('cleaned_data', [])
        print(f"\n📋 Vendor Classifications:")
        for i, row in enumerate(cleaned_data):
            vendor = row.get('standardized_vendor', 'N/A')
            category = row.get('category', 'N/A')
            original = unknown_data[i]['Merchant']
            print(f"   {original[:30]:<30} → {vendor} | {category}")
            
        return insights.get('ai_requests', 0) > 0
    else:
        print(f"❌ Error: {response.status_code} - {response.text}")
        return False

if __name__ == "__main__":
    test_unknown_vendors()