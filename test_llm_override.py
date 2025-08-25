#!/usr/bin/env python3
"""
Test if LLM override is working
"""

import requests
import json

def test_override():
    """Test if the real LLM override works"""
    
    # Simple test data
    data = [{
        "Date": "2023-01-01",
        "Merchant": "Test Vendor",
        "Amount": 10.0,
        "Notes": "Test transaction"
    }]
    
    # Test with mock (default)
    print("🧪 Testing Mock LLM (default)")
    payload_mock = {"data": data}
    
    response = requests.post('http://localhost:8080/process', json=payload_mock)
    if response.status_code == 200:
        result = response.json()
        print(f"   AI requests: {result.get('insights', {}).get('ai_requests', 0)}")
        print(f"   AI cost: ${result.get('insights', {}).get('ai_cost', 0.0):.4f}")
    else:
        print(f"   Error: {response.status_code}")
    
    # Test with real LLM override
    print("\n🧪 Testing Real LLM (override)")
    payload_real = {
        "data": data,
        "config": {"use_real_llm": True}
    }
    
    response = requests.post('http://localhost:8080/process', json=payload_real)
    if response.status_code == 200:
        result = response.json()
        print(f"   AI requests: {result.get('insights', {}).get('ai_requests', 0)}")
        print(f"   AI cost: ${result.get('insights', {}).get('ai_cost', 0.0):.4f}")
    else:
        print(f"   Error: {response.status_code}")

if __name__ == "__main__":
    test_override()