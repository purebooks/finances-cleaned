#!/usr/bin/env python3

import os
import requests
import json

# Ensure the API key is provided via environment (do not hardcode secrets)
if not os.environ.get('ANTHROPIC_API_KEY'):
    raise SystemExit("ANTHROPIC_API_KEY not set. Export it in your shell before running this test.")

# Test data - list of dictionaries format
test_data = [
    {
        "date": "2023-10-01",
        "merchant": "SQ *SQ *PATRIOT CAFE",
        "amount": "15.25",
        "description": "Morning coffee"
    }
]

# Test the API - send data in 'data' field like the interface does
url = "http://localhost:8080/process"
payload = {
    "user_intent": "standardize vendors and categorize",
    "data": test_data
}

print("🧪 Testing Live AI with API Key...")
print(f"🔑 API Key: {os.environ.get('ANTHROPIC_API_KEY', 'Not set')[:20]}...")

try:
    response = requests.post(url, json=payload, timeout=30)
    
    if response.status_code == 200:
        result = response.json()
        print("✅ Live AI test successful!")
        print(f"⏱️  Processing time: {result.get('processing_time', 'N/A')}s")
        print(f"🤖 AI requests: {result.get('ai_requests', 'N/A')}")
        print(f"💰 Cost: {result.get('cost', 'N/A')}")
        
        # Show the cleaned data
        cleaned_data = result.get('cleaned_data', [])
        if cleaned_data:
            print(f"📊 Cleaned {len(cleaned_data)} records")
            print("First record:")
            print(json.dumps(cleaned_data[0], indent=2))
    else:
        print(f"❌ API Error: {response.status_code}")
        print(response.text)
        
except Exception as e:
    print(f"❌ Test failed: {e}")

print("\n🎯 Ready to test with the interface!") 