#!/usr/bin/env python3

import requests
import json

# Test with the exact same data that the interface would send
test_data = [
    {"Date": "2023-10-01", "Merchant": "SQ *SQ *PATRIOT CAFE", "Amount": "15.25", "Notes": "Morning coffee"},
    {"Date": "2023/10/01", "Merchant": "uber trip", "Amount": "-22.50", "Notes": "Ride to meeting"},
    {"Date": "10-02-2023", "Merchant": "AMZ*Amazon.com", "Amount": "", "Notes": "Office supplies, pens"},
    {"Date": "", "Merchant": "GOOGLE *GSUITE", "Amount": "12.00", "Notes": "Monthly subscription"},
    {"Date": "2023-10-03", "Merchant": "", "Amount": "9.99", "Notes": "Music streaming"},
    {"Date": "2023-10-04", "Merchant": "apple.com/bill", "Amount": "9.99", "Notes": "iCloud Storage"},
    {"Date": "2023/10/01", "Merchant": "uber trip", "Amount": "-22.50", "Notes": "Duplicate ride entry for testing"},
    {"Date": "Oct 5 2023", "Merchant": "Random Local Grocer", "Amount": "55.43", "Notes": "Team lunch catering"}
]

payload = {
    "user_intent": "find duplicates and standardize vendors",
    "data": test_data
}

print("🧪 Debug API Test")
print(f"📊 Sending {len(test_data)} records")
print(f"📝 First record: {test_data[0]}")
print(f"🔍 Payload structure: {list(payload.keys())}")

try:
    response = requests.post('http://localhost:8080/process', json=payload, timeout=30)
    
    print(f"📡 Response status: {response.status_code}")
    print(f"📄 Response headers: {dict(response.headers)}")
    
    if response.status_code == 200:
        result = response.json()
        print("✅ Success!")
        print(f"📊 Processed {len(result.get('cleaned_data', []))} records")
        print(f"⏱️  Time: {result.get('processing_time', 'N/A')}s")
    else:
        print(f"❌ Error: {response.status_code}")
        print(f"📄 Response: {response.text}")
        
except Exception as e:
    print(f"❌ Exception: {e}")

print("\n�� Test complete!") 