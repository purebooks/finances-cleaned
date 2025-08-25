#!/usr/bin/env python3
"""
Simple debug to trace exactly what's happening
"""

import requests
import json

def test_simple_case():
    """Test with just 5 simple rows"""
    
    test_data = [
        {"Date": "2023-01-01", "Merchant": "Starbucks", "Amount": 5.75, "Notes": "Coffee"},
        {"Date": "2023-01-02", "Merchant": "Shell", "Amount": 45.20, "Notes": "Gas"},
        {"Date": "2023-01-03", "Merchant": "Amazon", "Amount": 25.99, "Notes": "Books"},
        {"Date": "2023-01-04", "Merchant": "Netflix", "Amount": 15.99, "Notes": "Subscription"},
        {"Date": "2023-01-05", "Merchant": "Uber", "Amount": 12.50, "Notes": "Ride"}
    ]
    
    payload = {
        "user_intent": "clean and standardize",
        "data": test_data
    }
    
    print(f"📤 Sending {len(test_data)} rows:")
    for i, row in enumerate(test_data):
        print(f"   Row {i+1}: {row['Merchant']} - ${row['Amount']}")
    
    try:
        response = requests.post("http://localhost:8080/process", json=payload, timeout=30)
        
        if response.status_code == 200:
            result = response.json()
            cleaned_data = result.get("cleaned_data", [])
            
            print(f"\n📥 Received {len(cleaned_data)} rows:")
            for i, row in enumerate(cleaned_data):
                merchant = row.get("Merchant", row.get("merchant", "Unknown"))
                amount = row.get("Amount", row.get("amount", 0))
                print(f"   Row {i+1}: {merchant} - ${amount}")
            
            # Print key insights
            insights = result.get("insights", {})
            print(f"\n📊 Processing insights:")
            print(f"   Rows processed: {insights.get('rows_processed', '?')}")
            print(f"   Processing time: {insights.get('processing_time', '?'):.3f}s")
            
            # Save full response for debugging
            with open("simple_debug_response.json", "w") as f:
                json.dump(result, f, indent=2)
            print(f"\n💾 Full response saved to simple_debug_response.json")
            
        else:
            print(f"❌ Failed: HTTP {response.status_code}")
            print(f"Response: {response.text}")
            
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    test_simple_case()