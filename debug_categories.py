#!/usr/bin/env python3
"""
Debug category classification specifically
"""

import requests
import json

def test_category_assignment():
    """Test if categories are being assigned correctly"""
    
    # Test with well-known vendors that should get clear categories
    test_data = [
        {"Date": "2023-01-01", "Merchant": "Starbucks", "Amount": 5.75, "Notes": "Coffee"},
        {"Date": "2023-01-02", "Merchant": "Netflix", "Amount": 15.99, "Notes": "Subscription"},
        {"Date": "2023-01-03", "Merchant": "Shell", "Amount": 45.20, "Notes": "Gas"},
        {"Date": "2023-01-04", "Merchant": "Amazon", "Amount": 25.99, "Notes": "Office supplies"},
    ]
    
    payload = {
        "user_intent": "categorize transactions",
        "data": test_data
    }
    
    print(f"🧪 Testing category assignment with {len(test_data)} well-known vendors")
    
    try:
        response = requests.post("http://localhost:8080/process", json=payload, timeout=30)
        
        if response.status_code == 200:
            result = response.json()
            cleaned_data = result.get("cleaned_data", [])
            
            print(f"\n📊 Results:")
            for i, row in enumerate(cleaned_data):
                merchant = row.get("Merchant", "Unknown")
                category = row.get("category", "NOT_FOUND")
                standardized_vendor = row.get("standardized_vendor", "NOT_FOUND")
                
                print(f"   Row {i+1}: {merchant}")
                print(f"      Category: {category}")
                print(f"      Standardized: {standardized_vendor}")
                print(f"      All fields: {list(row.keys())}")
                print()
            
            # Check insights
            insights = result.get("insights", {})
            print(f"📈 Insights:")
            print(f"   AI requests: {insights.get('ai_requests', '?')}")
            print(f"   AI cost: ${insights.get('ai_cost', '?')}")
            
            # Save for debugging
            with open("debug_categories_response.json", "w") as f:
                json.dump(result, f, indent=2)
            print(f"\n💾 Full response saved to debug_categories_response.json")
            
        else:
            print(f"❌ Failed: HTTP {response.status_code}")
            print(f"Response: {response.text}")
            
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    test_category_assignment()