#!/usr/bin/env python3
"""
Test the difference between small and large dataset processing
"""

import requests
import json

def test_dataset_size_behavior():
    """Test how the system behaves with different dataset sizes"""
    
    base_row = {
        "Date": "2023-01-01", 
        "Merchant": "Starbucks", 
        "Amount": 5.75, 
        "Notes": "Coffee"
    }
    
    sizes_to_test = [1, 5, 10, 25, 50, 100]
    
    for size in sizes_to_test:
        print(f"\n🧪 Testing with {size} rows...")
        
        # Create test data
        test_data = []
        merchants = ["Starbucks", "Netflix", "Shell", "Amazon", "Uber"]
        for i in range(size):
            row = dict(base_row)
            row["Merchant"] = merchants[i % len(merchants)]
            row["Notes"] = f"Transaction {i+1}"
            test_data.append(row)
        
        payload = {
            "user_intent": "categorize all transactions",
            "data": test_data
        }
        
        try:
            response = requests.post("http://localhost:8080/process", json=payload, timeout=30)
            
            if response.status_code == 200:
                result = response.json()
                cleaned_data = result.get("cleaned_data", [])
                
                # Check if first row has category
                has_category = len(cleaned_data) > 0 and "category" in cleaned_data[0]
                
                # Count how many rows have categories
                rows_with_categories = sum(1 for row in cleaned_data if "category" in row)
                
                print(f"   ✅ {len(cleaned_data)} rows returned")
                print(f"   📂 Categories: {rows_with_categories}/{len(cleaned_data)} rows have category field")
                
                if has_category:
                    sample_category = cleaned_data[0].get("category", "NONE")
                    print(f"   📋 Sample category: {sample_category}")
                    
                    # Show all categories if small dataset
                    if size <= 10:
                        categories = [row.get("category", "NONE") for row in cleaned_data]
                        print(f"   📋 All categories: {categories}")
                
                # Check processing insights
                insights = result.get("insights", {})
                ai_requests = insights.get("ai_requests", 0)
                print(f"   🤖 AI requests: {ai_requests}")
                
            else:
                print(f"   ❌ Failed: HTTP {response.status_code}")
                
        except Exception as e:
            print(f"   ❌ Error: {e}")

if __name__ == "__main__":
    test_dataset_size_behavior()