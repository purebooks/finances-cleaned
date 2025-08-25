#!/usr/bin/env python3
"""
Debug script to understand what's happening with large dataset processing
"""

import requests
import json
import pandas as pd

def test_small_vs_large():
    """Test with progressively larger datasets to find the breaking point"""
    
    # Create simple test data
    base_row = {
        "Date": "2023-01-01",
        "Merchant": "Test Merchant",
        "Amount": 100.0,
        "Notes": "Test transaction"
    }
    
    test_sizes = [1, 5, 10, 50, 100, 500]
    
    for size in test_sizes:
        print(f"\n🧪 Testing with {size} rows...")
        
        # Create test data
        test_data = [dict(base_row, **{"Notes": f"Test transaction {i+1}"}) for i in range(size)]
        
        payload = {
            "user_intent": "clean and standardize",
            "data": test_data
        }
        
        try:
            response = requests.post("http://localhost:8080/process", json=payload, timeout=30)
            
            if response.status_code == 200:
                result = response.json()
                cleaned_count = len(result.get("cleaned_data", []))
                print(f"   ✅ Success: {cleaned_count} rows processed")
                
                if cleaned_count != size:
                    print(f"   ⚠️  Warning: Expected {size} rows, got {cleaned_count}")
                    
            else:
                print(f"   ❌ Failed: HTTP {response.status_code}")
                print(f"   Response: {response.text[:200]}...")
                
        except Exception as e:
            print(f"   ❌ Error: {e}")

def test_csv_format():
    """Test the actual CSV format we're using"""
    print("\n🧪 Testing actual CSV format...")
    
    # Load a small sample from our 500-row test
    df = pd.read_csv("test_500_rows.csv")
    sample_data = df.head(10).fillna("").to_dict('records')
    
    # Clean up any problematic values
    for row in sample_data:
        row['Amount'] = pd.to_numeric(row['Amount'], errors='coerce')
        if pd.isna(row['Amount']):
            row['Amount'] = 0.0
    
    payload = {
        "user_intent": "clean and standardize",
        "data": sample_data
    }
    
    print(f"Sample data (first row): {sample_data[0]}")
    
    try:
        response = requests.post("http://localhost:8080/process", json=payload, timeout=30)
        
        if response.status_code == 200:
            result = response.json()
            cleaned_count = len(result.get("cleaned_data", []))
            print(f"✅ Success: {cleaned_count} rows processed from CSV sample")
            
            # Show the first processed row
            if result.get("cleaned_data"):
                print(f"First processed row: {result['cleaned_data'][0]}")
                
        else:
            print(f"❌ Failed: HTTP {response.status_code}")
            print(f"Response: {response.text[:500]}...")
            
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    test_small_vs_large()
    test_csv_format()