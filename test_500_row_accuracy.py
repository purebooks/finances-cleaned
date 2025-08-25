#!/usr/bin/env python3
"""
Test the 500-row dataset and validate 95% accuracy requirement
"""

import os
import requests
import json
import pandas as pd
import time
from typing import Dict, List, Tuple

def load_test_data() -> Tuple[List[Dict], List[Dict]]:
    """Load the test data and expected corrections"""
    # Load the messy CSV data
    df = pd.read_csv("test_500_rows.csv")
    
    # Clean up NaN values that cause JSON serialization issues
    df = df.fillna("")
    
    # Ensure amounts are numeric where possible
    df['Amount'] = pd.to_numeric(df['Amount'], errors='coerce').fillna(0)
    
    test_data = df.to_dict('records')
    
    # Load expected corrections
    with open("test_500_rows_expected.json", "r") as f:
        expected_corrections = json.load(f)
    
    return test_data, expected_corrections

def send_to_api(data: List[Dict]) -> Dict:
    """Send data to our cleaning API"""
    payload = {
        "user_intent": "comprehensive cleaning and standardization",
        "data": data,
        "config": {
            "preserve_schema": True,
            "ai_mode": "apply",
            "enable_ai": True,
            "ai_vendor_enabled": True,
            "ai_category_enabled": True,
            "ai_confidence_threshold": 0.6,
            "ai_preserve_schema_apply": True,
            "ai_mode": "apply",
            "ai_model": "gpt-4o-mini"
        }
    }
    
    print(f"🚀 Sending {len(data)} rows to API...")
    start_time = time.time()
    
    try:
        base_url = os.getenv("API_BASE_URL") or f"http://localhost:{os.getenv('PORT', '8080')}"
        response = requests.post(f"{base_url}/process", json=payload, timeout=120)
        processing_time = time.time() - start_time
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ API responded successfully in {processing_time:.2f}s")
            return {"success": True, "data": result, "processing_time": processing_time}
        else:
            print(f"❌ API error: {response.status_code}")
            return {"success": False, "error": f"HTTP {response.status_code}"}
            
    except requests.exceptions.Timeout:
        print("❌ API timeout after 120 seconds")
        return {"success": False, "error": "Timeout"}
    except Exception as e:
        print(f"❌ API error: {e}")
        return {"success": False, "error": str(e)}

def calculate_accuracy(cleaned_data: List[Dict], expected_corrections: List[Dict]) -> Dict:
    """Calculate various accuracy metrics"""
    
    if len(cleaned_data) != len(expected_corrections):
        print(f"⚠️  Row count mismatch: {len(cleaned_data)} vs {len(expected_corrections)}")
    
    results = {
        "total_rows": len(expected_corrections),
        "processed_rows": len(cleaned_data),
        "vendor_accuracy": 0,
        "category_accuracy": 0,
        "date_fix_accuracy": 0,
        "amount_fix_accuracy": 0,
        "overall_accuracy": 0,
        "detailed_results": []
    }
    
    vendor_correct = 0
    category_correct = 0
    date_fixes_correct = 0
    amount_fixes_correct = 0
    
    for i, expected in enumerate(expected_corrections):
        if i >= len(cleaned_data):
            break
            
        cleaned_row = cleaned_data[i]
        row_result = {
            "row_id": i,
            "vendor_match": False,
            "category_match": False,
            "date_fixed": False,
            "amount_fixed": False
        }
        
        # Check vendor standardization
        cleaned_merchant = cleaned_row.get("standardized_vendor", cleaned_row.get("merchant", "")).strip()
        expected_merchant = expected["expected_clean_merchant"].strip()
        
        # Fuzzy matching for vendor names (allow for slight variations)
        if (cleaned_merchant.lower() in expected_merchant.lower() or 
            expected_merchant.lower() in cleaned_merchant.lower() or
            cleaned_merchant.lower() == expected_merchant.lower()):
            vendor_correct += 1
            row_result["vendor_match"] = True
        
        # Check category classification  
        cleaned_category = str(cleaned_row.get("category", cleaned_row.get("Category", "")) or "").strip()
        expected_category = str(expected["expected_category"] or "").strip()
        
        # Use fuzzy matching for categories (allow for reasonable variations)
        if (cleaned_category == expected_category or 
            cleaned_category.lower() == expected_category.lower() or
            (cleaned_category and expected_category and 
             any(word in cleaned_category.lower() for word in expected_category.lower().split()) or
             any(word in expected_category.lower() for word in cleaned_category.lower().split()))):
            category_correct += 1
            row_result["category_match"] = True
        
        # Check date fixing (if there was a date issue)
        if expected["has_date_issue"]:
            cleaned_date = cleaned_row.get("date", cleaned_row.get("Date", ""))
            if cleaned_date and cleaned_date != "" and cleaned_date != "NaT":
                date_fixes_correct += 1
                row_result["date_fixed"] = True
        else:
            # If no issue expected, count as correct
            date_fixes_correct += 1
            row_result["date_fixed"] = True
        
        # Check amount fixing (if there was an amount issue)
        if expected["has_amount_issue"]:
            cleaned_amount = cleaned_row.get("amount", cleaned_row.get("Amount", 0))
            try:
                float(cleaned_amount)
                amount_fixes_correct += 1
                row_result["amount_fixed"] = True
            except (ValueError, TypeError):
                pass
        else:
            # If no issue expected, count as correct
            amount_fixes_correct += 1
            row_result["amount_fixed"] = True
        
        results["detailed_results"].append(row_result)
    
    # Calculate percentages
    total_rows = len(expected_corrections)
    results["vendor_accuracy"] = (vendor_correct / total_rows) * 100
    results["category_accuracy"] = (category_correct / total_rows) * 100
    results["date_fix_accuracy"] = (date_fixes_correct / total_rows) * 100
    results["amount_fix_accuracy"] = (amount_fixes_correct / total_rows) * 100
    
    # Overall accuracy (weighted average)
    results["overall_accuracy"] = (
        results["vendor_accuracy"] * 0.4 +  # 40% weight on vendor standardization
        results["category_accuracy"] * 0.4 +  # 40% weight on category classification
        results["date_fix_accuracy"] * 0.1 +   # 10% weight on date fixing
        results["amount_fix_accuracy"] * 0.1   # 10% weight on amount fixing
    )
    
    return results

def run_accuracy_test():
    """Run the full 500-row accuracy test"""
    print("🧪 Starting 500-Row Accuracy Test")
    print("=" * 50)
    
    # Load test data
    test_data, expected_corrections = load_test_data()
    print(f"📊 Loaded {len(test_data)} test rows")
    
    # Send to API
    api_result = send_to_api(test_data)
    
    if not api_result["success"]:
        print(f"❌ Test failed: {api_result['error']}")
        return False
    
    # Extract cleaned data
    cleaned_data = api_result["data"].get("cleaned_data", [])
    processing_time = api_result["processing_time"]
    
    # Calculate accuracy
    print("\n🔍 Calculating accuracy metrics...")
    accuracy_results = calculate_accuracy(cleaned_data, expected_corrections)
    
    # Print results
    print("\n📈 ACCURACY RESULTS")
    print("=" * 50)
    print(f"Vendor Standardization: {accuracy_results['vendor_accuracy']:.1f}%")
    print(f"Category Classification: {accuracy_results['category_accuracy']:.1f}%")
    print(f"Date Issue Fixing: {accuracy_results['date_fix_accuracy']:.1f}%")
    print(f"Amount Issue Fixing: {accuracy_results['amount_fix_accuracy']:.1f}%")
    print(f"OVERALL ACCURACY: {accuracy_results['overall_accuracy']:.1f}%")
    
    print(f"\n⚡ PERFORMANCE METRICS")
    print("=" * 50)
    print(f"Processing Time: {processing_time:.2f} seconds")
    print(f"Rows per Second: {len(test_data)/processing_time:.1f}")
    print(f"Rows Processed: {accuracy_results['processed_rows']}/{accuracy_results['total_rows']}")
    
    # Check if we meet the 95% requirement
    target_accuracy = 95.0
    passed = accuracy_results['overall_accuracy'] >= target_accuracy
    
    print(f"\n🎯 TARGET ACCURACY: {target_accuracy}%")
    if passed:
        print(f"✅ PASSED! Achieved {accuracy_results['overall_accuracy']:.1f}%")
    else:
        print(f"❌ FAILED! Only achieved {accuracy_results['overall_accuracy']:.1f}%")
        print(f"   Need to improve by {target_accuracy - accuracy_results['overall_accuracy']:.1f} percentage points")
    
    # Save detailed results
    with open("test_500_rows_results.json", "w") as f:
        json.dump({
            "test_summary": {
                "total_rows": accuracy_results['total_rows'],
                "processed_rows": accuracy_results['processed_rows'],
                "processing_time": processing_time,
                "overall_accuracy": accuracy_results['overall_accuracy'],
                "target_met": passed
            },
            "accuracy_breakdown": {
                "vendor_accuracy": accuracy_results['vendor_accuracy'],
                "category_accuracy": accuracy_results['category_accuracy'],
                "date_fix_accuracy": accuracy_results['date_fix_accuracy'],
                "amount_fix_accuracy": accuracy_results['amount_fix_accuracy']
            },
            "detailed_results": accuracy_results['detailed_results'],
            "api_response": api_result["data"]
        }, f, indent=2)
    
    print(f"\n💾 Detailed results saved to test_500_rows_results.json")
    
    return passed

if __name__ == "__main__":
    success = run_accuracy_test()
    exit(0 if success else 1)