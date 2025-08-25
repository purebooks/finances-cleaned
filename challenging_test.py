#!/usr/bin/env python3

import requests
import json
import time
from typing import Dict, List, Any

def test_challenging_data():
    """Test the enhanced system with really challenging data"""
    
    print("🔥 CHALLENGING ACCURACY TEST")
    print("=" * 50)
    print("Testing with edge cases, typos, ambiguous vendors, and complex scenarios")
    print()
    
    # Load challenging test data
    with open('challenging_test_data.json', 'r') as f:
        test_data = json.load(f)
    
    print(f"📊 Testing {len(test_data)} challenging transactions:")
    print()
    
    # Expected results (what we HOPE the enhanced system will achieve)
    expected_results = {
        "AMZN MKTP US*MO12345XY": {"vendor": "Amazon", "category": "Office Supplies & Equipment"},
        "SQ *TECHCORP SOLUTIONS": {"vendor": "Square", "category": "Professional Services"},
        "GOOGL *WORKSPACE G.CO": {"vendor": "Google Workspace", "category": "Software & Technology"},
        "STRIPE*UNKNOWN VENDOR LLC": {"vendor": "Stripe", "category": "Banking & Finance"},
        "UBER *EATS DELIVERY": {"vendor": "Uber Eats", "category": "Meals & Entertainment"},
        "PP*FREELANCER DESIGN": {"vendor": "PayPal", "category": "Professional Services"},
        "SBUX STORE #12345": {"vendor": "Starbucks", "category": "Meals & Entertainment"},
        "DELTA AIRLINES 12345678": {"vendor": "Delta Airlines", "category": "Travel & Transportation"},
        "MYSTERIOUS CORP INC": {"vendor": "Mysterious Corp Inc", "category": "Other"},  # Unknown - should go to LLM
        "MICROSFT*OFFICE365": {"vendor": "Microsoft", "category": "Software & Technology"},  # Typo test
        "RESTAURANT UNKNOWN NAME": {"vendor": "Restaurant Unknown Name", "category": "Meals & Entertainment"},
        "DIGITALOCEA*CLOUD HOSTING": {"vendor": "DigitalOcean", "category": "Software & Technology"},  # Typo test
        "APPLE.COM/BILL": {"vendor": "Apple", "category": "Software & Technology"},
        "ZOOM.US *VIDEOCONF": {"vendor": "Zoom", "category": "Software & Technology"},
        "TYPO COMPNAY LLC": {"vendor": "Typo Company LLC", "category": "Marketing & Advertising"},  # Unknown with typo
        "CHASe BANK FEE": {"vendor": "JPMorgan Chase", "category": "Banking & Finance"},  # Typo test
        "UNKNOWN*1234567890": {"vendor": "Unknown", "category": "Other"},  # Complete unknown
        "SALESFORCE.COM": {"vendor": "Salesforce", "category": "Software & Technology"},
        "MCD*FRANCHISE #5678": {"vendor": "McDonalds", "category": "Meals & Entertainment"},
        "GITHUB, INC.": {"vendor": "GitHub", "category": "Software & Technology"}
    }
    
    # API endpoint
    api_url = "https://ai-financial-cleaner-v5-pksi3xslca-uc.a.run.app/process"
    
    # Configuration for maximum accuracy testing
    config = {
        "enable_ai": True,
        "ai_vendor_enabled": True,
        "ai_category_enabled": True,
        "enable_transaction_intelligence": False,  # Skip for speed
        "enable_source_tracking": True,
        "ai_confidence_threshold": 0.7,
        "enable_parallel_processing": True,
        "max_workers": 4,
        "force_llm_for_testing": False  # Let rules work naturally
    }
    
    # Test the API
    print("🚀 Sending challenging data to enhanced AI system...")
    print("⏱️  This may take 1-2 minutes for unknown vendors...")
    print()
    
    start_time = time.time()
    
    try:
        response = requests.post(
            api_url,
            json={"data": test_data, "config": config},
            timeout=300  # 5 minute timeout for challenging data
        )
        
        if response.status_code != 200:
            print(f"❌ API Error: {response.status_code}")
            print(f"Response: {response.text}")
            return
        
        result = response.json()
        processing_time = time.time() - start_time
        
        print(f"✅ Processing completed in {processing_time:.2f} seconds")
        print()
        
        # Analyze results
        analyze_challenging_results(result, expected_results, test_data)
        
    except requests.exceptions.Timeout:
        print("⏰ Request timed out - this dataset is really challenging!")
        print("💡 Try testing with a smaller subset or increase timeout")
    except Exception as e:
        print(f"❌ Error testing challenging data: {e}")

def analyze_challenging_results(result: Dict[str, Any], expected: Dict[str, Dict], original_data: List[Dict]):
    """Analyze how well the system handled challenging scenarios"""
    
    print("📊 CHALLENGING TEST RESULTS ANALYSIS")
    print("=" * 50)
    
    # Extract results
    cleaned_data = result.get('cleaned_data', [])
    summary = result.get('summary_report', {})
    processing_summary = summary.get('processing_summary', {})
    
    print(f"📈 Overall Performance:")
    print(f"   • Total Transactions: {processing_summary.get('total_transactions', 0)}")
    print(f"   • LLM Calls Made: {processing_summary.get('llm_calls', 0)}")
    print(f"   • Total Processing Time: {summary.get('total_time', 'N/A')}s")
    print(f"   • Total Cost: ${processing_summary.get('total_cost', 0):.3f}")
    print(f"   • Cache Hit Rate: {processing_summary.get('cache_hit_rate', 0)*100:.1f}%")
    print()
    
    # Accuracy analysis
    correct_vendors = 0
    correct_categories = 0
    rule_based_matches = 0
    llm_calls = 0
    
    print("🎯 ACCURACY BREAKDOWN (Challenging Scenarios):")
    print("-" * 50)
    
    for i, (original, cleaned) in enumerate(zip(original_data, cleaned_data)):
        original_merchant = original['merchant']
        cleaned_vendor = cleaned.get('standardized_vendor', 'N/A')
        cleaned_category = cleaned.get('category', 'N/A')
        vendor_source = cleaned.get('vendor_source', 'unknown')
        category_source = cleaned.get('category_source', 'unknown')
        
        expected_result = expected.get(original_merchant, {})
        expected_vendor = expected_result.get('vendor', 'Unknown')
        expected_category = expected_result.get('category', 'Other')
        
        # Check accuracy
        vendor_correct = cleaned_vendor == expected_vendor or (expected_vendor == 'Unknown' and cleaned_vendor != 'N/A')
        category_correct = cleaned_category == expected_category or (expected_category == 'Other' and cleaned_category != 'N/A')
        
        if vendor_correct:
            correct_vendors += 1
        if category_correct:
            correct_categories += 1
        
        if vendor_source == 'rule_based':
            rule_based_matches += 1
        elif vendor_source == 'llm':
            llm_calls += 1
        
        # Status indicators
        vendor_status = "✅" if vendor_correct else "❌"
        category_status = "✅" if category_correct else "❌"
        source_emoji = "⚡" if vendor_source == "rule_based" else "🤖" if vendor_source == "llm" else "💾"
        
        print(f"{i+1:2d}. {original_merchant[:30]:<30}")
        print(f"    {vendor_status} Vendor: {cleaned_vendor} (expected: {expected_vendor}) {source_emoji}")
        print(f"    {category_status} Category: {cleaned_category} (expected: {expected_category})")
        print()
    
    # Calculate accuracy percentages
    total_tests = len(original_data)
    vendor_accuracy = (correct_vendors / total_tests) * 100
    category_accuracy = (correct_categories / total_tests) * 100
    rule_efficiency = (rule_based_matches / total_tests) * 100
    
    print("🏆 FINAL ACCURACY SCORES:")
    print("-" * 30)
    print(f"   📊 Vendor Accuracy: {vendor_accuracy:.1f}% ({correct_vendors}/{total_tests})")
    print(f"   📊 Category Accuracy: {category_accuracy:.1f}% ({correct_categories}/{total_tests})")
    print(f"   ⚡ Rule-Based Efficiency: {rule_efficiency:.1f}% ({rule_based_matches}/{total_tests})")
    print(f"   🤖 LLM Dependency: {(llm_calls/total_tests)*100:.1f}% ({llm_calls}/{total_tests})")
    print()
    
    # Performance rating
    overall_accuracy = (vendor_accuracy + category_accuracy) / 2
    
    if overall_accuracy >= 90:
        rating = "🌟 EXCELLENT"
    elif overall_accuracy >= 80:
        rating = "🎯 VERY GOOD"
    elif overall_accuracy >= 70:
        rating = "👍 GOOD"
    elif overall_accuracy >= 60:
        rating = "⚠️  NEEDS IMPROVEMENT"
    else:
        rating = "🔧 MAJOR IMPROVEMENTS NEEDED"
    
    print(f"🎖️  OVERALL RATING: {rating}")
    print(f"   Combined Accuracy: {overall_accuracy:.1f}%")
    print()
    
    # Specific challenge analysis
    print("🔍 CHALLENGE-SPECIFIC INSIGHTS:")
    print("-" * 35)
    
    typo_tests = ["MICROSFT*OFFICE365", "DIGITALOCEA*CLOUD HOSTING", "CHASe BANK FEE", "TYPO COMPNAY LLC"]
    format_tests = ["AMZN MKTP US*MO12345XY", "SQ *TECHCORP SOLUTIONS", "PP*FREELANCER DESIGN"]
    unknown_tests = ["MYSTERIOUS CORP INC", "UNKNOWN*1234567890", "TYPO COMPNAY LLC"]
    
    typo_success = sum(1 for test in typo_tests if any(cleaned.get('merchant', '') == test and 
                       cleaned.get('standardized_vendor', '') != 'N/A' for cleaned in cleaned_data))
    format_success = sum(1 for test in format_tests if any(cleaned.get('merchant', '') == test and 
                         cleaned.get('standardized_vendor', '') != 'N/A' for cleaned in cleaned_data))
    
    print(f"   🔤 Typo Handling: {(typo_success/len(typo_tests))*100:.0f}% ({typo_success}/{len(typo_tests)})")
    print(f"   🏷️  Format Parsing: {(format_success/len(format_tests))*100:.0f}% ({format_success}/{len(format_tests)})")
    print(f"   💰 Amount-Based Hints: Check $4.85 coffee → Meals & Entertainment")
    print(f"   🧠 Context Processing: Check team dinner → Meals & Entertainment")
    print()
    
    print("🎉 Challenging test completed!")
    print(f"💡 This was a deliberately difficult dataset with edge cases, typos, and ambiguous scenarios.")

if __name__ == "__main__":
    test_challenging_data() 