#!/usr/bin/env python3
"""
Test Enhanced Caching System
Demonstrates improved cache hit rates with vendor normalization and amount ranges
"""
import requests
import json

def test_caching_improvements():
    """Test the enhanced caching system with similar transactions"""
    
    # Test data designed to demonstrate cache improvements
    cache_test_data = [
        # Group 1: Same vendor, different store numbers (should HIT cache)
        {
            "merchant": "STARBUCKS STORE #1234",
            "amount": 4.85,
            "description": "Morning coffee",
            "date": "2024-01-15"
        },
        {
            "merchant": "STARBUCKS STORE #5678", # Should hit cache due to normalization
            "amount": 4.90,  # Should hit cache due to amount range (both in "micro" range)
            "description": "Afternoon coffee",
            "date": "2024-01-16"
        },
        {
            "merchant": "STARBUCKS STORE #9999", # Should hit cache
            "amount": 4.75,  # Should hit cache (same range)
            "description": "Evening coffee",
            "date": "2024-01-17"
        },
        
        # Group 2: Payment processor variations (should HIT cache)
        {
            "merchant": "SQ *TECH CAFE DOWNTOWN",
            "amount": 12.50,
            "description": "Business lunch",
            "date": "2024-01-18"
        },
        {
            "merchant": "SQ *TECH CAFE DOWNTOWN", # Exact match - should hit
            "amount": 13.25,  # Similar amount range - should hit
            "description": "Another lunch",
            "date": "2024-01-19"
        },
        
        # Group 3: Amount range variations (should HIT cache)
        {
            "merchant": "AMAZON.COM",
            "amount": 25.99,
            "description": "Office supplies",
            "date": "2024-01-20"
        },
        {
            "merchant": "AMAZON.COM AMZN.MKP", # Should normalize to AMAZON.COM
            "amount": 27.50,  # Both in "medium" range ($15-50)
            "description": "More supplies",
            "date": "2024-01-21"
        },
        {
            "merchant": "AMAZON.COM",
            "amount": 24.75,  # Same range again
            "description": "Even more supplies",
            "date": "2024-01-22"
        }
    ]
    
    print("🧪 TESTING ENHANCED CACHING SYSTEM")
    print("=" * 50)
    
    url = "http://localhost:8080/process"
    
    # First run - populate cache
    print("📊 FIRST RUN - Populating Cache:")
    response1 = requests.post(url, json={
        "data": cache_test_data,
        "user_intent": "standard_clean",
        "config": {"use_real_llm": True, "use_mock": False}
    })
    
    if response1.status_code == 200:
        result1 = response1.json()
        insights1 = result1.get('insights', {})
        print(f"  ✅ Processed {len(result1.get('cleaned_data', []))} transactions")
        print(f"  🤖 AI requests: {insights1.get('ai_requests', 0)}")
        print(f"  💰 AI cost: ${insights1.get('ai_cost', 0):.4f}")
        print(f"  ⏱️  Processing time: {result1.get('processing_time', 0):.2f}s")
    else:
        print(f"❌ First run failed: {response1.status_code}")
        return
    
    print(f"\n🔄 SECOND RUN - Testing Cache Hits:")
    
    # Second run - should hit cache frequently
    response2 = requests.post(url, json={
        "data": cache_test_data,  # Same data
        "user_intent": "standard_clean",
        "config": {"use_real_llm": True, "use_mock": False}
    })
    
    if response2.status_code == 200:
        result2 = response2.json()
        insights2 = result2.get('insights', {})
        print(f"  ✅ Processed {len(result2.get('cleaned_data', []))} transactions")
        print(f"  🤖 AI requests: {insights2.get('ai_requests', 0)}")
        print(f"  💰 AI cost: ${insights2.get('ai_cost', 0):.4f}")
        print(f"  ⏱️  Processing time: {result2.get('processing_time', 0):.2f}s")
        
        # Calculate improvements
        cost_reduction = ((insights1.get('ai_cost', 0) - insights2.get('ai_cost', 0)) / max(insights1.get('ai_cost', 0), 0.001)) * 100
        time_reduction = ((result1.get('processing_time', 0) - result2.get('processing_time', 0)) / max(result1.get('processing_time', 0), 0.1)) * 100
        
        print(f"\n🎯 CACHING IMPROVEMENTS:")
        print(f"  💰 Cost reduction: {cost_reduction:.1f}%")
        print(f"  ⚡ Speed improvement: {time_reduction:.1f}%")
        
    else:
        print(f"❌ Second run failed: {response2.status_code}")
        return
    
    # Test vendor normalization examples
    print(f"\n🔍 VENDOR NORMALIZATION EXAMPLES:")
    test_vendors = [
        "STARBUCKS STORE #1234",
        "STARBUCKS STORE #5678", 
        "SQ *TECH CAFE DOWNTOWN",
        "AMAZON.COM AMZN.MKP",
        "PAYPAL *MYSTERIOUS CORP"
    ]
    
    # Import the cache class to test normalization
    try:
        import sys
        sys.path.append('.')
        from advanced_llm_components import IntelligentCache
        
        cache = IntelligentCache()
        for vendor in test_vendors:
            normalized = cache._normalize_vendor_name(vendor)
            print(f"  '{vendor}' → '{normalized}'")
            
    except ImportError as e:
        print(f"  Could not import cache class: {e}")

def test_amount_ranges():
    """Test amount range grouping"""
    
    print(f"\n📊 AMOUNT RANGE GROUPING:")
    amounts = [0, 2.50, 4.85, 12.50, 25.99, 89.99, 250.00, 1250.00]
    
    try:
        from advanced_llm_components import IntelligentCache
        cache = IntelligentCache()
        
        for amount in amounts:
            range_name = cache._get_amount_range(amount)
            print(f"  ${amount:>7.2f} → {range_name}")
            
    except ImportError as e:
        print(f"  Could not test amount ranges: {e}")

def main():
    """Main testing function"""
    test_caching_improvements()
    test_amount_ranges()
    
    print(f"\n💡 EXPECTED BENEFITS:")
    print(f"  • Similar vendors hit cache (Starbucks Store #1234 = Store #5678)")
    print(f"  • Similar amounts hit cache ($4.85 ≈ $4.90 both 'micro' range)")
    print(f"  • Payment processors normalized (SQ * prefix removed)")
    print(f"  • Significant cost reduction on repeated processing")

if __name__ == "__main__":
    main()