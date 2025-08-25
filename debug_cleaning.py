#!/usr/bin/env python3
"""
Debug the vendor cleaning logic
"""

from llm_client_v2 import LLMClient

def test_cleaning():
    client = LLMClient(use_mock=True)
    
    # Test some failing cases
    failing_cases = [
        "SQ *Netflix",
        "SAFEWAY", 
        "AUTO PAY Kroger",
        "Apple",
        "Google",
        "BestBuy",
        "Gas Company #123456",
        "Wells Fargo.COM"
    ]
    
    print("🧪 Testing vendor cleaning and category classification:")
    print("=" * 60)
    
    for vendor in failing_cases:
        cleaned = client._clean_vendor_for_category(vendor)
        response = client._mock_llm_response("category", vendor)
        print(f"\n🔍 '{vendor}'")
        print(f"   → Cleaned: '{cleaned}'")
        print(f"   → Category: {response}")

if __name__ == "__main__":
    test_cleaning()