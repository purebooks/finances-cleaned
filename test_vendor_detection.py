#!/usr/bin/env python3
"""
Test the unknown vendor detection logic
"""

def _is_unknown_vendor(merchant: str) -> bool:
    """Check if a vendor is likely unknown and would benefit from real LLM processing."""
    if not merchant or not isinstance(merchant, str):
        return False
    
    merchant_lower = merchant.lower()
    
    # Known vendor patterns (subset of our mock logic)
    known_patterns = [
        "google", "amazon", "netflix", "spotify", "apple", "microsoft", "adobe",
        "starbucks", "mcdonald", "chipotle", "subway", "pizza", "domino",
        "uber", "lyft", "shell", "chevron", "delta", "united", "southwest",
        "target", "walmart", "costco", "home depot", "best buy",
        "cvs", "walgreens", "whole foods", "safeway", "kroger",
        "bank", "chase", "wells fargo", "american express",
        "at&t", "verizon", "comcast", "hilton", "marriott", "airbnb"
    ]
    
    # Check if merchant contains any known patterns
    for pattern in known_patterns:
        if pattern in merchant_lower:
            return False
    
    # Check for business-type keywords that suggest category but unknown vendor
    business_keywords = ["corp", "inc", "llc", "ltd", "company", "consulting", 
                        "restaurant", "cafe", "store", "shop", "market"]
    
    for keyword in business_keywords:
        if keyword in merchant_lower:
            return True  # Unknown but identifiable business
    
    # If no patterns match, it's likely unknown
    return True

def test_detection():
    """Test the vendor detection logic"""
    
    test_cases = [
        # Known vendors (should return False)
        ("Starbucks", False),
        ("Netflix", False), 
        ("Amazon", False),
        ("UBER TRIP", False),
        ("SQ *Starbucks", False),  # Should detect "starbucks"
        
        # Unknown vendors (should return True)
        ("XKCD RANDOM CORP 9872", True),
        ("Mystery Consulting LLC", True),
        ("Random Restaurant ABC", True),
        ("Joe's Unknown Shop", True),
        ("QWERTY ZXCVBNM LTD", True),
    ]
    
    print("🧪 Testing Unknown Vendor Detection")
    print("=" * 50)
    
    for merchant, expected in test_cases:
        result = _is_unknown_vendor(merchant)
        status = "✅" if result == expected else "❌"
        print(f"{status} {merchant:<25} → {'Unknown' if result else 'Known':<7} (expected: {'Unknown' if expected else 'Known'})")
    
    # Count results
    correct = sum(1 for merchant, expected in test_cases if _is_unknown_vendor(merchant) == expected)
    total = len(test_cases)
    accuracy = (correct / total) * 100
    
    print(f"\n📊 Detection Accuracy: {correct}/{total} ({accuracy:.1f}%)")

if __name__ == "__main__":
    test_detection()