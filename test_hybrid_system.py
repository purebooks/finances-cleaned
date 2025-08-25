#!/usr/bin/env python3
"""
Test the new hybrid LLM system with different transaction types
"""
import requests
import json

# Test data with different types that should trigger different processing paths
test_data = [
    # HIGH-VALUE: Should trigger AI (>$500)
    {
        "merchant": "MYSTERIOUS CORP INC",
        "amount": 1250.00,
        "description": "Consulting services",
        "date": "2024-01-15"
    },
    
    # COMPLEX PROCESSOR: Should trigger AI (SQ *)
    {
        "merchant": "SQ *TECH STARTUP CAFE",
        "amount": 45.67,
        "description": "Business meeting lunch",
        "date": "2024-01-16"
    },
    
    # SIMPLE KNOWN: Should use rules (Starbucks)
    {
        "merchant": "STARBUCKS STORE #1234",
        "amount": 4.85,
        "description": "Morning coffee",
        "date": "2024-01-17"
    },
    
    # BUSINESS ENTITY: Should trigger AI (consulting keyword)
    {
        "merchant": "ACME CONSULTING SOLUTIONS",
        "amount": 75.00,
        "description": "Advisory session",
        "date": "2024-01-18"
    },
    
    # CRYPTIC VENDOR: Should trigger AI (complex pattern)
    {
        "merchant": "PMT*XYZ123*VENDOR.COM",
        "amount": 89.99,
        "description": "Monthly service",
        "date": "2024-01-19"
    }
]

def test_hybrid_system():
    """Test the improved hybrid system"""
    url = "http://localhost:8080/process"
    
    print("🧪 TESTING IMPROVED HYBRID LLM SYSTEM")
    print("=" * 50)
    
    response = requests.post(url, json={
        "data": test_data,
        "user_intent": "standard_clean",
        "config": {
            "use_real_llm": True,
            "use_mock": False
        }
    })
    
    if response.status_code == 200:
        result = response.json()
        
        print("✅ SUCCESS! Processing completed")
        print(f"📊 Processed {len(result.get('cleaned_data', []))} transactions")
        print(f"⚡ Processing time: {result.get('processing_time', 0):.2f}s")
        
        insights = result.get('insights', {})
        print(f"🤖 AI requests: {insights.get('ai_requests', 0)}")
        print(f"💰 AI cost: ${insights.get('ai_cost', 0):.4f}")
        
        print("\n🔍 DETAILED RESULTS:")
        print("-" * 40)
        
        for i, record in enumerate(result.get('cleaned_data', [])):
            amount = record.get('Amount', 0)
            vendor = record.get('Clean Vendor', 'Unknown')
            category = record.get('Category', 'Unknown')
            
            print(f"{i+1}. ${amount} - {vendor} → {category}")
            
        return True
    else:
        print(f"❌ ERROR {response.status_code}: {response.text}")
        return False

if __name__ == "__main__":
    test_hybrid_system()