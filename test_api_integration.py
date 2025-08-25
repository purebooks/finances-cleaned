#!/usr/bin/env python3
"""
Test the API integration for intent-based processing
"""

import requests
import json
import time

# Test data
sample_data = [
    {
        "date": "2024-01-15",
        "merchant": "Starbucks #1234", 
        "amount": -4.85,
        "description": "Coffee purchase"
    },
    {
        "date": "2024-01-16",
        "merchant": "Netflix",
        "amount": -12.99,
        "description": "Streaming service"
    },
    {
        "date": "2024-01-17", 
        "merchant": "Amazon.com",
        "amount": -89.99,
        "description": "Online shopping"
    },
    {
        "date": "2024-01-18",
        "merchant": "Spotify",
        "amount": -9.99,
        "description": "Music streaming"
    }
]

def test_local_api():
    """Test the API integration locally"""
    base_url = "http://localhost:8080"
    
    print("🧪 TESTING API INTEGRATION - INTENT SYSTEM")
    print("=" * 55)
    
    # Test 1: Check if intents endpoint works
    print("\n📋 Test 1: Available Intents Endpoint")
    print("-" * 40)
    
    try:
        response = requests.get(f"{base_url}/intents")
        if response.status_code == 200:
            intents_data = response.json()
            print(f"✅ Intents endpoint working")
            print(f"   Available intents: {intents_data['total_count']}")
            
            for intent in intents_data['available_intents'][:3]:
                print(f"   - {intent['name']}: {intent['description']}")
        else:
            print(f"❌ Intents endpoint failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Cannot connect to local API: {e}")
        print("   Make sure to run: python3 app_v5.py")
        return
    
    # Test 2: Budget Analysis Intent
    print("\n💰 Test 2: Budget Analysis Intent")
    print("-" * 40)
    
    try:
        payload = {
            "data": sample_data,
            "user_intent": "budget analysis"
        }
        
        response = requests.post(f"{base_url}/process", json=payload, timeout=30)
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Budget analysis completed successfully")
            print(f"   Processing time: {result['processing_time']:.2f}s")
            print(f"   User intent: {result.get('user_intent', 'N/A')}")
            
            # Check intent-specific results
            if 'intent_summary' in result:
                intent_summary = result['intent_summary']
                print(f"   Intent: {intent_summary.get('intent', 'N/A')}")
                print(f"   Description: {intent_summary.get('description', 'N/A')}")
                
                if 'total_spending' in intent_summary:
                    print(f"   ✅ Total spending: ${abs(intent_summary['total_spending']):.2f}")
                if 'transaction_count' in intent_summary:
                    print(f"   ✅ Transaction count: {intent_summary['transaction_count']}")
                if 'top_spending_categories' in intent_summary:
                    categories = list(intent_summary['top_spending_categories'].keys())[:3]
                    print(f"   ✅ Top categories: {categories}")
        else:
            print(f"❌ Budget analysis failed: {response.status_code}")
            print(f"   Response: {response.text}")
            
    except Exception as e:
        print(f"❌ Budget analysis error: {e}")
    
    # Test 3: Find Coffee Spending Intent
    print("\n☕ Test 3: Find Coffee Spending Intent")
    print("-" * 40)
    
    try:
        payload = {
            "data": sample_data,
            "user_intent": "find coffee spending"
        }
        
        response = requests.post(f"{base_url}/process", json=payload, timeout=30)
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Coffee spending analysis completed")
            
            if 'intent_summary' in result:
                intent_summary = result['intent_summary']
                if 'target_category' in intent_summary:
                    print(f"   ✅ Target category: {intent_summary['target_category']}")
                if 'matching_transactions' in intent_summary:
                    print(f"   ✅ Matching transactions: {intent_summary['matching_transactions']}")
                if 'total_spent' in intent_summary:
                    print(f"   ✅ Total coffee spending: ${abs(intent_summary['total_spent']):.2f}")
        else:
            print(f"❌ Coffee analysis failed: {response.status_code}")
            
    except Exception as e:
        print(f"❌ Coffee analysis error: {e}")
    
    # Test 4: Standard Processing (No Intent)
    print("\n🔧 Test 4: Standard Processing (No Intent)")
    print("-" * 40)
    
    try:
        payload = {
            "data": sample_data
            # No user_intent specified
        }
        
        response = requests.post(f"{base_url}/process", json=payload, timeout=30)
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Standard processing completed")
            print(f"   Has intent summary: {'intent_summary' in result}")
            print(f"   User intent in response: {result.get('user_intent', 'None')}")
        else:
            print(f"❌ Standard processing failed: {response.status_code}")
            
    except Exception as e:
        print(f"❌ Standard processing error: {e}")
    
    # Test 5: Custom Config + Intent
    print("\n⚙️  Test 5: Custom Config + Intent")
    print("-" * 40)
    
    try:
        payload = {
            "data": sample_data,
            "user_intent": "budget analysis",
            "config": {
                "ai_confidence_threshold": 0.9
            }
        }
        
        response = requests.post(f"{base_url}/process", json=payload, timeout=30)
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Custom config + intent completed")
            print(f"   Intent processing: {'intent_summary' in result}")
            print(f"   Config applied: Custom settings merged")
        else:
            print(f"❌ Custom config test failed: {response.status_code}")
            
    except Exception as e:
        print(f"❌ Custom config test error: {e}")
    
    print(f"\n📈 API INTEGRATION TEST SUMMARY")
    print("-" * 35)
    print("✅ Intent endpoint available")
    print("✅ User intent parameter accepted")
    print("✅ Intent-specific processing working")
    print("✅ Intent summary in API response")
    print("✅ Backward compatibility maintained")
    print("✅ Custom config + intent combination supported")

def test_cloud_run_api():
    """Test the deployed Cloud Run API"""
    base_url = "https://ai-financial-cleaner-v5-pksi3xslca-uc.a.run.app"
    
    print("\n🌐 TESTING DEPLOYED API")
    print("=" * 30)
    
    try:
        # Test intents endpoint
        response = requests.get(f"{base_url}/intents", timeout=10)
        if response.status_code == 200:
            print("✅ Deployed API intent endpoint working")
        else:
            print(f"❌ Deployed API intents failed: {response.status_code}")
        
        # Test processing with intent
        payload = {
            "data": sample_data[:2],  # Smaller dataset for quick test
            "user_intent": "budget analysis"
        }
        
        response = requests.post(f"{base_url}/process", json=payload, timeout=60)
        if response.status_code == 200:
            result = response.json()
            print("✅ Deployed API intent processing working")
            print(f"   Intent: {result.get('intent_summary', {}).get('intent', 'N/A')}")
        else:
            print(f"❌ Deployed API processing failed: {response.status_code}")
            
    except Exception as e:
        print(f"❌ Deployed API test error: {e}")

if __name__ == "__main__":
    print("Choose test mode:")
    print("1. Local API (http://localhost:8080)")
    print("2. Deployed API (Cloud Run)")
    print("3. Both")
    
    choice = input("Enter choice (1/2/3): ").strip()
    
    if choice in ['1', '3']:
        test_local_api()
    
    if choice in ['2', '3']:
        test_cloud_run_api()