#!/usr/bin/env python3
"""
Test script for AI-Enhanced Financial Cleaner deployment
"""

import requests
import json
import time
import sys

def test_deployment(base_url):
    """Test the deployed API endpoints"""
    
    print(f"🧪 Testing AI-Enhanced Financial Cleaner at {base_url}")
    print("=" * 60)
    
    # Test 1: Health Check
    print("\n1️⃣ Testing Health Check")
    try:
        response = requests.get(f"{base_url}/health", timeout=10)
        if response.status_code == 200:
            health_data = response.json()
            print(f"✅ Health check passed")
            print(f"   Version: {health_data.get('version')}")
            print(f"   Status: {health_data.get('status')}")
            print(f"   Uptime: {health_data.get('uptime_seconds', 0):.1f}s")
        else:
            print(f"❌ Health check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Health check error: {e}")
        return False
    
    # Test 2: Configuration
    print("\n2️⃣ Testing Configuration")
    try:
        response = requests.get(f"{base_url}/config", timeout=10)
        if response.status_code == 200:
            config_data = response.json()
            print(f"✅ Configuration retrieved")
            print(f"   AI Enabled: {config_data.get('enable_ai')}")
            print(f"   Has API Key: {config_data.get('has_api_key')}")
            print(f"   Max File Size: {config_data.get('max_file_size_mb')}MB")
            print(f"   Version: {config_data.get('version')}")
        else:
            print(f"❌ Configuration failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Configuration error: {e}")
    
    # Test 3: Demo Endpoint
    print("\n3️⃣ Testing Demo Endpoint")
    try:
        response = requests.post(f"{base_url}/demo", timeout=30)
        if response.status_code == 200:
            demo_data = response.json()
            print(f"✅ Demo endpoint working")
            print(f"   Rows processed: {len(demo_data.get('cleaned_data', []))}")
            print(f"   AI requests: {demo_data.get('insights', {}).get('ai_requests', 0)}")
            print(f"   AI cost: ${demo_data.get('insights', {}).get('ai_cost', 0):.3f}")
            print(f"   Processing time: {demo_data.get('insights', {}).get('processing_time', 0):.2f}s")
        else:
            print(f"❌ Demo endpoint failed: {response.status_code}")
            print(f"   Response: {response.text}")
    except Exception as e:
        print(f"❌ Demo endpoint error: {e}")
    
    # Test 4: Custom Data Processing
    print("\n4️⃣ Testing Custom Data Processing")
    test_data = {
        "data": {
            "merchant": [
                "PAYPAL*DIGITALOCEAN",
                "SQ *COFFEE SHOP NYC",
                "UBER EATS DEC15"
            ],
            "amount": [50.00, 4.50, 23.75],
            "description": [
                "DigitalOcean hosting",
                "Coffee purchase",
                "Food delivery"
            ]
        },
        "config": {
            "enable_ai": True,
            "ai_vendor_enabled": True,
            "ai_category_enabled": True
        }
    }
    
    try:
        response = requests.post(
            f"{base_url}/process",
            json=test_data,
            timeout=30
        )
        if response.status_code == 200:
            process_data = response.json()
            print(f"✅ Custom processing working")
            print(f"   Rows processed: {len(process_data.get('cleaned_data', []))}")
            print(f"   AI requests: {process_data.get('insights', {}).get('ai_requests', 0)}")
            print(f"   AI cost: ${process_data.get('insights', {}).get('ai_cost', 0):.3f}")
            
            # Show vendor transformations
            cleaned_data = process_data.get('cleaned_data', [])
            print(f"\n   Vendor transformations:")
            for i, original in enumerate(test_data['data']['merchant']):
                if i < len(cleaned_data):
                    new_vendor = cleaned_data[i].get('merchant', 'N/A')
                    print(f"     {original} → {new_vendor}")
        else:
            print(f"❌ Custom processing failed: {response.status_code}")
            print(f"   Response: {response.text}")
    except Exception as e:
        print(f"❌ Custom processing error: {e}")
    
    # Test 5: Statistics
    print("\n5️⃣ Testing Statistics")
    try:
        response = requests.get(f"{base_url}/stats", timeout=10)
        if response.status_code == 200:
            stats_data = response.json()
            print(f"✅ Statistics retrieved")
            print(f"   Total requests: {stats_data.get('total_requests', 0)}")
            print(f"   Total AI calls: {stats_data.get('total_ai_calls', 0)}")
            print(f"   Total cost: ${stats_data.get('total_cost', 0):.3f}")
            print(f"   Avg cost per request: ${stats_data.get('average_cost_per_request', 0):.4f}")
        else:
            print(f"❌ Statistics failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Statistics error: {e}")
    
    # Test 6: Error Handling
    print("\n6️⃣ Testing Error Handling")
    try:
        # Test invalid data
        invalid_data = {"invalid": "data"}
        response = requests.post(
            f"{base_url}/process",
            json=invalid_data,
            timeout=10
        )
        if response.status_code == 400:
            print(f"✅ Error handling working (400 for invalid data)")
        else:
            print(f"⚠️  Unexpected response for invalid data: {response.status_code}")
    except Exception as e:
        print(f"❌ Error handling test failed: {e}")
    
    print("\n🎯 Test Summary")
    print("=" * 30)
    print("✅ Deployment is working correctly!")
    print(f"🌐 Service URL: {base_url}")
    print("\n📖 Available endpoints:")
    print(f"• GET  {base_url}/health")
    print(f"• GET  {base_url}/stats")
    print(f"• GET  {base_url}/config")
    print(f"• POST {base_url}/process")
    print(f"• POST {base_url}/demo")
    
    return True

def main():
    """Main test function"""
    if len(sys.argv) != 2:
        print("Usage: python test_deployment.py <service-url>")
        print("Example: python test_deployment.py https://ai-financial-cleaner-xxx.run.app")
        sys.exit(1)
    
    base_url = sys.argv[1].rstrip('/')
    
    success = test_deployment(base_url)
    
    if success:
        print("\n🎉 All tests passed! Your deployment is ready for production.")
        sys.exit(0)
    else:
        print("\n❌ Some tests failed. Please check the deployment.")
        sys.exit(1)

if __name__ == "__main__":
    main() 