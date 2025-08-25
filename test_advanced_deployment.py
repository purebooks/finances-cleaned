#!/usr/bin/env python3
"""
Test script for Advanced LLM Flow Financial Cleaner v5.0
"""

import requests
import json
import subprocess
import sys

def get_access_token():
    """Get Google Cloud access token"""
    try:
        result = subprocess.run(['gcloud', 'auth', 'print-access-token'], 
                              capture_output=True, text=True, check=True)
        return result.stdout.strip()
    except subprocess.CalledProcessError as e:
        print(f"Error getting access token: {e}")
        return None

def test_service_health(base_url, access_token=None):
    """Test service health endpoint"""
    headers = {}
    if access_token:
        headers['Authorization'] = f'Bearer {access_token}'
    
    try:
        response = requests.get(f"{base_url}/health", headers=headers, timeout=10)
        print(f"Health check status: {response.status_code}")
        if response.status_code == 200:
            print("✅ Service is healthy!")
            return True
        else:
            print(f"❌ Health check failed: {response.text}")
            return False
    except Exception as e:
        print(f"❌ Error testing health: {e}")
        return False

def test_demo_endpoint(base_url, access_token=None):
    """Test demo endpoint"""
    headers = {'Content-Type': 'application/json'}
    if access_token:
        headers['Authorization'] = f'Bearer {access_token}'
    
    try:
        response = requests.post(f"{base_url}/demo", headers=headers, timeout=30)
        print(f"Demo endpoint status: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print("✅ Demo endpoint working!")
            print(f"   - Processed {len(data.get('demo_data', []))} transactions")
            print(f"   - AI calls: {data.get('summary_report', {}).get('processing_summary', {}).get('llm_calls', 0)}")
            return True
        else:
            print(f"❌ Demo endpoint failed: {response.text}")
            return False
    except Exception as e:
        print(f"❌ Error testing demo: {e}")
        return False

def test_process_endpoint(base_url, access_token=None):
    """Test process endpoint with sample data"""
    headers = {'Content-Type': 'application/json'}
    if access_token:
        headers['Authorization'] = f'Bearer {access_token}'
    
    sample_data = {
        "data": {
            "merchant": ["Google Cloud", "Amazon AWS", "Netflix", "Spotify"],
            "amount": [150.00, 89.99, 15.99, 9.99],
            "description": ["Cloud hosting", "Web services", "Streaming", "Music"]
        },
        "config": {
            "enable_ai": True,
            "ai_vendor_enabled": True,
            "ai_category_enabled": True,
            "enable_transaction_intelligence": True,
            "enable_source_tracking": True
        }
    }
    
    try:
        response = requests.post(f"{base_url}/process", 
                               headers=headers, 
                               json=sample_data, 
                               timeout=60)
        print(f"Process endpoint status: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print("✅ Process endpoint working!")
            print(f"   - Processed {len(data.get('cleaned_data', []))} transactions")
            print(f"   - Vendor standardizations: {data.get('summary_report', {}).get('processing_summary', {}).get('vendor_standardizations', 0)}")
            print(f"   - Category classifications: {data.get('summary_report', {}).get('processing_summary', {}).get('category_classifications', 0)}")
            print(f"   - LLM calls: {data.get('summary_report', {}).get('processing_summary', {}).get('llm_calls', 0)}")
            print(f"   - Cache hit rate: {data.get('summary_report', {}).get('cache_performance', {}).get('vendor_hit_rate', 0):.1%}")
            return True
        else:
            print(f"❌ Process endpoint failed: {response.text}")
            return False
    except Exception as e:
        print(f"❌ Error testing process: {e}")
        return False

def main():
    """Main test function"""
    print("🚀 Testing Advanced LLM Flow Financial Cleaner v5.0")
    print("=" * 60)
    
    # Service URL
    base_url = "https://ai-financial-cleaner-v5-pksi3xslca-uc.a.run.app"
    print(f"Testing service: {base_url}")
    print()
    
    # Get access token
    print("🔑 Getting access token...")
    access_token = get_access_token()
    if access_token:
        print("✅ Access token obtained")
    else:
        print("⚠️  No access token available")
    
    print()
    
    # Test health endpoint
    print("🏥 Testing health endpoint...")
    health_ok = test_service_health(base_url, access_token)
    print()
    
    # Test demo endpoint
    print("🎯 Testing demo endpoint...")
    demo_ok = test_demo_endpoint(base_url, access_token)
    print()
    
    # Test process endpoint
    print("⚙️  Testing process endpoint...")
    process_ok = test_process_endpoint(base_url, access_token)
    print()
    
    # Summary
    print("📊 Test Summary")
    print("=" * 60)
    print(f"Health Check: {'✅ PASS' if health_ok else '❌ FAIL'}")
    print(f"Demo Endpoint: {'✅ PASS' if demo_ok else '❌ FAIL'}")
    print(f"Process Endpoint: {'✅ PASS' if process_ok else '❌ FAIL'}")
    print()
    
    if health_ok and demo_ok and process_ok:
        print("🎉 All tests passed! Advanced LLM Flow is working correctly.")
        print()
        print("🔧 Advanced Features Verified:")
        print("   • Intelligent Rule > Cache > LLM Processing Flow")
        print("   • Source Tracking & Confidence Scoring")
        print("   • Transaction Intelligence (Tags, Insights, Risk)")
        print("   • Comprehensive Cost & Performance Tracking")
        print("   • Enhanced DataFrame with Attribution")
        print("   • Separate Intelligence Section")
        print()
        print("🌐 Next Steps:")
        print("   1. Open interface_v5.html in your browser")
        print("   2. Upload your financial data")
        print("   3. Configure processing options")
        print("   4. Process with advanced LLM flow")
    else:
        print("❌ Some tests failed. Check the service configuration.")
        print()
        print("🔧 Troubleshooting:")
        print("   1. Check if the service is deployed correctly")
        print("   2. Verify IAM permissions")
        print("   3. Check organization policies")
        print("   4. Review Cloud Run logs")

if __name__ == "__main__":
    main() 