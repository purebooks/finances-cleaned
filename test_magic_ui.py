#!/usr/bin/env python3
"""
Test the Magic Text Bar UI functionality
"""

import os
import webbrowser
import time
import json
from pathlib import Path

def create_sample_data():
    """Create sample financial data for testing"""
    sample_data = [
        {
            "date": "2024-01-15",
            "merchant": "Starbucks Store #1234",
            "amount": -4.85,
            "description": "Coffee and pastry",
            "row_id": 1,
            "export_timestamp": "2024-01-20T10:30:00Z"
        },
        {
            "date": "2024-01-16", 
            "merchant": "Netflix.com",
            "amount": -12.99,
            "description": "Monthly subscription",
            "row_id": 2,
            "export_timestamp": "2024-01-20T10:30:00Z"
        },
        {
            "date": "2024-01-17",
            "merchant": "Amazon.com AMZN.COM/BILL",
            "amount": -89.99,
            "description": "Online shopping order #123",
            "row_id": 3,
            "export_timestamp": "2024-01-20T10:30:00Z"
        },
        {
            "date": "2024-01-18",
            "merchant": "Spotify AB",
            "amount": -9.99,
            "description": "Music streaming premium",
            "row_id": 4,
            "export_timestamp": "2024-01-20T10:30:00Z"
        },
        {
            "date": "2024-01-19",
            "merchant": "Blue Bottle Coffee",
            "amount": -6.75,
            "description": "Iced coffee and muffin",
            "row_id": 5,
            "export_timestamp": "2024-01-20T10:30:00Z"
        },
        {
            "date": "2024-01-20",
            "merchant": "PAYPAL *DROPBOX",
            "amount": -11.99,
            "description": "Cloud storage subscription",
            "row_id": 6,
            "export_timestamp": "2024-01-20T10:30:00Z"
        }
    ]
    
    # Save as JSON
    with open('sample_financial_data.json', 'w') as f:
        json.dump(sample_data, f, indent=2)
    
    # Save as CSV
    import csv
    with open('sample_financial_data.csv', 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=sample_data[0].keys())
        writer.writeheader()
        writer.writerows(sample_data)
    
    print("✅ Created sample data files:")
    print("   - sample_financial_data.json")
    print("   - sample_financial_data.csv")

def test_magic_ui():
    """Test the Magic Text Bar UI"""
    print("🎯 TESTING MAGIC TEXT BAR UI")
    print("=" * 40)
    
    # Check if UI file exists
    ui_file = Path('magic_interface.html')
    if not ui_file.exists():
        print("❌ magic_interface.html not found")
        return False
    
    # Create sample data
    create_sample_data()
    
    # Get absolute path to UI file
    ui_path = ui_file.absolute()
    
    print(f"\n🌐 Opening Magic UI in browser...")
    print(f"   File: {ui_path}")
    
    # Open in browser
    webbrowser.open(f'file://{ui_path}')
    
    print("\n✨ MAGIC TEXT BAR UI TEST INSTRUCTIONS")
    print("-" * 45)
    print("1. 📊 Upload the sample CSV or JSON file")
    print("2. 🪄 Try these magic intents:")
    print("   • 'budget analysis'")
    print("   • 'find coffee spending'") 
    print("   • 'subscription audit'")
    print("   • 'expense report'")
    print("3. ✨ Click 'Clean My Data' to see the magic!")
    print("4. 📥 Download your results")
    
    print("\n🎯 TESTING SCENARIOS")
    print("-" * 20)
    print("• Test intent suggestions (click the chips)")
    print("• Test drag & drop file upload")
    print("• Test different intent phrases")
    print("• Test responsive design (resize browser)")
    print("• Test download functionality")
    
    print("\n🚨 TROUBLESHOOTING")
    print("-" * 18)
    print("• Make sure the API is running: python3 app_v5.py")
    print("• Check browser console for errors (F12)")
    print("• Verify API_BASE_URL in the HTML file")
    
    return True

def check_api_status():
    """Check if the API is running"""
    import requests
    
    try:
        response = requests.get('http://localhost:8080/health', timeout=5)
        if response.status_code == 200:
            print("✅ API is running and healthy")
            
            # Check intents endpoint
            intents_response = requests.get('http://localhost:8080/intents', timeout=5)
            if intents_response.status_code == 200:
                data = intents_response.json()
                print(f"✅ Intents endpoint working ({data['total_count']} intents)")
            else:
                print("⚠️  Intents endpoint not working")
                
            return True
        else:
            print(f"❌ API health check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ API not running: {e}")
        print("   Start it with: python3 app_v5.py")
        return False

def main():
    print("🎯 MAGIC TEXT BAR UI TESTER")
    print("=" * 30)
    
    # Check API status
    print("\n🔍 Checking API status...")
    api_running = check_api_status()
    
    if not api_running:
        print("\n⚠️  API is not running. Starting it now...")
        print("   You may need to start it manually: python3 app_v5.py")
        
    # Test UI
    print("\n🎨 Testing Magic UI...")
    success = test_magic_ui()
    
    if success:
        print("\n🎉 Magic Text Bar UI is ready!")
        print("   The interface should be open in your browser")
        
        if api_running:
            print("   ✅ Full functionality available")
        else:
            print("   ⚠️  Start the API for full functionality")
    else:
        print("\n❌ UI test failed")

if __name__ == "__main__":
    main()