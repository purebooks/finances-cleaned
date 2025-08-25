#!/usr/bin/env python3

import pandas as pd
import numpy as np
from flexible_column_detector import FlexibleColumnDetector, standardize_data_format

def test_various_formats():
    """Test the flexible detector with different financial data formats"""
    
    print("🧪 FLEXIBLE COLUMN DETECTOR TESTS")
    print("=" * 50)
    
    detector = FlexibleColumnDetector()
    
    # Test 1: Standard format (should be easy)
    print("\n📊 Test 1: Standard Format")
    print("-" * 30)
    
    standard_data = pd.DataFrame({
        'merchant': ['Google LLC', 'Netflix Inc', 'Uber Technologies'],
        'amount': [15.00, 12.99, 25.47],
        'date': ['2024-01-01', '2024-01-02', '2024-01-03'],
        'description': ['Cloud storage', 'Streaming service', 'Ride sharing']
    })
    
    mapping1 = detector.detect_columns(standard_data)
    print(f"✅ Standard format - Confidence: {mapping1.confidence:.2f}")
    print(f"   Amount: {mapping1.amount_column}")
    print(f"   Merchant: {mapping1.merchant_column}")
    print(f"   Date: {mapping1.date_column}")
    print(f"   Method: {mapping1.detection_method}")
    
    # Test 2: Bank export format (common alternative)
    print("\n🏦 Test 2: Bank Export Format")
    print("-" * 30)
    
    bank_data = pd.DataFrame({
        'Transaction Date': ['01/15/2024', '01/16/2024', '01/17/2024'],
        'Description': ['AMAZON.COM*MKT', 'STARBUCKS #1234', 'PAYPAL *TRANSFER'],
        'Amount': [-89.99, -4.85, -150.00],
        'Balance': [1500.00, 1495.15, 1345.15]
    })
    
    mapping2 = detector.detect_columns(bank_data)
    print(f"✅ Bank format - Confidence: {mapping2.confidence:.2f}")
    print(f"   Amount: {mapping2.amount_column}")
    print(f"   Merchant: {mapping2.merchant_column}")
    print(f"   Date: {mapping2.date_column}")
    print(f"   Method: {mapping2.detection_method}")
    
    # Test 3: Accounting software format
    print("\n📋 Test 3: Accounting Software Format")
    print("-" * 30)
    
    accounting_data = pd.DataFrame({
        'Vendor Name': ['Microsoft Corporation', 'Adobe Systems', 'Zoom Video'],
        'Cost': [99.00, 52.99, 19.99],
        'Invoice Date': ['2024-02-01', '2024-02-02', '2024-02-03'],
        'Category': ['Software', 'Creative Tools', 'Communication'],
        'Reference': ['INV-001', 'INV-002', 'INV-003']
    })
    
    mapping3 = detector.detect_columns(accounting_data)
    print(f"✅ Accounting format - Confidence: {mapping3.confidence:.2f}")
    print(f"   Amount: {mapping3.amount_column}")
    print(f"   Merchant: {mapping3.merchant_column}")
    print(f"   Date: {mapping3.date_column}")
    print(f"   Method: {mapping3.detection_method}")
    
    # Test 4: No headers (challenging)
    print("\n❓ Test 4: No Headers (Position-based)")
    print("-" * 30)
    
    no_headers_data = pd.DataFrame([
        ['2024-03-01', 'Target Corporation', 125.67],
        ['2024-03-02', 'Best Buy Co Inc', 299.99],
        ['2024-03-03', 'Home Depot Inc', 89.45]
    ], columns=['Column1', 'Column2', 'Column3'])
    
    mapping4 = detector.detect_columns(no_headers_data)
    print(f"✅ No headers - Confidence: {mapping4.confidence:.2f}")
    print(f"   Amount: {mapping4.amount_column}")
    print(f"   Merchant: {mapping4.merchant_column}")
    print(f"   Date: {mapping4.date_column}")
    print(f"   Method: {mapping4.detection_method}")
    
    # Test 5: Weird format (really challenging)
    print("\n🤔 Test 5: Unusual Format")
    print("-" * 30)
    
    weird_data = pd.DataFrame({
        'When': ['March 5, 2024', 'March 6, 2024'],
        'Who': ['STRIPE*UNKNOWN LLC', 'GOOGL *WORKSPACE'],
        'How Much': ['$49.99', '$12.00'],
        'Why': ['Monthly subscription', 'Cloud storage'],
        'ID': ['TXN-001', 'TXN-002']
    })
    
    mapping5 = detector.detect_columns(weird_data)
    print(f"✅ Unusual format - Confidence: {mapping5.confidence:.2f}")
    print(f"   Amount: {mapping5.amount_column}")
    print(f"   Merchant: {mapping5.merchant_column}")
    print(f"   Date: {mapping5.date_column}")
    print(f"   Method: {mapping5.detection_method}")
    
    # Test standardization
    print("\n🔄 Test 6: Data Standardization")
    print("-" * 30)
    
    standardized = standardize_data_format(bank_data, mapping2)
    print(f"✅ Standardized columns: {list(standardized.columns)}")
    print("Sample standardized data:")
    print(standardized.head())
    
    # Summary
    print(f"\n📈 SUMMARY")
    print("-" * 20)
    
    all_mappings = [mapping1, mapping2, mapping3, mapping4, mapping5]
    avg_confidence = sum(m.confidence for m in all_mappings) / len(all_mappings)
    success_rate = sum(1 for m in all_mappings if m.confidence > 0.5) / len(all_mappings)
    
    print(f"Average Confidence: {avg_confidence:.2f}")
    print(f"Success Rate (>50%): {success_rate*100:.0f}%")
    print(f"High Confidence (>80%): {sum(1 for m in all_mappings if m.confidence > 0.8)}/5")
    
    # Get suggestions for manual mapping
    print(f"\n💡 Test 7: Manual Mapping Suggestions")
    print("-" * 30)
    
    suggestions = detector.get_column_suggestions(weird_data)
    print("Suggestions for unusual format:")
    for field_type, candidates in suggestions.items():
        if candidates:
            print(f"   {field_type}: {candidates[:3]}")  # Top 3 suggestions

if __name__ == "__main__":
    test_various_formats() 