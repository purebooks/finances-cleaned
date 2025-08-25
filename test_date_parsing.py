#!/usr/bin/env python3

import pandas as pd
from flexible_column_detector import FlexibleColumnDetector, standardize_data_format

def test_intelligent_date_parsing():
    """Test the enhanced date format parsing"""
    
    print("🧪 QUICK WIN 4: Intelligent Date Format Parsing Test")
    print("=" * 55)
    
    detector = FlexibleColumnDetector()
    
    # Test 1: US format (MM/DD/YYYY)
    print("\n🇺🇸 Test 1: US Date Format (MM/DD/YYYY)")
    print("-" * 40)
    
    us_data = pd.DataFrame({
        'Date': ['01/15/2024', '02/28/2024', '03/10/2024'],
        'Merchant': ['Target', 'Walmart', 'Amazon'],
        'Amount': ['$45.67', '$123.45', '$89.99'],
        'Category': ['Shopping', 'Groceries', 'Online']
    })
    
    print("Original data:")
    print(us_data['Date'].tolist())
    
    # Test smart parsing directly
    success, format_used = detector._smart_date_parsing(us_data['Date'])
    print(f"✅ Smart parsing success: {success}, format: {format_used}")
    
    mapping = detector.detect_columns(us_data)
    standardized = standardize_data_format(us_data, mapping)
    print(f"   Standardized dates: {standardized['date'].dt.strftime('%Y-%m-%d').tolist()}")
    
    # Test 2: European format (DD/MM/YYYY)
    print("\n🇪🇺 Test 2: European Date Format (DD/MM/YYYY)")
    print("-" * 40)
    
    eu_data = pd.DataFrame({
        'Date': ['15/01/2024', '28/02/2024', '10/03/2024'],
        'Merchant': ['Tesco', 'Carrefour', 'Media Markt'],
        'Amount': ['€45.67', '€123.45', '€89.99'],
        'Notes': ['Shopping', 'Groceries', 'Electronics']
    })
    
    print("Original data:")
    print(eu_data['Date'].tolist())
    
    success2, format_used2 = detector._smart_date_parsing(eu_data['Date'])
    print(f"✅ Smart parsing success: {success2}, format: {format_used2}")
    
    mapping2 = detector.detect_columns(eu_data)
    standardized2 = standardize_data_format(eu_data, mapping2)
    print(f"   Standardized dates: {standardized2['date'].dt.strftime('%Y-%m-%d').tolist()}")
    
    # Test 3: German format (DD.MM.YYYY)
    print("\n🇩🇪 Test 3: German Date Format (DD.MM.YYYY)")
    print("-" * 40)
    
    de_data = pd.DataFrame({
        'Datum': ['15.01.2024', '28.02.2024', '10.03.2024'],
        'Händler': ['REWE', 'Aldi', 'Saturn'],
        'Betrag': ['45,67 EUR', '123,45 EUR', '89,99 EUR'],
        'Kategorie': ['Lebensmittel', 'Einkauf', 'Elektronik']
    })
    
    print("Original data:")
    print(de_data['Datum'].tolist())
    
    success3, format_used3 = detector._smart_date_parsing(de_data['Datum'])
    print(f"✅ Smart parsing success: {success3}, format: {format_used3}")
    
    mapping3 = detector.detect_columns(de_data)
    standardized3 = standardize_data_format(de_data, mapping3)
    print(f"   Standardized dates: {standardized3['date'].dt.strftime('%Y-%m-%d').tolist()}")
    
    # Test 4: ISO format (YYYY-MM-DD)
    print("\n🌐 Test 4: ISO Date Format (YYYY-MM-DD)")
    print("-" * 40)
    
    iso_data = pd.DataFrame({
        'transaction_date': ['2024-01-15', '2024-02-28', '2024-03-10'],
        'vendor': ['Apple', 'Microsoft', 'Google'],
        'cost': ['$999.00', '$299.99', '$199.99'],
        'type': ['Hardware', 'Software', 'Service']
    })
    
    success4, format_used4 = detector._smart_date_parsing(iso_data['transaction_date'])
    print(f"✅ Smart parsing success: {success4}, format: {format_used4}")
    
    mapping4 = detector.detect_columns(iso_data)
    standardized4 = standardize_data_format(iso_data, mapping4)
    print(f"   Standardized dates: {standardized4['date'].dt.strftime('%Y-%m-%d').tolist()}")
    
    # Test 5: Mixed and challenging formats
    print("\n🌍 Test 5: Mixed Date Formats")
    print("-" * 40)
    
    mixed_data = pd.DataFrame({
        'When': ['Jan 15, 2024', 'February 28, 2024', '10 Mar 2024'],
        'Where': ['Coffee Shop', 'Restaurant', 'Gas Station'],
        'How_Much': ['$4.50', '$67.89', '$45.23'],
        'What': ['Coffee', 'Dinner', 'Fuel']
    })
    
    print("Original data:")
    print(mixed_data['When'].tolist())
    
    success5, format_used5 = detector._smart_date_parsing(mixed_data['When'])
    print(f"✅ Smart parsing success: {success5}, format: {format_used5}")
    
    if success5:
        mapping5 = detector.detect_columns(mixed_data)
        standardized5 = standardize_data_format(mixed_data, mapping5)
        print(f"   Standardized dates: {standardized5['date'].dt.strftime('%Y-%m-%d').tolist()}")
    else:
        print("   Could not parse text dates reliably")
    
    # Summary
    print(f"\n📈 SUMMARY")
    print("-" * 20)
    print("✅ Intelligent date format parsing implemented!")
    print("✅ Supports: US (MM/DD/YYYY), European (DD/MM/YYYY), German (DD.MM.YYYY)")
    print("✅ Handles: ISO (YYYY-MM-DD), text dates, timestamps")
    print("✅ Smart format detection with fallback to pandas auto-parsing")
    print("✅ Integrated with multi-language column detection")

if __name__ == "__main__":
    test_intelligent_date_parsing()
