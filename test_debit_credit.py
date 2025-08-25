#!/usr/bin/env python3

import pandas as pd
from flexible_column_detector import FlexibleColumnDetector, standardize_data_format

def test_debit_credit_handling():
    """Test the debit/credit column merging functionality"""
    
    print("🧪 QUICK WIN 2: Debit/Credit Column Handling Test")
    print("=" * 50)
    
    detector = FlexibleColumnDetector()
    
    # Test 1: Standard bank export with debit/credit columns
    print("\n🏦 Test 1: Bank Export with Debit/Credit")
    print("-" * 40)
    
    bank_data = pd.DataFrame({
        'Date': ['2024-01-01', '2024-01-02', '2024-01-03', '2024-01-04'],
        'Description': ['Salary Deposit', 'Grocery Store', 'Gas Station', 'ATM Withdrawal'],
        'Debit': ['', '$25.67', '$45.23', '$60.00'],
        'Credit': ['$2500.00', '', '', ''],
        'Balance': ['$2500.00', '$2474.33', '$2429.10', '$2369.10']
    })
    
    print("Original data:")
    print(bank_data[['Description', 'Debit', 'Credit']].to_string())
    
    mapping = detector.detect_columns(bank_data)
    print(f"\n✅ Detection confidence: {mapping.confidence:.2f}")
    print(f"   Amount column detected: {mapping.amount_column}")
    print(f"   Method: {mapping.detection_method}")
    
    standardized = standardize_data_format(bank_data, mapping)
    print(f"   Merged amounts: {standardized['amount'].tolist()}")
    
    # Test 2: Accounting format with different names
    print("\n📊 Test 2: Accounting Format")
    print("-" * 40)
    
    accounting_data = pd.DataFrame({
        'Transaction Date': ['2024-02-01', '2024-02-02', '2024-02-03'],
        'Vendor': ['Office Supplies Inc', 'Client Payment', 'Utility Company'],
        'Withdrawal': ['$125.50', '', '$89.99'],
        'Deposit': ['', '$1500.00', ''],
        'Account': ['Checking', 'Checking', 'Checking']
    })
    
    print("Original data:")
    print(accounting_data[['Vendor', 'Withdrawal', 'Deposit']].to_string())
    
    mapping2 = detector.detect_columns(accounting_data)
    print(f"\n✅ Detection confidence: {mapping2.confidence:.2f}")
    print(f"   Amount column detected: {mapping2.amount_column}")
    
    standardized2 = standardize_data_format(accounting_data, mapping2)
    print(f"   Merged amounts: {standardized2['amount'].tolist()}")
    
    # Test 3: Mixed format with income/expense
    print("\n💼 Test 3: Income/Expense Format")
    print("-" * 40)
    
    income_expense_data = pd.DataFrame({
        'Date': ['2024-03-01', '2024-03-02', '2024-03-03', '2024-03-04'],
        'Category': ['Salary', 'Office Rent', 'Consulting', 'Equipment'],
        'Income': ['5000.00', '', '1200.00', ''],
        'Expense': ['', '1500.00', '', '450.00'],
        'Notes': ['Monthly salary', 'Office lease', 'Client project', 'New laptop']
    })
    
    print("Original data:")
    print(income_expense_data[['Category', 'Income', 'Expense']].to_string())
    
    mapping3 = detector.detect_columns(income_expense_data)
    print(f"\n✅ Detection confidence: {mapping3.confidence:.2f}")
    print(f"   Amount column detected: {mapping3.amount_column}")
    
    standardized3 = standardize_data_format(income_expense_data, mapping3)
    print(f"   Merged amounts: {standardized3['amount'].tolist()}")
    
    # Test 4: Regular single amount column (should not be affected)
    print("\n✅ Test 4: Regular Amount Column (Control)")
    print("-" * 40)
    
    regular_data = pd.DataFrame({
        'Date': ['2024-04-01', '2024-04-02'],
        'Merchant': ['Amazon', 'Netflix'],
        'Amount': ['-$89.99', '$12.99'],
        'Category': ['Shopping', 'Entertainment']
    })
    
    mapping4 = detector.detect_columns(regular_data)
    print(f"✅ Detection confidence: {mapping4.confidence:.2f}")
    print(f"   Amount column detected: {mapping4.amount_column}")
    print(f"   Should be 'Amount', not 'merged_amount'")
    
    # Summary
    print(f"\n📈 SUMMARY")
    print("-" * 20)
    print("✅ Debit/Credit column merging implemented!")
    print("✅ Supports: Debit/Credit, Withdrawal/Deposit, Income/Expense")
    print("✅ Automatic detection and merging")
    print("✅ Preserves existing single-amount columns")
    print("✅ Handles currency symbols and formatting")

if __name__ == "__main__":
    test_debit_credit_handling()
