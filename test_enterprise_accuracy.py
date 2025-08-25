#!/usr/bin/env python3
"""
Enterprise-grade accuracy testing for transaction categorization
Tests the cleaning system with challenging real-world data
"""

import json
import pandas as pd
import time
from datetime import datetime

# Import the production cleaning system
try:
    from production_cleaner_ai_v5 import AIEnhancedProductionCleanerV5
    from advanced_llm_components import AdvancedLLMProcessor
    from llm_client_v2 import LLMClient
    print("✅ Successfully imported production modules")
except ImportError as e:
    print(f"❌ Import error: {e}")
    exit(1)

def load_test_data():
    """Load the enterprise stress test data"""
    try:
        with open('enterprise_stress_test_data.json', 'r') as f:
            data = json.load(f)
        print(f"✅ Loaded {len(data)} test transactions")
        return data
    except FileNotFoundError:
        print("❌ enterprise_stress_test_data.json not found")
        exit(1)

def convert_to_dataframe(data):
    """Convert JSON data to DataFrame format"""
    df = pd.DataFrame(data)
    
    # Rename columns to match expected format
    column_mapping = {
        'date': 'Date',
        'merchant': 'Merchant', 
        'amount': 'Amount',
        'description': 'Description',
        'memo': 'Memo'
    }
    
    df = df.rename(columns=column_mapping)
    print(f"✅ Converted to DataFrame: {df.shape}")
    return df

def test_cleaning_accuracy(df):
    """Test the cleaning system accuracy"""
    print("\n🧪 TESTING CLEANING ACCURACY...")
    
    try:
        # Initialize the cleaning system with DataFrame and config
        config = {
            'use_real_llm': True,
            'ai_confidence_threshold': 0.85
        }
        cleaner = AIEnhancedProductionCleanerV5(df, config=config)
        
        start_time = time.time()
        
        # Process the data (the system processes the DataFrame passed to constructor)
        processed_df, metadata = cleaner.process_data()
        
        # Convert to list of dictionaries for analysis
        results = processed_df.to_dict('records')
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        print(f"⏱️  Processing time: {processing_time:.2f} seconds")
        print(f"📊 Processed {len(results)} transactions")
        
        return results, processing_time
        
    except Exception as e:
        print(f"❌ Cleaning error: {e}")
        return None, 0

def analyze_results(original_df, results):
    """Analyze the cleaning results for accuracy"""
    print("\n📊 ANALYZING RESULTS...")
    
    if results is None:
        print("❌ No results to analyze")
        return
    
    results_df = pd.DataFrame(results)
    
    # Check data completeness
    total_transactions = len(original_df)
    processed_transactions = len(results_df)
    
    print(f"Total input transactions: {total_transactions}")
    print(f"Successfully processed: {processed_transactions}")
    print(f"Success rate: {(processed_transactions/total_transactions)*100:.1f}%")
    
    # Analyze categorization results
    if not results_df.empty:
        print("\n🎯 CATEGORIZATION ANALYSIS:")
        
        # Category distribution
        if 'category' in results_df.columns:
            categories = results_df['category'].value_counts()
            print(f"Categories found: {len(categories)}")
            for cat, count in categories.head(10).items():
                print(f"  {cat}: {count} transactions")
        
        # Vendor standardization
        if 'standardized_vendor' in results_df.columns:
            vendors = results_df['standardized_vendor'].value_counts()
            print(f"\nVendors standardized: {len(vendors)}")
            for vendor, count in vendors.head(10).items():
                print(f"  {vendor}: {count} transactions")
        
        # Check for missing/null values
        print("\n🔍 DATA QUALITY CHECK:")
        for col in ['category', 'standardized_vendor', 'amount', 'transaction_date']:
            if col in results_df.columns:
                missing = results_df[col].isnull().sum()
                print(f"  {col}: {missing} missing values ({(missing/len(results_df))*100:.1f}%)")
    
    return results_df

def test_specific_challenges(results_df):
    """Test specific challenging scenarios"""
    print("\n🎯 CHALLENGE-SPECIFIC TESTING:")
    
    if results_df is None or results_df.empty:
        print("❌ No results to test")
        return
    
    # Test 1: Payment processor handling
    processor_tests = {
        'Square': 'SQ *',
        'Stripe': 'STRIPE*',
        'PayPal': 'PP*',
        'Amazon': 'AMZN',
        'Google': 'GOOGL*'
    }
    
    print("\n🔧 PAYMENT PROCESSOR HANDLING:")
    for processor, pattern in processor_tests.items():
        if 'original_merchant' in results_df.columns:
            matches = results_df[results_df['original_merchant'].str.contains(pattern, na=False)]
            if not matches.empty:
                vendor = matches.iloc[0]['standardized_vendor'] if 'standardized_vendor' in matches.columns else 'N/A'
                category = matches.iloc[0]['category'] if 'category' in matches.columns else 'N/A'
                print(f"  {processor}: {vendor} → {category}")
    
    # Test 2: Date format handling
    print("\n📅 DATE FORMAT HANDLING:")
    if 'transaction_date' in results_df.columns:
        valid_dates = results_df['transaction_date'].notna().sum()
        total_dates = len(results_df)
        print(f"  Valid dates parsed: {valid_dates}/{total_dates} ({(valid_dates/total_dates)*100:.1f}%)")
    
    # Test 3: Amount handling
    print("\n💰 AMOUNT PROCESSING:")
    if 'amount' in results_df.columns:
        valid_amounts = results_df['amount'].notna().sum()
        total_amounts = len(results_df)
        print(f"  Valid amounts parsed: {valid_amounts}/{total_amounts} ({(valid_amounts/total_amounts)*100:.1f}%)")

def main():
    """Main testing function"""
    print("🚀 ENTERPRISE ACCURACY TESTING")
    print("=" * 50)
    
    # Load test data
    test_data = load_test_data()
    
    # Convert to DataFrame
    df = convert_to_dataframe(test_data)
    
    # Test the cleaning system
    results, processing_time = test_cleaning_accuracy(df)
    
    # Analyze results
    results_df = analyze_results(df, results)
    
    # Test specific challenges
    test_specific_challenges(results_df)
    
    # Generate summary
    print("\n" + "=" * 50)
    print("🎯 ENTERPRISE READINESS ASSESSMENT:")
    
    if results and len(results) > 0:
        success_rate = (len(results) / len(df)) * 100
        avg_processing_time = processing_time / len(df)
        
        print(f"✅ Processing Success Rate: {success_rate:.1f}%")
        print(f"⚡ Average Processing Speed: {avg_processing_time:.3f}s per transaction")
        
        # Enterprise readiness scoring
        if success_rate >= 95 and avg_processing_time < 1.0:
            print("🏆 ENTERPRISE READY: High accuracy and good performance")
        elif success_rate >= 90:
            print("⚠️  NEAR ENTERPRISE READY: Good accuracy, may need optimization")
        else:
            print("❌ NOT ENTERPRISE READY: Needs improvement before backend launch")
    else:
        print("❌ FAILED: System unable to process test data")
    
    print("\n💡 Ready for backend partnerships when success rate > 95%")

if __name__ == "__main__":
    main()