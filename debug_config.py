#!/usr/bin/env python3
"""
Debug the actual configuration being used for large datasets
"""

import pandas as pd
from production_cleaner_ai_v5 import AIEnhancedProductionCleanerV5
from llm_client_v2 import LLMClient

def debug_config():
    """Debug what configuration is actually being used"""
    
    # Create test data - use 25 rows to trigger parallel processing
    test_data = []
    for i in range(25):
        test_data.append({
            "Date": "2023-01-01",
            "Merchant": f"Starbucks {i}",
            "Amount": 5.75,
            "Notes": f"Coffee {i}"
        })
    
    df = pd.DataFrame(test_data)
    print(f"📊 Created DataFrame with {len(df)} rows")
    
    # Initialize the cleaner (same as API does)
    client = LLMClient(use_mock=True)
    # Force sequential processing to test if parallel processing is the issue
    config = {"enable_parallel_processing": False, "enable_transaction_intelligence": False}
    cleaner = AIEnhancedProductionCleanerV5(df, config=config, llm_client=client, user_intent="categorize transactions")
    
    print(f"\n⚙️ Configuration being used:")
    config = cleaner.config
    for key, value in config.items():
        if 'ai' in key.lower() or 'category' in key.lower() or 'enable' in key.lower():
            print(f"   {key}: {value}")
    
    print(f"\n📋 DataFrame info:")
    print(f"   Rows: {len(df)}")
    print(f"   Columns: {list(df.columns)}")
    
    # Check if it will use parallel processing
    use_parallel = len(df) > 20 and config.get('enable_parallel_processing', True)
    print(f"\n🔀 Processing mode:")
    print(f"   Will use parallel: {use_parallel}")
    print(f"   Parallel threshold: > 20 rows")
    print(f"   Enable parallel config: {config.get('enable_parallel_processing', 'NOT_SET')}")
    
    # Check column detection
    vendor_col = cleaner._find_vendor_column(df)
    amount_col = cleaner._find_amount_column(df)
    print(f"\n🔍 Column detection:")
    print(f"   Vendor column: {vendor_col}")
    print(f"   Amount column: {amount_col}")
    
    # Test the actual processing
    print(f"\n🧪 Running actual processing...")
    try:
        cleaned_df, report = cleaner.process_data()
        
        # Check if categories were assigned
        has_category = 'category' in cleaned_df.columns
        print(f"   Category column exists: {has_category}")
        
        if has_category:
            category_count = cleaned_df['category'].notna().sum()
            print(f"   Rows with categories: {category_count}/{len(cleaned_df)}")
            
            sample_categories = cleaned_df['category'].value_counts().head()
            print(f"   Sample categories: {dict(sample_categories)}")
        else:
            print(f"   Available columns: {list(cleaned_df.columns)}")
        
        # Check insights
        insights = report.get('insights', {})
        print(f"\n📈 Processing insights:")
        print(f"   AI requests: {insights.get('ai_requests', 0)}")
        print(f"   Processing time: {insights.get('processing_time', 0):.3f}s")
        
    except Exception as e:
        print(f"   ❌ Error during processing: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_config()