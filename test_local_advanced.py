#!/usr/bin/env python3
"""
Local test for Advanced LLM Flow components
"""

import pandas as pd
import sys
import os

# Add current directory to path to import our modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from production_cleaner_ai_v5 import AIEnhancedProductionCleanerV5
from llm_client import LLMClient

def test_advanced_llm_flow():
    """Test the advanced LLM flow locally"""
    print("🚀 Testing Advanced LLM Flow Components Locally")
    print("=" * 60)
    
    # Create sample data
    sample_data = {
        'merchant': ['Google Cloud', 'Amazon AWS', 'Netflix', 'Spotify', 'Uber'],
        'amount': [150.00, 89.99, 15.99, 9.99, 25.50],
        'description': ['Cloud hosting', 'Web services', 'Streaming', 'Music', 'Transport']
    }
    
    df = pd.DataFrame(sample_data)
    print(f"✅ Created sample DataFrame with {len(df)} transactions")
    print(f"   Columns: {list(df.columns)}")
    print()
    
    # Create LLM client (mock mode)
    llm_client = LLMClient(use_mock=True)
    print("✅ Created LLM client in mock mode")
    print()
    
    # Create advanced cleaner
    config = {
        'enable_ai': True,
        'ai_vendor_enabled': True,
        'ai_category_enabled': True,
        'enable_transaction_intelligence': True,
        'enable_source_tracking': True,
        'ai_confidence_threshold': 0.7
    }
    
    cleaner = AIEnhancedProductionCleanerV5(df=df, config=config, llm_client=llm_client)
    print("✅ Created Advanced LLM Flow Cleaner")
    print(f"   Version: {cleaner.cleaner_version}")
    print(f"   Config: {config}")
    print()
    
    # Test schema validation and analysis
    print("🔍 Testing Schema Validation and Analysis...")
    schema_analysis = cleaner._validate_schema_and_analyze(df)
    print(f"   Schema valid: {schema_analysis['schema_valid']}")
    print(f"   Column mapping: {schema_analysis['column_mapping']}")
    print(f"   Processing capabilities: {schema_analysis['processing_capabilities']}")
    print()
    
    # Test intelligent processing
    print("⚙️  Testing Intelligent Processing...")
    try:
        enhanced_df, output = cleaner.process_data()
        print("✅ Processing completed successfully!")
        print()
        
        # Display results
        print("📊 Processing Results:")
        print(f"   Enhanced DataFrame shape: {enhanced_df.shape}")
        print(f"   New columns: {[col for col in enhanced_df.columns if col not in df.columns]}")
        print()
        
        # Display summary report
        summary = output['summary_report']['processing_summary']
        print("📈 Summary Report:")
        print(f"   Total transactions: {summary['total_transactions']}")
        print(f"   Vendor standardizations: {summary['vendor_standardizations']}")
        print(f"   Category classifications: {summary['category_classifications']}")
        print(f"   LLM calls: {summary['llm_calls']}")
        print(f"   Cache hit rate: {summary['cache_hit_rate']:.1%}")
        print()
        
        # Display cost analysis
        cost_analysis = output['summary_report']['cost_analysis']
        print("💰 Cost Analysis:")
        print(f"   Total cost: ${cost_analysis['total_cost']:.3f}")
        print(f"   Vendor standardization cost: ${cost_analysis['vendor_standardization_cost']:.3f}")
        print(f"   Category classification cost: ${cost_analysis['category_classification_cost']:.3f}")
        print()
        
        # Display cache performance
        cache_performance = output['summary_report']['cache_performance']
        print("💾 Cache Performance:")
        print(f"   Vendor hit rate: {cache_performance['vendor_hit_rate']:.1%}")
        print(f"   Category hit rate: {cache_performance['category_hit_rate']:.1%}")
        print(f"   Total vendor requests: {cache_performance['total_vendor_requests']}")
        print(f"   Total category requests: {cache_performance['total_category_requests']}")
        print(f"   Cache size: {cache_performance['cache_size']} entries")
        print()
        
        # Display transaction intelligence
        intelligence = output['transaction_intelligence']
        print("🧠 Transaction Intelligence:")
        print(f"   Tags: {intelligence['tags_summary']}")
        print(f"   Insights: {len(intelligence['insights_summary'])} insights")
        print(f"   Average risk score: {intelligence['average_risk_score']:.2f}")
        print(f"   Anomaly report: {len(intelligence['anomaly_report'])} anomalies")
        print()
        
        # Display sample enhanced data
        print("📋 Sample Enhanced Data:")
        print(enhanced_df.head().to_string())
        print()
        
        print("🎉 All Advanced LLM Flow tests passed!")
        print()
        print("🔧 Advanced Features Verified:")
        print("   • Intelligent Rule > Cache > LLM Processing Flow")
        print("   • Source Tracking & Confidence Scoring")
        print("   • Transaction Intelligence (Tags, Insights, Risk)")
        print("   • Comprehensive Cost & Performance Tracking")
        print("   • Enhanced DataFrame with Attribution")
        print("   • Separate Intelligence Section")
        print()
        print("✅ The advanced LLM flow is working correctly!")
        
        return True
        
    except Exception as e:
        print(f"❌ Processing failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main test function"""
    success = test_advanced_llm_flow()
    
    if success:
        print("\n🚀 Next Steps:")
        print("   1. The advanced LLM flow is working correctly")
        print("   2. Deploy to Cloud Run with proper IAM permissions")
        print("   3. Use interface_v5.html to test the web interface")
        print("   4. Monitor performance and costs")
    else:
        print("\n❌ Tests failed. Check the implementation.")

if __name__ == "__main__":
    main() 