#!/usr/bin/env python3
"""
Test script for parallel processing implementation
Tests both sequential and parallel modes with timing comparison
"""

import pandas as pd
import time
import json
from production_cleaner_ai_v5 import AIEnhancedProductionCleanerV5

def create_test_data(num_rows=50):
    """Create test financial data"""
    test_data = []
    vendors = [
        "Google LLC", "Meta Platforms", "Amazon.com", "Stripe Inc", 
        "PayPal Holdings", "Netflix Inc", "Spotify Technology",
        "Uber Technologies", "Lyft Inc", "DigitalOcean LLC",
        "Unknown Vendor 1", "Unknown Vendor 2", "Mysterious Corp",
        "Random Business", "Test Company", "Sample LLC"
    ]
    
    for i in range(num_rows):
        test_data.append({
            "merchant": vendors[i % len(vendors)],
            "amount": round(10 + (i * 2.5), 2),
            "description": f"Transaction {i+1} description",
            "date": f"2024-01-{(i % 30) + 1:02d}"
        })
    
    return test_data

def test_processing_modes():
    """Test both sequential and parallel processing"""
    print("🧪 Testing Parallel Processing Implementation")
    print("=" * 60)
    
    # Create test data
    test_rows = 30  # Start with moderate size for testing
    test_data = create_test_data(test_rows)
    df = pd.DataFrame(test_data)
    
    print(f"📊 Test data: {len(df)} rows")
    print(f"Sample vendors: {df['merchant'].unique()[:5].tolist()}")
    print()
    
    # Test configurations
    base_config = {
        'enable_ai': True,
        'ai_vendor_enabled': True,
        'ai_category_enabled': True,
        'enable_transaction_intelligence': False,  # Skip for speed
        'enable_source_tracking': True,
        'ai_confidence_threshold': 0.7,
        'force_llm_for_testing': True  # Force some LLM calls
    }
    
    # Test 1: Sequential Processing
    print("🔄 Test 1: Sequential Processing")
    print("-" * 40)
    
    sequential_config = {**base_config, 'enable_parallel_processing': False}
    processor_seq = AIEnhancedProductionCleanerV5()
    
    start_time = time.time()
    result_seq = processor_seq.process_data(test_data, sequential_config)
    seq_time = time.time() - start_time
    
    print(f"✅ Sequential completed in {seq_time:.2f} seconds")
    print(f"   - LLM Calls: {result_seq['summary_report']['llm_tracker']['total_calls']}")
    print(f"   - Total Cost: ${result_seq['summary_report']['llm_tracker']['total_cost']:.3f}")
    print(f"   - Rows Processed: {len(result_seq['cleaned_data'])}")
    print()
    
    # Test 2: Parallel Processing
    print("⚡ Test 2: Parallel Processing (4 workers)")
    print("-" * 40)
    
    parallel_config = {
        **base_config, 
        'enable_parallel_processing': True,
        'max_workers': 4
    }
    processor_par = AIEnhancedProductionCleanerV5()
    
    start_time = time.time()
    result_par = processor_par.process_data(test_data, parallel_config)
    par_time = time.time() - start_time
    
    print(f"✅ Parallel completed in {par_time:.2f} seconds")
    print(f"   - LLM Calls: {result_par['summary_report']['llm_tracker']['total_calls']}")
    print(f"   - Total Cost: ${result_par['summary_report']['llm_tracker']['total_cost']:.3f}")
    print(f"   - Rows Processed: {len(result_par['cleaned_data'])}")
    print()
    
    # Performance Analysis
    print("📊 Performance Analysis")
    print("-" * 40)
    speedup = seq_time / par_time if par_time > 0 else 1
    time_saved = seq_time - par_time
    efficiency = (speedup / 4) * 100  # 4 workers
    
    print(f"⏱️  Sequential Time: {seq_time:.2f}s")
    print(f"⚡ Parallel Time: {par_time:.2f}s")
    print(f"🚀 Speedup: {speedup:.2f}x")
    print(f"💾 Time Saved: {time_saved:.2f}s ({(time_saved/seq_time)*100:.1f}%)")
    print(f"📈 Efficiency: {efficiency:.1f}% (of theoretical max)")
    print()
    
    # Data Quality Check
    print("🔍 Data Quality Verification")
    print("-" * 40)
    
    # Check if both methods processed the same number of rows
    seq_rows = len(result_seq['cleaned_data'])
    par_rows = len(result_par['cleaned_data'])
    
    print(f"📊 Rows - Sequential: {seq_rows}, Parallel: {par_rows}")
    
    if seq_rows == par_rows:
        print("✅ Row count matches")
    else:
        print("❌ Row count mismatch!")
    
    # Check standardized vendors
    seq_vendors = set(result_seq['cleaned_data'].get('standardized_vendor', []))
    par_vendors = set(result_par['cleaned_data'].get('standardized_vendor', []))
    
    common_vendors = seq_vendors.intersection(par_vendors)
    print(f"🏢 Common standardized vendors: {len(common_vendors)}/{len(seq_vendors)}")
    
    # Projection for larger datasets
    print()
    print("📈 Projections for Larger Datasets")
    print("-" * 40)
    
    for size in [100, 500, 1000]:
        proj_seq_time = (seq_time / test_rows) * size
        proj_par_time = (par_time / test_rows) * size
        proj_speedup = proj_seq_time / proj_par_time
        
        print(f"📊 {size} rows:")
        print(f"   Sequential: ~{proj_seq_time/60:.1f} minutes")
        print(f"   Parallel: ~{proj_par_time/60:.1f} minutes")
        print(f"   Speedup: {proj_speedup:.1f}x")
        print()

if __name__ == "__main__":
    try:
        test_processing_modes()
        print("🎉 Parallel processing test completed successfully!")
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc() 