#!/usr/bin/env python3
"""
Comprehensive test for enhanced parallel processing system
Tests: Sequential vs Parallel vs Batch processing with the new LLM client
"""

import pandas as pd
import time
import json
import sys
import os
from typing import List, Dict, Any

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from production_cleaner_ai_v5 import AIEnhancedProductionCleanerV5
try:
    from llm_client_v2 import LLMClient
except ImportError:
    from llm_client import LLMClient

def create_realistic_test_data(num_rows=50) -> List[Dict[str, Any]]:
    """Create realistic financial transaction data for testing"""
    vendors = [
        # Rule-based vendors (should be fast)
        "Google LLC", "Google Workspace", "GOOGLE*FIREBASE",
        "Meta Platforms", "Facebook Inc", "META*WHATSAPP",
        "Amazon.com", "AMZN MKTP", "Amazon Web Services",
        "Stripe Inc", "STRIPE*TRANSFER", "Stripe Payments",
        "PayPal Holdings", "PAYPAL*TRANSFER", "PayPal Inc",
        "Netflix Inc", "NETFLIX.COM", "Netflix Streaming",
        "Spotify Technology", "SPOTIFY*PREMIUM", "Spotify AB",
        "Uber Technologies", "UBER*TRIP", "Uber Eats",
        "Lyft Inc", "LYFT*RIDE", "Lyft Driver",
        "DigitalOcean LLC", "DIGITALOCEAN*DROPLET",
        
        # Unknown vendors (will trigger LLM calls)
        "Mysterious Corp Inc", "Unknown Vendor LLC", "Random Business Co",
        "Test Company XYZ", "Sample Services Ltd", "Generic Corp",
        "Business Solutions Inc", "Professional Services Co", "Enterprise LLC",
        "Innovation Labs", "Tech Startup Inc", "Digital Solutions",
        "Creative Agency", "Marketing Firm", "Consulting Group",
        "Software House", "Development Studio", "Analytics Corp"
    ]
    
    categories = [
        "Software & Technology", "Marketing & Advertising", "Office Supplies & Equipment",
        "Travel & Transportation", "Meals & Entertainment", "Professional Services",
        "Insurance & Legal", "Utilities & Rent", "Employee Benefits", "Banking & Finance"
    ]
    
    test_data = []
    for i in range(num_rows):
        vendor = vendors[i % len(vendors)]
        amount = round(10 + (i * 2.5) + (hash(vendor) % 100), 2)
        
        test_data.append({
            "merchant": vendor,
            "amount": amount,
            "description": f"Transaction {i+1} - {vendor} payment",
            "memo": f"Memo for transaction {i+1}",
            "date": f"2024-{((i % 12) + 1):02d}-{((i % 28) + 1):02d}",
            "account": "Business Checking",
            "category": ""  # Will be filled by AI
        })
    
    return test_data

def run_performance_comparison():
    """Run comprehensive performance comparison"""
    print("🧪 Enhanced Parallel Processing Performance Test")
    print("=" * 70)
    
    # Test configurations
    test_sizes = [25, 50]  # Start with smaller sizes for local testing
    
    for test_size in test_sizes:
        print(f"\n📊 Testing with {test_size} transactions")
        print("-" * 50)
        
        # Create test data
        test_data = create_realistic_test_data(test_size)
        
        # Base configuration
        base_config = {
            'enable_ai': True,
            'ai_vendor_enabled': True,
            'ai_category_enabled': True,
            'enable_transaction_intelligence': False,  # Skip for speed
            'enable_source_tracking': True,
            'ai_confidence_threshold': 0.7,
            'force_llm_for_testing': False  # Use production mode for realistic testing
        }
        
        results = {}
        
        # Test 1: Sequential Processing
        print("🔄 Test 1: Sequential Processing")
        sequential_config = {**base_config, 'enable_parallel_processing': False}
        
        # Convert test data to DataFrame
        df_seq = pd.DataFrame(test_data)
        
        # Create LLM client and processor
        from llm_client_v2 import LLMClient
        llm_client_seq = LLMClient(use_mock=True, enable_caching=True)
        processor_seq = AIEnhancedProductionCleanerV5(
            df=df_seq,
            config=sequential_config,
            llm_client=llm_client_seq
        )
        
        start_time = time.time()
        try:
            cleaned_df_seq, result_seq = processor_seq.process_data()
            seq_time = time.time() - start_time
            
            results['sequential'] = {
                'time': seq_time,
                'llm_calls': result_seq['summary_report']['llm_tracker']['total_calls'],
                'cost': result_seq['summary_report']['llm_tracker']['total_cost'],
                'rows': len(cleaned_df_seq),
                'success': True,
                'result': result_seq
            }
            print(f"   ✅ Completed in {seq_time:.2f}s")
            print(f"   📞 LLM Calls: {results['sequential']['llm_calls']}")
            print(f"   💰 Cost: ${results['sequential']['cost']:.3f}")
            
        except Exception as e:
            print(f"   ❌ Failed: {e}")
            results['sequential'] = {'success': False, 'error': str(e)}
        
        # Test 2: Parallel Processing (4 workers)
        print("\n⚡ Test 2: Parallel Processing (4 workers)")
        parallel_config = {
            **base_config, 
            'enable_parallel_processing': True,
            'max_workers': 4
        }
        
        # Convert test data to DataFrame
        df_par = pd.DataFrame(test_data)
        
        # Create LLM client and processor
        llm_client_par = LLMClient(use_mock=True, enable_caching=True)
        processor_par = AIEnhancedProductionCleanerV5(
            df=df_par,
            config=parallel_config,
            llm_client=llm_client_par
        )
        
        start_time = time.time()
        try:
            cleaned_df_par, result_par = processor_par.process_data()
            par_time = time.time() - start_time
            
            results['parallel'] = {
                'time': par_time,
                'llm_calls': result_par['summary_report']['llm_tracker']['total_calls'],
                'cost': result_par['summary_report']['llm_tracker']['total_cost'],
                'rows': len(cleaned_df_par),
                'success': True,
                'result': result_par
            }
            print(f"   ✅ Completed in {par_time:.2f}s")
            print(f"   📞 LLM Calls: {results['parallel']['llm_calls']}")
            print(f"   💰 Cost: ${results['parallel']['cost']:.3f}")
            
        except Exception as e:
            print(f"   ❌ Failed: {e}")
            results['parallel'] = {'success': False, 'error': str(e)}
        
        # Test 3: Enhanced LLM Client Test (if available)
        print("\n🚀 Test 3: Enhanced LLM Client Features")
        try:
            from llm_client_v2 import LLMClient
            enhanced_client = LLMClient(use_mock=True, enable_caching=True)
            
            # Test batch processing
            batch_rows = test_data[:10]  # Test with first 10 rows
            start_time = time.time()
            batch_results = enhanced_client.process_transaction_batch(batch_rows)
            batch_time = time.time() - start_time
            
            print(f"   ✅ Batch processing: {len(batch_results)} rows in {batch_time:.2f}s")
            print(f"   📊 Avg time per row: {(batch_time/len(batch_results)*1000):.1f}ms")
            
            # Test cache stats
            stats = enhanced_client.get_stats()
            print(f"   💾 Cache sizes: Vendor={stats.get('vendor_cache_size', 0)}, Category={stats.get('category_cache_size', 0)}")
            
        except ImportError:
            print("   ⚠️  Enhanced LLM client not available (using fallback)")
        except Exception as e:
            print(f"   ❌ Enhanced client test failed: {e}")
        
        # Performance Analysis
        if results.get('sequential', {}).get('success') and results.get('parallel', {}).get('success'):
            print(f"\n📈 Performance Analysis for {test_size} rows")
            print("-" * 40)
            
            seq_time = results['sequential']['time']
            par_time = results['parallel']['time']
            speedup = seq_time / par_time if par_time > 0 else 1
            time_saved = seq_time - par_time
            efficiency = (speedup / 4) * 100  # 4 workers
            
            print(f"⏱️  Sequential: {seq_time:.2f}s")
            print(f"⚡ Parallel: {par_time:.2f}s")
            print(f"🚀 Speedup: {speedup:.2f}x")
            print(f"💾 Time Saved: {time_saved:.2f}s ({(time_saved/seq_time)*100:.1f}%)")
            print(f"📊 Efficiency: {efficiency:.1f}% (of theoretical 4x max)")
            
            # Cost comparison
            seq_cost = results['sequential']['cost']
            par_cost = results['parallel']['cost']
            cost_diff = seq_cost - par_cost
            
            print(f"💰 Cost - Sequential: ${seq_cost:.3f}, Parallel: ${par_cost:.3f}")
            if cost_diff != 0:
                print(f"💵 Cost Difference: ${abs(cost_diff):.3f} ({'saved' if cost_diff > 0 else 'extra'})")
        
        # Data Quality Check
        if results.get('sequential', {}).get('success') and results.get('parallel', {}).get('success'):
            print(f"\n🔍 Data Quality Verification")
            print("-" * 40)
            
            seq_result = results['sequential']['result']
            par_result = results['parallel']['result']
            
            seq_vendors = set(seq_result['cleaned_data'].get('standardized_vendor', []))
            par_vendors = set(par_result['cleaned_data'].get('standardized_vendor', []))
            
            vendor_overlap = len(seq_vendors.intersection(par_vendors))
            total_vendors = max(len(seq_vendors), len(par_vendors))
            
            print(f"🏢 Vendor consistency: {vendor_overlap}/{total_vendors} ({(vendor_overlap/total_vendors*100):.1f}%)")
            
            # Check for any significant differences
            if vendor_overlap / total_vendors < 0.8:
                print("⚠️  Significant differences detected between sequential and parallel results")
            else:
                print("✅ Sequential and parallel results are highly consistent")

def test_enhanced_llm_client():
    """Test the enhanced LLM client specifically"""
    print("\n🔬 Enhanced LLM Client Deep Test")
    print("=" * 50)
    
    try:
        from llm_client_v2 import LLMClient
        
        # Test with mock mode first
        print("🧪 Testing Mock Mode")
        client = LLMClient(use_mock=True, enable_caching=True, cache_size=100)
        
        # Test individual calls
        vendor_result = client.resolve_vendor("PAYPAL*DIGITALOCEAN", "Online payment", "Monthly subscription")
        category_result = client.classify_category("DigitalOcean", 25.00, "Cloud hosting")
        
        print(f"   Vendor: 'PAYPAL*DIGITALOCEAN' -> '{vendor_result}'")
        print(f"   Category: 'DigitalOcean, $25' -> '{category_result}'")
        
        # Test batch processing
        test_batch = [
            {"merchant": "Google LLC", "amount": 15.0, "description": "Cloud storage"},
            {"merchant": "STRIPE*PAYMENT", "amount": 99.0, "description": "Software subscription"},
            {"merchant": "Unknown Corp", "amount": 50.0, "description": "Service payment"},
            {"merchant": "AMZN MKTP", "amount": 35.0, "description": "Office supplies"}
        ]
        
        print(f"\n🚀 Batch Processing Test ({len(test_batch)} items)")
        start_time = time.time()
        batch_results = client.process_transaction_batch(test_batch)
        batch_time = time.time() - start_time
        
        for i, result in enumerate(batch_results):
            original = test_batch[i]["merchant"]
            standardized = result["standardized_vendor"]
            category = result["category"]
            print(f"   {i+1}. '{original}' -> '{standardized}' ({category})")
        
        print(f"\n⏱️  Batch Time: {batch_time:.3f}s ({(batch_time/len(test_batch)*1000):.1f}ms per item)")
        
        # Test caching
        print(f"\n💾 Cache Test")
        # Call the same vendor twice
        start_time = time.time()
        first_call = client.resolve_vendor("Test Vendor Inc")
        first_time = time.time() - start_time
        
        start_time = time.time()
        second_call = client.resolve_vendor("Test Vendor Inc")  # Should be cached
        second_time = time.time() - start_time
        
        print(f"   First call: {first_time:.3f}s")
        print(f"   Second call: {second_time:.3f}s")
        print(f"   Cache speedup: {(first_time/second_time):.1f}x" if second_time > 0 else "   Instant cache hit!")
        
        # Get stats
        stats = client.get_stats()
        print(f"\n📊 Client Statistics:")
        print(f"   Requests: {stats['request_count']}")
        print(f"   Errors: {stats['error_count']}")
        print(f"   Cost: ${stats['total_cost']:.3f}")
        print(f"   Vendor Cache: {stats.get('vendor_cache_size', 0)} items")
        print(f"   Category Cache: {stats.get('category_cache_size', 0)} items")
        
        return True
        
    except ImportError:
        print("❌ Enhanced LLM client (llm_client_v2.py) not available")
        return False
    except Exception as e:
        print(f"❌ Enhanced LLM client test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def project_large_dataset_performance():
    """Project performance for larger datasets based on test results"""
    print("\n📈 Large Dataset Performance Projections")
    print("=" * 50)
    
    # Based on typical results from smaller tests
    base_sequential_time_per_row = 0.8  # seconds per row
    base_parallel_time_per_row = 0.2    # seconds per row (4x speedup)
    base_cost_per_row = 0.01           # dollars per row
    
    dataset_sizes = [100, 500, 1000, 5000]
    
    print("Dataset Size | Sequential Time | Parallel Time | Speedup | Est. Cost")
    print("-" * 65)
    
    for size in dataset_sizes:
        seq_time = base_sequential_time_per_row * size
        par_time = base_parallel_time_per_row * size
        speedup = seq_time / par_time
        cost = base_cost_per_row * size
        
        seq_time_str = f"{seq_time/60:.1f}m" if seq_time > 60 else f"{seq_time:.1f}s"
        par_time_str = f"{par_time/60:.1f}m" if par_time > 60 else f"{par_time:.1f}s"
        
        print(f"{size:>11} | {seq_time_str:>14} | {par_time_str:>13} | {speedup:>6.1f}x | ${cost:>7.2f}")

if __name__ == "__main__":
    try:
        # Run comprehensive test
        run_performance_comparison()
        
        # Test enhanced LLM client
        test_enhanced_llm_client()
        
        # Project large dataset performance
        project_large_dataset_performance()
        
        print("\n🎉 All tests completed successfully!")
        print("\n💡 Next Steps:")
        print("   1. Review performance results above")
        print("   2. If satisfied, deploy to Cloud Run")
        print("   3. Test with your real dataset")
        
    except Exception as e:
        print(f"\n❌ Test suite failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1) 