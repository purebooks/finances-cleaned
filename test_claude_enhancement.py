#!/usr/bin/env python3
"""
Test Claude Enhancement Phase 1 - Demonstrate the enhanced classification with explanations
"""

import requests
import json

def test_claude_enhancement():
    print('🧠 CLAUDE ENHANCEMENT DEMONSTRATION')
    print('=' * 60)
    print('Testing strategic Claude integration for edge cases...')
    print()
    
    # Test cases designed to trigger Claude enhancement
    test_data = [
        # High-value transaction (should trigger explanation)
        {'Merchant': 'ACME Consulting LLC', 'Amount': 2500, 'Date': '2024-01-01', 'Notes': ''},
        
        # Unknown vendor (should trigger enhancement)
        {'Merchant': 'Random Business 123', 'Amount': 150, 'Date': '2024-01-02', 'Notes': ''},
        
        # Grocery stores (were "Other" before, should be enhanced)
        {'Merchant': 'TST* Kroger', 'Amount': 75, 'Date': '2024-01-03', 'Notes': ''},
        {'Merchant': 'Safeway LLC', 'Amount': 120, 'Date': '2024-01-04', 'Notes': ''},
        
        # Well-known vendor (should use rules, no enhancement needed)
        {'Merchant': 'Google', 'Amount': 25, 'Date': '2024-01-05', 'Notes': ''}
    ]
    
    payload = {'user_intent': 'comprehensive cleaning and standardization', 'data': test_data}
    
    print('🚀 Processing test cases...')
    response = requests.post('http://localhost:8080/process', json=payload, timeout=60)
    
    if response.status_code == 200:
        result = response.json()
        cleaned_data = result.get('cleaned_data', [])
        insights = result.get('insights', {})
        
        print(f'✅ Processing completed in {insights.get("processing_time", 0):.2f}s')
        print(f'💰 AI cost: ${insights.get("ai_cost", 0):.4f}')
        print()
        
        print('📊 CLAUDE ENHANCEMENT RESULTS:')
        print('=' * 60)
        
        claude_enhanced = 0
        for i, row in enumerate(cleaned_data):
            original = test_data[i]['Merchant']
            vendor = row.get('standardized_vendor', 'N/A')
            category = row.get('category', 'N/A')
            source = row.get('category_source', 'N/A')
            explanation = row.get('category_explanation', '')
            confidence = row.get('category_confidence', 'N/A')
            
            is_claude = source == 'claude_enhanced'
            if is_claude:
                claude_enhanced += 1
                status = '🧠 CLAUDE'
            else:
                status = '⚡ RULES'
            
            print(f'{i+1}. {status} | {original}')
            print(f'   → {vendor} | {category}')
            print(f'   Confidence: {confidence} | Source: {source}')
            if explanation:
                print(f'   💡 Explanation: {explanation}')
            print()
        
        print('🎯 ENHANCEMENT SUMMARY:')
        print(f'   Claude enhanced: {claude_enhanced}/{len(cleaned_data)} rows')
        print(f'   Rule-based: {len(cleaned_data) - claude_enhanced}/{len(cleaned_data)} rows')
        
        if claude_enhanced > 0:
            print(f'✅ Claude enhancement successfully providing:')
            print(f'   • Better classifications for edge cases')
            print(f'   • Detailed explanations for accounting')
            print(f'   • Context-aware business categorization')
        else:
            print(f'ℹ️  All cases handled by high-confidence rules')
            
        return claude_enhanced > 0
        
    else:
        print(f'❌ Error: {response.status_code}')
        return False

if __name__ == "__main__":
    test_claude_enhancement()