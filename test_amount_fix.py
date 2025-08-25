#!/usr/bin/env python3
"""
Test the amount parsing fix for None categories
"""

import requests
import json

def test_amount_fix():
    print('🧪 TESTING AMOUNT PARSING FIX...')
    print('=' * 50)
    
    # Test the problematic rows with bad amount formats
    test_data = [
        {'Merchant': 'Budget INC', 'Amount': '-$729.4', 'Date': '20230329', 'Notes': ''},
        {'Merchant': 'Apple', 'Amount': 0, 'Date': 'Jan 17, 2023', 'Notes': ''},
        {'Merchant': 'TST* Southwest Airlines', 'Amount': '$861.24', 'Date': '2023-02-16', 'Notes': ''},
        {'Merchant': 'AUTO PAY Hilton Hotels', 'Amount': '(324.83)', 'Date': '', 'Notes': ''},
        {'Merchant': 'Google', 'Amount': '-$947.0', 'Date': '', 'Notes': ''}
    ]

    print('TESTING PROBLEMATIC AMOUNT FORMATS:')
    for i, data in enumerate(test_data):
        merchant = data['Merchant']
        amount = data['Amount']
        print(f'{i+1}. {merchant:25s} | Amount: {amount}')

    payload = {'user_intent': 'comprehensive cleaning and standardization', 'data': test_data}
    response = requests.post('http://localhost:8080/process', json=payload, timeout=60)

    if response.status_code == 200:
        result = response.json()
        cleaned_data = result.get('cleaned_data', [])
        
        print('\nRESULTS:')
        none_count = 0
        for i, row in enumerate(cleaned_data):
            original = test_data[i]['Merchant']
            vendor = row.get('standardized_vendor', 'N/A')
            category = row.get('category', 'N/A') or 'None'
            parsed_amount = row.get('amount', 'N/A')
            
            if category == 'None' or not category:
                none_count += 1
                status = '❌'
            else:
                status = '✅'
                
            print(f'{status} {i+1}. {original:25s} → {vendor:15s} | {category:20s} | ${parsed_amount}')
        
        print(f'\nNone categories: {none_count}/{len(cleaned_data)}')
        
        if none_count == 0:
            print('🎉 AMOUNT PARSING FIX SUCCESSFUL!')
            print('All problematic rows now have proper categories!')
            return True
        else:
            print(f'❌ Still {none_count} None categories - need further debugging')
            return False

    else:
        print(f'Error: {response.status_code} - {response.text}')
        return False

if __name__ == "__main__":
    test_amount_fix()