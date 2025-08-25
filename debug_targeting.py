#!/usr/bin/env python3
"""
Debug the Conservative targeting logic
"""
import pandas as pd
import sys
sys.path.append('.')

# Test the targeting function directly
test_data = {
    'category': ['Meals & Entertainment', 'Meals & Entertainment', 'Office Supplies & Equipment', 
                'Travel & Transportation', 'Professional Services', 'Travel & Transportation', 'Other'],
    'category_confidence': [0.85, 0.82, 0.90, 0.88, 0.92, 0.95, 0.60],
    'amount': [4.85, 12.50, 25.99, 45.50, 1250.00, 2500.00, 750.00],
    'standardized_vendor': ['Starbucks', 'TECH CAFE DOWNTOWN', 'Amazon', 'Uber', 
                           'MYSTERIOUS CONSULTING', 'Enterprise', 'UNKNOWN VENDOR']
}

df = pd.DataFrame(test_data)

print("🧪 DEBUGGING CONSERVATIVE TARGETING")
print("=" * 50)

print("📊 Test Data:")
for idx, row in df.iterrows():
    print(f"  {idx}: ${row['amount']:>7.2f} - {row['standardized_vendor']} - {row['category']} (conf: {row['category_confidence']})")

print("\n🎯 Applying Conservative Logic:")

# Simulate the targeting logic
targets = []
for idx, row in df.iterrows():
    category = row.get('category', '')
    confidence = row.get('category_confidence', 1.0)
    amount = abs(float(row.get('amount', 0))) if pd.notna(row.get('amount')) else 0
    vendor = row.get('standardized_vendor', '')
    
    selected = False
    reason = ""
    
    # CONSERVATIVE TARGET: Only high-value transactions (>$1000) get Claude enhancement
    if amount > 1000:
        targets.append(idx)
        selected = True
        reason = f"High-value (${amount})"
    # OPTIONAL: Critical "Other" categories only if truly unknown AND significant amount
    elif category == 'Other' and amount > 500:
        targets.append(idx)
        selected = True
        reason = f"Unknown high-value (${amount})"
    else:
        reason = "Not selected (Conservative)"
    
    status = "🧠 SELECTED" if selected else "⚡ SKIPPED"
    print(f"  {idx}: {status} - {vendor} - {reason}")

print(f"\n💡 CONSERVATIVE RESULTS:")
print(f"  Targets: {len(targets)} out of {len(df)} transactions")
print(f"  Cost reduction: {((len(df) - len(targets)) / len(df)) * 100:.1f}%")
print(f"  Selected indices: {targets}")

print(f"\n🔍 EXPECTED BEHAVIOR:")
print(f"  • Only transactions >$1000 should be selected")
print(f"  • Plus 'Other' category if >$500")
print(f"  • Should select indices [4, 5, 6] (3 transactions)")

if len(targets) == 3 and set(targets) == {4, 5, 6}:
    print(f"  ✅ Logic working correctly!")
else:
    print(f"  ❌ Logic issue detected!")