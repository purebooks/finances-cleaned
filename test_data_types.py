#!/usr/bin/env python3

import pandas as pd

# Test data that matches our CSV
test_data = [
    {"Date": "2023-10-01", "Merchant": "SQ *SQ *PATRIOT CAFE", "Amount": "15.25", "Notes": "Morning coffee"},
    {"Date": "2023/10/01", "Merchant": "uber trip", "Amount": "-22.50", "Notes": "Ride to meeting"},
    {"Date": "10-02-2023", "Merchant": "AMZ*Amazon.com", "Amount": "", "Notes": "Office supplies, pens"},
    {"Date": "", "Merchant": "GOOGLE *GSUITE", "Amount": "12.00", "Notes": "Monthly subscription"},
    {"Date": "2023-10-03", "Merchant": "", "Amount": "9.99", "Notes": "Music streaming"},
    {"Date": "2023-10-04", "Merchant": "apple.com/bill", "Amount": "9.99", "Notes": "iCloud Storage"},
    {"Date": "2023/10/01", "Merchant": "uber trip", "Amount": "-22.50", "Notes": "Duplicate ride entry for testing"},
    {"Date": "Oct 5 2023", "Merchant": "Random Local Grocer", "Amount": "55.43", "Notes": "Team lunch catering"}
]

print("🧪 Testing DataFrame Creation")
print(f"📊 Input data: {len(test_data)} records")

# Create DataFrame
df = pd.DataFrame(test_data)

print(f"📋 DataFrame shape: {df.shape}")
print(f"📋 DataFrame columns: {list(df.columns)}")
print(f"📋 DataFrame dtypes:")
for col in df.columns:
    print(f"   {col}: {df[col].dtype}")

print(f"\n📋 Sample data:")
print(df.head(3))

print(f"\n🔍 Testing column operations:")
for col in df.columns:
    try:
        # Test if we can call .lower() on the column name
        col_lower = col.lower()
        print(f"   ✅ {col}: can call .lower() -> '{col_lower}'")
    except Exception as e:
        print(f"   ❌ {col}: error calling .lower() -> {e}")

print("\n�� Test complete!") 