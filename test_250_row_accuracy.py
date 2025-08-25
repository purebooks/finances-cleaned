#!/usr/bin/env python3
"""
Run a 250-row accuracy test with exact per-column checks and a concise mismatch review.
Uses the same dataset/expected files as the 500-row test, but only the first 250 rows.
"""

import os
import json
import time
from typing import Dict, List, Tuple

import pandas as pd
import requests


def load_test_data_250() -> Tuple[List[Dict], List[Dict]]:
    df = pd.read_csv("test_500_rows.csv").head(250)
    df = df.fillna("")
    df['Amount'] = pd.to_numeric(df['Amount'], errors='coerce').fillna(0)
    test_data = df.to_dict('records')

    with open("test_500_rows_expected.json", "r") as f:
        expected_all = json.load(f)
    expected = expected_all[:250]
    return test_data, expected


def send_to_api(data: List[Dict]) -> Dict:
    payload = {
        "user_intent": "comprehensive cleaning and standardization",
        "data": data,
        "config": {
            "preserve_schema": False,
            "enable_ai": True,
            "ai_vendor_enabled": True,
            "ai_category_enabled": True,
            "ai_confidence_threshold": 0.7,
            "ai_mode": "apply",
            "ai_model": "gpt-4o-mini"
        }
    }

    base_url = os.getenv("API_BASE_URL") or f"http://localhost:{os.getenv('PORT', '8080')}"
    start = time.time()
    resp = requests.post(f"{base_url}/process", json=payload, timeout=180)
    elapsed = time.time() - start
    if resp.status_code != 200:
        return {"success": False, "error": f"HTTP {resp.status_code}", "processing_time": elapsed}
    return {"success": True, "data": resp.json(), "processing_time": elapsed}


def exact_accuracy(cleaned: List[Dict], expected: List[Dict]) -> Dict:
    total = min(len(cleaned), len(expected))
    vendor_ok = 0
    category_ok = 0
    date_ok = 0
    amount_ok = 0
    mismatches: List[Dict] = []

    for i in range(total):
        row = cleaned[i]
        exp = expected[i]

        cleaned_vendor = str(
            row.get("standardized_vendor",
                row.get("Clean Vendor",
                    row.get("Merchant", row.get("merchant", ""))
                )
            )
        ).strip()
        expected_vendor = str(exp.get("expected_clean_merchant", "").strip())
        vmatch = cleaned_vendor == expected_vendor
        if vmatch:
            vendor_ok += 1

        cleaned_category = str(row.get("category", row.get("Category", "")).strip())
        expected_category = str(exp.get("expected_category", "").strip())
        cmatch = cleaned_category == expected_category
        if cmatch:
            category_ok += 1

        cleaned_date = str(row.get("date", row.get("Date", "")).strip())
        expected_date = str(exp.get("expected_date", "").strip())
        dmatch = (not expected_date) or (cleaned_date == expected_date)
        if dmatch:
            date_ok += 1

        cleaned_amount = row.get("amount", row.get("Amount", 0))
        try:
            amatch = abs(float(cleaned_amount) - float(exp.get("expected_amount", cleaned_amount))) < 1e-6
        except Exception:
            amatch = False
        if amatch:
            amount_ok += 1

        if not (vmatch and cmatch and dmatch and amatch):
            mismatches.append({
                "row": i,
                "vendor": {"expected": expected_vendor, "actual": cleaned_vendor},
                "category": {"expected": expected_category, "actual": cleaned_category},
                "date": {"expected": expected_date, "actual": cleaned_date},
                "amount": {"expected": exp.get("expected_amount"), "actual": cleaned_amount}
            })

    def pct(n: int) -> float:
        return (n / total) * 100 if total else 0.0

    return {
        "total": total,
        "vendor_accuracy": pct(vendor_ok),
        "category_accuracy": pct(category_ok),
        "date_accuracy": pct(date_ok),
        "amount_accuracy": pct(amount_ok),
        "overall_accuracy": (pct(vendor_ok) * 0.4 + pct(category_ok) * 0.4 + pct(date_ok) * 0.1 + pct(amount_ok) * 0.1),
        "mismatches": mismatches,
    }


def run():
    print("🧪 Starting 250-Row Accuracy Test")
    print("=" * 50)

    test_data, expected = load_test_data_250()
    print(f"📊 Loaded {len(test_data)} rows")

    api_result = send_to_api(test_data)
    if not api_result.get("success"):
        print(f"❌ API error: {api_result.get('error')}")
        return False

    resp = api_result["data"]
    cleaned = resp.get("cleaned_data", [])
    print(f"✅ API responded in {api_result['processing_time']:.2f}s | Returned {len(cleaned)} rows")

    print("\n🔍 Calculating exact accuracy...")
    metrics = exact_accuracy(cleaned, expected)

    print("\n📈 ACCURACY RESULTS")
    print("=" * 50)
    print(f"Vendor: {metrics['vendor_accuracy']:.1f}%  Category: {metrics['category_accuracy']:.1f}%  Date: {metrics['date_accuracy']:.1f}%  Amount: {metrics['amount_accuracy']:.1f}%")
    print(f"OVERALL: {metrics['overall_accuracy']:.1f}%  in {api_result['processing_time']:.1f}s")

    # Short review of mismatches
    if metrics["mismatches"]:
        print("\n🔎 Review (first 10 mismatches):")
        for m in metrics["mismatches"][:10]:
            i = m["row"] + 1
            print(f"{i:3d}: vendor exp='{m['vendor']['expected']}' got='{m['vendor']['actual']}', "
                  f"category exp='{m['category']['expected']}' got='{m['category']['actual']}', "
                  f"date exp='{m['date']['expected']}' got='{m['date']['actual']}', "
                  f"amount exp='{m['amount']['expected']}' got='{m['amount']['actual']}'")

    # Save detail
    with open("test_250_results.json", "w") as f:
        json.dump({
            "summary": {
                "total": metrics['total'],
                "vendor_accuracy": metrics['vendor_accuracy'],
                "category_accuracy": metrics['category_accuracy'],
                "date_accuracy": metrics['date_accuracy'],
                "amount_accuracy": metrics['amount_accuracy'],
                "overall_accuracy": metrics['overall_accuracy'],
                "processing_time": api_result['processing_time'],
            },
            "mismatches": metrics['mismatches']
        }, f, indent=2)
    print("\n💾 Saved details to test_250_results.json")
    return True


if __name__ == "__main__":
    ok = run()
    raise SystemExit(0 if ok else 1)


