#!/usr/bin/env python3
"""
200-row accuracy test (mix of seen/unseen): uses first 200 rows of the 500-row dataset
and checks accuracy with the unified API response (cleaned_data).
"""

import os
import json
import time
from typing import Dict, List, Tuple

import pandas as pd
import requests


def load_data_200() -> Tuple[List[Dict], List[Dict]]:
    df = pd.read_csv("test_500_rows.csv").head(200)
    df = df.fillna("")
    df['Amount'] = pd.to_numeric(df['Amount'], errors='coerce').fillna(0)
    test_data = df.to_dict('records')
    with open("test_500_rows_expected.json", "r") as f:
        expected = json.load(f)[:200]
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
            "ai_confidence_threshold": 0.7
        }
    }
    base_url = os.getenv("API_BASE_URL") or f"http://localhost:{os.getenv('PORT', '8080')}"
    t0 = time.time()
    resp = requests.post(f"{base_url}/process", json=payload, timeout=180)
    dt = time.time() - t0
    if resp.status_code != 200:
        return {"success": False, "error": f"HTTP {resp.status_code}", "processing_time": dt}
    return {"success": True, "data": resp.json(), "processing_time": dt}


def _fuzzy_contains(a: str, b: str) -> bool:
    if not a or not b:
        return False
    a = a.lower().strip()
    b = b.lower().strip()
    return a in b or b in a or a == b


def _fuzzy_category(a: str, b: str) -> bool:
    if not a or not b:
        return False
    al = a.lower().strip()
    bl = b.lower().strip()
    if al == bl:
        return True
    aw = al.split()
    bw = bl.split()
    return any(w in bl for w in aw) or any(w in al for w in bw)


def accuracy(cleaned: List[Dict], expected: List[Dict]) -> Dict:
    total = min(len(cleaned), len(expected))
    v_ok = c_ok = d_ok = a_ok = 0
    for i in range(total):
        row = cleaned[i]
        exp = expected[i]
        cv = str(row.get("standardized_vendor", row.get("merchant", "")).strip())
        ev = str(exp.get("expected_clean_merchant", "").strip())
        if _fuzzy_contains(cv, ev):
            v_ok += 1
        cc = str(row.get("category", row.get("Category", "")).strip())
        ec = str(exp.get("expected_category", "").strip())
        if _fuzzy_category(cc, ec):
            c_ok += 1
        cd = str(row.get("date", row.get("Date", "")).strip())
        ed = str(exp.get("expected_date", "").strip())
        if (not ed) or (cd == ed):
            d_ok += 1
        ca = row.get("amount", row.get("Amount", 0))
        try:
            ca = float(ca)
        except Exception:
            ca = 0.0
        ea = exp.get("expected_amount", ca)
        try:
            ea = float(ea) if ea is not None else ca
        except Exception:
            ea = ca
        if abs(ca - ea) < 1e-6:
            a_ok += 1

    def pct(n):
        return (n / total) * 100 if total else 0.0

    overall = pct(v_ok) * 0.4 + pct(c_ok) * 0.4 + pct(d_ok) * 0.1 + pct(a_ok) * 0.1
    return {
        "total": total,
        "vendor": pct(v_ok),
        "category": pct(c_ok),
        "date": pct(d_ok),
        "amount": pct(a_ok),
        "overall": overall,
    }


def run():
    print("🧪 200-row mixed accuracy test")
    X, Y = load_data_200()
    print(f"📊 Loaded {len(X)} rows")
    res = send_to_api(X)
    if not res.get("success"):
        print(f"❌ API error: {res.get('error')}")
        return False
    cleaned = res["data"].get("cleaned_data", [])
    print(f"✅ API ok in {res['processing_time']:.1f}s | returned {len(cleaned)} rows")
    m = accuracy(cleaned, Y)
    print(f"Vendor: {m['vendor']:.1f}%  Category: {m['category']:.1f}%  Date: {m['date']:.1f}%  Amount: {m['amount']:.1f}%")
    print(f"OVERALL: {m['overall']:.1f}%  on {m['total']} rows")
    return True


if __name__ == "__main__":
    raise SystemExit(0 if run() else 1)


