#!/usr/bin/env python3
import os
import json
import requests

data = [
  {
    "Transaction Date": "2025-08-12",
    "Details": "Recurring Payment - NETFLIX.COM",
    "Debit": "15.49",
    "Credit": None,
    "Posted By": "Netflix"
  },
  {
    "Transaction Date": "2025-08-12",
    "Details": "SQ *THE CORNER CAFE",
    "Debit": "22.45",
    "Credit": None,
    "Posted By": "SQ *THE CORNER CAFE"
  },
  {
    "Transaction Date": "11-AUG-2025",
    "Details": "Client Payment INV-1024",
    "Debit": None,
    "Credit": "2500.00",
    "Posted By": "Stripe"
  },
  {
    "Transaction Date": "2025-08-11",
    "Details": "VENMO PAYMENT to John Appleseed",
    "Debit": "(100.00)",
    "Credit": None,
    "Posted By": "VENMO"
  },
  {
    "Transaction Date": "2025-08-11",
    "Details": "AMZN Mktp US*A1B2C3D4",
    "Debit": "78.99",
    "Credit": None,
    "Posted By": "AMZN Mktp"
  },
  {
    "Transaction Date": "2025-08-10",
    "Details": "TST* The Local Coffee Spot",
    "Debit": "5.75",
    "Credit": None,
    "Posted By": "The Local Coffee Spot"
  },
  {
    "Transaction Date": "2025-08-10",
    "Details": "Online payment to COMCAST CABLE",
    "Debit": "85.50",
    "Credit": None,
    "Posted By": "COMCAST"
  },
  {
    "Transaction Date": "2025-08-09",
    "Details": "Purchase from UNITED 0167-1234567890 JFK",
    "Debit": "412.30",
    "Credit": None,
    "Posted By": None
  },
  {
    "Transaction Date": "2025-08-14",
    "Details": "TARGET #1234 - RETURN",
    "Debit": None,
    "Credit": "45.21",
    "Posted By": "TARGET"
  }
]

def normalize_rows(rows):
    norm = []
    def parse_amount(debit, credit):
        s = debit if debit not in (None, "") else credit
        if s in (None, ""):
            return 0.0
        txt = str(s).strip().replace(",", "")
        neg = False
        if txt.startswith("(") and txt.endswith(")"):
            neg = True
            txt = txt[1:-1]
        try:
            val = float(txt)
        except Exception:
            val = 0.0
        if credit not in (None, "") and (debit in (None, "")):
            # treat credit as negative outflow
            val = -val
        if neg:
            val = -val
        return abs(val)

    def extract_merchant(posted_by, details):
        m = (posted_by or "").strip()
        if not m and details:
            m = str(details).split(" ")[0]
        return m

    for r in rows:
        norm.append({
            "Date": r.get("Transaction Date", ""),
            "Merchant": extract_merchant(r.get("Posted By"), r.get("Details")),
            "Amount": parse_amount(r.get("Debit"), r.get("Credit")),
            "Description": r.get("Details", "")
        })
    return norm

payload = {
  "user_intent": "standard clean",
  "data": normalize_rows(data),
  "config": {
    "preserve_schema": True,
    "enable_ai": True,
    "ai_vendor_enabled": True,
    "ai_category_enabled": True,
    "ai_confidence_threshold": 0.7,
    "ai_preserve_schema_apply": True,
    "ai_mode": "apply"
  }
}

base_url = os.getenv("API_BASE_URL", "http://127.0.0.1:8081")
r = requests.post(f"{base_url}/process", json=payload, timeout=120)
print(f"HTTP {r.status_code}")
resp = r.json() if r.status_code == 200 else {}

rows = resp.get("cleaned_data", [])
print(f"Returned {len(rows)} rows")

def pick(row, *keys):
    out = {}
    for k in keys:
        if k in row:
            out[k] = row[k]
    return out

for i, row in enumerate(rows, 1):
    # Display key fields
    print(json.dumps({
        "#": i,
        **pick(row, "Transaction Date", "Date", "date"),
        **pick(row, "Amount", "amount"),
        **pick(row, "standardized_vendor", "Clean Vendor", "Merchant", "merchant"),
        **pick(row, "category", "Category"),
        **pick(row, "Details", "Description", "Description/Memo")
    }))


