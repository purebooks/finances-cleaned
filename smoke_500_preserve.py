#!/usr/bin/env python3
import os
import random
import time
import json
from typing import List, Dict, Any
from concurrent.futures import ThreadPoolExecutor, as_completed

import subprocess

try:
    import requests
except Exception as e:
    raise SystemExit("The 'requests' package is required. Please install it in your environment.")


def get_service_url() -> str:
    env = os.getenv("SERVICE_URL")
    if env:
        return env.strip()
    try:
        out = subprocess.check_output([
            "gcloud", "run", "services", "describe", "financial-cleaner-v2",
            "--region=us-central1", "--format=value(status.url)"
        ], text=True).strip()
        if out:
            return out
    except Exception:
        pass
    raise SystemExit("Could not resolve Cloud Run service URL. Set SERVICE_URL env var.")


def get_identity_token() -> str:
    try:
        out = subprocess.check_output(["gcloud", "auth", "print-identity-token"], text=True).strip()
        if out:
            return out
    except Exception:
        pass
    raise SystemExit("Could not get identity token. Ensure you are logged in: gcloud auth login")


def build_unseen_vendors() -> List[str]:
    return [
        # Unseen or less-common brands across categories
        "Omega Hardware", "Nova Print Co.", "SkyCart", "BlueFin Travel", "QuickShip.io",
        "ByteForge", "Nimbus HR", "Aegis Insurance", "CityFuel", "ParkLine", "CloudCore",
        "AlphaPay", "ZenCarto", "GreenLeaf Cafe", "Metro Diner", "Urban Staples",
        "BrightPixel", "CodeSmiths", "CoreStack", "PaperTrail", "MoveMax Rentals",
        "ShipMaster", "PostPro", "AdVenture Media", "SnapStorm Ads", "PinPoint Ads",
        "StreamBox", "DocuLocker", "DevFlow", "TaskPilot", "MeshWorks", "GridLabs",
        "OfficeBay", "PartsNexus", "RetailHub", "LocalMarket", "Prime Parcel",
        "U-Park Garage", "CityPark", "ParkEase", "SkyLodge Hotels", "Harbor Inn",
        "Summit Suites", "PayFlow", "Stripe Fees", "CardPro", "CapitalLink",
        "BlueShield", "SecureLife", "Town Water Co", "City Internet", "Metro Gas Co",
    ]


def build_known_vendors() -> List[str]:
    return [
        # Known-safe list including recent rules
        "Global Supplies", "Epsilon Goods", "Gamma Services", "Acme Corp", "Delta Manufacturing",
        "Alpha Traders", "Theta Apparel", "Beta Logistics", "Notion", "Geico", "Stamps.com",
        "ParkWhiz", "Figma", "Airtable", "Asana", "Monday", "Backblaze", "Wasabi",
        "Squarespace", "Wix", "Heroku", "Rippling", "BambooHR", "T-Mobile", "Cox",
        "CenturyLink", "Frontier", "ShipStation", "Pirate Ship", "Lowes", "OfficeMax",
        "Micro Center", "ParkWhiz", "SpotHero", "U-Haul",
    ]


def random_amount() -> float:
    # Mix small, medium, large
    bucket = random.random()
    if bucket < 0.3:
        return round(random.uniform(5, 25), 2)
    elif bucket < 0.7:
        return round(random.uniform(25, 300), 2)
    else:
        return round(random.uniform(300, 5000), 2)


def random_description(vendor: str) -> str:
    hints = [
        "subscription", "annual plan", "monthly fee", "maintenance", "support",
        "travel", "airfare", "hotel", "parking", "fuel", "office supplies",
        "shipping label", "packing", "ads campaign", "processing fees", "insurance premium",
        "internet charge", "utilities", "legal review", "consulting", "payroll"
    ]
    return random.choice(hints)


def build_rows(n: int) -> List[Dict[str, Any]]:
    unseen = build_unseen_vendors()
    known = build_known_vendors()
    pool = unseen + known
    rows: List[Dict[str, Any]] = []
    base_date = "2025-08-{:02d}".format(random.randint(1, 28))
    for i in range(n):
        v = random.choice(pool)
        rows.append({
            "Date": base_date,
            "Merchant": v,
            "Amount": random_amount(),
            "Description": random_description(v)
        })
    # Inject a couple of duplicates
    if n >= 10:
        rows[5] = rows[4].copy()
        rows[15] = rows[14].copy()
    return rows


def process_batch(service_url: str, token: str, batch: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    # Client-side robustness controls
    retries = int(os.getenv("SMOKE_RETRIES", "3"))
    timeout = int(os.getenv("SMOKE_TIMEOUT", "240"))

    payload = {
        "user_intent": "standard clean",
        "data": batch,
        "config": {
            "preserve_schema": True,
            "enable_ai": True,
            "ai_mode": "apply",
            "ai_vendor_enabled": False,
            "ai_category_enabled": True,
            "ai_preserve_schema_apply": True,
            "ai_confidence_threshold": 0.6,
            "ai_batch_size": 20,
            "ai_cost_cap_per_request": 0.5,
            "ai_model": "gpt-4o-mini"
        }
    }
    last_err = None
    for attempt in range(retries + 1):
        try:
            resp = requests.post(
                f"{service_url}/process",
                headers={
                    "Authorization": f"Bearer {token}",
                    "Content-Type": "application/json"
                },
                data=json.dumps(payload), timeout=timeout
            )
            # If service is public, try without auth on 403
            if resp.status_code == 403:
                resp = requests.post(
                    f"{service_url}/process",
                    headers={"Content-Type": "application/json"},
                    data=json.dumps(payload), timeout=timeout
                )
            resp.raise_for_status()
            data = resp.json()
            return data.get("cleaned_data", [])
        except Exception as e:
            last_err = e
            if attempt < retries:
                # Exponential backoff with jitter
                backoff = min(2 ** attempt, 8)
                time.sleep(backoff + random.uniform(0, 0.5))
            else:
                raise


def run_smoke_test(total_rows: int = 500, batch_size: int = 50, max_workers: int = 3) -> None:
    service_url = get_service_url()
    token = get_identity_token()
    # Allow env overrides for large runs
    total_rows = int(os.getenv("SMOKE_TOTAL", str(total_rows)))
    batch_size = int(os.getenv("SMOKE_BATCH", str(batch_size)))
    max_workers = int(os.getenv("SMOKE_WORKERS", str(max_workers)))
    rows = build_rows(total_rows)

    # Batch
    batches = [rows[i:i + batch_size] for i in range(0, len(rows), batch_size)]
    all_cleaned: List[Dict[str, Any]] = []
    started = time.time()

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = [ex.submit(process_batch, service_url, token, b) for b in batches]
        for fut in as_completed(futures):
            try:
                all_cleaned.extend(fut.result())
            except Exception as e:
                print(f"Batch failed: {e}")

    elapsed = time.time() - started

    # Metrics
    def get_cat(r: Dict[str, Any]) -> str:
        c = r.get("Category")
        if c is None:
            c = r.get("category", "")
        return (c or "").strip()

    total = len(all_cleaned)
    blanks = sum(1 for r in all_cleaned if get_cat(r) == "")
    other = sum(1 for r in all_cleaned if get_cat(r) == "Other")

    # Top categories
    from collections import Counter
    cats = Counter(get_cat(r) or "[blank]" for r in all_cleaned)
    top5 = cats.most_common(5)

    print(json.dumps({
        "total_input_rows": total_rows,
        "total_returned_rows": total,
        "processing_seconds": round(elapsed, 2),
        "blank_categories": blanks,
        "blank_rate": round(blanks / max(1, total), 3),
        "other_categories": other,
        "other_rate": round(other / max(1, total), 3),
        "top_categories": top5,
    }, indent=2))


if __name__ == "__main__":
    run_smoke_test()
