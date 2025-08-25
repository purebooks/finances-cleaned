import os
import re
import csv
import json
import requests


def _default_norm(text: str) -> str:
    s = str(text or '').strip().lower()
    s = re.sub(r"\s+", " ", s)
    return s


def load_rows(csv_path: str, take: int = 20):
    rows = []
    with open(csv_path, 'r') as f:
        r = csv.DictReader(f)
        for i, row in enumerate(r):
            try:
                row['Amount'] = float(row.get('Amount') or 0)
            except Exception:
                row['Amount'] = 0.0
            rows.append(row)
            if i + 1 >= take:
                break
    return rows


def fetch_cleaned(rows):
    api = os.getenv('API_BASE_URL', 'http://localhost:8103')
    payload = {
        'user_intent': 'comprehensive cleaning and standardization',
        'data': rows,
        'config': {
            'enable_ai': True,
            'ai_vendor_enabled': True,
            'ai_category_enabled': True,
            'preserve_schema': False,
            'enable_parallel_processing': True,
        },
    }
    resp = requests.post(f"{api}/process", json=payload, timeout=1200)
    resp.raise_for_status()
    return resp.json().get('cleaned_data', [])


def main():
    rows = load_rows('test_500_rows.csv', take=int(os.getenv('TAKE_ROWS', '20')))
    cleaned = fetch_cleaned(rows)

    # Map between gold taxonomy and user's taxonomy for display
    gold_to_yours = {
        'Software & Technology': 'Technology & Software',
        'Meals & Entertainment': 'Food & Dining',
        'Utilities & Rent': 'Utilities & Communications',
        'Banking & Finance': 'Financial Services',
    }

    try:
        import gold_mapping as gm
        vendor_map = gm.GOLD_VENDOR_CATEGORY_MAP
        norm = gm._normalize_vendor_name
        to_gold = gm.GOLD_CATEGORY_REMAP  # from user's -> gold
    except Exception:
        vendor_map = {}
        norm = _default_norm
        to_gold = {
            'Technology & Software': 'Software & Technology',
            'Food & Dining': 'Meals & Entertainment',
            'Utilities & Communications': 'Utilities & Rent',
            'Financial Services': 'Banking & Finance',
        }

    def propose_category(vendor: str, current_gold_cat: str):
        vnorm = norm(vendor)
        mapped = None
        if vnorm in vendor_map:
            mapped = vendor_map[vnorm]
        else:
            for k, v in vendor_map.items():
                if k in vnorm:
                    mapped = v
                    break
        # Proposed in user's taxonomy
        proposed_yours = mapped if mapped else gold_to_yours.get(current_gold_cat, current_gold_cat)
        # Proposed in gold taxonomy
        proposed_gold = to_gold.get(proposed_yours, proposed_yours)
        return proposed_yours, proposed_gold

    out = []
    for row in cleaned:
        vendor = (
            row.get('standardized_vendor')
            or row.get('Vendor')
            or row.get('Merchant')
            or ''
        )
        cur_cat = (row.get('category') or row.get('Category') or '').strip()
        yours_cur = gold_to_yours.get(cur_cat, cur_cat)
        proposed_yours, proposed_gold = propose_category(vendor, cur_cat)
        out.append({
            'merchant': (row.get('Merchant') or row.get('merchant') or '')[:60],
            'standardized_vendor': vendor[:60],
            'current_category_gold': cur_cat,
            'current_category_yours': yours_cur,
            'proposed_category_yours': proposed_yours,
            'proposed_category_gold': proposed_gold,
            'changed_in_yours': yours_cur != proposed_yours,
            'changed_in_gold': cur_cat != proposed_gold,
        })

    changed = [r for r in out if r['changed_in_gold'] or r['changed_in_yours']]
    print(json.dumps({
        'changed_rows': changed,
        'sample_all': out[:10]
    }, indent=2))


if __name__ == '__main__':
    main()


