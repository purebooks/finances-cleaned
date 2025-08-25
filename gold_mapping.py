import re
import os

ENABLE_GOLD_MAPPING = os.getenv('USE_GOLD_MAPPING', 'false').lower() == 'true'
EXTRA_MAP_PATH = os.getenv('GOLD_VENDOR_MAP_FILE')

GOLD_VENDOR_CATEGORY_MAP = {
    # Travel & Transportation (lodging and booking)
    'marriott': 'Travel & Transportation',
    'hilton': 'Travel & Transportation',
    'hyatt': 'Travel & Transportation',
    'holiday inn': 'Travel & Transportation',
    'sheraton': 'Travel & Transportation',
    'westin': 'Travel & Transportation',
    'doubletree': 'Travel & Transportation',
    'courtyard': 'Travel & Transportation',
    'fairfield inn': 'Travel & Transportation',
    'hampton inn': 'Travel & Transportation',
    'la quinta': 'Travel & Transportation',
    'motel 6': 'Travel & Transportation',
    'super 8': 'Travel & Transportation',
    'comfort inn': 'Travel & Transportation',
    'best western': 'Travel & Transportation',
    'extended stay': 'Travel & Transportation',
    'airbnb': 'Travel & Transportation',
    'vrbo': 'Travel & Transportation',
    'booking.com': 'Travel & Transportation',
    'expedia': 'Travel & Transportation',
    'hotels.com': 'Travel & Transportation',
    'priceline': 'Travel & Transportation',
    'kayak': 'Travel & Transportation',
    'orbitz': 'Travel & Transportation',
    'travelocity': 'Travel & Transportation',

    # Airlines
    'delta airlines': 'Travel & Transportation',
    'american airlines': 'Travel & Transportation',
    'united airlines': 'Travel & Transportation',
    'southwest airlines': 'Travel & Transportation',
    'jetblue': 'Travel & Transportation',
    'alaska airlines': 'Travel & Transportation',
    'spirit airlines': 'Travel & Transportation',
    'frontier airlines': 'Travel & Transportation',
    'allegiant': 'Travel & Transportation',
    'hawaiian airlines': 'Travel & Transportation',
    'lufthansa': 'Travel & Transportation',
    'british airways': 'Travel & Transportation',
    'air france': 'Travel & Transportation',
    'klm': 'Travel & Transportation',
    'emirates': 'Travel & Transportation',
    'qatar airways': 'Travel & Transportation',

    # Ground Transportation
    'uber': 'Travel & Transportation',
    'lyft': 'Travel & Transportation',
    'taxi': 'Travel & Transportation',
    'hertz': 'Travel & Transportation',
    'avis': 'Travel & Transportation',
    'enterprise': 'Travel & Transportation',
    'budget rental': 'Travel & Transportation',
    'alamo': 'Travel & Transportation',
    'national car rental': 'Travel & Transportation',
    'zipcar': 'Travel & Transportation',
    'car2go': 'Travel & Transportation',
    'amtrak': 'Travel & Transportation',
    'greyhound': 'Travel & Transportation',
    'megabus': 'Travel & Transportation',
    'parking': 'Travel & Transportation',
    'tollway': 'Travel & Transportation',
    'ez pass': 'Travel & Transportation',
    'fastrak': 'Travel & Transportation',

    # Technology & Software (will remap to Software & Technology)
    'microsoft': 'Technology & Software',
    'adobe': 'Technology & Software',
    'salesforce': 'Technology & Software',
    'oracle': 'Technology & Software',
    'sap': 'Technology & Software',
    'atlassian': 'Technology & Software',
    'slack': 'Technology & Software',
    'zoom': 'Technology & Software',
    'dropbox': 'Technology & Software',
    'box': 'Technology & Software',
    'google workspace': 'Technology & Software',
    'office 365': 'Technology & Software',
    'github': 'Technology & Software',
    'gitlab': 'Technology & Software',
    'aws': 'Technology & Software',
    'amazon web services': 'Technology & Software',
    'azure': 'Technology & Software',
    'google cloud': 'Technology & Software',
    'digitalocean': 'Technology & Software',
    'linode': 'Technology & Software',
    'vultr': 'Technology & Software',
    'cloudflare': 'Technology & Software',
    'mailchimp': 'Technology & Software',
    'hubspot': 'Technology & Software',
    'zendesk': 'Technology & Software',
    'intercom': 'Technology & Software',
    'twilio': 'Technology & Software',
    'sendgrid': 'Technology & Software',
    'stripe': 'Technology & Software',
    'paypal': 'Technology & Software',
    'square': 'Technology & Software',
    'quickbooks': 'Technology & Software',
    'xero': 'Technology & Software',
    'freshbooks': 'Technology & Software',
    'docusign': 'Technology & Software',
    'hellosign': 'Technology & Software',
    'pandadoc': 'Technology & Software',
    'canva': 'Technology & Software',
    'figma': 'Technology & Software',
    'sketch': 'Technology & Software',
    'notion': 'Technology & Software',
    'airtable': 'Technology & Software',
    'monday.com': 'Technology & Software',
    'asana': 'Technology & Software',
    'trello': 'Technology & Software',
    'jira': 'Technology & Software',
    'confluence': 'Technology & Software',

    # Office Supplies & Equipment
    'amazon': 'Office Supplies & Equipment',
    'staples': 'Office Supplies & Equipment',
    'office depot': 'Office Supplies & Equipment',
    'officemax': 'Office Supplies & Equipment',
    'best buy': 'Office Supplies & Equipment',
    'walmart': 'Office Supplies & Equipment',
    'target': 'Office Supplies & Equipment',
    'costco': 'Office Supplies & Equipment',
    'sams club': 'Office Supplies & Equipment',
    'bjs': 'Office Supplies & Equipment',
    'home depot': 'Office Supplies & Equipment',
    'lowes': 'Office Supplies & Equipment',
    'menards': 'Office Supplies & Equipment',
    'ace hardware': 'Office Supplies & Equipment',
    'uline': 'Office Supplies & Equipment',
    'grainger': 'Office Supplies & Equipment',
    'mcmaster carr': 'Office Supplies & Equipment',
    'fastenal': 'Office Supplies & Equipment',
    'dell': 'Office Supplies & Equipment',
    'hp': 'Office Supplies & Equipment',
    'lenovo': 'Office Supplies & Equipment',
    'apple': 'Office Supplies & Equipment',
    'newegg': 'Office Supplies & Equipment',
    'microcenter': 'Office Supplies & Equipment',
    "frys electronics": 'Office Supplies & Equipment',
    'radioshack': 'Office Supplies & Equipment',
    'b&h photo': 'Office Supplies & Equipment',
    'adorama': 'Office Supplies & Equipment',

    # Professional Services (incl. pharmacies, consultancies, tax/legal)
    'walgreens': 'Professional Services',
    'cvs': 'Professional Services',
    'cvs pharmacy': 'Professional Services',
    'rite aid': 'Professional Services',
    'duane reade': 'Professional Services',
    'kaiser': 'Professional Services',
    'anthem': 'Professional Services',
    'aetna': 'Professional Services',
    'cigna': 'Professional Services',
    'blue cross': 'Professional Services',
    'humana': 'Professional Services',
    'united healthcare': 'Professional Services',
    'deloitte': 'Professional Services',
    'pwc': 'Professional Services',
    'ernst & young': 'Professional Services',
    'kpmg': 'Professional Services',
    'mckinsey': 'Professional Services',
    'bain': 'Professional Services',
    'bcg': 'Professional Services',
    'accenture': 'Professional Services',
    'ibm': 'Professional Services',
    'capgemini': 'Professional Services',
    'wipro': 'Professional Services',
    'tcs': 'Professional Services',
    'infosys': 'Professional Services',
    'cognizant': 'Professional Services',
    'h&r block': 'Professional Services',
    'jackson hewitt': 'Professional Services',
    'liberty tax': 'Professional Services',
    'intuit': 'Professional Services',
    'turbotax': 'Professional Services',
    'legalzoom': 'Professional Services',
    'nolo': 'Professional Services',
    'rocket lawyer': 'Professional Services',
    'ups store': 'Professional Services',
    'fedex office': 'Professional Services',
    'kinkos': 'Professional Services',
    'minuteman press': 'Professional Services',
    'sir speedy': 'Professional Services',
    'vistaprint': 'Professional Services',
    '99designs': 'Professional Services',
    'fiverr': 'Professional Services',
    'upwork': 'Professional Services',
    'freelancer': 'Professional Services',
    'guru': 'Professional Services',
    'toptal': 'Professional Services',

    # Utilities & Communications (will remap to Utilities & Rent)
    'verizon': 'Utilities & Communications',
    'att': 'Utilities & Communications',
    'at&t': 'Utilities & Communications',
    'tmobile': 'Utilities & Communications',
    't-mobile': 'Utilities & Communications',
    'sprint': 'Utilities & Communications',
    'xfinity': 'Utilities & Communications',
    'comcast': 'Utilities & Communications',
    'cox': 'Utilities & Communications',
    'spectrum': 'Utilities & Communications',
    'charter': 'Utilities & Communications',
    'optimum': 'Utilities & Communications',
    'cablevision': 'Utilities & Communications',
    'directv': 'Utilities & Communications',
    'dish network': 'Utilities & Communications',
    'sling tv': 'Utilities & Communications',
    'hulu': 'Utilities & Communications',
    'netflix': 'Utilities & Communications',
    'amazon prime': 'Utilities & Communications',
    'disney plus': 'Utilities & Communications',
    'hbo max': 'Utilities & Communications',
    'paramount plus': 'Utilities & Communications',
    'peacock': 'Utilities & Communications',
    'apple tv': 'Utilities & Communications',
    'youtube tv': 'Utilities & Communications',
    'spotify': 'Utilities & Communications',
    'apple music': 'Utilities & Communications',
    'pandora': 'Utilities & Communications',
    'sirius xm': 'Utilities & Communications',
    'pge': 'Utilities & Communications',
    'pg&e': 'Utilities & Communications',
    'con edison': 'Utilities & Communications',
    'consolidated edison': 'Utilities & Communications',
    'duke energy': 'Utilities & Communications',
    'southern company': 'Utilities & Communications',
    'exelon': 'Utilities & Communications',
    'nextera': 'Utilities & Communications',
    'dominion energy': 'Utilities & Communications',
    'american electric': 'Utilities & Communications',
    'centerpoint': 'Utilities & Communications',
    'national grid': 'Utilities & Communications',
    'pseg': 'Utilities & Communications',
    'water department': 'Utilities & Communications',
    'water district': 'Utilities & Communications',
    'city of': 'Utilities & Communications',
    'waste management': 'Utilities & Communications',
    'republic services': 'Utilities & Communications',
    'waste connections': 'Utilities & Communications',

    # Food & Dining (will remap to Meals & Entertainment)
    'mcdonalds': 'Food & Dining',
    'burger king': 'Food & Dining',
    'wendys': 'Food & Dining',
    'taco bell': 'Food & Dining',
    'kfc': 'Food & Dining',
    'pizza hut': 'Food & Dining',
    'dominos': 'Food & Dining',
    'papa johns': 'Food & Dining',
    'subway': 'Food & Dining',
    'starbucks': 'Food & Dining',
    'dunkin': 'Food & Dining',
    'tim hortons': 'Food & Dining',
    'panera': 'Food & Dining',
    'chipotle': 'Food & Dining',
    'qdoba': 'Food & Dining',
    'five guys': 'Food & Dining',
    'in-n-out': 'Food & Dining',
    'shake shack': 'Food & Dining',
    'chick-fil-a': 'Food & Dining',
    'popeyes': 'Food & Dining',
    'applebees': 'Food & Dining',
    'olive garden': 'Food & Dining',
    'red lobster': 'Food & Dining',
    'outback': 'Food & Dining',
    'texas roadhouse': 'Food & Dining',
    'cheesecake factory': 'Food & Dining',
    'pf changs': 'Food & Dining',
    'bww': 'Food & Dining',
    'buffalo wild wings': 'Food & Dining',
    'hooters': 'Food & Dining',
    'ihop': 'Food & Dining',
    'dennys': 'Food & Dining',
    'cracker barrel': 'Food & Dining',
    'uber eats': 'Food & Dining',
    'doordash': 'Food & Dining',
    'grubhub': 'Food & Dining',
    'postmates': 'Food & Dining',
    'seamless': 'Food & Dining',
    'caviar': 'Food & Dining',
    'instacart': 'Food & Dining',
    'shipt': 'Food & Dining',
    'whole foods': 'Food & Dining',
    'trader joes': 'Food & Dining',
    'kroger': 'Food & Dining',
    'safeway': 'Food & Dining',
    'albertsons': 'Food & Dining',
    'publix': 'Food & Dining',
    'wegmans': 'Food & Dining',
    'harris teeter': 'Food & Dining',
    'giant': 'Food & Dining',
    'stop & shop': 'Food & Dining',
    'food lion': 'Food & Dining',
    'hy-vee': 'Food & Dining',
    'meijer': 'Food & Dining',
    'heb': 'Food & Dining',
    'winn dixie': 'Food & Dining',

    # Financial Services (will remap to Banking & Finance)
    'chase': 'Financial Services',
    'bank of america': 'Financial Services',
    'wells fargo': 'Financial Services',
    'citibank': 'Financial Services',
    'us bank': 'Financial Services',
    'pnc': 'Financial Services',
    'td bank': 'Financial Services',
    'regions': 'Financial Services',
    'suntrust': 'Financial Services',
    'bb&t': 'Financial Services',
    'fifth third': 'Financial Services',
    'huntington': 'Financial Services',
    'ally bank': 'Financial Services',
    'discover': 'Financial Services',
    'capital one': 'Financial Services',
    'american express': 'Financial Services',
    'visa': 'Financial Services',
    'mastercard': 'Financial Services',
    'venmo': 'Financial Services',
    'zelle': 'Financial Services',
    'cash app': 'Financial Services',
    'square cash': 'Financial Services',
    'western union': 'Financial Services',
    'moneygram': 'Financial Services',
    'charles schwab': 'Financial Services',
    'fidelity': 'Financial Services',
    'vanguard': 'Financial Services',
    'etrade': 'Financial Services',
    'td ameritrade': 'Financial Services',
    'robinhood': 'Financial Services',
    'webull': 'Financial Services',
    'interactive brokers': 'Financial Services',
    'merrill lynch': 'Financial Services',
    'morgan stanley': 'Financial Services',
    'ubs': 'Financial Services',
    'goldman sachs': 'Financial Services',
    'jpmorgan': 'Financial Services',
    'blackrock': 'Financial Services',
    'state farm': 'Financial Services',
    'geico': 'Financial Services',
    'progressive': 'Financial Services',
    'allstate': 'Financial Services',
    'farmers': 'Financial Services',
    'liberty mutual': 'Financial Services',
    'usaa': 'Financial Services',
    'nationwide': 'Financial Services',
    'travelers': 'Financial Services',
    'aig': 'Financial Services',
    'metlife': 'Financial Services',
    'prudential': 'Financial Services',
    'new york life': 'Financial Services',
    'northwestern mutual': 'Financial Services',
    'john hancock': 'Financial Services',
    'lincoln financial': 'Financial Services',

    # Government & Public Services (will be remapped to Other)
    'irs': 'Government & Public Services',
    'internal revenue': 'Government & Public Services',
    'social security': 'Government & Public Services',
    'medicare': 'Government & Public Services',
    'medicaid': 'Government & Public Services',
    'dmv': 'Government & Public Services',
    'department of motor vehicles': 'Government & Public Services',
    'post office': 'Government & Public Services',
    'usps': 'Government & Public Services',
    'federal': 'Government & Public Services',
    'state of': 'Government & Public Services',
    'county of': 'Government & Public Services',
    'city hall': 'Government & Public Services',
    'court': 'Government & Public Services',
    'courthouse': 'Government & Public Services',
    'clerk of court': 'Government & Public Services',
    'sheriff': 'Government & Public Services',
    'police': 'Government & Public Services',
    'fire department': 'Government & Public Services',
    'library': 'Government & Public Services',
    'school district': 'Government & Public Services',
    'university': 'Government & Public Services',
    'college': 'Government & Public Services',
    'parking authority': 'Government & Public Services',
    'transit authority': 'Government & Public Services',
    'housing authority': 'Government & Public Services',
    'health department': 'Government & Public Services',
    'veterans affairs': 'Government & Public Services',
    'unemployment': 'Government & Public Services',
    'workers comp': 'Government & Public Services',
}

GOLD_CATEGORY_REMAP = {
    # Map our broader labels → gold taxonomy labels
    'Technology & Software': 'Software & Technology',
    'Food & Dining': 'Meals & Entertainment',
    'Utilities & Communications': 'Utilities & Rent',
    'Financial Services': 'Banking & Finance',
    # Not present in gold taxonomy → map to Other
    'Government & Public Services': 'Other',
}

def _normalize_vendor_name(text: str) -> str:
    if not text:
        return ''
    t = str(text).strip()
    prefixes = [
        'PAYPAL *', 'PAYPAL*', 'SQ *', 'TST* ', 'TST*', 'AUTO PAY ',
        'AMZ*', 'AMZ *', 'AMAZON*', 'AMAZON *'
    ]
    for p in prefixes:
        if t.upper().startswith(p.upper()):
            t = t[len(p):].strip()
            break
    t = t.replace('*', '').replace('#', '').strip()
    t = re.sub(r"\s+", ' ', t)
    return t.lower()

def apply_gold_category_mapping(vendor: str, category: str) -> str:
    if not ENABLE_GOLD_MAPPING:
        return category
    v = _normalize_vendor_name(vendor)
    # Exact vendor mapping
    if v in GOLD_VENDOR_CATEGORY_MAP:
        mapped = GOLD_VENDOR_CATEGORY_MAP[v]
        return GOLD_CATEGORY_REMAP.get(mapped, mapped)
    # Keyword mapping
    for key, mapped in GOLD_VENDOR_CATEGORY_MAP.items():
        if key in v:
            mapped_cat = mapped
            return GOLD_CATEGORY_REMAP.get(mapped_cat, mapped_cat)
    # Global category remap (to gold taxonomy)
    return GOLD_CATEGORY_REMAP.get(category, category)

def load_extra_vendor_map() -> None:
    """Optionally load additional vendor→category pairs from JSON file.
    File format: { "vendor_key_lower": "Gold Category" }
    """
    global GOLD_VENDOR_CATEGORY_MAP
    path = EXTRA_MAP_PATH
    if not path:
        return
    try:
        import json
        with open(path, 'r') as f:
            data = json.load(f)
        if isinstance(data, dict):
            norm = {}
            for k, v in data.items():
                try:
                    k_norm = _normalize_vendor_name(k)
                except Exception:
                    k_norm = str(k).strip().lower()
                # Ensure category label aligns with gold taxonomy if provided in our taxonomy
                v_norm = GOLD_CATEGORY_REMAP.get(v, v)
                if k_norm and v_norm:
                    norm[k_norm] = v_norm
            GOLD_VENDOR_CATEGORY_MAP.update(norm)
    except Exception:
        pass

# Load extra map at import if configured
load_extra_vendor_map()


