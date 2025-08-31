import json
import os
import sys
import types

import pandas as pd

# Ensure repository root is importable
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT not in sys.path:
	sys.path.insert(0, ROOT)

from common_cleaner import CommonCleaner


def make_df():
	return pd.DataFrame([
		{"merchant": "PAYPAL*DIGITALOCEAN", "amount": "50.00", "description": "Hosting payment", "date": "2024/06/01"},
		{"merchant": "SQ *COFFEE SHOP NYC", "amount": "(4.50)", "description": " Coffee purchase ", "date": "06-02-2024"},
	])


def test_minimal_mode_preserves_values_except_trim():
	df = make_df()
	cleaner = CommonCleaner(config={"preserve_schema": True, "cleaning_mode": "minimal"})
	cleaned, _ = cleaner.clean(df)
	# Merchant should not be title-cased in minimal mode
	assert cleaned.loc[0, "merchant"] == "PAYPAL*DIGITALOCEAN"
	assert cleaned.loc[1, "merchant"].startswith("SQ *COFFEE SHOP NYC")
	# Description should be trimmed
	assert cleaned.loc[1, "description"] == "Coffee purchase"
	# Amount/date should not be normalized
	assert cleaned.loc[0, "amount"] == "50.00"
	assert cleaned.loc[0, "date"] == "2024/06/01"


def test_standard_mode_normalizes_numbers_and_dates_and_titles():
	df = make_df()
	cleaner = CommonCleaner(config={"preserve_schema": True, "cleaning_mode": "standard"})
	cleaned, _ = cleaner.clean(df)
	# Title-cased merchant
	assert cleaned.loc[0, "merchant"] != "PAYPAL*DIGITALOCEAN"
	# Numbers normalized
	assert cleaned.loc[0, "amount"] == 50.0
	assert cleaned.loc[1, "amount"] == -4.5
	# Dates normalized
	assert cleaned.loc[0, "date"] == "2024-06-01"
	assert cleaned.loc[1, "date"] == "2024-06-02"

