#!/usr/bin/env python3

from __future__ import annotations

import pandas as pd
from typing import Dict, Any


def apply_ai_overlook(df: pd.DataFrame, cfg: Dict[str, Any], llm_client) -> None:
	"""Fill blanks only for vendor and memo-like columns. In-place.
	Requires cfg.enable_ai and cfg.ai_overlook (or ai_overlook_enabled).
	llm_client must expose resolve_vendor(). Memo suggestions are caller-resolved if needed.
	"""
	try:
		if not bool(cfg.get('enable_ai', False)):
			return
		if not bool(cfg.get('ai_overlook', cfg.get('ai_overlook_enabled', False))):
			return

		# Normalize blanks
		string_nulls = {"nan", "n/a", "none", "null", "unknown"}
		for c in [
			'standardized_vendor','Merchant','Posted By','Vendor/Customer','vendor','merchant','Clean Vendor',
			'Description','Notes','Memo','description','notes','memo','Description/Memo','Category','category']:
			if c in df.columns:
				try:
					df[c] = df[c].apply(lambda v: '' if (v is None or (isinstance(v, float) and pd.isna(v)) or (str(v).strip().lower() in string_nulls)) else str(v))
				except Exception:
					pass

		vendor_col = next((c for c in ['standardized_vendor','Merchant','Posted By','Vendor/Customer','vendor','merchant','Clean Vendor'] if c in df.columns), None)
		desc_col = next((c for c in ['Description/Memo','Description','Notes','Memo','description','notes','memo'] if c in df.columns), None)
		if vendor_col:
			for idx in df.index:
				try:
					cur = str(df.at[idx, vendor_col] or '').strip()
				except Exception:
					cur = ''
				if not cur:
					desc_val = ''
					try:
						desc_val = str(df.at[idx, desc_col] or '') if desc_col else ''
					except Exception:
						pass
					try:
						resolved = llm_client.resolve_vendor(cur, description=desc_val)
						if resolved and resolved.lower() not in ('', 'unknown', 'unknown vendor'):
							df.at[idx, vendor_col] = resolved
					except Exception:
						pass
	except Exception:
		# soft-fail: overlook is optional
		return

