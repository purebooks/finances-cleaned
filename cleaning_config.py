#!/usr/bin/env python3

from __future__ import annotations

from typing import Dict, Any


SUPPORTED_TOGGLES = (
    'cleaning_mode',
    'enable_date_normalization',
    'enable_number_normalization',
    'enable_text_whitespace_trim',
    'enable_text_title_case',
    'enable_deduplication',
    'enable_math_recompute',
)


def build_cleaner_config(app_config: Dict[str, Any], overrides: Dict[str, Any] | None) -> Dict[str, Any]:
    """Build a CommonCleaner configuration dict from app defaults and request overrides.

    Always enforces preserve_schema=True; defaults cleaning_mode from app_config['default_cleaning_mode']
    if not provided.
    """
    cfg: Dict[str, Any] = {'preserve_schema': True}
    if isinstance(overrides, dict):
        for key in SUPPORTED_TOGGLES:
            if key in overrides:
                cfg[key] = overrides[key]
    if 'cleaning_mode' not in cfg and isinstance(app_config, dict):
        default_mode = str(app_config.get('default_cleaning_mode', 'minimal')).lower()
        cfg['cleaning_mode'] = default_mode
    return cfg

