#!/usr/bin/env python3

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, List

import pandas as pd


@dataclass
class ResolvedRoles:
    vendor: Optional[str]
    category: Optional[str]
    description: Optional[str]
    amount: Optional[str]
    date: Optional[str]


class FieldResolver:
    """Detects column roles in a DataFrame by headers and light content analysis.

    Roles:
      - vendor: first of [Vendor/Customer, Merchant, Payee, Name] (by header or substring)
      - category: first of [Category, Type, Classification, Group]
      - description: first of [Description, Memo, Notes]
      - amount: amount-like column (header hints or numeric content)
      - date: date-like column (header hints or date parse ratio)
    """

    def __init__(self, df: pd.DataFrame):
        self.df = df
        self._lower_to_actual: Dict[str, str] = {str(c).lower(): c for c in df.columns}

    def roles(self) -> ResolvedRoles:
        return ResolvedRoles(
            vendor=self.resolve("vendor"),
            category=self.resolve("category"),
            description=self.resolve("description"),
            amount=self.resolve("amount"),
            date=self.resolve("date"),
        )

    def resolve(self, role: str) -> Optional[str]:
        role = role.lower().strip()
        if role == "vendor":
            candidates = [
                "vendor/customer",
                "vendor / customer",
                "vendor",
                "customer",
                "merchant",
                "payee",
                "name",
                "business",
                "company",
            ]
            return self._match_first(candidates)
        if role == "category":
            candidates = [
                "category",
                "type",
                "classification",
                "class",
                "group",
            ]
            return self._match_first(candidates)
        if role == "description":
            candidates = [
                "description",
                "memo",
                "notes",
                "note",
                "details",
            ]
            return self._match_first(candidates)
        if role == "amount":
            return self._detect_amount_column()
        if role == "date":
            return self._detect_date_column()
        return None

    # ----------------------
    # Helpers
    # ----------------------
    def _match_first(self, candidate_keywords: List[str]) -> Optional[str]:
        if not self.df.columns.any:
            return None
        # Exact header matches first
        for key in candidate_keywords:
            actual = self._lower_to_actual.get(key)
            if actual is not None:
                return actual
        # Substring header matches second
        for key in candidate_keywords:
            for low, actual in self._lower_to_actual.items():
                if key in low:
                    return actual
        return None

    def _detect_amount_column(self) -> Optional[str]:
        # Header hints
        hints = ["amount", "price", "cost", "value", "total"]
        for key in hints:
            for low, actual in self._lower_to_actual.items():
                if key in low:
                    return actual
        # Content analysis: numeric-like distribution
        best_col: Optional[str] = None
        best_ratio = 0.0
        for col in self.df.columns:
            ser = self.df[col].dropna().astype(str)
            if ser.empty:
                continue
            # Count values with at least one digit
            ratio_digits = ser.str.contains(r"\d").mean()
            if ratio_digits > best_ratio:
                best_ratio = ratio_digits
                best_col = col
        return best_col if best_ratio >= 0.6 else None

    def _detect_date_column(self) -> Optional[str]:
        # Header hints
        hints = ["date", "transaction date", "posted", "timestamp", "time"]
        for key in hints:
            for low, actual in self._lower_to_actual.items():
                if key in low:
                    # Prefer date-like over time-only
                    if "time" in low and "date" not in low:
                        continue
                    return actual
        # Content analysis: parse to datetime and pick highest success ratio
        best_col: Optional[str] = None
        best_ratio = 0.0
        for col in self.df.columns:
            ser = self.df[col].dropna().astype(str)
            if ser.empty:
                continue
            try:
                parsed = pd.to_datetime(ser, errors="coerce")
                ratio = parsed.notna().mean()
                if ratio > best_ratio:
                    best_ratio = ratio
                    best_col = col
            except Exception:
                continue
        return best_col if best_ratio >= 0.6 else None


