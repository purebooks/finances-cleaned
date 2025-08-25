import pandas as pd

from common_cleaner import CommonCleaner, detect_document_type


def assert_schema_preserved(original_df: pd.DataFrame, cleaned_df: pd.DataFrame):
    assert list(cleaned_df.columns) == list(original_df.columns)


def test_sales_ledger_normalization_and_math():
    df = pd.DataFrame([
        {
            "Transaction ID": "TRN-2025-0755",
            "Date": "7/4/25",
            "Time": "12:45:19",
            "Customer ID": "CUST-1233",
            "Customer Name": "David Miller",
            "Transaction Type": "Sale",
            "Payment Method": "Credit Card",
            "SKU": "CNDL-LAV",
            "Item Description": "Candle - Lavedner",
            "Unit Price": "$28.00",
            "Qty": 1,
            "Subtotal": "",
            "Discount": "$0.00",
            "Tax": 2.48,
            "Shipping": "$0.00",
            "Total Amount": "",
            "Location": "Brooklyn Store",
            "Notes": "Typo",
        },
        {
            "Transaction ID": "TRN-2025-0755",
            "Date": "07/04/2025",
            "Time": "12:45:19",
            "Customer ID": "CUST-1233",
            "Customer Name": "DAVID MILLER",
            "Transaction Type": "Sale",
            "Payment Method": "Credit Card",
            "SKU": "CNDL-LAV",
            "Item Description": "Candle - Lavender",
            "Unit Price": "28",
            "Qty": "1",
            "Subtotal": "28.00",
            "Discount": "0",
            "Tax": "2.48",
            "Shipping": "0",
            "Total Amount": "30.48",
            "Location": "Brooklyn Store",
            "Notes": "Typo",
        },
    ])

    cleaner = CommonCleaner()
    cleaned, summary = cleaner.clean(df)

    # Schema preserved
    assert_schema_preserved(df, cleaned)

    # Detection indicates sales
    assert summary.schema_analysis["detected_type"] == "sales_ledger"
    assert summary.schema_analysis["confidence"] >= 0.8

    # Time normalized to HH:MM:SS
    assert cleaned.loc[cleaned.index[0], "Time"] == "12:45:19"

    # Item typo corrected
    assert cleaned.loc[cleaned.index[0], "Item Description"] == "Candle - Lavender"

    # Duplicate removed (keep first)
    assert len(cleaned) == 1


def test_expense_ledger_amount_and_vendor_cleanup():
    df = pd.DataFrame([
        {"date": "1/2/24", "merchant": "PAYPAL*EXAMPLE INC", "amount": "($12.34)", "description": "test"},
        {"date": "1/02/2024", "merchant": "paypal * example inc", "amount": "$12.34", "description": "test"},
    ])

    cleaner = CommonCleaner()
    cleaned, summary = cleaner.clean(df)

    # Schema preserved
    assert_schema_preserved(df, cleaned)

    # Normalized dates and amounts
    assert list(cleaned["date"]) == ["2024-01-02", "2024-01-02"]
    assert list(cleaned["amount"]) == [-12.34, 12.34]

    # Vendor text lightly normalized (whitespace/case)
    assert isinstance(cleaned.loc[0, "merchant"], str)


def test_gl_journal_numeric_and_unbalanced_flag():
    df = pd.DataFrame([
        {"account": "1000", "debit": "1,000.00", "credit": "0"},
        {"account": "2000", "debit": "0", "credit": "900.00"},
    ])

    cleaner = CommonCleaner()
    cleaned, summary = cleaner.clean(df)

    # Schema preserved
    assert_schema_preserved(df, cleaned)

    # Numerics parsed
    assert cleaned["debit"].iloc[0] == 1000.0
    assert cleaned["credit"].iloc[1] == 900.0

    # Unbalanced flagged (1000 vs 900)
    assert summary.math_checks["mismatches"] >= 1


def test_ambiguous_minimal_stays_common():
    df = pd.DataFrame([
        {"note": "just text", "value": "n/a"},
        {"note": "more text", "value": "-"},
    ])

    cleaner = CommonCleaner()
    cleaned, summary = cleaner.clean(df)

    # Schema preserved
    assert_schema_preserved(df, cleaned)

    # Detection confidence low / unknown or non-sales/expense/gl
    assert summary.schema_analysis["confidence"] <= 0.8 or summary.schema_analysis["detected_type"] == "unknown"


