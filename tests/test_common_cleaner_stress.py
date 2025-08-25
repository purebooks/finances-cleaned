import random
import pandas as pd

from common_cleaner import CommonCleaner, detect_document_type


def test_numeric_parsing_variety():
    samples = [
        ("$1,234.56", 1234.56),
        ("(1,234.56)", -1234.56),
        ("-$1,234.56", -1234.56),
        ("USD 99.99", 99.99),
        ("  0  ", 0.0),
        ("N/A", "N/A"),
        ("Amount: 45", 45.0),
        ("€12,345", 12345.0),
        ("(£77.7) fee", -77.7),
    ]

    df = pd.DataFrame({"Amount": [s[0] for s in samples]})
    c = CommonCleaner()
    cleaned, _ = c.clean(df)

    for i, (_, expected) in enumerate(samples):
        if isinstance(expected, str):
            # Unparseable stays as-is
            assert cleaned.loc[i, "Amount"] == samples[i][0]
        else:
            assert cleaned.loc[i, "Amount"] == expected


def test_date_parsing_variety():
    cases = [
        ("2025-07-01", "2025-07-01"),
        ("7/1/25", "2025-07-01"),
        ("01-Jul-2025", "2025-07-01"),
        ("July 2, 2025", "2025-07-02"),
        ("11 July 2025", "2025-07-11"),
        ("bad-date", "bad-date"),
    ]

    df = pd.DataFrame({"Date": [d for d, _ in cases]})
    c = CommonCleaner()
    cleaned, _ = c.clean(df)
    for i, (_, iso) in enumerate(cases):
        assert cleaned.loc[i, "Date"] == iso


def test_sales_respect_blanks_and_flag_mismatch():
    # Has ≥4 sales signals to trigger type-specific processing
    df = pd.DataFrame([
        {"SKU": "ABC", "Item Description": "X", "Qty": 2, "Unit Price": 10.0, "Subtotal": "", "Total Amount": ""},
        {"SKU": "ABC", "Item Description": "X", "Qty": 2, "Unit Price": 10.0, "Subtotal": 15.0, "Total Amount": 18.0},
    ])

    c = CommonCleaner()
    cleaned, summary = c.clean(df)

    # Row 0: blanks filled
    assert cleaned.loc[0, "Subtotal"] == 20.0
    assert cleaned.loc[0, "Total Amount"] == 20.0

    # Row 1: mismatch detected, but values not overwritten
    assert cleaned.loc[1, "Subtotal"] == 15.0
    assert summary.math_checks["mismatches"] >= 1


def test_schema_invariance_on_wide_columns():
    # Build a wide DF with mixed types
    cols = [f"col{i}" for i in range(50)]
    data = []
    random.seed(0)
    for _ in range(10):
        row = []
        for j in range(50):
            if j % 5 == 0:
                row.append("01/01/25")
            elif j % 5 == 1:
                row.append("$1,234.56")
            elif j % 5 == 2:
                row.append("Some text   with   spaces")
            elif j % 5 == 3:
                row.append(123)
            else:
                row.append(None)
        data.append(row)
    df = pd.DataFrame(data, columns=cols)

    c = CommonCleaner()
    cleaned, _ = c.clean(df)
    assert list(cleaned.columns) == cols
    # Spot check a few cells transformed
    assert cleaned.iloc[0, 0] == "2025-01-01"
    assert cleaned.iloc[0, 1] == 1234.56
    assert cleaned.iloc[0, 2] == "Some text with spaces"


def test_detection_threshold_blocks_type_specific():
    # Only 3 sales signals (below required strong signal count), so no type-specific
    df = pd.DataFrame([
        {"Item Description": "X", "Qty": 2, "Total Amount": ""},  # missing SKU/Unit Price/Subtotal
    ])

    c = CommonCleaner()
    cleaned, summary = c.clean(df)

    # We expect to stay in common path; Total remains blank (no fill without subtotal)
    assert cleaned.loc[0, "Total Amount"] == ""
    assert summary.schema_analysis["confidence"] < 0.8


