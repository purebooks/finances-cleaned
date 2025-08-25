import pandas as pd

from common_cleaner import CommonCleaner


def test_mixed_expense_dataset_normalization_and_dedup():
    cols = [
        "Transaction ID",
        "Date",
        "Description",
        "Vendor/Customer",
        "Category",
        "Amount",
        "Notes",
    ]

    rows = [
        ["701", "01-Jul-2025", "Coffee Beans", "Supreme Roasters", "COGS", "$450.25", ""],
        ["702", "July 2, 2025", "UTILITIES", "CT Power & Light", "Utilities", "$325.00", ""],
        ["703", "7/3/25", "Payroll", "Staff", "Payroll Expense", "$1,250.75", ""],
        ["704", "2025-07-05", "Flour, Sugar, Milk", "SuperMart", "Food Supplies", "215.5", "Text receipt"],
        ["705", "7/5/25", "Marketing", "Facebook Ads", "Marketing", "$75.00", ""],
        ["706", "7/6/25", "Cash Sale", "Walk-in Customer", "Sales Revenue", "$1,850.00", ""],
        ["", "7/8/25", "Ink and Paper", "Office Depot", "Supplies", "$45.99", "For the printer"],
        ["708", "7/9/25", "Dinner with friend", "The Oak Grill", "Miscellaneous", "$112.50", "Personal"],
        ["709", "7/10/25", "Payment for Inv 23A", "Smith Catering", "Accounts Receivable", "$500.00", ""],
        ["710", "11 July 2025", "New Espresso Machine", "Pro Kitchen Supply", "Equipment", "$3,500.00", "Capital expense"],
        ["711", "July 12", "Refund for stale pastry", "Walk-in", "Sales Revenue", "$(5.50)", ""],
        ["701", "7/13/25", "Coffee Beans", "Supreme Roasters", "COGS", "$450.25", "auto-payment"],
        ["712", "7/15/25", "Greese trap cleaning", "Sanitation Co.", "Maintanance", "$150.00", "Typo in service name"],
        ["713", "7/17/25", "misc", "petty cash", "Miscellaneous", "$50.00", "What was this for?"],
        ["714", "7/20/25", "rent", "Landlord LLC", "Rent Expense", "2000", ""],
        ["715", "7/22/25", "Sales", "", "Sales", "$1,980.45", ""],
    ]

    df = pd.DataFrame(rows, columns=cols)

    cleaner = CommonCleaner()
    cleaned, summary = cleaner.clean(df)

    # Schema preserved
    assert list(cleaned.columns) == cols

    # Duplicate removal by Transaction ID (701 repeated)
    assert len(cleaned) == len(df) - 1
    assert summary.processing_summary["duplicates_removed"] == 1

    # Dates normalized to ISO (spot-check a few)
    # Note: 'July 12' is parsed to current year (expected 2025 in test environment)
    expectations = {
        0: "2025-07-01",
        1: "2025-07-02",
        2: "2025-07-03",
        3: "2025-07-05",
    }
    for idx, iso in expectations.items():
        # Adjust for dedup effect: after dedup, the first 701 row remains, the second (index 11) removed
        if idx < 11:
            assert cleaned.iloc[idx]["Date"] == iso

    # Amounts parsed (signs and commas/parentheses)
    # First row $450.25 => 450.25
    assert cleaned.iloc[0]["Amount"] == 450.25
    # Payroll $1,250.75 => 1250.75
    assert cleaned.iloc[2]["Amount"] == 1250.75
    # Refund $(5.50) => -5.5 (row was before duplicate removal index 10; after dedup it remains)
    # Find by Description match
    refund_row = cleaned[cleaned["Description"] == "Refund for stale pastry"].iloc[0]
    assert refund_row["Amount"] == -5.5

    # Vendor/Customer lightly title-cased
    assert cleaned.iloc[1]["Vendor/Customer"] == "Ct Power & Light"
    assert cleaned.iloc[13 - 1]["Vendor/Customer"] == "Petty Cash"  # index shifted by one due to dedup

    # Category trimmed (no invention or auto-correction of typos)
    maint_row = cleaned[cleaned["Description"] == "Greese trap cleaning"].iloc[0]
    assert maint_row["Category"] == "Maintanance"


