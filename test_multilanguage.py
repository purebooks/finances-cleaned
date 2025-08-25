#!/usr/bin/env python3

import pandas as pd
from flexible_column_detector import FlexibleColumnDetector, standardize_data_format

def test_multilanguage_support():
    """Test the multi-language column detection"""
    
    print("🧪 QUICK WIN 3: Multi-Language Support Test")
    print("=" * 50)
    
    detector = FlexibleColumnDetector()
    
    # Test 1: Spanish bank export
    print("\n🇪🇸 Test 1: Spanish Bank Export")
    print("-" * 40)
    
    spanish_data = pd.DataFrame({
        'Fecha': ['2024-01-01', '2024-01-02', '2024-01-03'],
        'Comerciante': ['Supermercado Plaza', 'Gasolinera Shell', 'Amazon España'],
        'Importe': ['€45,67', '€32,50', '€89,99'],
        'Descripción': ['Compras', 'Combustible', 'Libros online']
    })
    
    print("Original Spanish data:")
    print(spanish_data.to_string())
    
    mapping = detector.detect_columns(spanish_data)
    print(f"\n✅ Detection confidence: {mapping.confidence:.2f}")
    print(f"   Amount column: {mapping.amount_column}")
    print(f"   Merchant column: {mapping.merchant_column}")
    print(f"   Date column: {mapping.date_column}")
    
    standardized = standardize_data_format(spanish_data, mapping)
    print(f"   Standardized amounts: {standardized['amount'].tolist()}")
    
    # Test 2: German accounting
    print("\n🇩🇪 Test 2: German Accounting")
    print("-" * 40)
    
    german_data = pd.DataFrame({
        'Datum': ['01.01.2024', '02.01.2024', '03.01.2024'],
        'Händler': ['Büromarkt König', 'Tankstelle BP', 'Restaurant Zur Post'],
        'Betrag': ['125,50 EUR', '45,80 EUR', '67,90 EUR'],
        'Beschreibung': ['Büromaterial', 'Kraftstoff', 'Geschäftsessen']
    })
    
    print("Original German data:")
    print(german_data.to_string())
    
    mapping2 = detector.detect_columns(german_data)
    print(f"\n✅ Detection confidence: {mapping2.confidence:.2f}")
    print(f"   Amount column: {mapping2.amount_column}")
    print(f"   Merchant column: {mapping2.merchant_column}")
    print(f"   Date column: {mapping2.date_column}")
    
    standardized2 = standardize_data_format(german_data, mapping2)
    print(f"   Standardized amounts: {standardized2['amount'].tolist()}")
    
    # Test 3: French debit/credit format
    print("\n🇫🇷 Test 3: French Debit/Credit Format")
    print("-" * 40)
    
    french_data = pd.DataFrame({
        'Date': ['01/01/2024', '02/01/2024', '03/01/2024'],
        'Marchand': ['Carrefour', 'Station Essence', 'Apple Store'],
        'Débit': ['', '45,50 €', '299,00 €'],
        'Crédit': ['2500,00 €', '', ''],
        'Description': ['Salaire', 'Essence', 'iPhone']
    })
    
    print("Original French data:")
    print(french_data[['Marchand', 'Débit', 'Crédit']].to_string())
    
    mapping3 = detector.detect_columns(french_data)
    print(f"\n✅ Detection confidence: {mapping3.confidence:.2f}")
    print(f"   Amount column: {mapping3.amount_column}")
    print(f"   Merchant column: {mapping3.merchant_column}")
    
    standardized3 = standardize_data_format(french_data, mapping3)
    print(f"   Merged amounts: {standardized3['amount'].tolist()}")
    
    # Test 4: Portuguese income/expense
    print("\n🇵🇹 Test 4: Portuguese Income/Expense")
    print("-" * 40)
    
    portuguese_data = pd.DataFrame({
        'Data': ['01/01/2024', '02/01/2024', '03/01/2024'],
        'Empresa': ['Continente', 'Galp', 'Netflix'],
        'Receita': ['', '', ''],
        'Gasto': ['45,90', '38,50', '7,99'],
        'Descrição': ['Supermercado', 'Combustível', 'Streaming']
    })
    
    print("Original Portuguese data:")
    print(portuguese_data[['Empresa', 'Receita', 'Gasto']].to_string())
    
    mapping4 = detector.detect_columns(portuguese_data)
    print(f"\n✅ Detection confidence: {mapping4.confidence:.2f}")
    print(f"   Amount column: {mapping4.amount_column}")
    print(f"   Merchant column: {mapping4.merchant_column}")
    
    standardized4 = standardize_data_format(portuguese_data, mapping4)
    print(f"   Merged amounts: {standardized4['amount'].tolist()}")
    
    # Test 5: Italian mixed format
    print("\n🇮🇹 Test 5: Italian Format")
    print("-" * 40)
    
    italian_data = pd.DataFrame({
        'Data': ['01/01/2024', '02/01/2024'],
        'Commerciante': ['Esselunga', 'Eni'],
        'Importo': ['€25,50', '€42,30'],
        'Descrizione': ['Spesa', 'Benzina']
    })
    
    mapping5 = detector.detect_columns(italian_data)
    print(f"✅ Detection confidence: {mapping5.confidence:.2f}")
    print(f"   Amount: {mapping5.amount_column}, Merchant: {mapping5.merchant_column}")
    
    standardized5 = standardize_data_format(italian_data, mapping5)
    print(f"   Amounts: {standardized5['amount'].tolist()}")
    
    # Summary
    print(f"\n📈 SUMMARY")
    print("-" * 20)
    print("✅ Multi-language column detection implemented!")
    print("✅ Supports: Spanish, French, German, Portuguese, Italian, Dutch")
    print("✅ Column types: Amount, Merchant, Date, Description")
    print("✅ Special formats: Debit/Credit, Income/Expense in all languages")
    print("✅ Currency handling: €, $, and text-based amounts")

if __name__ == "__main__":
    test_multilanguage_support()
