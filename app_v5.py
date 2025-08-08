#!/usr/bin/env python3
"""
AI-Enhanced Financial Data Cleaner API v5.0
Advanced LLM Flow with Intelligent Processing
Production-ready Flask application for Cloud Run deployment
"""

import os
import logging
import time
import uuid
from datetime import datetime
from typing import Dict, Any, Optional, Tuple

import pandas as pd
from flask import Flask, request, jsonify
from flask_cors import CORS
import json

try:
    from llm_client_v2 import LLMClient
except ImportError:
    from llm_client import LLMClient
from production_cleaner_ai_v5 import AIEnhancedProductionCleanerV5
from flexible_column_detector import FlexibleColumnDetector

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- App Setup ---
app = Flask(__name__)
CORS(app)

# --- Configuration ---
APP_CONFIG = {
    'anthropic_api_key': os.getenv('ANTHROPIC_API_KEY'),
    'enable_ai': os.getenv('ENABLE_AI', 'true').lower() == 'true',
    'port': int(os.getenv('PORT', 8080)),
    'debug': os.getenv('FLASK_ENV') == 'development',
    'max_file_size_mb': 50,
    'version': '5.0.0'
}

# --- Global State ---
llm_client = None
start_time = datetime.utcnow()

# --- Helper Functions ---
def standardize_date(date_string):
    """
    Intelligently parses a date string from various formats into YYYY-MM-DD.
    Returns None if parsing fails.
    """
    if not isinstance(date_string, str) or pd.isna(date_string):
        return None
    try:
        # Use pandas to_datetime which is robust
        parsed_date = pd.to_datetime(date_string, errors='coerce')
        return parsed_date.strftime('%Y-%m-%d') if pd.notna(parsed_date) else None
    except (ValueError, TypeError):
        return None

def standardize_amount(amount_val):
    """
    Sanitizes an amount value to a standard float.
    Handles currency symbols, commas, parentheses for negatives, and other text.
    """
    if amount_val is None or pd.isna(amount_val):
        return 0.00
    
    s_amount = str(amount_val).strip()
    
    is_negative = False
    # Check for accounting-style negatives, e.g., (15.25)
    if s_amount.startswith('(') and s_amount.endswith(')'):
        is_negative = True
        s_amount = s_amount[1:-1]
        
    # Use regex to find the first valid number (int or float) in the string
    # This will strip out currency symbols, text like "Amount:", etc.
    import re
    match = re.search(r'[\d,]+\.?\d*', s_amount)
    
    if match:
        num_str = match.group(0).replace(',', '')
        try:
            amount = float(num_str)
            # If the original string had a negative sign, respect it
            if '-' in s_amount:
                amount = -abs(amount)
            # Apply negative sign if it was in accounting format
            elif is_negative:
                amount = -abs(amount)
            return amount
        except (ValueError, TypeError):
            return 0.00
            
    return 0.00

def get_llm_client() -> LLMClient:
    """Initializes and returns a singleton LLM client."""
    global llm_client
    if llm_client is None:
        use_mock = not APP_CONFIG['anthropic_api_key'] or not APP_CONFIG['enable_ai']
        if use_mock:
            logger.warning("Using mock AI client. Set ANTHROPIC_API_KEY for live AI.")
        llm_client = LLMClient(api_key=APP_CONFIG['anthropic_api_key'], use_mock=use_mock)
    return llm_client

def preprocess_and_standardize_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    The core data purification and standardization pipeline.
    This runs BEFORE any AI processing.
    """
    # Find the most likely date and amount columns
    date_col = next((col for col in df.columns if col.lower() in ['date', 'transaction date']), None)
    amount_col = next((col for col in df.columns if col.lower() in ['amount', 'price', 'cost', 'value', 'total']), None)
    
    # Standardize dates
    if date_col:
        df[date_col] = df[date_col].apply(standardize_date)
        df = df.rename(columns={date_col: 'Transaction Date'}) # Standardize name
    else:
        df['Transaction Date'] = None

    # Standardize amounts
    if amount_col:
        df[amount_col] = df[amount_col].apply(standardize_amount)
        df = df.rename(columns={amount_col: 'Amount'}) # Standardize name
    else:
        df['Amount'] = 0.00
        
    # --- Consolidate Description-like Columns (Moved from /process) ---
    description_cols = [col for col in df.columns if col.lower() in ['description', 'memo', 'notes']]
    if len(description_cols) > 1:
        df['Consolidated Description'] = df[description_cols].astype(str).agg(' | '.join, axis=1)
        df['Consolidated Description'] = df['Consolidated Description'].str.replace('nan', '').str.replace(r'\s*\|\s*\|*\s*', ' | ', regex=True).str.strip(' |')
        df = df.drop(columns=description_cols)
        df = df.rename(columns={'Consolidated Description': 'Description'})
    elif len(description_cols) == 1:
        df = df.rename(columns={description_cols[0]: 'Description'})

    # --- Phase 3 & 4: AI Imputation and "Needs Review" Flag ---
    merchant_col = next((col for col in df.columns if col.lower() in ['merchant', 'vendor', 'store']), 'Merchant')
    df = df.rename(columns={merchant_col: 'Merchant'}) if merchant_col in df.columns else df.assign(Merchant=None)

    df['Needs Review'] = False
    df['Suggestions'] = [[] for _ in range(len(df))]

    missing_merchant_mask = df['Merchant'].isnull() | (df['Merchant'] == '')
    if missing_merchant_mask.any():
        logger.info(f"Found {missing_merchant_mask.sum()} rows with missing merchants. Attempting AI imputation.")
        imputation_client = get_llm_client()
        
        for index, row in df[missing_merchant_mask].iterrows():
            description = row.get('Description', '')
            amount = row.get('Amount', 0.0)
            
            if description:
                try:
                    suggestions = imputation_client.suggest_vendors_from_description(description, amount)
                    if suggestions:
                        df.loc[index, 'Suggestions'] = [suggestions] # Nest suggestions in a list for the cell
                        df.loc[index, 'Needs Review'] = True
                        df.loc[index, 'Merchant'] = '[Vendor Missing]' # Keep placeholder for now
                        logger.info(f"AI suggested vendors {suggestions} for row {index}")
                    else:
                        df.loc[index, 'Merchant'] = '[Vendor Missing]'
                        df.loc[index, 'Needs Review'] = True
                except Exception as e:
                    logger.error(f"AI imputation failed for row {index}: {e}")
                    df.loc[index, 'Merchant'] = '[Vendor Missing]'
                    df.loc[index, 'Needs Review'] = True
            else:
                df.loc[index, 'Merchant'] = '[Vendor Missing]'
                df.loc[index, 'Needs Review'] = True
                
    # Final fallback for any remaining missing merchants
    df['Merchant'] = df['Merchant'].fillna('[Vendor Missing]')
    
    return df



def _is_unknown_vendor(merchant: str, amount: float = 0) -> bool:
    """Enhanced check for vendors that would benefit from AI processing."""
    if not merchant or not isinstance(merchant, str):
        return False
    
    merchant_lower = merchant.lower()
    
    # HIGH PRIORITY: Always use AI for high-value transactions
    if amount > 500:
        return True
    
    # HIGH PRIORITY: Complex payment processor patterns need AI parsing
    complex_processors = ["paypal *", "sq *", "stripe*", "tst*", "pp*", "venmo*"]
    if any(merchant.upper().startswith(proc.upper()) for proc in complex_processors):
        return True
    
    # HIGH PRIORITY: Cryptic/coded merchant names benefit from AI
    if any(char in merchant for char in ["*", "#", ".", "1234567890"]) and len(merchant) > 10:
        return True
    
    # Known vendor patterns (confident rule-based processing)
    known_patterns = [
        "google", "amazon", "netflix", "spotify", "apple", "microsoft", "adobe",
        "starbucks", "mcdonald", "chipotle", "subway", "pizza", "domino",
        "uber", "lyft", "shell", "chevron", "delta", "united", "southwest",
        "target", "walmart", "costco", "home depot", "best buy",
        "cvs", "walgreens", "whole foods", "safeway", "kroger",
        "bank", "chase", "wells fargo", "american express",
        "at&t", "verizon", "comcast", "hilton", "marriott", "airbnb"
    ]
    
    # Check if merchant contains any known patterns
    for pattern in known_patterns:
        if pattern in merchant_lower:
            return False
    
    # MEDIUM PRIORITY: Business entities that need context understanding
    business_keywords = ["corp", "inc", "llc", "ltd", "company", "consulting", 
                        "restaurant", "cafe", "store", "shop", "market", "solutions",
                        "services", "group", "enterprises", "technologies"]
    
    for keyword in business_keywords:
        if keyword in merchant_lower:
            return True  # AI can provide better context and categorization
    
    # MEDIUM PRIORITY: Subscription-like patterns
    if any(word in merchant_lower for word in ["subscription", "monthly", "annual", "renewal"]):
        return True
    
    # If no patterns match, likely needs AI processing
    return True

def safe_dataframe_to_json(df: pd.DataFrame) -> list:
    """Converts a DataFrame to a list of records, safely handling NaT and NaN."""
    import numpy as np
    import math
    import ast
    from collections import OrderedDict
    
    df_copy = df.copy()
    
    # Handle datetime columns
    for col in df_copy.select_dtypes(include=['datetime64[ns]']).columns:
        df_copy[col] = df_copy[col].apply(lambda x: x.isoformat() if pd.notna(x) else None)
    
    # Replace all NaN values with None (which becomes null in JSON)
    df_copy = df_copy.replace({np.nan: None})
    
    # Use pandas to_dict with orient='records' to preserve order, then reorder manually
    records = []
    for _, row in df_copy.iterrows():
        # Create record in exact order we want
        record = {
            'Transaction Date': row.get('Transaction Date', ''),
            'Amount': row.get('Amount', ''),
            'Clean Vendor': row.get('Clean Vendor', ''),
            'Category': row.get('Category', ''),
            'Description/Memo': row.get('Description/Memo', ''),
            'Needs Review': row.get('Needs Review', False),
            'Suggestions': row.get('Suggestions', [])
        }
        
        # Extract vendor suggestions from explanation if present
        explanation = row.get('vendor_explanation', '') or row.get('explanation', '')
        if explanation and 'AI vendor suggestions:' in explanation:
            try:
                suggestions_str = explanation.split('AI vendor suggestions:')[1].strip()
                suggestions = ast.literal_eval(suggestions_str) if suggestions_str.startswith('[') else []
                if isinstance(suggestions, list):
                    record['Vendor Suggestions'] = suggestions
            except Exception:
                record['Vendor Suggestions'] = []
        # Clean up NaN values
        for key, value in record.items():
            if pd.isna(value) or (isinstance(value, float) and math.isnan(value)):
                record[key] = None
            elif value == 'nan' or str(value).strip() == '':
                if key in ['Clean Vendor', 'Category']:
                    record[key] = f'[{key.replace("Clean ", "").replace("/Memo", "")} Missing]'
                else:
                    record[key] = None
        
        records.append(record)
    
    return records

def create_demo_data():
    """Creates sample data for demo endpoint."""
    return {
        "merchant": [
            "PAYPAL*DIGITALOCEAN",
            "SQ *COFFEE SHOP NYC", 
            "UBER EATS DEC15",
            "AMAZON.COM*AMZN.COM/BILL",
            "NETFLIX.COM"
        ],
        "amount": [50.00, 4.50, 23.75, 12.99, 15.99],
        "description": [
            "DigitalOcean hosting payment",
            "Coffee purchase at local shop",
            "Food delivery service",
            "Amazon Prime subscription",
            "Netflix streaming service"
        ]
    }

# --- API Endpoints ---
@app.route('/health', methods=['GET'])
def health_check():
    """Confirms the API is running."""
    uptime = (datetime.utcnow() - start_time).total_seconds()
    return jsonify({
        'status': 'healthy', 
        'timestamp': datetime.utcnow().isoformat(),
        'version': APP_CONFIG['version'],
        'uptime_seconds': uptime
    })

@app.route('/config', methods=['GET'])
def get_config():
    """Returns API configuration."""
    return jsonify({
        'enable_ai': APP_CONFIG['enable_ai'],
        'has_api_key': bool(APP_CONFIG['anthropic_api_key']),
        'max_file_size_mb': APP_CONFIG['max_file_size_mb'],
        'version': APP_CONFIG['version']
    })

@app.route('/demo', methods=['POST'])
def demo_endpoint():
    """Demo endpoint with sample data processing."""
    request_id = f"demo-{uuid.uuid4().hex[:8]}"
    start_time = time.time()
    
    try:
        demo_data = create_demo_data()
        
        # Use FlexibleColumnDetector to handle the demo data
        detector = FlexibleColumnDetector()
        df = detector.normalize_to_dataframe(demo_data)

        logger.info(f"[{request_id}] Processing demo data: {len(df)} rows")

        client = get_llm_client()
        cleaner = AIEnhancedProductionCleanerV5(df, config=None, llm_client=client, user_intent="Standard cleaning")
        cleaned_df, report = cleaner.process_data()

        processing_time = time.time() - start_time
        
        # Extract insights from the report
        insights = report.get('insights', {})
        
        return jsonify({
            'cleaned_data': safe_dataframe_to_json(cleaned_df),
            'summary_report': report.get('summary_report', {}),
            'insights': {
                'ai_requests': insights.get('ai_requests', 0),
                'ai_cost': insights.get('ai_cost', 0.0),
                'processing_time': processing_time,
                'rows_processed': len(cleaned_df)
            },
            'processing_time': processing_time,
            'request_id': request_id,
        })

    except Exception as e:
        logger.error(f"[{request_id}] Demo processing error: {e}", exc_info=True)
        return jsonify({'error': 'Demo processing failed.', 'details': str(e)}), 500

@app.route('/stats', methods=['GET'])
def get_stats():
    """Returns API statistics."""
    uptime = (datetime.utcnow() - start_time).total_seconds()
    return jsonify({
        'uptime_seconds': uptime,
        'version': APP_CONFIG['version'],
        'ai_enabled': APP_CONFIG['enable_ai'],
        'has_api_key': bool(APP_CONFIG['anthropic_api_key'])
    })

@app.route('/process', methods=['POST'])
def process_data():
    """Main data processing endpoint."""
    request_id = f"req-{uuid.uuid4().hex[:8]}"
    start_time = time.time()
    
    try:
        # --- NEW DEBUG LOGGING ---
        raw_body = request.get_data(as_text=True)
        logger.info(f"[{request_id}] ----- RAW REQUEST BODY -----")
        logger.info(raw_body)
        logger.info(f"[{request_id}] --- END RAW REQUEST BODY ---")

        request_data = request.get_json()
        logger.info(f"[{request_id}] Parsed JSON type: {type(request_data)}")
        
        user_intent = request_data.get('user_intent', '')
        
        # Extract the actual data from the request
        data = request_data.get('data', request_data)
        logger.info(f"[{request_id}] Extracted 'data' block type: {type(data)}")
        
        # Use FlexibleColumnDetector to handle various input formats
        detector = FlexibleColumnDetector()
        df = detector.normalize_to_dataframe(data)
        
        # --- 🚀 NEW: Run the full pre-processing pipeline ---
        df = preprocess_and_standardize_data(df)

        if df.empty:
            return jsonify({'error': 'No valid transaction data found.'}), 400

        logger.info(f"[{request_id}] Processing {len(df)} rows. Intent: '{user_intent or 'None'}'")

        # Check for config override in request
        config_override = request_data.get('config', {})
        use_real_llm = config_override.get('use_real_llm', True) or not config_override.get('use_mock', False)
        
        if use_real_llm and APP_CONFIG['anthropic_api_key']:
            # Smart hybrid approach: pre-analyze vendors
            unknown_vendors = []
            for row in df.itertuples():
                merchant = str(getattr(row, 'Merchant', '') or getattr(row, 'merchant', ''))
                amount = abs(float(getattr(row, 'Amount', 0) or getattr(row, 'amount', 0) or 0))
                if merchant and _is_unknown_vendor(merchant, amount):
                    unknown_vendors.append(merchant)
            
            if unknown_vendors:
                logger.info(f"[{request_id}] Found {len(unknown_vendors)} unknown vendors, using REAL LLM for enhancement")
                client = LLMClient(api_key=APP_CONFIG['anthropic_api_key'], use_mock=False)
            else:
                logger.info(f"[{request_id}] All vendors recognized, using efficient Python rules")
                client = get_llm_client()
        else:
            # Use default (mock) client
            client = get_llm_client()
            
        # Pass through request-level config so flags like force_llm_for_testing/use_mock take effect
        cleaner = AIEnhancedProductionCleanerV5(df, config=config_override, llm_client=client, user_intent=user_intent)
        cleaned_df, report = cleaner.process_data()

        processing_time = time.time() - start_time
        logger.info(f"[{request_id}] Processing finished in {processing_time:.2f}s")
        
        # --- DEBUG: Log DataFrame shapes ---
        logger.info(f"[{request_id}] Input DataFrame shape: {df.shape}")
        logger.info(f"[{request_id}] Output DataFrame shape: {cleaned_df.shape}")
        logger.info(f"[{request_id}] Output DataFrame columns: {list(cleaned_df.columns)}")
        if len(cleaned_df) <= 5:
            logger.info(f"[{request_id}] Output DataFrame preview:\n{cleaned_df}")
        else:
            logger.info(f"[{request_id}] Output DataFrame head:\n{cleaned_df.head()}")

        # Extract insights from the report
        insights = report.get('insights', {})
        
        # Create vendor transformations summary
        vendor_transformations = []
        if 'vendor_standardization' in report.get('summary_report', {}):
            for original, standardized in report['summary_report']['vendor_standardization'].items():
                vendor_transformations.append(f"{original} → {standardized}")

        # Clean up the output for professional display
        display_df = cleaned_df.copy()
        
        # Create the exact columns we want in the exact order requested
        final_df = pd.DataFrame()
        
        # Column 1: Transaction Date - find, standardize, and assign
        date_col = next((col for col in display_df.columns if col.lower() in ['date', 'transaction date']), None)
        if date_col:
            final_df['Transaction Date'] = display_df[date_col].apply(standardize_date)
        else:
            final_df['Transaction Date'] = '[Date Missing]'

            
                # Column 2: Amount - find, standardize, and assign
        amount_col = next((col for col in display_df.columns if col.lower() in ['amount', 'price', 'cost', 'value', 'total']), None)
        if amount_col:
            final_df['Amount'] = display_df[amount_col].apply(standardize_amount)
        else:
            final_df['Amount'] = 0.00

            
        # Column 3: Clean Vendor - check multiple possible names
        vendor_col = None
        for col in ['standardized_vendor', 'Merchant', 'merchant', 'vendor', 'store', 'business']:
            if col in display_df.columns:
                vendor_col = col
                break
                
        if vendor_col:
            final_df['Clean Vendor'] = display_df[vendor_col]
        else:
            final_df['Clean Vendor'] = ''
            
        # Column 4: Category  
        if 'category' in display_df.columns:
            final_df['Category'] = display_df['category']
        else:
            final_df['Category'] = ''
            
        # Column 5: Description/Memo - check multiple possible names and combine if needed
        description_col = None
        memo_col = None
        
        for col in ['Notes', 'notes', 'description', 'Description', 'desc', 'details']:
            if col in display_df.columns:
                description_col = col
                break
                
        for col in ['memo', 'Memo', 'comment', 'Comment']:
            if col in display_df.columns:
                memo_col = col
                break
        
        if description_col and memo_col:
            # Combine description and memo
            final_df['Description/Memo'] = (display_df[description_col].fillna('').astype(str) + 
                                          ' | ' + display_df[memo_col].fillna('').astype(str)).str.strip(' | ')
        elif description_col:
            final_df['Description/Memo'] = display_df[description_col].fillna('')
        elif memo_col:
            final_df['Description/Memo'] = display_df[memo_col].fillna('')
        else:
            final_df['Description/Memo'] = ''
        
        # Clean up missing values properly
        final_df['Clean Vendor'] = final_df['Clean Vendor'].apply(
            lambda x: '[Vendor Missing]' if pd.isna(x) or not str(x).strip() or str(x) == 'nan' else str(x).strip()
        )
        final_df['Category'] = final_df['Category'].apply(
            lambda x: '[Category Missing]' if pd.isna(x) or not str(x).strip() or str(x) == 'nan' else str(x).strip()
        )
        final_df['Transaction Date'] = final_df['Transaction Date'].apply(
            lambda x: '' if pd.isna(x) or str(x) == 'nan' else str(x).strip()
        )
        
        # Ensure the DataFrame columns are in the exact order we want
        column_order = ['Transaction Date', 'Amount', 'Clean Vendor', 'Category', 'Description/Memo']
        final_df = final_df[column_order]
        
        # Convert to JSON preserving order by manually building the response
        cleaned_records = []
        for _, row in final_df.iterrows():
            record = {}
            for col in column_order:
                value = row[col]
                if pd.isna(value):
                    if col == 'Clean Vendor':
                        value = '[Vendor Missing]'
                    elif col == 'Category':
                        value = '[Category Missing]'
                    else:
                        value = None
                record[col] = value
            cleaned_records.append(record)
        
        display_df = final_df
        
        return jsonify({
            'cleaned_data': cleaned_records,
            'summary_report': report.get('summary_report', {}),
            'insights': {
                'ai_requests': insights.get('ai_requests', 0),
                'ai_cost': insights.get('ai_cost', 0.0),
                'processing_time': processing_time,
                'rows_processed': len(cleaned_df),
                'vendor_transformations': vendor_transformations
            },
            'processing_time': processing_time,
            'request_id': request_id,
        })

    except Exception as e:
        logger.error(f"[{request_id}] Unhandled error: {e}", exc_info=True)
        return jsonify({'error': 'An internal server error occurred.', 'details': str(e)}), 500

if __name__ == '__main__':
    try:
        get_llm_client() # Initialize client on startup
        logger.info(f"API starting on port {APP_CONFIG['port']}")
        app.run(host='0.0.0.0', port=APP_CONFIG['port'], debug=APP_CONFIG['debug'])
    except Exception as e:
        logger.critical(f"Failed to start API: {e}", exc_info=True)
