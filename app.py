#!/usr/bin/env python3
"""
AI-Enhanced Financial Data Cleaner API
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

from llm_client import LLMClient
from production_cleaner_ai import AIEnhancedProductionCleaner

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for web frontend

# Global statistics tracking
app_stats = {
    'total_requests': 0,
    'total_ai_calls': 0,
    'total_cost': 0.0,
    'start_time': datetime.utcnow().isoformat(),
    'version': 'v4.2'
}

# Configuration from environment variables
APP_CONFIG = {
    'max_file_size': int(os.getenv('MAX_FILE_SIZE', '50')) * 1024 * 1024,  # 50MB default
    'enable_ai': os.getenv('ENABLE_AI', 'true').lower() == 'true',
    'ai_confidence_threshold': float(os.getenv('AI_CONFIDENCE_THRESHOLD', '0.7')),
    'log_level': os.getenv('LOG_LEVEL', 'INFO'),
    'anthropic_api_key': os.getenv('ANTHROPIC_API_KEY'),
}

def create_llm_client() -> LLMClient:
    """Create LLM client with proper configuration"""
    use_mock = not APP_CONFIG['anthropic_api_key'] or not APP_CONFIG['enable_ai']
    
    if use_mock:
        logger.warning("Using mock AI client - set ANTHROPIC_API_KEY for live AI")
    
    return LLMClient(
        api_key=APP_CONFIG['anthropic_api_key'],
        use_mock=use_mock,
        enable_caching=True
    )

def generate_request_id() -> str:
    """Generate unique request ID for tracking"""
    return f"req-{uuid.uuid4().hex[:8]}"

def validate_input_data(data: Dict[str, Any]) -> Tuple[bool, str]:
    """Validate input data structure"""
    if not isinstance(data, dict):
        return False, "Data must be a dictionary"
    
    required_fields = ['merchant']
    if not all(field in data for field in required_fields):
        return False, f"Missing required fields: {required_fields}"
    
    # Validate data types
    for field, values in data.items():
        if not isinstance(values, list):
            return False, f"Field '{field}' must be a list"
    
    # Validate consistent lengths
    lengths = [len(values) for values in data.values()]
    if len(set(lengths)) > 1:
        return False, "All data fields must have the same length"
    
    return True, ""

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint for Cloud Run"""
    return jsonify({
        'status': 'healthy',
        'version': app_stats['version'],
        'timestamp': datetime.utcnow().isoformat(),
        'uptime_seconds': (datetime.utcnow() - datetime.fromisoformat(app_stats['start_time'])).total_seconds()
    })

@app.route('/ready', methods=['GET'])
def readiness_check():
    """Readiness check endpoint"""
    try:
        # Test AI client initialization
        llm_client = create_llm_client()
        return jsonify({
            'status': 'ready',
            'ai_enabled': APP_CONFIG['enable_ai'],
            'mock_mode': llm_client.use_mock
        })
    except Exception as e:
        logger.error(f"Readiness check failed: {e}")
        return jsonify({
            'status': 'not_ready',
            'error': str(e)
        }), 503

@app.route('/stats', methods=['GET'])
def get_statistics():
    """Get application statistics"""
    return jsonify({
        **app_stats,
        'uptime_seconds': (datetime.utcnow() - datetime.fromisoformat(app_stats['start_time'])).total_seconds(),
        'average_cost_per_request': app_stats['total_cost'] / max(app_stats['total_requests'], 1),
        'requests_per_hour': app_stats['total_requests'] / max((datetime.utcnow() - datetime.fromisoformat(app_stats['start_time'])).total_seconds() / 3600, 1)
    })

@app.route('/process', methods=['POST'])
def process_financial_data():
    """Main endpoint for processing financial data"""
    request_id = generate_request_id()
    start_time = time.time()
    
    logger.info(f"[{request_id}] Processing request started")
    
    try:
        # Validate request
        if not request.is_json:
            return jsonify({
                'error': 'Request must be JSON',
                'request_id': request_id
            }), 400
        
        request_data = request.get_json()
        
        # Extract data and config
        data = request_data.get('data', {})
        config = request_data.get('config', {})
        
        # Validate input data
        is_valid, error_msg = validate_input_data(data)
        if not is_valid:
            return jsonify({
                'error': error_msg,
                'request_id': request_id
            }), 400
        
        # Convert to DataFrame
        df = pd.DataFrame(data)
        
        # Check file size limits
        if len(df) > 100000:  # 100k rows limit
            return jsonify({
                'error': 'Dataset too large. Maximum 100,000 transactions allowed.',
                'request_id': request_id
            }), 413
        
        logger.info(f"[{request_id}] Processing {len(df)} transactions")
        
        # Merge config with defaults
        processing_config = {
            'enable_ai': APP_CONFIG['enable_ai'],
            'ai_confidence_threshold': APP_CONFIG['ai_confidence_threshold'],
            'ai_vendor_enabled': True,
            'ai_category_enabled': True,
            'ai_analysis_enabled': False,  # Expensive, disabled by default
            **config  # User overrides
        }
        
        # Create LLM client and cleaner
        llm_client = create_llm_client()
        cleaner = AIEnhancedProductionCleaner(
            df=df,
            config=processing_config,
            llm_client=llm_client
        )
        
        # Process the data
        cleaned_df, insights = cleaner.process_data()
        
        # Update global statistics
        app_stats['total_requests'] += 1
        app_stats['total_ai_calls'] += insights.get('ai_requests', 0)
        app_stats['total_cost'] += insights.get('ai_cost', 0.0)
        
        processing_time = time.time() - start_time
        
        # Prepare response
        response = {
            'cleaned_data': cleaned_df.to_dict('records'),
            'insights': {
                **insights,
                'processing_time': processing_time,
                'request_id': request_id
            },
            'status': 'success',
            'request_id': request_id
        }
        
        logger.info(f"[{request_id}] Processing completed successfully in {processing_time:.2f}s")
        logger.info(f"[{request_id}] AI requests: {insights.get('ai_requests', 0)}, Cost: ${insights.get('ai_cost', 0):.3f}")
        
        return jsonify(response)
        
    except Exception as e:
        processing_time = time.time() - start_time
        logger.error(f"[{request_id}] Processing failed after {processing_time:.2f}s: {str(e)}")
        
        return jsonify({
            'error': 'Internal processing error',
            'message': str(e),
            'request_id': request_id,
            'processing_time': processing_time
        }), 500

@app.route('/demo', methods=['POST'])
def demo_endpoint():
    """Demo endpoint with sample data for testing"""
    request_id = generate_request_id()
    
    # Sample messy financial data
    sample_data = {
        'merchant': [
            'PAYPAL*DIGITALOCEAN',
            'SQ *COFFEE SHOP NYC',
            'UBER EATS DEC15',
            'MONTHLY.RECURRING.CHG*NETFLIX',
            'POS PURCHASE - STARBUCKS #12345',
            'WM SUPERCENTER #1234',
            'TARGET DEBIT CRD PURCHASE'
        ],
        'amount': [50.00, 4.50, 23.75, 15.99, 6.85, 234.56, 89.12],
        'description': [
            'DigitalOcean hosting',
            'Coffee purchase',
            'Food delivery',
            'Netflix subscription',
            'Coffee purchase',
            'Grocery shopping',
            'Shopping'
        ],
        'memo': [
            'Monthly hosting',
            'Morning coffee',
            'Lunch delivery',
            'Entertainment',
            'Coffee break',
            'Weekly shopping',
            'Target run'
        ]
    }
    
    # Use the main processing endpoint
    request.json = {
        'data': sample_data,
        'config': {
            'enable_ai': True,
            'ai_vendor_enabled': True,
            'ai_category_enabled': True
        }
    }
    
    return process_financial_data()

@app.route('/config', methods=['GET'])
def get_configuration():
    """Get current configuration"""
    return jsonify({
        'max_file_size_mb': APP_CONFIG['max_file_size'] / (1024 * 1024),
        'enable_ai': APP_CONFIG['enable_ai'],
        'ai_confidence_threshold': APP_CONFIG['ai_confidence_threshold'],
        'version': app_stats['version'],
        'has_api_key': bool(APP_CONFIG['anthropic_api_key'])
    })

@app.errorhandler(413)
def request_entity_too_large(error):
    """Handle file too large errors"""
    return jsonify({
        'error': 'Request entity too large',
        'max_size_mb': APP_CONFIG['max_file_size'] / (1024 * 1024)
    }), 413

@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors"""
    return jsonify({
        'error': 'Endpoint not found',
        'available_endpoints': [
            'POST /process - Process financial data',
            'POST /demo - Demo with sample data',
            'GET /health - Health check',
            'GET /ready - Readiness check',
            'GET /stats - Usage statistics',
            'GET /config - Configuration'
        ]
    }), 404

@app.errorhandler(500)
def internal_error(error):
    """Handle internal server errors"""
    logger.error(f"Internal server error: {error}")
    return jsonify({
        'error': 'Internal server error',
        'message': 'An unexpected error occurred'
    }), 500

if __name__ == '__main__':
    # Configure Flask for production
    port = int(os.getenv('PORT', 8080))
    debug = os.getenv('FLASK_ENV') == 'development'
    
    # Set max content length
    app.config['MAX_CONTENT_LENGTH'] = APP_CONFIG['max_file_size']
    
    logger.info(f"Starting AI-Enhanced Financial Cleaner API v{app_stats['version']}")
    logger.info(f"AI Enabled: {APP_CONFIG['enable_ai']}")
    logger.info(f"Port: {port}")
    logger.info(f"Debug: {debug}")
    
    # Start the application
    app.run(
        host='0.0.0.0',
        port=port,
        debug=debug,
        threaded=True
    ) 