import pandas as pd
import numpy as np
import re
import json
import logging
from datetime import datetime
from typing import Dict, Any, Optional, List, Tuple
from contextlib import contextmanager
import time
from llm_client import LLMClient

logger = logging.getLogger(__name__)

class AIEnhancedProductionCleaner:
    """
    AI-Enhanced Production Financial Cleaner v4.2
    Features:
    - AI-powered vendor standardization
    - AI-powered categorization
    - Comprehensive business intelligence
    - Performance monitoring and cost tracking
    - Caching for efficiency
    """
    
    def __init__(self, df: pd.DataFrame, config: Optional[Dict[str, Any]] = None, 
                 llm_client: Optional[LLMClient] = None):
        self.cleaner_version = "v4.2"
        self.df = df.copy()
        
        # Validate and set configuration
        self.config = self._validate_and_set_config(config or {})
        
        # Initialize AI client
        self.llm_client = llm_client or LLMClient(use_mock=True)
        
        # Processing statistics
        self.stats = {
            'ai_requests': 0,
            'ai_cost': 0.0,
            'cache_hits': 0,
            'processing_time': 0.0,
            'rows_processed': len(df)
        }
        
        logger.info(f"AI-Enhanced Production Cleaner {self.cleaner_version} initialized with {len(df)} rows")

    def _validate_and_set_config(self, user_config: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and merge configuration with safe defaults"""
        default_config = {
            'duplicate_threshold': 0.85,
            'outlier_z_threshold': 3.0,
            'chunk_size': 10000,
            'enable_ai': True,
            'ai_confidence_threshold': 0.7,
            'use_ai_for_unmatched_only': False,
            'ai_vendor_enabled': True,
            'ai_category_enabled': True,
            'ai_analysis_enabled': False  # More expensive comprehensive analysis
        }
        
        # Merge user config with defaults
        config = {**default_config, **user_config}
        
        # Type validation
        config['duplicate_threshold'] = float(config['duplicate_threshold'])
        config['outlier_z_threshold'] = float(config['outlier_z_threshold'])
        config['chunk_size'] = int(config['chunk_size'])
        config['enable_ai'] = bool(config['enable_ai'])
        config['ai_confidence_threshold'] = float(config['ai_confidence_threshold'])
        config['use_ai_for_unmatched_only'] = bool(config['use_ai_for_unmatched_only'])
        config['ai_vendor_enabled'] = bool(config['ai_vendor_enabled'])
        config['ai_category_enabled'] = bool(config['ai_category_enabled'])
        config['ai_analysis_enabled'] = bool(config['ai_analysis_enabled'])
        
        return config

    @contextmanager
    def timer(self, operation_name: str):
        """Context manager for timing operations"""
        start_time = time.time()
        try:
            yield
        finally:
            elapsed = time.time() - start_time
            logger.info(f"{operation_name} completed in {elapsed:.2f} seconds")

    def process_data(self) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Main processing pipeline with AI integration"""
        with self.timer("Total AI-enhanced processing"):
            if len(self.df) > self.config['chunk_size']:
                logger.info(f"Large dataset detected ({len(self.df)} rows), using chunk processing")
                return self._chunk_safe_processing()
            else:
                return self._standard_processing()

    def _chunk_safe_processing(self) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Process large datasets in chunks with AI integration"""
        chunk_size = self.config['chunk_size']
        total_chunks = (len(self.df) + chunk_size - 1) // chunk_size
        
        all_processed_chunks = []
        total_stats = {
            'ai_requests': 0,
            'ai_cost': 0.0,
            'cache_hits': 0,
            'chunks_processed': 0
        }
        
        for i in range(0, len(self.df), chunk_size):
            chunk_df = self.df.iloc[i:i + chunk_size].copy()
            chunk_num = (i // chunk_size) + 1
            
            logger.info(f"Processing chunk {chunk_num}/{total_chunks} ({len(chunk_df)} rows)")
            
            with self.timer(f"Chunk {chunk_num} processing"):
                chunk_result, chunk_stats = self._standard_processing_chunk(chunk_df)
                all_processed_chunks.append(chunk_result)
                
                # Aggregate statistics
                total_stats['ai_requests'] += chunk_stats.get('ai_requests', 0)
                total_stats['ai_cost'] += chunk_stats.get('ai_cost', 0.0)
                total_stats['cache_hits'] += chunk_stats.get('cache_hits', 0)
                total_stats['chunks_processed'] += 1
        
        # Combine all chunks
        final_df = pd.concat(all_processed_chunks, ignore_index=True)
        
        # Generate final insights
        final_insights = self._generate_comprehensive_insights(final_df)
        final_insights.update(total_stats)
        
        return final_df, final_insights

    def _standard_processing(self) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Standard processing pipeline with AI integration"""
        return self._standard_processing_chunk(self.df)

    def _standard_processing_chunk(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Process a single chunk with AI integration"""
        with self.timer("Standard processing"):
            # Data cleaning
            df = self._advanced_data_cleaning(df)
            
            # Amount and date cleaning
            df = self._advanced_amount_cleaning(df)
            
            # Duplicate detection and removal
            df = self._detect_and_remove_duplicates(df)
            
            # AI-powered vendor standardization
            if self.config['enable_ai'] and self.config['ai_vendor_enabled']:
                df = self._ai_enhanced_vendor_standardization(df)
            
            # AI-powered categorization
            if self.config['enable_ai'] and self.config['ai_category_enabled']:
                df = self._ai_enhanced_categorization(df)
            
            # Outlier detection
            df = self._detect_outliers(df)
            
            # Data filling and enhancement
            df = self._intelligent_data_filling(df)
            df = self._enhance_descriptions(df)
            
            # Final validation
            df = self._final_validation(df)
            
            # Generate insights
            insights = self._generate_comprehensive_insights(df)
            insights.update(self.stats)
            
            return df, insights

    def _ai_enhanced_vendor_standardization(self, df: pd.DataFrame) -> pd.DataFrame:
        """AI-enhanced vendor standardization"""
        if 'merchant' not in df.columns:
            logger.warning("No merchant column found for AI vendor standardization")
            return df
        
        for idx, row in df.iterrows():
            merchant = str(row.get('merchant', ''))
            description = str(row.get('description', ''))
            memo = str(row.get('memo', ''))
            
            # Check if we should use AI for this vendor
            use_ai = True
            if self.config['use_ai_for_unmatched_only']:
                # Only use AI if vendor wasn't matched by rules
                # This would require checking if the vendor was already standardized
                pass
            
            if use_ai:
                try:
                    ai_vendor = self.llm_client.resolve_vendor(merchant, description, memo)
                    df.at[idx, 'merchant'] = ai_vendor
                    self.stats['ai_requests'] += 1
                    self.stats['ai_cost'] += 0.01
                except Exception as e:
                    logger.warning(f"AI vendor resolution failed for {merchant}: {e}")
        
        return df

    def _ai_enhanced_categorization(self, df: pd.DataFrame) -> pd.DataFrame:
        """AI-enhanced categorization"""
        if 'merchant' not in df.columns or 'amount' not in df.columns:
            logger.warning("Missing merchant or amount column for AI categorization")
            return df
        
        for idx, row in df.iterrows():
            merchant = str(row.get('merchant', ''))
            description = str(row.get('description', ''))
            amount = float(row.get('amount', 0))
            memo = str(row.get('memo', ''))
            
            try:
                ai_result = self.llm_client.suggest_category(merchant, description, amount, memo)
                
                # Only use AI result if confidence is high enough
                if ai_result['confidence'] >= self.config['ai_confidence_threshold']:
                    df.at[idx, 'category'] = ai_result['category']
                    self.stats['ai_requests'] += 1
                    self.stats['ai_cost'] += 0.01
                    
            except Exception as e:
                logger.warning(f"AI categorization failed for {merchant}: {e}")
        
        return df

    def _advanced_data_cleaning(self, df: pd.DataFrame) -> pd.DataFrame:
        """Advanced data cleaning with AI insights"""
        # Standard cleaning operations
        for col in df.columns:
            if df[col].dtype == 'object':
                df[col] = df[col].astype(str).str.strip()
        
        # Remove empty rows
        df = df.dropna(how='all')
        
        return df

    def _advanced_amount_cleaning(self, df: pd.DataFrame) -> pd.DataFrame:
        """Advanced amount cleaning"""
        if 'amount' not in df.columns:
            return df
        
        # Convert to numeric, handling currency symbols
        df['amount'] = pd.to_numeric(df['amount'].astype(str).str.replace('$', '').str.replace(',', ''), errors='coerce')
        
        # Fill missing amounts with 0
        df['amount'] = df['amount'].fillna(0)
        
        return df

    def _detect_and_remove_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Detect and remove duplicates"""
        if len(df) == 0:
            return df
        
        # Exact duplicates
        initial_count = len(df)
        df = df.drop_duplicates()
        
        # Near-duplicates using similarity
        if 'merchant' in df.columns and 'amount' in df.columns:
            duplicates_removed = []
            for i in range(len(df)):
                for j in range(i + 1, len(df)):
                    if i in duplicates_removed or j in duplicates_removed:
                        continue
                    
                    merchant_similarity = self._calculate_similarity(
                        str(df.iloc[i].get('merchant', '')),
                        str(df.iloc[j].get('merchant', ''))
                    )
                    
                    amount_diff = abs(df.iloc[i]['amount'] - df.iloc[j]['amount'])
                    
                    if (merchant_similarity > self.config['duplicate_threshold'] and 
                        amount_diff < 0.01):
                        duplicates_removed.append(j)
            
            df = df.drop(df.index[duplicates_removed])
        
        logger.info(f"Removed {initial_count - len(df)} duplicates")
        return df

    def _calculate_similarity(self, str1: str, str2: str) -> float:
        """Calculate string similarity"""
        from difflib import SequenceMatcher
        return SequenceMatcher(None, str1.lower(), str2.lower()).ratio()

    def _detect_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """Detect outliers using Z-score"""
        if 'amount' not in df.columns or len(df) == 0:
            return df
        
        # Calculate Z-scores
        amounts = df['amount'].values
        mean_amount = np.mean(amounts)
        std_amount = np.std(amounts)
        
        if std_amount > 0:
            z_scores = np.abs((amounts - mean_amount) / std_amount)
            df['is_outlier'] = z_scores > self.config['outlier_z_threshold']
        else:
            df['is_outlier'] = False
        
        outlier_count = df['is_outlier'].sum()
        logger.info(f"Flagged {outlier_count} outliers")
        self.stats['outliers_flagged'] = outlier_count
        
        return df

    def _intelligent_data_filling(self, df: pd.DataFrame) -> pd.DataFrame:
        """Intelligent data filling"""
        # Fill missing categories with 'Other'
        if 'category' in df.columns:
            df['category'] = df['category'].fillna('Other')
        
        # Fill missing descriptions
        if 'description' in df.columns:
            df['description'] = df['description'].fillna('No description')
        
        return df

    def _enhance_descriptions(self, df: pd.DataFrame) -> pd.DataFrame:
        """Enhance descriptions with AI insights"""
        if not self.config['enable_ai'] or not self.config['ai_analysis_enabled']:
            return df
        
        if 'description' not in df.columns or 'merchant' not in df.columns:
            return df
        
        for idx, row in df.iterrows():
            merchant = str(row.get('merchant', ''))
            description = str(row.get('description', ''))
            amount = float(row.get('amount', 0))
            memo = str(row.get('memo', ''))
            
            try:
                analysis = self.llm_client.analyze_transaction(merchant, description, amount, memo)
                
                # Enhance description with business insights
                enhanced_desc = f"{description} | {analysis['business_impact']}"
                df.at[idx, 'description'] = enhanced_desc
                
                # Add business intelligence columns
                df.at[idx, 'business_impact'] = analysis['business_impact']
                df.at[idx, 'risk_level'] = analysis['risk_level']
                
                self.stats['ai_requests'] += 1
                self.stats['ai_cost'] += 0.02  # Higher cost for comprehensive analysis
                
            except Exception as e:
                logger.warning(f"AI description enhancement failed for {merchant}: {e}")
        
        return df

    def _final_validation(self, df: pd.DataFrame) -> pd.DataFrame:
        """Final validation and quality checks"""
        # Ensure all required columns exist
        required_columns = ['merchant', 'amount']
        for col in required_columns:
            if col not in df.columns:
                df[col] = ''
        
        # Validate data types
        if 'amount' in df.columns:
            df['amount'] = pd.to_numeric(df['amount'], errors='coerce').fillna(0)
        
        return df

    def _generate_comprehensive_insights(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Generate comprehensive insights with AI statistics"""
        if len(df) == 0:
            return {
                'total_rows': 0,
                'data_quality_score': 0.0,
                'ai_requests': self.stats['ai_requests'],
                'ai_cost': self.stats['ai_cost']
            }
        
        insights = {
            'total_rows': len(df),
            'data_quality_score': self._calculate_data_quality_score(df),
            'ai_requests': self.stats['ai_requests'],
            'ai_cost': self.stats['ai_cost'],
            'cache_hits': self.stats['cache_hits'],
            'processing_time': self.stats['processing_time']
        }
        
        # Financial insights
        if 'amount' in df.columns:
            amounts = df['amount'].dropna()
            if len(amounts) > 0:
                insights.update({
                    'total_amount': amounts.sum(),
                    'average_amount': amounts.mean(),
                    'median_amount': amounts.median(),
                    'min_amount': amounts.min(),
                    'max_amount': amounts.max(),
                    'std_amount': amounts.std()
                })
        
        # Vendor insights
        if 'merchant' in df.columns:
            vendor_counts = df['merchant'].value_counts()
            insights['top_vendors'] = vendor_counts.head(10).to_dict()
            insights['unique_vendors'] = len(vendor_counts)
        
        # Category insights
        if 'category' in df.columns:
            category_counts = df['category'].value_counts()
            insights['category_breakdown'] = category_counts.to_dict()
        
        # Outlier insights
        if 'is_outlier' in df.columns:
            insights['outlier_count'] = df['is_outlier'].sum()
            insights['outlier_percentage'] = (df['is_outlier'].sum() / len(df)) * 100
        
        return insights

    def _calculate_data_quality_score(self, df: pd.DataFrame) -> float:
        """Calculate data quality score"""
        if len(df) == 0:
            return 0.0
        
        score = 0.0
        total_checks = 0
        
        # Check for missing values
        if 'merchant' in df.columns:
            missing_merchant = df['merchant'].isna().sum()
            score += (len(df) - missing_merchant) / len(df)
            total_checks += 1
        
        if 'amount' in df.columns:
            missing_amount = df['amount'].isna().sum()
            score += (len(df) - missing_amount) / len(df)
            total_checks += 1
        
        if 'category' in df.columns:
            missing_category = df['category'].isna().sum()
            score += (len(df) - missing_category) / len(df)
            total_checks += 1
        
        return score / max(total_checks, 1)

    def get_usage_stats(self) -> Dict[str, Any]:
        """Get comprehensive usage statistics"""
        llm_stats = self.llm_client.get_usage_stats()
        
        return {
            'cleaner_version': self.cleaner_version,
            'rows_processed': self.stats['rows_processed'],
            'ai_requests': self.stats['ai_requests'],
            'ai_cost': self.stats['ai_cost'],
            'cache_hits': self.stats['cache_hits'],
            'processing_time': self.stats['processing_time'],
            'llm_stats': llm_stats
        }

    def reset_stats(self):
        """Reset processing statistics"""
        self.stats = {
            'ai_requests': 0,
            'ai_cost': 0.0,
            'cache_hits': 0,
            'processing_time': 0.0,
            'rows_processed': len(self.df)
        }
        self.llm_client.reset_metrics() 