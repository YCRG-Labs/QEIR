"""
Enhanced FRED API Data Collector for QE Hypothesis Testing

This module extends the existing QEIR data collection capabilities with
hypothesis-specific data series mapping and collection methods for testing
three specific QE hypotheses:

1. Central Bank Reaction and Confidence Effects Testing
2. QE Impact on Private Investment Analysis  
3. International QE Effects and Currency Analysis

Author: QE Research Team
Date: 2025
Version: 1.0
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
import time
import warnings
from dataclasses import dataclass

try:
    from fredapi import Fred
    import requests
except ImportError as e:
    raise ImportError(f"Required dependencies not installed: {e}")

from ..config import QEIRConfig
from .data_structures import HypothesisData
from .api_error_handler import FREDAPIErrorHandler, RetryConfig, ComprehensiveLogger
from .data_quality_validator import AdvancedDataValidator, QualityMetrics, TemporalAlignment








class HypothesisDataCollector:
    """
    Enhanced data collector class that extends existing QEIR data collection capabilities
    with hypothesis-specific data series mapping and collection methods.
    """
    
    def __init__(self, fred_api_key: str, config: Optional[QEIRConfig] = None, 
                 retry_config: Optional[RetryConfig] = None):
        """
        Initialize the hypothesis data collector.
        
        Args:
            fred_api_key: FRED API key for data access
            config: QEIR configuration object
            retry_config: Configuration for API retry mechanisms
        """
        self.fred_api_key = fred_api_key
        self.config = config or QEIRConfig()
        
        # Initialize FRED API client with error handling
        if not fred_api_key or fred_api_key == "YOUR_FRED_API_KEY_HERE":
            raise ValueError("Valid FRED API key is required")
        
        # Initialize robust API error handler
        self.api_handler = FREDAPIErrorHandler(fred_api_key, retry_config)
        
        # Setup comprehensive logging
        self.comprehensive_logger = ComprehensiveLogger()
        self.logger = logging.getLogger(__name__)
        
        # Advanced data quality validator
        self.validator = AdvancedDataValidator()
        
        # Define hypothesis-specific series mappings
        self._setup_series_mappings()
    
    def _setup_series_mappings(self):
        """Setup FRED series mappings for each hypothesis"""
        
        # Hypothesis 1: Central Bank Reaction and Confidence Effects
        self.hypothesis1_series = {
            # Central Bank Reaction Strength (γ₁) proxies
            'fed_total_assets': 'WALCL',                    # Fed total assets
            'fed_treasury_holdings': 'WTREGEN',             # Fed Treasury holdings
            'fed_mbs_holdings': 'WSHOMCB',                  # Fed MBS holdings
            'monetary_base': 'BOGMBASE',                    # Monetary base
            'fomc_meeting_frequency': 'FEDFUNDS',           # Fed funds rate (proxy for policy activity)
            
            # Confidence Effects (λ₂) proxies
            'consumer_confidence': 'UMCSENT',               # University of Michigan Consumer Sentiment
            'business_confidence': 'BSCICP03USM665S',       # Business confidence indicator
            'financial_stress_index': 'STLFSI2',            # St. Louis Fed Financial Stress Index
            'vix_index': 'VIXCLS',                          # VIX volatility index
            'credit_spreads': 'BAMLH0A0HYM2',              # High yield credit spreads
            
            # Debt Service Burden
            'federal_interest_payments': 'A091RC1Q027SBEA', # Federal interest payments
            'gdp_nominal': 'GDP',                           # Nominal GDP
            'federal_debt_total': 'GFDEBTN',                # Total federal debt
            
            # Long-term Yields
            'treasury_10y': 'DGS10',                        # 10-year Treasury yield
            'treasury_30y': 'DGS30',                        # 30-year Treasury yield
            'tips_10y': 'DFII10',                           # 10-year TIPS yield
        }
        
        # Hypothesis 2: QE Impact on Private Investment
        self.hypothesis2_series = {
            # QE Intensity measures - FIXED SERIES IDs
            'fed_securities_held': 'WALCL',                 # Use Fed total assets as proxy
            'treasury_outstanding': 'GFDEBTN',              # Use total federal debt as proxy
            'mbs_outstanding': 'MBST',                      # MBS outstanding
            
            # Private Investment measures
            'private_fixed_investment': 'PNFI',             # Private nonresidential fixed investment
            'equipment_investment': 'PNFIC1',               # Equipment investment
            'structures_investment': 'PNFIC96',             # Structures investment
            'intellectual_property': 'Y033RC1Q027SBEA',     # Intellectual property investment
            
            # Market Distortion (μ₂) proxies
            'treasury_bid_ask_spread': 'TB3SMFFM',          # 3-month Treasury bill secondary market rate
            'corporate_bond_spreads': 'BAMLC0A0CM',         # Corporate bond spreads
            'mortgage_spreads': 'MORTGAGE30US',             # 30-year mortgage rate
            'liquidity_premium': 'T10Y2Y',                  # 10Y-2Y Treasury spread
            
            # Interest Rate Channel
            'federal_funds_rate': 'FEDFUNDS',               # Federal funds rate
            'prime_rate': 'DPRIME',                         # Prime loan rate
            'corporate_aaa_rate': 'DAAA',                   # Corporate AAA bond rate
            'mortgage_rate_30y': 'MORTGAGE30US',            # 30-year mortgage rate
        }
        
        # Hypothesis 3: International QE Effects and Currency
        self.hypothesis3_series = {
            # Foreign Bond Holdings - FIXED SERIES IDs
            'foreign_treasury_holdings': 'FDHBFIN',         # Foreign holdings of US Treasuries
            'china_treasury_holdings': 'FDHBFIN',           # Use total foreign holdings as proxy
            'japan_treasury_holdings': 'FDHBFIN',           # Use total foreign holdings as proxy
            'foreign_agency_holdings': 'FDHBATN',           # Foreign agency debt holdings
            
            # Exchange Rates
            'trade_weighted_dollar': 'DTWEXBGS',            # Trade weighted US dollar index
            'eur_usd_rate': 'DEXUSEU',                      # EUR/USD exchange rate
            'jpy_usd_rate': 'DEXJPUS',                      # JPY/USD exchange rate
            'cny_usd_rate': 'DEXCHUS',                      # CNY/USD exchange rate
            
            # Inflation Measures
            'cpi_all_items': 'CPIAUCSL',                    # Consumer Price Index
            'pce_price_index': 'PCEPI',                     # PCE Price Index
            'import_price_index': 'IR',                     # Import price index
            'breakeven_inflation_5y': 'T5YIE',              # 5-year breakeven inflation
            'breakeven_inflation_10y': 'T10YIE',            # 10-year breakeven inflation
            
            # Capital Flows - FIXED SERIES IDs
            'portfolio_flows': 'TRESEGUSM052N',             # Use official reserve assets as proxy
            'direct_investment_flows': 'TRESEGUSM052N',     # Use official reserve assets as proxy
            'official_reserve_assets': 'TRESEGUSM052N',     # Official reserve assets
        }
        
        # Alternative series for fallback
        self.alternative_series = {
            # Hypothesis 1 alternatives
            'consumer_confidence': ['UMCSENT', 'CSCICP03USM665S', 'ICSENT'],
            'business_confidence': ['BSCICP03USM665S', 'USSLIND', 'NAPM'],
            'financial_stress_index': ['STLFSI2', 'NFCI', 'ANFCI'],
            
            # Hypothesis 2 alternatives
            'private_fixed_investment': ['PNFI', 'GPDI', 'PINVEST'],
            'equipment_investment': ['PNFIC1', 'PNFIC96', 'EQUIPINV'],
            
            # Hypothesis 3 alternatives
            'trade_weighted_dollar': ['DTWEXBGS', 'DTWEXM', 'DTWEXAFEGS'],
            'import_price_index': ['IR', 'IMPGS', 'IMPCH'],
        }
    
    def collect_hypothesis1_data(self, start_date: str, end_date: str) -> Dict[str, pd.Series]:
        """
        Collect data for Hypothesis 1: Central Bank Reaction and Confidence Effects Testing
        
        Args:
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            
        Returns:
            Dictionary of pandas Series with hypothesis 1 data
        """
        self.logger.info("Collecting Hypothesis 1 data: Central Bank Reaction and Confidence Effects")
        
        data = {}
        failed_series = []
        
        for series_name, fred_id in self.hypothesis1_series.items():
            # Get fallback series if available
            fallback_ids = self.alternative_series.get(series_name, [])
            
            # Use robust API handler with fallback
            result = self.api_handler.get_series_with_fallback(
                fred_id, start_date, end_date, fallback_ids
            )
            
            # Log the result
            self.comprehensive_logger.log_api_call(fred_id, result)
            
            if result.success and result.data is not None and not result.data.empty:
                data[series_name] = result.data
                self.logger.info(f"Downloaded {series_name}: {len(result.data)} observations")
            else:
                failed_series.append(series_name)
                self.logger.error(f"All attempts failed for {series_name}: {result.error_message}")
        
        if failed_series:
            self.logger.warning(f"Failed to collect {len(failed_series)} series: {failed_series}")
        
        return data
    
    def collect_hypothesis2_data(self, start_date: str, end_date: str) -> Dict[str, pd.Series]:
        """
        Collect data for Hypothesis 2: QE Impact on Private Investment Analysis
        
        Args:
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            
        Returns:
            Dictionary of pandas Series with hypothesis 2 data
        """
        self.logger.info("Collecting Hypothesis 2 data: QE Impact on Private Investment")
        
        data = {}
        failed_series = []
        
        for series_name, fred_id in self.hypothesis2_series.items():
            # Get fallback series if available
            fallback_ids = self.alternative_series.get(series_name, [])
            
            # Use robust API handler with fallback
            result = self.api_handler.get_series_with_fallback(
                fred_id, start_date, end_date, fallback_ids
            )
            
            # Log the result
            self.comprehensive_logger.log_api_call(fred_id, result)
            
            if result.success and result.data is not None and not result.data.empty:
                data[series_name] = result.data
                self.logger.info(f"Downloaded {series_name}: {len(result.data)} observations")
            else:
                failed_series.append(series_name)
                self.logger.error(f"All attempts failed for {series_name}: {result.error_message}")
        
        if failed_series:
            self.logger.warning(f"Failed to collect {len(failed_series)} series: {failed_series}")
        
        return data
    
    def collect_hypothesis3_data(self, start_date: str, end_date: str) -> Dict[str, pd.Series]:
        """
        Collect data for Hypothesis 3: International QE Effects and Currency Analysis
        
        Args:
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            
        Returns:
            Dictionary of pandas Series with hypothesis 3 data
        """
        self.logger.info("Collecting Hypothesis 3 data: International QE Effects and Currency")
        
        data = {}
        failed_series = []
        
        for series_name, fred_id in self.hypothesis3_series.items():
            # Get fallback series if available
            fallback_ids = self.alternative_series.get(series_name, [])
            
            # Use robust API handler with fallback
            result = self.api_handler.get_series_with_fallback(
                fred_id, start_date, end_date, fallback_ids
            )
            
            # Log the result
            self.comprehensive_logger.log_api_call(fred_id, result)
            
            if result.success and result.data is not None and not result.data.empty:
                data[series_name] = result.data
                self.logger.info(f"Downloaded {series_name}: {len(result.data)} observations")
            else:
                failed_series.append(series_name)
                self.logger.error(f"All attempts failed for {series_name}: {result.error_message}")
        
        if failed_series:
            self.logger.warning(f"Failed to collect {len(failed_series)} series: {failed_series}")
        
        return data
    
    def validate_data_quality(self, data: Dict[str, pd.Series]) -> Dict[str, Any]:
        """
        Validate data quality using advanced validation system
        
        Args:
            data: Dictionary of pandas Series to validate
            
        Returns:
            Dictionary containing comprehensive validation results and quality metrics
        """
        self.logger.info("Validating data quality with advanced validation system...")
        
        # Use the advanced validator to generate comprehensive quality report
        quality_report = self.validator.generate_quality_report(data)
        
        # Log summary information
        summary = quality_report['summary']
        self.logger.info(f"Data quality validation complete. Overall quality score: {summary['overall_quality_score']:.1f}")
        self.logger.info(f"High quality series: {summary['high_quality_series']}/{summary['total_series']}")
        
        if summary['critical_issues'] > 0:
            self.logger.error(f"Critical issues found: {summary['critical_issues']}")
        if summary['error_issues'] > 0:
            self.logger.warning(f"Error issues found: {summary['error_issues']}")
        if summary['warning_issues'] > 0:
            self.logger.info(f"Warning issues found: {summary['warning_issues']}")
        
        return quality_report
    
    def collect_all_hypothesis_data(self, start_date: str, end_date: str) -> HypothesisData:
        """
        Collect data for all three hypotheses and return structured HypothesisData object
        
        Args:
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            
        Returns:
            HypothesisData object containing all collected data
        """
        self.logger.info("Collecting data for all hypotheses...")
        
        # Collect data for each hypothesis
        h1_data = self.collect_hypothesis1_data(start_date, end_date)
        h2_data = self.collect_hypothesis2_data(start_date, end_date)
        h3_data = self.collect_hypothesis3_data(start_date, end_date)
        
        # Create structured HypothesisData object
        hypothesis_data = HypothesisData()
        
        # Map Hypothesis 1 data with fallbacks
        hypothesis_data.central_bank_reaction = (
            h1_data.get('fed_total_assets') if h1_data.get('fed_total_assets') is not None else
            h1_data.get('monetary_base') if h1_data.get('monetary_base') is not None else
            h1_data.get('fed_treasury_holdings')
        )
        
        hypothesis_data.confidence_effects = (
            h1_data.get('consumer_confidence') if h1_data.get('consumer_confidence') is not None else
            h1_data.get('business_confidence') if h1_data.get('business_confidence') is not None else
            h1_data.get('vix_index')
        )
        
        # Create debt service burden ratio if possible
        interest_payments = h1_data.get('federal_interest_payments')
        gdp = h1_data.get('gdp_nominal')
        
        if interest_payments is not None and gdp is not None:
            # Align the series and calculate ratio
            common_dates = interest_payments.index.intersection(gdp.index)
            if len(common_dates) > 0:
                aligned_payments = interest_payments.loc[common_dates]
                aligned_gdp = gdp.loc[common_dates]
                hypothesis_data.debt_service_burden = (aligned_payments / aligned_gdp) * 100
            else:
                hypothesis_data.debt_service_burden = interest_payments
        else:
            hypothesis_data.debt_service_burden = interest_payments if interest_payments is not None else h1_data.get('federal_debt_total')
        
        hypothesis_data.long_term_yields = (
            h1_data.get('treasury_10y') if h1_data.get('treasury_10y') is not None else
            h1_data.get('treasury_30y') if h1_data.get('treasury_30y') is not None else
            h1_data.get('tips_10y')
        )
        
        # Map Hypothesis 2 data with fallbacks
        hypothesis_data.qe_intensity = (
            h2_data.get('fed_securities_held') if h2_data.get('fed_securities_held') is not None else
            h2_data.get('fed_total_assets') if h2_data.get('fed_total_assets') is not None else
            h2_data.get('monetary_base')
        )
        
        hypothesis_data.private_investment = (
            h2_data.get('private_fixed_investment') if h2_data.get('private_fixed_investment') is not None else
            h2_data.get('equipment_investment') if h2_data.get('equipment_investment') is not None else
            h2_data.get('structures_investment')
        )
        
        hypothesis_data.market_distortions = (
            h2_data.get('corporate_bond_spreads') if h2_data.get('corporate_bond_spreads') is not None else
            h2_data.get('mortgage_spreads') if h2_data.get('mortgage_spreads') is not None else
            h2_data.get('liquidity_premium')
        )
        
        hypothesis_data.interest_rate_channel = (
            h2_data.get('federal_funds_rate') if h2_data.get('federal_funds_rate') is not None else
            h2_data.get('prime_rate') if h2_data.get('prime_rate') is not None else
            h2_data.get('corporate_aaa_rate')
        )
        
        # Map Hypothesis 3 data with fallbacks
        hypothesis_data.foreign_bond_holdings = (
            h3_data.get('foreign_treasury_holdings') if h3_data.get('foreign_treasury_holdings') is not None else
            h3_data.get('china_treasury_holdings') if h3_data.get('china_treasury_holdings') is not None else
            h3_data.get('foreign_agency_holdings')
        )
        
        hypothesis_data.exchange_rate = (
            h3_data.get('trade_weighted_dollar') if h3_data.get('trade_weighted_dollar') is not None else
            h3_data.get('eur_usd_rate') if h3_data.get('eur_usd_rate') is not None else
            h3_data.get('jpy_usd_rate')
        )
        
        hypothesis_data.inflation_measures = (
            h3_data.get('cpi_all_items') if h3_data.get('cpi_all_items') is not None else
            h3_data.get('pce_price_index') if h3_data.get('pce_price_index') is not None else
            h3_data.get('import_price_index')
        )
        
        hypothesis_data.capital_flows = (
            h3_data.get('portfolio_flows') if h3_data.get('portfolio_flows') is not None else
            h3_data.get('direct_investment_flows') if h3_data.get('direct_investment_flows') is not None else
            h3_data.get('official_reserve_assets')
        )
        
        # Set common metadata
        all_data = {**h1_data, **h2_data, **h3_data}
        if all_data:
            # Get date range from all series
            all_dates = []
            for series in all_data.values():
                if hasattr(series, 'index'):
                    all_dates.extend(series.index.tolist())
            
            if all_dates:
                hypothesis_data.dates = pd.DatetimeIndex(sorted(set(all_dates)))
        
        hypothesis_data.metadata = {
            'collection_timestamp': datetime.now().isoformat(),
            'start_date': start_date,
            'end_date': end_date,
            'hypothesis1_series_count': len(h1_data),
            'hypothesis2_series_count': len(h2_data),
            'hypothesis3_series_count': len(h3_data),
            'total_series_count': len(all_data)
        }
        
        self.logger.info(f"Collected data for all hypotheses. Total series: {len(all_data)}")
        
        return hypothesis_data
    
    def collect_and_process_all_data(self, start_date: str, end_date: str, 
                                   processing_config: Optional[Dict[str, Any]] = None) -> Tuple[HypothesisData, Dict[str, Any]]:
        """
        Collect and process data for all hypotheses with complete pipeline
        
        Args:
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            processing_config: Optional processing configuration dictionary
            
        Returns:
            Tuple of (processed HypothesisData, processing report)
        """
        self.logger.info("Collecting and processing data for all hypotheses...")
        
        # Import here to avoid circular import
        from .data_processor import DataProcessor, ProcessingConfig
        
        # Collect raw data
        raw_hypothesis_data = self.collect_all_hypothesis_data(start_date, end_date)
        
        # Set up data processor with custom config if provided
        if processing_config:
            config = ProcessingConfig(**processing_config)
            data_processor = DataProcessor(config)
        else:
            data_processor = DataProcessor()
        
        # Process the data
        processed_data, processing_report = data_processor.process_hypothesis_data(raw_hypothesis_data)
        
        self.logger.info("Data collection and processing completed")
        
        return processed_data, processing_report
    
    def get_api_statistics(self) -> Dict[str, Any]:
        """
        Get comprehensive API call statistics and performance metrics
        
        Returns:
            Dictionary with API statistics and error breakdown
        """
        stats = self.api_handler.get_call_statistics()
        
        # Log summary statistics
        self.comprehensive_logger.log_statistics_summary(stats)
        
        return stats
    
    def reset_api_statistics(self):
        """Reset API call statistics"""
        self.api_handler.reset_statistics()
        self.logger.info("API statistics reset")
    
    def handle_data_gaps(self, data: Dict[str, pd.Series], 
                        method: str = 'interpolate') -> Dict[str, pd.Series]:
        """
        Handle gaps in time series data using various methods
        
        Args:
            data: Dictionary of time series with potential gaps
            method: Gap handling method ('interpolate', 'forward_fill', 'backward_fill', 'drop')
            
        Returns:
            Dictionary of time series with gaps handled
        """
        self.logger.info(f"Handling data gaps using method: {method}")
        
        processed_data = {}
        
        for series_name, series in data.items():
            if series is None or series.empty:
                processed_data[series_name] = series
                continue
            
            original_missing = series.isna().sum()
            
            if method == 'interpolate':
                # Use linear interpolation for numeric data
                processed_series = series.interpolate(method='linear', limit_direction='both')
            elif method == 'forward_fill':
                processed_series = series.fillna(method='ffill')
            elif method == 'backward_fill':
                processed_series = series.fillna(method='bfill')
            elif method == 'drop':
                processed_series = series.dropna()
            else:
                self.logger.warning(f"Unknown gap handling method: {method}, using interpolation")
                processed_series = series.interpolate(method='linear', limit_direction='both')
            
            final_missing = processed_series.isna().sum()
            filled_count = original_missing - final_missing
            
            if filled_count > 0:
                self.logger.info(f"Filled {filled_count} gaps in {series_name}")
            
            processed_data[series_name] = processed_series
        
        return processed_data
    
    def align_temporal_data(self, data: Dict[str, pd.Series], 
                           target_frequency: Optional[str] = None) -> Dict[str, pd.Series]:
        """
        Align time series data to common frequency and date range
        
        Args:
            data: Dictionary of time series to align
            target_frequency: Target frequency ('D', 'M', 'Q', etc.). If None, uses most common frequency
            
        Returns:
            Dictionary of aligned time series
        """
        self.logger.info("Aligning temporal data...")
        
        # Analyze temporal alignment
        alignment_analysis = self.validator.analyze_temporal_alignment(data)
        
        if not target_frequency:
            target_frequency = alignment_analysis.frequency
            if target_frequency == "unknown":
                target_frequency = 'M'  # Default to monthly
                self.logger.warning("Could not determine common frequency, defaulting to monthly")
        
        # Get common date range
        common_start = alignment_analysis.common_start_date
        common_end = alignment_analysis.common_end_date
        
        self.logger.info(f"Aligning to frequency: {target_frequency}, period: {common_start} to {common_end}")
        
        aligned_data = {}
        
        for series_name, series in data.items():
            if series is None or series.empty:
                aligned_data[series_name] = series
                continue
            
            try:
                # Filter to common period
                filtered_series = series[(series.index >= common_start) & (series.index <= common_end)]
                
                # Resample to target frequency
                if target_frequency in ['D', 'B']:  # Daily frequencies
                    resampled_series = filtered_series.resample(target_frequency).mean()
                elif target_frequency in ['M', 'MS']:  # Monthly frequencies
                    resampled_series = filtered_series.resample(target_frequency).mean()
                elif target_frequency in ['Q', 'QS']:  # Quarterly frequencies
                    resampled_series = filtered_series.resample(target_frequency).mean()
                elif target_frequency in ['A', 'AS']:  # Annual frequencies
                    resampled_series = filtered_series.resample(target_frequency).mean()
                else:
                    # For other frequencies, try direct resampling
                    resampled_series = filtered_series.resample(target_frequency).mean()
                
                aligned_data[series_name] = resampled_series
                self.logger.info(f"Aligned {series_name}: {len(filtered_series)} -> {len(resampled_series)} observations")
                
            except Exception as e:
                self.logger.error(f"Failed to align {series_name}: {e}")
                aligned_data[series_name] = series  # Keep original if alignment fails
        
        return aligned_data
    
    def flag_quality_issues(self, data: Dict[str, pd.Series]) -> Dict[str, Dict[str, Any]]:
        """
        Flag data quality issues and provide detailed flagging information
        
        Args:
            data: Dictionary of time series to analyze
            
        Returns:
            Dictionary with quality flags and issue details for each series
        """
        self.logger.info("Flagging data quality issues...")
        
        quality_flags = {}
        
        for series_name, series in data.items():
            if series is None or series.empty:
                quality_flags[series_name] = {
                    'overall_flag': 'CRITICAL',
                    'issues': ['Series is empty or None'],
                    'usable_for_analysis': False,
                    'recommendations': ['Check data source and collection process']
                }
                continue
            
            # Get comprehensive quality metrics
            metrics = self.validator.validate_series_comprehensive(series, series_name)
            
            # Determine overall flag based on quality score and issues
            if metrics.quality_score >= 80:
                overall_flag = 'GOOD'
            elif metrics.quality_score >= 60:
                overall_flag = 'ACCEPTABLE'
            elif metrics.quality_score >= 40:
                overall_flag = 'POOR'
            else:
                overall_flag = 'CRITICAL'
            
            # Check for critical issues that make data unusable
            critical_issues = [issue for issue in metrics.issues if issue.severity.value == 'critical']
            error_issues = [issue for issue in metrics.issues if issue.severity.value == 'error']
            
            usable_for_analysis = (
                len(critical_issues) == 0 and 
                len(error_issues) <= 2 and 
                metrics.quality_score >= 30
            )
            
            # Collect issue messages
            issue_messages = [issue.message for issue in metrics.issues]
            
            # Collect recommendations
            recommendations = [issue.recommendation for issue in metrics.issues if issue.recommendation]
            
            quality_flags[series_name] = {
                'overall_flag': overall_flag,
                'quality_score': metrics.quality_score,
                'usable_for_analysis': usable_for_analysis,
                'issues': issue_messages,
                'recommendations': recommendations,
                'missing_percentage': metrics.missing_percentage,
                'outlier_percentage': metrics.outlier_percentage,
                'completeness_score': metrics.completeness_score,
                'consistency_score': metrics.consistency_score,
                'accuracy_score': metrics.accuracy_score
            }
        
        # Log summary
        good_series = sum(1 for flags in quality_flags.values() if flags['overall_flag'] == 'GOOD')
        acceptable_series = sum(1 for flags in quality_flags.values() if flags['overall_flag'] == 'ACCEPTABLE')
        poor_series = sum(1 for flags in quality_flags.values() if flags['overall_flag'] == 'POOR')
        critical_series = sum(1 for flags in quality_flags.values() if flags['overall_flag'] == 'CRITICAL')
        
        self.logger.info(f"Quality flagging complete: {good_series} good, {acceptable_series} acceptable, "
                        f"{poor_series} poor, {critical_series} critical")
        
        return quality_flags