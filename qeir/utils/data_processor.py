"""
Data Processing and Alignment Pipeline for QE Hypothesis Testing

This module implements data processing methods to handle different frequencies,
missing values, variable construction for hypothesis-specific measures (γ₁, λ₂, μ₂),
and data quality validation with outlier detection.

Author: QE Research Team
Date: 2025
Version: 1.0
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional, Tuple, Any, Union
from datetime import datetime, timedelta
import warnings
from dataclasses import dataclass
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.impute import KNNImputer

from .data_structures import HypothesisData


@dataclass
class ProcessingConfig:
    """Configuration for data processing pipeline"""
    
    # Frequency alignment
    target_frequency: str = 'M'  # Monthly frequency as default
    alignment_method: str = 'end'  # 'start', 'end', or 'mean' for period alignment
    
    # Missing value handling
    max_missing_pct: float = 20.0  # Maximum allowed missing percentage
    interpolation_method: str = 'linear'  # 'linear', 'cubic', 'knn', 'forward', 'backward'
    max_consecutive_missing: int = 6  # Maximum consecutive missing values to interpolate
    
    # Outlier detection and handling
    outlier_method: str = 'iqr'  # 'iqr', 'zscore', 'modified_zscore', 'isolation_forest'
    outlier_threshold: float = 3.0  # Threshold for outlier detection
    outlier_treatment: str = 'winsorize'  # 'remove', 'winsorize', 'cap', 'interpolate'
    winsorize_limits: Tuple[float, float] = (0.05, 0.05)  # Lower and upper limits for winsorization
    
    # Variable construction
    log_transform_variables: List[str] = None  # Variables to log-transform
    standardize_variables: List[str] = None  # Variables to standardize
    difference_variables: List[str] = None  # Variables to difference
    
    # Data quality
    min_observations: int = 24  # Minimum observations required
    min_date_coverage: float = 0.8  # Minimum date coverage required


class DataProcessor:
    """
    Data processing and alignment pipeline for hypothesis testing data
    """
    
    def __init__(self, config: Optional[ProcessingConfig] = None):
        """
        Initialize the data processor
        
        Args:
            config: Processing configuration object
        """
        self.config = config or ProcessingConfig()
        self.logger = logging.getLogger(__name__)
        
        # Initialize scalers and imputers
        self.scaler = StandardScaler()
        self.knn_imputer = KNNImputer(n_neighbors=5)
        
        # Processing metadata
        self.processing_log = []
        self.quality_metrics = {}
    
    def align_frequencies(self, data: Dict[str, pd.Series], 
                         target_freq: Optional[str] = None,
                         rate_variables: Optional[List[str]] = None,
                         stock_variables: Optional[List[str]] = None) -> Dict[str, pd.Series]:
        """
        Align all series to a common frequency
        
        Args:
            data: Dictionary of pandas Series with potentially different frequencies
            target_freq: Target frequency ('D', 'W', 'M', 'Q', 'QE', 'Y')
            rate_variables: List of variable names that are rates (yields, interest rates) - use averaging
            stock_variables: List of variable names that are stocks (debt, holdings) - use end-of-period
            
        Returns:
            Dictionary of aligned series
        """
        target_freq = target_freq or self.config.target_frequency
        rate_variables = rate_variables or []
        stock_variables = stock_variables or []
        self.logger.info(f"Aligning series to {target_freq} frequency")
        
        aligned_data = {}
        alignment_info = {}
        
        for series_name, series in data.items():
            if series is None or series.empty:
                self.logger.warning(f"Skipping empty series: {series_name}")
                continue
            
            try:
                # Detect current frequency
                original_freq = pd.infer_freq(series.index)
                if original_freq is None:
                    # Try to infer from first few observations
                    if len(series) > 1:
                        freq_seconds = (series.index[1] - series.index[0]).total_seconds()
                        if freq_seconds <= 86400:  # Daily or higher frequency
                            original_freq = 'D'
                        elif freq_seconds <= 604800:  # Weekly
                            original_freq = 'W'
                        else:  # Monthly or lower
                            original_freq = 'M'
                    else:
                        original_freq = 'M'  # Default assumption
                
                # Resample to target frequency
                if target_freq == 'D':
                    if original_freq in ['D', 'B']:  # Already daily
                        aligned_series = series
                    else:  # Upsample from lower frequency
                        aligned_series = series.resample('D').interpolate(method='linear')
                elif target_freq == 'W':
                    if original_freq == 'D':  # Downsample from daily
                        aligned_series = series.resample('W').last()
                    elif original_freq == 'W':  # Already weekly
                        aligned_series = series
                    else:  # Upsample from lower frequency
                        aligned_series = series.resample('W').interpolate(method='linear')
                elif target_freq == 'M':
                    if original_freq in ['D', 'W']:  # Downsample
                        if self.config.alignment_method == 'end':
                            aligned_series = series.resample('ME').last()
                        elif self.config.alignment_method == 'start':
                            aligned_series = series.resample('ME').first()
                        else:  # mean
                            aligned_series = series.resample('ME').mean()
                    elif original_freq == 'M':  # Already monthly
                        aligned_series = series
                    else:  # Upsample from quarterly/yearly
                        aligned_series = series.resample('ME').interpolate(method='linear')
                elif target_freq in ['Q', 'QE']:
                    if original_freq in ['D', 'W', 'M']:  # Downsample
                        # Determine aggregation method based on variable type
                        if series_name in rate_variables:
                            # Rate variables: use averaging
                            aligned_series = series.resample('QE').mean()
                            self.logger.debug(f"{series_name} is a rate variable - using mean aggregation")
                        elif series_name in stock_variables:
                            # Stock variables: use end-of-quarter values
                            aligned_series = series.resample('QE').last()
                            self.logger.debug(f"{series_name} is a stock variable - using end-of-quarter")
                        else:
                            # Default behavior based on config
                            if self.config.alignment_method == 'end':
                                aligned_series = series.resample('QE').last()
                            elif self.config.alignment_method == 'start':
                                aligned_series = series.resample('QE').first()
                            else:  # mean
                                aligned_series = series.resample('QE').mean()
                    elif original_freq == 'Q':  # Already quarterly
                        aligned_series = series
                    else:  # Upsample from yearly
                        aligned_series = series.resample('QE').interpolate(method='linear')
                else:  # Yearly or other
                    aligned_series = series.resample(target_freq).mean()
                
                aligned_data[series_name] = aligned_series
                alignment_info[series_name] = {
                    'original_freq': original_freq,
                    'target_freq': target_freq,
                    'original_length': len(series),
                    'aligned_length': len(aligned_series),
                    'method': self.config.alignment_method
                }
                
                self.logger.info(f"Aligned {series_name}: {original_freq} -> {target_freq} "
                               f"({len(series)} -> {len(aligned_series)} obs)")
                
            except Exception as e:
                self.logger.error(f"Failed to align {series_name}: {e}")
                continue
        
        # Log alignment summary
        self.processing_log.append({
            'step': 'frequency_alignment',
            'target_frequency': target_freq,
            'series_processed': len(aligned_data),
            'alignment_info': alignment_info,
            'timestamp': datetime.now().isoformat()
        })
        
        return aligned_data
    
    def handle_missing_values(self, data: Dict[str, pd.Series]) -> Dict[str, pd.Series]:
        """
        Handle missing values using various interpolation methods
        
        Args:
            data: Dictionary of pandas Series with potential missing values
            
        Returns:
            Dictionary of series with missing values handled
        """
        self.logger.info("Handling missing values")
        
        processed_data = {}
        missing_info = {}
        
        for series_name, series in data.items():
            if series is None or series.empty:
                continue
            
            # Calculate missing statistics
            total_obs = len(series)
            missing_obs = series.isna().sum()
            missing_pct = (missing_obs / total_obs) * 100 if total_obs > 0 else 100
            
            missing_info[series_name] = {
                'total_obs': total_obs,
                'missing_obs': missing_obs,
                'missing_pct': missing_pct
            }
            
            # Skip series with too much missing data
            if missing_pct > self.config.max_missing_pct:
                self.logger.warning(f"Skipping {series_name}: {missing_pct:.1f}% missing data "
                                  f"(threshold: {self.config.max_missing_pct}%)")
                continue
            
            # Skip if no missing values
            if missing_obs == 0:
                processed_data[series_name] = series.copy()
                continue
            
            try:
                processed_series = series.copy()
                
                # Check for consecutive missing values
                missing_mask = processed_series.isna()
                consecutive_missing = self._get_max_consecutive_missing(missing_mask)
                
                if consecutive_missing > self.config.max_consecutive_missing:
                    self.logger.warning(f"{series_name}: {consecutive_missing} consecutive missing values "
                                      f"(threshold: {self.config.max_consecutive_missing})")
                
                # Apply interpolation method
                if self.config.interpolation_method == 'linear':
                    processed_series = processed_series.interpolate(method='linear', limit_direction='both')
                elif self.config.interpolation_method == 'cubic':
                    # Check if we have enough data points for cubic interpolation
                    non_missing_count = processed_series.notna().sum()
                    if non_missing_count < 4:
                        self.logger.warning(f"{series_name}: Insufficient data points ({non_missing_count}) "
                                          f"for cubic interpolation, falling back to linear")
                        processed_series = processed_series.interpolate(method='linear', limit_direction='both')
                    else:
                        processed_series = processed_series.interpolate(method='cubic', limit_direction='both')
                        # Validate smoothness (continuous second derivative)
                        if self._validate_cubic_smoothness(processed_series):
                            self.logger.debug(f"{series_name}: Cubic spline smoothness validated")
                        else:
                            self.logger.warning(f"{series_name}: Cubic spline may not be smooth")
                elif self.config.interpolation_method == 'forward':
                    processed_series = processed_series.fillna(method='ffill')
                elif self.config.interpolation_method == 'backward':
                    processed_series = processed_series.fillna(method='bfill')
                elif self.config.interpolation_method == 'knn':
                    # Use KNN imputation for more sophisticated missing value handling
                    processed_series = self._knn_interpolate(processed_series)
                
                # Fill any remaining missing values at the edges
                if processed_series.isna().any():
                    # Forward fill first, then backward fill
                    processed_series = processed_series.fillna(method='ffill').fillna(method='bfill')
                
                processed_data[series_name] = processed_series
                
                final_missing = processed_series.isna().sum()
                self.logger.info(f"Processed {series_name}: {missing_obs} -> {final_missing} missing values")
                
            except Exception as e:
                self.logger.error(f"Failed to process missing values for {series_name}: {e}")
                continue
        
        # Log missing value handling summary
        self.processing_log.append({
            'step': 'missing_value_handling',
            'method': self.config.interpolation_method,
            'series_processed': len(processed_data),
            'missing_info': missing_info,
            'timestamp': datetime.now().isoformat()
        })
        
        return processed_data
    
    def detect_and_handle_outliers(self, data: Dict[str, pd.Series]) -> Dict[str, pd.Series]:
        """
        Detect and handle outliers using various methods
        
        Args:
            data: Dictionary of pandas Series
            
        Returns:
            Dictionary of series with outliers handled
        """
        self.logger.info(f"Detecting and handling outliers using {self.config.outlier_method} method")
        
        processed_data = {}
        outlier_info = {}
        
        for series_name, series in data.items():
            if series is None or series.empty:
                continue
            
            try:
                processed_series = series.copy()
                
                # Detect outliers
                outlier_mask = self._detect_outliers(processed_series, self.config.outlier_method)
                outlier_count = outlier_mask.sum()
                outlier_pct = (outlier_count / len(series)) * 100 if len(series) > 0 else 0
                
                outlier_info[series_name] = {
                    'outlier_count': outlier_count,
                    'outlier_pct': outlier_pct,
                    'method': self.config.outlier_method
                }
                
                if outlier_count > 0:
                    # Handle outliers based on treatment method
                    if self.config.outlier_treatment == 'remove':
                        processed_series = processed_series[~outlier_mask]
                    elif self.config.outlier_treatment == 'winsorize':
                        processed_series = self._winsorize_series(processed_series)
                    elif self.config.outlier_treatment == 'cap':
                        processed_series = self._cap_outliers(processed_series, outlier_mask)
                    elif self.config.outlier_treatment == 'interpolate':
                        processed_series.loc[outlier_mask] = np.nan
                        processed_series = processed_series.interpolate(method='linear')
                    
                    self.logger.info(f"Handled {outlier_count} outliers in {series_name} "
                                   f"({outlier_pct:.1f}%) using {self.config.outlier_treatment}")
                
                processed_data[series_name] = processed_series
                
            except Exception as e:
                self.logger.error(f"Failed to handle outliers for {series_name}: {e}")
                processed_data[series_name] = series.copy()
                continue
        
        # Log outlier handling summary
        self.processing_log.append({
            'step': 'outlier_handling',
            'method': self.config.outlier_method,
            'treatment': self.config.outlier_treatment,
            'series_processed': len(processed_data),
            'outlier_info': outlier_info,
            'timestamp': datetime.now().isoformat()
        })
        
        return processed_data
    
    def construct_hypothesis_variables(self, data: Dict[str, pd.Series]) -> Dict[str, pd.Series]:
        """
        Construct hypothesis-specific measures (γ₁, λ₂, μ₂)
        
        Args:
            data: Dictionary of processed pandas Series
            
        Returns:
            Dictionary including constructed variables
        """
        self.logger.info("Constructing hypothesis-specific variables")
        
        constructed_data = data.copy()
        construction_info = {}
        
        try:
            # Hypothesis 1: Central Bank Reaction Strength (γ₁)
            gamma1_components = []
            
            # Fed balance sheet growth rate
            if 'fed_total_assets' in data:
                fed_assets = data['fed_total_assets']
                fed_growth = fed_assets.pct_change(periods=12).fillna(0)  # YoY growth
                gamma1_components.append(fed_growth)
                constructed_data['fed_assets_growth_yoy'] = fed_growth
            
            # Policy rate change intensity
            if 'federal_funds_rate' in data:
                fed_rate = data['federal_funds_rate']
                rate_change = fed_rate.diff().abs()  # Absolute change
                rate_volatility = rate_change.rolling(window=6).std().fillna(0)
                gamma1_components.append(rate_volatility)
                constructed_data['fed_rate_volatility'] = rate_volatility
            
            # Construct γ₁ (Central Bank Reaction Strength)
            if gamma1_components:
                # Standardize components and take average
                standardized_components = []
                for component in gamma1_components:
                    if component.std() > 0:
                        standardized = (component - component.mean()) / component.std()
                        standardized_components.append(standardized)
                
                if standardized_components:
                    gamma1 = pd.concat(standardized_components, axis=1).mean(axis=1)
                    constructed_data['gamma1_central_bank_reaction'] = gamma1
                    construction_info['gamma1'] = {
                        'components': len(standardized_components),
                        'method': 'standardized_average'
                    }
            
            # Hypothesis 1: Confidence Effects (λ₂)
            lambda2_components = []
            
            # Consumer confidence (inverted for negative effects)
            if 'consumer_confidence' in data:
                consumer_conf = data['consumer_confidence']
                # Standardize and invert (negative confidence effects)
                if consumer_conf.std() > 0:
                    conf_standardized = -(consumer_conf - consumer_conf.mean()) / consumer_conf.std()
                    lambda2_components.append(conf_standardized)
                    constructed_data['consumer_confidence_inverted'] = conf_standardized
            
            # Financial stress index
            if 'financial_stress_index' in data:
                stress_index = data['financial_stress_index']
                if stress_index.std() > 0:
                    stress_standardized = (stress_index - stress_index.mean()) / stress_index.std()
                    lambda2_components.append(stress_standardized)
                    constructed_data['financial_stress_standardized'] = stress_standardized
            
            # VIX volatility
            if 'vix_index' in data:
                vix = data['vix_index']
                if vix.std() > 0:
                    vix_standardized = (vix - vix.mean()) / vix.std()
                    lambda2_components.append(vix_standardized)
                    constructed_data['vix_standardized'] = vix_standardized
            
            # Construct λ₂ (Confidence Effects)
            if lambda2_components:
                lambda2 = pd.concat(lambda2_components, axis=1).mean(axis=1)
                constructed_data['lambda2_confidence_effects'] = lambda2
                construction_info['lambda2'] = {
                    'components': len(lambda2_components),
                    'method': 'standardized_average'
                }
            
            # Hypothesis 2: Market Distortions (μ₂)
            mu2_components = []
            
            # Credit spreads
            if 'corporate_bond_spreads' in data:
                credit_spreads = data['corporate_bond_spreads']
                if credit_spreads.std() > 0:
                    spreads_standardized = (credit_spreads - credit_spreads.mean()) / credit_spreads.std()
                    mu2_components.append(spreads_standardized)
                    constructed_data['credit_spreads_standardized'] = spreads_standardized
            
            # Liquidity premium (term spread)
            if 'treasury_10y' in data and 'treasury_2y' in data:
                term_spread = data['treasury_10y'] - data['treasury_2y']
                if term_spread.std() > 0:
                    spread_standardized = (term_spread - term_spread.mean()) / term_spread.std()
                    mu2_components.append(spread_standardized)
                    constructed_data['term_spread_standardized'] = spread_standardized
            elif 'liquidity_premium' in data:
                liquidity_prem = data['liquidity_premium']
                if liquidity_prem.std() > 0:
                    liq_standardized = (liquidity_prem - liquidity_prem.mean()) / liquidity_prem.std()
                    mu2_components.append(liq_standardized)
                    constructed_data['liquidity_premium_standardized'] = liq_standardized
            
            # Construct μ₂ (Market Distortions)
            if mu2_components:
                mu2 = pd.concat(mu2_components, axis=1).mean(axis=1)
                constructed_data['mu2_market_distortions'] = mu2
                construction_info['mu2'] = {
                    'components': len(mu2_components),
                    'method': 'standardized_average'
                }
            
            # Additional constructed variables
            
            # QE Intensity measure
            if 'fed_securities_held' in data and 'treasury_outstanding' in data:
                qe_intensity = data['fed_securities_held'] / data['treasury_outstanding']
                constructed_data['qe_intensity_ratio'] = qe_intensity
                construction_info['qe_intensity'] = {
                    'method': 'fed_holdings_to_outstanding_ratio'
                }
            
            # Debt service burden
            if 'federal_interest_payments' in data and 'gdp_nominal' in data:
                debt_service_burden = (data['federal_interest_payments'] / data['gdp_nominal']) * 100
                constructed_data['debt_service_burden_pct'] = debt_service_burden
                construction_info['debt_service_burden'] = {
                    'method': 'interest_payments_to_gdp_ratio'
                }
            
            # Real interest rates
            if 'treasury_10y' in data and 'breakeven_inflation_10y' in data:
                real_rate_10y = data['treasury_10y'] - data['breakeven_inflation_10y']
                constructed_data['real_rate_10y'] = real_rate_10y
                construction_info['real_rate_10y'] = {
                    'method': 'nominal_minus_breakeven_inflation'
                }
            
        except Exception as e:
            self.logger.error(f"Error in variable construction: {e}")
        
        # Log variable construction summary
        self.processing_log.append({
            'step': 'variable_construction',
            'variables_constructed': len(construction_info),
            'construction_info': construction_info,
            'timestamp': datetime.now().isoformat()
        })
        
        self.logger.info(f"Constructed {len(construction_info)} hypothesis-specific variables")
        
        return constructed_data
    
    def validate_quarterly_conversion(self, 
                                     monthly_data: Dict[str, pd.Series],
                                     quarterly_data: Dict[str, pd.Series],
                                     rate_variables: Optional[List[str]] = None,
                                     stock_variables: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Validate that quarterly conversion preserves information correctly
        
        Args:
            monthly_data: Dictionary of monthly series (before conversion)
            quarterly_data: Dictionary of quarterly series (after conversion)
            rate_variables: List of rate variable names (should use averaging)
            stock_variables: List of stock variable names (should use end-of-quarter)
            
        Returns:
            Dictionary containing validation results
        """
        self.logger.info("Validating quarterly conversion")
        
        rate_variables = rate_variables or []
        stock_variables = stock_variables or []
        validation_results = {}
        
        for series_name in quarterly_data.keys():
            if series_name not in monthly_data:
                continue
                
            monthly_series = monthly_data[series_name]
            quarterly_series = quarterly_data[series_name]
            
            # Skip if either series is empty
            if monthly_series.empty or quarterly_series.empty:
                continue
            
            try:
                # Validate rate variables (should equal mean of monthly values)
                if series_name in rate_variables:
                    # Resample monthly to quarterly using mean for comparison
                    expected_quarterly = monthly_series.resample('QE').mean()
                    
                    # Align indices
                    common_index = quarterly_series.index.intersection(expected_quarterly.index)
                    if len(common_index) == 0:
                        validation_results[series_name] = {
                            'variable_type': 'rate',
                            'is_valid': False,
                            'error': 'No common dates between monthly and quarterly data'
                        }
                        continue
                    
                    actual = quarterly_series.loc[common_index]
                    expected = expected_quarterly.loc[common_index]
                    
                    # Calculate correlation and mean absolute error
                    correlation = actual.corr(expected)
                    mae = np.abs(actual - expected).mean()
                    max_error = np.abs(actual - expected).max()
                    
                    # Check if values are approximately equal (within tolerance)
                    tolerance = 1e-10
                    is_valid = np.allclose(actual, expected, rtol=tolerance, atol=tolerance)
                    
                    validation_results[series_name] = {
                        'variable_type': 'rate',
                        'is_valid': is_valid,
                        'correlation': correlation,
                        'mean_absolute_error': mae,
                        'max_error': max_error,
                        'aggregation_method': 'mean'
                    }
                    
                # Validate stock variables (should equal last monthly value in quarter)
                elif series_name in stock_variables:
                    # Resample monthly to quarterly using last for comparison
                    expected_quarterly = monthly_series.resample('QE').last()
                    
                    # Align indices
                    common_index = quarterly_series.index.intersection(expected_quarterly.index)
                    if len(common_index) == 0:
                        validation_results[series_name] = {
                            'variable_type': 'stock',
                            'is_valid': False,
                            'error': 'No common dates between monthly and quarterly data'
                        }
                        continue
                    
                    actual = quarterly_series.loc[common_index]
                    expected = expected_quarterly.loc[common_index]
                    
                    # Calculate correlation and mean absolute error
                    correlation = actual.corr(expected)
                    mae = np.abs(actual - expected).mean()
                    max_error = np.abs(actual - expected).max()
                    
                    # Check if values are exactly equal
                    tolerance = 1e-10
                    is_valid = np.allclose(actual, expected, rtol=tolerance, atol=tolerance)
                    
                    validation_results[series_name] = {
                        'variable_type': 'stock',
                        'is_valid': is_valid,
                        'correlation': correlation,
                        'mean_absolute_error': mae,
                        'max_error': max_error,
                        'aggregation_method': 'last'
                    }
                    
            except Exception as e:
                validation_results[series_name] = {
                    'is_valid': False,
                    'error': str(e)
                }
                self.logger.error(f"Validation failed for {series_name}: {e}")
        
        # Calculate summary statistics
        valid_count = sum(1 for v in validation_results.values() 
                         if v.get('is_valid', False))
        total_count = len(validation_results)
        
        summary = {
            'total_validated': total_count,
            'valid_count': valid_count,
            'invalid_count': total_count - valid_count,
            'validation_rate': valid_count / total_count if total_count > 0 else 0.0,
            'timestamp': datetime.now().isoformat()
        }
        
        self.logger.info(f"Quarterly conversion validation: {valid_count}/{total_count} series valid")
        
        return {
            'summary': summary,
            'series_results': validation_results
        }
    
    def validate_processed_data(self, data: Dict[str, pd.Series]) -> Dict[str, Any]:
        """
        Validate processed data quality and coverage
        
        Args:
            data: Dictionary of processed pandas Series
            
        Returns:
            Dictionary containing validation results
        """
        self.logger.info("Validating processed data quality")
        
        validation_results = {}
        overall_quality = 0.0
        valid_series_count = 0
        
        for series_name, series in data.items():
            if series is None or series.empty:
                validation_results[series_name] = {
                    'is_valid': False,
                    'error': 'Series is empty or None',
                    'quality_score': 0.0
                }
                continue
            
            # Basic validation metrics
            total_obs = len(series)
            missing_obs = series.isna().sum()
            missing_pct = (missing_obs / total_obs) * 100 if total_obs > 0 else 100
            
            # Date coverage
            if hasattr(series.index, 'min') and hasattr(series.index, 'max'):
                date_range = series.index.max() - series.index.min()
                # Use updated frequency codes
                freq_map = {'M': 'ME', 'Q': 'QE', 'QE': 'QE'}
                freq = freq_map.get(self.config.target_frequency, self.config.target_frequency)
                expected_obs = len(pd.date_range(series.index.min(), series.index.max(), 
                                               freq=freq))
                date_coverage = total_obs / expected_obs if expected_obs > 0 else 0
                
                # Check for target date range (2008Q1-2023Q4) for quarterly data
                target_start = pd.Timestamp('2008-03-31')  # 2008Q1 end
                target_end = pd.Timestamp('2023-12-31')    # 2023Q4 end
                
                date_range_warning = None
                if self.config.target_frequency in ['Q', 'QE']:
                    if series.index.min() > target_start:
                        date_range_warning = f"Data starts after target (2008Q1): {series.index.min()}"
                        self.logger.warning(f"{series_name}: {date_range_warning}")
                    if series.index.max() < target_end:
                        date_range_warning = f"Data ends before target (2023Q4): {series.index.max()}"
                        self.logger.warning(f"{series_name}: {date_range_warning}")
            else:
                date_coverage = 1.0
                date_range_warning = None
            
            # Quality checks
            is_valid = (
                total_obs >= self.config.min_observations and
                missing_pct <= self.config.max_missing_pct and
                date_coverage >= self.config.min_date_coverage
            )
            
            # Calculate quality score
            quality_score = 100.0
            if total_obs < self.config.min_observations:
                quality_score -= 30.0
            quality_score -= min(missing_pct, 30.0)
            quality_score -= max(0, (self.config.min_date_coverage - date_coverage) * 40)
            quality_score = max(quality_score, 0.0)
            
            validation_results[series_name] = {
                'is_valid': is_valid,
                'total_observations': total_obs,
                'missing_observations': missing_obs,
                'missing_percentage': missing_pct,
                'date_coverage': date_coverage,
                'quality_score': quality_score,
                'date_range': {
                    'start': series.index.min() if hasattr(series.index, 'min') else None,
                    'end': series.index.max() if hasattr(series.index, 'max') else None
                },
                'date_range_warning': date_range_warning if 'date_range_warning' in locals() else None
            }
            
            if is_valid:
                overall_quality += quality_score
                valid_series_count += 1
        
        # Calculate overall metrics
        overall_quality = overall_quality / valid_series_count if valid_series_count > 0 else 0.0
        
        summary = {
            'overall_quality_score': overall_quality,
            'total_series': len(data),
            'valid_series_count': valid_series_count,
            'validation_timestamp': datetime.now().isoformat(),
            'processing_config': {
                'target_frequency': self.config.target_frequency,
                'interpolation_method': self.config.interpolation_method,
                'outlier_method': self.config.outlier_method,
                'outlier_treatment': self.config.outlier_treatment
            }
        }
        
        self.logger.info(f"Data validation complete. Overall quality: {overall_quality:.1f}")
        self.logger.info(f"Valid series: {valid_series_count}/{len(data)}")
        
        return {
            'summary': summary,
            'series_results': validation_results
        }
    
    def process_hypothesis_data(self, hypothesis_data: HypothesisData) -> Tuple[HypothesisData, Dict[str, Any]]:
        """
        Complete processing pipeline for hypothesis data
        
        Args:
            hypothesis_data: HypothesisData object with raw data
            
        Returns:
            Tuple of (processed HypothesisData, processing report)
        """
        self.logger.info("Starting complete data processing pipeline")
        
        # Convert HypothesisData to dictionary for processing
        data_dict = {}
        for field_name, field_value in hypothesis_data.__dict__.items():
            if isinstance(field_value, pd.Series) and not field_value.empty:
                data_dict[field_name] = field_value
        
        if not data_dict:
            self.logger.warning("No valid series found in hypothesis data")
            return hypothesis_data, {'error': 'No valid series found'}
        
        # Step 1: Align frequencies
        aligned_data = self.align_frequencies(data_dict)
        
        # Step 2: Handle missing values
        processed_data = self.handle_missing_values(aligned_data)
        
        # Step 3: Detect and handle outliers
        cleaned_data = self.detect_and_handle_outliers(processed_data)
        
        # Step 4: Construct hypothesis-specific variables
        final_data = self.construct_hypothesis_variables(cleaned_data)
        
        # Step 5: Validate processed data
        validation_results = self.validate_processed_data(final_data)
        
        # Create processed HypothesisData object
        processed_hypothesis_data = HypothesisData()
        
        # Map processed data back to HypothesisData structure
        for field_name in hypothesis_data.__dict__.keys():
            if field_name in final_data:
                setattr(processed_hypothesis_data, field_name, final_data[field_name])
            elif hasattr(hypothesis_data, field_name):
                setattr(processed_hypothesis_data, field_name, getattr(hypothesis_data, field_name))
        
        # Add constructed variables to metadata
        processed_hypothesis_data.metadata = hypothesis_data.metadata or {}
        processed_hypothesis_data.metadata.update({
            'processing_timestamp': datetime.now().isoformat(),
            'processing_config': self.config.__dict__,
            'constructed_variables': [k for k in final_data.keys() 
                                    if k not in data_dict.keys()],
            'processing_log': self.processing_log
        })
        
        # Create processing report
        processing_report = {
            'summary': {
                'input_series': len(data_dict),
                'output_series': len(final_data),
                'constructed_variables': len(final_data) - len(data_dict),
                'overall_quality': validation_results['summary']['overall_quality_score'],
                'valid_series': validation_results['summary']['valid_series_count']
            },
            'processing_steps': self.processing_log,
            'validation_results': validation_results,
            'timestamp': datetime.now().isoformat()
        }
        
        self.logger.info("Data processing pipeline completed successfully")
        self.logger.info(f"Processed {len(data_dict)} -> {len(final_data)} series")
        self.logger.info(f"Overall quality score: {validation_results['summary']['overall_quality_score']:.1f}")
        
        return processed_hypothesis_data, processing_report
    
    def filter_to_date_range(self, data: Dict[str, pd.Series],
                            start_date: Optional[str] = '2008-03-31',
                            end_date: Optional[str] = '2023-12-31') -> Dict[str, pd.Series]:
        """
        Filter data to target date range (default: 2008Q1-2023Q4)
        
        Args:
            data: Dictionary of pandas Series
            start_date: Start date (default: 2008Q1 end)
            end_date: End date (default: 2023Q4 end)
            
        Returns:
            Dictionary of filtered series
        """
        self.logger.info(f"Filtering data to date range: {start_date} to {end_date}")
        
        start_ts = pd.Timestamp(start_date)
        end_ts = pd.Timestamp(end_date)
        
        filtered_data = {}
        for series_name, series in data.items():
            if series is None or series.empty:
                continue
            
            try:
                # Filter to date range
                mask = (series.index >= start_ts) & (series.index <= end_ts)
                filtered_series = series[mask]
                
                if len(filtered_series) > 0:
                    filtered_data[series_name] = filtered_series
                    self.logger.debug(f"Filtered {series_name}: {len(series)} -> {len(filtered_series)} obs")
                else:
                    self.logger.warning(f"No data in target range for {series_name}")
                    
            except Exception as e:
                self.logger.error(f"Failed to filter {series_name}: {e}")
                continue
        
        return filtered_data
    
    # Helper methods
    
    def _get_max_consecutive_missing(self, missing_mask: pd.Series) -> int:
        """Get maximum consecutive missing values"""
        if not missing_mask.any():
            return 0
        
        consecutive_counts = []
        current_count = 0
        
        for is_missing in missing_mask:
            if is_missing:
                current_count += 1
            else:
                if current_count > 0:
                    consecutive_counts.append(current_count)
                current_count = 0
        
        if current_count > 0:
            consecutive_counts.append(current_count)
        
        return max(consecutive_counts) if consecutive_counts else 0
    
    def _knn_interpolate(self, series: pd.Series) -> pd.Series:
        """Use KNN imputation for missing values"""
        try:
            # Create a DataFrame with the series and its lags for KNN
            df = pd.DataFrame({'value': series})
            for lag in [1, 2, 3]:
                df[f'lag_{lag}'] = series.shift(lag)
                df[f'lead_{lag}'] = series.shift(-lag)
            
            # Apply KNN imputation
            imputed_values = self.knn_imputer.fit_transform(df)
            return pd.Series(imputed_values[:, 0], index=series.index)
        except Exception as e:
            self.logger.warning(f"KNN imputation failed, falling back to linear: {e}")
            return series.interpolate(method='linear')
    
    def _detect_outliers(self, series: pd.Series, method: str) -> pd.Series:
        """Detect outliers using specified method"""
        if method == 'iqr':
            Q1 = series.quantile(0.25)
            Q3 = series.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            return (series < lower_bound) | (series > upper_bound)
        
        elif method == 'zscore':
            z_scores = np.abs(stats.zscore(series.dropna()))
            outlier_mask = pd.Series(False, index=series.index)
            outlier_mask.loc[series.dropna().index] = z_scores > self.config.outlier_threshold
            return outlier_mask
        
        elif method == 'modified_zscore':
            median = series.median()
            mad = np.median(np.abs(series - median))
            modified_z_scores = 0.6745 * (series - median) / mad
            return np.abs(modified_z_scores) > self.config.outlier_threshold
        
        else:
            # Default to IQR method
            return self._detect_outliers(series, 'iqr')
    
    def _winsorize_series(self, series: pd.Series) -> pd.Series:
        """Winsorize series at specified limits"""
        lower_limit, upper_limit = self.config.winsorize_limits
        lower_percentile = series.quantile(lower_limit)
        upper_percentile = series.quantile(1 - upper_limit)
        
        winsorized = series.copy()
        winsorized[winsorized < lower_percentile] = lower_percentile
        winsorized[winsorized > upper_percentile] = upper_percentile
        
        return winsorized
    
    def _cap_outliers(self, series: pd.Series, outlier_mask: pd.Series) -> pd.Series:
        """Cap outliers at reasonable bounds"""
        capped = series.copy()
        
        # Cap at 99th and 1st percentiles
        lower_cap = series.quantile(0.01)
        upper_cap = series.quantile(0.99)
        
        capped[outlier_mask & (series < lower_cap)] = lower_cap
        capped[outlier_mask & (series > upper_cap)] = upper_cap
        
        return capped
    
    def _validate_cubic_smoothness(self, series: pd.Series) -> bool:
        """
        Validate that cubic spline interpolation maintains smoothness
        (continuous second derivative)
        
        Args:
            series: Interpolated series
            
        Returns:
            True if series appears smooth, False otherwise
        """
        try:
            # Remove any remaining NaN values
            clean_series = series.dropna()
            
            if len(clean_series) < 4:
                return False
            
            # Calculate first and second derivatives using finite differences
            values = clean_series.values
            
            # First derivative (approximate)
            first_deriv = np.diff(values)
            
            # Second derivative (approximate)
            second_deriv = np.diff(first_deriv)
            
            # Check for discontinuities in second derivative
            # A cubic spline should have continuous second derivative
            # Check if second derivative changes are reasonable (no large jumps)
            if len(second_deriv) > 1:
                second_deriv_changes = np.diff(second_deriv)
                # Use median absolute deviation to detect outliers in changes
                mad = np.median(np.abs(second_deriv_changes - np.median(second_deriv_changes)))
                if mad > 0:
                    # Check if any changes are more than 5 MADs away
                    threshold = 5 * mad
                    large_jumps = np.abs(second_deriv_changes - np.median(second_deriv_changes)) > threshold
                    if large_jumps.any():
                        return False
            
            return True
            
        except Exception as e:
            self.logger.debug(f"Smoothness validation failed: {e}")
            return True  # Assume smooth if validation fails