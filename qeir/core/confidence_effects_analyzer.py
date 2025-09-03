"""
Confidence Effects Measurement and Interaction Analysis

This module implements confidence effects measurement from FRED data and interaction analysis
between central bank reaction (γ₁) and confidence effects (λ₂) for Hypothesis 1 testing.
"""

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import pearsonr, spearmanr
import statsmodels.api as sm
from statsmodels.regression.linear_model import OLS
from statsmodels.stats.diagnostic import het_breuschpagan, het_white
from statsmodels.stats.stattools import durbin_watson
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
import logging
import warnings

warnings.filterwarnings('ignore')


@dataclass
class ConfidenceEffectMeasures:
    """Data structure for confidence effect measures from FRED data"""
    
    # Consumer confidence measures
    consumer_confidence_index: Optional[pd.Series] = None
    consumer_sentiment_michigan: Optional[pd.Series] = None
    consumer_expectations: Optional[pd.Series] = None
    
    # Business confidence measures
    business_confidence_index: Optional[pd.Series] = None
    small_business_optimism: Optional[pd.Series] = None
    ceo_confidence_index: Optional[pd.Series] = None
    
    # Financial stress measures
    financial_stress_index: Optional[pd.Series] = None
    vix_volatility_index: Optional[pd.Series] = None
    credit_spread_investment_grade: Optional[pd.Series] = None
    credit_spread_high_yield: Optional[pd.Series] = None
    
    # Market-based confidence proxies
    equity_risk_premium: Optional[pd.Series] = None
    term_premium: Optional[pd.Series] = None
    liquidity_premium: Optional[pd.Series] = None
    
    # Composite measures
    composite_confidence_index: Optional[pd.Series] = None
    financial_conditions_index: Optional[pd.Series] = None
    
    def get_available_measures(self) -> List[str]:
        """Get list of available confidence measures"""
        available = []
        for field_name, field_value in self.__dict__.items():
            if field_value is not None and isinstance(field_value, pd.Series):
                available.append(field_name)
        return available
    
    def get_measure(self, measure_name: str) -> pd.Series:
        """Get specific confidence measure by name"""
        if hasattr(self, measure_name):
            measure = getattr(self, measure_name)
            if measure is not None:
                return measure
            else:
                raise ValueError(f"Confidence measure '{measure_name}' is None")
        else:
            raise ValueError(f"Unknown confidence measure: {measure_name}")


class ConfidenceEffectsProcessor:
    """
    Processor for creating confidence effect proxies from FRED data.
    
    Handles data cleaning, normalization, and creation of composite confidence measures.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.processed_measures = None
        self.normalization_params = {}
        
    def process_raw_confidence_data(self, 
                                  raw_data: Dict[str, pd.Series],
                                  normalization_method: str = 'zscore',
                                  handle_missing: str = 'interpolate') -> ConfidenceEffectMeasures:
        """
        Process raw confidence data from FRED into standardized measures.
        
        Args:
            raw_data: Dictionary of raw confidence series from FRED
            normalization_method: Method for normalization ('zscore', 'minmax', 'none')
            handle_missing: Method for handling missing values ('interpolate', 'forward_fill', 'drop')
            
        Returns:
            ConfidenceEffectMeasures object with processed data
        """
        processed_data = {}
        
        for series_name, series_data in raw_data.items():
            if series_data is None or len(series_data) == 0:
                self.logger.warning(f"Skipping empty series: {series_name}")
                continue
            
            # Handle missing values
            if handle_missing == 'interpolate':
                series_clean = series_data.interpolate(method='linear', limit_direction='both')
            elif handle_missing == 'forward_fill':
                series_clean = series_data.fillna(method='ffill')
            elif handle_missing == 'drop':
                series_clean = series_data.dropna()
            else:
                series_clean = series_data
            
            # Normalize data
            if normalization_method == 'zscore':
                mean_val = series_clean.mean()
                std_val = series_clean.std()
                if std_val > 0:
                    series_normalized = (series_clean - mean_val) / std_val
                    self.normalization_params[series_name] = {'mean': mean_val, 'std': std_val, 'method': 'zscore'}
                else:
                    series_normalized = series_clean
                    self.normalization_params[series_name] = {'method': 'none', 'reason': 'zero_std'}
                    
            elif normalization_method == 'minmax':
                min_val = series_clean.min()
                max_val = series_clean.max()
                if max_val > min_val:
                    series_normalized = (series_clean - min_val) / (max_val - min_val)
                    self.normalization_params[series_name] = {'min': min_val, 'max': max_val, 'method': 'minmax'}
                else:
                    series_normalized = series_clean
                    self.normalization_params[series_name] = {'method': 'none', 'reason': 'constant_values'}
                    
            else:  # normalization_method == 'none'
                series_normalized = series_clean
                self.normalization_params[series_name] = {'method': 'none'}
            
            processed_data[series_name] = series_normalized
        
        # Create ConfidenceEffectMeasures object
        measures = ConfidenceEffectMeasures()
        
        # Map processed data to appropriate fields
        field_mapping = {
            'consumer_confidence': 'consumer_confidence_index',
            'consumer_sentiment': 'consumer_sentiment_michigan',
            'consumer_expectations': 'consumer_expectations',
            'business_confidence': 'business_confidence_index',
            'small_business_optimism': 'small_business_optimism',
            'ceo_confidence': 'ceo_confidence_index',
            'financial_stress': 'financial_stress_index',
            'vix': 'vix_volatility_index',
            'credit_spread_ig': 'credit_spread_investment_grade',
            'credit_spread_hy': 'credit_spread_high_yield',
            'equity_risk_premium': 'equity_risk_premium',
            'term_premium': 'term_premium',
            'liquidity_premium': 'liquidity_premium'
        }
        
        for raw_name, processed_series in processed_data.items():
            if raw_name in field_mapping:
                setattr(measures, field_mapping[raw_name], processed_series)
            else:
                # Try direct mapping
                if hasattr(measures, raw_name):
                    setattr(measures, raw_name, processed_series)
        
        self.processed_measures = measures
        return measures
    
    def create_composite_confidence_index(self, 
                                        measures: ConfidenceEffectMeasures,
                                        method: str = 'pca',
                                        weights: Optional[Dict[str, float]] = None) -> pd.Series:
        """
        Create composite confidence index from individual measures.
        
        Args:
            measures: ConfidenceEffectMeasures object
            method: Method for creating composite ('pca', 'weighted_average', 'equal_weight')
            weights: Dictionary of weights for weighted_average method
            
        Returns:
            Composite confidence index as pandas Series
        """
        available_measures = measures.get_available_measures()
        
        if len(available_measures) == 0:
            raise ValueError("No confidence measures available for composite index")
        
        # Get all available series and align dates
        series_list = []
        series_names = []
        
        for measure_name in available_measures:
            series = measures.get_measure(measure_name)
            if series is not None and len(series) > 0:
                series_list.append(series)
                series_names.append(measure_name)
        
        if len(series_list) == 0:
            raise ValueError("No valid confidence series found")
        
        # Align all series to common dates
        common_index = series_list[0].index
        for series in series_list[1:]:
            common_index = common_index.intersection(series.index)
        
        if len(common_index) < 10:
            raise ValueError(f"Insufficient common observations: {len(common_index)}")
        
        # Create aligned DataFrame
        aligned_data = pd.DataFrame()
        for i, series in enumerate(series_list):
            aligned_data[series_names[i]] = series.loc[common_index]
        
        # Remove any remaining NaN values
        aligned_data = aligned_data.dropna()
        
        if len(aligned_data) < 10:
            raise ValueError("Insufficient data after removing NaN values")
        
        # Create composite index
        if method == 'pca':
            composite_index = self._create_pca_composite(aligned_data)
        elif method == 'weighted_average':
            composite_index = self._create_weighted_composite(aligned_data, weights, series_names)
        elif method == 'equal_weight':
            composite_index = self._create_equal_weight_composite(aligned_data)
        else:
            raise ValueError(f"Unknown composite method: {method}")
        
        # Store composite index
        measures.composite_confidence_index = composite_index
        
        return composite_index
    
    def _create_pca_composite(self, data: pd.DataFrame) -> pd.Series:
        """Create PCA-based composite index"""
        from sklearn.decomposition import PCA
        from sklearn.preprocessing import StandardScaler
        
        # Standardize data
        scaler = StandardScaler()
        data_scaled = scaler.fit_transform(data)
        
        # Apply PCA
        pca = PCA(n_components=1)
        pca_result = pca.fit_transform(data_scaled)
        
        # Create series
        composite_index = pd.Series(
            pca_result.flatten(),
            index=data.index,
            name='composite_confidence_pca'
        )
        
        self.logger.info(f"PCA composite created. Explained variance: {pca.explained_variance_ratio_[0]:.3f}")
        
        return composite_index
    
    def _create_weighted_composite(self, 
                                 data: pd.DataFrame, 
                                 weights: Optional[Dict[str, float]], 
                                 series_names: List[str]) -> pd.Series:
        """Create weighted average composite index"""
        if weights is None:
            # Equal weights
            weights = {name: 1.0 / len(series_names) for name in series_names}
        else:
            # Normalize weights
            total_weight = sum(weights.values())
            weights = {k: v / total_weight for k, v in weights.items()}
        
        # Calculate weighted average
        composite_values = np.zeros(len(data))
        for col_name in data.columns:
            if col_name in weights:
                composite_values += data[col_name].values * weights[col_name]
        
        composite_index = pd.Series(
            composite_values,
            index=data.index,
            name='composite_confidence_weighted'
        )
        
        return composite_index
    
    def _create_equal_weight_composite(self, data: pd.DataFrame) -> pd.Series:
        """Create equal-weighted composite index"""
        composite_index = data.mean(axis=1)
        composite_index.name = 'composite_confidence_equal_weight'
        return composite_index


class InteractionAnalyzer:
    """
    Analyzer for interaction effects between central bank reaction (γ₁) and confidence effects (λ₂).
    
    Implements statistical significance tests for interaction effects and provides
    comprehensive interaction analysis.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.interaction_results = None
        
    def analyze_interactions(self,
                           dependent_var: pd.Series,
                           central_bank_reaction: pd.Series,
                           confidence_effects: pd.Series,
                           control_variables: Optional[pd.DataFrame] = None,
                           interaction_types: List[str] = ['multiplicative', 'threshold']) -> Dict[str, Any]:
        """
        Comprehensive analysis of interaction effects between γ₁ and λ₂.
        
        Args:
            dependent_var: Dependent variable (e.g., long-term yields)
            central_bank_reaction: Central bank reaction strength (γ₁)
            confidence_effects: Confidence effects (λ₂)
            control_variables: Additional control variables
            interaction_types: Types of interactions to test
            
        Returns:
            Dictionary with comprehensive interaction analysis results
        """
        # Align all series
        common_index = dependent_var.index.intersection(central_bank_reaction.index)\
                                          .intersection(confidence_effects.index)
        
        if control_variables is not None:
            common_index = common_index.intersection(control_variables.index)
        
        if len(common_index) < 20:
            raise ValueError(f"Insufficient common observations: {len(common_index)}")
        
        y = dependent_var.loc[common_index]
        gamma1 = central_bank_reaction.loc[common_index]
        lambda2 = confidence_effects.loc[common_index]
        
        if control_variables is not None:
            controls = control_variables.loc[common_index]
        else:
            controls = None
        
        results = {
            'data_info': {
                'observations': len(common_index),
                'start_date': common_index.min().strftime('%Y-%m-%d'),
                'end_date': common_index.max().strftime('%Y-%m-%d')
            },
            'interaction_analyses': {}
        }
        
        # Multiplicative interaction analysis
        if 'multiplicative' in interaction_types:
            mult_results = self._analyze_multiplicative_interaction(y, gamma1, lambda2, controls)
            results['interaction_analyses']['multiplicative'] = mult_results
        
        # Threshold interaction analysis
        if 'threshold' in interaction_types:
            threshold_results = self._analyze_threshold_interaction(y, gamma1, lambda2, controls)
            results['interaction_analyses']['threshold'] = threshold_results
        
        # Correlation analysis
        correlation_results = self._analyze_correlations(y, gamma1, lambda2)
        results['correlation_analysis'] = correlation_results
        
        # Regime-dependent interaction analysis
        regime_results = self._analyze_regime_dependent_interactions(y, gamma1, lambda2)
        results['regime_dependent_analysis'] = regime_results
        
        self.interaction_results = results
        return results
    
    def _analyze_multiplicative_interaction(self,
                                          y: pd.Series,
                                          gamma1: pd.Series,
                                          lambda2: pd.Series,
                                          controls: Optional[pd.DataFrame]) -> Dict[str, Any]:
        """Analyze multiplicative interaction: γ₁ * λ₂"""
        
        # Create interaction term
        interaction_term = gamma1 * lambda2
        
        # Prepare regression data
        X_data = {
            'gamma1': gamma1,
            'lambda2': lambda2,
            'interaction': interaction_term
        }
        
        if controls is not None:
            for col in controls.columns:
                X_data[f'control_{col}'] = controls[col]
        
        X_df = pd.DataFrame(X_data, index=y.index)
        X_with_const = sm.add_constant(X_df)
        
        # Fit regression
        model = OLS(y, X_with_const).fit()
        
        # Extract interaction coefficient and significance
        interaction_coeff = model.params['interaction']
        interaction_se = model.bse['interaction']
        interaction_tstat = model.tvalues['interaction']
        interaction_pvalue = model.pvalues['interaction']
        
        # Calculate marginal effects
        marginal_effects = self._calculate_marginal_effects(gamma1, lambda2, interaction_coeff, 
                                                          model.params['gamma1'], model.params['lambda2'])
        
        # Diagnostic tests
        diagnostics = self._run_regression_diagnostics(model, X_with_const, y)
        
        return {
            'model_summary': {
                'r_squared': model.rsquared,
                'adj_r_squared': model.rsquared_adj,
                'f_statistic': model.fvalue,
                'f_pvalue': model.f_pvalue,
                'observations': model.nobs
            },
            'interaction_coefficient': {
                'coefficient': interaction_coeff,
                'std_error': interaction_se,
                't_statistic': interaction_tstat,
                'p_value': interaction_pvalue,
                'significant_5pct': interaction_pvalue < 0.05,
                'significant_1pct': interaction_pvalue < 0.01
            },
            'main_effects': {
                'gamma1': {
                    'coefficient': model.params['gamma1'],
                    'p_value': model.pvalues['gamma1'],
                    'significant': model.pvalues['gamma1'] < 0.05
                },
                'lambda2': {
                    'coefficient': model.params['lambda2'],
                    'p_value': model.pvalues['lambda2'],
                    'significant': model.pvalues['lambda2'] < 0.05
                }
            },
            'marginal_effects': marginal_effects,
            'diagnostics': diagnostics,
            'interpretation': self._interpret_multiplicative_interaction(
                interaction_coeff, interaction_pvalue, model.params['gamma1'], model.params['lambda2']
            )
        }
    
    def _analyze_threshold_interaction(self,
                                     y: pd.Series,
                                     gamma1: pd.Series,
                                     lambda2: pd.Series,
                                     controls: Optional[pd.DataFrame]) -> Dict[str, Any]:
        """Analyze threshold-based interaction effects"""
        
        # Define thresholds based on percentiles
        gamma1_thresholds = [np.percentile(gamma1, p) for p in [25, 50, 75]]
        lambda2_thresholds = [np.percentile(lambda2, p) for p in [25, 50, 75]]
        
        threshold_results = {}
        
        # Test different threshold combinations
        for i, gamma_thresh in enumerate(gamma1_thresholds):
            for j, lambda_thresh in enumerate(lambda2_thresholds):
                
                # Create regime indicators
                high_gamma = (gamma1 > gamma_thresh).astype(int)
                high_lambda = (lambda2 > lambda_thresh).astype(int)
                
                # Create interaction terms
                gamma_lambda_interaction = high_gamma * high_lambda
                
                # Prepare regression
                X_data = {
                    'gamma1': gamma1,
                    'lambda2': lambda2,
                    'high_gamma': high_gamma,
                    'high_lambda': high_lambda,
                    'high_gamma_high_lambda': gamma_lambda_interaction
                }
                
                if controls is not None:
                    for col in controls.columns:
                        X_data[f'control_{col}'] = controls[col]
                
                X_df = pd.DataFrame(X_data, index=y.index)
                X_with_const = sm.add_constant(X_df)
                
                try:
                    model = OLS(y, X_with_const).fit()
                    
                    threshold_results[f'gamma_p{25+i*25}_lambda_p{25+j*25}'] = {
                        'gamma_threshold': gamma_thresh,
                        'lambda_threshold': lambda_thresh,
                        'interaction_coefficient': model.params['high_gamma_high_lambda'],
                        'interaction_pvalue': model.pvalues['high_gamma_high_lambda'],
                        'r_squared': model.rsquared,
                        'significant': model.pvalues['high_gamma_high_lambda'] < 0.05,
                        'regime_observations': {
                            'low_gamma_low_lambda': int(np.sum((high_gamma == 0) & (high_lambda == 0))),
                            'high_gamma_low_lambda': int(np.sum((high_gamma == 1) & (high_lambda == 0))),
                            'low_gamma_high_lambda': int(np.sum((high_gamma == 0) & (high_lambda == 1))),
                            'high_gamma_high_lambda': int(np.sum((high_gamma == 1) & (high_lambda == 1)))
                        }
                    }
                    
                except Exception as e:
                    threshold_results[f'gamma_p{25+i*25}_lambda_p{25+j*25}'] = {
                        'error': str(e),
                        'gamma_threshold': gamma_thresh,
                        'lambda_threshold': lambda_thresh
                    }
        
        # Find best threshold combination
        best_threshold = None
        best_r_squared = -1
        
        for threshold_name, threshold_result in threshold_results.items():
            if 'r_squared' in threshold_result and threshold_result['r_squared'] > best_r_squared:
                best_r_squared = threshold_result['r_squared']
                best_threshold = threshold_name
        
        return {
            'threshold_combinations': threshold_results,
            'best_threshold_combination': best_threshold,
            'best_r_squared': best_r_squared
        }
    
    def _analyze_correlations(self,
                            y: pd.Series,
                            gamma1: pd.Series,
                            lambda2: pd.Series) -> Dict[str, Any]:
        """Analyze correlations between variables"""
        
        # Create interaction term
        interaction = gamma1 * lambda2
        
        # Calculate correlations
        correlations = {}
        
        # Pearson correlations
        correlations['pearson'] = {
            'y_gamma1': pearsonr(y, gamma1),
            'y_lambda2': pearsonr(y, lambda2),
            'y_interaction': pearsonr(y, interaction),
            'gamma1_lambda2': pearsonr(gamma1, lambda2)
        }
        
        # Spearman correlations
        correlations['spearman'] = {
            'y_gamma1': spearmanr(y, gamma1),
            'y_lambda2': spearmanr(y, lambda2),
            'y_interaction': spearmanr(y, interaction),
            'gamma1_lambda2': spearmanr(gamma1, lambda2)
        }
        
        # Format results
        formatted_correlations = {}
        for corr_type, corr_dict in correlations.items():
            formatted_correlations[corr_type] = {}
            for var_pair, (corr_coeff, p_value) in corr_dict.items():
                formatted_correlations[corr_type][var_pair] = {
                    'correlation': corr_coeff,
                    'p_value': p_value,
                    'significant': p_value < 0.05
                }
        
        return formatted_correlations
    
    def _analyze_regime_dependent_interactions(self,
                                             y: pd.Series,
                                             gamma1: pd.Series,
                                             lambda2: pd.Series) -> Dict[str, Any]:
        """Analyze interactions that depend on economic regimes"""
        
        # Define regimes based on dependent variable (e.g., high vs low yield periods)
        y_median = y.median()
        high_yield_regime = (y > y_median).astype(int)
        
        regime_results = {}
        
        # Analyze interaction in each regime
        for regime_value, regime_name in [(0, 'low_yield'), (1, 'high_yield')]:
            regime_mask = (high_yield_regime == regime_value)
            
            if np.sum(regime_mask) < 10:
                regime_results[regime_name] = {'error': 'Insufficient observations'}
                continue
            
            # Subset data for regime
            y_regime = y[regime_mask]
            gamma1_regime = gamma1[regime_mask]
            lambda2_regime = lambda2[regime_mask]
            
            # Create interaction term
            interaction_regime = gamma1_regime * lambda2_regime
            
            # Regression for this regime
            X_regime = pd.DataFrame({
                'gamma1': gamma1_regime,
                'lambda2': lambda2_regime,
                'interaction': interaction_regime
            })
            X_regime_const = sm.add_constant(X_regime)
            
            try:
                model_regime = OLS(y_regime, X_regime_const).fit()
                
                regime_results[regime_name] = {
                    'observations': int(np.sum(regime_mask)),
                    'r_squared': model_regime.rsquared,
                    'interaction_coefficient': model_regime.params['interaction'],
                    'interaction_pvalue': model_regime.pvalues['interaction'],
                    'interaction_significant': model_regime.pvalues['interaction'] < 0.05,
                    'main_effects': {
                        'gamma1_coeff': model_regime.params['gamma1'],
                        'lambda2_coeff': model_regime.params['lambda2'],
                        'gamma1_pvalue': model_regime.pvalues['gamma1'],
                        'lambda2_pvalue': model_regime.pvalues['lambda2']
                    }
                }
                
            except Exception as e:
                regime_results[regime_name] = {'error': str(e)}
        
        # Test for regime differences
        regime_comparison = self._compare_regime_interactions(regime_results)
        
        return {
            'regime_results': regime_results,
            'regime_comparison': regime_comparison
        }
    
    def _calculate_marginal_effects(self,
                                  gamma1: pd.Series,
                                  lambda2: pd.Series,
                                  interaction_coeff: float,
                                  gamma1_coeff: float,
                                  lambda2_coeff: float) -> Dict[str, Any]:
        """Calculate marginal effects of interaction"""
        
        # Marginal effect of gamma1 depends on lambda2 level
        # ∂y/∂γ₁ = β₁ + θ*λ₂
        marginal_gamma1 = gamma1_coeff + interaction_coeff * lambda2
        
        # Marginal effect of lambda2 depends on gamma1 level
        # ∂y/∂λ₂ = β₂ + θ*γ₁
        marginal_lambda2 = lambda2_coeff + interaction_coeff * gamma1
        
        return {
            'marginal_gamma1': {
                'mean': marginal_gamma1.mean(),
                'std': marginal_gamma1.std(),
                'min': marginal_gamma1.min(),
                'max': marginal_gamma1.max(),
                'percentiles': {
                    '25th': marginal_gamma1.quantile(0.25),
                    '50th': marginal_gamma1.quantile(0.50),
                    '75th': marginal_gamma1.quantile(0.75)
                }
            },
            'marginal_lambda2': {
                'mean': marginal_lambda2.mean(),
                'std': marginal_lambda2.std(),
                'min': marginal_lambda2.min(),
                'max': marginal_lambda2.max(),
                'percentiles': {
                    '25th': marginal_lambda2.quantile(0.25),
                    '50th': marginal_lambda2.quantile(0.50),
                    '75th': marginal_lambda2.quantile(0.75)
                }
            }
        }
    
    def _run_regression_diagnostics(self, model, X, y) -> Dict[str, Any]:
        """Run regression diagnostics"""
        
        diagnostics = {}
        
        # Heteroskedasticity tests
        try:
            bp_stat, bp_pvalue, _, _ = het_breuschpagan(model.resid, X)
            diagnostics['breusch_pagan'] = {
                'statistic': bp_stat,
                'p_value': bp_pvalue,
                'homoskedastic': bp_pvalue > 0.05
            }
        except Exception:
            diagnostics['breusch_pagan'] = {'error': 'Test failed'}
        
        try:
            white_stat, white_pvalue, _, _ = het_white(model.resid, X)
            diagnostics['white_test'] = {
                'statistic': white_stat,
                'p_value': white_pvalue,
                'homoskedastic': white_pvalue > 0.05
            }
        except Exception:
            diagnostics['white_test'] = {'error': 'Test failed'}
        
        # Durbin-Watson test for autocorrelation
        try:
            dw_stat = durbin_watson(model.resid)
            diagnostics['durbin_watson'] = {
                'statistic': dw_stat,
                'interpretation': 'positive_autocorr' if dw_stat < 1.5 else 'negative_autocorr' if dw_stat > 2.5 else 'no_autocorr'
            }
        except Exception:
            diagnostics['durbin_watson'] = {'error': 'Test failed'}
        
        # Normality of residuals
        try:
            from scipy.stats import jarque_bera
            jb_stat, jb_pvalue = jarque_bera(model.resid)
            diagnostics['jarque_bera'] = {
                'statistic': jb_stat,
                'p_value': jb_pvalue,
                'normal_residuals': jb_pvalue > 0.05
            }
        except Exception:
            diagnostics['jarque_bera'] = {'error': 'Test failed'}
        
        return diagnostics
    
    def _interpret_multiplicative_interaction(self,
                                            interaction_coeff: float,
                                            interaction_pvalue: float,
                                            gamma1_coeff: float,
                                            lambda2_coeff: float) -> Dict[str, str]:
        """Interpret multiplicative interaction results"""
        
        interpretation = {}
        
        # Statistical significance
        if interaction_pvalue < 0.01:
            interpretation['significance'] = "Highly significant interaction effect (p < 0.01)"
        elif interaction_pvalue < 0.05:
            interpretation['significance'] = "Significant interaction effect (p < 0.05)"
        else:
            interpretation['significance'] = "No significant interaction effect (p >= 0.05)"
        
        # Economic interpretation
        if interaction_pvalue < 0.05:
            if interaction_coeff > 0:
                interpretation['economic_meaning'] = "Positive interaction: Central bank reaction and confidence effects reinforce each other"
                interpretation['policy_implication'] = "When both central bank reaction and confidence effects are strong, their combined impact is amplified"
            else:
                interpretation['economic_meaning'] = "Negative interaction: Central bank reaction and confidence effects offset each other"
                interpretation['policy_implication'] = "Strong central bank reactions may be less effective when confidence effects are also strong"
        else:
            interpretation['economic_meaning'] = "No significant interaction between central bank reaction and confidence effects"
            interpretation['policy_implication'] = "Central bank reaction and confidence effects appear to operate independently"
        
        # Magnitude assessment
        abs_interaction = abs(interaction_coeff)
        abs_main_effects = abs(gamma1_coeff) + abs(lambda2_coeff)
        
        if abs_interaction > 0.5 * abs_main_effects:
            interpretation['magnitude'] = "Large interaction effect relative to main effects"
        elif abs_interaction > 0.1 * abs_main_effects:
            interpretation['magnitude'] = "Moderate interaction effect relative to main effects"
        else:
            interpretation['magnitude'] = "Small interaction effect relative to main effects"
        
        return interpretation
    
    def _compare_regime_interactions(self, regime_results: Dict[str, Any]) -> Dict[str, Any]:
        """Compare interaction effects across regimes"""
        
        comparison = {}
        
        # Extract interaction coefficients
        regime_coeffs = {}
        regime_pvalues = {}
        
        for regime_name, regime_result in regime_results.items():
            if 'interaction_coefficient' in regime_result:
                regime_coeffs[regime_name] = regime_result['interaction_coefficient']
                regime_pvalues[regime_name] = regime_result['interaction_pvalue']
        
        if len(regime_coeffs) >= 2:
            regime_names = list(regime_coeffs.keys())
            
            # Compare coefficients
            coeff_diff = regime_coeffs[regime_names[1]] - regime_coeffs[regime_names[0]]
            
            comparison['coefficient_difference'] = coeff_diff
            comparison['regime_coefficients'] = regime_coeffs
            comparison['regime_pvalues'] = regime_pvalues
            
            # Interpretation
            if abs(coeff_diff) > 0.1:  # Arbitrary threshold
                comparison['interpretation'] = f"Substantial difference in interaction effects between {regime_names[0]} and {regime_names[1]} regimes"
            else:
                comparison['interpretation'] = f"Similar interaction effects between {regime_names[0]} and {regime_names[1]} regimes"
        
        return comparison
    
    def get_interaction_summary(self) -> Dict[str, Any]:
        """Get summary of all interaction analyses"""
        
        if self.interaction_results is None:
            raise ValueError("No interaction analysis results available. Run analyze_interactions() first.")
        
        summary = {
            'data_summary': self.interaction_results['data_info'],
            'key_findings': {},
            'statistical_significance': {},
            'economic_interpretation': {}
        }
        
        # Multiplicative interaction summary
        if 'multiplicative' in self.interaction_results['interaction_analyses']:
            mult_results = self.interaction_results['interaction_analyses']['multiplicative']
            
            summary['key_findings']['multiplicative_interaction'] = {
                'coefficient': mult_results['interaction_coefficient']['coefficient'],
                'significant': mult_results['interaction_coefficient']['significant_5pct'],
                'r_squared': mult_results['model_summary']['r_squared']
            }
            
            summary['statistical_significance']['multiplicative'] = mult_results['interaction_coefficient']['p_value']
            summary['economic_interpretation']['multiplicative'] = mult_results['interpretation']['economic_meaning']
        
        # Threshold interaction summary
        if 'threshold' in self.interaction_results['interaction_analyses']:
            threshold_results = self.interaction_results['interaction_analyses']['threshold']
            
            summary['key_findings']['threshold_interaction'] = {
                'best_combination': threshold_results['best_threshold_combination'],
                'best_r_squared': threshold_results['best_r_squared']
            }
        
        # Correlation summary
        if 'correlation_analysis' in self.interaction_results:
            corr_results = self.interaction_results['correlation_analysis']
            
            summary['key_findings']['correlations'] = {
                'gamma1_lambda2_correlation': corr_results['pearson']['gamma1_lambda2']['correlation'],
                'interaction_y_correlation': corr_results['pearson']['y_interaction']['correlation']
            }
        
        return summary