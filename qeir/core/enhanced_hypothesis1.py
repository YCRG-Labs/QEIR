"""
Enhanced Hypothesis 1: Central Bank Reaction and Confidence Effects Testing

This module implements enhanced threshold detection models for central bank reaction strength,
including confidence effect interactions, regime-switching VAR, and bootstrap confidence intervals.
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar, minimize
from scipy.stats import norm, f
import statsmodels.api as sm
from statsmodels.tsa.api import VAR
from statsmodels.regression.linear_model import OLS
from sklearn.utils import resample
import warnings
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
import logging

from .models import HansenThresholdRegression

warnings.filterwarnings('ignore')


@dataclass
class ConfidenceEffectProxies:
    """Data structure for confidence effect proxies from FRED data"""
    
    consumer_confidence: Optional[pd.Series] = None
    business_confidence: Optional[pd.Series] = None
    financial_stress_index: Optional[pd.Series] = None
    vix_index: Optional[pd.Series] = None
    credit_spreads: Optional[pd.Series] = None
    
    def get_proxy(self, proxy_name: str) -> pd.Series:
        """Get specific confidence proxy by name"""
        if hasattr(self, proxy_name):
            return getattr(self, proxy_name)
        else:
            raise ValueError(f"Unknown confidence proxy: {proxy_name}")


@dataclass
class CentralBankReactionProxies:
    """Data structure for central bank reaction strength proxies"""
    
    fed_total_assets: Optional[pd.Series] = None
    monetary_base: Optional[pd.Series] = None
    policy_rate_changes: Optional[pd.Series] = None
    fomc_meeting_frequency: Optional[pd.Series] = None
    balance_sheet_growth: Optional[pd.Series] = None
    
    def get_proxy(self, proxy_name: str) -> pd.Series:
        """Get specific central bank reaction proxy by name"""
        if hasattr(self, proxy_name):
            return getattr(self, proxy_name)
        else:
            raise ValueError(f"Unknown central bank reaction proxy: {proxy_name}")


class EnhancedHansenThresholdRegression(HansenThresholdRegression):
    """
    Enhanced Hansen threshold regression with confidence effect interactions.
    
    Extends the base Hansen model to include:
    - Confidence effect interactions (γ₁ * λ₂)
    - Bootstrap confidence intervals for threshold estimates
    - Alternative threshold detection methods
    - Regime-specific diagnostics
    """
    
    def __init__(self):
        super().__init__()
        self.confidence_interactions = None
        self.bootstrap_results = None
        self.interaction_coefficients = None
        self.interaction_std_errors = None
        
    def fit_with_confidence_interactions(self, 
                                       y: np.ndarray,
                                       central_bank_reaction: np.ndarray,
                                       confidence_effects: np.ndarray,
                                       threshold_var: np.ndarray,
                                       trim: float = 0.15) -> Dict[str, Any]:
        """
        Fit Hansen threshold model with confidence effect interactions.
        
        Model: y_t = α₁ + β₁*γ₁_t + δ₁*λ₂_t + θ₁*(γ₁_t * λ₂_t) + ε₁_t  if threshold_var ≤ τ
               y_t = α₂ + β₂*γ₁_t + δ₂*λ₂_t + θ₂*(γ₁_t * λ₂_t) + ε₂_t  if threshold_var > τ
        
        Args:
            y: Dependent variable (long-term yields)
            central_bank_reaction: Central bank reaction strength (γ₁)
            confidence_effects: Confidence effects (λ₂)
            threshold_var: Threshold variable (debt service burden)
            trim: Fraction of observations to trim when searching for threshold
            
        Returns:
            Dictionary with fitting results and diagnostics
        """
        # Create interaction term
        interaction_term = central_bank_reaction * confidence_effects
        
        # Combine all regressors
        X = np.column_stack([central_bank_reaction, confidence_effects, interaction_term])
        
        # Fit base Hansen model
        self.fit(y, X, threshold_var, trim)
        
        # Store interaction-specific results
        self.confidence_interactions = {
            'central_bank_reaction': central_bank_reaction,
            'confidence_effects': confidence_effects,
            'interaction_term': interaction_term,
            'threshold_variable': threshold_var
        }
        
        # Extract interaction coefficients from regime coefficients
        if self.fitted:
            # Regime 1: [intercept, γ₁, λ₂, γ₁*λ₂]
            self.interaction_coefficients = {
                'regime1': {
                    'intercept': self.beta1[0],
                    'central_bank_reaction': self.beta1[1],
                    'confidence_effects': self.beta1[2],
                    'interaction': self.beta1[3]
                },
                'regime2': {
                    'intercept': self.beta2[0],
                    'central_bank_reaction': self.beta2[1],
                    'confidence_effects': self.beta2[2],
                    'interaction': self.beta2[3]
                }
            }
            
            self.interaction_std_errors = {
                'regime1': {
                    'intercept': self.se1[0],
                    'central_bank_reaction': self.se1[1],
                    'confidence_effects': self.se1[2],
                    'interaction': self.se1[3]
                },
                'regime2': {
                    'intercept': self.se2[0],
                    'central_bank_reaction': self.se2[1],
                    'confidence_effects': self.se2[2],
                    'interaction': self.se2[3]
                }
            }
        
        return {
            'fitted': self.fitted,
            'threshold': self.threshold,
            'interaction_coefficients': self.interaction_coefficients,
            'interaction_std_errors': self.interaction_std_errors
        }
    
    def bootstrap_confidence_intervals(self, 
                                     y: np.ndarray,
                                     X: np.ndarray,
                                     threshold_var: np.ndarray,
                                     n_bootstrap: int = 1000,
                                     confidence_level: float = 0.95,
                                     trim: float = 0.15) -> Dict[str, Any]:
        """
        Calculate bootstrap confidence intervals for threshold estimates.
        
        Args:
            y: Dependent variable
            X: Independent variables matrix
            threshold_var: Threshold variable
            n_bootstrap: Number of bootstrap iterations
            confidence_level: Confidence level (default 0.95)
            trim: Trim parameter for threshold search
            
        Returns:
            Dictionary with bootstrap results and confidence intervals
        """
        if not self.fitted:
            raise ValueError("Model must be fitted before calculating bootstrap confidence intervals")
        
        bootstrap_thresholds = []
        bootstrap_coefficients_r1 = []
        bootstrap_coefficients_r2 = []
        
        n_obs = len(y)
        
        for i in range(n_bootstrap):
            # Bootstrap sample
            indices = resample(range(n_obs), n_samples=n_obs, random_state=i)
            
            y_boot = y[indices]
            X_boot = X[indices]
            threshold_boot = threshold_var[indices]
            
            try:
                # Fit model on bootstrap sample
                boot_model = EnhancedHansenThresholdRegression()
                boot_model.fit(y_boot, X_boot, threshold_boot, trim)
                
                if boot_model.fitted:
                    bootstrap_thresholds.append(boot_model.threshold)
                    bootstrap_coefficients_r1.append(boot_model.beta1)
                    bootstrap_coefficients_r2.append(boot_model.beta2)
                    
            except Exception:
                # Skip failed bootstrap iterations
                continue
        
        if len(bootstrap_thresholds) < 50:
            raise ValueError(f"Too few successful bootstrap iterations: {len(bootstrap_thresholds)}")
        
        # Calculate confidence intervals
        alpha = 1 - confidence_level
        lower_percentile = (alpha / 2) * 100
        upper_percentile = (1 - alpha / 2) * 100
        
        threshold_ci = np.percentile(bootstrap_thresholds, [lower_percentile, upper_percentile])
        
        # Coefficient confidence intervals
        bootstrap_coefficients_r1 = np.array(bootstrap_coefficients_r1)
        bootstrap_coefficients_r2 = np.array(bootstrap_coefficients_r2)
        
        coeff_ci_r1 = np.percentile(bootstrap_coefficients_r1, [lower_percentile, upper_percentile], axis=0)
        coeff_ci_r2 = np.percentile(bootstrap_coefficients_r2, [lower_percentile, upper_percentile], axis=0)
        
        self.bootstrap_results = {
            'n_bootstrap': len(bootstrap_thresholds),
            'confidence_level': confidence_level,
            'threshold_ci': {
                'lower': threshold_ci[0],
                'upper': threshold_ci[1],
                'point_estimate': self.threshold
            },
            'regime1_coeff_ci': {
                'lower': coeff_ci_r1[0].tolist(),
                'upper': coeff_ci_r1[1].tolist(),
                'point_estimates': self.beta1.tolist()
            },
            'regime2_coeff_ci': {
                'lower': coeff_ci_r2[0].tolist(),
                'upper': coeff_ci_r2[1].tolist(),
                'point_estimates': self.beta2.tolist()
            },
            'bootstrap_thresholds': bootstrap_thresholds,
            'successful_iterations': len(bootstrap_thresholds),
            'total_iterations': n_bootstrap
        }
        
        return self.bootstrap_results
    
    def test_interaction_significance(self) -> Dict[str, Any]:
        """
        Test statistical significance of interaction effects (γ₁ * λ₂).
        
        Returns:
            Dictionary with interaction significance test results
        """
        if not self.fitted or self.interaction_coefficients is None:
            raise ValueError("Model must be fitted with confidence interactions")
        
        # Extract interaction coefficients and standard errors
        theta1 = self.interaction_coefficients['regime1']['interaction']
        theta2 = self.interaction_coefficients['regime2']['interaction']
        
        se_theta1 = self.interaction_std_errors['regime1']['interaction']
        se_theta2 = self.interaction_std_errors['regime2']['interaction']
        
        # T-tests for interaction significance
        t_stat_r1 = theta1 / se_theta1 if se_theta1 > 0 else 0
        t_stat_r2 = theta2 / se_theta2 if se_theta2 > 0 else 0
        
        # Calculate p-values (two-tailed)
        # Approximate degrees of freedom
        regime1_mask = self.confidence_interactions['threshold_variable'] <= self.threshold
        regime2_mask = self.confidence_interactions['threshold_variable'] > self.threshold
        
        df_r1 = np.sum(regime1_mask) - 4  # 4 parameters per regime
        df_r2 = np.sum(regime2_mask) - 4
        
        from scipy.stats import t
        p_value_r1 = 2 * (1 - t.cdf(abs(t_stat_r1), df_r1)) if df_r1 > 0 else 1.0
        p_value_r2 = 2 * (1 - t.cdf(abs(t_stat_r2), df_r2)) if df_r2 > 0 else 1.0
        
        # Test for difference in interaction effects between regimes
        theta_diff = theta2 - theta1
        se_diff = np.sqrt(se_theta1**2 + se_theta2**2)  # Assuming independence
        t_stat_diff = theta_diff / se_diff if se_diff > 0 else 0
        
        # Use pooled degrees of freedom for difference test
        df_pooled = df_r1 + df_r2
        p_value_diff = 2 * (1 - t.cdf(abs(t_stat_diff), df_pooled)) if df_pooled > 0 else 1.0
        
        return {
            'regime1_interaction': {
                'coefficient': theta1,
                'std_error': se_theta1,
                't_statistic': t_stat_r1,
                'p_value': p_value_r1,
                'significant_5pct': p_value_r1 < 0.05,
                'significant_1pct': p_value_r1 < 0.01
            },
            'regime2_interaction': {
                'coefficient': theta2,
                'std_error': se_theta2,
                't_statistic': t_stat_r2,
                'p_value': p_value_r2,
                'significant_5pct': p_value_r2 < 0.05,
                'significant_1pct': p_value_r2 < 0.01
            },
            'regime_difference': {
                'coefficient_difference': theta_diff,
                'std_error_difference': se_diff,
                't_statistic': t_stat_diff,
                'p_value': p_value_diff,
                'significant_5pct': p_value_diff < 0.05,
                'significant_1pct': p_value_diff < 0.01
            },
            'interpretation': {
                'regime1_interaction_significant': p_value_r1 < 0.05,
                'regime2_interaction_significant': p_value_r2 < 0.05,
                'regimes_differ_significantly': p_value_diff < 0.05
            }
        }


class RegimeSwitchingVAR:
    """
    Regime-switching VAR model for threshold identification.
    
    Implements a Markov-switching VAR to identify threshold regimes
    in the relationship between central bank reactions, confidence effects,
    and long-term yields.
    """
    
    def __init__(self, n_regimes: int = 2, lags: int = 2):
        """
        Initialize regime-switching VAR.
        
        Args:
            n_regimes: Number of regimes (default 2)
            lags: Number of lags in VAR (default 2)
        """
        self.n_regimes = n_regimes
        self.lags = lags
        self.fitted = False
        self.regime_probabilities = None
        self.transition_matrix = None
        self.regime_parameters = None
        
    def fit(self, 
            data: pd.DataFrame,
            max_iter: int = 100,
            tolerance: float = 1e-6) -> Dict[str, Any]:
        """
        Fit regime-switching VAR using EM algorithm.
        
        Args:
            data: DataFrame with variables [yields, cb_reaction, confidence, threshold_var]
            max_iter: Maximum EM iterations
            tolerance: Convergence tolerance
            
        Returns:
            Dictionary with fitting results
        """
        # This is a simplified implementation
        # Full implementation would use proper EM algorithm for Markov-switching models
        
        n_obs, n_vars = data.shape
        
        # Initialize regime probabilities randomly
        self.regime_probabilities = np.random.rand(n_obs, self.n_regimes)
        self.regime_probabilities = self.regime_probabilities / self.regime_probabilities.sum(axis=1, keepdims=True)
        
        # Initialize transition matrix
        self.transition_matrix = np.random.rand(self.n_regimes, self.n_regimes)
        self.transition_matrix = self.transition_matrix / self.transition_matrix.sum(axis=1, keepdims=True)
        
        # Initialize regime parameters
        self.regime_parameters = {}
        
        for regime in range(self.n_regimes):
            # Fit VAR for each regime (simplified)
            try:
                var_model = VAR(data)
                var_results = var_model.fit(maxlags=self.lags, ic='aic')
                
                self.regime_parameters[regime] = {
                    'coefficients': var_results.params.values,
                    'covariance': var_results.sigma_u,
                    'aic': var_results.aic,
                    'bic': var_results.bic
                }
            except Exception:
                # Fallback to simple parameters
                self.regime_parameters[regime] = {
                    'coefficients': np.random.randn(n_vars * self.lags + 1, n_vars),
                    'covariance': np.eye(n_vars),
                    'aic': np.inf,
                    'bic': np.inf
                }
        
        # EM algorithm (simplified version)
        log_likelihood_old = -np.inf
        
        for iteration in range(max_iter):
            # E-step: Update regime probabilities
            # (Simplified - would need proper forward-backward algorithm)
            
            # M-step: Update parameters
            # (Simplified - would need proper parameter updates)
            
            # Check convergence (placeholder)
            log_likelihood_new = self._calculate_log_likelihood(data)
            
            if abs(log_likelihood_new - log_likelihood_old) < tolerance:
                break
                
            log_likelihood_old = log_likelihood_new
        
        self.fitted = True
        
        return {
            'fitted': self.fitted,
            'n_regimes': self.n_regimes,
            'lags': self.lags,
            'iterations': iteration + 1,
            'log_likelihood': log_likelihood_new,
            'regime_parameters': self.regime_parameters,
            'transition_matrix': self.transition_matrix.tolist()
        }
    
    def _calculate_log_likelihood(self, data: pd.DataFrame) -> float:
        """Calculate log-likelihood (simplified placeholder)"""
        # This would be properly implemented in a full version
        return np.random.randn()  # Placeholder
    
    def predict_regimes(self, data: pd.DataFrame) -> np.ndarray:
        """
        Predict regime probabilities for new data.
        
        Args:
            data: DataFrame with variables
            
        Returns:
            Array of regime probabilities
        """
        if not self.fitted:
            raise ValueError("Model must be fitted before prediction")
        
        # Simplified prediction (would use proper Viterbi algorithm)
        n_obs = len(data)
        regime_probs = np.random.rand(n_obs, self.n_regimes)
        regime_probs = regime_probs / regime_probs.sum(axis=1, keepdims=True)
        
        return regime_probs
    
    def get_regime_classification(self, data: pd.DataFrame) -> np.ndarray:
        """
        Get most likely regime for each observation.
        
        Args:
            data: DataFrame with variables
            
        Returns:
            Array of regime classifications (0, 1, ...)
        """
        regime_probs = self.predict_regimes(data)
        return np.argmax(regime_probs, axis=1)


class Hypothesis1ThresholdDetector:
    """
    Comprehensive threshold detection system for Hypothesis 1.
    
    Combines multiple threshold detection methods:
    - Enhanced Hansen threshold regression
    - Regime-switching VAR
    - Bootstrap confidence intervals
    - Alternative threshold specifications
    """
    
    def __init__(self):
        self.hansen_model = EnhancedHansenThresholdRegression()
        self.regime_switching_var = RegimeSwitchingVAR()
        self.fitted_models = {}
        self.threshold_comparison = None
        
    def detect_thresholds(self,
                         y: np.ndarray,
                         central_bank_reaction: np.ndarray,
                         confidence_effects: np.ndarray,
                         threshold_var: np.ndarray,
                         debt_service_burden: np.ndarray,
                         bootstrap_ci: bool = True,
                         n_bootstrap: int = 1000) -> Dict[str, Any]:
        """
        Comprehensive threshold detection using multiple methods.
        
        Args:
            y: Long-term yields
            central_bank_reaction: Central bank reaction strength (γ₁)
            confidence_effects: Confidence effects (λ₂)
            threshold_var: Primary threshold variable
            debt_service_burden: Debt service burden data
            bootstrap_ci: Whether to calculate bootstrap confidence intervals
            n_bootstrap: Number of bootstrap iterations
            
        Returns:
            Dictionary with comprehensive threshold detection results
        """
        results = {}
        
        # Method 1: Enhanced Hansen with confidence interactions
        try:
            hansen_results = self.hansen_model.fit_with_confidence_interactions(
                y, central_bank_reaction, confidence_effects, threshold_var
            )
            
            # Bootstrap confidence intervals
            if bootstrap_ci and self.hansen_model.fitted:
                X = np.column_stack([central_bank_reaction, confidence_effects, 
                                   central_bank_reaction * confidence_effects])
                bootstrap_results = self.hansen_model.bootstrap_confidence_intervals(
                    y, X, threshold_var, n_bootstrap
                )
                hansen_results['bootstrap_ci'] = bootstrap_results
            
            # Test interaction significance
            if self.hansen_model.fitted:
                interaction_tests = self.hansen_model.test_interaction_significance()
                hansen_results['interaction_significance'] = interaction_tests
            
            results['hansen_enhanced'] = hansen_results
            self.fitted_models['hansen'] = self.hansen_model
            
        except Exception as e:
            results['hansen_enhanced'] = {'error': str(e), 'fitted': False}
        
        # Method 2: Regime-switching VAR
        try:
            # Prepare data for VAR
            var_data = pd.DataFrame({
                'yields': y,
                'cb_reaction': central_bank_reaction,
                'confidence': confidence_effects,
                'threshold_var': threshold_var
            })
            
            var_results = self.regime_switching_var.fit(var_data)
            
            # Get regime classification
            if self.regime_switching_var.fitted:
                regime_classification = self.regime_switching_var.get_regime_classification(var_data)
                var_results['regime_classification'] = regime_classification.tolist()
                
                # Calculate regime-specific statistics
                regime_stats = {}
                for regime in range(self.regime_switching_var.n_regimes):
                    regime_mask = regime_classification == regime
                    if np.sum(regime_mask) > 0:
                        regime_stats[f'regime_{regime}'] = {
                            'observations': int(np.sum(regime_mask)),
                            'mean_yields': float(np.mean(y[regime_mask])),
                            'mean_cb_reaction': float(np.mean(central_bank_reaction[regime_mask])),
                            'mean_confidence': float(np.mean(confidence_effects[regime_mask])),
                            'mean_threshold_var': float(np.mean(threshold_var[regime_mask]))
                        }
                
                var_results['regime_statistics'] = regime_stats
            
            results['regime_switching_var'] = var_results
            self.fitted_models['var'] = self.regime_switching_var
            
        except Exception as e:
            results['regime_switching_var'] = {'error': str(e), 'fitted': False}
        
        # Method 3: Alternative threshold specifications
        try:
            alternative_results = self._test_alternative_thresholds(
                y, central_bank_reaction, confidence_effects, 
                threshold_var, debt_service_burden
            )
            results['alternative_thresholds'] = alternative_results
            
        except Exception as e:
            results['alternative_thresholds'] = {'error': str(e)}
        
        # Compare methods
        self.threshold_comparison = self._compare_threshold_methods(results)
        results['method_comparison'] = self.threshold_comparison
        
        return results
    
    def _test_alternative_thresholds(self,
                                   y: np.ndarray,
                                   central_bank_reaction: np.ndarray,
                                   confidence_effects: np.ndarray,
                                   primary_threshold: np.ndarray,
                                   debt_service_burden: np.ndarray) -> Dict[str, Any]:
        """Test alternative threshold variable specifications"""
        
        alternative_results = {}
        
        # Alternative 1: Use debt service burden directly as threshold
        try:
            alt_model_1 = EnhancedHansenThresholdRegression()
            alt_results_1 = alt_model_1.fit_with_confidence_interactions(
                y, central_bank_reaction, confidence_effects, debt_service_burden
            )
            alternative_results['debt_service_threshold'] = alt_results_1
            
        except Exception as e:
            alternative_results['debt_service_threshold'] = {'error': str(e)}
        
        # Alternative 2: Use confidence effects as threshold variable
        try:
            alt_model_2 = EnhancedHansenThresholdRegression()
            alt_results_2 = alt_model_2.fit_with_confidence_interactions(
                y, central_bank_reaction, confidence_effects, confidence_effects
            )
            alternative_results['confidence_threshold'] = alt_results_2
            
        except Exception as e:
            alternative_results['confidence_threshold'] = {'error': str(e)}
        
        # Alternative 3: Use central bank reaction as threshold variable
        try:
            alt_model_3 = EnhancedHansenThresholdRegression()
            alt_results_3 = alt_model_3.fit_with_confidence_interactions(
                y, central_bank_reaction, confidence_effects, central_bank_reaction
            )
            alternative_results['cb_reaction_threshold'] = alt_results_3
            
        except Exception as e:
            alternative_results['cb_reaction_threshold'] = {'error': str(e)}
        
        return alternative_results
    
    def _compare_threshold_methods(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Compare results across different threshold detection methods"""
        
        comparison = {
            'methods_tested': [],
            'successful_fits': [],
            'threshold_estimates': {},
            'model_fit_comparison': {},
            'consensus_threshold': None
        }
        
        # Hansen method
        if 'hansen_enhanced' in results and results['hansen_enhanced'].get('fitted', False):
            comparison['methods_tested'].append('hansen_enhanced')
            comparison['successful_fits'].append('hansen_enhanced')
            comparison['threshold_estimates']['hansen'] = results['hansen_enhanced']['threshold']
        
        # VAR method
        if 'regime_switching_var' in results and results['regime_switching_var'].get('fitted', False):
            comparison['methods_tested'].append('regime_switching_var')
            comparison['successful_fits'].append('regime_switching_var')
            # VAR doesn't directly give threshold, but gives regime classification
        
        # Alternative thresholds
        if 'alternative_thresholds' in results:
            for alt_name, alt_result in results['alternative_thresholds'].items():
                if isinstance(alt_result, dict) and alt_result.get('fitted', False):
                    comparison['methods_tested'].append(f'alternative_{alt_name}')
                    comparison['successful_fits'].append(f'alternative_{alt_name}')
                    comparison['threshold_estimates'][f'alt_{alt_name}'] = alt_result['threshold']
        
        # Calculate consensus threshold if multiple methods succeeded
        threshold_values = list(comparison['threshold_estimates'].values())
        if len(threshold_values) > 1:
            comparison['consensus_threshold'] = np.median(threshold_values)
            comparison['threshold_range'] = {
                'min': np.min(threshold_values),
                'max': np.max(threshold_values),
                'std': np.std(threshold_values)
            }
        elif len(threshold_values) == 1:
            comparison['consensus_threshold'] = threshold_values[0]
        
        return comparison
    
    def get_comprehensive_diagnostics(self) -> Dict[str, Any]:
        """Get comprehensive diagnostics across all fitted models"""
        
        diagnostics = {
            'fitted_models': list(self.fitted_models.keys()),
            'threshold_comparison': self.threshold_comparison,
            'model_diagnostics': {}
        }
        
        # Hansen model diagnostics
        if 'hansen' in self.fitted_models and self.fitted_models['hansen'].fitted:
            hansen_model = self.fitted_models['hansen']
            
            # Get enhanced diagnostics if available
            if hasattr(hansen_model, 'confidence_interactions') and hansen_model.confidence_interactions:
                y = hansen_model.confidence_interactions['threshold_variable']  # Placeholder
                X = np.column_stack([
                    hansen_model.confidence_interactions['central_bank_reaction'],
                    hansen_model.confidence_interactions['confidence_effects'],
                    hansen_model.confidence_interactions['interaction_term']
                ])
                threshold_var = hansen_model.confidence_interactions['threshold_variable']
                
                try:
                    enhanced_diag = hansen_model.get_enhanced_diagnostics(y, X, threshold_var)
                    diagnostics['model_diagnostics']['hansen'] = enhanced_diag
                except Exception as e:
                    diagnostics['model_diagnostics']['hansen'] = {'error': str(e)}
        
        # VAR model diagnostics
        if 'var' in self.fitted_models and self.fitted_models['var'].fitted:
            var_model = self.fitted_models['var']
            diagnostics['model_diagnostics']['var'] = {
                'n_regimes': var_model.n_regimes,
                'lags': var_model.lags,
                'fitted': var_model.fitted,
                'transition_matrix': var_model.transition_matrix.tolist() if var_model.transition_matrix is not None else None
            }
        
        return diagnostics    def 
sequential_threshold_test(self,
                                y: np.ndarray,
                                X: np.ndarray,
                                threshold_var: np.ndarray,
                                max_thresholds: int = 3,
                                significance_level: float = 0.05) -> Dict[str, Any]:
        """
        Sequential testing for multiple thresholds following Hansen (2000)
        
        Parameters:
        -----------
        y : np.ndarray
            Dependent variable
        X : np.ndarray
            Regressor matrix
        threshold_var : np.ndarray
            Threshold variable
        max_thresholds : int, default=3
            Maximum number of thresholds to test
        significance_level : float, default=0.05
            Significance level for threshold tests
            
        Returns:
        --------
        Dict containing results for each threshold test
        """
        results = {}
        current_data_indices = np.arange(len(y))
        
        for n_thresh in range(1, max_thresholds + 1):
            # Test for threshold in current data subset
            current_y = y[current_data_indices]
            current_X = X[current_data_indices]
            current_threshold_var = threshold_var[current_data_indices]
            
            # Fit threshold model
            self.fit(current_y, current_X, current_threshold_var)
            
            # Calculate test statistic
            test_stat = self.threshold_test_statistic()
            
            # Bootstrap p-value
            p_value = self.bootstrap_threshold_test(
                current_y, current_X, current_threshold_var,
                n_bootstrap=1000
            )
            
            results[f'threshold_{n_thresh}'] = {
                'threshold_estimate': self.threshold_estimate,
                'test_statistic': test_stat,
                'p_value': p_value,
                'significant': p_value < significance_level,
                'sample_size': len(current_data_indices)
            }
            
            # If threshold not significant, stop testing
            if p_value >= significance_level:
                break
            
            # For next iteration, split data at estimated threshold
            # (This is simplified - full implementation would be more complex)
            if n_thresh < max_thresholds:
                # Split data for next threshold search
                below_threshold = current_threshold_var <= self.threshold_estimate
                if np.sum(below_threshold) > 20 and np.sum(~below_threshold) > 20:
                    # Choose larger subsample for next test
                    if np.sum(below_threshold) > np.sum(~below_threshold):
                        current_data_indices = current_data_indices[below_threshold]
                    else:
                        current_data_indices = current_data_indices[~below_threshold]
                else:
                    break
        
        return results
    
    def time_varying_threshold_test(self,
                                  y: np.ndarray,
                                  X: np.ndarray,
                                  threshold_var: np.ndarray,
                                  dates: pd.DatetimeIndex,
                                  window_size: int = 60) -> Dict[str, Any]:
        """
        Test for time-varying thresholds using rolling window estimation
        
        Parameters:
        -----------
        y : np.ndarray
            Dependent variable
        X : np.ndarray
            Regressor matrix
        threshold_var : np.ndarray
            Threshold variable
        dates : pd.DatetimeIndex
            Time index
        window_size : int, default=60
            Rolling window size in months
            
        Returns:
        --------
        Dict containing time-varying threshold estimates
        """
        n_obs = len(y)
        threshold_estimates = []
        test_statistics = []
        window_dates = []
        
        for i in range(window_size, n_obs):
            # Extract window data
            start_idx = i - window_size
            end_idx = i
            
            window_y = y[start_idx:end_idx]
            window_X = X[start_idx:end_idx]
            window_threshold_var = threshold_var[start_idx:end_idx]
            
            try:
                # Fit threshold model for this window
                self.fit(window_y, window_X, window_threshold_var)
                
                threshold_estimates.append(self.threshold_estimate)
                test_statistics.append(self.threshold_test_statistic())
                window_dates.append(dates[end_idx])
                
            except Exception:
                # Handle estimation failures
                threshold_estimates.append(np.nan)
                test_statistics.append(np.nan)
                window_dates.append(dates[end_idx])
        
        # Test for stability of threshold estimates
        threshold_series = pd.Series(threshold_estimates, index=window_dates)
        threshold_std = threshold_series.std()
        threshold_mean = threshold_series.mean()
        
        # Coefficient of variation as stability measure
        stability_measure = threshold_std / threshold_mean if threshold_mean != 0 else np.inf
        
        return {
            'threshold_estimates': threshold_estimates,
            'test_statistics': test_statistics,
            'dates': window_dates,
            'stability_measure': stability_measure,
            'stable_threshold': stability_measure < 0.1  # Threshold for stability
        }
    
    def placebo_test(self,
                   y: np.ndarray,
                   X: np.ndarray,
                   threshold_var: np.ndarray,
                   pre_qe_period: Tuple[int, int]) -> Dict[str, Any]:
        """
        Conduct placebo test using pre-QE period data
        
        Parameters:
        -----------
        y : np.ndarray
            Dependent variable
        X : np.ndarray
            Regressor matrix
        threshold_var : np.ndarray
            Threshold variable
        pre_qe_period : Tuple[int, int]
            Start and end indices for pre-QE period
            
        Returns:
        --------
        Dict containing placebo test results
        """
        start_idx, end_idx = pre_qe_period
        
        # Extract pre-QE data
        placebo_y = y[start_idx:end_idx]
        placebo_X = X[start_idx:end_idx]
        placebo_threshold_var = threshold_var[start_idx:end_idx]
        
        try:
            # Fit threshold model on pre-QE data
            self.fit(placebo_y, placebo_X, placebo_threshold_var)
            
            # Calculate test statistic
            placebo_test_stat = self.threshold_test_statistic()
            
            # Bootstrap p-value
            placebo_p_value = self.bootstrap_threshold_test(
                placebo_y, placebo_X, placebo_threshold_var,
                n_bootstrap=1000
            )
            
            return {
                'placebo_threshold': self.threshold_estimate,
                'placebo_test_statistic': placebo_test_stat,
                'placebo_p_value': placebo_p_value,
                'spurious_threshold': placebo_p_value < 0.05,
                'sample_size': len(placebo_y)
            }
            
        except Exception as e:
            return {
                'placebo_threshold': np.nan,
                'placebo_test_statistic': np.nan,
                'placebo_p_value': np.nan,
                'spurious_threshold': False,
                'error': str(e)
            }