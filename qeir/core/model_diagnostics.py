"""
Statistical Model Diagnostics and Robustness Testing Framework

This module implements comprehensive diagnostic tests for model assumptions and specification,
bootstrap procedures for robust inference, and alternative estimation methods for robustness.
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Optional, Any, Tuple, Union, Callable
from dataclasses import dataclass, field
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Statistical libraries
from scipy import stats
from scipy.stats import jarque_bera, shapiro, anderson, kstest, normaltest
from statsmodels.stats.diagnostic import (
    het_breuschpagan, het_white, acorr_breusch_godfrey,
    linear_harvey_collier, het_arch, acorr_ljungbox
)
from statsmodels.stats.stattools import durbin_watson, omni_normtest
from statsmodels.tsa.stattools import adfuller, kpss, coint
from statsmodels.stats.outliers_influence import OLSInfluence
import matplotlib.pyplot as plt
import seaborn as sns

# Bootstrap libraries
from sklearn.utils import resample
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# Optional bootstrap libraries
try:
    from arch.bootstrap import IIDBootstrap, StationaryBootstrap, CircularBlockBootstrap
    ARCH_AVAILABLE = True
except ImportError:
    ARCH_AVAILABLE = False
    # Create dummy classes for when arch is not available
    class IIDBootstrap:
        def __init__(self, *args, **kwargs):
            raise ImportError("arch package not available. Install with: pip install arch")
        
        def bootstrap(self, *args, **kwargs):
            raise ImportError("arch package not available. Install with: pip install arch")
    
    class StationaryBootstrap:
        def __init__(self, *args, **kwargs):
            raise ImportError("arch package not available. Install with: pip install arch")
        
        def bootstrap(self, *args, **kwargs):
            raise ImportError("arch package not available. Install with: pip install arch")
    
    class CircularBlockBootstrap:
        def __init__(self, *args, **kwargs):
            raise ImportError("arch package not available. Install with: pip install arch")
        
        def bootstrap(self, *args, **kwargs):
            raise ImportError("arch package not available. Install with: pip install arch")


@dataclass
class DiagnosticConfig:
    """Configuration for model diagnostics and robustness testing"""
    
    # Normality tests
    normality_tests: List[str] = field(default_factory=lambda: [
        'jarque_bera', 'shapiro', 'anderson', 'kolmogorov_smirnov', 'dagostino'
    ])
    normality_alpha: float = 0.05
    
    # Heteroskedasticity tests
    heteroskedasticity_tests: List[str] = field(default_factory=lambda: [
        'breusch_pagan', 'white', 'arch'
    ])
    het_alpha: float = 0.05
    
    # Autocorrelation tests
    autocorrelation_tests: List[str] = field(default_factory=lambda: [
        'breusch_godfrey', 'ljung_box', 'durbin_watson'
    ])
    autocorr_lags: int = 10
    autocorr_alpha: float = 0.05
    
    # Linearity tests
    linearity_tests: List[str] = field(default_factory=lambda: [
        'harvey_collier', 'rainbow'
    ])
    linearity_alpha: float = 0.05
    
    # Stationarity tests
    stationarity_tests: List[str] = field(default_factory=lambda: [
        'adf', 'kpss', 'phillips_perron'
    ])
    stationarity_alpha: float = 0.05
    
    # Bootstrap settings
    bootstrap_method: str = 'iid'  # Options: iid, stationary, circular_block
    bootstrap_samples: int = 1000
    bootstrap_block_size: Optional[int] = None
    bootstrap_confidence_levels: List[float] = field(default_factory=lambda: [0.05, 0.95])
    
    # Robustness settings
    alternative_estimators: List[str] = field(default_factory=lambda: [
        'huber', 'quantile', 'theil_sen', 'ransac'
    ])
    quantile_levels: List[float] = field(default_factory=lambda: [0.25, 0.5, 0.75])
    
    # Outlier detection
    outlier_methods: List[str] = field(default_factory=lambda: [
        'z_score', 'iqr', 'isolation_forest', 'local_outlier_factor'
    ])
    outlier_threshold: float = 3.0
    
    # Influence diagnostics
    influence_measures: List[str] = field(default_factory=lambda: [
        'cooks_distance', 'leverage', 'studentized_residuals', 'dffits'
    ])
    
    # Visualization settings
    create_diagnostic_plots: bool = True
    plot_style: str = 'seaborn'
    figure_size: Tuple[int, int] = (12, 8)


@dataclass
class DiagnosticResults:
    """Results from model diagnostic tests"""
    
    model_name: str
    model_type: str
    
    # Normality test results
    normality_tests: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    normality_passed: bool = False
    
    # Heteroskedasticity test results
    heteroskedasticity_tests: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    homoskedasticity_passed: bool = False
    
    # Autocorrelation test results
    autocorrelation_tests: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    no_autocorrelation_passed: bool = False
    
    # Linearity test results
    linearity_tests: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    linearity_passed: bool = False
    
    # Stationarity test results
    stationarity_tests: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    stationarity_passed: bool = False
    
    # Outlier detection results
    outlier_detection: Dict[str, Any] = field(default_factory=dict)
    outliers_detected: List[int] = field(default_factory=list)
    
    # Influence diagnostics
    influence_diagnostics: Dict[str, Any] = field(default_factory=dict)
    influential_observations: List[int] = field(default_factory=list)
    
    # Overall assessment
    overall_assessment: Dict[str, Any] = field(default_factory=dict)
    diagnostic_score: float = 0.0
    
    # Metadata
    n_observations: int = 0
    n_parameters: int = 0
    diagnostic_timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class BootstrapResults:
    """Results from bootstrap procedures"""
    
    method: str
    n_bootstrap_samples: int
    
    # Bootstrap estimates
    bootstrap_estimates: np.ndarray
    original_estimate: float
    
    # Confidence intervals
    confidence_intervals: Dict[str, Tuple[float, float]] = field(default_factory=dict)
    
    # Bootstrap statistics
    bootstrap_mean: float = 0.0
    bootstrap_std: float = 0.0
    bootstrap_bias: float = 0.0
    
    # Distribution properties
    bootstrap_distribution: Dict[str, Any] = field(default_factory=dict)
    
    # Metadata
    bootstrap_timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


class StatisticalModelDiagnostics:
    """
    Comprehensive diagnostic testing framework for statistical models.
    
    Implements Requirements 5.5, 5.6 for model assumption testing and robustness.
    """
    
    def __init__(self, config: Optional[DiagnosticConfig] = None):
        """
        Initialize diagnostic framework.
        
        Args:
            config: DiagnosticConfig for diagnostic parameters
        """
        self.config = config or DiagnosticConfig()
        self.logger = logging.getLogger(__name__)
        
        # Storage for diagnostic results
        self.diagnostic_results: Dict[str, DiagnosticResults] = {}
        self.bootstrap_results: Dict[str, BootstrapResults] = {}
        
        self.logger.info("StatisticalModelDiagnostics initialized")
    
    def run_comprehensive_diagnostics(self,
                                    model: Any,
                                    model_name: str,
                                    X: Union[np.ndarray, pd.DataFrame],
                                    y: Union[np.ndarray, pd.Series],
                                    residuals: Optional[np.ndarray] = None,
                                    fitted_values: Optional[np.ndarray] = None) -> DiagnosticResults:
        """
        Run comprehensive diagnostic tests for a statistical model.
        
        Args:
            model: Fitted statistical model
            model_name: Name identifier for the model
            X: Feature matrix
            y: Target variable
            residuals: Model residuals (if available)
            fitted_values: Model fitted values (if available)
            
        Returns:
            DiagnosticResults with comprehensive test results
        """
        self.logger.info(f"Running comprehensive diagnostics for model: {model_name}")
        
        # Convert inputs to numpy arrays
        if isinstance(X, pd.DataFrame):
            X_array = X.values
        else:
            X_array = X
        
        if isinstance(y, pd.Series):
            y_array = y.values
        else:
            y_array = y
        
        # Extract or compute residuals and fitted values
        if residuals is None or fitted_values is None:
            residuals, fitted_values = self._extract_model_residuals_and_fitted(
                model, X_array, y_array
            )
        
        # Initialize diagnostic results
        results = DiagnosticResults(
            model_name=model_name,
            model_type=self._determine_model_type(model),
            n_observations=len(y_array),
            n_parameters=self._estimate_n_parameters(model, X_array)
        )
        
        # Run normality tests
        if residuals is not None:
            results.normality_tests = self._test_normality(residuals)
            results.normality_passed = self._assess_normality_tests(results.normality_tests)
        
        # Run heteroskedasticity tests
        if residuals is not None and fitted_values is not None:
            results.heteroskedasticity_tests = self._test_heteroskedasticity(
                residuals, fitted_values, X_array
            )
            results.homoskedasticity_passed = self._assess_heteroskedasticity_tests(
                results.heteroskedasticity_tests
            )
        
        # Run autocorrelation tests
        if residuals is not None:
            results.autocorrelation_tests = self._test_autocorrelation(residuals)
            results.no_autocorrelation_passed = self._assess_autocorrelation_tests(
                results.autocorrelation_tests
            )
        
        # Run linearity tests
        if fitted_values is not None:
            results.linearity_tests = self._test_linearity(y_array, fitted_values, X_array)
            results.linearity_passed = self._assess_linearity_tests(results.linearity_tests)
        
        # Run stationarity tests
        results.stationarity_tests = self._test_stationarity(y_array)
        results.stationarity_passed = self._assess_stationarity_tests(results.stationarity_tests)
        
        # Outlier detection
        if residuals is not None:
            results.outlier_detection = self._detect_outliers(residuals, y_array, X_array)
            results.outliers_detected = results.outlier_detection.get('outlier_indices', [])
        
        # Influence diagnostics
        if hasattr(model, 'fittedvalues') and hasattr(model, 'resid'):
            results.influence_diagnostics = self._compute_influence_diagnostics(model, X_array, y_array)
            results.influential_observations = results.influence_diagnostics.get('influential_indices', [])
        
        # Overall assessment
        results.overall_assessment = self._compute_overall_assessment(results)
        results.diagnostic_score = results.overall_assessment.get('diagnostic_score', 0.0)
        
        # Store results
        self.diagnostic_results[model_name] = results
        
        self.logger.info(f"Comprehensive diagnostics completed for {model_name}")
        return results
    
    def _extract_model_residuals_and_fitted(self,
                                          model: Any,
                                          X: np.ndarray,
                                          y: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Extract residuals and fitted values from model."""
        try:
            # Try common attributes first
            if hasattr(model, 'resid') and hasattr(model, 'fittedvalues'):
                return model.resid, model.fittedvalues
            
            elif hasattr(model, 'residuals') and hasattr(model, 'fitted_values'):
                return model.residuals, model.fitted_values
            
            # Try to compute from predict method
            elif hasattr(model, 'predict'):
                try:
                    fitted_values = model.predict(X)
                    residuals = y - fitted_values
                    return residuals, fitted_values
                except:
                    pass
            
            # For specific model types
            if 'Hansen' in model.__class__.__name__:
                # Hansen threshold model
                if hasattr(model, 'residuals1') and hasattr(model, 'residuals2'):
                    # Combine residuals from both regimes
                    residuals = np.concatenate([model.residuals1, model.residuals2])
                    fitted_values = y - residuals[:len(y)]
                    return residuals[:len(y)], fitted_values
            
            elif 'LocalProjections' in model.__class__.__name__:
                # Local projections model
                if hasattr(model, 'results') and hasattr(model.results, 'resid'):
                    return model.results.resid, model.results.fittedvalues
            
            # Fallback: return None
            return None, None
            
        except Exception as e:
            self.logger.warning(f"Error extracting residuals and fitted values: {str(e)}")
            return None, None
    
    def _determine_model_type(self, model: Any) -> str:
        """Determine the type of statistical model."""
        model_class = model.__class__.__name__
        
        if 'Hansen' in model_class:
            return 'threshold_regression'
        elif 'LocalProjections' in model_class:
            return 'local_projections'
        elif 'VAR' in model_class:
            return 'vector_autoregression'
        elif 'ARIMA' in model_class:
            return 'arima'
        elif 'OLS' in model_class:
            return 'ols_regression'
        else:
            return 'unknown'
    
    def _estimate_n_parameters(self, model: Any, X: np.ndarray) -> int:
        """Estimate number of parameters in the model."""
        try:
            if hasattr(model, 'params'):
                return len(model.params)
            elif hasattr(model, 'beta1') and hasattr(model, 'beta2'):
                # Hansen model
                return len(model.beta1) + len(model.beta2) + 1
            elif hasattr(model, 'n_params'):
                return model.n_params
            else:
                # Default estimate
                return X.shape[1] + 1
        except:
            return X.shape[1] + 1
    
    def _test_normality(self, residuals: np.ndarray) -> Dict[str, Dict[str, Any]]:
        """Run normality tests on residuals."""
        normality_results = {}
        
        # Remove NaN values
        clean_residuals = residuals[~np.isnan(residuals)]
        
        if len(clean_residuals) < 3:
            return {'error': 'Insufficient data for normality tests'}
        
        # Jarque-Bera test
        if 'jarque_bera' in self.config.normality_tests:
            try:
                jb_stat, jb_pvalue = jarque_bera(clean_residuals)
                normality_results['jarque_bera'] = {
                    'statistic': jb_stat,
                    'p_value': jb_pvalue,
                    'reject_normality': jb_pvalue < self.config.normality_alpha,
                    'interpretation': 'Reject normality' if jb_pvalue < self.config.normality_alpha else 'Fail to reject normality'
                }
            except Exception as e:
                normality_results['jarque_bera'] = {'error': str(e)}
        
        # Shapiro-Wilk test
        if 'shapiro' in self.config.normality_tests and len(clean_residuals) <= 5000:
            try:
                sw_stat, sw_pvalue = shapiro(clean_residuals)
                normality_results['shapiro'] = {
                    'statistic': sw_stat,
                    'p_value': sw_pvalue,
                    'reject_normality': sw_pvalue < self.config.normality_alpha,
                    'interpretation': 'Reject normality' if sw_pvalue < self.config.normality_alpha else 'Fail to reject normality'
                }
            except Exception as e:
                normality_results['shapiro'] = {'error': str(e)}
        
        # Anderson-Darling test
        if 'anderson' in self.config.normality_tests:
            try:
                ad_result = anderson(clean_residuals, dist='norm')
                # Use 5% significance level (index 2)
                critical_value = ad_result.critical_values[2]
                normality_results['anderson'] = {
                    'statistic': ad_result.statistic,
                    'critical_value': critical_value,
                    'reject_normality': ad_result.statistic > critical_value,
                    'interpretation': 'Reject normality' if ad_result.statistic > critical_value else 'Fail to reject normality'
                }
            except Exception as e:
                normality_results['anderson'] = {'error': str(e)}
        
        # Kolmogorov-Smirnov test
        if 'kolmogorov_smirnov' in self.config.normality_tests:
            try:
                # Test against standard normal distribution
                standardized_residuals = (clean_residuals - np.mean(clean_residuals)) / np.std(clean_residuals)
                ks_stat, ks_pvalue = kstest(standardized_residuals, 'norm')
                normality_results['kolmogorov_smirnov'] = {
                    'statistic': ks_stat,
                    'p_value': ks_pvalue,
                    'reject_normality': ks_pvalue < self.config.normality_alpha,
                    'interpretation': 'Reject normality' if ks_pvalue < self.config.normality_alpha else 'Fail to reject normality'
                }
            except Exception as e:
                normality_results['kolmogorov_smirnov'] = {'error': str(e)}
        
        # D'Agostino's normality test
        if 'dagostino' in self.config.normality_tests:
            try:
                da_stat, da_pvalue = normaltest(clean_residuals)
                normality_results['dagostino'] = {
                    'statistic': da_stat,
                    'p_value': da_pvalue,
                    'reject_normality': da_pvalue < self.config.normality_alpha,
                    'interpretation': 'Reject normality' if da_pvalue < self.config.normality_alpha else 'Fail to reject normality'
                }
            except Exception as e:
                normality_results['dagostino'] = {'error': str(e)}
        
        return normality_results
    
    def _assess_normality_tests(self, normality_tests: Dict[str, Dict[str, Any]]) -> bool:
        """Assess overall normality based on test results."""
        if not normality_tests or 'error' in normality_tests:
            return False
        
        # Count tests that fail to reject normality
        valid_tests = [test for test in normality_tests.values() if 'error' not in test]
        if not valid_tests:
            return False
        
        non_rejecting_tests = sum(1 for test in valid_tests if not test.get('reject_normality', True))
        
        # Pass if majority of tests fail to reject normality
        return non_rejecting_tests > len(valid_tests) / 2
    
    def _test_heteroskedasticity(self,
                               residuals: np.ndarray,
                               fitted_values: np.ndarray,
                               X: np.ndarray) -> Dict[str, Dict[str, Any]]:
        """Run heteroskedasticity tests."""
        het_results = {}
        
        # Clean data
        valid_idx = ~(np.isnan(residuals) | np.isnan(fitted_values))
        clean_residuals = residuals[valid_idx]
        clean_fitted = fitted_values[valid_idx]
        clean_X = X[valid_idx] if len(X) == len(residuals) else X
        
        if len(clean_residuals) < 10:
            return {'error': 'Insufficient data for heteroskedasticity tests'}
        
        # Breusch-Pagan test
        if 'breusch_pagan' in self.config.heteroskedasticity_tests:
            try:
                # Need to create a design matrix
                if clean_X.ndim == 1:
                    clean_X = clean_X.reshape(-1, 1)
                
                # Add constant term
                X_with_const = np.column_stack([np.ones(len(clean_X)), clean_X])
                
                bp_stat, bp_pvalue, _, _ = het_breuschpagan(clean_residuals, X_with_const)
                het_results['breusch_pagan'] = {
                    'statistic': bp_stat,
                    'p_value': bp_pvalue,
                    'reject_homoskedasticity': bp_pvalue < self.config.het_alpha,
                    'interpretation': 'Reject homoskedasticity (heteroskedasticity present)' if bp_pvalue < self.config.het_alpha else 'Fail to reject homoskedasticity'
                }
            except Exception as e:
                het_results['breusch_pagan'] = {'error': str(e)}
        
        # White test
        if 'white' in self.config.heteroskedasticity_tests:
            try:
                if clean_X.ndim == 1:
                    clean_X = clean_X.reshape(-1, 1)
                
                X_with_const = np.column_stack([np.ones(len(clean_X)), clean_X])
                
                white_stat, white_pvalue, _, _ = het_white(clean_residuals, X_with_const)
                het_results['white'] = {
                    'statistic': white_stat,
                    'p_value': white_pvalue,
                    'reject_homoskedasticity': white_pvalue < self.config.het_alpha,
                    'interpretation': 'Reject homoskedasticity (heteroskedasticity present)' if white_pvalue < self.config.het_alpha else 'Fail to reject homoskedasticity'
                }
            except Exception as e:
                het_results['white'] = {'error': str(e)}
        
        # ARCH test
        if 'arch' in self.config.heteroskedasticity_tests:
            try:
                arch_stat, arch_pvalue, _, _ = het_arch(clean_residuals, maxlag=min(5, len(clean_residuals)//4))
                het_results['arch'] = {
                    'statistic': arch_stat,
                    'p_value': arch_pvalue,
                    'reject_homoskedasticity': arch_pvalue < self.config.het_alpha,
                    'interpretation': 'Reject homoskedasticity (ARCH effects present)' if arch_pvalue < self.config.het_alpha else 'Fail to reject homoskedasticity'
                }
            except Exception as e:
                het_results['arch'] = {'error': str(e)}
        
        return het_results
    
    def _assess_heteroskedasticity_tests(self, het_tests: Dict[str, Dict[str, Any]]) -> bool:
        """Assess overall homoskedasticity based on test results."""
        if not het_tests or 'error' in het_tests:
            return False
        
        valid_tests = [test for test in het_tests.values() if 'error' not in test]
        if not valid_tests:
            return False
        
        # Pass if majority of tests fail to reject homoskedasticity
        non_rejecting_tests = sum(1 for test in valid_tests if not test.get('reject_homoskedasticity', True))
        return non_rejecting_tests > len(valid_tests) / 2
    
    def _test_autocorrelation(self, residuals: np.ndarray) -> Dict[str, Dict[str, Any]]:
        """Run autocorrelation tests on residuals."""
        autocorr_results = {}
        
        clean_residuals = residuals[~np.isnan(residuals)]
        
        if len(clean_residuals) < 10:
            return {'error': 'Insufficient data for autocorrelation tests'}
        
        # Breusch-Godfrey test
        if 'breusch_godfrey' in self.config.autocorrelation_tests:
            try:
                # Create simple design matrix for BG test
                X_simple = np.ones((len(clean_residuals), 1))  # Just intercept
                bg_stat, bg_pvalue, _, _ = acorr_breusch_godfrey(clean_residuals, X_simple, nlags=self.config.autocorr_lags)
                autocorr_results['breusch_godfrey'] = {
                    'statistic': bg_stat,
                    'p_value': bg_pvalue,
                    'reject_no_autocorrelation': bg_pvalue < self.config.autocorr_alpha,
                    'interpretation': 'Reject no autocorrelation (autocorrelation present)' if bg_pvalue < self.config.autocorr_alpha else 'Fail to reject no autocorrelation'
                }
            except Exception as e:
                autocorr_results['breusch_godfrey'] = {'error': str(e)}
        
        # Ljung-Box test
        if 'ljung_box' in self.config.autocorrelation_tests:
            try:
                lb_result = acorr_ljungbox(clean_residuals, lags=self.config.autocorr_lags, return_df=True)
                # Use the last lag's p-value
                lb_pvalue = lb_result['lb_pvalue'].iloc[-1]
                lb_stat = lb_result['lb_stat'].iloc[-1]
                
                autocorr_results['ljung_box'] = {
                    'statistic': lb_stat,
                    'p_value': lb_pvalue,
                    'reject_no_autocorrelation': lb_pvalue < self.config.autocorr_alpha,
                    'interpretation': 'Reject no autocorrelation (autocorrelation present)' if lb_pvalue < self.config.autocorr_alpha else 'Fail to reject no autocorrelation'
                }
            except Exception as e:
                autocorr_results['ljung_box'] = {'error': str(e)}
        
        # Durbin-Watson test
        if 'durbin_watson' in self.config.autocorrelation_tests:
            try:
                dw_stat = durbin_watson(clean_residuals)
                # DW statistic interpretation: values around 2 indicate no autocorrelation
                # Values < 1.5 or > 2.5 suggest autocorrelation
                autocorr_present = dw_stat < 1.5 or dw_stat > 2.5
                
                autocorr_results['durbin_watson'] = {
                    'statistic': dw_stat,
                    'reject_no_autocorrelation': autocorr_present,
                    'interpretation': f'DW = {dw_stat:.3f}. ' + ('Autocorrelation likely present' if autocorr_present else 'No strong evidence of autocorrelation')
                }
            except Exception as e:
                autocorr_results['durbin_watson'] = {'error': str(e)}
        
        return autocorr_results
    
    def _assess_autocorrelation_tests(self, autocorr_tests: Dict[str, Dict[str, Any]]) -> bool:
        """Assess overall absence of autocorrelation based on test results."""
        if not autocorr_tests or 'error' in autocorr_tests:
            return False
        
        valid_tests = [test for test in autocorr_tests.values() if 'error' not in test]
        if not valid_tests:
            return False
        
        # Pass if majority of tests fail to reject no autocorrelation
        non_rejecting_tests = sum(1 for test in valid_tests if not test.get('reject_no_autocorrelation', True))
        return non_rejecting_tests > len(valid_tests) / 2
    
    def _test_linearity(self,
                       y: np.ndarray,
                       fitted_values: np.ndarray,
                       X: np.ndarray) -> Dict[str, Dict[str, Any]]:
        """Run linearity tests."""
        linearity_results = {}
        
        # Clean data
        valid_idx = ~(np.isnan(y) | np.isnan(fitted_values))
        clean_y = y[valid_idx]
        clean_fitted = fitted_values[valid_idx]
        clean_X = X[valid_idx] if len(X) == len(y) else X
        
        if len(clean_y) < 10:
            return {'error': 'Insufficient data for linearity tests'}
        
        # Harvey-Collier test
        if 'harvey_collier' in self.config.linearity_tests:
            try:
                if clean_X.ndim == 1:
                    clean_X = clean_X.reshape(-1, 1)
                
                X_with_const = np.column_stack([np.ones(len(clean_X)), clean_X])
                
                hc_stat, hc_pvalue = linear_harvey_collier(clean_y, X_with_const)
                linearity_results['harvey_collier'] = {
                    'statistic': hc_stat,
                    'p_value': hc_pvalue,
                    'reject_linearity': hc_pvalue < self.config.linearity_alpha,
                    'interpretation': 'Reject linearity (non-linear relationship)' if hc_pvalue < self.config.linearity_alpha else 'Fail to reject linearity'
                }
            except Exception as e:
                linearity_results['harvey_collier'] = {'error': str(e)}
        
        # Rainbow test (simplified implementation)
        if 'rainbow' in self.config.linearity_tests:
            try:
                # Simple rainbow test: compare fit of full model vs restricted model
                # This is a simplified version
                n = len(clean_y)
                mid_start = n // 4
                mid_end = 3 * n // 4
                
                # Fit on middle portion
                y_mid = clean_y[mid_start:mid_end]
                fitted_mid = clean_fitted[mid_start:mid_end]
                
                # Calculate RSS for middle portion
                rss_mid = np.sum((y_mid - fitted_mid) ** 2)
                rss_full = np.sum((clean_y - clean_fitted) ** 2)
                
                # F-statistic (simplified)
                f_stat = ((rss_full - rss_mid) / (n - len(y_mid))) / (rss_mid / len(y_mid))
                
                # Approximate p-value using F-distribution
                from scipy.stats import f
                p_value = 1 - f.cdf(f_stat, n - len(y_mid), len(y_mid))
                
                linearity_results['rainbow'] = {
                    'statistic': f_stat,
                    'p_value': p_value,
                    'reject_linearity': p_value < self.config.linearity_alpha,
                    'interpretation': 'Reject linearity (non-linear relationship)' if p_value < self.config.linearity_alpha else 'Fail to reject linearity'
                }
            except Exception as e:
                linearity_results['rainbow'] = {'error': str(e)}
        
        return linearity_results
    
    def _assess_linearity_tests(self, linearity_tests: Dict[str, Dict[str, Any]]) -> bool:
        """Assess overall linearity based on test results."""
        if not linearity_tests or 'error' in linearity_tests:
            return False
        
        valid_tests = [test for test in linearity_tests.values() if 'error' not in test]
        if not valid_tests:
            return False
        
        # Pass if majority of tests fail to reject linearity
        non_rejecting_tests = sum(1 for test in valid_tests if not test.get('reject_linearity', True))
        return non_rejecting_tests > len(valid_tests) / 2
    
    def _test_stationarity(self, y: np.ndarray) -> Dict[str, Dict[str, Any]]:
        """Run stationarity tests on the time series."""
        stationarity_results = {}
        
        clean_y = y[~np.isnan(y)]
        
        if len(clean_y) < 10:
            return {'error': 'Insufficient data for stationarity tests'}
        
        # Augmented Dickey-Fuller test
        if 'adf' in self.config.stationarity_tests:
            try:
                adf_stat, adf_pvalue, adf_lags, adf_nobs, adf_critical, adf_icbest = adfuller(clean_y)
                stationarity_results['adf'] = {
                    'statistic': adf_stat,
                    'p_value': adf_pvalue,
                    'critical_values': adf_critical,
                    'reject_unit_root': adf_pvalue < self.config.stationarity_alpha,
                    'interpretation': 'Reject unit root (stationary)' if adf_pvalue < self.config.stationarity_alpha else 'Fail to reject unit root (non-stationary)'
                }
            except Exception as e:
                stationarity_results['adf'] = {'error': str(e)}
        
        # KPSS test
        if 'kpss' in self.config.stationarity_tests:
            try:
                kpss_stat, kpss_pvalue, kpss_lags, kpss_critical = kpss(clean_y)
                stationarity_results['kpss'] = {
                    'statistic': kpss_stat,
                    'p_value': kpss_pvalue,
                    'critical_values': kpss_critical,
                    'reject_stationarity': kpss_pvalue < self.config.stationarity_alpha,
                    'interpretation': 'Reject stationarity (non-stationary)' if kpss_pvalue < self.config.stationarity_alpha else 'Fail to reject stationarity (stationary)'
                }
            except Exception as e:
                stationarity_results['kpss'] = {'error': str(e)}
        
        return stationarity_results
    
    def _assess_stationarity_tests(self, stationarity_tests: Dict[str, Dict[str, Any]]) -> bool:
        """Assess overall stationarity based on test results."""
        if not stationarity_tests or 'error' in stationarity_tests:
            return False
        
        # For stationarity, we need ADF to reject unit root AND KPSS to fail to reject stationarity
        adf_result = stationarity_tests.get('adf', {})
        kpss_result = stationarity_tests.get('kpss', {})
        
        adf_stationary = adf_result.get('reject_unit_root', False)
        kpss_stationary = not kpss_result.get('reject_stationarity', True)
        
        # Both tests should agree on stationarity
        return adf_stationary and kpss_stationary
    
    def _detect_outliers(self,
                        residuals: np.ndarray,
                        y: np.ndarray,
                        X: np.ndarray) -> Dict[str, Any]:
        """Detect outliers using multiple methods."""
        outlier_results = {}
        
        clean_residuals = residuals[~np.isnan(residuals)]
        
        if len(clean_residuals) < 10:
            return {'error': 'Insufficient data for outlier detection'}
        
        # Z-score method
        if 'z_score' in self.config.outlier_methods:
            try:
                z_scores = np.abs(stats.zscore(clean_residuals))
                z_outliers = np.where(z_scores > self.config.outlier_threshold)[0]
                outlier_results['z_score'] = {
                    'outlier_indices': z_outliers.tolist(),
                    'n_outliers': len(z_outliers),
                    'outlier_percentage': len(z_outliers) / len(clean_residuals) * 100
                }
            except Exception as e:
                outlier_results['z_score'] = {'error': str(e)}
        
        # IQR method
        if 'iqr' in self.config.outlier_methods:
            try:
                Q1 = np.percentile(clean_residuals, 25)
                Q3 = np.percentile(clean_residuals, 75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                iqr_outliers = np.where((clean_residuals < lower_bound) | (clean_residuals > upper_bound))[0]
                outlier_results['iqr'] = {
                    'outlier_indices': iqr_outliers.tolist(),
                    'n_outliers': len(iqr_outliers),
                    'outlier_percentage': len(iqr_outliers) / len(clean_residuals) * 100,
                    'bounds': {'lower': lower_bound, 'upper': upper_bound}
                }
            except Exception as e:
                outlier_results['iqr'] = {'error': str(e)}
        
        # Combine outlier indices
        all_outliers = set()
        for method_result in outlier_results.values():
            if 'outlier_indices' in method_result:
                all_outliers.update(method_result['outlier_indices'])
        
        outlier_results['combined'] = {
            'outlier_indices': sorted(list(all_outliers)),
            'n_outliers': len(all_outliers),
            'outlier_percentage': len(all_outliers) / len(clean_residuals) * 100
        }
        
        return outlier_results
    
    def _compute_influence_diagnostics(self,
                                     model: Any,
                                     X: np.ndarray,
                                     y: np.ndarray) -> Dict[str, Any]:
        """Compute influence diagnostics for the model."""
        influence_results = {}
        
        try:
            # This requires statsmodels OLS-like interface
            if hasattr(model, 'get_influence'):
                influence = model.get_influence()
                
                # Cook's distance
                if 'cooks_distance' in self.config.influence_measures:
                    cooks_d = influence.cooks_distance[0]
                    # Threshold: 4/n
                    threshold = 4 / len(y)
                    influential_cooks = np.where(cooks_d > threshold)[0]
                    
                    influence_results['cooks_distance'] = {
                        'values': cooks_d.tolist(),
                        'threshold': threshold,
                        'influential_indices': influential_cooks.tolist(),
                        'n_influential': len(influential_cooks)
                    }
                
                # Leverage
                if 'leverage' in self.config.influence_measures:
                    leverage = influence.hat_matrix_diag
                    # Threshold: 2*p/n where p is number of parameters
                    p = X.shape[1] + 1
                    threshold = 2 * p / len(y)
                    high_leverage = np.where(leverage > threshold)[0]
                    
                    influence_results['leverage'] = {
                        'values': leverage.tolist(),
                        'threshold': threshold,
                        'high_leverage_indices': high_leverage.tolist(),
                        'n_high_leverage': len(high_leverage)
                    }
                
                # Studentized residuals
                if 'studentized_residuals' in self.config.influence_measures:
                    student_resid = influence.resid_studentized_external
                    # Threshold: |t| > 2
                    threshold = 2
                    outlier_resid = np.where(np.abs(student_resid) > threshold)[0]
                    
                    influence_results['studentized_residuals'] = {
                        'values': student_resid.tolist(),
                        'threshold': threshold,
                        'outlier_indices': outlier_resid.tolist(),
                        'n_outliers': len(outlier_resid)
                    }
            
            else:
                influence_results['error'] = 'Model does not support influence diagnostics'
        
        except Exception as e:
            influence_results['error'] = str(e)
        
        return influence_results
    
    def _compute_overall_assessment(self, results: DiagnosticResults) -> Dict[str, Any]:
        """Compute overall diagnostic assessment."""
        assessment = {}
        
        # Count passed tests
        tests_passed = 0
        total_tests = 0
        
        if results.normality_passed:
            tests_passed += 1
        total_tests += 1
        
        if results.homoskedasticity_passed:
            tests_passed += 1
        total_tests += 1
        
        if results.no_autocorrelation_passed:
            tests_passed += 1
        total_tests += 1
        
        if results.linearity_passed:
            tests_passed += 1
        total_tests += 1
        
        if results.stationarity_passed:
            tests_passed += 1
        total_tests += 1
        
        # Calculate diagnostic score
        diagnostic_score = tests_passed / total_tests if total_tests > 0 else 0.0
        
        # Overall assessment
        if diagnostic_score >= 0.8:
            overall_status = 'Excellent'
        elif diagnostic_score >= 0.6:
            overall_status = 'Good'
        elif diagnostic_score >= 0.4:
            overall_status = 'Fair'
        else:
            overall_status = 'Poor'
        
        assessment = {
            'diagnostic_score': diagnostic_score,
            'tests_passed': tests_passed,
            'total_tests': total_tests,
            'overall_status': overall_status,
            'recommendations': self._generate_recommendations(results),
            'summary': {
                'normality': 'Pass' if results.normality_passed else 'Fail',
                'homoskedasticity': 'Pass' if results.homoskedasticity_passed else 'Fail',
                'no_autocorrelation': 'Pass' if results.no_autocorrelation_passed else 'Fail',
                'linearity': 'Pass' if results.linearity_passed else 'Fail',
                'stationarity': 'Pass' if results.stationarity_passed else 'Fail'
            }
        }
        
        return assessment
    
    def _generate_recommendations(self, results: DiagnosticResults) -> List[str]:
        """Generate recommendations based on diagnostic results."""
        recommendations = []
        
        if not results.normality_passed:
            recommendations.append("Consider robust estimation methods or data transformation for non-normal residuals")
        
        if not results.homoskedasticity_passed:
            recommendations.append("Use heteroskedasticity-robust standard errors or weighted least squares")
        
        if not results.no_autocorrelation_passed:
            recommendations.append("Consider adding lagged variables or using HAC standard errors for autocorrelation")
        
        if not results.linearity_passed:
            recommendations.append("Investigate non-linear relationships or polynomial terms")
        
        if not results.stationarity_passed:
            recommendations.append("Consider differencing the series or using cointegration methods")
        
        if len(results.outliers_detected) > 0:
            recommendations.append(f"Investigate {len(results.outliers_detected)} detected outliers")
        
        if len(results.influential_observations) > 0:
            recommendations.append(f"Examine {len(results.influential_observations)} influential observations")
        
        if not recommendations:
            recommendations.append("Model diagnostics look good - no major issues detected")
        
        return recommendations


class BootstrapInference:
    """
    Bootstrap procedures for robust statistical inference.
    
    Implements Requirements 5.5, 5.6 for robust inference and uncertainty quantification.
    """
    
    def __init__(self, config: Optional[DiagnosticConfig] = None):
        """
        Initialize bootstrap inference framework.
        
        Args:
            config: DiagnosticConfig for bootstrap parameters
        """
        self.config = config or DiagnosticConfig()
        self.logger = logging.getLogger(__name__)
        
        # Storage for bootstrap results
        self.bootstrap_results: Dict[str, BootstrapResults] = {}
        
        self.logger.info("BootstrapInference initialized")
    
    def bootstrap_model_parameters(self,
                                 model: Any,
                                 X: Union[np.ndarray, pd.DataFrame],
                                 y: Union[np.ndarray, pd.Series],
                                 parameter_extractor: Callable[[Any], float],
                                 model_name: str) -> BootstrapResults:
        """
        Bootstrap model parameters for robust inference.
        
        Args:
            model: Statistical model to bootstrap
            X: Feature matrix
            y: Target variable
            parameter_extractor: Function to extract parameter of interest from fitted model
            model_name: Name identifier for the model
            
        Returns:
            BootstrapResults with bootstrap estimates and confidence intervals
        """
        self.logger.info(f"Running bootstrap inference for model: {model_name}")
        
        # Convert to numpy arrays
        if isinstance(X, pd.DataFrame):
            X_array = X.values
        else:
            X_array = X
        
        if isinstance(y, pd.Series):
            y_array = y.values
        else:
            y_array = y
        
        # Get original parameter estimate
        try:
            original_estimate = parameter_extractor(model)
        except Exception as e:
            self.logger.error(f"Error extracting original parameter: {str(e)}")
            raise
        
        # Initialize bootstrap - use simple resampling if arch not available
        if not ARCH_AVAILABLE or self.config.bootstrap_method == 'simple':
            # Simple bootstrap resampling
            bootstrap_estimates = []
            data_combined = np.column_stack([y_array, X_array])
            
            for i in range(self.config.bootstrap_samples):
                try:
                    # Simple bootstrap resample
                    boot_indices = np.random.choice(len(data_combined), size=len(data_combined), replace=True)
                    boot_data = data_combined[boot_indices]
                    boot_y = boot_data[:, 0]
                    boot_X = boot_data[:, 1:]
                    
                    # Fit model on bootstrap sample
                    boot_model = self._fit_bootstrap_model(model, boot_X, boot_y)
                    
                    # Extract parameter
                    boot_estimate = parameter_extractor(boot_model)
                    bootstrap_estimates.append(boot_estimate)
                    
                except Exception as e:
                    self.logger.warning(f"Bootstrap iteration {i} failed: {str(e)}")
                    continue
        else:
            # Use arch bootstrap methods
            if self.config.bootstrap_method == 'iid':
                bootstrap = IIDBootstrap(np.column_stack([y_array, X_array]))
            elif self.config.bootstrap_method == 'stationary':
                block_size = self.config.bootstrap_block_size or max(1, int(len(y_array) ** 0.25))
                bootstrap = StationaryBootstrap(block_size, np.column_stack([y_array, X_array]))
            elif self.config.bootstrap_method == 'circular_block':
                block_size = self.config.bootstrap_block_size or max(1, int(len(y_array) ** 0.25))
                bootstrap = CircularBlockBootstrap(block_size, np.column_stack([y_array, X_array]))
            else:
                raise ValueError(f"Unknown bootstrap method: {self.config.bootstrap_method}")
            
            # Run bootstrap
            bootstrap_estimates = []
            
            for i, data in enumerate(bootstrap.bootstrap(self.config.bootstrap_samples)):
                try:
                    # Extract bootstrap sample
                    boot_data = data[0][0]  # Get the data array
                    boot_y = boot_data[:, 0]
                    boot_X = boot_data[:, 1:]
                    
                    # Fit model on bootstrap sample
                    boot_model = self._fit_bootstrap_model(model, boot_X, boot_y)
                    
                    # Extract parameter
                    boot_estimate = parameter_extractor(boot_model)
                    bootstrap_estimates.append(boot_estimate)
                    
                except Exception as e:
                    self.logger.warning(f"Bootstrap iteration {i} failed: {str(e)}")
                    continue
        
        if not bootstrap_estimates:
            raise ValueError("All bootstrap iterations failed")
        
        bootstrap_estimates = np.array(bootstrap_estimates)
        
        # Calculate confidence intervals
        confidence_intervals = {}
        for level in self.config.bootstrap_confidence_levels:
            alpha = level if level < 0.5 else 1 - level
            lower_percentile = (alpha / 2) * 100
            upper_percentile = (1 - alpha / 2) * 100
            
            ci_lower = np.percentile(bootstrap_estimates, lower_percentile)
            ci_upper = np.percentile(bootstrap_estimates, upper_percentile)
            
            confidence_intervals[f"{int((1-alpha)*100)}%"] = (ci_lower, ci_upper)
        
        # Calculate bootstrap statistics
        bootstrap_mean = np.mean(bootstrap_estimates)
        bootstrap_std = np.std(bootstrap_estimates)
        bootstrap_bias = bootstrap_mean - original_estimate
        
        # Distribution properties
        bootstrap_distribution = {
            'mean': bootstrap_mean,
            'std': bootstrap_std,
            'min': np.min(bootstrap_estimates),
            'max': np.max(bootstrap_estimates),
            'skewness': stats.skew(bootstrap_estimates),
            'kurtosis': stats.kurtosis(bootstrap_estimates),
            'percentiles': {
                '5%': np.percentile(bootstrap_estimates, 5),
                '25%': np.percentile(bootstrap_estimates, 25),
                '50%': np.percentile(bootstrap_estimates, 50),
                '75%': np.percentile(bootstrap_estimates, 75),
                '95%': np.percentile(bootstrap_estimates, 95)
            }
        }
        
        # Create results
        results = BootstrapResults(
            method=self.config.bootstrap_method,
            n_bootstrap_samples=len(bootstrap_estimates),
            bootstrap_estimates=bootstrap_estimates,
            original_estimate=original_estimate,
            confidence_intervals=confidence_intervals,
            bootstrap_mean=bootstrap_mean,
            bootstrap_std=bootstrap_std,
            bootstrap_bias=bootstrap_bias,
            bootstrap_distribution=bootstrap_distribution
        )
        
        # Store results
        self.bootstrap_results[model_name] = results
        
        self.logger.info(f"Bootstrap inference completed for {model_name}")
        return results
    
    def _fit_bootstrap_model(self, original_model: Any, X: np.ndarray, y: np.ndarray) -> Any:
        """Fit model on bootstrap sample."""
        # Create a copy of the original model
        model_class = original_model.__class__
        
        try:
            # Try to create new instance with same parameters
            if hasattr(original_model, 'get_params'):
                # Scikit-learn style
                params = original_model.get_params()
                boot_model = model_class(**params)
            else:
                # Create basic instance
                boot_model = model_class()
            
            # Fit the model
            if 'Hansen' in model_class.__name__:
                # Hansen threshold model
                threshold_var = X[:, 0] if X.shape[1] > 0 else y
                boot_model.fit(y, X, threshold_var)
            elif 'LocalProjections' in model_class.__name__:
                # Local projections model
                shock = X[:, 0] if X.shape[1] > 0 else np.zeros(len(y))
                controls = pd.DataFrame(X[:, 1:]) if X.shape[1] > 1 else None
                boot_model.fit(pd.Series(y), pd.Series(shock), controls)
            else:
                # Generic model
                boot_model.fit(X, y)
            
            return boot_model
            
        except Exception as e:
            self.logger.warning(f"Error fitting bootstrap model: {str(e)}")
            raise
    
    def bootstrap_prediction_intervals(self,
                                     model: Any,
                                     X_train: Union[np.ndarray, pd.DataFrame],
                                     y_train: Union[np.ndarray, pd.Series],
                                     X_test: Union[np.ndarray, pd.DataFrame],
                                     model_name: str) -> Dict[str, np.ndarray]:
        """
        Generate bootstrap prediction intervals.
        
        Args:
            model: Fitted statistical model
            X_train: Training features
            y_train: Training target
            X_test: Test features for prediction
            model_name: Name identifier for the model
            
        Returns:
            Dictionary with prediction intervals
        """
        self.logger.info(f"Generating bootstrap prediction intervals for {model_name}")
        
        # Convert to numpy arrays
        if isinstance(X_train, pd.DataFrame):
            X_train_array = X_train.values
        else:
            X_train_array = X_train
        
        if isinstance(y_train, pd.Series):
            y_train_array = y_train.values
        else:
            y_train_array = y_train
        
        if isinstance(X_test, pd.DataFrame):
            X_test_array = X_test.values
        else:
            X_test_array = X_test
        
        # Initialize bootstrap - use simple resampling if arch not available
        bootstrap_predictions = []
        
        if not ARCH_AVAILABLE or self.config.bootstrap_method == 'simple':
            # Simple bootstrap resampling
            data_combined = np.column_stack([y_train_array, X_train_array])
            
            for i in range(self.config.bootstrap_samples):
                try:
                    # Simple bootstrap resample
                    boot_indices = np.random.choice(len(data_combined), size=len(data_combined), replace=True)
                    boot_data = data_combined[boot_indices]
                    boot_y = boot_data[:, 0]
                    boot_X = boot_data[:, 1:]
                    
                    # Fit model on bootstrap sample
                    boot_model = self._fit_bootstrap_model(model, boot_X, boot_y)
                    
                    # Make predictions
                    if hasattr(boot_model, 'predict'):
                        boot_pred = boot_model.predict(X_test_array)
                    else:
                        # Fallback prediction
                        boot_pred = np.full(len(X_test_array), np.mean(boot_y))
                    
                    bootstrap_predictions.append(boot_pred)
                    
                except Exception as e:
                    self.logger.warning(f"Bootstrap prediction iteration {i} failed: {str(e)}")
                    continue
        else:
            # Use arch bootstrap methods
            if self.config.bootstrap_method == 'iid':
                bootstrap = IIDBootstrap(np.column_stack([y_train_array, X_train_array]))
            else:
                block_size = self.config.bootstrap_block_size or max(1, int(len(y_train_array) ** 0.25))
                if self.config.bootstrap_method == 'stationary':
                    bootstrap = StationaryBootstrap(block_size, np.column_stack([y_train_array, X_train_array]))
                else:
                    bootstrap = CircularBlockBootstrap(block_size, np.column_stack([y_train_array, X_train_array]))
            
            # Collect bootstrap predictions
            for i, data in enumerate(bootstrap.bootstrap(self.config.bootstrap_samples)):
                try:
                    # Extract bootstrap sample
                    boot_data = data[0][0]
                    boot_y = boot_data[:, 0]
                    boot_X = boot_data[:, 1:]
                    
                    # Fit model on bootstrap sample
                    boot_model = self._fit_bootstrap_model(model, boot_X, boot_y)
                    
                    # Make predictions
                    if hasattr(boot_model, 'predict'):
                        boot_pred = boot_model.predict(X_test_array)
                    else:
                        # Fallback prediction
                        boot_pred = np.full(len(X_test_array), np.mean(boot_y))
                    
                    bootstrap_predictions.append(boot_pred)
                    
                except Exception as e:
                    self.logger.warning(f"Bootstrap prediction iteration {i} failed: {str(e)}")
                    continue
        
        if not bootstrap_predictions:
            raise ValueError("All bootstrap prediction iterations failed")
        
        bootstrap_predictions = np.array(bootstrap_predictions)
        
        # Calculate prediction intervals
        prediction_intervals = {}
        
        for level in self.config.bootstrap_confidence_levels:
            alpha = level if level < 0.5 else 1 - level
            lower_percentile = (alpha / 2) * 100
            upper_percentile = (1 - alpha / 2) * 100
            
            pi_lower = np.percentile(bootstrap_predictions, lower_percentile, axis=0)
            pi_upper = np.percentile(bootstrap_predictions, upper_percentile, axis=0)
            
            prediction_intervals[f"{int((1-alpha)*100)}%_lower"] = pi_lower
            prediction_intervals[f"{int((1-alpha)*100)}%_upper"] = pi_upper
        
        # Add mean prediction
        prediction_intervals['mean'] = np.mean(bootstrap_predictions, axis=0)
        prediction_intervals['std'] = np.std(bootstrap_predictions, axis=0)
        
        self.logger.info(f"Bootstrap prediction intervals generated for {model_name}")
        return prediction_intervals


class AlternativeEstimationMethods:
    """
    Alternative estimation methods for robustness testing.
    
    Implements Requirements 5.5, 5.6 for alternative estimators and robustness.
    """
    
    def __init__(self, config: Optional[DiagnosticConfig] = None):
        """
        Initialize alternative estimation framework.
        
        Args:
            config: DiagnosticConfig for estimation parameters
        """
        self.config = config or DiagnosticConfig()
        self.logger = logging.getLogger(__name__)
        
        # Storage for estimation results
        self.estimation_results: Dict[str, Dict[str, Any]] = {}
        
        self.logger.info("AlternativeEstimationMethods initialized")
    
    def fit_robust_estimators(self,
                            X: Union[np.ndarray, pd.DataFrame],
                            y: Union[np.ndarray, pd.Series],
                            model_name: str) -> Dict[str, Any]:
        """
        Fit multiple robust estimators for comparison.
        
        Args:
            X: Feature matrix
            y: Target variable
            model_name: Name identifier for the model
            
        Returns:
            Dictionary with results from different robust estimators
        """
        self.logger.info(f"Fitting robust estimators for {model_name}")
        
        # Convert to numpy arrays
        if isinstance(X, pd.DataFrame):
            X_array = X.values
            feature_names = X.columns.tolist()
        else:
            X_array = X
            feature_names = [f"feature_{i}" for i in range(X.shape[1])]
        
        if isinstance(y, pd.Series):
            y_array = y.values
        else:
            y_array = y
        
        robust_results = {}
        
        # Huber regression (robust to outliers)
        if 'huber' in self.config.alternative_estimators:
            try:
                from sklearn.linear_model import HuberRegressor
                
                huber = HuberRegressor(epsilon=1.35, max_iter=100)
                huber.fit(X_array, y_array)
                
                # Calculate predictions and metrics
                y_pred = huber.predict(X_array)
                r2 = r2_score(y_array, y_pred)
                mse = mean_squared_error(y_array, y_pred)
                
                robust_results['huber'] = {
                    'model': huber,
                    'coefficients': huber.coef_.tolist(),
                    'intercept': huber.intercept_,
                    'r2_score': r2,
                    'mse': mse,
                    'predictions': y_pred,
                    'feature_names': feature_names
                }
                
            except Exception as e:
                robust_results['huber'] = {'error': str(e)}
        
        # Quantile regression
        if 'quantile' in self.config.alternative_estimators:
            try:
                from sklearn.linear_model import QuantileRegressor
                
                quantile_results = {}
                for quantile in self.config.quantile_levels:
                    qr = QuantileRegressor(quantile=quantile, alpha=0.01, solver='highs')
                    qr.fit(X_array, y_array)
                    
                    y_pred = qr.predict(X_array)
                    
                    quantile_results[f'q{quantile}'] = {
                        'model': qr,
                        'coefficients': qr.coef_.tolist(),
                        'intercept': qr.intercept_,
                        'predictions': y_pred,
                        'quantile': quantile
                    }
                
                robust_results['quantile'] = quantile_results
                
            except Exception as e:
                robust_results['quantile'] = {'error': str(e)}
        
        # Theil-Sen estimator (robust to outliers)
        if 'theil_sen' in self.config.alternative_estimators:
            try:
                from sklearn.linear_model import TheilSenRegressor
                
                theil_sen = TheilSenRegressor(random_state=42, max_subpopulation=1e4)
                theil_sen.fit(X_array, y_array)
                
                y_pred = theil_sen.predict(X_array)
                r2 = r2_score(y_array, y_pred)
                mse = mean_squared_error(y_array, y_pred)
                
                robust_results['theil_sen'] = {
                    'model': theil_sen,
                    'coefficients': theil_sen.coef_.tolist(),
                    'intercept': theil_sen.intercept_,
                    'r2_score': r2,
                    'mse': mse,
                    'predictions': y_pred,
                    'feature_names': feature_names
                }
                
            except Exception as e:
                robust_results['theil_sen'] = {'error': str(e)}
        
        # RANSAC (robust to outliers)
        if 'ransac' in self.config.alternative_estimators:
            try:
                from sklearn.linear_model import RANSACRegressor
                
                ransac = RANSACRegressor(random_state=42, max_trials=100)
                ransac.fit(X_array, y_array)
                
                y_pred = ransac.predict(X_array)
                r2 = r2_score(y_array, y_pred)
                mse = mean_squared_error(y_array, y_pred)
                
                robust_results['ransac'] = {
                    'model': ransac,
                    'coefficients': ransac.estimator_.coef_.tolist(),
                    'intercept': ransac.estimator_.intercept_,
                    'r2_score': r2,
                    'mse': mse,
                    'predictions': y_pred,
                    'inlier_mask': ransac.inlier_mask_.tolist(),
                    'n_inliers': np.sum(ransac.inlier_mask_),
                    'feature_names': feature_names
                }
                
            except Exception as e:
                robust_results['ransac'] = {'error': str(e)}
        
        # Store results
        self.estimation_results[model_name] = robust_results
        
        self.logger.info(f"Robust estimators fitted for {model_name}")
        return robust_results
    
    def compare_estimators(self, model_name: str) -> Dict[str, Any]:
        """
        Compare results from different robust estimators.
        
        Args:
            model_name: Name identifier for the model
            
        Returns:
            Dictionary with comparison results
        """
        if model_name not in self.estimation_results:
            raise ValueError(f"No estimation results found for {model_name}")
        
        results = self.estimation_results[model_name]
        comparison = {}
        
        # Extract performance metrics
        performance_data = []
        
        for estimator_name, estimator_results in results.items():
            if 'error' in estimator_results:
                continue
            
            if estimator_name == 'quantile':
                # Handle quantile regression separately
                for q_name, q_results in estimator_results.items():
                    if 'error' not in q_results:
                        performance_data.append({
                            'estimator': f"{estimator_name}_{q_name}",
                            'r2_score': np.nan,  # Not applicable for quantile regression
                            'mse': np.nan,
                            'n_parameters': len(q_results.get('coefficients', [])) + 1
                        })
            else:
                performance_data.append({
                    'estimator': estimator_name,
                    'r2_score': estimator_results.get('r2_score', np.nan),
                    'mse': estimator_results.get('mse', np.nan),
                    'n_parameters': len(estimator_results.get('coefficients', [])) + 1
                })
        
        if performance_data:
            comparison_df = pd.DataFrame(performance_data)
            comparison['performance_table'] = comparison_df
            
            # Best estimator by R²
            valid_r2 = comparison_df.dropna(subset=['r2_score'])
            if not valid_r2.empty:
                best_r2_idx = valid_r2['r2_score'].idxmax()
                comparison['best_by_r2'] = {
                    'estimator': valid_r2.loc[best_r2_idx, 'estimator'],
                    'r2_score': valid_r2.loc[best_r2_idx, 'r2_score']
                }
            
            # Best estimator by MSE
            valid_mse = comparison_df.dropna(subset=['mse'])
            if not valid_mse.empty:
                best_mse_idx = valid_mse['mse'].idxmin()
                comparison['best_by_mse'] = {
                    'estimator': valid_mse.loc[best_mse_idx, 'estimator'],
                    'mse': valid_mse.loc[best_mse_idx, 'mse']
                }
        
        # Coefficient comparison
        coefficient_comparison = {}
        for estimator_name, estimator_results in results.items():
            if 'error' in estimator_results or estimator_name == 'quantile':
                continue
            
            coefficients = estimator_results.get('coefficients', [])
            intercept = estimator_results.get('intercept', 0)
            
            coefficient_comparison[estimator_name] = {
                'coefficients': coefficients,
                'intercept': intercept
            }
        
        if coefficient_comparison:
            comparison['coefficient_comparison'] = coefficient_comparison
            
            # Calculate coefficient stability (standard deviation across estimators)
            all_coefficients = []
            for est_results in coefficient_comparison.values():
                all_coefficients.append(est_results['coefficients'])
            
            if all_coefficients and len(all_coefficients) > 1:
                coeff_array = np.array(all_coefficients)
                coeff_std = np.std(coeff_array, axis=0)
                coeff_mean = np.mean(coeff_array, axis=0)
                
                comparison['coefficient_stability'] = {
                    'mean_coefficients': coeff_mean.tolist(),
                    'std_coefficients': coeff_std.tolist(),
                    'coefficient_of_variation': (coeff_std / np.abs(coeff_mean)).tolist()
                }
        
        return comparison
    
    def plot_robust_comparison(self, model_name: str, figsize: Tuple[int, int] = (12, 8)) -> plt.Figure:
        """
        Create visualization comparing robust estimators.
        
        Args:
            model_name: Name identifier for the model
            figsize: Figure size
            
        Returns:
            Matplotlib figure
        """
        if model_name not in self.estimation_results:
            raise ValueError(f"No estimation results found for {model_name}")
        
        results = self.estimation_results[model_name]
        comparison = self.compare_estimators(model_name)
        
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        
        # Performance comparison
        if 'performance_table' in comparison:
            perf_df = comparison['performance_table']
            
            # R² comparison
            valid_r2 = perf_df.dropna(subset=['r2_score'])
            if not valid_r2.empty:
                axes[0, 0].bar(valid_r2['estimator'], valid_r2['r2_score'])
                axes[0, 0].set_title('R² Score Comparison')
                axes[0, 0].set_ylabel('R² Score')
                axes[0, 0].tick_params(axis='x', rotation=45)
            
            # MSE comparison
            valid_mse = perf_df.dropna(subset=['mse'])
            if not valid_mse.empty:
                axes[0, 1].bar(valid_mse['estimator'], valid_mse['mse'])
                axes[0, 1].set_title('MSE Comparison')
                axes[0, 1].set_ylabel('MSE')
                axes[0, 1].tick_params(axis='x', rotation=45)
        
        # Coefficient comparison
        if 'coefficient_comparison' in comparison:
            coeff_comp = comparison['coefficient_comparison']
            
            # Plot coefficients
            estimators = list(coeff_comp.keys())
            n_coeffs = len(coeff_comp[estimators[0]]['coefficients'])
            
            x = np.arange(n_coeffs)
            width = 0.8 / len(estimators)
            
            for i, (est_name, est_data) in enumerate(coeff_comp.items()):
                axes[1, 0].bar(x + i * width, est_data['coefficients'], 
                              width, label=est_name, alpha=0.7)
            
            axes[1, 0].set_title('Coefficient Comparison')
            axes[1, 0].set_ylabel('Coefficient Value')
            axes[1, 0].set_xlabel('Coefficient Index')
            axes[1, 0].legend()
        
        # Coefficient stability
        if 'coefficient_stability' in comparison:
            stability = comparison['coefficient_stability']
            
            axes[1, 1].bar(range(len(stability['coefficient_of_variation'])), 
                          stability['coefficient_of_variation'])
            axes[1, 1].set_title('Coefficient Stability (CV)')
            axes[1, 1].set_ylabel('Coefficient of Variation')
            axes[1, 1].set_xlabel('Coefficient Index')
        
        plt.tight_layout()
        return fig


# Utility functions for diagnostic plotting
def plot_diagnostic_summary(diagnostic_results: DiagnosticResults, 
                           figsize: Tuple[int, int] = (15, 10)) -> plt.Figure:
    """
    Create comprehensive diagnostic plot summary.
    
    Args:
        diagnostic_results: DiagnosticResults object
        figsize: Figure size
        
    Returns:
        Matplotlib figure with diagnostic plots
    """
    fig, axes = plt.subplots(3, 3, figsize=figsize)
    
    # Test results summary
    test_results = {
        'Normality': 'Pass' if diagnostic_results.normality_passed else 'Fail',
        'Homoskedasticity': 'Pass' if diagnostic_results.homoskedasticity_passed else 'Fail',
        'No Autocorrelation': 'Pass' if diagnostic_results.no_autocorrelation_passed else 'Fail',
        'Linearity': 'Pass' if diagnostic_results.linearity_passed else 'Fail',
        'Stationarity': 'Pass' if diagnostic_results.stationarity_passed else 'Fail'
    }
    
    # Summary bar chart
    test_names = list(test_results.keys())
    test_values = [1 if result == 'Pass' else 0 for result in test_results.values()]
    colors = ['green' if val == 1 else 'red' for val in test_values]
    
    axes[0, 0].bar(test_names, test_values, color=colors, alpha=0.7)
    axes[0, 0].set_title('Diagnostic Test Summary')
    axes[0, 0].set_ylabel('Pass (1) / Fail (0)')
    axes[0, 0].tick_params(axis='x', rotation=45)
    
    # Diagnostic score
    score = diagnostic_results.diagnostic_score
    axes[0, 1].pie([score, 1-score], labels=['Passed', 'Failed'], 
                   colors=['green', 'red'], autopct='%1.1f%%')
    axes[0, 1].set_title(f'Overall Diagnostic Score: {score:.2f}')
    
    # Outlier summary
    if diagnostic_results.outlier_detection:
        outlier_methods = []
        outlier_counts = []
        
        for method, results in diagnostic_results.outlier_detection.items():
            if isinstance(results, dict) and 'n_outliers' in results:
                outlier_methods.append(method)
                outlier_counts.append(results['n_outliers'])
        
        if outlier_methods:
            axes[0, 2].bar(outlier_methods, outlier_counts)
            axes[0, 2].set_title('Outliers Detected by Method')
            axes[0, 2].set_ylabel('Number of Outliers')
            axes[0, 2].tick_params(axis='x', rotation=45)
    
    # Test statistics visualization
    row_idx = 1
    col_idx = 0
    
    # Normality test p-values
    if diagnostic_results.normality_tests:
        norm_tests = []
        norm_pvalues = []
        
        for test_name, test_results in diagnostic_results.normality_tests.items():
            if 'p_value' in test_results:
                norm_tests.append(test_name)
                norm_pvalues.append(test_results['p_value'])
        
        if norm_tests:
            axes[row_idx, col_idx].bar(norm_tests, norm_pvalues)
            axes[row_idx, col_idx].axhline(y=0.05, color='red', linestyle='--', alpha=0.7)
            axes[row_idx, col_idx].set_title('Normality Test P-values')
            axes[row_idx, col_idx].set_ylabel('P-value')
            axes[row_idx, col_idx].tick_params(axis='x', rotation=45)
            col_idx += 1
    
    # Heteroskedasticity test p-values
    if diagnostic_results.heteroskedasticity_tests:
        het_tests = []
        het_pvalues = []
        
        for test_name, test_results in diagnostic_results.heteroskedasticity_tests.items():
            if 'p_value' in test_results:
                het_tests.append(test_name)
                het_pvalues.append(test_results['p_value'])
        
        if het_tests:
            axes[row_idx, col_idx].bar(het_tests, het_pvalues)
            axes[row_idx, col_idx].axhline(y=0.05, color='red', linestyle='--', alpha=0.7)
            axes[row_idx, col_idx].set_title('Heteroskedasticity Test P-values')
            axes[row_idx, col_idx].set_ylabel('P-value')
            axes[row_idx, col_idx].tick_params(axis='x', rotation=45)
            col_idx += 1
    
    # Autocorrelation test p-values
    if diagnostic_results.autocorrelation_tests:
        autocorr_tests = []
        autocorr_pvalues = []
        
        for test_name, test_results in diagnostic_results.autocorrelation_tests.items():
            if 'p_value' in test_results:
                autocorr_tests.append(test_name)
                autocorr_pvalues.append(test_results['p_value'])
        
        if autocorr_tests and col_idx < 3:
            axes[row_idx, col_idx].bar(autocorr_tests, autocorr_pvalues)
            axes[row_idx, col_idx].axhline(y=0.05, color='red', linestyle='--', alpha=0.7)
            axes[row_idx, col_idx].set_title('Autocorrelation Test P-values')
            axes[row_idx, col_idx].set_ylabel('P-value')
            axes[row_idx, col_idx].tick_params(axis='x', rotation=45)
    
    # Fill remaining subplots with text summaries
    for i in range(3):
        for j in range(3):
            if i >= 2:  # Last row
                axes[i, j].text(0.1, 0.5, f"Model: {diagnostic_results.model_name}\n"
                                         f"Type: {diagnostic_results.model_type}\n"
                                         f"N obs: {diagnostic_results.n_observations}\n"
                                         f"N params: {diagnostic_results.n_parameters}\n"
                                         f"Score: {diagnostic_results.diagnostic_score:.3f}",
                               transform=axes[i, j].transAxes, fontsize=10,
                               verticalalignment='center')
                axes[i, j].set_xticks([])
                axes[i, j].set_yticks([])
    
    plt.tight_layout()
    return fig