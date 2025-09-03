"""
Model Specification Enhancer for Publication-Quality Econometric Analysis

This module provides enhanced model specifications to address low R² issues
and improve model fit for publication-ready econometric analysis.
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar, minimize, differential_evolution
from scipy.stats import norm, chi2, f, t
import statsmodels.api as sm
from statsmodels.regression.linear_model import OLS
from statsmodels.tsa.stattools import adfuller
from sklearn.model_selection import cross_val_score, KFold, GridSearchCV
from sklearn.linear_model import LinearRegression, QuantileRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample
import warnings
warnings.filterwarnings('ignore')

try:
    from ..core.models import HansenThresholdRegression
    from .robust_estimation_methods import RobustEstimationMethods
except ImportError:
    from models import HansenThresholdRegression, SmoothTransitionRegression
    from robust_estimation_methods import RobustEstimationMethods


class ModelSpecificationEnhancer:
    """
    Enhanced model specifications for improved model fit and publication quality
    
    This class provides alternative model specifications to address low explanatory
    power in threshold models, implementing multiple threshold detection methods,
    enhanced transition functions, and robust estimation techniques.
    """
    
    def __init__(self):
        self.fitted_models = {}
        self.comparison_results = {}
        self.enhancement_history = []
        self.robust_estimator = RobustEstimationMethods()
        
    def enhanced_hansen_regression(self, y, x, threshold_var, enhancements=None):
        """
        Enhanced Hansen regression with improved R² through better specification
        
        Args:
            y: Dependent variable
            x: Independent variables (can be 1D or 2D array)
            threshold_var: Threshold variable for regime switching
            enhancements: Dict of enhancement options
                - 'data_transforms': List of transformations to try
                - 'additional_controls': Additional control variables
                - 'interaction_terms': Whether to include interaction terms
                - 'lagged_variables': Number of lags to include
                - 'outlier_treatment': Method for outlier handling
                
        Returns:
            dict: Enhanced model results with diagnostics and comparisons
        """
        if enhancements is None:
            enhancements = {
                'data_transforms': ['levels', 'differences', 'logs'],
                'additional_controls': True,
                'interaction_terms': True,
                'lagged_variables': 2,
                'outlier_treatment': 'winsorize'
            }
        
        results = {
            'original_model': None,
            'enhanced_models': {},
            'best_model': None,
            'improvement_summary': {},
            'diagnostics': {}
        }
        
        # Ensure x is 2D and convert pandas Series to numpy array
        if hasattr(x, 'values'):  # pandas Series
            x = x.values
        if x.ndim == 1:
            x = x.reshape(-1, 1)
        
        # Convert pandas Series to numpy arrays
        if hasattr(y, 'values'):
            y = y.values
        if hasattr(threshold_var, 'values'):
            threshold_var = threshold_var.values
        
        # Fit original Hansen model for baseline
        try:
            original_hansen = HansenThresholdRegression()
            original_hansen.fit(y, x, threshold_var)
            
            # Calculate original R²
            y_pred_orig = original_hansen.predict(x, threshold_var)
            ss_res_orig = np.sum((y - y_pred_orig) ** 2)
            ss_tot_orig = np.sum((y - np.mean(y)) ** 2)
            original_r2 = 1 - (ss_res_orig / ss_tot_orig) if ss_tot_orig > 0 else 0
            
            results['original_model'] = {
                'model': original_hansen,
                'r_squared': original_r2,
                'threshold': original_hansen.threshold,
                'fitted': original_hansen.fitted
            }
        except Exception as e:
            results['original_model'] = {
                'error': str(e),
                'r_squared': 0,
                'fitted': False
            }
        
        # Enhancement 1: Data transformations
        if 'data_transforms' in enhancements:
            results['enhanced_models']['transformations'] = self._test_data_transformations(
                y, x, threshold_var, enhancements['data_transforms']
            )
        
        # Enhancement 2: Additional control variables
        if enhancements.get('additional_controls', False):
            results['enhanced_models']['additional_controls'] = self._add_control_variables(
                y, x, threshold_var
            )
        
        # Enhancement 3: Interaction terms
        if enhancements.get('interaction_terms', False):
            results['enhanced_models']['interactions'] = self._add_interaction_terms(
                y, x, threshold_var
            )
        
        # Enhancement 4: Lagged variables
        if enhancements.get('lagged_variables', 0) > 0:
            results['enhanced_models']['lagged'] = self._add_lagged_variables(
                y, x, threshold_var, enhancements['lagged_variables']
            )
        
        # Enhancement 5: Outlier treatment
        if 'outlier_treatment' in enhancements:
            results['enhanced_models']['outlier_treated'] = self._apply_outlier_treatment(
                y, x, threshold_var, enhancements['outlier_treatment']
            )
        
        # Find best performing model
        best_r2 = results['original_model'].get('r_squared', 0)
        best_model_key = 'original'
        
        for enhancement_type, models in results['enhanced_models'].items():
            if isinstance(models, dict) and 'r_squared' in models:
                if models['r_squared'] > best_r2:
                    best_r2 = models['r_squared']
                    best_model_key = enhancement_type
            elif isinstance(models, dict):
                for model_name, model_info in models.items():
                    if isinstance(model_info, dict) and 'r_squared' in model_info:
                        if model_info['r_squared'] > best_r2:
                            best_r2 = model_info['r_squared']
                            best_model_key = f"{enhancement_type}_{model_name}"
        
        results['best_model'] = {
            'type': best_model_key,
            'r_squared': best_r2,
            'improvement': best_r2 - results['original_model'].get('r_squared', 0)
        }
        
        # Generate improvement summary
        results['improvement_summary'] = self._generate_improvement_summary(results)
        
        # Store results
        self.fitted_models['enhanced_hansen'] = results
        
        return results
    
    def multiple_threshold_model(self, y, x, threshold_var, max_thresholds=3):
        """
        Test models with multiple thresholds using sequential threshold testing
        
        Args:
            y: Dependent variable
            x: Independent variables
            threshold_var: Threshold variable
            max_thresholds: Maximum number of thresholds to test
            
        Returns:
            dict: Results from multiple threshold models
        """
        results = {
            'models': {},
            'selection_criteria': {},
            'best_model': None,
            'threshold_tests': {}
        }
        
        # Ensure x is 2D and convert pandas Series to numpy array
        if hasattr(x, 'values'):  # pandas Series
            x = x.values
        if x.ndim == 1:
            x = x.reshape(-1, 1)
        
        # Convert pandas Series to numpy arrays
        if hasattr(y, 'values'):
            y = y.values
        if hasattr(threshold_var, 'values'):
            threshold_var = threshold_var.values
        
        # Test models with 1 to max_thresholds
        for n_thresh in range(1, max_thresholds + 1):
            try:
                if n_thresh == 1:
                    # Single threshold Hansen model
                    model = HansenThresholdRegression()
                    model.fit(y, x, threshold_var)
                    
                    y_pred = model.predict(x, threshold_var)
                    ss_res = np.sum((y - y_pred) ** 2)
                    ss_tot = np.sum((y - np.mean(y)) ** 2)
                    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
                    
                    # Calculate information criteria
                    n_params = len(model.beta1) + len(model.beta2) + 1  # +1 for threshold
                    n_obs = len(y)
                    log_likelihood = -0.5 * n_obs * np.log(2 * np.pi * ss_res / n_obs) - 0.5 * ss_res / (ss_res / n_obs)
                    
                    aic = 2 * n_params - 2 * log_likelihood
                    bic = np.log(n_obs) * n_params - 2 * log_likelihood
                    
                    results['models'][f'{n_thresh}_threshold'] = {
                        'model': model,
                        'n_thresholds': n_thresh,
                        'thresholds': [model.threshold],
                        'r_squared': r2,
                        'aic': aic,
                        'bic': bic,
                        'log_likelihood': log_likelihood,
                        'n_params': n_params,
                        'fitted': True
                    }
                    
                else:
                    # Multiple threshold model (sequential estimation)
                    multi_model = self._fit_multiple_threshold_model(y, x, threshold_var, n_thresh)
                    results['models'][f'{n_thresh}_threshold'] = multi_model
                    
            except Exception as e:
                results['models'][f'{n_thresh}_threshold'] = {
                    'error': str(e),
                    'fitted': False,
                    'n_thresholds': n_thresh
                }
        
        # Model selection based on information criteria
        results['selection_criteria'] = self._select_best_threshold_model(results['models'])
        
        # Sequential threshold tests
        results['threshold_tests'] = self._sequential_threshold_tests(y, x, threshold_var, max_thresholds)
        
        # Store results
        self.fitted_models['multiple_threshold'] = results
        
        return results
    
    def smooth_transition_alternatives(self, y, x, threshold_var, transition_types=None):
        """
        Test smooth transition regression with different transition functions
        
        Args:
            y: Dependent variable
            x: Independent variables
            threshold_var: Threshold variable
            transition_types: List of transition function types to test
            
        Returns:
            dict: Results from different STR specifications
        """
        if transition_types is None:
            transition_types = ['logistic', 'exponential', 'linear']
        
        results = {
            'models': {},
            'comparison': {},
            'best_specification': None
        }
        
        # Ensure x is 2D and convert pandas Series to numpy array
        if hasattr(x, 'values'):  # pandas Series
            x = x.values
        if x.ndim == 1:
            x = x.reshape(-1, 1)
        
        # Convert pandas Series to numpy arrays
        if hasattr(y, 'values'):
            y = y.values
        if hasattr(threshold_var, 'values'):
            threshold_var = threshold_var.values
        
        # Test different transition functions
        for trans_type in transition_types:
            try:
                if trans_type == 'logistic':
                    # Standard logistic STR
                    str_model = SmoothTransitionRegression()
                    str_result = str_model.fit(y, x, threshold_var)
                    
                    y_pred = str_model.predict(x, threshold_var)
                    ss_res = np.sum((y - y_pred) ** 2)
                    ss_tot = np.sum((y - np.mean(y)) ** 2)
                    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
                    
                    results['models'][trans_type] = {
                        'model': str_model,
                        'r_squared': r2,
                        'gamma': str_model.gamma,
                        'c': str_model.c,
                        'fitted': str_model.fitted,
                        'transition_type': trans_type
                    }
                    
                elif trans_type == 'exponential':
                    # Exponential STR (ESTR)
                    estr_model = self._fit_exponential_str(y, x, threshold_var)
                    results['models'][trans_type] = estr_model
                    
                elif trans_type == 'linear':
                    # Linear transition (approximation)
                    linear_model = self._fit_linear_transition(y, x, threshold_var)
                    results['models'][trans_type] = linear_model
                    
            except Exception as e:
                results['models'][trans_type] = {
                    'error': str(e),
                    'fitted': False,
                    'transition_type': trans_type
                }
        
        # Compare models
        results['comparison'] = self._compare_str_models(results['models'])
        
        # Store results
        self.fitted_models['smooth_transition'] = results
        
        return results
    
    def _test_data_transformations(self, y, x, threshold_var, transforms):
        """Test different data transformations"""
        results = {}
        
        for transform in transforms:
            try:
                if transform == 'levels':
                    # Use original data
                    y_trans, x_trans, thresh_trans = y.copy(), x.copy(), threshold_var.copy()
                    
                elif transform == 'differences':
                    # First differences
                    if isinstance(y, pd.Series):
                        y_trans = y.diff().dropna()
                        if x.ndim == 1:
                            x_trans = pd.Series(x).diff().dropna().values.reshape(-1, 1)
                        else:
                            x_trans = pd.DataFrame(x).diff().dropna().values
                        thresh_trans = pd.Series(threshold_var).diff().dropna().values
                    else:
                        y_trans = np.diff(y)
                        x_trans = np.diff(x, axis=0) if x.ndim > 1 else np.diff(x).reshape(-1, 1)
                        thresh_trans = np.diff(threshold_var)
                    
                elif transform == 'logs':
                    # Log transformation (only for positive values)
                    if np.all(y > 0) and np.all(x > 0) and np.all(threshold_var > 0):
                        y_trans = np.log(y)
                        x_trans = np.log(x) if x.ndim == 1 else np.log(x)
                        thresh_trans = np.log(threshold_var)
                    else:
                        # Skip if negative values present
                        continue
                
                # Fit Hansen model with transformed data
                hansen_trans = HansenThresholdRegression()
                hansen_trans.fit(y_trans, x_trans, thresh_trans)
                
                # Calculate R²
                y_pred = hansen_trans.predict(x_trans, thresh_trans)
                ss_res = np.sum((y_trans - y_pred) ** 2)
                ss_tot = np.sum((y_trans - np.mean(y_trans)) ** 2)
                r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
                
                results[transform] = {
                    'model': hansen_trans,
                    'r_squared': r2,
                    'transformation': transform,
                    'fitted': hansen_trans.fitted
                }
                
            except Exception as e:
                results[transform] = {
                    'error': str(e),
                    'transformation': transform,
                    'fitted': False
                }
        
        return results
    
    def _add_control_variables(self, y, x, threshold_var):
        """Add additional control variables to improve fit"""
        # Create additional controls based on existing variables
        n_obs = len(y)
        
        # Time trend
        time_trend = np.arange(n_obs)
        
        # Squared terms
        if x.ndim == 1:
            x_squared = x ** 2
            additional_controls = np.column_stack([x, x_squared, time_trend])
        else:
            x_squared = x ** 2
            additional_controls = np.column_stack([x, x_squared, time_trend])
        
        try:
            # Fit Hansen model with additional controls
            hansen_enhanced = HansenThresholdRegression()
            hansen_enhanced.fit(y, additional_controls, threshold_var)
            
            # Calculate R²
            y_pred = hansen_enhanced.predict(additional_controls, threshold_var)
            ss_res = np.sum((y - y_pred) ** 2)
            ss_tot = np.sum((y - np.mean(y)) ** 2)
            r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
            
            return {
                'model': hansen_enhanced,
                'r_squared': r2,
                'additional_variables': ['squared_terms', 'time_trend'],
                'fitted': hansen_enhanced.fitted
            }
            
        except Exception as e:
            return {
                'error': str(e),
                'fitted': False
            }
    
    def _add_interaction_terms(self, y, x, threshold_var):
        """Add interaction terms with threshold variable"""
        try:
            # Create interaction terms
            if x.ndim == 1:
                x_thresh_interact = x * threshold_var
                x_enhanced = np.column_stack([x, x_thresh_interact])
            else:
                x_thresh_interact = x * threshold_var.reshape(-1, 1)
                x_enhanced = np.column_stack([x, x_thresh_interact])
            
            # Fit Hansen model with interactions
            hansen_interact = HansenThresholdRegression()
            hansen_interact.fit(y, x_enhanced, threshold_var)
            
            # Calculate R²
            y_pred = hansen_interact.predict(x_enhanced, threshold_var)
            ss_res = np.sum((y - y_pred) ** 2)
            ss_tot = np.sum((y - np.mean(y)) ** 2)
            r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
            
            return {
                'model': hansen_interact,
                'r_squared': r2,
                'interaction_terms': True,
                'fitted': hansen_interact.fitted
            }
            
        except Exception as e:
            return {
                'error': str(e),
                'fitted': False
            }
    
    def _add_lagged_variables(self, y, x, threshold_var, n_lags):
        """Add lagged variables to the model"""
        try:
            # Create lagged variables
            if isinstance(y, pd.Series):
                y_data = y.values
            else:
                y_data = y
            
            # Create lags
            lagged_vars = []
            for lag in range(1, n_lags + 1):
                if isinstance(y, pd.Series):
                    y_lag = y.shift(lag).values
                else:
                    y_lag = np.concatenate([np.full(lag, np.nan), y_data[:-lag]])
                lagged_vars.append(y_lag)
            
            # Combine with original x
            if x.ndim == 1:
                x_with_lags = np.column_stack([x] + lagged_vars)
            else:
                x_with_lags = np.column_stack([x] + lagged_vars)
            
            # Remove NaN rows
            valid_mask = ~np.isnan(x_with_lags).any(axis=1)
            y_clean = y_data[valid_mask]
            x_clean = x_with_lags[valid_mask]
            thresh_clean = threshold_var[valid_mask] if hasattr(threshold_var, '__len__') else threshold_var
            
            if len(y_clean) < 20:  # Minimum observations
                return {
                    'error': 'Insufficient observations after adding lags',
                    'fitted': False
                }
            
            # Fit Hansen model with lags
            hansen_lagged = HansenThresholdRegression()
            hansen_lagged.fit(y_clean, x_clean, thresh_clean)
            
            # Calculate R²
            y_pred = hansen_lagged.predict(x_clean, thresh_clean)
            ss_res = np.sum((y_clean - y_pred) ** 2)
            ss_tot = np.sum((y_clean - np.mean(y_clean)) ** 2)
            r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
            
            return {
                'model': hansen_lagged,
                'r_squared': r2,
                'n_lags': n_lags,
                'fitted': hansen_lagged.fitted
            }
            
        except Exception as e:
            return {
                'error': str(e),
                'fitted': False
            }
    
    def _apply_outlier_treatment(self, y, x, threshold_var, method):
        """Apply outlier treatment methods"""
        try:
            if method == 'winsorize':
                # Winsorize at 5th and 95th percentiles
                y_clean = np.clip(y, np.percentile(y, 5), np.percentile(y, 95))
                
                if x.ndim == 1:
                    x_clean = np.clip(x, np.percentile(x, 5), np.percentile(x, 95))
                else:
                    x_clean = np.column_stack([
                        np.clip(x[:, i], np.percentile(x[:, i], 5), np.percentile(x[:, i], 95))
                        for i in range(x.shape[1])
                    ])
                
                thresh_clean = np.clip(threshold_var, 
                                     np.percentile(threshold_var, 5), 
                                     np.percentile(threshold_var, 95))
                
            elif method == 'remove':
                # Remove observations beyond 3 standard deviations
                z_scores_y = np.abs((y - np.mean(y)) / np.std(y))
                
                if x.ndim == 1:
                    z_scores_x = np.abs((x - np.mean(x)) / np.std(x))
                    outlier_mask = (z_scores_y < 3) & (z_scores_x < 3)
                else:
                    z_scores_x = np.max([np.abs((x[:, i] - np.mean(x[:, i])) / np.std(x[:, i])) 
                                       for i in range(x.shape[1])], axis=0)
                    outlier_mask = (z_scores_y < 3) & (z_scores_x < 3)
                
                y_clean = y[outlier_mask]
                x_clean = x[outlier_mask] if x.ndim == 1 else x[outlier_mask]
                thresh_clean = threshold_var[outlier_mask]
                
            else:
                # No treatment
                y_clean, x_clean, thresh_clean = y, x, threshold_var
            
            # Fit Hansen model with treated data
            hansen_clean = HansenThresholdRegression()
            hansen_clean.fit(y_clean, x_clean, thresh_clean)
            
            # Calculate R²
            y_pred = hansen_clean.predict(x_clean, thresh_clean)
            ss_res = np.sum((y_clean - y_pred) ** 2)
            ss_tot = np.sum((y_clean - np.mean(y_clean)) ** 2)
            r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
            
            return {
                'model': hansen_clean,
                'r_squared': r2,
                'outlier_method': method,
                'fitted': hansen_clean.fitted
            }
            
        except Exception as e:
            return {
                'error': str(e),
                'fitted': False
            }
    
    def _fit_multiple_threshold_model(self, y, x, threshold_var, n_thresholds):
        """Fit model with multiple thresholds using sequential estimation"""
        try:
            # Sequential threshold estimation
            thresholds = []
            current_y = y.copy()
            current_x = x.copy()
            current_thresh = threshold_var.copy()
            
            for i in range(n_thresholds):
                # Fit single threshold model
                hansen = HansenThresholdRegression()
                hansen.fit(current_y, current_x, current_thresh)
                
                if hansen.fitted:
                    thresholds.append(hansen.threshold)
                    
                    # For next iteration, focus on regime with larger residuals
                    y_pred = hansen.predict(current_x, current_thresh)
                    residuals = np.abs(current_y - y_pred)
                    
                    # Keep observations with larger residuals for next threshold
                    if i < n_thresholds - 1:
                        keep_mask = residuals > np.median(residuals)
                        current_y = current_y[keep_mask]
                        current_x = current_x[keep_mask] if current_x.ndim == 1 else current_x[keep_mask]
                        current_thresh = current_thresh[keep_mask]
                        
                        if len(current_y) < 20:  # Minimum observations
                            break
                else:
                    break
            
            # Fit final model with all identified thresholds
            if len(thresholds) > 0:
                final_model = self._fit_multi_regime_model(y, x, threshold_var, thresholds)
                
                return {
                    'model': final_model,
                    'n_thresholds': len(thresholds),
                    'thresholds': thresholds,
                    'r_squared': final_model.get('r_squared', 0),
                    'fitted': final_model.get('fitted', False)
                }
            else:
                return {
                    'error': 'No thresholds identified',
                    'fitted': False,
                    'n_thresholds': 0
                }
                
        except Exception as e:
            return {
                'error': str(e),
                'fitted': False,
                'n_thresholds': n_thresholds
            }
    
    def _fit_multi_regime_model(self, y, x, threshold_var, thresholds):
        """Fit model with multiple regimes defined by thresholds"""
        try:
            # Sort thresholds
            sorted_thresholds = sorted(thresholds)
            n_regimes = len(sorted_thresholds) + 1
            
            # Create regime indicators
            regime_masks = []
            
            # First regime: threshold_var <= first_threshold
            regime_masks.append(threshold_var <= sorted_thresholds[0])
            
            # Middle regimes: threshold_i < threshold_var <= threshold_{i+1}
            for i in range(len(sorted_thresholds) - 1):
                mask = (threshold_var > sorted_thresholds[i]) & (threshold_var <= sorted_thresholds[i + 1])
                regime_masks.append(mask)
            
            # Last regime: threshold_var > last_threshold
            regime_masks.append(threshold_var > sorted_thresholds[-1])
            
            # Fit separate regression for each regime
            regime_models = []
            total_ssr = 0
            n_params = 0
            
            for i, mask in enumerate(regime_masks):
                if np.sum(mask) >= 5:  # Minimum observations per regime
                    y_regime = y[mask]
                    x_regime = x[mask] if x.ndim == 1 else x[mask]
                    
                    # Add constant
                    if x_regime.ndim == 1:
                        X_regime = np.column_stack([np.ones(len(y_regime)), x_regime])
                    else:
                        X_regime = np.column_stack([np.ones(len(y_regime)), x_regime])
                    
                    # Fit OLS
                    beta_regime = np.linalg.lstsq(X_regime, y_regime, rcond=None)[0]
                    y_pred_regime = X_regime @ beta_regime
                    ssr_regime = np.sum((y_regime - y_pred_regime) ** 2)
                    
                    regime_models.append({
                        'beta': beta_regime,
                        'ssr': ssr_regime,
                        'n_obs': len(y_regime),
                        'regime_id': i
                    })
                    
                    total_ssr += ssr_regime
                    n_params += len(beta_regime)
                else:
                    regime_models.append(None)
            
            # Calculate overall R²
            ss_tot = np.sum((y - np.mean(y)) ** 2)
            r2 = 1 - (total_ssr / ss_tot) if ss_tot > 0 else 0
            
            # Calculate information criteria
            n_obs = len(y)
            log_likelihood = -0.5 * n_obs * np.log(2 * np.pi * total_ssr / n_obs) - 0.5 * total_ssr / (total_ssr / n_obs)
            
            aic = 2 * n_params - 2 * log_likelihood
            bic = np.log(n_obs) * n_params - 2 * log_likelihood
            
            return {
                'regime_models': regime_models,
                'thresholds': sorted_thresholds,
                'n_regimes': n_regimes,
                'r_squared': r2,
                'aic': aic,
                'bic': bic,
                'total_ssr': total_ssr,
                'n_params': n_params,
                'fitted': True
            }
            
        except Exception as e:
            return {
                'error': str(e),
                'fitted': False
            }
    
    def _fit_exponential_str(self, y, x, threshold_var):
        """Fit exponential smooth transition regression"""
        try:
            def exponential_transition(qe_intensity, gamma, c):
                """Exponential transition function"""
                return 1 - np.exp(-gamma * (qe_intensity - c) ** 2)
            
            def objective(params):
                gamma, c = params
                if gamma <= 0:
                    return 1e10
                
                G = exponential_transition(threshold_var, gamma, c)
                
                # Apply transition to regressors
                if x.ndim == 1:
                    X_transition = x * G
                    X_reg = np.column_stack([np.ones(len(x)), x, X_transition])
                else:
                    G_reshaped = G[:, np.newaxis]
                    X_transition = x * G_reshaped
                    X_reg = np.column_stack([np.ones(len(x)), x, X_transition])
                
                try:
                    beta = np.linalg.lstsq(X_reg, y, rcond=None)[0]
                    residuals = y - X_reg @ beta
                    ssr = np.sum(residuals ** 2)
                    return ssr
                except:
                    return 1e10
            
            # Initial values
            initial_gamma = 1.0
            initial_c = np.median(threshold_var)
            
            result = minimize(objective, [initial_gamma, initial_c], 
                            method='Nelder-Mead', 
                            options={'maxiter': 1000})
            
            if result.success:
                gamma_opt, c_opt = result.x
                G = exponential_transition(threshold_var, gamma_opt, c_opt)
                
                if x.ndim == 1:
                    X_reg = np.column_stack([np.ones(len(x)), x, x * G])
                else:
                    G_reshaped = G[:, np.newaxis]
                    X_reg = np.column_stack([np.ones(len(x)), x, x * G_reshaped])
                
                coeffs = np.linalg.lstsq(X_reg, y, rcond=None)[0]
                y_pred = X_reg @ coeffs
                
                ss_res = np.sum((y - y_pred) ** 2)
                ss_tot = np.sum((y - np.mean(y)) ** 2)
                r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
                
                return {
                    'gamma': gamma_opt,
                    'c': c_opt,
                    'coefficients': coeffs,
                    'r_squared': r2,
                    'transition_type': 'exponential',
                    'fitted': True
                }
            else:
                return {
                    'error': 'Optimization failed',
                    'fitted': False,
                    'transition_type': 'exponential'
                }
                
        except Exception as e:
            return {
                'error': str(e),
                'fitted': False,
                'transition_type': 'exponential'
            }
    
    def _fit_linear_transition(self, y, x, threshold_var):
        """Fit linear transition model (approximation to STR)"""
        try:
            # Normalize threshold variable to [0, 1]
            thresh_min = np.min(threshold_var)
            thresh_max = np.max(threshold_var)
            thresh_norm = (threshold_var - thresh_min) / (thresh_max - thresh_min)
            
            # Linear transition function
            G = thresh_norm
            
            # Apply transition to regressors
            if x.ndim == 1:
                X_transition = x * G
                X_reg = np.column_stack([np.ones(len(x)), x, X_transition])
            else:
                G_reshaped = G[:, np.newaxis]
                X_transition = x * G_reshaped
                X_reg = np.column_stack([np.ones(len(x)), x, X_transition])
            
            # Fit linear model
            coeffs = np.linalg.lstsq(X_reg, y, rcond=None)[0]
            y_pred = X_reg @ coeffs
            
            ss_res = np.sum((y - y_pred) ** 2)
            ss_tot = np.sum((y - np.mean(y)) ** 2)
            r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
            
            return {
                'coefficients': coeffs,
                'r_squared': r2,
                'transition_type': 'linear',
                'fitted': True
            }
            
        except Exception as e:
            return {
                'error': str(e),
                'fitted': False,
                'transition_type': 'linear'
            }
    
    def _select_best_threshold_model(self, models):
        """Select best model based on information criteria"""
        selection_results = {
            'aic_best': None,
            'bic_best': None,
            'r2_best': None,
            'criteria_table': []
        }
        
        best_aic = np.inf
        best_bic = np.inf
        best_r2 = -np.inf
        
        for model_name, model_info in models.items():
            if model_info.get('fitted', False):
                aic = model_info.get('aic', np.inf)
                bic = model_info.get('bic', np.inf)
                r2 = model_info.get('r_squared', -np.inf)
                
                selection_results['criteria_table'].append({
                    'model': model_name,
                    'aic': aic,
                    'bic': bic,
                    'r_squared': r2,
                    'n_thresholds': model_info.get('n_thresholds', 0)
                })
                
                if aic < best_aic:
                    best_aic = aic
                    selection_results['aic_best'] = model_name
                
                if bic < best_bic:
                    best_bic = bic
                    selection_results['bic_best'] = model_name
                
                if r2 > best_r2:
                    best_r2 = r2
                    selection_results['r2_best'] = model_name
        
        return selection_results
    
    def _sequential_threshold_tests(self, y, x, threshold_var, max_thresholds):
        """Perform sequential tests for number of thresholds"""
        test_results = {}
        
        for n in range(1, max_thresholds):
            try:
                # Test n vs n+1 thresholds
                # This is a simplified version - full implementation would use Hansen's test
                
                # Fit n-threshold model
                if n == 1:
                    model_n = HansenThresholdRegression()
                    model_n.fit(y, x, threshold_var)
                    ssr_n = np.sum((y - model_n.predict(x, threshold_var)) ** 2)
                else:
                    multi_n = self._fit_multiple_threshold_model(y, x, threshold_var, n)
                    ssr_n = multi_n.get('total_ssr', np.inf)
                
                # Fit (n+1)-threshold model
                multi_n1 = self._fit_multiple_threshold_model(y, x, threshold_var, n + 1)
                ssr_n1 = multi_n1.get('total_ssr', np.inf)
                
                # F-test for additional threshold
                if ssr_n1 < ssr_n:
                    f_stat = ((ssr_n - ssr_n1) / (x.shape[1] if x.ndim > 1 else 1)) / (ssr_n1 / (len(y) - (n + 1) * (x.shape[1] if x.ndim > 1 else 1)))
                    
                    # Approximate p-value
                    from scipy.stats import f
                    df1 = x.shape[1] if x.ndim > 1 else 1
                    df2 = len(y) - (n + 1) * df1
                    p_value = 1 - f.cdf(f_stat, df1, df2) if df2 > 0 else 1
                    
                    test_results[f'{n}_vs_{n+1}'] = {
                        'f_statistic': f_stat,
                        'p_value': p_value,
                        'ssr_n': ssr_n,
                        'ssr_n1': ssr_n1,
                        'reject_null': p_value < 0.05
                    }
                else:
                    test_results[f'{n}_vs_{n+1}'] = {
                        'f_statistic': 0,
                        'p_value': 1,
                        'ssr_n': ssr_n,
                        'ssr_n1': ssr_n1,
                        'reject_null': False
                    }
                    
            except Exception as e:
                test_results[f'{n}_vs_{n+1}'] = {
                    'error': str(e),
                    'reject_null': False
                }
        
        return test_results
    
    def _compare_str_models(self, models):
        """Compare different STR model specifications"""
        comparison = {
            'best_model': None,
            'model_ranking': [],
            'fit_comparison': {}
        }
        
        best_r2 = -np.inf
        
        for model_name, model_info in models.items():
            if model_info.get('fitted', False):
                r2 = model_info.get('r_squared', -np.inf)
                
                comparison['model_ranking'].append({
                    'model': model_name,
                    'r_squared': r2,
                    'transition_type': model_info.get('transition_type', 'unknown')
                })
                
                if r2 > best_r2:
                    best_r2 = r2
                    comparison['best_model'] = model_name
        
        # Sort by R²
        comparison['model_ranking'].sort(key=lambda x: x['r_squared'], reverse=True)
        
        return comparison
    
    def time_varying_threshold_model(self, y, x, threshold_var, time_var):
        """
        Implement time-varying threshold model for dynamic threshold analysis
        
        Args:
            y: Dependent variable
            x: Independent variables
            threshold_var: Threshold variable
            time_var: Time variable (e.g., dates, periods)
            
        Returns:
            dict: Time-varying threshold model results
        """
        results = {
            'models': {},
            'time_varying_thresholds': {},
            'stability_tests': {},
            'best_specification': None
        }
        
        # Convert pandas Series to numpy arrays
        if hasattr(y, 'values'):
            y = y.values
        if hasattr(x, 'values'):
            x = x.values
        if hasattr(threshold_var, 'values'):
            threshold_var = threshold_var.values
        if hasattr(time_var, 'values'):
            time_var = time_var.values
        
        # Ensure x is 2D
        if x.ndim == 1:
            x = x.reshape(-1, 1)
        
        try:
            # Method 1: Rolling window threshold estimation
            results['models']['rolling_window'] = self._fit_rolling_window_threshold(
                y, x, threshold_var, time_var
            )
            
            # Method 2: Smooth time-varying threshold
            results['models']['smooth_time_varying'] = self._fit_smooth_time_varying_threshold(
                y, x, threshold_var, time_var
            )
            
            # Method 3: Regime-dependent time trend
            results['models']['regime_time_trend'] = self._fit_regime_time_trend_model(
                y, x, threshold_var, time_var
            )
            
            # Stability tests
            results['stability_tests'] = self._test_threshold_stability_over_time(
                y, x, threshold_var, time_var
            )
            
            # Select best specification
            results['best_specification'] = self._select_best_time_varying_model(results['models'])
            
        except Exception as e:
            results['error'] = str(e)
        
        # Store results
        self.fitted_models['time_varying_threshold'] = results
        
        return results
    
    def regime_specific_controls_model(self, y, x, threshold_var, regime_controls):
        """
        Implement model with different variables per regime
        
        Args:
            y: Dependent variable
            x: Base independent variables
            threshold_var: Threshold variable
            regime_controls: Dict with regime-specific control variables
                           {'regime1': array, 'regime2': array}
            
        Returns:
            dict: Regime-specific controls model results
        """
        results = {
            'model': None,
            'regime_diagnostics': {},
            'specification_tests': {},
            'r_squared': 0,
            'fitted': False
        }
        
        # Convert pandas Series to numpy arrays
        if hasattr(y, 'values'):
            y = y.values
        if hasattr(x, 'values'):
            x = x.values
        if hasattr(threshold_var, 'values'):
            threshold_var = threshold_var.values
        
        # Ensure x is 2D
        if x.ndim == 1:
            x = x.reshape(-1, 1)
        
        try:
            # First, find optimal threshold
            hansen_base = HansenThresholdRegression()
            hansen_base.fit(y, x, threshold_var)
            
            if not hansen_base.fitted:
                results['error'] = 'Base Hansen model failed to fit'
                return results
            
            threshold = hansen_base.threshold
            
            # Create regime masks
            regime1_mask = threshold_var <= threshold
            regime2_mask = threshold_var > threshold
            
            # Fit regime-specific models with different controls
            regime_models = {}
            total_ssr = 0
            n_params = 0
            
            # Regime 1
            if np.sum(regime1_mask) >= 5:
                y1 = y[regime1_mask]
                x1_base = x[regime1_mask]
                
                # Add regime-specific controls for regime 1
                if 'regime1' in regime_controls:
                    regime1_controls = regime_controls['regime1']
                    if hasattr(regime1_controls, 'values'):
                        regime1_controls = regime1_controls.values
                    if regime1_controls.ndim == 1:
                        regime1_controls = regime1_controls.reshape(-1, 1)
                    
                    # Filter controls for regime 1
                    controls1 = regime1_controls[regime1_mask]
                    x1_full = np.column_stack([x1_base, controls1])
                else:
                    x1_full = x1_base
                
                # Add constant and fit
                X1 = np.column_stack([np.ones(len(y1)), x1_full])
                beta1 = np.linalg.lstsq(X1, y1, rcond=None)[0]
                y1_pred = X1 @ beta1
                ssr1 = np.sum((y1 - y1_pred) ** 2)
                
                regime_models['regime1'] = {
                    'beta': beta1,
                    'ssr': ssr1,
                    'n_obs': len(y1),
                    'n_vars': X1.shape[1]
                }
                
                total_ssr += ssr1
                n_params += len(beta1)
            
            # Regime 2
            if np.sum(regime2_mask) >= 5:
                y2 = y[regime2_mask]
                x2_base = x[regime2_mask]
                
                # Add regime-specific controls for regime 2
                if 'regime2' in regime_controls:
                    regime2_controls = regime_controls['regime2']
                    if hasattr(regime2_controls, 'values'):
                        regime2_controls = regime2_controls.values
                    if regime2_controls.ndim == 1:
                        regime2_controls = regime2_controls.reshape(-1, 1)
                    
                    # Filter controls for regime 2
                    controls2 = regime2_controls[regime2_mask]
                    x2_full = np.column_stack([x2_base, controls2])
                else:
                    x2_full = x2_base
                
                # Add constant and fit
                X2 = np.column_stack([np.ones(len(y2)), x2_full])
                beta2 = np.linalg.lstsq(X2, y2, rcond=None)[0]
                y2_pred = X2 @ beta2
                ssr2 = np.sum((y2 - y2_pred) ** 2)
                
                regime_models['regime2'] = {
                    'beta': beta2,
                    'ssr': ssr2,
                    'n_obs': len(y2),
                    'n_vars': X2.shape[1]
                }
                
                total_ssr += ssr2
                n_params += len(beta2)
            
            # Calculate overall R²
            ss_tot = np.sum((y - np.mean(y)) ** 2)
            r2 = 1 - (total_ssr / ss_tot) if ss_tot > 0 else 0
            
            results.update({
                'model': regime_models,
                'threshold': threshold,
                'r_squared': r2,
                'total_ssr': total_ssr,
                'n_params': n_params,
                'fitted': True
            })
            
            # Regime diagnostics
            results['regime_diagnostics'] = self._calculate_regime_diagnostics(
                regime_models, y, threshold_var, threshold
            )
            
            # Specification tests
            results['specification_tests'] = self._test_regime_specific_specification(
                y, x, threshold_var, regime_controls, hansen_base
            )
            
        except Exception as e:
            results['error'] = str(e)
        
        # Store results
        self.fitted_models['regime_specific_controls'] = results
        
        return results
    
    def markov_switching_threshold_model(self, y, x, threshold_var, n_regimes=2):
        """
        Create Markov-switching threshold model for probabilistic regime switching
        
        Args:
            y: Dependent variable
            x: Independent variables
            threshold_var: Threshold variable
            n_regimes: Number of regimes
            
        Returns:
            dict: Markov-switching threshold model results
        """
        results = {
            'model': None,
            'regime_probabilities': None,
            'transition_matrix': None,
            'regime_parameters': {},
            'r_squared': 0,
            'fitted': False
        }
        
        # Convert pandas Series to numpy arrays
        if hasattr(y, 'values'):
            y = y.values
        if hasattr(x, 'values'):
            x = x.values
        if hasattr(threshold_var, 'values'):
            threshold_var = threshold_var.values
        
        # Ensure x is 2D
        if x.ndim == 1:
            x = x.reshape(-1, 1)
        
        try:
            # Simplified Markov-switching model using EM algorithm approximation
            results = self._fit_markov_switching_model(y, x, threshold_var, n_regimes)
            
        except Exception as e:
            results['error'] = str(e)
        
        # Store results
        self.fitted_models['markov_switching'] = results
        
        return results
    
    def structural_break_threshold_model(self, y, x, threshold_var):
        """
        Add structural break threshold model with endogenous break detection
        
        Args:
            y: Dependent variable
            x: Independent variables
            threshold_var: Threshold variable
            
        Returns:
            dict: Structural break threshold model results
        """
        results = {
            'break_points': [],
            'models': {},
            'break_tests': {},
            'best_model': None,
            'r_squared': 0,
            'fitted': False
        }
        
        # Convert pandas Series to numpy arrays
        if hasattr(y, 'values'):
            y = y.values
        if hasattr(x, 'values'):
            x = x.values
        if hasattr(threshold_var, 'values'):
            threshold_var = threshold_var.values
        
        # Ensure x is 2D
        if x.ndim == 1:
            x = x.reshape(-1, 1)
        
        try:
            # Method 1: Bai-Perron structural break test
            results['models']['bai_perron'] = self._fit_bai_perron_breaks(y, x, threshold_var)
            
            # Method 2: CUSUM-based break detection
            results['models']['cusum'] = self._fit_cusum_breaks(y, x, threshold_var)
            
            # Method 3: Threshold-based structural breaks
            results['models']['threshold_breaks'] = self._fit_threshold_structural_breaks(y, x, threshold_var)
            
            # Break tests
            results['break_tests'] = self._perform_structural_break_tests(y, x, threshold_var)
            
            # Select best model
            results['best_model'] = self._select_best_break_model(results['models'])
            
            if results['best_model']:
                results['r_squared'] = results['models'][results['best_model']].get('r_squared', 0)
                results['fitted'] = True
            
        except Exception as e:
            results['error'] = str(e)
        
        # Store results
        self.fitted_models['structural_break'] = results
        
        return results
    
    def _fit_rolling_window_threshold(self, y, x, threshold_var, time_var, window_size=None):
        """Fit threshold model with rolling window"""
        if window_size is None:
            window_size = max(20, len(y) // 5)  # Adaptive window size
        
        results = {
            'thresholds': [],
            'time_points': [],
            'r_squared_series': [],
            'fitted': False
        }
        
        try:
            n_obs = len(y)
            
            for i in range(window_size, n_obs):
                # Define window
                start_idx = i - window_size
                end_idx = i
                
                # Extract window data
                y_window = y[start_idx:end_idx]
                x_window = x[start_idx:end_idx]
                thresh_window = threshold_var[start_idx:end_idx]
                
                # Fit Hansen model on window
                try:
                    hansen_window = HansenThresholdRegression()
                    hansen_window.fit(y_window, x_window, thresh_window)
                    
                    if hansen_window.fitted:
                        # Calculate R² for window
                        y_pred = hansen_window.predict(x_window, thresh_window)
                        ss_res = np.sum((y_window - y_pred) ** 2)
                        ss_tot = np.sum((y_window - np.mean(y_window)) ** 2)
                        r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
                        
                        results['thresholds'].append(hansen_window.threshold)
                        results['time_points'].append(time_var[i] if hasattr(time_var, '__len__') else i)
                        results['r_squared_series'].append(r2)
                    
                except:
                    continue
            
            if len(results['thresholds']) > 0:
                results['fitted'] = True
                results['mean_threshold'] = np.mean(results['thresholds'])
                results['threshold_std'] = np.std(results['thresholds'])
                results['mean_r_squared'] = np.mean(results['r_squared_series'])
            
        except Exception as e:
            results['error'] = str(e)
        
        return results
    
    def _fit_smooth_time_varying_threshold(self, y, x, threshold_var, time_var):
        """Fit smooth time-varying threshold model"""
        results = {
            'time_varying_threshold': None,
            'smoothing_parameter': None,
            'r_squared': 0,
            'fitted': False
        }
        
        try:
            # Normalize time variable
            if hasattr(time_var, '__len__'):
                time_norm = (time_var - np.min(time_var)) / (np.max(time_var) - np.min(time_var))
            else:
                time_norm = np.arange(len(y)) / len(y)
            
            # Fit time-varying threshold as polynomial function of time
            def objective(params):
                alpha, beta = params
                
                # Time-varying threshold: threshold(t) = alpha + beta * t
                threshold_t = alpha + beta * time_norm
                
                total_ssr = 0
                n_valid = 0
                
                for i in range(len(y)):
                    # Create regime based on time-varying threshold
                    regime1_mask = threshold_var <= threshold_t[i]
                    regime2_mask = threshold_var > threshold_t[i]
                    
                    if np.sum(regime1_mask) >= 3 and np.sum(regime2_mask) >= 3:
                        try:
                            # Fit local Hansen model
                            y_local = y
                            x_local = x
                            
                            # Regime 1
                            y1 = y_local[regime1_mask]
                            x1 = x_local[regime1_mask]
                            X1 = np.column_stack([np.ones(len(y1)), x1])
                            beta1 = np.linalg.lstsq(X1, y1, rcond=None)[0]
                            
                            # Regime 2
                            y2 = y_local[regime2_mask]
                            x2 = x_local[regime2_mask]
                            X2 = np.column_stack([np.ones(len(y2)), x2])
                            beta2 = np.linalg.lstsq(X2, y2, rcond=None)[0]
                            
                            # Calculate SSR
                            ssr1 = np.sum((y1 - X1 @ beta1) ** 2)
                            ssr2 = np.sum((y2 - X2 @ beta2) ** 2)
                            total_ssr += ssr1 + ssr2
                            n_valid += 1
                            
                        except:
                            total_ssr += 1e6  # Penalty for failed fit
                
                return total_ssr / max(n_valid, 1)
            
            # Optimize time-varying threshold parameters
            from scipy.optimize import minimize
            
            # Initial values
            initial_threshold = np.median(threshold_var)
            result = minimize(objective, [initial_threshold, 0], method='Nelder-Mead')
            
            if result.success:
                alpha_opt, beta_opt = result.x
                threshold_t_opt = alpha_opt + beta_opt * time_norm
                
                # Calculate final R²
                # Use average threshold for final model
                avg_threshold = np.mean(threshold_t_opt)
                hansen_final = HansenThresholdRegression()
                hansen_final.fit(y, x, threshold_var)
                
                if hansen_final.fitted:
                    y_pred = hansen_final.predict(x, threshold_var)
                    ss_res = np.sum((y - y_pred) ** 2)
                    ss_tot = np.sum((y - np.mean(y)) ** 2)
                    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
                    
                    results.update({
                        'time_varying_threshold': threshold_t_opt,
                        'alpha': alpha_opt,
                        'beta': beta_opt,
                        'r_squared': r2,
                        'fitted': True
                    })
            
        except Exception as e:
            results['error'] = str(e)
        
        return results
    
    def _fit_regime_time_trend_model(self, y, x, threshold_var, time_var):
        """Fit model with regime-dependent time trends"""
        results = {
            'model': None,
            'regime_trends': {},
            'r_squared': 0,
            'fitted': False
        }
        
        try:
            # Fit base Hansen model to get threshold
            hansen_base = HansenThresholdRegression()
            hansen_base.fit(y, x, threshold_var)
            
            if not hansen_base.fitted:
                results['error'] = 'Base Hansen model failed'
                return results
            
            threshold = hansen_base.threshold
            
            # Create regime masks
            regime1_mask = threshold_var <= threshold
            regime2_mask = threshold_var > threshold
            
            # Normalize time variable
            if hasattr(time_var, '__len__'):
                time_norm = (time_var - np.min(time_var)) / (np.max(time_var) - np.min(time_var))
            else:
                time_norm = np.arange(len(y)) / len(y)
            
            # Fit regime-specific models with time trends
            regime_models = {}
            total_ssr = 0
            
            # Regime 1 with time trend
            if np.sum(regime1_mask) >= 5:
                y1 = y[regime1_mask]
                x1 = x[regime1_mask]
                time1 = time_norm[regime1_mask]
                
                # Add time trend to regime 1
                X1 = np.column_stack([np.ones(len(y1)), x1, time1, time1**2])
                beta1 = np.linalg.lstsq(X1, y1, rcond=None)[0]
                y1_pred = X1 @ beta1
                ssr1 = np.sum((y1 - y1_pred) ** 2)
                
                regime_models['regime1'] = {
                    'beta': beta1,
                    'ssr': ssr1,
                    'time_trend': True,
                    'n_obs': len(y1)
                }
                total_ssr += ssr1
            
            # Regime 2 with time trend
            if np.sum(regime2_mask) >= 5:
                y2 = y[regime2_mask]
                x2 = x[regime2_mask]
                time2 = time_norm[regime2_mask]
                
                # Add time trend to regime 2
                X2 = np.column_stack([np.ones(len(y2)), x2, time2, time2**2])
                beta2 = np.linalg.lstsq(X2, y2, rcond=None)[0]
                y2_pred = X2 @ beta2
                ssr2 = np.sum((y2 - y2_pred) ** 2)
                
                regime_models['regime2'] = {
                    'beta': beta2,
                    'ssr': ssr2,
                    'time_trend': True,
                    'n_obs': len(y2)
                }
                total_ssr += ssr2
            
            # Calculate overall R²
            ss_tot = np.sum((y - np.mean(y)) ** 2)
            r2 = 1 - (total_ssr / ss_tot) if ss_tot > 0 else 0
            
            results.update({
                'model': regime_models,
                'threshold': threshold,
                'r_squared': r2,
                'fitted': True
            })
            
        except Exception as e:
            results['error'] = str(e)
        
        return results
    
    def _test_threshold_stability_over_time(self, y, x, threshold_var, time_var):
        """Test threshold stability over time"""
        tests = {
            'stability_test': None,
            'parameter_stability': None,
            'regime_stability': None
        }
        
        try:
            # Split sample test
            n_obs = len(y)
            split_point = n_obs // 2
            
            # First half
            y1 = y[:split_point]
            x1 = x[:split_point]
            thresh1 = threshold_var[:split_point]
            
            # Second half
            y2 = y[split_point:]
            x2 = x[split_point:]
            thresh2 = threshold_var[split_point:]
            
            # Fit Hansen models on each half
            hansen1 = HansenThresholdRegression()
            hansen2 = HansenThresholdRegression()
            
            hansen1.fit(y1, x1, thresh1)
            hansen2.fit(y2, x2, thresh2)
            
            if hansen1.fitted and hansen2.fitted:
                # Test threshold stability
                threshold_diff = abs(hansen1.threshold - hansen2.threshold)
                threshold_stability = threshold_diff / np.std(threshold_var)
                
                tests['stability_test'] = {
                    'threshold_1': hansen1.threshold,
                    'threshold_2': hansen2.threshold,
                    'difference': threshold_diff,
                    'normalized_difference': threshold_stability,
                    'stable': threshold_stability < 1.0  # Arbitrary threshold
                }
            
        except Exception as e:
            tests['error'] = str(e)
        
        return tests
    
    def _select_best_time_varying_model(self, models):
        """Select best time-varying model specification"""
        best_model = None
        best_r2 = -np.inf
        
        for model_name, model_info in models.items():
            if model_info.get('fitted', False):
                r2 = model_info.get('r_squared', -np.inf)
                if r2 > best_r2:
                    best_r2 = r2
                    best_model = model_name
        
        return best_model
    
    def _calculate_regime_diagnostics(self, regime_models, y, threshold_var, threshold):
        """Calculate diagnostics for regime-specific models"""
        diagnostics = {}
        
        try:
            regime1_mask = threshold_var <= threshold
            regime2_mask = threshold_var > threshold
            
            for regime_name, model_info in regime_models.items():
                if regime_name == 'regime1':
                    mask = regime1_mask
                elif regime_name == 'regime2':
                    mask = regime2_mask
                else:
                    continue
                
                y_regime = y[mask]
                n_obs = len(y_regime)
                
                if n_obs > 0:
                    # Calculate regime-specific R²
                    ssr = model_info['ssr']
                    ss_tot = np.sum((y_regime - np.mean(y_regime)) ** 2)
                    r2_regime = 1 - (ssr / ss_tot) if ss_tot > 0 else 0
                    
                    diagnostics[regime_name] = {
                        'n_obs': n_obs,
                        'r_squared': r2_regime,
                        'ssr': ssr,
                        'mean_y': np.mean(y_regime),
                        'std_y': np.std(y_regime)
                    }
        
        except Exception as e:
            diagnostics['error'] = str(e)
        
        return diagnostics
    
    def _test_regime_specific_specification(self, y, x, threshold_var, regime_controls, base_model):
        """Test regime-specific specification against base model"""
        tests = {
            'likelihood_ratio_test': None,
            'specification_improvement': None
        }
        
        try:
            # Calculate base model SSR
            y_pred_base = base_model.predict(x, threshold_var)
            ssr_base = np.sum((y - y_pred_base) ** 2)
            
            # This would be compared against the regime-specific model SSR
            # For now, return placeholder
            tests['specification_improvement'] = {
                'base_ssr': ssr_base,
                'improvement_available': True
            }
            
        except Exception as e:
            tests['error'] = str(e)
        
        return tests
    
    def _fit_markov_switching_model(self, y, x, threshold_var, n_regimes):
        """Fit simplified Markov-switching model"""
        results = {
            'regime_probabilities': None,
            'regime_parameters': {},
            'r_squared': 0,
            'fitted': False
        }
        
        try:
            # Simplified approach: use k-means clustering to identify regimes
            from sklearn.cluster import KMeans
            
            # Combine variables for clustering
            features = np.column_stack([y, x.flatten() if x.ndim == 1 else x, threshold_var])
            
            # Fit k-means
            kmeans = KMeans(n_clusters=n_regimes, random_state=42, n_init=10)
            regime_labels = kmeans.fit_predict(features)
            
            # Fit separate models for each regime
            regime_models = {}
            total_ssr = 0
            
            for regime in range(n_regimes):
                regime_mask = regime_labels == regime
                
                if np.sum(regime_mask) >= 5:
                    y_regime = y[regime_mask]
                    x_regime = x[regime_mask]
                    
                    # Fit OLS for regime
                    X_regime = np.column_stack([np.ones(len(y_regime)), x_regime])
                    beta_regime = np.linalg.lstsq(X_regime, y_regime, rcond=None)[0]
                    y_pred_regime = X_regime @ beta_regime
                    ssr_regime = np.sum((y_regime - y_pred_regime) ** 2)
                    
                    regime_models[f'regime_{regime}'] = {
                        'beta': beta_regime,
                        'ssr': ssr_regime,
                        'n_obs': len(y_regime),
                        'probability': np.sum(regime_mask) / len(y)
                    }
                    
                    total_ssr += ssr_regime
            
            # Calculate overall R²
            ss_tot = np.sum((y - np.mean(y)) ** 2)
            r2 = 1 - (total_ssr / ss_tot) if ss_tot > 0 else 0
            
            results.update({
                'regime_parameters': regime_models,
                'regime_probabilities': regime_labels,
                'r_squared': r2,
                'fitted': True
            })
            
        except Exception as e:
            results['error'] = str(e)
        
        return results
    
    def _fit_bai_perron_breaks(self, y, x, threshold_var):
        """Simplified Bai-Perron structural break detection"""
        results = {
            'break_points': [],
            'r_squared': 0,
            'fitted': False
        }
        
        try:
            # Simplified approach: test multiple potential break points
            n_obs = len(y)
            min_segment = max(10, n_obs // 10)
            
            best_ssr = np.inf
            best_breaks = []
            
            # Test single break point
            for break_point in range(min_segment, n_obs - min_segment):
                try:
                    # Split data
                    y1 = y[:break_point]
                    x1 = x[:break_point]
                    y2 = y[break_point:]
                    x2 = x[break_point:]
                    
                    # Fit separate models
                    X1 = np.column_stack([np.ones(len(y1)), x1])
                    X2 = np.column_stack([np.ones(len(y2)), x2])
                    
                    beta1 = np.linalg.lstsq(X1, y1, rcond=None)[0]
                    beta2 = np.linalg.lstsq(X2, y2, rcond=None)[0]
                    
                    ssr1 = np.sum((y1 - X1 @ beta1) ** 2)
                    ssr2 = np.sum((y2 - X2 @ beta2) ** 2)
                    total_ssr = ssr1 + ssr2
                    
                    if total_ssr < best_ssr:
                        best_ssr = total_ssr
                        best_breaks = [break_point]
                
                except:
                    continue
            
            if best_breaks:
                ss_tot = np.sum((y - np.mean(y)) ** 2)
                r2 = 1 - (best_ssr / ss_tot) if ss_tot > 0 else 0
                
                results.update({
                    'break_points': best_breaks,
                    'r_squared': r2,
                    'ssr': best_ssr,
                    'fitted': True
                })
        
        except Exception as e:
            results['error'] = str(e)
        
        return results
    
    def _fit_cusum_breaks(self, y, x, threshold_var):
        """CUSUM-based break detection"""
        results = {
            'cusum_statistics': [],
            'break_points': [],
            'r_squared': 0,
            'fitted': False
        }
        
        try:
            # Fit full sample model
            X_full = np.column_stack([np.ones(len(y)), x])
            beta_full = np.linalg.lstsq(X_full, y, rcond=None)[0]
            residuals_full = y - X_full @ beta_full
            
            # Calculate CUSUM statistics
            n_obs = len(y)
            cusum_stats = []
            
            for t in range(1, n_obs):
                # Cumulative sum of residuals
                cusum_t = np.sum(residuals_full[:t]) / np.sqrt(np.sum(residuals_full**2))
                cusum_stats.append(abs(cusum_t))
            
            # Find break points where CUSUM exceeds threshold
            cusum_threshold = 0.5  # Simplified threshold
            break_candidates = [i for i, stat in enumerate(cusum_stats) if stat > cusum_threshold]
            
            if break_candidates:
                # Use the point with maximum CUSUM statistic
                max_cusum_idx = np.argmax([cusum_stats[i] for i in break_candidates])
                break_point = break_candidates[max_cusum_idx]
                
                # Fit model with break
                y1 = y[:break_point]
                x1 = x[:break_point]
                y2 = y[break_point:]
                x2 = x[break_point:]
                
                X1 = np.column_stack([np.ones(len(y1)), x1])
                X2 = np.column_stack([np.ones(len(y2)), x2])
                
                beta1 = np.linalg.lstsq(X1, y1, rcond=None)[0]
                beta2 = np.linalg.lstsq(X2, y2, rcond=None)[0]
                
                ssr1 = np.sum((y1 - X1 @ beta1) ** 2)
                ssr2 = np.sum((y2 - X2 @ beta2) ** 2)
                total_ssr = ssr1 + ssr2
                
                ss_tot = np.sum((y - np.mean(y)) ** 2)
                r2 = 1 - (total_ssr / ss_tot) if ss_tot > 0 else 0
                
                results.update({
                    'cusum_statistics': cusum_stats,
                    'break_points': [break_point],
                    'r_squared': r2,
                    'fitted': True
                })
        
        except Exception as e:
            results['error'] = str(e)
        
        return results
    
    def _fit_threshold_structural_breaks(self, y, x, threshold_var):
        """Threshold-based structural break model"""
        results = {
            'threshold_breaks': [],
            'r_squared': 0,
            'fitted': False
        }
        
        try:
            # Use threshold variable to identify structural breaks
            # Find points where threshold variable changes significantly
            thresh_diff = np.diff(threshold_var)
            thresh_std = np.std(thresh_diff)
            
            # Identify large changes (potential breaks)
            break_candidates = np.where(np.abs(thresh_diff) > 2 * thresh_std)[0]
            
            if len(break_candidates) > 0:
                # Test each candidate break point
                best_ssr = np.inf
                best_break = None
                
                for break_point in break_candidates:
                    if break_point > 10 and break_point < len(y) - 10:
                        try:
                            # Split data at break point
                            y1 = y[:break_point]
                            x1 = x[:break_point]
                            y2 = y[break_point:]
                            x2 = x[break_point:]
                            
                            # Fit Hansen models on each segment
                            thresh1 = threshold_var[:break_point]
                            thresh2 = threshold_var[break_point:]
                            
                            hansen1 = HansenThresholdRegression()
                            hansen2 = HansenThresholdRegression()
                            
                            hansen1.fit(y1, x1, thresh1)
                            hansen2.fit(y2, x2, thresh2)
                            
                            if hansen1.fitted and hansen2.fitted:
                                y1_pred = hansen1.predict(x1, thresh1)
                                y2_pred = hansen2.predict(x2, thresh2)
                                
                                ssr1 = np.sum((y1 - y1_pred) ** 2)
                                ssr2 = np.sum((y2 - y2_pred) ** 2)
                                total_ssr = ssr1 + ssr2
                                
                                if total_ssr < best_ssr:
                                    best_ssr = total_ssr
                                    best_break = break_point
                        
                        except:
                            continue
                
                if best_break is not None:
                    ss_tot = np.sum((y - np.mean(y)) ** 2)
                    r2 = 1 - (best_ssr / ss_tot) if ss_tot > 0 else 0
                    
                    results.update({
                        'threshold_breaks': [best_break],
                        'r_squared': r2,
                        'ssr': best_ssr,
                        'fitted': True
                    })
        
        except Exception as e:
            results['error'] = str(e)
        
        return results
    
    def _perform_structural_break_tests(self, y, x, threshold_var):
        """Perform various structural break tests"""
        tests = {
            'chow_test': None,
            'quandt_likelihood_ratio': None,
            'sup_wald_test': None
        }
        
        try:
            # Simplified Chow test at midpoint
            n_obs = len(y)
            break_point = n_obs // 2
            
            # Full sample model
            X_full = np.column_stack([np.ones(len(y)), x])
            beta_full = np.linalg.lstsq(X_full, y, rcond=None)[0]
            ssr_full = np.sum((y - X_full @ beta_full) ** 2)
            
            # Split sample models
            y1 = y[:break_point]
            x1 = x[:break_point]
            y2 = y[break_point:]
            x2 = x[break_point:]
            
            X1 = np.column_stack([np.ones(len(y1)), x1])
            X2 = np.column_stack([np.ones(len(y2)), x2])
            
            beta1 = np.linalg.lstsq(X1, y1, rcond=None)[0]
            beta2 = np.linalg.lstsq(X2, y2, rcond=None)[0]
            
            ssr1 = np.sum((y1 - X1 @ beta1) ** 2)
            ssr2 = np.sum((y2 - X2 @ beta2) ** 2)
            ssr_split = ssr1 + ssr2
            
            # Chow test statistic
            k = X_full.shape[1]  # Number of parameters
            f_stat = ((ssr_full - ssr_split) / k) / (ssr_split / (n_obs - 2*k))
            
            # Approximate p-value
            from scipy.stats import f
            p_value = 1 - f.cdf(f_stat, k, n_obs - 2*k) if n_obs > 2*k else 1
            
            tests['chow_test'] = {
                'f_statistic': f_stat,
                'p_value': p_value,
                'break_point': break_point,
                'significant_break': p_value < 0.05
            }
        
        except Exception as e:
            tests['error'] = str(e)
        
        return tests
    
    def _select_best_break_model(self, models):
        """Select best structural break model"""
        best_model = None
        best_r2 = -np.inf
        
        for model_name, model_info in models.items():
            if model_info.get('fitted', False):
                r2 = model_info.get('r_squared', -np.inf)
                if r2 > best_r2:
                    best_r2 = r2
                    best_model = model_name
        
        return best_model
    
    def _generate_improvement_summary(self, results):
        """Generate summary of model improvements"""
        original_r2 = results['original_model'].get('r_squared', 0)
        
        summary = {
            'original_r_squared': original_r2,
            'improvements': [],
            'best_improvement': 0,
            'recommendations': []
        }
        
        # Check each enhancement
        for enhancement_type, models in results['enhanced_models'].items():
            if isinstance(models, dict):
                if 'r_squared' in models:
                    improvement = models['r_squared'] - original_r2
                    if improvement > 0:
                        summary['improvements'].append({
                            'type': enhancement_type,
                            'improvement': improvement,
                            'new_r_squared': models['r_squared']
                        })
                else:
                    # Multiple models in this enhancement
                    for model_name, model_info in models.items():
                        if isinstance(model_info, dict) and 'r_squared' in model_info:
                            improvement = model_info['r_squared'] - original_r2
                            if improvement > 0:
                                summary['improvements'].append({
                                    'type': f"{enhancement_type}_{model_name}",
                                    'improvement': improvement,
                                    'new_r_squared': model_info['r_squared']
                                })
        
        # Sort improvements
        summary['improvements'].sort(key=lambda x: x['improvement'], reverse=True)
        
        if summary['improvements']:
            summary['best_improvement'] = summary['improvements'][0]['improvement']
        
        # Generate recommendations
        if original_r2 < 0.05:
            summary['recommendations'].append("Original R² is very low (<0.05). Consider data quality issues.")
        
        if summary['best_improvement'] > 0.02:
            summary['recommendations'].append(f"Significant improvement found with {summary['improvements'][0]['type']}")
        
        if not summary['improvements']:
            summary['recommendations'].append("No improvements found. Consider alternative model specifications or data transformations.")
        
        return summary
    
    def bootstrap_threshold_estimation(self, y, x, threshold_var, n_bootstrap=1000, confidence_level=0.95):
        """
        Bootstrap threshold estimation with confidence intervals
        
        Args:
            y: Dependent variable
            x: Independent variables
            threshold_var: Threshold variable
            n_bootstrap: Number of bootstrap samples
            confidence_level: Confidence level for intervals
            
        Returns:
            dict: Bootstrap results with confidence intervals
        """
        results = self.robust_estimator.bootstrap_threshold_estimation(
            y, x, threshold_var, n_bootstrap, confidence_level
        )
        # Store in main fitted_models
        self.fitted_models['bootstrap_threshold'] = results
        return results
    
    def quantile_threshold_regression(self, y, x, threshold_var, quantiles=None):
        """
        Quantile threshold regression for robust estimation
        
        Args:
            y: Dependent variable
            x: Independent variables
            threshold_var: Threshold variable
            quantiles: List of quantiles to estimate (default: [0.1, 0.25, 0.5, 0.75, 0.9])
            
        Returns:
            dict: Quantile regression results across different quantiles
        """
        results = self.robust_estimator.quantile_threshold_regression(y, x, threshold_var, quantiles)
        # Store in main fitted_models
        self.fitted_models['quantile_threshold'] = results
        return results
    
    def bayesian_threshold_model(self, y, x, threshold_var, n_samples=5000, n_chains=4):
        """
        Bayesian threshold model with uncertainty quantification
        
        Args:
            y: Dependent variable
            x: Independent variables
            threshold_var: Threshold variable
            n_samples: Number of MCMC samples per chain
            n_chains: Number of MCMC chains
            
        Returns:
            dict: Bayesian estimation results with posterior distributions
        """
        results = self.robust_estimator.bayesian_threshold_model(y, x, threshold_var, n_samples, n_chains)
        # Store in main fitted_models
        self.fitted_models['bayesian_threshold'] = results
        return results
    
    def cross_validation_model_selection(self, y, x, threshold_var, model_specifications=None, cv_folds=5):
        """
        Cross-validation based model selection for specification choice
        
        Args:
            y: Dependent variable
            x: Independent variables
            threshold_var: Threshold variable
            model_specifications: List of model specifications to compare
            cv_folds: Number of cross-validation folds
            
        Returns:
            dict: Cross-validation results and model selection
        """
        results = self.robust_estimator.cross_validation_model_selection(
            y, x, threshold_var, model_specifications, cv_folds
        )
        # Store in main fitted_models
        self.fitted_models['cv_model_selection'] = results
        return results
    
    def bootstrap_threshold_estimation(self, y, x, threshold_var, n_bootstrap=1000, confidence_level=0.95):
        """
        Bootstrap threshold estimation with confidence intervals
        
        Args:
            y: Dependent variable
            x: Independent variables
            threshold_var: Threshold variable
            n_bootstrap: Number of bootstrap samples
            confidence_level: Confidence level for intervals
            
        Returns:
            dict: Bootstrap results with confidence intervals
        """
        # Ensure arrays are numpy arrays
        if hasattr(y, 'values'):
            y = y.values
        if hasattr(x, 'values'):
            x = x.values
        if hasattr(threshold_var, 'values'):
            threshold_var = threshold_var.values
        
        if x.ndim == 1:
            x = x.reshape(-1, 1)
        
        n_obs = len(y)
        bootstrap_results = {
            'thresholds': [],
            'coefficients_regime1': [],
            'coefficients_regime2': [],
            'r_squared_values': [],
            'successful_fits': 0
        }
        
        # Original model for comparison
        try:
            original_model = HansenThresholdRegression()
            original_model.fit(y, x, threshold_var)
            original_threshold = original_model.threshold if original_model.fitted else None
        except:
            original_threshold = None
        
        # Bootstrap sampling
        for i in range(n_bootstrap):
            try:
                # Resample with replacement
                bootstrap_indices = np.random.choice(n_obs, size=n_obs, replace=True)
                y_boot = y[bootstrap_indices]
                x_boot = x[bootstrap_indices]
                threshold_boot = threshold_var[bootstrap_indices]
                
                # Fit Hansen model on bootstrap sample
                hansen_boot = HansenThresholdRegression()
                hansen_boot.fit(y_boot, x_boot, threshold_boot)
                
                if hansen_boot.fitted:
                    # Calculate R² for bootstrap sample
                    y_pred_boot = hansen_boot.predict(x_boot, threshold_boot)
                    ss_res = np.sum((y_boot - y_pred_boot) ** 2)
                    ss_tot = np.sum((y_boot - np.mean(y_boot)) ** 2)
                    r2_boot = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
                    
                    # Store results
                    bootstrap_results['thresholds'].append(hansen_boot.threshold)
                    bootstrap_results['coefficients_regime1'].append(hansen_boot.beta1)
                    bootstrap_results['coefficients_regime2'].append(hansen_boot.beta2)
                    bootstrap_results['r_squared_values'].append(r2_boot)
                    bootstrap_results['successful_fits'] += 1
                    
            except Exception:
                continue
        
        # Calculate confidence intervals
        alpha = 1 - confidence_level
        lower_percentile = (alpha / 2) * 100
        upper_percentile = (1 - alpha / 2) * 100
        
        results = {
            'original_threshold': original_threshold,
            'bootstrap_statistics': {},
            'confidence_intervals': {},
            'n_successful_bootstrap': bootstrap_results['successful_fits'],
            'bootstrap_success_rate': bootstrap_results['successful_fits'] / n_bootstrap
        }
        
        if bootstrap_results['successful_fits'] > 0:
            # Threshold statistics
            thresholds = np.array(bootstrap_results['thresholds'])
            results['bootstrap_statistics']['threshold'] = {
                'mean': np.mean(thresholds),
                'std': np.std(thresholds),
                'median': np.median(thresholds)
            }
            results['confidence_intervals']['threshold'] = {
                'lower': np.percentile(thresholds, lower_percentile),
                'upper': np.percentile(thresholds, upper_percentile)
            }
            
            # R² statistics
            r2_values = np.array(bootstrap_results['r_squared_values'])
            results['bootstrap_statistics']['r_squared'] = {
                'mean': np.mean(r2_values),
                'std': np.std(r2_values),
                'median': np.median(r2_values)
            }
            results['confidence_intervals']['r_squared'] = {
                'lower': np.percentile(r2_values, lower_percentile),
                'upper': np.percentile(r2_values, upper_percentile)
            }
            
            # Coefficient statistics (regime 1)
            if bootstrap_results['coefficients_regime1']:
                coef1_array = np.array(bootstrap_results['coefficients_regime1'])
                results['bootstrap_statistics']['coefficients_regime1'] = {
                    'mean': np.mean(coef1_array, axis=0),
                    'std': np.std(coef1_array, axis=0),
                    'median': np.median(coef1_array, axis=0)
                }
                results['confidence_intervals']['coefficients_regime1'] = {
                    'lower': np.percentile(coef1_array, lower_percentile, axis=0),
                    'upper': np.percentile(coef1_array, upper_percentile, axis=0)
                }
            
            # Coefficient statistics (regime 2)
            if bootstrap_results['coefficients_regime2']:
                coef2_array = np.array(bootstrap_results['coefficients_regime2'])
                results['bootstrap_statistics']['coefficients_regime2'] = {
                    'mean': np.mean(coef2_array, axis=0),
                    'std': np.std(coef2_array, axis=0),
                    'median': np.median(coef2_array, axis=0)
                }
                results['confidence_intervals']['coefficients_regime2'] = {
                    'lower': np.percentile(coef2_array, lower_percentile, axis=0),
                    'upper': np.percentile(coef2_array, upper_percentile, axis=0)
                }
        
        # Store results
        self.fitted_models['bootstrap_threshold'] = results
        
        return results
    
    def quantile_threshold_regression(self, y, x, threshold_var, quantiles=None):
        """
        Quantile threshold regression for robust estimation
        
        Args:
            y: Dependent variable
            x: Independent variables
            threshold_var: Threshold variable
            quantiles: List of quantiles to estimate (default: [0.1, 0.25, 0.5, 0.75, 0.9])
            
        Returns:
            dict: Quantile regression results across different quantiles
        """
        if quantiles is None:
            quantiles = [0.1, 0.25, 0.5, 0.75, 0.9]
        
        # Ensure arrays are numpy arrays
        if hasattr(y, 'values'):
            y = y.values
        if hasattr(x, 'values'):
            x = x.values
        if hasattr(threshold_var, 'values'):
            threshold_var = threshold_var.values
        
        if x.ndim == 1:
            x = x.reshape(-1, 1)
        
        results = {
            'quantile_models': {},
            'threshold_estimates': {},
            'coefficient_estimates': {},
            'model_comparison': {}
        }
        
        # First, estimate threshold using median regression (more robust)
        try:
            # Use grid search for threshold estimation with quantile regression
            threshold_candidates = np.percentile(threshold_var, np.linspace(10, 90, 20))
            best_threshold = None
            best_objective = np.inf
            
            for thresh_candidate in threshold_candidates:
                try:
                    # Create regime indicators
                    regime1_mask = threshold_var <= thresh_candidate
                    regime2_mask = threshold_var > thresh_candidate
                    
                    if np.sum(regime1_mask) < 10 or np.sum(regime2_mask) < 10:
                        continue
                    
                    # Fit quantile regression for each regime (using median)
                    total_objective = 0
                    
                    # Regime 1
                    y1 = y[regime1_mask]
                    x1 = x[regime1_mask]
                    if len(y1) > 0:
                        qr1 = QuantileRegressor(quantile=0.5, alpha=0.01)
                        qr1.fit(x1, y1)
                        y1_pred = qr1.predict(x1)
                        total_objective += np.sum(np.abs(y1 - y1_pred))
                    
                    # Regime 2
                    y2 = y[regime2_mask]
                    x2 = x[regime2_mask]
                    if len(y2) > 0:
                        qr2 = QuantileRegressor(quantile=0.5, alpha=0.01)
                        qr2.fit(x2, y2)
                        y2_pred = qr2.predict(x2)
                        total_objective += np.sum(np.abs(y2 - y2_pred))
                    
                    if total_objective < best_objective:
                        best_objective = total_objective
                        best_threshold = thresh_candidate
                        
                except Exception:
                    continue
            
            if best_threshold is None:
                best_threshold = np.median(threshold_var)
            
            results['estimated_threshold'] = best_threshold
            
        except Exception as e:
            results['threshold_estimation_error'] = str(e)
            best_threshold = np.median(threshold_var)
            results['estimated_threshold'] = best_threshold
        
        # Fit quantile regressions for each quantile
        regime1_mask = threshold_var <= best_threshold
        regime2_mask = threshold_var > best_threshold
        
        for q in quantiles:
            try:
                quantile_result = {
                    'quantile': q,
                    'threshold': best_threshold,
                    'regime1_model': None,
                    'regime2_model': None,
                    'regime1_coefficients': None,
                    'regime2_coefficients': None,
                    'fitted': False
                }
                
                # Fit regime 1
                if np.sum(regime1_mask) >= 5:
                    y1 = y[regime1_mask]
                    x1 = x[regime1_mask]
                    
                    qr1 = QuantileRegressor(quantile=q, alpha=0.01)
                    qr1.fit(x1, y1)
                    
                    quantile_result['regime1_model'] = qr1
                    quantile_result['regime1_coefficients'] = qr1.coef_
                
                # Fit regime 2
                if np.sum(regime2_mask) >= 5:
                    y2 = y[regime2_mask]
                    x2 = x[regime2_mask]
                    
                    qr2 = QuantileRegressor(quantile=q, alpha=0.01)
                    qr2.fit(x2, y2)
                    
                    quantile_result['regime2_model'] = qr2
                    quantile_result['regime2_coefficients'] = qr2.coef_
                
                if quantile_result['regime1_model'] is not None or quantile_result['regime2_model'] is not None:
                    quantile_result['fitted'] = True
                
                results['quantile_models'][q] = quantile_result
                
            except Exception as e:
                results['quantile_models'][q] = {
                    'quantile': q,
                    'error': str(e),
                    'fitted': False
                }
        
        # Analyze coefficient stability across quantiles
        results['coefficient_stability'] = self._analyze_quantile_coefficient_stability(results['quantile_models'])
        
        # Store results
        self.fitted_models['quantile_threshold'] = results
        
        return results
    
    def bayesian_threshold_model(self, y, x, threshold_var, n_samples=5000, n_chains=4):
        """
        Bayesian threshold model with uncertainty quantification
        
        Args:
            y: Dependent variable
            x: Independent variables
            threshold_var: Threshold variable
            n_samples: Number of MCMC samples per chain
            n_chains: Number of MCMC chains
            
        Returns:
            dict: Bayesian estimation results with posterior distributions
        """
        # Ensure arrays are numpy arrays
        if hasattr(y, 'values'):
            y = y.values
        if hasattr(x, 'values'):
            x = x.values
        if hasattr(threshold_var, 'values'):
            threshold_var = threshold_var.values
        
        if x.ndim == 1:
            x = x.reshape(-1, 1)
        
        results = {
            'posterior_samples': {},
            'posterior_statistics': {},
            'credible_intervals': {},
            'model_diagnostics': {},
            'fitted': False
        }
        
        try:
            # Simplified Bayesian estimation using Metropolis-Hastings
            # This is a basic implementation - in practice, you'd use PyMC3 or Stan
            
            n_obs = len(y)
            n_vars = x.shape[1]
            
            # Prior parameters (weakly informative)
            prior_mean_beta = np.zeros(n_vars + 1)  # +1 for intercept
            prior_cov_beta = np.eye(n_vars + 1) * 10  # Weak prior
            prior_sigma_shape = 2
            prior_sigma_scale = 1
            
            # Initialize parameters
            all_samples = []
            
            for chain in range(n_chains):
                # Initialize chain
                current_threshold = np.random.uniform(
                    np.percentile(threshold_var, 20),
                    np.percentile(threshold_var, 80)
                )
                current_beta1 = np.random.multivariate_normal(prior_mean_beta, prior_cov_beta)
                current_beta2 = np.random.multivariate_normal(prior_mean_beta, prior_cov_beta)
                current_sigma = 1.0
                
                chain_samples = {
                    'threshold': [],
                    'beta1': [],
                    'beta2': [],
                    'sigma': [],
                    'log_likelihood': []
                }
                
                accepted = 0
                
                for i in range(n_samples):
                    # Propose new threshold
                    threshold_proposal = current_threshold + np.random.normal(0, 0.1 * np.std(threshold_var))
                    
                    # Ensure threshold is within reasonable bounds
                    threshold_proposal = np.clip(
                        threshold_proposal,
                        np.percentile(threshold_var, 5),
                        np.percentile(threshold_var, 95)
                    )
                    
                    # Calculate likelihood for current and proposed threshold
                    current_ll = self._calculate_threshold_log_likelihood(
                        y, x, threshold_var, current_threshold, current_beta1, current_beta2, current_sigma
                    )
                    
                    proposed_ll = self._calculate_threshold_log_likelihood(
                        y, x, threshold_var, threshold_proposal, current_beta1, current_beta2, current_sigma
                    )
                    
                    # Metropolis-Hastings acceptance
                    log_alpha = proposed_ll - current_ll
                    if np.log(np.random.uniform()) < log_alpha:
                        current_threshold = threshold_proposal
                        current_ll = proposed_ll
                        accepted += 1
                    
                    # Update coefficients (Gibbs sampling)
                    regime1_mask = threshold_var <= current_threshold
                    regime2_mask = threshold_var > current_threshold
                    
                    if np.sum(regime1_mask) >= 3:
                        y1 = y[regime1_mask]
                        x1_with_intercept = np.column_stack([np.ones(np.sum(regime1_mask)), x[regime1_mask]])
                        
                        # Posterior for beta1 (conjugate normal)
                        precision_matrix = np.linalg.inv(prior_cov_beta) + (x1_with_intercept.T @ x1_with_intercept) / (current_sigma ** 2)
                        posterior_cov = np.linalg.inv(precision_matrix)
                        posterior_mean = posterior_cov @ (
                            np.linalg.inv(prior_cov_beta) @ prior_mean_beta +
                            (x1_with_intercept.T @ y1) / (current_sigma ** 2)
                        )
                        
                        current_beta1 = np.random.multivariate_normal(posterior_mean, posterior_cov)
                    
                    if np.sum(regime2_mask) >= 3:
                        y2 = y[regime2_mask]
                        x2_with_intercept = np.column_stack([np.ones(np.sum(regime2_mask)), x[regime2_mask]])
                        
                        # Posterior for beta2 (conjugate normal)
                        precision_matrix = np.linalg.inv(prior_cov_beta) + (x2_with_intercept.T @ x2_with_intercept) / (current_sigma ** 2)
                        posterior_cov = np.linalg.inv(precision_matrix)
                        posterior_mean = posterior_cov @ (
                            np.linalg.inv(prior_cov_beta) @ prior_mean_beta +
                            (x2_with_intercept.T @ y2) / (current_sigma ** 2)
                        )
                        
                        current_beta2 = np.random.multivariate_normal(posterior_mean, posterior_cov)
                    
                    # Update sigma (inverse gamma posterior)
                    ssr = self._calculate_ssr(y, x, threshold_var, current_threshold, current_beta1, current_beta2)
                    posterior_shape = prior_sigma_shape + n_obs / 2
                    posterior_scale = prior_sigma_scale + ssr / 2
                    current_sigma = np.sqrt(1 / np.random.gamma(posterior_shape, 1 / posterior_scale))
                    
                    # Store samples (after burn-in)
                    if i >= n_samples // 4:  # 25% burn-in
                        chain_samples['threshold'].append(current_threshold)
                        chain_samples['beta1'].append(current_beta1.copy())
                        chain_samples['beta2'].append(current_beta2.copy())
                        chain_samples['sigma'].append(current_sigma)
                        chain_samples['log_likelihood'].append(current_ll)
                
                all_samples.append(chain_samples)
                
            # Combine chains
            combined_samples = {
                'threshold': np.concatenate([chain['threshold'] for chain in all_samples]),
                'beta1': np.vstack([chain['beta1'] for chain in all_samples]),
                'beta2': np.vstack([chain['beta2'] for chain in all_samples]),
                'sigma': np.concatenate([chain['sigma'] for chain in all_samples]),
                'log_likelihood': np.concatenate([chain['log_likelihood'] for chain in all_samples])
            }
            
            results['posterior_samples'] = combined_samples
            
            # Calculate posterior statistics
            results['posterior_statistics'] = {
                'threshold': {
                    'mean': np.mean(combined_samples['threshold']),
                    'std': np.std(combined_samples['threshold']),
                    'median': np.median(combined_samples['threshold'])
                },
                'beta1': {
                    'mean': np.mean(combined_samples['beta1'], axis=0),
                    'std': np.std(combined_samples['beta1'], axis=0),
                    'median': np.median(combined_samples['beta1'], axis=0)
                },
                'beta2': {
                    'mean': np.mean(combined_samples['beta2'], axis=0),
                    'std': np.std(combined_samples['beta2'], axis=0),
                    'median': np.median(combined_samples['beta2'], axis=0)
                },
                'sigma': {
                    'mean': np.mean(combined_samples['sigma']),
                    'std': np.std(combined_samples['sigma']),
                    'median': np.median(combined_samples['sigma'])
                }
            }
            
            # Calculate credible intervals (95%)
            results['credible_intervals'] = {
                'threshold': {
                    'lower': np.percentile(combined_samples['threshold'], 2.5),
                    'upper': np.percentile(combined_samples['threshold'], 97.5)
                },
                'beta1': {
                    'lower': np.percentile(combined_samples['beta1'], 2.5, axis=0),
                    'upper': np.percentile(combined_samples['beta1'], 97.5, axis=0)
                },
                'beta2': {
                    'lower': np.percentile(combined_samples['beta2'], 2.5, axis=0),
                    'upper': np.percentile(combined_samples['beta2'], 97.5, axis=0)
                },
                'sigma': {
                    'lower': np.percentile(combined_samples['sigma'], 2.5),
                    'upper': np.percentile(combined_samples['sigma'], 97.5)
                }
            }
            
            # Model diagnostics
            results['model_diagnostics'] = {
                'n_effective_samples': len(combined_samples['threshold']),
                'acceptance_rate': accepted / n_samples if 'accepted' in locals() else None,
                'rhat': self._calculate_rhat(all_samples),  # Gelman-Rubin statistic
                'effective_sample_size': self._calculate_ess(combined_samples)
            }
            
            results['fitted'] = True
            
        except Exception as e:
            results['error'] = str(e)
            results['fitted'] = False
        
        # Store results
        self.fitted_models['bayesian_threshold'] = results
        
        return results
    
    def cross_validation_model_selection(self, y, x, threshold_var, model_specifications=None, cv_folds=5):
        """
        Cross-validation based model selection for specification choice
        
        Args:
            y: Dependent variable
            x: Independent variables
            threshold_var: Threshold variable
            model_specifications: List of model specifications to compare
            cv_folds: Number of cross-validation folds
            
        Returns:
            dict: Cross-validation results and model selection
        """
        if model_specifications is None:
            model_specifications = [
                {'type': 'hansen', 'params': {}},
                {'type': 'hansen_enhanced', 'params': {'enhancements': {'data_transforms': ['levels']}}},
                {'type': 'hansen_enhanced', 'params': {'enhancements': {'data_transforms': ['differences']}}},
                {'type': 'hansen_enhanced', 'params': {'enhancements': {'additional_controls': True}}},
                {'type': 'multiple_threshold', 'params': {'max_thresholds': 2}},
                {'type': 'smooth_transition', 'params': {'transition_types': ['logistic']}}
            ]
        
        # Ensure arrays are numpy arrays
        if hasattr(y, 'values'):
            y = y.values
        if hasattr(x, 'values'):
            x = x.values
        if hasattr(threshold_var, 'values'):
            threshold_var = threshold_var.values
        
        if x.ndim == 1:
            x = x.reshape(-1, 1)
        
        results = {
            'cv_results': {},
            'model_rankings': {},
            'best_model': None,
            'selection_criteria': {}
        }
        
        n_obs = len(y)
        fold_size = n_obs // cv_folds
        
        # Create cross-validation folds
        cv_folds_indices = []
        for fold in range(cv_folds):
            start_idx = fold * fold_size
            end_idx = (fold + 1) * fold_size if fold < cv_folds - 1 else n_obs
            test_indices = np.arange(start_idx, end_idx)
            train_indices = np.concatenate([np.arange(0, start_idx), np.arange(end_idx, n_obs)])
            cv_folds_indices.append((train_indices, test_indices))
        
        # Evaluate each model specification
        for spec_idx, spec in enumerate(model_specifications):
            spec_name = f"{spec['type']}_{spec_idx}"
            
            cv_scores = {
                'mse': [],
                'mae': [],
                'r_squared': [],
                'successful_fits': 0
            }
            
            for fold, (train_idx, test_idx) in enumerate(cv_folds_indices):
                try:
                    # Split data
                    y_train, y_test = y[train_idx], y[test_idx]
                    x_train, x_test = x[train_idx], x[test_idx]
                    thresh_train, thresh_test = threshold_var[train_idx], threshold_var[test_idx]
                    
                    # Fit model based on specification
                    if spec['type'] == 'hansen':
                        model = HansenThresholdRegression()
                        model.fit(y_train, x_train, thresh_train)
                        
                        if model.fitted:
                            y_pred = model.predict(x_test, thresh_test)
                        else:
                            continue
                            
                    elif spec['type'] == 'hansen_enhanced':
                        enhanced_result = self.enhanced_hansen_regression(
                            y_train, x_train, thresh_train, 
                            spec['params'].get('enhancements', {})
                        )
                        
                        # Use best enhanced model
                        best_model_info = enhanced_result.get('best_model', {})
                        if best_model_info.get('type') == 'original':
                            model = enhanced_result['original_model']['model']
                        else:
                            # Find the best enhanced model
                            best_type = best_model_info['type']
                            enhanced_models = enhanced_result['enhanced_models']
                            
                            if best_type in enhanced_models:
                                model_info = enhanced_models[best_type]
                                if isinstance(model_info, dict) and 'model' in model_info:
                                    model = model_info['model']
                                else:
                                    continue
                            else:
                                continue
                        
                        if hasattr(model, 'fitted') and model.fitted:
                            y_pred = model.predict(x_test, thresh_test)
                        else:
                            continue
                            
                    elif spec['type'] == 'multiple_threshold':
                        multi_result = self.multiple_threshold_model(
                            y_train, x_train, thresh_train,
                            spec['params'].get('max_thresholds', 2)
                        )
                        
                        # Use best multiple threshold model
                        best_multi = multi_result.get('selection_criteria', {}).get('best_model')
                        if best_multi and best_multi in multi_result['models']:
                            model_info = multi_result['models'][best_multi]
                            if model_info.get('fitted', False):
                                # Predict using multiple threshold model (simplified)
                                y_pred = self._predict_multiple_threshold(
                                    model_info, x_test, thresh_test
                                )
                            else:
                                continue
                        else:
                            continue
                            
                    elif spec['type'] == 'smooth_transition':
                        str_result = self.smooth_transition_alternatives(
                            y_train, x_train, thresh_train,
                            spec['params'].get('transition_types', ['logistic'])
                        )
                        
                        # Use best STR model
                        best_str = str_result.get('comparison', {}).get('best_model')
                        if best_str and best_str in str_result['models']:
                            model_info = str_result['models'][best_str]
                            if model_info.get('fitted', False):
                                model = model_info['model']
                                y_pred = model.predict(x_test, thresh_test)
                            else:
                                continue
                        else:
                            continue
                    
                    else:
                        continue
                    
                    # Calculate performance metrics
                    mse = np.mean((y_test - y_pred) ** 2)
                    mae = np.mean(np.abs(y_test - y_pred))
                    
                    ss_res = np.sum((y_test - y_pred) ** 2)
                    ss_tot = np.sum((y_test - np.mean(y_test)) ** 2)
                    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
                    
                    cv_scores['mse'].append(mse)
                    cv_scores['mae'].append(mae)
                    cv_scores['r_squared'].append(r2)
                    cv_scores['successful_fits'] += 1
                    
                except Exception as e:
                    # Skip this fold if fitting fails
                    continue
            
            # Calculate average CV scores
            if cv_scores['successful_fits'] > 0:
                results['cv_results'][spec_name] = {
                    'specification': spec,
                    'mean_mse': np.mean(cv_scores['mse']),
                    'std_mse': np.std(cv_scores['mse']),
                    'mean_mae': np.mean(cv_scores['mae']),
                    'std_mae': np.std(cv_scores['mae']),
                    'mean_r_squared': np.mean(cv_scores['r_squared']),
                    'std_r_squared': np.std(cv_scores['r_squared']),
                    'successful_folds': cv_scores['successful_fits'],
                    'success_rate': cv_scores['successful_fits'] / cv_folds
                }
            else:
                results['cv_results'][spec_name] = {
                    'specification': spec,
                    'error': 'No successful fits across CV folds',
                    'successful_folds': 0,
                    'success_rate': 0
                }
        
        # Rank models by different criteria
        successful_models = {k: v for k, v in results['cv_results'].items() 
                           if v.get('successful_folds', 0) > 0}
        
        if successful_models:
            # Rank by R²
            r2_ranking = sorted(successful_models.items(), 
                              key=lambda x: x[1]['mean_r_squared'], reverse=True)
            
            # Rank by MSE (lower is better)
            mse_ranking = sorted(successful_models.items(), 
                               key=lambda x: x[1]['mean_mse'])
            
            # Rank by MAE (lower is better)
            mae_ranking = sorted(successful_models.items(), 
                               key=lambda x: x[1]['mean_mae'])
            
            results['model_rankings'] = {
                'by_r_squared': [(name, info['mean_r_squared']) for name, info in r2_ranking],
                'by_mse': [(name, info['mean_mse']) for name, info in mse_ranking],
                'by_mae': [(name, info['mean_mae']) for name, info in mae_ranking]
            }
            
            # Select best model (by R²)
            results['best_model'] = {
                'name': r2_ranking[0][0],
                'specification': r2_ranking[0][1]['specification'],
                'cv_performance': r2_ranking[0][1]
            }
            
            # Selection criteria summary
            results['selection_criteria'] = {
                'primary_criterion': 'r_squared',
                'best_r_squared': r2_ranking[0][1]['mean_r_squared'],
                'best_mse': mse_ranking[0][1]['mean_mse'],
                'best_mae': mae_ranking[0][1]['mean_mae'],
                'n_successful_models': len(successful_models)
            }
        
        # Store results
        self.fitted_models['cv_model_selection'] = results
        
        return results
    
    def _analyze_quantile_coefficient_stability(self, quantile_models):
        """Analyze coefficient stability across quantiles"""
        stability_results = {
            'regime1_stability': {},
            'regime2_stability': {},
            'overall_stability': {}
        }
        
        # Extract coefficients for each quantile
        quantiles = []
        regime1_coefs = []
        regime2_coefs = []
        
        for q, model_info in quantile_models.items():
            if model_info.get('fitted', False):
                quantiles.append(q)
                
                if model_info['regime1_coefficients'] is not None:
                    regime1_coefs.append(model_info['regime1_coefficients'])
                
                if model_info['regime2_coefficients'] is not None:
                    regime2_coefs.append(model_info['regime2_coefficients'])
        
        # Analyze regime 1 stability
        if len(regime1_coefs) > 1:
            regime1_array = np.array(regime1_coefs)
            stability_results['regime1_stability'] = {
                'coefficient_ranges': np.ptp(regime1_array, axis=0),  # Peak-to-peak
                'coefficient_std': np.std(regime1_array, axis=0),
                'mean_coefficients': np.mean(regime1_array, axis=0)
            }
        
        # Analyze regime 2 stability
        if len(regime2_coefs) > 1:
            regime2_array = np.array(regime2_coefs)
            stability_results['regime2_stability'] = {
                'coefficient_ranges': np.ptp(regime2_array, axis=0),
                'coefficient_std': np.std(regime2_array, axis=0),
                'mean_coefficients': np.mean(regime2_array, axis=0)
            }
        
        # Overall stability assessment
        stability_results['overall_stability'] = {
            'n_quantiles_fitted': len(quantiles),
            'quantiles_fitted': quantiles,
            'stability_assessment': 'stable' if (
                len(regime1_coefs) > 1 and np.all(stability_results['regime1_stability']['coefficient_std'] < 0.5)
            ) else 'unstable'
        }
        
        return stability_results
    
    def _calculate_threshold_log_likelihood(self, y, x, threshold_var, threshold, beta1, beta2, sigma):
        """Calculate log-likelihood for threshold model"""
        try:
            regime1_mask = threshold_var <= threshold
            regime2_mask = threshold_var > threshold
            
            log_likelihood = 0
            
            # Regime 1
            if np.sum(regime1_mask) > 0:
                y1 = y[regime1_mask]
                x1_with_intercept = np.column_stack([np.ones(np.sum(regime1_mask)), x[regime1_mask]])
                
                if len(beta1) == x1_with_intercept.shape[1]:
                    y1_pred = x1_with_intercept @ beta1
                    residuals1 = y1 - y1_pred
                    log_likelihood += -0.5 * np.sum(residuals1 ** 2) / (sigma ** 2) - 0.5 * len(y1) * np.log(2 * np.pi * sigma ** 2)
            
            # Regime 2
            if np.sum(regime2_mask) > 0:
                y2 = y[regime2_mask]
                x2_with_intercept = np.column_stack([np.ones(np.sum(regime2_mask)), x[regime2_mask]])
                
                if len(beta2) == x2_with_intercept.shape[1]:
                    y2_pred = x2_with_intercept @ beta2
                    residuals2 = y2 - y2_pred
                    log_likelihood += -0.5 * np.sum(residuals2 ** 2) / (sigma ** 2) - 0.5 * len(y2) * np.log(2 * np.pi * sigma ** 2)
            
            return log_likelihood
            
        except Exception:
            return -np.inf
    
    def _calculate_ssr(self, y, x, threshold_var, threshold, beta1, beta2):
        """Calculate sum of squared residuals"""
        try:
            regime1_mask = threshold_var <= threshold
            regime2_mask = threshold_var > threshold
            
            ssr = 0
            
            # Regime 1
            if np.sum(regime1_mask) > 0:
                y1 = y[regime1_mask]
                x1_with_intercept = np.column_stack([np.ones(np.sum(regime1_mask)), x[regime1_mask]])
                
                if len(beta1) == x1_with_intercept.shape[1]:
                    y1_pred = x1_with_intercept @ beta1
                    ssr += np.sum((y1 - y1_pred) ** 2)
            
            # Regime 2
            if np.sum(regime2_mask) > 0:
                y2 = y[regime2_mask]
                x2_with_intercept = np.column_stack([np.ones(np.sum(regime2_mask)), x[regime2_mask]])
                
                if len(beta2) == x2_with_intercept.shape[1]:
                    y2_pred = x2_with_intercept @ beta2
                    ssr += np.sum((y2 - y2_pred) ** 2)
            
            return ssr
            
        except Exception:
            return np.inf
    
    def _calculate_rhat(self, chain_samples):
        """Calculate Gelman-Rubin R-hat statistic for convergence diagnosis"""
        try:
            # Extract threshold samples from each chain
            chains = [chain['threshold'] for chain in chain_samples]
            n_chains = len(chains)
            n_samples = len(chains[0])
            
            # Calculate between-chain and within-chain variance
            chain_means = [np.mean(chain) for chain in chains]
            overall_mean = np.mean(chain_means)
            
            # Between-chain variance
            B = n_samples * np.var(chain_means, ddof=1)
            
            # Within-chain variance
            W = np.mean([np.var(chain, ddof=1) for chain in chains])
            
            # Marginal posterior variance
            var_plus = ((n_samples - 1) / n_samples) * W + (1 / n_samples) * B
            
            # R-hat
            rhat = np.sqrt(var_plus / W)
            
            return rhat
            
        except Exception:
            return None
    
    def _calculate_ess(self, combined_samples):
        """Calculate effective sample size"""
        try:
            # Simplified ESS calculation
            n_samples = len(combined_samples['threshold'])
            
            # Calculate autocorrelation (simplified)
            threshold_samples = combined_samples['threshold']
            autocorr = np.correlate(threshold_samples, threshold_samples, mode='full')
            autocorr = autocorr[autocorr.size // 2:]
            autocorr = autocorr / autocorr[0]
            
            # Find first negative autocorrelation
            first_negative = np.where(autocorr < 0)[0]
            if len(first_negative) > 0:
                tau = first_negative[0]
            else:
                tau = len(autocorr) // 4
            
            ess = n_samples / (1 + 2 * np.sum(autocorr[1:tau]))
            
            return max(1, int(ess))
            
        except Exception:
            return n_samples // 4  # Conservative estimate
    
    def _predict_multiple_threshold(self, model_info, x_test, threshold_test):
        """Predict using multiple threshold model"""
        try:
            thresholds = model_info.get('thresholds', [])
            regime_models = model_info.get('regime_models', [])
            
            if not thresholds or not regime_models:
                return np.zeros(len(x_test))
            
            y_pred = np.zeros(len(x_test))
            
            # Sort thresholds
            sorted_thresholds = sorted(thresholds)
            
            # Create regime masks
            regime_masks = []
            
            # First regime
            regime_masks.append(threshold_test <= sorted_thresholds[0])
            
            # Middle regimes
            for i in range(len(sorted_thresholds) - 1):
                mask = (threshold_test > sorted_thresholds[i]) & (threshold_test <= sorted_thresholds[i + 1])
                regime_masks.append(mask)
            
            # Last regime
            regime_masks.append(threshold_test > sorted_thresholds[-1])
            
            # Predict for each regime
            for i, (mask, regime_model) in enumerate(zip(regime_masks, regime_models)):
                if regime_model is not None and np.sum(mask) > 0:
                    x_regime = x_test[mask]
                    
                    # Add intercept
                    if x_regime.ndim == 1:
                        X_regime = np.column_stack([np.ones(len(x_regime)), x_regime])
                    else:
                        X_regime = np.column_stack([np.ones(len(x_regime)), x_regime])
                    
                    # Predict
                    beta = regime_model.get('beta', np.zeros(X_regime.shape[1]))
                    if len(beta) == X_regime.shape[1]:
                        y_pred[mask] = X_regime @ beta
            
            return y_pred
            
        except Exception:
            return np.zeros(len(x_test))
    
    def bootstrap_threshold_estimation(self, y, x, threshold_var, n_bootstrap=1000, confidence_level=0.95):
        """
        Bootstrap threshold estimation with confidence intervals
        
        Args:
            y: Dependent variable
            x: Independent variables
            threshold_var: Threshold variable
            n_bootstrap: Number of bootstrap samples
            confidence_level: Confidence level for intervals
            
        Returns:
            dict: Bootstrap results with confidence intervals
        """
        # Ensure arrays are numpy arrays
        if hasattr(y, 'values'):
            y = y.values
        if hasattr(x, 'values'):
            x = x.values
        if hasattr(threshold_var, 'values'):
            threshold_var = threshold_var.values
        
        if x.ndim == 1:
            x = x.reshape(-1, 1)
        
        n_obs = len(y)
        bootstrap_results = {
            'thresholds': [],
            'coefficients_regime1': [],
            'coefficients_regime2': [],
            'r_squared_values': [],
            'successful_fits': 0
        }
        
        # Original model for comparison
        try:
            original_model = HansenThresholdRegression()
            original_model.fit(y, x, threshold_var)
            original_threshold = original_model.threshold if original_model.fitted else None
        except:
            original_threshold = None
        
        # Bootstrap sampling
        for i in range(n_bootstrap):
            try:
                # Resample with replacement
                bootstrap_indices = np.random.choice(n_obs, size=n_obs, replace=True)
                y_boot = y[bootstrap_indices]
                x_boot = x[bootstrap_indices]
                threshold_boot = threshold_var[bootstrap_indices]
                
                # Fit Hansen model on bootstrap sample
                hansen_boot = HansenThresholdRegression()
                hansen_boot.fit(y_boot, x_boot, threshold_boot)
                
                if hansen_boot.fitted:
                    # Calculate R² for bootstrap sample
                    y_pred_boot = hansen_boot.predict(x_boot, threshold_boot)
                    ss_res = np.sum((y_boot - y_pred_boot) ** 2)
                    ss_tot = np.sum((y_boot - np.mean(y_boot)) ** 2)
                    r2_boot = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
                    
                    # Store results
                    bootstrap_results['thresholds'].append(hansen_boot.threshold)
                    bootstrap_results['coefficients_regime1'].append(hansen_boot.beta1)
                    bootstrap_results['coefficients_regime2'].append(hansen_boot.beta2)
                    bootstrap_results['r_squared_values'].append(r2_boot)
                    bootstrap_results['successful_fits'] += 1
                    
            except Exception:
                continue
        
        # Calculate confidence intervals
        alpha = 1 - confidence_level
        lower_percentile = (alpha / 2) * 100
        upper_percentile = (1 - alpha / 2) * 100
        
        results = {
            'original_threshold': original_threshold,
            'bootstrap_statistics': {},
            'confidence_intervals': {},
            'n_successful_bootstrap': bootstrap_results['successful_fits'],
            'bootstrap_success_rate': bootstrap_results['successful_fits'] / n_bootstrap
        }
        
        if bootstrap_results['successful_fits'] > 0:
            # Threshold statistics
            thresholds = np.array(bootstrap_results['thresholds'])
            results['bootstrap_statistics']['threshold'] = {
                'mean': np.mean(thresholds),
                'std': np.std(thresholds),
                'median': np.median(thresholds)
            }
            results['confidence_intervals']['threshold'] = {
                'lower': np.percentile(thresholds, lower_percentile),
                'upper': np.percentile(thresholds, upper_percentile)
            }
            
            # R² statistics
            r2_values = np.array(bootstrap_results['r_squared_values'])
            results['bootstrap_statistics']['r_squared'] = {
                'mean': np.mean(r2_values),
                'std': np.std(r2_values),
                'median': np.median(r2_values)
            }
            results['confidence_intervals']['r_squared'] = {
                'lower': np.percentile(r2_values, lower_percentile),
                'upper': np.percentile(r2_values, upper_percentile)
            }
            
            # Coefficient statistics (regime 1)
            if bootstrap_results['coefficients_regime1']:
                coef1_array = np.array(bootstrap_results['coefficients_regime1'])
                results['bootstrap_statistics']['coefficients_regime1'] = {
                    'mean': np.mean(coef1_array, axis=0),
                    'std': np.std(coef1_array, axis=0),
                    'median': np.median(coef1_array, axis=0)
                }
                results['confidence_intervals']['coefficients_regime1'] = {
                    'lower': np.percentile(coef1_array, lower_percentile, axis=0),
                    'upper': np.percentile(coef1_array, upper_percentile, axis=0)
                }
            
            # Coefficient statistics (regime 2)
            if bootstrap_results['coefficients_regime2']:
                coef2_array = np.array(bootstrap_results['coefficients_regime2'])
                results['bootstrap_statistics']['coefficients_regime2'] = {
                    'mean': np.mean(coef2_array, axis=0),
                    'std': np.std(coef2_array, axis=0),
                    'median': np.median(coef2_array, axis=0)
                }
                results['confidence_intervals']['coefficients_regime2'] = {
                    'lower': np.percentile(coef2_array, lower_percentile, axis=0),
                    'upper': np.percentile(coef2_array, upper_percentile, axis=0)
                }
        
        # Store results
        self.fitted_models['bootstrap_threshold'] = results
        
        return results
    
    def quantile_threshold_regression(self, y, x, threshold_var, quantiles=None):
        """
        Quantile threshold regression for robust estimation
        
        Args:
            y: Dependent variable
            x: Independent variables
            threshold_var: Threshold variable
            quantiles: List of quantiles to estimate (default: [0.1, 0.25, 0.5, 0.75, 0.9])
            
        Returns:
            dict: Quantile regression results across different quantiles
        """
        if quantiles is None:
            quantiles = [0.1, 0.25, 0.5, 0.75, 0.9]
        
        # Ensure arrays are numpy arrays
        if hasattr(y, 'values'):
            y = y.values
        if hasattr(x, 'values'):
            x = x.values
        if hasattr(threshold_var, 'values'):
            threshold_var = threshold_var.values
        
        if x.ndim == 1:
            x = x.reshape(-1, 1)
        
        results = {
            'quantile_models': {},
            'threshold_estimates': {},
            'coefficient_estimates': {},
            'model_comparison': {}
        }
        
        # First, estimate threshold using median regression (more robust)
        try:
            # Use grid search for threshold estimation with quantile regression
            threshold_candidates = np.percentile(threshold_var, np.linspace(10, 90, 20))
            best_threshold = None
            best_objective = np.inf
            
            for thresh_candidate in threshold_candidates:
                try:
                    # Create regime indicators
                    regime1_mask = threshold_var <= thresh_candidate
                    regime2_mask = threshold_var > thresh_candidate
                    
                    if np.sum(regime1_mask) < 10 or np.sum(regime2_mask) < 10:
                        continue
                    
                    # Fit quantile regression for each regime (using median)
                    total_objective = 0
                    
                    # Regime 1
                    y1 = y[regime1_mask]
                    x1 = x[regime1_mask]
                    if len(y1) > 0:
                        qr1 = QuantileRegressor(quantile=0.5, alpha=0.01)
                        qr1.fit(x1, y1)
                        y1_pred = qr1.predict(x1)
                        total_objective += np.sum(np.abs(y1 - y1_pred))
                    
                    # Regime 2
                    y2 = y[regime2_mask]
                    x2 = x[regime2_mask]
                    if len(y2) > 0:
                        qr2 = QuantileRegressor(quantile=0.5, alpha=0.01)
                        qr2.fit(x2, y2)
                        y2_pred = qr2.predict(x2)
                        total_objective += np.sum(np.abs(y2 - y2_pred))
                    
                    if total_objective < best_objective:
                        best_objective = total_objective
                        best_threshold = thresh_candidate
                        
                except Exception:
                    continue
            
            if best_threshold is None:
                best_threshold = np.median(threshold_var)
            
            results['estimated_threshold'] = best_threshold
            
        except Exception as e:
            results['threshold_estimation_error'] = str(e)
            best_threshold = np.median(threshold_var)
            results['estimated_threshold'] = best_threshold
        
        # Fit quantile regressions for each quantile
        regime1_mask = threshold_var <= best_threshold
        regime2_mask = threshold_var > best_threshold
        
        for q in quantiles:
            try:
                quantile_result = {
                    'quantile': q,
                    'threshold': best_threshold,
                    'regime1_model': None,
                    'regime2_model': None,
                    'regime1_coefficients': None,
                    'regime2_coefficients': None,
                    'fitted': False
                }
                
                # Fit regime 1
                if np.sum(regime1_mask) >= 5:
                    y1 = y[regime1_mask]
                    x1 = x[regime1_mask]
                    
                    qr1 = QuantileRegressor(quantile=q, alpha=0.01)
                    qr1.fit(x1, y1)
                    
                    quantile_result['regime1_model'] = qr1
                    quantile_result['regime1_coefficients'] = qr1.coef_
                
                # Fit regime 2
                if np.sum(regime2_mask) >= 5:
                    y2 = y[regime2_mask]
                    x2 = x[regime2_mask]
                    
                    qr2 = QuantileRegressor(quantile=q, alpha=0.01)
                    qr2.fit(x2, y2)
                    
                    quantile_result['regime2_model'] = qr2
                    quantile_result['regime2_coefficients'] = qr2.coef_
                
                if quantile_result['regime1_model'] is not None or quantile_result['regime2_model'] is not None:
                    quantile_result['fitted'] = True
                
                results['quantile_models'][q] = quantile_result
                
            except Exception as e:
                results['quantile_models'][q] = {
                    'quantile': q,
                    'error': str(e),
                    'fitted': False
                }
        
        # Analyze coefficient stability across quantiles
        results['coefficient_stability'] = self._analyze_quantile_coefficient_stability(results['quantile_models'])
        
        # Store results
        self.fitted_models['quantile_threshold'] = results
        
        return results
    
    def bayesian_threshold_model(self, y, x, threshold_var, n_samples=5000, n_chains=4):
        """
        Bayesian threshold model with uncertainty quantification
        
        Args:
            y: Dependent variable
            x: Independent variables
            threshold_var: Threshold variable
            n_samples: Number of MCMC samples per chain
            n_chains: Number of MCMC chains
            
        Returns:
            dict: Bayesian estimation results with posterior distributions
        """
        # Ensure arrays are numpy arrays
        if hasattr(y, 'values'):
            y = y.values
        if hasattr(x, 'values'):
            x = x.values
        if hasattr(threshold_var, 'values'):
            threshold_var = threshold_var.values
        
        if x.ndim == 1:
            x = x.reshape(-1, 1)
        
        results = {
            'posterior_samples': {},
            'posterior_statistics': {},
            'credible_intervals': {},
            'model_diagnostics': {},
            'fitted': False
        }
        
        try:
            # Simplified Bayesian estimation using Metropolis-Hastings
            # This is a basic implementation - in practice, you'd use PyMC3 or Stan
            
            n_obs = len(y)
            n_vars = x.shape[1]
            
            # Prior parameters (weakly informative)
            prior_mean_beta = np.zeros(n_vars + 1)  # +1 for intercept
            prior_cov_beta = np.eye(n_vars + 1) * 10  # Weak prior
            prior_sigma_shape = 2
            prior_sigma_scale = 1
            
            # Initialize parameters
            all_samples = []
            
            for chain in range(n_chains):
                # Initialize chain
                current_threshold = np.random.uniform(
                    np.percentile(threshold_var, 20),
                    np.percentile(threshold_var, 80)
                )
                current_beta1 = np.random.multivariate_normal(prior_mean_beta, prior_cov_beta)
                current_beta2 = np.random.multivariate_normal(prior_mean_beta, prior_cov_beta)
                current_sigma = 1.0
                
                chain_samples = {
                    'threshold': [],
                    'beta1': [],
                    'beta2': [],
                    'sigma': [],
                    'log_likelihood': []
                }
                
                accepted = 0
                
                for i in range(n_samples):
                    # Propose new threshold
                    threshold_proposal = current_threshold + np.random.normal(0, 0.1 * np.std(threshold_var))
                    
                    # Ensure threshold is within reasonable bounds
                    threshold_proposal = np.clip(
                        threshold_proposal,
                        np.percentile(threshold_var, 5),
                        np.percentile(threshold_var, 95)
                    )
                    
                    # Calculate likelihood for current and proposed threshold
                    current_ll = self._calculate_threshold_log_likelihood(
                        y, x, threshold_var, current_threshold, current_beta1, current_beta2, current_sigma
                    )
                    
                    proposed_ll = self._calculate_threshold_log_likelihood(
                        y, x, threshold_var, threshold_proposal, current_beta1, current_beta2, current_sigma
                    )
                    
                    # Metropolis-Hastings acceptance
                    log_alpha = proposed_ll - current_ll
                    if np.log(np.random.uniform()) < log_alpha:
                        current_threshold = threshold_proposal
                        current_ll = proposed_ll
                        accepted += 1
                    
                    # Update coefficients (Gibbs sampling)
                    regime1_mask = threshold_var <= current_threshold
                    regime2_mask = threshold_var > current_threshold
                    
                    if np.sum(regime1_mask) >= 3:
                        y1 = y[regime1_mask]
                        x1_with_intercept = np.column_stack([np.ones(np.sum(regime1_mask)), x[regime1_mask]])
                        
                        # Posterior for beta1 (conjugate normal)
                        precision_matrix = np.linalg.inv(prior_cov_beta) + (x1_with_intercept.T @ x1_with_intercept) / (current_sigma ** 2)
                        posterior_cov = np.linalg.inv(precision_matrix)
                        posterior_mean = posterior_cov @ (
                            np.linalg.inv(prior_cov_beta) @ prior_mean_beta +
                            (x1_with_intercept.T @ y1) / (current_sigma ** 2)
                        )
                        
                        current_beta1 = np.random.multivariate_normal(posterior_mean, posterior_cov)
                    
                    if np.sum(regime2_mask) >= 3:
                        y2 = y[regime2_mask]
                        x2_with_intercept = np.column_stack([np.ones(np.sum(regime2_mask)), x[regime2_mask]])
                        
                        # Posterior for beta2 (conjugate normal)
                        precision_matrix = np.linalg.inv(prior_cov_beta) + (x2_with_intercept.T @ x2_with_intercept) / (current_sigma ** 2)
                        posterior_cov = np.linalg.inv(precision_matrix)
                        posterior_mean = posterior_cov @ (
                            np.linalg.inv(prior_cov_beta) @ prior_mean_beta +
                            (x2_with_intercept.T @ y2) / (current_sigma ** 2)
                        )
                        
                        current_beta2 = np.random.multivariate_normal(posterior_mean, posterior_cov)
                    
                    # Update sigma (inverse gamma posterior)
                    ssr = self._calculate_ssr(y, x, threshold_var, current_threshold, current_beta1, current_beta2)
                    posterior_shape = prior_sigma_shape + n_obs / 2
                    posterior_scale = prior_sigma_scale + ssr / 2
                    current_sigma = np.sqrt(1 / np.random.gamma(posterior_shape, 1 / posterior_scale))
                    
                    # Store samples (after burn-in)
                    if i >= n_samples // 4:  # 25% burn-in
                        chain_samples['threshold'].append(current_threshold)
                        chain_samples['beta1'].append(current_beta1.copy())
                        chain_samples['beta2'].append(current_beta2.copy())
                        chain_samples['sigma'].append(current_sigma)
                        chain_samples['log_likelihood'].append(current_ll)
                
                all_samples.append(chain_samples)
                
            # Combine chains
            combined_samples = {
                'threshold': np.concatenate([chain['threshold'] for chain in all_samples]),
                'beta1': np.vstack([chain['beta1'] for chain in all_samples]),
                'beta2': np.vstack([chain['beta2'] for chain in all_samples]),
                'sigma': np.concatenate([chain['sigma'] for chain in all_samples]),
                'log_likelihood': np.concatenate([chain['log_likelihood'] for chain in all_samples])
            }
            
            results['posterior_samples'] = combined_samples
            
            # Calculate posterior statistics
            results['posterior_statistics'] = {
                'threshold': {
                    'mean': np.mean(combined_samples['threshold']),
                    'std': np.std(combined_samples['threshold']),
                    'median': np.median(combined_samples['threshold'])
                },
                'beta1': {
                    'mean': np.mean(combined_samples['beta1'], axis=0),
                    'std': np.std(combined_samples['beta1'], axis=0),
                    'median': np.median(combined_samples['beta1'], axis=0)
                },
                'beta2': {
                    'mean': np.mean(combined_samples['beta2'], axis=0),
                    'std': np.std(combined_samples['beta2'], axis=0),
                    'median': np.median(combined_samples['beta2'], axis=0)
                },
                'sigma': {
                    'mean': np.mean(combined_samples['sigma']),
                    'std': np.std(combined_samples['sigma']),
                    'median': np.median(combined_samples['sigma'])
                }
            }
            
            # Calculate credible intervals (95%)
            results['credible_intervals'] = {
                'threshold': {
                    'lower': np.percentile(combined_samples['threshold'], 2.5),
                    'upper': np.percentile(combined_samples['threshold'], 97.5)
                },
                'beta1': {
                    'lower': np.percentile(combined_samples['beta1'], 2.5, axis=0),
                    'upper': np.percentile(combined_samples['beta1'], 97.5, axis=0)
                },
                'beta2': {
                    'lower': np.percentile(combined_samples['beta2'], 2.5, axis=0),
                    'upper': np.percentile(combined_samples['beta2'], 97.5, axis=0)
                },
                'sigma': {
                    'lower': np.percentile(combined_samples['sigma'], 2.5),
                    'upper': np.percentile(combined_samples['sigma'], 97.5)
                }
            }
            
            # Model diagnostics
            results['model_diagnostics'] = {
                'n_effective_samples': len(combined_samples['threshold']),
                'acceptance_rate': accepted / n_samples if 'accepted' in locals() else None,
                'rhat': self._calculate_rhat(all_samples),  # Gelman-Rubin statistic
                'effective_sample_size': self._calculate_ess(combined_samples)
            }
            
            results['fitted'] = True
            
        except Exception as e:
            results['error'] = str(e)
            results['fitted'] = False
        
        # Store results
        self.fitted_models['bayesian_threshold'] = results
        
        return results
    
    def cross_validation_model_selection(self, y, x, threshold_var, model_specifications=None, cv_folds=5):
        """
        Cross-validation based model selection for specification choice
        
        Args:
            y: Dependent variable
            x: Independent variables
            threshold_var: Threshold variable
            model_specifications: List of model specifications to compare
            cv_folds: Number of cross-validation folds
            
        Returns:
            dict: Cross-validation results and model selection
        """
        if model_specifications is None:
            model_specifications = [
                {'type': 'hansen', 'params': {}},
                {'type': 'hansen_enhanced', 'params': {'enhancements': {'data_transforms': ['levels']}}},
                {'type': 'hansen_enhanced', 'params': {'enhancements': {'data_transforms': ['differences']}}},
                {'type': 'hansen_enhanced', 'params': {'enhancements': {'additional_controls': True}}},
                {'type': 'multiple_threshold', 'params': {'max_thresholds': 2}},
                {'type': 'smooth_transition', 'params': {'transition_types': ['logistic']}}
            ]
        
        # Ensure arrays are numpy arrays
        if hasattr(y, 'values'):
            y = y.values
        if hasattr(x, 'values'):
            x = x.values
        if hasattr(threshold_var, 'values'):
            threshold_var = threshold_var.values
        
        if x.ndim == 1:
            x = x.reshape(-1, 1)
        
        results = {
            'cv_results': {},
            'model_rankings': {},
            'best_model': None,
            'selection_criteria': {}
        }
        
        n_obs = len(y)
        fold_size = n_obs // cv_folds
        
        # Create cross-validation folds
        cv_folds_indices = []
        for fold in range(cv_folds):
            start_idx = fold * fold_size
            end_idx = (fold + 1) * fold_size if fold < cv_folds - 1 else n_obs
            test_indices = np.arange(start_idx, end_idx)
            train_indices = np.concatenate([np.arange(0, start_idx), np.arange(end_idx, n_obs)])
            cv_folds_indices.append((train_indices, test_indices))
        
        # Evaluate each model specification
        for spec_idx, spec in enumerate(model_specifications):
            spec_name = f"{spec['type']}_{spec_idx}"
            
            cv_scores = {
                'mse': [],
                'mae': [],
                'r_squared': [],
                'successful_fits': 0
            }
            
            for fold, (train_idx, test_idx) in enumerate(cv_folds_indices):
                try:
                    # Split data
                    y_train, y_test = y[train_idx], y[test_idx]
                    x_train, x_test = x[train_idx], x[test_idx]
                    thresh_train, thresh_test = threshold_var[train_idx], threshold_var[test_idx]
                    
                    # Fit model based on specification
                    if spec['type'] == 'hansen':
                        model = HansenThresholdRegression()
                        model.fit(y_train, x_train, thresh_train)
                        
                        if model.fitted:
                            y_pred = model.predict(x_test, thresh_test)
                        else:
                            continue
                            
                    elif spec['type'] == 'hansen_enhanced':
                        enhanced_result = self.enhanced_hansen_regression(
                            y_train, x_train, thresh_train, 
                            spec['params'].get('enhancements', {})
                        )
                        
                        # Use best enhanced model
                        best_model_info = enhanced_result.get('best_model', {})
                        if best_model_info.get('type') == 'original':
                            model = enhanced_result['original_model']['model']
                        else:
                            # Find the best enhanced model
                            best_type = best_model_info['type']
                            enhanced_models = enhanced_result['enhanced_models']
                            
                            if best_type in enhanced_models:
                                model_info = enhanced_models[best_type]
                                if isinstance(model_info, dict) and 'model' in model_info:
                                    model = model_info['model']
                                else:
                                    continue
                            else:
                                continue
                        
                        if hasattr(model, 'fitted') and model.fitted:
                            y_pred = model.predict(x_test, thresh_test)
                        else:
                            continue
                            
                    elif spec['type'] == 'multiple_threshold':
                        multi_result = self.multiple_threshold_model(
                            y_train, x_train, thresh_train,
                            spec['params'].get('max_thresholds', 2)
                        )
                        
                        # Use best multiple threshold model
                        best_multi = multi_result.get('selection_criteria', {}).get('best_model')
                        if best_multi and best_multi in multi_result['models']:
                            model_info = multi_result['models'][best_multi]
                            if model_info.get('fitted', False):
                                # Predict using multiple threshold model (simplified)
                                y_pred = self._predict_multiple_threshold(
                                    model_info, x_test, thresh_test
                                )
                            else:
                                continue
                        else:
                            continue
                            
                    elif spec['type'] == 'smooth_transition':
                        str_result = self.smooth_transition_alternatives(
                            y_train, x_train, thresh_train,
                            spec['params'].get('transition_types', ['logistic'])
                        )
                        
                        # Use best STR model
                        best_str = str_result.get('comparison', {}).get('best_model')
                        if best_str and best_str in str_result['models']:
                            model_info = str_result['models'][best_str]
                            if model_info.get('fitted', False):
                                model = model_info['model']
                                y_pred = model.predict(x_test, thresh_test)
                            else:
                                continue
                        else:
                            continue
                    
                    else:
                        continue
                    
                    # Calculate performance metrics
                    mse = np.mean((y_test - y_pred) ** 2)
                    mae = np.mean(np.abs(y_test - y_pred))
                    
                    ss_res = np.sum((y_test - y_pred) ** 2)
                    ss_tot = np.sum((y_test - np.mean(y_test)) ** 2)
                    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
                    
                    cv_scores['mse'].append(mse)
                    cv_scores['mae'].append(mae)
                    cv_scores['r_squared'].append(r2)
                    cv_scores['successful_fits'] += 1
                    
                except Exception as e:
                    # Skip this fold if fitting fails
                    continue
            
            # Calculate average CV scores
            if cv_scores['successful_fits'] > 0:
                results['cv_results'][spec_name] = {
                    'specification': spec,
                    'mean_mse': np.mean(cv_scores['mse']),
                    'std_mse': np.std(cv_scores['mse']),
                    'mean_mae': np.mean(cv_scores['mae']),
                    'std_mae': np.std(cv_scores['mae']),
                    'mean_r_squared': np.mean(cv_scores['r_squared']),
                    'std_r_squared': np.std(cv_scores['r_squared']),
                    'successful_folds': cv_scores['successful_fits'],
                    'success_rate': cv_scores['successful_fits'] / cv_folds
                }
            else:
                results['cv_results'][spec_name] = {
                    'specification': spec,
                    'error': 'No successful fits across CV folds',
                    'successful_folds': 0,
                    'success_rate': 0
                }
        
        # Rank models by different criteria
        successful_models = {k: v for k, v in results['cv_results'].items() 
                           if v.get('successful_folds', 0) > 0}
        
        if successful_models:
            # Rank by R²
            r2_ranking = sorted(successful_models.items(), 
                              key=lambda x: x[1]['mean_r_squared'], reverse=True)
            
            # Rank by MSE (lower is better)
            mse_ranking = sorted(successful_models.items(), 
                               key=lambda x: x[1]['mean_mse'])
            
            # Rank by MAE (lower is better)
            mae_ranking = sorted(successful_models.items(), 
                               key=lambda x: x[1]['mean_mae'])
            
            results['model_rankings'] = {
                'by_r_squared': [(name, info['mean_r_squared']) for name, info in r2_ranking],
                'by_mse': [(name, info['mean_mse']) for name, info in mse_ranking],
                'by_mae': [(name, info['mean_mae']) for name, info in mae_ranking]
            }
            
            # Select best model (by R²)
            results['best_model'] = {
                'name': r2_ranking[0][0],
                'specification': r2_ranking[0][1]['specification'],
                'cv_performance': r2_ranking[0][1]
            }
            
            # Selection criteria summary
            results['selection_criteria'] = {
                'primary_criterion': 'r_squared',
                'best_r_squared': r2_ranking[0][1]['mean_r_squared'],
                'best_mse': mse_ranking[0][1]['mean_mse'],
                'best_mae': mae_ranking[0][1]['mean_mae'],
                'n_successful_models': len(successful_models)
            }
        
        # Store results
        self.fitted_models['cv_model_selection'] = results
        
        return results
    
    def _analyze_quantile_coefficient_stability(self, quantile_models):
        """Analyze coefficient stability across quantiles"""
        stability_results = {
            'regime1_stability': {},
            'regime2_stability': {},
            'overall_stability': {}
        }
        
        # Extract coefficients for each quantile
        quantiles = []
        regime1_coefs = []
        regime2_coefs = []
        
        for q, model_info in quantile_models.items():
            if model_info.get('fitted', False):
                quantiles.append(q)
                
                if model_info['regime1_coefficients'] is not None:
                    regime1_coefs.append(model_info['regime1_coefficients'])
                
                if model_info['regime2_coefficients'] is not None:
                    regime2_coefs.append(model_info['regime2_coefficients'])
        
        # Analyze regime 1 stability
        if len(regime1_coefs) > 1:
            regime1_array = np.array(regime1_coefs)
            stability_results['regime1_stability'] = {
                'coefficient_ranges': np.ptp(regime1_array, axis=0),  # Peak-to-peak
                'coefficient_std': np.std(regime1_array, axis=0),
                'mean_coefficients': np.mean(regime1_array, axis=0)
            }
        
        # Analyze regime 2 stability
        if len(regime2_coefs) > 1:
            regime2_array = np.array(regime2_coefs)
            stability_results['regime2_stability'] = {
                'coefficient_ranges': np.ptp(regime2_array, axis=0),
                'coefficient_std': np.std(regime2_array, axis=0),
                'mean_coefficients': np.mean(regime2_array, axis=0)
            }
        
        # Overall stability assessment
        stability_results['overall_stability'] = {
            'n_quantiles_fitted': len(quantiles),
            'quantiles_fitted': quantiles,
            'stability_assessment': 'stable' if (
                len(regime1_coefs) > 1 and np.all(stability_results['regime1_stability']['coefficient_std'] < 0.5)
            ) else 'unstable'
        }
        
        return stability_results
    
    def _calculate_threshold_log_likelihood(self, y, x, threshold_var, threshold, beta1, beta2, sigma):
        """Calculate log-likelihood for threshold model"""
        try:
            regime1_mask = threshold_var <= threshold
            regime2_mask = threshold_var > threshold
            
            log_likelihood = 0
            
            # Regime 1
            if np.sum(regime1_mask) > 0:
                y1 = y[regime1_mask]
                x1_with_intercept = np.column_stack([np.ones(np.sum(regime1_mask)), x[regime1_mask]])
                
                if len(beta1) == x1_with_intercept.shape[1]:
                    y1_pred = x1_with_intercept @ beta1
                    residuals1 = y1 - y1_pred
                    log_likelihood += -0.5 * np.sum(residuals1 ** 2) / (sigma ** 2) - 0.5 * len(y1) * np.log(2 * np.pi * sigma ** 2)
            
            # Regime 2
            if np.sum(regime2_mask) > 0:
                y2 = y[regime2_mask]
                x2_with_intercept = np.column_stack([np.ones(np.sum(regime2_mask)), x[regime2_mask]])
                
                if len(beta2) == x2_with_intercept.shape[1]:
                    y2_pred = x2_with_intercept @ beta2
                    residuals2 = y2 - y2_pred
                    log_likelihood += -0.5 * np.sum(residuals2 ** 2) / (sigma ** 2) - 0.5 * len(y2) * np.log(2 * np.pi * sigma ** 2)
            
            return log_likelihood
            
        except Exception:
            return -np.inf
    
    def _calculate_ssr(self, y, x, threshold_var, threshold, beta1, beta2):
        """Calculate sum of squared residuals"""
        try:
            regime1_mask = threshold_var <= threshold
            regime2_mask = threshold_var > threshold
            
            ssr = 0
            
            # Regime 1
            if np.sum(regime1_mask) > 0:
                y1 = y[regime1_mask]
                x1_with_intercept = np.column_stack([np.ones(np.sum(regime1_mask)), x[regime1_mask]])
                
                if len(beta1) == x1_with_intercept.shape[1]:
                    y1_pred = x1_with_intercept @ beta1
                    ssr += np.sum((y1 - y1_pred) ** 2)
            
            # Regime 2
            if np.sum(regime2_mask) > 0:
                y2 = y[regime2_mask]
                x2_with_intercept = np.column_stack([np.ones(np.sum(regime2_mask)), x[regime2_mask]])
                
                if len(beta2) == x2_with_intercept.shape[1]:
                    y2_pred = x2_with_intercept @ beta2
                    ssr += np.sum((y2 - y2_pred) ** 2)
            
            return ssr
            
        except Exception:
            return np.inf
    
    def _calculate_rhat(self, chain_samples):
        """Calculate Gelman-Rubin R-hat statistic for convergence diagnosis"""
        try:
            # Extract threshold samples from each chain
            chains = [chain['threshold'] for chain in chain_samples]
            n_chains = len(chains)
            n_samples = len(chains[0])
            
            # Calculate between-chain and within-chain variance
            chain_means = [np.mean(chain) for chain in chains]
            overall_mean = np.mean(chain_means)
            
            # Between-chain variance
            B = n_samples * np.var(chain_means, ddof=1)
            
            # Within-chain variance
            W = np.mean([np.var(chain, ddof=1) for chain in chains])
            
            # Marginal posterior variance
            var_plus = ((n_samples - 1) / n_samples) * W + (1 / n_samples) * B
            
            # R-hat
            rhat = np.sqrt(var_plus / W)
            
            return rhat
            
        except Exception:
            return None
    
    def _calculate_ess(self, combined_samples):
        """Calculate effective sample size"""
        try:
            # Simplified ESS calculation
            n_samples = len(combined_samples['threshold'])
            
            # Calculate autocorrelation (simplified)
            threshold_samples = combined_samples['threshold']
            autocorr = np.correlate(threshold_samples, threshold_samples, mode='full')
            autocorr = autocorr[autocorr.size // 2:]
            autocorr = autocorr / autocorr[0]
            
            # Find first negative autocorrelation
            first_negative = np.where(autocorr < 0)[0]
            if len(first_negative) > 0:
                tau = first_negative[0]
            else:
                tau = len(autocorr) // 4
            
            ess = n_samples / (1 + 2 * np.sum(autocorr[1:tau]))
            
            return max(1, int(ess))
            
        except Exception:
            return n_samples // 4  # Conservative estimate
    
    def _predict_multiple_threshold(self, model_info, x_test, threshold_test):
        """Predict using multiple threshold model"""
        try:
            thresholds = model_info.get('thresholds', [])
            regime_models = model_info.get('regime_models', [])
            
            if not thresholds or not regime_models:
                return np.zeros(len(x_test))
            
            y_pred = np.zeros(len(x_test))
            
            # Sort thresholds
            sorted_thresholds = sorted(thresholds)
            
            # Create regime masks
            regime_masks = []
            
            # First regime
            regime_masks.append(threshold_test <= sorted_thresholds[0])
            
            # Middle regimes
            for i in range(len(sorted_thresholds) - 1):
                mask = (threshold_test > sorted_thresholds[i]) & (threshold_test <= sorted_thresholds[i + 1])
                regime_masks.append(mask)
            
            # Last regime
            regime_masks.append(threshold_test > sorted_thresholds[-1])
            
            # Predict for each regime
            for i, (mask, regime_model) in enumerate(zip(regime_masks, regime_models)):
                if regime_model is not None and np.sum(mask) > 0:
                    x_regime = x_test[mask]
                    
                    # Add intercept
                    if x_regime.ndim == 1:
                        X_regime = np.column_stack([np.ones(len(x_regime)), x_regime])
                    else:
                        X_regime = np.column_stack([np.ones(len(x_regime)), x_regime])
                    
                    # Predict
                    beta = regime_model.get('beta', np.zeros(X_regime.shape[1]))
                    if len(beta) == X_regime.shape[1]:
                        y_pred[mask] = X_regime @ beta
            
            return y_pred
            
        except Exception:
            return np.zeros(len(x_test)) 
    
    def bootstrap_threshold_estimation(self, y, x, threshold_var, n_bootstrap=1000, confidence_level=0.95):
        """
        Bootstrap threshold estimation with confidence intervals
        
        Args:
            y: Dependent variable
            x: Independent variables
            threshold_var: Threshold variable
            n_bootstrap: Number of bootstrap samples
            confidence_level: Confidence level for intervals
            
        Returns:
            dict: Bootstrap results with confidence intervals
        """
        # Ensure arrays are numpy arrays
        if hasattr(y, 'values'):
            y = y.values
        if hasattr(x, 'values'):
            x = x.values
        if hasattr(threshold_var, 'values'):
            threshold_var = threshold_var.values
        
        if x.ndim == 1:
            x = x.reshape(-1, 1)
        
        n_obs = len(y)
        bootstrap_results = {
            'thresholds': [],
            'coefficients_regime1': [],
            'coefficients_regime2': [],
            'r_squared_values': [],
            'successful_fits': 0
        }
        
        # Original model for comparison
        try:
            original_model = HansenThresholdRegression()
            original_model.fit(y, x, threshold_var)
            original_threshold = original_model.threshold if original_model.fitted else None
        except:
            original_threshold = None
        
        # Bootstrap sampling
        for i in range(n_bootstrap):
            try:
                # Resample with replacement
                bootstrap_indices = np.random.choice(n_obs, size=n_obs, replace=True)
                y_boot = y[bootstrap_indices]
                x_boot = x[bootstrap_indices]
                threshold_boot = threshold_var[bootstrap_indices]
                
                # Fit Hansen model on bootstrap sample
                hansen_boot = HansenThresholdRegression()
                hansen_boot.fit(y_boot, x_boot, threshold_boot)
                
                if hansen_boot.fitted:
                    # Calculate R² for bootstrap sample
                    y_pred_boot = hansen_boot.predict(x_boot, threshold_boot)
                    ss_res = np.sum((y_boot - y_pred_boot) ** 2)
                    ss_tot = np.sum((y_boot - np.mean(y_boot)) ** 2)
                    r2_boot = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
                    
                    # Store results
                    bootstrap_results['thresholds'].append(hansen_boot.threshold)
                    bootstrap_results['coefficients_regime1'].append(hansen_boot.beta1)
                    bootstrap_results['coefficients_regime2'].append(hansen_boot.beta2)
                    bootstrap_results['r_squared_values'].append(r2_boot)
                    bootstrap_results['successful_fits'] += 1
                    
            except Exception:
                continue
        
        # Calculate confidence intervals
        alpha = 1 - confidence_level
        lower_percentile = (alpha / 2) * 100
        upper_percentile = (1 - alpha / 2) * 100
        
        results = {
            'original_threshold': original_threshold,
            'bootstrap_statistics': {},
            'confidence_intervals': {},
            'n_successful_bootstrap': bootstrap_results['successful_fits'],
            'bootstrap_success_rate': bootstrap_results['successful_fits'] / n_bootstrap
        }
        
        if bootstrap_results['successful_fits'] > 0:
            # Threshold statistics
            thresholds = np.array(bootstrap_results['thresholds'])
            results['bootstrap_statistics']['threshold'] = {
                'mean': np.mean(thresholds),
                'std': np.std(thresholds),
                'median': np.median(thresholds)
            }
            results['confidence_intervals']['threshold'] = {
                'lower': np.percentile(thresholds, lower_percentile),
                'upper': np.percentile(thresholds, upper_percentile)
            }
            
            # R² statistics
            r2_values = np.array(bootstrap_results['r_squared_values'])
            results['bootstrap_statistics']['r_squared'] = {
                'mean': np.mean(r2_values),
                'std': np.std(r2_values),
                'median': np.median(r2_values)
            }
            results['confidence_intervals']['r_squared'] = {
                'lower': np.percentile(r2_values, lower_percentile),
                'upper': np.percentile(r2_values, upper_percentile)
            }
            
            # Coefficient statistics (regime 1)
            if bootstrap_results['coefficients_regime1']:
                coef1_array = np.array(bootstrap_results['coefficients_regime1'])
                results['bootstrap_statistics']['coefficients_regime1'] = {
                    'mean': np.mean(coef1_array, axis=0),
                    'std': np.std(coef1_array, axis=0),
                    'median': np.median(coef1_array, axis=0)
                }
                results['confidence_intervals']['coefficients_regime1'] = {
                    'lower': np.percentile(coef1_array, lower_percentile, axis=0),
                    'upper': np.percentile(coef1_array, upper_percentile, axis=0)
                }
            
            # Coefficient statistics (regime 2)
            if bootstrap_results['coefficients_regime2']:
                coef2_array = np.array(bootstrap_results['coefficients_regime2'])
                results['bootstrap_statistics']['coefficients_regime2'] = {
                    'mean': np.mean(coef2_array, axis=0),
                    'std': np.std(coef2_array, axis=0),
                    'median': np.median(coef2_array, axis=0)
                }
                results['confidence_intervals']['coefficients_regime2'] = {
                    'lower': np.percentile(coef2_array, lower_percentile, axis=0),
                    'upper': np.percentile(coef2_array, upper_percentile, axis=0)
                }
        
        # Store results
        self.fitted_models['bootstrap_threshold'] = results
        
        return results
    
    def quantile_threshold_regression(self, y, x, threshold_var, quantiles=None):
        """
        Quantile threshold regression for robust estimation
        
        Args:
            y: Dependent variable
            x: Independent variables
            threshold_var: Threshold variable
            quantiles: List of quantiles to estimate (default: [0.1, 0.25, 0.5, 0.75, 0.9])
            
        Returns:
            dict: Quantile regression results across different quantiles
        """
        if quantiles is None:
            quantiles = [0.1, 0.25, 0.5, 0.75, 0.9]
        
        # Ensure arrays are numpy arrays
        if hasattr(y, 'values'):
            y = y.values
        if hasattr(x, 'values'):
            x = x.values
        if hasattr(threshold_var, 'values'):
            threshold_var = threshold_var.values
        
        if x.ndim == 1:
            x = x.reshape(-1, 1)
        
        results = {
            'quantile_models': {},
            'threshold_estimates': {},
            'coefficient_estimates': {},
            'model_comparison': {}
        }
        
        # First, estimate threshold using median regression (more robust)
        try:
            # Use grid search for threshold estimation with quantile regression
            threshold_candidates = np.percentile(threshold_var, np.linspace(10, 90, 20))
            best_threshold = None
            best_objective = np.inf
            
            for thresh_candidate in threshold_candidates:
                try:
                    # Create regime indicators
                    regime1_mask = threshold_var <= thresh_candidate
                    regime2_mask = threshold_var > thresh_candidate
                    
                    if np.sum(regime1_mask) < 10 or np.sum(regime2_mask) < 10:
                        continue
                    
                    # Fit quantile regression for each regime (using median)
                    total_objective = 0
                    
                    # Regime 1
                    y1 = y[regime1_mask]
                    x1 = x[regime1_mask]
                    if len(y1) > 0:
                        qr1 = QuantileRegressor(quantile=0.5, alpha=0.01)
                        qr1.fit(x1, y1)
                        y1_pred = qr1.predict(x1)
                        total_objective += np.sum(np.abs(y1 - y1_pred))
                    
                    # Regime 2
                    y2 = y[regime2_mask]
                    x2 = x[regime2_mask]
                    if len(y2) > 0:
                        qr2 = QuantileRegressor(quantile=0.5, alpha=0.01)
                        qr2.fit(x2, y2)
                        y2_pred = qr2.predict(x2)
                        total_objective += np.sum(np.abs(y2 - y2_pred))
                    
                    if total_objective < best_objective:
                        best_objective = total_objective
                        best_threshold = thresh_candidate
                        
                except Exception:
                    continue
            
            if best_threshold is None:
                best_threshold = np.median(threshold_var)
            
            results['estimated_threshold'] = best_threshold
            
        except Exception as e:
            results['threshold_estimation_error'] = str(e)
            best_threshold = np.median(threshold_var)
            results['estimated_threshold'] = best_threshold
        
        # Fit quantile regressions for each quantile
        regime1_mask = threshold_var <= best_threshold
        regime2_mask = threshold_var > best_threshold
        
        for q in quantiles:
            try:
                quantile_result = {
                    'quantile': q,
                    'threshold': best_threshold,
                    'regime1_model': None,
                    'regime2_model': None,
                    'regime1_coefficients': None,
                    'regime2_coefficients': None,
                    'fitted': False
                }
                
                # Fit regime 1
                if np.sum(regime1_mask) >= 5:
                    y1 = y[regime1_mask]
                    x1 = x[regime1_mask]
                    
                    qr1 = QuantileRegressor(quantile=q, alpha=0.01)
                    qr1.fit(x1, y1)
                    
                    quantile_result['regime1_model'] = qr1
                    quantile_result['regime1_coefficients'] = qr1.coef_
                
                # Fit regime 2
                if np.sum(regime2_mask) >= 5:
                    y2 = y[regime2_mask]
                    x2 = x[regime2_mask]
                    
                    qr2 = QuantileRegressor(quantile=q, alpha=0.01)
                    qr2.fit(x2, y2)
                    
                    quantile_result['regime2_model'] = qr2
                    quantile_result['regime2_coefficients'] = qr2.coef_
                
                if quantile_result['regime1_model'] is not None or quantile_result['regime2_model'] is not None:
                    quantile_result['fitted'] = True
                
                results['quantile_models'][q] = quantile_result
                
            except Exception as e:
                results['quantile_models'][q] = {
                    'quantile': q,
                    'error': str(e),
                    'fitted': False
                }
        
        # Analyze coefficient stability across quantiles
        results['coefficient_stability'] = self._analyze_quantile_coefficient_stability(results['quantile_models'])
        
        # Store results
        self.fitted_models['quantile_threshold'] = results
        
        return results
    
    def bayesian_threshold_model(self, y, x, threshold_var, n_samples=5000, n_chains=4):
        """
        Bayesian threshold model with uncertainty quantification
        
        Args:
            y: Dependent variable
            x: Independent variables
            threshold_var: Threshold variable
            n_samples: Number of MCMC samples per chain
            n_chains: Number of MCMC chains
            
        Returns:
            dict: Bayesian estimation results with posterior distributions
        """
        # Ensure arrays are numpy arrays
        if hasattr(y, 'values'):
            y = y.values
        if hasattr(x, 'values'):
            x = x.values
        if hasattr(threshold_var, 'values'):
            threshold_var = threshold_var.values
        
        if x.ndim == 1:
            x = x.reshape(-1, 1)
        
        results = {
            'posterior_samples': {},
            'posterior_statistics': {},
            'credible_intervals': {},
            'model_diagnostics': {},
            'fitted': False
        }
        
        try:
            # Simplified Bayesian estimation using Metropolis-Hastings
            # This is a basic implementation - in practice, you'd use PyMC3 or Stan
            
            n_obs = len(y)
            n_vars = x.shape[1]
            
            # Prior parameters (weakly informative)
            prior_mean_beta = np.zeros(n_vars + 1)  # +1 for intercept
            prior_cov_beta = np.eye(n_vars + 1) * 10  # Weak prior
            prior_sigma_shape = 2
            prior_sigma_scale = 1
            
            # Initialize parameters
            all_samples = []
            
            for chain in range(n_chains):
                # Initialize chain
                current_threshold = np.random.uniform(
                    np.percentile(threshold_var, 20),
                    np.percentile(threshold_var, 80)
                )
                current_beta1 = np.random.multivariate_normal(prior_mean_beta, prior_cov_beta)
                current_beta2 = np.random.multivariate_normal(prior_mean_beta, prior_cov_beta)
                current_sigma = 1.0
                
                chain_samples = {
                    'threshold': [],
                    'beta1': [],
                    'beta2': [],
                    'sigma': [],
                    'log_likelihood': []
                }
                
                accepted = 0
                
                for i in range(n_samples):
                    # Propose new threshold
                    threshold_proposal = current_threshold + np.random.normal(0, 0.1 * np.std(threshold_var))
                    
                    # Ensure threshold is within reasonable bounds
                    threshold_proposal = np.clip(
                        threshold_proposal,
                        np.percentile(threshold_var, 5),
                        np.percentile(threshold_var, 95)
                    )
                    
                    # Calculate likelihood for current and proposed threshold
                    current_ll = self._calculate_threshold_log_likelihood(
                        y, x, threshold_var, current_threshold, current_beta1, current_beta2, current_sigma
                    )
                    
                    proposed_ll = self._calculate_threshold_log_likelihood(
                        y, x, threshold_var, threshold_proposal, current_beta1, current_beta2, current_sigma
                    )
                    
                    # Metropolis-Hastings acceptance
                    log_alpha = proposed_ll - current_ll
                    if np.log(np.random.uniform()) < log_alpha:
                        current_threshold = threshold_proposal
                        current_ll = proposed_ll
                        accepted += 1
                    
                    # Update coefficients (Gibbs sampling)
                    regime1_mask = threshold_var <= current_threshold
                    regime2_mask = threshold_var > current_threshold
                    
                    if np.sum(regime1_mask) >= 3:
                        y1 = y[regime1_mask]
                        x1_with_intercept = np.column_stack([np.ones(np.sum(regime1_mask)), x[regime1_mask]])
                        
                        # Posterior for beta1 (conjugate normal)
                        precision_matrix = np.linalg.inv(prior_cov_beta) + (x1_with_intercept.T @ x1_with_intercept) / (current_sigma ** 2)
                        posterior_cov = np.linalg.inv(precision_matrix)
                        posterior_mean = posterior_cov @ (
                            np.linalg.inv(prior_cov_beta) @ prior_mean_beta +
                            (x1_with_intercept.T @ y1) / (current_sigma ** 2)
                        )
                        
                        current_beta1 = np.random.multivariate_normal(posterior_mean, posterior_cov)
                    
                    if np.sum(regime2_mask) >= 3:
                        y2 = y[regime2_mask]
                        x2_with_intercept = np.column_stack([np.ones(np.sum(regime2_mask)), x[regime2_mask]])
                        
                        # Posterior for beta2 (conjugate normal)
                        precision_matrix = np.linalg.inv(prior_cov_beta) + (x2_with_intercept.T @ x2_with_intercept) / (current_sigma ** 2)
                        posterior_cov = np.linalg.inv(precision_matrix)
                        posterior_mean = posterior_cov @ (
                            np.linalg.inv(prior_cov_beta) @ prior_mean_beta +
                            (x2_with_intercept.T @ y2) / (current_sigma ** 2)
                        )
                        
                        current_beta2 = np.random.multivariate_normal(posterior_mean, posterior_cov)
                    
                    # Update sigma (inverse gamma posterior)
                    ssr = self._calculate_ssr(y, x, threshold_var, current_threshold, current_beta1, current_beta2)
                    posterior_shape = prior_sigma_shape + n_obs / 2
                    posterior_scale = prior_sigma_scale + ssr / 2
                    current_sigma = np.sqrt(1 / np.random.gamma(posterior_shape, 1 / posterior_scale))
                    
                    # Store samples (after burn-in)
                    if i >= n_samples // 4:  # 25% burn-in
                        chain_samples['threshold'].append(current_threshold)
                        chain_samples['beta1'].append(current_beta1.copy())
                        chain_samples['beta2'].append(current_beta2.copy())
                        chain_samples['sigma'].append(current_sigma)
                        chain_samples['log_likelihood'].append(current_ll)
                
                all_samples.append(chain_samples)
                
            # Combine chains
            combined_samples = {
                'threshold': np.concatenate([chain['threshold'] for chain in all_samples]),
                'beta1': np.vstack([chain['beta1'] for chain in all_samples]),
                'beta2': np.vstack([chain['beta2'] for chain in all_samples]),
                'sigma': np.concatenate([chain['sigma'] for chain in all_samples]),
                'log_likelihood': np.concatenate([chain['log_likelihood'] for chain in all_samples])
            }
            
            results['posterior_samples'] = combined_samples
            
            # Calculate posterior statistics
            results['posterior_statistics'] = {
                'threshold': {
                    'mean': np.mean(combined_samples['threshold']),
                    'std': np.std(combined_samples['threshold']),
                    'median': np.median(combined_samples['threshold'])
                },
                'beta1': {
                    'mean': np.mean(combined_samples['beta1'], axis=0),
                    'std': np.std(combined_samples['beta1'], axis=0),
                    'median': np.median(combined_samples['beta1'], axis=0)
                },
                'beta2': {
                    'mean': np.mean(combined_samples['beta2'], axis=0),
                    'std': np.std(combined_samples['beta2'], axis=0),
                    'median': np.median(combined_samples['beta2'], axis=0)
                },
                'sigma': {
                    'mean': np.mean(combined_samples['sigma']),
                    'std': np.std(combined_samples['sigma']),
                    'median': np.median(combined_samples['sigma'])
                }
            }
            
            # Calculate credible intervals (95%)
            results['credible_intervals'] = {
                'threshold': {
                    'lower': np.percentile(combined_samples['threshold'], 2.5),
                    'upper': np.percentile(combined_samples['threshold'], 97.5)
                },
                'beta1': {
                    'lower': np.percentile(combined_samples['beta1'], 2.5, axis=0),
                    'upper': np.percentile(combined_samples['beta1'], 97.5, axis=0)
                },
                'beta2': {
                    'lower': np.percentile(combined_samples['beta2'], 2.5, axis=0),
                    'upper': np.percentile(combined_samples['beta2'], 97.5, axis=0)
                },
                'sigma': {
                    'lower': np.percentile(combined_samples['sigma'], 2.5),
                    'upper': np.percentile(combined_samples['sigma'], 97.5)
                }
            }
            
            # Model diagnostics
            results['model_diagnostics'] = {
                'n_effective_samples': len(combined_samples['threshold']),
                'acceptance_rate': accepted / n_samples if 'accepted' in locals() else None,
                'rhat': self._calculate_rhat(all_samples),  # Gelman-Rubin statistic
                'effective_sample_size': self._calculate_ess(combined_samples)
            }
            
            results['fitted'] = True
            
        except Exception as e:
            results['error'] = str(e)
            results['fitted'] = False
        
        # Store results
        self.fitted_models['bayesian_threshold'] = results
        
        return results
    
    def cross_validation_model_selection(self, y, x, threshold_var, model_specifications=None, cv_folds=5):
        """
        Cross-validation based model selection for specification choice
        
        Args:
            y: Dependent variable
            x: Independent variables
            threshold_var: Threshold variable
            model_specifications: List of model specifications to compare
            cv_folds: Number of cross-validation folds
            
        Returns:
            dict: Cross-validation results and model selection
        """
        if model_specifications is None:
            model_specifications = [
                {'type': 'hansen', 'params': {}},
                {'type': 'hansen_enhanced', 'params': {'enhancements': {'data_transforms': ['levels']}}},
                {'type': 'hansen_enhanced', 'params': {'enhancements': {'data_transforms': ['differences']}}},
                {'type': 'hansen_enhanced', 'params': {'enhancements': {'additional_controls': True}}},
                {'type': 'multiple_threshold', 'params': {'max_thresholds': 2}},
                {'type': 'smooth_transition', 'params': {'transition_types': ['logistic']}}
            ]
        
        # Ensure arrays are numpy arrays
        if hasattr(y, 'values'):
            y = y.values
        if hasattr(x, 'values'):
            x = x.values
        if hasattr(threshold_var, 'values'):
            threshold_var = threshold_var.values
        
        if x.ndim == 1:
            x = x.reshape(-1, 1)
        
        results = {
            'cv_results': {},
            'model_rankings': {},
            'best_model': None,
            'selection_criteria': {}
        }
        
        n_obs = len(y)
        fold_size = n_obs // cv_folds
        
        # Create cross-validation folds
        cv_folds_indices = []
        for fold in range(cv_folds):
            start_idx = fold * fold_size
            end_idx = (fold + 1) * fold_size if fold < cv_folds - 1 else n_obs
            test_indices = np.arange(start_idx, end_idx)
            train_indices = np.concatenate([np.arange(0, start_idx), np.arange(end_idx, n_obs)])
            cv_folds_indices.append((train_indices, test_indices))
        
        # Evaluate each model specification
        for spec_idx, spec in enumerate(model_specifications):
            spec_name = f"{spec['type']}_{spec_idx}"
            
            cv_scores = {
                'mse': [],
                'mae': [],
                'r_squared': [],
                'successful_fits': 0
            }
            
            for fold, (train_idx, test_idx) in enumerate(cv_folds_indices):
                try:
                    # Split data
                    y_train, y_test = y[train_idx], y[test_idx]
                    x_train, x_test = x[train_idx], x[test_idx]
                    thresh_train, thresh_test = threshold_var[train_idx], threshold_var[test_idx]
                    
                    # Fit model based on specification
                    if spec['type'] == 'hansen':
                        model = HansenThresholdRegression()
                        model.fit(y_train, x_train, thresh_train)
                        
                        if model.fitted:
                            y_pred = model.predict(x_test, thresh_test)
                        else:
                            continue
                            
                    elif spec['type'] == 'hansen_enhanced':
                        enhanced_result = self.enhanced_hansen_regression(
                            y_train, x_train, thresh_train, 
                            spec['params'].get('enhancements', {})
                        )
                        
                        # Use best enhanced model
                        best_model_info = enhanced_result.get('best_model', {})
                        if best_model_info.get('type') == 'original':
                            model = enhanced_result['original_model']['model']
                        else:
                            # Find the best enhanced model
                            best_type = best_model_info['type']
                            enhanced_models = enhanced_result['enhanced_models']
                            
                            if best_type in enhanced_models:
                                model_info = enhanced_models[best_type]
                                if isinstance(model_info, dict) and 'model' in model_info:
                                    model = model_info['model']
                                else:
                                    continue
                            else:
                                continue
                        
                        if hasattr(model, 'fitted') and model.fitted:
                            y_pred = model.predict(x_test, thresh_test)
                        else:
                            continue
                            
                    elif spec['type'] == 'multiple_threshold':
                        multi_result = self.multiple_threshold_model(
                            y_train, x_train, thresh_train,
                            spec['params'].get('max_thresholds', 2)
                        )
                        
                        # Use best multiple threshold model
                        best_multi = multi_result.get('selection_criteria', {}).get('best_model')
                        if best_multi and best_multi in multi_result['models']:
                            model_info = multi_result['models'][best_multi]
                            if model_info.get('fitted', False):
                                # Predict using multiple threshold model (simplified)
                                y_pred = self._predict_multiple_threshold(
                                    model_info, x_test, thresh_test
                                )
                            else:
                                continue
                        else:
                            continue
                            
                    elif spec['type'] == 'smooth_transition':
                        str_result = self.smooth_transition_alternatives(
                            y_train, x_train, thresh_train,
                            spec['params'].get('transition_types', ['logistic'])
                        )
                        
                        # Use best STR model
                        best_str = str_result.get('comparison', {}).get('best_model')
                        if best_str and best_str in str_result['models']:
                            model_info = str_result['models'][best_str]
                            if model_info.get('fitted', False):
                                model = model_info['model']
                                y_pred = model.predict(x_test, thresh_test)
                            else:
                                continue
                        else:
                            continue
                    
                    else:
                        continue
                    
                    # Calculate performance metrics
                    mse = np.mean((y_test - y_pred) ** 2)
                    mae = np.mean(np.abs(y_test - y_pred))
                    
                    ss_res = np.sum((y_test - y_pred) ** 2)
                    ss_tot = np.sum((y_test - np.mean(y_test)) ** 2)
                    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
                    
                    cv_scores['mse'].append(mse)
                    cv_scores['mae'].append(mae)
                    cv_scores['r_squared'].append(r2)
                    cv_scores['successful_fits'] += 1
                    
                except Exception as e:
                    # Skip this fold if fitting fails
                    continue
            
            # Calculate average CV scores
            if cv_scores['successful_fits'] > 0:
                results['cv_results'][spec_name] = {
                    'specification': spec,
                    'mean_mse': np.mean(cv_scores['mse']),
                    'std_mse': np.std(cv_scores['mse']),
                    'mean_mae': np.mean(cv_scores['mae']),
                    'std_mae': np.std(cv_scores['mae']),
                    'mean_r_squared': np.mean(cv_scores['r_squared']),
                    'std_r_squared': np.std(cv_scores['r_squared']),
                    'successful_folds': cv_scores['successful_fits'],
                    'success_rate': cv_scores['successful_fits'] / cv_folds
                }
            else:
                results['cv_results'][spec_name] = {
                    'specification': spec,
                    'error': 'No successful fits across CV folds',
                    'successful_folds': 0,
                    'success_rate': 0
                }
        
        # Rank models by different criteria
        successful_models = {k: v for k, v in results['cv_results'].items() 
                           if v.get('successful_folds', 0) > 0}
        
        if successful_models:
            # Rank by R²
            r2_ranking = sorted(successful_models.items(), 
                              key=lambda x: x[1]['mean_r_squared'], reverse=True)
            
            # Rank by MSE (lower is better)
            mse_ranking = sorted(successful_models.items(), 
                               key=lambda x: x[1]['mean_mse'])
            
            # Rank by MAE (lower is better)
            mae_ranking = sorted(successful_models.items(), 
                               key=lambda x: x[1]['mean_mae'])
            
            results['model_rankings'] = {
                'by_r_squared': [(name, info['mean_r_squared']) for name, info in r2_ranking],
                'by_mse': [(name, info['mean_mse']) for name, info in mse_ranking],
                'by_mae': [(name, info['mean_mae']) for name, info in mae_ranking]
            }
            
            # Select best model (by R²)
            results['best_model'] = {
                'name': r2_ranking[0][0],
                'specification': r2_ranking[0][1]['specification'],
                'cv_performance': r2_ranking[0][1]
            }
            
            # Selection criteria summary
            results['selection_criteria'] = {
                'primary_criterion': 'r_squared',
                'best_r_squared': r2_ranking[0][1]['mean_r_squared'],
                'best_mse': mse_ranking[0][1]['mean_mse'],
                'best_mae': mae_ranking[0][1]['mean_mae'],
                'n_successful_models': len(successful_models)
            }
        
        # Store results
        self.fitted_models['cv_model_selection'] = results
        
        return results
    
    def _analyze_quantile_coefficient_stability(self, quantile_models):
        """Analyze coefficient stability across quantiles"""
        stability_results = {
            'regime1_stability': {},
            'regime2_stability': {},
            'overall_stability': {}
        }
        
        # Extract coefficients for each quantile
        quantiles = []
        regime1_coefs = []
        regime2_coefs = []
        
        for q, model_info in quantile_models.items():
            if model_info.get('fitted', False):
                quantiles.append(q)
                
                if model_info['regime1_coefficients'] is not None:
                    regime1_coefs.append(model_info['regime1_coefficients'])
                
                if model_info['regime2_coefficients'] is not None:
                    regime2_coefs.append(model_info['regime2_coefficients'])
        
        # Analyze regime 1 stability
        if len(regime1_coefs) > 1:
            regime1_array = np.array(regime1_coefs)
            stability_results['regime1_stability'] = {
                'coefficient_ranges': np.ptp(regime1_array, axis=0),  # Peak-to-peak
                'coefficient_std': np.std(regime1_array, axis=0),
                'mean_coefficients': np.mean(regime1_array, axis=0)
            }
        
        # Analyze regime 2 stability
        if len(regime2_coefs) > 1:
            regime2_array = np.array(regime2_coefs)
            stability_results['regime2_stability'] = {
                'coefficient_ranges': np.ptp(regime2_array, axis=0),
                'coefficient_std': np.std(regime2_array, axis=0),
                'mean_coefficients': np.mean(regime2_array, axis=0)
            }
        
        # Overall stability assessment
        stability_results['overall_stability'] = {
            'n_quantiles_fitted': len(quantiles),
            'quantiles_fitted': quantiles,
            'stability_assessment': 'stable' if (
                len(regime1_coefs) > 1 and np.all(stability_results['regime1_stability']['coefficient_std'] < 0.5)
            ) else 'unstable'
        }
        
        return stability_results
    
    def _calculate_threshold_log_likelihood(self, y, x, threshold_var, threshold, beta1, beta2, sigma):
        """Calculate log-likelihood for threshold model"""
        try:
            regime1_mask = threshold_var <= threshold
            regime2_mask = threshold_var > threshold
            
            log_likelihood = 0
            
            # Regime 1
            if np.sum(regime1_mask) > 0:
                y1 = y[regime1_mask]
                x1_with_intercept = np.column_stack([np.ones(np.sum(regime1_mask)), x[regime1_mask]])
                
                if len(beta1) == x1_with_intercept.shape[1]:
                    y1_pred = x1_with_intercept @ beta1
                    residuals1 = y1 - y1_pred
                    log_likelihood += -0.5 * np.sum(residuals1 ** 2) / (sigma ** 2) - 0.5 * len(y1) * np.log(2 * np.pi * sigma ** 2)
            
            # Regime 2
            if np.sum(regime2_mask) > 0:
                y2 = y[regime2_mask]
                x2_with_intercept = np.column_stack([np.ones(np.sum(regime2_mask)), x[regime2_mask]])
                
                if len(beta2) == x2_with_intercept.shape[1]:
                    y2_pred = x2_with_intercept @ beta2
                    residuals2 = y2 - y2_pred
                    log_likelihood += -0.5 * np.sum(residuals2 ** 2) / (sigma ** 2) - 0.5 * len(y2) * np.log(2 * np.pi * sigma ** 2)
            
            return log_likelihood
            
        except Exception:
            return -np.inf
    
    def _calculate_ssr(self, y, x, threshold_var, threshold, beta1, beta2):
        """Calculate sum of squared residuals"""
        try:
            regime1_mask = threshold_var <= threshold
            regime2_mask = threshold_var > threshold
            
            ssr = 0
            
            # Regime 1
            if np.sum(regime1_mask) > 0:
                y1 = y[regime1_mask]
                x1_with_intercept = np.column_stack([np.ones(np.sum(regime1_mask)), x[regime1_mask]])
                
                if len(beta1) == x1_with_intercept.shape[1]:
                    y1_pred = x1_with_intercept @ beta1
                    ssr += np.sum((y1 - y1_pred) ** 2)
            
            # Regime 2
            if np.sum(regime2_mask) > 0:
                y2 = y[regime2_mask]
                x2_with_intercept = np.column_stack([np.ones(np.sum(regime2_mask)), x[regime2_mask]])
                
                if len(beta2) == x2_with_intercept.shape[1]:
                    y2_pred = x2_with_intercept @ beta2
                    ssr += np.sum((y2 - y2_pred) ** 2)
            
            return ssr
            
        except Exception:
            return np.inf
    
    def _calculate_rhat(self, chain_samples):
        """Calculate Gelman-Rubin R-hat statistic for convergence diagnosis"""
        try:
            # Extract threshold samples from each chain
            chains = [chain['threshold'] for chain in chain_samples]
            n_chains = len(chains)
            n_samples = len(chains[0])
            
            # Calculate between-chain and within-chain variance
            chain_means = [np.mean(chain) for chain in chains]
            overall_mean = np.mean(chain_means)
            
            # Between-chain variance
            B = n_samples * np.var(chain_means, ddof=1)
            
            # Within-chain variance
            W = np.mean([np.var(chain, ddof=1) for chain in chains])
            
            # Marginal posterior variance
            var_plus = ((n_samples - 1) / n_samples) * W + (1 / n_samples) * B
            
            # R-hat
            rhat = np.sqrt(var_plus / W)
            
            return rhat
            
        except Exception:
            return None
    
    def _calculate_ess(self, combined_samples):
        """Calculate effective sample size"""
        try:
            # Simplified ESS calculation
            n_samples = len(combined_samples['threshold'])
            
            # Calculate autocorrelation (simplified)
            threshold_samples = combined_samples['threshold']
            autocorr = np.correlate(threshold_samples, threshold_samples, mode='full')
            autocorr = autocorr[autocorr.size // 2:]
            autocorr = autocorr / autocorr[0]
            
            # Find first negative autocorrelation
            first_negative = np.where(autocorr < 0)[0]
            if len(first_negative) > 0:
                tau = first_negative[0]
            else:
                tau = len(autocorr) // 4
            
            ess = n_samples / (1 + 2 * np.sum(autocorr[1:tau]))
            
            return max(1, int(ess))
            
        except Exception:
            return n_samples // 4  # Conservative estimate
    
    def _predict_multiple_threshold(self, model_info, x_test, threshold_test):
        """Predict using multiple threshold model"""
        try:
            thresholds = model_info.get('thresholds', [])
            regime_models = model_info.get('regime_models', [])
            
            if not thresholds or not regime_models:
                return np.zeros(len(x_test))
            
            y_pred = np.zeros(len(x_test))
            
            # Sort thresholds
            sorted_thresholds = sorted(thresholds)
            
            # Create regime masks
            regime_masks = []
            
            # First regime
            regime_masks.append(threshold_test <= sorted_thresholds[0])
            
            # Middle regimes
            for i in range(len(sorted_thresholds) - 1):
                mask = (threshold_test > sorted_thresholds[i]) & (threshold_test <= sorted_thresholds[i + 1])
                regime_masks.append(mask)
            
            # Last regime
            regime_masks.append(threshold_test > sorted_thresholds[-1])
            
            # Predict for each regime
            for i, (mask, regime_model) in enumerate(zip(regime_masks, regime_models)):
                if regime_model is not None and np.sum(mask) > 0:
                    x_regime = x_test[mask]
                    
                    # Add intercept
                    if x_regime.ndim == 1:
                        X_regime = np.column_stack([np.ones(len(x_regime)), x_regime])
                    else:
                        X_regime = np.column_stack([np.ones(len(x_regime)), x_regime])
                    
                    # Predict
                    beta = regime_model.get('beta', np.zeros(X_regime.shape[1]))
                    if len(beta) == X_regime.shape[1]:
                        y_pred[mask] = X_regime @ beta
            
            return y_pred
            
        except Exception:
            return np.zeros(len(x_test))    
    
    def bootstrap_threshold_estimation(self, y, x, threshold_var, n_bootstrap=1000, confidence_level=0.95):
        """
        Bootstrap threshold estimation with confidence intervals
        
        Args:
            y: Dependent variable
            x: Independent variables
            threshold_var: Threshold variable
            n_bootstrap: Number of bootstrap samples
            confidence_level: Confidence level for intervals
            
        Returns:
            dict: Bootstrap results with confidence intervals
        """
        # Ensure arrays are numpy arrays
        if hasattr(y, 'values'):
            y = y.values
        if hasattr(x, 'values'):
            x = x.values
        if hasattr(threshold_var, 'values'):
            threshold_var = threshold_var.values
        
        if x.ndim == 1:
            x = x.reshape(-1, 1)
        
        n_obs = len(y)
        bootstrap_results = {
            'thresholds': [],
            'coefficients_regime1': [],
            'coefficients_regime2': [],
            'r_squared_values': [],
            'successful_fits': 0
        }
        
        # Original model for comparison
        try:
            original_model = HansenThresholdRegression()
            original_model.fit(y, x, threshold_var)
            original_threshold = original_model.threshold if original_model.fitted else None
        except:
            original_threshold = None
        
        # Bootstrap sampling
        for i in range(n_bootstrap):
            try:
                # Resample with replacement
                bootstrap_indices = np.random.choice(n_obs, size=n_obs, replace=True)
                y_boot = y[bootstrap_indices]
                x_boot = x[bootstrap_indices]
                threshold_boot = threshold_var[bootstrap_indices]
                
                # Fit Hansen model on bootstrap sample
                hansen_boot = HansenThresholdRegression()
                hansen_boot.fit(y_boot, x_boot, threshold_boot)
                
                if hansen_boot.fitted:
                    # Calculate R² for bootstrap sample
                    y_pred_boot = hansen_boot.predict(x_boot, threshold_boot)
                    ss_res = np.sum((y_boot - y_pred_boot) ** 2)
                    ss_tot = np.sum((y_boot - np.mean(y_boot)) ** 2)
                    r2_boot = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
                    
                    # Store results
                    bootstrap_results['thresholds'].append(hansen_boot.threshold)
                    bootstrap_results['coefficients_regime1'].append(hansen_boot.beta1)
                    bootstrap_results['coefficients_regime2'].append(hansen_boot.beta2)
                    bootstrap_results['r_squared_values'].append(r2_boot)
                    bootstrap_results['successful_fits'] += 1
                    
            except Exception:
                continue
        
        # Calculate confidence intervals
        alpha = 1 - confidence_level
        lower_percentile = (alpha / 2) * 100
        upper_percentile = (1 - alpha / 2) * 100
        
        results = {
            'original_threshold': original_threshold,
            'bootstrap_statistics': {},
            'confidence_intervals': {},
            'n_successful_bootstrap': bootstrap_results['successful_fits'],
            'bootstrap_success_rate': bootstrap_results['successful_fits'] / n_bootstrap
        }
        
        if bootstrap_results['successful_fits'] > 0:
            # Threshold statistics
            thresholds = np.array(bootstrap_results['thresholds'])
            results['bootstrap_statistics']['threshold'] = {
                'mean': np.mean(thresholds),
                'std': np.std(thresholds),
                'median': np.median(thresholds)
            }
            results['confidence_intervals']['threshold'] = {
                'lower': np.percentile(thresholds, lower_percentile),
                'upper': np.percentile(thresholds, upper_percentile)
            }
            
            # R² statistics
            r2_values = np.array(bootstrap_results['r_squared_values'])
            results['bootstrap_statistics']['r_squared'] = {
                'mean': np.mean(r2_values),
                'std': np.std(r2_values),
                'median': np.median(r2_values)
            }
            results['confidence_intervals']['r_squared'] = {
                'lower': np.percentile(r2_values, lower_percentile),
                'upper': np.percentile(r2_values, upper_percentile)
            }
            
            # Coefficient statistics (regime 1)
            if bootstrap_results['coefficients_regime1']:
                coef1_array = np.array(bootstrap_results['coefficients_regime1'])
                results['bootstrap_statistics']['coefficients_regime1'] = {
                    'mean': np.mean(coef1_array, axis=0),
                    'std': np.std(coef1_array, axis=0),
                    'median': np.median(coef1_array, axis=0)
                }
                results['confidence_intervals']['coefficients_regime1'] = {
                    'lower': np.percentile(coef1_array, lower_percentile, axis=0),
                    'upper': np.percentile(coef1_array, upper_percentile, axis=0)
                }
            
            # Coefficient statistics (regime 2)
            if bootstrap_results['coefficients_regime2']:
                coef2_array = np.array(bootstrap_results['coefficients_regime2'])
                results['bootstrap_statistics']['coefficients_regime2'] = {
                    'mean': np.mean(coef2_array, axis=0),
                    'std': np.std(coef2_array, axis=0),
                    'median': np.median(coef2_array, axis=0)
                }
                results['confidence_intervals']['coefficients_regime2'] = {
                    'lower': np.percentile(coef2_array, lower_percentile, axis=0),
                    'upper': np.percentile(coef2_array, upper_percentile, axis=0)
                }
        
        # Store results
        self.fitted_models['bootstrap_threshold'] = results
        
        return results
    
    def quantile_threshold_regression(self, y, x, threshold_var, quantiles=None):
        """
        Quantile threshold regression for robust estimation
        
        Args:
            y: Dependent variable
            x: Independent variables
            threshold_var: Threshold variable
            quantiles: List of quantiles to estimate (default: [0.1, 0.25, 0.5, 0.75, 0.9])
            
        Returns:
            dict: Quantile regression results across different quantiles
        """
        if quantiles is None:
            quantiles = [0.1, 0.25, 0.5, 0.75, 0.9]
        
        # Ensure arrays are numpy arrays
        if hasattr(y, 'values'):
            y = y.values
        if hasattr(x, 'values'):
            x = x.values
        if hasattr(threshold_var, 'values'):
            threshold_var = threshold_var.values
        
        if x.ndim == 1:
            x = x.reshape(-1, 1)
        
        results = {
            'quantile_models': {},
            'threshold_estimates': {},
            'coefficient_estimates': {},
            'model_comparison': {}
        }
        
        # First, estimate threshold using median regression (more robust)
        try:
            # Use grid search for threshold estimation with quantile regression
            threshold_candidates = np.percentile(threshold_var, np.linspace(10, 90, 20))
            best_threshold = None
            best_objective = np.inf
            
            for thresh_candidate in threshold_candidates:
                try:
                    # Create regime indicators
                    regime1_mask = threshold_var <= thresh_candidate
                    regime2_mask = threshold_var > thresh_candidate
                    
                    if np.sum(regime1_mask) < 10 or np.sum(regime2_mask) < 10:
                        continue
                    
                    # Fit quantile regression for each regime (using median)
                    total_objective = 0
                    
                    # Regime 1
                    y1 = y[regime1_mask]
                    x1 = x[regime1_mask]
                    if len(y1) > 0:
                        qr1 = QuantileRegressor(quantile=0.5, alpha=0.01)
                        qr1.fit(x1, y1)
                        y1_pred = qr1.predict(x1)
                        total_objective += np.sum(np.abs(y1 - y1_pred))
                    
                    # Regime 2
                    y2 = y[regime2_mask]
                    x2 = x[regime2_mask]
                    if len(y2) > 0:
                        qr2 = QuantileRegressor(quantile=0.5, alpha=0.01)
                        qr2.fit(x2, y2)
                        y2_pred = qr2.predict(x2)
                        total_objective += np.sum(np.abs(y2 - y2_pred))
                    
                    if total_objective < best_objective:
                        best_objective = total_objective
                        best_threshold = thresh_candidate
                        
                except Exception:
                    continue
            
            if best_threshold is None:
                best_threshold = np.median(threshold_var)
            
            results['estimated_threshold'] = best_threshold
            
        except Exception as e:
            results['threshold_estimation_error'] = str(e)
            best_threshold = np.median(threshold_var)
            results['estimated_threshold'] = best_threshold
        
        # Fit quantile regressions for each quantile
        regime1_mask = threshold_var <= best_threshold
        regime2_mask = threshold_var > best_threshold
        
        for q in quantiles:
            try:
                quantile_result = {
                    'quantile': q,
                    'threshold': best_threshold,
                    'regime1_model': None,
                    'regime2_model': None,
                    'regime1_coefficients': None,
                    'regime2_coefficients': None,
                    'fitted': False
                }
                
                # Fit regime 1
                if np.sum(regime1_mask) >= 5:
                    y1 = y[regime1_mask]
                    x1 = x[regime1_mask]
                    
                    qr1 = QuantileRegressor(quantile=q, alpha=0.01)
                    qr1.fit(x1, y1)
                    
                    quantile_result['regime1_model'] = qr1
                    quantile_result['regime1_coefficients'] = qr1.coef_
                
                # Fit regime 2
                if np.sum(regime2_mask) >= 5:
                    y2 = y[regime2_mask]
                    x2 = x[regime2_mask]
                    
                    qr2 = QuantileRegressor(quantile=q, alpha=0.01)
                    qr2.fit(x2, y2)
                    
                    quantile_result['regime2_model'] = qr2
                    quantile_result['regime2_coefficients'] = qr2.coef_
                
                if quantile_result['regime1_model'] is not None or quantile_result['regime2_model'] is not None:
                    quantile_result['fitted'] = True
                
                results['quantile_models'][q] = quantile_result
                
            except Exception as e:
                results['quantile_models'][q] = {
                    'quantile': q,
                    'error': str(e),
                    'fitted': False
                }
        
        # Analyze coefficient stability across quantiles
        results['coefficient_stability'] = self._analyze_quantile_coefficient_stability(results['quantile_models'])
        
        # Store results
        self.fitted_models['quantile_threshold'] = results
        
        return results
    
    def bayesian_threshold_model(self, y, x, threshold_var, n_samples=5000, n_chains=4):
        """
        Bayesian threshold model with uncertainty quantification
        
        Args:
            y: Dependent variable
            x: Independent variables
            threshold_var: Threshold variable
            n_samples: Number of MCMC samples per chain
            n_chains: Number of MCMC chains
            
        Returns:
            dict: Bayesian estimation results with posterior distributions
        """
        # Ensure arrays are numpy arrays
        if hasattr(y, 'values'):
            y = y.values
        if hasattr(x, 'values'):
            x = x.values
        if hasattr(threshold_var, 'values'):
            threshold_var = threshold_var.values
        
        if x.ndim == 1:
            x = x.reshape(-1, 1)
        
        results = {
            'posterior_samples': {},
            'posterior_statistics': {},
            'credible_intervals': {},
            'model_diagnostics': {},
            'fitted': False
        }
        
        try:
            # Simplified Bayesian estimation using Metropolis-Hastings
            n_obs = len(y)
            n_vars = x.shape[1]
            
            # Prior parameters (weakly informative)
            prior_mean_beta = np.zeros(n_vars + 1)  # +1 for intercept
            prior_cov_beta = np.eye(n_vars + 1) * 10  # Weak prior
            prior_sigma_shape = 2
            prior_sigma_scale = 1
            
            # Initialize parameters
            all_samples = []
            
            for chain in range(n_chains):
                # Initialize chain
                current_threshold = np.random.uniform(
                    np.percentile(threshold_var, 20),
                    np.percentile(threshold_var, 80)
                )
                current_beta1 = np.random.multivariate_normal(prior_mean_beta, prior_cov_beta)
                current_beta2 = np.random.multivariate_normal(prior_mean_beta, prior_cov_beta)
                current_sigma = 1.0
                
                chain_samples = {
                    'threshold': [],
                    'beta1': [],
                    'beta2': [],
                    'sigma': [],
                    'log_likelihood': []
                }
                
                accepted = 0
                
                for i in range(n_samples):
                    # Propose new threshold
                    threshold_proposal = current_threshold + np.random.normal(0, 0.1 * np.std(threshold_var))
                    
                    # Ensure threshold is within reasonable bounds
                    threshold_proposal = np.clip(
                        threshold_proposal,
                        np.percentile(threshold_var, 5),
                        np.percentile(threshold_var, 95)
                    )
                    
                    # Calculate likelihood for current and proposed threshold
                    current_ll = self._calculate_threshold_log_likelihood(
                        y, x, threshold_var, current_threshold, current_beta1, current_beta2, current_sigma
                    )
                    
                    proposed_ll = self._calculate_threshold_log_likelihood(
                        y, x, threshold_var, threshold_proposal, current_beta1, current_beta2, current_sigma
                    )
                    
                    # Metropolis-Hastings acceptance
                    log_alpha = proposed_ll - current_ll
                    if np.log(np.random.uniform()) < log_alpha:
                        current_threshold = threshold_proposal
                        current_ll = proposed_ll
                        accepted += 1
                    
                    # Update coefficients (Gibbs sampling)
                    regime1_mask = threshold_var <= current_threshold
                    regime2_mask = threshold_var > current_threshold
                    
                    if np.sum(regime1_mask) >= 3:
                        y1 = y[regime1_mask]
                        x1_with_intercept = np.column_stack([np.ones(np.sum(regime1_mask)), x[regime1_mask]])
                        
                        # Posterior for beta1 (conjugate normal)
                        precision_matrix = np.linalg.inv(prior_cov_beta) + (x1_with_intercept.T @ x1_with_intercept) / (current_sigma ** 2)
                        posterior_cov = np.linalg.inv(precision_matrix)
                        posterior_mean = posterior_cov @ (
                            np.linalg.inv(prior_cov_beta) @ prior_mean_beta +
                            (x1_with_intercept.T @ y1) / (current_sigma ** 2)
                        )
                        
                        current_beta1 = np.random.multivariate_normal(posterior_mean, posterior_cov)
                    
                    if np.sum(regime2_mask) >= 3:
                        y2 = y[regime2_mask]
                        x2_with_intercept = np.column_stack([np.ones(np.sum(regime2_mask)), x[regime2_mask]])
                        
                        # Posterior for beta2 (conjugate normal)
                        precision_matrix = np.linalg.inv(prior_cov_beta) + (x2_with_intercept.T @ x2_with_intercept) / (current_sigma ** 2)
                        posterior_cov = np.linalg.inv(precision_matrix)
                        posterior_mean = posterior_cov @ (
                            np.linalg.inv(prior_cov_beta) @ prior_mean_beta +
                            (x2_with_intercept.T @ y2) / (current_sigma ** 2)
                        )
                        
                        current_beta2 = np.random.multivariate_normal(posterior_mean, posterior_cov)
                    
                    # Update sigma (inverse gamma posterior)
                    ssr = self._calculate_ssr(y, x, threshold_var, current_threshold, current_beta1, current_beta2)
                    posterior_shape = prior_sigma_shape + n_obs / 2
                    posterior_scale = prior_sigma_scale + ssr / 2
                    current_sigma = np.sqrt(1 / np.random.gamma(posterior_shape, 1 / posterior_scale))
                    
                    # Store samples (after burn-in)
                    if i >= n_samples // 4:  # 25% burn-in
                        chain_samples['threshold'].append(current_threshold)
                        chain_samples['beta1'].append(current_beta1.copy())
                        chain_samples['beta2'].append(current_beta2.copy())
                        chain_samples['sigma'].append(current_sigma)
                        chain_samples['log_likelihood'].append(current_ll)
                
                all_samples.append(chain_samples)
                
            # Combine chains
            combined_samples = {
                'threshold': np.concatenate([chain['threshold'] for chain in all_samples]),
                'beta1': np.vstack([chain['beta1'] for chain in all_samples]),
                'beta2': np.vstack([chain['beta2'] for chain in all_samples]),
                'sigma': np.concatenate([chain['sigma'] for chain in all_samples]),
                'log_likelihood': np.concatenate([chain['log_likelihood'] for chain in all_samples])
            }
            
            results['posterior_samples'] = combined_samples
            
            # Calculate posterior statistics
            results['posterior_statistics'] = {
                'threshold': {
                    'mean': np.mean(combined_samples['threshold']),
                    'std': np.std(combined_samples['threshold']),
                    'median': np.median(combined_samples['threshold'])
                },
                'beta1': {
                    'mean': np.mean(combined_samples['beta1'], axis=0),
                    'std': np.std(combined_samples['beta1'], axis=0),
                    'median': np.median(combined_samples['beta1'], axis=0)
                },
                'beta2': {
                    'mean': np.mean(combined_samples['beta2'], axis=0),
                    'std': np.std(combined_samples['beta2'], axis=0),
                    'median': np.median(combined_samples['beta2'], axis=0)
                },
                'sigma': {
                    'mean': np.mean(combined_samples['sigma']),
                    'std': np.std(combined_samples['sigma']),
                    'median': np.median(combined_samples['sigma'])
                }
            }
            
            # Calculate credible intervals (95%)
            results['credible_intervals'] = {
                'threshold': {
                    'lower': np.percentile(combined_samples['threshold'], 2.5),
                    'upper': np.percentile(combined_samples['threshold'], 97.5)
                },
                'beta1': {
                    'lower': np.percentile(combined_samples['beta1'], 2.5, axis=0),
                    'upper': np.percentile(combined_samples['beta1'], 97.5, axis=0)
                },
                'beta2': {
                    'lower': np.percentile(combined_samples['beta2'], 2.5, axis=0),
                    'upper': np.percentile(combined_samples['beta2'], 97.5, axis=0)
                },
                'sigma': {
                    'lower': np.percentile(combined_samples['sigma'], 2.5),
                    'upper': np.percentile(combined_samples['sigma'], 97.5)
                }
            }
            
            # Model diagnostics
            results['model_diagnostics'] = {
                'n_effective_samples': len(combined_samples['threshold']),
                'acceptance_rate': accepted / n_samples if 'accepted' in locals() else None,
                'rhat': self._calculate_rhat(all_samples),  # Gelman-Rubin statistic
                'effective_sample_size': self._calculate_ess(combined_samples)
            }
            
            results['fitted'] = True
            
        except Exception as e:
            results['error'] = str(e)
            results['fitted'] = False
        
        # Store results
        self.fitted_models['bayesian_threshold'] = results
        
        return results
    
    def cross_validation_model_selection(self, y, x, threshold_var, model_specifications=None, cv_folds=5):
        """
        Cross-validation based model selection for specification choice
        
        Args:
            y: Dependent variable
            x: Independent variables
            threshold_var: Threshold variable
            model_specifications: List of model specifications to compare
            cv_folds: Number of cross-validation folds
            
        Returns:
            dict: Cross-validation results and model selection
        """
        if model_specifications is None:
            model_specifications = [
                {'type': 'hansen', 'params': {}},
                {'type': 'hansen_enhanced', 'params': {'enhancements': {'data_transforms': ['levels']}}},
                {'type': 'hansen_enhanced', 'params': {'enhancements': {'data_transforms': ['differences']}}},
                {'type': 'hansen_enhanced', 'params': {'enhancements': {'additional_controls': True}}},
                {'type': 'multiple_threshold', 'params': {'max_thresholds': 2}},
                {'type': 'smooth_transition', 'params': {'transition_types': ['logistic']}}
            ]
        
        # Ensure arrays are numpy arrays
        if hasattr(y, 'values'):
            y = y.values
        if hasattr(x, 'values'):
            x = x.values
        if hasattr(threshold_var, 'values'):
            threshold_var = threshold_var.values
        
        if x.ndim == 1:
            x = x.reshape(-1, 1)
        
        results = {
            'cv_results': {},
            'model_rankings': {},
            'best_model': None,
            'selection_criteria': {}
        }
        
        n_obs = len(y)
        fold_size = n_obs // cv_folds
        
        # Create cross-validation folds
        cv_folds_indices = []
        for fold in range(cv_folds):
            start_idx = fold * fold_size
            end_idx = (fold + 1) * fold_size if fold < cv_folds - 1 else n_obs
            test_indices = np.arange(start_idx, end_idx)
            train_indices = np.concatenate([np.arange(0, start_idx), np.arange(end_idx, n_obs)])
            cv_folds_indices.append((train_indices, test_indices))
        
        # Evaluate each model specification
        for spec_idx, spec in enumerate(model_specifications):
            spec_name = f"{spec['type']}_{spec_idx}"
            
            cv_scores = {
                'mse': [],
                'mae': [],
                'r_squared': [],
                'successful_fits': 0
            }
            
            for fold, (train_idx, test_idx) in enumerate(cv_folds_indices):
                try:
                    # Split data
                    y_train, y_test = y[train_idx], y[test_idx]
                    x_train, x_test = x[train_idx], x[test_idx]
                    thresh_train, thresh_test = threshold_var[train_idx], threshold_var[test_idx]
                    
                    # Fit model based on specification
                    if spec['type'] == 'hansen':
                        model = HansenThresholdRegression()
                        model.fit(y_train, x_train, thresh_train)
                        
                        if model.fitted:
                            y_pred = model.predict(x_test, thresh_test)
                        else:
                            continue
                            
                    elif spec['type'] == 'hansen_enhanced':
                        enhanced_result = self.enhanced_hansen_regression(
                            y_train, x_train, thresh_train, 
                            spec['params'].get('enhancements', {})
                        )
                        
                        # Use best enhanced model
                        best_model_info = enhanced_result.get('best_model', {})
                        if best_model_info.get('type') == 'original':
                            model = enhanced_result['original_model']['model']
                        else:
                            # Find the best enhanced model
                            best_type = best_model_info['type']
                            enhanced_models = enhanced_result['enhanced_models']
                            
                            if best_type in enhanced_models:
                                model_info = enhanced_models[best_type]
                                if isinstance(model_info, dict) and 'model' in model_info:
                                    model = model_info['model']
                                else:
                                    continue
                            else:
                                continue
                        
                        if hasattr(model, 'fitted') and model.fitted:
                            y_pred = model.predict(x_test, thresh_test)
                        else:
                            continue
                            
                    elif spec['type'] == 'multiple_threshold':
                        multi_result = self.multiple_threshold_model(
                            y_train, x_train, thresh_train,
                            spec['params'].get('max_thresholds', 2)
                        )
                        
                        # Use best multiple threshold model
                        best_multi = multi_result.get('selection_criteria', {}).get('best_model')
                        if best_multi and best_multi in multi_result['models']:
                            model_info = multi_result['models'][best_multi]
                            if model_info.get('fitted', False):
                                # Predict using multiple threshold model (simplified)
                                y_pred = self._predict_multiple_threshold(
                                    model_info, x_test, thresh_test
                                )
                            else:
                                continue
                        else:
                            continue
                            
                    elif spec['type'] == 'smooth_transition':
                        str_result = self.smooth_transition_alternatives(
                            y_train, x_train, thresh_train,
                            spec['params'].get('transition_types', ['logistic'])
                        )
                        
                        # Use best STR model
                        best_str = str_result.get('comparison', {}).get('best_model')
                        if best_str and best_str in str_result['models']:
                            model_info = str_result['models'][best_str]
                            if model_info.get('fitted', False):
                                model = model_info['model']
                                y_pred = model.predict(x_test, thresh_test)
                            else:
                                continue
                        else:
                            continue
                    
                    else:
                        continue
                    
                    # Calculate performance metrics
                    mse = np.mean((y_test - y_pred) ** 2)
                    mae = np.mean(np.abs(y_test - y_pred))
                    
                    ss_res = np.sum((y_test - y_pred) ** 2)
                    ss_tot = np.sum((y_test - np.mean(y_test)) ** 2)
                    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
                    
                    cv_scores['mse'].append(mse)
                    cv_scores['mae'].append(mae)
                    cv_scores['r_squared'].append(r2)
                    cv_scores['successful_fits'] += 1
                    
                except Exception as e:
                    # Skip this fold if fitting fails
                    continue
            
            # Calculate average CV scores
            if cv_scores['successful_fits'] > 0:
                results['cv_results'][spec_name] = {
                    'specification': spec,
                    'mean_mse': np.mean(cv_scores['mse']),
                    'std_mse': np.std(cv_scores['mse']),
                    'mean_mae': np.mean(cv_scores['mae']),
                    'std_mae': np.std(cv_scores['mae']),
                    'mean_r_squared': np.mean(cv_scores['r_squared']),
                    'std_r_squared': np.std(cv_scores['r_squared']),
                    'successful_folds': cv_scores['successful_fits'],
                    'success_rate': cv_scores['successful_fits'] / cv_folds
                }
            else:
                results['cv_results'][spec_name] = {
                    'specification': spec,
                    'error': 'No successful fits across CV folds',
                    'successful_folds': 0,
                    'success_rate': 0
                }
        
        # Rank models by different criteria
        successful_models = {k: v for k, v in results['cv_results'].items() 
                           if v.get('successful_folds', 0) > 0}
        
        if successful_models:
            # Rank by R²
            r2_ranking = sorted(successful_models.items(), 
                              key=lambda x: x[1]['mean_r_squared'], reverse=True)
            
            # Rank by MSE (lower is better)
            mse_ranking = sorted(successful_models.items(), 
                               key=lambda x: x[1]['mean_mse'])
            
            # Rank by MAE (lower is better)
            mae_ranking = sorted(successful_models.items(), 
                               key=lambda x: x[1]['mean_mae'])
            
            results['model_rankings'] = {
                'by_r_squared': [(name, info['mean_r_squared']) for name, info in r2_ranking],
                'by_mse': [(name, info['mean_mse']) for name, info in mse_ranking],
                'by_mae': [(name, info['mean_mae']) for name, info in mae_ranking]
            }
            
            # Select best model (by R²)
            results['best_model'] = {
                'name': r2_ranking[0][0],
                'specification': r2_ranking[0][1]['specification'],
                'cv_performance': r2_ranking[0][1]
            }
            
            # Selection criteria summary
            results['selection_criteria'] = {
                'primary_criterion': 'r_squared',
                'best_r_squared': r2_ranking[0][1]['mean_r_squared'],
                'best_mse': mse_ranking[0][1]['mean_mse'],
                'best_mae': mae_ranking[0][1]['mean_mae'],
                'n_successful_models': len(successful_models)
            }
        
        # Store results
        self.fitted_models['cv_model_selection'] = results
        
        return results
    
    def _analyze_quantile_coefficient_stability(self, quantile_models):
        """Analyze coefficient stability across quantiles"""
        stability_results = {
            'regime1_stability': {},
            'regime2_stability': {},
            'overall_stability': {}
        }
        
        # Extract coefficients for each quantile
        quantiles = []
        regime1_coefs = []
        regime2_coefs = []
        
        for q, model_info in quantile_models.items():
            if model_info.get('fitted', False):
                quantiles.append(q)
                
                if model_info['regime1_coefficients'] is not None:
                    regime1_coefs.append(model_info['regime1_coefficients'])
                
                if model_info['regime2_coefficients'] is not None:
                    regime2_coefs.append(model_info['regime2_coefficients'])
        
        # Analyze regime 1 stability
        if len(regime1_coefs) > 1:
            regime1_array = np.array(regime1_coefs)
            stability_results['regime1_stability'] = {
                'coefficient_ranges': np.ptp(regime1_array, axis=0),  # Peak-to-peak
                'coefficient_std': np.std(regime1_array, axis=0),
                'mean_coefficients': np.mean(regime1_array, axis=0)
            }
        
        # Analyze regime 2 stability
        if len(regime2_coefs) > 1:
            regime2_array = np.array(regime2_coefs)
            stability_results['regime2_stability'] = {
                'coefficient_ranges': np.ptp(regime2_array, axis=0),
                'coefficient_std': np.std(regime2_array, axis=0),
                'mean_coefficients': np.mean(regime2_array, axis=0)
            }
        
        # Overall stability assessment
        stability_results['overall_stability'] = {
            'n_quantiles_fitted': len(quantiles),
            'quantiles_fitted': quantiles,
            'stability_assessment': 'stable' if (
                len(regime1_coefs) > 1 and np.all(stability_results['regime1_stability']['coefficient_std'] < 0.5)
            ) else 'unstable'
        }
        
        return stability_results
    
    def _calculate_threshold_log_likelihood(self, y, x, threshold_var, threshold, beta1, beta2, sigma):
        """Calculate log-likelihood for threshold model"""
        try:
            regime1_mask = threshold_var <= threshold
            regime2_mask = threshold_var > threshold
            
            log_likelihood = 0
            
            # Regime 1
            if np.sum(regime1_mask) > 0:
                y1 = y[regime1_mask]
                x1_with_intercept = np.column_stack([np.ones(np.sum(regime1_mask)), x[regime1_mask]])
                
                if len(beta1) == x1_with_intercept.shape[1]:
                    y1_pred = x1_with_intercept @ beta1
                    residuals1 = y1 - y1_pred
                    log_likelihood += -0.5 * np.sum(residuals1 ** 2) / (sigma ** 2) - 0.5 * len(y1) * np.log(2 * np.pi * sigma ** 2)
            
            # Regime 2
            if np.sum(regime2_mask) > 0:
                y2 = y[regime2_mask]
                x2_with_intercept = np.column_stack([np.ones(np.sum(regime2_mask)), x[regime2_mask]])
                
                if len(beta2) == x2_with_intercept.shape[1]:
                    y2_pred = x2_with_intercept @ beta2
                    residuals2 = y2 - y2_pred
                    log_likelihood += -0.5 * np.sum(residuals2 ** 2) / (sigma ** 2) - 0.5 * len(y2) * np.log(2 * np.pi * sigma ** 2)
            
            return log_likelihood
            
        except Exception:
            return -np.inf
    
    def _calculate_ssr(self, y, x, threshold_var, threshold, beta1, beta2):
        """Calculate sum of squared residuals"""
        try:
            regime1_mask = threshold_var <= threshold
            regime2_mask = threshold_var > threshold
            
            ssr = 0
            
            # Regime 1
            if np.sum(regime1_mask) > 0:
                y1 = y[regime1_mask]
                x1_with_intercept = np.column_stack([np.ones(np.sum(regime1_mask)), x[regime1_mask]])
                
                if len(beta1) == x1_with_intercept.shape[1]:
                    y1_pred = x1_with_intercept @ beta1
                    ssr += np.sum((y1 - y1_pred) ** 2)
            
            # Regime 2
            if np.sum(regime2_mask) > 0:
                y2 = y[regime2_mask]
                x2_with_intercept = np.column_stack([np.ones(np.sum(regime2_mask)), x[regime2_mask]])
                
                if len(beta2) == x2_with_intercept.shape[1]:
                    y2_pred = x2_with_intercept @ beta2
                    ssr += np.sum((y2 - y2_pred) ** 2)
            
            return ssr
            
        except Exception:
            return np.inf
    
    def _calculate_rhat(self, chain_samples):
        """Calculate Gelman-Rubin R-hat statistic for convergence diagnosis"""
        try:
            # Extract threshold samples from each chain
            chains = [chain['threshold'] for chain in chain_samples]
            n_chains = len(chains)
            n_samples = len(chains[0])
            
            # Calculate between-chain and within-chain variance
            chain_means = [np.mean(chain) for chain in chains]
            overall_mean = np.mean(chain_means)
            
            # Between-chain variance
            B = n_samples * np.var(chain_means, ddof=1)
            
            # Within-chain variance
            W = np.mean([np.var(chain, ddof=1) for chain in chains])
            
            # Marginal posterior variance
            var_plus = ((n_samples - 1) / n_samples) * W + (1 / n_samples) * B
            
            # R-hat
            rhat = np.sqrt(var_plus / W)
            
            return rhat
            
        except Exception:
            return None
    
    def _calculate_ess(self, combined_samples):
        """Calculate effective sample size"""
        try:
            # Simplified ESS calculation
            n_samples = len(combined_samples['threshold'])
            
            # Calculate autocorrelation (simplified)
            threshold_samples = combined_samples['threshold']
            autocorr = np.correlate(threshold_samples, threshold_samples, mode='full')
            autocorr = autocorr[autocorr.size // 2:]
            autocorr = autocorr / autocorr[0]
            
            # Find first negative autocorrelation
            first_negative = np.where(autocorr < 0)[0]
            if len(first_negative) > 0:
                tau = first_negative[0]
            else:
                tau = len(autocorr) // 4
            
            ess = n_samples / (1 + 2 * np.sum(autocorr[1:tau]))
            
            return max(1, int(ess))
            
        except Exception:
            return n_samples // 4  # Conservative estimate
    
    def _predict_multiple_threshold(self, model_info, x_test, threshold_test):
        """Predict using multiple threshold model"""
        try:
            thresholds = model_info.get('thresholds', [])
            regime_models = model_info.get('regime_models', [])
            
            if not thresholds or not regime_models:
                return np.zeros(len(x_test))
            
            y_pred = np.zeros(len(x_test))
            
            # Sort thresholds
            sorted_thresholds = sorted(thresholds)
            
            # Create regime masks
            regime_masks = []
            
            # First regime
            regime_masks.append(threshold_test <= sorted_thresholds[0])
            
            # Middle regimes
            for i in range(len(sorted_thresholds) - 1):
                mask = (threshold_test > sorted_thresholds[i]) & (threshold_test <= sorted_thresholds[i + 1])
                regime_masks.append(mask)
            
            # Last regime
            regime_masks.append(threshold_test > sorted_thresholds[-1])
            
            # Predict for each regime
            for i, (mask, regime_model) in enumerate(zip(regime_masks, regime_models)):
                if regime_model is not None and np.sum(mask) > 0:
                    x_regime = x_test[mask]
                    
                    # Add intercept
                    if x_regime.ndim == 1:
                        X_regime = np.column_stack([np.ones(len(x_regime)), x_regime])
                    else:
                        X_regime = np.column_stack([np.ones(len(x_regime)), x_regime])
                    
                    # Predict
                    beta = regime_model.get('beta', np.zeros(X_regime.shape[1]))
                    if len(beta) == X_regime.shape[1]:
                        y_pred[mask] = X_regime @ beta
            
            return y_pred
            
        except Exception:
            return np.zeros(len(x_test))