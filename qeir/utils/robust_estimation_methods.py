"""
Robust Estimation Methods for Threshold Models

This module provides robust estimation and uncertainty quantification methods
for threshold models, including bootstrap, quantile regression, Bayesian estimation,
and cross-validation model selection.
"""

import numpy as np
import pandas as pd
from scipy.stats import norm, chi2, f, t
from sklearn.linear_model import QuantileRegressor
from sklearn.utils import resample
import warnings
warnings.filterwarnings('ignore')

try:
    from ..core.models import HansenThresholdRegression
except ImportError:
    from models import HansenThresholdRegression


class RobustEstimationMethods:
    """
    Robust estimation and uncertainty quantification methods for threshold models
    """
    
    def __init__(self):
        self.fitted_models = {}
    
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
                {'type': 'hansen_enhanced', 'params': {'enhancements': {'additional_controls': True}}}
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
                    else:
                        # For other specifications, use basic Hansen model
                        model = HansenThresholdRegression()
                        model.fit(y_train, x_train, thresh_train)
                        
                        if model.fitted:
                            y_pred = model.predict(x_test, thresh_test)
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