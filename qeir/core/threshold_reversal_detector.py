"""
Threshold Reversal Detection and Validation

This module implements statistical tests for threshold significance, visualization methods
for regime-switching behavior, and robustness tests across different threshold specifications
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import f, chi2, norm
import statsmodels.api as sm
from statsmodels.regression.linear_model import OLS
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.tsa.stattools import adfuller, kpss
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
import logging
import warnings

warnings.filterwarnings('ignore')


@dataclass
class ThresholdTestResults:
    """Data structure for threshold significance test results"""
    
    # Basic threshold information
    threshold_value: float
    threshold_variable_name: str
    
    # Statistical tests
    sup_wald_test: Optional[Dict[str, Any]] = None
    likelihood_ratio_test: Optional[Dict[str, Any]] = None
    bootstrap_test: Optional[Dict[str, Any]] = None
    
    # Regime information
    regime1_observations: int = 0
    regime2_observations: int = 0
    regime1_statistics: Optional[Dict[str, Any]] = None
    regime2_statistics: Optional[Dict[str, Any]] = None
    
    # Model comparison
    linear_model_results: Optional[Dict[str, Any]] = None
    threshold_model_results: Optional[Dict[str, Any]] = None
    model_comparison: Optional[Dict[str, Any]] = None
    
    # Significance assessment
    threshold_significant: bool = False
    significance_level: float = 0.05
    test_summary: Optional[Dict[str, str]] = None


class ThresholdSignificanceTests:
    """
    Statistical tests for threshold significance.
    
    Implements various tests to determine if an identified threshold is statistically significant,
    including sup-Wald tests, likelihood ratio tests, and bootstrap procedures.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
    def test_threshold_significance(self,
                                  y: np.ndarray,
                                  X: np.ndarray,
                                  threshold_var: np.ndarray,
                                  threshold_value: float,
                                  test_types: List[str] = ['sup_wald', 'likelihood_ratio', 'bootstrap'],
                                  significance_level: float = 0.05,
                                  n_bootstrap: int = 1000) -> ThresholdTestResults:
        """
        Comprehensive threshold significance testing.
        
        Args:
            y: Dependent variable
            X: Independent variables matrix
            threshold_var: Threshold variable
            threshold_value: Estimated threshold value
            test_types: Types of tests to perform
            significance_level: Significance level for tests
            n_bootstrap: Number of bootstrap iterations
            
        Returns:
            ThresholdTestResults object with comprehensive test results
        """
        results = ThresholdTestResults(
            threshold_value=threshold_value,
            threshold_variable_name="threshold_variable",
            significance_level=significance_level
        )
        
        # Create regime indicators
        regime1_mask = threshold_var <= threshold_value
        regime2_mask = threshold_var > threshold_value
        
        results.regime1_observations = int(np.sum(regime1_mask))
        results.regime2_observations = int(np.sum(regime2_mask))
        
        # Calculate regime-specific statistics
        results.regime1_statistics = self._calculate_regime_statistics(y[regime1_mask], X[regime1_mask])
        results.regime2_statistics = self._calculate_regime_statistics(y[regime2_mask], X[regime2_mask])
        
        # Fit linear and threshold models for comparison
        results.linear_model_results = self._fit_linear_model(y, X)
        results.threshold_model_results = self._fit_threshold_model(y, X, threshold_var, threshold_value)
        
        # Run significance tests
        if 'sup_wald' in test_types:
            results.sup_wald_test = self._sup_wald_test(y, X, threshold_var, threshold_value)
        
        if 'likelihood_ratio' in test_types:
            results.likelihood_ratio_test = self._likelihood_ratio_test(
                results.linear_model_results, results.threshold_model_results
            )
        
        if 'bootstrap' in test_types:
            results.bootstrap_test = self._bootstrap_threshold_test(
                y, X, threshold_var, threshold_value, n_bootstrap
            )
        
        # Model comparison
        results.model_comparison = self._compare_models(
            results.linear_model_results, results.threshold_model_results
        )
        
        # Overall significance assessment
        results.threshold_significant, results.test_summary = self._assess_overall_significance(
            results, significance_level
        )
        
        return results
    
    def _calculate_regime_statistics(self, y_regime: np.ndarray, X_regime: np.ndarray) -> Dict[str, Any]:
        """Calculate statistics for a specific regime"""
        
        if len(y_regime) == 0:
            return {'error': 'No observations in regime'}
        
        stats_dict = {
            'observations': len(y_regime),
            'y_mean': float(np.mean(y_regime)),
            'y_std': float(np.std(y_regime)),
            'y_min': float(np.min(y_regime)),
            'y_max': float(np.max(y_regime))
        }
        
        # Fit regression for this regime
        if len(y_regime) > X_regime.shape[1] + 1:  # Need more obs than parameters
            try:
                X_regime_const = sm.add_constant(X_regime)
                model = OLS(y_regime, X_regime_const).fit()
                
                stats_dict.update({
                    'r_squared': model.rsquared,
                    'adj_r_squared': model.rsquared_adj,
                    'f_statistic': model.fvalue,
                    'f_pvalue': model.f_pvalue,
                    'coefficients': model.params.tolist(),
                    'std_errors': model.bse.tolist(),
                    'pvalues': model.pvalues.tolist()
                })
                
            except Exception as e:
                stats_dict['regression_error'] = str(e)
        else:
            stats_dict['regression_error'] = 'Insufficient observations for regression'
        
        return stats_dict
    
    def _fit_linear_model(self, y: np.ndarray, X: np.ndarray) -> Dict[str, Any]:
        """Fit linear model without threshold"""
        
        try:
            X_const = sm.add_constant(X)
            model = OLS(y, X_const).fit()
            
            return {
                'fitted': True,
                'r_squared': model.rsquared,
                'adj_r_squared': model.rsquared_adj,
                'aic': model.aic,
                'bic': model.bic,
                'log_likelihood': model.llf,
                'f_statistic': model.fvalue,
                'f_pvalue': model.f_pvalue,
                'coefficients': model.params.tolist(),
                'std_errors': model.bse.tolist(),
                'pvalues': model.pvalues.tolist(),
                'residual_sum_squares': np.sum(model.resid**2),
                'n_parameters': len(model.params)
            }
            
        except Exception as e:
            return {'fitted': False, 'error': str(e)}
    
    def _fit_threshold_model(self, 
                           y: np.ndarray, 
                           X: np.ndarray, 
                           threshold_var: np.ndarray, 
                           threshold_value: float) -> Dict[str, Any]:
        """Fit threshold model"""
        
        try:
            regime1_mask = threshold_var <= threshold_value
            regime2_mask = threshold_var > threshold_value
            
            # Fit regime 1
            if np.sum(regime1_mask) > X.shape[1] + 1:
                X1_const = sm.add_constant(X[regime1_mask])
                model1 = OLS(y[regime1_mask], X1_const).fit()
                regime1_results = {
                    'fitted': True,
                    'coefficients': model1.params.tolist(),
                    'std_errors': model1.bse.tolist(),
                    'r_squared': model1.rsquared,
                    'residual_sum_squares': np.sum(model1.resid**2),
                    'log_likelihood': model1.llf
                }
            else:
                regime1_results = {'fitted': False, 'error': 'Insufficient observations'}
            
            # Fit regime 2
            if np.sum(regime2_mask) > X.shape[1] + 1:
                X2_const = sm.add_constant(X[regime2_mask])
                model2 = OLS(y[regime2_mask], X2_const).fit()
                regime2_results = {
                    'fitted': True,
                    'coefficients': model2.params.tolist(),
                    'std_errors': model2.bse.tolist(),
                    'r_squared': model2.rsquared,
                    'residual_sum_squares': np.sum(model2.resid**2),
                    'log_likelihood': model2.llf
                }
            else:
                regime2_results = {'fitted': False, 'error': 'Insufficient observations'}
            
            # Combined results
            if regime1_results['fitted'] and regime2_results['fitted']:
                total_rss = regime1_results['residual_sum_squares'] + regime2_results['residual_sum_squares']
                total_ll = regime1_results['log_likelihood'] + regime2_results['log_likelihood']
                n_params = len(regime1_results['coefficients']) + len(regime2_results['coefficients'])
                
                # Calculate AIC and BIC for threshold model
                n_obs = len(y)
                aic = 2 * n_params - 2 * total_ll
                bic = np.log(n_obs) * n_params - 2 * total_ll
                
                return {
                    'fitted': True,
                    'regime1': regime1_results,
                    'regime2': regime2_results,
                    'total_residual_sum_squares': total_rss,
                    'total_log_likelihood': total_ll,
                    'n_parameters': n_params,
                    'aic': aic,
                    'bic': bic,
                    'threshold_value': threshold_value
                }
            else:
                return {
                    'fitted': False,
                    'regime1': regime1_results,
                    'regime2': regime2_results,
                    'error': 'One or both regimes could not be fitted'
                }
                
        except Exception as e:
            return {'fitted': False, 'error': str(e)}
    
    def _sup_wald_test(self, 
                      y: np.ndarray, 
                      X: np.ndarray, 
                      threshold_var: np.ndarray, 
                      threshold_value: float) -> Dict[str, Any]:
        """
        Supremum Wald test for threshold significance.
        
        Tests the null hypothesis of no threshold against the alternative of a threshold.
        """
        try:
            # Create regime indicators
            regime1_mask = threshold_var <= threshold_value
            regime2_mask = threshold_var > threshold_value
            
            # Check if we have sufficient observations in each regime
            if np.sum(regime1_mask) < X.shape[1] + 2 or np.sum(regime2_mask) < X.shape[1] + 2:
                return {'error': 'Insufficient observations in one or both regimes for sup-Wald test'}
            
            # Fit unrestricted model (with threshold)
            X1_const = sm.add_constant(X[regime1_mask])
            X2_const = sm.add_constant(X[regime2_mask])
            
            model1 = OLS(y[regime1_mask], X1_const).fit()
            model2 = OLS(y[regime2_mask], X2_const).fit()
            
            # Fit restricted model (without threshold)
            X_const = sm.add_constant(X)
            model_restricted = OLS(y, X_const).fit()
            
            # Calculate Wald statistic for coefficient equality
            # H0: β1 = β2 (coefficients are equal across regimes)
            
            beta1 = model1.params.values
            beta2 = model2.params.values
            
            # Covariance matrices
            cov1 = model1.cov_params().values
            cov2 = model2.cov_params().values
            
            # Difference in coefficients
            beta_diff = beta1 - beta2
            
            # Combined covariance matrix (assuming independence)
            cov_combined = cov1 + cov2
            
            # Wald statistic
            try:
                wald_stat = beta_diff.T @ np.linalg.inv(cov_combined) @ beta_diff
                df = len(beta1)  # Degrees of freedom
                p_value = 1 - chi2.cdf(wald_stat, df)
                
                return {
                    'statistic': float(wald_stat),
                    'degrees_freedom': df,
                    'p_value': float(p_value),
                    'critical_value_5pct': chi2.ppf(0.95, df),
                    'significant_5pct': p_value < 0.05,
                    'significant_1pct': p_value < 0.01,
                    'test_type': 'sup_wald'
                }
                
            except np.linalg.LinAlgError:
                # Fallback: use F-test approach
                rss_restricted = np.sum(model_restricted.resid**2)
                rss_unrestricted = np.sum(model1.resid**2) + np.sum(model2.resid**2)
                
                n = len(y)
                k = len(beta1)
                
                f_stat = ((rss_restricted - rss_unrestricted) / k) / (rss_unrestricted / (n - 2*k))
                p_value = 1 - f.cdf(f_stat, k, n - 2*k)
                
                return {
                    'statistic': float(f_stat),
                    'degrees_freedom': (k, n - 2*k),
                    'p_value': float(p_value),
                    'critical_value_5pct': f.ppf(0.95, k, n - 2*k),
                    'significant_5pct': p_value < 0.05,
                    'significant_1pct': p_value < 0.01,
                    'test_type': 'sup_wald_f_version'
                }
                
        except Exception as e:
            return {'error': str(e), 'test_type': 'sup_wald'}
    
    def _likelihood_ratio_test(self, 
                             linear_results: Dict[str, Any], 
                             threshold_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Likelihood ratio test comparing linear vs threshold model.
        """
        if not linear_results.get('fitted', False) or not threshold_results.get('fitted', False):
            return {'error': 'One or both models not fitted properly'}
        
        try:
            # Log-likelihoods
            ll_restricted = linear_results['log_likelihood']
            ll_unrestricted = threshold_results['total_log_likelihood']
            
            # Degrees of freedom (difference in number of parameters)
            df = threshold_results['n_parameters'] - linear_results['n_parameters']
            
            # LR statistic
            lr_stat = 2 * (ll_unrestricted - ll_restricted)
            
            # P-value
            p_value = 1 - chi2.cdf(lr_stat, df)
            
            return {
                'statistic': float(lr_stat),
                'degrees_freedom': df,
                'p_value': float(p_value),
                'critical_value_5pct': chi2.ppf(0.95, df),
                'significant_5pct': p_value < 0.05,
                'significant_1pct': p_value < 0.01,
                'log_likelihood_restricted': ll_restricted,
                'log_likelihood_unrestricted': ll_unrestricted,
                'test_type': 'likelihood_ratio'
            }
            
        except Exception as e:
            return {'error': str(e), 'test_type': 'likelihood_ratio'}
    
    def _bootstrap_threshold_test(self, 
                                y: np.ndarray, 
                                X: np.ndarray, 
                                threshold_var: np.ndarray, 
                                threshold_value: float, 
                                n_bootstrap: int) -> Dict[str, Any]:
        """
        Bootstrap test for threshold significance.
        """
        try:
            from sklearn.utils import resample
            
            # Original test statistic (using RSS difference)
            linear_model = self._fit_linear_model(y, X)
            threshold_model = self._fit_threshold_model(y, X, threshold_var, threshold_value)
            
            if not linear_model['fitted'] or not threshold_model['fitted']:
                return {'error': 'Could not fit models for bootstrap test'}
            
            original_stat = (linear_model['residual_sum_squares'] - 
                           threshold_model['total_residual_sum_squares'])
            
            # Bootstrap under null hypothesis (linear model)
            X_const = sm.add_constant(X)
            linear_ols = OLS(y, X_const).fit()
            
            bootstrap_stats = []
            
            for i in range(n_bootstrap):
                # Generate bootstrap sample under null
                y_boot = linear_ols.fittedvalues + resample(linear_ols.resid, n_samples=len(y), random_state=i)
                
                # Fit models on bootstrap sample
                try:
                    boot_linear = self._fit_linear_model(y_boot, X)
                    boot_threshold = self._fit_threshold_model(y_boot, X, threshold_var, threshold_value)
                    
                    if boot_linear['fitted'] and boot_threshold['fitted']:
                        boot_stat = (boot_linear['residual_sum_squares'] - 
                                   boot_threshold['total_residual_sum_squares'])
                        bootstrap_stats.append(boot_stat)
                        
                except Exception:
                    continue
            
            if len(bootstrap_stats) < 50:
                return {'error': f'Too few successful bootstrap iterations: {len(bootstrap_stats)}'}
            
            # Calculate p-value
            p_value = np.mean(np.array(bootstrap_stats) >= original_stat)
            
            return {
                'statistic': float(original_stat),
                'p_value': float(p_value),
                'bootstrap_iterations': len(bootstrap_stats),
                'bootstrap_stats_mean': float(np.mean(bootstrap_stats)),
                'bootstrap_stats_std': float(np.std(bootstrap_stats)),
                'significant_5pct': p_value < 0.05,
                'significant_1pct': p_value < 0.01,
                'test_type': 'bootstrap'
            }
            
        except Exception as e:
            return {'error': str(e), 'test_type': 'bootstrap'}
    
    def _compare_models(self, 
                       linear_results: Dict[str, Any], 
                       threshold_results: Dict[str, Any]) -> Dict[str, Any]:
        """Compare linear vs threshold model performance"""
        
        if not linear_results.get('fitted', False) or not threshold_results.get('fitted', False):
            return {'error': 'One or both models not fitted properly'}
        
        comparison = {
            'linear_model': {
                'aic': linear_results['aic'],
                'bic': linear_results['bic'],
                'r_squared': linear_results.get('r_squared', np.nan),
                'log_likelihood': linear_results['log_likelihood']
            },
            'threshold_model': {
                'aic': threshold_results['aic'],
                'bic': threshold_results['bic'],
                'log_likelihood': threshold_results['total_log_likelihood']
            },
            'model_selection': {}
        }
        
        # Model selection criteria
        comparison['model_selection']['aic_prefers_threshold'] = threshold_results['aic'] < linear_results['aic']
        comparison['model_selection']['bic_prefers_threshold'] = threshold_results['bic'] < linear_results['bic']
        comparison['model_selection']['ll_improvement'] = (threshold_results['total_log_likelihood'] - 
                                                         linear_results['log_likelihood'])
        
        # Overall recommendation
        aic_threshold = comparison['model_selection']['aic_prefers_threshold']
        bic_threshold = comparison['model_selection']['bic_prefers_threshold']
        
        if aic_threshold and bic_threshold:
            comparison['model_selection']['recommendation'] = 'threshold_model'
        elif not aic_threshold and not bic_threshold:
            comparison['model_selection']['recommendation'] = 'linear_model'
        else:
            comparison['model_selection']['recommendation'] = 'mixed_evidence'
        
        return comparison
    
    def _assess_overall_significance(self, 
                                   results: ThresholdTestResults, 
                                   significance_level: float) -> Tuple[bool, Dict[str, str]]:
        """Assess overall threshold significance based on all tests"""
        
        test_results = []
        test_summary = {}
        
        # Sup-Wald test
        if results.sup_wald_test and 'p_value' in results.sup_wald_test:
            sup_wald_sig = results.sup_wald_test['p_value'] < significance_level
            test_results.append(sup_wald_sig)
            test_summary['sup_wald'] = f"{'Significant' if sup_wald_sig else 'Not significant'} (p={results.sup_wald_test['p_value']:.4f})"
        
        # Likelihood ratio test
        if results.likelihood_ratio_test and 'p_value' in results.likelihood_ratio_test:
            lr_sig = results.likelihood_ratio_test['p_value'] < significance_level
            test_results.append(lr_sig)
            test_summary['likelihood_ratio'] = f"{'Significant' if lr_sig else 'Not significant'} (p={results.likelihood_ratio_test['p_value']:.4f})"
        
        # Bootstrap test
        if results.bootstrap_test and 'p_value' in results.bootstrap_test:
            bootstrap_sig = results.bootstrap_test['p_value'] < significance_level
            test_results.append(bootstrap_sig)
            test_summary['bootstrap'] = f"{'Significant' if bootstrap_sig else 'Not significant'} (p={results.bootstrap_test['p_value']:.4f})"
        
        # Model comparison
        if results.model_comparison and 'model_selection' in results.model_comparison:
            model_prefers_threshold = results.model_comparison['model_selection']['recommendation'] == 'threshold_model'
            test_summary['model_selection'] = f"Model selection {'favors' if model_prefers_threshold else 'does not favor'} threshold model"
        
        # Overall assessment
        if len(test_results) == 0:
            overall_significant = False
            test_summary['overall'] = "No valid tests completed"
        else:
            # Threshold is significant if majority of tests are significant
            significant_count = sum(test_results)
            total_tests = len(test_results)
            overall_significant = significant_count > total_tests / 2
            
            test_summary['overall'] = f"Threshold {'is' if overall_significant else 'is not'} significant ({significant_count}/{total_tests} tests significant)"
        
        return overall_significant, test_summary


class RegimeSwitchingVisualizer:
    """
    Visualization methods for regime-switching behavior.
    
    Creates publication-quality plots showing threshold effects, regime transitions,
    and model diagnostics.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
    def plot_threshold_effects(self,
                             y: pd.Series,
                             threshold_var: pd.Series,
                             threshold_value: float,
                             central_bank_reaction: Optional[pd.Series] = None,
                             confidence_effects: Optional[pd.Series] = None,
                             save_path: Optional[str] = None,
                             figsize: Tuple[int, int] = (15, 10)) -> plt.Figure:
        """
        Create comprehensive threshold effects visualization.
        
        Args:
            y: Dependent variable (e.g., long-term yields)
            threshold_var: Threshold variable (e.g., debt service burden)
            threshold_value: Estimated threshold value
            central_bank_reaction: Central bank reaction strength (γ₁)
            confidence_effects: Confidence effects (λ₂)
            save_path: Path to save the figure
            figsize: Figure size
            
        Returns:
            matplotlib Figure object
        """
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        fig.suptitle('Threshold Effects Analysis', fontsize=16, fontweight='bold')
        
        # Align all series
        common_index = y.index.intersection(threshold_var.index)
        if central_bank_reaction is not None:
            common_index = common_index.intersection(central_bank_reaction.index)
        if confidence_effects is not None:
            common_index = common_index.intersection(confidence_effects.index)
        
        y_aligned = y.loc[common_index]
        threshold_aligned = threshold_var.loc[common_index]
        
        # Create regime indicators
        regime1_mask = threshold_aligned <= threshold_value
        regime2_mask = threshold_aligned > threshold_value
        
        # Plot 1: Time series with regime coloring
        ax1 = axes[0, 0]
        ax1.plot(common_index[regime1_mask], y_aligned[regime1_mask], 
                'o', color='blue', alpha=0.6, label=f'Regime 1 (≤{threshold_value:.3f})', markersize=3)
        ax1.plot(common_index[regime2_mask], y_aligned[regime2_mask], 
                'o', color='red', alpha=0.6, label=f'Regime 2 (>{threshold_value:.3f})', markersize=3)
        ax1.set_title('Dependent Variable by Regime')
        ax1.set_ylabel('Dependent Variable')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Scatter plot of dependent variable vs threshold variable
        ax2 = axes[0, 1]
        ax2.scatter(threshold_aligned[regime1_mask], y_aligned[regime1_mask], 
                   color='blue', alpha=0.6, label='Regime 1', s=20)
        ax2.scatter(threshold_aligned[regime2_mask], y_aligned[regime2_mask], 
                   color='red', alpha=0.6, label='Regime 2', s=20)
        ax2.axvline(x=threshold_value, color='black', linestyle='--', linewidth=2, label='Threshold')
        ax2.set_xlabel('Threshold Variable')
        ax2.set_ylabel('Dependent Variable')
        ax2.set_title('Threshold Relationship')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Regime probabilities over time (if available)
        ax3 = axes[1, 0]
        regime_indicator = regime2_mask.astype(int)
        ax3.fill_between(common_index, 0, regime_indicator, alpha=0.3, color='red', label='Regime 2')
        ax3.fill_between(common_index, regime_indicator, 1, alpha=0.3, color='blue', label='Regime 1')
        ax3.set_title('Regime Classification Over Time')
        ax3.set_ylabel('Regime Probability')
        ax3.set_ylim(0, 1)
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Distribution of threshold variable with threshold line
        ax4 = axes[1, 1]
        ax4.hist(threshold_aligned, bins=30, alpha=0.7, color='gray', edgecolor='black')
        ax4.axvline(x=threshold_value, color='red', linestyle='--', linewidth=2, label='Threshold')
        ax4.set_xlabel('Threshold Variable')
        ax4.set_ylabel('Frequency')
        ax4.set_title('Distribution of Threshold Variable')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"Threshold effects plot saved to {save_path}")
        
        return fig
    
    def plot_interaction_effects(self,
                               y: pd.Series,
                               central_bank_reaction: pd.Series,
                               confidence_effects: pd.Series,
                               threshold_var: pd.Series,
                               threshold_value: float,
                               save_path: Optional[str] = None,
                               figsize: Tuple[int, int] = (15, 8)) -> plt.Figure:
        """
        Visualize interaction effects between central bank reaction and confidence effects.
        """
        fig, axes = plt.subplots(1, 3, figsize=figsize)
        fig.suptitle('Interaction Effects Analysis', fontsize=16, fontweight='bold')
        
        # Align all series
        common_index = y.index.intersection(central_bank_reaction.index)\
                              .intersection(confidence_effects.index)\
                              .intersection(threshold_var.index)
        
        y_aligned = y.loc[common_index]
        gamma1_aligned = central_bank_reaction.loc[common_index]
        lambda2_aligned = confidence_effects.loc[common_index]
        threshold_aligned = threshold_var.loc[common_index]
        
        # Create regime indicators
        regime1_mask = threshold_aligned <= threshold_value
        regime2_mask = threshold_aligned > threshold_value
        
        # Create interaction term
        interaction = gamma1_aligned * lambda2_aligned
        
        # Plot 1: 3D scatter plot (projected to 2D)
        ax1 = axes[0]
        scatter1 = ax1.scatter(gamma1_aligned[regime1_mask], lambda2_aligned[regime1_mask], 
                              c=y_aligned[regime1_mask], cmap='Blues', alpha=0.6, s=30, label='Regime 1')
        scatter2 = ax1.scatter(gamma1_aligned[regime2_mask], lambda2_aligned[regime2_mask], 
                              c=y_aligned[regime2_mask], cmap='Reds', alpha=0.6, s=30, label='Regime 2')
        ax1.set_xlabel('Central Bank Reaction (γ₁)')
        ax1.set_ylabel('Confidence Effects (λ₂)')
        ax1.set_title('Interaction Space by Regime')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Interaction term vs dependent variable
        ax2 = axes[1]
        ax2.scatter(interaction[regime1_mask], y_aligned[regime1_mask], 
                   color='blue', alpha=0.6, label='Regime 1', s=20)
        ax2.scatter(interaction[regime2_mask], y_aligned[regime2_mask], 
                   color='red', alpha=0.6, label='Regime 2', s=20)
        ax2.set_xlabel('Interaction Term (γ₁ × λ₂)')
        ax2.set_ylabel('Dependent Variable')
        ax2.set_title('Interaction Effects')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Marginal effects
        ax3 = axes[2]
        
        # Calculate marginal effects (simplified)
        gamma1_bins = pd.qcut(gamma1_aligned, q=5, duplicates='drop')
        lambda2_bins = pd.qcut(lambda2_aligned, q=5, duplicates='drop')
        
        # Create heatmap data
        marginal_effects = pd.DataFrame(index=gamma1_bins.cat.categories, 
                                      columns=lambda2_bins.cat.categories)
        
        for i, gamma_bin in enumerate(gamma1_bins.cat.categories):
            for j, lambda_bin in enumerate(lambda2_bins.cat.categories):
                mask = (gamma1_bins == gamma_bin) & (lambda2_bins == lambda_bin)
                if mask.sum() > 0:
                    marginal_effects.iloc[i, j] = y_aligned[mask].mean()
        
        # Plot heatmap
        im = ax3.imshow(marginal_effects.values.astype(float), cmap='RdYlBu', aspect='auto')
        ax3.set_title('Marginal Effects Heatmap')
        ax3.set_xlabel('Confidence Effects Quintiles')
        ax3.set_ylabel('Central Bank Reaction Quintiles')
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax3)
        cbar.set_label('Mean Dependent Variable')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"Interaction effects plot saved to {save_path}")
        
        return fig
    
    def plot_robustness_analysis(self,
                               robustness_results: Dict[str, Any],
                               save_path: Optional[str] = None,
                               figsize: Tuple[int, int] = (15, 10)) -> plt.Figure:
        """
        Visualize robustness test results across different specifications.
        """
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        fig.suptitle('Robustness Analysis', fontsize=16, fontweight='bold')
        
        # Extract threshold estimates from different specifications
        threshold_estimates = []
        specification_names = []
        
        for spec_name, spec_results in robustness_results.items():
            if isinstance(spec_results, dict) and 'threshold_value' in spec_results:
                threshold_estimates.append(spec_results['threshold_value'])
                specification_names.append(spec_name)
        
        if len(threshold_estimates) > 0:
            # Plot 1: Threshold estimates across specifications
            ax1 = axes[0, 0]
            bars = ax1.bar(range(len(threshold_estimates)), threshold_estimates, 
                          color='skyblue', edgecolor='navy', alpha=0.7)
            ax1.set_xlabel('Specification')
            ax1.set_ylabel('Threshold Estimate')
            ax1.set_title('Threshold Estimates Across Specifications')
            ax1.set_xticks(range(len(specification_names)))
            ax1.set_xticklabels(specification_names, rotation=45, ha='right')
            ax1.grid(True, alpha=0.3)
            
            # Add value labels on bars
            for i, bar in enumerate(bars):
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.3f}', ha='center', va='bottom')
        
        # Plot 2: Distribution of threshold estimates
        if len(threshold_estimates) > 1:
            ax2 = axes[0, 1]
            ax2.hist(threshold_estimates, bins=min(10, len(threshold_estimates)), 
                    alpha=0.7, color='lightgreen', edgecolor='darkgreen')
            ax2.axvline(np.mean(threshold_estimates), color='red', linestyle='--', 
                       label=f'Mean: {np.mean(threshold_estimates):.3f}')
            ax2.axvline(np.median(threshold_estimates), color='blue', linestyle='--', 
                       label=f'Median: {np.median(threshold_estimates):.3f}')
            ax2.set_xlabel('Threshold Value')
            ax2.set_ylabel('Frequency')
            ax2.set_title('Distribution of Threshold Estimates')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
        
        # Plot 3: Model fit comparison (R-squared)
        r_squared_values = []
        for spec_name, spec_results in robustness_results.items():
            if isinstance(spec_results, dict) and 'r_squared' in spec_results:
                r_squared_values.append(spec_results['r_squared'])
            else:
                r_squared_values.append(np.nan)
        
        if len(r_squared_values) > 0 and not all(np.isnan(r_squared_values)):
            ax3 = axes[1, 0]
            valid_indices = [i for i, val in enumerate(r_squared_values) if not np.isnan(val)]
            valid_r_squared = [r_squared_values[i] for i in valid_indices]
            valid_names = [specification_names[i] for i in valid_indices]
            
            bars = ax3.bar(range(len(valid_r_squared)), valid_r_squared, 
                          color='orange', edgecolor='darkorange', alpha=0.7)
            ax3.set_xlabel('Specification')
            ax3.set_ylabel('R-squared')
            ax3.set_title('Model Fit Across Specifications')
            ax3.set_xticks(range(len(valid_names)))
            ax3.set_xticklabels(valid_names, rotation=45, ha='right')
            ax3.grid(True, alpha=0.3)
            
            # Add value labels
            for i, bar in enumerate(bars):
                height = bar.get_height()
                ax3.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.3f}', ha='center', va='bottom')
        
        # Plot 4: Significance test results
        significance_results = []
        for spec_name, spec_results in robustness_results.items():
            if isinstance(spec_results, dict) and 'threshold_significant' in spec_results:
                significance_results.append(1 if spec_results['threshold_significant'] else 0)
            else:
                significance_results.append(0)
        
        if len(significance_results) > 0:
            ax4 = axes[1, 1]
            colors = ['red' if sig == 0 else 'green' for sig in significance_results]
            bars = ax4.bar(range(len(significance_results)), significance_results, 
                          color=colors, alpha=0.7)
            ax4.set_xlabel('Specification')
            ax4.set_ylabel('Threshold Significant')
            ax4.set_title('Threshold Significance Across Specifications')
            ax4.set_xticks(range(len(specification_names)))
            ax4.set_xticklabels(specification_names, rotation=45, ha='right')
            ax4.set_ylim(0, 1.1)
            ax4.grid(True, alpha=0.3)
            
            # Add percentage of significant results
            pct_significant = np.mean(significance_results) * 100
            ax4.text(0.02, 0.98, f'{pct_significant:.1f}% significant', 
                    transform=ax4.transAxes, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"Robustness analysis plot saved to {save_path}")
        
        return fig


class RobustnessTestSuite:
    """
    Comprehensive robustness testing across different threshold specifications.
    
    Tests threshold stability across different:
    - Time periods
    - Variable specifications
    - Estimation methods
    - Trimming parameters
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.threshold_tester = ThresholdSignificanceTests()
        
    def run_comprehensive_robustness_tests(self,
                                         y: pd.Series,
                                         central_bank_reaction: pd.Series,
                                         confidence_effects: pd.Series,
                                         threshold_var: pd.Series,
                                         debt_service_burden: pd.Series,
                                         baseline_threshold: float) -> Dict[str, Any]:
        """
        Run comprehensive robustness tests across different specifications.
        
        Args:
            y: Dependent variable
            central_bank_reaction: Central bank reaction strength (γ₁)
            confidence_effects: Confidence effects (λ₂)
            threshold_var: Primary threshold variable
            debt_service_burden: Alternative threshold variable
            baseline_threshold: Baseline threshold estimate
            
        Returns:
            Dictionary with comprehensive robustness test results
        """
        robustness_results = {
            'baseline': {
                'threshold_value': baseline_threshold,
                'specification': 'baseline'
            }
        }
        
        # Align all data
        common_index = y.index.intersection(central_bank_reaction.index)\
                              .intersection(confidence_effects.index)\
                              .intersection(threshold_var.index)\
                              .intersection(debt_service_burden.index)
        
        y_aligned = y.loc[common_index].values
        gamma1_aligned = central_bank_reaction.loc[common_index].values
        lambda2_aligned = confidence_effects.loc[common_index].values
        threshold_aligned = threshold_var.loc[common_index].values
        debt_service_aligned = debt_service_burden.loc[common_index].values
        
        # Create interaction term
        interaction = gamma1_aligned * lambda2_aligned
        X_baseline = np.column_stack([gamma1_aligned, lambda2_aligned, interaction])
        
        # Test 1: Alternative trimming parameters
        self.logger.info("Testing alternative trimming parameters")
        trim_values = [0.10, 0.15, 0.20, 0.25, 0.30]
        
        for trim in trim_values:
            try:
                from .enhanced_hypothesis1 import EnhancedHansenThresholdRegression
                
                alt_model = EnhancedHansenThresholdRegression()
                alt_model.fit(y_aligned, X_baseline, threshold_aligned, trim=trim)
                
                if alt_model.fitted:
                    # Test significance
                    test_results = self.threshold_tester.test_threshold_significance(
                        y_aligned, X_baseline, threshold_aligned, alt_model.threshold
                    )
                    
                    robustness_results[f'trim_{trim}'] = {
                        'threshold_value': alt_model.threshold,
                        'threshold_significant': test_results.threshold_significant,
                        'specification': f'trim_{trim}',
                        'r_squared': self._calculate_threshold_r_squared(
                            y_aligned, X_baseline, threshold_aligned, alt_model.threshold
                        )
                    }
                    
            except Exception as e:
                robustness_results[f'trim_{trim}'] = {
                    'error': str(e),
                    'specification': f'trim_{trim}'
                }
        
        # Test 2: Alternative threshold variables
        self.logger.info("Testing alternative threshold variables")
        
        # Use debt service burden as threshold variable
        try:
            from .enhanced_hypothesis1 import EnhancedHansenThresholdRegression
            
            alt_model_debt = EnhancedHansenThresholdRegression()
            alt_model_debt.fit(y_aligned, X_baseline, debt_service_aligned, trim=0.15)
            
            if alt_model_debt.fitted:
                test_results = self.threshold_tester.test_threshold_significance(
                    y_aligned, X_baseline, debt_service_aligned, alt_model_debt.threshold
                )
                
                robustness_results['debt_service_threshold'] = {
                    'threshold_value': alt_model_debt.threshold,
                    'threshold_significant': test_results.threshold_significant,
                    'specification': 'debt_service_threshold',
                    'r_squared': self._calculate_threshold_r_squared(
                        y_aligned, X_baseline, debt_service_aligned, alt_model_debt.threshold
                    )
                }
                
        except Exception as e:
            robustness_results['debt_service_threshold'] = {
                'error': str(e),
                'specification': 'debt_service_threshold'
            }
        
        # Use confidence effects as threshold variable
        try:
            alt_model_conf = EnhancedHansenThresholdRegression()
            alt_model_conf.fit(y_aligned, X_baseline, lambda2_aligned, trim=0.15)
            
            if alt_model_conf.fitted:
                test_results = self.threshold_tester.test_threshold_significance(
                    y_aligned, X_baseline, lambda2_aligned, alt_model_conf.threshold
                )
                
                robustness_results['confidence_threshold'] = {
                    'threshold_value': alt_model_conf.threshold,
                    'threshold_significant': test_results.threshold_significant,
                    'specification': 'confidence_threshold',
                    'r_squared': self._calculate_threshold_r_squared(
                        y_aligned, X_baseline, lambda2_aligned, alt_model_conf.threshold
                    )
                }
                
        except Exception as e:
            robustness_results['confidence_threshold'] = {
                'error': str(e),
                'specification': 'confidence_threshold'
            }
        
        # Test 3: Alternative variable specifications
        self.logger.info("Testing alternative variable specifications")
        
        # Without interaction term
        try:
            X_no_interaction = np.column_stack([gamma1_aligned, lambda2_aligned])
            
            alt_model_no_int = EnhancedHansenThresholdRegression()
            alt_model_no_int.fit(y_aligned, X_no_interaction, threshold_aligned, trim=0.15)
            
            if alt_model_no_int.fitted:
                test_results = self.threshold_tester.test_threshold_significance(
                    y_aligned, X_no_interaction, threshold_aligned, alt_model_no_int.threshold
                )
                
                robustness_results['no_interaction'] = {
                    'threshold_value': alt_model_no_int.threshold,
                    'threshold_significant': test_results.threshold_significant,
                    'specification': 'no_interaction',
                    'r_squared': self._calculate_threshold_r_squared(
                        y_aligned, X_no_interaction, threshold_aligned, alt_model_no_int.threshold
                    )
                }
                
        except Exception as e:
            robustness_results['no_interaction'] = {
                'error': str(e),
                'specification': 'no_interaction'
            }
        
        # Test 4: Sub-sample stability
        self.logger.info("Testing sub-sample stability")
        
        n_obs = len(y_aligned)
        if n_obs > 100:  # Only test if we have sufficient observations
            
            # First half
            try:
                mid_point = n_obs // 2
                
                alt_model_first = EnhancedHansenThresholdRegression()
                alt_model_first.fit(
                    y_aligned[:mid_point], 
                    X_baseline[:mid_point], 
                    threshold_aligned[:mid_point], 
                    trim=0.15
                )
                
                if alt_model_first.fitted:
                    robustness_results['first_half'] = {
                        'threshold_value': alt_model_first.threshold,
                        'specification': 'first_half',
                        'observations': mid_point
                    }
                    
            except Exception as e:
                robustness_results['first_half'] = {
                    'error': str(e),
                    'specification': 'first_half'
                }
            
            # Second half
            try:
                alt_model_second = EnhancedHansenThresholdRegression()
                alt_model_second.fit(
                    y_aligned[mid_point:], 
                    X_baseline[mid_point:], 
                    threshold_aligned[mid_point:], 
                    trim=0.15
                )
                
                if alt_model_second.fitted:
                    robustness_results['second_half'] = {
                        'threshold_value': alt_model_second.threshold,
                        'specification': 'second_half',
                        'observations': n_obs - mid_point
                    }
                    
            except Exception as e:
                robustness_results['second_half'] = {
                    'error': str(e),
                    'specification': 'second_half'
                }
        
        # Calculate robustness summary statistics
        robustness_summary = self._calculate_robustness_summary(robustness_results)
        
        return {
            'robustness_tests': robustness_results,
            'robustness_summary': robustness_summary
        }
    
    def _calculate_threshold_r_squared(self,
                                     y: np.ndarray,
                                     X: np.ndarray,
                                     threshold_var: np.ndarray,
                                     threshold_value: float) -> float:
        """Calculate R-squared for threshold model"""
        
        try:
            regime1_mask = threshold_var <= threshold_value
            regime2_mask = threshold_var > threshold_value
            
            if np.sum(regime1_mask) < X.shape[1] + 1 or np.sum(regime2_mask) < X.shape[1] + 1:
                return np.nan
            
            # Fit regime models
            X1_const = sm.add_constant(X[regime1_mask])
            X2_const = sm.add_constant(X[regime2_mask])
            
            model1 = OLS(y[regime1_mask], X1_const).fit()
            model2 = OLS(y[regime2_mask], X2_const).fit()
            
            # Calculate overall R-squared
            y_pred = np.zeros_like(y)
            y_pred[regime1_mask] = model1.predict(X1_const)
            y_pred[regime2_mask] = model2.predict(X2_const)
            
            ss_res = np.sum((y - y_pred) ** 2)
            ss_tot = np.sum((y - np.mean(y)) ** 2)
            
            r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
            return r_squared
            
        except Exception:
            return np.nan
    
    def _calculate_robustness_summary(self, robustness_results: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate summary statistics for robustness tests"""
        
        # Extract threshold values
        threshold_values = []
        significant_count = 0
        total_tests = 0
        
        for spec_name, spec_results in robustness_results.items():
            if isinstance(spec_results, dict) and 'threshold_value' in spec_results:
                threshold_values.append(spec_results['threshold_value'])
                
                if 'threshold_significant' in spec_results:
                    total_tests += 1
                    if spec_results['threshold_significant']:
                        significant_count += 1
        
        summary = {}
        
        if len(threshold_values) > 0:
            summary['threshold_statistics'] = {
                'mean': float(np.mean(threshold_values)),
                'median': float(np.median(threshold_values)),
                'std': float(np.std(threshold_values)),
                'min': float(np.min(threshold_values)),
                'max': float(np.max(threshold_values)),
                'range': float(np.max(threshold_values) - np.min(threshold_values)),
                'coefficient_of_variation': float(np.std(threshold_values) / np.mean(threshold_values))
            }
        
        if total_tests > 0:
            summary['significance_statistics'] = {
                'total_tests': total_tests,
                'significant_tests': significant_count,
                'percentage_significant': float(significant_count / total_tests * 100)
            }
        
        # Robustness assessment
        if len(threshold_values) > 1:
            cv = np.std(threshold_values) / np.mean(threshold_values)
            
            if cv < 0.1:
                robustness_level = 'high'
            elif cv < 0.3:
                robustness_level = 'moderate'
            else:
                robustness_level = 'low'
            
            summary['robustness_assessment'] = {
                'level': robustness_level,
                'coefficient_of_variation': cv,
                'interpretation': f"Threshold estimates show {robustness_level} robustness across specifications"
            }
        
        return summary