"""
Model Diagnostics Module for QE Paper Revisions

This module provides comprehensive diagnostic tools for econometric models
used in the QE analysis, addressing reviewer concerns about model validity
and robustness.
"""

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import jarque_bera, normaltest, shapiro
import statsmodels.api as sm
from statsmodels.stats.diagnostic import het_breuschpagan, het_white
from statsmodels.stats.stattools import durbin_watson
from statsmodels.tsa.stattools import adfuller, kpss
import warnings
warnings.filterwarnings('ignore')

class ModelDiagnostics:
    """
    Comprehensive model diagnostics for QE analysis models
    
    Addresses Requirements 6.1 and 6.2:
    - Hansen threshold regression diagnostics for R² improvement
    - Local projections diagnostics with robustness testing
    - Sample size diagnostics for varying observations
    """
    
    def __init__(self):
        self.diagnostic_results = {}
        
    def hansen_regression_diagnostics(self, model, y, x, threshold_var, 
                                    threshold_value=None, regime_labels=None):
        """
        Comprehensive diagnostics for Hansen threshold regression
        
        Args:
            model: Fitted HansenThresholdRegression object
            y: Dependent variable
            x: Independent variables
            threshold_var: Threshold variable
            threshold_value: Optional specific threshold to test
            regime_labels: Optional labels for regimes
            
        Returns:
            dict: Comprehensive diagnostic results with R² improvement suggestions
        """
        if not hasattr(model, 'fitted') or not model.fitted:
            raise ValueError("Model must be fitted before running diagnostics")
            
        diagnostics = {}
        
        # Use model's threshold if not specified
        if threshold_value is None:
            threshold_value = model.threshold
            
        # Create regime masks
        regime1_mask = threshold_var <= threshold_value
        regime2_mask = threshold_var > threshold_value
        
        # Basic model statistics
        diagnostics['threshold_value'] = threshold_value
        diagnostics['regime1_obs'] = np.sum(regime1_mask)
        diagnostics['regime2_obs'] = np.sum(regime2_mask)
        diagnostics['total_obs'] = len(y)
        
        # Calculate R² for each regime and overall
        regime1_r2, regime2_r2, overall_r2 = self._calculate_regime_r_squared(
            model, y, x, regime1_mask, regime2_mask
        )
        
        diagnostics['regime1_r2'] = regime1_r2
        diagnostics['regime2_r2'] = regime2_r2
        diagnostics['overall_r2'] = overall_r2
        
        # R² improvement suggestions
        diagnostics['r2_suggestions'] = self._generate_r2_improvement_suggestions(
            overall_r2, regime1_r2, regime2_r2
        )
        
        # Residual diagnostics for each regime
        diagnostics['regime1_residual_tests'] = self._residual_diagnostics(
            model, y[regime1_mask], x[regime1_mask], regime='regime1'
        )
        diagnostics['regime2_residual_tests'] = self._residual_diagnostics(
            model, y[regime2_mask], x[regime2_mask], regime='regime2'
        )
        
        # Threshold stability tests
        diagnostics['threshold_stability'] = self._threshold_stability_tests(
            y, x, threshold_var, threshold_value
        )
        
        # Model specification tests
        diagnostics['specification_tests'] = self._specification_tests(
            model, y, x, threshold_var
        )
        
        # Sample adequacy tests
        diagnostics['sample_adequacy'] = self._sample_adequacy_tests_hansen(
            regime1_mask, regime2_mask, x.shape[1] if x.ndim > 1 else 1
        )
        
        self.diagnostic_results['hansen_regression'] = diagnostics
        return diagnostics
    
    def local_projections_diagnostics(self, model, y, shock, controls=None, 
                                    max_horizon=None):
        """
        Comprehensive diagnostics for Local Projections with robustness testing
        
        Args:
            model: Fitted LocalProjections object
            y: Outcome variable
            shock: Shock variable
            controls: Control variables
            max_horizon: Maximum horizon to test
            
        Returns:
            dict: Comprehensive diagnostic results with robustness tests
        """
        if not hasattr(model, 'fitted') or not model.fitted:
            raise ValueError("Model must be fitted before running diagnostics")
            
        diagnostics = {}
        
        if max_horizon is None:
            max_horizon = model.max_horizon
            
        # Horizon-specific diagnostics
        horizon_diagnostics = {}
        
        for h in range(min(max_horizon + 1, len(model.results))):
            if model.results.get(h) is not None:
                horizon_diag = self._single_horizon_diagnostics(
                    model.results[h], h, y, shock, controls
                )
                horizon_diagnostics[h] = horizon_diag
                
        diagnostics['horizon_diagnostics'] = horizon_diagnostics
        
        # Overall robustness tests
        diagnostics['robustness_tests'] = self._local_projections_robustness_tests(
            model, y, shock, controls, max_horizon
        )
        
        # Impulse response stability
        diagnostics['impulse_response_stability'] = self._impulse_response_stability_tests(
            model, max_horizon
        )
        
        # Serial correlation tests across horizons
        diagnostics['serial_correlation_tests'] = self._serial_correlation_across_horizons(
            model, max_horizon
        )
        
        # Specification tests
        diagnostics['specification_tests'] = self._local_projections_specification_tests(
            model, y, shock, controls
        )
        
        self.diagnostic_results['local_projections'] = diagnostics
        return diagnostics
    
    def sample_size_diagnostics(self, datasets, model_types, min_obs_threshold=30):
        """
        Diagnostics for handling varying sample sizes across specifications
        
        Args:
            datasets: List of datasets with varying sample sizes
            model_types: List of model types corresponding to datasets
            min_obs_threshold: Minimum observations threshold
            
        Returns:
            dict: Sample size adequacy and recommendations
        """
        diagnostics = {}
        
        # Basic sample size statistics
        sample_sizes = [len(data) for data in datasets]
        diagnostics['sample_sizes'] = sample_sizes
        diagnostics['min_sample_size'] = min(sample_sizes)
        diagnostics['max_sample_size'] = max(sample_sizes)
        diagnostics['mean_sample_size'] = np.mean(sample_sizes)
        diagnostics['sample_size_variation'] = np.std(sample_sizes)
        
        # Adequacy tests
        diagnostics['adequacy_tests'] = []
        
        for i, (data, model_type) in enumerate(zip(datasets, model_types)):
            adequacy = self._assess_sample_adequacy(
                len(data), model_type, min_obs_threshold
            )
            adequacy['dataset_index'] = i
            diagnostics['adequacy_tests'].append(adequacy)
            
        # Power analysis for varying sample sizes
        diagnostics['power_analysis'] = self._sample_size_power_analysis(
            sample_sizes, model_types
        )
        
        # Recommendations for sample size issues
        diagnostics['recommendations'] = self._sample_size_recommendations(
            sample_sizes, model_types, min_obs_threshold
        )
        
        self.diagnostic_results['sample_size'] = diagnostics
        return diagnostics
    
    def _calculate_regime_r_squared(self, model, y, x, regime1_mask, regime2_mask):
        """Calculate R² for each regime and overall model"""
        
        # Regime 1 R²
        if np.sum(regime1_mask) > 0:
            y1 = y[regime1_mask]
            X1 = np.column_stack([np.ones(np.sum(regime1_mask)), x[regime1_mask]])
            y1_pred = X1 @ model.beta1
            ss_res1 = np.sum((y1 - y1_pred) ** 2)
            ss_tot1 = np.sum((y1 - np.mean(y1)) ** 2)
            regime1_r2 = 1 - (ss_res1 / ss_tot1) if ss_tot1 > 0 else 0
        else:
            regime1_r2 = 0
            
        # Regime 2 R²
        if np.sum(regime2_mask) > 0:
            y2 = y[regime2_mask]
            X2 = np.column_stack([np.ones(np.sum(regime2_mask)), x[regime2_mask]])
            y2_pred = X2 @ model.beta2
            ss_res2 = np.sum((y2 - y2_pred) ** 2)
            ss_tot2 = np.sum((y2 - np.mean(y2)) ** 2)
            regime2_r2 = 1 - (ss_res2 / ss_tot2) if ss_tot2 > 0 else 0
        else:
            regime2_r2 = 0
            
        # Overall R²
        y_pred = model.predict(x, model.threshold * np.ones(len(x)))
        ss_res_total = np.sum((y - y_pred) ** 2)
        ss_tot_total = np.sum((y - np.mean(y)) ** 2)
        overall_r2 = 1 - (ss_res_total / ss_tot_total) if ss_tot_total > 0 else 0
        
        return regime1_r2, regime2_r2, overall_r2
    
    def _generate_r2_improvement_suggestions(self, overall_r2, regime1_r2, regime2_r2):
        """Generate specific suggestions for improving R²"""
        suggestions = []
        
        if overall_r2 < 0.01:
            suggestions.append("Very low overall R² detected. Consider:")
            suggestions.append("- Adding relevant control variables (VIX, term spread, etc.)")
            suggestions.append("- Using first differences instead of levels")
            suggestions.append("- Checking for structural breaks beyond the threshold")
            suggestions.append("- Considering alternative functional forms")
            
        if abs(regime1_r2 - regime2_r2) > 0.1:
            suggestions.append("Large R² difference between regimes suggests:")
            suggestions.append("- Regime-specific control variables may be needed")
            suggestions.append("- Different model specifications for each regime")
            
        if regime1_r2 < 0.05 or regime2_r2 < 0.05:
            low_regime = "Regime 1" if regime1_r2 < 0.05 else "Regime 2"
            suggestions.append(f"{low_regime} has very low R². Consider:")
            suggestions.append("- Regime-specific variables")
            suggestions.append("- Interaction terms with threshold variable")
            
        return suggestions
    
    def _residual_diagnostics(self, model, y, x, regime='unknown'):
        """Comprehensive residual diagnostics for a regime"""
        
        # Calculate residuals
        if regime == 'regime1':
            X_reg = np.column_stack([np.ones(len(y)), x])
            residuals = y - X_reg @ model.beta1
        elif regime == 'regime2':
            X_reg = np.column_stack([np.ones(len(y)), x])
            residuals = y - X_reg @ model.beta2
        else:
            # For general case, use model prediction
            residuals = y - model.predict(x, model.threshold * np.ones(len(x)))
            
        diagnostics = {}
        
        # Normality tests
        if len(residuals) > 3:
            try:
                jb_stat, jb_pvalue = jarque_bera(residuals)
                diagnostics['jarque_bera'] = {'statistic': jb_stat, 'p_value': jb_pvalue}
            except:
                diagnostics['jarque_bera'] = {'statistic': np.nan, 'p_value': np.nan}
                
            try:
                sw_stat, sw_pvalue = shapiro(residuals)
                diagnostics['shapiro_wilk'] = {'statistic': sw_stat, 'p_value': sw_pvalue}
            except:
                diagnostics['shapiro_wilk'] = {'statistic': np.nan, 'p_value': np.nan}
        
        # Heteroskedasticity tests
        if len(residuals) > 5 and x.ndim > 0:
            try:
                X_reg = np.column_stack([np.ones(len(y)), x])
                bp_stat, bp_pvalue, _, _ = het_breuschpagan(residuals, X_reg)
                diagnostics['breusch_pagan'] = {'statistic': bp_stat, 'p_value': bp_pvalue}
            except:
                diagnostics['breusch_pagan'] = {'statistic': np.nan, 'p_value': np.nan}
        
        # Serial correlation (if time series)
        if len(residuals) > 2:
            try:
                dw_stat = durbin_watson(residuals)
                diagnostics['durbin_watson'] = dw_stat
            except:
                diagnostics['durbin_watson'] = np.nan
        
        # Basic residual statistics
        diagnostics['mean'] = np.mean(residuals)
        diagnostics['std'] = np.std(residuals)
        diagnostics['skewness'] = stats.skew(residuals)
        diagnostics['kurtosis'] = stats.kurtosis(residuals)
        
        return diagnostics
    
    def _threshold_stability_tests(self, y, x, threshold_var, threshold_value):
        """Test stability of threshold estimate"""
        
        stability_tests = {}
        
        # Test threshold in different subsamples
        n = len(y)
        subsample_size = n // 2
        
        if subsample_size > 20:  # Minimum for meaningful test
            # First half
            mid_point = n // 2
            y1, x1, thresh1 = y[:mid_point], x[:mid_point], threshold_var[:mid_point]
            
            # Second half  
            y2, x2, thresh2 = y[mid_point:], x[mid_point:], threshold_var[mid_point:]
            
            # Estimate threshold in each subsample
            import sys
            import os
            sys.path.append(os.path.dirname(__file__))
            from models import HansenThresholdRegression
            
            try:
                model1 = HansenThresholdRegression()
                model1.fit(y1, x1, thresh1)
                thresh_est1 = model1.threshold
                
                model2 = HansenThresholdRegression()
                model2.fit(y2, x2, thresh2)
                thresh_est2 = model2.threshold
                
                stability_tests['subsample_thresholds'] = {
                    'first_half': thresh_est1,
                    'second_half': thresh_est2,
                    'difference': abs(thresh_est1 - thresh_est2),
                    'relative_difference': abs(thresh_est1 - thresh_est2) / threshold_value
                }
                
            except:
                stability_tests['subsample_thresholds'] = {
                    'error': 'Could not estimate thresholds in subsamples'
                }
        
        # Bootstrap confidence interval for threshold
        stability_tests['bootstrap_ci'] = self._bootstrap_threshold_ci(
            y, x, threshold_var, n_bootstrap=100
        )
        
        return stability_tests
    
    def _bootstrap_threshold_ci(self, y, x, threshold_var, n_bootstrap=100):
        """Bootstrap confidence interval for threshold estimate"""
        
        # Import here to avoid circular imports
        import sys
        import os
        sys.path.append(os.path.dirname(__file__))
        from models import HansenThresholdRegression
        
        bootstrap_thresholds = []
        n = len(y)
        
        for _ in range(n_bootstrap):
            # Bootstrap sample
            indices = np.random.choice(n, size=n, replace=True)
            y_boot = y[indices]
            x_boot = x[indices]
            thresh_boot = threshold_var[indices]
            
            try:
                model_boot = HansenThresholdRegression()
                model_boot.fit(y_boot, x_boot, thresh_boot)
                if model_boot.threshold is not None:
                    bootstrap_thresholds.append(model_boot.threshold)
            except:
                continue
                
        if len(bootstrap_thresholds) > 10:
            return {
                'lower_ci': np.percentile(bootstrap_thresholds, 2.5),
                'upper_ci': np.percentile(bootstrap_thresholds, 97.5),
                'mean': np.mean(bootstrap_thresholds),
                'std': np.std(bootstrap_thresholds),
                'successful_bootstraps': len(bootstrap_thresholds)
            }
        else:
            return {'error': 'Insufficient successful bootstrap samples'}
    
    def _specification_tests(self, model, y, x, threshold_var):
        """Model specification tests"""
        
        tests = {}
        
        # Linearity test (compare with linear model)
        try:
            X_linear = np.column_stack([np.ones(len(y)), x])
            linear_model = sm.OLS(y, X_linear).fit()
            
            # Calculate predictions
            y_pred_threshold = model.predict(x, threshold_var)
            y_pred_linear = linear_model.predict(X_linear)
            
            # Compare R²
            ss_res_threshold = np.sum((y - y_pred_threshold) ** 2)
            ss_res_linear = np.sum((y - y_pred_linear) ** 2)
            
            tests['linearity_test'] = {
                'threshold_ssr': ss_res_threshold,
                'linear_ssr': ss_res_linear,
                'improvement': (ss_res_linear - ss_res_threshold) / ss_res_linear,
                'f_statistic': ((ss_res_linear - ss_res_threshold) / 2) / (ss_res_threshold / (len(y) - 4))
            }
            
        except:
            tests['linearity_test'] = {'error': 'Could not perform linearity test'}
        
        return tests
    
    def _single_horizon_diagnostics(self, horizon_model, horizon, y, shock, controls):
        """Diagnostics for a single horizon in local projections"""
        
        diagnostics = {}
        
        # Basic model statistics
        diagnostics['horizon'] = horizon
        diagnostics['r_squared'] = horizon_model.rsquared
        diagnostics['adj_r_squared'] = horizon_model.rsquared_adj
        diagnostics['n_obs'] = horizon_model.nobs
        diagnostics['f_statistic'] = horizon_model.fvalue
        diagnostics['f_pvalue'] = horizon_model.f_pvalue
        
        # Coefficient diagnostics
        shock_coeff_name = horizon_model.params.index[1]  # Assuming shock is first regressor
        diagnostics['shock_coefficient'] = horizon_model.params[shock_coeff_name]
        diagnostics['shock_std_error'] = horizon_model.bse[shock_coeff_name]
        diagnostics['shock_t_stat'] = horizon_model.tvalues[shock_coeff_name]
        diagnostics['shock_p_value'] = horizon_model.pvalues[shock_coeff_name]
        
        # Residual diagnostics
        residuals = horizon_model.resid
        diagnostics['residual_diagnostics'] = self._residual_diagnostics_ols(residuals)
        
        return diagnostics
    
    def _residual_diagnostics_ols(self, residuals):
        """Residual diagnostics for OLS models"""
        
        diagnostics = {}
        
        # Basic statistics
        diagnostics['mean'] = np.mean(residuals)
        diagnostics['std'] = np.std(residuals)
        diagnostics['skewness'] = stats.skew(residuals)
        diagnostics['kurtosis'] = stats.kurtosis(residuals)
        
        # Normality tests
        if len(residuals) > 3:
            try:
                jb_stat, jb_pvalue = jarque_bera(residuals)
                diagnostics['jarque_bera'] = {'statistic': jb_stat, 'p_value': jb_pvalue}
            except:
                diagnostics['jarque_bera'] = {'statistic': np.nan, 'p_value': np.nan}
        
        # Serial correlation
        if len(residuals) > 2:
            try:
                dw_stat = durbin_watson(residuals)
                diagnostics['durbin_watson'] = dw_stat
            except:
                diagnostics['durbin_watson'] = np.nan
        
        return diagnostics
    
    def _local_projections_robustness_tests(self, model, y, shock, controls, max_horizon):
        """Robustness tests for local projections"""
        
        robustness = {}
        
        # Test with different lag specifications
        robustness['lag_robustness'] = self._test_lag_specifications(
            y, shock, controls, max_horizon
        )
        
        # Test with different control specifications
        robustness['control_robustness'] = self._test_control_specifications(
            y, shock, controls, max_horizon
        )
        
        # Subsample stability
        robustness['subsample_stability'] = self._test_subsample_stability(
            y, shock, controls, max_horizon
        )
        
        return robustness
    
    def _test_lag_specifications(self, y, shock, controls, max_horizon):
        """Test robustness to different lag specifications"""
        
        import sys
        import os
        sys.path.append(os.path.dirname(__file__))
        from models import LocalProjections
        
        lag_tests = {}
        
        for lags in [2, 4, 6, 8]:
            try:
                lp_model = LocalProjections(max_horizon=min(max_horizon, 10))
                lp_model.fit(y, shock, controls, lags=lags)
                
                # Get impulse responses
                ir = lp_model.get_impulse_responses()
                
                lag_tests[f'lags_{lags}'] = {
                    'peak_response': ir['coefficient'].max() if len(ir) > 0 else np.nan,
                    'peak_horizon': ir.loc[ir['coefficient'].idxmax(), 'horizon'] if len(ir) > 0 else np.nan,
                    'significant_horizons': np.sum(np.abs(ir['coefficient']) > 1.96 * np.abs(ir['coefficient'] / ir['coefficient'])) if len(ir) > 0 else 0
                }
                
            except:
                lag_tests[f'lags_{lags}'] = {'error': 'Could not estimate model'}
        
        return lag_tests
    
    def _test_control_specifications(self, y, shock, controls, max_horizon):
        """Test robustness to different control specifications"""
        
        import sys
        import os
        sys.path.append(os.path.dirname(__file__))
        from models import LocalProjections
        
        control_tests = {}
        
        # Test without controls
        try:
            lp_no_controls = LocalProjections(max_horizon=min(max_horizon, 10))
            lp_no_controls.fit(y, shock, controls=None)
            ir_no_controls = lp_no_controls.get_impulse_responses()
            
            control_tests['no_controls'] = {
                'peak_response': ir_no_controls['coefficient'].max() if len(ir_no_controls) > 0 else np.nan,
                'peak_horizon': ir_no_controls.loc[ir_no_controls['coefficient'].idxmax(), 'horizon'] if len(ir_no_controls) > 0 else np.nan
            }
        except:
            control_tests['no_controls'] = {'error': 'Could not estimate model'}
        
        # Test with controls (if provided)
        if controls is not None:
            try:
                lp_with_controls = LocalProjections(max_horizon=min(max_horizon, 10))
                lp_with_controls.fit(y, shock, controls)
                ir_with_controls = lp_with_controls.get_impulse_responses()
                
                control_tests['with_controls'] = {
                    'peak_response': ir_with_controls['coefficient'].max() if len(ir_with_controls) > 0 else np.nan,
                    'peak_horizon': ir_with_controls.loc[ir_with_controls['coefficient'].idxmax(), 'horizon'] if len(ir_with_controls) > 0 else np.nan
                }
            except:
                control_tests['with_controls'] = {'error': 'Could not estimate model'}
        
        return control_tests
    
    def _test_subsample_stability(self, y, shock, controls, max_horizon):
        """Test stability across subsamples"""
        
        import sys
        import os
        sys.path.append(os.path.dirname(__file__))
        from models import LocalProjections
        
        stability_tests = {}
        
        n = len(y)
        if n > 40:  # Minimum for meaningful subsample analysis
            
            # First half
            mid_point = n // 2
            try:
                lp_first = LocalProjections(max_horizon=min(max_horizon, 10))
                lp_first.fit(y[:mid_point], shock[:mid_point], 
                           controls[:mid_point] if controls is not None else None)
                ir_first = lp_first.get_impulse_responses()
                
                stability_tests['first_half'] = {
                    'peak_response': ir_first['coefficient'].max() if len(ir_first) > 0 else np.nan,
                    'observations': mid_point
                }
            except:
                stability_tests['first_half'] = {'error': 'Could not estimate model'}
            
            # Second half
            try:
                lp_second = LocalProjections(max_horizon=min(max_horizon, 10))
                lp_second.fit(y[mid_point:], shock[mid_point:], 
                            controls[mid_point:] if controls is not None else None)
                ir_second = lp_second.get_impulse_responses()
                
                stability_tests['second_half'] = {
                    'peak_response': ir_second['coefficient'].max() if len(ir_second) > 0 else np.nan,
                    'observations': n - mid_point
                }
            except:
                stability_tests['second_half'] = {'error': 'Could not estimate model'}
        
        return stability_tests
    
    def _impulse_response_stability_tests(self, model, max_horizon):
        """Test stability of impulse responses"""
        
        stability = {}
        
        # Get impulse responses
        try:
            ir = model.get_impulse_responses()
            
            if len(ir) > 0:
                # Peak response analysis
                stability['peak_response'] = ir['coefficient'].max()
                stability['peak_horizon'] = ir.loc[ir['coefficient'].idxmax(), 'horizon']
                stability['trough_response'] = ir['coefficient'].min()
                stability['trough_horizon'] = ir.loc[ir['coefficient'].idxmin(), 'horizon']
                
                # Significance analysis
                significant_mask = (ir['lower_ci'] > 0) | (ir['upper_ci'] < 0)
                stability['significant_horizons'] = ir.loc[significant_mask, 'horizon'].tolist()
                stability['first_significant_horizon'] = ir.loc[significant_mask, 'horizon'].min() if significant_mask.any() else None
                stability['last_significant_horizon'] = ir.loc[significant_mask, 'horizon'].max() if significant_mask.any() else None
                
                # Persistence analysis
                stability['response_persistence'] = self._calculate_response_persistence(ir)
                
            else:
                stability['error'] = 'No impulse responses available'
                
        except:
            stability['error'] = 'Could not analyze impulse responses'
        
        return stability
    
    def _calculate_response_persistence(self, ir):
        """Calculate persistence of impulse responses"""
        
        if len(ir) == 0:
            return np.nan
            
        # Half-life calculation
        peak_response = ir['coefficient'].max()
        half_peak = peak_response / 2
        
        # Find horizon where response falls below half of peak
        below_half = ir[ir['coefficient'] < half_peak]
        if len(below_half) > 0:
            half_life = below_half['horizon'].min()
        else:
            half_life = ir['horizon'].max()  # Response doesn't decay to half
            
        return {
            'half_life': half_life,
            'peak_response': peak_response,
            'final_response': ir['coefficient'].iloc[-1],
            'decay_rate': (peak_response - ir['coefficient'].iloc[-1]) / len(ir)
        }
    
    def _serial_correlation_across_horizons(self, model, max_horizon):
        """Test for serial correlation across horizons"""
        
        serial_tests = {}
        
        for h in range(min(max_horizon + 1, len(model.results))):
            if model.results.get(h) is not None:
                residuals = model.results[h].resid
                
                if len(residuals) > 2:
                    try:
                        dw_stat = durbin_watson(residuals)
                        serial_tests[f'horizon_{h}'] = {
                            'durbin_watson': dw_stat,
                            'serial_correlation_concern': dw_stat < 1.5 or dw_stat > 2.5
                        }
                    except:
                        serial_tests[f'horizon_{h}'] = {'error': 'Could not calculate DW statistic'}
        
        return serial_tests
    
    def _local_projections_specification_tests(self, model, y, shock, controls):
        """Specification tests for local projections"""
        
        tests = {}
        
        # Test for structural breaks
        tests['structural_breaks'] = self._test_structural_breaks_lp(y, shock, controls)
        
        # Test for nonlinearity
        tests['nonlinearity'] = self._test_nonlinearity_lp(y, shock, controls)
        
        return tests
    
    def _test_structural_breaks_lp(self, y, shock, controls):
        """Test for structural breaks in local projections"""
        
        # Simple Chow test approach
        n = len(y)
        if n < 40:
            return {'error': 'Insufficient observations for structural break test'}
        
        import sys
        import os
        sys.path.append(os.path.dirname(__file__))
        from models import LocalProjections
        
        # Split sample at midpoint
        mid_point = n // 2
        
        try:
            # First subsample
            lp1 = LocalProjections(max_horizon=5)
            lp1.fit(y[:mid_point], shock[:mid_point], 
                   controls[:mid_point] if controls is not None else None)
            
            # Second subsample
            lp2 = LocalProjections(max_horizon=5)
            lp2.fit(y[mid_point:], shock[mid_point:], 
                   controls[mid_point:] if controls is not None else None)
            
            # Compare responses
            ir1 = lp1.get_impulse_responses()
            ir2 = lp2.get_impulse_responses()
            
            if len(ir1) > 0 and len(ir2) > 0:
                # Simple comparison of peak responses
                peak1 = ir1['coefficient'].max()
                peak2 = ir2['coefficient'].max()
                
                return {
                    'first_half_peak': peak1,
                    'second_half_peak': peak2,
                    'difference': abs(peak1 - peak2),
                    'relative_difference': abs(peak1 - peak2) / max(abs(peak1), abs(peak2)) if max(abs(peak1), abs(peak2)) > 0 else 0
                }
            else:
                return {'error': 'Could not estimate impulse responses for subsamples'}
                
        except:
            return {'error': 'Could not perform structural break test'}
    
    def _test_nonlinearity_lp(self, y, shock, controls):
        """Test for nonlinearity in local projections"""
        
        # Test by adding squared shock term
        import sys
        import os
        sys.path.append(os.path.dirname(__file__))
        from models import LocalProjections
        
        try:
            # Create squared shock
            shock_squared = shock ** 2
            
            # Combine controls
            if controls is not None:
                extended_controls = pd.concat([controls, shock_squared], axis=1)
            else:
                extended_controls = shock_squared
            
            # Estimate with nonlinear term
            lp_nonlinear = LocalProjections(max_horizon=5)
            lp_nonlinear.fit(y, shock, extended_controls)
            
            # Compare with linear specification
            lp_linear = LocalProjections(max_horizon=5)
            lp_linear.fit(y, shock, controls)
            
            # Simple comparison of fit
            ir_nonlinear = lp_nonlinear.get_impulse_responses()
            ir_linear = lp_linear.get_impulse_responses()
            
            if len(ir_nonlinear) > 0 and len(ir_linear) > 0:
                return {
                    'nonlinear_peak': ir_nonlinear['coefficient'].max(),
                    'linear_peak': ir_linear['coefficient'].max(),
                    'improvement': abs(ir_nonlinear['coefficient'].max()) - abs(ir_linear['coefficient'].max())
                }
            else:
                return {'error': 'Could not estimate nonlinear specification'}
                
        except:
            return {'error': 'Could not perform nonlinearity test'}
    
    def _assess_sample_adequacy(self, sample_size, model_type, min_threshold):
        """Assess adequacy of sample size for specific model type"""
        
        adequacy = {}
        adequacy['sample_size'] = sample_size
        adequacy['model_type'] = model_type
        adequacy['min_threshold'] = min_threshold
        
        # Model-specific requirements
        if model_type == 'hansen_threshold':
            required_obs = 40  # Minimum for threshold search
            adequacy['required_obs'] = required_obs
            adequacy['adequate'] = sample_size >= required_obs
            
        elif model_type == 'local_projections':
            required_obs = 50  # For meaningful impulse responses
            adequacy['required_obs'] = required_obs
            adequacy['adequate'] = sample_size >= required_obs
            
        elif model_type == 'iv_regression':
            required_obs = 30  # For IV estimation
            adequacy['required_obs'] = required_obs
            adequacy['adequate'] = sample_size >= required_obs
            
        else:
            adequacy['required_obs'] = min_threshold
            adequacy['adequate'] = sample_size >= min_threshold
        
        # Power considerations
        adequacy['power_assessment'] = self._assess_statistical_power(
            sample_size, model_type
        )
        
        return adequacy
    
    def _assess_statistical_power(self, sample_size, model_type):
        """Assess statistical power for given sample size"""
        
        # Rough power assessment based on sample size
        if sample_size < 30:
            power_level = 'Low'
            power_score = 0.3
        elif sample_size < 50:
            power_level = 'Moderate'
            power_score = 0.6
        elif sample_size < 100:
            power_level = 'Good'
            power_score = 0.8
        else:
            power_level = 'High'
            power_score = 0.9
            
        return {
            'power_level': power_level,
            'power_score': power_score,
            'recommendation': self._power_recommendation(power_level, model_type)
        }
    
    def _power_recommendation(self, power_level, model_type):
        """Generate recommendations based on power assessment"""
        
        if power_level == 'Low':
            return f"Low power for {model_type}. Consider combining datasets or using more efficient estimators."
        elif power_level == 'Moderate':
            return f"Moderate power for {model_type}. Results should be interpreted cautiously."
        else:
            return f"Adequate power for {model_type}."
    
    def _sample_size_power_analysis(self, sample_sizes, model_types):
        """Power analysis across different sample sizes"""
        
        power_analysis = {}
        
        for i, (size, model_type) in enumerate(zip(sample_sizes, model_types)):
            power_analysis[f'dataset_{i}'] = self._assess_statistical_power(size, model_type)
        
        # Overall assessment
        min_power = min([pa['power_score'] for pa in power_analysis.values()])
        mean_power = np.mean([pa['power_score'] for pa in power_analysis.values()])
        
        power_analysis['overall'] = {
            'min_power_score': min_power,
            'mean_power_score': mean_power,
            'overall_assessment': 'Adequate' if min_power > 0.6 else 'Concerning'
        }
        
        return power_analysis
    
    def _sample_size_recommendations(self, sample_sizes, model_types, min_threshold):
        """Generate recommendations for sample size issues"""
        
        recommendations = []
        
        min_size = min(sample_sizes)
        max_size = max(sample_sizes)
        
        if min_size < min_threshold:
            recommendations.append(f"Minimum sample size ({min_size}) below threshold ({min_threshold})")
            recommendations.append("Consider: data pooling, longer time periods, or alternative specifications")
        
        if max_size / min_size > 2:
            recommendations.append("Large variation in sample sizes across specifications")
            recommendations.append("Consider: balanced panels, consistent time periods, or robustness checks")
        
        # Model-specific recommendations
        for size, model_type in zip(sample_sizes, model_types):
            if model_type == 'hansen_threshold' and size < 40:
                recommendations.append(f"Hansen threshold regression needs more observations (current: {size}, recommended: 40+)")
            elif model_type == 'local_projections' and size < 50:
                recommendations.append(f"Local projections need more observations (current: {size}, recommended: 50+)")
        
        return recommendations
    
    def generate_diagnostic_report(self, output_file=None):
        """Generate comprehensive diagnostic report"""
        
        report = []
        report.append("=" * 60)
        report.append("COMPREHENSIVE MODEL DIAGNOSTICS REPORT")
        report.append("=" * 60)
        report.append("")
        
        # Hansen Regression Diagnostics
        if 'hansen_regression' in self.diagnostic_results:
            report.append("HANSEN THRESHOLD REGRESSION DIAGNOSTICS")
            report.append("-" * 40)
            
            hr_diag = self.diagnostic_results['hansen_regression']
            
            report.append(f"Threshold Value: {hr_diag.get('threshold_value', 'N/A')}")
            report.append(f"Regime 1 Observations: {hr_diag.get('regime1_obs', 'N/A')}")
            report.append(f"Regime 2 Observations: {hr_diag.get('regime2_obs', 'N/A')}")
            report.append(f"Overall R²: {hr_diag.get('overall_r2', 'N/A'):.4f}")
            report.append(f"Regime 1 R²: {hr_diag.get('regime1_r2', 'N/A'):.4f}")
            report.append(f"Regime 2 R²: {hr_diag.get('regime2_r2', 'N/A'):.4f}")
            report.append("")
            
            if 'r2_suggestions' in hr_diag:
                report.append("R² IMPROVEMENT SUGGESTIONS:")
                for suggestion in hr_diag['r2_suggestions']:
                    report.append(f"  • {suggestion}")
                report.append("")
        
        # Local Projections Diagnostics
        if 'local_projections' in self.diagnostic_results:
            report.append("LOCAL PROJECTIONS DIAGNOSTICS")
            report.append("-" * 40)
            
            lp_diag = self.diagnostic_results['local_projections']
            
            if 'impulse_response_stability' in lp_diag:
                stability = lp_diag['impulse_response_stability']
                report.append(f"Peak Response: {stability.get('peak_response', 'N/A')}")
                report.append(f"Peak Horizon: {stability.get('peak_horizon', 'N/A')}")
                report.append(f"Significant Horizons: {stability.get('significant_horizons', 'N/A')}")
                report.append("")
        
        # Sample Size Diagnostics
        if 'sample_size' in self.diagnostic_results:
            report.append("SAMPLE SIZE DIAGNOSTICS")
            report.append("-" * 40)
            
            ss_diag = self.diagnostic_results['sample_size']
            
            report.append(f"Sample Size Range: {ss_diag.get('min_sample_size', 'N/A')} - {ss_diag.get('max_sample_size', 'N/A')}")
            report.append(f"Mean Sample Size: {ss_diag.get('mean_sample_size', 'N/A'):.1f}")
            report.append(f"Sample Size Variation: {ss_diag.get('sample_size_variation', 'N/A'):.1f}")
            report.append("")
            
            if 'recommendations' in ss_diag:
                report.append("RECOMMENDATIONS:")
                for rec in ss_diag['recommendations']:
                    report.append(f"  • {rec}")
                report.append("")
        
        report_text = "\n".join(report)
        
        if output_file:
            with open(output_file, 'w') as f:
                f.write(report_text)
        
        return report_text
    
    def _sample_adequacy_tests_hansen(self, regime1_mask, regime2_mask, n_vars):
        """Sample adequacy tests specific to Hansen threshold regression"""
        
        adequacy = {}
        
        regime1_obs = np.sum(regime1_mask)
        regime2_obs = np.sum(regime2_mask)
        total_obs = len(regime1_mask)
        
        adequacy['regime1_obs'] = regime1_obs
        adequacy['regime2_obs'] = regime2_obs
        adequacy['total_obs'] = total_obs
        
        # Minimum observations per regime
        min_obs_per_regime = max(10, 2 * n_vars)
        
        adequacy['regime1_adequate'] = regime1_obs >= min_obs_per_regime
        adequacy['regime2_adequate'] = regime2_obs >= min_obs_per_regime
        adequacy['overall_adequate'] = adequacy['regime1_adequate'] and adequacy['regime2_adequate']
        
        # Balance between regimes
        regime_balance = min(regime1_obs, regime2_obs) / max(regime1_obs, regime2_obs)
        adequacy['regime_balance'] = regime_balance
        adequacy['balanced_regimes'] = regime_balance > 0.3  # At least 30% in smaller regime
        
        # Recommendations
        recommendations = []
        if not adequacy['regime1_adequate']:
            recommendations.append(f"Regime 1 has insufficient observations ({regime1_obs} < {min_obs_per_regime})")
        if not adequacy['regime2_adequate']:
            recommendations.append(f"Regime 2 has insufficient observations ({regime2_obs} < {min_obs_per_regime})")
        if not adequacy['balanced_regimes']:
            recommendations.append(f"Regimes are unbalanced (ratio: {regime_balance:.2f})")
            
        adequacy['recommendations'] = recommendations
        
        return adequacy