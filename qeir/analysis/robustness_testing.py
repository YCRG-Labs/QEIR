"""
Comprehensive Robustness Testing Framework for QE Analysis

This module provides systematic validation across specifications, temporal periods,
identification strategies, and model assumptions to ensure robust empirical findings.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Callable
import warnings
from datetime import datetime, timedelta
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error
import scipy.stats as stats
from dataclasses import dataclass


@dataclass
class RobustnessResult:
    """Container for robustness test results"""
    test_name: str
    specification: str
    coefficient: float
    std_error: float
    p_value: float
    confidence_interval: Tuple[float, float]
    sample_size: int
    r_squared: float
    additional_stats: Dict[str, Any]


class RobustnessTestSuite:
    """
    Comprehensive robustness testing framework for QE analysis
    
    Provides systematic validation across different specifications, temporal periods,
    identification strategies, and model assumptions.
    """
    
    def __init__(self, base_data: pd.DataFrame, qe_start_date: str = '2008-11-01'):
        """
        Initialize robustness testing suite
        
        Parameters:
        -----------
        base_data : pd.DataFrame
            Base dataset with all variables
        qe_start_date : str
            Start date for QE period analysis
        """
        self.base_data = base_data.copy()
        self.qe_start_date = pd.to_datetime(qe_start_date)
        self.results = {}
        
        # Ensure date column is datetime
        if 'date' in self.base_data.columns:
            self.base_data['date'] = pd.to_datetime(self.base_data['date'])
            self.base_data = self.base_data.set_index('date')
        
    def temporal_robustness_test(self, 
                               model_func: Callable,
                               dependent_var: str,
                               independent_vars: List[str],
                               qe_episode_definitions: Optional[Dict[str, List[str]]] = None) -> Dict[str, RobustnessResult]:
        """
        Test robustness across different QE episode definitions and temporal periods
        
        Parameters:
        -----------
        model_func : Callable
            Function that fits the econometric model
        dependent_var : str
            Name of dependent variable
        independent_vars : List[str]
            List of independent variable names
        qe_episode_definitions : Dict[str, List[str]], optional
            Different QE episode definitions to test
            
        Returns:
        --------
        Dict[str, RobustnessResult]
            Results for each temporal specification
        """
        if qe_episode_definitions is None:
            qe_episode_definitions = {
                'qe1_only': ['2008-11-01', '2010-06-30'],
                'qe1_qe2': ['2008-11-01', '2012-12-31'],
                'all_qe': ['2008-11-01', '2020-12-31'],
                'post_crisis': ['2010-01-01', '2020-12-31'],
                'pre_covid': ['2008-11-01', '2019-12-31']
            }
        
        results = {}
        
        for period_name, date_range in qe_episode_definitions.items():
            try:
                # Filter data for specific period
                start_date = pd.to_datetime(date_range[0])
                end_date = pd.to_datetime(date_range[1])
                
                period_data = self.base_data.loc[start_date:end_date].copy()
                
                if len(period_data) < 20:  # Minimum sample size
                    warnings.warn(f"Insufficient data for period {period_name}")
                    continue
                
                # Fit model for this period
                model_result = model_func(
                    data=period_data,
                    dependent_var=dependent_var,
                    independent_vars=independent_vars
                )
                
                # Extract key coefficient (assuming first independent variable is main QE measure)
                main_coef = model_result.params[independent_vars[0]]
                main_se = model_result.bse[independent_vars[0]]
                main_pval = model_result.pvalues[independent_vars[0]]
                
                # Calculate confidence interval
                ci_lower = main_coef - 1.96 * main_se
                ci_upper = main_coef + 1.96 * main_se
                
                results[period_name] = RobustnessResult(
                    test_name='temporal_robustness',
                    specification=period_name,
                    coefficient=main_coef,
                    std_error=main_se,
                    p_value=main_pval,
                    confidence_interval=(ci_lower, ci_upper),
                    sample_size=len(period_data),
                    r_squared=model_result.rsquared if hasattr(model_result, 'rsquared') else np.nan,
                    additional_stats={'period': date_range}
                )
                
            except Exception as e:
                warnings.warn(f"Error in temporal robustness test for {period_name}: {str(e)}")
                continue
        
        self.results['temporal_robustness'] = results
        return results
    
    def multiple_threshold_test(self,
                              threshold_var: str,
                              dependent_var: str,
                              qe_var: str,
                              control_vars: List[str],
                              max_thresholds: int = 3) -> Dict:
        """
        Test for multiple thresholds following Hansen (2000) sequential testing
        
        Parameters:
        -----------
        threshold_var : str
            Variable to test for thresholds (e.g., debt service burden)
        dependent_var : str
            Dependent variable
        qe_var : str
            QE intensity variable
        control_vars : List[str]
            Control variables
        max_thresholds : int, default=3
            Maximum number of thresholds to test
            
        Returns:
        --------
        Dict containing threshold test results
        """
        from qeir.core.threshold_regression import EnhancedThresholdRegression
        
        data = self.base_data.dropna(subset=[threshold_var, dependent_var, qe_var] + control_vars)
        
        threshold_results = {}
        current_data = data.copy()
        
        for n_thresh in range(1, max_thresholds + 1):
            try:
                # Fit threshold model
                threshold_model = EnhancedThresholdRegression(
                    confidence_interactions=True,
                    bootstrap_iterations=1000
                )
                
                result = threshold_model.fit(
                    data=current_data,
                    dependent_var=dependent_var,
                    threshold_var=threshold_var,
                    qe_var=qe_var,
                    control_vars=control_vars
                )
                
                threshold_results[f'threshold_{n_thresh}'] = {
                    'threshold_estimate': result['threshold_estimate'],
                    'confidence_interval': result['confidence_interval'],
                    'test_statistic': result['test_statistic'],
                    'p_value': result['p_value'],
                    'significant': result['p_value'] < 0.05
                }
                
                # If not significant, stop testing
                if result['p_value'] >= 0.05:
                    break
                    
            except Exception as e:
                warnings.warn(f"Error in threshold test {n_thresh}: {str(e)}")
                break
        
        return threshold_results
    
    def driscoll_kraay_standard_errors(self,
                                     model_result,
                                     data: pd.DataFrame,
                                     max_lags: int = 4) -> Dict:
        """
        Compute Driscoll-Kraay standard errors robust to serial correlation
        and cross-sectional dependence
        
        Parameters:
        -----------
        model_result : regression result object
            Fitted regression model
        data : pd.DataFrame
            Data used in regression
        max_lags : int, default=4
            Maximum number of lags for HAC correction
            
        Returns:
        --------
        Dict containing corrected standard errors and test statistics
        """
        try:
            # Extract residuals and design matrix
            residuals = model_result.resid
            X = model_result.model.exog
            
            # Compute Driscoll-Kraay covariance matrix
            n_obs = len(residuals)
            k_vars = X.shape[1]
            
            # Newey-West type correction with Driscoll-Kraay modification
            XpX_inv = np.linalg.inv(X.T @ X)
            
            # Initialize covariance matrix
            S = np.zeros((k_vars, k_vars))
            
            # Add contemporaneous term
            for t in range(n_obs):
                x_t = X[t:t+1, :].T
                S += x_t @ x_t.T * (residuals[t] ** 2)
            
            # Add lagged terms with Bartlett weights
            for lag in range(1, min(max_lags + 1, n_obs)):
                weight = 1 - lag / (max_lags + 1)  # Bartlett kernel
                
                for t in range(lag, n_obs):
                    x_t = X[t:t+1, :].T
                    x_t_lag = X[t-lag:t-lag+1, :].T
                    
                    cross_term = (x_t @ x_t_lag.T * residuals[t] * residuals[t-lag] +
                                 x_t_lag @ x_t.T * residuals[t-lag] * residuals[t])
                    
                    S += weight * cross_term
            
            # Final covariance matrix
            V_dk = XpX_inv @ S @ XpX_inv / n_obs
            
            # Standard errors
            se_dk = np.sqrt(np.diag(V_dk))
            
            # T-statistics
            t_stats_dk = model_result.params / se_dk
            
            # P-values (two-tailed)
            p_values_dk = 2 * (1 - stats.t.cdf(np.abs(t_stats_dk), n_obs - k_vars))
            
            return {
                'standard_errors': se_dk,
                't_statistics': t_stats_dk,
                'p_values': p_values_dk,
                'covariance_matrix': V_dk,
                'method': 'Driscoll-Kraay'
            }
            
        except Exception as e:
            warnings.warn(f"Error computing Driscoll-Kraay standard errors: {str(e)}")
            return {
                'standard_errors': model_result.bse,
                't_statistics': model_result.tvalues,
                'p_values': model_result.pvalues,
                'method': 'OLS (fallback)'
            }
                
                period_data = self.base_data[
                    (self.base_data.index >= start_date) & 
                    (self.base_data.index <= end_date)
                ].copy()
                
                if len(period_data) < 20:  # Minimum sample size check
                    warnings.warn(f"Insufficient data for period {period_name}: {len(period_data)} observations")
                    continue
                
                # Fit model for this period
                model_result = model_func(
                    data=period_data,
                    dependent_var=dependent_var,
                    independent_vars=independent_vars
                )
                
                # Extract key coefficient (assume first independent variable is main QE variable)
                main_coef = model_result.params[independent_vars[0]]
                main_se = model_result.bse[independent_vars[0]]
                main_pval = model_result.pvalues[independent_vars[0]]
                
                # Calculate confidence interval
                ci_lower = main_coef - 1.96 * main_se
                ci_upper = main_coef + 1.96 * main_se
                
                results[period_name] = RobustnessResult(
                    test_name="temporal_robustness",
                    specification=period_name,
                    coefficient=main_coef,
                    std_error=main_se,
                    p_value=main_pval,
                    confidence_interval=(ci_lower, ci_upper),
                    sample_size=len(period_data),
                    r_squared=getattr(model_result, 'rsquared', np.nan),
                    additional_stats={
                        'period_start': start_date,
                        'period_end': end_date,
                        'f_statistic': getattr(model_result, 'fvalue', np.nan)
                    }
                )
                
            except Exception as e:
                warnings.warn(f"Error in temporal robustness test for {period_name}: {str(e)}")
                continue
        
        self.results['temporal_robustness'] = results
        return results
    
    def identification_robustness_test(self,
                                     iv_model_func: Callable,
                                     dependent_var: str,
                                     endogenous_vars: List[str],
                                     instrument_sets: Dict[str, List[str]]) -> Dict[str, RobustnessResult]:
        """
        Test robustness across multiple instrument specifications
        
        Parameters:
        -----------
        iv_model_func : Callable
            Function that fits IV regression model
        dependent_var : str
            Name of dependent variable
        endogenous_vars : List[str]
            List of endogenous variables
        instrument_sets : Dict[str, List[str]]
            Different instrument sets to test
            
        Returns:
        --------
        Dict[str, RobustnessResult]
            Results for each instrument specification
        """
        results = {}
        
        # Filter to QE period
        qe_data = self.base_data[self.base_data.index >= self.qe_start_date].copy()
        
        for instrument_name, instruments in instrument_sets.items():
            try:
                # Check if all instruments are available
                missing_instruments = [inst for inst in instruments if inst not in qe_data.columns]
                if missing_instruments:
                    warnings.warn(f"Missing instruments for {instrument_name}: {missing_instruments}")
                    continue
                
                # Fit IV model with this instrument set
                model_result = iv_model_func(
                    data=qe_data,
                    dependent_var=dependent_var,
                    endogenous_vars=endogenous_vars,
                    instruments=instruments
                )
                
                # Extract main coefficient (first endogenous variable)
                main_coef = model_result.params[endogenous_vars[0]]
                main_se = model_result.std_errors[endogenous_vars[0]]
                main_pval = model_result.pvalues[endogenous_vars[0]]
                
                # Calculate confidence interval
                ci_lower = main_coef - 1.96 * main_se
                ci_upper = main_coef + 1.96 * main_se
                
                # Get instrument diagnostics
                first_stage_f = getattr(model_result, 'first_stage', {}).get('f_statistic', np.nan)
                j_statistic = getattr(model_result, 'j_statistic', np.nan)
                
                results[instrument_name] = RobustnessResult(
                    test_name="identification_robustness",
                    specification=instrument_name,
                    coefficient=main_coef,
                    std_error=main_se,
                    p_value=main_pval,
                    confidence_interval=(ci_lower, ci_upper),
                    sample_size=len(qe_data),
                    r_squared=getattr(model_result, 'rsquared', np.nan),
                    additional_stats={
                        'first_stage_f': first_stage_f,
                        'j_statistic': j_statistic,
                        'num_instruments': len(instruments),
                        'instruments': instruments
                    }
                )
                
            except Exception as e:
                warnings.warn(f"Error in identification robustness test for {instrument_name}: {str(e)}")
                continue
        
        self.results['identification_robustness'] = results
        return results
    
    def model_specification_robustness_test(self,
                                          model_specifications: Dict[str, Dict],
                                          dependent_var: str) -> Dict[str, RobustnessResult]:
        """
        Test robustness across alternative functional forms and specifications
        
        Parameters:
        -----------
        model_specifications : Dict[str, Dict]
            Dictionary of model specifications with parameters
        dependent_var : str
            Name of dependent variable
            
        Returns:
        --------
        Dict[str, RobustnessResult]
            Results for each model specification
        """
        results = {}
        
        # Filter to QE period
        qe_data = self.base_data[self.base_data.index >= self.qe_start_date].copy()
        
        for spec_name, spec_params in model_specifications.items():
            try:
                model_func = spec_params['model_func']
                model_args = spec_params.get('args', {})
                
                # Fit model with this specification
                model_result = model_func(
                    data=qe_data,
                    dependent_var=dependent_var,
                    **model_args
                )
                
                # Extract main QE coefficient
                qe_var = spec_params.get('qe_variable', 'qe_intensity')
                if qe_var not in model_result.params.index:
                    warnings.warn(f"QE variable {qe_var} not found in {spec_name} results")
                    continue
                
                main_coef = model_result.params[qe_var]
                main_se = model_result.bse[qe_var]
                main_pval = model_result.pvalues[qe_var]
                
                # Calculate confidence interval
                ci_lower = main_coef - 1.96 * main_se
                ci_upper = main_coef + 1.96 * main_se
                
                results[spec_name] = RobustnessResult(
                    test_name="model_specification_robustness",
                    specification=spec_name,
                    coefficient=main_coef,
                    std_error=main_se,
                    p_value=main_pval,
                    confidence_interval=(ci_lower, ci_upper),
                    sample_size=len(qe_data),
                    r_squared=getattr(model_result, 'rsquared', np.nan),
                    additional_stats={
                        'model_type': spec_params.get('model_type', 'unknown'),
                        'specification_details': model_args
                    }
                )
                
            except Exception as e:
                warnings.warn(f"Error in model specification robustness test for {spec_name}: {str(e)}")
                continue
        
        self.results['model_specification_robustness'] = results
        return results
    
    def subsample_robustness_test(self,
                                model_func: Callable,
                                dependent_var: str,
                                independent_vars: List[str],
                                subsample_strategies: Optional[Dict[str, Dict]] = None) -> Dict[str, RobustnessResult]:
        """
        Test stability across different time periods and subsamples
        
        Parameters:
        -----------
        model_func : Callable
            Function that fits the econometric model
        dependent_var : str
            Name of dependent variable
        independent_vars : List[str]
            List of independent variable names
        subsample_strategies : Dict[str, Dict], optional
            Different subsampling strategies to test
            
        Returns:
        --------
        Dict[str, RobustnessResult]
            Results for each subsample
        """
        if subsample_strategies is None:
            subsample_strategies = {
                'first_half': {'method': 'split', 'fraction': 0.5, 'part': 'first'},
                'second_half': {'method': 'split', 'fraction': 0.5, 'part': 'second'},
                'exclude_crisis': {'method': 'exclude_dates', 'start': '2008-09-01', 'end': '2009-06-30'},
                'exclude_covid': {'method': 'exclude_dates', 'start': '2020-03-01', 'end': '2020-12-31'},
                'rolling_5yr': {'method': 'rolling', 'window_years': 5}
            }
        
        results = {}
        
        # Filter to QE period
        qe_data = self.base_data[self.base_data.index >= self.qe_start_date].copy()
        
        for strategy_name, strategy_params in subsample_strategies.items():
            try:
                method = strategy_params['method']
                
                if method == 'split':
                    # Split sample
                    n_obs = len(qe_data)
                    split_point = int(n_obs * strategy_params['fraction'])
                    
                    if strategy_params['part'] == 'first':
                        subsample_data = qe_data.iloc[:split_point]
                    else:
                        subsample_data = qe_data.iloc[split_point:]
                        
                elif method == 'exclude_dates':
                    # Exclude specific date range
                    exclude_start = pd.to_datetime(strategy_params['start'])
                    exclude_end = pd.to_datetime(strategy_params['end'])
                    
                    subsample_data = qe_data[
                        ~((qe_data.index >= exclude_start) & (qe_data.index <= exclude_end))
                    ]
                    
                elif method == 'rolling':
                    # Rolling window analysis (use most recent window)
                    window_years = strategy_params['window_years']
                    end_date = qe_data.index.max()
                    start_date = end_date - pd.DateOffset(years=window_years)
                    
                    subsample_data = qe_data[qe_data.index >= start_date]
                    
                else:
                    warnings.warn(f"Unknown subsample method: {method}")
                    continue
                
                if len(subsample_data) < 20:  # Minimum sample size check
                    warnings.warn(f"Insufficient data for subsample {strategy_name}: {len(subsample_data)} observations")
                    continue
                
                # Fit model on subsample
                model_result = model_func(
                    data=subsample_data,
                    dependent_var=dependent_var,
                    independent_vars=independent_vars
                )
                
                # Extract key coefficient
                main_coef = model_result.params[independent_vars[0]]
                main_se = model_result.bse[independent_vars[0]]
                main_pval = model_result.pvalues[independent_vars[0]]
                
                # Calculate confidence interval
                ci_lower = main_coef - 1.96 * main_se
                ci_upper = main_coef + 1.96 * main_se
                
                results[strategy_name] = RobustnessResult(
                    test_name="subsample_robustness",
                    specification=strategy_name,
                    coefficient=main_coef,
                    std_error=main_se,
                    p_value=main_pval,
                    confidence_interval=(ci_lower, ci_upper),
                    sample_size=len(subsample_data),
                    r_squared=getattr(model_result, 'rsquared', np.nan),
                    additional_stats={
                        'subsample_method': method,
                        'subsample_start': subsample_data.index.min(),
                        'subsample_end': subsample_data.index.max(),
                        'strategy_params': strategy_params
                    }
                )
                
            except Exception as e:
                warnings.warn(f"Error in subsample robustness test for {strategy_name}: {str(e)}")
                continue
        
        self.results['subsample_robustness'] = results
        return results
    
    def generate_robustness_summary(self) -> pd.DataFrame:
        """
        Generate comprehensive summary of all robustness tests
        
        Returns:
        --------
        pd.DataFrame
            Summary table of all robustness test results
        """
        summary_data = []
        
        for test_type, test_results in self.results.items():
            for spec_name, result in test_results.items():
                summary_data.append({
                    'test_type': result.test_name,
                    'specification': result.specification,
                    'coefficient': result.coefficient,
                    'std_error': result.std_error,
                    'p_value': result.p_value,
                    'ci_lower': result.confidence_interval[0],
                    'ci_upper': result.confidence_interval[1],
                    'sample_size': result.sample_size,
                    'r_squared': result.r_squared,
                    'significant_5pct': result.p_value < 0.05,
                    'significant_1pct': result.p_value < 0.01
                })
        
        return pd.DataFrame(summary_data)
    
    def test_coefficient_stability(self, test_type: str, significance_level: float = 0.05) -> Dict[str, Any]:
        """
        Test stability of coefficients across specifications within a test type
        
        Parameters:
        -----------
        test_type : str
            Type of robustness test to analyze
        significance_level : float
            Significance level for stability tests
            
        Returns:
        --------
        Dict[str, Any]
            Stability test results
        """
        if test_type not in self.results:
            raise ValueError(f"Test type {test_type} not found in results")
        
        results = self.results[test_type]
        coefficients = [r.coefficient for r in results.values()]
        std_errors = [r.std_error for r in results.values()]
        
        if len(coefficients) < 2:
            return {'error': 'Insufficient specifications for stability test'}
        
        # Calculate coefficient statistics
        coef_mean = np.mean(coefficients)
        coef_std = np.std(coefficients)
        coef_min = np.min(coefficients)
        coef_max = np.max(coefficients)
        coef_range = coef_max - coef_min
        
        # Test for coefficient equality (simple F-test approximation)
        # This is a simplified version - in practice you'd want more sophisticated tests
        weighted_mean = np.average(coefficients, weights=[1/se**2 for se in std_errors])
        chi_squared_stat = sum([(coef - weighted_mean)**2 / se**2 for coef, se in zip(coefficients, std_errors)])
        p_value_stability = 1 - stats.chi2.cdf(chi_squared_stat, len(coefficients) - 1)
        
        return {
            'test_type': test_type,
            'num_specifications': len(coefficients),
            'coefficient_mean': coef_mean,
            'coefficient_std': coef_std,
            'coefficient_min': coef_min,
            'coefficient_max': coef_max,
            'coefficient_range': coef_range,
            'relative_range': coef_range / abs(coef_mean) if coef_mean != 0 else np.inf,
            'weighted_mean': weighted_mean,
            'stability_test_statistic': chi_squared_stat,
            'stability_p_value': p_value_stability,
            'coefficients_stable': p_value_stability > significance_level
        }


class CrossValidationFramework:
    """
    Cross-validation framework for out-of-sample testing of QE models
    
    Provides time series cross-validation, rolling window validation,
    and holdout sample validation for predictive model assessment.
    """
    
    def __init__(self, data: pd.DataFrame, target_variable: str):
        """
        Initialize cross-validation framework
        
        Parameters:
        -----------
        data : pd.DataFrame
            Time series data with datetime index
        target_variable : str
            Name of target variable for prediction
        """
        self.data = data.copy()
        self.target_variable = target_variable
        
        # Ensure datetime index
        if 'date' in self.data.columns:
            self.data['date'] = pd.to_datetime(self.data['date'])
            self.data = self.data.set_index('date')
        
        self.cv_results = {}
    
    def time_series_cross_validation(self,
                                   model_func: Callable,
                                   feature_vars: List[str],
                                   n_splits: int = 5,
                                   test_size_months: int = 12) -> Dict[str, Any]:
        """
        Perform time series cross-validation with expanding window
        
        Parameters:
        -----------
        model_func : Callable
            Function that fits and predicts with the model
        feature_vars : List[str]
            List of feature variable names
        n_splits : int
            Number of cross-validation splits
        test_size_months : int
            Size of test set in months
            
        Returns:
        --------
        Dict[str, Any]
            Cross-validation results including metrics and predictions
        """
        cv_scores = []
        predictions = []
        actuals = []
        
        data_array = self.data[feature_vars + [self.target_variable]].dropna()
        
        # Create time series splits
        # For monthly data, test_size should be in number of observations, not days
        test_size_obs = min(test_size_months, len(data_array) // (n_splits + 1))
        tscv = TimeSeriesSplit(n_splits=n_splits, test_size=test_size_obs)
        
        for fold, (train_idx, test_idx) in enumerate(tscv.split(data_array)):
            try:
                # Split data
                train_data = data_array.iloc[train_idx]
                test_data = data_array.iloc[test_idx]
                
                # Fit model on training data
                model = model_func(
                    train_data=train_data,
                    feature_vars=feature_vars,
                    target_var=self.target_variable
                )
                
                # Make predictions on test data
                test_features = test_data[feature_vars]
                test_targets = test_data[self.target_variable]
                
                pred = model.predict(test_features)
                
                # Calculate metrics
                mse = mean_squared_error(test_targets, pred)
                mae = mean_absolute_error(test_targets, pred)
                rmse = np.sqrt(mse)
                
                # Calculate R-squared for out-of-sample
                ss_res = np.sum((test_targets - pred) ** 2)
                ss_tot = np.sum((test_targets - np.mean(test_targets)) ** 2)
                r2_oos = 1 - (ss_res / ss_tot) if ss_tot != 0 else np.nan
                
                cv_scores.append({
                    'fold': fold,
                    'train_start': train_data.index.min(),
                    'train_end': train_data.index.max(),
                    'test_start': test_data.index.min(),
                    'test_end': test_data.index.max(),
                    'train_size': len(train_data),
                    'test_size': len(test_data),
                    'mse': mse,
                    'mae': mae,
                    'rmse': rmse,
                    'r2_oos': r2_oos
                })
                
                # Store predictions for analysis
                for i, (actual, predicted) in enumerate(zip(test_targets, pred)):
                    predictions.append({
                        'fold': fold,
                        'date': test_data.index[i],
                        'actual': actual,
                        'predicted': predicted,
                        'error': actual - predicted
                    })
                
            except Exception as e:
                warnings.warn(f"Error in CV fold {fold}: {str(e)}")
                continue
        
        # Calculate overall statistics
        overall_metrics = {
            'mean_mse': np.mean([score['mse'] for score in cv_scores]),
            'std_mse': np.std([score['mse'] for score in cv_scores]),
            'mean_mae': np.mean([score['mae'] for score in cv_scores]),
            'std_mae': np.std([score['mae'] for score in cv_scores]),
            'mean_rmse': np.mean([score['rmse'] for score in cv_scores]),
            'std_rmse': np.std([score['rmse'] for score in cv_scores]),
            'mean_r2_oos': np.mean([score['r2_oos'] for score in cv_scores if not np.isnan(score['r2_oos'])]),
            'std_r2_oos': np.std([score['r2_oos'] for score in cv_scores if not np.isnan(score['r2_oos'])])
        }
        
        results = {
            'method': 'time_series_cross_validation',
            'n_splits': n_splits,
            'test_size_months': test_size_months,
            'fold_results': cv_scores,
            'predictions': predictions,
            'overall_metrics': overall_metrics,
            'feature_vars': feature_vars,
            'target_var': self.target_variable
        }
        
        self.cv_results['time_series_cv'] = results
        return results  
  
    def rolling_window_validation(self,
                                 model_func: Callable,
                                 feature_vars: List[str],
                                 window_size_months: int = 60,
                                 step_size_months: int = 12) -> Dict[str, Any]:
        """
        Perform rolling window validation for dynamic model performance assessment
        
        Parameters:
        -----------
        model_func : Callable
            Function that fits and predicts with the model
        feature_vars : List[str]
            List of feature variable names
        window_size_months : int
            Size of training window in months
        step_size_months : int
            Step size between windows in months
            
        Returns:
        --------
        Dict[str, Any]
            Rolling window validation results
        """
        data_clean = self.data[feature_vars + [self.target_variable]].dropna()
        
        # Calculate window parameters
        window_size_days = window_size_months * 30  # Approximate
        step_size_days = step_size_months * 30
        
        results = []
        predictions = []
        
        start_date = data_clean.index.min()
        end_date = data_clean.index.max()
        
        current_start = start_date
        
        while current_start + pd.Timedelta(days=window_size_days + 30) <= end_date:
            try:
                # Define training and test windows
                train_end = current_start + pd.Timedelta(days=window_size_days)
                test_start = train_end
                test_end = test_start + pd.Timedelta(days=30)  # 1 month ahead prediction
                
                # Extract data
                train_data = data_clean[
                    (data_clean.index >= current_start) & 
                    (data_clean.index < train_end)
                ]
                
                test_data = data_clean[
                    (data_clean.index >= test_start) & 
                    (data_clean.index < test_end)
                ]
                
                if len(train_data) < 24 or len(test_data) == 0:  # Minimum 2 years training
                    current_start += pd.Timedelta(days=step_size_days)
                    continue
                
                # Fit model
                model = model_func(
                    train_data=train_data,
                    feature_vars=feature_vars,
                    target_var=self.target_variable
                )
                
                # Make predictions
                test_features = test_data[feature_vars]
                test_targets = test_data[self.target_variable]
                
                pred = model.predict(test_features)
                
                # Calculate metrics
                mse = mean_squared_error(test_targets, pred)
                mae = mean_absolute_error(test_targets, pred)
                rmse = np.sqrt(mse)
                
                # Directional accuracy (for financial time series)
                if len(test_targets) > 1:
                    actual_direction = np.sign(test_targets.diff().dropna())
                    pred_direction = np.sign(pd.Series(pred).diff().dropna())
                    directional_accuracy = np.mean(actual_direction == pred_direction)
                else:
                    directional_accuracy = np.nan
                
                results.append({
                    'train_start': current_start,
                    'train_end': train_end,
                    'test_start': test_start,
                    'test_end': test_end,
                    'train_size': len(train_data),
                    'test_size': len(test_data),
                    'mse': mse,
                    'mae': mae,
                    'rmse': rmse,
                    'directional_accuracy': directional_accuracy
                })
                
                # Store individual predictions
                for i, (actual, predicted) in enumerate(zip(test_targets, pred)):
                    predictions.append({
                        'window_start': current_start,
                        'date': test_data.index[i],
                        'actual': actual,
                        'predicted': predicted,
                        'error': actual - predicted
                    })
                
            except Exception as e:
                warnings.warn(f"Error in rolling window starting {current_start}: {str(e)}")
            
            current_start += pd.Timedelta(days=step_size_days)
        
        # Calculate summary statistics
        if results:
            summary_metrics = {
                'mean_mse': np.mean([r['mse'] for r in results]),
                'std_mse': np.std([r['mse'] for r in results]),
                'mean_mae': np.mean([r['mae'] for r in results]),
                'std_mae': np.std([r['mae'] for r in results]),
                'mean_rmse': np.mean([r['rmse'] for r in results]),
                'std_rmse': np.std([r['rmse'] for r in results]),
                'mean_directional_accuracy': np.mean([r['directional_accuracy'] for r in results if not np.isnan(r['directional_accuracy'])]),
                'num_windows': len(results)
            }
        else:
            summary_metrics = {}
        
        rolling_results = {
            'method': 'rolling_window_validation',
            'window_size_months': window_size_months,
            'step_size_months': step_size_months,
            'window_results': results,
            'predictions': predictions,
            'summary_metrics': summary_metrics,
            'feature_vars': feature_vars,
            'target_var': self.target_variable
        }
        
        self.cv_results['rolling_window'] = rolling_results
        return rolling_results
    
    def holdout_sample_validation(self,
                                model_func: Callable,
                                feature_vars: List[str],
                                holdout_fraction: float = 0.2,
                                holdout_method: str = 'recent') -> Dict[str, Any]:
        """
        Perform holdout sample validation for final model assessment
        
        Parameters:
        -----------
        model_func : Callable
            Function that fits and predicts with the model
        feature_vars : List[str]
            List of feature variable names
        holdout_fraction : float
            Fraction of data to hold out for testing
        holdout_method : str
            Method for selecting holdout sample ('recent', 'random', 'middle')
            
        Returns:
        --------
        Dict[str, Any]
            Holdout validation results
        """
        data_clean = self.data[feature_vars + [self.target_variable]].dropna()
        n_obs = len(data_clean)
        holdout_size = int(n_obs * holdout_fraction)
        
        if holdout_method == 'recent':
            # Use most recent data as holdout
            train_data = data_clean.iloc[:-holdout_size]
            test_data = data_clean.iloc[-holdout_size:]
            
        elif holdout_method == 'random':
            # Random holdout (not recommended for time series, but included for completeness)
            np.random.seed(42)  # For reproducibility
            test_indices = np.random.choice(n_obs, holdout_size, replace=False)
            train_indices = np.setdiff1d(np.arange(n_obs), test_indices)
            
            train_data = data_clean.iloc[train_indices]
            test_data = data_clean.iloc[test_indices]
            
        elif holdout_method == 'middle':
            # Use middle portion as holdout (for testing structural stability)
            start_idx = (n_obs - holdout_size) // 2
            end_idx = start_idx + holdout_size
            
            train_data = pd.concat([
                data_clean.iloc[:start_idx],
                data_clean.iloc[end_idx:]
            ])
            test_data = data_clean.iloc[start_idx:end_idx]
            
        else:
            raise ValueError(f"Unknown holdout method: {holdout_method}")
        
        try:
            # Fit model on training data
            model = model_func(
                train_data=train_data,
                feature_vars=feature_vars,
                target_var=self.target_variable
            )
            
            # Make predictions on test data
            test_features = test_data[feature_vars]
            test_targets = test_data[self.target_variable]
            
            pred = model.predict(test_features)
            
            # Calculate comprehensive metrics
            mse = mean_squared_error(test_targets, pred)
            mae = mean_absolute_error(test_targets, pred)
            rmse = np.sqrt(mse)
            
            # R-squared out-of-sample
            ss_res = np.sum((test_targets - pred) ** 2)
            ss_tot = np.sum((test_targets - np.mean(test_targets)) ** 2)
            r2_oos = 1 - (ss_res / ss_tot) if ss_tot != 0 else np.nan
            
            # Mean absolute percentage error
            mape = np.mean(np.abs((test_targets - pred) / test_targets)) * 100
            
            # Directional accuracy
            if len(test_targets) > 1:
                actual_direction = np.sign(test_targets.diff().dropna())
                pred_direction = np.sign(pd.Series(pred).diff().dropna())
                directional_accuracy = np.mean(actual_direction == pred_direction)
            else:
                directional_accuracy = np.nan
            
            # Prediction intervals (if model supports it)
            try:
                pred_intervals = model.predict_intervals(test_features, alpha=0.05)
                coverage_rate = np.mean(
                    (test_targets >= pred_intervals[:, 0]) & 
                    (test_targets <= pred_intervals[:, 1])
                )
            except:
                pred_intervals = None
                coverage_rate = np.nan
            
            # Store individual predictions
            predictions = []
            for i, (actual, predicted) in enumerate(zip(test_targets, pred)):
                pred_dict = {
                    'date': test_data.index[i],
                    'actual': actual,
                    'predicted': predicted,
                    'error': actual - predicted,
                    'abs_error': abs(actual - predicted),
                    'pct_error': ((actual - predicted) / actual * 100) if actual != 0 else np.nan
                }
                
                if pred_intervals is not None:
                    pred_dict['lower_bound'] = pred_intervals[i, 0]
                    pred_dict['upper_bound'] = pred_intervals[i, 1]
                    pred_dict['in_interval'] = (actual >= pred_intervals[i, 0]) and (actual <= pred_intervals[i, 1])
                
                predictions.append(pred_dict)
            
            results = {
                'method': 'holdout_sample_validation',
                'holdout_method': holdout_method,
                'holdout_fraction': holdout_fraction,
                'train_size': len(train_data),
                'test_size': len(test_data),
                'train_period': (train_data.index.min(), train_data.index.max()),
                'test_period': (test_data.index.min(), test_data.index.max()),
                'metrics': {
                    'mse': mse,
                    'mae': mae,
                    'rmse': rmse,
                    'r2_oos': r2_oos,
                    'mape': mape,
                    'directional_accuracy': directional_accuracy,
                    'coverage_rate': coverage_rate
                },
                'predictions': predictions,
                'feature_vars': feature_vars,
                'target_var': self.target_variable
            }
            
        except Exception as e:
            results = {
                'method': 'holdout_sample_validation',
                'error': str(e),
                'holdout_method': holdout_method,
                'holdout_fraction': holdout_fraction
            }
        
        self.cv_results['holdout'] = results
        return results
    
    def generate_cv_summary(self) -> pd.DataFrame:
        """
        Generate summary of all cross-validation results
        
        Returns:
        --------
        pd.DataFrame
            Summary of cross-validation metrics
        """
        summary_data = []
        
        for cv_method, cv_result in self.cv_results.items():
            if 'error' in cv_result:
                continue
                
            if cv_method == 'time_series_cv':
                metrics = cv_result['overall_metrics']
                summary_data.append({
                    'method': 'Time Series CV',
                    'n_folds': cv_result['n_splits'],
                    'mean_mse': metrics.get('mean_mse', np.nan),
                    'std_mse': metrics.get('std_mse', np.nan),
                    'mean_mae': metrics.get('mean_mae', np.nan),
                    'mean_r2_oos': metrics.get('mean_r2_oos', np.nan),
                    'std_r2_oos': metrics.get('std_r2_oos', np.nan)
                })
                
            elif cv_method == 'rolling_window':
                metrics = cv_result['summary_metrics']
                summary_data.append({
                    'method': 'Rolling Window',
                    'n_windows': metrics.get('num_windows', 0),
                    'mean_mse': metrics.get('mean_mse', np.nan),
                    'std_mse': metrics.get('std_mse', np.nan),
                    'mean_mae': metrics.get('mean_mae', np.nan),
                    'mean_directional_accuracy': metrics.get('mean_directional_accuracy', np.nan),
                    'window_size_months': cv_result['window_size_months']
                })
                
            elif cv_method == 'holdout':
                if 'metrics' in cv_result:
                    metrics = cv_result['metrics']
                    summary_data.append({
                        'method': f"Holdout ({cv_result['holdout_method']})",
                        'test_size': cv_result['test_size'],
                        'mse': metrics.get('mse', np.nan),
                        'mae': metrics.get('mae', np.nan),
                        'r2_oos': metrics.get('r2_oos', np.nan),
                        'mape': metrics.get('mape', np.nan),
                        'directional_accuracy': metrics.get('directional_accuracy', np.nan),
                        'coverage_rate': metrics.get('coverage_rate', np.nan)
                    })
        
        return pd.DataFrame(summary_data)


class SensitivityAnalyzer:
    """
    Sensitivity analysis framework for testing robustness to parameter and assumption changes
    
    Provides systematic testing of key parameters, thresholds, and modeling assumptions
    to ensure robust empirical findings.
    """
    
    def __init__(self, base_data: pd.DataFrame, base_model_func: Callable):
        """
        Initialize sensitivity analyzer
        
        Parameters:
        -----------
        base_data : pd.DataFrame
            Base dataset for analysis
        base_model_func : Callable
            Base model function for sensitivity testing
        """
        self.base_data = base_data.copy()
        self.base_model_func = base_model_func
        self.sensitivity_results = {}
        
        # Ensure datetime index
        if 'date' in self.base_data.columns:
            self.base_data['date'] = pd.to_datetime(self.base_data['date'])
            self.base_data = self.base_data.set_index('date')
    
    def threshold_sensitivity_test(self,
                                 threshold_variable: str,
                                 base_threshold: float = 0.003,  # 0.3%
                                 threshold_range: Tuple[float, float] = (0.001, 0.006),
                                 n_thresholds: int = 11,
                                 dependent_var: str = 'investment_growth') -> Dict[str, Any]:
        """
        Test sensitivity of results to threshold parameter changes
        
        Parameters:
        -----------
        threshold_variable : str
            Name of threshold variable (e.g., 'qe_intensity')
        base_threshold : float
            Base threshold value (0.3% = 0.003)
        threshold_range : Tuple[float, float]
            Range of thresholds to test
        n_thresholds : int
            Number of threshold values to test
        dependent_var : str
            Dependent variable for threshold regression
            
        Returns:
        --------
        Dict[str, Any]
            Threshold sensitivity results
        """
        # Generate threshold values to test
        threshold_values = np.linspace(threshold_range[0], threshold_range[1], n_thresholds)
        
        results = []
        
        for threshold in threshold_values:
            try:
                # Create threshold dummy variable
                data_with_threshold = self.base_data.copy()
                data_with_threshold['threshold_dummy'] = (
                    data_with_threshold[threshold_variable] > threshold
                ).astype(int)
                
                # Fit threshold model
                model_result = self.base_model_func(
                    data=data_with_threshold,
                    dependent_var=dependent_var,
                    threshold_var=threshold_variable,
                    threshold_value=threshold
                )
                
                # Extract key statistics
                if hasattr(model_result, 'params'):
                    # Linear threshold model
                    threshold_coef = model_result.params.get('threshold_dummy', np.nan)
                    threshold_se = model_result.bse.get('threshold_dummy', np.nan)
                    threshold_pval = model_result.pvalues.get('threshold_dummy', np.nan)
                    r_squared = getattr(model_result, 'rsquared', np.nan)
                else:
                    # Hansen threshold model
                    threshold_coef = getattr(model_result, 'threshold_effect', np.nan)
                    threshold_se = getattr(model_result, 'threshold_se', np.nan)
                    threshold_pval = getattr(model_result, 'threshold_pvalue', np.nan)
                    r_squared = getattr(model_result, 'rsquared', np.nan)
                
                # Calculate observations in each regime
                low_regime_obs = np.sum(data_with_threshold[threshold_variable] <= threshold)
                high_regime_obs = np.sum(data_with_threshold[threshold_variable] > threshold)
                
                results.append({
                    'threshold_value': threshold,
                    'threshold_coef': threshold_coef,
                    'threshold_se': threshold_se,
                    'threshold_pval': threshold_pval,
                    'r_squared': r_squared,
                    'low_regime_obs': low_regime_obs,
                    'high_regime_obs': high_regime_obs,
                    'regime_balance': min(low_regime_obs, high_regime_obs) / max(low_regime_obs, high_regime_obs),
                    'significant_5pct': threshold_pval < 0.05 if not np.isnan(threshold_pval) else False,
                    'significant_1pct': threshold_pval < 0.01 if not np.isnan(threshold_pval) else False
                })
                
            except Exception as e:
                warnings.warn(f"Error testing threshold {threshold}: {str(e)}")
                continue
        
        # Calculate sensitivity statistics
        valid_results = [r for r in results if not np.isnan(r['threshold_coef'])]
        
        if valid_results:
            coefficients = [r['threshold_coef'] for r in valid_results]
            p_values = [r['threshold_pval'] for r in valid_results]
            
            sensitivity_stats = {
                'coefficient_range': max(coefficients) - min(coefficients),
                'coefficient_std': np.std(coefficients),
                'coefficient_cv': np.std(coefficients) / abs(np.mean(coefficients)) if np.mean(coefficients) != 0 else np.inf,
                'significant_fraction': np.mean([r['significant_5pct'] for r in valid_results]),
                'base_threshold_rank': None,
                'optimal_threshold': None
            }
            
            # Find base threshold rank
            base_result = next((r for r in valid_results if abs(r['threshold_value'] - base_threshold) < 1e-6), None)
            if base_result:
                base_coef = base_result['threshold_coef']
                sensitivity_stats['base_threshold_rank'] = sum(1 for r in valid_results if abs(r['threshold_coef']) > abs(base_coef)) + 1
            
            # Find optimal threshold (highest absolute coefficient with good regime balance)
            balanced_results = [r for r in valid_results if r['regime_balance'] > 0.1]  # At least 10% in each regime
            if balanced_results:
                optimal_result = max(balanced_results, key=lambda x: abs(x['threshold_coef']))
                sensitivity_stats['optimal_threshold'] = optimal_result['threshold_value']
        else:
            sensitivity_stats = {}
        
        threshold_results = {
            'test_type': 'threshold_sensitivity',
            'threshold_variable': threshold_variable,
            'base_threshold': base_threshold,
            'threshold_range': threshold_range,
            'n_thresholds': n_thresholds,
            'results': results,
            'sensitivity_stats': sensitivity_stats,
            'dependent_var': dependent_var
        }
        
        self.sensitivity_results['threshold_sensitivity'] = threshold_results
        return threshold_results
    
    def channel_decomposition_sensitivity_test(self,
                                             base_split: Tuple[float, float] = (0.6, 0.4),
                                             split_range: Tuple[float, float] = (0.4, 0.8),
                                             n_splits: int = 9,
                                             dependent_var: str = 'investment_growth') -> Dict[str, Any]:
        """
        Test sensitivity of channel decomposition to different splits
        
        Parameters:
        -----------
        base_split : Tuple[float, float]
            Base channel split (market distortion, interest rate)
        split_range : Tuple[float, float]
            Range for market distortion channel share
        n_splits : int
            Number of splits to test
        dependent_var : str
            Dependent variable for decomposition
            
        Returns:
        --------
        Dict[str, Any]
            Channel decomposition sensitivity results
        """
        # Generate split values to test (market distortion channel share)
        market_distortion_shares = np.linspace(split_range[0], split_range[1], n_splits)
        
        results = []
        
        for md_share in market_distortion_shares:
            ir_share = 1.0 - md_share  # Interest rate channel share
            
            try:
                # Create decomposed variables
                data_decomposed = self.base_data.copy()
                
                # Assume we have market distortion and interest rate channel variables
                if 'market_distortion_channel' in data_decomposed.columns and 'interest_rate_channel' in data_decomposed.columns:
                    data_decomposed['combined_effect'] = (
                        md_share * data_decomposed['market_distortion_channel'] +
                        ir_share * data_decomposed['interest_rate_channel']
                    )
                else:
                    # Create synthetic decomposition for testing
                    data_decomposed['combined_effect'] = (
                        md_share * data_decomposed.get('qe_intensity', 0) +
                        ir_share * data_decomposed.get('yield_change', 0)
                    )
                
                # Fit model with decomposed effect
                model_result = self.base_model_func(
                    data=data_decomposed,
                    dependent_var=dependent_var,
                    independent_vars=['combined_effect'],
                    decomposition_weights=(md_share, ir_share)
                )
                
                # Extract results
                combined_coef = model_result.params.get('combined_effect', np.nan)
                combined_se = model_result.bse.get('combined_effect', np.nan)
                combined_pval = model_result.pvalues.get('combined_effect', np.nan)
                r_squared = getattr(model_result, 'rsquared', np.nan)
                
                # Calculate implied individual channel effects
                if not np.isnan(combined_coef):
                    implied_md_effect = combined_coef * md_share
                    implied_ir_effect = combined_coef * ir_share
                else:
                    implied_md_effect = np.nan
                    implied_ir_effect = np.nan
                
                results.append({
                    'md_share': md_share,
                    'ir_share': ir_share,
                    'combined_coef': combined_coef,
                    'combined_se': combined_se,
                    'combined_pval': combined_pval,
                    'r_squared': r_squared,
                    'implied_md_effect': implied_md_effect,
                    'implied_ir_effect': implied_ir_effect,
                    'significant_5pct': combined_pval < 0.05 if not np.isnan(combined_pval) else False,
                    'significant_1pct': combined_pval < 0.01 if not np.isnan(combined_pval) else False
                })
                
            except Exception as e:
                warnings.warn(f"Error testing split MD:{md_share:.1%}/IR:{ir_share:.1%}: {str(e)}")
                continue
        
        # Calculate sensitivity statistics
        valid_results = [r for r in results if not np.isnan(r['combined_coef'])]
        
        if valid_results:
            combined_coefs = [r['combined_coef'] for r in valid_results]
            md_effects = [r['implied_md_effect'] for r in valid_results if not np.isnan(r['implied_md_effect'])]
            ir_effects = [r['implied_ir_effect'] for r in valid_results if not np.isnan(r['implied_ir_effect'])]
            
            sensitivity_stats = {
                'combined_coef_range': max(combined_coefs) - min(combined_coefs),
                'combined_coef_std': np.std(combined_coefs),
                'md_effect_range': max(md_effects) - min(md_effects) if md_effects else np.nan,
                'ir_effect_range': max(ir_effects) - min(ir_effects) if ir_effects else np.nan,
                'significant_fraction': np.mean([r['significant_5pct'] for r in valid_results]),
                'base_split_rank': None,
                'optimal_split': None
            }
            
            # Find base split rank
            base_md_share = base_split[0]
            base_result = next((r for r in valid_results if abs(r['md_share'] - base_md_share) < 0.01), None)
            if base_result:
                base_coef = abs(base_result['combined_coef'])
                sensitivity_stats['base_split_rank'] = sum(1 for r in valid_results if abs(r['combined_coef']) > base_coef) + 1
            
            # Find optimal split (highest significance and effect size)
            significant_results = [r for r in valid_results if r['significant_5pct']]
            if significant_results:
                optimal_result = max(significant_results, key=lambda x: abs(x['combined_coef']))
                sensitivity_stats['optimal_split'] = (optimal_result['md_share'], optimal_result['ir_share'])
        else:
            sensitivity_stats = {}
        
        decomposition_results = {
            'test_type': 'channel_decomposition_sensitivity',
            'base_split': base_split,
            'split_range': split_range,
            'n_splits': n_splits,
            'results': results,
            'sensitivity_stats': sensitivity_stats,
            'dependent_var': dependent_var
        }
        
        self.sensitivity_results['channel_decomposition_sensitivity'] = decomposition_results
        return decomposition_results
    
    def instrument_choice_sensitivity_test(self,
                                         instrument_sets: Dict[str, List[str]],
                                         dependent_var: str,
                                         endogenous_vars: List[str]) -> Dict[str, Any]:
        """
        Test sensitivity to different instrument choices for identification
        
        Parameters:
        -----------
        instrument_sets : Dict[str, List[str]]
            Different sets of instruments to test
        dependent_var : str
            Dependent variable for IV regression
        endogenous_vars : List[str]
            Endogenous variables
            
        Returns:
        --------
        Dict[str, Any]
            Instrument choice sensitivity results
        """
        results = []
        
        for instrument_name, instruments in instrument_sets.items():
            try:
                # Check instrument availability
                available_instruments = [inst for inst in instruments if inst in self.base_data.columns]
                if len(available_instruments) < len(endogenous_vars):
                    warnings.warn(f"Insufficient instruments for {instrument_name}: need {len(endogenous_vars)}, have {len(available_instruments)}")
                    continue
                
                # Fit IV model
                model_result = self.base_model_func(
                    data=self.base_data,
                    dependent_var=dependent_var,
                    endogenous_vars=endogenous_vars,
                    instruments=available_instruments,
                    model_type='iv'
                )
                
                # Extract main coefficient (first endogenous variable)
                main_var = endogenous_vars[0]
                main_coef = model_result.params.get(main_var, np.nan)
                main_se = model_result.std_errors.get(main_var, np.nan)
                main_pval = model_result.pvalues.get(main_var, np.nan)
                
                # Get instrument diagnostics
                first_stage_f = getattr(model_result, 'first_stage_f_statistic', np.nan)
                j_statistic = getattr(model_result, 'j_statistic', np.nan)
                j_pvalue = getattr(model_result, 'j_pvalue', np.nan)
                
                # Weak instrument test
                weak_instrument = first_stage_f < 10 if not np.isnan(first_stage_f) else True
                
                # Overidentification test (if overidentified)
                overidentified = len(available_instruments) > len(endogenous_vars)
                overid_rejected = j_pvalue < 0.05 if not np.isnan(j_pvalue) and overidentified else False
                
                results.append({
                    'instrument_set': instrument_name,
                    'instruments': available_instruments,
                    'num_instruments': len(available_instruments),
                    'main_coef': main_coef,
                    'main_se': main_se,
                    'main_pval': main_pval,
                    'first_stage_f': first_stage_f,
                    'j_statistic': j_statistic,
                    'j_pvalue': j_pvalue,
                    'weak_instrument': weak_instrument,
                    'overid_rejected': overid_rejected,
                    'overidentified': overidentified,
                    'significant_5pct': main_pval < 0.05 if not np.isnan(main_pval) else False,
                    'significant_1pct': main_pval < 0.01 if not np.isnan(main_pval) else False,
                    'valid_instruments': not weak_instrument and not overid_rejected
                })
                
            except Exception as e:
                warnings.warn(f"Error testing instruments {instrument_name}: {str(e)}")
                continue
        
        # Calculate sensitivity statistics
        valid_results = [r for r in results if not np.isnan(r['main_coef']) and r['valid_instruments']]
        all_results = [r for r in results if not np.isnan(r['main_coef'])]
        
        if valid_results:
            valid_coefs = [r['main_coef'] for r in valid_results]
            all_coefs = [r['main_coef'] for r in all_results]
            
            sensitivity_stats = {
                'valid_instruments_fraction': len(valid_results) / len(results) if results else 0,
                'valid_coef_range': max(valid_coefs) - min(valid_coefs),
                'valid_coef_std': np.std(valid_coefs),
                'valid_coef_cv': np.std(valid_coefs) / abs(np.mean(valid_coefs)) if np.mean(valid_coefs) != 0 else np.inf,
                'all_coef_range': max(all_coefs) - min(all_coefs) if all_coefs else np.nan,
                'significant_fraction': np.mean([r['significant_5pct'] for r in valid_results]),
                'mean_first_stage_f': np.mean([r['first_stage_f'] for r in valid_results if not np.isnan(r['first_stage_f'])]),
                'best_instrument_set': None
            }
            
            # Find best instrument set (highest F-stat among significant results)
            significant_valid = [r for r in valid_results if r['significant_5pct']]
            if significant_valid:
                best_result = max(significant_valid, key=lambda x: x['first_stage_f'] if not np.isnan(x['first_stage_f']) else 0)
                sensitivity_stats['best_instrument_set'] = best_result['instrument_set']
        else:
            sensitivity_stats = {
                'valid_instruments_fraction': 0,
                'error': 'No valid instrument sets found'
            }
        
        instrument_results = {
            'test_type': 'instrument_choice_sensitivity',
            'instrument_sets': list(instrument_sets.keys()),
            'dependent_var': dependent_var,
            'endogenous_vars': endogenous_vars,
            'results': results,
            'sensitivity_stats': sensitivity_stats
        }
        
        self.sensitivity_results['instrument_choice_sensitivity'] = instrument_results
        return instrument_results
    
    def generate_sensitivity_summary(self) -> pd.DataFrame:
        """
        Generate comprehensive summary of all sensitivity analyses
        
        Returns:
        --------
        pd.DataFrame
            Summary of sensitivity test results
        """
        summary_data = []
        
        for test_type, test_results in self.sensitivity_results.items():
            if test_type == 'threshold_sensitivity':
                stats = test_results['sensitivity_stats']
                summary_data.append({
                    'test_type': 'Threshold Sensitivity',
                    'parameter': f"Threshold ({test_results['threshold_variable']})",
                    'base_value': test_results['base_threshold'],
                    'test_range': f"{test_results['threshold_range'][0]:.3f} - {test_results['threshold_range'][1]:.3f}",
                    'coefficient_range': stats.get('coefficient_range', np.nan),
                    'coefficient_cv': stats.get('coefficient_cv', np.nan),
                    'significant_fraction': stats.get('significant_fraction', np.nan),
                    'optimal_value': stats.get('optimal_threshold', np.nan)
                })
                
            elif test_type == 'channel_decomposition_sensitivity':
                stats = test_results['sensitivity_stats']
                summary_data.append({
                    'test_type': 'Channel Decomposition',
                    'parameter': 'Market Distortion Share',
                    'base_value': test_results['base_split'][0],
                    'test_range': f"{test_results['split_range'][0]:.1%} - {test_results['split_range'][1]:.1%}",
                    'coefficient_range': stats.get('combined_coef_range', np.nan),
                    'significant_fraction': stats.get('significant_fraction', np.nan),
                    'optimal_value': stats.get('optimal_split', [np.nan])[0] if stats.get('optimal_split') else np.nan
                })
                
            elif test_type == 'instrument_choice_sensitivity':
                stats = test_results['sensitivity_stats']
                summary_data.append({
                    'test_type': 'Instrument Choice',
                    'parameter': 'Instrument Set',
                    'num_sets_tested': len(test_results['instrument_sets']),
                    'valid_fraction': stats.get('valid_instruments_fraction', np.nan),
                    'coefficient_range': stats.get('valid_coef_range', np.nan),
                    'coefficient_cv': stats.get('valid_coef_cv', np.nan),
                    'significant_fraction': stats.get('significant_fraction', np.nan),
                    'best_set': stats.get('best_instrument_set', 'None')
                })
        
        return pd.DataFrame(summary_data)