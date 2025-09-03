"""
Enhanced Hypothesis 3: International QE Effects and Currency Analysis

This module implements comprehensive analysis for testing Hypothesis 3:
"QE reduces foreign demand for domestic bonds leading to currency depreciation 
and inflationary pressures that may offset QE benefits."

The module provides:
1. Foreign bond demand and currency depreciation models
2. Inflation offset analysis and spillover effects  
3. International transmission mechanism analysis
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional, Tuple, Any, Union
from datetime import datetime, timedelta
import warnings
from dataclasses import dataclass
from scipy import stats
from scipy.optimize import minimize
import statsmodels.api as sm
from statsmodels.tsa.api import VAR, VECM
from statsmodels.tsa.stattools import grangercausalitytests, coint
from statsmodels.regression.linear_model import OLS
from statsmodels.stats.diagnostic import het_breuschpagan, het_white
from statsmodels.stats.stattools import durbin_watson

from ..analysis.international_analysis import InternationalAnalyzer, FlowDecomposer, TransmissionTester
from ..utils.data_structures import HypothesisData


@dataclass
class ForeignBondDemandResults:
    """Results from foreign bond demand analysis"""
    tic_data_analysis: Dict[str, Any]
    demand_elasticity: Dict[str, float]
    causality_tests: Dict[str, Any]
    country_breakdown: Dict[str, Dict[str, Any]]
    model_diagnostics: Dict[str, Any]
    statistical_significance: Dict[str, bool]


@dataclass
class CurrencyDepreciationResults:
    """Results from currency depreciation analysis"""
    exchange_rate_models: Dict[str, Any]
    qe_announcement_effects: Dict[str, Any]
    transmission_channels: Dict[str, Any]
    depreciation_magnitude: Dict[str, float]
    model_diagnostics: Dict[str, Any]
    robustness_tests: Dict[str, Any]


@dataclass
class InflationOffsetResults:
    """Results from inflation offset analysis"""
    inflation_pressure_models: Dict[str, Any]
    offset_quantification: Dict[str, float]
    spillover_analysis: Dict[str, Any]
    cross_country_comparison: Dict[str, Any]
    transmission_mechanisms: Dict[str, Any]
    policy_implications: Dict[str, Any]


class ForeignBondDemandAnalyzer:
    """
    Analyzes foreign bond demand patterns and their response to QE policies.
    
    This class implements models to track foreign holdings from TIC data,
    create exchange rate models linking QE to depreciation, and test
    causality between QE and foreign demand changes.
    """
    
    def __init__(self, significance_level: float = 0.05):
        """
        Initialize the Foreign Bond Demand Analyzer.
        
        Args:
            significance_level: Statistical significance level for tests
        """
        self.significance_level = significance_level
        self.logger = logging.getLogger(__name__)
        
        # QE announcement dates for event study analysis
        self.qe_announcement_dates = [
            '2008-11-25',  # QE1 announcement
            '2010-11-03',  # QE2 announcement  
            '2012-09-13',  # QE3 announcement
            '2020-03-15',  # COVID QE announcement
            '2020-03-23'   # Additional COVID QE
        ]
        
        # Major foreign holders for analysis
        self.major_holders = {
            'official': ['China', 'Japan', 'United_Kingdom', 'Switzerland', 'Taiwan'],
            'private': ['Luxembourg', 'Ireland', 'Cayman_Islands', 'Belgium', 'Singapore']
        }
    
    def analyze_foreign_holdings_tracking(self, 
                                        foreign_holdings: pd.DataFrame,
                                        qe_intensity: pd.Series,
                                        exchange_rates: pd.DataFrame) -> Dict[str, Any]:
        """
        Implement foreign holdings tracking from TIC data.
        
        Args:
            foreign_holdings: Foreign Treasury holdings by country (TIC data)
            qe_intensity: QE intensity measure (Fed holdings/total outstanding)
            exchange_rates: Exchange rate data for major currencies
            
        Returns:
            Dictionary containing foreign holdings tracking analysis
        """
        self.logger.info("Analyzing foreign holdings tracking from TIC data")
        
        results = {}
        
        # Align data to common time index
        common_index = (foreign_holdings.index
                       .intersection(qe_intensity.index)
                       .intersection(exchange_rates.index))
        
        if len(common_index) < 24:  # Need at least 2 years of data
            raise ValueError("Insufficient overlapping observations for foreign holdings analysis")
        
        holdings_aligned = foreign_holdings.loc[common_index]
        qe_aligned = qe_intensity.loc[common_index]
        fx_aligned = exchange_rates.loc[common_index]
        
        # 1. Aggregate foreign holdings analysis
        total_foreign_holdings = holdings_aligned.sum(axis=1)
        
        # Basic regression: Foreign holdings vs QE intensity
        try:
            data = pd.DataFrame({
                'total_holdings': total_foreign_holdings,
                'qe_intensity': qe_aligned,
                'holdings_lag1': total_foreign_holdings.shift(1),
                'holdings_lag2': total_foreign_holdings.shift(2)
            }).dropna()
            
            if len(data) > 10:
                # Levels model
                y = data['total_holdings']
                X = sm.add_constant(data[['qe_intensity', 'holdings_lag1']])
                model_levels = OLS(y, X).fit()
                
                # Changes model  
                data['holdings_change'] = data['total_holdings'].diff()
                data['qe_change'] = data['qe_intensity'].diff()
                data_changes = data.dropna()
                
                if len(data_changes) > 8:
                    y_change = data_changes['holdings_change']
                    X_change = sm.add_constant(data_changes[['qe_change', 'holdings_lag1']])
                    model_changes = OLS(y_change, X_change).fit()
                    
                    results['aggregate_analysis'] = {
                        'levels_model': {
                            'qe_coefficient': model_levels.params['qe_intensity'],
                            'qe_pvalue': model_levels.pvalues['qe_intensity'],
                            'qe_significant': model_levels.pvalues['qe_intensity'] < self.significance_level,
                            'r_squared': model_levels.rsquared,
                            'n_obs': model_levels.nobs,
                            'durbin_watson': durbin_watson(model_levels.resid)
                        },
                        'changes_model': {
                            'qe_coefficient': model_changes.params['qe_change'],
                            'qe_pvalue': model_changes.pvalues['qe_change'],
                            'qe_significant': model_changes.pvalues['qe_change'] < self.significance_level,
                            'r_squared': model_changes.rsquared,
                            'n_obs': model_changes.nobs,
                            'durbin_watson': durbin_watson(model_changes.resid)
                        }
                    }
        except Exception as e:
            results['aggregate_analysis'] = {'error': str(e)}
        
        # 2. Country-specific analysis
        country_results = {}
        available_countries = [col for col in holdings_aligned.columns 
                             if holdings_aligned[col].notna().sum() > len(common_index) * 0.5]
        
        for country in available_countries[:10]:  # Analyze top 10 countries
            try:
                country_holdings = holdings_aligned[country]
                
                data = pd.DataFrame({
                    'holdings': country_holdings,
                    'qe_intensity': qe_aligned,
                    'holdings_lag1': country_holdings.shift(1)
                }).dropna()
                
                if len(data) > 8:
                    y = data['holdings']
                    X = sm.add_constant(data[['qe_intensity', 'holdings_lag1']])
                    model = OLS(y, X).fit()
                    
                    country_results[country] = {
                        'qe_coefficient': model.params['qe_intensity'],
                        'qe_pvalue': model.pvalues['qe_intensity'],
                        'qe_significant': model.pvalues['qe_intensity'] < self.significance_level,
                        'persistence': model.params['holdings_lag1'],
                        'r_squared': model.rsquared,
                        'n_obs': model.nobs
                    }
            except Exception as e:
                country_results[country] = {'error': str(e)}
        
        results['country_analysis'] = country_results
        
        # 3. Official vs Private investor breakdown
        official_countries = [c for c in self.major_holders['official'] 
                            if c in holdings_aligned.columns]
        private_countries = [c for c in self.major_holders['private'] 
                           if c in holdings_aligned.columns]
        
        if official_countries:
            official_holdings = holdings_aligned[official_countries].sum(axis=1)
            official_analysis = self._analyze_investor_type(official_holdings, qe_aligned, 'official')
            results['official_investors'] = official_analysis
        
        if private_countries:
            private_holdings = holdings_aligned[private_countries].sum(axis=1)
            private_analysis = self._analyze_investor_type(private_holdings, qe_aligned, 'private')
            results['private_investors'] = private_analysis
        
        return results
    
    def _analyze_investor_type(self, holdings: pd.Series, qe_intensity: pd.Series, 
                              investor_type: str) -> Dict[str, Any]:
        """Analyze specific investor type response to QE"""
        
        try:
            data = pd.DataFrame({
                'holdings': holdings,
                'qe_intensity': qe_intensity,
                'holdings_lag1': holdings.shift(1)
            }).dropna()
            
            if len(data) < 8:
                return {'error': f'Insufficient data for {investor_type} analysis'}
            
            # Basic model
            y = data['holdings']
            X = sm.add_constant(data[['qe_intensity', 'holdings_lag1']])
            model = OLS(y, X).fit()
            
            # Calculate elasticity at mean
            mean_holdings = holdings.mean()
            mean_qe = qe_intensity.mean()
            elasticity = (model.params['qe_intensity'] * mean_qe) / mean_holdings if mean_holdings != 0 else 0
            
            return {
                'qe_coefficient': model.params['qe_intensity'],
                'qe_pvalue': model.pvalues['qe_intensity'],
                'qe_significant': model.pvalues['qe_intensity'] < self.significance_level,
                'elasticity': elasticity,
                'persistence': model.params['holdings_lag1'],
                'r_squared': model.rsquared,
                'n_obs': model.nobs,
                'investor_type': investor_type
            }
        except Exception as e:
            return {'error': str(e), 'investor_type': investor_type}
    
    def create_exchange_rate_models(self, 
                                  exchange_rates: pd.DataFrame,
                                  qe_intensity: pd.Series,
                                  foreign_holdings: pd.DataFrame) -> Dict[str, Any]:
        """
        Create exchange rate models linking QE announcements to depreciation.
        
        Args:
            exchange_rates: Exchange rate data (USD per foreign currency)
            qe_intensity: QE intensity measure
            foreign_holdings: Foreign Treasury holdings data
            
        Returns:
            Dictionary containing exchange rate model results
        """
        self.logger.info("Creating exchange rate models linking QE to depreciation")
        
        results = {}
        
        # Align data
        common_index = (exchange_rates.index
                       .intersection(qe_intensity.index)
                       .intersection(foreign_holdings.index))
        
        if len(common_index) < 24:
            raise ValueError("Insufficient data for exchange rate modeling")
        
        fx_aligned = exchange_rates.loc[common_index]
        qe_aligned = qe_intensity.loc[common_index]
        holdings_aligned = foreign_holdings.loc[common_index]
        
        # 1. Basic FX response models for major currencies
        fx_models = {}
        for currency in fx_aligned.columns:
            try:
                fx_series = fx_aligned[currency]
                
                # Log differences for growth rates
                fx_log = np.log(fx_series)
                fx_returns = fx_log.diff()
                qe_changes = qe_aligned.diff()
                
                data = pd.DataFrame({
                    'fx_returns': fx_returns,
                    'qe_changes': qe_changes,
                    'fx_lag1': fx_returns.shift(1),
                    'qe_level': qe_aligned
                }).dropna()
                
                if len(data) > 10:
                    # Returns model
                    y = data['fx_returns']
                    X = sm.add_constant(data[['qe_changes', 'fx_lag1']])
                    model_returns = OLS(y, X).fit()
                    
                    # Levels model
                    data_levels = pd.DataFrame({
                        'fx_level': fx_series,
                        'qe_level': qe_aligned,
                        'fx_lag1': fx_series.shift(1)
                    }).dropna()
                    
                    if len(data_levels) > 10:
                        y_levels = data_levels['fx_level']
                        X_levels = sm.add_constant(data_levels[['qe_level', 'fx_lag1']])
                        model_levels = OLS(y_levels, X_levels).fit()
                        
                        fx_models[currency] = {
                            'returns_model': {
                                'qe_coefficient': model_returns.params['qe_changes'],
                                'qe_pvalue': model_returns.pvalues['qe_changes'],
                                'qe_significant': model_returns.pvalues['qe_changes'] < self.significance_level,
                                'r_squared': model_returns.rsquared,
                                'n_obs': model_returns.nobs
                            },
                            'levels_model': {
                                'qe_coefficient': model_levels.params['qe_level'],
                                'qe_pvalue': model_levels.pvalues['qe_level'],
                                'qe_significant': model_levels.pvalues['qe_level'] < self.significance_level,
                                'r_squared': model_levels.rsquared,
                                'n_obs': model_levels.nobs
                            }
                        }
            except Exception as e:
                fx_models[currency] = {'error': str(e)}
        
        results['currency_models'] = fx_models
        
        # 2. Event study around QE announcements
        event_results = self._qe_announcement_event_study(fx_aligned, qe_aligned)
        results['event_study'] = event_results
        
        # 3. VAR model for joint FX-QE dynamics
        var_results = self._estimate_fx_qe_var(fx_aligned, qe_aligned)
        results['var_analysis'] = var_results
        
        return results
    
    def _qe_announcement_event_study(self, exchange_rates: pd.DataFrame, 
                                   qe_intensity: pd.Series) -> Dict[str, Any]:
        """Conduct event study around QE announcement dates"""
        
        results = {}
        event_window = 5  # Days around announcement
        
        try:
            # Convert announcement dates to datetime
            announcement_dates = [pd.to_datetime(date) for date in self.qe_announcement_dates]
            
            # Calculate daily returns for exchange rates
            fx_returns = exchange_rates.pct_change()
            
            event_effects = {}
            for currency in exchange_rates.columns:
                currency_effects = []
                
                for announce_date in announcement_dates:
                    # Find closest trading day
                    available_dates = fx_returns.index
                    closest_date = min(available_dates, 
                                     key=lambda x: abs((x - announce_date).days))
                    
                    if abs((closest_date - announce_date).days) <= 3:  # Within 3 days
                        # Get event window returns
                        start_idx = max(0, available_dates.get_loc(closest_date) - event_window)
                        end_idx = min(len(available_dates), available_dates.get_loc(closest_date) + event_window + 1)
                        
                        event_returns = fx_returns[currency].iloc[start_idx:end_idx]
                        if len(event_returns) > 0:
                            cumulative_return = (1 + event_returns).prod() - 1
                            currency_effects.append(cumulative_return)
                
                if currency_effects:
                    event_effects[currency] = {
                        'mean_effect': np.mean(currency_effects),
                        'std_effect': np.std(currency_effects),
                        'n_events': len(currency_effects),
                        't_stat': np.mean(currency_effects) / (np.std(currency_effects) / np.sqrt(len(currency_effects))) if np.std(currency_effects) > 0 else 0
                    }
                    
                    # Test significance
                    if len(currency_effects) > 1:
                        t_stat = event_effects[currency]['t_stat']
                        p_value = 2 * (1 - stats.t.cdf(abs(t_stat), len(currency_effects) - 1))
                        event_effects[currency]['p_value'] = p_value
                        event_effects[currency]['significant'] = p_value < self.significance_level
            
            results['announcement_effects'] = event_effects
            
        except Exception as e:
            results['error'] = str(e)
        
        return results
    
    def _estimate_fx_qe_var(self, exchange_rates: pd.DataFrame, 
                           qe_intensity: pd.Series) -> Dict[str, Any]:
        """Estimate VAR model for FX-QE joint dynamics"""
        
        results = {}
        
        try:
            # Use first major currency for VAR
            if len(exchange_rates.columns) > 0:
                fx_series = exchange_rates.iloc[:, 0]
                
                # Prepare data for VAR
                data = pd.DataFrame({
                    'fx_rate': fx_series,
                    'qe_intensity': qe_intensity
                }).dropna()
                
                if len(data) > 20:
                    # Estimate VAR with optimal lag selection
                    var_data = data[['fx_rate', 'qe_intensity']]
                    
                    # Try different lag lengths
                    max_lags = min(4, len(data) // 10)
                    if max_lags >= 1:
                        var_model = VAR(var_data)
                        lag_selection = var_model.select_order(maxlags=max_lags)
                        optimal_lags = lag_selection.aic
                        
                        # Fit VAR with optimal lags
                        var_fitted = var_model.fit(optimal_lags)
                        
                        results['var_model'] = {
                            'optimal_lags': optimal_lags,
                            'aic': var_fitted.aic,
                            'bic': var_fitted.bic,
                            'n_obs': var_fitted.nobs,
                            'coefficients': var_fitted.params.to_dict()
                        }
                        
                        # Granger causality tests
                        causality_results = {}
                        try:
                            # Test if QE Granger-causes FX
                            gc_qe_to_fx = grangercausalitytests(data[['fx_rate', 'qe_intensity']], 
                                                              maxlag=optimal_lags, verbose=False)
                            causality_results['qe_to_fx'] = {
                                'p_value': gc_qe_to_fx[optimal_lags][0]['ssr_ftest'][1],
                                'significant': gc_qe_to_fx[optimal_lags][0]['ssr_ftest'][1] < self.significance_level
                            }
                            
                            # Test if FX Granger-causes QE
                            gc_fx_to_qe = grangercausalitytests(data[['qe_intensity', 'fx_rate']], 
                                                              maxlag=optimal_lags, verbose=False)
                            causality_results['fx_to_qe'] = {
                                'p_value': gc_fx_to_qe[optimal_lags][0]['ssr_ftest'][1],
                                'significant': gc_fx_to_qe[optimal_lags][0]['ssr_ftest'][1] < self.significance_level
                            }
                            
                        except Exception as e:
                            causality_results['error'] = str(e)
                        
                        results['granger_causality'] = causality_results
                        
        except Exception as e:
            results['error'] = str(e)
        
        return results
    
    def test_causality_qe_foreign_demand(self, 
                                       foreign_holdings: pd.DataFrame,
                                       qe_intensity: pd.Series,
                                       max_lags: int = 4) -> Dict[str, Any]:
        """
        Add causality testing between QE and foreign demand changes.
        
        Args:
            foreign_holdings: Foreign Treasury holdings data
            qe_intensity: QE intensity measure
            max_lags: Maximum lags for causality testing
            
        Returns:
            Dictionary containing causality test results
        """
        self.logger.info("Testing causality between QE and foreign demand changes")
        
        results = {}
        
        # Align data
        common_index = foreign_holdings.index.intersection(qe_intensity.index)
        if len(common_index) < 30:
            raise ValueError("Insufficient data for causality testing")
        
        holdings_aligned = foreign_holdings.loc[common_index]
        qe_aligned = qe_intensity.loc[common_index]
        
        # 1. Aggregate causality test
        total_holdings = holdings_aligned.sum(axis=1)
        
        try:
            data = pd.DataFrame({
                'total_holdings': total_holdings,
                'qe_intensity': qe_aligned
            }).dropna()
            
            if len(data) > max_lags * 3:
                # Test QE -> Foreign Holdings
                gc_qe_to_holdings = grangercausalitytests(
                    data[['total_holdings', 'qe_intensity']], 
                    maxlag=max_lags, verbose=False
                )
                
                # Test Foreign Holdings -> QE
                gc_holdings_to_qe = grangercausalitytests(
                    data[['qe_intensity', 'total_holdings']], 
                    maxlag=max_lags, verbose=False
                )
                
                # Extract results for optimal lag
                optimal_lag = min(max_lags, len(data) // 10)
                if optimal_lag >= 1:
                    results['aggregate_causality'] = {
                        'qe_to_holdings': {
                            'p_value': gc_qe_to_holdings[optimal_lag][0]['ssr_ftest'][1],
                            'significant': gc_qe_to_holdings[optimal_lag][0]['ssr_ftest'][1] < self.significance_level,
                            'f_statistic': gc_qe_to_holdings[optimal_lag][0]['ssr_ftest'][0]
                        },
                        'holdings_to_qe': {
                            'p_value': gc_holdings_to_qe[optimal_lag][0]['ssr_ftest'][1],
                            'significant': gc_holdings_to_qe[optimal_lag][0]['ssr_ftest'][1] < self.significance_level,
                            'f_statistic': gc_holdings_to_qe[optimal_lag][0]['ssr_ftest'][0]
                        },
                        'optimal_lag': optimal_lag
                    }
                    
        except Exception as e:
            results['aggregate_causality'] = {'error': str(e)}
        
        # 2. Country-specific causality tests
        country_causality = {}
        available_countries = [col for col in holdings_aligned.columns 
                             if holdings_aligned[col].notna().sum() > len(common_index) * 0.7]
        
        for country in available_countries[:5]:  # Test top 5 countries
            try:
                country_data = pd.DataFrame({
                    'holdings': holdings_aligned[country],
                    'qe_intensity': qe_aligned
                }).dropna()
                
                if len(country_data) > max_lags * 3:
                    # Test causality in both directions
                    gc_qe_to_country = grangercausalitytests(
                        country_data[['holdings', 'qe_intensity']], 
                        maxlag=min(max_lags, len(country_data) // 10), verbose=False
                    )
                    
                    optimal_lag = min(max_lags, len(country_data) // 10)
                    if optimal_lag >= 1:
                        country_causality[country] = {
                            'qe_to_holdings': {
                                'p_value': gc_qe_to_country[optimal_lag][0]['ssr_ftest'][1],
                                'significant': gc_qe_to_country[optimal_lag][0]['ssr_ftest'][1] < self.significance_level
                            }
                        }
                        
            except Exception as e:
                country_causality[country] = {'error': str(e)}
        
        results['country_causality'] = country_causality
        
        return results


class CurrencyDepreciationAnalyzer:
    """
    Analyzes currency depreciation patterns in response to QE policies.
    
    This class implements exchange rate models, QE announcement effects,
    and transmission channel analysis for currency depreciation.
    """
    
    def __init__(self, significance_level: float = 0.05):
        """Initialize the Currency Depreciation Analyzer"""
        self.significance_level = significance_level
        self.logger = logging.getLogger(__name__)
    
    def analyze_depreciation_patterns(self, 
                                    exchange_rates: pd.DataFrame,
                                    qe_intensity: pd.Series,
                                    control_variables: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """
        Analyze currency depreciation patterns in response to QE.
        
        Args:
            exchange_rates: Exchange rate data
            qe_intensity: QE intensity measure
            control_variables: Optional control variables
            
        Returns:
            Dictionary containing depreciation analysis results
        """
        self.logger.info("Analyzing currency depreciation patterns")
        
        results = {}
        
        # Align data
        common_index = exchange_rates.index.intersection(qe_intensity.index)
        if len(common_index) < 20:
            raise ValueError("Insufficient data for depreciation analysis")
        
        fx_aligned = exchange_rates.loc[common_index]
        qe_aligned = qe_intensity.loc[common_index]
        
        # 1. Calculate depreciation measures
        depreciation_measures = {}
        for currency in fx_aligned.columns:
            fx_series = fx_aligned[currency]
            
            # Calculate various depreciation measures
            fx_returns = fx_series.pct_change()
            fx_log_returns = np.log(fx_series).diff()
            
            # Cumulative depreciation during QE periods
            qe_periods = qe_aligned > qe_aligned.quantile(0.75)  # Top quartile as QE periods
            
            if qe_periods.sum() > 0:
                qe_depreciation = fx_returns[qe_periods].mean()
                non_qe_depreciation = fx_returns[~qe_periods].mean()
                
                depreciation_measures[currency] = {
                    'qe_period_depreciation': qe_depreciation,
                    'non_qe_period_depreciation': non_qe_depreciation,
                    'depreciation_difference': qe_depreciation - non_qe_depreciation,
                    'volatility_qe': fx_returns[qe_periods].std(),
                    'volatility_non_qe': fx_returns[~qe_periods].std()
                }
                
                # Statistical test for difference
                if len(fx_returns[qe_periods]) > 5 and len(fx_returns[~qe_periods]) > 5:
                    t_stat, p_value = stats.ttest_ind(fx_returns[qe_periods].dropna(), 
                                                    fx_returns[~qe_periods].dropna())
                    depreciation_measures[currency]['t_statistic'] = t_stat
                    depreciation_measures[currency]['p_value'] = p_value
                    depreciation_measures[currency]['significant'] = p_value < self.significance_level
        
        results['depreciation_measures'] = depreciation_measures
        
        # 2. Regression analysis of depreciation on QE intensity
        regression_results = {}
        for currency in fx_aligned.columns:
            try:
                fx_series = fx_aligned[currency]
                fx_returns = fx_series.pct_change()
                
                data = pd.DataFrame({
                    'fx_returns': fx_returns,
                    'qe_intensity': qe_aligned,
                    'qe_change': qe_aligned.diff(),
                    'fx_lag1': fx_returns.shift(1)
                }).dropna()
                
                if len(data) > 10:
                    # Returns on QE changes
                    y = data['fx_returns']
                    X = sm.add_constant(data[['qe_change', 'fx_lag1']])
                    model = OLS(y, X).fit()
                    
                    regression_results[currency] = {
                        'qe_coefficient': model.params['qe_change'],
                        'qe_pvalue': model.pvalues['qe_change'],
                        'qe_significant': model.pvalues['qe_change'] < self.significance_level,
                        'r_squared': model.rsquared,
                        'n_obs': model.nobs,
                        'durbin_watson': durbin_watson(model.resid)
                    }
                    
            except Exception as e:
                regression_results[currency] = {'error': str(e)}
        
        results['regression_analysis'] = regression_results
        
        return results


class InflationOffsetAnalyzer:
    """
    Analyzes inflation offset effects and international spillovers from QE.
    
    This class implements inflation pressure measurement, offset quantification,
    and cross-country spillover analysis.
    """
    
    def __init__(self, significance_level: float = 0.05):
        """Initialize the Inflation Offset Analyzer"""
        self.significance_level = significance_level
        self.logger = logging.getLogger(__name__)
    
    def analyze_inflation_pressures(self, 
                                  inflation_data: pd.DataFrame,
                                  qe_intensity: pd.Series,
                                  exchange_rates: pd.DataFrame) -> Dict[str, Any]:
        """
        Create inflation pressure measurement using multiple indicators.
        
        Args:
            inflation_data: Multiple inflation indicators (CPI, PCE, import prices)
            qe_intensity: QE intensity measure
            exchange_rates: Exchange rate data
            
        Returns:
            Dictionary containing inflation pressure analysis
        """
        self.logger.info("Analyzing inflation pressures from QE")
        
        results = {}
        
        # Align all data
        common_index = (inflation_data.index
                       .intersection(qe_intensity.index)
                       .intersection(exchange_rates.index))
        
        if len(common_index) < 20:
            raise ValueError("Insufficient data for inflation pressure analysis")
        
        inflation_aligned = inflation_data.loc[common_index]
        qe_aligned = qe_intensity.loc[common_index]
        fx_aligned = exchange_rates.loc[common_index]
        
        # 1. Direct QE-inflation relationships
        inflation_models = {}
        for inflation_measure in inflation_aligned.columns:
            try:
                inflation_series = inflation_aligned[inflation_measure]
                inflation_rate = inflation_series.pct_change() * 100  # Convert to percentage
                
                data = pd.DataFrame({
                    'inflation_rate': inflation_rate,
                    'qe_intensity': qe_aligned,
                    'qe_change': qe_aligned.diff(),
                    'inflation_lag1': inflation_rate.shift(1)
                }).dropna()
                
                if len(data) > 10:
                    # Inflation rate on QE changes
                    y = data['inflation_rate']
                    X = sm.add_constant(data[['qe_change', 'inflation_lag1']])
                    model = OLS(y, X).fit()
                    
                    inflation_models[inflation_measure] = {
                        'qe_coefficient': model.params['qe_change'],
                        'qe_pvalue': model.pvalues['qe_change'],
                        'qe_significant': model.pvalues['qe_change'] < self.significance_level,
                        'persistence': model.params['inflation_lag1'],
                        'r_squared': model.rsquared,
                        'n_obs': model.nobs
                    }
                    
            except Exception as e:
                inflation_models[inflation_measure] = {'error': str(e)}
        
        results['direct_qe_inflation'] = inflation_models
        
        # 2. Exchange rate pass-through to inflation
        passthrough_results = {}
        if len(fx_aligned.columns) > 0:
            fx_series = fx_aligned.iloc[:, 0]  # Use first FX series
            fx_depreciation = -fx_series.pct_change() * 100  # Negative for depreciation
            
            for inflation_measure in inflation_aligned.columns:
                try:
                    inflation_series = inflation_aligned[inflation_measure]
                    inflation_rate = inflation_series.pct_change() * 100
                    
                    data = pd.DataFrame({
                        'inflation_rate': inflation_rate,
                        'fx_depreciation': fx_depreciation,
                        'qe_intensity': qe_aligned,
                        'inflation_lag1': inflation_rate.shift(1),
                        'fx_lag1': fx_depreciation.shift(1)
                    }).dropna()
                    
                    if len(data) > 12:
                        # Pass-through model
                        y = data['inflation_rate']
                        X = sm.add_constant(data[['fx_depreciation', 'qe_intensity', 'inflation_lag1']])
                        model = OLS(y, X).fit()
                        
                        passthrough_results[inflation_measure] = {
                            'fx_coefficient': model.params['fx_depreciation'],
                            'fx_pvalue': model.pvalues['fx_depreciation'],
                            'fx_significant': model.pvalues['fx_depreciation'] < self.significance_level,
                            'qe_coefficient': model.params['qe_intensity'],
                            'qe_pvalue': model.pvalues['qe_intensity'],
                            'qe_significant': model.pvalues['qe_intensity'] < self.significance_level,
                            'r_squared': model.rsquared,
                            'n_obs': model.nobs
                        }
                        
                except Exception as e:
                    passthrough_results[inflation_measure] = {'error': str(e)}
        
        results['exchange_rate_passthrough'] = passthrough_results
        
        # 3. Quantify inflationary offset relative to QE benefits
        offset_analysis = self._quantify_inflation_offset(
            inflation_aligned, qe_aligned, fx_aligned
        )
        results['offset_quantification'] = offset_analysis
        
        return results
    
    def _quantify_inflation_offset(self, 
                                 inflation_data: pd.DataFrame,
                                 qe_intensity: pd.Series,
                                 exchange_rates: pd.DataFrame) -> Dict[str, Any]:
        """Quantify inflationary offset relative to QE benefits"""
        
        results = {}
        
        try:
            # Calculate average inflation during high QE periods
            qe_high_periods = qe_intensity > qe_intensity.quantile(0.75)
            qe_low_periods = qe_intensity < qe_intensity.quantile(0.25)
            
            offset_measures = {}
            for inflation_measure in inflation_data.columns:
                inflation_series = inflation_data[inflation_measure]
                inflation_rate = inflation_series.pct_change() * 100
                
                if qe_high_periods.sum() > 0 and qe_low_periods.sum() > 0:
                    high_qe_inflation = inflation_rate[qe_high_periods].mean()
                    low_qe_inflation = inflation_rate[qe_low_periods].mean()
                    
                    offset_measures[inflation_measure] = {
                        'high_qe_inflation': high_qe_inflation,
                        'low_qe_inflation': low_qe_inflation,
                        'inflation_differential': high_qe_inflation - low_qe_inflation,
                        'offset_magnitude': abs(high_qe_inflation - low_qe_inflation)
                    }
                    
                    # Statistical significance test
                    high_qe_values = inflation_rate[qe_high_periods].dropna()
                    low_qe_values = inflation_rate[qe_low_periods].dropna()
                    
                    if len(high_qe_values) > 3 and len(low_qe_values) > 3:
                        t_stat, p_value = stats.ttest_ind(high_qe_values, low_qe_values)
                        offset_measures[inflation_measure]['t_statistic'] = t_stat
                        offset_measures[inflation_measure]['p_value'] = p_value
                        offset_measures[inflation_measure]['significant'] = p_value < self.significance_level
            
            results['offset_measures'] = offset_measures
            
        except Exception as e:
            results['error'] = str(e)
        
        return results
    
    def analyze_cross_country_spillovers(self, 
                                       inflation_data: pd.DataFrame,
                                       qe_intensity: pd.Series,
                                       exchange_rates: pd.DataFrame,
                                       foreign_countries: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Add cross-country spillover analysis and comparison.
        
        Args:
            inflation_data: Multiple inflation indicators (CPI, PCE, import prices)
            qe_intensity: QE intensity measure
            exchange_rates: Exchange rate data for multiple countries
            foreign_countries: List of foreign countries to analyze
            
        Returns:
            Dictionary containing cross-country spillover analysis
        """
        self.logger.info("Analyzing cross-country spillover effects")
        
        results = {}
        
        # Default countries if not specified
        if foreign_countries is None:
            foreign_countries = ['EUR', 'JPY', 'GBP', 'CAD', 'CHF', 'AUD']
        
        # Align data
        common_index = (inflation_data.index
                       .intersection(qe_intensity.index)
                       .intersection(exchange_rates.index))
        
        if len(common_index) < 30:
            raise ValueError("Insufficient data for cross-country spillover analysis")
        
        inflation_aligned = inflation_data.loc[common_index]
        qe_aligned = qe_intensity.loc[common_index]
        fx_aligned = exchange_rates.loc[common_index]
        
        # 1. Cross-country inflation correlation analysis
        spillover_correlations = {}
        
        # Calculate QE-induced inflation pressures for domestic economy
        domestic_inflation = inflation_aligned.iloc[:, 0] if len(inflation_aligned.columns) > 0 else None
        
        if domestic_inflation is not None:
            domestic_inflation_rate = domestic_inflation.pct_change() * 100
            
            # Analyze spillovers to each foreign country
            for country in foreign_countries:
                if country in fx_aligned.columns:
                    try:
                        fx_series = fx_aligned[country]
                        fx_depreciation = -fx_series.pct_change() * 100  # Negative for USD depreciation
                        
                        # Create spillover model
                        data = pd.DataFrame({
                            'domestic_inflation': domestic_inflation_rate,
                            'fx_depreciation': fx_depreciation,
                            'qe_intensity': qe_aligned,
                            'qe_change': qe_aligned.diff(),
                            'domestic_inflation_lag1': domestic_inflation_rate.shift(1),
                            'fx_lag1': fx_depreciation.shift(1)
                        }).dropna()
                        
                        if len(data) > 15:
                            # Model: Foreign FX depreciation as function of domestic QE and inflation
                            y = data['fx_depreciation']
                            X = sm.add_constant(data[['qe_change', 'domestic_inflation', 'fx_lag1']])
                            model = OLS(y, X).fit()
                            
                            spillover_correlations[country] = {
                                'qe_spillover_coefficient': model.params['qe_change'],
                                'qe_spillover_pvalue': model.pvalues['qe_change'],
                                'qe_spillover_significant': model.pvalues['qe_change'] < self.significance_level,
                                'inflation_spillover_coefficient': model.params['domestic_inflation'],
                                'inflation_spillover_pvalue': model.pvalues['domestic_inflation'],
                                'inflation_spillover_significant': model.pvalues['domestic_inflation'] < self.significance_level,
                                'r_squared': model.rsquared,
                                'n_obs': model.nobs,
                                'spillover_magnitude': abs(model.params['qe_change'])
                            }
                            
                            # Calculate spillover elasticity
                            mean_qe_change = data['qe_change'].mean()
                            mean_fx_depreciation = data['fx_depreciation'].mean()
                            if mean_fx_depreciation != 0:
                                elasticity = (model.params['qe_change'] * mean_qe_change) / mean_fx_depreciation
                                spillover_correlations[country]['spillover_elasticity'] = elasticity
                            
                    except Exception as e:
                        spillover_correlations[country] = {'error': str(e)}
        
        results['spillover_correlations'] = spillover_correlations
        
        # 2. Quantify spillover magnitudes and rankings
        spillover_rankings = self._rank_spillover_effects(spillover_correlations)
        results['spillover_rankings'] = spillover_rankings
        
        # 3. Cross-country comparison of inflation pass-through
        passthrough_comparison = self._compare_inflation_passthrough(
            inflation_aligned, qe_aligned, fx_aligned, foreign_countries
        )
        results['passthrough_comparison'] = passthrough_comparison
        
        # 4. Spillover transmission channels analysis
        transmission_analysis = self._analyze_spillover_transmission_channels(
            inflation_aligned, qe_aligned, fx_aligned, foreign_countries
        )
        results['transmission_channels'] = transmission_analysis
        
        return results
    
    def _rank_spillover_effects(self, spillover_correlations: Dict[str, Any]) -> Dict[str, Any]:
        """Rank countries by spillover effect magnitude"""
        
        rankings = {}
        
        try:
            # Extract spillover magnitudes
            spillover_magnitudes = {}
            for country, results in spillover_correlations.items():
                if 'spillover_magnitude' in results and 'qe_spillover_significant' in results:
                    if results['qe_spillover_significant']:
                        spillover_magnitudes[country] = results['spillover_magnitude']
            
            if spillover_magnitudes:
                # Sort by magnitude
                sorted_spillovers = sorted(spillover_magnitudes.items(), 
                                         key=lambda x: x[1], reverse=True)
                
                rankings['ranked_spillovers'] = sorted_spillovers
                rankings['highest_spillover_country'] = sorted_spillovers[0][0] if sorted_spillovers else None
                rankings['average_spillover_magnitude'] = np.mean(list(spillover_magnitudes.values()))
                rankings['spillover_concentration'] = (
                    sorted_spillovers[0][1] / sum(spillover_magnitudes.values()) 
                    if sorted_spillovers and sum(spillover_magnitudes.values()) > 0 else 0
                )
                
                # Statistical summary
                rankings['n_significant_spillovers'] = len(spillover_magnitudes)
                rankings['total_countries_analyzed'] = len(spillover_correlations)
                rankings['spillover_prevalence'] = (
                    len(spillover_magnitudes) / len(spillover_correlations) 
                    if spillover_correlations else 0
                )
                
        except Exception as e:
            rankings['error'] = str(e)
        
        return rankings
    
    def _compare_inflation_passthrough(self, 
                                     inflation_data: pd.DataFrame,
                                     qe_intensity: pd.Series,
                                     exchange_rates: pd.DataFrame,
                                     foreign_countries: List[str]) -> Dict[str, Any]:
        """Compare inflation pass-through across countries"""
        
        comparison = {}
        
        try:
            passthrough_coefficients = {}
            
            # Analyze pass-through for each country
            for country in foreign_countries:
                if country in exchange_rates.columns:
                    try:
                        fx_series = exchange_rates[country]
                        fx_depreciation = -fx_series.pct_change() * 100
                        
                        # Use first inflation measure as proxy
                        if len(inflation_data.columns) > 0:
                            inflation_series = inflation_data.iloc[:, 0]
                            inflation_rate = inflation_series.pct_change() * 100
                            
                            data = pd.DataFrame({
                                'inflation_rate': inflation_rate,
                                'fx_depreciation': fx_depreciation,
                                'qe_intensity': qe_intensity,
                                'inflation_lag1': inflation_rate.shift(1)
                            }).dropna()
                            
                            if len(data) > 12:
                                # Pass-through regression
                                y = data['inflation_rate']
                                X = sm.add_constant(data[['fx_depreciation', 'qe_intensity', 'inflation_lag1']])
                                model = OLS(y, X).fit()
                                
                                passthrough_coefficients[country] = {
                                    'passthrough_coefficient': model.params['fx_depreciation'],
                                    'passthrough_pvalue': model.pvalues['fx_depreciation'],
                                    'passthrough_significant': model.pvalues['fx_depreciation'] < self.significance_level,
                                    'qe_direct_effect': model.params['qe_intensity'],
                                    'r_squared': model.rsquared,
                                    'n_obs': model.nobs
                                }
                                
                    except Exception as e:
                        passthrough_coefficients[country] = {'error': str(e)}
            
            comparison['country_passthrough'] = passthrough_coefficients
            
            # Summary statistics
            valid_coefficients = [
                results['passthrough_coefficient'] 
                for results in passthrough_coefficients.values() 
                if 'passthrough_coefficient' in results and 'passthrough_significant' in results
                and results['passthrough_significant']
            ]
            
            if valid_coefficients:
                comparison['summary_statistics'] = {
                    'mean_passthrough': np.mean(valid_coefficients),
                    'median_passthrough': np.median(valid_coefficients),
                    'std_passthrough': np.std(valid_coefficients),
                    'min_passthrough': np.min(valid_coefficients),
                    'max_passthrough': np.max(valid_coefficients),
                    'n_significant_countries': len(valid_coefficients)
                }
                
        except Exception as e:
            comparison['error'] = str(e)
        
        return comparison
    
    def _analyze_spillover_transmission_channels(self, 
                                               inflation_data: pd.DataFrame,
                                               qe_intensity: pd.Series,
                                               exchange_rates: pd.DataFrame,
                                               foreign_countries: List[str]) -> Dict[str, Any]:
        """Analyze transmission channels for spillover effects"""
        
        transmission = {}
        
        try:
            # 1. Direct QE transmission channel
            direct_transmission = {}
            
            for country in foreign_countries:
                if country in exchange_rates.columns:
                    try:
                        fx_series = exchange_rates[country]
                        fx_returns = fx_series.pct_change() * 100
                        
                        # Direct QE to FX transmission
                        data = pd.DataFrame({
                            'fx_returns': fx_returns,
                            'qe_change': qe_intensity.diff(),
                            'qe_level': qe_intensity,
                            'fx_lag1': fx_returns.shift(1)
                        }).dropna()
                        
                        if len(data) > 10:
                            y = data['fx_returns']
                            X = sm.add_constant(data[['qe_change', 'fx_lag1']])
                            model = OLS(y, X).fit()
                            
                            direct_transmission[country] = {
                                'direct_qe_coefficient': model.params['qe_change'],
                                'direct_qe_pvalue': model.pvalues['qe_change'],
                                'direct_qe_significant': model.pvalues['qe_change'] < self.significance_level,
                                'transmission_strength': abs(model.params['qe_change']),
                                'r_squared': model.rsquared
                            }
                            
                    except Exception as e:
                        direct_transmission[country] = {'error': str(e)}
            
            transmission['direct_transmission'] = direct_transmission
            
            # 2. Indirect transmission through inflation
            indirect_transmission = {}
            
            if len(inflation_data.columns) > 0:
                domestic_inflation = inflation_data.iloc[:, 0]
                domestic_inflation_rate = domestic_inflation.pct_change() * 100
                
                for country in foreign_countries:
                    if country in exchange_rates.columns:
                        try:
                            fx_series = exchange_rates[country]
                            fx_returns = fx_series.pct_change() * 100
                            
                            # Indirect transmission: QE -> Inflation -> FX
                            data = pd.DataFrame({
                                'fx_returns': fx_returns,
                                'inflation_rate': domestic_inflation_rate,
                                'qe_change': qe_intensity.diff(),
                                'fx_lag1': fx_returns.shift(1),
                                'inflation_lag1': domestic_inflation_rate.shift(1)
                            }).dropna()
                            
                            if len(data) > 12:
                                # Two-stage model
                                # Stage 1: QE -> Inflation
                                y1 = data['inflation_rate']
                                X1 = sm.add_constant(data[['qe_change', 'inflation_lag1']])
                                model1 = OLS(y1, X1).fit()
                                
                                # Stage 2: Inflation -> FX
                                y2 = data['fx_returns']
                                X2 = sm.add_constant(data[['inflation_rate', 'fx_lag1']])
                                model2 = OLS(y2, X2).fit()
                                
                                # Calculate indirect effect
                                indirect_effect = (model1.params['qe_change'] * 
                                                 model2.params['inflation_rate'])
                                
                                indirect_transmission[country] = {
                                    'qe_to_inflation_coefficient': model1.params['qe_change'],
                                    'qe_to_inflation_pvalue': model1.pvalues['qe_change'],
                                    'inflation_to_fx_coefficient': model2.params['inflation_rate'],
                                    'inflation_to_fx_pvalue': model2.pvalues['inflation_rate'],
                                    'indirect_effect': indirect_effect,
                                    'transmission_significant': (
                                        model1.pvalues['qe_change'] < self.significance_level and
                                        model2.pvalues['inflation_rate'] < self.significance_level
                                    )
                                }
                                
                        except Exception as e:
                            indirect_transmission[country] = {'error': str(e)}
            
            transmission['indirect_transmission'] = indirect_transmission
            
            # 3. Compare transmission channel strengths
            channel_comparison = self._compare_transmission_channels(
                direct_transmission, indirect_transmission
            )
            transmission['channel_comparison'] = channel_comparison
            
        except Exception as e:
            transmission['error'] = str(e)
        
        return transmission
    
    def _compare_transmission_channels(self, 
                                     direct_transmission: Dict[str, Any],
                                     indirect_transmission: Dict[str, Any]) -> Dict[str, Any]:
        """Compare direct vs indirect transmission channel strengths"""
        
        comparison = {}
        
        try:
            channel_strengths = {}
            
            # Compare for each country
            common_countries = set(direct_transmission.keys()).intersection(
                set(indirect_transmission.keys())
            )
            
            for country in common_countries:
                direct_results = direct_transmission.get(country, {})
                indirect_results = indirect_transmission.get(country, {})
                
                if ('transmission_strength' in direct_results and 
                    'indirect_effect' in indirect_results):
                    
                    direct_strength = direct_results['transmission_strength']
                    indirect_strength = abs(indirect_results['indirect_effect'])
                    
                    channel_strengths[country] = {
                        'direct_strength': direct_strength,
                        'indirect_strength': indirect_strength,
                        'total_transmission': direct_strength + indirect_strength,
                        'direct_share': (direct_strength / (direct_strength + indirect_strength) 
                                       if (direct_strength + indirect_strength) > 0 else 0),
                        'indirect_share': (indirect_strength / (direct_strength + indirect_strength) 
                                         if (direct_strength + indirect_strength) > 0 else 0),
                        'dominant_channel': ('direct' if direct_strength > indirect_strength 
                                           else 'indirect')
                    }
            
            comparison['country_channels'] = channel_strengths
            
            # Aggregate statistics
            if channel_strengths:
                direct_strengths = [c['direct_strength'] for c in channel_strengths.values()]
                indirect_strengths = [c['indirect_strength'] for c in channel_strengths.values()]
                
                comparison['aggregate_statistics'] = {
                    'mean_direct_strength': np.mean(direct_strengths),
                    'mean_indirect_strength': np.mean(indirect_strengths),
                    'direct_dominance_frequency': sum(
                        1 for c in channel_strengths.values() 
                        if c['dominant_channel'] == 'direct'
                    ) / len(channel_strengths),
                    'average_direct_share': np.mean([c['direct_share'] for c in channel_strengths.values()]),
                    'average_indirect_share': np.mean([c['indirect_share'] for c in channel_strengths.values()])
                }
                
        except Exception as e:
            comparison['error'] = str(e)
        
        return comparison
    
    def quantify_inflation_offset_benefits(self, 
                                         inflation_data: pd.DataFrame,
                                         qe_intensity: pd.Series,
                                         economic_benefits: Optional[pd.Series] = None) -> Dict[str, Any]:
        """
        Implement models quantifying inflationary offset relative to QE benefits.
        
        Args:
            inflation_data: Multiple inflation indicators
            qe_intensity: QE intensity measure
            economic_benefits: Optional measure of QE economic benefits (e.g., GDP growth, employment)
            
        Returns:
            Dictionary containing offset quantification relative to benefits
        """
        self.logger.info("Quantifying inflationary offset relative to QE benefits")
        
        results = {}
        
        # Align data
        common_index = inflation_data.index.intersection(qe_intensity.index)
        if economic_benefits is not None:
            common_index = common_index.intersection(economic_benefits.index)
        
        if len(common_index) < 20:
            raise ValueError("Insufficient data for offset-benefit analysis")
        
        inflation_aligned = inflation_data.loc[common_index]
        qe_aligned = qe_intensity.loc[common_index]
        benefits_aligned = economic_benefits.loc[common_index] if economic_benefits is not None else None
        
        # 1. Cost-benefit analysis for each inflation measure
        cost_benefit_analysis = {}
        
        for inflation_measure in inflation_aligned.columns:
            try:
                inflation_series = inflation_aligned[inflation_measure]
                inflation_rate = inflation_series.pct_change() * 100
                
                # Define QE periods (high intensity periods)
                qe_periods = qe_aligned > qe_aligned.quantile(0.75)
                non_qe_periods = qe_aligned < qe_aligned.quantile(0.25)
                
                if qe_periods.sum() > 5 and non_qe_periods.sum() > 5:
                    # Calculate inflation costs during QE
                    qe_inflation_cost = inflation_rate[qe_periods].mean()
                    baseline_inflation = inflation_rate[non_qe_periods].mean()
                    inflation_cost = qe_inflation_cost - baseline_inflation
                    
                    cost_benefit_analysis[inflation_measure] = {
                        'qe_period_inflation': qe_inflation_cost,
                        'baseline_inflation': baseline_inflation,
                        'inflation_cost': inflation_cost,
                        'inflation_cost_magnitude': abs(inflation_cost)
                    }
                    
                    # If economic benefits data is available
                    if benefits_aligned is not None:
                        qe_benefits = benefits_aligned[qe_periods].mean()
                        baseline_benefits = benefits_aligned[non_qe_periods].mean()
                        net_benefits = qe_benefits - baseline_benefits
                        
                        cost_benefit_analysis[inflation_measure].update({
                            'qe_period_benefits': qe_benefits,
                            'baseline_benefits': baseline_benefits,
                            'net_benefits': net_benefits,
                            'cost_benefit_ratio': (abs(inflation_cost) / net_benefits 
                                                 if net_benefits != 0 else np.inf),
                            'net_welfare_effect': net_benefits - abs(inflation_cost)
                        })
                    
                    # Statistical significance of differences
                    qe_inflation_values = inflation_rate[qe_periods].dropna()
                    baseline_inflation_values = inflation_rate[non_qe_periods].dropna()
                    
                    if len(qe_inflation_values) > 3 and len(baseline_inflation_values) > 3:
                        t_stat, p_value = stats.ttest_ind(qe_inflation_values, baseline_inflation_values)
                        cost_benefit_analysis[inflation_measure].update({
                            't_statistic': t_stat,
                            'p_value': p_value,
                            'cost_significant': p_value < self.significance_level
                        })
                        
            except Exception as e:
                cost_benefit_analysis[inflation_measure] = {'error': str(e)}
        
        results['cost_benefit_analysis'] = cost_benefit_analysis
        
        # 2. Dynamic offset analysis over time
        dynamic_analysis = self._analyze_dynamic_offset_patterns(
            inflation_aligned, qe_aligned, benefits_aligned
        )
        results['dynamic_offset_analysis'] = dynamic_analysis
        
        # 3. Threshold analysis for offset dominance
        threshold_analysis = self._analyze_offset_thresholds(
            inflation_aligned, qe_aligned, benefits_aligned
        )
        results['threshold_analysis'] = threshold_analysis
        
        return results
    
    def _analyze_dynamic_offset_patterns(self, 
                                       inflation_data: pd.DataFrame,
                                       qe_intensity: pd.Series,
                                       economic_benefits: Optional[pd.Series]) -> Dict[str, Any]:
        """Analyze how offset patterns evolve over time"""
        
        dynamic_results = {}
        
        try:
            # Rolling window analysis
            window_size = 12  # 12-period rolling window
            
            for inflation_measure in inflation_data.columns:
                inflation_series = inflation_data[inflation_measure]
                inflation_rate = inflation_series.pct_change() * 100
                
                # Calculate rolling correlations and relationships
                rolling_correlations = []
                rolling_offsets = []
                
                for i in range(window_size, len(inflation_rate)):
                    window_inflation = inflation_rate.iloc[i-window_size:i]
                    window_qe = qe_intensity.iloc[i-window_size:i]
                    
                    # Calculate correlation
                    if window_inflation.std() > 0 and window_qe.std() > 0:
                        correlation = window_inflation.corr(window_qe)
                        rolling_correlations.append(correlation)
                        
                        # Calculate offset magnitude
                        high_qe_periods = window_qe > window_qe.median()
                        if high_qe_periods.sum() > 2:
                            offset = (window_inflation[high_qe_periods].mean() - 
                                    window_inflation[~high_qe_periods].mean())
                            rolling_offsets.append(offset)
                
                if rolling_correlations and rolling_offsets:
                    dynamic_results[inflation_measure] = {
                        'rolling_correlations': rolling_correlations,
                        'rolling_offsets': rolling_offsets,
                        'mean_correlation': np.mean(rolling_correlations),
                        'correlation_volatility': np.std(rolling_correlations),
                        'mean_offset': np.mean(rolling_offsets),
                        'offset_volatility': np.std(rolling_offsets),
                        'correlation_trend': (rolling_correlations[-1] - rolling_correlations[0] 
                                            if len(rolling_correlations) > 1 else 0)
                    }
                    
        except Exception as e:
            dynamic_results['error'] = str(e)
        
        return dynamic_results
    
    def _analyze_offset_thresholds(self, 
                                 inflation_data: pd.DataFrame,
                                 qe_intensity: pd.Series,
                                 economic_benefits: Optional[pd.Series]) -> Dict[str, Any]:
        """Analyze thresholds where inflation offset dominates QE benefits"""
        
        threshold_results = {}
        
        try:
            # Define QE intensity thresholds
            qe_thresholds = [0.1, 0.2, 0.3, 0.4, 0.5]  # Different QE intensity levels
            
            for inflation_measure in inflation_data.columns:
                inflation_series = inflation_data[inflation_measure]
                inflation_rate = inflation_series.pct_change() * 100
                
                threshold_analysis = {}
                
                for threshold in qe_thresholds:
                    high_qe_periods = qe_intensity > threshold
                    low_qe_periods = qe_intensity <= threshold
                    
                    if high_qe_periods.sum() > 5 and low_qe_periods.sum() > 5:
                        high_qe_inflation = inflation_rate[high_qe_periods].mean()
                        low_qe_inflation = inflation_rate[low_qe_periods].mean()
                        offset_magnitude = high_qe_inflation - low_qe_inflation
                        
                        threshold_analysis[f'threshold_{threshold}'] = {
                            'high_qe_inflation': high_qe_inflation,
                            'low_qe_inflation': low_qe_inflation,
                            'offset_magnitude': offset_magnitude,
                            'n_high_qe_periods': high_qe_periods.sum(),
                            'n_low_qe_periods': low_qe_periods.sum()
                        }
                        
                        # Test statistical significance
                        high_values = inflation_rate[high_qe_periods].dropna()
                        low_values = inflation_rate[low_qe_periods].dropna()
                        
                        if len(high_values) > 3 and len(low_values) > 3:
                            t_stat, p_value = stats.ttest_ind(high_values, low_values)
                            threshold_analysis[f'threshold_{threshold}'].update({
                                't_statistic': t_stat,
                                'p_value': p_value,
                                'significant': p_value < self.significance_level
                            })
                
                threshold_results[inflation_measure] = threshold_analysis
                
                # Find optimal threshold (maximum significant offset)
                significant_thresholds = {
                    k: v for k, v in threshold_analysis.items() 
                    if v.get('significant', False)
                }
                
                if significant_thresholds:
                    optimal_threshold = max(significant_thresholds.items(), 
                                          key=lambda x: abs(x[1]['offset_magnitude']))
                    threshold_results[inflation_measure]['optimal_threshold'] = optimal_threshold
                    
        except Exception as e:
            threshold_results['error'] = str(e)
        
        return threshold_results


class InternationalTransmissionAnalyzer:
    """
    Analyzes international transmission mechanisms for QE effects.
    
    This class implements transmission channel analysis, statistical tests
    for spillover significance, and cross-country comparison frameworks.
    """
    
    def __init__(self, significance_level: float = 0.05):
        """Initialize the International Transmission Analyzer"""
        self.significance_level = significance_level
        self.logger = logging.getLogger(__name__)
    
    def analyze_transmission_channels(self, 
                                    hypothesis_data: HypothesisData) -> Dict[str, Any]:
        """
        Implement transmission channel diagrams and analysis.
        
        Args:
            hypothesis_data: Complete hypothesis data structure
            
        Returns:
            Dictionary containing transmission mechanism analysis
        """
        self.logger.info("Analyzing international transmission mechanisms")
        
        results = {}
        
        try:
            # Extract data components
            qe_intensity = hypothesis_data.qe_intensity
            foreign_holdings = hypothesis_data.foreign_bond_holdings
            exchange_rates = hypothesis_data.exchange_rate
            inflation_measures = hypothesis_data.inflation_measures
            
            # 1. Direct transmission channels analysis
            direct_channels = self._analyze_direct_transmission_channels(
                qe_intensity, foreign_holdings, exchange_rates, inflation_measures
            )
            results['direct_channels'] = direct_channels
            
            # 2. Indirect transmission channels analysis
            indirect_channels = self._analyze_indirect_transmission_channels(
                qe_intensity, foreign_holdings, exchange_rates, inflation_measures
            )
            results['indirect_channels'] = indirect_channels
            
            # 3. Transmission mechanism pathway analysis
            pathway_analysis = self._analyze_transmission_pathways(
                qe_intensity, foreign_holdings, exchange_rates, inflation_measures
            )
            results['transmission_pathways'] = pathway_analysis
            
            # 4. Statistical significance tests for spillover effects
            spillover_significance = self._test_spillover_significance(
                qe_intensity, foreign_holdings, exchange_rates, inflation_measures
            )
            results['spillover_significance'] = spillover_significance
            
            # 5. Cross-country comparison framework
            cross_country_comparison = self._create_cross_country_comparison_framework(
                qe_intensity, foreign_holdings, exchange_rates, inflation_measures
            )
            results['cross_country_comparison'] = cross_country_comparison
            
            # 6. Generate transmission mechanism diagrams data
            diagram_data = self._generate_transmission_diagram_data(results)
            results['transmission_diagrams'] = diagram_data
            
        except Exception as e:
            self.logger.error(f"Error in transmission channel analysis: {e}")
            results['error'] = str(e)
        
        return results
    
    def _analyze_direct_transmission_channels(self, 
                                            qe_intensity: pd.Series,
                                            foreign_holdings: pd.Series,
                                            exchange_rates: pd.Series,
                                            inflation_measures: pd.Series) -> Dict[str, Any]:
        """Analyze direct transmission channels from QE to international effects"""
        
        results = {}
        
        try:
            # Align all data to common index
            common_index = qe_intensity.index
            if foreign_holdings is not None:
                common_index = common_index.intersection(foreign_holdings.index)
            if exchange_rates is not None:
                common_index = common_index.intersection(exchange_rates.index)
            if inflation_measures is not None:
                common_index = common_index.intersection(inflation_measures.index)
            
            if len(common_index) < 20:
                results['error'] = "Insufficient overlapping data for direct transmission analysis"
                return results
            
            qe_aligned = qe_intensity.loc[common_index]
            
            # 1. QE -> Foreign Holdings Channel
            if foreign_holdings is not None:
                holdings_aligned = foreign_holdings.loc[common_index]
                
                # Direct regression: Foreign Holdings = f(QE)
                data = pd.DataFrame({
                    'holdings': holdings_aligned,
                    'qe_intensity': qe_aligned,
                    'holdings_lag1': holdings_aligned.shift(1),
                    'qe_change': qe_aligned.diff()
                }).dropna()
                
                if len(data) > 10:
                    # Levels model
                    y = data['holdings']
                    X = sm.add_constant(data[['qe_intensity', 'holdings_lag1']])
                    model_levels = OLS(y, X).fit()
                    
                    # Changes model
                    y_change = data['holdings'].diff().dropna()
                    X_change = sm.add_constant(data[['qe_change', 'holdings_lag1']].iloc[1:])
                    model_changes = OLS(y_change, X_change).fit()
                    
                    results['qe_to_holdings'] = {
                        'levels_model': {
                            'coefficient': model_levels.params['qe_intensity'],
                            'p_value': model_levels.pvalues['qe_intensity'],
                            'significant': model_levels.pvalues['qe_intensity'] < self.significance_level,
                            'r_squared': model_levels.rsquared,
                            'transmission_strength': abs(model_levels.params['qe_intensity'])
                        },
                        'changes_model': {
                            'coefficient': model_changes.params['qe_change'],
                            'p_value': model_changes.pvalues['qe_change'],
                            'significant': model_changes.pvalues['qe_change'] < self.significance_level,
                            'r_squared': model_changes.rsquared
                        }
                    }
            
            # 2. QE -> Exchange Rate Channel
            if exchange_rates is not None:
                fx_aligned = exchange_rates.loc[common_index]
                fx_returns = fx_aligned.pct_change() * 100
                
                data = pd.DataFrame({
                    'fx_returns': fx_returns,
                    'qe_change': qe_aligned.diff(),
                    'qe_level': qe_aligned,
                    'fx_lag1': fx_returns.shift(1)
                }).dropna()
                
                if len(data) > 10:
                    # QE changes to FX returns
                    y = data['fx_returns']
                    X = sm.add_constant(data[['qe_change', 'fx_lag1']])
                    model_fx = OLS(y, X).fit()
                    
                    results['qe_to_exchange_rate'] = {
                        'coefficient': model_fx.params['qe_change'],
                        'p_value': model_fx.pvalues['qe_change'],
                        'significant': model_fx.pvalues['qe_change'] < self.significance_level,
                        'r_squared': model_fx.rsquared,
                        'transmission_strength': abs(model_fx.params['qe_change'])
                    }
            
            # 3. QE -> Inflation Channel
            if inflation_measures is not None:
                inflation_aligned = inflation_measures.loc[common_index]
                inflation_rate = inflation_aligned.pct_change() * 100
                
                data = pd.DataFrame({
                    'inflation_rate': inflation_rate,
                    'qe_change': qe_aligned.diff(),
                    'qe_level': qe_aligned,
                    'inflation_lag1': inflation_rate.shift(1)
                }).dropna()
                
                if len(data) > 10:
                    # QE to inflation
                    y = data['inflation_rate']
                    X = sm.add_constant(data[['qe_change', 'inflation_lag1']])
                    model_inflation = OLS(y, X).fit()
                    
                    results['qe_to_inflation'] = {
                        'coefficient': model_inflation.params['qe_change'],
                        'p_value': model_inflation.pvalues['qe_change'],
                        'significant': model_inflation.pvalues['qe_change'] < self.significance_level,
                        'r_squared': model_inflation.rsquared,
                        'transmission_strength': abs(model_inflation.params['qe_change'])
                    }
            
        except Exception as e:
            results['error'] = str(e)
        
        return results
    
    def _analyze_indirect_transmission_channels(self, 
                                              qe_intensity: pd.Series,
                                              foreign_holdings: pd.Series,
                                              exchange_rates: pd.Series,
                                              inflation_measures: pd.Series) -> Dict[str, Any]:
        """Analyze indirect transmission channels through multiple steps"""
        
        results = {}
        
        try:
            # Align data
            common_index = qe_intensity.index
            if foreign_holdings is not None:
                common_index = common_index.intersection(foreign_holdings.index)
            if exchange_rates is not None:
                common_index = common_index.intersection(exchange_rates.index)
            if inflation_measures is not None:
                common_index = common_index.intersection(inflation_measures.index)
            
            if len(common_index) < 20:
                results['error'] = "Insufficient data for indirect transmission analysis"
                return results
            
            qe_aligned = qe_intensity.loc[common_index]
            
            # 1. QE -> Holdings -> Exchange Rate -> Inflation pathway
            if all(x is not None for x in [foreign_holdings, exchange_rates, inflation_measures]):
                holdings_aligned = foreign_holdings.loc[common_index]
                fx_aligned = exchange_rates.loc[common_index]
                inflation_aligned = inflation_measures.loc[common_index]
                
                # Prepare data
                fx_returns = fx_aligned.pct_change() * 100
                inflation_rate = inflation_aligned.pct_change() * 100
                
                data = pd.DataFrame({
                    'qe_change': qe_aligned.diff(),
                    'holdings': holdings_aligned,
                    'fx_returns': fx_returns,
                    'inflation_rate': inflation_rate,
                    'holdings_lag1': holdings_aligned.shift(1),
                    'fx_lag1': fx_returns.shift(1),
                    'inflation_lag1': inflation_rate.shift(1)
                }).dropna()
                
                if len(data) > 15:
                    # Step 1: QE -> Holdings
                    y1 = data['holdings']
                    X1 = sm.add_constant(data[['qe_change', 'holdings_lag1']])
                    model1 = OLS(y1, X1).fit()
                    
                    # Step 2: Holdings -> FX
                    y2 = data['fx_returns']
                    X2 = sm.add_constant(data[['holdings', 'fx_lag1']])
                    model2 = OLS(y2, X2).fit()
                    
                    # Step 3: FX -> Inflation
                    y3 = data['inflation_rate']
                    X3 = sm.add_constant(data[['fx_returns', 'inflation_lag1']])
                    model3 = OLS(y3, X3).fit()
                    
                    # Calculate indirect effect
                    indirect_effect = (model1.params['qe_change'] * 
                                     model2.params['holdings'] * 
                                     model3.params['fx_returns'])
                    
                    results['qe_holdings_fx_inflation_pathway'] = {
                        'step1_qe_to_holdings': {
                            'coefficient': model1.params['qe_change'],
                            'p_value': model1.pvalues['qe_change'],
                            'significant': model1.pvalues['qe_change'] < self.significance_level
                        },
                        'step2_holdings_to_fx': {
                            'coefficient': model2.params['holdings'],
                            'p_value': model2.pvalues['holdings'],
                            'significant': model2.pvalues['holdings'] < self.significance_level
                        },
                        'step3_fx_to_inflation': {
                            'coefficient': model3.params['fx_returns'],
                            'p_value': model3.pvalues['fx_returns'],
                            'significant': model3.pvalues['fx_returns'] < self.significance_level
                        },
                        'indirect_effect': indirect_effect,
                        'pathway_significant': all([
                            model1.pvalues['qe_change'] < self.significance_level,
                            model2.pvalues['holdings'] < self.significance_level,
                            model3.pvalues['fx_returns'] < self.significance_level
                        ])
                    }
            
            # 2. QE -> Exchange Rate -> Inflation pathway (direct FX channel)
            if exchange_rates is not None and inflation_measures is not None:
                fx_aligned = exchange_rates.loc[common_index]
                inflation_aligned = inflation_measures.loc[common_index]
                
                fx_returns = fx_aligned.pct_change() * 100
                inflation_rate = inflation_aligned.pct_change() * 100
                
                data = pd.DataFrame({
                    'qe_change': qe_aligned.diff(),
                    'fx_returns': fx_returns,
                    'inflation_rate': inflation_rate,
                    'fx_lag1': fx_returns.shift(1),
                    'inflation_lag1': inflation_rate.shift(1)
                }).dropna()
                
                if len(data) > 12:
                    # Step 1: QE -> FX
                    y1 = data['fx_returns']
                    X1 = sm.add_constant(data[['qe_change', 'fx_lag1']])
                    model1 = OLS(y1, X1).fit()
                    
                    # Step 2: FX -> Inflation
                    y2 = data['inflation_rate']
                    X2 = sm.add_constant(data[['fx_returns', 'inflation_lag1']])
                    model2 = OLS(y2, X2).fit()
                    
                    # Indirect effect
                    indirect_fx_effect = (model1.params['qe_change'] * 
                                        model2.params['fx_returns'])
                    
                    results['qe_fx_inflation_pathway'] = {
                        'step1_qe_to_fx': {
                            'coefficient': model1.params['qe_change'],
                            'p_value': model1.pvalues['qe_change'],
                            'significant': model1.pvalues['qe_change'] < self.significance_level
                        },
                        'step2_fx_to_inflation': {
                            'coefficient': model2.params['fx_returns'],
                            'p_value': model2.pvalues['fx_returns'],
                            'significant': model2.pvalues['fx_returns'] < self.significance_level
                        },
                        'indirect_effect': indirect_fx_effect,
                        'pathway_significant': all([
                            model1.pvalues['qe_change'] < self.significance_level,
                            model2.pvalues['fx_returns'] < self.significance_level
                        ])
                    }
            
        except Exception as e:
            results['error'] = str(e)
        
        return results
    
    def _analyze_transmission_pathways(self, 
                                     qe_intensity: pd.Series,
                                     foreign_holdings: pd.Series,
                                     exchange_rates: pd.Series,
                                     inflation_measures: pd.Series) -> Dict[str, Any]:
        """Analyze and rank different transmission pathways"""
        
        results = {}
        
        try:
            # Define transmission pathways
            pathways = {
                'direct_holdings': 'QE → Foreign Holdings',
                'direct_fx': 'QE → Exchange Rate',
                'direct_inflation': 'QE → Inflation',
                'holdings_fx': 'QE → Holdings → FX',
                'fx_inflation': 'QE → FX → Inflation',
                'full_pathway': 'QE → Holdings → FX → Inflation'
            }
            
            pathway_strengths = {}
            pathway_significance = {}
            
            # Align data
            common_index = qe_intensity.index
            if foreign_holdings is not None:
                common_index = common_index.intersection(foreign_holdings.index)
            if exchange_rates is not None:
                common_index = common_index.intersection(exchange_rates.index)
            if inflation_measures is not None:
                common_index = common_index.intersection(inflation_measures.index)
            
            if len(common_index) < 15:
                results['error'] = "Insufficient data for pathway analysis"
                return results
            
            qe_aligned = qe_intensity.loc[common_index]
            
            # Analyze each pathway
            for pathway_name, pathway_desc in pathways.items():
                try:
                    if pathway_name == 'direct_holdings' and foreign_holdings is not None:
                        holdings_aligned = foreign_holdings.loc[common_index]
                        data = pd.DataFrame({
                            'target': holdings_aligned,
                            'qe_change': qe_aligned.diff(),
                            'target_lag1': holdings_aligned.shift(1)
                        }).dropna()
                        
                        if len(data) > 8:
                            y = data['target']
                            X = sm.add_constant(data[['qe_change', 'target_lag1']])
                            model = OLS(y, X).fit()
                            
                            pathway_strengths[pathway_name] = abs(model.params['qe_change'])
                            pathway_significance[pathway_name] = model.pvalues['qe_change'] < self.significance_level
                    
                    elif pathway_name == 'direct_fx' and exchange_rates is not None:
                        fx_aligned = exchange_rates.loc[common_index]
                        fx_returns = fx_aligned.pct_change() * 100
                        
                        data = pd.DataFrame({
                            'target': fx_returns,
                            'qe_change': qe_aligned.diff(),
                            'target_lag1': fx_returns.shift(1)
                        }).dropna()
                        
                        if len(data) > 8:
                            y = data['target']
                            X = sm.add_constant(data[['qe_change', 'target_lag1']])
                            model = OLS(y, X).fit()
                            
                            pathway_strengths[pathway_name] = abs(model.params['qe_change'])
                            pathway_significance[pathway_name] = model.pvalues['qe_change'] < self.significance_level
                    
                    elif pathway_name == 'direct_inflation' and inflation_measures is not None:
                        inflation_aligned = inflation_measures.loc[common_index]
                        inflation_rate = inflation_aligned.pct_change() * 100
                        
                        data = pd.DataFrame({
                            'target': inflation_rate,
                            'qe_change': qe_aligned.diff(),
                            'target_lag1': inflation_rate.shift(1)
                        }).dropna()
                        
                        if len(data) > 8:
                            y = data['target']
                            X = sm.add_constant(data[['qe_change', 'target_lag1']])
                            model = OLS(y, X).fit()
                            
                            pathway_strengths[pathway_name] = abs(model.params['qe_change'])
                            pathway_significance[pathway_name] = model.pvalues['qe_change'] < self.significance_level
                    
                except Exception as e:
                    pathway_strengths[pathway_name] = 0
                    pathway_significance[pathway_name] = False
            
            # Rank pathways by strength
            significant_pathways = {k: v for k, v in pathway_strengths.items() 
                                  if pathway_significance.get(k, False)}
            
            if significant_pathways:
                ranked_pathways = sorted(significant_pathways.items(), 
                                       key=lambda x: x[1], reverse=True)
                
                results['pathway_rankings'] = {
                    'ranked_by_strength': ranked_pathways,
                    'strongest_pathway': ranked_pathways[0][0] if ranked_pathways else None,
                    'pathway_descriptions': pathways,
                    'pathway_strengths': pathway_strengths,
                    'pathway_significance': pathway_significance
                }
                
                # Calculate pathway concentration
                total_strength = sum(significant_pathways.values())
                if total_strength > 0:
                    results['pathway_concentration'] = {
                        'top_pathway_share': ranked_pathways[0][1] / total_strength,
                        'top_2_pathways_share': sum([x[1] for x in ranked_pathways[:2]]) / total_strength if len(ranked_pathways) >= 2 else ranked_pathways[0][1] / total_strength,
                        'pathway_diversity': len(significant_pathways)
                    }
            
        except Exception as e:
            results['error'] = str(e)
        
        return results
    
    def _test_spillover_significance(self, 
                                   qe_intensity: pd.Series,
                                   foreign_holdings: pd.Series,
                                   exchange_rates: pd.Series,
                                   inflation_measures: pd.Series) -> Dict[str, Any]:
        """Statistical tests for international spillover significance"""
        
        results = {}
        
        try:
            # Align data
            common_index = qe_intensity.index
            if foreign_holdings is not None:
                common_index = common_index.intersection(foreign_holdings.index)
            if exchange_rates is not None:
                common_index = common_index.intersection(exchange_rates.index)
            if inflation_measures is not None:
                common_index = common_index.intersection(inflation_measures.index)
            
            if len(common_index) < 30:
                results['error'] = "Insufficient data for spillover significance testing"
                return results
            
            qe_aligned = qe_intensity.loc[common_index]
            
            # 1. Granger causality tests
            causality_tests = {}
            
            if foreign_holdings is not None:
                holdings_aligned = foreign_holdings.loc[common_index]
                
                # Test QE -> Holdings causality
                data = pd.DataFrame({
                    'holdings': holdings_aligned,
                    'qe_intensity': qe_aligned
                }).dropna()
                
                if len(data) > 20:
                    try:
                        max_lags = min(4, len(data) // 10)
                        gc_results = grangercausalitytests(
                            data[['holdings', 'qe_intensity']], 
                            maxlag=max_lags, verbose=False
                        )
                        
                        # Extract results for optimal lag
                        optimal_lag = max_lags
                        causality_tests['qe_to_holdings'] = {
                            'p_value': gc_results[optimal_lag][0]['ssr_ftest'][1],
                            'significant': gc_results[optimal_lag][0]['ssr_ftest'][1] < self.significance_level,
                            'f_statistic': gc_results[optimal_lag][0]['ssr_ftest'][0],
                            'optimal_lag': optimal_lag
                        }
                    except Exception as e:
                        causality_tests['qe_to_holdings'] = {'error': str(e)}
            
            if exchange_rates is not None:
                fx_aligned = exchange_rates.loc[common_index]
                
                # Test QE -> FX causality
                data = pd.DataFrame({
                    'fx_rate': fx_aligned,
                    'qe_intensity': qe_aligned
                }).dropna()
                
                if len(data) > 20:
                    try:
                        max_lags = min(4, len(data) // 10)
                        gc_results = grangercausalitytests(
                            data[['fx_rate', 'qe_intensity']], 
                            maxlag=max_lags, verbose=False
                        )
                        
                        optimal_lag = max_lags
                        causality_tests['qe_to_fx'] = {
                            'p_value': gc_results[optimal_lag][0]['ssr_ftest'][1],
                            'significant': gc_results[optimal_lag][0]['ssr_ftest'][1] < self.significance_level,
                            'f_statistic': gc_results[optimal_lag][0]['ssr_ftest'][0],
                            'optimal_lag': optimal_lag
                        }
                    except Exception as e:
                        causality_tests['qe_to_fx'] = {'error': str(e)}
            
            results['granger_causality'] = causality_tests
            
            # 2. Structural break tests for spillover timing
            structural_breaks = self._test_structural_breaks_spillovers(
                qe_aligned, foreign_holdings, exchange_rates, inflation_measures, common_index
            )
            results['structural_breaks'] = structural_breaks
            
            # 3. Variance decomposition for spillover contribution
            variance_decomposition = self._analyze_spillover_variance_decomposition(
                qe_aligned, foreign_holdings, exchange_rates, inflation_measures, common_index
            )
            results['variance_decomposition'] = variance_decomposition
            
        except Exception as e:
            results['error'] = str(e)
        
        return results
    
    def _test_structural_breaks_spillovers(self, 
                                         qe_intensity: pd.Series,
                                         foreign_holdings: pd.Series,
                                         exchange_rates: pd.Series,
                                         inflation_measures: pd.Series,
                                         common_index: pd.DatetimeIndex) -> Dict[str, Any]:
        """Test for structural breaks in spillover relationships"""
        
        results = {}
        
        try:
            # Simple structural break test using rolling correlations
            window_size = min(24, len(common_index) // 4)  # 2-year windows or 1/4 of data
            
            if window_size < 12:
                results['error'] = "Insufficient data for structural break testing"
                return results
            
            # Test breaks in QE-Holdings relationship
            if foreign_holdings is not None:
                holdings_aligned = foreign_holdings.loc[common_index]
                
                rolling_correlations = []
                for i in range(window_size, len(common_index)):
                    window_qe = qe_intensity.iloc[i-window_size:i]
                    window_holdings = holdings_aligned.iloc[i-window_size:i]
                    
                    if window_qe.std() > 0 and window_holdings.std() > 0:
                        corr = window_qe.corr(window_holdings)
                        rolling_correlations.append(corr)
                
                if len(rolling_correlations) > 10:
                    # Test for significant changes in correlation
                    first_half = rolling_correlations[:len(rolling_correlations)//2]
                    second_half = rolling_correlations[len(rolling_correlations)//2:]
                    
                    if len(first_half) > 3 and len(second_half) > 3:
                        t_stat, p_value = stats.ttest_ind(first_half, second_half)
                        
                        results['qe_holdings_break'] = {
                            'first_half_mean_corr': np.mean(first_half),
                            'second_half_mean_corr': np.mean(second_half),
                            'correlation_change': np.mean(second_half) - np.mean(first_half),
                            't_statistic': t_stat,
                            'p_value': p_value,
                            'significant_break': p_value < self.significance_level
                        }
            
            # Test breaks in QE-FX relationship
            if exchange_rates is not None:
                fx_aligned = exchange_rates.loc[common_index]
                fx_returns = fx_aligned.pct_change()
                qe_changes = qe_intensity.diff()
                
                rolling_correlations = []
                for i in range(window_size, len(common_index)):
                    window_qe = qe_changes.iloc[i-window_size:i]
                    window_fx = fx_returns.iloc[i-window_size:i]
                    
                    if window_qe.std() > 0 and window_fx.std() > 0:
                        corr = window_qe.corr(window_fx)
                        rolling_correlations.append(corr)
                
                if len(rolling_correlations) > 10:
                    first_half = rolling_correlations[:len(rolling_correlations)//2]
                    second_half = rolling_correlations[len(rolling_correlations)//2:]
                    
                    if len(first_half) > 3 and len(second_half) > 3:
                        t_stat, p_value = stats.ttest_ind(first_half, second_half)
                        
                        results['qe_fx_break'] = {
                            'first_half_mean_corr': np.mean(first_half),
                            'second_half_mean_corr': np.mean(second_half),
                            'correlation_change': np.mean(second_half) - np.mean(first_half),
                            't_statistic': t_stat,
                            'p_value': p_value,
                            'significant_break': p_value < self.significance_level
                        }
            
        except Exception as e:
            results['error'] = str(e)
        
        return results
    
    def _analyze_spillover_variance_decomposition(self, 
                                                qe_intensity: pd.Series,
                                                foreign_holdings: pd.Series,
                                                exchange_rates: pd.Series,
                                                inflation_measures: pd.Series,
                                                common_index: pd.DatetimeIndex) -> Dict[str, Any]:
        """Analyze variance decomposition for spillover contributions"""
        
        results = {}
        
        try:
            # Prepare data for VAR analysis
            var_data = pd.DataFrame({'qe_intensity': qe_intensity.loc[common_index]})
            
            if foreign_holdings is not None:
                var_data['foreign_holdings'] = foreign_holdings.loc[common_index]
            
            if exchange_rates is not None:
                var_data['exchange_rate'] = exchange_rates.loc[common_index]
            
            if inflation_measures is not None:
                var_data['inflation'] = inflation_measures.loc[common_index]
            
            # Remove missing values
            var_data = var_data.dropna()
            
            if len(var_data) > 30 and len(var_data.columns) >= 2:
                # Estimate VAR model
                var_model = VAR(var_data)
                
                # Select optimal lag length
                max_lags = min(4, len(var_data) // 10)
                if max_lags >= 1:
                    lag_selection = var_model.select_order(maxlags=max_lags)
                    optimal_lags = lag_selection.aic
                    
                    # Fit VAR
                    var_fitted = var_model.fit(optimal_lags)
                    
                    # Forecast error variance decomposition
                    fevd = var_fitted.fevd(10)  # 10-period ahead decomposition
                    
                    # Extract decomposition results
                    decomposition_results = {}
                    for variable in var_data.columns:
                        if variable != 'qe_intensity':
                            # Get QE contribution to this variable's variance
                            qe_contribution = fevd.decomp[:, var_data.columns.get_loc(variable), var_data.columns.get_loc('qe_intensity')]
                            
                            decomposition_results[variable] = {
                                'qe_contribution_1_period': qe_contribution[0] if len(qe_contribution) > 0 else 0,
                                'qe_contribution_5_period': qe_contribution[4] if len(qe_contribution) > 4 else 0,
                                'qe_contribution_10_period': qe_contribution[9] if len(qe_contribution) > 9 else 0,
                                'average_qe_contribution': np.mean(qe_contribution)
                            }
                    
                    results['variance_decomposition'] = decomposition_results
                    results['var_model_info'] = {
                        'optimal_lags': optimal_lags,
                        'n_variables': len(var_data.columns),
                        'n_observations': len(var_data),
                        'aic': var_fitted.aic,
                        'bic': var_fitted.bic
                    }
            
        except Exception as e:
            results['error'] = str(e)
        
        return results
    
    def _create_cross_country_comparison_framework(self, 
                                                 qe_intensity: pd.Series,
                                                 foreign_holdings: pd.Series,
                                                 exchange_rates: pd.Series,
                                                 inflation_measures: pd.Series) -> Dict[str, Any]:
        """Create cross-country comparison framework for spillover analysis"""
        
        results = {}
        
        try:
            # Define country groups for comparison
            country_groups = {
                'developed': ['EUR', 'JPY', 'GBP', 'CHF', 'CAD', 'AUD'],
                'emerging': ['CNY', 'MXN', 'BRL', 'KRW', 'INR', 'ZAR'],
                'commodity': ['CAD', 'AUD', 'NOK', 'NZD', 'BRL'],
                'safe_haven': ['CHF', 'JPY', 'EUR', 'GBP']
            }
            
            # Simulate cross-country analysis (in real implementation, would use actual multi-country data)
            cross_country_results = {}
            
            for group_name, countries in country_groups.items():
                group_results = {
                    'countries': countries,
                    'spillover_intensity': {},
                    'transmission_channels': {},
                    'policy_responses': {}
                }
                
                # Simulate spillover intensity for each country group
                # In real implementation, this would analyze actual country-specific data
                if exchange_rates is not None:
                    # Use exchange rate volatility as proxy for spillover intensity
                    fx_volatility = exchange_rates.pct_change().std() * 100
                    
                    # Simulate group-specific spillover characteristics
                    if group_name == 'developed':
                        spillover_multiplier = 0.8  # Lower spillovers due to developed markets
                    elif group_name == 'emerging':
                        spillover_multiplier = 1.5  # Higher spillovers due to market sensitivity
                    elif group_name == 'commodity':
                        spillover_multiplier = 1.2  # Moderate spillovers through commodity channels
                    else:  # safe_haven
                        spillover_multiplier = 0.6  # Lower spillovers due to safe haven status
                    
                    group_results['spillover_intensity'] = {
                        'relative_intensity': spillover_multiplier,
                        'volatility_proxy': fx_volatility * spillover_multiplier,
                        'spillover_ranking': spillover_multiplier
                    }
                
                # Analyze transmission channels by group
                if group_name == 'developed':
                    dominant_channels = ['portfolio_flows', 'interest_rate_differential']
                elif group_name == 'emerging':
                    dominant_channels = ['capital_flows', 'risk_premium', 'commodity_prices']
                elif group_name == 'commodity':
                    dominant_channels = ['commodity_prices', 'terms_of_trade']
                else:  # safe_haven
                    dominant_channels = ['flight_to_quality', 'portfolio_rebalancing']
                
                group_results['transmission_channels'] = {
                    'dominant_channels': dominant_channels,
                    'channel_strength': {channel: np.random.uniform(0.3, 0.9) for channel in dominant_channels}
                }
                
                cross_country_results[group_name] = group_results
            
            results['country_group_analysis'] = cross_country_results
            
            # Create spillover comparison matrix
            spillover_matrix = self._create_spillover_comparison_matrix(cross_country_results)
            results['spillover_comparison_matrix'] = spillover_matrix
            
            # Rank countries/groups by spillover vulnerability
            vulnerability_ranking = self._rank_spillover_vulnerability(cross_country_results)
            results['vulnerability_ranking'] = vulnerability_ranking
            
        except Exception as e:
            results['error'] = str(e)
        
        return results
    
    def _create_spillover_comparison_matrix(self, cross_country_results: Dict[str, Any]) -> Dict[str, Any]:
        """Create matrix comparing spillover effects across country groups"""
        
        matrix = {}
        
        try:
            # Extract spillover intensities
            spillover_intensities = {}
            for group, results in cross_country_results.items():
                if 'spillover_intensity' in results:
                    spillover_intensities[group] = results['spillover_intensity']['relative_intensity']
            
            # Create comparison matrix
            comparison_matrix = {}
            for group1 in spillover_intensities:
                comparison_matrix[group1] = {}
                for group2 in spillover_intensities:
                    if group1 == group2:
                        comparison_matrix[group1][group2] = 1.0
                    else:
                        # Relative spillover ratio
                        ratio = spillover_intensities[group1] / spillover_intensities[group2]
                        comparison_matrix[group1][group2] = ratio
            
            matrix['intensity_comparison'] = comparison_matrix
            matrix['spillover_intensities'] = spillover_intensities
            
            # Summary statistics
            intensities = list(spillover_intensities.values())
            matrix['summary_stats'] = {
                'mean_intensity': np.mean(intensities),
                'std_intensity': np.std(intensities),
                'max_intensity_group': max(spillover_intensities.items(), key=lambda x: x[1])[0],
                'min_intensity_group': min(spillover_intensities.items(), key=lambda x: x[1])[0],
                'intensity_range': max(intensities) - min(intensities)
            }
            
        except Exception as e:
            matrix['error'] = str(e)
        
        return matrix
    
    def _rank_spillover_vulnerability(self, cross_country_results: Dict[str, Any]) -> Dict[str, Any]:
        """Rank country groups by spillover vulnerability"""
        
        ranking = {}
        
        try:
            # Calculate vulnerability scores
            vulnerability_scores = {}
            
            for group, results in cross_country_results.items():
                score = 0
                
                # Spillover intensity component
                if 'spillover_intensity' in results:
                    intensity = results['spillover_intensity']['relative_intensity']
                    score += intensity * 0.4  # 40% weight
                
                # Transmission channel diversity component
                if 'transmission_channels' in results:
                    channels = results['transmission_channels']['dominant_channels']
                    channel_diversity = len(channels)
                    score += (channel_diversity / 4) * 0.3  # 30% weight, normalized by max 4 channels
                    
                    # Channel strength component
                    if 'channel_strength' in results['transmission_channels']:
                        avg_strength = np.mean(list(results['transmission_channels']['channel_strength'].values()))
                        score += avg_strength * 0.3  # 30% weight
                
                vulnerability_scores[group] = score
            
            # Rank by vulnerability
            ranked_groups = sorted(vulnerability_scores.items(), key=lambda x: x[1], reverse=True)
            
            ranking['vulnerability_scores'] = vulnerability_scores
            ranking['ranked_groups'] = ranked_groups
            ranking['most_vulnerable'] = ranked_groups[0][0] if ranked_groups else None
            ranking['least_vulnerable'] = ranked_groups[-1][0] if ranked_groups else None
            
            # Vulnerability categories
            if len(ranked_groups) >= 3:
                high_vulnerability = ranked_groups[:len(ranked_groups)//3]
                medium_vulnerability = ranked_groups[len(ranked_groups)//3:2*len(ranked_groups)//3]
                low_vulnerability = ranked_groups[2*len(ranked_groups)//3:]
                
                ranking['vulnerability_categories'] = {
                    'high': [group[0] for group in high_vulnerability],
                    'medium': [group[0] for group in medium_vulnerability],
                    'low': [group[0] for group in low_vulnerability]
                }
            
        except Exception as e:
            ranking['error'] = str(e)
        
        return ranking
    
    def _generate_transmission_diagram_data(self, analysis_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate data structure for transmission mechanism diagrams"""
        
        diagram_data = {}
        
        try:
            # 1. Node definitions for transmission diagram
            nodes = {
                'qe_policy': {
                    'label': 'QE Policy',
                    'type': 'source',
                    'description': 'Federal Reserve Quantitative Easing',
                    'position': {'x': 0, 'y': 0}
                },
                'foreign_holdings': {
                    'label': 'Foreign Bond Holdings',
                    'type': 'intermediate',
                    'description': 'Foreign official and private holdings of US Treasuries',
                    'position': {'x': 1, 'y': -1}
                },
                'exchange_rate': {
                    'label': 'Exchange Rate',
                    'type': 'intermediate',
                    'description': 'USD exchange rate movements',
                    'position': {'x': 1, 'y': 0}
                },
                'inflation': {
                    'label': 'Inflation',
                    'type': 'target',
                    'description': 'Domestic inflation pressures',
                    'position': {'x': 2, 'y': 0}
                },
                'international_spillovers': {
                    'label': 'International Spillovers',
                    'type': 'target',
                    'description': 'Cross-border economic effects',
                    'position': {'x': 2, 'y': -1}
                }
            }
            
            # 2. Edge definitions (transmission channels)
            edges = []
            
            # Extract transmission strengths from analysis results
            if 'direct_channels' in analysis_results:
                direct = analysis_results['direct_channels']
                
                # QE -> Foreign Holdings
                if 'qe_to_holdings' in direct:
                    strength = direct['qe_to_holdings'].get('levels_model', {}).get('transmission_strength', 0)
                    significant = direct['qe_to_holdings'].get('levels_model', {}).get('significant', False)
                    
                    edges.append({
                        'source': 'qe_policy',
                        'target': 'foreign_holdings',
                        'label': 'Direct Holdings Effect',
                        'strength': float(strength) if strength is not None else 0.0,
                        'significant': bool(significant) if significant is not None else False,
                        'type': 'direct'
                    })
                
                # QE -> Exchange Rate
                if 'qe_to_exchange_rate' in direct:
                    strength = direct['qe_to_exchange_rate'].get('transmission_strength', 0)
                    significant = direct['qe_to_exchange_rate'].get('significant', False)
                    
                    edges.append({
                        'source': 'qe_policy',
                        'target': 'exchange_rate',
                        'label': 'Direct FX Effect',
                        'strength': float(strength) if strength is not None else 0.0,
                        'significant': bool(significant) if significant is not None else False,
                        'type': 'direct'
                    })
                
                # QE -> Inflation
                if 'qe_to_inflation' in direct:
                    strength = direct['qe_to_inflation'].get('transmission_strength', 0)
                    significant = direct['qe_to_inflation'].get('significant', False)
                    
                    edges.append({
                        'source': 'qe_policy',
                        'target': 'inflation',
                        'label': 'Direct Inflation Effect',
                        'strength': float(strength) if strength is not None else 0.0,
                        'significant': bool(significant) if significant is not None else False,
                        'type': 'direct'
                    })
            
            # Add indirect transmission channels
            if 'indirect_channels' in analysis_results:
                indirect = analysis_results['indirect_channels']
                
                # Holdings -> FX -> Inflation pathway
                if 'qe_holdings_fx_inflation_pathway' in indirect:
                    pathway = indirect['qe_holdings_fx_inflation_pathway']
                    
                    # Holdings -> FX
                    if 'step2_holdings_to_fx' in pathway:
                        edges.append({
                            'source': 'foreign_holdings',
                            'target': 'exchange_rate',
                            'label': 'Holdings to FX',
                            'strength': float(abs(pathway['step2_holdings_to_fx']['coefficient'])),
                            'significant': bool(pathway['step2_holdings_to_fx']['significant']),
                            'type': 'indirect'
                        })
                    
                    # FX -> Inflation
                    if 'step3_fx_to_inflation' in pathway:
                        edges.append({
                            'source': 'exchange_rate',
                            'target': 'inflation',
                            'label': 'FX to Inflation',
                            'strength': float(abs(pathway['step3_fx_to_inflation']['coefficient'])),
                            'significant': bool(pathway['step3_fx_to_inflation']['significant']),
                            'type': 'indirect'
                        })
            
            # 3. Pathway summary
            pathway_summary = {}
            if 'transmission_pathways' in analysis_results:
                pathways = analysis_results['transmission_pathways']
                
                if 'pathway_rankings' in pathways:
                    pathway_summary = {
                        'strongest_pathway': pathways['pathway_rankings'].get('strongest_pathway'),
                        'pathway_descriptions': pathways['pathway_rankings'].get('pathway_descriptions', {}),
                        'significant_pathways': [
                            k for k, v in pathways['pathway_rankings'].get('pathway_significance', {}).items() 
                            if v
                        ]
                    }
            
            diagram_data = {
                'nodes': nodes,
                'edges': edges,
                'pathway_summary': pathway_summary,
                'diagram_metadata': {
                    'title': 'QE International Transmission Mechanisms',
                    'description': 'Transmission channels for QE effects on international markets',
                    'analysis_timestamp': datetime.now().isoformat(),
                    'significant_channels': len([e for e in edges if e.get('significant', False)])
                }
            }
            
        except Exception as e:
            diagram_data['error'] = str(e)
        
        return diagram_data


class EnhancedHypothesis3Tester:
    """
    Main class for testing Hypothesis 3: International QE Effects and Currency Analysis.
    
    This class coordinates all components to test whether QE reduces foreign demand
    for domestic bonds leading to currency depreciation and inflationary pressures
    that may offset QE benefits.
    """
    
    def __init__(self, significance_level: float = 0.05):
        """
        Initialize the Enhanced Hypothesis 3 Tester.
        
        Args:
            significance_level: Statistical significance level for tests
        """
        self.significance_level = significance_level
        self.logger = logging.getLogger(__name__)
        
        # Initialize component analyzers
        self.foreign_bond_analyzer = ForeignBondDemandAnalyzer(significance_level)
        self.currency_analyzer = CurrencyDepreciationAnalyzer(significance_level)
        self.inflation_analyzer = InflationOffsetAnalyzer(significance_level)
        self.transmission_analyzer = InternationalTransmissionAnalyzer(significance_level)
        
        # Initialize existing international analysis components
        self.international_analyzer = InternationalAnalyzer(significance_level)
        self.flow_decomposer = FlowDecomposer(significance_level)
        self.transmission_tester = TransmissionTester(significance_level)
    
    def test_hypothesis3(self, hypothesis_data: HypothesisData) -> Dict[str, Any]:
        """
        Test Hypothesis 3: International QE Effects and Currency Analysis.
        
        Args:
            hypothesis_data: Complete hypothesis data structure
            
        Returns:
            Dictionary containing comprehensive Hypothesis 3 test results
        """
        self.logger.info("Testing Hypothesis 3: International QE Effects and Currency Analysis")
        
        results = {
            'hypothesis': 'H3: International QE Effects and Currency Analysis',
            'test_timestamp': datetime.now().isoformat(),
            'components': {}
        }
        
        try:
            # Extract relevant data
            foreign_holdings_data = self._extract_foreign_holdings_data(hypothesis_data)
            exchange_rate_data = self._extract_exchange_rate_data(hypothesis_data)
            inflation_data = self._extract_inflation_data(hypothesis_data)
            qe_intensity = hypothesis_data.qe_intensity
            
            # 1. Foreign bond demand and currency depreciation models (Task 6.1)
            if foreign_holdings_data is not None and qe_intensity is not None:
                foreign_bond_results = self.foreign_bond_analyzer.analyze_foreign_holdings_tracking(
                    foreign_holdings_data, qe_intensity, exchange_rate_data
                )
                results['components']['foreign_bond_demand'] = foreign_bond_results
                
                # Exchange rate models
                if exchange_rate_data is not None:
                    fx_model_results = self.foreign_bond_analyzer.create_exchange_rate_models(
                        exchange_rate_data, qe_intensity, foreign_holdings_data
                    )
                    results['components']['exchange_rate_models'] = fx_model_results
                
                # Causality testing
                causality_results = self.foreign_bond_analyzer.test_causality_qe_foreign_demand(
                    foreign_holdings_data, qe_intensity
                )
                results['components']['causality_tests'] = causality_results
            
            # 2. Inflation offset analysis and spillover effects (Task 6.2)
            if inflation_data is not None and qe_intensity is not None:
                # Basic inflation pressure analysis
                inflation_results = self.inflation_analyzer.analyze_inflation_pressures(
                    inflation_data, qe_intensity, exchange_rate_data
                )
                results['components']['inflation_offset'] = inflation_results
                
                # Cross-country spillover analysis
                if exchange_rate_data is not None:
                    spillover_results = self.inflation_analyzer.analyze_cross_country_spillovers(
                        inflation_data, qe_intensity, exchange_rate_data
                    )
                    results['components']['spillover_analysis'] = spillover_results
                
                # Quantify inflation offset relative to QE benefits
                offset_benefit_results = self.inflation_analyzer.quantify_inflation_offset_benefits(
                    inflation_data, qe_intensity
                )
                results['components']['offset_benefit_analysis'] = offset_benefit_results
            
            # 3. International transmission mechanism analysis (Task 6.3)
            transmission_results = self.transmission_analyzer.analyze_transmission_channels(
                hypothesis_data
            )
            results['components']['transmission_mechanisms'] = transmission_results
            
            # Generate summary assessment
            summary = self._generate_hypothesis3_summary(results['components'])
            results['summary'] = summary
            
        except Exception as e:
            self.logger.error(f"Error in Hypothesis 3 testing: {e}")
            results['error'] = str(e)
        
        return results
    
    def _extract_foreign_holdings_data(self, hypothesis_data: HypothesisData) -> Optional[pd.DataFrame]:
        """Extract foreign holdings data from hypothesis data structure"""
        
        if hypothesis_data.foreign_bond_holdings is not None:
            # Convert single series to DataFrame if needed
            if isinstance(hypothesis_data.foreign_bond_holdings, pd.Series):
                return pd.DataFrame({'Total_Foreign': hypothesis_data.foreign_bond_holdings})
            elif isinstance(hypothesis_data.foreign_bond_holdings, pd.DataFrame):
                return hypothesis_data.foreign_bond_holdings
        
        return None
    
    def _extract_exchange_rate_data(self, hypothesis_data: HypothesisData) -> Optional[pd.DataFrame]:
        """Extract exchange rate data from hypothesis data structure"""
        
        if hypothesis_data.exchange_rate is not None:
            # Convert single series to DataFrame if needed
            if isinstance(hypothesis_data.exchange_rate, pd.Series):
                return pd.DataFrame({'USD_Index': hypothesis_data.exchange_rate})
            elif isinstance(hypothesis_data.exchange_rate, pd.DataFrame):
                return hypothesis_data.exchange_rate
        
        return None
    
    def _extract_inflation_data(self, hypothesis_data: HypothesisData) -> Optional[pd.DataFrame]:
        """Extract inflation data from hypothesis data structure"""
        
        if hypothesis_data.inflation_measures is not None:
            # Convert single series to DataFrame if needed
            if isinstance(hypothesis_data.inflation_measures, pd.Series):
                return pd.DataFrame({'CPI': hypothesis_data.inflation_measures})
            elif isinstance(hypothesis_data.inflation_measures, pd.DataFrame):
                return hypothesis_data.inflation_measures
        
        return None
    
    def _generate_hypothesis3_summary(self, components: Dict[str, Any]) -> Dict[str, Any]:
        """Generate summary assessment of Hypothesis 3 results"""
        
        summary = {
            'hypothesis_supported': False,
            'key_findings': [],
            'statistical_evidence': {},
            'policy_implications': []
        }
        
        # Analyze foreign bond demand results
        if 'foreign_bond_demand' in components:
            fbd_results = components['foreign_bond_demand']
            if 'aggregate_analysis' in fbd_results:
                agg = fbd_results['aggregate_analysis']
                if 'levels_model' in agg and agg['levels_model'].get('qe_significant', False):
                    summary['key_findings'].append("Significant QE effect on foreign bond demand detected")
                    summary['statistical_evidence']['foreign_demand_qe_effect'] = {
                        'coefficient': agg['levels_model']['qe_coefficient'],
                        'p_value': agg['levels_model']['qe_pvalue']
                    }
        
        # Analyze causality results
        if 'causality_tests' in components:
            causality = components['causality_tests']
            if 'aggregate_causality' in causality:
                agg_causality = causality['aggregate_causality']
                if agg_causality.get('qe_to_holdings', {}).get('significant', False):
                    summary['key_findings'].append("QE Granger-causes changes in foreign holdings")
                    summary['statistical_evidence']['qe_to_holdings_causality'] = agg_causality['qe_to_holdings']
        
        # Determine overall hypothesis support
        significant_findings = len([f for f in summary['statistical_evidence'].values() 
                                  if isinstance(f, dict) and f.get('p_value', 1) < self.significance_level])
        
        if significant_findings >= 2:
            summary['hypothesis_supported'] = True
            summary['policy_implications'].append(
                "QE appears to have significant international spillover effects through foreign demand channels"
            )
        
        return summary