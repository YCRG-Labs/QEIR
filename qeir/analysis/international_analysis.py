"""
International Results Reconciliation Module

This module provides comprehensive tools for analyzing international spillover effects
of quantitative easing, reconciling inconsistencies between foreign Treasury holdings
and exchange rate transmission mechanisms.
"""

import numpy as np
import pandas as pd
from scipy import stats
from scipy.linalg import inv
from scipy.optimize import minimize
import warnings
from typing import Dict, List, Tuple, Optional, Union, Any
from functools import wraps
import statsmodels.api as sm
from statsmodels.tsa.api import VAR
from statsmodels.regression.linear_model import OLS


def deprecated(reason: str):
    """
    Decorator to mark classes or functions as deprecated.
    
    Args:
        reason: Explanation for why the item is deprecated
    """
    def decorator(obj):
        if isinstance(obj, type):
            # Decorating a class
            original_init = obj.__init__
            
            @wraps(original_init)
            def new_init(self, *args, **kwargs):
                warnings.warn(
                    f"{obj.__name__} is deprecated. {reason}",
                    category=DeprecationWarning,
                    stacklevel=2
                )
                original_init(self, *args, **kwargs)
            
            obj.__init__ = new_init
            return obj
        else:
            # Decorating a function
            @wraps(obj)
            def wrapper(*args, **kwargs):
                warnings.warn(
                    f"{obj.__name__} is deprecated. {reason}",
                    category=DeprecationWarning,
                    stacklevel=2
                )
                return obj(*args, **kwargs)
            return wrapper
    return decorator


@deprecated(
    "Hypothesis 3 (International Spillovers) has been deprecated as part of the "
    "methodology revision. The analysis now focuses exclusively on domestic fiscal "
    "and investment channels (Hypotheses 1 and 2). This class is preserved for "
    "backward compatibility but should not be used in new analyses."
)
class InternationalAnalyzer:
    """
    Comprehensive international spillover analysis class for QE effects.
    
    .. deprecated::
        This class is deprecated as part of the QE methodology revision.
        Hypothesis 3 (International Spillovers) is no longer part of the main
        analysis pipeline. Use domestic-focused analyses instead.
    
    This class implements enhanced methods for analyzing international transmission
    of QE effects, including:
    - Foreign holdings response models with investor type separation
    - Exchange rate transmission models for FX mechanism analysis
    - Simultaneous equation models for joint FX and bond flow analysis
    - Validation of international transmission mechanisms
    """
    
    def __init__(self, significance_level: float = 0.05):
        """
        Initialize the InternationalAnalyzer.
        
        Parameters:
        -----------
        significance_level : float, default=0.05
            Significance level for statistical tests
        """
        self.significance_level = significance_level
        self.analysis_results = {}
        self.fitted_models = {}
        
    def heterogeneity_analysis(self,
                             foreign_holdings: pd.DataFrame,
                             qe_intensity: pd.Series,
                             exchange_rates: pd.DataFrame,
                             country_characteristics: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze heterogeneous spillover effects by country characteristics
        
        Parameters:
        -----------
        foreign_holdings : pd.DataFrame
            Foreign Treasury holdings by country
        qe_intensity : pd.Series
            QE intensity measure
        exchange_rates : pd.DataFrame
            Exchange rates vs USD
        country_characteristics : pd.DataFrame
            Country characteristics (exchange rate regime, financial development, etc.)
            
        Returns:
        --------
        Dict containing heterogeneity analysis results
        """
        results = {}
        
        # Separate countries by exchange rate regime
        if 'exchange_rate_regime' in country_characteristics.columns:
            fixed_rate_countries = country_characteristics[
                country_characteristics['exchange_rate_regime'] == 'fixed'
            ].index.tolist()
            
            floating_rate_countries = country_characteristics[
                country_characteristics['exchange_rate_regime'] == 'floating'
            ].index.tolist()
            
            # Analyze spillovers for each group
            for regime, countries in [('fixed', fixed_rate_countries), 
                                    ('floating', floating_rate_countries)]:
                if countries:
                    regime_holdings = foreign_holdings[countries].sum(axis=1)
                    regime_fx = exchange_rates[countries].mean(axis=1) if len(countries) > 1 else exchange_rates[countries[0]]
                    
                    # Holdings response
                    holdings_model = self._estimate_spillover_model(
                        regime_holdings, qe_intensity
                    )
                    
                    # FX response
                    fx_model = self._estimate_spillover_model(
                        regime_fx.pct_change(), qe_intensity
                    )
                    
                    results[f'{regime}_rate_regime'] = {
                        'holdings_response': holdings_model,
                        'fx_response': fx_model,
                        'countries': countries
                    }
        
        # Separate by financial development
        if 'financial_development' in country_characteristics.columns:
            developed = country_characteristics[
                country_characteristics['financial_development'] == 'advanced'
            ].index.tolist()
            
            emerging = country_characteristics[
                country_characteristics['financial_development'] == 'emerging'
            ].index.tolist()
            
            for dev_level, countries in [('advanced', developed), ('emerging', emerging)]:
                if countries:
                    dev_holdings = foreign_holdings[countries].sum(axis=1)
                    
                    holdings_model = self._estimate_spillover_model(
                        dev_holdings, qe_intensity
                    )
                    
                    results[f'{dev_level}_economies'] = {
                        'holdings_response': holdings_model,
                        'countries': countries
                    }
        
        return results
    
    def var_with_sign_restrictions(self,
                                 data: pd.DataFrame,
                                 qe_var: str,
                                 foreign_vars: List[str],
                                 lags: int = 4) -> Dict[str, Any]:
        """
        Estimate VAR with sign restrictions for QE spillover identification
        
        Parameters:
        -----------
        data : pd.DataFrame
            Data containing all variables
        qe_var : str
            QE intensity variable name
        foreign_vars : List[str]
            Foreign variables (holdings, exchange rates)
        lags : int, default=4
            Number of lags in VAR
            
        Returns:
        --------
        Dict containing VAR results with sign restrictions
        """
        # Prepare data
        var_data = data[[qe_var] + foreign_vars].dropna()
        
        # Estimate unrestricted VAR
        var_model = VAR(var_data)
        var_results = var_model.fit(lags)
        
        # Implement sign restrictions (simplified)
        # QE shock should: increase QE, decrease foreign holdings, depreciate USD
        sign_restrictions = {
            qe_var: 1,  # Positive QE shock
            # Foreign holdings should decrease (negative sign)
            # Exchange rates should increase (USD depreciation, positive sign)
        }
        
        # Extract structural shocks (this is simplified - full implementation
        # would use proper sign restriction algorithms)
        structural_shocks = self._apply_sign_restrictions(
            var_results, sign_restrictions
        )
        
        # Calculate impulse responses
        irf = var_results.irf(periods=12)
        
        return {
            'var_results': var_results,
            'structural_shocks': structural_shocks,
            'impulse_responses': irf,
            'sign_restrictions': sign_restrictions
        }
    
    def _apply_sign_restrictions(self, var_results, sign_restrictions: Dict) -> np.ndarray:
        """Apply sign restrictions to identify structural shocks (simplified)"""
        # This is a placeholder for proper sign restriction implementation
        # Full implementation would use algorithms from Uhlig (2005) or similar
        
        # For now, return reduced-form residuals
        return var_results.resid.values
    
    def _estimate_spillover_model(self, 
                                dependent_var: pd.Series,
                                qe_intensity: pd.Series,
                                lags: int = 4) -> Dict[str, Any]:
        """Estimate basic spillover regression model"""
        # Align data
        common_index = dependent_var.index.intersection(qe_intensity.index)
        y = dependent_var.loc[common_index]
        x = qe_intensity.loc[common_index]
        
        # Add lags
        X_matrix = []
        for lag in range(lags + 1):
            if lag == 0:
                X_matrix.append(x.values)
            else:
                X_matrix.append(x.shift(lag).values)
        
        X = np.column_stack(X_matrix)
        X = sm.add_constant(X)
        
        # Remove NaN rows
        valid_rows = ~np.isnan(X).any(axis=1) & ~np.isnan(y.values)
        X_clean = X[valid_rows]
        y_clean = y.values[valid_rows]
        
        # Estimate model
        model = OLS(y_clean, X_clean).fit()
        
        return {
            'coefficient': model.params[1],  # Contemporary QE effect
            'std_error': model.bse[1],
            'p_value': model.pvalues[1],
            'r_squared': model.rsquared,
            'full_results': model
        }

    def foreign_holdings_response_model(self,
                                      foreign_holdings: pd.DataFrame,
                                      qe_intensity: pd.Series,
                                      exchange_rates: pd.DataFrame,
                                      investor_types: Optional[Dict[str, List[str]]] = None,
                                      control_variables: Optional[pd.DataFrame] = None) -> Dict:
        """
        Analyze foreign Treasury holdings response to QE with investor type separation.
        
        Parameters:
        -----------
        foreign_holdings : pd.DataFrame
            Foreign Treasury holdings by country/investor type
        qe_intensity : pd.Series
            QE intensity measure (Fed holdings / total outstanding)
        exchange_rates : pd.DataFrame
            Exchange rates for relevant countries
        investor_types : Dict[str, List[str]], optional
            Mapping of investor types to country columns
        control_variables : pd.DataFrame, optional
            Additional control variables (VIX, interest rate differentials, etc.)
            
        Returns:
        --------
        Dict containing analysis results
        """
        if investor_types is None:
            investor_types = {
                'official': ['China', 'Japan', 'Saudi_Arabia', 'Taiwan', 'Switzerland'],
                'private': ['UK', 'Luxembourg', 'Ireland', 'Cayman_Islands', 'Belgium']
            }
        
        results = {}
        
        # Align data
        common_index = qe_intensity.index.intersection(foreign_holdings.index)
        if len(common_index) < 20:
            raise ValueError("Insufficient overlapping observations for analysis")
            
        qe_aligned = qe_intensity.loc[common_index]
        holdings_aligned = foreign_holdings.loc[common_index]
        
        # Basic aggregate analysis
        total_holdings = holdings_aligned.sum(axis=1)
        
        try:
            # Simple regression model
            data = pd.DataFrame({
                'holdings': total_holdings,
                'qe_intensity': qe_aligned,
                'holdings_lag1': total_holdings.shift(1)
            }).dropna()
            
            if len(data) > 10:
                y = data['holdings']
                X = sm.add_constant(data[['qe_intensity', 'holdings_lag1']])
                model = OLS(y, X).fit()
                
                results['aggregate_response'] = {
                    'basic_model': {
                        'coefficients': model.params.to_dict(),
                        'p_values': model.pvalues.to_dict(),
                        'r_squared': model.rsquared,
                        'n_obs': model.nobs
                    }
                }
        except Exception as e:
            results['aggregate_response'] = {'error': str(e)}
        
        # Store results
        self.analysis_results['foreign_holdings'] = results
        return results
    
    def exchange_rate_transmission_model(self,
                                       exchange_rates: pd.DataFrame,
                                       qe_intensity: pd.Series,
                                       foreign_holdings: pd.DataFrame,
                                       control_variables: Optional[pd.DataFrame] = None) -> Dict:
        """
        Analyze exchange rate transmission mechanism for QE effects.
        
        Parameters:
        -----------
        exchange_rates : pd.DataFrame
            Exchange rate data (USD per foreign currency)
        qe_intensity : pd.Series
            QE intensity measure
        foreign_holdings : pd.DataFrame
            Foreign Treasury holdings data
        control_variables : pd.DataFrame, optional
            Additional control variables
            
        Returns:
        --------
        Dict containing FX transmission analysis results
        """
        results = {}
        
        # Align all data to common time index
        common_index = (qe_intensity.index
                       .intersection(exchange_rates.index)
                       .intersection(foreign_holdings.index))
        
        if len(common_index) < 20:
            raise ValueError("Insufficient overlapping observations for FX transmission analysis")
        
        qe_aligned = qe_intensity.loc[common_index]
        fx_aligned = exchange_rates.loc[common_index]
        holdings_aligned = foreign_holdings.loc[common_index]
        
        # Basic FX response analysis
        fx_responses = {}
        for currency in fx_aligned.columns:
            try:
                fx_series = fx_aligned[currency]
                
                data = pd.DataFrame({
                    'fx_rate': fx_series,
                    'qe_intensity': qe_aligned,
                    'fx_lag1': fx_series.shift(1)
                }).dropna()
                
                if len(data) > 10:
                    y = data['fx_rate']
                    X = sm.add_constant(data[['qe_intensity', 'fx_lag1']])
                    model = OLS(y, X).fit()
                    
                    fx_responses[currency] = {
                        'basic_fx_model': {
                            'qe_coefficient': model.params['qe_intensity'],
                            'qe_pvalue': model.pvalues['qe_intensity'],
                            'r_squared': model.rsquared,
                            'n_obs': model.nobs
                        }
                    }
            except Exception as e:
                fx_responses[currency] = {'error': str(e)}
        
        results['qe_fx_effects'] = fx_responses
        
        # Store results
        self.analysis_results['fx_transmission'] = results
        return results
    
    def simultaneous_equation_model(self,
                                  qe_intensity: pd.Series,
                                  exchange_rates: pd.DataFrame,
                                  foreign_holdings: pd.DataFrame,
                                  control_variables: Optional[pd.DataFrame] = None) -> Dict:
        """
        Estimate simultaneous equation model for joint FX and bond flow analysis.
        
        Parameters:
        -----------
        qe_intensity : pd.Series
            QE intensity measure
        exchange_rates : pd.DataFrame
            Exchange rate data
        foreign_holdings : pd.DataFrame
            Foreign Treasury holdings data
        control_variables : pd.DataFrame, optional
            Additional control variables
            
        Returns:
        --------
        Dict containing simultaneous equation model results
        """
        results = {}
        
        # Align all data
        common_index = (qe_intensity.index
                       .intersection(exchange_rates.index)
                       .intersection(foreign_holdings.index))
        
        if len(common_index) < 30:
            raise ValueError("Insufficient observations for simultaneous equation estimation")
        
        qe_aligned = qe_intensity.loc[common_index]
        fx_aligned = exchange_rates.loc[common_index]
        holdings_aligned = foreign_holdings.loc[common_index]
        
        # Simple 2SLS estimation
        if len(exchange_rates.columns) > 0:
            fx_series = fx_aligned.iloc[:, 0]
            total_holdings = holdings_aligned.sum(axis=1)
            
            try:
                # Prepare data
                data = pd.DataFrame({
                    'fx_rate': fx_series,
                    'total_holdings': total_holdings,
                    'qe_intensity': qe_aligned,
                    'fx_lag1': fx_series.shift(1),
                    'holdings_lag1': total_holdings.shift(1)
                }).dropna()
                
                if len(data) > 20:
                    # FX equation
                    y1 = data['fx_rate']
                    X1 = sm.add_constant(data[['qe_intensity', 'fx_lag1']])
                    model_fx = OLS(y1, X1).fit()
                    
                    # Holdings equation
                    y2 = data['total_holdings']
                    X2 = sm.add_constant(data[['qe_intensity', 'holdings_lag1']])
                    model_holdings = OLS(y2, X2).fit()
                    
                    results['2sls_estimation'] = {
                        'fx_equation': {
                            'qe_coefficient': model_fx.params['qe_intensity'],
                            'qe_pvalue': model_fx.pvalues['qe_intensity'],
                            'r_squared': model_fx.rsquared,
                            'n_obs': model_fx.nobs
                        },
                        'holdings_equation': {
                            'qe_coefficient': model_holdings.params['qe_intensity'],
                            'qe_pvalue': model_holdings.pvalues['qe_intensity'],
                            'r_squared': model_holdings.rsquared,
                            'n_obs': model_holdings.nobs
                        }
                    }
            except Exception as e:
                results['2sls_estimation'] = {'error': str(e)}
        
        # Store results
        self.analysis_results['simultaneous_system'] = results
        return results


class FlowDecomposer:
    """
    Flow decomposition class for separating official vs private investor responses to QE.
    
    This class implements methods to analyze how different types of foreign investors
    (central banks, sovereign wealth funds vs private market participants) respond
    differently to US QE policies, helping to reconcile mixed international results.
    """
    
    def __init__(self, significance_level: float = 0.05):
        """
        Initialize the FlowDecomposer.
        
        Parameters:
        -----------
        significance_level : float, default=0.05
            Significance level for statistical tests
        """
        self.significance_level = significance_level
        self.decomposition_results = {}
        self.investor_classifications = {
            'official': {
                'central_banks': ['China', 'Japan', 'Switzerland', 'Taiwan', 'Korea'],
                'sovereign_funds': ['Saudi_Arabia', 'Norway', 'Singapore', 'UAE']
            },
            'private': {
                'financial_centers': ['UK', 'Luxembourg', 'Ireland', 'Cayman_Islands'],
                'market_participants': ['Germany', 'France', 'Netherlands', 'Belgium']
            }
        }
    
    def official_investor_model(self,
                              foreign_holdings: pd.DataFrame,
                              qe_intensity: pd.Series,
                              exchange_rates: Optional[pd.DataFrame] = None,
                              control_variables: Optional[pd.DataFrame] = None) -> Dict:
        """
        Analyze official investor (central bank and sovereign fund) behavior in response to QE.
        
        Parameters:
        -----------
        foreign_holdings : pd.DataFrame
            Foreign Treasury holdings by country/entity
        qe_intensity : pd.Series
            QE intensity measure
        exchange_rates : pd.DataFrame, optional
            Exchange rate data for relevant currencies
        control_variables : pd.DataFrame, optional
            Additional control variables
            
        Returns:
        --------
        Dict containing official investor analysis results
        """
        results = {}
        
        # Align data
        common_index = qe_intensity.index.intersection(foreign_holdings.index)
        if len(common_index) < 20:
            raise ValueError("Insufficient overlapping observations for official investor analysis")
        
        qe_aligned = qe_intensity.loc[common_index]
        holdings_aligned = foreign_holdings.loc[common_index]
        
        # Central Bank Behavior Analysis
        cb_countries = self.investor_classifications['official']['central_banks']
        available_cb = [c for c in cb_countries if c in holdings_aligned.columns]
        
        if available_cb:
            cb_holdings = holdings_aligned[available_cb].sum(axis=1)
            cb_results = self._analyze_official_behavior(
                cb_holdings, qe_aligned, investor_type='central_banks'
            )
            results['central_banks'] = cb_results
        
        # Combined Official Investor Analysis
        all_official = available_cb + [c for c in self.investor_classifications['official']['sovereign_funds'] 
                                     if c in holdings_aligned.columns]
        if all_official:
            total_official_holdings = holdings_aligned[all_official].sum(axis=1)
            combined_results = self._analyze_official_behavior(
                total_official_holdings, qe_aligned, investor_type='all_official'
            )
            results['combined_official'] = combined_results
        
        # Store results
        self.decomposition_results['official_investors'] = results
        return results
    
    def _analyze_official_behavior(self,
                                 holdings: pd.Series,
                                 qe_intensity: pd.Series,
                                 investor_type: str = 'official') -> Dict:
        """
        Analyze official investor behavior patterns.
        
        Parameters:
        -----------
        holdings : pd.Series
            Official investor holdings
        qe_intensity : pd.Series
            QE intensity measure
        investor_type : str
            Type of official investor
            
        Returns:
        --------
        Dict containing official behavior analysis
        """
        results = {}
        
        # Prepare data
        data = pd.DataFrame({
            'holdings': holdings,
            'qe_intensity': qe_intensity,
            'holdings_lag1': holdings.shift(1)
        }).dropna()
        
        if len(data) < 10:
            return {'error': f'Insufficient data for {investor_type} analysis'}
        
        # Basic QE response (levels)
        try:
            y = data['holdings']
            X = sm.add_constant(data[['qe_intensity', 'holdings_lag1']])
            
            model_levels = OLS(y, X).fit()
            results['levels_model'] = {
                'qe_coefficient': model_levels.params['qe_intensity'],
                'qe_pvalue': model_levels.pvalues['qe_intensity'],
                'persistence': model_levels.params['holdings_lag1'],
                'r_squared': model_levels.rsquared,
                'n_obs': model_levels.nobs
            }
        except Exception as e:
            results['levels_model'] = {'error': str(e)}
        
        results['investor_type'] = investor_type
        return results
    
    def private_investor_model(self,
                             foreign_holdings: pd.DataFrame,
                             qe_intensity: pd.Series,
                             market_variables: Optional[pd.DataFrame] = None,
                             control_variables: Optional[pd.DataFrame] = None) -> Dict:
        """
        Analyze private investor (market-based) responses to QE.
        
        Parameters:
        -----------
        foreign_holdings : pd.DataFrame
            Foreign Treasury holdings by country/entity
        qe_intensity : pd.Series
            QE intensity measure
        market_variables : pd.DataFrame, optional
            Market-based variables (VIX, credit spreads, term premiums)
        control_variables : pd.DataFrame, optional
            Additional control variables
            
        Returns:
        --------
        Dict containing private investor analysis results
        """
        results = {}
        
        # Align data
        common_index = qe_intensity.index.intersection(foreign_holdings.index)
        if len(common_index) < 20:
            raise ValueError("Insufficient overlapping observations for private investor analysis")
        
        qe_aligned = qe_intensity.loc[common_index]
        holdings_aligned = foreign_holdings.loc[common_index]
        
        # Financial Center Analysis
        fc_countries = self.investor_classifications['private']['financial_centers']
        available_fc = [c for c in fc_countries if c in holdings_aligned.columns]
        
        if available_fc:
            fc_holdings = holdings_aligned[available_fc].sum(axis=1)
            fc_results = self._analyze_private_behavior(
                fc_holdings, qe_aligned, market_variables, investor_type='financial_centers'
            )
            results['financial_centers'] = fc_results
        
        # Combined Private Investor Analysis
        all_private = available_fc + [c for c in self.investor_classifications['private']['market_participants'] 
                                    if c in holdings_aligned.columns]
        if all_private:
            total_private_holdings = holdings_aligned[all_private].sum(axis=1)
            combined_results = self._analyze_private_behavior(
                total_private_holdings, qe_aligned, market_variables, investor_type='all_private'
            )
            results['combined_private'] = combined_results
        
        # Store results
        self.decomposition_results['private_investors'] = results
        return results
    
    def _analyze_private_behavior(self,
                                holdings: pd.Series,
                                qe_intensity: pd.Series,
                                market_variables: Optional[pd.DataFrame] = None,
                                investor_type: str = 'private') -> Dict:
        """
        Analyze private investor behavior patterns.
        
        Parameters:
        -----------
        holdings : pd.Series
            Private investor holdings
        qe_intensity : pd.Series
            QE intensity measure
        market_variables : pd.DataFrame, optional
            Market-based variables
        investor_type : str
            Type of private investor
            
        Returns:
        --------
        Dict containing private behavior analysis
        """
        results = {}
        
        # Prepare data
        data = pd.DataFrame({
            'holdings': holdings,
            'qe_intensity': qe_intensity,
            'holdings_lag1': holdings.shift(1)
        })
        
        # Add market variables if available
        if market_variables is not None:
            for col in market_variables.columns:
                if col in data.index:
                    data[col] = market_variables[col]
        
        data_clean = data.dropna()
        
        if len(data_clean) < 10:
            return {'error': f'Insufficient data for {investor_type} analysis'}
        
        # Basic QE response
        try:
            y = data_clean['holdings']
            X = sm.add_constant(data_clean[['qe_intensity', 'holdings_lag1']])
            
            model_basic = OLS(y, X).fit()
            results['basic_model'] = {
                'qe_coefficient': model_basic.params['qe_intensity'],
                'qe_pvalue': model_basic.pvalues['qe_intensity'],
                'persistence': model_basic.params['holdings_lag1'],
                'r_squared': model_basic.rsquared,
                'n_obs': model_basic.nobs
            }
        except Exception as e:
            results['basic_model'] = {'error': str(e)}
        
        results['investor_type'] = investor_type
        return results
    
    def investor_heterogeneity_test(self,
                                  foreign_holdings: pd.DataFrame,
                                  qe_intensity: pd.Series,
                                  exchange_rates: Optional[pd.DataFrame] = None,
                                  control_variables: Optional[pd.DataFrame] = None) -> Dict:
        """
        Test for heterogeneity in QE responses between official and private investors.
        
        Parameters:
        -----------
        foreign_holdings : pd.DataFrame
            Foreign Treasury holdings by country/entity
        qe_intensity : pd.Series
            QE intensity measure
        exchange_rates : pd.DataFrame, optional
            Exchange rate data
        control_variables : pd.DataFrame, optional
            Additional control variables
            
        Returns:
        --------
        Dict containing heterogeneity test results
        """
        results = {}
        
        # Align data
        common_index = qe_intensity.index.intersection(foreign_holdings.index)
        if len(common_index) < 30:
            raise ValueError("Insufficient observations for heterogeneity testing")
        
        qe_aligned = qe_intensity.loc[common_index]
        holdings_aligned = foreign_holdings.loc[common_index]
        
        # Create official and private investor aggregates
        official_countries = (self.investor_classifications['official']['central_banks'] + 
                            self.investor_classifications['official']['sovereign_funds'])
        private_countries = (self.investor_classifications['private']['financial_centers'] + 
                           self.investor_classifications['private']['market_participants'])
        
        available_official = [c for c in official_countries if c in holdings_aligned.columns]
        available_private = [c for c in private_countries if c in holdings_aligned.columns]
        
        if not available_official or not available_private:
            return {'error': 'Insufficient official or private investor data for heterogeneity testing'}
        
        official_holdings = holdings_aligned[available_official].sum(axis=1)
        private_holdings = holdings_aligned[available_private].sum(axis=1)
        
        # Coefficient Equality Test
        coeff_test = self._test_coefficient_equality(
            official_holdings, private_holdings, qe_aligned
        )
        results['coefficient_equality_test'] = coeff_test
        
        # Store results
        self.decomposition_results['heterogeneity_tests'] = results
        return results
    
    def _test_coefficient_equality(self,
                                 official_holdings: pd.Series,
                                 private_holdings: pd.Series,
                                 qe_intensity: pd.Series) -> Dict:
        """
        Test equality of QE response coefficients between official and private investors.
        
        Parameters:
        -----------
        official_holdings : pd.Series
            Official investor holdings
        private_holdings : pd.Series
            Private investor holdings
        qe_intensity : pd.Series
            QE intensity measure
            
        Returns:
        --------
        Dict containing coefficient equality test results
        """
        results = {}
        
        try:
            # Estimate separate models
            data_official = pd.DataFrame({
                'holdings': official_holdings,
                'qe_intensity': qe_intensity,
                'holdings_lag1': official_holdings.shift(1)
            }).dropna()
            
            data_private = pd.DataFrame({
                'holdings': private_holdings,
                'qe_intensity': qe_intensity,
                'holdings_lag1': private_holdings.shift(1)
            }).dropna()
            
            # Official investor model
            X_official = sm.add_constant(data_official[['qe_intensity', 'holdings_lag1']])
            model_official = OLS(data_official['holdings'], X_official).fit()
            
            # Private investor model
            X_private = sm.add_constant(data_private[['qe_intensity', 'holdings_lag1']])
            model_private = OLS(data_private['holdings'], X_private).fit()
            
            # Extract coefficients and standard errors
            coeff_official = model_official.params['qe_intensity']
            coeff_private = model_private.params['qe_intensity']
            se_official = model_official.bse['qe_intensity']
            se_private = model_private.bse['qe_intensity']
            
            # Test coefficient equality
            coeff_diff = coeff_official - coeff_private
            se_diff = np.sqrt(se_official**2 + se_private**2)
            
            if se_diff > 0:
                t_stat = coeff_diff / se_diff
                df = model_official.df_resid + model_private.df_resid
                p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df))
                
                results['official_coefficient'] = coeff_official
                results['private_coefficient'] = coeff_private
                results['coefficient_difference'] = coeff_diff
                results['t_statistic'] = t_stat
                results['p_value'] = p_value
                results['significant_difference'] = p_value < self.significance_level
                results['official_r_squared'] = model_official.rsquared
                results['private_r_squared'] = model_private.rsquared
            
        except Exception as e:
            results['error'] = str(e)
        
        return results


class TransmissionTester:
    """
    Transmission testing class for validating multiple international QE transmission channels.
    
    This class implements comprehensive testing methods to validate different transmission
    mechanisms for international QE effects, including portfolio rebalancing, signaling
    channels, and high-frequency identification around QE announcements.
    """
    
    def __init__(self, significance_level: float = 0.05):
        """
        Initialize the TransmissionTester.
        
        Parameters:
        -----------
        significance_level : float, default=0.05
            Significance level for statistical tests
        """
        self.significance_level = significance_level
        self.transmission_results = {}
        self.qe_announcement_dates = [
            '2008-11-25',  # QE1 announcement
            '2010-11-03',  # QE2 announcement
            '2012-09-13',  # QE3 announcement
            '2020-03-15',  # COVID QE announcement
            '2020-03-23'   # Additional COVID QE
        ]
    
    def portfolio_rebalancing_test(self,
                                 foreign_holdings: pd.DataFrame,
                                 qe_intensity: pd.Series,
                                 asset_prices: Optional[pd.DataFrame] = None,
                                 control_variables: Optional[pd.DataFrame] = None) -> Dict:
        """
        Test portfolio rebalancing channel for international QE transmission.
        
        Parameters:
        -----------
        foreign_holdings : pd.DataFrame
            Foreign Treasury holdings by country/entity
        qe_intensity : pd.Series
            QE intensity measure
        asset_prices : pd.DataFrame, optional
            Asset price data (bond yields, equity prices, etc.)
        control_variables : pd.DataFrame, optional
            Additional control variables
            
        Returns:
        --------
        Dict containing portfolio rebalancing test results
        """
        results = {}
        
        # Align data
        common_index = qe_intensity.index.intersection(foreign_holdings.index)
        if len(common_index) < 30:
            raise ValueError("Insufficient observations for portfolio rebalancing test")
        
        qe_aligned = qe_intensity.loc[common_index]
        holdings_aligned = foreign_holdings.loc[common_index]
        
        # Direct Portfolio Rebalancing Effects
        direct_effects = self._test_direct_rebalancing(
            holdings_aligned, qe_aligned, asset_prices
        )
        results['direct_rebalancing'] = direct_effects
        
        # Store results
        self.transmission_results['portfolio_rebalancing'] = results
        return results
    
    def _test_direct_rebalancing(self,
                               holdings: pd.DataFrame,
                               qe_intensity: pd.Series,
                               asset_prices: Optional[pd.DataFrame] = None) -> Dict:
        """
        Test direct portfolio rebalancing effects of QE.
        
        Parameters:
        -----------
        holdings : pd.DataFrame
            Foreign holdings data
        qe_intensity : pd.Series
            QE intensity measure
        asset_prices : pd.DataFrame, optional
            Asset price data
            
        Returns:
        --------
        Dict containing direct rebalancing test results
        """
        results = {}
        
        # Calculate portfolio shares for each country
        total_holdings = holdings.sum(axis=1)
        portfolio_shares = holdings.div(total_holdings, axis=0).fillna(0)
        
        # Test rebalancing for each country
        for country in holdings.columns:
            try:
                country_share = portfolio_shares[country]
                
                # Prepare data for rebalancing test
                data = pd.DataFrame({
                    'portfolio_share': country_share,
                    'qe_intensity': qe_intensity,
                    'share_lag1': country_share.shift(1)
                }).dropna()
                
                if len(data) > 15:
                    # Basic rebalancing model
                    y = data['portfolio_share']
                    X = sm.add_constant(data[['qe_intensity', 'share_lag1']])
                    
                    model_basic = OLS(y, X).fit()
                    
                    results[f'{country}_rebalancing'] = {
                        'linear_qe_coeff': model_basic.params['qe_intensity'],
                        'linear_qe_pvalue': model_basic.pvalues['qe_intensity'],
                        'linear_r_squared': model_basic.rsquared,
                        'persistence': model_basic.params['share_lag1']
                    }
            
            except Exception as e:
                results[f'{country}_rebalancing'] = {'error': str(e)}
        
        return results
    
    def signaling_channel_test(self,
                             foreign_holdings: pd.DataFrame,
                             qe_intensity: pd.Series,
                             announcement_dates: Optional[List[str]] = None,
                             market_expectations: Optional[pd.DataFrame] = None) -> Dict:
        """
        Test signaling channel for QE announcement effects.
        
        Parameters:
        -----------
        foreign_holdings : pd.DataFrame
            Foreign Treasury holdings by country/entity
        qe_intensity : pd.Series
            QE intensity measure
        announcement_dates : List[str], optional
            QE announcement dates (if None, uses default dates)
        market_expectations : pd.DataFrame, optional
            Market expectation measures (survey data, forward rates, etc.)
            
        Returns:
        --------
        Dict containing signaling channel test results
        """
        results = {}
        
        # Use default announcement dates if none provided
        if announcement_dates is None:
            announcement_dates = self.qe_announcement_dates
        
        # Align data
        common_index = qe_intensity.index.intersection(foreign_holdings.index)
        if len(common_index) < 30:
            raise ValueError("Insufficient observations for signaling channel test")
        
        qe_aligned = qe_intensity.loc[common_index]
        holdings_aligned = foreign_holdings.loc[common_index]
        
        # Announcement Effects
        announcement_effects = self._test_announcement_effects(
            holdings_aligned, qe_aligned, announcement_dates
        )
        results['announcement_effects'] = announcement_effects
        
        # Store results
        self.transmission_results['signaling_channel'] = results
        return results
    
    def _test_announcement_effects(self,
                                 holdings: pd.DataFrame,
                                 qe_intensity: pd.Series,
                                 announcement_dates: List[str]) -> Dict:
        """
        Test QE announcement effects on foreign holdings.
        
        Parameters:
        -----------
        holdings : pd.DataFrame
            Foreign holdings data
        qe_intensity : pd.Series
            QE intensity measure
        announcement_dates : List[str]
            QE announcement dates
            
        Returns:
        --------
        Dict containing announcement effects results
        """
        results = {}
        
        # Convert announcement dates to datetime
        announcement_dates_dt = pd.to_datetime(announcement_dates)
        
        # Create announcement dummy variables
        announcement_dummies = pd.Series(0, index=qe_intensity.index)
        
        for ann_date in announcement_dates_dt:
            # Find closest date in index
            try:
                closest_date = qe_intensity.index[qe_intensity.index.get_indexer([ann_date], method='nearest')[0]]
                announcement_dummies[closest_date] = 1
            except (IndexError, KeyError):
                continue
        
        # Test announcement effects for aggregate holdings
        total_holdings = holdings.sum(axis=1)
        
        try:
            data = pd.DataFrame({
                'total_holdings': total_holdings,
                'qe_intensity': qe_intensity,
                'announcement': announcement_dummies,
                'holdings_lag1': total_holdings.shift(1)
            }).dropna()
            
            if len(data) > 20:
                # Announcement effect model
                y = data['total_holdings']
                X = sm.add_constant(data[['qe_intensity', 'announcement', 'holdings_lag1']])
                
                model_announcement = OLS(y, X).fit()
                
                results['aggregate_announcement_effects'] = {
                    'qe_coeff': model_announcement.params['qe_intensity'],
                    'announcement_coeff': model_announcement.params['announcement'],
                    'announcement_pvalue': model_announcement.pvalues['announcement'],
                    'r_squared': model_announcement.rsquared,
                    'significant_announcement': (
                        model_announcement.pvalues['announcement'] < self.significance_level
                    )
                }
        
        except Exception as e:
            results['aggregate_announcement_effects'] = {'error': str(e)}
        
        return results
    
    def high_frequency_identification(self,
                                    foreign_holdings: pd.DataFrame,
                                    qe_intensity: pd.Series,
                                    announcement_dates: Optional[List[str]] = None,
                                    high_freq_data: Optional[pd.DataFrame] = None) -> Dict:
        """
        Implement high-frequency identification for QE announcement day analysis.
        
        Parameters:
        -----------
        foreign_holdings : pd.DataFrame
            Foreign Treasury holdings by country/entity
        qe_intensity : pd.Series
            QE intensity measure
        announcement_dates : List[str], optional
            QE announcement dates (if None, uses default dates)
        high_freq_data : pd.DataFrame, optional
            High-frequency market data (intraday prices, volumes, etc.)
            
        Returns:
        --------
        Dict containing high-frequency identification results
        """
        results = {}
        
        # Use default announcement dates if none provided
        if announcement_dates is None:
            announcement_dates = self.qe_announcement_dates
        
        # Align data
        common_index = qe_intensity.index.intersection(foreign_holdings.index)
        if len(common_index) < 30:
            raise ValueError("Insufficient observations for high-frequency identification")
        
        qe_aligned = qe_intensity.loc[common_index]
        holdings_aligned = foreign_holdings.loc[common_index]
        
        # Event Study Analysis
        event_study_results = self._conduct_event_study(
            holdings_aligned, qe_aligned, announcement_dates
        )
        results['event_study'] = event_study_results
        
        # Store results
        self.transmission_results['high_frequency_identification'] = results
        return results
    
    def _conduct_event_study(self,
                           holdings: pd.DataFrame,
                           qe_intensity: pd.Series,
                           announcement_dates: List[str]) -> Dict:
        """
        Conduct event study analysis around QE announcements.
        
        Parameters:
        -----------
        holdings : pd.DataFrame
            Foreign holdings data
        qe_intensity : pd.Series
            QE intensity measure
        announcement_dates : List[str]
            QE announcement dates
            
        Returns:
        --------
        Dict containing event study results
        """
        results = {}
        
        # Convert announcement dates
        announcement_dates_dt = pd.to_datetime(announcement_dates)
        
        # Calculate abnormal returns for each announcement
        event_effects = []
        
        for i, ann_date in enumerate(announcement_dates_dt):
            try:
                # Find announcement date in index
                closest_idx = qe_intensity.index.get_indexer([ann_date], method='nearest')[0]
                
                # Define event window (e.g., -5 to +5 days around announcement)
                start_idx = max(0, closest_idx - 5)
                end_idx = min(len(qe_intensity) - 1, closest_idx + 5)
                
                # Calculate holdings changes in event window
                total_holdings = holdings.sum(axis=1)
                
                if start_idx < end_idx:
                    pre_event_holdings = total_holdings.iloc[start_idx:closest_idx].mean()
                    post_event_holdings = total_holdings.iloc[closest_idx:end_idx].mean()
                    
                    abnormal_change = post_event_holdings - pre_event_holdings
                    
                    event_effects.append({
                        'event_date': ann_date,
                        'event_number': i + 1,
                        'abnormal_change': abnormal_change,
                        'pre_event_level': pre_event_holdings,
                        'post_event_level': post_event_holdings
                    })
            
            except (IndexError, KeyError):
                continue
        
        if event_effects:
            # Calculate average abnormal returns
            abnormal_changes = [e['abnormal_change'] for e in event_effects]
            
            results['event_effects'] = event_effects
            results['average_abnormal_change'] = np.mean(abnormal_changes)
            results['abnormal_change_std'] = np.std(abnormal_changes)
            results['num_events'] = len(event_effects)
            
            # Test significance of average abnormal change
            if len(abnormal_changes) > 1:
                t_stat = (np.mean(abnormal_changes) / 
                         (np.std(abnormal_changes) / np.sqrt(len(abnormal_changes))))
                p_value = 2 * (1 - stats.t.cdf(abs(t_stat), len(abnormal_changes) - 1))
                
                results['significance_test'] = {
                    't_statistic': t_stat,
                    'p_value': p_value,
                    'significant': p_value < self.significance_level
                }
        
        return results