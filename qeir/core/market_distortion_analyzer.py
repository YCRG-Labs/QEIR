"""
Market Distortion vs Interest Rate Channel Decomposition

This module implements sophisticated analysis to separate QE transmission effects
through interest rate channels vs market distortion channels (μ₂), with statistical
tests for dominance of distortion effects.

Key Components:
1. Market Distortion Proxies (μ₂) from bid-ask spreads and liquidity measures
2. Channel Decomposition Models separating interest rate and distortion effects
3. Statistical Tests for Dominance of Distortion Effects
4. Robustness Testing across different specifications
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from scipy import stats
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
from statsmodels.tsa.api import VAR


@dataclass
class MarketDistortionConfig:
    """Configuration for market distortion analysis"""
    
    # Distortion proxy construction
    bid_ask_weight: float = 0.4  # Weight for bid-ask spread component
    liquidity_weight: float = 0.3  # Weight for liquidity measures
    volatility_weight: float = 0.3  # Weight for volatility measures
    
    # Normalization and scaling
    use_standardization: bool = True
    rolling_window: int = 12  # Months for rolling statistics
    
    # Channel decomposition
    decomposition_method: str = 'interaction'  # 'interaction', 'structural', 'var'
    bootstrap_iterations: int = 1000
    confidence_level: float = 0.95
    
    # Dominance testing
    dominance_test_type: str = 'one_sided'  # 'one_sided', 'two_sided'
    significance_level: float = 0.05


@dataclass
class DistortionProxyResults:
    """Results from market distortion proxy construction"""
    
    # Constructed proxy
    distortion_proxy: pd.Series = field(default_factory=pd.Series)
    
    # Component contributions
    bid_ask_component: pd.Series = field(default_factory=pd.Series)
    liquidity_component: pd.Series = field(default_factory=pd.Series)
    volatility_component: pd.Series = field(default_factory=pd.Series)
    
    # Proxy statistics
    proxy_mean: float = 0.0
    proxy_std: float = 0.0
    proxy_skewness: float = 0.0
    proxy_kurtosis: float = 0.0
    
    # Component correlations
    component_correlations: pd.DataFrame = field(default_factory=pd.DataFrame)
    
    # Quality metrics
    construction_success: bool = False
    missing_data_pct: float = 0.0
    observations: int = 0


@dataclass
class ChannelDecompositionResults:
    """Results from channel decomposition analysis"""
    
    # Interest rate channel results
    interest_rate_effect: float = 0.0
    interest_rate_se: float = 0.0
    interest_rate_pvalue: float = 1.0
    interest_rate_ci_lower: float = 0.0
    interest_rate_ci_upper: float = 0.0
    
    # Market distortion channel results
    distortion_effect: float = 0.0
    distortion_se: float = 0.0
    distortion_pvalue: float = 1.0
    distortion_ci_lower: float = 0.0
    distortion_ci_upper: float = 0.0
    
    # Dominance test results
    distortion_dominance: bool = False
    dominance_test_stat: float = 0.0
    dominance_pvalue: float = 1.0
    dominance_ci_lower: float = 0.0
    dominance_ci_upper: float = 0.0
    
    # Decomposition statistics
    total_effect: float = 0.0
    distortion_share: float = 0.0
    interest_rate_share: float = 0.0
    
    # Model diagnostics
    r_squared: float = 0.0
    adjusted_r_squared: float = 0.0
    f_statistic: float = 0.0
    f_pvalue: float = 1.0
    observations: int = 0
    fitted: bool = False
    
    # Bootstrap results
    bootstrap_results: Dict[str, Any] = field(default_factory=dict)


class MarketDistortionProxyBuilder:
    """
    Constructs market distortion proxies (μ₂) from bid-ask spreads and liquidity measures
    """
    
    def __init__(self, config: Optional[MarketDistortionConfig] = None):
        """
        Initialize market distortion proxy builder
        
        Args:
            config: Configuration for distortion proxy construction
        """
        self.config = config or MarketDistortionConfig()
        self.logger = logging.getLogger(__name__)
        
        # Storage for results
        self.results: Optional[DistortionProxyResults] = None
        
    def build_distortion_proxy(self,
                              bid_ask_spreads: pd.Series,
                              liquidity_measures: pd.Series,
                              volatility_measures: pd.Series,
                              additional_proxies: Optional[Dict[str, pd.Series]] = None) -> DistortionProxyResults:
        """
        Build comprehensive market distortion proxy from multiple components
        
        Args:
            bid_ask_spreads: Bid-ask spread measures
            liquidity_measures: Market liquidity indicators
            volatility_measures: Market volatility measures
            additional_proxies: Additional distortion proxies
            
        Returns:
            DistortionProxyResults object
        """
        self.logger.info("Building market distortion proxy")
        
        results = DistortionProxyResults()
        
        try:
            # Align all series to common dates
            common_dates = bid_ask_spreads.index.intersection(liquidity_measures.index)\
                                                .intersection(volatility_measures.index)
            
            if len(common_dates) < 20:
                raise ValueError(f"Insufficient overlapping data: only {len(common_dates)} observations")
            
            # Align data
            bid_ask_aligned = bid_ask_spreads.loc[common_dates]
            liquidity_aligned = liquidity_measures.loc[common_dates]
            volatility_aligned = volatility_measures.loc[common_dates]
            
            # Standardize components if requested
            if self.config.use_standardization:
                bid_ask_std = (bid_ask_aligned - bid_ask_aligned.mean()) / bid_ask_aligned.std()
                liquidity_std = (liquidity_aligned - liquidity_aligned.mean()) / liquidity_aligned.std()
                volatility_std = (volatility_aligned - volatility_aligned.mean()) / volatility_aligned.std()
            else:
                bid_ask_std = bid_ask_aligned
                liquidity_std = liquidity_aligned
                volatility_std = volatility_aligned
            
            # Store standardized components
            results.bid_ask_component = bid_ask_std
            results.liquidity_component = liquidity_std
            results.volatility_component = volatility_std
            
            # Construct weighted composite proxy
            distortion_proxy = (
                self.config.bid_ask_weight * bid_ask_std +
                self.config.liquidity_weight * liquidity_std +
                self.config.volatility_weight * volatility_std
            )
            
            # Handle additional proxies if provided
            if additional_proxies:
                additional_weight = 1.0 - (self.config.bid_ask_weight + 
                                         self.config.liquidity_weight + 
                                         self.config.volatility_weight)
                
                if additional_weight > 0:
                    additional_component = pd.Series(0.0, index=common_dates)
                    weight_per_additional = additional_weight / len(additional_proxies)
                    
                    for name, proxy in additional_proxies.items():
                        proxy_aligned = proxy.loc[common_dates]
                        if self.config.use_standardization:
                            proxy_std = (proxy_aligned - proxy_aligned.mean()) / proxy_aligned.std()
                        else:
                            proxy_std = proxy_aligned
                        
                        additional_component += weight_per_additional * proxy_std
                    
                    distortion_proxy += additional_component
            
            # Apply rolling smoothing if specified
            if self.config.rolling_window > 1:
                distortion_proxy = distortion_proxy.rolling(
                    window=self.config.rolling_window, 
                    center=True
                ).mean().bfill().ffill()
            
            # Store final proxy
            results.distortion_proxy = distortion_proxy
            
            # Calculate proxy statistics
            results.proxy_mean = distortion_proxy.mean()
            results.proxy_std = distortion_proxy.std()
            results.proxy_skewness = distortion_proxy.skew()
            results.proxy_kurtosis = distortion_proxy.kurtosis()
            results.observations = len(distortion_proxy)
            results.missing_data_pct = distortion_proxy.isna().sum() / len(distortion_proxy) * 100
            
            # Calculate component correlations
            component_df = pd.DataFrame({
                'bid_ask': results.bid_ask_component,
                'liquidity': results.liquidity_component,
                'volatility': results.volatility_component,
                'composite': results.distortion_proxy
            })
            results.component_correlations = component_df.corr()
            
            results.construction_success = True
            self.logger.info(f"Successfully built distortion proxy with {results.observations} observations")
            
        except Exception as e:
            self.logger.error(f"Error building distortion proxy: {e}")
            results.construction_success = False
            results.distortion_proxy = pd.Series(dtype=float)
        
        self.results = results
        return results


class ChannelDecompositionAnalyzer:
    """
    Analyzes channel decomposition separating interest rate and market distortion effects
    """
    
    def __init__(self, config: Optional[MarketDistortionConfig] = None):
        """
        Initialize channel decomposition analyzer
        
        Args:
            config: Configuration for channel decomposition
        """
        self.config = config or MarketDistortionConfig()
        self.logger = logging.getLogger(__name__)
        
        # Storage for results
        self.results: Optional[ChannelDecompositionResults] = None
        
    def decompose_channels(self,
                          dependent_variable: pd.Series,
                          qe_intensity: pd.Series,
                          market_distortions: pd.Series,
                          interest_rates: pd.Series,
                          controls: Optional[pd.DataFrame] = None) -> ChannelDecompositionResults:
        """
        Decompose QE transmission into interest rate and market distortion channels
        
        Args:
            dependent_variable: Outcome variable (e.g., investment growth)
            qe_intensity: QE intensity measure
            market_distortions: Market distortion proxy (μ₂)
            interest_rates: Interest rate channel proxy
            controls: Additional control variables
            
        Returns:
            ChannelDecompositionResults object
        """
        self.logger.info(f"Decomposing channels using {self.config.decomposition_method} method")
        
        results = ChannelDecompositionResults()
        
        try:
            # Align all series to common dates
            common_dates = dependent_variable.index.intersection(qe_intensity.index)\
                                                  .intersection(market_distortions.index)\
                                                  .intersection(interest_rates.index)
            
            if len(common_dates) < 20:
                raise ValueError(f"Insufficient overlapping data: only {len(common_dates)} observations")
            
            # Align data
            y = dependent_variable.loc[common_dates]
            qe = qe_intensity.loc[common_dates]
            distortions = market_distortions.loc[common_dates]
            rates = interest_rates.loc[common_dates]
            
            # Handle controls
            if controls is not None:
                controls_aligned = controls.loc[common_dates]
            else:
                controls_aligned = None
            
            # Apply interaction decomposition (simplified for now)
            results = self._interaction_decomposition(y, qe, distortions, rates, controls_aligned)
            
            results.fitted = True
            self.logger.info("Channel decomposition completed successfully")
            
        except Exception as e:
            self.logger.error(f"Error in channel decomposition: {e}")
            results.fitted = False
            results = ChannelDecompositionResults()  # Return empty results
        
        self.results = results
        return results
    
    def _interaction_decomposition(self,
                                 y: pd.Series,
                                 qe: pd.Series,
                                 distortions: pd.Series,
                                 rates: pd.Series,
                                 controls: Optional[pd.DataFrame] = None) -> ChannelDecompositionResults:
        """
        Interaction-based channel decomposition
        
        Model: y = α + β₁*(QE × Interest_Rate) + β₂*(QE × Distortion) + γ*Controls + ε
        """
        results = ChannelDecompositionResults()
        
        # Create interaction terms
        qe_rate_interaction = qe * rates
        qe_distortion_interaction = qe * distortions
        
        # Prepare regression data
        X_data = {
            'qe_rate_interaction': qe_rate_interaction,
            'qe_distortion_interaction': qe_distortion_interaction
        }
        
        # Add controls if provided
        if controls is not None:
            for col in controls.columns:
                X_data[f'control_{col}'] = controls[col]
        
        # Create design matrix
        X_df = pd.DataFrame(X_data)
        X_reg = sm.add_constant(X_df)
        
        # Fit regression
        model = sm.OLS(y.values, X_reg).fit()
        
        # Extract results
        if len(model.params) >= 3:  # Constant + 2 main coefficients
            # Interest rate channel effect
            results.interest_rate_effect = model.params[1]
            results.interest_rate_se = model.bse[1]
            results.interest_rate_pvalue = model.pvalues[1]
            
            # Market distortion channel effect
            results.distortion_effect = model.params[2]
            results.distortion_se = model.bse[2]
            results.distortion_pvalue = model.pvalues[2]
            
            # Calculate confidence intervals
            alpha = 1 - self.config.confidence_level
            t_crit = stats.t.ppf(1 - alpha/2, model.df_resid)
            
            results.interest_rate_ci_lower = results.interest_rate_effect - t_crit * results.interest_rate_se
            results.interest_rate_ci_upper = results.interest_rate_effect + t_crit * results.interest_rate_se
            
            results.distortion_ci_lower = results.distortion_effect - t_crit * results.distortion_se
            results.distortion_ci_upper = results.distortion_effect + t_crit * results.distortion_se
            
            # Test for dominance of distortion effects
            dominance_results = self._test_dominance(
                results.distortion_effect, results.distortion_se,
                results.interest_rate_effect, results.interest_rate_se,
                model.df_resid
            )
            
            results.distortion_dominance = dominance_results['dominance']
            results.dominance_test_stat = dominance_results['test_stat']
            results.dominance_pvalue = dominance_results['pvalue']
            results.dominance_ci_lower = dominance_results['ci_lower']
            results.dominance_ci_upper = dominance_results['ci_upper']
            
            # Calculate effect shares
            total_abs_effect = abs(results.interest_rate_effect) + abs(results.distortion_effect)
            if total_abs_effect > 0:
                results.interest_rate_share = abs(results.interest_rate_effect) / total_abs_effect
                results.distortion_share = abs(results.distortion_effect) / total_abs_effect
            
            results.total_effect = results.interest_rate_effect + results.distortion_effect
            
            # Model diagnostics
            results.r_squared = model.rsquared
            results.adjusted_r_squared = model.rsquared_adj
            results.f_statistic = model.fvalue
            results.f_pvalue = model.f_pvalue
            results.observations = int(model.nobs)
        
        return results
    
    def _test_dominance(self,
                       distortion_effect: float,
                       distortion_se: float,
                       interest_rate_effect: float,
                       interest_rate_se: float,
                       df: int) -> Dict[str, Any]:
        """
        Test for dominance of distortion effects over interest rate effects
        
        H0: β_distortion <= β_interest_rate
        H1: β_distortion > β_interest_rate
        """
        # Calculate test statistic
        effect_diff = distortion_effect - interest_rate_effect
        se_diff = np.sqrt(distortion_se**2 + interest_rate_se**2)
        
        if se_diff > 0:
            if self.config.dominance_test_type == 'one_sided':
                # One-sided test for dominance
                test_stat = effect_diff / se_diff
                pvalue = 1 - stats.t.cdf(test_stat, df)
                dominance = pvalue < self.config.significance_level
            else:
                # Two-sided test for difference
                test_stat = abs(effect_diff) / se_diff
                pvalue = 2 * (1 - stats.t.cdf(test_stat, df))
                dominance = pvalue < self.config.significance_level and distortion_effect > interest_rate_effect
            
            # Confidence interval for difference
            alpha = 1 - self.config.confidence_level
            t_crit = stats.t.ppf(1 - alpha/2, df)
            ci_lower = effect_diff - t_crit * se_diff
            ci_upper = effect_diff + t_crit * se_diff
        else:
            test_stat = 0.0
            pvalue = 1.0
            dominance = False
            ci_lower = 0.0
            ci_upper = 0.0
        
        return {
            'dominance': dominance,
            'test_stat': test_stat,
            'pvalue': pvalue,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper
        }