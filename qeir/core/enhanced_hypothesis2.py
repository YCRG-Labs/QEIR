"""
Enhanced Hypothesis 2: QE Impact on Private Investment Analysis

This module implements comprehensive analysis of how intensive QE reduces long-term 
private investment when market distortions dominate interest rate effects.

Key Components:
1. QE Intensity Measurement from Fed balance sheet data
2. Private Investment Response Models using Local Projections
3. Market Distortion vs Interest Rate Channel Decomposition
4. Statistical Tests for Dominance of Distortion Effects
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
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import statsmodels.api as sm
from statsmodels.tsa.api import VAR
from statsmodels.regression.linear_model import OLS

from .models import LocalProjections, InstrumentalVariablesRegression
from ..utils.data_structures import HypothesisData


@dataclass
class QEIntensityConfig:
    """Configuration for QE intensity measurement"""
    
    # QE intensity calculation methods
    intensity_method: str = 'holdings_ratio'  # 'holdings_ratio', 'balance_sheet_growth', 'combined'
    normalization_method: str = 'outstanding_securities'  # 'outstanding_securities', 'gdp', 'monetary_base'
    
    # QE episode identification
    qe_threshold: float = 0.05  # Minimum balance sheet growth to qualify as QE
    min_episode_duration: int = 3  # Minimum months for QE episode
    
    # Intensive vs standard QE classification
    intensive_threshold: float = 0.15  # Threshold for intensive QE classification
    use_percentile_threshold: bool = True  # Use percentile-based threshold
    intensive_percentile: float = 75.0  # Percentile for intensive QE threshold


@dataclass
class InvestmentImpactConfig:
    """Configuration for investment impact analysis"""
    
    # Local projections settings
    max_horizon: int = 20  # Maximum horizon for impulse responses
    lags: int = 4  # Number of lags in local projections
    
    # Investment measures
    short_term_horizon: int = 4  # Quarters for short-term effects
    long_term_horizon: int = 12  # Quarters for long-term effects
    
    # Channel decomposition
    use_instrumental_variables: bool = True
    bootstrap_iterations: int = 1000
    confidence_level: float = 0.95


@dataclass
class ChannelDecompositionResults:
    """Results from channel decomposition analysis"""
    
    # Step 1: Channel effects (QE → channels)
    rate_channel_beta: float = 0.0  # βr: Effect of QE on interest rates
    rate_channel_se: float = 0.0
    rate_channel_pvalue: float = 1.0
    
    distortion_channel_beta: float = 0.0  # βD: Effect of QE on distortions
    distortion_channel_se: float = 0.0
    distortion_channel_pvalue: float = 1.0
    
    # Step 2: Investment responses (channels → investment)
    investment_rate_response: Dict[int, float] = None  # ρh by horizon
    investment_distortion_response: Dict[int, float] = None  # κh by horizon
    
    # Step 3: Decomposed effects
    rate_contributions: Dict[int, float] = None  # ψ^rate_h by horizon
    distortion_contributions: Dict[int, float] = None  # ψ^dist_h by horizon
    total_effects: Dict[int, float] = None  # ψh by horizon
    
    # Step 4: Channel shares
    distortion_share: float = 0.0  # Target: 0.65
    rate_share: float = 0.0  # Target: 0.35
    cumulative_effect_12q: float = 0.0  # Target: -2.7 pp
    
    # Validation
    shares_sum_to_one: bool = False  # distortion_share + rate_share ≈ 1
    distortion_dominates: bool = False  # distortion_share > rate_share
    meets_target_shares: bool = False  # Within ±5% of targets
    
    # Legacy fields (for backward compatibility)
    interest_rate_effect: float = 0.0
    interest_rate_se: float = 0.0
    interest_rate_pvalue: float = 1.0
    distortion_effect: float = 0.0
    distortion_se: float = 0.0
    distortion_pvalue: float = 1.0
    distortion_dominance: bool = False
    dominance_test_stat: float = 0.0
    dominance_pvalue: float = 1.0
    total_effect: float = 0.0
    interest_rate_share: float = 0.0
    
    # Model diagnostics
    r_squared: float = 0.0
    observations: int = 0
    fitted: bool = False
    
    def __post_init__(self):
        """Initialize mutable default values"""
        if self.investment_rate_response is None:
            self.investment_rate_response = {}
        if self.investment_distortion_response is None:
            self.investment_distortion_response = {}
        if self.rate_contributions is None:
            self.rate_contributions = {}
        if self.distortion_contributions is None:
            self.distortion_contributions = {}
        if self.total_effects is None:
            self.total_effects = {}


@dataclass
class InvestmentImpactResults:
    """Results from investment impact analysis"""
    
    # Short-term effects
    short_term_effect: float = 0.0
    short_term_se: float = 0.0
    short_term_pvalue: float = 1.0
    
    # Long-term effects
    long_term_effect: float = 0.0
    long_term_se: float = 0.0
    long_term_pvalue: float = 1.0
    
    # Cumulative effects (Σ(h=0 to H) ψ_h)
    cumulative_effect: float = 0.0
    cumulative_se: float = 0.0
    cumulative_lower_ci: float = 0.0
    cumulative_upper_ci: float = 0.0
    cumulative_horizons: int = 0
    
    # Impulse response function
    impulse_responses: pd.DataFrame = field(default_factory=pd.DataFrame)
    confidence_intervals: pd.DataFrame = field(default_factory=pd.DataFrame)
    
    # Channel decomposition
    channel_decomposition: ChannelDecompositionResults = field(default_factory=ChannelDecompositionResults)
    
    # Model diagnostics
    local_projections_results: Dict[str, Any] = field(default_factory=dict)
    iv_results: Optional[Dict[str, Any]] = None
    
    # QE intensity analysis
    qe_intensity_stats: Dict[str, Any] = field(default_factory=dict)
    
    # Metadata
    analysis_timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    config_used: Optional[InvestmentImpactConfig] = None


class QEIntensityMeasurer:
    """
    Measures QE intensity from Fed balance sheet data with multiple methodologies
    """
    
    def __init__(self, config: Optional[QEIntensityConfig] = None):
        """
        Initialize QE intensity measurer
        
        Args:
            config: Configuration for QE intensity measurement
        """
        self.config = config or QEIntensityConfig()
        self.logger = logging.getLogger(__name__)
        
        # Storage for calculated measures
        self.qe_intensity_series: Optional[pd.Series] = None
        self.qe_episodes: Optional[pd.DataFrame] = None
        self.intensive_episodes: Optional[pd.DataFrame] = None
        
    def calculate_qe_intensity(self, 
                              fed_holdings: pd.Series,
                              outstanding_securities: Optional[pd.Series] = None,
                              gdp: Optional[pd.Series] = None,
                              monetary_base: Optional[pd.Series] = None) -> pd.Series:
        """
        Calculate QE intensity using specified method
        
        Args:
            fed_holdings: Fed securities holdings
            outstanding_securities: Total outstanding securities for normalization
            gdp: GDP for normalization
            monetary_base: Monetary base for normalization
            
        Returns:
            QE intensity series
        """
        self.logger.info(f"Calculating QE intensity using {self.config.intensity_method} method")
        
        if self.config.intensity_method == 'holdings_ratio':
            if outstanding_securities is None:
                raise ValueError("Outstanding securities required for holdings_ratio method")
            
            # QE intensity = Fed holdings / Total outstanding securities
            qe_intensity = fed_holdings / outstanding_securities
            
        elif self.config.intensity_method == 'balance_sheet_growth':
            # QE intensity = YoY growth rate of Fed holdings
            qe_intensity = fed_holdings.pct_change(periods=12)
            
        elif self.config.intensity_method == 'combined':
            # Combined measure using both holdings ratio and growth
            if outstanding_securities is None:
                raise ValueError("Outstanding securities required for combined method")
            
            holdings_ratio = fed_holdings / outstanding_securities
            growth_rate = fed_holdings.pct_change(periods=12)
            
            # Standardize both measures and take average
            holdings_ratio_std = (holdings_ratio - holdings_ratio.mean()) / holdings_ratio.std()
            growth_rate_std = (growth_rate - growth_rate.mean()) / growth_rate.std()
            
            qe_intensity = (holdings_ratio_std + growth_rate_std) / 2
            
        else:
            raise ValueError(f"Unknown intensity method: {self.config.intensity_method}")
        
        # Apply normalization if specified
        if self.config.normalization_method == 'gdp' and gdp is not None:
            qe_intensity = qe_intensity / gdp * 100  # As percentage of GDP
        elif self.config.normalization_method == 'monetary_base' and monetary_base is not None:
            qe_intensity = qe_intensity / monetary_base
        
        # Clean the series
        qe_intensity = qe_intensity.dropna()
        
        self.qe_intensity_series = qe_intensity
        self.logger.info(f"Calculated QE intensity series with {len(qe_intensity)} observations")
        
        return qe_intensity
    
    def identify_qe_episodes(self, qe_intensity: Optional[pd.Series] = None) -> pd.DataFrame:
        """
        Identify QE episodes based on intensity thresholds
        
        Args:
            qe_intensity: QE intensity series (uses stored series if None)
            
        Returns:
            DataFrame with QE episode information
        """
        if qe_intensity is None:
            qe_intensity = self.qe_intensity_series
            
        if qe_intensity is None:
            raise ValueError("QE intensity series not available")
        
        self.logger.info("Identifying QE episodes")
        
        # Identify periods above threshold
        above_threshold = qe_intensity > self.config.qe_threshold
        
        # Find episode start and end dates
        episodes = []
        in_episode = False
        episode_start = None
        
        for date, is_above in above_threshold.items():
            if is_above and not in_episode:
                # Start of new episode
                episode_start = date
                in_episode = True
            elif not is_above and in_episode:
                # End of episode
                episode_end = date
                episode_duration = len(qe_intensity.loc[episode_start:episode_end])
                
                if episode_duration >= self.config.min_episode_duration:
                    episode_intensity = qe_intensity.loc[episode_start:episode_end]
                    episodes.append({
                        'start_date': episode_start,
                        'end_date': episode_end,
                        'duration_months': episode_duration,
                        'mean_intensity': episode_intensity.mean(),
                        'max_intensity': episode_intensity.max(),
                        'cumulative_intensity': episode_intensity.sum()
                    })
                
                in_episode = False
        
        # Handle case where series ends during an episode
        if in_episode and episode_start is not None:
            episode_end = qe_intensity.index[-1]
            episode_duration = len(qe_intensity.loc[episode_start:episode_end])
            
            if episode_duration >= self.config.min_episode_duration:
                episode_intensity = qe_intensity.loc[episode_start:episode_end]
                episodes.append({
                    'start_date': episode_start,
                    'end_date': episode_end,
                    'duration_months': episode_duration,
                    'mean_intensity': episode_intensity.mean(),
                    'max_intensity': episode_intensity.max(),
                    'cumulative_intensity': episode_intensity.sum()
                })
        
        episodes_df = pd.DataFrame(episodes)
        self.qe_episodes = episodes_df
        
        self.logger.info(f"Identified {len(episodes_df)} QE episodes")
        
        return episodes_df
    
    def classify_intensive_episodes(self, episodes: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Classify QE episodes as intensive vs standard
        
        Args:
            episodes: QE episodes DataFrame (uses stored episodes if None)
            
        Returns:
            DataFrame with intensive episode classification
        """
        if episodes is None:
            episodes = self.qe_episodes
            
        if episodes is None or episodes.empty:
            raise ValueError("QE episodes not available")
        
        self.logger.info("Classifying intensive QE episodes")
        
        episodes_classified = episodes.copy()
        
        if self.config.use_percentile_threshold:
            # Use percentile-based threshold
            intensity_threshold = np.percentile(episodes['mean_intensity'], self.config.intensive_percentile)
        else:
            # Use fixed threshold
            intensity_threshold = self.config.intensive_threshold
        
        # Classify episodes
        episodes_classified['is_intensive'] = episodes_classified['mean_intensity'] > intensity_threshold
        episodes_classified['intensity_threshold'] = intensity_threshold
        
        # Calculate additional metrics for intensive episodes
        intensive_episodes = episodes_classified[episodes_classified['is_intensive']].copy()
        
        if not intensive_episodes.empty:
            intensive_episodes['intensity_percentile'] = intensive_episodes['mean_intensity'].rank(pct=True) * 100
            
            # Calculate relative intensity (compared to threshold)
            intensive_episodes['relative_intensity'] = (
                intensive_episodes['mean_intensity'] / intensity_threshold
            )
        
        self.intensive_episodes = intensive_episodes
        
        intensive_count = len(intensive_episodes)
        total_count = len(episodes_classified)
        
        self.logger.info(f"Classified {intensive_count}/{total_count} episodes as intensive QE")
        
        return episodes_classified
    
    def get_qe_intensity_statistics(self) -> Dict[str, Any]:
        """
        Get comprehensive statistics on QE intensity measures
        
        Returns:
            Dictionary with QE intensity statistics
        """
        if self.qe_intensity_series is None:
            return {'error': 'QE intensity series not calculated'}
        
        stats_dict = {
            'series_statistics': {
                'mean': self.qe_intensity_series.mean(),
                'std': self.qe_intensity_series.std(),
                'min': self.qe_intensity_series.min(),
                'max': self.qe_intensity_series.max(),
                'median': self.qe_intensity_series.median(),
                'skewness': self.qe_intensity_series.skew(),
                'kurtosis': self.qe_intensity_series.kurtosis(),
                'observations': len(self.qe_intensity_series)
            },
            'episode_statistics': {},
            'intensive_statistics': {}
        }
        
        if self.qe_episodes is not None and not self.qe_episodes.empty:
            stats_dict['episode_statistics'] = {
                'total_episodes': len(self.qe_episodes),
                'mean_duration': self.qe_episodes['duration_months'].mean(),
                'mean_intensity': self.qe_episodes['mean_intensity'].mean(),
                'total_qe_months': self.qe_episodes['duration_months'].sum(),
                'max_episode_intensity': self.qe_episodes['max_intensity'].max()
            }
        
        if self.intensive_episodes is not None and not self.intensive_episodes.empty:
            stats_dict['intensive_statistics'] = {
                'intensive_episodes': len(self.intensive_episodes),
                'intensive_share': len(self.intensive_episodes) / len(self.qe_episodes) if self.qe_episodes is not None else 0,
                'mean_intensive_duration': self.intensive_episodes['duration_months'].mean(),
                'mean_intensive_intensity': self.intensive_episodes['mean_intensity'].mean(),
                'max_intensive_intensity': self.intensive_episodes['max_intensity'].max()
            }
        
        return stats_dict


class InvestmentImpactAnalyzer:
    """
    Analyzes private investment response to QE using Local Projections and channel decomposition
    """
    
    def __init__(self, config: Optional[InvestmentImpactConfig] = None):
        """
        Initialize investment impact analyzer
        
        Args:
            config: Configuration for investment impact analysis
        """
        self.config = config or InvestmentImpactConfig()
        self.logger = logging.getLogger(__name__)
        
        # Initialize models
        self.local_projections = LocalProjections(max_horizon=self.config.max_horizon)
        self.iv_model = InstrumentalVariablesRegression() if self.config.use_instrumental_variables else None
        
        # Storage for results
        self.results: Optional[InvestmentImpactResults] = None
        
    def analyze_investment_response(self,
                                  private_investment: pd.Series,
                                  qe_intensity: pd.Series,
                                  market_distortions: pd.Series,
                                  interest_rates: pd.Series,
                                  controls: Optional[pd.DataFrame] = None) -> InvestmentImpactResults:
        """
        Analyze private investment response to QE intensity
        
        Args:
            private_investment: Private investment series
            qe_intensity: QE intensity measure
            market_distortions: Market distortion proxy (μ₂)
            interest_rates: Interest rate channel proxy
            controls: Additional control variables
            
        Returns:
            InvestmentImpactResults object
        """
        self.logger.info("Analyzing private investment response to QE")
        
        # Initialize results
        results = InvestmentImpactResults(config_used=self.config)
        
        # Align all series to common dates
        common_dates = private_investment.index.intersection(qe_intensity.index)\
                                              .intersection(market_distortions.index)\
                                              .intersection(interest_rates.index)
        
        if len(common_dates) < 20:
            raise ValueError(f"Insufficient overlapping data: only {len(common_dates)} observations")
        
        # Align data
        investment_aligned = private_investment.loc[common_dates]
        qe_aligned = qe_intensity.loc[common_dates]
        distortions_aligned = market_distortions.loc[common_dates]
        rates_aligned = interest_rates.loc[common_dates]
        
        # Validate and prepare control variables (Requirement 6.4)
        if controls is not None:
            # Validate controls type
            if not isinstance(controls, (pd.DataFrame, pd.Series)):
                raise TypeError("controls must be a pandas DataFrame or Series")
            
            # Check for missing values in controls
            if isinstance(controls, pd.DataFrame):
                if controls.isnull().any().any():
                    n_missing = controls.isnull().sum().sum()
                    raise ValueError(f"Controls contain {n_missing} missing values. "
                                   "Please handle missing values before analysis.")
            else:
                if controls.isnull().any():
                    raise ValueError("Controls contain missing values. "
                                   "Please handle missing values before analysis.")
            
            # Align controls to common dates
            controls_aligned = controls.loc[common_dates]
            
            # Verify alignment worked correctly
            if len(controls_aligned) != len(common_dates):
                raise ValueError(f"Controls alignment failed: expected {len(common_dates)} "
                               f"observations but got {len(controls_aligned)}")
            
            # Combine with built-in controls
            control_df = pd.DataFrame({
                'market_distortions': distortions_aligned,
                'interest_rates': rates_aligned
            })
            control_df = pd.concat([control_df, controls_aligned], axis=1)
        else:
            control_df = pd.DataFrame({
                'market_distortions': distortions_aligned,
                'interest_rates': rates_aligned
            })
        
        # Final validation: ensure control_df has no missing values
        if control_df.isnull().any().any():
            n_missing = control_df.isnull().sum().sum()
            raise ValueError(f"Control DataFrame contains {n_missing} missing values after alignment. "
                           "This may indicate data quality issues.")
        
        try:
            # Fit Local Projections model
            self.logger.info("Fitting Local Projections model")
            self.local_projections.fit(
                y=investment_aligned,
                shock=qe_aligned,
                controls=control_df,
                lags=self.config.lags
            )
            
            # Extract impulse responses
            if self.local_projections.fitted and self.local_projections.results:
                horizons = list(range(self.config.max_horizon + 1))
                impulse_responses = []
                confidence_lower = []
                confidence_upper = []
                
                for h in horizons:
                    if h in self.local_projections.results:
                        result = self.local_projections.results[h]
                        if hasattr(result, 'params') and len(result.params) > 0:
                            # QE shock coefficient (first coefficient after constant)
                            coef = result.params[1] if len(result.params) > 1 else result.params[0]
                            se = result.bse[1] if len(result.bse) > 1 else result.bse[0]
                            
                            # Calculate confidence intervals
                            alpha = 1 - self.config.confidence_level
                            t_crit = stats.t.ppf(1 - alpha/2, result.df_resid)
                            
                            impulse_responses.append(coef)
                            confidence_lower.append(coef - t_crit * se)
                            confidence_upper.append(coef + t_crit * se)
                        else:
                            impulse_responses.append(0.0)
                            confidence_lower.append(0.0)
                            confidence_upper.append(0.0)
                    else:
                        impulse_responses.append(0.0)
                        confidence_lower.append(0.0)
                        confidence_upper.append(0.0)
                
                # Store impulse response results
                results.impulse_responses = pd.DataFrame({
                    'horizon': horizons,
                    'response': impulse_responses
                })
                
                results.confidence_intervals = pd.DataFrame({
                    'horizon': horizons,
                    'lower': confidence_lower,
                    'upper': confidence_upper
                })
                
                # Calculate short-term and long-term effects
                if len(impulse_responses) > self.config.short_term_horizon:
                    results.short_term_effect = np.mean(impulse_responses[1:self.config.short_term_horizon+1])
                    results.short_term_se = np.std(impulse_responses[1:self.config.short_term_horizon+1]) / np.sqrt(self.config.short_term_horizon)
                
                if len(impulse_responses) > self.config.long_term_horizon:
                    results.long_term_effect = np.mean(impulse_responses[self.config.long_term_horizon:])
                    results.long_term_se = np.std(impulse_responses[self.config.long_term_horizon:]) / np.sqrt(len(impulse_responses) - self.config.long_term_horizon)
                
                # Calculate cumulative effect using LocalProjections method
                try:
                    cumulative_results = self.local_projections.compute_cumulative_effect(
                        shock_idx=1,  # QE shock is first variable after constant
                        max_horizon=self.config.long_term_horizon  # Use long_term_horizon (default 12 quarters)
                    )
                    
                    results.cumulative_effect = cumulative_results['cumulative_effect']
                    results.cumulative_se = cumulative_results['cumulative_se']
                    results.cumulative_lower_ci = cumulative_results['cumulative_lower_ci']
                    results.cumulative_upper_ci = cumulative_results['cumulative_upper_ci']
                    results.cumulative_horizons = cumulative_results['horizons_included']
                    
                    self.logger.info(f"Cumulative effect over {results.cumulative_horizons} quarters: "
                                   f"{results.cumulative_effect:.4f} [{results.cumulative_lower_ci:.4f}, "
                                   f"{results.cumulative_upper_ci:.4f}]")
                except Exception as e:
                    self.logger.warning(f"Could not compute cumulative effect: {e}")
                    results.cumulative_effect = 0.0
                    results.cumulative_se = 0.0
                
                # Store local projections results
                results.local_projections_results = {
                    'fitted': self.local_projections.fitted,
                    'max_horizon': self.config.max_horizon,
                    'lags': self.config.lags,
                    'observations': len(common_dates)
                }
                
            else:
                self.logger.warning("Local Projections model failed to fit properly")
                
        except Exception as e:
            self.logger.error(f"Error in Local Projections analysis: {e}")
            results.local_projections_results = {'error': str(e), 'fitted': False}
        
        # Instrumental Variables analysis if enabled
        if self.config.use_instrumental_variables and self.iv_model is not None:
            try:
                self.logger.info("Running Instrumental Variables analysis")
                
                # Prepare IV regression
                # Dependent variable: investment growth
                y_iv = investment_aligned.pct_change().dropna()
                
                # Endogenous variable: QE intensity
                # Exogenous variables: market distortions
                X_iv = np.column_stack([
                    qe_aligned.loc[y_iv.index].values,
                    distortions_aligned.loc[y_iv.index].values
                ])
                
                # Instruments: lagged interest rates and lagged QE intensity
                Z_iv = np.column_stack([
                    rates_aligned.shift(1).loc[y_iv.index].fillna(0).values,
                    qe_aligned.shift(1).loc[y_iv.index].fillna(0).values,
                    distortions_aligned.loc[y_iv.index].values  # Exogenous variables also go in Z
                ])
                
                # Fit IV model
                self.iv_model.fit(
                    y=y_iv.values,
                    X=X_iv,
                    Z=Z_iv,
                    endogenous_idx=[0]  # QE intensity is endogenous
                )
                
                if self.iv_model.fitted:
                    results.iv_results = {
                        'fitted': True,
                        'first_stage_f_stats': [fs.fvalue for fs in self.iv_model.first_stage_results],
                        'second_stage_params': self.iv_model.second_stage_results.params.tolist(),
                        'second_stage_pvalues': self.iv_model.second_stage_results.pvalues.tolist(),
                        'second_stage_rsquared': self.iv_model.second_stage_results.rsquared
                    }
                else:
                    results.iv_results = {'fitted': False, 'error': 'IV model failed to fit'}
                    
            except Exception as e:
                self.logger.error(f"Error in IV analysis: {e}")
                results.iv_results = {'error': str(e), 'fitted': False}
        
        # Channel decomposition analysis
        try:
            channel_results = self._decompose_channels(
                investment_aligned, qe_aligned, distortions_aligned, rates_aligned
            )
            results.channel_decomposition = channel_results
        except Exception as e:
            self.logger.error(f"Error in channel decomposition: {e}")
            results.channel_decomposition = ChannelDecompositionResults()
        
        # Store QE intensity statistics
        qe_measurer = QEIntensityMeasurer()
        qe_measurer.qe_intensity_series = qe_aligned
        results.qe_intensity_stats = qe_measurer.get_qe_intensity_statistics()
        
        self.results = results
        self.logger.info("Investment impact analysis completed")
        
        return results
    
    def _decompose_channels(self,
                           investment: pd.Series,
                           qe_intensity: pd.Series,
                           market_distortions: pd.Series,
                           interest_rates: pd.Series) -> ChannelDecompositionResults:
        """
        Decompose QE effects into interest rate and market distortion channels.
        
        This method now uses the StructuralChannelDecomposer for rigorous
        two-step structural decomposition following Requirements 7.1-7.7.
        
        Args:
            investment: Private investment series
            qe_intensity: QE intensity measure (used as QE shocks)
            market_distortions: Market distortion proxy
            interest_rates: Interest rate proxy
            
        Returns:
            ChannelDecompositionResults object
            
        Requirements: 7.7
        """
        self.logger.info("Decomposing transmission channels using structural decomposition")
        
        results = ChannelDecompositionResults()
        
        try:
            # Import the new decomposer
            from qeir.analysis.channel_decomposition import (
                StructuralChannelDecomposer,
                ChannelDecompositionConfig
            )
            
            # Prepare investment growth
            investment_growth = investment.pct_change().dropna()
            
            # Align all series
            common_idx = investment_growth.index.intersection(qe_intensity.index)\
                                                .intersection(market_distortions.index)\
                                                .intersection(interest_rates.index)
            
            if len(common_idx) < 10:
                raise ValueError("Insufficient data for channel decomposition")
            
            investment_growth_aligned = investment_growth.loc[common_idx]
            qe_aligned = qe_intensity.loc[common_idx]
            distortions_aligned = market_distortions.loc[common_idx]
            rates_aligned = interest_rates.loc[common_idx]
            
            # Create decomposer with configuration
            config = ChannelDecompositionConfig(
                max_horizon=self.config.max_horizon,
                use_hac_errors=True,
                hac_lags=4
            )
            decomposer = StructuralChannelDecomposer(config=config)
            
            # Run full structural decomposition
            decomp_results = decomposer.run_full_decomposition(
                qe_shocks=qe_aligned,
                interest_rates=rates_aligned,
                distortion_index=distortions_aligned,
                investment_growth=investment_growth_aligned,
                controls=None
            )
            
            # Extract results and populate ChannelDecompositionResults
            channel_effects = decomp_results['channel_effects']
            investment_responses = decomp_results['investment_responses']
            decomposition = decomp_results['decomposition']
            channel_shares = decomp_results['channel_shares']
            
            # Step 1: Channel effects
            results.rate_channel_beta = channel_effects['rate_channel']['beta']
            results.rate_channel_se = channel_effects['rate_channel']['se']
            results.rate_channel_pvalue = channel_effects['rate_channel']['pvalue']
            
            results.distortion_channel_beta = channel_effects['distortion_channel']['beta']
            results.distortion_channel_se = channel_effects['distortion_channel']['se']
            results.distortion_channel_pvalue = channel_effects['distortion_channel']['pvalue']
            
            # Step 2: Investment responses
            results.investment_rate_response = {
                h: resp['rho'] for h, resp in investment_responses.items()
            }
            results.investment_distortion_response = {
                h: resp['kappa'] for h, resp in investment_responses.items()
            }
            
            # Step 3: Decomposed effects
            results.rate_contributions = decomposition['rate_contributions']
            results.distortion_contributions = decomposition['distortion_contributions']
            results.total_effects = decomposition['total_effects']
            
            # Step 4: Channel shares
            results.distortion_share = channel_shares['distortion_share']
            results.rate_share = channel_shares['rate_share']
            results.cumulative_effect_12q = channel_shares['cumulative_total_effect']
            
            # Validation flags
            results.shares_sum_to_one = channel_shares['shares_sum_to_one']
            results.distortion_dominates = results.distortion_share > results.rate_share
            results.meets_target_shares = channel_shares['meets_target_shares']
            
            # Populate legacy fields for backward compatibility
            results.interest_rate_effect = results.rate_channel_beta
            results.interest_rate_se = results.rate_channel_se
            results.interest_rate_pvalue = results.rate_channel_pvalue
            results.distortion_effect = results.distortion_channel_beta
            results.distortion_se = results.distortion_channel_se
            results.distortion_pvalue = results.distortion_channel_pvalue
            results.interest_rate_share = results.rate_share
            results.total_effect = channel_shares['cumulative_total_effect']
            
            # Dominance test
            results.distortion_dominance = results.distortion_dominates
            
            # Model diagnostics
            results.observations = len(common_idx)
            results.fitted = True
            
            self.logger.info(f"Structural decomposition completed: "
                           f"Distortion={results.distortion_share:.2%}, "
                           f"Rate={results.rate_share:.2%}")
            
        except ImportError as e:
            self.logger.warning(f"Could not import StructuralChannelDecomposer, falling back to legacy method: {e}")
            # Fall back to legacy implementation
            results = self._decompose_channels_legacy(investment, qe_intensity, market_distortions, interest_rates)
            
        except Exception as e:
            self.logger.error(f"Error in structural channel decomposition: {e}")
            self.logger.info("Attempting legacy channel decomposition")
            try:
                results = self._decompose_channels_legacy(investment, qe_intensity, market_distortions, interest_rates)
            except Exception as e2:
                self.logger.error(f"Legacy decomposition also failed: {e2}")
                results.fitted = False
        
        return results
    
    def _decompose_channels_legacy(self,
                                   investment: pd.Series,
                                   qe_intensity: pd.Series,
                                   market_distortions: pd.Series,
                                   interest_rates: pd.Series) -> ChannelDecompositionResults:
        """
        Legacy channel decomposition method (for backward compatibility).
        
        Args:
            investment: Private investment series
            qe_intensity: QE intensity measure
            market_distortions: Market distortion proxy
            interest_rates: Interest rate proxy
            
        Returns:
            ChannelDecompositionResults object
        """
        self.logger.info("Using legacy channel decomposition method")
        
        results = ChannelDecompositionResults()
        
        try:
            # Prepare data for regression
            y = investment.pct_change().dropna()
            
            # Align all series
            common_idx = y.index.intersection(qe_intensity.index)\
                                .intersection(market_distortions.index)\
                                .intersection(interest_rates.index)
            
            if len(common_idx) < 10:
                raise ValueError("Insufficient data for channel decomposition")
            
            y_aligned = y.loc[common_idx]
            qe_aligned = qe_intensity.loc[common_idx]
            distortions_aligned = market_distortions.loc[common_idx]
            rates_aligned = interest_rates.loc[common_idx]
            
            # Channel decomposition regression:
            # Investment_growth = α + β₁*QE*Interest_Rate_Channel + β₂*QE*Distortion_Channel + ε
            
            # Create interaction terms
            qe_rate_interaction = qe_aligned * rates_aligned
            qe_distortion_interaction = qe_aligned * distortions_aligned
            
            # Standardize variables for better interpretation
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(np.column_stack([
                qe_rate_interaction.values,
                qe_distortion_interaction.values
            ]))
            
            # Fit regression
            X_reg = sm.add_constant(X_scaled)
            model = sm.OLS(y_aligned.values, X_reg).fit()
            
            if len(model.params) >= 3:  # Constant + 2 coefficients
                # Interest rate channel effect
                results.interest_rate_effect = model.params[1]
                results.interest_rate_se = model.bse[1]
                results.interest_rate_pvalue = model.pvalues[1]
                
                # Market distortion channel effect
                results.distortion_effect = model.params[2]
                results.distortion_se = model.bse[2]
                results.distortion_pvalue = model.pvalues[2]
                
                # Test for dominance of distortion effects
                # H0: β₂ <= β₁ vs H1: β₂ > β₁
                dominance_test_stat = (results.distortion_effect - results.interest_rate_effect) / \
                                    np.sqrt(results.distortion_se**2 + results.interest_rate_se**2)
                
                results.dominance_test_stat = dominance_test_stat
                results.dominance_pvalue = 1 - stats.norm.cdf(dominance_test_stat)
                results.distortion_dominance = results.dominance_pvalue < 0.05
                
                # Calculate effect shares
                total_abs_effect = abs(results.interest_rate_effect) + abs(results.distortion_effect)
                if total_abs_effect > 0:
                    results.interest_rate_share = abs(results.interest_rate_effect) / total_abs_effect
                    results.distortion_share = abs(results.distortion_effect) / total_abs_effect
                
                results.total_effect = results.interest_rate_effect + results.distortion_effect
                results.r_squared = model.rsquared
                results.observations = len(y_aligned)
                results.fitted = True
                
            else:
                self.logger.warning("Insufficient parameters in channel decomposition model")
                
        except Exception as e:
            self.logger.error(f"Error in legacy channel decomposition: {e}")
            results.fitted = False
        
        return results
    
    def test_distortion_dominance(self, 
                                 investment_results: InvestmentImpactResults,
                                 significance_level: float = 0.05) -> Dict[str, Any]:
        """
        Test statistical significance of distortion effect dominance
        
        Args:
            investment_results: Results from investment impact analysis
            significance_level: Significance level for tests
            
        Returns:
            Dictionary with dominance test results
        """
        self.logger.info("Testing distortion effect dominance")
        
        channel_results = investment_results.channel_decomposition
        
        if not channel_results.fitted:
            return {
                'error': 'Channel decomposition not fitted',
                'dominance_detected': False
            }
        
        # Multiple tests for robustness
        tests = {}
        
        # Test 1: Direct coefficient comparison
        tests['coefficient_comparison'] = {
            'distortion_effect': channel_results.distortion_effect,
            'interest_rate_effect': channel_results.interest_rate_effect,
            'distortion_larger': abs(channel_results.distortion_effect) > abs(channel_results.interest_rate_effect),
            'distortion_significant': channel_results.distortion_pvalue < significance_level,
            'interest_rate_significant': channel_results.interest_rate_pvalue < significance_level
        }
        
        # Test 2: Statistical dominance test
        tests['statistical_dominance'] = {
            'test_statistic': channel_results.dominance_test_stat,
            'p_value': channel_results.dominance_pvalue,
            'significant_dominance': channel_results.dominance_pvalue < significance_level,
            'dominance_detected': channel_results.distortion_dominance
        }
        
        # Test 3: Effect share analysis
        tests['effect_shares'] = {
            'distortion_share': channel_results.distortion_share,
            'interest_rate_share': channel_results.interest_rate_share,
            'distortion_majority': channel_results.distortion_share > 0.5
        }
        
        # Overall assessment
        dominance_criteria_met = sum([
            tests['coefficient_comparison']['distortion_larger'],
            tests['statistical_dominance']['significant_dominance'],
            tests['effect_shares']['distortion_majority']
        ])
        
        overall_result = {
            'dominance_criteria_met': dominance_criteria_met,
            'total_criteria': 3,
            'strong_dominance': dominance_criteria_met >= 2,
            'weak_dominance': dominance_criteria_met >= 1,
            'no_dominance': dominance_criteria_met == 0
        }
        
        return {
            'tests': tests,
            'overall_assessment': overall_result,
            'significance_level': significance_level,
            'model_r_squared': channel_results.r_squared,
            'observations': channel_results.observations
        }


class EnhancedHypothesis2Tester:
    """
    Main class for comprehensive Hypothesis 2 testing: QE Impact on Private Investment Analysis
    """
    
    def __init__(self, 
                 qe_config: Optional[QEIntensityConfig] = None,
                 investment_config: Optional[InvestmentImpactConfig] = None):
        """
        Initialize enhanced Hypothesis 2 tester
        
        Args:
            qe_config: Configuration for QE intensity measurement
            investment_config: Configuration for investment impact analysis
        """
        self.qe_config = qe_config or QEIntensityConfig()
        self.investment_config = investment_config or InvestmentImpactConfig()
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self.qe_measurer = QEIntensityMeasurer(self.qe_config)
        self.investment_analyzer = InvestmentImpactAnalyzer(self.investment_config)
        
        # Storage for results
        self.results: Optional[InvestmentImpactResults] = None
        
    def test_hypothesis2(self, data: HypothesisData) -> InvestmentImpactResults:
        """
        Complete Hypothesis 2 testing pipeline
        
        Args:
            data: HypothesisData object with required series
            
        Returns:
            InvestmentImpactResults object
        """
        self.logger.info("Starting comprehensive Hypothesis 2 testing")
        
        # Extract required data
        try:
            # QE intensity calculation
            if data.qe_intensity is not None:
                qe_intensity = data.qe_intensity
            else:
                # Calculate from Fed holdings if available
                if hasattr(data, 'fed_securities_held') and data.fed_securities_held is not None:
                    outstanding = getattr(data, 'treasury_outstanding', None)
                    qe_intensity = self.qe_measurer.calculate_qe_intensity(
                        fed_holdings=data.fed_securities_held,
                        outstanding_securities=outstanding
                    )
                else:
                    raise ValueError("QE intensity data not available")
            
            # Private investment
            if data.private_investment is None:
                raise ValueError("Private investment data not available")
            
            # Market distortions
            if data.market_distortions is None:
                raise ValueError("Market distortions data not available")
            
            # Interest rate channel
            if data.interest_rate_channel is None:
                raise ValueError("Interest rate channel data not available")
            
            # Run investment impact analysis
            results = self.investment_analyzer.analyze_investment_response(
                private_investment=data.private_investment,
                qe_intensity=qe_intensity,
                market_distortions=data.market_distortions,
                interest_rates=data.interest_rate_channel
            )
            
            # Test distortion dominance
            dominance_results = self.investment_analyzer.test_distortion_dominance(results)
            
            # Add dominance test results to main results
            if 'dominance_test_results' not in results.local_projections_results:
                results.local_projections_results['dominance_test_results'] = dominance_results
            
            self.results = results
            self.logger.info("Hypothesis 2 testing completed successfully")
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error in Hypothesis 2 testing: {e}")
            # Return error results
            error_results = InvestmentImpactResults(
                config_used=self.investment_config
            )
            error_results.local_projections_results = {'error': str(e), 'fitted': False}
            return error_results