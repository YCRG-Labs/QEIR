"""
Investment Impact Visualization and Analysis

This module implements comprehensive visualization and analysis for Hypothesis 2:
QE Impact on Private Investment Analysis. It creates decomposition charts showing
interest rate vs distortion contributions, adds confidence intervals and statistical
significance testing, and provides robustness tests across different QE episode definitions.

Key Components:
1. Decomposition Charts showing interest rate vs distortion contributions
2. Confidence Intervals and Statistical Significance Testing
3. Robustness Tests across different QE episode definitions
4. Publication-ready visualization outputs

Author: QE Research Team
Date: 2025
Version: 1.0
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from datetime import datetime
import warnings
from pathlib import Path
warnings.filterwarnings('ignore')

from scipy import stats
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm

from .publication_visualization import PublicationVisualizationSuite
from ..core.enhanced_hypothesis2 import InvestmentImpactResults, ChannelDecompositionResults
from ..core.market_distortion_analyzer import ChannelDecompositionResults as MarketChannelResults


@dataclass
class InvestmentVisualizationConfig:
    """Configuration for investment impact visualization"""
    
    # Figure settings
    figure_style: str = 'economics_journal'  # Publication style
    save_formats: List[str] = field(default_factory=lambda: ['png', 'pdf'])
    dpi: int = 300
    
    # Decomposition chart settings
    show_confidence_intervals: bool = True
    confidence_level: float = 0.95
    bar_alpha: float = 0.7
    
    # Robustness testing
    robustness_episodes: List[str] = field(default_factory=lambda: [
        'all_episodes', 'intensive_only', 'crisis_periods', 'post_crisis'
    ])
    
    # Statistical testing
    significance_level: float = 0.05
    bootstrap_iterations: int = 1000
    
    # Color scheme
    interest_rate_color: str = '#1f77b4'  # Blue
    distortion_color: str = '#ff7f0e'     # Orange
    total_effect_color: str = '#2ca02c'   # Green
    significance_color: str = '#d62728'   # Red


@dataclass
class RobustnessTestResults:
    """Results from robustness testing across different QE episode definitions"""
    
    # Episode definitions tested
    episode_definitions: List[str] = field(default_factory=list)
    
    # Results for each definition
    interest_rate_effects: List[float] = field(default_factory=list)
    distortion_effects: List[float] = field(default_factory=list)
    dominance_results: List[bool] = field(default_factory=list)
    
    # Statistical significance
    interest_rate_pvalues: List[float] = field(default_factory=list)
    distortion_pvalues: List[float] = field(default_factory=list)
    dominance_pvalues: List[float] = field(default_factory=list)
    
    # Confidence intervals
    interest_rate_ci_lower: List[float] = field(default_factory=list)
    interest_rate_ci_upper: List[float] = field(default_factory=list)
    distortion_ci_lower: List[float] = field(default_factory=list)
    distortion_ci_upper: List[float] = field(default_factory=list)
    
    # Model fit statistics
    r_squared_values: List[float] = field(default_factory=list)
    observations: List[int] = field(default_factory=list)
    
    # Summary statistics
    mean_interest_rate_effect: float = 0.0
    mean_distortion_effect: float = 0.0
    consistency_score: float = 0.0  # Measure of result consistency across definitions
    
    # Metadata
    analysis_timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


class InvestmentImpactVisualizer:
    """
    Creates comprehensive visualizations for investment impact analysis
    """
    
    def __init__(self, config: Optional[InvestmentVisualizationConfig] = None):
        """
        Initialize investment impact visualizer
        
        Args:
            config: Configuration for visualization settings
        """
        self.config = config or InvestmentVisualizationConfig()
        self.logger = logging.getLogger(__name__)
        
        # Initialize publication visualization suite
        self.pub_viz = PublicationVisualizationSuite(style=self.config.figure_style)
        
        # Storage for results
        self.robustness_results: Optional[RobustnessTestResults] = None
        
    def create_channel_decomposition_chart(self,
                                         channel_results: Union[ChannelDecompositionResults, MarketChannelResults],
                                         title: str = "QE Transmission Channel Decomposition",
                                         save_path: Optional[str] = None) -> plt.Figure:
        """
        Create decomposition chart showing interest rate vs distortion contributions
        
        Args:
            channel_results: Channel decomposition results
            title: Chart title
            save_path: Path to save the figure
            
        Returns:
            matplotlib Figure object
        """
        self.logger.info("Creating channel decomposition chart")
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Extract effects and confidence intervals
        interest_rate_effect = channel_results.interest_rate_effect
        distortion_effect = channel_results.distortion_effect
        
        # Calculate confidence intervals if not available
        if hasattr(channel_results, 'interest_rate_ci_lower') and hasattr(channel_results, 'interest_rate_ci_upper'):
            interest_rate_ci = [channel_results.interest_rate_ci_lower, channel_results.interest_rate_ci_upper]
        else:
            # Calculate from standard errors if available
            alpha = 1 - self.config.confidence_level
            t_crit = 1.96  # Approximate for large samples
            ir_margin = t_crit * channel_results.interest_rate_se
            interest_rate_ci = [interest_rate_effect - ir_margin, interest_rate_effect + ir_margin]
        
        if hasattr(channel_results, 'distortion_ci_lower') and hasattr(channel_results, 'distortion_ci_upper'):
            distortion_ci = [channel_results.distortion_ci_lower, channel_results.distortion_ci_upper]
        else:
            # Calculate from standard errors if available
            alpha = 1 - self.config.confidence_level
            t_crit = 1.96  # Approximate for large samples
            dist_margin = t_crit * channel_results.distortion_se
            distortion_ci = [distortion_effect - dist_margin, distortion_effect + dist_margin]
        
        # Left panel: Channel effects comparison
        channels = ['Interest Rate\nChannel', 'Market Distortion\nChannel']
        effects = [interest_rate_effect, distortion_effect]
        colors = [self.config.interest_rate_color, self.config.distortion_color]
        
        bars = ax1.bar(channels, effects, color=colors, alpha=self.config.bar_alpha,
                      edgecolor='black', linewidth=0.8)
        
        # Add confidence intervals if requested
        if self.config.show_confidence_intervals:
            ci_lower = [interest_rate_ci[0], distortion_ci[0]]
            ci_upper = [interest_rate_ci[1], distortion_ci[1]]
            
            ax1.errorbar(range(len(channels)), effects,
                        yerr=[np.array(effects) - np.array(ci_lower),
                              np.array(ci_upper) - np.array(effects)],
                        fmt='none', color='black', capsize=4, capthick=1.5)
        
        # Add significance indicators
        if channel_results.interest_rate_pvalue < self.config.significance_level:
            ax1.text(0, interest_rate_effect + 0.01, '***' if channel_results.interest_rate_pvalue < 0.01 else '*',
                    ha='center', va='bottom', fontsize=12, color=self.config.significance_color)
        
        if channel_results.distortion_pvalue < self.config.significance_level:
            ax1.text(1, distortion_effect + 0.01, '***' if channel_results.distortion_pvalue < 0.01 else '*',
                    ha='center', va='bottom', fontsize=12, color=self.config.significance_color)
        
        # Add value labels on bars
        for i, (bar, effect) in enumerate(zip(bars, effects)):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                    f'{effect:.3f}', ha='center', va='bottom', fontweight='bold')
        
        ax1.set_ylabel('Effect Size')
        ax1.set_title('Channel Effects Comparison')
        ax1.grid(True, alpha=0.3)
        ax1.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        
        # Right panel: Effect shares pie chart
        if hasattr(channel_results, 'interest_rate_share') and hasattr(channel_results, 'distortion_share'):
            shares = [channel_results.interest_rate_share * 100, channel_results.distortion_share * 100]
            labels = [f'Interest Rate\n({shares[0]:.1f}%)', f'Market Distortion\n({shares[1]:.1f}%)']
            
            wedges, texts, autotexts = ax2.pie(shares, labels=labels, colors=colors,
                                              autopct='', startangle=90,
                                              textprops={'fontsize': 10})
            
            # Highlight dominant channel
            if channel_results.distortion_dominance:
                wedges[1].set_edgecolor(self.config.significance_color)
                wedges[1].set_linewidth(3)
                ax2.text(0, -1.3, f'Distortion Dominance: p = {channel_results.dominance_pvalue:.3f}',
                        ha='center', fontsize=10, color=self.config.significance_color,
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        else:
            # Fallback: simple shares calculation
            total_abs_effect = abs(interest_rate_effect) + abs(distortion_effect)
            if total_abs_effect > 0:
                ir_share = abs(interest_rate_effect) / total_abs_effect * 100
                dist_share = abs(distortion_effect) / total_abs_effect * 100
                
                shares = [ir_share, dist_share]
                labels = [f'Interest Rate\n({ir_share:.1f}%)', f'Market Distortion\n({dist_share:.1f}%)']
                
                ax2.pie(shares, labels=labels, colors=colors, autopct='', startangle=90,
                       textprops={'fontsize': 10})
        
        ax2.set_title('Relative Channel Contributions')
        
        # Overall figure formatting
        fig.suptitle(title, fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            self._save_figure(fig, save_path)
        
        return fig
    
    def create_impulse_response_chart(self,
                                    impulse_responses: pd.DataFrame,
                                    confidence_intervals: pd.DataFrame,
                                    title: str = "Investment Response to QE Shock",
                                    save_path: Optional[str] = None) -> plt.Figure:
        """
        Create impulse response chart with confidence intervals
        
        Args:
            impulse_responses: DataFrame with 'horizon' and 'response' columns
            confidence_intervals: DataFrame with 'horizon', 'lower', 'upper' columns
            title: Chart title
            save_path: Path to save the figure
            
        Returns:
            matplotlib Figure object
        """
        self.logger.info("Creating impulse response chart")
        
        fig, ax = self.pub_viz.publication_figure_template(double_column=True)
        
        horizons = impulse_responses['horizon']
        responses = impulse_responses['response']
        
        # Plot impulse response
        ax.plot(horizons, responses, color=self.config.total_effect_color,
               linewidth=2, marker='o', markersize=4, label='Point Estimate')
        
        # Add confidence intervals
        if not confidence_intervals.empty and self.config.show_confidence_intervals:
            ax.fill_between(confidence_intervals['horizon'],
                           confidence_intervals['lower'],
                           confidence_intervals['upper'],
                           alpha=0.3, color=self.config.total_effect_color,
                           label=f'{int(self.config.confidence_level*100)}% Confidence Interval')
        
        # Add zero line
        ax.axhline(y=0, color='black', linestyle='--', alpha=0.7)
        
        # Formatting
        ax.set_xlabel('Quarters After QE Shock')
        ax.set_ylabel('Investment Response (%)')
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Highlight significant periods
        if not confidence_intervals.empty:
            significant_periods = (confidence_intervals['lower'] > 0) | (confidence_intervals['upper'] < 0)
            if significant_periods.any():
                sig_horizons = confidence_intervals.loc[significant_periods, 'horizon']
                sig_responses = impulse_responses.loc[impulse_responses['horizon'].isin(sig_horizons), 'response']
                ax.scatter(sig_horizons, sig_responses, color=self.config.significance_color,
                          s=30, marker='*', label='Statistically Significant', zorder=5)
                ax.legend()
        
        plt.tight_layout()
        
        if save_path:
            self._save_figure(fig, save_path)
        
        return fig
    
    def create_robustness_comparison_chart(self,
                                         robustness_results: RobustnessTestResults,
                                         title: str = "Robustness Across QE Episode Definitions",
                                         save_path: Optional[str] = None) -> plt.Figure:
        """
        Create robustness comparison chart across different QE episode definitions
        
        Args:
            robustness_results: Results from robustness testing
            title: Chart title
            save_path: Path to save the figure
            
        Returns:
            matplotlib Figure object
        """
        self.logger.info("Creating robustness comparison chart")
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
        
        episodes = robustness_results.episode_definitions
        n_episodes = len(episodes)
        
        if n_episodes == 0:
            ax1.text(0.5, 0.5, 'No robustness results available',
                    ha='center', va='center', transform=ax1.transAxes)
            return fig
        
        x_pos = np.arange(n_episodes)
        
        # Panel 1: Interest Rate Effects
        ir_effects = robustness_results.interest_rate_effects
        ir_ci_lower = robustness_results.interest_rate_ci_lower
        ir_ci_upper = robustness_results.interest_rate_ci_upper
        ir_pvalues = robustness_results.interest_rate_pvalues
        
        bars1 = ax1.bar(x_pos, ir_effects, color=self.config.interest_rate_color,
                       alpha=self.config.bar_alpha, edgecolor='black')
        
        # Add confidence intervals
        if len(ir_ci_lower) == n_episodes and len(ir_ci_upper) == n_episodes:
            ax1.errorbar(x_pos, ir_effects,
                        yerr=[np.array(ir_effects) - np.array(ir_ci_lower),
                              np.array(ir_ci_upper) - np.array(ir_effects)],
                        fmt='none', color='black', capsize=3)
        
        # Mark significant results
        for i, (bar, pval) in enumerate(zip(bars1, ir_pvalues)):
            if pval < self.config.significance_level:
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                        '***' if pval < 0.01 else '*',
                        ha='center', va='bottom', color=self.config.significance_color)
        
        ax1.set_xticks(x_pos)
        ax1.set_xticklabels(episodes, rotation=45, ha='right')
        ax1.set_ylabel('Interest Rate Effect')
        ax1.set_title('Interest Rate Channel Effects')
        ax1.grid(True, alpha=0.3)
        ax1.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        
        # Panel 2: Market Distortion Effects
        dist_effects = robustness_results.distortion_effects
        dist_ci_lower = robustness_results.distortion_ci_lower
        dist_ci_upper = robustness_results.distortion_ci_upper
        dist_pvalues = robustness_results.distortion_pvalues
        
        bars2 = ax2.bar(x_pos, dist_effects, color=self.config.distortion_color,
                       alpha=self.config.bar_alpha, edgecolor='black')
        
        # Add confidence intervals
        if len(dist_ci_lower) == n_episodes and len(dist_ci_upper) == n_episodes:
            ax2.errorbar(x_pos, dist_effects,
                        yerr=[np.array(dist_effects) - np.array(dist_ci_lower),
                              np.array(dist_ci_upper) - np.array(dist_effects)],
                        fmt='none', color='black', capsize=3)
        
        # Mark significant results
        for i, (bar, pval) in enumerate(zip(bars2, dist_pvalues)):
            if pval < self.config.significance_level:
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                        '***' if pval < 0.01 else '*',
                        ha='center', va='bottom', color=self.config.significance_color)
        
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels(episodes, rotation=45, ha='right')
        ax2.set_ylabel('Market Distortion Effect')
        ax2.set_title('Market Distortion Channel Effects')
        ax2.grid(True, alpha=0.3)
        ax2.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        
        # Panel 3: Dominance Test Results
        dominance_results = robustness_results.dominance_results
        dominance_pvalues = robustness_results.dominance_pvalues
        
        # Create binary dominance indicators
        dominance_binary = [1 if dom else 0 for dom in dominance_results]
        colors = [self.config.significance_color if dom else 'gray' for dom in dominance_results]
        
        bars3 = ax3.bar(x_pos, dominance_binary, color=colors,
                       alpha=self.config.bar_alpha, edgecolor='black')
        
        # Add p-values as text
        for i, (bar, pval) in enumerate(zip(bars3, dominance_pvalues)):
            ax3.text(bar.get_x() + bar.get_width()/2., 0.5,
                    f'p={pval:.3f}', ha='center', va='center',
                    rotation=90, fontsize=8)
        
        ax3.set_xticks(x_pos)
        ax3.set_xticklabels(episodes, rotation=45, ha='right')
        ax3.set_ylabel('Distortion Dominance')
        ax3.set_title('Distortion Dominance Test Results')
        ax3.set_ylim(0, 1.2)
        ax3.grid(True, alpha=0.3)
        
        # Panel 4: Model Fit Comparison
        r_squared = robustness_results.r_squared_values
        observations = robustness_results.observations
        
        if len(r_squared) == n_episodes:
            bars4 = ax4.bar(x_pos, r_squared, color=self.config.total_effect_color,
                           alpha=self.config.bar_alpha, edgecolor='black')
            
            # Add observation counts as text
            for i, (bar, obs) in enumerate(zip(bars4, observations)):
                height = bar.get_height()
                ax4.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                        f'N={obs}', ha='center', va='bottom', fontsize=8)
            
            ax4.set_xticks(x_pos)
            ax4.set_xticklabels(episodes, rotation=45, ha='right')
            ax4.set_ylabel('R-squared')
            ax4.set_title('Model Fit Across Specifications')
            ax4.grid(True, alpha=0.3)
        
        fig.suptitle(title, fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            self._save_figure(fig, save_path)
        
        return fig
    
    def run_robustness_tests(self,
                           investment_data: pd.Series,
                           qe_intensity_data: pd.Series,
                           market_distortions: pd.Series,
                           interest_rates: pd.Series,
                           episode_definitions: Optional[Dict[str, pd.Series]] = None) -> RobustnessTestResults:
        """
        Run robustness tests across different QE episode definitions
        
        Args:
            investment_data: Private investment series
            qe_intensity_data: QE intensity measure
            market_distortions: Market distortion proxy
            interest_rates: Interest rate proxy
            episode_definitions: Dictionary of episode definitions with boolean masks
            
        Returns:
            RobustnessTestResults object
        """
        self.logger.info("Running robustness tests across QE episode definitions")
        
        results = RobustnessTestResults()
        
        # Default episode definitions if not provided
        if episode_definitions is None:
            episode_definitions = self._create_default_episode_definitions(qe_intensity_data)
        
        results.episode_definitions = list(episode_definitions.keys())
        
        # Align all data to common dates
        common_dates = investment_data.index.intersection(qe_intensity_data.index)\
                                           .intersection(market_distortions.index)\
                                           .intersection(interest_rates.index)
        
        investment_aligned = investment_data.loc[common_dates]
        qe_aligned = qe_intensity_data.loc[common_dates]
        distortions_aligned = market_distortions.loc[common_dates]
        rates_aligned = interest_rates.loc[common_dates]
        
        # Run analysis for each episode definition
        for episode_name, episode_mask in episode_definitions.items():
            try:
                self.logger.info(f"Testing episode definition: {episode_name}")
                
                # Apply episode mask
                episode_mask_aligned = episode_mask.loc[common_dates]
                episode_dates = common_dates[episode_mask_aligned]
                
                if len(episode_dates) < 20:
                    self.logger.warning(f"Insufficient data for {episode_name}: {len(episode_dates)} observations")
                    continue
                
                # Subset data to episode
                y_episode = investment_aligned.loc[episode_dates].pct_change().dropna()
                qe_episode = qe_aligned.loc[y_episode.index]
                distortions_episode = distortions_aligned.loc[y_episode.index]
                rates_episode = rates_aligned.loc[y_episode.index]
                
                # Run channel decomposition
                channel_results = self._run_channel_decomposition(
                    y_episode, qe_episode, distortions_episode, rates_episode
                )
                
                # Store results
                results.interest_rate_effects.append(channel_results['interest_rate_effect'])
                results.distortion_effects.append(channel_results['distortion_effect'])
                results.dominance_results.append(channel_results['distortion_dominance'])
                
                results.interest_rate_pvalues.append(channel_results['interest_rate_pvalue'])
                results.distortion_pvalues.append(channel_results['distortion_pvalue'])
                results.dominance_pvalues.append(channel_results['dominance_pvalue'])
                
                results.interest_rate_ci_lower.append(channel_results['interest_rate_ci_lower'])
                results.interest_rate_ci_upper.append(channel_results['interest_rate_ci_upper'])
                results.distortion_ci_lower.append(channel_results['distortion_ci_lower'])
                results.distortion_ci_upper.append(channel_results['distortion_ci_upper'])
                
                results.r_squared_values.append(channel_results['r_squared'])
                results.observations.append(len(y_episode))
                
            except Exception as e:
                self.logger.error(f"Error testing {episode_name}: {e}")
                continue
        
        # Calculate summary statistics
        if results.interest_rate_effects:
            results.mean_interest_rate_effect = np.mean(results.interest_rate_effects)
            results.mean_distortion_effect = np.mean(results.distortion_effects)
            
            # Calculate consistency score (1 - coefficient of variation)
            ir_cv = np.std(results.interest_rate_effects) / abs(results.mean_interest_rate_effect) if results.mean_interest_rate_effect != 0 else float('inf')
            dist_cv = np.std(results.distortion_effects) / abs(results.mean_distortion_effect) if results.mean_distortion_effect != 0 else float('inf')
            
            # Handle infinite CV values
            if ir_cv == float('inf') and dist_cv == float('inf'):
                results.consistency_score = 0.0
            elif ir_cv == float('inf'):
                results.consistency_score = max(0.0, 1 - dist_cv)
            elif dist_cv == float('inf'):
                results.consistency_score = max(0.0, 1 - ir_cv)
            else:
                results.consistency_score = max(0.0, 1 - (ir_cv + dist_cv) / 2)
        
        self.robustness_results = results
        self.logger.info(f"Completed robustness testing across {len(results.episode_definitions)} definitions")
        
        return results
    
    def _create_default_episode_definitions(self, qe_intensity: pd.Series) -> Dict[str, pd.Series]:
        """
        Create default episode definitions for robustness testing
        
        Args:
            qe_intensity: QE intensity series
            
        Returns:
            Dictionary of episode definitions
        """
        episode_definitions = {}
        
        # All episodes
        episode_definitions['all_episodes'] = pd.Series(True, index=qe_intensity.index)
        
        # Intensive QE episodes (top 25% of intensity)
        intensity_threshold = qe_intensity.quantile(0.75)
        episode_definitions['intensive_only'] = qe_intensity > intensity_threshold
        
        # Crisis periods (2008-2010, 2020-2021) - approximate
        crisis_mask = pd.Series(False, index=qe_intensity.index)
        for date in qe_intensity.index:
            if (2008 <= date.year <= 2010) or (2020 <= date.year <= 2021):
                crisis_mask.loc[date] = True
        episode_definitions['crisis_periods'] = crisis_mask
        
        # Post-crisis periods (2011-2019)
        post_crisis_mask = pd.Series(False, index=qe_intensity.index)
        for date in qe_intensity.index:
            if 2011 <= date.year <= 2019:
                post_crisis_mask.loc[date] = True
        episode_definitions['post_crisis'] = post_crisis_mask
        
        return episode_definitions
    
    def _run_channel_decomposition(self,
                                 investment: pd.Series,
                                 qe_intensity: pd.Series,
                                 market_distortions: pd.Series,
                                 interest_rates: pd.Series) -> Dict[str, Any]:
        """
        Run channel decomposition analysis for robustness testing
        
        Args:
            investment: Investment growth series
            qe_intensity: QE intensity measure
            market_distortions: Market distortion proxy
            interest_rates: Interest rate proxy
            
        Returns:
            Dictionary with decomposition results
        """
        # Create interaction terms
        qe_rate_interaction = qe_intensity * interest_rates
        qe_distortion_interaction = qe_intensity * market_distortions
        
        # Prepare regression data
        X_data = pd.DataFrame({
            'qe_rate_interaction': qe_rate_interaction,
            'qe_distortion_interaction': qe_distortion_interaction
        })
        
        # Add constant
        X_reg = sm.add_constant(X_data)
        
        # Fit regression
        model = sm.OLS(investment.values, X_reg).fit()
        
        # Extract results
        results = {
            'interest_rate_effect': 0.0,
            'distortion_effect': 0.0,
            'interest_rate_pvalue': 1.0,
            'distortion_pvalue': 1.0,
            'interest_rate_ci_lower': 0.0,
            'interest_rate_ci_upper': 0.0,
            'distortion_ci_lower': 0.0,
            'distortion_ci_upper': 0.0,
            'distortion_dominance': False,
            'dominance_pvalue': 1.0,
            'r_squared': 0.0
        }
        
        if len(model.params) >= 3:  # Constant + 2 coefficients
            # Interest rate channel effect
            results['interest_rate_effect'] = model.params.iloc[1]
            results['interest_rate_pvalue'] = model.pvalues.iloc[1]
            
            # Market distortion channel effect
            results['distortion_effect'] = model.params.iloc[2]
            results['distortion_pvalue'] = model.pvalues.iloc[2]
            
            # Calculate confidence intervals
            alpha = 1 - self.config.confidence_level
            t_crit = stats.t.ppf(1 - alpha/2, model.df_resid)
            
            results['interest_rate_ci_lower'] = results['interest_rate_effect'] - t_crit * model.bse.iloc[1]
            results['interest_rate_ci_upper'] = results['interest_rate_effect'] + t_crit * model.bse.iloc[1]
            results['distortion_ci_lower'] = results['distortion_effect'] - t_crit * model.bse.iloc[2]
            results['distortion_ci_upper'] = results['distortion_effect'] + t_crit * model.bse.iloc[2]
            
            # Test for dominance
            effect_diff = results['distortion_effect'] - results['interest_rate_effect']
            se_diff = np.sqrt(model.bse.iloc[2]**2 + model.bse.iloc[1]**2)
            
            if se_diff > 0:
                dominance_test_stat = effect_diff / se_diff
                results['dominance_pvalue'] = 1 - stats.t.cdf(dominance_test_stat, model.df_resid)
                results['distortion_dominance'] = results['dominance_pvalue'] < self.config.significance_level
            
            results['r_squared'] = model.rsquared
        
        return results
    
    def create_comprehensive_analysis_report(self,
                                           investment_results: InvestmentImpactResults,
                                           robustness_results: Optional[RobustnessTestResults] = None,
                                           output_dir: str = "investment_analysis_output") -> Dict[str, str]:
        """
        Create comprehensive analysis report with all visualizations
        
        Args:
            investment_results: Main investment impact results
            robustness_results: Robustness test results
            output_dir: Directory to save outputs
            
        Returns:
            Dictionary with paths to created files
        """
        self.logger.info("Creating comprehensive investment analysis report")
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        created_files = {}
        
        # 1. Channel decomposition chart
        if hasattr(investment_results, 'channel_decomposition') and investment_results.channel_decomposition.fitted:
            decomp_path = output_path / "channel_decomposition"
            fig1 = self.create_channel_decomposition_chart(
                investment_results.channel_decomposition,
                save_path=str(decomp_path)
            )
            created_files['channel_decomposition'] = str(decomp_path)
            plt.close(fig1)
        
        # 2. Impulse response chart
        if not investment_results.impulse_responses.empty:
            impulse_path = output_path / "impulse_responses"
            fig2 = self.create_impulse_response_chart(
                investment_results.impulse_responses,
                investment_results.confidence_intervals,
                save_path=str(impulse_path)
            )
            created_files['impulse_responses'] = str(impulse_path)
            plt.close(fig2)
        
        # 3. Robustness comparison chart
        if robustness_results is not None and robustness_results.episode_definitions:
            robust_path = output_path / "robustness_comparison"
            fig3 = self.create_robustness_comparison_chart(
                robustness_results,
                save_path=str(robust_path)
            )
            created_files['robustness_comparison'] = str(robust_path)
            plt.close(fig3)
        
        # 4. Summary statistics table
        summary_path = output_path / "summary_statistics.txt"
        self._create_summary_statistics_table(investment_results, robustness_results, summary_path)
        created_files['summary_statistics'] = str(summary_path)
        
        self.logger.info(f"Created comprehensive analysis report in {output_dir}")
        return created_files
    
    def _create_summary_statistics_table(self,
                                       investment_results: InvestmentImpactResults,
                                       robustness_results: Optional[RobustnessTestResults],
                                       save_path: Path) -> None:
        """
        Create summary statistics table
        
        Args:
            investment_results: Main investment impact results
            robustness_results: Robustness test results
            save_path: Path to save the table
        """
        with open(save_path, 'w') as f:
            f.write("INVESTMENT IMPACT ANALYSIS - SUMMARY STATISTICS\n")
            f.write("=" * 60 + "\n\n")
            
            # Main results
            f.write("MAIN RESULTS:\n")
            f.write("-" * 20 + "\n")
            f.write(f"Short-term effect: {investment_results.short_term_effect:.4f} (p={investment_results.short_term_pvalue:.3f})\n")
            f.write(f"Long-term effect: {investment_results.long_term_effect:.4f} (p={investment_results.long_term_pvalue:.3f})\n")
            
            if hasattr(investment_results, 'channel_decomposition') and investment_results.channel_decomposition.fitted:
                cd = investment_results.channel_decomposition
                f.write(f"\nCHANNEL DECOMPOSITION:\n")
                f.write("-" * 25 + "\n")
                f.write(f"Interest rate effect: {cd.interest_rate_effect:.4f} (p={cd.interest_rate_pvalue:.3f})\n")
                f.write(f"Market distortion effect: {cd.distortion_effect:.4f} (p={cd.distortion_pvalue:.3f})\n")
                f.write(f"Distortion dominance: {'Yes' if cd.distortion_dominance else 'No'} (p={cd.dominance_pvalue:.3f})\n")
                f.write(f"R-squared: {cd.r_squared:.4f}\n")
            
            # Robustness results
            if robustness_results is not None and robustness_results.episode_definitions:
                f.write(f"\nROBUSTNESS TESTS:\n")
                f.write("-" * 20 + "\n")
                f.write(f"Episode definitions tested: {len(robustness_results.episode_definitions)}\n")
                f.write(f"Mean interest rate effect: {robustness_results.mean_interest_rate_effect:.4f}\n")
                f.write(f"Mean distortion effect: {robustness_results.mean_distortion_effect:.4f}\n")
                f.write(f"Consistency score: {robustness_results.consistency_score:.4f}\n")
                
                # Dominance results across specifications
                dominance_count = sum(robustness_results.dominance_results)
                f.write(f"Distortion dominance in {dominance_count}/{len(robustness_results.dominance_results)} specifications\n")
    
    def _save_figure(self, fig: plt.Figure, save_path: str) -> None:
        """
        Save figure in multiple formats
        
        Args:
            fig: matplotlib Figure object
            save_path: Base path for saving (without extension)
        """
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        for fmt in self.config.save_formats:
            file_path = save_path.with_suffix(f'.{fmt}')
            fig.savefig(file_path, format=fmt, dpi=self.config.dpi,
                       bbox_inches='tight', facecolor='white', edgecolor='none')