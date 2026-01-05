"""
Publication-Quality Visualization Suite for Economics Research

This module provides publication-ready visualization capabilities specifically designed
for economics journals, with consistent styling, high-quality figures, and comprehensive
diagnostic visualizations.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union, Any
import warnings
from pathlib import Path

# Suppress matplotlib warnings for cleaner output
warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib')

class PublicationVisualizationSuite:
    """
    Publication-quality visualization suite for economics research.
    
    Provides journal-ready figures with consistent styling, proper formatting,
    and comprehensive diagnostic capabilities suitable for top-tier economics journals.
    """
    
    def __init__(self, style: str = 'economics_journal'):
        """
        Initialize the publication visualization suite.
        
        Parameters:
        -----------
        style : str
            Style configuration to use ('economics_journal', 'aer', 'qje', 'jpe')
        """
        self.style = style
        self.style_config = self._load_style_config(style)
        self._setup_matplotlib_params()
        
    def _load_style_config(self, style: str) -> Dict[str, Any]:
        """
        Load style configuration for different journal standards.
        
        Parameters:
        -----------
        style : str
            Style name to load
            
        Returns:
        --------
        Dict[str, Any]
            Style configuration dictionary
        """
        base_config = {
            'figure_size_single': (3.5, 2.8),  # Single column width
            'figure_size_double': (7.0, 5.6),  # Double column width
            'dpi': 300,
            'font_family': 'serif',
            'font_serif': ['Times New Roman', 'Times', 'serif'],
            'font_size_base': 10,
            'font_size_title': 12,
            'font_size_label': 10,
            'font_size_tick': 9,
            'font_size_legend': 9,
            'line_width': 1.2,
            'marker_size': 4,
            'grid_alpha': 0.3,
            'spine_width': 0.8,
        }
        
        # Journal-specific configurations
        journal_configs = {
            'economics_journal': {
                **base_config,
                'color_palette': ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'],
                'confidence_alpha': 0.2,
                'grid_style': '--',
                'legend_frameon': False,
            },
            'aer': {
                **base_config,
                'color_palette': ['#000000', '#666666', '#999999', '#cccccc'],
                'confidence_alpha': 0.15,
                'grid_style': ':',
                'legend_frameon': True,
            },
            'qje': {
                **base_config,
                'color_palette': ['#2E4057', '#048A81', '#54C6EB', '#F18F01', '#C73E1D'],
                'confidence_alpha': 0.25,
                'grid_style': '-',
                'legend_frameon': False,
            },
            'jpe': {
                **base_config,
                'color_palette': ['#1B263B', '#415A77', '#778DA9', '#E0E1DD'],
                'confidence_alpha': 0.2,
                'grid_style': '--',
                'legend_frameon': True,
            }
        }
        
        return journal_configs.get(style, journal_configs['economics_journal'])
    
    def _setup_matplotlib_params(self):
        """Set up matplotlib parameters for publication quality."""
        plt.rcParams.update({
            'figure.dpi': self.style_config['dpi'],
            'savefig.dpi': self.style_config['dpi'],
            'font.family': self.style_config['font_family'],
            'font.serif': self.style_config['font_serif'],
            'font.size': self.style_config['font_size_base'],
            'axes.titlesize': self.style_config['font_size_title'],
            'axes.labelsize': self.style_config['font_size_label'],
            'xtick.labelsize': self.style_config['font_size_tick'],
            'ytick.labelsize': self.style_config['font_size_tick'],
            'legend.fontsize': self.style_config['font_size_legend'],
            'lines.linewidth': self.style_config['line_width'],
            'lines.markersize': self.style_config['marker_size'],
            'axes.linewidth': self.style_config['spine_width'],
            'grid.alpha': self.style_config['grid_alpha'],
            'grid.linestyle': self.style_config['grid_style'],
            'legend.frameon': self.style_config['legend_frameon'],
            'axes.spines.top': False,
            'axes.spines.right': False,
            'figure.autolayout': True,
        })
    
    def publication_figure_template(self, 
                                  figsize: Optional[Tuple[float, float]] = None,
                                  double_column: bool = False) -> Tuple[plt.Figure, plt.Axes]:
        """
        Create a publication-ready figure template with consistent formatting.
        
        Parameters:
        -----------
        figsize : Optional[Tuple[float, float]]
            Custom figure size, if None uses style defaults
        double_column : bool
            Whether to use double column width
            
        Returns:
        --------
        Tuple[plt.Figure, plt.Axes]
            Figure and axes objects
        """
        if figsize is None:
            if double_column:
                figsize = self.style_config['figure_size_double']
            else:
                figsize = self.style_config['figure_size_single']
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # Apply consistent styling
        ax.grid(True, alpha=self.style_config['grid_alpha'], 
                linestyle=self.style_config['grid_style'])
        ax.set_axisbelow(True)
        
        return fig, ax
    
    def create_threshold_analysis_figure(self, 
                                       threshold_results: Dict[str, Any],
                                       confidence_intervals: Optional[Dict[str, np.ndarray]] = None,
                                       save_path: Optional[str] = None,
                                       title: Optional[str] = None) -> plt.Figure:
        """
        Create publication-quality threshold analysis figure with confidence intervals.
        
        Parameters:
        -----------
        threshold_results : Dict[str, Any]
            Dictionary containing threshold analysis results with keys:
            - 'threshold_values': array of threshold values
            - 'coefficients': array of coefficient estimates
            - 'threshold_estimate': estimated threshold value
            - 'regime_1_coef': coefficient for regime 1
            - 'regime_2_coef': coefficient for regime 2
        confidence_intervals : Optional[Dict[str, np.ndarray]]
            Confidence intervals for coefficients
        save_path : Optional[str]
            Path to save the figure
        title : Optional[str]
            Figure title
            
        Returns:
        --------
        plt.Figure
            The created figure
        """
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=self.style_config['figure_size_double'])
        
        colors = self.style_config['color_palette']
        
        # Top panel: Threshold search results
        threshold_vals = threshold_results.get('threshold_values', [])
        coefficients = threshold_results.get('coefficients', [])
        
        if len(threshold_vals) > 0 and len(coefficients) > 0:
            ax1.plot(threshold_vals, coefficients, color=colors[0], 
                    linewidth=self.style_config['line_width'])
            
            # Mark estimated threshold
            threshold_est = threshold_results.get('threshold_estimate')
            if threshold_est is not None:
                ax1.axvline(threshold_est, color=colors[1], linestyle='--', 
                           label=f'Estimated Threshold: {threshold_est:.3f}')
                ax1.legend()
        
        ax1.set_xlabel('Threshold Variable')
        ax1.set_ylabel('Coefficient Estimate')
        ax1.set_title('Threshold Search Results')
        ax1.grid(True, alpha=self.style_config['grid_alpha'])
        
        # Bottom panel: Regime-specific effects
        regimes = ['Regime 1\n(Below Threshold)', 'Regime 2\n(Above Threshold)']
        regime_coefs = [
            threshold_results.get('regime_1_coef', 0),
            threshold_results.get('regime_2_coef', 0)
        ]
        
        bars = ax2.bar(regimes, regime_coefs, color=[colors[0], colors[1]], 
                      alpha=0.7, edgecolor='black', linewidth=0.8)
        
        # Add confidence intervals if provided
        if confidence_intervals:
            regime_1_ci = confidence_intervals.get('regime_1', [0, 0])
            regime_2_ci = confidence_intervals.get('regime_2', [0, 0])
            
            ci_lower = [regime_1_ci[0], regime_2_ci[0]]
            ci_upper = [regime_1_ci[1], regime_2_ci[1]]
            
            ax2.errorbar(range(len(regimes)), regime_coefs, 
                        yerr=[np.array(regime_coefs) - np.array(ci_lower),
                              np.array(ci_upper) - np.array(regime_coefs)],
                        fmt='none', color='black', capsize=3, capthick=1)
        
        ax2.set_ylabel('Coefficient Estimate')
        ax2.set_title('Regime-Specific Effects')
        ax2.grid(True, alpha=self.style_config['grid_alpha'])
        ax2.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        
        # Add value labels on bars
        for bar, coef in zip(bars, regime_coefs):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{coef:.3f}', ha='center', va='bottom')
        
        if title:
            fig.suptitle(title, fontsize=self.style_config['font_size_title'])
        
        plt.tight_layout()
        
        if save_path:
            self._save_figure(fig, save_path)
        
        return fig
    
    def journal_style_config(self, journal: str = None) -> Dict[str, Any]:
        """
        Get or update journal-specific style configuration.
        
        Parameters:
        -----------
        journal : str, optional
            Journal name to get configuration for
            
        Returns:
        --------
        Dict[str, Any]
            Style configuration dictionary
        """
        if journal:
            return self._load_style_config(journal)
        return self.style_config
    
    def _save_figure(self, fig: plt.Figure, save_path: str, 
                    formats: List[str] = None) -> None:
        """
        Save figure in publication-ready formats.
        
        Parameters:
        -----------
        fig : plt.Figure
            Figure to save
        save_path : str
            Base path for saving (without extension)
        formats : List[str], optional
            List of formats to save ['png', 'pdf', 'eps']
        """
        if formats is None:
            formats = ['png', 'pdf']
        
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        for fmt in formats:
            file_path = save_path.with_suffix(f'.{fmt}')
            fig.savefig(file_path, format=fmt, dpi=self.style_config['dpi'],
                       bbox_inches='tight', facecolor='white', edgecolor='none')
    
    def create_coefficient_plot(self, 
                               coefficients: pd.DataFrame,
                               title: str = "Coefficient Estimates",
                               save_path: Optional[str] = None) -> plt.Figure:
        """
        Create publication-quality coefficient plot with confidence intervals.
        
        Parameters:
        -----------
        coefficients : pd.DataFrame
            DataFrame with columns: 'coef', 'se', 'ci_lower', 'ci_upper'
        title : str
            Plot title
        save_path : Optional[str]
            Path to save figure
            
        Returns:
        --------
        plt.Figure
            The created figure
        """
        fig, ax = self.publication_figure_template()
        
        y_pos = np.arange(len(coefficients))
        colors = self.style_config['color_palette']
        
        # Plot coefficients
        ax.errorbar(coefficients['coef'], y_pos, 
                   xerr=[coefficients['coef'] - coefficients['ci_lower'],
                         coefficients['ci_upper'] - coefficients['coef']],
                   fmt='o', color=colors[0], capsize=3, capthick=1,
                   markersize=self.style_config['marker_size'])
        
        # Add vertical line at zero
        ax.axvline(x=0, color='black', linestyle='-', alpha=0.5)
        
        # Customize axes
        ax.set_yticks(y_pos)
        ax.set_yticklabels(coefficients.index)
        ax.set_xlabel('Coefficient Estimate')
        ax.set_title(title)
        ax.grid(True, alpha=self.style_config['grid_alpha'])
        
        plt.tight_layout()
        
        if save_path:
            self._save_figure(fig, save_path)
        
        return fig
    
    def create_time_series_plot(self, 
                               data: pd.DataFrame,
                               y_cols: List[str],
                               x_col: str = None,
                               title: str = "Time Series Plot",
                               save_path: Optional[str] = None) -> plt.Figure:
        """
        Create publication-quality time series plot.
        
        Parameters:
        -----------
        data : pd.DataFrame
            Data to plot
        y_cols : List[str]
            Column names for y-axis variables
        x_col : str, optional
            Column name for x-axis (uses index if None)
        title : str
            Plot title
        save_path : Optional[str]
            Path to save figure
            
        Returns:
        --------
        plt.Figure
            The created figure
        """
        fig, ax = self.publication_figure_template(double_column=True)
        
        colors = self.style_config['color_palette']
        x_data = data[x_col] if x_col else data.index
        
        for i, col in enumerate(y_cols):
            ax.plot(x_data, data[col], color=colors[i % len(colors)], 
                   label=col, linewidth=self.style_config['line_width'])
        
        ax.set_xlabel(x_col if x_col else 'Time')
        ax.set_ylabel('Value')
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=self.style_config['grid_alpha'])
        
        plt.tight_layout()
        
        if save_path:
            self._save_figure(fig, save_path)
        
        return fig
    
    def create_scatter_plot(self, 
                           x_data: np.ndarray,
                           y_data: np.ndarray,
                           fit_line: bool = True,
                           title: str = "Scatter Plot",
                           xlabel: str = "X Variable",
                           ylabel: str = "Y Variable",
                           save_path: Optional[str] = None) -> plt.Figure:
        """
        Create publication-quality scatter plot with optional fit line.
        
        Parameters:
        -----------
        x_data : np.ndarray
            X-axis data
        y_data : np.ndarray
            Y-axis data
        fit_line : bool
            Whether to add regression line
        title : str
            Plot title
        xlabel : str
            X-axis label
        ylabel : str
            Y-axis label
        save_path : Optional[str]
            Path to save figure
            
        Returns:
        --------
        plt.Figure
            The created figure
        """
        fig, ax = self.publication_figure_template()
        
        colors = self.style_config['color_palette']
        
        # Create scatter plot
        ax.scatter(x_data, y_data, color=colors[0], alpha=0.6,
                  s=self.style_config['marker_size']**2)
        
        # Add fit line if requested
        if fit_line:
            z = np.polyfit(x_data, y_data, 1)
            p = np.poly1d(z)
            ax.plot(x_data, p(x_data), color=colors[1], 
                   linestyle='--', linewidth=self.style_config['line_width'])
            
            # Add R-squared
            correlation_matrix = np.corrcoef(x_data, y_data)
            r_squared = correlation_matrix[0, 1]**2
            ax.text(0.05, 0.95, f'R² = {r_squared:.3f}', 
                   transform=ax.transAxes, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, alpha=self.style_config['grid_alpha'])
        
        plt.tight_layout()
        
        if save_path:
            self._save_figure(fig, save_path)
        
        return fig    

    def create_diagnostic_panel_figure(self, 
                                     diagnostics: Dict[str, Any],
                                     save_path: Optional[str] = None,
                                     title: Optional[str] = None) -> plt.Figure:
        """
        Create comprehensive diagnostic panel with residual plots and Q-Q plots.
        
        Parameters:
        -----------
        diagnostics : Dict[str, Any]
            Dictionary containing diagnostic results with keys:
            - 'residuals': array of residuals
            - 'fitted_values': array of fitted values
            - 'standardized_residuals': standardized residuals
            - 'leverage': leverage values (optional)
            - 'cooks_distance': Cook's distance values (optional)
        save_path : Optional[str]
            Path to save the figure
        title : Optional[str]
            Figure title
            
        Returns:
        --------
        plt.Figure
            The created diagnostic panel figure
        """
        fig = plt.figure(figsize=self.style_config['figure_size_double'])
        
        # Create 2x2 subplot layout
        gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
        
        colors = self.style_config['color_palette']
        
        # Extract diagnostic data
        residuals = diagnostics.get('residuals', [])
        fitted_values = diagnostics.get('fitted_values', [])
        std_residuals = diagnostics.get('standardized_residuals', [])
        
        if len(residuals) == 0:
            # Handle empty diagnostics
            ax = fig.add_subplot(gs[:, :])
            ax.text(0.5, 0.5, 'No diagnostic data available', 
                   ha='center', va='center', transform=ax.transAxes)
            return fig
        
        # 1. Residuals vs Fitted Values
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.scatter(fitted_values, residuals, color=colors[0], alpha=0.6,
                   s=self.style_config['marker_size']**2)
        ax1.axhline(y=0, color='red', linestyle='--', alpha=0.8)
        
        # Add LOWESS smooth line
        if len(fitted_values) > 10:
            try:
                from scipy.stats import linregress
                # Simple moving average as LOWESS alternative
                sorted_indices = np.argsort(fitted_values)
                window_size = max(10, len(fitted_values) // 10)
                smooth_fitted = []
                smooth_residuals = []
                
                for i in range(0, len(fitted_values) - window_size + 1, window_size // 2):
                    end_idx = min(i + window_size, len(fitted_values))
                    indices = sorted_indices[i:end_idx]
                    smooth_fitted.append(np.mean(fitted_values[indices]))
                    smooth_residuals.append(np.mean(residuals[indices]))
                
                ax1.plot(smooth_fitted, smooth_residuals, color=colors[1], 
                        linewidth=self.style_config['line_width'])
            except:
                pass  # Skip smooth line if calculation fails
        
        ax1.set_xlabel('Fitted Values')
        ax1.set_ylabel('Residuals')
        ax1.set_title('Residuals vs Fitted')
        ax1.grid(True, alpha=self.style_config['grid_alpha'])
        
        # 2. Q-Q Plot (Normal Probability Plot)
        ax2 = fig.add_subplot(gs[0, 1])
        
        if len(std_residuals) > 0:
            # Sort standardized residuals
            sorted_residuals = np.sort(std_residuals)
            n = len(sorted_residuals)
            
            # Theoretical quantiles (standard normal)
            theoretical_quantiles = np.array([
                np.percentile(np.random.standard_normal(10000), 100 * (i + 0.5) / n)
                for i in range(n)
            ])
            
            ax2.scatter(theoretical_quantiles, sorted_residuals, 
                       color=colors[0], alpha=0.6,
                       s=self.style_config['marker_size']**2)
            
            # Add reference line (y = x)
            min_val = min(np.min(theoretical_quantiles), np.min(sorted_residuals))
            max_val = max(np.max(theoretical_quantiles), np.max(sorted_residuals))
            ax2.plot([min_val, max_val], [min_val, max_val], 
                    color='red', linestyle='--', alpha=0.8)
        
        ax2.set_xlabel('Theoretical Quantiles')
        ax2.set_ylabel('Sample Quantiles')
        ax2.set_title('Normal Q-Q Plot')
        ax2.grid(True, alpha=self.style_config['grid_alpha'])
        
        # 3. Scale-Location Plot (Standardized residuals vs Fitted)
        ax3 = fig.add_subplot(gs[1, 0])
        
        if len(std_residuals) > 0:
            sqrt_abs_residuals = np.sqrt(np.abs(std_residuals))
            ax3.scatter(fitted_values, sqrt_abs_residuals, 
                       color=colors[0], alpha=0.6,
                       s=self.style_config['marker_size']**2)
            
            # Add smooth line
            if len(fitted_values) > 10:
                try:
                    sorted_indices = np.argsort(fitted_values)
                    window_size = max(10, len(fitted_values) // 10)
                    smooth_fitted = []
                    smooth_sqrt_residuals = []
                    
                    for i in range(0, len(fitted_values) - window_size + 1, window_size // 2):
                        end_idx = min(i + window_size, len(fitted_values))
                        indices = sorted_indices[i:end_idx]
                        smooth_fitted.append(np.mean(fitted_values[indices]))
                        smooth_sqrt_residuals.append(np.mean(sqrt_abs_residuals[indices]))
                    
                    ax3.plot(smooth_fitted, smooth_sqrt_residuals, color=colors[1], 
                            linewidth=self.style_config['line_width'])
                except:
                    pass
        
        ax3.set_xlabel('Fitted Values')
        ax3.set_ylabel('√|Standardized Residuals|')
        ax3.set_title('Scale-Location')
        ax3.grid(True, alpha=self.style_config['grid_alpha'])
        
        # 4. Residuals vs Leverage (if leverage data available)
        ax4 = fig.add_subplot(gs[1, 1])
        
        leverage = diagnostics.get('leverage', [])
        cooks_distance = diagnostics.get('cooks_distance', [])
        
        if len(leverage) > 0 and len(std_residuals) > 0:
            ax4.scatter(leverage, std_residuals, color=colors[0], alpha=0.6,
                       s=self.style_config['marker_size']**2)
            
            # Add Cook's distance contours if available
            if len(cooks_distance) > 0:
                # Highlight points with high Cook's distance
                high_cooks = cooks_distance > 4 / len(cooks_distance)
                if np.any(high_cooks):
                    ax4.scatter(leverage[high_cooks], std_residuals[high_cooks], 
                               color=colors[3], s=self.style_config['marker_size']**2 * 2,
                               label='High Cook\'s Distance')
                    ax4.legend()
            
            ax4.set_xlabel('Leverage')
            ax4.set_ylabel('Standardized Residuals')
            ax4.set_title('Residuals vs Leverage')
        else:
            # Histogram of residuals as alternative
            ax4.hist(residuals, bins=20, color=colors[0], alpha=0.7, edgecolor='black')
            ax4.set_xlabel('Residuals')
            ax4.set_ylabel('Frequency')
            ax4.set_title('Residual Distribution')
        
        ax4.grid(True, alpha=self.style_config['grid_alpha'])
        
        if title:
            fig.suptitle(title, fontsize=self.style_config['font_size_title'])
        
        if save_path:
            self._save_figure(fig, save_path)
        
        return fig
    
    def create_model_fit_comparison_figure(self, 
                                         model_results: Dict[str, Dict[str, Any]],
                                         save_path: Optional[str] = None,
                                         title: str = "Model Fit Comparison") -> plt.Figure:
        """
        Create model fit comparison figure for specification comparison.
        
        Parameters:
        -----------
        model_results : Dict[str, Dict[str, Any]]
            Dictionary with model names as keys and results as values.
            Each result should contain: 'r_squared', 'aic', 'bic', 'log_likelihood'
        save_path : Optional[str]
            Path to save the figure
        title : str
            Figure title
            
        Returns:
        --------
        plt.Figure
            The created comparison figure
        """
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, 
                                                     figsize=self.style_config['figure_size_double'])
        
        colors = self.style_config['color_palette']
        
        model_names = list(model_results.keys())
        n_models = len(model_names)
        
        if n_models == 0:
            ax1.text(0.5, 0.5, 'No model results available', 
                    ha='center', va='center', transform=ax1.transAxes)
            return fig
        
        # Extract metrics
        r_squared = [model_results[name].get('r_squared', 0) for name in model_names]
        aic = [model_results[name].get('aic', 0) for name in model_names]
        bic = [model_results[name].get('bic', 0) for name in model_names]
        log_likelihood = [model_results[name].get('log_likelihood', 0) for name in model_names]
        
        x_pos = np.arange(n_models)
        
        # 1. R-squared comparison
        bars1 = ax1.bar(x_pos, r_squared, color=colors[0], alpha=0.7, edgecolor='black')
        ax1.set_xlabel('Models')
        ax1.set_ylabel('R-squared')
        ax1.set_title('Model R-squared Comparison')
        ax1.set_xticks(x_pos)
        ax1.set_xticklabels(model_names, rotation=45, ha='right')
        ax1.grid(True, alpha=self.style_config['grid_alpha'])
        
        # Add value labels on bars
        for bar, r2 in zip(bars1, r_squared):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.001,
                    f'{r2:.3f}', ha='center', va='bottom', fontsize=8)
        
        # 2. AIC comparison (lower is better)
        bars2 = ax2.bar(x_pos, aic, color=colors[1], alpha=0.7, edgecolor='black')
        ax2.set_xlabel('Models')
        ax2.set_ylabel('AIC')
        ax2.set_title('AIC Comparison (Lower is Better)')
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels(model_names, rotation=45, ha='right')
        ax2.grid(True, alpha=self.style_config['grid_alpha'])
        
        # Highlight best AIC
        if len(aic) > 0:
            best_aic_idx = np.argmin(aic)
            bars2[best_aic_idx].set_color(colors[3])
        
        # 3. BIC comparison (lower is better)
        bars3 = ax3.bar(x_pos, bic, color=colors[2], alpha=0.7, edgecolor='black')
        ax3.set_xlabel('Models')
        ax3.set_ylabel('BIC')
        ax3.set_title('BIC Comparison (Lower is Better)')
        ax3.set_xticks(x_pos)
        ax3.set_xticklabels(model_names, rotation=45, ha='right')
        ax3.grid(True, alpha=self.style_config['grid_alpha'])
        
        # Highlight best BIC
        if len(bic) > 0:
            best_bic_idx = np.argmin(bic)
            bars3[best_bic_idx].set_color(colors[3])
        
        # 4. Log-likelihood comparison (higher is better)
        bars4 = ax4.bar(x_pos, log_likelihood, color=colors[4 % len(colors)], 
                       alpha=0.7, edgecolor='black')
        ax4.set_xlabel('Models')
        ax4.set_ylabel('Log-Likelihood')
        ax4.set_title('Log-Likelihood Comparison (Higher is Better)')
        ax4.set_xticks(x_pos)
        ax4.set_xticklabels(model_names, rotation=45, ha='right')
        ax4.grid(True, alpha=self.style_config['grid_alpha'])
        
        # Highlight best log-likelihood
        if len(log_likelihood) > 0:
            best_ll_idx = np.argmax(log_likelihood)
            bars4[best_ll_idx].set_color(colors[3])
        
        fig.suptitle(title, fontsize=self.style_config['font_size_title'])
        plt.tight_layout()
        
        if save_path:
            self._save_figure(fig, save_path)
        
        return fig
    
    def threshold_stability_visualization(self, 
                                        stability_results: Dict[str, Any],
                                        save_path: Optional[str] = None,
                                        title: str = "Threshold Stability Analysis") -> plt.Figure:
        """
        Create threshold stability visualization showing confidence intervals.
        
        Parameters:
        -----------
        stability_results : Dict[str, Any]
            Dictionary containing stability analysis results with keys:
            - 'threshold_estimates': array of threshold estimates across samples
            - 'confidence_intervals': array of confidence intervals
            - 'sample_periods': array of sample period labels
            - 'coefficients': array of coefficient estimates across samples
        save_path : Optional[str]
            Path to save the figure
        title : str
            Figure title
            
        Returns:
        --------
        plt.Figure
            The created stability visualization
        """
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=self.style_config['figure_size_double'])
        
        colors = self.style_config['color_palette']
        
        # Extract stability data
        threshold_estimates = stability_results.get('threshold_estimates', [])
        confidence_intervals = stability_results.get('confidence_intervals', [])
        sample_periods = stability_results.get('sample_periods', [])
        coefficients = stability_results.get('coefficients', [])
        
        if len(threshold_estimates) == 0:
            ax1.text(0.5, 0.5, 'No stability data available', 
                    ha='center', va='center', transform=ax1.transAxes)
            return fig
        
        x_pos = np.arange(len(threshold_estimates))
        
        # 1. Threshold stability over time/samples
        ax1.plot(x_pos, threshold_estimates, 'o-', color=colors[0], 
                linewidth=self.style_config['line_width'],
                markersize=self.style_config['marker_size'])
        
        # Add confidence intervals if available
        if len(confidence_intervals) > 0 and len(confidence_intervals) == len(threshold_estimates):
            ci_lower = [ci[0] if isinstance(ci, (list, tuple)) else ci - 0.1 
                       for ci in confidence_intervals]
            ci_upper = [ci[1] if isinstance(ci, (list, tuple)) else ci + 0.1 
                       for ci in confidence_intervals]
            
            ax1.fill_between(x_pos, ci_lower, ci_upper, 
                           color=colors[0], alpha=self.style_config['confidence_alpha'],
                           label='95% Confidence Interval')
            ax1.legend()
        
        ax1.set_xlabel('Sample Period')
        ax1.set_ylabel('Threshold Estimate')
        ax1.set_title('Threshold Stability Across Samples')
        ax1.grid(True, alpha=self.style_config['grid_alpha'])
        
        # Set x-axis labels if sample periods provided
        if len(sample_periods) == len(threshold_estimates):
            ax1.set_xticks(x_pos)
            ax1.set_xticklabels(sample_periods, rotation=45, ha='right')
        
        # 2. Coefficient stability
        if len(coefficients) > 0:
            # If coefficients is a 2D array (multiple coefficients per sample)
            if isinstance(coefficients[0], (list, tuple, np.ndarray)):
                n_coefs = len(coefficients[0])
                for i in range(min(n_coefs, len(colors))):
                    coef_series = [coef[i] for coef in coefficients]
                    ax2.plot(x_pos, coef_series, 'o-', 
                            color=colors[i], 
                            linewidth=self.style_config['line_width'],
                            markersize=self.style_config['marker_size'],
                            label=f'Coefficient {i+1}')
                ax2.legend()
            else:
                # Single coefficient series
                ax2.plot(x_pos, coefficients, 'o-', color=colors[1], 
                        linewidth=self.style_config['line_width'],
                        markersize=self.style_config['marker_size'])
        
        ax2.set_xlabel('Sample Period')
        ax2.set_ylabel('Coefficient Estimate')
        ax2.set_title('Coefficient Stability Across Samples')
        ax2.grid(True, alpha=self.style_config['grid_alpha'])
        ax2.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        
        # Set x-axis labels if sample periods provided
        if len(sample_periods) == len(threshold_estimates):
            ax2.set_xticks(x_pos)
            ax2.set_xticklabels(sample_periods, rotation=45, ha='right')
        
        fig.suptitle(title, fontsize=self.style_config['font_size_title'])
        plt.tight_layout()
        
        if save_path:
            self._save_figure(fig, save_path)
        
        return fig
    
    def regime_analysis_visualization(self, 
                                    regime_results: Dict[str, Any],
                                    save_path: Optional[str] = None,
                                    title: str = "Regime-Specific Analysis") -> plt.Figure:
        """
        Create regime analysis visualization for regime-specific diagnostics.
        
        Parameters:
        -----------
        regime_results : Dict[str, Any]
            Dictionary containing regime analysis results with keys:
            - 'regime_1': dict with 'observations', 'coefficients', 'residuals'
            - 'regime_2': dict with 'observations', 'coefficients', 'residuals'
            - 'threshold_variable': array of threshold variable values
            - 'regime_indicator': array indicating regime membership
        save_path : Optional[str]
            Path to save the figure
        title : str
            Figure title
            
        Returns:
        --------
        plt.Figure
            The created regime analysis visualization
        """
        fig = plt.figure(figsize=self.style_config['figure_size_double'])
        gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
        
        colors = self.style_config['color_palette']
        
        # Extract regime data
        regime_1 = regime_results.get('regime_1', {})
        regime_2 = regime_results.get('regime_2', {})
        threshold_var = regime_results.get('threshold_variable', [])
        regime_indicator = regime_results.get('regime_indicator', [])
        
        # 1. Regime distribution
        ax1 = fig.add_subplot(gs[0, 0])
        
        if len(regime_indicator) > 0:
            regime_counts = [np.sum(regime_indicator == 0), np.sum(regime_indicator == 1)]
            regime_labels = ['Regime 1', 'Regime 2']
            
            bars = ax1.bar(regime_labels, regime_counts, 
                          color=[colors[0], colors[1]], alpha=0.7, edgecolor='black')
            
            # Add percentage labels
            total_obs = sum(regime_counts)
            for bar, count in zip(bars, regime_counts):
                height = bar.get_height()
                percentage = (count / total_obs) * 100
                ax1.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                        f'{count}\n({percentage:.1f}%)', 
                        ha='center', va='bottom')
        
        ax1.set_ylabel('Number of Observations')
        ax1.set_title('Regime Distribution')
        ax1.grid(True, alpha=self.style_config['grid_alpha'])
        
        # 2. Threshold variable distribution by regime
        ax2 = fig.add_subplot(gs[0, 1])
        
        if len(threshold_var) > 0 and len(regime_indicator) > 0:
            regime_1_data = threshold_var[regime_indicator == 0]
            regime_2_data = threshold_var[regime_indicator == 1]
            
            ax2.hist(regime_1_data, bins=20, alpha=0.6, color=colors[0], 
                    label='Regime 1', density=True)
            ax2.hist(regime_2_data, bins=20, alpha=0.6, color=colors[1], 
                    label='Regime 2', density=True)
            ax2.legend()
        
        ax2.set_xlabel('Threshold Variable')
        ax2.set_ylabel('Density')
        ax2.set_title('Threshold Variable by Regime')
        ax2.grid(True, alpha=self.style_config['grid_alpha'])
        
        # 3. Regime-specific coefficient comparison
        ax3 = fig.add_subplot(gs[1, 0])
        
        regime_1_coefs = regime_1.get('coefficients', [])
        regime_2_coefs = regime_2.get('coefficients', [])
        
        if len(regime_1_coefs) > 0 and len(regime_2_coefs) > 0:
            # Assume coefficients are arrays of coefficient values
            n_coefs = min(len(regime_1_coefs), len(regime_2_coefs))
            x_pos = np.arange(n_coefs)
            width = 0.35
            
            bars1 = ax3.bar(x_pos - width/2, regime_1_coefs[:n_coefs], width,
                           color=colors[0], alpha=0.7, label='Regime 1')
            bars2 = ax3.bar(x_pos + width/2, regime_2_coefs[:n_coefs], width,
                           color=colors[1], alpha=0.7, label='Regime 2')
            
            ax3.set_xlabel('Coefficient Index')
            ax3.set_ylabel('Coefficient Value')
            ax3.set_title('Regime-Specific Coefficients')
            ax3.set_xticks(x_pos)
            ax3.legend()
            ax3.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        
        ax3.grid(True, alpha=self.style_config['grid_alpha'])
        
        # 4. Regime-specific residual analysis
        ax4 = fig.add_subplot(gs[1, 1])
        
        regime_1_residuals = regime_1.get('residuals', [])
        regime_2_residuals = regime_2.get('residuals', [])
        
        if len(regime_1_residuals) > 0 and len(regime_2_residuals) > 0:
            # Box plot of residuals by regime
            residual_data = [regime_1_residuals, regime_2_residuals]
            box_plot = ax4.boxplot(residual_data, tick_labels=['Regime 1', 'Regime 2'],
                                  patch_artist=True)
            
            # Color the boxes
            for patch, color in zip(box_plot['boxes'], [colors[0], colors[1]]):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)
        
        ax4.set_ylabel('Residuals')
        ax4.set_title('Residual Distribution by Regime')
        ax4.grid(True, alpha=self.style_config['grid_alpha'])
        ax4.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        
        fig.suptitle(title, fontsize=self.style_config['font_size_title'])
        
        if save_path:
            self._save_figure(fig, save_path)
        
        return fig    

    def create_robustness_visualization(self, 
                                      robustness_results: Dict[str, Any],
                                      save_path: Optional[str] = None,
                                      title: str = "Robustness Analysis") -> plt.Figure:
        """
        Create robustness visualization for coefficient stability plots.
        
        Parameters:
        -----------
        robustness_results : Dict[str, Any]
            Dictionary containing robustness analysis results with keys:
            - 'specifications': list of specification names
            - 'coefficients': 2D array of coefficients [spec x coef]
            - 'confidence_intervals': 3D array of CIs [spec x coef x 2]
            - 'coefficient_names': list of coefficient names
        save_path : Optional[str]
            Path to save the figure
        title : str
            Figure title
            
        Returns:
        --------
        plt.Figure
            The created robustness visualization
        """
        fig, ax = self.publication_figure_template(double_column=True)
        
        colors = self.style_config['color_palette']
        
        # Extract robustness data
        specifications = robustness_results.get('specifications', [])
        coefficients = robustness_results.get('coefficients', [])
        confidence_intervals = robustness_results.get('confidence_intervals', [])
        coefficient_names = robustness_results.get('coefficient_names', [])
        
        if len(specifications) == 0 or len(coefficients) == 0:
            ax.text(0.5, 0.5, 'No robustness data available', 
                   ha='center', va='center', transform=ax.transAxes)
            return fig
        
        # Convert to numpy arrays for easier handling
        coefficients = np.array(coefficients)
        if len(confidence_intervals) > 0:
            confidence_intervals = np.array(confidence_intervals)
        
        n_specs = len(specifications)
        n_coefs = coefficients.shape[1] if coefficients.ndim > 1 else 1
        
        # Create coefficient stability plot
        x_pos = np.arange(n_specs)
        
        # Plot each coefficient across specifications
        for i in range(min(n_coefs, len(colors))):
            if coefficients.ndim > 1:
                coef_values = coefficients[:, i]
            else:
                coef_values = coefficients
                
            color = colors[i % len(colors)]
            label = coefficient_names[i] if i < len(coefficient_names) else f'Coefficient {i+1}'
            
            # Plot coefficient line
            ax.plot(x_pos, coef_values, 'o-', color=color, 
                   linewidth=self.style_config['line_width'],
                   markersize=self.style_config['marker_size'],
                   label=label)
            
            # Add confidence intervals if available
            if len(confidence_intervals) > 0 and confidence_intervals.shape[0] == n_specs:
                if confidence_intervals.ndim == 3 and i < confidence_intervals.shape[1]:
                    ci_lower = confidence_intervals[:, i, 0]
                    ci_upper = confidence_intervals[:, i, 1]
                    
                    ax.fill_between(x_pos, ci_lower, ci_upper, 
                                   color=color, alpha=self.style_config['confidence_alpha'])
        
        # Customize plot
        ax.set_xlabel('Model Specification')
        ax.set_ylabel('Coefficient Estimate')
        ax.set_title(title)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(specifications, rotation=45, ha='right')
        ax.grid(True, alpha=self.style_config['grid_alpha'])
        ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        
        if n_coefs > 1:
            ax.legend()
        
        plt.tight_layout()
        
        if save_path:
            self._save_figure(fig, save_path)
        
        return fig
    
    def sensitivity_analysis_visualization(self, 
                                         sensitivity_results: Dict[str, Any],
                                         save_path: Optional[str] = None,
                                         title: str = "Sensitivity Analysis") -> plt.Figure:
        """
        Create sensitivity analysis visualization with tornado plots for key parameters.
        
        Parameters:
        -----------
        sensitivity_results : Dict[str, Any]
            Dictionary containing sensitivity analysis results with keys:
            - 'parameter_names': list of parameter names
            - 'low_values': array of results with low parameter values
            - 'high_values': array of results with high parameter values
            - 'baseline_value': baseline result value
            - 'parameter_ranges': dict with 'low' and 'high' parameter values
        save_path : Optional[str]
            Path to save the figure
        title : str
            Figure title
            
        Returns:
        --------
        plt.Figure
            The created tornado plot
        """
        fig, ax = self.publication_figure_template(double_column=True)
        
        colors = self.style_config['color_palette']
        
        # Extract sensitivity data
        parameter_names = sensitivity_results.get('parameter_names', [])
        low_values = sensitivity_results.get('low_values', [])
        high_values = sensitivity_results.get('high_values', [])
        baseline_value = sensitivity_results.get('baseline_value', 0)
        
        if len(parameter_names) == 0 or len(low_values) == 0 or len(high_values) == 0:
            ax.text(0.5, 0.5, 'No sensitivity data available', 
                   ha='center', va='center', transform=ax.transAxes)
            return fig
        
        # Calculate sensitivity ranges
        low_impact = np.array(low_values) - baseline_value
        high_impact = np.array(high_values) - baseline_value
        
        # Sort by total impact (absolute range)
        total_impact = np.abs(high_impact - low_impact)
        sort_indices = np.argsort(total_impact)[::-1]  # Descending order
        
        # Reorder data by impact
        parameter_names = [parameter_names[i] for i in sort_indices]
        low_impact = low_impact[sort_indices]
        high_impact = high_impact[sort_indices]
        
        y_pos = np.arange(len(parameter_names))
        
        # Create tornado plot
        for i, (low, high) in enumerate(zip(low_impact, high_impact)):
            # Determine which side is positive/negative
            left_val = min(low, high)
            right_val = max(low, high)
            
            # Plot bars
            if left_val < 0:
                ax.barh(i, left_val, color=colors[1], alpha=0.7, 
                       label='Negative Impact' if i == 0 else "")
            if right_val > 0:
                ax.barh(i, right_val, color=colors[0], alpha=0.7,
                       label='Positive Impact' if i == 0 else "")
            
            # Add value labels
            if abs(left_val) > abs(right_val) * 0.1:  # Only show if significant
                ax.text(left_val - 0.01 * abs(left_val), i, f'{left_val:.3f}', 
                       ha='right', va='center', fontsize=8)
            if abs(right_val) > abs(left_val) * 0.1:
                ax.text(right_val + 0.01 * abs(right_val), i, f'{right_val:.3f}', 
                       ha='left', va='center', fontsize=8)
        
        # Customize plot
        ax.set_yticks(y_pos)
        ax.set_yticklabels(parameter_names)
        ax.set_xlabel('Impact on Result (Change from Baseline)')
        ax.set_title(title)
        ax.axvline(x=0, color='black', linestyle='-', alpha=0.8)
        ax.grid(True, alpha=self.style_config['grid_alpha'], axis='x')
        
        # Add baseline reference
        ax.text(0.02, 0.98, f'Baseline: {baseline_value:.3f}', 
               transform=ax.transAxes, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Add legend if we have both positive and negative impacts
        if any(low_impact < 0) and any(high_impact > 0):
            ax.legend()
        
        plt.tight_layout()
        
        if save_path:
            self._save_figure(fig, save_path)
        
        return fig
    
    def subsample_stability_plots(self, 
                                subsample_results: Dict[str, Any],
                                save_path: Optional[str] = None,
                                title: str = "Subsample Stability Analysis") -> plt.Figure:
        """
        Create subsample stability plots for temporal robustness analysis.
        
        Parameters:
        -----------
        subsample_results : Dict[str, Any]
            Dictionary containing subsample analysis results with keys:
            - 'time_periods': list of time period labels
            - 'coefficients': 2D array of coefficients [period x coef]
            - 'confidence_intervals': 3D array of CIs [period x coef x 2]
            - 'coefficient_names': list of coefficient names
            - 'sample_sizes': array of sample sizes for each period
        save_path : Optional[str]
            Path to save the figure
        title : str
            Figure title
            
        Returns:
        --------
        plt.Figure
            The created subsample stability visualization
        """
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=self.style_config['figure_size_double'])
        
        colors = self.style_config['color_palette']
        
        # Extract subsample data
        time_periods = subsample_results.get('time_periods', [])
        coefficients = subsample_results.get('coefficients', [])
        confidence_intervals = subsample_results.get('confidence_intervals', [])
        coefficient_names = subsample_results.get('coefficient_names', [])
        sample_sizes = subsample_results.get('sample_sizes', [])
        
        if len(time_periods) == 0 or len(coefficients) == 0:
            ax1.text(0.5, 0.5, 'No subsample data available', 
                    ha='center', va='center', transform=ax1.transAxes)
            return fig
        
        # Convert to numpy arrays
        coefficients = np.array(coefficients)
        if len(confidence_intervals) > 0:
            confidence_intervals = np.array(confidence_intervals)
        
        x_pos = np.arange(len(time_periods))
        n_coefs = coefficients.shape[1] if coefficients.ndim > 1 else 1
        
        # Top panel: Coefficient evolution over time
        for i in range(min(n_coefs, len(colors))):
            if coefficients.ndim > 1:
                coef_values = coefficients[:, i]
            else:
                coef_values = coefficients
                
            color = colors[i % len(colors)]
            label = coefficient_names[i] if i < len(coefficient_names) else f'Coefficient {i+1}'
            
            # Plot coefficient evolution
            ax1.plot(x_pos, coef_values, 'o-', color=color, 
                    linewidth=self.style_config['line_width'],
                    markersize=self.style_config['marker_size'],
                    label=label)
            
            # Add confidence intervals if available
            if len(confidence_intervals) > 0 and confidence_intervals.shape[0] == len(time_periods):
                if confidence_intervals.ndim == 3 and i < confidence_intervals.shape[1]:
                    ci_lower = confidence_intervals[:, i, 0]
                    ci_upper = confidence_intervals[:, i, 1]
                    
                    ax1.fill_between(x_pos, ci_lower, ci_upper, 
                                    color=color, alpha=self.style_config['confidence_alpha'])
        
        ax1.set_xlabel('Time Period')
        ax1.set_ylabel('Coefficient Estimate')
        ax1.set_title('Coefficient Evolution Over Time')
        ax1.set_xticks(x_pos)
        ax1.set_xticklabels(time_periods, rotation=45, ha='right')
        ax1.grid(True, alpha=self.style_config['grid_alpha'])
        ax1.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        
        if n_coefs > 1:
            ax1.legend()
        
        # Bottom panel: Sample size evolution
        if len(sample_sizes) > 0:
            ax2.bar(x_pos, sample_sizes, color=colors[0], alpha=0.7, edgecolor='black')
            
            # Add value labels on bars
            for i, size in enumerate(sample_sizes):
                ax2.text(i, size + max(sample_sizes) * 0.01, str(size), 
                        ha='center', va='bottom', fontsize=8)
        
        ax2.set_xlabel('Time Period')
        ax2.set_ylabel('Sample Size')
        ax2.set_title('Sample Size by Period')
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels(time_periods, rotation=45, ha='right')
        ax2.grid(True, alpha=self.style_config['grid_alpha'])
        
        fig.suptitle(title, fontsize=self.style_config['font_size_title'])
        plt.tight_layout()
        
        if save_path:
            self._save_figure(fig, save_path)
        
        return fig
    
    def specification_comparison_heatmap(self, 
                                       comparison_results: Dict[str, Any],
                                       save_path: Optional[str] = None,
                                       title: str = "Model Performance Comparison") -> plt.Figure:
        """
        Create specification comparison heatmap for model performance comparison.
        
        Parameters:
        -----------
        comparison_results : Dict[str, Any]
            Dictionary containing comparison results with keys:
            - 'model_names': list of model names
            - 'metrics': list of metric names (e.g., ['R²', 'AIC', 'BIC'])
            - 'values': 2D array of metric values [model x metric]
            - 'higher_is_better': list of booleans indicating if higher values are better
        save_path : Optional[str]
            Path to save the figure
        title : str
            Figure title
            
        Returns:
        --------
        plt.Figure
            The created heatmap visualization
        """
        fig, ax = self.publication_figure_template(double_column=True)
        
        # Extract comparison data
        model_names = comparison_results.get('model_names', [])
        metrics = comparison_results.get('metrics', [])
        values = comparison_results.get('values', [])
        higher_is_better = comparison_results.get('higher_is_better', [])
        
        if len(model_names) == 0 or len(metrics) == 0 or len(values) == 0:
            ax.text(0.5, 0.5, 'No comparison data available', 
                   ha='center', va='center', transform=ax.transAxes)
            return fig
        
        # Convert to numpy array and normalize for heatmap
        values = np.array(values)
        normalized_values = np.zeros_like(values)
        
        # Normalize each metric (column) to 0-1 scale
        for j in range(values.shape[1]):
            col_values = values[:, j]
            if len(higher_is_better) > j:
                if higher_is_better[j]:
                    # Higher is better: normalize so highest = 1
                    normalized_values[:, j] = (col_values - col_values.min()) / (col_values.max() - col_values.min())
                else:
                    # Lower is better: normalize so lowest = 1
                    normalized_values[:, j] = (col_values.max() - col_values) / (col_values.max() - col_values.min())
            else:
                # Default: higher is better
                normalized_values[:, j] = (col_values - col_values.min()) / (col_values.max() - col_values.min())
        
        # Create heatmap
        im = ax.imshow(normalized_values, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
        
        # Set ticks and labels
        ax.set_xticks(np.arange(len(metrics)))
        ax.set_yticks(np.arange(len(model_names)))
        ax.set_xticklabels(metrics)
        ax.set_yticklabels(model_names)
        
        # Rotate the tick labels and set their alignment
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        
        # Add text annotations with actual values
        for i in range(len(model_names)):
            for j in range(len(metrics)):
                text = ax.text(j, i, f'{values[i, j]:.3f}', 
                             ha="center", va="center", color="black", fontsize=8)
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_label('Normalized Performance (1 = Best)', rotation=270, labelpad=15)
        
        ax.set_title(title)
        plt.tight_layout()
        
        if save_path:
            self._save_figure(fig, save_path)
        
        return fig