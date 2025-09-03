"""
Hypothesis-Specific Plotting Functions

This module contains specialized plotting functions for each of the three
QE hypotheses, focusing on the specific econometric methods and results.

Author: Kiro AI Assistant
Date: 2025-09-02
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple

class HypothesisPlotter:
    """
    Specialized plotting functions for QE hypothesis testing results
    """
    
    def __init__(self):
        """Initialize with publication-quality settings"""
        plt.style.use('seaborn-v0_8-whitegrid')
        sns.set_palette("husl")
        
        self.colors = {
            'primary': '#2E86AB',
            'secondary': '#A23B72', 
            'accent': '#F18F01',
            'neutral': '#C73E1D',
            'success': '#28a745',
            'warning': '#ffc107',
            'danger': '#dc3545'
        }
    
    def plot_threshold_regression(self, threshold_var: pd.Series, 
                                dependent_var: pd.Series,
                                threshold_value: float,
                                title: str = "Threshold Regression Results",
                                figsize: tuple = (12, 8)):
        """
        Plot threshold regression results for Hypothesis 1
        
        Args:
            threshold_var: Threshold variable (debt service burden)
            dependent_var: Dependent variable (long-term yields)
            threshold_value: Estimated threshold value
            title: Plot title
            figsize: Figure size
        """
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        
        # Align data
        common_dates = threshold_var.index.intersection(dependent_var.index)
        if len(common_dates) < 10:
            print("Insufficient data for threshold regression plot")
            return None, None
        
        threshold_aligned = threshold_var.loc[common_dates]
        dependent_aligned = dependent_var.loc[common_dates]
        
        # Plot 1: Scatter plot with threshold line
        regime1_mask = threshold_aligned <= threshold_value
        regime2_mask = threshold_aligned > threshold_value
        
        axes[0, 0].scatter(threshold_aligned[regime1_mask], dependent_aligned[regime1_mask], 
                          alpha=0.7, s=50, color=self.colors['primary'], label='Regime 1')
        axes[0, 0].scatter(threshold_aligned[regime2_mask], dependent_aligned[regime2_mask], 
                          alpha=0.7, s=50, color=self.colors['secondary'], label='Regime 2')
        axes[0, 0].axvline(threshold_value, color='red', linestyle='--', linewidth=2, 
                          label=f'Threshold = {threshold_value:.3f}')
        axes[0, 0].set_xlabel('Threshold Variable')
        axes[0, 0].set_ylabel('Dependent Variable')
        axes[0, 0].set_title('Regime Classification')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Time series with regime shading
        axes[0, 1].plot(common_dates, dependent_aligned, 'k-', linewidth=2, alpha=0.8)
        
        for i, date in enumerate(common_dates):
            color = self.colors['primary'] if regime1_mask.iloc[i] else self.colors['secondary']
            axes[0, 1].axvspan(date, date, alpha=0.1, color=color)
        
        axes[0, 1].set_xlabel('Date')
        axes[0, 1].set_ylabel('Dependent Variable')
        axes[0, 1].set_title('Time Series with Regime Classification')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: Threshold variable over time
        axes[1, 0].plot(common_dates, threshold_aligned, color=self.colors['accent'], linewidth=2)
        axes[1, 0].axhline(threshold_value, color='red', linestyle='--', linewidth=2)
        axes[1, 0].fill_between(common_dates, threshold_aligned.min(), threshold_value,
                               alpha=0.3, color=self.colors['primary'], label='Regime 1')
        axes[1, 0].fill_between(common_dates, threshold_value, threshold_aligned.max(),
                               alpha=0.3, color=self.colors['secondary'], label='Regime 2')
        axes[1, 0].set_xlabel('Date')
        axes[1, 0].set_ylabel('Threshold Variable')
        axes[1, 0].set_title('Threshold Variable Evolution')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Plot 4: Regime statistics
        regime1_stats = {
            'Observations': regime1_mask.sum(),
            'Mean Dependent': dependent_aligned[regime1_mask].mean(),
            'Std Dependent': dependent_aligned[regime1_mask].std()
        }
        
        regime2_stats = {
            'Observations': regime2_mask.sum(),
            'Mean Dependent': dependent_aligned[regime2_mask].mean(),
            'Std Dependent': dependent_aligned[regime2_mask].std()
        }
        
        x_pos = np.arange(len(regime1_stats))
        width = 0.35
        
        axes[1, 1].bar(x_pos - width/2, list(regime1_stats.values()), width, 
                      label='Regime 1', color=self.colors['primary'], alpha=0.7)
        axes[1, 1].bar(x_pos + width/2, list(regime2_stats.values()), width,
                      label='Regime 2', color=self.colors['secondary'], alpha=0.7)
        
        axes[1, 1].set_xlabel('Statistics')
        axes[1, 1].set_ylabel('Value')
        axes[1, 1].set_title('Regime Comparison')
        axes[1, 1].set_xticks(x_pos)
        axes[1, 1].set_xticklabels(list(regime1_stats.keys()), rotation=45)
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3, axis='y')
        
        plt.suptitle(title, fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        return fig, axes
    
    def plot_impulse_responses(self, horizons: np.ndarray,
                              responses: Dict[str, np.ndarray],
                              confidence_bands: Optional[Dict[str, Tuple[np.ndarray, np.ndarray]]] = None,
                              title: str = "Impulse Response Functions",
                              figsize: tuple = (15, 10)):
        """
        Plot impulse response functions for Hypothesis 2
        
        Args:
            horizons: Array of time horizons
            responses: Dictionary of response series
            confidence_bands: Optional confidence bands
            title: Plot title
            figsize: Figure size
        """
        n_responses = len(responses)
        cols = 2
        rows = (n_responses + 1) // 2
        
        fig, axes = plt.subplots(rows, cols, figsize=figsize)
        if n_responses == 1:
            axes = [axes]
        elif rows == 1:
            axes = axes.reshape(1, -1)
        
        colors = [self.colors['primary'], self.colors['secondary'], 
                 self.colors['accent'], self.colors['neutral']]
        
        for i, (response_name, response_values) in enumerate(responses.items()):
            row, col = i // cols, i % cols
            ax = axes[row, col] if rows > 1 else axes[col]
            
            # Plot main response
            ax.plot(horizons, response_values, color=colors[i % len(colors)], 
                   linewidth=3, marker='o', markersize=4, label=response_name)
            
            # Add confidence bands if available
            if confidence_bands and response_name in confidence_bands:
                lower, upper = confidence_bands[response_name]
                ax.fill_between(horizons, lower, upper, alpha=0.3, 
                              color=colors[i % len(colors)])
            
            # Add zero line
            ax.axhline(0, color='black', linestyle='--', alpha=0.5)
            
            ax.set_xlabel('Quarters')
            ax.set_ylabel('Response (%)')
            ax.set_title(f'{response_name} Response to QE Shock')
            ax.grid(True, alpha=0.3)
            ax.legend()
        
        # Hide empty subplots
        for i in range(n_responses, rows * cols):
            row, col = i // cols, i % cols
            ax = axes[row, col] if rows > 1 else axes[col]
            ax.axis('off')
        
        plt.suptitle(title, fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        return fig, axes
    
    def plot_international_spillovers(self, data_dict: Dict[str, pd.Series],
                                    correlation_matrix: Optional[pd.DataFrame] = None,
                                    title: str = "International Spillover Analysis",
                                    figsize: tuple = (16, 12)):
        """
        Plot international spillover analysis for Hypothesis 3
        
        Args:
            data_dict: Dictionary of international variables
            correlation_matrix: Correlation matrix of variables
            title: Plot title
            figsize: Figure size
        """
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        
        # Plot 1: Time series of international variables
        for i, (name, series) in enumerate(data_dict.items()):
            if series is not None and not series.empty:
                color = plt.cm.Set3(i / len(data_dict))
                axes[0, 0].plot(series.index, series.values, label=name, 
                              linewidth=2, alpha=0.8, color=color)
        
        axes[0, 0].set_xlabel('Date')
        axes[0, 0].set_ylabel('Value')
        axes[0, 0].set_title('International Variables Over Time')
        axes[0, 0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Correlation heatmap
        if correlation_matrix is not None:
            im = axes[0, 1].imshow(correlation_matrix.values, cmap='RdBu_r', 
                                 vmin=-1, vmax=1, aspect='auto')
            axes[0, 1].set_xticks(range(len(correlation_matrix.columns)))
            axes[0, 1].set_yticks(range(len(correlation_matrix.index)))
            axes[0, 1].set_xticklabels(correlation_matrix.columns, rotation=45)
            axes[0, 1].set_yticklabels(correlation_matrix.index)
            axes[0, 1].set_title('Cross-Country Correlations')
            
            # Add correlation values
            for i in range(len(correlation_matrix.index)):
                for j in range(len(correlation_matrix.columns)):
                    text = axes[0, 1].text(j, i, f'{correlation_matrix.iloc[i, j]:.2f}',
                                         ha="center", va="center", color="black", fontweight='bold')
            
            plt.colorbar(im, ax=axes[0, 1], shrink=0.8)
        
        # Plot 3: Exchange rate volatility (spillover intensity proxy)
        if 'exchange_rate' in data_dict and data_dict['exchange_rate'] is not None:
            exchange_rate = data_dict['exchange_rate']
            volatility = exchange_rate.rolling(window=12).std()
            volatility = volatility.dropna()
            
            if len(volatility) > 0:
                axes[1, 0].plot(volatility.index, volatility.values, 
                              color=self.colors['accent'], linewidth=2)
                axes[1, 0].set_xlabel('Date')
                axes[1, 0].set_ylabel('Volatility')
                axes[1, 0].set_title('Exchange Rate Volatility (Spillover Intensity)')
                axes[1, 0].grid(True, alpha=0.3)
        
        # Plot 4: Capital flow dynamics
        if 'capital_flows' in data_dict and data_dict['capital_flows'] is not None:
            capital_flows = data_dict['capital_flows']
            flow_changes = capital_flows.pct_change() * 100
            flow_changes = flow_changes.dropna()
            
            if len(flow_changes) > 0:
                axes[1, 1].plot(flow_changes.index, flow_changes.values, 
                              color=self.colors['neutral'], linewidth=2, alpha=0.7)
                axes[1, 1].axhline(0, color='black', linestyle='--', alpha=0.5)
                axes[1, 1].set_xlabel('Date')
                axes[1, 1].set_ylabel('Change (%)')
                axes[1, 1].set_title('Capital Flow Dynamics')
                axes[1, 1].grid(True, alpha=0.3)
        
        plt.suptitle(title, fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        return fig, axes
    
    def plot_model_diagnostics(self, results_dict: Dict[str, Any],
                             title: str = "Model Diagnostics",
                             figsize: tuple = (14, 10)):
        """
        Plot comprehensive model diagnostics
        
        Args:
            results_dict: Dictionary containing model results and diagnostics
            title: Plot title
            figsize: Figure size
        """
        fig, axes = plt.subplots(2, 3, figsize=figsize)
        
        # Plot 1: Model fit statistics
        if 'fit_statistics' in results_dict:
            fit_stats = results_dict['fit_statistics']
            bars = axes[0, 0].bar(fit_stats.keys(), fit_stats.values(), 
                                color=self.colors['primary'], alpha=0.7)
            axes[0, 0].set_ylabel('R-squared')
            axes[0, 0].set_title('Model Fit Statistics')
            axes[0, 0].set_ylim(0, 1)
            axes[0, 0].tick_params(axis='x', rotation=45)
            axes[0, 0].grid(True, alpha=0.3, axis='y')
            
            # Add value labels
            for bar, value in zip(bars, fit_stats.values()):
                height = bar.get_height()
                axes[0, 0].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                               f'{value:.3f}', ha='center', va='bottom')
        
        # Plot 2: Residual analysis
        if 'residuals' in results_dict:
            residuals = results_dict['residuals']
            axes[0, 1].hist(residuals, bins=30, alpha=0.7, color=self.colors['secondary'],
                          edgecolor='black')
            axes[0, 1].set_xlabel('Residuals')
            axes[0, 1].set_ylabel('Frequency')
            axes[0, 1].set_title('Residual Distribution')
            axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: Parameter stability
        if 'parameter_evolution' in results_dict:
            param_evolution = results_dict['parameter_evolution']
            for param_name, param_values in param_evolution.items():
                axes[0, 2].plot(param_values, label=param_name, linewidth=2)
            axes[0, 2].set_xlabel('Time Period')
            axes[0, 2].set_ylabel('Parameter Value')
            axes[0, 2].set_title('Parameter Stability')
            axes[0, 2].legend()
            axes[0, 2].grid(True, alpha=0.3)
        
        # Plot 4: Forecast accuracy
        if 'forecast_errors' in results_dict:
            forecast_errors = results_dict['forecast_errors']
            axes[1, 0].plot(forecast_errors, color=self.colors['accent'], linewidth=2)
            axes[1, 0].axhline(0, color='black', linestyle='--', alpha=0.5)
            axes[1, 0].set_xlabel('Forecast Horizon')
            axes[1, 0].set_ylabel('Forecast Error')
            axes[1, 0].set_title('Forecast Accuracy')
            axes[1, 0].grid(True, alpha=0.3)
        
        # Plot 5: Statistical tests
        if 'statistical_tests' in results_dict:
            test_results = results_dict['statistical_tests']
            test_names = list(test_results.keys())
            test_pvalues = [test_results[name].get('p_value', 1) for name in test_names]
            
            colors = ['green' if p < 0.05 else 'red' for p in test_pvalues]
            bars = axes[1, 1].bar(test_names, test_pvalues, color=colors, alpha=0.7)
            axes[1, 1].axhline(0.05, color='red', linestyle='--', alpha=0.8, 
                             label='5% Significance Level')
            axes[1, 1].set_ylabel('P-value')
            axes[1, 1].set_title('Statistical Test Results')
            axes[1, 1].tick_params(axis='x', rotation=45)
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3, axis='y')
        
        # Plot 6: Data quality summary
        if 'data_quality' in results_dict:
            quality_metrics = results_dict['data_quality']
            quality_names = list(quality_metrics.keys())
            quality_scores = list(quality_metrics.values())
            
            bars = axes[1, 2].barh(quality_names, quality_scores, 
                                 color=self.colors['neutral'], alpha=0.7)
            axes[1, 2].set_xlabel('Quality Score')
            axes[1, 2].set_title('Data Quality Assessment')
            axes[1, 2].set_xlim(0, 100)
            axes[1, 2].grid(True, alpha=0.3, axis='x')
        
        plt.suptitle(title, fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        return fig, axes