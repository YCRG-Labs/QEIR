"""
Data Visualization Utilities for QE Hypothesis Testing

This module provides utility functions for creating quick data visualizations
and exploratory plots during the analysis process.

Author: Kiro AI Assistant
Date: 2025-09-02
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any

class DataVisualizer:
    """
    Utility class for quick data visualization and exploration
    """
    
    def __init__(self):
        """Initialize the data visualizer with default settings"""
        plt.style.use('seaborn-v0_8-whitegrid')
        sns.set_palette("husl")
    
    def quick_time_series(self, data_dict: Dict[str, pd.Series], 
                         title: str = "Time Series Plot",
                         figsize: tuple = (12, 8)):
        """
        Create a quick time series plot of multiple variables
        
        Args:
            data_dict: Dictionary of pandas Series to plot
            title: Plot title
            figsize: Figure size tuple
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        for name, series in data_dict.items():
            if series is not None and not series.empty:
                ax.plot(series.index, series.values, label=name, linewidth=2, alpha=0.8)
        
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel('Date')
        ax.set_ylabel('Value')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig, ax
    
    def correlation_heatmap(self, data_dict: Dict[str, pd.Series],
                           title: str = "Correlation Matrix",
                           figsize: tuple = (10, 8)):
        """
        Create a correlation heatmap of multiple variables
        
        Args:
            data_dict: Dictionary of pandas Series
            title: Plot title
            figsize: Figure size tuple
        """
        # Align data to common dates
        valid_data = {k: v for k, v in data_dict.items() if v is not None and not v.empty}
        
        if len(valid_data) < 2:
            print("Need at least 2 valid series for correlation analysis")
            return None, None
        
        # Find common dates
        common_dates = None
        for series in valid_data.values():
            if common_dates is None:
                common_dates = series.index
            else:
                common_dates = common_dates.intersection(series.index)
        
        if len(common_dates) < 10:
            print("Insufficient overlapping data for correlation analysis")
            return None, None
        
        # Create aligned dataframe
        aligned_data = {}
        for name, series in valid_data.items():
            aligned_data[name] = series.loc[common_dates]
        
        df = pd.DataFrame(aligned_data)
        correlation_matrix = df.corr()
        
        # Create heatmap
        fig, ax = plt.subplots(figsize=figsize)
        sns.heatmap(correlation_matrix, annot=True, cmap='RdBu_r', center=0,
                   square=True, fmt='.3f', cbar_kws={'label': 'Correlation'})
        ax.set_title(title, fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        return fig, ax
    
    def scatter_with_trend(self, x_series: pd.Series, y_series: pd.Series,
                          x_label: str = "X Variable", y_label: str = "Y Variable",
                          title: str = "Scatter Plot with Trend",
                          figsize: tuple = (10, 6)):
        """
        Create a scatter plot with trend line
        
        Args:
            x_series: X-axis data
            y_series: Y-axis data
            x_label: X-axis label
            y_label: Y-axis label
            title: Plot title
            figsize: Figure size tuple
        """
        # Align data
        common_dates = x_series.index.intersection(y_series.index)
        if len(common_dates) < 10:
            print("Insufficient overlapping data for scatter plot")
            return None, None
        
        x_aligned = x_series.loc[common_dates]
        y_aligned = y_series.loc[common_dates]
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # Scatter plot
        ax.scatter(x_aligned, y_aligned, alpha=0.6, s=50)
        
        # Trend line
        z = np.polyfit(x_aligned, y_aligned, 1)
        p = np.poly1d(z)
        ax.plot(x_aligned, p(x_aligned), "r--", alpha=0.8, linewidth=2)
        
        # Calculate correlation
        correlation = np.corrcoef(x_aligned, y_aligned)[0, 1]
        
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.set_title(f"{title}\nCorrelation: {correlation:.3f}")
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig, ax
    
    def distribution_comparison(self, data_dict: Dict[str, pd.Series],
                              title: str = "Distribution Comparison",
                              figsize: tuple = (12, 8)):
        """
        Create distribution comparison plots
        
        Args:
            data_dict: Dictionary of pandas Series
            title: Plot title
            figsize: Figure size tuple
        """
        valid_data = {k: v.dropna() for k, v in data_dict.items() 
                     if v is not None and not v.empty}
        
        if not valid_data:
            print("No valid data for distribution comparison")
            return None, None
        
        n_vars = len(valid_data)
        fig, axes = plt.subplots(2, n_vars, figsize=figsize)
        
        if n_vars == 1:
            axes = axes.reshape(2, 1)
        
        for i, (name, series) in enumerate(valid_data.items()):
            # Histogram
            axes[0, i].hist(series, bins=30, alpha=0.7, edgecolor='black')
            axes[0, i].set_title(f'{name} - Histogram')
            axes[0, i].set_ylabel('Frequency')
            axes[0, i].grid(True, alpha=0.3)
            
            # Box plot
            axes[1, i].boxplot(series)
            axes[1, i].set_title(f'{name} - Box Plot')
            axes[1, i].set_ylabel('Value')
            axes[1, i].grid(True, alpha=0.3)
        
        plt.suptitle(title, fontsize=16, fontweight='bold')
        plt.tight_layout()
        return fig, axes


def quick_data_overview(data, save_path: Optional[str] = None):
    """
    Create a quick overview of hypothesis data
    
    Args:
        data: HypothesisData object
        save_path: Optional path to save the figure
    """
    visualizer = DataVisualizer()
    
    # Collect all available data
    all_data = {}
    
    if data.central_bank_reaction is not None:
        all_data['Central Bank Reaction'] = data.central_bank_reaction
    if data.confidence_effects is not None:
        all_data['Confidence Effects'] = data.confidence_effects
    if data.debt_service_burden is not None:
        all_data['Debt Service Burden'] = data.debt_service_burden
    if data.long_term_yields is not None:
        all_data['Long-term Yields'] = data.long_term_yields
    if data.qe_intensity is not None:
        all_data['QE Intensity'] = data.qe_intensity
    if data.private_investment is not None:
        all_data['Private Investment'] = data.private_investment
    if data.market_distortions is not None:
        all_data['Market Distortions'] = data.market_distortions
    if data.exchange_rate is not None:
        all_data['Exchange Rate'] = data.exchange_rate
    if data.inflation_measures is not None:
        all_data['Inflation Measures'] = data.inflation_measures
    if data.capital_flows is not None:
        all_data['Capital Flows'] = data.capital_flows
    
    if not all_data:
        print("No data available for visualization")
        return None
    
    # Create time series plot
    fig, ax = visualizer.quick_time_series(all_data, "QE Hypothesis Data Overview")
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Data overview saved to: {save_path}")
    
    return fig, ax