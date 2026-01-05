"""
Publication Output Generator

This module provides utilities for generating publication-ready LaTeX tables
and figures for the revised QE methodology analysis.

Author: QE Research Team
Date: 2025
Version: 1.0
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
import warnings


class PublicationOutputGenerator:
    """
    Generator for publication-ready tables and figures.
    
    This class provides methods to format estimation results into LaTeX tables
    and generate publication-quality figures for the revised QE methodology paper.
    
    Attributes:
        output_dir: Directory for saving output files
        figure_format: Format for saving figures (default: 'pdf')
        table_format: Format for saving tables (default: 'tex')
    """
    
    def __init__(
        self,
        output_dir: str = "output",
        figure_format: str = "pdf",
        table_format: str = "tex"
    ):
        """
        Initialize the publication output generator.
        
        Args:
            output_dir: Directory path for saving outputs
            figure_format: Format for figures ('pdf', 'png', 'svg')
            table_format: Format for tables ('tex', 'csv', 'xlsx')
        """
        self.output_dir = Path(output_dir)
        self.figure_format = figure_format
        self.table_format = table_format
        
        # Create output directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "tables").mkdir(exist_ok=True)
        (self.output_dir / "figures").mkdir(exist_ok=True)
        
        # Set matplotlib style for publication quality
        plt.style.use('seaborn-v0_8-paper')
        plt.rcParams['font.family'] = 'serif'
        plt.rcParams['font.size'] = 10
        plt.rcParams['axes.labelsize'] = 10
        plt.rcParams['axes.titlesize'] = 11
        plt.rcParams['xtick.labelsize'] = 9
        plt.rcParams['ytick.labelsize'] = 9
        plt.rcParams['legend.fontsize'] = 9
        plt.rcParams['figure.titlesize'] = 11
    
    # =========================================================================
    # Formatting Utilities
    # =========================================================================
    
    def format_number(
        self,
        value: float,
        decimals: int = 3,
        percentage: bool = False
    ) -> str:
        """
        Format a number for publication display.
        
        Args:
            value: Number to format
            decimals: Number of decimal places
            percentage: Whether to display as percentage
            
        Returns:
            Formatted string
        """
        if pd.isna(value):
            return "—"
        
        if percentage:
            return f"{value * 100:.{decimals}f}"
        else:
            return f"{value:.{decimals}f}"
    
    def format_standard_error(
        self,
        se: float,
        decimals: int = 3
    ) -> str:
        """
        Format standard error in parentheses.
        
        Args:
            se: Standard error value
            decimals: Number of decimal places
            
        Returns:
            Formatted string like "(0.123)"
        """
        if pd.isna(se):
            return "—"
        return f"({se:.{decimals}f})"
    
    def add_significance_stars(
        self,
        pvalue: float
    ) -> str:
        """
        Add significance stars based on p-value.
        
        Args:
            pvalue: P-value from statistical test
            
        Returns:
            String with stars: *** (p<0.01), ** (p<0.05), * (p<0.10)
        """
        if pd.isna(pvalue):
            return ""
        
        if pvalue < 0.01:
            return "***"
        elif pvalue < 0.05:
            return "**"
        elif pvalue < 0.10:
            return "*"
        else:
            return ""
    
    def format_coefficient_row(
        self,
        coef: float,
        se: float,
        pvalue: float,
        decimals: int = 3
    ) -> Tuple[str, str]:
        """
        Format coefficient and standard error for table display.
        
        Args:
            coef: Coefficient estimate
            se: Standard error
            pvalue: P-value
            decimals: Number of decimal places
            
        Returns:
            Tuple of (coefficient_string, se_string)
        """
        coef_str = self.format_number(coef, decimals) + self.add_significance_stars(pvalue)
        se_str = self.format_standard_error(se, decimals)
        return coef_str, se_str
    
    def create_latex_table_header(
        self,
        caption: str,
        label: str,
        columns: List[str],
        alignment: Optional[str] = None
    ) -> str:
        """
        Create LaTeX table header.
        
        Args:
            caption: Table caption
            label: LaTeX label for referencing
            columns: List of column names
            alignment: Column alignment string (e.g., 'lcccc')
            
        Returns:
            LaTeX header string
        """
        if alignment is None:
            alignment = 'l' + 'c' * (len(columns) - 1)
        
        header = "\\begin{table}[htbp]\n"
        header += "\\centering\n"
        header += f"\\caption{{{caption}}}\n"
        header += f"\\label{{{label}}}\n"
        header += f"\\begin{{tabular}}{{{alignment}}}\n"
        header += "\\hline\\hline\n"
        header += " & ".join(columns) + " \\\\\n"
        header += "\\hline\n"
        
        return header
    
    def create_latex_table_footer(
        self,
        notes: Optional[str] = None
    ) -> str:
        """
        Create LaTeX table footer.
        
        Args:
            notes: Optional table notes
            
        Returns:
            LaTeX footer string
        """
        footer = "\\hline\\hline\n"
        footer += "\\end{tabular}\n"
        
        if notes:
            footer += f"\\begin{{tablenotes}}\n"
            footer += f"\\small\n"
            footer += f"\\item {notes}\n"
            footer += f"\\end{{tablenotes}}\n"
        
        footer += "\\end{table}\n"
        
        return footer
    
    def save_table(
        self,
        content: str,
        filename: str
    ) -> Path:
        """
        Save table to file.
        
        Args:
            content: Table content (LaTeX or other format)
            filename: Output filename
            
        Returns:
            Path to saved file
        """
        filepath = self.output_dir / "tables" / filename
        with open(filepath, 'w') as f:
            f.write(content)
        return filepath
    
    def save_figure(
        self,
        fig: plt.Figure,
        filename: str,
        dpi: int = 300
    ) -> Path:
        """
        Save figure to file.
        
        Args:
            fig: Matplotlib figure object
            filename: Output filename (without extension)
            dpi: Resolution for raster formats
            
        Returns:
            Path to saved file
        """
        filepath = self.output_dir / "figures" / f"{filename}.{self.figure_format}"
        fig.savefig(filepath, format=self.figure_format, dpi=dpi, bbox_inches='tight')
        plt.close(fig)
        return filepath

    # =========================================================================
    # Main Results Tables
    # =========================================================================
    
    def generate_table2_threshold_regression(
        self,
        results: Dict[str, Any],
        filename: str = "table2_threshold_regression.tex"
    ) -> Path:
        """
        Generate Table 2: Threshold Regression Estimates.
        
        This table presents the threshold estimate and regime-specific
        parameter estimates from the Hansen threshold regression model.
        
        Args:
            results: Dictionary containing threshold regression results with keys:
                - 'threshold_estimate': Estimated threshold value
                - 'threshold_ci_lower': Lower bound of 95% CI
                - 'threshold_ci_upper': Upper bound of 95% CI
                - 'regime1_qe_effect': QE effect in low-fiscal regime
                - 'regime1_qe_se': Standard error
                - 'regime1_qe_pvalue': P-value
                - 'regime2_qe_effect': QE effect in high-fiscal regime
                - 'regime2_qe_se': Standard error
                - 'regime2_qe_pvalue': P-value
                - 'first_stage_f_stat': First-stage F-statistic
                - 'observations_total': Total observations
                - 'regime1_observations': Observations in regime 1
                - 'regime2_observations': Observations in regime 2
            filename: Output filename
            
        Returns:
            Path to saved table file
        """
        # Extract results
        threshold = results['threshold_estimate']
        ci_lower = results['threshold_ci_lower']
        ci_upper = results['threshold_ci_upper']
        
        # Format coefficient rows
        regime1_coef, regime1_se = self.format_coefficient_row(
            results['regime1_qe_effect'],
            results['regime1_qe_se'],
            results['regime1_qe_pvalue']
        )
        
        regime2_coef, regime2_se = self.format_coefficient_row(
            results['regime2_qe_effect'],
            results['regime2_qe_se'],
            results['regime2_qe_pvalue']
        )
        
        # Create table
        caption = "Threshold Regression Estimates: Fiscal Regime Effects on QE Transmission"
        label = "tab:threshold_regression"
        columns = ["", "Low Fiscal Stress", "High Fiscal Stress"]
        
        content = self.create_latex_table_header(caption, label, columns, "lcc")
        
        # Add threshold information
        content += f"\\multicolumn{{3}}{{l}}{{\\textit{{Threshold Estimate}}}} \\\\\n"
        content += f"Debt-to-GDP Threshold ($\\tau$) & \\multicolumn{{2}}{{c}}{{{self.format_number(threshold, 3)}}} \\\\\n"
        content += f"95\\% Confidence Interval & \\multicolumn{{2}}{{c}}{{[{self.format_number(ci_lower, 3)}, {self.format_number(ci_upper, 3)}]}} \\\\\n"
        content += "\\hline\n"
        
        # Add regime-specific effects
        content += f"\\multicolumn{{3}}{{l}}{{\\textit{{QE Effects on 10Y Yields (basis points)}}}} \\\\\n"
        content += f"QE Shock Coefficient & {regime1_coef} & {regime2_coef} \\\\\n"
        content += f" & {regime1_se} & {regime2_se} \\\\\n"
        content += "\\hline\n"
        
        # Add diagnostics
        content += f"\\multicolumn{{3}}{{l}}{{\\textit{{Diagnostics}}}} \\\\\n"
        content += f"First-Stage F-Statistic & \\multicolumn{{2}}{{c}}{{{self.format_number(results['first_stage_f_stat'], 2)}}} \\\\\n"
        content += f"Observations & {results['regime1_observations']} & {results['regime2_observations']} \\\\\n"
        content += f"Total Observations & \\multicolumn{{2}}{{c}}{{{results['observations_total']}}} \\\\\n"
        
        notes = "Notes: Standard errors in parentheses. *** p<0.01, ** p<0.05, * p<0.10. " \
                "QE shocks are instrumented using high-frequency FOMC surprises. " \
                "Low fiscal stress regime: Debt-to-GDP $\\leq \\tau$. " \
                "High fiscal stress regime: Debt-to-GDP $> \\tau$."
        
        content += self.create_latex_table_footer(notes)
        
        return self.save_table(content, filename)
    
    def generate_table3_regime_effects(
        self,
        results: Dict[str, Any],
        filename: str = "table3_regime_effects.tex"
    ) -> Path:
        """
        Generate Table 3: Regime-Specific QE Effects.
        
        This table presents detailed regime-specific effects including
        attenuation calculations and statistical tests.
        
        Args:
            results: Dictionary containing regime effect results with keys:
                - 'regime1_qe_effect': QE effect in low-fiscal regime (bps)
                - 'regime1_qe_se': Standard error
                - 'regime1_qe_pvalue': P-value
                - 'regime2_qe_effect': QE effect in high-fiscal regime (bps)
                - 'regime2_qe_se': Standard error
                - 'regime2_qe_pvalue': P-value
                - 'attenuation_pct': Percentage attenuation
                - 'attenuation_significant': Whether difference is significant
                - 'regime1_r_squared': R-squared for regime 1
                - 'regime2_r_squared': R-squared for regime 2
            filename: Output filename
            
        Returns:
            Path to saved table file
        """
        # Format coefficient rows
        regime1_coef, regime1_se = self.format_coefficient_row(
            results['regime1_qe_effect'],
            results['regime1_qe_se'],
            results['regime1_qe_pvalue']
        )
        
        regime2_coef, regime2_se = self.format_coefficient_row(
            results['regime2_qe_effect'],
            results['regime2_qe_se'],
            results['regime2_qe_pvalue']
        )
        
        # Calculate attenuation
        attenuation = results['attenuation_pct']
        attenuation_sig = "***" if results.get('attenuation_significant', False) else ""
        
        # Create table
        caption = "Regime-Specific QE Effects and Attenuation"
        label = "tab:regime_effects"
        columns = ["", "Low Fiscal Stress", "High Fiscal Stress", "Attenuation"]
        
        content = self.create_latex_table_header(caption, label, columns, "lccc")
        
        # Add QE effects
        content += f"\\multicolumn{{4}}{{l}}{{\\textit{{QE Effects on 10Y Yields}}}} \\\\\n"
        content += f"Effect (basis points) & {regime1_coef} & {regime2_coef} & \\\\\n"
        content += f" & {regime1_se} & {regime2_se} & \\\\\n"
        content += "\\hline\n"
        
        # Add attenuation
        content += f"\\multicolumn{{4}}{{l}}{{\\textit{{Attenuation Analysis}}}} \\\\\n"
        content += f"Percentage Reduction & & & {self.format_number(attenuation, 1)}\\%{attenuation_sig} \\\\\n"
        content += f"Formula & \\multicolumn{{3}}{{l}}{{$(\\beta_1 - \\beta_2) / \\beta_1 \\times 100$}} \\\\\n"
        content += "\\hline\n"
        
        # Add model fit
        content += f"\\multicolumn{{4}}{{l}}{{\\textit{{Model Fit}}}} \\\\\n"
        content += f"R-squared & {self.format_number(results['regime1_r_squared'], 3)} & {self.format_number(results['regime2_r_squared'], 3)} & \\\\\n"
        
        notes = "Notes: Standard errors in parentheses. *** p<0.01, ** p<0.05, * p<0.10. " \
                "Attenuation measures the percentage reduction in QE effectiveness " \
                "when moving from low to high fiscal stress regime."
        
        content += self.create_latex_table_footer(notes)
        
        return self.save_table(content, filename)
    
    def generate_table4_channel_decomposition(
        self,
        results: Dict[str, Any],
        filename: str = "table4_channel_decomposition.tex"
    ) -> Path:
        """
        Generate Table 4: Channel Decomposition Results.
        
        This table presents the structural decomposition of QE transmission
        into interest rate and market distortion channels.
        
        Args:
            results: Dictionary containing channel decomposition results with keys:
                - 'rate_channel_beta': Effect of QE on interest rates
                - 'rate_channel_se': Standard error
                - 'rate_channel_pvalue': P-value
                - 'distortion_channel_beta': Effect of QE on distortions
                - 'distortion_channel_se': Standard error
                - 'distortion_channel_pvalue': P-value
                - 'distortion_share': Share of total effect (0-1)
                - 'rate_share': Share of total effect (0-1)
                - 'cumulative_effect_12q': Cumulative investment effect
                - 'meets_target_shares': Whether shares meet targets
            filename: Output filename
            
        Returns:
            Path to saved table file
        """
        # Format coefficient rows
        rate_coef, rate_se = self.format_coefficient_row(
            results['rate_channel_beta'],
            results['rate_channel_se'],
            results['rate_channel_pvalue']
        )
        
        distortion_coef, distortion_se = self.format_coefficient_row(
            results['distortion_channel_beta'],
            results['distortion_channel_se'],
            results['distortion_channel_pvalue']
        )
        
        # Create table
        caption = "Structural Channel Decomposition of QE Transmission"
        label = "tab:channel_decomposition"
        columns = ["Channel", "QE Effect", "Share of Total"]
        
        content = self.create_latex_table_header(caption, label, columns, "lcc")
        
        # Add channel effects
        content += f"\\multicolumn{{3}}{{l}}{{\\textit{{Step 1: QE Effects on Channels}}}} \\\\\n"
        content += f"Interest Rate Channel ($\\beta_r$) & {rate_coef} & \\\\\n"
        content += f" & {rate_se} & \\\\\n"
        content += f"Market Distortion Channel ($\\beta_D$) & {distortion_coef} & \\\\\n"
        content += f" & {distortion_se} & \\\\\n"
        content += "\\hline\n"
        
        # Add channel shares
        content += f"\\multicolumn{{3}}{{l}}{{\\textit{{Step 2: Channel Shares}}}} \\\\\n"
        content += f"Interest Rate Channel & & {self.format_number(results['rate_share'], 1, percentage=True)}\\% \\\\\n"
        content += f"Market Distortion Channel & & {self.format_number(results['distortion_share'], 1, percentage=True)}\\% \\\\\n"
        content += "\\hline\n"
        
        # Add cumulative effect
        content += f"\\multicolumn{{3}}{{l}}{{\\textit{{Investment Response}}}} \\\\\n"
        content += f"Cumulative Effect (12 quarters) & {self.format_number(results['cumulative_effect_12q'], 2)} pp & \\\\\n"
        
        target_met = "Yes" if results.get('meets_target_shares', False) else "No"
        content += f"Meets Target Shares (65\\%/35\\%) & {target_met} & \\\\\n"
        
        notes = "Notes: Standard errors in parentheses. *** p<0.01, ** p<0.05, * p<0.10. " \
                "Channel shares calculated as cumulative contribution over 12 quarters. " \
                "Target shares: 65% distortion channel, 35% interest rate channel (±5%)."
        
        content += self.create_latex_table_footer(notes)
        
        return self.save_table(content, filename)

    # =========================================================================
    # Robustness Tables
    # =========================================================================
    
    def generate_table5_double_threshold(
        self,
        results: Dict[str, Any],
        filename: str = "table5_double_threshold.tex"
    ) -> Path:
        """
        Generate Table 5: Double-Threshold Model Results.
        
        This table presents results from the three-regime threshold model
        as a robustness check on the baseline two-regime specification.
        
        Args:
            results: Dictionary containing double-threshold results with keys:
                - 'threshold1_estimate': First threshold
                - 'threshold2_estimate': Second threshold
                - 'regime1_qe_effect': QE effect in low regime
                - 'regime1_qe_se': Standard error
                - 'regime1_qe_pvalue': P-value
                - 'regime2_qe_effect': QE effect in medium regime
                - 'regime2_qe_se': Standard error
                - 'regime2_qe_pvalue': P-value
                - 'regime3_qe_effect': QE effect in high regime
                - 'regime3_qe_se': Standard error
                - 'regime3_qe_pvalue': P-value
                - 'regime1_observations': Observations in regime 1
                - 'regime2_observations': Observations in regime 2
                - 'regime3_observations': Observations in regime 3
            filename: Output filename
            
        Returns:
            Path to saved table file
        """
        # Format coefficient rows
        regime1_coef, regime1_se = self.format_coefficient_row(
            results['regime1_qe_effect'],
            results['regime1_qe_se'],
            results['regime1_qe_pvalue']
        )
        
        regime2_coef, regime2_se = self.format_coefficient_row(
            results['regime2_qe_effect'],
            results['regime2_qe_se'],
            results['regime2_qe_pvalue']
        )
        
        regime3_coef, regime3_se = self.format_coefficient_row(
            results['regime3_qe_effect'],
            results['regime3_qe_se'],
            results['regime3_qe_pvalue']
        )
        
        # Create table
        caption = "Double-Threshold Model: Three-Regime Specification"
        label = "tab:double_threshold"
        columns = ["", "Low Regime", "Medium Regime", "High Regime"]
        
        content = self.create_latex_table_header(caption, label, columns, "lccc")
        
        # Add threshold information
        content += f"\\multicolumn{{4}}{{l}}{{\\textit{{Threshold Estimates}}}} \\\\\n"
        content += f"First Threshold ($\\tau_1$) & \\multicolumn{{3}}{{l}}{{{self.format_number(results['threshold1_estimate'], 3)}}} \\\\\n"
        content += f"Second Threshold ($\\tau_2$) & \\multicolumn{{3}}{{l}}{{{self.format_number(results['threshold2_estimate'], 3)}}} \\\\\n"
        content += "\\hline\n"
        
        # Add regime-specific effects
        content += f"\\multicolumn{{4}}{{l}}{{\\textit{{QE Effects (basis points)}}}} \\\\\n"
        content += f"QE Shock Coefficient & {regime1_coef} & {regime2_coef} & {regime3_coef} \\\\\n"
        content += f" & {regime1_se} & {regime2_se} & {regime3_se} \\\\\n"
        content += "\\hline\n"
        
        # Add observations
        content += f"Observations & {results['regime1_observations']} & {results['regime2_observations']} & {results['regime3_observations']} \\\\\n"
        
        notes = "Notes: Standard errors in parentheses. *** p<0.01, ** p<0.05, * p<0.10. " \
                "Three-regime model allows for non-linear effects across fiscal stress levels. " \
                "Low regime: Debt-to-GDP $\\leq \\tau_1$. " \
                "Medium regime: $\\tau_1 <$ Debt-to-GDP $\\leq \\tau_2$. " \
                "High regime: Debt-to-GDP $> \\tau_2$."
        
        content += self.create_latex_table_footer(notes)
        
        return self.save_table(content, filename)
    
    def generate_table6_alternative_fiscal(
        self,
        results: Dict[str, List[Dict[str, Any]]],
        filename: str = "table6_alternative_fiscal.tex"
    ) -> Path:
        """
        Generate Table 6: Alternative Fiscal Indicators.
        
        This table presents threshold regression results using different
        fiscal stress indicators as robustness checks.
        
        Args:
            results: Dictionary with keys for each fiscal indicator:
                - 'debt_to_gdp': Results using debt-to-GDP (baseline)
                - 'gross_debt': Results using gross debt-to-GDP
                - 'primary_deficit': Results using primary deficit-to-GDP
                - 'r_minus_g': Results using r-g differential
                - 'cbo_fiscal_gap': Results using CBO fiscal gap
                Each containing:
                    - 'threshold_estimate': Threshold value
                    - 'regime1_qe_effect': Low regime effect
                    - 'regime1_qe_se': Standard error
                    - 'regime2_qe_effect': High regime effect
                    - 'regime2_qe_se': Standard error
                    - 'attenuation_pct': Attenuation percentage
            filename: Output filename
            
        Returns:
            Path to saved table file
        """
        indicators = [
            ('debt_to_gdp', 'Debt-to-GDP (Baseline)'),
            ('gross_debt', 'Gross Debt-to-GDP'),
            ('primary_deficit', 'Primary Deficit-to-GDP'),
            ('r_minus_g', 'r-g Differential'),
            ('cbo_fiscal_gap', 'CBO Fiscal Gap')
        ]
        
        # Create table
        caption = "Robustness to Alternative Fiscal Stress Indicators"
        label = "tab:alternative_fiscal"
        columns = ["Fiscal Indicator", "Threshold", "Low Regime", "High Regime", "Attenuation"]
        
        content = self.create_latex_table_header(caption, label, columns, "lcccc")
        
        for key, name in indicators:
            if key not in results:
                continue
                
            res = results[key]
            
            # Format coefficients
            regime1_coef, regime1_se = self.format_coefficient_row(
                res['regime1_qe_effect'],
                res['regime1_qe_se'],
                res.get('regime1_qe_pvalue', 0.05)
            )
            
            regime2_coef, regime2_se = self.format_coefficient_row(
                res['regime2_qe_effect'],
                res['regime2_qe_se'],
                res.get('regime2_qe_pvalue', 0.05)
            )
            
            threshold = self.format_number(res['threshold_estimate'], 3)
            attenuation = self.format_number(res['attenuation_pct'], 1) + "\\%"
            
            content += f"{name} & {threshold} & {regime1_coef} & {regime2_coef} & {attenuation} \\\\\n"
            content += f" & & {regime1_se} & {regime2_se} & \\\\\n"
        
        notes = "Notes: Standard errors in parentheses. *** p<0.01, ** p<0.05, * p<0.10. " \
                "Each row presents results from threshold regression using a different fiscal stress indicator. " \
                "Attenuation measures percentage reduction in QE effectiveness across regimes."
        
        content += self.create_latex_table_footer(notes)
        
        return self.save_table(content, filename)
    
    def generate_table7_alternative_distortion(
        self,
        results: Dict[str, Dict[str, Any]],
        filename: str = "table7_alternative_distortion.tex"
    ) -> Path:
        """
        Generate Table 7: Alternative Distortion Measures.
        
        This table presents channel decomposition results using different
        market distortion measures as robustness checks.
        
        Args:
            results: Dictionary with keys for each distortion measure:
                - 'baseline': Three-component index (baseline)
                - 'fails_to_deliver': Using fails-to-deliver
                - 'repo_specialness': Using repo specialness
                - 'clearing_volumes': Using clearing volumes
                - 'dealer_capital': Using dealer capital ratios
                Each containing:
                    - 'distortion_share': Share of total effect
                    - 'rate_share': Share of total effect
                    - 'cumulative_effect_12q': Cumulative investment effect
            filename: Output filename
            
        Returns:
            Path to saved table file
        """
        measures = [
            ('baseline', 'Three-Component Index (Baseline)'),
            ('fails_to_deliver', 'Fails-to-Deliver'),
            ('repo_specialness', 'Repo Specialness'),
            ('clearing_volumes', 'Clearing Volumes'),
            ('dealer_capital', 'Dealer Capital Ratios')
        ]
        
        # Create table
        caption = "Robustness to Alternative Market Distortion Measures"
        label = "tab:alternative_distortion"
        columns = ["Distortion Measure", "Distortion Share", "Rate Share", "Cumulative Effect"]
        
        content = self.create_latex_table_header(caption, label, columns, "lccc")
        
        for key, name in measures:
            if key not in results:
                continue
                
            res = results[key]
            
            distortion_share = self.format_number(res['distortion_share'], 1, percentage=True) + "\\%"
            rate_share = self.format_number(res['rate_share'], 1, percentage=True) + "\\%"
            cumulative = self.format_number(res['cumulative_effect_12q'], 2) + " pp"
            
            content += f"{name} & {distortion_share} & {rate_share} & {cumulative} \\\\\n"
        
        notes = "Notes: Channel shares calculated as cumulative contribution over 12 quarters. " \
                "Each row uses a different measure of market distortions in the structural decomposition. " \
                "Target shares: 65% distortion channel, 35% interest rate channel."
        
        content += self.create_latex_table_footer(notes)
        
        return self.save_table(content, filename)
    
    def generate_table8_shadow_rate(
        self,
        results: Dict[str, Dict[str, Any]],
        filename: str = "table8_shadow_rate.tex"
    ) -> Path:
        """
        Generate Table 8: Shadow Rate Conditioning.
        
        This table presents results split by Wu-Xia shadow rate levels
        to examine heterogeneity across monetary policy regimes.
        
        Args:
            results: Dictionary with keys for each shadow rate regime:
                - 'full_sample': Full sample results
                - 'shadow_rate_low': Results when shadow rate < -2%
                - 'shadow_rate_medium': Results when -2% ≤ shadow rate < 0%
                - 'shadow_rate_high': Results when shadow rate ≥ 0%
                Each containing:
                    - 'regime1_qe_effect': Low fiscal regime effect
                    - 'regime1_qe_se': Standard error
                    - 'regime2_qe_effect': High fiscal regime effect
                    - 'regime2_qe_se': Standard error
                    - 'attenuation_pct': Attenuation percentage
                    - 'observations': Number of observations
            filename: Output filename
            
        Returns:
            Path to saved table file
        """
        regimes = [
            ('full_sample', 'Full Sample'),
            ('shadow_rate_low', 'Shadow Rate < -2\\%'),
            ('shadow_rate_medium', 'Shadow Rate [-2\\%, 0\\%)'),
            ('shadow_rate_high', 'Shadow Rate $\\geq$ 0\\%')
        ]
        
        # Create table
        caption = "QE Effects Conditional on Wu-Xia Shadow Rate"
        label = "tab:shadow_rate"
        columns = ["Sample", "Low Fiscal", "High Fiscal", "Attenuation", "N"]
        
        content = self.create_latex_table_header(caption, label, columns, "lcccc")
        
        for key, name in regimes:
            if key not in results:
                continue
                
            res = results[key]
            
            # Format coefficients
            regime1_coef, regime1_se = self.format_coefficient_row(
                res['regime1_qe_effect'],
                res['regime1_qe_se'],
                res.get('regime1_qe_pvalue', 0.05)
            )
            
            regime2_coef, regime2_se = self.format_coefficient_row(
                res['regime2_qe_effect'],
                res['regime2_qe_se'],
                res.get('regime2_qe_pvalue', 0.05)
            )
            
            attenuation = self.format_number(res['attenuation_pct'], 1) + "\\%"
            n_obs = str(res['observations'])
            
            content += f"{name} & {regime1_coef} & {regime2_coef} & {attenuation} & {n_obs} \\\\\n"
            content += f" & {regime1_se} & {regime2_se} & & \\\\\n"
        
        notes = "Notes: Standard errors in parentheses. *** p<0.01, ** p<0.05, * p<0.10. " \
                "Sample split by Wu-Xia shadow rate to examine heterogeneity across monetary policy regimes. " \
                "Shadow rate accounts for unconventional policy when nominal rates hit zero lower bound."
        
        content += self.create_latex_table_footer(notes)
        
        return self.save_table(content, filename)
    
    def generate_table9_data_sources(
        self,
        data_dict: Dict[str, Dict[str, str]],
        filename: str = "table9_data_sources.tex"
    ) -> Path:
        """
        Generate Table 9: Data Sources and Definitions.
        
        This table documents all variables used in the analysis with
        their definitions and data sources.
        
        Args:
            data_dict: Dictionary mapping variable names to info dicts with keys:
                - 'description': Variable description
                - 'source': Data source
                - 'frequency': Data frequency
                - 'sample_period': Sample period
            filename: Output filename
            
        Returns:
            Path to saved table file
        """
        # Create table
        caption = "Data Sources and Variable Definitions"
        label = "tab:data_sources"
        columns = ["Variable", "Description", "Source", "Frequency"]
        
        content = self.create_latex_table_header(caption, label, columns, "llll")
        
        # Sort variables by category
        categories = {
            'Dependent Variables': [],
            'Treatment Variables': [],
            'Threshold Variables': [],
            'Channel Variables': [],
            'Control Variables': [],
            'Instruments': []
        }
        
        for var_name, var_info in data_dict.items():
            category = var_info.get('category', 'Other')
            if category in categories:
                categories[category].append((var_name, var_info))
        
        # Add variables by category
        for category, variables in categories.items():
            if not variables:
                continue
                
            content += f"\\multicolumn{{4}}{{l}}{{\\textit{{{category}}}}} \\\\\n"
            
            for var_name, var_info in variables:
                desc = var_info['description']
                source = var_info['source']
                freq = var_info['frequency']
                
                # Escape special LaTeX characters
                desc = desc.replace('&', '\\&').replace('%', '\\%')
                source = source.replace('&', '\\&')
                
                content += f"{var_name} & {desc} & {source} & {freq} \\\\\n"
            
            content += "\\hline\n"
        
        notes = "Notes: All variables converted to quarterly frequency for analysis. " \
                "Sample period: 2008Q1-2023Q4 (64 quarters). " \
                "FRED = Federal Reserve Economic Data. " \
                "FRBNY = Federal Reserve Bank of New York. " \
                "CME = Chicago Mercantile Exchange."
        
        content += self.create_latex_table_footer(notes)
        
        return self.save_table(content, filename)

    # =========================================================================
    # Main Figures
    # =========================================================================
    
    def generate_figure1_smooth_transition(
        self,
        threshold_grid: np.ndarray,
        transition_function: np.ndarray,
        threshold_estimate: float,
        filename: str = "figure1_smooth_transition"
    ) -> Path:
        """
        Generate Figure 1: Smooth Transition Function.
        
        This figure plots the transition function showing how QE effects
        change smoothly across fiscal stress levels.
        
        Args:
            threshold_grid: Array of threshold values
            transition_function: Array of transition probabilities
            threshold_estimate: Estimated threshold value
            filename: Output filename (without extension)
            
        Returns:
            Path to saved figure file
        """
        fig, ax = plt.subplots(figsize=(8, 5))
        
        # Plot transition function
        ax.plot(threshold_grid, transition_function, 'b-', linewidth=2, label='Transition Function')
        
        # Add vertical line at threshold estimate
        ax.axvline(threshold_estimate, color='r', linestyle='--', linewidth=1.5, 
                   label=f'Threshold Estimate: {threshold_estimate:.3f}')
        
        # Add horizontal reference lines
        ax.axhline(0, color='gray', linestyle='-', linewidth=0.5, alpha=0.5)
        ax.axhline(1, color='gray', linestyle='-', linewidth=0.5, alpha=0.5)
        
        # Labels and formatting
        ax.set_xlabel('Debt-to-GDP Ratio', fontsize=11)
        ax.set_ylabel('Regime Probability', fontsize=11)
        ax.set_title('Smooth Transition Function Across Fiscal Regimes', fontsize=12)
        ax.legend(loc='best', frameon=True)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return self.save_figure(fig, filename)
    
    def generate_figure2_qe_episodes(
        self,
        episodes: Dict[str, Dict[str, Any]],
        filename: str = "figure2_qe_episodes"
    ) -> Path:
        """
        Generate Figure 2: Investment Response by QE Episode.
        
        This figure shows impulse response functions for investment
        separately for each QE episode (QE1, QE2, QE3, COVID-QE).
        
        Args:
            episodes: Dictionary with keys for each episode containing:
                - 'horizons': Array of horizons (0 to H)
                - 'effects': Array of investment effects
                - 'ci_lower': Lower confidence interval
                - 'ci_upper': Upper confidence interval
            filename: Output filename (without extension)
            
        Returns:
            Path to saved figure file
        """
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        axes = axes.flatten()
        
        episode_names = ['QE1', 'QE2', 'QE3', 'COVID-QE']
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
        
        for idx, (episode_key, episode_name) in enumerate(zip(episodes.keys(), episode_names)):
            if episode_key not in episodes:
                continue
                
            ax = axes[idx]
            data = episodes[episode_key]
            
            horizons = data['horizons']
            effects = data['effects']
            ci_lower = data['ci_lower']
            ci_upper = data['ci_upper']
            
            # Plot impulse response
            ax.plot(horizons, effects, color=colors[idx], linewidth=2, label='Point Estimate')
            ax.fill_between(horizons, ci_lower, ci_upper, color=colors[idx], alpha=0.2, label='95% CI')
            
            # Add zero line
            ax.axhline(0, color='black', linestyle='-', linewidth=0.5)
            
            # Labels and formatting
            ax.set_xlabel('Quarters After Shock', fontsize=10)
            ax.set_ylabel('Investment Growth (pp)', fontsize=10)
            ax.set_title(f'{episode_name}', fontsize=11, fontweight='bold')
            ax.legend(loc='best', frameon=True, fontsize=8)
            ax.grid(True, alpha=0.3)
        
        plt.suptitle('Investment Response to QE Shocks by Episode', fontsize=13, fontweight='bold')
        plt.tight_layout()
        return self.save_figure(fig, filename)
    
    def generate_figure3_ssr_minimization(
        self,
        threshold_grid: np.ndarray,
        ssr_values: np.ndarray,
        threshold_estimate: float,
        confidence_interval: Tuple[float, float],
        filename: str = "figure3_ssr_minimization"
    ) -> Path:
        """
        Generate Figure 3: SSR Minimization for Threshold Identification.
        
        This figure shows the sum of squared residuals across the threshold
        grid, identifying the optimal threshold value.
        
        Args:
            threshold_grid: Array of candidate threshold values
            ssr_values: Array of SSR values for each threshold
            threshold_estimate: Estimated threshold (minimum SSR)
            confidence_interval: Tuple of (lower, upper) CI bounds
            filename: Output filename (without extension)
            
        Returns:
            Path to saved figure file
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot SSR curve
        ax.plot(threshold_grid, ssr_values, 'b-', linewidth=2, label='Sum of Squared Residuals')
        
        # Mark threshold estimate
        min_ssr = np.min(ssr_values)
        ax.plot(threshold_estimate, min_ssr, 'ro', markersize=10, 
                label=f'Threshold Estimate: {threshold_estimate:.3f}')
        
        # Add confidence interval shading
        ci_lower, ci_upper = confidence_interval
        ax.axvspan(ci_lower, ci_upper, alpha=0.2, color='red', 
                   label=f'95% CI: [{ci_lower:.3f}, {ci_upper:.3f}]')
        
        # Labels and formatting
        ax.set_xlabel('Debt-to-GDP Ratio', fontsize=11)
        ax.set_ylabel('Sum of Squared Residuals', fontsize=11)
        ax.set_title('Threshold Identification via SSR Minimization', fontsize=12)
        ax.legend(loc='best', frameon=True)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return self.save_figure(fig, filename)
    
    def generate_figure4_regime_scatter(
        self,
        data: Dict[str, Any],
        filename: str = "figure4_regime_scatter"
    ) -> Path:
        """
        Generate Figure 4: Regime-Split Scatter Plots.
        
        This figure shows scatter plots of yield changes vs QE shocks
        separately for low and high fiscal stress regimes.
        
        Args:
            data: Dictionary containing:
                - 'qe_shocks': Array of QE shock values
                - 'yield_changes': Array of yield changes
                - 'regime_indicator': Boolean array (True = high regime)
                - 'threshold_estimate': Threshold value
                - 'regime1_fitted': Fitted values for low regime
                - 'regime2_fitted': Fitted values for high regime
            filename: Output filename (without extension)
            
        Returns:
            Path to saved figure file
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        qe_shocks = data['qe_shocks']
        yield_changes = data['yield_changes']
        regime_indicator = data['regime_indicator']
        
        # Low fiscal stress regime (left panel)
        low_regime_mask = ~regime_indicator
        ax1.scatter(qe_shocks[low_regime_mask], yield_changes[low_regime_mask], 
                   alpha=0.6, s=50, color='blue', label='Observations')
        
        if 'regime1_fitted' in data:
            ax1.plot(qe_shocks[low_regime_mask], data['regime1_fitted'], 
                    'r-', linewidth=2, label='Fitted Line')
        
        ax1.axhline(0, color='black', linestyle='-', linewidth=0.5)
        ax1.axvline(0, color='black', linestyle='-', linewidth=0.5)
        ax1.set_xlabel('QE Shock', fontsize=11)
        ax1.set_ylabel('10Y Yield Change (bps)', fontsize=11)
        ax1.set_title(f'Low Fiscal Stress (Debt-to-GDP ≤ {data["threshold_estimate"]:.3f})', 
                     fontsize=11, fontweight='bold')
        ax1.legend(loc='best', frameon=True)
        ax1.grid(True, alpha=0.3)
        
        # High fiscal stress regime (right panel)
        high_regime_mask = regime_indicator
        ax2.scatter(qe_shocks[high_regime_mask], yield_changes[high_regime_mask], 
                   alpha=0.6, s=50, color='red', label='Observations')
        
        if 'regime2_fitted' in data:
            ax2.plot(qe_shocks[high_regime_mask], data['regime2_fitted'], 
                    'r-', linewidth=2, label='Fitted Line')
        
        ax2.axhline(0, color='black', linestyle='-', linewidth=0.5)
        ax2.axvline(0, color='black', linestyle='-', linewidth=0.5)
        ax2.set_xlabel('QE Shock', fontsize=11)
        ax2.set_ylabel('10Y Yield Change (bps)', fontsize=11)
        ax2.set_title(f'High Fiscal Stress (Debt-to-GDP > {data["threshold_estimate"]:.3f})', 
                     fontsize=11, fontweight='bold')
        ax2.legend(loc='best', frameon=True)
        ax2.grid(True, alpha=0.3)
        
        plt.suptitle('QE Effects on Yields Across Fiscal Regimes', fontsize=13, fontweight='bold')
        plt.tight_layout()
        return self.save_figure(fig, filename)
    
    def generate_figure5_local_projections(
        self,
        horizons: np.ndarray,
        effects: np.ndarray,
        ci_lower: np.ndarray,
        ci_upper: np.ndarray,
        filename: str = "figure5_local_projections"
    ) -> Path:
        """
        Generate Figure 5: Local Projection Impulse Responses.
        
        This figure shows the dynamic investment response to QE shocks
        estimated via local projections.
        
        Args:
            horizons: Array of horizons (0 to H quarters)
            effects: Array of investment effects at each horizon
            ci_lower: Lower bound of 95% confidence interval
            ci_upper: Upper bound of 95% confidence interval
            filename: Output filename (without extension)
            
        Returns:
            Path to saved figure file
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot impulse response
        ax.plot(horizons, effects, 'b-', linewidth=2.5, marker='o', markersize=6, 
                label='Point Estimate')
        ax.fill_between(horizons, ci_lower, ci_upper, alpha=0.2, color='blue', 
                        label='95% Confidence Interval')
        
        # Add zero line
        ax.axhline(0, color='black', linestyle='-', linewidth=0.8)
        
        # Labels and formatting
        ax.set_xlabel('Quarters After QE Shock', fontsize=11)
        ax.set_ylabel('Investment Growth (percentage points)', fontsize=11)
        ax.set_title('Dynamic Investment Response to QE Shocks', fontsize=12, fontweight='bold')
        ax.legend(loc='best', frameon=True)
        ax.grid(True, alpha=0.3)
        
        # Set x-axis to show integer quarters
        ax.set_xticks(horizons[::2])  # Show every other quarter for clarity
        
        plt.tight_layout()
        return self.save_figure(fig, filename)
    
    def generate_figure6_channel_decomposition(
        self,
        channel_data: Dict[str, Any],
        filename: str = "figure6_channel_decomposition"
    ) -> Path:
        """
        Generate Figure 6: Bar Chart Decomposition of Channels.
        
        This figure shows the decomposition of total QE effects into
        interest rate and market distortion channels.
        
        Args:
            channel_data: Dictionary containing:
                - 'horizons': Array of horizons
                - 'rate_contributions': Array of rate channel contributions
                - 'distortion_contributions': Array of distortion channel contributions
                - 'total_effects': Array of total effects
                - 'rate_share': Overall rate channel share (0-1)
                - 'distortion_share': Overall distortion channel share (0-1)
            filename: Output filename (without extension)
            
        Returns:
            Path to saved figure file
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        horizons = channel_data['horizons']
        rate_contrib = channel_data['rate_contributions']
        distortion_contrib = channel_data['distortion_contributions']
        
        # Left panel: Stacked bar chart by horizon
        width = 0.6
        ax1.bar(horizons, rate_contrib, width, label='Interest Rate Channel', 
                color='#1f77b4', alpha=0.8)
        ax1.bar(horizons, distortion_contrib, width, bottom=rate_contrib, 
                label='Market Distortion Channel', color='#ff7f0e', alpha=0.8)
        
        # Add zero line
        ax1.axhline(0, color='black', linestyle='-', linewidth=0.5)
        
        ax1.set_xlabel('Quarters After Shock', fontsize=11)
        ax1.set_ylabel('Investment Effect (pp)', fontsize=11)
        ax1.set_title('Channel Contributions by Horizon', fontsize=11, fontweight='bold')
        ax1.legend(loc='best', frameon=True)
        ax1.grid(True, alpha=0.3, axis='y')
        
        # Right panel: Pie chart of cumulative shares
        shares = [channel_data['rate_share'] * 100, channel_data['distortion_share'] * 100]
        labels = [f"Interest Rate\n{shares[0]:.1f}%", f"Market Distortion\n{shares[1]:.1f}%"]
        colors = ['#1f77b4', '#ff7f0e']
        
        ax2.pie(shares, labels=labels, colors=colors, autopct='', startangle=90,
                textprops={'fontsize': 11, 'weight': 'bold'})
        ax2.set_title('Cumulative Channel Shares', fontsize=11, fontweight='bold')
        
        plt.suptitle('Structural Decomposition of QE Transmission Channels', 
                    fontsize=13, fontweight='bold')
        plt.tight_layout()
        return self.save_figure(fig, filename)

    # =========================================================================
    # Appendix Figures
    # =========================================================================
    
    def generate_appendix_figures(
        self,
        appendix_data: Dict[str, Any],
        filename_prefix: str = "appendix"
    ) -> List[Path]:
        """
        Generate all appendix figures (A1-A6).
        
        This method generates supplementary diagnostic figures for the appendix.
        
        Args:
            appendix_data: Dictionary containing data for all appendix figures:
                - 'a1_ssr_plot': Data for SSR plot (alternative view)
                - 'a2_regime_scatters': Data for regime scatter plots (extended)
                - 'a3_confidence_amplification': Data for confidence effects
                - 'a4_full_irf': Data for full impulse response functions
                - 'a5_decomposition_bars': Data for detailed decomposition
                - 'a6_robustness_comparison': Data for robustness comparison
            filename_prefix: Prefix for appendix figure filenames
            
        Returns:
            List of paths to saved figure files
        """
        saved_paths = []
        
        # Figure A1: Alternative SSR Plot
        if 'a1_ssr_plot' in appendix_data:
            path = self._generate_appendix_a1_ssr(
                appendix_data['a1_ssr_plot'],
                f"{filename_prefix}_a1_ssr_plot"
            )
            saved_paths.append(path)
        
        # Figure A2: Extended Regime Scatters
        if 'a2_regime_scatters' in appendix_data:
            path = self._generate_appendix_a2_scatters(
                appendix_data['a2_regime_scatters'],
                f"{filename_prefix}_a2_regime_scatters"
            )
            saved_paths.append(path)
        
        # Figure A3: Confidence Amplification
        if 'a3_confidence_amplification' in appendix_data:
            path = self._generate_appendix_a3_confidence(
                appendix_data['a3_confidence_amplification'],
                f"{filename_prefix}_a3_confidence_amplification"
            )
            saved_paths.append(path)
        
        # Figure A4: Full IRF
        if 'a4_full_irf' in appendix_data:
            path = self._generate_appendix_a4_full_irf(
                appendix_data['a4_full_irf'],
                f"{filename_prefix}_a4_full_irf"
            )
            saved_paths.append(path)
        
        # Figure A5: Detailed Decomposition
        if 'a5_decomposition_bars' in appendix_data:
            path = self._generate_appendix_a5_decomposition(
                appendix_data['a5_decomposition_bars'],
                f"{filename_prefix}_a5_decomposition_bars"
            )
            saved_paths.append(path)
        
        # Figure A6: Robustness Comparison
        if 'a6_robustness_comparison' in appendix_data:
            path = self._generate_appendix_a6_robustness(
                appendix_data['a6_robustness_comparison'],
                f"{filename_prefix}_a6_robustness_comparison"
            )
            saved_paths.append(path)
        
        return saved_paths
    
    def _generate_appendix_a1_ssr(
        self,
        data: Dict[str, Any],
        filename: str
    ) -> Path:
        """Generate Appendix Figure A1: Alternative SSR Plot."""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        threshold_grid = data['threshold_grid']
        ssr_values = data['ssr_values']
        threshold_estimate = data['threshold_estimate']
        
        # Normalize SSR for better visualization
        ssr_normalized = (ssr_values - np.min(ssr_values)) / (np.max(ssr_values) - np.min(ssr_values))
        
        ax.plot(threshold_grid, ssr_normalized, 'b-', linewidth=2)
        ax.axvline(threshold_estimate, color='r', linestyle='--', linewidth=1.5,
                   label=f'Threshold: {threshold_estimate:.3f}')
        
        ax.set_xlabel('Debt-to-GDP Ratio', fontsize=11)
        ax.set_ylabel('Normalized SSR', fontsize=11)
        ax.set_title('Appendix A1: Normalized Sum of Squared Residuals', fontsize=12)
        ax.legend(loc='best', frameon=True)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return self.save_figure(fig, filename)
    
    def _generate_appendix_a2_scatters(
        self,
        data: Dict[str, Any],
        filename: str
    ) -> Path:
        """Generate Appendix Figure A2: Extended Regime Scatter Plots."""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes = axes.flatten()
        
        variables = ['10Y Yields', '5Y Yields', '2Y Yields', 'Investment']
        
        for idx, var_name in enumerate(variables):
            if var_name not in data:
                continue
                
            ax = axes[idx]
            var_data = data[var_name]
            
            qe_shocks = var_data['qe_shocks']
            dependent_var = var_data['dependent_var']
            regime_indicator = var_data['regime_indicator']
            
            # Plot both regimes
            low_mask = ~regime_indicator
            high_mask = regime_indicator
            
            ax.scatter(qe_shocks[low_mask], dependent_var[low_mask], 
                      alpha=0.6, s=40, color='blue', label='Low Fiscal Stress')
            ax.scatter(qe_shocks[high_mask], dependent_var[high_mask], 
                      alpha=0.6, s=40, color='red', label='High Fiscal Stress')
            
            ax.axhline(0, color='black', linestyle='-', linewidth=0.5)
            ax.axvline(0, color='black', linestyle='-', linewidth=0.5)
            ax.set_xlabel('QE Shock', fontsize=10)
            ax.set_ylabel(f'{var_name} Change', fontsize=10)
            ax.set_title(f'{var_name}', fontsize=11, fontweight='bold')
            ax.legend(loc='best', frameon=True, fontsize=8)
            ax.grid(True, alpha=0.3)
        
        plt.suptitle('Appendix A2: Regime Effects Across Variables', fontsize=13, fontweight='bold')
        plt.tight_layout()
        return self.save_figure(fig, filename)
    
    def _generate_appendix_a3_confidence(
        self,
        data: Dict[str, Any],
        filename: str
    ) -> Path:
        """Generate Appendix Figure A3: Confidence Amplification Effects."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Left panel: Confidence effects over time
        dates = data['dates']
        confidence_proxy = data['confidence_proxy']
        qe_intensity = data['qe_intensity']
        
        ax1_twin = ax1.twinx()
        
        line1 = ax1.plot(dates, confidence_proxy, 'b-', linewidth=2, label='Confidence Proxy')
        line2 = ax1_twin.plot(dates, qe_intensity, 'r-', linewidth=2, label='QE Intensity')
        
        ax1.set_xlabel('Date', fontsize=11)
        ax1.set_ylabel('Confidence Proxy', fontsize=11, color='b')
        ax1_twin.set_ylabel('QE Intensity', fontsize=11, color='r')
        ax1.set_title('Confidence and QE Over Time', fontsize=11, fontweight='bold')
        
        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        ax1.legend(lines, labels, loc='best', frameon=True)
        ax1.grid(True, alpha=0.3)
        
        # Right panel: Confidence amplification scatter
        confidence_interaction = data['confidence_interaction']
        yield_changes = data['yield_changes']
        
        ax2.scatter(confidence_interaction, yield_changes, alpha=0.6, s=50, color='purple')
        
        # Add fitted line if available
        if 'fitted_line' in data:
            ax2.plot(confidence_interaction, data['fitted_line'], 'r-', linewidth=2)
        
        ax2.set_xlabel('Confidence × QE Interaction', fontsize=11)
        ax2.set_ylabel('Yield Change (bps)', fontsize=11)
        ax2.set_title('Confidence Amplification Effect', fontsize=11, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        plt.suptitle('Appendix A3: Confidence Channel Analysis', fontsize=13, fontweight='bold')
        plt.tight_layout()
        return self.save_figure(fig, filename)
    
    def _generate_appendix_a4_full_irf(
        self,
        data: Dict[str, Any],
        filename: str
    ) -> Path:
        """Generate Appendix Figure A4: Full Impulse Response Functions."""
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        variables = ['Investment', '10Y Yields', '5Y Yields', 
                    '2Y Yields', 'GDP Growth', 'Unemployment']
        
        for idx, var_name in enumerate(variables):
            if var_name not in data:
                continue
                
            ax = axes[idx]
            var_data = data[var_name]
            
            horizons = var_data['horizons']
            effects = var_data['effects']
            ci_lower = var_data['ci_lower']
            ci_upper = var_data['ci_upper']
            
            ax.plot(horizons, effects, 'b-', linewidth=2, marker='o', markersize=4)
            ax.fill_between(horizons, ci_lower, ci_upper, alpha=0.2, color='blue')
            ax.axhline(0, color='black', linestyle='-', linewidth=0.5)
            
            ax.set_xlabel('Quarters', fontsize=10)
            ax.set_ylabel('Effect', fontsize=10)
            ax.set_title(f'{var_name}', fontsize=11, fontweight='bold')
            ax.grid(True, alpha=0.3)
        
        plt.suptitle('Appendix A4: Full Set of Impulse Response Functions', 
                    fontsize=13, fontweight='bold')
        plt.tight_layout()
        return self.save_figure(fig, filename)
    
    def _generate_appendix_a5_decomposition(
        self,
        data: Dict[str, Any],
        filename: str
    ) -> Path:
        """Generate Appendix Figure A5: Detailed Channel Decomposition."""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Panel 1: Rate channel components
        ax1 = axes[0, 0]
        horizons = data['horizons']
        rate_direct = data['rate_direct']
        rate_indirect = data['rate_indirect']
        
        ax1.bar(horizons, rate_direct, width=0.4, label='Direct Effect', alpha=0.8)
        ax1.bar(horizons, rate_indirect, width=0.4, bottom=rate_direct, 
               label='Indirect Effect', alpha=0.8)
        ax1.set_xlabel('Quarters', fontsize=10)
        ax1.set_ylabel('Effect (pp)', fontsize=10)
        ax1.set_title('Interest Rate Channel Components', fontsize=11, fontweight='bold')
        ax1.legend(loc='best', frameon=True)
        ax1.grid(True, alpha=0.3, axis='y')
        
        # Panel 2: Distortion channel components
        ax2 = axes[0, 1]
        distortion_liquidity = data['distortion_liquidity']
        distortion_balance_sheet = data['distortion_balance_sheet']
        distortion_concentration = data['distortion_concentration']
        
        ax2.bar(horizons, distortion_liquidity, width=0.4, label='Liquidity', alpha=0.8)
        ax2.bar(horizons, distortion_balance_sheet, width=0.4, 
               bottom=distortion_liquidity, label='Balance Sheet', alpha=0.8)
        ax2.bar(horizons, distortion_concentration, width=0.4,
               bottom=distortion_liquidity + distortion_balance_sheet, 
               label='Concentration', alpha=0.8)
        ax2.set_xlabel('Quarters', fontsize=10)
        ax2.set_ylabel('Effect (pp)', fontsize=10)
        ax2.set_title('Market Distortion Channel Components', fontsize=11, fontweight='bold')
        ax2.legend(loc='best', frameon=True)
        ax2.grid(True, alpha=0.3, axis='y')
        
        # Panel 3: Cumulative contributions
        ax3 = axes[1, 0]
        cumulative_rate = np.cumsum(data['rate_contributions'])
        cumulative_distortion = np.cumsum(data['distortion_contributions'])
        
        ax3.plot(horizons, cumulative_rate, 'b-', linewidth=2, marker='o', 
                label='Rate Channel')
        ax3.plot(horizons, cumulative_distortion, 'r-', linewidth=2, marker='s', 
                label='Distortion Channel')
        ax3.set_xlabel('Quarters', fontsize=10)
        ax3.set_ylabel('Cumulative Effect (pp)', fontsize=10)
        ax3.set_title('Cumulative Channel Contributions', fontsize=11, fontweight='bold')
        ax3.legend(loc='best', frameon=True)
        ax3.grid(True, alpha=0.3)
        
        # Panel 4: Share evolution
        ax4 = axes[1, 1]
        total_effects = data['rate_contributions'] + data['distortion_contributions']
        rate_share_evolution = data['rate_contributions'] / total_effects * 100
        distortion_share_evolution = data['distortion_contributions'] / total_effects * 100
        
        ax4.plot(horizons, rate_share_evolution, 'b-', linewidth=2, marker='o', 
                label='Rate Share')
        ax4.plot(horizons, distortion_share_evolution, 'r-', linewidth=2, marker='s', 
                label='Distortion Share')
        ax4.axhline(35, color='b', linestyle='--', linewidth=1, alpha=0.5, label='Target: 35%')
        ax4.axhline(65, color='r', linestyle='--', linewidth=1, alpha=0.5, label='Target: 65%')
        ax4.set_xlabel('Quarters', fontsize=10)
        ax4.set_ylabel('Share (%)', fontsize=10)
        ax4.set_title('Channel Share Evolution', fontsize=11, fontweight='bold')
        ax4.legend(loc='best', frameon=True, fontsize=8)
        ax4.grid(True, alpha=0.3)
        
        plt.suptitle('Appendix A5: Detailed Channel Decomposition Analysis', 
                    fontsize=13, fontweight='bold')
        plt.tight_layout()
        return self.save_figure(fig, filename)
    
    def _generate_appendix_a6_robustness(
        self,
        data: Dict[str, Any],
        filename: str
    ) -> Path:
        """Generate Appendix Figure A6: Robustness Comparison."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Left panel: Threshold estimates across specifications
        specifications = data['specifications']
        threshold_estimates = data['threshold_estimates']
        threshold_cis = data['threshold_cis']
        
        y_pos = np.arange(len(specifications))
        
        ax1.barh(y_pos, threshold_estimates, xerr=threshold_cis, 
                capsize=5, alpha=0.7, color='steelblue')
        ax1.axvline(0.285, color='r', linestyle='--', linewidth=2, label='Baseline: 0.285')
        ax1.set_yticks(y_pos)
        ax1.set_yticklabels(specifications, fontsize=9)
        ax1.set_xlabel('Threshold Estimate', fontsize=11)
        ax1.set_title('Threshold Estimates Across Specifications', fontsize=11, fontweight='bold')
        ax1.legend(loc='best', frameon=True)
        ax1.grid(True, alpha=0.3, axis='x')
        
        # Right panel: Attenuation percentages
        attenuation_pcts = data['attenuation_pcts']
        
        ax2.barh(y_pos, attenuation_pcts, alpha=0.7, color='coral')
        ax2.axvline(63, color='r', linestyle='--', linewidth=2, label='Baseline: 63%')
        ax2.set_yticks(y_pos)
        ax2.set_yticklabels(specifications, fontsize=9)
        ax2.set_xlabel('Attenuation (%)', fontsize=11)
        ax2.set_title('Attenuation Across Specifications', fontsize=11, fontweight='bold')
        ax2.legend(loc='best', frameon=True)
        ax2.grid(True, alpha=0.3, axis='x')
        
        plt.suptitle('Appendix A6: Robustness of Main Results', fontsize=13, fontweight='bold')
        plt.tight_layout()
        return self.save_figure(fig, filename)
