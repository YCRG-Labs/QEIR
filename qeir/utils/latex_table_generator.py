"""
LaTeX Table Generator for QE Hypothesis Testing Results

This module provides automated LaTeX table formatting for statistical results
from the QE hypothesis testing framework, with proper notation, significance
indicators, and consistent formatting across all three hypotheses.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Union
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class LaTeXTableGenerator:
    """
    Generates publication-ready LaTeX tables for QE hypothesis testing results.
    
    Features:
    - Automated LaTeX formatting with proper statistical notation
    - Statistical significance indicators (*, **, ***)
    - Confidence intervals formatting
    - Consistent styling across all three hypotheses
    - Support for multiple model comparison tables
    """
    
    def __init__(self, output_dir: str = "output/tables"):
        """
        Initialize LaTeX table generator.
        
        Parameters:
        -----------
        output_dir : str
            Directory to save generated LaTeX tables
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # LaTeX formatting settings
        self.decimal_places = 3
        self.significance_levels = {0.01: '***', 0.05: '**', 0.10: '*'}
        
        # Table styling
        self.table_style = {
            'booktabs': True,
            'centering': True,
            'caption_position': 'top',
            'label_prefix': 'tab:'
        }
    
    def format_coefficient(self, coef: float, se: float, pvalue: float) -> str:
        """
        Format coefficient with standard error and significance stars.
        
        Parameters:
        -----------
        coef : float
            Coefficient estimate
        se : float
            Standard error
        pvalue : float
            P-value for significance testing
            
        Returns:
        --------
        str
            Formatted coefficient string with significance indicators
        """
        # Format coefficient
        coef_str = f"{coef:.{self.decimal_places}f}"
        
        # Add significance stars
        stars = ""
        for level, star in sorted(self.significance_levels.items()):
            if pvalue <= level:
                stars = star
                break
        
        # Format standard error
        se_str = f"({se:.{self.decimal_places}f})"
        
        return f"{coef_str}{stars} \\\\ {se_str}"
    
    def format_confidence_interval(self, lower: float, upper: float) -> str:
        """
        Format confidence interval for LaTeX display.
        
        Parameters:
        -----------
        lower : float
            Lower bound of confidence interval
        upper : float
            Upper bound of confidence interval
            
        Returns:
        --------
        str
            Formatted confidence interval string
        """
        return f"[{lower:.{self.decimal_places}f}, {upper:.{self.decimal_places}f}]"
    
    def create_hypothesis1_table(self, results: Dict[str, Any], 
                                filename: str = "hypothesis1_threshold_results.tex") -> str:
        """
        Create LaTeX table for Hypothesis 1 threshold effects results.
        
        Parameters:
        -----------
        results : Dict[str, Any]
            Results dictionary containing threshold regression results
        filename : str
            Output filename for LaTeX table
            
        Returns:
        --------
        str
            Path to generated LaTeX file
        """
        logger.info("Generating Hypothesis 1 threshold effects table")
        
        # Extract results
        hansen_results = results.get('hansen_results', {})
        threshold_value = hansen_results.get('threshold', 0.0)
        threshold_se = hansen_results.get('threshold_se', 0.0)
        threshold_pvalue = hansen_results.get('threshold_pvalue', 1.0)
        
        # Regime coefficients
        regime1_coefs = hansen_results.get('regime1_coefficients', {})
        regime2_coefs = hansen_results.get('regime2_coefficients', {})
        
        # Start LaTeX table
        latex_content = self._create_table_header(
            caption="Central Bank Reaction and Confidence Effects: Threshold Regression Results",
            label="hypothesis1_threshold",
            columns=["Variable", "Regime 1 (Low Threshold)", "Regime 2 (High Threshold)"]
        )
        
        # Add threshold information
        threshold_str = self.format_coefficient(threshold_value, threshold_se, threshold_pvalue)
        latex_content += f"\\multicolumn{{3}}{{c}}{{\\textbf{{Threshold Value: {threshold_value:.{self.decimal_places}f}}}}} \\\\\n"
        latex_content += "\\midrule\n"
        
        # Add coefficient rows
        variables = ['central_bank_reaction', 'confidence_effects', 'interaction_term', 'debt_service_burden']
        var_labels = ['Central Bank Reaction ($\\gamma_1$)', 'Confidence Effects ($\\lambda_2$)', 
                     'Interaction ($\\gamma_1 \\times \\lambda_2$)', 'Debt Service Burden']
        
        for var, label in zip(variables, var_labels):
            if var in regime1_coefs and var in regime2_coefs:
                r1_coef = self.format_coefficient(
                    regime1_coefs[var]['coef'],
                    regime1_coefs[var]['se'],
                    regime1_coefs[var]['pvalue']
                )
                r2_coef = self.format_coefficient(
                    regime2_coefs[var]['coef'],
                    regime2_coefs[var]['se'],
                    regime2_coefs[var]['pvalue']
                )
                latex_content += f"{label} & {r1_coef} & {r2_coef} \\\\\n"
        
        # Add model statistics
        latex_content += "\\midrule\n"
        latex_content += f"Observations & \\multicolumn{{2}}{{c}}{{{hansen_results.get('n_obs', 'N/A')}}} \\\\\n"
        latex_content += f"R-squared & \\multicolumn{{2}}{{c}}{{{hansen_results.get('r_squared', 0.0):.3f}}} \\\\\n"
        latex_content += f"F-statistic & \\multicolumn{{2}}{{c}}{{{hansen_results.get('f_stat', 0.0):.3f}}} \\\\\n"
        
        latex_content += self._create_table_footer()
        
        # Save to file
        output_path = self.output_dir / filename
        with open(output_path, 'w') as f:
            f.write(latex_content)
        
        logger.info(f"Hypothesis 1 table saved to {output_path}")
        return str(output_path)
    
    def create_hypothesis2_table(self, results: Dict[str, Any],
                                filename: str = "hypothesis2_investment_results.tex") -> str:
        """
        Create LaTeX table for Hypothesis 2 investment distortion results.
        
        Parameters:
        -----------
        results : Dict[str, Any]
            Results dictionary containing investment analysis results
        filename : str
            Output filename for LaTeX table
            
        Returns:
        --------
        str
            Path to generated LaTeX file
        """
        logger.info("Generating Hypothesis 2 investment distortion table")
        
        # Extract results
        decomposition_results = results.get('channel_decomposition', {})
        investment_results = results.get('investment_impact', {})
        
        # Start LaTeX table
        latex_content = self._create_table_header(
            caption="QE Impact on Private Investment: Channel Decomposition Analysis",
            label="hypothesis2_investment",
            columns=["Channel", "Short-term Effect", "Long-term Effect", "Dominance Test"]
        )
        
        # Interest rate channel
        ir_short = decomposition_results.get('interest_rate_channel', {}).get('short_term', {})
        ir_long = decomposition_results.get('interest_rate_channel', {}).get('long_term', {})
        ir_dominance = decomposition_results.get('interest_rate_channel', {}).get('dominance_pvalue', 1.0)
        
        if ir_short and ir_long:
            ir_short_str = self.format_coefficient(
                ir_short.get('coef', 0.0),
                ir_short.get('se', 0.0),
                ir_short.get('pvalue', 1.0)
            )
            ir_long_str = self.format_coefficient(
                ir_long.get('coef', 0.0),
                ir_long.get('se', 0.0),
                ir_long.get('pvalue', 1.0)
            )
            dominance_str = f"p = {ir_dominance:.3f}"
            latex_content += f"Interest Rate Channel & {ir_short_str} & {ir_long_str} & {dominance_str} \\\\\n"
        
        # Market distortion channel
        md_short = decomposition_results.get('market_distortion_channel', {}).get('short_term', {})
        md_long = decomposition_results.get('market_distortion_channel', {}).get('long_term', {})
        md_dominance = decomposition_results.get('market_distortion_channel', {}).get('dominance_pvalue', 1.0)
        
        if md_short and md_long:
            md_short_str = self.format_coefficient(
                md_short.get('coef', 0.0),
                md_short.get('se', 0.0),
                md_short.get('pvalue', 1.0)
            )
            md_long_str = self.format_coefficient(
                md_long.get('coef', 0.0),
                md_long.get('se', 0.0),
                md_long.get('pvalue', 1.0)
            )
            dominance_str = f"p = {md_dominance:.3f}"
            latex_content += f"Market Distortion Channel ($\\mu_2$) & {md_short_str} & {md_long_str} & {dominance_str} \\\\\n"
        
        # Add model statistics
        latex_content += "\\midrule\n"
        latex_content += f"QE Episodes & \\multicolumn{{3}}{{c}}{{{investment_results.get('qe_episodes', 'N/A')}}} \\\\\n"
        latex_content += f"Observations & \\multicolumn{{3}}{{c}}{{{investment_results.get('n_obs', 'N/A')}}} \\\\\n"
        latex_content += f"R-squared & \\multicolumn{{3}}{{c}}{{{investment_results.get('r_squared', 0.0):.3f}}} \\\\\n"
        
        latex_content += self._create_table_footer()
        
        # Save to file
        output_path = self.output_dir / filename
        with open(output_path, 'w') as f:
            f.write(latex_content)
        
        logger.info(f"Hypothesis 2 table saved to {output_path}")
        return str(output_path) 
   
    def create_hypothesis3_table(self, results: Dict[str, Any],
                                filename: str = "hypothesis3_spillover_results.tex") -> str:
        """
        Create LaTeX table for Hypothesis 3 international spillover results.
        
        Parameters:
        -----------
        results : Dict[str, Any]
            Results dictionary containing international spillover results
        filename : str
            Output filename for LaTeX table
            
        Returns:
        --------
        str
            Path to generated LaTeX file
        """
        logger.info("Generating Hypothesis 3 international spillover table")
        
        # Extract results
        spillover_results = results.get('spillover_analysis', {})
        currency_results = results.get('currency_effects', {})
        inflation_results = results.get('inflation_offset', {})
        
        # Start LaTeX table
        latex_content = self._create_table_header(
            caption="International QE Effects: Spillover and Currency Analysis",
            label="hypothesis3_spillover",
            columns=["Effect", "Coefficient", "95\\% CI", "Economic Significance"]
        )
        
        # Foreign bond demand effect
        foreign_demand = spillover_results.get('foreign_demand_effect', {})
        if foreign_demand:
            coef_str = self.format_coefficient(
                foreign_demand.get('coef', 0.0),
                foreign_demand.get('se', 0.0),
                foreign_demand.get('pvalue', 1.0)
            )
            ci_str = self.format_confidence_interval(
                foreign_demand.get('ci_lower', 0.0),
                foreign_demand.get('ci_upper', 0.0)
            )
            econ_sig = f"{foreign_demand.get('economic_magnitude', 0.0):.1f}\\% reduction"
            latex_content += f"Foreign Bond Demand & {coef_str} & {ci_str} & {econ_sig} \\\\\n"
        
        # Currency depreciation effect
        currency_effect = currency_results.get('depreciation_effect', {})
        if currency_effect:
            coef_str = self.format_coefficient(
                currency_effect.get('coef', 0.0),
                currency_effect.get('se', 0.0),
                currency_effect.get('pvalue', 1.0)
            )
            ci_str = self.format_confidence_interval(
                currency_effect.get('ci_lower', 0.0),
                currency_effect.get('ci_upper', 0.0)
            )
            econ_sig = f"{currency_effect.get('economic_magnitude', 0.0):.1f}\\% depreciation"
            latex_content += f"Currency Depreciation & {coef_str} & {ci_str} & {econ_sig} \\\\\n"
        
        # Inflation offset effect
        inflation_effect = inflation_results.get('offset_effect', {})
        if inflation_effect:
            coef_str = self.format_coefficient(
                inflation_effect.get('coef', 0.0),
                inflation_effect.get('se', 0.0),
                inflation_effect.get('pvalue', 1.0)
            )
            ci_str = self.format_confidence_interval(
                inflation_effect.get('ci_lower', 0.0),
                inflation_effect.get('ci_upper', 0.0)
            )
            econ_sig = f"{inflation_effect.get('offset_ratio', 0.0):.1f}\\% of QE benefit"
            latex_content += f"Inflationary Offset & {coef_str} & {ci_str} & {econ_sig} \\\\\n"
        
        # Add model statistics
        latex_content += "\\midrule\n"
        latex_content += f"Countries & \\multicolumn{{3}}{{c}}{{{spillover_results.get('n_countries', 'N/A')}}} \\\\\n"
        latex_content += f"Time Period & \\multicolumn{{3}}{{c}}{{{spillover_results.get('time_period', 'N/A')}}} \\\\\n"
        latex_content += f"Granger Causality p-value & \\multicolumn{{3}}{{c}}{{{spillover_results.get('granger_pvalue', 1.0):.3f}}} \\\\\n"
        
        latex_content += self._create_table_footer()
        
        # Save to file
        output_path = self.output_dir / filename
        with open(output_path, 'w') as f:
            f.write(latex_content)
        
        logger.info(f"Hypothesis 3 table saved to {output_path}")
        return str(output_path)
    
    def create_model_comparison_table(self, results: Dict[str, Any],
                                     filename: str = "model_comparison_results.tex") -> str:
        """
        Create LaTeX table comparing statistical and ML model results.
        
        Parameters:
        -----------
        results : Dict[str, Any]
            Results dictionary containing model comparison results
        filename : str
            Output filename for LaTeX table
            
        Returns:
        --------
        str
            Path to generated LaTeX file
        """
        logger.info("Generating model comparison table")
        
        # Extract model results
        statistical_models = results.get('statistical_models', {})
        ml_models = results.get('ml_models', {})
        ensemble_results = results.get('ensemble_results', {})
        
        # Start LaTeX table
        latex_content = self._create_table_header(
            caption="Model Comparison: Statistical vs Machine Learning Approaches",
            label="model_comparison",
            columns=["Model", "RMSE", "MAE", "R²", "AIC/BIC"]
        )
        
        # Statistical models section
        latex_content += "\\multicolumn{5}{c}{\\textbf{Statistical Models}} \\\\\n"
        latex_content += "\\midrule\n"
        
        stat_models = ['hansen_threshold', 'local_projections', 'var_model']
        stat_labels = ['Hansen Threshold', 'Local Projections', 'VAR Model']
        
        for model, label in zip(stat_models, stat_labels):
            if model in statistical_models:
                model_results = statistical_models[model]
                rmse = f"{model_results.get('rmse', 0.0):.4f}"
                mae = f"{model_results.get('mae', 0.0):.4f}"
                r2 = f"{model_results.get('r_squared', 0.0):.3f}"
                aic_bic = f"{model_results.get('aic', 0.0):.1f}/{model_results.get('bic', 0.0):.1f}"
                latex_content += f"{label} & {rmse} & {mae} & {r2} & {aic_bic} \\\\\n"
        
        # ML models section
        latex_content += "\\midrule\n"
        latex_content += "\\multicolumn{5}{c}{\\textbf{Machine Learning Models}} \\\\\n"
        latex_content += "\\midrule\n"
        
        ml_model_names = ['random_forest', 'gradient_boosting', 'neural_network']
        ml_labels = ['Random Forest', 'Gradient Boosting', 'Neural Network']
        
        for model, label in zip(ml_model_names, ml_labels):
            if model in ml_models:
                model_results = ml_models[model]
                rmse = f"{model_results.get('rmse', 0.0):.4f}"
                mae = f"{model_results.get('mae', 0.0):.4f}"
                r2 = f"{model_results.get('r_squared', 0.0):.3f}"
                cv_score = f"{model_results.get('cv_score', 0.0):.3f}"
                latex_content += f"{label} & {rmse} & {mae} & {r2} & {cv_score} \\\\\n"
        
        # Ensemble results
        if ensemble_results:
            latex_content += "\\midrule\n"
            latex_content += "\\multicolumn{5}{c}{\\textbf{Ensemble Methods}} \\\\\n"
            latex_content += "\\midrule\n"
            
            ensemble_rmse = f"{ensemble_results.get('rmse', 0.0):.4f}"
            ensemble_mae = f"{ensemble_results.get('mae', 0.0):.4f}"
            ensemble_r2 = f"{ensemble_results.get('r_squared', 0.0):.3f}"
            ensemble_weight = f"{ensemble_results.get('avg_weight', 0.0):.3f}"
            latex_content += f"Weighted Ensemble & {ensemble_rmse} & {ensemble_mae} & {ensemble_r2} & {ensemble_weight} \\\\\n"
        
        latex_content += self._create_table_footer()
        
        # Save to file
        output_path = self.output_dir / filename
        with open(output_path, 'w') as f:
            f.write(latex_content)
        
        logger.info(f"Model comparison table saved to {output_path}")
        return str(output_path)
    
    def create_robustness_table(self, results: Dict[str, Any],
                               filename: str = "robustness_results.tex") -> str:
        """
        Create LaTeX table for robustness test results.
        
        Parameters:
        -----------
        results : Dict[str, Any]
            Results dictionary containing robustness test results
        filename : str
            Output filename for LaTeX table
            
        Returns:
        --------
        str
            Path to generated LaTeX file
        """
        logger.info("Generating robustness test table")
        
        # Extract robustness results
        sensitivity_tests = results.get('sensitivity_analysis', {})
        bootstrap_results = results.get('bootstrap_results', {})
        alternative_specs = results.get('alternative_specifications', {})
        
        # Start LaTeX table
        latex_content = self._create_table_header(
            caption="Robustness Tests: Sensitivity Analysis and Alternative Specifications",
            label="robustness_tests",
            columns=["Test", "Baseline", "Alternative 1", "Alternative 2", "p-value"]
        )
        
        # Sensitivity tests
        if sensitivity_tests:
            latex_content += "\\multicolumn{5}{c}{\\textbf{Sensitivity Analysis}} \\\\\n"
            latex_content += "\\midrule\n"
            
            for test_name, test_results in sensitivity_tests.items():
                baseline = f"{test_results.get('baseline', 0.0):.3f}"
                alt1 = f"{test_results.get('alternative1', 0.0):.3f}"
                alt2 = f"{test_results.get('alternative2', 0.0):.3f}"
                pvalue = f"{test_results.get('stability_pvalue', 1.0):.3f}"
                
                test_label = test_name.replace('_', ' ').title()
                latex_content += f"{test_label} & {baseline} & {alt1} & {alt2} & {pvalue} \\\\\n"
        
        # Bootstrap results
        if bootstrap_results:
            latex_content += "\\midrule\n"
            latex_content += "\\multicolumn{5}{c}{\\textbf{Bootstrap Confidence Intervals}} \\\\\n"
            latex_content += "\\midrule\n"
            
            for param_name, boot_results in bootstrap_results.items():
                param_label = param_name.replace('_', ' ').title()
                baseline = f"{boot_results.get('point_estimate', 0.0):.3f}"
                ci_lower = f"{boot_results.get('ci_lower', 0.0):.3f}"
                ci_upper = f"{boot_results.get('ci_upper', 0.0):.3f}"
                bias = f"{boot_results.get('bias', 0.0):.4f}"
                
                latex_content += f"{param_label} & {baseline} & {ci_lower} & {ci_upper} & {bias} \\\\\n"
        
        latex_content += self._create_table_footer()
        
        # Save to file
        output_path = self.output_dir / filename
        with open(output_path, 'w') as f:
            f.write(latex_content)
        
        logger.info(f"Robustness table saved to {output_path}")
        return str(output_path)
    
    def _create_table_header(self, caption: str, label: str, columns: List[str]) -> str:
        """Create LaTeX table header with proper formatting."""
        n_cols = len(columns)
        col_spec = "l" + "c" * (n_cols - 1)
        
        header = f"""\\begin{{table}}[htbp]
\\centering
\\caption{{{caption}}}
\\label{{{self.table_style['label_prefix']}{label}}}
\\begin{{tabular}}{{{col_spec}}}
\\toprule
"""
        
        # Add column headers
        header += " & ".join(columns) + " \\\\\n"
        header += "\\midrule\n"
        
        return header
    
    def _create_table_footer(self) -> str:
        """Create LaTeX table footer with significance notes."""
        footer = """\\bottomrule
\\end{tabular}
\\begin{tablenotes}
\\small
\\item Note: Standard errors in parentheses. 
\\item *** p$<$0.01, ** p$<$0.05, * p$<$0.10
\\end{tablenotes}
\\end{table}

"""
        return footer
    
    def generate_all_tables(self, results: Dict[str, Any]) -> Dict[str, str]:
        """
        Generate all LaTeX tables for the hypothesis testing results.
        
        Parameters:
        -----------
        results : Dict[str, Any]
            Complete results dictionary from hypothesis testing framework
            
        Returns:
        --------
        Dict[str, str]
            Dictionary mapping table names to file paths
        """
        logger.info("Generating all LaTeX tables")
        
        generated_tables = {}
        
        # Generate hypothesis-specific tables
        if 'hypothesis1_results' in results:
            path = self.create_hypothesis1_table(results['hypothesis1_results'])
            generated_tables['hypothesis1'] = path
        
        if 'hypothesis2_results' in results:
            path = self.create_hypothesis2_table(results['hypothesis2_results'])
            generated_tables['hypothesis2'] = path
        
        if 'hypothesis3_results' in results:
            path = self.create_hypothesis3_table(results['hypothesis3_results'])
            generated_tables['hypothesis3'] = path
        
        # Generate comparison and robustness tables
        if 'model_comparison' in results:
            path = self.create_model_comparison_table(results['model_comparison'])
            generated_tables['model_comparison'] = path
        
        if 'robustness_tests' in results:
            path = self.create_robustness_table(results['robustness_tests'])
            generated_tables['robustness'] = path
        
        logger.info(f"Generated {len(generated_tables)} LaTeX tables")
        return generated_tables


def create_latex_tables(results: Dict[str, Any], output_dir: str = "output/tables") -> Dict[str, str]:
    """
    Convenience function to generate all LaTeX tables.
    
    Parameters:
    -----------
    results : Dict[str, Any]
        Complete results dictionary from hypothesis testing framework
    output_dir : str
        Directory to save generated LaTeX tables
        
    Returns:
    --------
    Dict[str, str]
        Dictionary mapping table names to file paths
    """
    generator = LaTeXTableGenerator(output_dir)
    return generator.generate_all_tables(results)