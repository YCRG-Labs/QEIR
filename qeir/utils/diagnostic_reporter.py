"""
Comprehensive Diagnostic and Robustness Reporter

This module provides comprehensive diagnostic plot generation, robustness test
summaries, and structured output organization for QE hypothesis testing results.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any, Optional, Tuple, Union
from pathlib import Path
import logging
import json
from datetime import datetime
import warnings
from scipy import stats
from statsmodels.stats.diagnostic import het_breuschpagan, het_white
from statsmodels.stats.stattools import jarque_bera
from statsmodels.tsa.stattools import adfuller

logger = logging.getLogger(__name__)


class DiagnosticReporter:
    """
    Generates comprehensive diagnostic reports and robustness test summaries
    for QE hypothesis testing results.
    
    Features:
    - Model diagnostic plots and statistical tests
    - Robustness test summaries and comparison tables
    - Structured output directory organization
    - Automated report generation with interpretation
    """
    
    def __init__(self, output_dir: str = "output/diagnostics"):
        """
        Initialize diagnostic reporter.
        
        Parameters:
        -----------
        output_dir : str
            Directory to save diagnostic reports and plots
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        (self.output_dir / "plots").mkdir(exist_ok=True)
        (self.output_dir / "tables").mkdir(exist_ok=True)
        (self.output_dir / "reports").mkdir(exist_ok=True)
        (self.output_dir / "robustness").mkdir(exist_ok=True)
        
        # Diagnostic test thresholds
        self.diagnostic_thresholds = {
            'normality_pvalue': 0.05,
            'heteroskedasticity_pvalue': 0.05,
            'autocorrelation_threshold': 0.1,
            'stationarity_pvalue': 0.05,
            'outlier_threshold': 3.0  # Standard deviations
        }
        
        # Report metadata
        self.report_metadata = {
            'generated_at': datetime.now().isoformat(),
            'version': '1.0.0',
            'diagnostic_tests_performed': []
        }
    
    def generate_model_diagnostics(self, results: Dict[str, Any], 
                                  model_name: str = "model") -> Dict[str, Any]:
        """
        Generate comprehensive model diagnostic plots and tests.
        
        Parameters:
        -----------
        results : Dict[str, Any]
            Model results dictionary containing residuals, fitted values, etc.
        model_name : str
            Name of the model for file naming
            
        Returns:
        --------
        Dict[str, Any]
            Dictionary containing diagnostic test results and plot paths
        """
        logger.info(f"Generating model diagnostics for {model_name}")
        
        diagnostic_results = {
            'model_name': model_name,
            'diagnostic_plots': {},
            'statistical_tests': {},
            'interpretation': {}
        }
        
        # Extract model components
        residuals = np.array(results.get('residuals', []))
        fitted_values = np.array(results.get('fitted_values', []))
        actual_values = np.array(results.get('actual_values', []))
        
        if len(residuals) == 0:
            logger.warning(f"No residuals found for {model_name}")
            return diagnostic_results
        
        # Generate diagnostic plots
        diagnostic_results['diagnostic_plots'] = self._create_diagnostic_plots(
            residuals, fitted_values, actual_values, model_name
        )
        
        # Perform statistical tests
        diagnostic_results['statistical_tests'] = self._perform_diagnostic_tests(
            residuals, fitted_values
        )
        
        # Generate interpretation
        diagnostic_results['interpretation'] = self._interpret_diagnostics(
            diagnostic_results['statistical_tests']
        )
        
        # Save diagnostic summary
        self._save_diagnostic_summary(diagnostic_results, model_name)
        
        return diagnostic_results
    
    def _create_diagnostic_plots(self, residuals: np.ndarray, 
                                fitted_values: np.ndarray,
                                actual_values: np.ndarray, 
                                model_name: str) -> Dict[str, str]:
        """Create comprehensive diagnostic plots."""
        plot_paths = {}
        
        # Create figure with subplots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'Model Diagnostics: {model_name.replace("_", " ").title()}', 
                    fontsize=16, fontweight='bold')
        
        # Plot 1: Residuals vs Fitted Values
        ax1.scatter(fitted_values, residuals, alpha=0.6, s=30)
        ax1.axhline(y=0, color='red', linestyle='--', alpha=0.8)
        ax1.set_xlabel('Fitted Values')
        ax1.set_ylabel('Residuals')
        ax1.set_title('Residuals vs Fitted Values')
        ax1.grid(True, alpha=0.3)
        
        # Add LOWESS smooth line
        try:
            from statsmodels.nonparametric.smoothers_lowess import lowess
            smoothed = lowess(residuals, fitted_values, frac=0.3)
            ax1.plot(smoothed[:, 0], smoothed[:, 1], color='blue', linewidth=2, 
                    label='LOWESS')
            ax1.legend()
        except ImportError:
            pass
        
        # Plot 2: Q-Q Plot for Normality
        stats.probplot(residuals, dist="norm", plot=ax2)
        ax2.set_title('Q-Q Plot (Normality Check)')
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Histogram of Residuals
        ax3.hist(residuals, bins=30, density=True, alpha=0.7, color='skyblue', 
                edgecolor='black')
        
        # Overlay normal distribution
        mu, sigma = np.mean(residuals), np.std(residuals)
        x = np.linspace(residuals.min(), residuals.max(), 100)
        ax3.plot(x, stats.norm.pdf(x, mu, sigma), 'r-', linewidth=2, 
                label=f'Normal(μ={mu:.3f}, σ={sigma:.3f})')
        ax3.set_xlabel('Residuals')
        ax3.set_ylabel('Density')
        ax3.set_title('Residual Distribution')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Actual vs Predicted
        if len(actual_values) > 0 and len(fitted_values) > 0:
            ax4.scatter(actual_values, fitted_values, alpha=0.6, s=30)
            
            # Add 45-degree line
            min_val = min(actual_values.min(), fitted_values.min())
            max_val = max(actual_values.max(), fitted_values.max())
            ax4.plot([min_val, max_val], [min_val, max_val], 'r--', 
                    linewidth=2, label='Perfect Fit')
            
            ax4.set_xlabel('Actual Values')
            ax4.set_ylabel('Predicted Values')
            ax4.set_title('Actual vs Predicted')
            ax4.legend()
            ax4.grid(True, alpha=0.3)
            
            # Calculate and display R²
            r_squared = np.corrcoef(actual_values, fitted_values)[0, 1]**2
            ax4.text(0.05, 0.95, f'R² = {r_squared:.3f}', 
                    transform=ax4.transAxes, fontsize=12, 
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        
        # Save diagnostic plots
        plot_path = self.output_dir / "plots" / f"{model_name}_diagnostics.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        plot_paths['main_diagnostics'] = str(plot_path)
        
        # Create additional specialized plots
        plot_paths.update(self._create_specialized_diagnostic_plots(
            residuals, fitted_values, model_name
        ))
        
        return plot_paths
    
    def _create_specialized_diagnostic_plots(self, residuals: np.ndarray,
                                           fitted_values: np.ndarray,
                                           model_name: str) -> Dict[str, str]:
        """Create specialized diagnostic plots."""
        plot_paths = {}
        
        # Autocorrelation plot
        if len(residuals) > 10:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
            
            # ACF plot
            try:
                from statsmodels.tsa.stattools import acf
                from statsmodels.graphics.tsaplots import plot_acf
                
                plot_acf(residuals, ax=ax1, lags=min(20, len(residuals)//4))
                ax1.set_title('Autocorrelation Function')
                
                # PACF plot
                from statsmodels.graphics.tsaplots import plot_pacf
                plot_pacf(residuals, ax=ax2, lags=min(20, len(residuals)//4))
                ax2.set_title('Partial Autocorrelation Function')
                
                plt.tight_layout()
                
                acf_path = self.output_dir / "plots" / f"{model_name}_autocorrelation.png"
                plt.savefig(acf_path, dpi=300, bbox_inches='tight')
                plt.close()
                
                plot_paths['autocorrelation'] = str(acf_path)
                
            except ImportError:
                plt.close()
                logger.warning("Statsmodels not available for ACF/PACF plots")
        
        # Scale-Location plot (for heteroskedasticity)
        fig, ax = plt.subplots(figsize=(8, 6))
        
        sqrt_abs_residuals = np.sqrt(np.abs(residuals))
        ax.scatter(fitted_values, sqrt_abs_residuals, alpha=0.6, s=30)
        
        # Add LOWESS smooth line
        try:
            from statsmodels.nonparametric.smoothers_lowess import lowess
            smoothed = lowess(sqrt_abs_residuals, fitted_values, frac=0.3)
            ax.plot(smoothed[:, 0], smoothed[:, 1], color='red', linewidth=2)
        except ImportError:
            pass
        
        ax.set_xlabel('Fitted Values')
        ax.set_ylabel('√|Residuals|')
        ax.set_title('Scale-Location Plot')
        ax.grid(True, alpha=0.3)
        
        scale_loc_path = self.output_dir / "plots" / f"{model_name}_scale_location.png"
        plt.savefig(scale_loc_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        plot_paths['scale_location'] = str(scale_loc_path)
        
        return plot_paths
    
    def _perform_diagnostic_tests(self, residuals: np.ndarray,
                                 fitted_values: np.ndarray) -> Dict[str, Any]:
        """Perform comprehensive statistical diagnostic tests."""
        test_results = {}
        
        # Normality tests
        test_results['normality'] = self._test_normality(residuals)
        
        # Heteroskedasticity tests
        if len(fitted_values) > 0:
            test_results['heteroskedasticity'] = self._test_heteroskedasticity(
                residuals, fitted_values
            )
        
        # Autocorrelation tests
        test_results['autocorrelation'] = self._test_autocorrelation(residuals)
        
        # Outlier detection
        test_results['outliers'] = self._detect_outliers(residuals)
        
        # Stationarity test
        test_results['stationarity'] = self._test_stationarity(residuals)
        
        return test_results
    
    def _test_normality(self, residuals: np.ndarray) -> Dict[str, Any]:
        """Test residual normality using multiple tests."""
        normality_results = {}
        
        # Jarque-Bera test
        try:
            jb_stat, jb_pvalue = jarque_bera(residuals)
            normality_results['jarque_bera'] = {
                'statistic': float(jb_stat),
                'pvalue': float(jb_pvalue),
                'interpretation': 'Normal' if jb_pvalue > self.diagnostic_thresholds['normality_pvalue'] else 'Non-normal'
            }
        except Exception as e:
            logger.warning(f"Jarque-Bera test failed: {e}")
            normality_results['jarque_bera'] = {'error': str(e)}
        
        # Shapiro-Wilk test (for smaller samples)
        if len(residuals) <= 5000:
            try:
                sw_stat, sw_pvalue = stats.shapiro(residuals)
                normality_results['shapiro_wilk'] = {
                    'statistic': float(sw_stat),
                    'pvalue': float(sw_pvalue),
                    'interpretation': 'Normal' if sw_pvalue > self.diagnostic_thresholds['normality_pvalue'] else 'Non-normal'
                }
            except Exception as e:
                logger.warning(f"Shapiro-Wilk test failed: {e}")
                normality_results['shapiro_wilk'] = {'error': str(e)}
        
        # Kolmogorov-Smirnov test
        try:
            ks_stat, ks_pvalue = stats.kstest(residuals, 'norm', 
                                            args=(np.mean(residuals), np.std(residuals)))
            normality_results['kolmogorov_smirnov'] = {
                'statistic': float(ks_stat),
                'pvalue': float(ks_pvalue),
                'interpretation': 'Normal' if ks_pvalue > self.diagnostic_thresholds['normality_pvalue'] else 'Non-normal'
            }
        except Exception as e:
            logger.warning(f"Kolmogorov-Smirnov test failed: {e}")
            normality_results['kolmogorov_smirnov'] = {'error': str(e)}
        
        return normality_results
    
    def _test_heteroskedasticity(self, residuals: np.ndarray,
                                fitted_values: np.ndarray) -> Dict[str, Any]:
        """Test for heteroskedasticity using multiple tests."""
        hetero_results = {}
        
        # Reshape for statsmodels
        X = fitted_values.reshape(-1, 1)
        
        # Breusch-Pagan test
        try:
            bp_stat, bp_pvalue, _, _ = het_breuschpagan(residuals, X)
            hetero_results['breusch_pagan'] = {
                'statistic': float(bp_stat),
                'pvalue': float(bp_pvalue),
                'interpretation': 'Homoskedastic' if bp_pvalue > self.diagnostic_thresholds['heteroskedasticity_pvalue'] else 'Heteroskedastic'
            }
        except Exception as e:
            logger.warning(f"Breusch-Pagan test failed: {e}")
            hetero_results['breusch_pagan'] = {'error': str(e)}
        
        # White test
        try:
            white_stat, white_pvalue, _, _ = het_white(residuals, X)
            hetero_results['white'] = {
                'statistic': float(white_stat),
                'pvalue': float(white_pvalue),
                'interpretation': 'Homoskedastic' if white_pvalue > self.diagnostic_thresholds['heteroskedasticity_pvalue'] else 'Heteroskedastic'
            }
        except Exception as e:
            logger.warning(f"White test failed: {e}")
            hetero_results['white'] = {'error': str(e)}
        
        return hetero_results
    
    def _test_autocorrelation(self, residuals: np.ndarray) -> Dict[str, Any]:
        """Test for autocorrelation in residuals."""
        autocorr_results = {}
        
        # Ljung-Box test
        try:
            from statsmodels.stats.diagnostic import acorr_ljungbox
            
            lb_results = acorr_ljungbox(residuals, lags=min(10, len(residuals)//4), 
                                       return_df=True)
            
            # Take the maximum p-value across lags
            min_pvalue = lb_results['lb_pvalue'].min()
            
            autocorr_results['ljung_box'] = {
                'min_pvalue': float(min_pvalue),
                'interpretation': 'No autocorrelation' if min_pvalue > self.diagnostic_thresholds['autocorrelation_threshold'] else 'Autocorrelation present'
            }
        except Exception as e:
            logger.warning(f"Ljung-Box test failed: {e}")
            autocorr_results['ljung_box'] = {'error': str(e)}
        
        # Durbin-Watson test
        try:
            from statsmodels.stats.stattools import durbin_watson
            
            dw_stat = durbin_watson(residuals)
            autocorr_results['durbin_watson'] = {
                'statistic': float(dw_stat),
                'interpretation': self._interpret_durbin_watson(dw_stat)
            }
        except Exception as e:
            logger.warning(f"Durbin-Watson test failed: {e}")
            autocorr_results['durbin_watson'] = {'error': str(e)}
        
        return autocorr_results
    
    def _interpret_durbin_watson(self, dw_stat: float) -> str:
        """Interpret Durbin-Watson statistic."""
        if dw_stat < 1.5:
            return "Positive autocorrelation"
        elif dw_stat > 2.5:
            return "Negative autocorrelation"
        else:
            return "No significant autocorrelation"
    
    def _detect_outliers(self, residuals: np.ndarray) -> Dict[str, Any]:
        """Detect outliers in residuals."""
        outlier_results = {}
        
        # Z-score method
        z_scores = np.abs(stats.zscore(residuals))
        outlier_indices = np.where(z_scores > self.diagnostic_thresholds['outlier_threshold'])[0]
        
        outlier_results['z_score'] = {
            'threshold': self.diagnostic_thresholds['outlier_threshold'],
            'n_outliers': len(outlier_indices),
            'outlier_percentage': (len(outlier_indices) / len(residuals)) * 100,
            'outlier_indices': outlier_indices.tolist()
        }
        
        # IQR method
        Q1 = np.percentile(residuals, 25)
        Q3 = np.percentile(residuals, 75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        iqr_outliers = np.where((residuals < lower_bound) | (residuals > upper_bound))[0]
        
        outlier_results['iqr'] = {
            'lower_bound': float(lower_bound),
            'upper_bound': float(upper_bound),
            'n_outliers': len(iqr_outliers),
            'outlier_percentage': (len(iqr_outliers) / len(residuals)) * 100,
            'outlier_indices': iqr_outliers.tolist()
        }
        
        return outlier_results
    
    def _test_stationarity(self, residuals: np.ndarray) -> Dict[str, Any]:
        """Test stationarity of residuals using ADF test."""
        stationarity_results = {}
        
        try:
            adf_stat, adf_pvalue, _, _, critical_values, _ = adfuller(residuals)
            
            stationarity_results['augmented_dickey_fuller'] = {
                'statistic': float(adf_stat),
                'pvalue': float(adf_pvalue),
                'critical_values': {k: float(v) for k, v in critical_values.items()},
                'interpretation': 'Stationary' if adf_pvalue < self.diagnostic_thresholds['stationarity_pvalue'] else 'Non-stationary'
            }
        except Exception as e:
            logger.warning(f"ADF test failed: {e}")
            stationarity_results['augmented_dickey_fuller'] = {'error': str(e)}
        
        return stationarity_results 
   
    def _interpret_diagnostics(self, test_results: Dict[str, Any]) -> Dict[str, str]:
        """Generate interpretation of diagnostic test results."""
        interpretations = {}
        
        # Overall model health assessment
        issues = []
        
        # Check normality
        normality_tests = test_results.get('normality', {})
        normality_issues = []
        for test_name, test_result in normality_tests.items():
            if isinstance(test_result, dict) and 'interpretation' in test_result:
                if test_result['interpretation'] == 'Non-normal':
                    normality_issues.append(test_name)
        
        if normality_issues:
            issues.append(f"Residuals show non-normality ({', '.join(normality_issues)})")
            interpretations['normality'] = "Residuals deviate from normality. Consider robust standard errors or transformation."
        else:
            interpretations['normality'] = "Residuals appear normally distributed."
        
        # Check heteroskedasticity
        hetero_tests = test_results.get('heteroskedasticity', {})
        hetero_issues = []
        for test_name, test_result in hetero_tests.items():
            if isinstance(test_result, dict) and 'interpretation' in test_result:
                if test_result['interpretation'] == 'Heteroskedastic':
                    hetero_issues.append(test_name)
        
        if hetero_issues:
            issues.append(f"Heteroskedasticity detected ({', '.join(hetero_issues)})")
            interpretations['heteroskedasticity'] = "Heteroskedasticity present. Consider robust standard errors or weighted least squares."
        else:
            interpretations['heteroskedasticity'] = "Homoskedasticity assumption satisfied."
        
        # Check autocorrelation
        autocorr_tests = test_results.get('autocorrelation', {})
        autocorr_issues = []
        for test_name, test_result in autocorr_tests.items():
            if isinstance(test_result, dict) and 'interpretation' in test_result:
                if 'autocorrelation' in test_result['interpretation'].lower() and 'no' not in test_result['interpretation'].lower():
                    autocorr_issues.append(test_name)
        
        if autocorr_issues:
            issues.append(f"Autocorrelation detected ({', '.join(autocorr_issues)})")
            interpretations['autocorrelation'] = "Autocorrelation present. Consider HAC standard errors or AR terms."
        else:
            interpretations['autocorrelation'] = "No significant autocorrelation detected."
        
        # Check outliers
        outlier_results = test_results.get('outliers', {})
        high_outlier_methods = []
        for method, outlier_data in outlier_results.items():
            if isinstance(outlier_data, dict) and 'outlier_percentage' in outlier_data:
                if outlier_data['outlier_percentage'] > 5:  # More than 5% outliers
                    high_outlier_methods.append(method)
        
        if high_outlier_methods:
            issues.append(f"High outlier percentage ({', '.join(high_outlier_methods)})")
            interpretations['outliers'] = "Significant outliers detected. Consider robust regression or outlier treatment."
        else:
            interpretations['outliers'] = "Outlier levels within acceptable range."
        
        # Overall assessment
        if not issues:
            interpretations['overall'] = "Model diagnostics look good. No major assumption violations detected."
        else:
            interpretations['overall'] = f"Model diagnostics reveal {len(issues)} potential issues: {'; '.join(issues)}."
        
        return interpretations
    
    def _save_diagnostic_summary(self, diagnostic_results: Dict[str, Any], 
                                model_name: str):
        """Save diagnostic summary to JSON file."""
        summary_path = self.output_dir / "reports" / f"{model_name}_diagnostic_summary.json"
        
        # Create serializable summary
        summary = {
            'model_name': diagnostic_results['model_name'],
            'diagnostic_plots': diagnostic_results['diagnostic_plots'],
            'statistical_tests': diagnostic_results['statistical_tests'],
            'interpretation': diagnostic_results['interpretation'],
            'metadata': self.report_metadata
        }
        
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        logger.info(f"Diagnostic summary saved to {summary_path}")
    
    def generate_robustness_report(self, robustness_results: Dict[str, Any]) -> str:
        """
        Generate comprehensive robustness test report.
        
        Parameters:
        -----------
        robustness_results : Dict[str, Any]
            Results from robustness testing framework
            
        Returns:
        --------
        str
            Path to generated robustness report
        """
        logger.info("Generating comprehensive robustness report")
        
        # Create robustness summary
        robustness_summary = {
            'sensitivity_analysis': self._summarize_sensitivity_tests(
                robustness_results.get('sensitivity_analysis', {})
            ),
            'bootstrap_results': self._summarize_bootstrap_results(
                robustness_results.get('bootstrap_results', {})
            ),
            'alternative_specifications': self._summarize_alternative_specs(
                robustness_results.get('alternative_specifications', {})
            ),
            'cross_validation': self._summarize_cross_validation(
                robustness_results.get('cross_validation', {})
            ),
            'stability_tests': self._summarize_stability_tests(
                robustness_results.get('stability_tests', {})
            )
        }
        
        # Generate robustness plots
        robustness_plots = self._create_robustness_plots(robustness_results)
        
        # Create comprehensive report
        report_content = self._create_robustness_report_content(
            robustness_summary, robustness_plots
        )
        
        # Save report
        report_path = self.output_dir / "reports" / "comprehensive_robustness_report.md"
        with open(report_path, 'w') as f:
            f.write(report_content)
        
        # Save JSON summary
        json_path = self.output_dir / "reports" / "robustness_summary.json"
        with open(json_path, 'w') as f:
            json.dump(robustness_summary, f, indent=2, default=str)
        
        logger.info(f"Robustness report saved to {report_path}")
        return str(report_path)
    
    def _summarize_sensitivity_tests(self, sensitivity_results: Dict[str, Any]) -> Dict[str, Any]:
        """Summarize sensitivity analysis results."""
        summary = {
            'tests_performed': list(sensitivity_results.keys()),
            'stability_assessment': {},
            'key_findings': []
        }
        
        for test_name, test_results in sensitivity_results.items():
            if isinstance(test_results, dict):
                baseline = test_results.get('baseline', 0)
                alternatives = [test_results.get(f'alternative{i}', 0) for i in range(1, 4)]
                
                # Calculate coefficient of variation
                all_values = [baseline] + [alt for alt in alternatives if alt != 0]
                if len(all_values) > 1:
                    cv = np.std(all_values) / np.abs(np.mean(all_values)) if np.mean(all_values) != 0 else np.inf
                    
                    stability_level = "High" if cv < 0.1 else "Medium" if cv < 0.3 else "Low"
                    summary['stability_assessment'][test_name] = {
                        'coefficient_of_variation': cv,
                        'stability_level': stability_level,
                        'range': [min(all_values), max(all_values)]
                    }
                    
                    if stability_level == "Low":
                        summary['key_findings'].append(
                            f"{test_name}: Low stability (CV={cv:.3f}), results vary significantly across specifications"
                        )
        
        return summary
    
    def _summarize_bootstrap_results(self, bootstrap_results: Dict[str, Any]) -> Dict[str, Any]:
        """Summarize bootstrap results."""
        summary = {
            'parameters_tested': list(bootstrap_results.keys()),
            'confidence_intervals': {},
            'bias_assessment': {},
            'key_findings': []
        }
        
        for param_name, boot_results in bootstrap_results.items():
            if isinstance(boot_results, dict):
                point_est = boot_results.get('point_estimate', 0)
                ci_lower = boot_results.get('ci_lower', 0)
                ci_upper = boot_results.get('ci_upper', 0)
                bias = boot_results.get('bias', 0)
                
                summary['confidence_intervals'][param_name] = {
                    'point_estimate': point_est,
                    'ci_lower': ci_lower,
                    'ci_upper': ci_upper,
                    'ci_width': ci_upper - ci_lower
                }
                
                # Assess bias
                relative_bias = abs(bias / point_est) if point_est != 0 else np.inf
                bias_level = "Low" if relative_bias < 0.05 else "Medium" if relative_bias < 0.15 else "High"
                
                summary['bias_assessment'][param_name] = {
                    'absolute_bias': bias,
                    'relative_bias': relative_bias,
                    'bias_level': bias_level
                }
                
                if bias_level == "High":
                    summary['key_findings'].append(
                        f"{param_name}: High bias detected (relative bias={relative_bias:.3f})"
                    )
        
        return summary
    
    def _summarize_alternative_specs(self, alt_spec_results: Dict[str, Any]) -> Dict[str, Any]:
        """Summarize alternative specification results."""
        summary = {
            'specifications_tested': list(alt_spec_results.keys()),
            'consistency_assessment': {},
            'key_findings': []
        }
        
        # Compare results across specifications
        if len(alt_spec_results) > 1:
            for param_name in ['main_coefficient', 'threshold_value', 'r_squared']:
                param_values = []
                for spec_name, spec_results in alt_spec_results.items():
                    if isinstance(spec_results, dict) and param_name in spec_results:
                        param_values.append(spec_results[param_name])
                
                if len(param_values) > 1:
                    cv = np.std(param_values) / np.abs(np.mean(param_values)) if np.mean(param_values) != 0 else np.inf
                    consistency_level = "High" if cv < 0.1 else "Medium" if cv < 0.3 else "Low"
                    
                    summary['consistency_assessment'][param_name] = {
                        'coefficient_of_variation': cv,
                        'consistency_level': consistency_level,
                        'values_across_specs': param_values
                    }
                    
                    if consistency_level == "Low":
                        summary['key_findings'].append(
                            f"{param_name}: Low consistency across specifications (CV={cv:.3f})"
                        )
        
        return summary
    
    def _summarize_cross_validation(self, cv_results: Dict[str, Any]) -> Dict[str, Any]:
        """Summarize cross-validation results."""
        summary = {
            'cv_method': cv_results.get('method', 'Unknown'),
            'n_folds': cv_results.get('n_folds', 0),
            'performance_metrics': {},
            'stability_assessment': {},
            'key_findings': []
        }
        
        # Summarize performance across folds
        for metric in ['rmse', 'mae', 'r_squared']:
            fold_values = cv_results.get(f'{metric}_folds', [])
            if fold_values:
                summary['performance_metrics'][metric] = {
                    'mean': np.mean(fold_values),
                    'std': np.std(fold_values),
                    'min': np.min(fold_values),
                    'max': np.max(fold_values)
                }
                
                # Assess stability
                cv_metric = np.std(fold_values) / np.abs(np.mean(fold_values)) if np.mean(fold_values) != 0 else np.inf
                stability_level = "High" if cv_metric < 0.1 else "Medium" if cv_metric < 0.3 else "Low"
                
                summary['stability_assessment'][metric] = {
                    'coefficient_of_variation': cv_metric,
                    'stability_level': stability_level
                }
                
                if stability_level == "Low":
                    summary['key_findings'].append(
                        f"{metric}: Low stability across CV folds (CV={cv_metric:.3f})"
                    )
        
        return summary
    
    def _summarize_stability_tests(self, stability_results: Dict[str, Any]) -> Dict[str, Any]:
        """Summarize stability test results."""
        summary = {
            'tests_performed': list(stability_results.keys()),
            'stability_periods': {},
            'break_points': {},
            'key_findings': []
        }
        
        for test_name, test_results in stability_results.items():
            if isinstance(test_results, dict):
                # Structural break tests
                if 'break_points' in test_results:
                    break_points = test_results['break_points']
                    summary['break_points'][test_name] = {
                        'n_breaks': len(break_points),
                        'break_dates': break_points,
                        'significance': test_results.get('break_significance', [])
                    }
                    
                    if len(break_points) > 0:
                        summary['key_findings'].append(
                            f"{test_name}: {len(break_points)} structural breaks detected"
                        )
                
                # Rolling window stability
                if 'rolling_coefficients' in test_results:
                    rolling_coefs = test_results['rolling_coefficients']
                    if rolling_coefs:
                        cv = np.std(rolling_coefs) / np.abs(np.mean(rolling_coefs)) if np.mean(rolling_coefs) != 0 else np.inf
                        stability_level = "High" if cv < 0.2 else "Medium" if cv < 0.5 else "Low"
                        
                        summary['stability_periods'][test_name] = {
                            'coefficient_of_variation': cv,
                            'stability_level': stability_level
                        }
                        
                        if stability_level == "Low":
                            summary['key_findings'].append(
                                f"{test_name}: Low parameter stability over time (CV={cv:.3f})"
                            )
        
        return summary
    
    def _create_robustness_plots(self, robustness_results: Dict[str, Any]) -> Dict[str, str]:
        """Create robustness visualization plots."""
        plot_paths = {}
        
        # Sensitivity analysis plot
        sensitivity_results = robustness_results.get('sensitivity_analysis', {})
        if sensitivity_results:
            plot_paths['sensitivity'] = self._plot_sensitivity_analysis(sensitivity_results)
        
        # Bootstrap results plot
        bootstrap_results = robustness_results.get('bootstrap_results', {})
        if bootstrap_results:
            plot_paths['bootstrap'] = self._plot_bootstrap_results(bootstrap_results)
        
        # Cross-validation plot
        cv_results = robustness_results.get('cross_validation', {})
        if cv_results:
            plot_paths['cross_validation'] = self._plot_cross_validation_results(cv_results)
        
        # Stability plot
        stability_results = robustness_results.get('stability_tests', {})
        if stability_results:
            plot_paths['stability'] = self._plot_stability_results(stability_results)
        
        return plot_paths
    
    def _plot_sensitivity_analysis(self, sensitivity_results: Dict[str, Any]) -> str:
        """Create sensitivity analysis visualization."""
        fig, ax = plt.subplots(figsize=(12, 8))
        
        test_names = []
        baseline_values = []
        alt_values = []
        
        for test_name, test_results in sensitivity_results.items():
            if isinstance(test_results, dict):
                test_names.append(test_name.replace('_', ' ').title())
                baseline_values.append(test_results.get('baseline', 0))
                
                # Collect alternative values
                alternatives = []
                for i in range(1, 4):
                    alt_val = test_results.get(f'alternative{i}', None)
                    if alt_val is not None:
                        alternatives.append(alt_val)
                alt_values.append(alternatives)
        
        if test_names and baseline_values:
            y_pos = np.arange(len(test_names))
            
            # Plot baseline values
            ax.barh(y_pos, baseline_values, alpha=0.7, label='Baseline', color='blue')
            
            # Plot alternative values as scatter points
            for i, alternatives in enumerate(alt_values):
                for j, alt_val in enumerate(alternatives):
                    ax.scatter(alt_val, i, color='red', alpha=0.7, s=50, 
                             marker='o' if j == 0 else 's' if j == 1 else '^')
            
            ax.set_yticks(y_pos)
            ax.set_yticklabels(test_names)
            ax.set_xlabel('Coefficient Value')
            ax.set_title('Sensitivity Analysis: Baseline vs Alternative Specifications')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plot_path = self.output_dir / "robustness" / "sensitivity_analysis.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return str(plot_path)
    
    def _plot_bootstrap_results(self, bootstrap_results: Dict[str, Any]) -> str:
        """Create bootstrap results visualization."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        param_names = []
        point_estimates = []
        ci_lowers = []
        ci_uppers = []
        biases = []
        
        for param_name, boot_results in bootstrap_results.items():
            if isinstance(boot_results, dict):
                param_names.append(param_name.replace('_', ' ').title())
                point_estimates.append(boot_results.get('point_estimate', 0))
                ci_lowers.append(boot_results.get('ci_lower', 0))
                ci_uppers.append(boot_results.get('ci_upper', 0))
                biases.append(boot_results.get('bias', 0))
        
        if param_names:
            y_pos = np.arange(len(param_names))
            
            # Plot 1: Confidence intervals
            ax1.errorbar(point_estimates, y_pos, 
                        xerr=[np.array(point_estimates) - np.array(ci_lowers),
                              np.array(ci_uppers) - np.array(point_estimates)],
                        fmt='o', capsize=5, capthick=2)
            ax1.set_yticks(y_pos)
            ax1.set_yticklabels(param_names)
            ax1.set_xlabel('Parameter Value')
            ax1.set_title('Bootstrap Confidence Intervals')
            ax1.grid(True, alpha=0.3)
            ax1.axvline(x=0, color='red', linestyle='--', alpha=0.5)
            
            # Plot 2: Bias assessment
            colors = ['green' if abs(b) < 0.01 else 'orange' if abs(b) < 0.05 else 'red' for b in biases]
            ax2.barh(y_pos, biases, color=colors, alpha=0.7)
            ax2.set_yticks(y_pos)
            ax2.set_yticklabels(param_names)
            ax2.set_xlabel('Bootstrap Bias')
            ax2.set_title('Bootstrap Bias Assessment')
            ax2.grid(True, alpha=0.3)
            ax2.axvline(x=0, color='black', linestyle='-', alpha=0.5)
        
        plt.tight_layout()
        
        plot_path = self.output_dir / "robustness" / "bootstrap_results.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return str(plot_path)
    
    def _plot_cross_validation_results(self, cv_results: Dict[str, Any]) -> str:
        """Create cross-validation results visualization."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes = axes.flatten()
        
        metrics = ['rmse', 'mae', 'r_squared']
        
        for i, metric in enumerate(metrics):
            if i < len(axes):
                fold_values = cv_results.get(f'{metric}_folds', [])
                if fold_values:
                    axes[i].boxplot(fold_values)
                    axes[i].set_title(f'{metric.upper()} Across CV Folds')
                    axes[i].set_ylabel(metric.upper())
                    axes[i].grid(True, alpha=0.3)
                    
                    # Add mean line
                    mean_val = np.mean(fold_values)
                    axes[i].axhline(y=mean_val, color='red', linestyle='--', 
                                   label=f'Mean: {mean_val:.4f}')
                    axes[i].legend()
        
        # Learning curve (if available)
        if 'learning_curve' in cv_results:
            learning_data = cv_results['learning_curve']
            train_sizes = learning_data.get('train_sizes', [])
            train_scores = learning_data.get('train_scores', [])
            val_scores = learning_data.get('val_scores', [])
            
            if train_sizes and train_scores and val_scores:
                axes[3].plot(train_sizes, np.mean(train_scores, axis=1), 'o-', 
                           label='Training Score')
                axes[3].plot(train_sizes, np.mean(val_scores, axis=1), 'o-', 
                           label='Validation Score')
                axes[3].fill_between(train_sizes, 
                                   np.mean(train_scores, axis=1) - np.std(train_scores, axis=1),
                                   np.mean(train_scores, axis=1) + np.std(train_scores, axis=1),
                                   alpha=0.3)
                axes[3].fill_between(train_sizes, 
                                   np.mean(val_scores, axis=1) - np.std(val_scores, axis=1),
                                   np.mean(val_scores, axis=1) + np.std(val_scores, axis=1),
                                   alpha=0.3)
                axes[3].set_xlabel('Training Set Size')
                axes[3].set_ylabel('Score')
                axes[3].set_title('Learning Curve')
                axes[3].legend()
                axes[3].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        plot_path = self.output_dir / "robustness" / "cross_validation_results.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return str(plot_path)
    
    def _plot_stability_results(self, stability_results: Dict[str, Any]) -> str:
        """Create stability test visualization."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes = axes.flatten()
        
        plot_idx = 0
        
        for test_name, test_results in stability_results.items():
            if plot_idx >= len(axes):
                break
                
            if isinstance(test_results, dict):
                # Rolling coefficients plot
                if 'rolling_coefficients' in test_results:
                    rolling_coefs = test_results['rolling_coefficients']
                    rolling_dates = test_results.get('rolling_dates', list(range(len(rolling_coefs))))
                    
                    axes[plot_idx].plot(rolling_dates, rolling_coefs, linewidth=2)
                    axes[plot_idx].set_title(f'Rolling Coefficients: {test_name.replace("_", " ").title()}')
                    axes[plot_idx].set_ylabel('Coefficient Value')
                    axes[plot_idx].grid(True, alpha=0.3)
                    
                    # Add confidence bands if available
                    if 'rolling_ci_lower' in test_results and 'rolling_ci_upper' in test_results:
                        ci_lower = test_results['rolling_ci_lower']
                        ci_upper = test_results['rolling_ci_upper']
                        axes[plot_idx].fill_between(rolling_dates, ci_lower, ci_upper, 
                                                   alpha=0.3, label='95% CI')
                        axes[plot_idx].legend()
                    
                    plot_idx += 1
        
        plt.tight_layout()
        
        plot_path = self.output_dir / "robustness" / "stability_results.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return str(plot_path)
    
    def _create_robustness_report_content(self, robustness_summary: Dict[str, Any],
                                         robustness_plots: Dict[str, str]) -> str:
        """Create comprehensive robustness report content."""
        report_lines = [
            "# Comprehensive Robustness Testing Report",
            "",
            f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "## Executive Summary",
            "",
            "This report presents the results of comprehensive robustness testing for the QE hypothesis testing framework.",
            "The analysis includes sensitivity analysis, bootstrap validation, alternative specifications, cross-validation,",
            "and stability testing to assess the reliability and robustness of the main findings.",
            "",
        ]
        
        # Add sensitivity analysis section
        if 'sensitivity_analysis' in robustness_summary:
            sens_summary = robustness_summary['sensitivity_analysis']
            report_lines.extend([
                "## Sensitivity Analysis",
                "",
                f"**Tests Performed:** {len(sens_summary.get('tests_performed', []))}",
                "",
                "### Key Findings:",
                ""
            ])
            
            for finding in sens_summary.get('key_findings', []):
                report_lines.append(f"- {finding}")
            
            report_lines.extend(["", "### Stability Assessment:", ""])
            
            for test_name, stability in sens_summary.get('stability_assessment', {}).items():
                stability_level = stability.get('stability_level', 'Unknown')
                cv = stability.get('coefficient_of_variation', 0)
                report_lines.append(f"- **{test_name}**: {stability_level} stability (CV: {cv:.3f})")
            
            if 'sensitivity' in robustness_plots:
                report_lines.extend([
                    "",
                    f"![Sensitivity Analysis]({robustness_plots['sensitivity']})",
                    ""
                ])
        
        # Add bootstrap section
        if 'bootstrap_results' in robustness_summary:
            boot_summary = robustness_summary['bootstrap_results']
            report_lines.extend([
                "## Bootstrap Validation",
                "",
                f"**Parameters Tested:** {len(boot_summary.get('parameters_tested', []))}",
                "",
                "### Bias Assessment:",
                ""
            ])
            
            for param, bias_info in boot_summary.get('bias_assessment', {}).items():
                bias_level = bias_info.get('bias_level', 'Unknown')
                rel_bias = bias_info.get('relative_bias', 0)
                report_lines.append(f"- **{param}**: {bias_level} bias (Relative: {rel_bias:.3f})")
            
            if 'bootstrap' in robustness_plots:
                report_lines.extend([
                    "",
                    f"![Bootstrap Results]({robustness_plots['bootstrap']})",
                    ""
                ])
        
        # Add cross-validation section
        if 'cross_validation' in robustness_summary:
            cv_summary = robustness_summary['cross_validation']
            report_lines.extend([
                "## Cross-Validation Results",
                "",
                f"**Method:** {cv_summary.get('cv_method', 'Unknown')}",
                f"**Number of Folds:** {cv_summary.get('n_folds', 0)}",
                "",
                "### Performance Metrics:",
                ""
            ])
            
            for metric, perf_info in cv_summary.get('performance_metrics', {}).items():
                mean_val = perf_info.get('mean', 0)
                std_val = perf_info.get('std', 0)
                report_lines.append(f"- **{metric.upper()}**: {mean_val:.4f} ± {std_val:.4f}")
            
            if 'cross_validation' in robustness_plots:
                report_lines.extend([
                    "",
                    f"![Cross-Validation Results]({robustness_plots['cross_validation']})",
                    ""
                ])
        
        # Add stability section
        if 'stability_tests' in robustness_summary:
            stab_summary = robustness_summary['stability_tests']
            report_lines.extend([
                "## Stability Testing",
                "",
                f"**Tests Performed:** {len(stab_summary.get('tests_performed', []))}",
                "",
                "### Key Findings:",
                ""
            ])
            
            for finding in stab_summary.get('key_findings', []):
                report_lines.append(f"- {finding}")
            
            if 'stability' in robustness_plots:
                report_lines.extend([
                    "",
                    f"![Stability Results]({robustness_plots['stability']})",
                    ""
                ])
        
        # Add conclusions
        report_lines.extend([
            "## Conclusions and Recommendations",
            "",
            "Based on the comprehensive robustness testing:",
            "",
            "1. **Model Reliability**: [Assessment based on test results]",
            "2. **Key Vulnerabilities**: [Identified weaknesses]",
            "3. **Recommended Actions**: [Suggested improvements]",
            "",
            "---",
            "",
            "*This report was automatically generated by the QE Hypothesis Testing Framework.*"
        ])
        
        return "\n".join(report_lines)
    
    def create_structured_output_directory(self, results: Dict[str, Any]) -> str:
        """
        Create structured output directory with clear file organization.
        
        Parameters:
        -----------
        results : Dict[str, Any]
            Complete results from hypothesis testing framework
            
        Returns:
        --------
        str
            Path to structured output directory
        """
        logger.info("Creating structured output directory")
        
        # Create main output structure
        output_structure = {
            'tables': ['latex', 'csv', 'excel'],
            'figures': ['publication', 'presentation', 'diagnostic'],
            'reports': ['diagnostic', 'robustness', 'summary'],
            'data': ['processed', 'results', 'metadata'],
            'code': ['scripts', 'notebooks', 'config']
        }
        
        # Create directory structure
        for main_dir, subdirs in output_structure.items():
            main_path = self.output_dir / main_dir
            main_path.mkdir(exist_ok=True)
            
            for subdir in subdirs:
                (main_path / subdir).mkdir(exist_ok=True)
        
        # Create README file
        readme_content = self._create_output_readme(output_structure)
        readme_path = self.output_dir / "README.md"
        with open(readme_path, 'w') as f:
            f.write(readme_content)
        
        # Create file index
        file_index = self._create_file_index(results)
        index_path = self.output_dir / "file_index.json"
        with open(index_path, 'w') as f:
            json.dump(file_index, f, indent=2, default=str)
        
        logger.info(f"Structured output directory created at {self.output_dir}")
        return str(self.output_dir)
    
    def _create_output_readme(self, output_structure: Dict[str, List[str]]) -> str:
        """Create README file for output directory."""
        readme_lines = [
            "# QE Hypothesis Testing Framework - Output Directory",
            "",
            f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "## Directory Structure",
            "",
        ]
        
        for main_dir, subdirs in output_structure.items():
            readme_lines.append(f"### {main_dir.title()}/")
            readme_lines.append("")
            
            for subdir in subdirs:
                readme_lines.append(f"- **{subdir}/**: {self._get_directory_description(main_dir, subdir)}")
            
            readme_lines.append("")
        
        readme_lines.extend([
            "## File Naming Conventions",
            "",
            "- **Tables**: `hypothesis{N}_{description}.{format}`",
            "- **Figures**: `hypothesis{N}_{plot_type}.{format}`",
            "- **Reports**: `{report_type}_report.{format}`",
            "- **Data**: `{data_type}_{timestamp}.{format}`",
            "",
            "## Usage Instructions",
            "",
            "1. **Publication Materials**: Use files in `tables/latex/` and `figures/publication/`",
            "2. **Diagnostic Review**: Check `reports/diagnostic/` for model validation",
            "3. **Robustness Assessment**: Review `reports/robustness/` for stability analysis",
            "4. **Data Access**: Processed data available in `data/processed/`",
            "",
            "## Contact",
            "",
            "For questions about this output, refer to the QE Hypothesis Testing Framework documentation.",
        ])
        
        return "\n".join(readme_lines)
    
    def _get_directory_description(self, main_dir: str, subdir: str) -> str:
        """Get description for directory structure."""
        descriptions = {
            ('tables', 'latex'): "LaTeX formatted tables for publication",
            ('tables', 'csv'): "CSV format tables for data analysis",
            ('tables', 'excel'): "Excel format tables for presentation",
            ('figures', 'publication'): "High-resolution figures for journal submission",
            ('figures', 'presentation'): "Presentation-ready figures",
            ('figures', 'diagnostic'): "Model diagnostic plots",
            ('reports', 'diagnostic'): "Model diagnostic reports and summaries",
            ('reports', 'robustness'): "Robustness testing reports",
            ('reports', 'summary'): "Executive summaries and key findings",
            ('data', 'processed'): "Cleaned and processed datasets",
            ('data', 'results'): "Model results and estimates",
            ('data', 'metadata'): "Data documentation and metadata",
            ('code', 'scripts'): "Analysis scripts and utilities",
            ('code', 'notebooks'): "Jupyter notebooks for exploration",
            ('code', 'config'): "Configuration files and parameters"
        }
        
        return descriptions.get((main_dir, subdir), "Additional files and outputs")
    
    def _create_file_index(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Create comprehensive file index."""
        file_index = {
            'created_at': datetime.now().isoformat(),
            'total_files': 0,
            'file_categories': {},
            'hypothesis_files': {},
            'key_outputs': {}
        }
        
        # Scan output directory for files
        for file_path in self.output_dir.rglob('*'):
            if file_path.is_file():
                file_index['total_files'] += 1
                
                # Categorize by directory
                relative_path = file_path.relative_to(self.output_dir)
                category = str(relative_path.parts[0]) if relative_path.parts else 'root'
                
                if category not in file_index['file_categories']:
                    file_index['file_categories'][category] = []
                
                file_index['file_categories'][category].append({
                    'filename': file_path.name,
                    'path': str(relative_path),
                    'size_bytes': file_path.stat().st_size,
                    'modified': datetime.fromtimestamp(file_path.stat().st_mtime).isoformat()
                })
        
        return file_index


def create_comprehensive_diagnostics(results: Dict[str, Any], 
                                   output_dir: str = "output/diagnostics") -> Dict[str, str]:
    """
    Convenience function to generate comprehensive diagnostic reports.
    
    Parameters:
    -----------
    results : Dict[str, Any]
        Complete results dictionary from hypothesis testing framework
    output_dir : str
        Directory to save diagnostic outputs
        
    Returns:
    --------
    Dict[str, str]
        Dictionary mapping report types to file paths
    """
    reporter = DiagnosticReporter(output_dir)
    
    generated_reports = {}
    
    # Generate model diagnostics for each hypothesis
    for hypothesis_name in ['hypothesis1', 'hypothesis2', 'hypothesis3']:
        if f'{hypothesis_name}_results' in results:
            hypothesis_results = results[f'{hypothesis_name}_results']
            diagnostics = reporter.generate_model_diagnostics(
                hypothesis_results, f'{hypothesis_name}_model'
            )
            generated_reports[f'{hypothesis_name}_diagnostics'] = diagnostics
    
    # Generate robustness report
    if 'robustness_tests' in results:
        robustness_report = reporter.generate_robustness_report(
            results['robustness_tests']
        )
        generated_reports['robustness_report'] = robustness_report
    
    # Create structured output directory
    output_dir_path = reporter.create_structured_output_directory(results)
    generated_reports['output_directory'] = output_dir_path
    
    return generated_reports