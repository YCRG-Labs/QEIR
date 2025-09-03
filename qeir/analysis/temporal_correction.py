"""
Temporal Scope Correction System for QE Analysis

This module implements the TemporalScopeCorrector class to enforce proper temporal
boundaries for quantitative easing analysis, ensuring focus on the 2008-2024 period
and providing validation for any pre-QE data usage.
"""

import pandas as pd
import numpy as np
import warnings
from datetime import datetime, date
from typing import Dict, List, Tuple, Optional, Union
import logging

class TemporalScopeCorrector:
    """
    Enforces temporal consistency for QE analysis by restricting data to the
    relevant QE implementation period (2008-2024) and validating any historical
    data usage.
    
    This addresses Requirement 1.1: Data Temporal Scope Correction
    """
    
    def __init__(self, qe_start_date: str = '2008-11-01', qe_end_date: str = '2024-12-31'):
        """
        Initialize TemporalScopeCorrector with QE period boundaries.
        
        Args:
            qe_start_date: Start of QE implementation period (default: Nov 2008)
            qe_end_date: End of analysis period (default: Dec 2024)
        """
        self.qe_start_date = pd.to_datetime(qe_start_date)
        self.qe_end_date = pd.to_datetime(qe_end_date)
        self.pre_qe_cutoff = pd.to_datetime('2008-01-01')
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        
        # Validation flags
        self.validation_results = {}
        
    def validate_temporal_scope(self, data: pd.DataFrame, 
                              date_column: Optional[str] = None,
                              strict_mode: bool = True) -> Dict:
        """
        Validate temporal scope of dataset and flag any pre-QE data usage.
        
        Args:
            data: DataFrame with time-indexed data or date column
            date_column: Name of date column if not using index
            strict_mode: If True, raises warnings for pre-QE data
            
        Returns:
            Dictionary with validation results and recommendations
        """
        validation_report = {
            'total_observations': len(data),
            'date_range': None,
            'pre_qe_observations': 0,
            'qe_period_observations': 0,
            'post_qe_observations': 0,
            'temporal_consistency': True,
            'warnings': [],
            'recommendations': []
        }
        
        try:
            # Extract date information
            if date_column:
                dates = pd.to_datetime(data[date_column])
            else:
                if hasattr(data.index, 'to_pydatetime'):
                    dates = data.index
                else:
                    dates = pd.to_datetime(data.index)
            
            validation_report['date_range'] = (dates.min(), dates.max())
            
            # Categorize observations by period
            pre_qe_mask = dates < self.qe_start_date
            qe_period_mask = (dates >= self.qe_start_date) & (dates <= self.qe_end_date)
            post_qe_mask = dates > self.qe_end_date
            
            validation_report['pre_qe_observations'] = pre_qe_mask.sum()
            validation_report['qe_period_observations'] = qe_period_mask.sum()
            validation_report['post_qe_observations'] = post_qe_mask.sum()
            
            # Check for temporal consistency issues
            if validation_report['pre_qe_observations'] > 0:
                validation_report['temporal_consistency'] = False
                warning_msg = (f"Found {validation_report['pre_qe_observations']} observations "
                             f"before QE period ({self.qe_start_date.date()})")
                validation_report['warnings'].append(warning_msg)
                
                if strict_mode:
                    self.logger.warning(warning_msg)
                    
                validation_report['recommendations'].append(
                    "Consider removing pre-QE data or provide theoretical justification"
                )
            
            # Check if sufficient QE period data exists
            if validation_report['qe_period_observations'] < 100:
                warning_msg = (f"Only {validation_report['qe_period_observations']} observations "
                             f"in QE period - may be insufficient for robust analysis")
                validation_report['warnings'].append(warning_msg)
                validation_report['recommendations'].append(
                    "Consider extending data collection or using higher frequency data"
                )
            
            # Check for future data
            if validation_report['post_qe_observations'] > 0:
                info_msg = (f"Found {validation_report['post_qe_observations']} observations "
                          f"after analysis period ({self.qe_end_date.date()})")
                validation_report['warnings'].append(info_msg)
                
            self.validation_results = validation_report
            
        except Exception as e:
            validation_report['error'] = str(e)
            self.logger.error(f"Temporal validation failed: {e}")
            
        return validation_report
    
    def create_qe_focused_dataset(self, data: pd.DataFrame,
                                date_column: Optional[str] = None,
                                preserve_pre_qe: bool = False) -> Dict[str, pd.DataFrame]:
        """
        Create QE-focused dataset by filtering to 2008-2024 period.
        
        Args:
            data: Input DataFrame
            date_column: Name of date column if not using index
            preserve_pre_qe: If True, also return pre-QE data separately
            
        Returns:
            Dictionary containing filtered datasets
        """
        datasets = {}
        
        try:
            # Extract date information
            if date_column:
                dates = pd.to_datetime(data[date_column])
                data_with_dates = data.copy()
            else:
                if hasattr(data.index, 'to_pydatetime'):
                    dates = data.index
                else:
                    dates = pd.to_datetime(data.index)
                data_with_dates = data.copy()
            
            # Create QE period mask
            qe_period_mask = (dates >= self.qe_start_date) & (dates <= self.qe_end_date)
            
            # Filter to QE period
            if date_column:
                qe_focused_data = data_with_dates[qe_period_mask].copy()
            else:
                qe_focused_data = data_with_dates.loc[qe_period_mask].copy()
            
            datasets['qe_focused'] = qe_focused_data
            
            # Preserve pre-QE data if requested
            if preserve_pre_qe:
                pre_qe_mask = dates < self.qe_start_date
                if date_column:
                    pre_qe_data = data_with_dates[pre_qe_mask].copy()
                else:
                    pre_qe_data = data_with_dates.loc[pre_qe_mask].copy()
                datasets['pre_qe'] = pre_qe_data
            
            # Log dataset creation
            try:
                if hasattr(qe_focused_data.index, 'date'):
                    start_date = qe_focused_data.index.min().date()
                    end_date = qe_focused_data.index.max().date()
                else:
                    start_date = pd.to_datetime(qe_focused_data.index.min()).date()
                    end_date = pd.to_datetime(qe_focused_data.index.max()).date()
                
                self.logger.info(f"Created QE-focused dataset: {len(qe_focused_data)} observations "
                               f"from {start_date} to {end_date}")
            except:
                self.logger.info(f"Created QE-focused dataset: {len(qe_focused_data)} observations")
            
            if preserve_pre_qe and 'pre_qe' in datasets:
                self.logger.info(f"Preserved pre-QE dataset: {len(datasets['pre_qe'])} observations")
                
        except Exception as e:
            self.logger.error(f"Dataset creation failed: {e}")
            raise ValueError(f"Failed to create QE-focused dataset: {e}")
            
        return datasets
    
    def get_qe_period_info(self) -> Dict:
        """
        Get information about the defined QE period.
        
        Returns:
            Dictionary with QE period details
        """
        return {
            'qe_start_date': self.qe_start_date,
            'qe_end_date': self.qe_end_date,
            'qe_period_years': (self.qe_end_date - self.qe_start_date).days / 365.25,
            'pre_qe_cutoff': self.pre_qe_cutoff
        }
    
    def check_date_alignment(self, *datasets: pd.DataFrame) -> Dict:
        """
        Check temporal alignment across multiple datasets.
        
        Args:
            *datasets: Variable number of DataFrames to check
            
        Returns:
            Dictionary with alignment analysis
        """
        alignment_report = {
            'datasets_count': len(datasets),
            'common_date_range': None,
            'alignment_issues': [],
            'recommendations': []
        }
        
        if len(datasets) < 2:
            alignment_report['alignment_issues'].append("Need at least 2 datasets for alignment check")
            return alignment_report
        
        try:
            # Extract date ranges from each dataset
            date_ranges = []
            for i, df in enumerate(datasets):
                if hasattr(df.index, 'to_pydatetime'):
                    dates = df.index
                else:
                    dates = pd.to_datetime(df.index)
                date_ranges.append((dates.min(), dates.max()))
            
            # Find common date range
            common_start = max(dr[0] for dr in date_ranges)
            common_end = min(dr[1] for dr in date_ranges)
            
            if common_start <= common_end:
                alignment_report['common_date_range'] = (common_start, common_end)
                common_days = (common_end - common_start).days
                
                if common_days < 365:
                    alignment_report['alignment_issues'].append(
                        f"Common date range is only {common_days} days"
                    )
                    alignment_report['recommendations'].append(
                        "Consider extending data collection period"
                    )
            else:
                alignment_report['alignment_issues'].append("No overlapping date range found")
                alignment_report['recommendations'].append(
                    "Check data sources and collection periods"
                )
                
        except Exception as e:
            alignment_report['error'] = str(e)
            
        return alignment_report
    
    def pre_qe_data_validator(self, data: pd.DataFrame,
                            analysis_description: str = "QE analysis",
                            justification: Optional[str] = None,
                            date_column: Optional[str] = None) -> Dict:
        """
        Validate and flag pre-QE data usage with justification requirements.
        
        Args:
            data: DataFrame to validate
            analysis_description: Description of the analysis being performed
            justification: Theoretical/empirical justification for pre-QE data usage
            date_column: Name of date column if not using index
            
        Returns:
            Dictionary with validation results and warnings
        """
        validation_report = {
            'analysis_description': analysis_description,
            'pre_qe_data_present': False,
            'pre_qe_observations': 0,
            'pre_qe_percentage': 0.0,
            'justification_provided': justification is not None,
            'justification': justification,
            'warnings': [],
            'recommendations': [],
            'validation_status': 'PASS'
        }
        
        try:
            # Extract date information
            if date_column:
                dates = pd.to_datetime(data[date_column])
            else:
                if hasattr(data.index, 'to_pydatetime'):
                    dates = data.index
                else:
                    dates = pd.to_datetime(data.index)
            
            # Check for pre-QE data
            pre_qe_mask = dates < self.qe_start_date
            validation_report['pre_qe_observations'] = pre_qe_mask.sum()
            validation_report['pre_qe_percentage'] = (pre_qe_mask.sum() / len(data)) * 100
            validation_report['pre_qe_data_present'] = validation_report['pre_qe_observations'] > 0
            
            if validation_report['pre_qe_data_present']:
                # Generate warnings based on severity
                if validation_report['pre_qe_percentage'] > 50:
                    severity = "CRITICAL"
                    validation_report['validation_status'] = 'FAIL'
                elif validation_report['pre_qe_percentage'] > 25:
                    severity = "HIGH"
                    validation_report['validation_status'] = 'WARNING'
                else:
                    severity = "MEDIUM"
                    validation_report['validation_status'] = 'WARNING'
                
                warning_msg = (f"{severity}: {analysis_description} includes "
                             f"{validation_report['pre_qe_observations']} pre-QE observations "
                             f"({validation_report['pre_qe_percentage']:.1f}% of data)")
                validation_report['warnings'].append(warning_msg)
                
                # Check justification adequacy
                if not justification:
                    validation_report['warnings'].append(
                        "No theoretical justification provided for pre-QE data inclusion"
                    )
                    validation_report['recommendations'].append(
                        "Provide economic theory explaining relevance of pre-QE period to QE analysis"
                    )
                    validation_report['recommendations'].append(
                        "Consider restricting analysis to QE period (2008-2024) only"
                    )
                else:
                    # Validate justification quality
                    justification_quality = self._assess_justification_quality(justification)
                    if justification_quality['score'] < 0.6:
                        validation_report['warnings'].append(
                            f"Justification may be insufficient: {justification_quality['issues']}"
                        )
                        validation_report['recommendations'].extend(justification_quality['recommendations'])
                
                # Add robustness check recommendations
                validation_report['recommendations'].append(
                    "Conduct robustness checks excluding pre-QE data"
                )
                validation_report['recommendations'].append(
                    "Test temporal stability of relationships across pre-QE and QE periods"
                )
                
                # Log warning
                self.logger.warning(warning_msg)
                
        except Exception as e:
            validation_report['error'] = str(e)
            validation_report['validation_status'] = 'ERROR'
            self.logger.error(f"Pre-QE data validation failed: {e}")
        
        return validation_report
    
    def _assess_justification_quality(self, justification: str) -> Dict:
        """
        Assess the quality of theoretical justification for pre-QE data usage.
        
        Args:
            justification: Text justification provided
            
        Returns:
            Dictionary with quality assessment
        """
        assessment = {
            'score': 0.0,
            'issues': [],
            'recommendations': []
        }
        
        justification_lower = justification.lower()
        
        # Check for key theoretical concepts
        theoretical_keywords = [
            'structural', 'long-term', 'baseline', 'trend', 'equilibrium',
            'historical', 'comparison', 'benchmark', 'control', 'counterfactual'
        ]
        
        keyword_score = sum(1 for keyword in theoretical_keywords 
                          if keyword in justification_lower) / len(theoretical_keywords)
        
        # Check for economic reasoning
        economic_keywords = [
            'monetary policy', 'interest rate', 'financial crisis', 'recession',
            'economic cycle', 'market conditions', 'liquidity', 'credit'
        ]
        
        economic_score = sum(1 for keyword in economic_keywords 
                           if keyword in justification_lower) / len(economic_keywords)
        
        # Check for methodological considerations
        method_keywords = [
            'identification', 'causal', 'endogeneity', 'instrument', 'robustness',
            'specification', 'estimation', 'model'
        ]
        
        method_score = sum(1 for keyword in method_keywords 
                         if keyword in justification_lower) / len(method_keywords)
        
        # Calculate overall score
        assessment['score'] = (keyword_score * 0.3 + economic_score * 0.4 + method_score * 0.3)
        
        # Generate recommendations based on weaknesses
        if keyword_score < 0.3:
            assessment['issues'].append("Lacks clear theoretical framework")
            assessment['recommendations'].append(
                "Strengthen theoretical justification with economic theory"
            )
        
        if economic_score < 0.3:
            assessment['issues'].append("Limited economic reasoning")
            assessment['recommendations'].append(
                "Explain economic mechanisms linking pre-QE and QE periods"
            )
        
        if method_score < 0.3:
            assessment['issues'].append("Insufficient methodological consideration")
            assessment['recommendations'].append(
                "Address potential methodological issues from temporal heterogeneity"
            )
        
        if len(justification) < 100:
            assessment['issues'].append("Justification too brief")
            assessment['recommendations'].append(
                "Provide more detailed theoretical explanation"
            )
        
        return assessment
    
    def create_warning_system(self, warning_level: str = 'MEDIUM') -> Dict:
        """
        Create a systematic warning system for pre-QE data usage.
        
        Args:
            warning_level: Threshold for warnings ('LOW', 'MEDIUM', 'HIGH', 'CRITICAL')
            
        Returns:
            Dictionary with warning system configuration
        """
        warning_thresholds = {
            'LOW': {'pre_qe_pct': 10, 'min_justification_score': 0.3},
            'MEDIUM': {'pre_qe_pct': 25, 'min_justification_score': 0.5},
            'HIGH': {'pre_qe_pct': 50, 'min_justification_score': 0.7},
            'CRITICAL': {'pre_qe_pct': 75, 'min_justification_score': 0.8}
        }
        
        if warning_level not in warning_thresholds:
            warning_level = 'MEDIUM'
        
        system_config = {
            'warning_level': warning_level,
            'thresholds': warning_thresholds[warning_level],
            'required_checks': [
                'temporal_consistency_check',
                'justification_adequacy_check',
                'robustness_requirement_check'
            ],
            'automatic_actions': {
                'flag_pre_qe_usage': True,
                'require_justification': True,
                'suggest_robustness_checks': True,
                'log_warnings': True
            }
        }
        
        return system_config
    
    def batch_validate_analyses(self, analyses: Dict[str, Dict]) -> Dict:
        """
        Validate multiple analyses for pre-QE data usage.
        
        Args:
            analyses: Dictionary of analysis_name -> {data, description, justification}
            
        Returns:
            Dictionary with batch validation results
        """
        batch_results = {
            'total_analyses': len(analyses),
            'analyses_with_pre_qe': 0,
            'analyses_without_justification': 0,
            'critical_issues': 0,
            'individual_results': {},
            'summary_recommendations': []
        }
        
        for analysis_name, analysis_info in analyses.items():
            try:
                data = analysis_info['data']
                description = analysis_info.get('description', analysis_name)
                justification = analysis_info.get('justification', None)
                
                validation_result = self.pre_qe_data_validator(
                    data=data,
                    analysis_description=description,
                    justification=justification
                )
                
                batch_results['individual_results'][analysis_name] = validation_result
                
                # Update summary statistics
                if validation_result['pre_qe_data_present']:
                    batch_results['analyses_with_pre_qe'] += 1
                
                if validation_result['pre_qe_data_present'] and not validation_result['justification_provided']:
                    batch_results['analyses_without_justification'] += 1
                
                if validation_result['validation_status'] == 'FAIL':
                    batch_results['critical_issues'] += 1
                    
            except Exception as e:
                batch_results['individual_results'][analysis_name] = {
                    'error': str(e),
                    'validation_status': 'ERROR'
                }
        
        # Generate summary recommendations
        if batch_results['analyses_with_pre_qe'] > 0:
            batch_results['summary_recommendations'].append(
                f"{batch_results['analyses_with_pre_qe']} analyses include pre-QE data"
            )
        
        if batch_results['analyses_without_justification'] > 0:
            batch_results['summary_recommendations'].append(
                f"{batch_results['analyses_without_justification']} analyses lack justification for pre-QE data"
            )
        
        if batch_results['critical_issues'] > 0:
            batch_results['summary_recommendations'].append(
                f"{batch_results['critical_issues']} analyses have critical temporal scope issues"
            )
        
        return batch_results
    
    def subsample_temporal_robustness(self, data: pd.DataFrame,
                                    analysis_function: callable,
                                    start_dates: Optional[List[str]] = None,
                                    date_column: Optional[str] = None) -> Dict:
        """
        Test robustness of analysis results across different temporal starting points.
        
        Args:
            data: DataFrame with time series data
            analysis_function: Function that takes data and returns analysis results
            start_dates: List of alternative start dates to test
            date_column: Name of date column if not using index
            
        Returns:
            Dictionary with robustness test results
        """
        if start_dates is None:
            # Default alternative start dates for QE analysis
            start_dates = [
                '2007-01-01',  # Pre-crisis baseline
                '2008-01-01',  # Crisis start
                '2008-09-01',  # Lehman collapse
                '2008-11-01',  # QE1 start (default)
                '2009-01-01',  # Post-crisis
                '2010-01-01'   # Recovery period
            ]
        
        robustness_results = {
            'baseline_start_date': self.qe_start_date.strftime('%Y-%m-%d'),
            'alternative_start_dates': start_dates,
            'results_by_start_date': {},
            'stability_metrics': {},
            'robustness_summary': {}
        }
        
        try:
            # Extract date information
            if date_column:
                dates = pd.to_datetime(data[date_column])
                data_with_dates = data.copy()
            else:
                if hasattr(data.index, 'to_pydatetime'):
                    dates = data.index
                else:
                    dates = pd.to_datetime(data.index)
                data_with_dates = data.copy()
            
            # Run analysis for each start date
            baseline_result = None
            all_results = []
            
            for start_date_str in start_dates:
                start_date = pd.to_datetime(start_date_str)
                
                # Filter data from start date
                if date_column:
                    mask = pd.to_datetime(data_with_dates[date_column]) >= start_date
                    subset_data = data_with_dates[mask].copy()
                else:
                    mask = dates >= start_date
                    subset_data = data_with_dates.loc[mask].copy()
                
                if len(subset_data) < 100:  # Minimum observations
                    robustness_results['results_by_start_date'][start_date_str] = {
                        'error': 'Insufficient observations',
                        'observations': len(subset_data)
                    }
                    continue
                
                try:
                    # Run analysis function
                    result = analysis_function(subset_data)
                    robustness_results['results_by_start_date'][start_date_str] = {
                        'result': result,
                        'observations': len(subset_data),
                        'date_range': (subset_data.index.min(), subset_data.index.max())
                    }
                    
                    # Store baseline result (default QE start date)
                    if start_date_str == self.qe_start_date.strftime('%Y-%m-%d'):
                        baseline_result = result
                    
                    all_results.append(result)
                    
                except Exception as e:
                    robustness_results['results_by_start_date'][start_date_str] = {
                        'error': str(e),
                        'observations': len(subset_data)
                    }
            
            # Calculate stability metrics
            if len(all_results) >= 2:
                robustness_results['stability_metrics'] = self._calculate_stability_metrics(
                    all_results, baseline_result
                )
                
                # Generate robustness summary
                robustness_results['robustness_summary'] = self._generate_robustness_summary(
                    robustness_results['stability_metrics']
                )
                
        except Exception as e:
            robustness_results['error'] = str(e)
            self.logger.error(f"Temporal robustness testing failed: {e}")
        
        return robustness_results
    
    def _calculate_stability_metrics(self, all_results: List, baseline_result) -> Dict:
        """
        Calculate stability metrics across different temporal specifications.
        
        Args:
            all_results: List of analysis results from different start dates
            baseline_result: Baseline result for comparison
            
        Returns:
            Dictionary with stability metrics
        """
        stability_metrics = {
            'coefficient_stability': {},
            'significance_stability': {},
            'overall_stability_score': 0.0
        }
        
        try:
            # Extract coefficients if results are dictionaries with coefficient info
            if isinstance(baseline_result, dict) and 'coefficients' in baseline_result:
                baseline_coeffs = baseline_result['coefficients']
                
                # Collect coefficients from all results
                all_coeffs = []
                for result in all_results:
                    if isinstance(result, dict) and 'coefficients' in result:
                        all_coeffs.append(result['coefficients'])
                
                if len(all_coeffs) >= 2:
                    # Calculate coefficient stability
                    for coeff_name in baseline_coeffs.keys():
                        coeff_values = [coeffs.get(coeff_name, np.nan) for coeffs in all_coeffs]
                        coeff_values = [v for v in coeff_values if not np.isnan(v)]
                        
                        if len(coeff_values) >= 2:
                            stability_metrics['coefficient_stability'][coeff_name] = {
                                'mean': np.mean(coeff_values),
                                'std': np.std(coeff_values),
                                'cv': np.std(coeff_values) / abs(np.mean(coeff_values)) if np.mean(coeff_values) != 0 else np.inf,
                                'range': (np.min(coeff_values), np.max(coeff_values))
                            }
            
            # Extract significance results if available
            if isinstance(baseline_result, dict) and 'p_values' in baseline_result:
                baseline_pvals = baseline_result['p_values']
                
                # Collect p-values from all results
                all_pvals = []
                for result in all_results:
                    if isinstance(result, dict) and 'p_values' in result:
                        all_pvals.append(result['p_values'])
                
                if len(all_pvals) >= 2:
                    # Calculate significance stability
                    for var_name in baseline_pvals.keys():
                        pval_values = [pvals.get(var_name, np.nan) for pvals in all_pvals]
                        pval_values = [v for v in pval_values if not np.isnan(v)]
                        
                        if len(pval_values) >= 2:
                            significant_count = sum(1 for p in pval_values if p < 0.05)
                            stability_metrics['significance_stability'][var_name] = {
                                'significance_rate': significant_count / len(pval_values),
                                'mean_p_value': np.mean(pval_values),
                                'p_value_range': (np.min(pval_values), np.max(pval_values))
                            }
            
            # Calculate overall stability score
            coeff_scores = []
            if stability_metrics['coefficient_stability']:
                for coeff_name, metrics in stability_metrics['coefficient_stability'].items():
                    # Lower coefficient of variation indicates higher stability
                    cv = metrics['cv']
                    if cv != np.inf and not np.isnan(cv):
                        stability_score = max(0, 1 - min(cv, 1))  # Cap CV at 1 for scoring
                        coeff_scores.append(stability_score)
            
            sig_scores = []
            if stability_metrics['significance_stability']:
                for var_name, metrics in stability_metrics['significance_stability'].items():
                    # Higher significance rate indicates more stable results
                    sig_rate = metrics['significance_rate']
                    sig_scores.append(sig_rate)
            
            # Combine scores
            all_scores = coeff_scores + sig_scores
            if all_scores:
                stability_metrics['overall_stability_score'] = np.mean(all_scores)
            
        except Exception as e:
            stability_metrics['error'] = str(e)
        
        return stability_metrics
    
    def _generate_robustness_summary(self, stability_metrics: Dict) -> Dict:
        """
        Generate human-readable robustness summary.
        
        Args:
            stability_metrics: Dictionary with calculated stability metrics
            
        Returns:
            Dictionary with robustness assessment
        """
        summary = {
            'overall_assessment': 'UNKNOWN',
            'stability_score': stability_metrics.get('overall_stability_score', 0.0),
            'key_findings': [],
            'recommendations': []
        }
        
        stability_score = summary['stability_score']
        
        # Overall assessment based on stability score
        if stability_score >= 0.8:
            summary['overall_assessment'] = 'HIGHLY_ROBUST'
            summary['key_findings'].append("Results are highly stable across temporal specifications")
        elif stability_score >= 0.6:
            summary['overall_assessment'] = 'MODERATELY_ROBUST'
            summary['key_findings'].append("Results show moderate stability across temporal specifications")
        elif stability_score >= 0.4:
            summary['overall_assessment'] = 'WEAKLY_ROBUST'
            summary['key_findings'].append("Results show limited stability across temporal specifications")
            summary['recommendations'].append("Consider sensitivity analysis for key findings")
        else:
            summary['overall_assessment'] = 'NOT_ROBUST'
            summary['key_findings'].append("Results are not stable across temporal specifications")
            summary['recommendations'].append("Reconsider temporal scope and model specification")
        
        # Specific coefficient stability assessment
        if 'coefficient_stability' in stability_metrics:
            unstable_coeffs = []
            for coeff_name, metrics in stability_metrics['coefficient_stability'].items():
                if metrics['cv'] > 0.5:  # High coefficient of variation
                    unstable_coeffs.append(coeff_name)
            
            if unstable_coeffs:
                summary['key_findings'].append(
                    f"Coefficients with high variability: {', '.join(unstable_coeffs)}"
                )
                summary['recommendations'].append(
                    "Investigate sources of coefficient instability"
                )
        
        # Significance stability assessment
        if 'significance_stability' in stability_metrics:
            unstable_significance = []
            for var_name, metrics in stability_metrics['significance_stability'].items():
                if metrics['significance_rate'] < 0.7:  # Less than 70% significance rate
                    unstable_significance.append(var_name)
            
            if unstable_significance:
                summary['key_findings'].append(
                    f"Variables with unstable significance: {', '.join(unstable_significance)}"
                )
                summary['recommendations'].append(
                    "Consider robustness of inference for key variables"
                )
        
        return summary
    
    def compare_full_vs_qe_sample(self, data: pd.DataFrame,
                                analysis_function: callable,
                                date_column: Optional[str] = None) -> Dict:
        """
        Compare analysis results between full sample and QE-only sample.
        
        Args:
            data: Full dataset
            analysis_function: Function that takes data and returns analysis results
            date_column: Name of date column if not using index
            
        Returns:
            Dictionary with comparison results
        """
        comparison_results = {
            'full_sample_results': None,
            'qe_sample_results': None,
            'comparison_metrics': {},
            'recommendations': []
        }
        
        try:
            # Run analysis on full sample
            comparison_results['full_sample_results'] = {
                'result': analysis_function(data),
                'observations': len(data),
                'date_range': (data.index.min(), data.index.max())
            }
            
            # Create QE-only sample
            qe_datasets = self.create_qe_focused_dataset(data, date_column=date_column)
            qe_data = qe_datasets['qe_focused']
            
            # Run analysis on QE sample
            comparison_results['qe_sample_results'] = {
                'result': analysis_function(qe_data),
                'observations': len(qe_data),
                'date_range': (qe_data.index.min(), qe_data.index.max())
            }
            
            # Calculate comparison metrics
            comparison_results['comparison_metrics'] = self._compare_analysis_results(
                comparison_results['full_sample_results']['result'],
                comparison_results['qe_sample_results']['result']
            )
            
            # Generate recommendations
            comparison_results['recommendations'] = self._generate_comparison_recommendations(
                comparison_results['comparison_metrics']
            )
            
        except Exception as e:
            comparison_results['error'] = str(e)
            self.logger.error(f"Full vs QE sample comparison failed: {e}")
        
        return comparison_results
    
    def _compare_analysis_results(self, full_result, qe_result) -> Dict:
        """
        Compare two analysis results and calculate difference metrics.
        
        Args:
            full_result: Results from full sample analysis
            qe_result: Results from QE-only sample analysis
            
        Returns:
            Dictionary with comparison metrics
        """
        comparison_metrics = {
            'coefficient_differences': {},
            'significance_changes': {},
            'overall_similarity': 0.0
        }
        
        try:
            # Compare coefficients if available
            if (isinstance(full_result, dict) and isinstance(qe_result, dict) and
                'coefficients' in full_result and 'coefficients' in qe_result):
                
                full_coeffs = full_result['coefficients']
                qe_coeffs = qe_result['coefficients']
                
                for coeff_name in full_coeffs.keys():
                    if coeff_name in qe_coeffs:
                        full_val = full_coeffs[coeff_name]
                        qe_val = qe_coeffs[coeff_name]
                        
                        if full_val != 0:
                            pct_change = ((qe_val - full_val) / full_val) * 100
                        else:
                            pct_change = np.inf if qe_val != 0 else 0
                        
                        comparison_metrics['coefficient_differences'][coeff_name] = {
                            'full_sample': full_val,
                            'qe_sample': qe_val,
                            'absolute_difference': qe_val - full_val,
                            'percent_change': pct_change
                        }
            
            # Compare significance if available
            if (isinstance(full_result, dict) and isinstance(qe_result, dict) and
                'p_values' in full_result and 'p_values' in qe_result):
                
                full_pvals = full_result['p_values']
                qe_pvals = qe_result['p_values']
                
                for var_name in full_pvals.keys():
                    if var_name in qe_pvals:
                        full_sig = full_pvals[var_name] < 0.05
                        qe_sig = qe_pvals[var_name] < 0.05
                        
                        comparison_metrics['significance_changes'][var_name] = {
                            'full_sample_significant': full_sig,
                            'qe_sample_significant': qe_sig,
                            'significance_changed': full_sig != qe_sig,
                            'full_p_value': full_pvals[var_name],
                            'qe_p_value': qe_pvals[var_name]
                        }
            
            # Calculate overall similarity score
            similarity_scores = []
            
            # Coefficient similarity
            if comparison_metrics['coefficient_differences']:
                coeff_similarities = []
                for coeff_name, metrics in comparison_metrics['coefficient_differences'].items():
                    pct_change = abs(metrics['percent_change'])
                    if pct_change != np.inf and not np.isnan(pct_change):
                        # Similarity decreases with percentage change
                        similarity = max(0, 1 - min(pct_change / 100, 1))
                        coeff_similarities.append(similarity)
                
                if coeff_similarities:
                    similarity_scores.append(np.mean(coeff_similarities))
            
            # Significance similarity
            if comparison_metrics['significance_changes']:
                sig_agreements = []
                for var_name, metrics in comparison_metrics['significance_changes'].items():
                    # Agreement if significance status is the same
                    agreement = 1 if not metrics['significance_changed'] else 0
                    sig_agreements.append(agreement)
                
                if sig_agreements:
                    similarity_scores.append(np.mean(sig_agreements))
            
            if similarity_scores:
                comparison_metrics['overall_similarity'] = np.mean(similarity_scores)
            
        except Exception as e:
            comparison_metrics['error'] = str(e)
        
        return comparison_metrics
    
    def _generate_comparison_recommendations(self, comparison_metrics: Dict) -> List[str]:
        """
        Generate recommendations based on full vs QE sample comparison.
        
        Args:
            comparison_metrics: Dictionary with comparison metrics
            
        Returns:
            List of recommendation strings
        """
        recommendations = []
        
        overall_similarity = comparison_metrics.get('overall_similarity', 0.0)
        
        if overall_similarity >= 0.8:
            recommendations.append("Results are highly consistent between full and QE samples")
            recommendations.append("Pre-QE data does not substantially affect conclusions")
        elif overall_similarity >= 0.6:
            recommendations.append("Results show moderate consistency between samples")
            recommendations.append("Consider reporting both full and QE-only results")
        else:
            recommendations.append("Results differ substantially between full and QE samples")
            recommendations.append("Recommend focusing on QE-only sample for main analysis")
            recommendations.append("Investigate sources of temporal instability")
        
        # Specific coefficient recommendations
        if 'coefficient_differences' in comparison_metrics:
            large_changes = []
            for coeff_name, metrics in comparison_metrics['coefficient_differences'].items():
                pct_change = abs(metrics['percent_change'])
                if pct_change > 50 and pct_change != np.inf:
                    large_changes.append(coeff_name)
            
            if large_changes:
                recommendations.append(
                    f"Large coefficient changes detected for: {', '.join(large_changes)}"
                )
        
        # Significance change recommendations
        if 'significance_changes' in comparison_metrics:
            sig_changes = []
            for var_name, metrics in comparison_metrics['significance_changes'].items():
                if metrics['significance_changed']:
                    sig_changes.append(var_name)
            
            if sig_changes:
                recommendations.append(
                    f"Significance changes detected for: {', '.join(sig_changes)}"
                )
                recommendations.append("Consider robustness of inference")
        
        return recommendations
    
    def test_qe_episode_stability(self, data: pd.DataFrame,
                                analysis_function: callable,
                                episode_definitions: Optional[Dict] = None,
                                date_column: Optional[str] = None) -> Dict:
        """
        Test stability of results across different QE episodes.
        
        Args:
            data: DataFrame with time series data
            analysis_function: Function that takes data and returns analysis results
            episode_definitions: Dictionary defining QE episodes
            date_column: Name of date column if not using index
            
        Returns:
            Dictionary with episode stability results
        """
        if episode_definitions is None:
            # Default QE episode definitions
            episode_definitions = {
                'QE1': ('2008-11-01', '2010-03-31'),
                'QE2': ('2010-11-01', '2011-06-30'),
                'QE3': ('2012-09-01', '2014-10-31'),
                'COVID_QE': ('2020-03-01', '2021-12-31')
            }
        
        episode_results = {
            'episode_definitions': episode_definitions,
            'results_by_episode': {},
            'cross_episode_stability': {},
            'recommendations': []
        }
        
        try:
            # Extract date information
            if date_column:
                dates = pd.to_datetime(data[date_column])
                data_with_dates = data.copy()
            else:
                if hasattr(data.index, 'to_pydatetime'):
                    dates = data.index
                else:
                    dates = pd.to_datetime(data.index)
                data_with_dates = data.copy()
            
            # Run analysis for each episode
            episode_analysis_results = []
            
            for episode_name, (start_date, end_date) in episode_definitions.items():
                start_dt = pd.to_datetime(start_date)
                end_dt = pd.to_datetime(end_date)
                
                # Filter data to episode
                if date_column:
                    episode_mask = ((pd.to_datetime(data_with_dates[date_column]) >= start_dt) & 
                                  (pd.to_datetime(data_with_dates[date_column]) <= end_dt))
                    episode_data = data_with_dates[episode_mask].copy()
                else:
                    episode_mask = (dates >= start_dt) & (dates <= end_dt)
                    episode_data = data_with_dates.loc[episode_mask].copy()
                
                if len(episode_data) < 50:  # Minimum observations for episode analysis
                    episode_results['results_by_episode'][episode_name] = {
                        'error': 'Insufficient observations for episode analysis',
                        'observations': len(episode_data)
                    }
                    continue
                
                try:
                    # Run analysis for episode
                    result = analysis_function(episode_data)
                    episode_results['results_by_episode'][episode_name] = {
                        'result': result,
                        'observations': len(episode_data),
                        'date_range': (episode_data.index.min(), episode_data.index.max())
                    }
                    episode_analysis_results.append(result)
                    
                except Exception as e:
                    episode_results['results_by_episode'][episode_name] = {
                        'error': str(e),
                        'observations': len(episode_data)
                    }
            
            # Calculate cross-episode stability
            if len(episode_analysis_results) >= 2:
                episode_results['cross_episode_stability'] = self._calculate_stability_metrics(
                    episode_analysis_results, episode_analysis_results[0]
                )
                
                # Generate recommendations
                stability_score = episode_results['cross_episode_stability'].get('overall_stability_score', 0.0)
                
                if stability_score >= 0.7:
                    episode_results['recommendations'].append("Results are stable across QE episodes")
                elif stability_score >= 0.5:
                    episode_results['recommendations'].append("Results show moderate stability across episodes")
                    episode_results['recommendations'].append("Consider episode-specific analysis")
                else:
                    episode_results['recommendations'].append("Results vary significantly across QE episodes")
                    episode_results['recommendations'].append("Investigate episode-specific mechanisms")
                    episode_results['recommendations'].append("Consider time-varying parameter models")
            
        except Exception as e:
            episode_results['error'] = str(e)
            self.logger.error(f"QE episode stability testing failed: {e}")
        
        return episode_results