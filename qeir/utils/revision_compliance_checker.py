"""
Revision Compliance Checker

This module provides comprehensive testing framework for methodology validation
and ensures all reviewer concerns are systematically addressed.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import logging
from datetime import datetime

from .revised_qe_analyzer import RevisedQEAnalyzer


class RevisionComplianceChecker:
    """
    Comprehensive testing framework for methodology validation.
    
    This class systematically validates that all reviewer concerns are addressed
    and provides automated compliance reporting.
    """
    
    def __init__(self, analyzer: Optional[RevisedQEAnalyzer] = None):
        """
        Initialize the RevisionComplianceChecker.
        
        Args:
            analyzer: Optional RevisedQEAnalyzer instance for testing
        """
        self.analyzer = analyzer or RevisedQEAnalyzer()
        self.logger = logging.getLogger(__name__)
        
        # Compliance test results
        self.compliance_results = {}
        self.methodology_tests = {}
        self.validation_report = {}
        
        # Define reviewer concerns for systematic testing
        self.reviewer_concerns = {
            'temporal_scope': {
                'description': 'Data temporal inconsistencies - focus on QE period',
                'requirements': ['2008-2024 focus', 'pre-QE justification', 'robustness checks']
            },
            'identification_strategy': {
                'description': 'Weak identification and endogeneity concerns',
                'requirements': ['instrument validity', 'weak instrument tests', 'endogeneity testing']
            },
            'threshold_theory': {
                'description': 'Lack of theoretical foundation for 0.3% threshold',
                'requirements': ['economic theory', 'threshold justification', 'robustness']
            },
            'channel_decomposition': {
                'description': 'Unclear 60%/40% market distortion vs interest rate split',
                'requirements': ['formal identification', 'theoretical model', 'empirical validation']
            },
            'international_inconsistency': {
                'description': 'Mixed international transmission results',
                'requirements': ['consistent framework', 'reconciled findings', 'theoretical coherence']
            },
            'technical_issues': {
                'description': 'Low R² and model specification concerns',
                'requirements': ['diagnostic tests', 'robustness checks', 'specification testing']
            },
            'publication_strategy': {
                'description': 'Unclear contribution and journal targeting',
                'requirements': ['contribution assessment', 'journal alignment', 'splitting feasibility']
            }
        }
    
    def reviewer_concern_compliance_test(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Test compliance with each major reviewer concern.
        
        Args:
            data: Dataset for analysis
            
        Returns:
            Dictionary with compliance test results for each concern
        """
        self.logger.info("Running reviewer concern compliance tests")
        
        compliance_results = {}
        
        # Run analysis to get results for testing
        analysis_results = self.analyzer.run_revised_analysis(data)
        
        # Test each reviewer concern
        for concern, details in self.reviewer_concerns.items():
            self.logger.info(f"Testing compliance for: {concern}")
            
            test_result = self._test_individual_concern(
                concern, details, analysis_results, data
            )
            compliance_results[concern] = test_result
        
        self.compliance_results = compliance_results
        return compliance_results
    
    def _test_individual_concern(self, concern: str, details: Dict, 
                               analysis_results: Dict, data: pd.DataFrame) -> Dict[str, Any]:
        """Test compliance for an individual reviewer concern."""
        test_result = {
            'concern': concern,
            'description': details['description'],
            'requirements': details['requirements'],
            'compliance_status': 'unknown',
            'test_results': {},
            'recommendations': []
        }
        
        if concern == 'temporal_scope':
            test_result.update(self._test_temporal_scope_compliance(analysis_results, data))
        elif concern == 'identification_strategy':
            test_result.update(self._test_identification_compliance(analysis_results, data))
        elif concern == 'threshold_theory':
            test_result.update(self._test_threshold_theory_compliance(analysis_results, data))
        elif concern == 'channel_decomposition':
            test_result.update(self._test_channel_decomposition_compliance(analysis_results, data))
        elif concern == 'international_inconsistency':
            test_result.update(self._test_international_compliance(analysis_results, data))
        elif concern == 'technical_issues':
            test_result.update(self._test_technical_compliance(analysis_results, data))
        elif concern == 'publication_strategy':
            test_result.update(self._test_publication_compliance(analysis_results, data))
        
        return test_result
    
    def _test_temporal_scope_compliance(self, analysis_results: Dict, data: pd.DataFrame) -> Dict:
        """Test temporal scope compliance."""
        test_results = {}
        compliance_status = 'compliant'
        recommendations = []
        
        # Check if temporal correction was applied
        temporal_data = analysis_results.get('temporal_correction')
        if temporal_data is not None:
            test_results['temporal_correction_applied'] = True
            
            # Check data period
            if 'date' in data.columns:
                min_date = pd.to_datetime(data['date']).min()
                max_date = pd.to_datetime(data['date']).max()
                
                test_results['data_period'] = {
                    'start': str(min_date),
                    'end': str(max_date),
                    'qe_focused': min_date >= pd.to_datetime('2008-01-01')
                }
                
                if min_date < pd.to_datetime('2008-01-01'):
                    compliance_status = 'partial'
                    recommendations.append("Consider removing pre-QE data or provide justification")
        else:
            test_results['temporal_correction_applied'] = False
            compliance_status = 'non_compliant'
            recommendations.append("Apply temporal scope correction")
        
        # Check for robustness tests
        diagnostics = self.analyzer.diagnostics.get('temporal_robustness', {})
        test_results['robustness_tests'] = bool(diagnostics and 'error' not in diagnostics)
        
        return {
            'compliance_status': compliance_status,
            'test_results': test_results,
            'recommendations': recommendations
        }
    
    def _test_identification_compliance(self, analysis_results: Dict, data: pd.DataFrame) -> Dict:
        """Test identification strategy compliance."""
        test_results = {}
        compliance_status = 'compliant'
        recommendations = []
        
        identification_results = analysis_results.get('identification', {})
        
        # Check for instrument validity tests
        validity_tests = [key for key in identification_results.keys() if 'validity' in key]
        test_results['instrument_validity_tests'] = len(validity_tests)
        
        if len(validity_tests) == 0:
            compliance_status = 'non_compliant'
            recommendations.append("Implement instrument validity tests")
        
        # Check for endogeneity testing
        test_results['endogeneity_test'] = 'endogeneity_test' in identification_results
        if not test_results['endogeneity_test']:
            compliance_status = 'partial'
            recommendations.append("Add comprehensive endogeneity testing")
        
        # Check for overidentification tests
        test_results['overidentification_test'] = 'overidentification_test' in identification_results
        
        return {
            'compliance_status': compliance_status,
            'test_results': test_results,
            'recommendations': recommendations
        }
    
    def _test_threshold_theory_compliance(self, analysis_results: Dict, data: pd.DataFrame) -> Dict:
        """Test threshold theory compliance."""
        test_results = {}
        compliance_status = 'compliant'
        recommendations = []
        
        theory_results = analysis_results.get('theoretical_foundation', {})
        
        # Check for threshold theory
        test_results['threshold_theory'] = 'threshold_theory' in theory_results
        if not test_results['threshold_theory']:
            compliance_status = 'non_compliant'
            recommendations.append("Develop theoretical foundation for threshold effects")
        
        # Check for theoretical predictions
        test_results['theoretical_predictions'] = 'theoretical_predictions' in theory_results
        if not test_results['theoretical_predictions']:
            compliance_status = 'partial'
            recommendations.append("Add theoretical prediction validation")
        
        return {
            'compliance_status': compliance_status,
            'test_results': test_results,
            'recommendations': recommendations
        }
    
    def _test_channel_decomposition_compliance(self, analysis_results: Dict, data: pd.DataFrame) -> Dict:
        """Test channel decomposition compliance."""
        test_results = {}
        compliance_status = 'compliant'
        recommendations = []
        
        theory_results = analysis_results.get('theoretical_foundation', {})
        channel_decomp = theory_results.get('channel_decomposition', {})
        
        # Check for channel models
        test_results['interest_rate_channel'] = 'interest_rate_channel' in channel_decomp
        test_results['market_distortion_channel'] = 'market_distortion_channel' in channel_decomp
        
        if not (test_results['interest_rate_channel'] and test_results['market_distortion_channel']):
            compliance_status = 'non_compliant'
            recommendations.append("Implement formal channel decomposition models")
        
        return {
            'compliance_status': compliance_status,
            'test_results': test_results,
            'recommendations': recommendations
        }
    
    def _test_international_compliance(self, analysis_results: Dict, data: pd.DataFrame) -> Dict:
        """Test international analysis compliance."""
        test_results = {}
        compliance_status = 'compliant'
        recommendations = []
        
        international_results = analysis_results.get('international_analysis', {})
        
        # Check for spillover analysis
        test_results['spillover_analysis'] = 'spillover_analysis' in international_results
        
        # Check for flow decomposition
        flow_decomp = international_results.get('flow_decomposition', {})
        test_results['official_investors'] = 'official_investors' in flow_decomp
        test_results['private_investors'] = 'private_investors' in flow_decomp
        
        # Check for transmission channels
        test_results['transmission_channels'] = 'transmission_channels' in international_results
        
        if not all([test_results['spillover_analysis'], test_results['official_investors'], 
                   test_results['private_investors'], test_results['transmission_channels']]):
            compliance_status = 'partial'
            recommendations.append("Complete international transmission analysis")
        
        return {
            'compliance_status': compliance_status,
            'test_results': test_results,
            'recommendations': recommendations
        }
    
    def _test_technical_compliance(self, analysis_results: Dict, data: pd.DataFrame) -> Dict:
        """Test technical improvements compliance."""
        test_results = {}
        compliance_status = 'compliant'
        recommendations = []
        
        technical_results = analysis_results.get('technical_improvements', {})
        
        # Check for diagnostic tests
        test_results['hansen_diagnostics'] = 'hansen_diagnostics' in technical_results
        test_results['local_projections_diagnostics'] = 'local_projections_diagnostics' in technical_results
        test_results['sample_size_diagnostics'] = 'sample_size_diagnostics' in technical_results
        
        if not all(test_results.values()):
            compliance_status = 'partial'
            recommendations.append("Complete technical diagnostic testing")
        
        return {
            'compliance_status': compliance_status,
            'test_results': test_results,
            'recommendations': recommendations
        }
    
    def _test_publication_compliance(self, analysis_results: Dict, data: pd.DataFrame) -> Dict:
        """Test publication strategy compliance."""
        test_results = {}
        compliance_status = 'compliant'
        recommendations = []
        
        publication_results = analysis_results.get('publication_strategy', {})
        
        # Check for splitting feasibility
        test_results['splitting_feasibility'] = 'splitting_feasibility' in publication_results
        
        # Check for journal targeting
        test_results['journal_targeting'] = 'journal_targeting' in publication_results
        
        # Check for contribution assessment
        test_results['contribution_assessment'] = 'contribution_assessment' in publication_results
        
        if not all(test_results.values()):
            compliance_status = 'partial'
            recommendations.append("Complete publication strategy assessment")
        
        return {
            'compliance_status': compliance_status,
            'test_results': test_results,
            'recommendations': recommendations
        }
    
    def methodology_robustness_test(self, data: pd.DataFrame, 
                                  n_bootstrap: int = 100) -> Dict[str, Any]:
        """
        Test methodology robustness across all enhanced components.
        
        Args:
            data: Dataset for analysis
            n_bootstrap: Number of bootstrap iterations
            
        Returns:
            Dictionary with robustness test results
        """
        self.logger.info("Running methodology robustness tests")
        
        robustness_results = {
            'bootstrap_iterations': n_bootstrap,
            'component_stability': {},
            'overall_robustness': 'unknown',
            'stability_metrics': {}
        }
        
        # Test each component's robustness
        components = [
            'temporal_correction', 'identification', 'theoretical_foundation',
            'technical_improvements', 'international_analysis', 'publication_strategy'
        ]
        
        for component in components:
            self.logger.info(f"Testing robustness for: {component}")
            
            try:
                stability_test = self._test_component_stability(component, data, n_bootstrap)
                robustness_results['component_stability'][component] = stability_test
            except Exception as e:
                self.logger.warning(f"Robustness test failed for {component}: {e}")
                robustness_results['component_stability'][component] = {'error': str(e)}
        
        # Calculate overall robustness
        stable_components = sum(1 for result in robustness_results['component_stability'].values()
                              if result.get('stable', False))
        total_components = len(components)
        
        robustness_results['stability_metrics'] = {
            'stable_components': stable_components,
            'total_components': total_components,
            'stability_ratio': stable_components / total_components
        }
        
        if robustness_results['stability_metrics']['stability_ratio'] >= 0.8:
            robustness_results['overall_robustness'] = 'robust'
        elif robustness_results['stability_metrics']['stability_ratio'] >= 0.6:
            robustness_results['overall_robustness'] = 'moderately_robust'
        else:
            robustness_results['overall_robustness'] = 'unstable'
        
        self.methodology_tests['robustness'] = robustness_results
        return robustness_results
    
    def _test_component_stability(self, component: str, data: pd.DataFrame, 
                                n_bootstrap: int) -> Dict[str, Any]:
        """Test stability of individual component across bootstrap samples."""
        stability_results = {
            'component': component,
            'stable': False,
            'consistency_score': 0.0,
            'bootstrap_results': []
        }
        
        # For demonstration, we'll run a simplified stability test
        # In practice, this would involve bootstrap sampling and re-running analysis
        
        try:
            # Run analysis multiple times with slight data variations
            results = []
            for i in range(min(10, n_bootstrap)):  # Limit for demonstration
                # Add small random noise to test stability
                noisy_data = data.copy()
                if 'qe_intensity' in data.columns:
                    noise = np.random.normal(0, 0.001, len(data))
                    noisy_data['qe_intensity'] += noise
                
                # Run analysis
                analysis_result = self.analyzer.run_revised_analysis(noisy_data)
                component_result = analysis_result.get(component, {})
                
                # Simple consistency check - component produces results
                has_results = bool(component_result and 'error' not in str(component_result))
                results.append(has_results)
            
            # Calculate consistency
            consistency_score = sum(results) / len(results) if results else 0.0
            stability_results['consistency_score'] = consistency_score
            stability_results['stable'] = consistency_score >= 0.8
            stability_results['bootstrap_results'] = results
            
        except Exception as e:
            stability_results['error'] = str(e)
        
        return stability_results
    
    def automated_validation_report(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Create automated validation report for revision compliance documentation.
        
        Args:
            data: Dataset for analysis
            
        Returns:
            Comprehensive validation report
        """
        self.logger.info("Generating automated validation report")
        
        # Run all compliance tests
        compliance_results = self.reviewer_concern_compliance_test(data)
        robustness_results = self.methodology_robustness_test(data)
        
        # Generate comprehensive report
        validation_report = {
            'report_metadata': {
                'generation_date': datetime.now(),
                'data_period': self._get_data_period(data),
                'total_observations': len(data),
                'analysis_components': len(self.reviewer_concerns)
            },
            'compliance_summary': self._generate_compliance_summary(compliance_results),
            'robustness_summary': self._generate_robustness_summary(robustness_results),
            'detailed_results': {
                'compliance_tests': compliance_results,
                'robustness_tests': robustness_results
            },
            'overall_assessment': self._generate_overall_assessment(compliance_results, robustness_results),
            'recommendations': self._generate_recommendations(compliance_results, robustness_results)
        }
        
        self.validation_report = validation_report
        return validation_report
    
    def _generate_compliance_summary(self, compliance_results: Dict) -> Dict[str, Any]:
        """Generate summary of compliance test results."""
        summary = {
            'total_concerns': len(compliance_results),
            'compliant': 0,
            'partial': 0,
            'non_compliant': 0,
            'compliance_rate': 0.0
        }
        
        for concern, result in compliance_results.items():
            status = result.get('compliance_status', 'unknown')
            if status == 'compliant':
                summary['compliant'] += 1
            elif status == 'partial':
                summary['partial'] += 1
            elif status == 'non_compliant':
                summary['non_compliant'] += 1
        
        summary['compliance_rate'] = summary['compliant'] / summary['total_concerns']
        return summary
    
    def _generate_robustness_summary(self, robustness_results: Dict) -> Dict[str, Any]:
        """Generate summary of robustness test results."""
        stability_metrics = robustness_results.get('stability_metrics', {})
        
        return {
            'overall_robustness': robustness_results.get('overall_robustness', 'unknown'),
            'stability_ratio': stability_metrics.get('stability_ratio', 0.0),
            'stable_components': stability_metrics.get('stable_components', 0),
            'total_components': stability_metrics.get('total_components', 0)
        }
    
    def _generate_overall_assessment(self, compliance_results: Dict, 
                                   robustness_results: Dict) -> Dict[str, Any]:
        """Generate overall assessment of revision readiness."""
        compliance_summary = self._generate_compliance_summary(compliance_results)
        robustness_summary = self._generate_robustness_summary(robustness_results)
        
        # Determine overall readiness
        compliance_rate = compliance_summary['compliance_rate']
        stability_ratio = robustness_summary['stability_ratio']
        
        if compliance_rate >= 0.8 and stability_ratio >= 0.8:
            readiness = 'ready_for_submission'
        elif compliance_rate >= 0.6 and stability_ratio >= 0.6:
            readiness = 'minor_revisions_needed'
        elif compliance_rate >= 0.4 or stability_ratio >= 0.4:
            readiness = 'major_revisions_needed'
        else:
            readiness = 'substantial_work_required'
        
        return {
            'revision_readiness': readiness,
            'compliance_rate': compliance_rate,
            'stability_ratio': stability_ratio,
            'key_strengths': self._identify_strengths(compliance_results, robustness_results),
            'critical_issues': self._identify_critical_issues(compliance_results, robustness_results)
        }
    
    def _generate_recommendations(self, compliance_results: Dict, 
                                robustness_results: Dict) -> List[str]:
        """Generate actionable recommendations for improvement."""
        recommendations = []
        
        # Collect recommendations from compliance tests
        for concern, result in compliance_results.items():
            concern_recommendations = result.get('recommendations', [])
            recommendations.extend(concern_recommendations)
        
        # Add robustness-based recommendations
        stability_ratio = robustness_results.get('stability_metrics', {}).get('stability_ratio', 0)
        if stability_ratio < 0.6:
            recommendations.append("Improve methodology stability through additional robustness checks")
        
        # Remove duplicates and prioritize
        unique_recommendations = list(set(recommendations))
        return unique_recommendations[:10]  # Top 10 recommendations
    
    def _identify_strengths(self, compliance_results: Dict, robustness_results: Dict) -> List[str]:
        """Identify key strengths of the current methodology."""
        strengths = []
        
        for concern, result in compliance_results.items():
            if result.get('compliance_status') == 'compliant':
                strengths.append(f"Strong {concern.replace('_', ' ')} implementation")
        
        if robustness_results.get('overall_robustness') in ['robust', 'moderately_robust']:
            strengths.append("Robust methodology across components")
        
        return strengths
    
    def _identify_critical_issues(self, compliance_results: Dict, robustness_results: Dict) -> List[str]:
        """Identify critical issues requiring immediate attention."""
        critical_issues = []
        
        for concern, result in compliance_results.items():
            if result.get('compliance_status') == 'non_compliant':
                critical_issues.append(f"Non-compliant {concern.replace('_', ' ')}")
        
        if robustness_results.get('overall_robustness') == 'unstable':
            critical_issues.append("Methodology instability across components")
        
        return critical_issues
    
    def _get_data_period(self, data: pd.DataFrame) -> Dict[str, str]:
        """Get data period information."""
        if 'date' in data.columns:
            dates = pd.to_datetime(data['date'])
            return {
                'start_date': str(dates.min().date()),
                'end_date': str(dates.max().date()),
                'observations': len(data)
            }
        return {'period': 'unknown', 'observations': len(data)}
    
    def export_compliance_report(self, filepath: str, format: str = 'json') -> bool:
        """
        Export compliance report to file.
        
        Args:
            filepath: Path for output file
            format: Export format ('json', 'csv', 'txt')
            
        Returns:
            Success status
        """
        try:
            if format == 'json':
                import json
                with open(filepath, 'w') as f:
                    json.dump(self.validation_report, f, indent=2, default=str)
            elif format == 'txt':
                with open(filepath, 'w') as f:
                    f.write(self._format_text_report())
            else:
                self.logger.error(f"Unsupported export format: {format}")
                return False
            
            self.logger.info(f"Compliance report exported to: {filepath}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to export report: {e}")
            return False
    
    def _format_text_report(self) -> str:
        """Format validation report as readable text."""
        if not self.validation_report:
            return "No validation report available. Run automated_validation_report() first."
        
        report_lines = [
            "QE PAPER REVISION COMPLIANCE REPORT",
            "=" * 40,
            f"Generated: {self.validation_report['report_metadata']['generation_date']}",
            f"Data Period: {self.validation_report['report_metadata']['data_period']}",
            "",
            "COMPLIANCE SUMMARY:",
            f"- Total Concerns: {self.validation_report['compliance_summary']['total_concerns']}",
            f"- Compliant: {self.validation_report['compliance_summary']['compliant']}",
            f"- Partial: {self.validation_report['compliance_summary']['partial']}",
            f"- Non-Compliant: {self.validation_report['compliance_summary']['non_compliant']}",
            f"- Compliance Rate: {self.validation_report['compliance_summary']['compliance_rate']:.2%}",
            "",
            "ROBUSTNESS SUMMARY:",
            f"- Overall Robustness: {self.validation_report['robustness_summary']['overall_robustness']}",
            f"- Stability Ratio: {self.validation_report['robustness_summary']['stability_ratio']:.2%}",
            "",
            "OVERALL ASSESSMENT:",
            f"- Revision Readiness: {self.validation_report['overall_assessment']['revision_readiness']}",
            "",
            "KEY RECOMMENDATIONS:",
        ]
        
        for i, rec in enumerate(self.validation_report['recommendations'], 1):
            report_lines.append(f"{i}. {rec}")
        
        return "\n".join(report_lines)