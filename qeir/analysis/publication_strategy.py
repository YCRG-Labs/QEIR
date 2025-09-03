"""
Publication Strategy Implementation System

This module provides tools for analyzing publication strategies, including paper splitting
feasibility, journal targeting, and contribution validation for QE research papers.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum
import warnings


class JournalType(Enum):
    """Enumeration of target journals for QE research."""
    JME = "Journal of Monetary Economics"
    AEJ_MACRO = "AEJ: Macroeconomics"
    JIMF = "Journal of International Money and Finance"
    JMCB = "Journal of Money, Credit and Banking"


@dataclass
class PaperMetrics:
    """Metrics for evaluating paper quality and contribution."""
    sample_size: int
    time_period: Tuple[str, str]
    methodology_count: int
    robustness_tests: int
    theoretical_foundation: float  # Score 0-1
    empirical_novelty: float  # Score 0-1
    policy_relevance: float  # Score 0-1
    statistical_significance: float  # Proportion of significant results


@dataclass
class SplittingAssessment:
    """Assessment results for paper splitting feasibility."""
    is_feasible: bool
    confidence_score: float
    paper1_strength: float
    paper2_strength: float
    overlap_concerns: List[str]
    recommendations: List[str]


class PublicationAnalyzer:
    """
    Analyzes publication strategies for QE research, including paper splitting feasibility
    and standalone contribution assessment.
    """
    
    def __init__(self):
        """Initialize the PublicationAnalyzer."""
        self.min_sample_size = 100
        self.min_robustness_tests = 3
        self.min_theoretical_score = 0.6
        self.min_novelty_score = 0.7
        
    def paper_splitting_feasibility(self, 
                                  hypothesis1_metrics: PaperMetrics,
                                  hypothesis2_metrics: PaperMetrics,
                                  hypothesis3_metrics: PaperMetrics) -> SplittingAssessment:
        """
        Analyze feasibility of splitting comprehensive QE study into focused papers.
        
        Args:
            hypothesis1_metrics: Metrics for threshold effects hypothesis
            hypothesis2_metrics: Metrics for investment channels hypothesis  
            hypothesis3_metrics: Metrics for international spillovers hypothesis
            
        Returns:
            SplittingAssessment with feasibility analysis and recommendations
        """
        # Assess domestic effects paper (Hypotheses 1 & 2)
        domestic_strength = self._assess_paper_strength(
            [hypothesis1_metrics, hypothesis2_metrics], "domestic"
        )
        
        # Assess international effects paper (Hypothesis 3)
        international_strength = self._assess_paper_strength(
            [hypothesis3_metrics], "international"
        )
        
        # Check for methodological overlap
        overlap_concerns = self._identify_overlap_concerns(
            hypothesis1_metrics, hypothesis2_metrics, hypothesis3_metrics
        )
        
        # Determine overall feasibility
        is_feasible = (
            domestic_strength >= 0.7 and 
            international_strength >= 0.7 and
            len(overlap_concerns) <= 3
        )
        
        confidence_score = min(domestic_strength, international_strength) * (1 - len(overlap_concerns) * 0.05)
        
        recommendations = self._generate_splitting_recommendations(
            domestic_strength, international_strength, overlap_concerns
        )
        
        return SplittingAssessment(
            is_feasible=is_feasible,
            confidence_score=confidence_score,
            paper1_strength=domestic_strength,
            paper2_strength=international_strength,
            overlap_concerns=overlap_concerns,
            recommendations=recommendations
        )
    
    def hypothesis_independence_test(self, 
                                   hypothesis1_results: Dict[str, Any],
                                   hypothesis2_results: Dict[str, Any],
                                   hypothesis3_results: Dict[str, Any]) -> Dict[str, float]:
        """
        Test independence of hypotheses for separate paper viability.
        
        Args:
            hypothesis1_results: Results from threshold effects analysis
            hypothesis2_results: Results from investment channels analysis
            hypothesis3_results: Results from international spillovers analysis
            
        Returns:
            Dictionary with independence scores for each hypothesis pair
        """
        independence_scores = {}
        
        # Test H1-H2 independence (domestic effects paper)
        h1_h2_independence = self._calculate_hypothesis_independence(
            hypothesis1_results, hypothesis2_results
        )
        independence_scores['h1_h2'] = h1_h2_independence
        
        # Test H1-H3 independence
        h1_h3_independence = self._calculate_hypothesis_independence(
            hypothesis1_results, hypothesis3_results
        )
        independence_scores['h1_h3'] = h1_h3_independence
        
        # Test H2-H3 independence  
        h2_h3_independence = self._calculate_hypothesis_independence(
            hypothesis2_results, hypothesis3_results
        )
        independence_scores['h2_h3'] = h2_h3_independence
        
        # Overall independence assessment
        independence_scores['overall'] = np.mean(list(independence_scores.values()))
        
        return independence_scores
    
    def sample_size_adequacy_test(self, 
                                paper_metrics: List[PaperMetrics],
                                target_journal: JournalType) -> Dict[str, Any]:
        """
        Test sample size adequacy for individual paper requirements.
        
        Args:
            paper_metrics: List of metrics for papers being assessed
            target_journal: Target journal for publication
            
        Returns:
            Dictionary with adequacy assessment results
        """
        journal_requirements = self._get_journal_sample_requirements(target_journal)
        
        adequacy_results = {}
        
        for i, metrics in enumerate(paper_metrics):
            paper_name = f"paper_{i+1}"
            
            # Check minimum sample size
            meets_min_size = metrics.sample_size >= journal_requirements['min_sample_size']
            
            # Check time period coverage
            period_years = self._calculate_period_years(metrics.time_period)
            meets_period_req = period_years >= journal_requirements['min_time_period']
            
            # Check robustness requirements
            meets_robustness = metrics.robustness_tests >= journal_requirements['min_robustness_tests']
            
            # Calculate adequacy score
            adequacy_score = np.mean([meets_min_size, meets_period_req, meets_robustness])
            
            adequacy_results[paper_name] = {
                'adequate': adequacy_score >= 0.8,
                'score': adequacy_score,
                'sample_size_ok': meets_min_size,
                'time_period_ok': meets_period_req,
                'robustness_ok': meets_robustness,
                'recommendations': self._generate_adequacy_recommendations(
                    meets_min_size, meets_period_req, meets_robustness
                )
            }
        
        return adequacy_results
    
    def _assess_paper_strength(self, metrics_list: List[PaperMetrics], paper_type: str) -> float:
        """Assess overall strength of a paper based on its metrics."""
        if not metrics_list:
            return 0.0
            
        # Aggregate metrics across hypotheses
        total_sample = sum(m.sample_size for m in metrics_list)
        avg_theoretical = np.mean([m.theoretical_foundation for m in metrics_list])
        avg_novelty = np.mean([m.empirical_novelty for m in metrics_list])
        avg_policy = np.mean([m.policy_relevance for m in metrics_list])
        total_robustness = sum(m.robustness_tests for m in metrics_list)
        avg_significance = np.mean([m.statistical_significance for m in metrics_list])
        
        # Calculate strength components
        sample_strength = min(total_sample / 200, 1.0)  # Normalize to 200 obs
        theoretical_strength = avg_theoretical
        novelty_strength = avg_novelty
        policy_strength = avg_policy
        robustness_strength = min(total_robustness / 5, 1.0)  # Normalize to 5 tests
        significance_strength = avg_significance
        
        # Weight components based on paper type
        if paper_type == "domestic":
            weights = [0.15, 0.25, 0.20, 0.15, 0.15, 0.10]  # Emphasize theory and novelty
        else:  # international
            weights = [0.20, 0.20, 0.25, 0.10, 0.15, 0.10]  # Emphasize novelty and sample
            
        components = [sample_strength, theoretical_strength, novelty_strength, 
                     policy_strength, robustness_strength, significance_strength]
        
        return np.average(components, weights=weights)
    
    def _identify_overlap_concerns(self, h1_metrics: PaperMetrics, 
                                 h2_metrics: PaperMetrics, 
                                 h3_metrics: PaperMetrics) -> List[str]:
        """Identify potential overlap concerns between hypotheses."""
        concerns = []
        
        # Check temporal overlap
        h1_period = set(range(int(h1_metrics.time_period[0][:4]), 
                             int(h1_metrics.time_period[1][:4]) + 1))
        h2_period = set(range(int(h2_metrics.time_period[0][:4]), 
                             int(h2_metrics.time_period[1][:4]) + 1))
        h3_period = set(range(int(h3_metrics.time_period[0][:4]), 
                             int(h3_metrics.time_period[1][:4]) + 1))
        
        if len(h1_period.intersection(h2_period)) / len(h1_period.union(h2_period)) > 0.8:
            concerns.append("High temporal overlap between H1 and H2")
            
        if len(h1_period.intersection(h3_period)) / len(h1_period.union(h3_period)) > 0.6:
            concerns.append("Significant temporal overlap between domestic and international analysis")
        
        # Check methodological overlap
        if (h1_metrics.methodology_count + h2_metrics.methodology_count) < 4:
            concerns.append("Limited methodological diversity for domestic effects paper")
            
        if h3_metrics.methodology_count < 2:
            concerns.append("Limited methodological approaches for international paper")
        
        return concerns
    
    def _calculate_hypothesis_independence(self, results1: Dict[str, Any], 
                                        results2: Dict[str, Any]) -> float:
        """Calculate independence score between two hypotheses."""
        # Check for shared variables
        vars1 = set(results1.get('variables', []))
        vars2 = set(results2.get('variables', []))
        shared_vars = len(vars1.intersection(vars2))
        total_vars = len(vars1.union(vars2))
        
        variable_independence = 1 - (shared_vars / max(total_vars, 1))
        
        # Check for shared methodology
        methods1 = set(results1.get('methods', []))
        methods2 = set(results2.get('methods', []))
        shared_methods = len(methods1.intersection(methods2))
        total_methods = len(methods1.union(methods2))
        
        method_independence = 1 - (shared_methods / max(total_methods, 1))
        
        # Check for shared data sources
        data1 = set(results1.get('data_sources', []))
        data2 = set(results2.get('data_sources', []))
        shared_data = len(data1.intersection(data2))
        total_data = len(data1.union(data2))
        
        data_independence = 1 - (shared_data / max(total_data, 1))
        
        # Weighted average
        return np.average([variable_independence, method_independence, data_independence], 
                         weights=[0.4, 0.4, 0.2])
    
    def _get_journal_sample_requirements(self, journal: JournalType) -> Dict[str, int]:
        """Get sample size and other requirements for target journals."""
        requirements = {
            JournalType.JME: {
                'min_sample_size': 150,
                'min_time_period': 10,
                'min_robustness_tests': 4
            },
            JournalType.AEJ_MACRO: {
                'min_sample_size': 120,
                'min_time_period': 8,
                'min_robustness_tests': 3
            },
            JournalType.JIMF: {
                'min_sample_size': 100,
                'min_time_period': 6,
                'min_robustness_tests': 3
            },
            JournalType.JMCB: {
                'min_sample_size': 80,
                'min_time_period': 5,
                'min_robustness_tests': 2
            }
        }
        return requirements[journal]
    
    def _calculate_period_years(self, time_period: Tuple[str, str]) -> int:
        """Calculate number of years in time period."""
        start_year = int(time_period[0][:4])
        end_year = int(time_period[1][:4])
        return end_year - start_year + 1
    
    def _generate_adequacy_recommendations(self, size_ok: bool, period_ok: bool, 
                                        robustness_ok: bool) -> List[str]:
        """Generate recommendations for improving sample adequacy."""
        recommendations = []
        
        if not size_ok:
            recommendations.append("Increase sample size through data expansion or pooling")
        if not period_ok:
            recommendations.append("Extend time period coverage or justify shorter period")
        if not robustness_ok:
            recommendations.append("Add additional robustness tests and sensitivity analysis")
            
        return recommendations
    
    def _generate_splitting_recommendations(self, domestic_strength: float, 
                                         international_strength: float,
                                         overlap_concerns: List[str]) -> List[str]:
        """Generate recommendations for paper splitting strategy."""
        recommendations = []
        
        if domestic_strength >= 0.8 and international_strength >= 0.8:
            recommendations.append("Strong case for splitting into two focused papers")
        elif domestic_strength >= 0.7 or international_strength >= 0.7:
            recommendations.append("Consider splitting with additional strengthening of weaker paper")
        else:
            recommendations.append("Maintain unified approach and strengthen overall contribution")
            
        if len(overlap_concerns) > 2:
            recommendations.append("Address methodological and temporal overlap before splitting")
            
        if domestic_strength < 0.6:
            recommendations.append("Strengthen theoretical foundation and empirical robustness for domestic effects")
            
        if international_strength < 0.6:
            recommendations.append("Enhance international analysis methodology and sample coverage")
            
        return recommendations

@dataclass
class JournalAlignment:
    """Results of journal alignment assessment."""
    journal: JournalType
    alignment_score: float
    methodology_fit: float
    scope_fit: float
    contribution_fit: float
    recommendations: List[str]
    required_changes: List[str]


class JournalTargeter:
    """
    Analyzes methodology alignment with target journals for QE research papers.
    """
    
    def __init__(self):
        """Initialize the JournalTargeter."""
        self.journal_preferences = self._initialize_journal_preferences()
    
    def jme_alignment_test(self, paper_metrics: PaperMetrics, 
                          methodology_details: Dict[str, Any]) -> JournalAlignment:
        """
        Test alignment with Journal of Monetary Economics requirements.
        
        Args:
            paper_metrics: Metrics for the paper being assessed
            methodology_details: Details about methodology used
            
        Returns:
            JournalAlignment assessment for JME
        """
        journal = JournalType.JME
        preferences = self.journal_preferences[journal]
        
        # Assess methodology fit
        methodology_fit = self._assess_methodology_fit(methodology_details, preferences['methodology'])
        
        # Assess scope fit
        scope_fit = self._assess_scope_fit(paper_metrics, preferences['scope'])
        
        # Assess contribution fit
        contribution_fit = self._assess_contribution_fit(paper_metrics, preferences['contribution'])
        
        # Calculate overall alignment
        alignment_score = np.average([methodology_fit, scope_fit, contribution_fit], 
                                   weights=[0.4, 0.3, 0.3])
        
        # Generate recommendations
        recommendations = self._generate_journal_recommendations(
            journal, methodology_fit, scope_fit, contribution_fit
        )
        
        # Identify required changes
        required_changes = self._identify_required_changes(
            journal, methodology_details, paper_metrics
        )
        
        return JournalAlignment(
            journal=journal,
            alignment_score=alignment_score,
            methodology_fit=methodology_fit,
            scope_fit=scope_fit,
            contribution_fit=contribution_fit,
            recommendations=recommendations,
            required_changes=required_changes
        )
    
    def aej_macro_alignment_test(self, paper_metrics: PaperMetrics,
                               methodology_details: Dict[str, Any]) -> JournalAlignment:
        """
        Test alignment with AEJ: Macroeconomics requirements.
        
        Args:
            paper_metrics: Metrics for the paper being assessed
            methodology_details: Details about methodology used
            
        Returns:
            JournalAlignment assessment for AEJ: Macro
        """
        journal = JournalType.AEJ_MACRO
        preferences = self.journal_preferences[journal]
        
        # Assess methodology fit (AEJ: Macro emphasizes theoretical rigor)
        methodology_fit = self._assess_methodology_fit(methodology_details, preferences['methodology'])
        
        # Assess scope fit (prefers focused contributions)
        scope_fit = self._assess_scope_fit(paper_metrics, preferences['scope'])
        
        # Assess contribution fit (values theoretical innovation)
        contribution_fit = self._assess_contribution_fit(paper_metrics, preferences['contribution'])
        
        # Calculate overall alignment (higher weight on theory)
        alignment_score = np.average([methodology_fit, scope_fit, contribution_fit], 
                                   weights=[0.3, 0.3, 0.4])
        
        recommendations = self._generate_journal_recommendations(
            journal, methodology_fit, scope_fit, contribution_fit
        )
        
        required_changes = self._identify_required_changes(
            journal, methodology_details, paper_metrics
        )
        
        return JournalAlignment(
            journal=journal,
            alignment_score=alignment_score,
            methodology_fit=methodology_fit,
            scope_fit=scope_fit,
            contribution_fit=contribution_fit,
            recommendations=recommendations,
            required_changes=required_changes
        )
    
    def jimf_alignment_test(self, paper_metrics: PaperMetrics,
                          methodology_details: Dict[str, Any]) -> JournalAlignment:
        """
        Test alignment with Journal of International Money and Finance requirements.
        
        Args:
            paper_metrics: Metrics for the paper being assessed
            methodology_details: Details about methodology used
            
        Returns:
            JournalAlignment assessment for JIMF
        """
        journal = JournalType.JIMF
        preferences = self.journal_preferences[journal]
        
        # Assess methodology fit (JIMF accepts diverse methods)
        methodology_fit = self._assess_methodology_fit(methodology_details, preferences['methodology'])
        
        # Assess scope fit (must have international focus)
        scope_fit = self._assess_scope_fit(paper_metrics, preferences['scope'])
        
        # Assess contribution fit (values policy relevance)
        contribution_fit = self._assess_contribution_fit(paper_metrics, preferences['contribution'])
        
        # Calculate overall alignment (higher weight on scope for international journal)
        alignment_score = np.average([methodology_fit, scope_fit, contribution_fit], 
                                   weights=[0.25, 0.45, 0.3])
        
        recommendations = self._generate_journal_recommendations(
            journal, methodology_fit, scope_fit, contribution_fit
        )
        
        required_changes = self._identify_required_changes(
            journal, methodology_details, paper_metrics
        )
        
        return JournalAlignment(
            journal=journal,
            alignment_score=alignment_score,
            methodology_fit=methodology_fit,
            scope_fit=scope_fit,
            contribution_fit=contribution_fit,
            recommendations=recommendations,
            required_changes=required_changes
        )
    
    def compare_journal_alignment(self, paper_metrics: PaperMetrics,
                                methodology_details: Dict[str, Any]) -> Dict[JournalType, JournalAlignment]:
        """
        Compare alignment across multiple target journals.
        
        Args:
            paper_metrics: Metrics for the paper being assessed
            methodology_details: Details about methodology used
            
        Returns:
            Dictionary mapping journals to their alignment assessments
        """
        alignments = {}
        
        alignments[JournalType.JME] = self.jme_alignment_test(paper_metrics, methodology_details)
        alignments[JournalType.AEJ_MACRO] = self.aej_macro_alignment_test(paper_metrics, methodology_details)
        alignments[JournalType.JIMF] = self.jimf_alignment_test(paper_metrics, methodology_details)
        
        # Also test JMCB for completeness
        alignments[JournalType.JMCB] = self._jmcb_alignment_test(paper_metrics, methodology_details)
        
        return alignments
    
    def _initialize_journal_preferences(self) -> Dict[JournalType, Dict[str, Any]]:
        """Initialize journal preference profiles."""
        return {
            JournalType.JME: {
                'methodology': {
                    'preferred_methods': ['iv_regression', 'hansen_threshold', 'local_projections', 'var'],
                    'theoretical_rigor': 0.9,
                    'empirical_robustness': 0.9,
                    'identification_standards': 0.95
                },
                'scope': {
                    'monetary_policy_focus': 0.95,
                    'central_banking': 0.9,
                    'international_scope': 0.6,
                    'policy_relevance': 0.8
                },
                'contribution': {
                    'theoretical_novelty': 0.8,
                    'empirical_novelty': 0.9,
                    'policy_implications': 0.7,
                    'methodological_innovation': 0.8
                }
            },
            JournalType.AEJ_MACRO: {
                'methodology': {
                    'preferred_methods': ['dsge', 'var', 'theoretical_models', 'calibration'],
                    'theoretical_rigor': 0.95,
                    'empirical_robustness': 0.8,
                    'identification_standards': 0.85
                },
                'scope': {
                    'macroeconomic_focus': 0.95,
                    'business_cycles': 0.8,
                    'international_scope': 0.7,
                    'policy_relevance': 0.9
                },
                'contribution': {
                    'theoretical_novelty': 0.95,
                    'empirical_novelty': 0.7,
                    'policy_implications': 0.8,
                    'methodological_innovation': 0.9
                }
            },
            JournalType.JIMF: {
                'methodology': {
                    'preferred_methods': ['panel_var', 'event_study', 'iv_regression', 'diff_in_diff'],
                    'theoretical_rigor': 0.7,
                    'empirical_robustness': 0.85,
                    'identification_standards': 0.8
                },
                'scope': {
                    'international_focus': 0.95,
                    'exchange_rates': 0.9,
                    'capital_flows': 0.9,
                    'policy_relevance': 0.85
                },
                'contribution': {
                    'theoretical_novelty': 0.6,
                    'empirical_novelty': 0.85,
                    'policy_implications': 0.9,
                    'methodological_innovation': 0.7
                }
            },
            JournalType.JMCB: {
                'methodology': {
                    'preferred_methods': ['var', 'iv_regression', 'panel_methods', 'time_series'],
                    'theoretical_rigor': 0.75,
                    'empirical_robustness': 0.8,
                    'identification_standards': 0.75
                },
                'scope': {
                    'banking_focus': 0.8,
                    'monetary_policy': 0.85,
                    'financial_markets': 0.9,
                    'policy_relevance': 0.8
                },
                'contribution': {
                    'theoretical_novelty': 0.7,
                    'empirical_novelty': 0.8,
                    'policy_implications': 0.85,
                    'methodological_innovation': 0.75
                }
            }
        }
    
    def _assess_methodology_fit(self, methodology_details: Dict[str, Any], 
                              preferences: Dict[str, Any]) -> float:
        """Assess how well methodology fits journal preferences."""
        methods_used = set(methodology_details.get('methods', []))
        preferred_methods = set(preferences['preferred_methods'])
        
        # Calculate method overlap
        method_overlap = len(methods_used.intersection(preferred_methods)) / len(preferred_methods)
        
        # Assess theoretical rigor
        theoretical_score = methodology_details.get('theoretical_rigor', 0.5)
        theoretical_fit = min(theoretical_score / preferences['theoretical_rigor'], 1.0)
        
        # Assess empirical robustness
        robustness_score = methodology_details.get('robustness_tests', 0) / 5.0  # Normalize to 5 tests
        robustness_fit = min(robustness_score / preferences['empirical_robustness'], 1.0)
        
        # Assess identification standards
        identification_score = methodology_details.get('identification_strength', 0.5)
        identification_fit = min(identification_score / preferences['identification_standards'], 1.0)
        
        # Weighted average
        return np.average([method_overlap, theoretical_fit, robustness_fit, identification_fit],
                         weights=[0.3, 0.25, 0.25, 0.2])
    
    def _assess_scope_fit(self, paper_metrics: PaperMetrics, preferences: Dict[str, Any]) -> float:
        """Assess how well paper scope fits journal preferences."""
        scope_scores = []
        
        # Policy relevance fit
        policy_fit = min(paper_metrics.policy_relevance / preferences.get('policy_relevance', 0.8), 1.0)
        scope_scores.append(policy_fit)
        
        # Add specific scope assessments based on available preferences
        for scope_type, required_level in preferences.items():
            if scope_type != 'policy_relevance':
                # Use paper metrics as proxy for scope alignment
                if 'international' in scope_type:
                    actual_level = 0.8 if 'international' in str(paper_metrics.time_period) else 0.3
                elif 'monetary' in scope_type or 'central' in scope_type:
                    actual_level = 0.9  # QE research is inherently monetary policy focused
                else:
                    actual_level = 0.7  # Default moderate alignment
                    
                scope_fit = min(actual_level / required_level, 1.0)
                scope_scores.append(scope_fit)
        
        return np.mean(scope_scores)
    
    def _assess_contribution_fit(self, paper_metrics: PaperMetrics, preferences: Dict[str, Any]) -> float:
        """Assess how well paper contribution fits journal preferences."""
        # Theoretical novelty fit
        theoretical_fit = min(paper_metrics.theoretical_foundation / preferences['theoretical_novelty'], 1.0)
        
        # Empirical novelty fit
        empirical_fit = min(paper_metrics.empirical_novelty / preferences['empirical_novelty'], 1.0)
        
        # Policy implications fit
        policy_fit = min(paper_metrics.policy_relevance / preferences['policy_implications'], 1.0)
        
        # Methodological innovation (use methodology count as proxy)
        method_innovation = min(paper_metrics.methodology_count / 3.0, 1.0)
        method_fit = min(method_innovation / preferences['methodological_innovation'], 1.0)
        
        return np.average([theoretical_fit, empirical_fit, policy_fit, method_fit],
                         weights=[0.3, 0.3, 0.2, 0.2])
    
    def _generate_journal_recommendations(self, journal: JournalType, 
                                        methodology_fit: float, scope_fit: float,
                                        contribution_fit: float) -> List[str]:
        """Generate recommendations for improving journal alignment."""
        recommendations = []
        
        if methodology_fit < 0.7:
            if journal == JournalType.JME:
                recommendations.append("Strengthen identification strategy and add more robustness tests")
            elif journal == JournalType.AEJ_MACRO:
                recommendations.append("Develop stronger theoretical foundation and formal model")
            elif journal == JournalType.JIMF:
                recommendations.append("Add international comparative analysis and policy discussion")
        
        if scope_fit < 0.7:
            if journal == JournalType.JME:
                recommendations.append("Emphasize monetary policy transmission mechanisms")
            elif journal == JournalType.AEJ_MACRO:
                recommendations.append("Connect findings to broader macroeconomic theory")
            elif journal == JournalType.JIMF:
                recommendations.append("Expand international scope and cross-country analysis")
        
        if contribution_fit < 0.7:
            recommendations.append(f"Strengthen novelty and policy relevance for {journal.value}")
        
        return recommendations
    
    def _identify_required_changes(self, journal: JournalType, 
                                 methodology_details: Dict[str, Any],
                                 paper_metrics: PaperMetrics) -> List[str]:
        """Identify specific changes required for journal submission."""
        required_changes = []
        
        preferences = self.journal_preferences[journal]
        
        # Check sample size requirements
        min_sample = self._get_min_sample_size(journal)
        if paper_metrics.sample_size < min_sample:
            required_changes.append(f"Increase sample size to at least {min_sample} observations")
        
        # Check robustness requirements
        min_robustness = preferences['methodology']['empirical_robustness'] * 5  # Scale to number of tests
        if paper_metrics.robustness_tests < min_robustness:
            required_changes.append(f"Add {int(min_robustness - paper_metrics.robustness_tests)} more robustness tests")
        
        # Check theoretical requirements
        if journal == JournalType.AEJ_MACRO and paper_metrics.theoretical_foundation < 0.8:
            required_changes.append("Develop formal theoretical model with mathematical derivations")
        
        # Check international scope for JIMF
        if journal == JournalType.JIMF:
            methods_used = methodology_details.get('methods', [])
            if not any('international' in method or 'cross_country' in method for method in methods_used):
                required_changes.append("Add cross-country analysis or international comparison")
        
        return required_changes
    
    def _get_min_sample_size(self, journal: JournalType) -> int:
        """Get minimum sample size for journal."""
        min_sizes = {
            JournalType.JME: 150,
            JournalType.AEJ_MACRO: 120,
            JournalType.JIMF: 100,
            JournalType.JMCB: 80
        }
        return min_sizes[journal]
    
    def _jmcb_alignment_test(self, paper_metrics: PaperMetrics,
                           methodology_details: Dict[str, Any]) -> JournalAlignment:
        """Test alignment with Journal of Money, Credit and Banking."""
        journal = JournalType.JMCB
        preferences = self.journal_preferences[journal]
        
        methodology_fit = self._assess_methodology_fit(methodology_details, preferences['methodology'])
        scope_fit = self._assess_scope_fit(paper_metrics, preferences['scope'])
        contribution_fit = self._assess_contribution_fit(paper_metrics, preferences['contribution'])
        
        alignment_score = np.average([methodology_fit, scope_fit, contribution_fit], 
                                   weights=[0.35, 0.35, 0.3])
        
        recommendations = self._generate_journal_recommendations(
            journal, methodology_fit, scope_fit, contribution_fit
        )
        
        required_changes = self._identify_required_changes(
            journal, methodology_details, paper_metrics
        )
        
        return JournalAlignment(
            journal=journal,
            alignment_score=alignment_score,
            methodology_fit=methodology_fit,
            scope_fit=scope_fit,
            contribution_fit=contribution_fit,
            recommendations=recommendations,
            required_changes=required_changes
        )
@dataclass
class ContributionAssessment:
    """Assessment of paper's standalone contribution."""
    paper_type: str
    contribution_score: float
    theoretical_contribution: float
    empirical_contribution: float
    methodological_contribution: float
    policy_contribution: float
    novelty_assessment: Dict[str, float]
    publication_readiness: bool
    strengths: List[str]
    weaknesses: List[str]
    improvement_recommendations: List[str]


class ContributionValidator:
    """
    Validates standalone contributions of QE research papers for publication assessment.
    """
    
    def __init__(self):
        """Initialize the ContributionValidator."""
        self.min_contribution_threshold = 0.7
        self.novelty_categories = [
            'theoretical_innovation', 'empirical_methodology', 'data_sources',
            'identification_strategy', 'policy_insights', 'international_scope'
        ]
    
    def domestic_effects_contribution_test(self, 
                                         h1_metrics: PaperMetrics,
                                         h2_metrics: PaperMetrics,
                                         methodology_details: Dict[str, Any]) -> ContributionAssessment:
        """
        Test standalone contribution of domestic effects paper (Hypotheses 1 & 2).
        
        Args:
            h1_metrics: Metrics for threshold effects hypothesis
            h2_metrics: Metrics for investment channels hypothesis
            methodology_details: Details about methodology used
            
        Returns:
            ContributionAssessment for domestic effects paper
        """
        paper_type = "Domestic QE Effects"
        
        # Assess theoretical contribution
        theoretical_contribution = self._assess_theoretical_contribution(
            [h1_metrics, h2_metrics], methodology_details, "domestic"
        )
        
        # Assess empirical contribution
        empirical_contribution = self._assess_empirical_contribution(
            [h1_metrics, h2_metrics], methodology_details, "domestic"
        )
        
        # Assess methodological contribution
        methodological_contribution = self._assess_methodological_contribution(
            methodology_details, "domestic"
        )
        
        # Assess policy contribution
        policy_contribution = self._assess_policy_contribution(
            [h1_metrics, h2_metrics], "domestic"
        )
        
        # Calculate overall contribution score
        contribution_score = np.average([
            theoretical_contribution, empirical_contribution,
            methodological_contribution, policy_contribution
        ], weights=[0.3, 0.3, 0.2, 0.2])
        
        # Assess novelty across categories
        novelty_assessment = self._assess_novelty_dimensions(
            [h1_metrics, h2_metrics], methodology_details, "domestic"
        )
        
        # Determine publication readiness
        publication_readiness = (
            contribution_score >= self.min_contribution_threshold and
            np.mean(list(novelty_assessment.values())) >= 0.6
        )
        
        # Identify strengths and weaknesses
        strengths, weaknesses = self._identify_strengths_weaknesses(
            theoretical_contribution, empirical_contribution,
            methodological_contribution, policy_contribution, "domestic"
        )
        
        # Generate improvement recommendations
        improvement_recommendations = self._generate_improvement_recommendations(
            theoretical_contribution, empirical_contribution,
            methodological_contribution, policy_contribution, "domestic"
        )
        
        return ContributionAssessment(
            paper_type=paper_type,
            contribution_score=contribution_score,
            theoretical_contribution=theoretical_contribution,
            empirical_contribution=empirical_contribution,
            methodological_contribution=methodological_contribution,
            policy_contribution=policy_contribution,
            novelty_assessment=novelty_assessment,
            publication_readiness=publication_readiness,
            strengths=strengths,
            weaknesses=weaknesses,
            improvement_recommendations=improvement_recommendations
        )
    
    def international_effects_contribution_test(self,
                                              h3_metrics: PaperMetrics,
                                              methodology_details: Dict[str, Any]) -> ContributionAssessment:
        """
        Test standalone contribution of international effects paper (Hypothesis 3).
        
        Args:
            h3_metrics: Metrics for international spillovers hypothesis
            methodology_details: Details about methodology used
            
        Returns:
            ContributionAssessment for international effects paper
        """
        paper_type = "International QE Spillovers"
        
        # Assess theoretical contribution
        theoretical_contribution = self._assess_theoretical_contribution(
            [h3_metrics], methodology_details, "international"
        )
        
        # Assess empirical contribution
        empirical_contribution = self._assess_empirical_contribution(
            [h3_metrics], methodology_details, "international"
        )
        
        # Assess methodological contribution
        methodological_contribution = self._assess_methodological_contribution(
            methodology_details, "international"
        )
        
        # Assess policy contribution
        policy_contribution = self._assess_policy_contribution(
            [h3_metrics], "international"
        )
        
        # Calculate overall contribution score (higher weight on empirical for international)
        contribution_score = np.average([
            theoretical_contribution, empirical_contribution,
            methodological_contribution, policy_contribution
        ], weights=[0.2, 0.4, 0.2, 0.2])
        
        # Assess novelty across categories
        novelty_assessment = self._assess_novelty_dimensions(
            [h3_metrics], methodology_details, "international"
        )
        
        # Determine publication readiness
        publication_readiness = (
            contribution_score >= self.min_contribution_threshold and
            np.mean(list(novelty_assessment.values())) >= 0.6
        )
        
        # Identify strengths and weaknesses
        strengths, weaknesses = self._identify_strengths_weaknesses(
            theoretical_contribution, empirical_contribution,
            methodological_contribution, policy_contribution, "international"
        )
        
        # Generate improvement recommendations
        improvement_recommendations = self._generate_improvement_recommendations(
            theoretical_contribution, empirical_contribution,
            methodological_contribution, policy_contribution, "international"
        )
        
        return ContributionAssessment(
            paper_type=paper_type,
            contribution_score=contribution_score,
            theoretical_contribution=theoretical_contribution,
            empirical_contribution=empirical_contribution,
            methodological_contribution=methodological_contribution,
            policy_contribution=policy_contribution,
            novelty_assessment=novelty_assessment,
            publication_readiness=publication_readiness,
            strengths=strengths,
            weaknesses=weaknesses,
            improvement_recommendations=improvement_recommendations
        )
    
    def novelty_assessment_test(self, 
                              paper_metrics: List[PaperMetrics],
                              methodology_details: Dict[str, Any],
                              paper_type: str) -> Dict[str, Any]:
        """
        Comprehensive novelty assessment for paper's unique contribution.
        
        Args:
            paper_metrics: List of metrics for hypotheses in the paper
            methodology_details: Details about methodology used
            paper_type: Type of paper ("domestic" or "international")
            
        Returns:
            Dictionary with detailed novelty assessment
        """
        novelty_scores = self._assess_novelty_dimensions(
            paper_metrics, methodology_details, paper_type
        )
        
        # Calculate overall novelty score
        overall_novelty = np.mean(list(novelty_scores.values()))
        
        # Identify strongest novelty dimensions
        strongest_dimensions = sorted(novelty_scores.items(), 
                                    key=lambda x: x[1], reverse=True)[:3]
        
        # Identify areas needing improvement
        weak_dimensions = [(k, v) for k, v in novelty_scores.items() if v < 0.5]
        
        # Generate novelty enhancement recommendations
        enhancement_recommendations = self._generate_novelty_recommendations(
            novelty_scores, paper_type
        )
        
        # Assess competitive advantage
        competitive_advantage = self._assess_competitive_advantage(
            novelty_scores, methodology_details, paper_type
        )
        
        return {
            'overall_novelty': overall_novelty,
            'dimension_scores': novelty_scores,
            'strongest_dimensions': strongest_dimensions,
            'weak_dimensions': weak_dimensions,
            'enhancement_recommendations': enhancement_recommendations,
            'competitive_advantage': competitive_advantage,
            'publication_potential': overall_novelty >= 0.6
        }
    
    def compare_contribution_strategies(self,
                                      unified_metrics: List[PaperMetrics],
                                      split_domestic_metrics: List[PaperMetrics],
                                      split_international_metrics: List[PaperMetrics],
                                      methodology_details: Dict[str, Any]) -> Dict[str, Any]:
        """
        Compare contribution strength between unified vs split paper strategies.
        
        Args:
            unified_metrics: Metrics for unified paper approach
            split_domestic_metrics: Metrics for domestic effects paper
            split_international_metrics: Metrics for international effects paper
            methodology_details: Details about methodology used
            
        Returns:
            Dictionary comparing strategies and recommending optimal approach
        """
        # Assess unified paper contribution
        unified_contribution = self._assess_unified_contribution(
            unified_metrics, methodology_details
        )
        
        # Assess split papers contributions
        domestic_assessment = self.domestic_effects_contribution_test(
            split_domestic_metrics[0], split_domestic_metrics[1], methodology_details
        )
        
        international_assessment = self.international_effects_contribution_test(
            split_international_metrics[0], methodology_details
        )
        
        # Calculate combined split contribution
        split_contribution = np.average([
            domestic_assessment.contribution_score,
            international_assessment.contribution_score
        ])
        
        # Determine optimal strategy
        if split_contribution > unified_contribution + 0.1:
            recommendation = "split"
            rationale = "Split papers show significantly stronger individual contributions"
        elif unified_contribution > split_contribution + 0.1:
            recommendation = "unified"
            rationale = "Unified paper provides stronger overall contribution"
        else:
            recommendation = "either"
            rationale = "Both strategies show similar contribution strength"
        
        return {
            'unified_contribution': unified_contribution,
            'split_contribution': split_contribution,
            'domestic_contribution': domestic_assessment.contribution_score,
            'international_contribution': international_assessment.contribution_score,
            'recommended_strategy': recommendation,
            'rationale': rationale,
            'domestic_readiness': domestic_assessment.publication_readiness,
            'international_readiness': international_assessment.publication_readiness,
            'strategy_comparison': {
                'unified_strengths': self._get_unified_strengths(unified_metrics),
                'split_strengths': [domestic_assessment.strengths, international_assessment.strengths],
                'unified_weaknesses': self._get_unified_weaknesses(unified_metrics),
                'split_weaknesses': [domestic_assessment.weaknesses, international_assessment.weaknesses]
            }
        }
    
    def _assess_theoretical_contribution(self, 
                                       metrics_list: List[PaperMetrics],
                                       methodology_details: Dict[str, Any],
                                       paper_type: str) -> float:
        """Assess theoretical contribution of the paper."""
        # Average theoretical foundation across hypotheses
        avg_theoretical = np.mean([m.theoretical_foundation for m in metrics_list])
        
        # Assess theoretical innovation based on methodology
        theoretical_methods = methodology_details.get('theoretical_methods', [])
        innovation_bonus = len(theoretical_methods) * 0.1
        
        # Paper type specific adjustments
        if paper_type == "domestic":
            # Domestic papers need strong threshold theory
            threshold_theory = methodology_details.get('threshold_theory_strength', 0.5)
            type_adjustment = threshold_theory * 0.2
        else:  # international
            # International papers need transmission mechanism theory
            transmission_theory = methodology_details.get('transmission_theory_strength', 0.5)
            type_adjustment = transmission_theory * 0.2
        
        return min(avg_theoretical + innovation_bonus + type_adjustment, 1.0)
    
    def _assess_empirical_contribution(self,
                                     metrics_list: List[PaperMetrics],
                                     methodology_details: Dict[str, Any],
                                     paper_type: str) -> float:
        """Assess empirical contribution of the paper."""
        # Average empirical novelty and statistical significance
        avg_novelty = np.mean([m.empirical_novelty for m in metrics_list])
        avg_significance = np.mean([m.statistical_significance for m in metrics_list])
        
        # Assess data contribution
        data_novelty = methodology_details.get('data_novelty', 0.5)
        
        # Assess identification strength
        identification_strength = methodology_details.get('identification_strength', 0.5)
        
        # Sample size adequacy
        total_sample = sum(m.sample_size for m in metrics_list)
        sample_adequacy = min(total_sample / 200, 1.0)  # Normalize to 200 obs
        
        return np.average([avg_novelty, avg_significance, data_novelty, 
                          identification_strength, sample_adequacy],
                         weights=[0.3, 0.2, 0.2, 0.2, 0.1])
    
    def _assess_methodological_contribution(self,
                                          methodology_details: Dict[str, Any],
                                          paper_type: str) -> float:
        """Assess methodological contribution of the paper."""
        methods_used = methodology_details.get('methods', [])
        
        # Count advanced methods
        advanced_methods = ['hansen_threshold', 'iv_regression', 'local_projections', 
                          'panel_var', 'structural_breaks']
        advanced_count = len([m for m in methods_used if m in advanced_methods])
        
        # Assess robustness
        robustness_tests = methodology_details.get('robustness_tests', 0)
        robustness_score = min(robustness_tests / 5.0, 1.0)
        
        # Assess innovation in methodology
        methodological_innovation = methodology_details.get('methodological_innovation', 0.5)
        
        # Paper type specific methods
        if paper_type == "domestic":
            preferred_methods = ['hansen_threshold', 'str_model', 'iv_regression']
        else:  # international
            preferred_methods = ['panel_var', 'event_study', 'cross_country']
        
        method_fit = len([m for m in methods_used if m in preferred_methods]) / len(preferred_methods)
        
        return np.average([advanced_count/3, robustness_score, methodological_innovation, method_fit],
                         weights=[0.3, 0.3, 0.2, 0.2])
    
    def _assess_policy_contribution(self,
                                  metrics_list: List[PaperMetrics],
                                  paper_type: str) -> float:
        """Assess policy contribution of the paper."""
        # Average policy relevance across hypotheses
        avg_policy = np.mean([m.policy_relevance for m in metrics_list])
        
        # Paper type specific policy relevance
        if paper_type == "domestic":
            # Domestic papers contribute to QE design and implementation
            policy_multiplier = 1.1
        else:  # international
            # International papers contribute to coordination and spillover management
            policy_multiplier = 1.0
        
        return min(avg_policy * policy_multiplier, 1.0)
    
    def _assess_novelty_dimensions(self,
                                 metrics_list: List[PaperMetrics],
                                 methodology_details: Dict[str, Any],
                                 paper_type: str) -> Dict[str, float]:
        """Assess novelty across different dimensions."""
        novelty_scores = {}
        
        # Theoretical innovation
        novelty_scores['theoretical_innovation'] = np.mean([m.theoretical_foundation for m in metrics_list])
        
        # Empirical methodology
        methods_novelty = methodology_details.get('methodological_innovation', 0.5)
        novelty_scores['empirical_methodology'] = methods_novelty
        
        # Data sources
        data_novelty = methodology_details.get('data_novelty', 0.5)
        novelty_scores['data_sources'] = data_novelty
        
        # Identification strategy
        identification_novelty = methodology_details.get('identification_strength', 0.5)
        novelty_scores['identification_strategy'] = identification_novelty
        
        # Policy insights
        novelty_scores['policy_insights'] = np.mean([m.policy_relevance for m in metrics_list])
        
        # International scope (higher for international papers)
        if paper_type == "international":
            novelty_scores['international_scope'] = 0.9
        else:
            novelty_scores['international_scope'] = 0.3
        
        return novelty_scores
    
    def _identify_strengths_weaknesses(self,
                                     theoretical: float, empirical: float,
                                     methodological: float, policy: float,
                                     paper_type: str) -> Tuple[List[str], List[str]]:
        """Identify paper strengths and weaknesses."""
        strengths = []
        weaknesses = []
        
        contributions = {
            'theoretical': theoretical,
            'empirical': empirical,
            'methodological': methodological,
            'policy': policy
        }
        
        for contrib_type, score in contributions.items():
            if score >= 0.8:
                strengths.append(f"Strong {contrib_type} contribution (score: {score:.2f})")
            elif score < 0.5:
                weaknesses.append(f"Weak {contrib_type} contribution (score: {score:.2f})")
        
        # Paper type specific assessments
        if paper_type == "domestic":
            if theoretical >= 0.7 and methodological >= 0.7:
                strengths.append("Well-suited for top-tier monetary economics journals")
            if empirical < 0.6:
                weaknesses.append("Empirical robustness may need strengthening for JME/AEJ")
        else:  # international
            if empirical >= 0.8:
                strengths.append("Strong empirical foundation for international finance journals")
            if theoretical < 0.6:
                weaknesses.append("May benefit from stronger theoretical framework")
        
        return strengths, weaknesses
    
    def _generate_improvement_recommendations(self,
                                           theoretical: float, empirical: float,
                                           methodological: float, policy: float,
                                           paper_type: str) -> List[str]:
        """Generate recommendations for improving contribution."""
        recommendations = []
        
        if theoretical < 0.7:
            if paper_type == "domestic":
                recommendations.append("Develop stronger theoretical foundation for threshold effects")
            else:
                recommendations.append("Enhance theoretical framework for international transmission")
        
        if empirical < 0.7:
            recommendations.append("Add more robustness tests and sensitivity analysis")
            recommendations.append("Consider expanding sample size or time period")
        
        if methodological < 0.7:
            recommendations.append("Incorporate additional advanced econometric methods")
            recommendations.append("Strengthen identification strategy")
        
        if policy < 0.6:
            recommendations.append("Enhance policy implications and practical relevance")
            recommendations.append("Connect findings to current policy debates")
        
        return recommendations
    
    def _generate_novelty_recommendations(self,
                                        novelty_scores: Dict[str, float],
                                        paper_type: str) -> List[str]:
        """Generate recommendations for enhancing novelty."""
        recommendations = []
        
        for dimension, score in novelty_scores.items():
            if score < 0.5:
                if dimension == 'theoretical_innovation':
                    recommendations.append("Develop more innovative theoretical insights")
                elif dimension == 'empirical_methodology':
                    recommendations.append("Adopt cutting-edge econometric methods")
                elif dimension == 'data_sources':
                    recommendations.append("Incorporate novel or proprietary datasets")
                elif dimension == 'identification_strategy':
                    recommendations.append("Develop more creative identification approaches")
                elif dimension == 'policy_insights':
                    recommendations.append("Generate more actionable policy recommendations")
        
        return recommendations
    
    def _assess_competitive_advantage(self,
                                    novelty_scores: Dict[str, float],
                                    methodology_details: Dict[str, Any],
                                    paper_type: str) -> Dict[str, Any]:
        """Assess competitive advantage in the literature."""
        # Identify unique selling points
        strong_dimensions = [k for k, v in novelty_scores.items() if v >= 0.8]
        
        # Assess methodological advantages
        advanced_methods = methodology_details.get('methods', [])
        method_advantages = [m for m in advanced_methods if m in 
                           ['hansen_threshold', 'local_projections', 'panel_var']]
        
        # Assess data advantages
        data_advantages = []
        if methodology_details.get('data_novelty', 0) >= 0.7:
            data_advantages.append("Novel dataset or data construction")
        
        # Overall competitive position
        total_advantages = len(strong_dimensions) + len(method_advantages) + len(data_advantages)
        
        if total_advantages >= 4:
            competitive_position = "Strong"
        elif total_advantages >= 2:
            competitive_position = "Moderate"
        else:
            competitive_position = "Weak"
        
        return {
            'competitive_position': competitive_position,
            'unique_selling_points': strong_dimensions,
            'methodological_advantages': method_advantages,
            'data_advantages': data_advantages,
            'total_advantage_count': total_advantages
        }
    
    def _assess_unified_contribution(self,
                                   unified_metrics: List[PaperMetrics],
                                   methodology_details: Dict[str, Any]) -> float:
        """Assess contribution of unified paper approach."""
        # Average across all hypotheses
        avg_theoretical = np.mean([m.theoretical_foundation for m in unified_metrics])
        avg_empirical = np.mean([m.empirical_novelty for m in unified_metrics])
        avg_policy = np.mean([m.policy_relevance for m in unified_metrics])
        
        # Methodological diversity bonus for unified approach
        method_count = len(methodology_details.get('methods', []))
        diversity_bonus = min(method_count / 5.0, 0.2)
        
        # Scope bonus for comprehensive coverage
        scope_bonus = 0.1 if len(unified_metrics) >= 3 else 0
        
        base_contribution = np.average([avg_theoretical, avg_empirical, avg_policy],
                                     weights=[0.35, 0.35, 0.3])
        
        return min(base_contribution + diversity_bonus + scope_bonus, 1.0)
    
    def _get_unified_strengths(self, unified_metrics: List[PaperMetrics]) -> List[str]:
        """Get strengths of unified paper approach."""
        strengths = ["Comprehensive coverage of QE effects"]
        
        if len(unified_metrics) >= 3:
            strengths.append("Multiple complementary hypotheses")
        
        avg_novelty = np.mean([m.empirical_novelty for m in unified_metrics])
        if avg_novelty >= 0.8:
            strengths.append("Strong empirical novelty across hypotheses")
        
        return strengths
    
    def _get_unified_weaknesses(self, unified_metrics: List[PaperMetrics]) -> List[str]:
        """Get weaknesses of unified paper approach."""
        weaknesses = []
        
        # Check for weak individual hypotheses
        weak_hypotheses = [i for i, m in enumerate(unified_metrics) 
                          if m.empirical_novelty < 0.6]
        
        if len(weak_hypotheses) > 0:
            weaknesses.append(f"Weak contribution from hypothesis {weak_hypotheses[0] + 1}")
        
        # Check for length/complexity concerns
        total_methods = sum(m.methodology_count for m in unified_metrics)
        if total_methods > 6:
            weaknesses.append("May be too complex for single paper")
        
        return weaknesses