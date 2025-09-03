"""
Local Projections Validation Against Known Impulse Response Patterns

This module validates our Local Projections implementation against known
impulse response patterns from the econometric literature.
"""

import numpy as np
import pandas as pd
import pytest
from scipy import stats
from sklearn.metrics import mean_squared_error
import sys
import os

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.models import LocalProjections


class LocalProjectionsValidator:
    """Validates Local Projections against known impulse response patterns."""
    
    def __init__(self):
        self.tolerance = 0.1  # 10% tolerance for impulse response comparisons
        self.literature_benchmarks = self._load_literature_benchmarks()
    
    def _load_literature_benchmarks(self):
        """
        Load benchmark impulse response patterns from the literature.
        
        Based on Jordà (2005) and subsequent applications in monetary policy
        and QE literature.
        """
        return {
            'jorda_2005_monetary': {
                'description': 'Jordà (2005) monetary policy shock responses',
                'peak_horizon': 2,  # Peak response at 2 quarters
                'peak_magnitude_range': (0.3, 0.7),
                'persistence_quarters': 6,
                'decay_pattern': 'exponential',
                'confidence_interval_coverage': 0.95
            },
            'qe_investment_response': {
                'description': 'QE effects on investment from literature',
                'peak_horizon': 3,  # Peak at 3 quarters
                'peak_magnitude_range': (0.2, 0.5),
                'persistence_quarters': 8,
                'decay_pattern': 'gradual',
                'confidence_interval_coverage': 0.90
            },
            'financial_shock_response': {
                'description': 'Financial shock impulse responses',
                'peak_horizon': 1,  # Immediate peak
                'peak_magnitude_range': (0.4, 0.8),
                'persistence_quarters': 4,
                'decay_pattern': 'fast',
                'confidence_interval_coverage': 0.95
            }
        }
    
    def generate_jorda_2005_style_data(self, n_obs=200, n_horizons=8, seed=42):
        """
        Generate data matching Jordà (2005) monetary policy study structure.
        """
        np.random.seed(seed)
        
        # True impulse response: peaks at horizon 2, then decays exponentially
        true_impulse_response = np.array([
            0.0,   # Horizon 0: no immediate effect
            0.2,   # Horizon 1: building up
            0.5,   # Horizon 2: peak effect
            0.4,   # Horizon 3: decay starts
            0.25,  # Horizon 4: continued decay
            0.15,  # Horizon 5: further decay
            0.08,  # Horizon 6: small effect
            0.03   # Horizon 7: nearly zero
        ])
        
        # Generate monetary policy shock (instrument)
        monetary_shock = np.random.normal(0, 1, n_obs)
        
        # Generate control variables (typical macro controls)
        gdp_growth_lag = np.random.normal(0.02, 0.01, n_obs)
        inflation_lag = np.random.normal(0.025, 0.008, n_obs)
        interest_rate_lag = np.random.normal(0.04, 0.015, n_obs)
        
        controls = np.column_stack([gdp_growth_lag, inflation_lag, interest_rate_lag])
        
        # Generate dependent variable (e.g., output growth)
        y = np.zeros(n_obs)
        
        for t in range(4, n_obs):  # Start after initial lags
            # Add AR component
            for lag in range(1, 4):
                if t - lag >= 0:
                    y[t] += 0.1 * y[t - lag]
            
            # Add impulse response effects
            for h in range(min(len(true_impulse_response), t + 1)):
                if t - h >= 0:
                    y[t] += true_impulse_response[h] * monetary_shock[t - h]
            
            # Add control variable effects
            y[t] += 0.3 * gdp_growth_lag[t] + 0.2 * inflation_lag[t] - 0.1 * interest_rate_lag[t]
            
            # Add realistic noise
            y[t] += np.random.normal(0, 0.01)
        
        return {
            'y': y,
            'shock': monetary_shock,
            'controls': controls,
            'true_impulse_response': true_impulse_response,
            'n_horizons': n_horizons,
            'benchmark': 'jorda_2005_monetary'
        }
    
    def generate_qe_investment_response_data(self, n_obs=180, n_horizons=10, seed=42):
        """
        Generate data matching QE investment response patterns from literature.
        """
        np.random.seed(seed)
        
        # True impulse response: gradual build-up, peak at horizon 3, slow decay
        true_impulse_response = np.array([
            0.0,   # Horizon 0: no immediate effect
            0.1,   # Horizon 1: small initial effect
            0.25,  # Horizon 2: building up
            0.4,   # Horizon 3: peak effect
            0.35,  # Horizon 4: slight decline
            0.28,  # Horizon 5: gradual decay
            0.2,   # Horizon 6: continued decline
            0.12,  # Horizon 7: smaller effect
            0.06,  # Horizon 8: fading
            0.02   # Horizon 9: nearly zero
        ])
        
        # Generate QE shock
        qe_shock = np.random.normal(0, 1, n_obs)
        
        # Generate financial controls
        bond_yield_lag = np.random.normal(0.03, 0.01, n_obs)
        vix_lag = np.random.lognormal(3, 0.3, n_obs)
        credit_spread_lag = np.random.normal(0.02, 0.005, n_obs)
        
        controls = np.column_stack([bond_yield_lag, vix_lag, credit_spread_lag])
        
        # Generate dependent variable (investment growth)
        y = np.zeros(n_obs)
        
        for t in range(4, n_obs):
            # Add AR component (investment has persistence)
            for lag in range(1, 3):
                if t - lag >= 0:
                    y[t] += 0.15 * y[t - lag]
            
            # Add QE impulse response effects
            for h in range(min(len(true_impulse_response), t + 1)):
                if t - h >= 0:
                    y[t] += true_impulse_response[h] * qe_shock[t - h]
            
            # Add control effects
            y[t] += (-0.2 * bond_yield_lag[t] - 0.001 * vix_lag[t] - 
                    0.3 * credit_spread_lag[t])
            
            # Add noise (higher for financial data)
            y[t] += np.random.normal(0, 0.015)
        
        return {
            'y': y,
            'shock': qe_shock,
            'controls': controls,
            'true_impulse_response': true_impulse_response,
            'n_horizons': n_horizons,
            'benchmark': 'qe_investment_response'
        }
    
    def validate_impulse_response_accuracy(self, data, estimated_ir, benchmark_name):
        """
        Validate estimated impulse response against true response.
        """
        true_ir = data['true_impulse_response']
        benchmark = self.literature_benchmarks[benchmark_name]
        
        results = {
            'impulse_response_mse': None,
            'peak_timing_accuracy': False,
            'peak_magnitude_accuracy': False,
            'persistence_accuracy': False,
            'shape_correlation': None,
            'validation_passed': False
        }
        
        try:
            # Ensure same length for comparison
            min_length = min(len(true_ir), len(estimated_ir))
            true_ir_comp = true_ir[:min_length]
            estimated_ir_comp = estimated_ir[:min_length]
            
            # Calculate MSE
            mse = mean_squared_error(true_ir_comp, estimated_ir_comp)
            results['impulse_response_mse'] = mse
            
            # Check peak timing
            true_peak_horizon = np.argmax(np.abs(true_ir_comp))
            estimated_peak_horizon = np.argmax(np.abs(estimated_ir_comp))
            peak_timing_error = abs(true_peak_horizon - estimated_peak_horizon)
            results['peak_timing_accuracy'] = peak_timing_error <= 1  # Allow 1 period difference
            results['true_peak_horizon'] = true_peak_horizon
            results['estimated_peak_horizon'] = estimated_peak_horizon
            
            # Check peak magnitude
            true_peak_magnitude = np.max(np.abs(true_ir_comp))
            estimated_peak_magnitude = np.max(np.abs(estimated_ir_comp))
            peak_magnitude_error = abs(true_peak_magnitude - estimated_peak_magnitude)
            
            # Check if within literature range
            lit_range = benchmark['peak_magnitude_range']
            magnitude_in_range = (lit_range[0] <= estimated_peak_magnitude <= lit_range[1])
            magnitude_close_to_true = peak_magnitude_error < self.tolerance
            
            results['peak_magnitude_accuracy'] = magnitude_in_range and magnitude_close_to_true
            results['true_peak_magnitude'] = true_peak_magnitude
            results['estimated_peak_magnitude'] = estimated_peak_magnitude
            
            # Check persistence (how long effects last)
            significance_threshold = 0.05  # 5% of peak
            true_persistence = np.sum(np.abs(true_ir_comp) > significance_threshold * true_peak_magnitude)
            estimated_persistence = np.sum(np.abs(estimated_ir_comp) > significance_threshold * estimated_peak_magnitude)
            persistence_error = abs(true_persistence - estimated_persistence)
            
            results['persistence_accuracy'] = persistence_error <= 2  # Allow 2 period difference
            results['true_persistence'] = true_persistence
            results['estimated_persistence'] = estimated_persistence
            
            # Check shape correlation
            if len(true_ir_comp) > 1 and len(estimated_ir_comp) > 1:
                correlation = np.corrcoef(true_ir_comp, estimated_ir_comp)[0, 1]
                results['shape_correlation'] = correlation
                shape_ok = correlation > 0.7  # Strong positive correlation
            else:
                shape_ok = False
            
            # Overall validation
            results['validation_passed'] = (
                mse < self.tolerance and
                results['peak_timing_accuracy'] and
                results['peak_magnitude_accuracy'] and
                results['persistence_accuracy'] and
                shape_ok
            )
            
        except Exception as e:
            results['error'] = str(e)
            results['validation_passed'] = False
        
        return results
    
    def validate_confidence_intervals(self, lp_model, data):
        """
        Validate confidence interval coverage and properties.
        """
        results = {
            'confidence_intervals_available': False,
            'coverage_adequate': False,
            'intervals_reasonable': False,
            'validation_passed': False
        }
        
        try:
            # Check if confidence intervals are available
            if hasattr(lp_model, 'get_impulse_responses'):
                ir_results = lp_model.get_impulse_responses()
                if 'lower_ci' in ir_results.columns and 'upper_ci' in ir_results.columns:
                    results['confidence_intervals_available'] = True
                    
                    ci_lower = ir_results['lower_ci'].values
                    ci_upper = ir_results['upper_ci'].values
                    point_estimates = ir_results['coefficient'].values
                
                # Check that intervals are reasonable (upper > lower)
                intervals_valid = np.all(ci_upper >= ci_lower)
                
                # Check that point estimates are within intervals
                point_in_intervals = np.all(
                    (point_estimates >= ci_lower) & (point_estimates <= ci_upper)
                )
                
                results['intervals_reasonable'] = intervals_valid and point_in_intervals
                
                # For coverage, we'd need multiple simulations, so we approximate
                # by checking that intervals are not too narrow or too wide
                interval_widths = ci_upper - ci_lower
                avg_width = np.mean(interval_widths)
                point_std = np.std(point_estimates)
                
                # Reasonable if average width is between 1-4 times point estimate std
                width_reasonable = 1.0 * point_std <= avg_width <= 4.0 * point_std
                results['coverage_adequate'] = width_reasonable
                results['average_interval_width'] = avg_width
                results['point_estimate_std'] = point_std
                
            # Overall validation
            results['validation_passed'] = (
                results['confidence_intervals_available'] and
                results['intervals_reasonable'] and
                results['coverage_adequate']
            )
            
        except Exception as e:
            results['error'] = str(e)
            results['validation_passed'] = False
        
        return results
    
    def validate_against_jorda_2005(self):
        """
        Validate against Jordà (2005) monetary policy benchmark.
        """
        print("Validating against Jordà (2005) monetary policy benchmark...")
        
        # Generate Jordà 2005 style data
        data = self.generate_jorda_2005_style_data()
        
        # Fit Local Projections model
        lp_model = LocalProjections()
        
        try:
            # Convert to pandas Series/DataFrame for LocalProjections
            y_series = pd.Series(data['y'])
            shock_series = pd.Series(data['shock'])
            controls_df = pd.DataFrame(data['controls']) if data['controls'] is not None else None
            
            lp_model.fit(
                y=y_series,
                shock=shock_series,
                controls=controls_df,
                lags=4
            )
            
            # Validate impulse response
            ir_data = lp_model.get_impulse_responses()
            estimated_ir = ir_data['coefficient'].values
            ir_results = self.validate_impulse_response_accuracy(
                data, estimated_ir, 'jorda_2005_monetary'
            )
            
            # Validate confidence intervals
            ci_results = self.validate_confidence_intervals(lp_model, data)
            
            # Combine results
            results = {
                'model_converged': True,
                'impulse_response_validation': ir_results,
                'confidence_interval_validation': ci_results,
                'validation_passed': (
                    ir_results.get('validation_passed', False) and
                    ci_results.get('validation_passed', False)
                )
            }
            
        except Exception as e:
            results = {
                'model_converged': False,
                'error': str(e),
                'validation_passed': False
            }
        
        return results
    
    def validate_against_qe_literature(self):
        """
        Validate against QE investment response literature.
        """
        print("Validating against QE investment response literature...")
        
        # Generate QE style data
        data = self.generate_qe_investment_response_data()
        
        # Fit Local Projections model
        lp_model = LocalProjections()
        
        try:
            # Convert to pandas Series/DataFrame for LocalProjections
            y_series = pd.Series(data['y'])
            shock_series = pd.Series(data['shock'])
            controls_df = pd.DataFrame(data['controls']) if data['controls'] is not None else None
            
            lp_model.fit(
                y=y_series,
                shock=shock_series,
                controls=controls_df,
                lags=3
            )
            
            # Validate impulse response
            ir_data = lp_model.get_impulse_responses()
            estimated_ir = ir_data['coefficient'].values
            ir_results = self.validate_impulse_response_accuracy(
                data, estimated_ir, 'qe_investment_response'
            )
            
            # Validate confidence intervals
            ci_results = self.validate_confidence_intervals(lp_model, data)
            
            # Combine results
            results = {
                'model_converged': True,
                'impulse_response_validation': ir_results,
                'confidence_interval_validation': ci_results,
                'validation_passed': (
                    ir_results.get('validation_passed', False) and
                    ci_results.get('validation_passed', False)
                )
            }
            
        except Exception as e:
            results = {
                'model_converged': False,
                'error': str(e),
                'validation_passed': False
            }
        
        return results
    
    def run_comprehensive_lp_validation(self):
        """
        Run comprehensive validation of Local Projections implementation.
        """
        print("="*60)
        print("LOCAL PROJECTIONS VALIDATION AGAINST LITERATURE")
        print("="*60)
        
        validation_results = {
            'jorda_2005_validation': {},
            'qe_literature_validation': {},
            'overall_validation_passed': False
        }
        
        try:
            # Validate against Jordà (2005)
            jorda_results = self.validate_against_jorda_2005()
            validation_results['jorda_2005_validation'] = jorda_results
            
            # Validate against QE literature
            qe_results = self.validate_against_qe_literature()
            validation_results['qe_literature_validation'] = qe_results
            
            # Overall assessment
            all_passed = (
                jorda_results.get('validation_passed', False) and
                qe_results.get('validation_passed', False)
            )
            
            validation_results['overall_validation_passed'] = all_passed
            
            # Print summary
            print("\nVALIDATION SUMMARY:")
            print("-" * 40)
            print(f"Jordà (2005) Benchmark: {'PASS' if jorda_results.get('validation_passed') else 'FAIL'}")
            print(f"QE Literature Benchmark: {'PASS' if qe_results.get('validation_passed') else 'FAIL'}")
            print(f"\nOverall LP Validation: {'PASS' if all_passed else 'FAIL'}")
            print("="*60)
            
        except Exception as e:
            validation_results['error'] = str(e)
            validation_results['overall_validation_passed'] = False
            print(f"Local Projections validation failed with error: {e}")
        
        return validation_results


def run_local_projections_validation():
    """Run Local Projections validation."""
    validator = LocalProjectionsValidator()
    return validator.run_comprehensive_lp_validation()


if __name__ == "__main__":
    results = run_local_projections_validation()
    exit_code = 0 if results['overall_validation_passed'] else 1
    exit(exit_code)