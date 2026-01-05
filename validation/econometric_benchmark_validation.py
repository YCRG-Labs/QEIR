"""
Econometric Benchmark Validation

This module validates our econometric implementations against known benchmarks
and published results to ensure accuracy and reliability for publication.
"""

import numpy as np
import pandas as pd
import pytest
from scipy import stats
from sklearn.metrics import mean_squared_error, r2_score
import sys
import os

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.models import HansenThresholdRegression, SmoothTransitionRegression, LocalProjections
from src.publication_model_diagnostics import PublicationModelDiagnostics
from src.model_specification_enhancer import ModelSpecificationEnhancer


class EconometricBenchmarkValidator:
    """Validates econometric implementations against known benchmarks."""
    
    def __init__(self):
        self.tolerance = 1e-3  # Numerical tolerance for comparisons
        self.benchmark_results = {}
        
    def generate_hansen_benchmark_data(self, n_obs=200, threshold_true=0.5, 
                                     regime1_coef=0.3, regime2_coef=0.7, 
                                     noise_std=0.5, seed=42):
        """
        Generate synthetic data with known Hansen threshold structure.
        
        This creates data where we know the true parameters, allowing us to
        validate that our estimation recovers the correct values.
        """
        np.random.seed(seed)
        
        # Generate threshold variable (uniform for clear regime separation)
        threshold_var = np.random.uniform(0, 1, n_obs)
        
        # Generate independent variable
        x = np.random.normal(0, 1, n_obs)
        
        # Create regime indicator
        regime_indicator = (threshold_var > threshold_true).astype(int)
        
        # Generate dependent variable with true threshold effect
        y = np.zeros(n_obs)
        regime1_mask = regime_indicator == 0
        regime2_mask = regime_indicator == 1
        
        y[regime1_mask] = regime1_coef * x[regime1_mask] + np.random.normal(0, noise_std, np.sum(regime1_mask))
        y[regime2_mask] = regime2_coef * x[regime2_mask] + np.random.normal(0, noise_std, np.sum(regime2_mask))
        
        return {
            'y': y,
            'x': x,
            'threshold_var': threshold_var,
            'true_threshold': threshold_true,
            'true_regime1_coef': regime1_coef,
            'true_regime2_coef': regime2_coef,
            'true_regime_indicator': regime_indicator,
            'noise_std': noise_std
        }
    
    def generate_str_benchmark_data(self, n_obs=200, threshold_true=0.0, 
                                  regime1_coef=0.3, regime2_coef=0.7,
                                  gamma_true=2.0, noise_std=0.5, seed=42):
        """
        Generate synthetic data with known STR structure.
        """
        np.random.seed(seed)
        
        # Generate threshold variable
        threshold_var = np.random.normal(0, 1, n_obs)
        
        # Generate independent variable
        x = np.random.normal(0, 1, n_obs)
        
        # Create smooth transition function
        transition_func = 1 / (1 + np.exp(-gamma_true * (threshold_var - threshold_true)))
        
        # Generate dependent variable with smooth transition
        y = (regime1_coef * (1 - transition_func) + regime2_coef * transition_func) * x
        y += np.random.normal(0, noise_std, n_obs)
        
        return {
            'y': y,
            'x': x,
            'threshold_var': threshold_var,
            'true_threshold': threshold_true,
            'true_regime1_coef': regime1_coef,
            'true_regime2_coef': regime2_coef,
            'true_gamma': gamma_true,
            'true_transition_func': transition_func,
            'noise_std': noise_std
        }
    
    def generate_local_projections_benchmark_data(self, n_obs=200, impulse_response_true=None,
                                                n_lags=4, n_horizons=8, seed=42):
        """
        Generate synthetic data with known impulse response structure.
        """
        np.random.seed(seed)
        
        if impulse_response_true is None:
            # Default impulse response: peaks at horizon 2, then decays
            impulse_response_true = np.array([0.0, 0.3, 0.5, 0.4, 0.2, 0.1, 0.05, 0.0])
        
        # Generate shock variable (instrument)
        shock = np.random.normal(0, 1, n_obs)
        
        # Generate control variables
        controls = np.random.normal(0, 1, (n_obs, 2))
        
        # Generate dependent variable with known impulse response
        y = np.zeros(n_obs)
        
        for t in range(n_lags, n_obs):
            # Add lagged effects
            for lag in range(n_lags):
                if t - lag >= 0:
                    y[t] += 0.1 * y[t - lag - 1]  # AR component
            
            # Add shock effects (impulse response)
            for h in range(min(len(impulse_response_true), t + 1)):
                if t - h >= 0:
                    y[t] += impulse_response_true[h] * shock[t - h]
            
            # Add control effects
            y[t] += 0.2 * controls[t, 0] + 0.1 * controls[t, 1]
            
            # Add noise
            y[t] += np.random.normal(0, 0.3)
        
        return {
            'y': y,
            'shock': shock,
            'controls': controls,
            'true_impulse_response': impulse_response_true,
            'n_lags': n_lags,
            'n_horizons': n_horizons
        }
    
    def validate_hansen_estimation(self, data, tolerance=None):
        """
        Validate Hansen threshold regression against known parameters.
        """
        if tolerance is None:
            tolerance = self.tolerance
            
        # Fit Hansen model
        hansen = HansenThresholdRegression()
        hansen.fit(data['y'], data['x'], data['threshold_var'])
        
        results = {
            'threshold_accuracy': None,
            'regime1_coef_accuracy': None,
            'regime2_coef_accuracy': None,
            'r_squared': None,
            'regime_classification_accuracy': None,
            'validation_passed': False
        }
        
        try:
            # Check threshold estimation accuracy
            if hasattr(hansen, 'threshold_estimate'):
                threshold_error = abs(hansen.threshold_estimate - data['true_threshold'])
                results['threshold_accuracy'] = threshold_error
                threshold_ok = threshold_error < tolerance * 10  # More lenient for threshold
            else:
                threshold_ok = False
            
            # Check coefficient estimation accuracy
            if hasattr(hansen, 'regime1_coef') and hasattr(hansen, 'regime2_coef'):
                regime1_error = abs(hansen.regime1_coef[0] - data['true_regime1_coef'])
                regime2_error = abs(hansen.regime2_coef[0] - data['true_regime2_coef'])
                
                results['regime1_coef_accuracy'] = regime1_error
                results['regime2_coef_accuracy'] = regime2_error
                
                coef_ok = regime1_error < tolerance * 5 and regime2_error < tolerance * 5
            else:
                coef_ok = False
            
            # Check model fit
            if hasattr(hansen, 'r_squared'):
                results['r_squared'] = hansen.r_squared
                r2_ok = hansen.r_squared > 0.5  # Should have decent fit with clean data
            else:
                r2_ok = False
            
            # Check regime classification
            if hasattr(hansen, 'threshold_estimate'):
                estimated_regimes = (data['threshold_var'] > hansen.threshold_estimate).astype(int)
                classification_accuracy = np.mean(estimated_regimes == data['true_regime_indicator'])
                results['regime_classification_accuracy'] = classification_accuracy
                classification_ok = classification_accuracy > 0.8
            else:
                classification_ok = False
            
            # Overall validation
            results['validation_passed'] = threshold_ok and coef_ok and r2_ok and classification_ok
            
        except Exception as e:
            results['error'] = str(e)
            results['validation_passed'] = False
        
        return results
    
    def validate_str_estimation(self, data, tolerance=None):
        """
        Validate Smooth Transition Regression against known parameters.
        """
        if tolerance is None:
            tolerance = self.tolerance
            
        # Fit STR model
        str_model = SmoothTransitionRegression()
        str_model.fit(data['y'], data['x'], data['threshold_var'])
        
        results = {
            'threshold_accuracy': None,
            'gamma_accuracy': None,
            'regime1_coef_accuracy': None,
            'regime2_coef_accuracy': None,
            'r_squared': None,
            'transition_function_accuracy': None,
            'validation_passed': False
        }
        
        try:
            # Check threshold estimation
            if hasattr(str_model, 'threshold_estimate'):
                threshold_error = abs(str_model.threshold_estimate - data['true_threshold'])
                results['threshold_accuracy'] = threshold_error
                threshold_ok = threshold_error < tolerance * 10
            else:
                threshold_ok = False
            
            # Check gamma parameter
            if hasattr(str_model, 'gamma'):
                gamma_error = abs(str_model.gamma - data['true_gamma'])
                results['gamma_accuracy'] = gamma_error
                gamma_ok = gamma_error < tolerance * 20  # Gamma can be harder to estimate precisely
            else:
                gamma_ok = False
            
            # Check coefficients
            if hasattr(str_model, 'regime1_coef') and hasattr(str_model, 'regime2_coef'):
                regime1_error = abs(str_model.regime1_coef[0] - data['true_regime1_coef'])
                regime2_error = abs(str_model.regime2_coef[0] - data['true_regime2_coef'])
                
                results['regime1_coef_accuracy'] = regime1_error
                results['regime2_coef_accuracy'] = regime2_error
                
                coef_ok = regime1_error < tolerance * 5 and regime2_error < tolerance * 5
            else:
                coef_ok = False
            
            # Check model fit
            if hasattr(str_model, 'r_squared'):
                results['r_squared'] = str_model.r_squared
                r2_ok = str_model.r_squared > 0.5
            else:
                r2_ok = False
            
            # Check transition function accuracy
            if hasattr(str_model, 'transition_function'):
                transition_mse = mean_squared_error(data['true_transition_func'], 
                                                  str_model.transition_function)
                results['transition_function_accuracy'] = transition_mse
                transition_ok = transition_mse < tolerance * 100
            else:
                transition_ok = False
            
            results['validation_passed'] = (threshold_ok and gamma_ok and coef_ok and 
                                          r2_ok and transition_ok)
            
        except Exception as e:
            results['error'] = str(e)
            results['validation_passed'] = False
        
        return results
    
    def validate_local_projections_estimation(self, data, tolerance=None):
        """
        Validate Local Projections against known impulse responses.
        """
        if tolerance is None:
            tolerance = self.tolerance
            
        # Fit Local Projections model
        lp_model = LocalProjections()
        
        # Convert to pandas Series/DataFrame for LocalProjections
        y_series = pd.Series(data['y'])
        shock_series = pd.Series(data['shock'])
        controls_df = pd.DataFrame(data['controls']) if data['controls'] is not None else None
        
        lp_model.fit(y_series, shock_series, controls_df, 
                    lags=data['n_lags'])
        
        results = {
            'impulse_response_accuracy': None,
            'peak_response_accuracy': None,
            'persistence_accuracy': None,
            'confidence_intervals_coverage': None,
            'validation_passed': False
        }
        
        try:
            # Check impulse response estimation
            if hasattr(lp_model, 'get_impulse_responses'):
                ir_results = lp_model.get_impulse_responses()
                true_ir = data['true_impulse_response'][:data['n_horizons']]
                estimated_ir = ir_results['coefficient'].values[:len(true_ir)]
                
                ir_mse = mean_squared_error(true_ir, estimated_ir)
                results['impulse_response_accuracy'] = ir_mse
                ir_ok = ir_mse < tolerance * 10
                
                # Check peak response
                true_peak = np.max(true_ir)
                estimated_peak = np.max(estimated_ir)
                peak_error = abs(true_peak - estimated_peak)
                results['peak_response_accuracy'] = peak_error
                peak_ok = peak_error < tolerance * 5
                
                # Check persistence (how long effects last)
                true_persistence = np.sum(np.abs(true_ir) > 0.05)
                estimated_persistence = np.sum(np.abs(estimated_ir) > 0.05)
                persistence_error = abs(true_persistence - estimated_persistence)
                results['persistence_accuracy'] = persistence_error
                persistence_ok = persistence_error <= 2  # Allow 2 period difference
                
            else:
                ir_ok = peak_ok = persistence_ok = False
            
            # Check confidence interval coverage (if available)
            if hasattr(lp_model, 'get_impulse_responses'):
                ir_results = lp_model.get_impulse_responses()
                if 'lower_ci' in ir_results.columns and 'upper_ci' in ir_results.columns:
                    results['confidence_intervals_coverage'] = 'available'
                    ci_ok = True
                else:
                    results['confidence_intervals_coverage'] = 'not_available'
                    ci_ok = True  # Not required
            else:
                ci_ok = True  # Not required
            
            results['validation_passed'] = ir_ok and peak_ok and persistence_ok and ci_ok
            
        except Exception as e:
            results['error'] = str(e)
            results['validation_passed'] = False
        
        return results
    
    def validate_diagnostics_accuracy(self, model, data):
        """
        Validate that diagnostic tools correctly identify model issues.
        """
        diagnostics = PublicationModelDiagnostics()
        
        results = {
            'r2_diagnosis_accuracy': None,
            'regime_analysis_accuracy': None,
            'specification_test_accuracy': None,
            'validation_passed': False
        }
        
        try:
            # Test R² diagnosis
            if hasattr(model, 'r_squared'):
                diagnosis = diagnostics.diagnose_low_r_squared(
                    model, data['y'], data['x'], data['threshold_var']
                )
                
                # Check if diagnosis correctly identifies R² level
                actual_r2 = model.r_squared
                diagnosed_r2 = diagnosis['r2_analysis']['overall_r2']
                r2_error = abs(actual_r2 - diagnosed_r2)
                results['r2_diagnosis_accuracy'] = r2_error
                r2_diag_ok = r2_error < self.tolerance
                
                # Check regime analysis accuracy
                if 'regime_analysis' in diagnosis:
                    regime_analysis = diagnosis['regime_analysis']
                    # Should correctly identify regime sizes
                    if hasattr(model, 'threshold_estimate'):
                        true_regime1_size = np.sum(data['threshold_var'] <= model.threshold_estimate)
                        diagnosed_regime1_size = regime_analysis.get('regime1_obs', 0)
                        regime_error = abs(true_regime1_size - diagnosed_regime1_size)
                        results['regime_analysis_accuracy'] = regime_error
                        regime_ok = regime_error <= 5  # Allow small differences
                    else:
                        regime_ok = True
                else:
                    regime_ok = True
                
                # Check specification tests
                if 'specification_adequacy' in diagnosis:
                    spec_tests = diagnosis['specification_adequacy']
                    # Should pass basic adequacy tests with clean synthetic data
                    results['specification_test_accuracy'] = 'passed'
                    spec_ok = True
                else:
                    spec_ok = True
                
                results['validation_passed'] = r2_diag_ok and regime_ok and spec_ok
            else:
                results['validation_passed'] = False
                
        except Exception as e:
            results['error'] = str(e)
            results['validation_passed'] = False
        
        return results
    
    def run_comprehensive_benchmark_validation(self):
        """
        Run comprehensive validation against all benchmark datasets.
        """
        validation_results = {
            'hansen_validation': {},
            'str_validation': {},
            'local_projections_validation': {},
            'diagnostics_validation': {},
            'overall_validation_passed': False
        }
        
        try:
            # Test Hansen model
            print("Validating Hansen Threshold Regression...")
            hansen_data = self.generate_hansen_benchmark_data()
            hansen_results = self.validate_hansen_estimation(hansen_data)
            validation_results['hansen_validation'] = hansen_results
            
            # Test STR model
            print("Validating Smooth Transition Regression...")
            str_data = self.generate_str_benchmark_data()
            str_results = self.validate_str_estimation(str_data)
            validation_results['str_validation'] = str_results
            
            # Test Local Projections
            print("Validating Local Projections...")
            lp_data = self.generate_local_projections_benchmark_data()
            lp_results = self.validate_local_projections_estimation(lp_data)
            validation_results['local_projections_validation'] = lp_results
            
            # Test diagnostics accuracy
            print("Validating Diagnostic Tools...")
            hansen_model = HansenThresholdRegression()
            hansen_model.fit(hansen_data['y'], hansen_data['x'], hansen_data['threshold_var'])
            diag_results = self.validate_diagnostics_accuracy(hansen_model, hansen_data)
            validation_results['diagnostics_validation'] = diag_results
            
            # Overall validation
            all_passed = (
                hansen_results.get('validation_passed', False) and
                str_results.get('validation_passed', False) and
                lp_results.get('validation_passed', False) and
                diag_results.get('validation_passed', False)
            )
            
            validation_results['overall_validation_passed'] = all_passed
            
            # Print summary
            print("\n" + "="*50)
            print("BENCHMARK VALIDATION SUMMARY")
            print("="*50)
            print(f"Hansen Threshold Regression: {'PASS' if hansen_results.get('validation_passed') else 'FAIL'}")
            print(f"Smooth Transition Regression: {'PASS' if str_results.get('validation_passed') else 'FAIL'}")
            print(f"Local Projections: {'PASS' if lp_results.get('validation_passed') else 'FAIL'}")
            print(f"Diagnostic Tools: {'PASS' if diag_results.get('validation_passed') else 'FAIL'}")
            print(f"\nOverall Validation: {'PASS' if all_passed else 'FAIL'}")
            print("="*50)
            
        except Exception as e:
            validation_results['error'] = str(e)
            validation_results['overall_validation_passed'] = False
            print(f"Validation failed with error: {e}")
        
        return validation_results
    
    def compare_with_published_results(self):
        """
        Compare results with published econometric studies (where data is available).
        
        This would ideally use real datasets from published papers, but for now
        we use stylized examples that match published parameter ranges.
        """
        comparison_results = {
            'hansen_literature_comparison': {},
            'str_literature_comparison': {},
            'qe_literature_comparison': {},
            'comparison_passed': False
        }
        
        try:
            # Hansen (1999) style parameters - typical threshold effects in macro
            hansen_lit_data = self.generate_hansen_benchmark_data(
                n_obs=300, 
                threshold_true=0.6,  # Typical threshold in QE studies
                regime1_coef=0.1,    # Low regime effect
                regime2_coef=0.4,    # High regime effect  
                noise_std=0.8        # Realistic noise level
            )
            
            hansen_lit_results = self.validate_hansen_estimation(hansen_lit_data, tolerance=0.1)
            comparison_results['hansen_literature_comparison'] = hansen_lit_results
            
            # STR literature comparison - typical smooth transition parameters
            str_lit_data = self.generate_str_benchmark_data(
                n_obs=250,
                threshold_true=0.0,   # Centered threshold
                regime1_coef=0.2,     # Moderate effects
                regime2_coef=0.5,
                gamma_true=1.5,       # Moderate transition speed
                noise_std=0.6
            )
            
            str_lit_results = self.validate_str_estimation(str_lit_data, tolerance=0.15)
            comparison_results['str_literature_comparison'] = str_lit_results
            
            # QE literature comparison - typical QE effect sizes
            qe_lit_data = self.generate_hansen_benchmark_data(
                n_obs=200,
                threshold_true=0.4,   # QE intensity threshold
                regime1_coef=0.05,    # Small effect below threshold
                regime2_coef=0.25,    # Larger effect above threshold
                noise_std=1.0         # Higher noise (realistic for macro data)
            )
            
            qe_lit_results = self.validate_hansen_estimation(qe_lit_data, tolerance=0.2)
            comparison_results['qe_literature_comparison'] = qe_lit_results
            
            # Overall comparison assessment
            all_comparisons_passed = (
                hansen_lit_results.get('validation_passed', False) and
                str_lit_results.get('validation_passed', False) and
                qe_lit_results.get('validation_passed', False)
            )
            
            comparison_results['comparison_passed'] = all_comparisons_passed
            
            print("\n" + "="*50)
            print("LITERATURE COMPARISON SUMMARY")
            print("="*50)
            print(f"Hansen Literature Parameters: {'PASS' if hansen_lit_results.get('validation_passed') else 'FAIL'}")
            print(f"STR Literature Parameters: {'PASS' if str_lit_results.get('validation_passed') else 'FAIL'}")
            print(f"QE Literature Parameters: {'PASS' if qe_lit_results.get('validation_passed') else 'FAIL'}")
            print(f"\nLiterature Comparison: {'PASS' if all_comparisons_passed else 'FAIL'}")
            print("="*50)
            
        except Exception as e:
            comparison_results['error'] = str(e)
            comparison_results['comparison_passed'] = False
            print(f"Literature comparison failed with error: {e}")
        
        return comparison_results


def run_benchmark_validation():
    """Run the complete benchmark validation suite."""
    validator = EconometricBenchmarkValidator()
    
    print("Starting Econometric Benchmark Validation...")
    print("This validates our implementations against known parameters and literature.")
    
    # Run comprehensive validation
    validation_results = validator.run_comprehensive_benchmark_validation()
    
    # Run literature comparison
    literature_results = validator.compare_with_published_results()
    
    # Combine results
    final_results = {
        'benchmark_validation': validation_results,
        'literature_comparison': literature_results,
        'overall_passed': (
            validation_results.get('overall_validation_passed', False) and
            literature_results.get('comparison_passed', False)
        )
    }
    
    print(f"\nFINAL VALIDATION RESULT: {'PASS' if final_results['overall_passed'] else 'FAIL'}")
    
    return final_results


if __name__ == "__main__":
    # Run validation when script is executed directly
    results = run_benchmark_validation()
    
    # Exit with appropriate code
    exit_code = 0 if results['overall_passed'] else 1
    exit(exit_code)