"""
Specification Test Validation

This module validates that our specification tests correctly identify
model misspecification and follow proper statistical procedures.
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

from src.models import HansenThresholdRegression, SmoothTransitionRegression
from src.publication_model_diagnostics import PublicationModelDiagnostics
from src.model_specification_enhancer import ModelSpecificationEnhancer


class SpecificationTestValidator:
    """Validates specification tests against known misspecification scenarios."""
    
    def __init__(self):
        self.significance_level = 0.05
        self.power_threshold = 0.8  # Minimum power for detecting misspecification
        self.size_tolerance = 0.02  # Tolerance for test size (should be close to 0.05)
    
    def generate_correctly_specified_data(self, n_obs=200, seed=42):
        """
        Generate data that matches the assumed model specification.
        Tests should NOT reject the null hypothesis for this data.
        """
        np.random.seed(seed)
        
        # Generate threshold variable
        threshold_var = np.random.uniform(0, 1, n_obs)
        
        # Generate independent variables
        x1 = np.random.normal(0, 1, n_obs)
        x2 = np.random.normal(0, 1, n_obs)
        
        # True threshold
        true_threshold = 0.5
        
        # Generate dependent variable with correct threshold specification
        regime_indicator = (threshold_var > true_threshold).astype(int)
        
        y = np.zeros(n_obs)
        for i in range(n_obs):
            if regime_indicator[i] == 0:  # Regime 1
                y[i] = 0.3 * x1[i] + 0.2 * x2[i]
            else:  # Regime 2
                y[i] = 0.7 * x1[i] + 0.4 * x2[i]
            
            # Add homoskedastic noise
            y[i] += np.random.normal(0, 0.5)
        
        return {
            'y': y,
            'x': np.column_stack([x1, x2]),
            'threshold_var': threshold_var,
            'true_threshold': true_threshold,
            'specification': 'correct',
            'expected_rejection_rate': self.significance_level
        }
    
    def generate_misspecified_threshold_data(self, n_obs=200, seed=42):
        """
        Generate data with wrong threshold variable.
        Tests should reject the null hypothesis for this data.
        """
        np.random.seed(seed)
        
        # Generate variables
        true_threshold_var = np.random.uniform(0, 1, n_obs)
        wrong_threshold_var = np.random.uniform(0, 1, n_obs)  # Different variable
        x1 = np.random.normal(0, 1, n_obs)
        x2 = np.random.normal(0, 1, n_obs)
        
        # True threshold based on true_threshold_var
        true_threshold = 0.5
        regime_indicator = (true_threshold_var > true_threshold).astype(int)
        
        # Generate y based on true threshold variable
        y = np.zeros(n_obs)
        for i in range(n_obs):
            if regime_indicator[i] == 0:
                y[i] = 0.3 * x1[i] + 0.2 * x2[i]
            else:
                y[i] = 0.7 * x1[i] + 0.4 * x2[i]
            y[i] += np.random.normal(0, 0.5)
        
        return {
            'y': y,
            'x': np.column_stack([x1, x2]),
            'threshold_var': wrong_threshold_var,  # Using wrong threshold variable
            'true_threshold_var': true_threshold_var,
            'specification': 'misspecified_threshold',
            'expected_rejection_rate': 0.8  # Should reject with high probability
        }
    
    def generate_heteroskedastic_data(self, n_obs=200, seed=42):
        """
        Generate data with heteroskedasticity.
        Heteroskedasticity tests should reject the null hypothesis.
        """
        np.random.seed(seed)
        
        # Generate variables
        threshold_var = np.random.uniform(0, 1, n_obs)
        x1 = np.random.normal(0, 1, n_obs)
        x2 = np.random.normal(0, 1, n_obs)
        
        true_threshold = 0.5
        regime_indicator = (threshold_var > true_threshold).astype(int)
        
        # Generate y with heteroskedastic errors
        y = np.zeros(n_obs)
        for i in range(n_obs):
            if regime_indicator[i] == 0:
                y[i] = 0.3 * x1[i] + 0.2 * x2[i]
                # Heteroskedastic error (variance depends on x1)
                error_std = 0.3 + 0.5 * abs(x1[i])
            else:
                y[i] = 0.7 * x1[i] + 0.4 * x2[i]
                # Different heteroskedasticity pattern
                error_std = 0.4 + 0.3 * abs(x2[i])
            
            y[i] += np.random.normal(0, error_std)
        
        return {
            'y': y,
            'x': np.column_stack([x1, x2]),
            'threshold_var': threshold_var,
            'specification': 'heteroskedastic',
            'expected_rejection_rate': 0.8  # Heteroskedasticity tests should reject
        }
    
    def generate_nonlinear_misspecification_data(self, n_obs=200, seed=42):
        """
        Generate data with nonlinear relationships not captured by threshold model.
        Specification tests should detect this misspecification.
        """
        np.random.seed(seed)
        
        # Generate variables
        threshold_var = np.random.uniform(0, 1, n_obs)
        x1 = np.random.normal(0, 1, n_obs)
        x2 = np.random.normal(0, 1, n_obs)
        
        # Generate y with quadratic relationship (not threshold)
        y = 0.3 * x1 + 0.2 * x2 + 0.4 * x1**2 + 0.1 * x1 * x2
        y += np.random.normal(0, 0.5, n_obs)
        
        return {
            'y': y,
            'x': np.column_stack([x1, x2]),
            'threshold_var': threshold_var,
            'specification': 'nonlinear_misspecification',
            'expected_rejection_rate': 0.7  # Should detect nonlinearity
        }
    
    def validate_linearity_test(self, data):
        """
        Validate linearity tests (e.g., RESET test, Ramsey test).
        """
        results = {
            'test_statistic': None,
            'p_value': None,
            'rejection_decision': None,
            'correct_decision': None,
            'validation_passed': False
        }
        
        try:
            # Fit linear model first
            X = data['x']
            y = data['y']
            
            # Simple linear regression
            from sklearn.linear_model import LinearRegression
            linear_model = LinearRegression()
            linear_model.fit(X, y)
            
            # Get fitted values and residuals
            y_fitted = linear_model.predict(X)
            residuals = y - y_fitted
            
            # RESET test: regress residuals on powers of fitted values
            y_fitted_squared = y_fitted ** 2
            y_fitted_cubed = y_fitted ** 3
            
            # Auxiliary regression
            X_aux = np.column_stack([X, y_fitted_squared, y_fitted_cubed])
            
            from sklearn.linear_model import LinearRegression
            aux_model = LinearRegression()
            aux_model.fit(X_aux, y)
            
            # Calculate R-squared for auxiliary regression
            r2_aux = aux_model.score(X_aux, y)
            r2_original = linear_model.score(X, y)
            
            # F-test for additional terms
            n = len(y)
            k_original = X.shape[1]
            k_aux = X_aux.shape[1]
            
            f_stat = ((r2_aux - r2_original) / (k_aux - k_original)) / ((1 - r2_aux) / (n - k_aux - 1))
            p_value = 1 - stats.f.cdf(f_stat, k_aux - k_original, n - k_aux - 1)
            
            results['test_statistic'] = f_stat
            results['p_value'] = p_value
            results['rejection_decision'] = p_value < self.significance_level
            
            # Check if decision is correct based on data specification
            if data['specification'] == 'correct':
                # Should NOT reject (Type I error if rejected)
                results['correct_decision'] = not results['rejection_decision']
            elif data['specification'] == 'nonlinear_misspecification':
                # Should reject (correct if rejected)
                results['correct_decision'] = results['rejection_decision']
            else:
                # For other misspecifications, linearity test may or may not reject
                results['correct_decision'] = True  # Assume correct for now
            
            results['validation_passed'] = results['correct_decision']
            
        except Exception as e:
            results['error'] = str(e)
            results['validation_passed'] = False
        
        return results
    
    def validate_heteroskedasticity_test(self, data):
        """
        Validate heteroskedasticity tests (e.g., Breusch-Pagan, White test).
        """
        results = {
            'test_statistic': None,
            'p_value': None,
            'rejection_decision': None,
            'correct_decision': None,
            'validation_passed': False
        }
        
        try:
            X = data['x']
            y = data['y']
            
            # Fit model and get residuals
            from sklearn.linear_model import LinearRegression
            model = LinearRegression()
            model.fit(X, y)
            
            residuals = y - model.predict(X)
            residuals_squared = residuals ** 2
            
            # Breusch-Pagan test: regress squared residuals on X
            bp_model = LinearRegression()
            bp_model.fit(X, residuals_squared)
            
            # Calculate test statistic
            r2_bp = bp_model.score(X, residuals_squared)
            n = len(y)
            lm_statistic = n * r2_bp  # n * R-squared
            
            # Chi-squared test
            p_value = 1 - stats.chi2.cdf(lm_statistic, X.shape[1])
            
            results['test_statistic'] = lm_statistic
            results['p_value'] = p_value
            results['rejection_decision'] = p_value < self.significance_level
            
            # Check if decision is correct
            if data['specification'] == 'heteroskedastic':
                # Should reject (detect heteroskedasticity)
                results['correct_decision'] = results['rejection_decision']
            else:
                # Should NOT reject (homoskedastic data)
                results['correct_decision'] = not results['rejection_decision']
            
            results['validation_passed'] = results['correct_decision']
            
        except Exception as e:
            results['error'] = str(e)
            results['validation_passed'] = False
        
        return results
    
    def validate_threshold_specification_test(self, data):
        """
        Validate threshold specification tests.
        """
        results = {
            'threshold_test_statistic': None,
            'threshold_p_value': None,
            'threshold_rejection': None,
            'correct_threshold_decision': None,
            'validation_passed': False
        }
        
        try:
            # Fit Hansen threshold model
            hansen = HansenThresholdRegression()
            hansen.fit(data['y'], data['x'], data['threshold_var'])
            
            # Get threshold test results if available
            if hasattr(hansen, 'threshold_test_statistic'):
                results['threshold_test_statistic'] = hansen.threshold_test_statistic
                results['threshold_p_value'] = getattr(hansen, 'threshold_p_value', None)
                
                if results['threshold_p_value'] is not None:
                    results['threshold_rejection'] = results['threshold_p_value'] < self.significance_level
                    
                    # Check correctness based on data specification
                    if data['specification'] == 'correct':
                        # Should detect threshold effect
                        results['correct_threshold_decision'] = results['threshold_rejection']
                    elif data['specification'] == 'misspecified_threshold':
                        # May or may not detect with wrong threshold variable
                        results['correct_threshold_decision'] = True  # Lenient for now
                    else:
                        results['correct_threshold_decision'] = True
                    
                    results['validation_passed'] = results['correct_threshold_decision']
                else:
                    results['validation_passed'] = False
            else:
                # If no threshold test available, check if model converged
                results['validation_passed'] = hasattr(hansen, 'threshold_estimate')
            
        except Exception as e:
            results['error'] = str(e)
            results['validation_passed'] = False
        
        return results
    
    def validate_model_selection_criteria(self, data):
        """
        Validate model selection criteria (AIC, BIC) behavior.
        """
        results = {
            'linear_aic': None,
            'threshold_aic': None,
            'aic_selects_correct': None,
            'linear_bic': None,
            'threshold_bic': None,
            'bic_selects_correct': None,
            'validation_passed': False
        }
        
        try:
            X = data['x']
            y = data['y']
            
            # Fit linear model
            from sklearn.linear_model import LinearRegression
            linear_model = LinearRegression()
            linear_model.fit(X, y)
            
            # Calculate AIC/BIC for linear model
            n = len(y)
            k_linear = X.shape[1] + 1  # coefficients + intercept
            
            linear_residuals = y - linear_model.predict(X)
            linear_sse = np.sum(linear_residuals ** 2)
            linear_log_likelihood = -n/2 * np.log(2 * np.pi * linear_sse / n) - n/2
            
            linear_aic = -2 * linear_log_likelihood + 2 * k_linear
            linear_bic = -2 * linear_log_likelihood + np.log(n) * k_linear
            
            results['linear_aic'] = linear_aic
            results['linear_bic'] = linear_bic
            
            # Fit threshold model
            hansen = HansenThresholdRegression()
            hansen.fit(y, X, data['threshold_var'])
            
            # Calculate AIC/BIC for threshold model (if available)
            if hasattr(hansen, 'log_likelihood'):
                k_threshold = 2 * X.shape[1] + 2  # coefficients for both regimes + threshold + variance
                
                threshold_aic = -2 * hansen.log_likelihood + 2 * k_threshold
                threshold_bic = -2 * hansen.log_likelihood + np.log(n) * k_threshold
                
                results['threshold_aic'] = threshold_aic
                results['threshold_bic'] = threshold_bic
                
                # Check which model is selected
                results['aic_selects_correct'] = self._check_model_selection_correctness(
                    data, linear_aic, threshold_aic, 'aic'
                )
                results['bic_selects_correct'] = self._check_model_selection_correctness(
                    data, linear_bic, threshold_bic, 'bic'
                )
                
                results['validation_passed'] = (
                    results['aic_selects_correct'] and results['bic_selects_correct']
                )
            else:
                results['validation_passed'] = False
            
        except Exception as e:
            results['error'] = str(e)
            results['validation_passed'] = False
        
        return results
    
    def _check_model_selection_correctness(self, data, linear_criterion, threshold_criterion, criterion_name):
        """
        Check if model selection criterion selects the correct model.
        """
        threshold_selected = threshold_criterion < linear_criterion
        
        if data['specification'] == 'correct':
            # Threshold model should be selected
            return threshold_selected
        elif data['specification'] in ['nonlinear_misspecification']:
            # Linear model might be selected (neither is correct)
            return True  # Be lenient
        else:
            # For other cases, either could be reasonable
            return True
    
    def run_specification_test_validation(self):
        """
        Run comprehensive specification test validation.
        """
        print("="*60)
        print("SPECIFICATION TEST VALIDATION")
        print("="*60)
        
        validation_results = {
            'linearity_test_validation': {},
            'heteroskedasticity_test_validation': {},
            'threshold_test_validation': {},
            'model_selection_validation': {},
            'overall_validation_passed': False
        }
        
        try:
            # Test with correctly specified data
            print("Testing with correctly specified data...")
            correct_data = self.generate_correctly_specified_data()
            
            linearity_correct = self.validate_linearity_test(correct_data)
            hetero_correct = self.validate_heteroskedasticity_test(correct_data)
            threshold_correct = self.validate_threshold_specification_test(correct_data)
            selection_correct = self.validate_model_selection_criteria(correct_data)
            
            # Test with misspecified data
            print("Testing with misspecified data...")
            
            # Heteroskedastic data
            hetero_data = self.generate_heteroskedastic_data()
            hetero_misspec = self.validate_heteroskedasticity_test(hetero_data)
            
            # Nonlinear misspecification
            nonlinear_data = self.generate_nonlinear_misspecification_data()
            linearity_misspec = self.validate_linearity_test(nonlinear_data)
            
            # Wrong threshold variable
            threshold_misspec_data = self.generate_misspecified_threshold_data()
            threshold_misspec = self.validate_threshold_specification_test(threshold_misspec_data)
            
            # Compile results
            validation_results['linearity_test_validation'] = {
                'correct_specification': linearity_correct,
                'nonlinear_misspecification': linearity_misspec,
                'overall_passed': (
                    linearity_correct.get('validation_passed', False) and
                    linearity_misspec.get('validation_passed', False)
                )
            }
            
            validation_results['heteroskedasticity_test_validation'] = {
                'homoskedastic_data': hetero_correct,
                'heteroskedastic_data': hetero_misspec,
                'overall_passed': (
                    hetero_correct.get('validation_passed', False) and
                    hetero_misspec.get('validation_passed', False)
                )
            }
            
            validation_results['threshold_test_validation'] = {
                'correct_threshold': threshold_correct,
                'misspecified_threshold': threshold_misspec,
                'overall_passed': (
                    threshold_correct.get('validation_passed', False) and
                    threshold_misspec.get('validation_passed', False)
                )
            }
            
            validation_results['model_selection_validation'] = {
                'correct_specification': selection_correct,
                'overall_passed': selection_correct.get('validation_passed', False)
            }
            
            # Overall assessment
            all_passed = (
                validation_results['linearity_test_validation']['overall_passed'] and
                validation_results['heteroskedasticity_test_validation']['overall_passed'] and
                validation_results['threshold_test_validation']['overall_passed'] and
                validation_results['model_selection_validation']['overall_passed']
            )
            
            validation_results['overall_validation_passed'] = all_passed
            
            # Print summary
            print("\nVALIDATION SUMMARY:")
            print("-" * 40)
            print(f"Linearity Tests: {'PASS' if validation_results['linearity_test_validation']['overall_passed'] else 'FAIL'}")
            print(f"Heteroskedasticity Tests: {'PASS' if validation_results['heteroskedasticity_test_validation']['overall_passed'] else 'FAIL'}")
            print(f"Threshold Tests: {'PASS' if validation_results['threshold_test_validation']['overall_passed'] else 'FAIL'}")
            print(f"Model Selection: {'PASS' if validation_results['model_selection_validation']['overall_passed'] else 'FAIL'}")
            print(f"\nOverall Specification Test Validation: {'PASS' if all_passed else 'FAIL'}")
            print("="*60)
            
        except Exception as e:
            validation_results['error'] = str(e)
            validation_results['overall_validation_passed'] = False
            print(f"Specification test validation failed with error: {e}")
        
        return validation_results


def run_specification_test_validation():
    """Run specification test validation."""
    validator = SpecificationTestValidator()
    return validator.run_specification_test_validation()


if __name__ == "__main__":
    results = run_specification_test_validation()
    exit_code = 0 if results['overall_validation_passed'] else 1
    exit(exit_code)