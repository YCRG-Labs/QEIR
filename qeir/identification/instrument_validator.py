"""
Instrument Validation Module

This module provides tools for validating instrumental variables in 2SLS estimation,
including weak instrument tests and first-stage diagnostics.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Tuple
import statsmodels.api as sm
from scipy import stats
import warnings


class InstrumentValidator:
    """
    Validator for instrumental variables in 2SLS estimation.
    
    Provides methods for:
    - First-stage F-statistic calculation
    - Weak instrument testing
    - Instrument relevance diagnostics
    - Overidentification tests
    """
    
    def __init__(self, weak_instrument_threshold: float = 10.0):
        """
        Initialize instrument validator.
        
        Args:
            weak_instrument_threshold: Minimum F-statistic for strong instruments
                                      (Stock-Yogo critical value, default 10.0)
        """
        self.weak_instrument_threshold = weak_instrument_threshold
        self.first_stage_results = None
        
    def compute_first_stage_fstat(
        self,
        endogenous: np.ndarray,
        instruments: np.ndarray,
        controls: Optional[np.ndarray] = None,
        return_details: bool = False
    ) -> float:
        """
        Compute first-stage F-statistic for instrument strength.
        
        Tests whether instruments are relevant for the endogenous variable
        by regressing endogenous on instruments and controls.
        
        Args:
            endogenous: Endogenous variable (n x 1)
            instruments: Instrumental variables (n x k)
            controls: Optional control variables (n x p)
            return_details: If True, return full first-stage results
            
        Returns:
            First-stage F-statistic (or dict if return_details=True)
        """
        # Ensure arrays are 2D
        if endogenous.ndim == 1:
            endogenous = endogenous.reshape(-1, 1)
        if instruments.ndim == 1:
            instruments = instruments.reshape(-1, 1)
        
        # Build regressor matrix
        if controls is not None:
            if controls.ndim == 1:
                controls = controls.reshape(-1, 1)
            X = np.column_stack([instruments, controls])
            n_instruments = instruments.shape[1]
        else:
            X = instruments
            n_instruments = instruments.shape[1]
        
        # Add constant
        X = sm.add_constant(X)
        
        # Run first-stage regression
        model = sm.OLS(endogenous, X)
        results = model.fit()
        
        # Compute F-statistic for instruments
        # Test that all instrument coefficients are jointly zero
        # Instrument coefficients are positions 1 to n_instruments+1 (after constant)
        instrument_indices = list(range(1, n_instruments + 1))
        
        # F-test for joint significance of instruments
        f_test = results.f_test(np.eye(len(results.params))[instrument_indices])
        
        # Handle different return types from f_test
        if hasattr(f_test.fvalue, '__getitem__'):
            f_stat = f_test.fvalue[0][0] if isinstance(f_test.fvalue[0], (list, np.ndarray)) else f_test.fvalue[0]
        else:
            f_stat = float(f_test.fvalue)
        
        p_value = f_test.pvalue
        
        # Store results
        self.first_stage_results = {
            'f_statistic': f_stat,
            'p_value': p_value,
            'r_squared': results.rsquared,
            'adj_r_squared': results.rsquared_adj,
            'n_observations': len(endogenous),
            'n_instruments': n_instruments,
            'coefficients': results.params,
            'std_errors': results.bse,
            'full_results': results
        }
        
        if return_details:
            return self.first_stage_results
        else:
            return f_stat
    
    def weak_instrument_test(
        self,
        endogenous: np.ndarray,
        instruments: np.ndarray,
        controls: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """
        Test for weak instruments using Stock-Yogo critical values.
        
        Args:
            endogenous: Endogenous variable
            instruments: Instrumental variables
            controls: Optional control variables
            
        Returns:
            Dictionary with test results and diagnostics
        """
        # Compute first-stage F-statistic
        first_stage = self.compute_first_stage_fstat(
            endogenous, instruments, controls, return_details=True
        )
        
        f_stat = first_stage['f_statistic']
        
        # Test against threshold
        is_strong = f_stat > self.weak_instrument_threshold
        
        # Compute partial R-squared (R² from instruments only)
        # This requires running regression with and without instruments
        if controls is not None:
            if controls.ndim == 1:
                controls = controls.reshape(-1, 1)
            X_controls = sm.add_constant(controls)
            model_controls = sm.OLS(endogenous, X_controls)
            results_controls = model_controls.fit()
            r2_controls = results_controls.rsquared
        else:
            r2_controls = 0.0
        
        r2_full = first_stage['r_squared']
        partial_r2 = r2_full - r2_controls
        
        test_results = {
            'f_statistic': f_stat,
            'p_value': first_stage['p_value'],
            'threshold': self.weak_instrument_threshold,
            'is_strong_instrument': is_strong,
            'r_squared': r2_full,
            'partial_r_squared': partial_r2,
            'n_instruments': first_stage['n_instruments'],
            'n_observations': first_stage['n_observations'],
            'interpretation': self._interpret_weak_test(f_stat, is_strong)
        }
        
        return test_results
    
    def _interpret_weak_test(self, f_stat: float, is_strong: bool) -> str:
        """
        Provide interpretation of weak instrument test.
        
        Args:
            f_stat: First-stage F-statistic
            is_strong: Whether instrument passes strength test
            
        Returns:
            Interpretation string
        """
        if is_strong:
            if f_stat > 20:
                return f"Strong instruments (F={f_stat:.2f} >> 10). Instruments are highly relevant."
            else:
                return f"Adequate instruments (F={f_stat:.2f} > 10). Instruments pass strength test."
        else:
            return (
                f"Weak instruments (F={f_stat:.2f} < 10). "
                f"IV estimates may be biased. Consider alternative instruments."
            )
    
    def validate_instruments_for_regression(
        self,
        endogenous: np.ndarray,
        instruments: np.ndarray,
        controls: Optional[np.ndarray] = None,
        raise_on_weak: bool = False
    ) -> Dict[str, Any]:
        """
        Comprehensive validation of instruments for IV regression.
        
        Args:
            endogenous: Endogenous variable
            instruments: Instrumental variables
            controls: Optional control variables
            raise_on_weak: If True, raise error for weak instruments
            
        Returns:
            Dictionary with validation results
        """
        # Run weak instrument test
        weak_test = self.weak_instrument_test(endogenous, instruments, controls)
        
        # Check for weak instruments
        if not weak_test['is_strong_instrument']:
            message = (
                f"Weak instruments detected: F-statistic = {weak_test['f_statistic']:.2f} "
                f"< {self.weak_instrument_threshold}"
            )
            if raise_on_weak:
                raise ValueError(message)
            else:
                warnings.warn(message)
        
        # Additional diagnostics
        validation = {
            'weak_instrument_test': weak_test,
            'valid_for_iv': weak_test['is_strong_instrument'],
            'warnings': []
        }
        
        # Check for sufficient observations
        n_obs = weak_test['n_observations']
        n_params = weak_test['n_instruments'] + (controls.shape[1] if controls is not None else 0)
        
        if n_obs < 3 * n_params:
            validation['warnings'].append(
                f"Limited observations: {n_obs} observations for {n_params} parameters"
            )
        
        # Check partial R²
        if weak_test['partial_r_squared'] < 0.01:
            validation['warnings'].append(
                f"Low partial R²: {weak_test['partial_r_squared']:.4f}. "
                f"Instruments explain little variation in endogenous variable."
            )
        
        return validation
    
    def compute_fstat_for_all_instruments(
        self,
        endogenous_vars: Dict[str, np.ndarray],
        instruments: Dict[str, np.ndarray],
        controls: Optional[np.ndarray] = None
    ) -> pd.DataFrame:
        """
        Compute first-stage F-statistics for multiple endogenous variables and instruments.
        
        Args:
            endogenous_vars: Dictionary mapping variable names to arrays
            instruments: Dictionary mapping instrument names to arrays
            controls: Optional control variables (same for all)
            
        Returns:
            DataFrame with F-statistics for each endogenous-instrument pair
        """
        results = []
        
        for endog_name, endog_var in endogenous_vars.items():
            for inst_name, inst_var in instruments.items():
                try:
                    f_stat = self.compute_first_stage_fstat(
                        endog_var, inst_var, controls
                    )
                    
                    results.append({
                        'endogenous': endog_name,
                        'instrument': inst_name,
                        'f_statistic': f_stat,
                        'is_strong': f_stat > self.weak_instrument_threshold
                    })
                except Exception as e:
                    warnings.warn(
                        f"Error computing F-stat for {endog_name} ~ {inst_name}: {e}"
                    )
                    results.append({
                        'endogenous': endog_name,
                        'instrument': inst_name,
                        'f_statistic': np.nan,
                        'is_strong': False
                    })
        
        return pd.DataFrame(results)
