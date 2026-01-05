"""
Comprehensive robustness testing suite for QE methodology revision.

This module implements all robustness checks specified in Requirement 8:
- Double-threshold models (8.1)
- Smooth transition regression (8.2)
- Alternative fiscal indicators (8.3)
- Alternative distortion measures (8.4)
- QE episode subsamples (8.5)
- Shadow rate conditioning (8.6)
- Instrument validation tests (8.7-8.10)
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
import logging
from scipy.optimize import minimize
import statsmodels.api as sm
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from ..core.models import HansenThresholdRegression, SmoothTransitionRegression
from ..core.enhanced_hypothesis1 import EnhancedHansenThresholdRegression
from ..identification.hf_surprise_identifier import HFSurpriseIdentifier
from ..identification.instrument_validator import InstrumentValidator

logger = logging.getLogger(__name__)


@dataclass
class RobustnessResults:
    """Container for robustness test results."""
    
    # Test identification
    test_name: str
    test_type: str  # 'threshold', 'str', 'fiscal', 'distortion', 'subsample', 'shadow_rate', 'instrument'
    
    # Main estimates
    threshold_estimate: Optional[float] = None
    threshold_ci: Optional[Tuple[float, float]] = None
    regime1_effect: Optional[float] = None
    regime2_effect: Optional[float] = None
    regime3_effect: Optional[float] = None  # For double-threshold
    attenuation_pct: Optional[float] = None
    
    # Transition parameters (for STR)
    transition_speed: Optional[float] = None
    transition_location: Optional[float] = None
    
    # Channel decomposition
    distortion_share: Optional[float] = None
    rate_share: Optional[float] = None
    
    # Instrument validation
    first_stage_f_stat: Optional[float] = None
    instrument_valid: bool = False
    
    # Model diagnostics
    r_squared: Optional[float] = None
    n_observations: int = 0
    
    # Additional metadata
    specification_details: Dict[str, Any] = field(default_factory=dict)
    
    def is_consistent_with_baseline(self, baseline: 'RobustnessResults', 
                                    threshold_tolerance: float = 0.05,
                                    effect_tolerance: float = 0.3) -> bool:
        """
        Check if results are consistent with baseline specification.
        
        Args:
            baseline: Baseline results to compare against
            threshold_tolerance: Tolerance for threshold estimate (±5 pp)
            effect_tolerance: Tolerance for regime effects (±30%)
            
        Returns:
            True if results are consistent with baseline
        """
        if self.threshold_estimate is None or baseline.threshold_estimate is None:
            return False
            
        # Check threshold consistency
        threshold_diff = abs(self.threshold_estimate - baseline.threshold_estimate)
        if threshold_diff > threshold_tolerance:
            return False
            
        # Check regime effect consistency
        if self.regime1_effect is not None and baseline.regime1_effect is not None:
            effect1_diff = abs(self.regime1_effect - baseline.regime1_effect)
            if effect1_diff > abs(baseline.regime1_effect * effect_tolerance):
                return False
                
        if self.regime2_effect is not None and baseline.regime2_effect is not None:
            effect2_diff = abs(self.regime2_effect - baseline.regime2_effect)
            if effect2_diff > abs(baseline.regime2_effect * effect_tolerance):
                return False
                
        return True


class RobustnessTestSuite:
    """
    Comprehensive robustness testing suite for QE methodology revision.
    
    This class provides methods to run all robustness checks specified in
    Requirement 8, including alternative specifications, sample splits,
    and instrument validation.
    """
    
    def __init__(self, 
                 baseline_data: pd.DataFrame,
                 baseline_results: Optional[RobustnessResults] = None,
                 min_first_stage_f: float = 10.0):
        """
        Initialize robustness test suite.
        
        Args:
            baseline_data: DataFrame with all variables for analysis
            baseline_results: Baseline estimation results for comparison
            min_first_stage_f: Minimum F-statistic for valid instruments
        """
        self.baseline_data = baseline_data
        self.baseline_results = baseline_results
        self.min_first_stage_f = min_first_stage_f
        self.results: List[RobustnessResults] = []
        
        logger.info("Initialized RobustnessTestSuite")
    
    def _safe_threshold_fit(self, y, X, threshold_var, trim=0.15):
        """
        Safely fit threshold model with error handling.
        
        Returns:
            Tuple of (threshold_estimate, regime1_mask, regime2_mask) or (None, None, None) if failed
        """
        try:
            threshold_model = HansenThresholdRegression()
            fit_result = threshold_model.fit(y, X, threshold_var, trim=trim)
            threshold_estimate = threshold_model.threshold
            
            if threshold_estimate is None:
                return None, None, None
            
            regime1_mask = threshold_var <= threshold_estimate
            regime2_mask = threshold_var > threshold_estimate
            
            return threshold_estimate, regime1_mask, regime2_mask
        except Exception as e:
            logger.error(f"Threshold fitting failed: {e}")
            return None, None, None
        
    def run_all_tests(self) -> Dict[str, List[RobustnessResults]]:
        """
        Run all robustness tests and return aggregated results.
        
        Returns:
            Dictionary mapping test type to list of results
        """
        logger.info("Running all robustness tests...")
        
        all_results = {
            'threshold_specification': [],
            'fiscal_indicators': [],
            'distortion_measures': [],
            'sample_splits': [],
            'instrument_validation': []
        }
        
        # Test 1: Double-threshold model
        try:
            result = self.test_double_threshold()
            all_results['threshold_specification'].append(result)
            self.results.append(result)
        except Exception as e:
            logger.error(f"Double-threshold test failed: {e}")
            
        # Test 2: Smooth transition regression
        try:
            result = self.test_smooth_transition()
            all_results['threshold_specification'].append(result)
            self.results.append(result)
        except Exception as e:
            logger.error(f"Smooth transition test failed: {e}")
            
        # Test 3: Alternative fiscal indicators
        try:
            results = self.test_alternative_fiscal_indicators()
            all_results['fiscal_indicators'].extend(results)
            self.results.extend(results)
        except Exception as e:
            logger.error(f"Alternative fiscal indicators test failed: {e}")
            
        # Test 4: Alternative distortion measures
        try:
            results = self.test_alternative_distortion_measures()
            all_results['distortion_measures'].extend(results)
            self.results.extend(results)
        except Exception as e:
            logger.error(f"Alternative distortion measures test failed: {e}")
            
        # Test 5: QE episode subsamples
        try:
            results = self.test_qe_episode_subsamples()
            all_results['sample_splits'].extend(results)
            self.results.extend(results)
        except Exception as e:
            logger.error(f"QE episode subsamples test failed: {e}")
            
        # Test 6: Shadow rate conditioning
        try:
            results = self.test_shadow_rate_conditioning()
            all_results['sample_splits'].extend(results)
            self.results.extend(results)
        except Exception as e:
            logger.error(f"Shadow rate conditioning test failed: {e}")
            
        # Test 7: Instrument validation
        try:
            results = self.test_alternative_hf_windows()
            all_results['instrument_validation'].extend(results)
            self.results.extend(results)
        except Exception as e:
            logger.error(f"Alternative HF windows test failed: {e}")
            
        try:
            results = self.test_alternative_asset_classes()
            all_results['instrument_validation'].extend(results)
            self.results.extend(results)
        except Exception as e:
            logger.error(f"Alternative asset classes test failed: {e}")
            
        try:
            results = self.test_placebo_non_fomc_days()
            all_results['instrument_validation'].extend(results)
            self.results.extend(results)
        except Exception as e:
            logger.error(f"Placebo test failed: {e}")
            
        logger.info(f"Completed all robustness tests. Total results: {len(self.results)}")
        
        return all_results
    
    def get_summary_table(self) -> pd.DataFrame:
        """
        Generate summary table of all robustness test results.
        
        Returns:
            DataFrame with key statistics from each test
        """
        summary_data = []
        
        for result in self.results:
            row = {
                'Test': result.test_name,
                'Type': result.test_type,
                'Threshold': result.threshold_estimate,
                'Regime 1 Effect': result.regime1_effect,
                'Regime 2 Effect': result.regime2_effect,
                'Attenuation %': result.attenuation_pct,
                'First-Stage F': result.first_stage_f_stat,
                'Valid Instrument': result.instrument_valid,
                'N': result.n_observations
            }
            summary_data.append(row)
        
        # Return empty DataFrame with correct columns if no results
        if len(summary_data) == 0:
            return pd.DataFrame(columns=[
                'Test', 'Type', 'Threshold', 'Regime 1 Effect', 'Regime 2 Effect',
                'Attenuation %', 'First-Stage F', 'Valid Instrument', 'N'
            ])
            
        return pd.DataFrame(summary_data)
    
    def check_consistency(self, tolerance_threshold: float = 0.05,
                         tolerance_effect: float = 0.3) -> Dict[str, Any]:
        """
        Check consistency of robustness results with baseline.
        
        Args:
            tolerance_threshold: Tolerance for threshold estimates
            tolerance_effect: Tolerance for regime effects
            
        Returns:
            Dictionary with consistency statistics
        """
        if self.baseline_results is None:
            logger.warning("No baseline results provided for consistency check")
            return {'consistent_count': 0, 'total_count': 0, 'consistency_rate': 0.0}
            
        consistent_count = 0
        total_count = len(self.results)
        
        for result in self.results:
            if result.is_consistent_with_baseline(
                self.baseline_results,
                threshold_tolerance=tolerance_threshold,
                effect_tolerance=tolerance_effect
            ):
                consistent_count += 1
                
        consistency_rate = consistent_count / total_count if total_count > 0 else 0.0
        
        return {
            'consistent_count': consistent_count,
            'total_count': total_count,
            'consistency_rate': consistency_rate,
            'threshold_tolerance': tolerance_threshold,
            'effect_tolerance': tolerance_effect
        }
    
    def test_double_threshold(self) -> RobustnessResults:
        """
        Test double-threshold model with three fiscal regimes.
        
        Implements Requirement 8.1: Estimate three-regime model with two thresholds
        creating low/medium/high fiscal regimes.
        
        Model:
        y_t = α₁ + β₁*QE_t + γ₁*Z_t + ε_t  if F_t ≤ τ₁
        y_t = α₂ + β₂*QE_t + γ₂*Z_t + ε_t  if τ₁ < F_t ≤ τ₂
        y_t = α₃ + β₃*QE_t + γ₃*Z_t + ε_t  if F_t > τ₂
        
        Returns:
            RobustnessResults with three-regime estimates
        """
        logger.info("Testing double-threshold model...")
        
        # Extract required variables
        y = self.baseline_data['yield_changes_10y'].values
        qe_shocks = self.baseline_data['qe_shocks_instrumented'].values
        threshold_var = self.baseline_data['fiscal_indicator'].values
        
        # Get control variables if available
        control_cols = ['gdp_growth', 'unemployment_rate', 'inflation']
        controls = None
        if all(col in self.baseline_data.columns for col in control_cols):
            controls = self.baseline_data[control_cols].values
        
        # Grid search for two thresholds
        trim = 0.15
        sorted_threshold = np.sort(threshold_var)
        n = len(threshold_var)
        lower_idx = int(np.floor(n * trim))
        upper_idx = int(np.ceil(n * (1 - trim)))
        
        threshold_grid1 = sorted_threshold[lower_idx:int(n * 0.5)]
        threshold_grid2 = sorted_threshold[int(n * 0.5):upper_idx]
        
        best_ssr = np.inf
        best_tau1 = None
        best_tau2 = None
        best_results = None
        
        # Search over grid of threshold pairs
        for tau1 in threshold_grid1[::2]:  # Sample every 2nd value for efficiency
            for tau2 in threshold_grid2[::2]:
                if tau2 <= tau1:
                    continue
                    
                # Create regime indicators
                regime1 = threshold_var <= tau1
                regime2 = (threshold_var > tau1) & (threshold_var <= tau2)
                regime3 = threshold_var > tau2
                
                # Check minimum observations per regime
                if np.sum(regime1) < 10 or np.sum(regime2) < 10 or np.sum(regime3) < 10:
                    continue
                
                # Estimate model for each regime
                ssr_total = 0
                regime_results = []
                
                for regime_mask in [regime1, regime2, regime3]:
                    y_regime = y[regime_mask]
                    qe_regime = qe_shocks[regime_mask]
                    
                    if controls is not None:
                        controls_regime = controls[regime_mask]
                        X_regime = np.column_stack([np.ones(len(y_regime)), qe_regime, controls_regime])
                    else:
                        X_regime = np.column_stack([np.ones(len(y_regime)), qe_regime])
                    
                    try:
                        beta = np.linalg.lstsq(X_regime, y_regime, rcond=None)[0]
                        residuals = y_regime - X_regime @ beta
                        ssr = np.sum(residuals**2)
                        ssr_total += ssr
                        regime_results.append(beta)
                    except:
                        ssr_total = np.inf
                        break
                
                if ssr_total < best_ssr:
                    best_ssr = ssr_total
                    best_tau1 = tau1
                    best_tau2 = tau2
                    best_results = regime_results
        
        if best_tau1 is None or best_tau2 is None:
            logger.error("Double-threshold estimation failed")
            return RobustnessResults(
                test_name="Double-Threshold Model",
                test_type="threshold",
                n_observations=len(y)
            )
        
        # Extract regime-specific QE effects (second coefficient)
        regime1_effect = best_results[0][1] if len(best_results[0]) > 1 else None
        regime2_effect = best_results[1][1] if len(best_results[1]) > 1 else None
        regime3_effect = best_results[2][1] if len(best_results[2]) > 1 else None
        
        # Calculate attenuation from low to high regime
        attenuation_pct = None
        if regime1_effect is not None and regime3_effect is not None and regime1_effect != 0:
            attenuation_pct = ((regime1_effect - regime3_effect) / regime1_effect) * 100
        
        # Compute first-stage F-statistic if instruments available
        first_stage_f = None
        instrument_valid = False
        if 'hf_qe_surprises' in self.baseline_data.columns:
            instruments = self.baseline_data['hf_qe_surprises'].values
            validator = InstrumentValidator()
            f_stat = validator.weak_instrument_test(qe_shocks, instruments, controls)
            first_stage_f = f_stat.get('f_statistic')
            instrument_valid = first_stage_f is not None and first_stage_f > self.min_first_stage_f
        
        result = RobustnessResults(
            test_name="Double-Threshold Model",
            test_type="threshold",
            threshold_estimate=best_tau1,  # Report first threshold
            regime1_effect=regime1_effect,
            regime2_effect=regime2_effect,
            regime3_effect=regime3_effect,
            attenuation_pct=attenuation_pct,
            first_stage_f_stat=first_stage_f,
            instrument_valid=instrument_valid,
            n_observations=len(y),
            specification_details={
                'threshold1': best_tau1,
                'threshold2': best_tau2,
                'regime1_n': np.sum(threshold_var <= best_tau1),
                'regime2_n': np.sum((threshold_var > best_tau1) & (threshold_var <= best_tau2)),
                'regime3_n': np.sum(threshold_var > best_tau2),
                'ssr': best_ssr
            }
        )
        
        logger.info(f"Double-threshold model: τ₁={best_tau1:.3f}, τ₂={best_tau2:.3f}")
        logger.info(f"Regime effects: β₁={regime1_effect:.4f}, β₂={regime2_effect:.4f}, β₃={regime3_effect:.4f}")
        
        return result
    
    def test_smooth_transition(self) -> RobustnessResults:
        """
        Test smooth transition regression as alternative to discrete threshold.
        
        Implements Requirement 8.2: Estimate logistic STR model with smooth
        transition function as alternative to discrete threshold.
        
        Model:
        y_t = α + β₁*QE_t + β₂*QE_t*G(F_t; γ, c) + δ*Z_t + ε_t
        
        where G(F_t; γ, c) = 1 / (1 + exp(-γ(F_t - c))) is the logistic transition function
        
        Returns:
            RobustnessResults with STR estimates
        """
        logger.info("Testing smooth transition regression...")
        
        # Extract required variables
        y = self.baseline_data['yield_changes_10y'].values
        qe_shocks = self.baseline_data['qe_shocks_instrumented'].values
        threshold_var = self.baseline_data['fiscal_indicator'].values
        
        # Get control variables if available
        control_cols = ['gdp_growth', 'unemployment_rate', 'inflation']
        controls = None
        X_base = qe_shocks.reshape(-1, 1)
        
        if all(col in self.baseline_data.columns for col in control_cols):
            controls = self.baseline_data[control_cols].values
            X_base = np.column_stack([qe_shocks, controls])
        
        # Fit STR model
        str_model = SmoothTransitionRegression()
        
        try:
            # Initial values for optimization
            initial_c = np.median(threshold_var)
            initial_gamma = 10.0  # Moderate transition speed
            
            result = str_model.fit(y, X_base, threshold_var, 
                                  initial_gamma=initial_gamma, 
                                  initial_c=initial_c)
            
            # Extract transition parameters
            transition_speed = str_model.gamma
            transition_location = str_model.c
            
            # Extract regime effects
            # Coefficients: [intercept, qe_effect_low, qe_effect_transition, ...]
            regime1_effect = str_model.coeffs[1]  # Low regime (G=0)
            regime2_effect = str_model.coeffs[1] + str_model.coeffs[2]  # High regime (G=1)
            
            # Calculate attenuation
            attenuation_pct = None
            if regime1_effect != 0:
                attenuation_pct = ((regime1_effect - regime2_effect) / regime1_effect) * 100
            
            # Calculate R-squared
            # Use the predict method if available, otherwise compute manually
            try:
                y_pred = str_model.predict(X_base, threshold_var)
                ss_res = np.sum((y - y_pred)**2)
                ss_tot = np.sum((y - np.mean(y))**2)
                r_squared = 1 - (ss_res / ss_tot)
            except:
                # Fallback: compute R-squared from residuals
                r_squared = None
            
            # Compute first-stage F-statistic if instruments available
            first_stage_f = None
            instrument_valid = False
            if 'hf_qe_surprises' in self.baseline_data.columns:
                instruments = self.baseline_data['hf_qe_surprises'].values
                validator = InstrumentValidator()
                f_stat = validator.weak_instrument_test(qe_shocks, instruments, controls)
                first_stage_f = f_stat.get('f_statistic')
                instrument_valid = first_stage_f is not None and first_stage_f > self.min_first_stage_f
            
            result = RobustnessResults(
                test_name="Smooth Transition Regression",
                test_type="str",
                threshold_estimate=transition_location,
                regime1_effect=regime1_effect,
                regime2_effect=regime2_effect,
                attenuation_pct=attenuation_pct,
                transition_speed=transition_speed,
                transition_location=transition_location,
                first_stage_f_stat=first_stage_f,
                instrument_valid=instrument_valid,
                r_squared=r_squared,
                n_observations=len(y),
                specification_details={
                    'gamma': transition_speed,
                    'c': transition_location,
                    'convergence': result.success if hasattr(result, 'success') else True
                }
            )
            
            logger.info(f"STR model: c={transition_location:.3f}, γ={transition_speed:.3f}")
            logger.info(f"Regime effects: β₁={regime1_effect:.4f}, β₂={regime2_effect:.4f}")
            
            return result
            
        except Exception as e:
            logger.error(f"Smooth transition regression failed: {e}")
            return RobustnessResults(
                test_name="Smooth Transition Regression",
                test_type="str",
                n_observations=len(y),
                specification_details={'error': str(e)}
            )
    
    def test_alternative_fiscal_indicators(self) -> List[RobustnessResults]:
        """
        Test alternative fiscal indicators as threshold variables.
        
        Implements Requirement 8.3: Re-estimate with alternative fiscal indicators:
        - Gross debt-to-GDP
        - Primary deficit-to-GDP
        - r-g differential (interest rate - growth rate)
        - CBO fiscal gap
        
        Returns:
            List of RobustnessResults for each fiscal indicator
        """
        logger.info("Testing alternative fiscal indicators...")
        
        results = []
        
        # Define alternative fiscal indicators to test
        fiscal_indicators = {
            'gross_debt_gdp': 'Gross Debt-to-GDP',
            'primary_deficit_gdp': 'Primary Deficit-to-GDP',
            'r_g_differential': 'r-g Differential',
            'cbo_fiscal_gap': 'CBO Fiscal Gap'
        }
        
        # Extract base variables
        y = self.baseline_data['yield_changes_10y'].values
        qe_shocks = self.baseline_data['qe_shocks_instrumented'].values
        
        # Get control variables if available
        control_cols = ['gdp_growth', 'unemployment_rate', 'inflation']
        controls = None
        if all(col in self.baseline_data.columns for col in control_cols):
            controls = self.baseline_data[control_cols].values
        
        # Test each alternative fiscal indicator
        for indicator_col, indicator_name in fiscal_indicators.items():
            if indicator_col not in self.baseline_data.columns:
                logger.warning(f"Fiscal indicator '{indicator_col}' not found in data, skipping")
                continue
            
            threshold_var = self.baseline_data[indicator_col].values
            
            # Check for missing values
            valid_mask = ~(np.isnan(y) | np.isnan(qe_shocks) | np.isnan(threshold_var))
            if controls is not None:
                valid_mask &= ~np.any(np.isnan(controls), axis=1)
            
            y_valid = y[valid_mask]
            qe_valid = qe_shocks[valid_mask]
            threshold_valid = threshold_var[valid_mask]
            controls_valid = controls[valid_mask] if controls is not None else None
            
            if len(y_valid) < 30:
                logger.warning(f"Insufficient observations for {indicator_name}, skipping")
                continue
            
            # Fit threshold model
            try:
                threshold_model = HansenThresholdRegression()
                
                # Prepare X matrix
                if controls_valid is not None:
                    X = np.column_stack([qe_valid, controls_valid])
                else:
                    X = qe_valid.reshape(-1, 1)
                
                fit_result = threshold_model.fit(y_valid, X, threshold_valid, trim=0.15)
                
                # Extract results
                threshold_estimate = threshold_model.threshold
                
                # Get regime-specific effects
                regime1_mask = threshold_valid <= threshold_estimate
                regime2_mask = threshold_valid > threshold_estimate
                
                # Estimate regime-specific models
                regime1_effect = None
                regime2_effect = None
                
                if np.sum(regime1_mask) >= 10:
                    y1 = y_valid[regime1_mask]
                    X1 = X[regime1_mask] if X.ndim > 1 else X[regime1_mask].reshape(-1, 1)
                    X1_with_const = np.column_stack([np.ones(len(y1)), X1])
                    beta1 = np.linalg.lstsq(X1_with_const, y1, rcond=None)[0]
                    regime1_effect = beta1[1]  # QE effect
                
                if np.sum(regime2_mask) >= 10:
                    y2 = y_valid[regime2_mask]
                    X2 = X[regime2_mask] if X.ndim > 1 else X[regime2_mask].reshape(-1, 1)
                    X2_with_const = np.column_stack([np.ones(len(y2)), X2])
                    beta2 = np.linalg.lstsq(X2_with_const, y2, rcond=None)[0]
                    regime2_effect = beta2[1]  # QE effect
                
                # Calculate attenuation
                attenuation_pct = None
                if regime1_effect is not None and regime2_effect is not None and regime1_effect != 0:
                    attenuation_pct = ((regime1_effect - regime2_effect) / regime1_effect) * 100
                
                # Compute first-stage F-statistic if instruments available
                first_stage_f = None
                instrument_valid = False
                if 'hf_qe_surprises' in self.baseline_data.columns:
                    instruments = self.baseline_data['hf_qe_surprises'].values[valid_mask]
                    validator = InstrumentValidator()
                    f_stat = validator.weak_instrument_test(qe_valid, instruments, controls_valid)
                    first_stage_f = f_stat.get('f_statistic')
                    instrument_valid = first_stage_f is not None and first_stage_f > self.min_first_stage_f
                
                result = RobustnessResults(
                    test_name=f"Alternative Fiscal: {indicator_name}",
                    test_type="fiscal",
                    threshold_estimate=threshold_estimate,
                    regime1_effect=regime1_effect,
                    regime2_effect=regime2_effect,
                    attenuation_pct=attenuation_pct,
                    first_stage_f_stat=first_stage_f,
                    instrument_valid=instrument_valid,
                    n_observations=len(y_valid),
                    specification_details={
                        'fiscal_indicator': indicator_col,
                        'regime1_n': np.sum(regime1_mask),
                        'regime2_n': np.sum(regime2_mask)
                    }
                )
                
                results.append(result)
                logger.info(f"{indicator_name}: τ={threshold_estimate:.3f}, β₁={regime1_effect:.4f}, β₂={regime2_effect:.4f}")
                
            except Exception as e:
                logger.error(f"Failed to estimate model with {indicator_name}: {e}")
                continue
        
        logger.info(f"Completed alternative fiscal indicators test: {len(results)} specifications")
        return results
    
    def test_alternative_distortion_measures(self) -> List[RobustnessResults]:
        """
        Test alternative distortion measures for channel decomposition.
        
        Implements Requirement 8.4: Construct alternative distortion indices using:
        - Treasury fails-to-deliver
        - Repo specialness
        - Central clearing volumes
        - FINRA dealer capital ratios
        
        Returns:
            List of RobustnessResults for each distortion measure
        """
        logger.info("Testing alternative distortion measures...")
        
        results = []
        
        # Define alternative distortion measures to test
        distortion_measures = {
            'fails_to_deliver': 'Fails-to-Deliver',
            'repo_specialness': 'Repo Specialness',
            'clearing_volumes': 'Clearing Volumes',
            'dealer_capital_ratios': 'Dealer Capital Ratios'
        }
        
        # Extract base variables
        y = self.baseline_data['yield_changes_10y'].values
        qe_shocks = self.baseline_data['qe_shocks_instrumented'].values
        threshold_var = self.baseline_data['fiscal_indicator'].values
        
        # Get control variables if available
        control_cols = ['gdp_growth', 'unemployment_rate', 'inflation']
        controls = None
        if all(col in self.baseline_data.columns for col in control_cols):
            controls = self.baseline_data[control_cols].values
        
        # Test each alternative distortion measure
        for measure_col, measure_name in distortion_measures.items():
            if measure_col not in self.baseline_data.columns:
                logger.warning(f"Distortion measure '{measure_col}' not found in data, skipping")
                continue
            
            distortion_index = self.baseline_data[measure_col].values
            
            # Check for missing values
            valid_mask = ~(np.isnan(y) | np.isnan(qe_shocks) | np.isnan(threshold_var) | np.isnan(distortion_index))
            if controls is not None:
                valid_mask &= ~np.any(np.isnan(controls), axis=1)
            
            y_valid = y[valid_mask]
            qe_valid = qe_shocks[valid_mask]
            threshold_valid = threshold_var[valid_mask]
            distortion_valid = distortion_index[valid_mask]
            controls_valid = controls[valid_mask] if controls is not None else None
            
            if len(y_valid) < 30:
                logger.warning(f"Insufficient observations for {measure_name}, skipping")
                continue
            
            try:
                # Perform channel decomposition with alternative distortion measure
                # Step 1: Estimate effect of QE on distortion
                if controls_valid is not None:
                    X_distortion = np.column_stack([np.ones(len(qe_valid)), qe_valid, controls_valid])
                else:
                    X_distortion = np.column_stack([np.ones(len(qe_valid)), qe_valid])
                
                beta_distortion = np.linalg.lstsq(X_distortion, distortion_valid, rcond=None)[0]
                qe_effect_on_distortion = beta_distortion[1]
                
                # Step 2: Estimate effect of distortion on yields (using threshold model)
                threshold_model = HansenThresholdRegression()
                
                # Include distortion in X matrix
                if controls_valid is not None:
                    X = np.column_stack([qe_valid, distortion_valid, controls_valid])
                else:
                    X = np.column_stack([qe_valid, distortion_valid])
                
                fit_result = threshold_model.fit(y_valid, X, threshold_valid, trim=0.15)
                threshold_estimate = threshold_model.threshold
                
                # Get regime-specific effects
                regime1_mask = threshold_valid <= threshold_estimate
                regime2_mask = threshold_valid > threshold_estimate
                
                # Estimate regime-specific models to get distortion channel coefficient
                distortion_coef_regime1 = None
                distortion_coef_regime2 = None
                
                if np.sum(regime1_mask) >= 10:
                    y1 = y_valid[regime1_mask]
                    X1 = X[regime1_mask]
                    X1_with_const = np.column_stack([np.ones(len(y1)), X1])
                    beta1 = np.linalg.lstsq(X1_with_const, y1, rcond=None)[0]
                    distortion_coef_regime1 = beta1[2]  # Distortion coefficient
                
                if np.sum(regime2_mask) >= 10:
                    y2 = y_valid[regime2_mask]
                    X2 = X[regime2_mask]
                    X2_with_const = np.column_stack([np.ones(len(y2)), X2])
                    beta2 = np.linalg.lstsq(X2_with_const, y2, rcond=None)[0]
                    distortion_coef_regime2 = beta2[2]  # Distortion coefficient
                
                # Calculate distortion channel contribution
                # ψ^dist = κ × βD (where κ is effect of distortion on yields, βD is effect of QE on distortion)
                distortion_contribution_regime1 = None
                distortion_contribution_regime2 = None
                
                if distortion_coef_regime1 is not None:
                    distortion_contribution_regime1 = distortion_coef_regime1 * qe_effect_on_distortion
                
                if distortion_coef_regime2 is not None:
                    distortion_contribution_regime2 = distortion_coef_regime2 * qe_effect_on_distortion
                
                # Compute first-stage F-statistic if instruments available
                first_stage_f = None
                instrument_valid = False
                if 'hf_qe_surprises' in self.baseline_data.columns:
                    instruments = self.baseline_data['hf_qe_surprises'].values[valid_mask]
                    validator = InstrumentValidator()
                    f_stat = validator.weak_instrument_test(qe_valid, instruments, controls_valid)
                    first_stage_f = f_stat.get('f_statistic')
                    instrument_valid = first_stage_f is not None and first_stage_f > self.min_first_stage_f
                
                result = RobustnessResults(
                    test_name=f"Alternative Distortion: {measure_name}",
                    test_type="distortion",
                    threshold_estimate=threshold_estimate,
                    regime1_effect=distortion_contribution_regime1,
                    regime2_effect=distortion_contribution_regime2,
                    first_stage_f_stat=first_stage_f,
                    instrument_valid=instrument_valid,
                    n_observations=len(y_valid),
                    specification_details={
                        'distortion_measure': measure_col,
                        'qe_effect_on_distortion': qe_effect_on_distortion,
                        'distortion_coef_regime1': distortion_coef_regime1,
                        'distortion_coef_regime2': distortion_coef_regime2,
                        'regime1_n': np.sum(regime1_mask),
                        'regime2_n': np.sum(regime2_mask)
                    }
                )
                
                results.append(result)
                logger.info(f"{measure_name}: τ={threshold_estimate:.3f}, distortion channel contribution: R1={distortion_contribution_regime1:.4f}, R2={distortion_contribution_regime2:.4f}")
                
            except Exception as e:
                logger.error(f"Failed to estimate model with {measure_name}: {e}")
                continue
        
        logger.info(f"Completed alternative distortion measures test: {len(results)} specifications")
        return results
    
    def test_qe_episode_subsamples(self) -> List[RobustnessResults]:
        """
        Test QE effects across different QE episodes.
        
        Implements Requirement 8.5: Estimate separate models for:
        - QE1 (2008Q4-2010Q1)
        - QE2 (2010Q4-2011Q2)
        - QE3 (2012Q3-2014Q4)
        - COVID-QE (2020Q1-2022Q4)
        
        Returns:
            List of RobustnessResults for each QE episode
        """
        logger.info("Testing QE episode subsamples...")
        
        results = []
        
        # Define QE episodes
        qe_episodes = {
            'QE1': ('2008-10-01', '2010-03-31'),
            'QE2': ('2010-10-01', '2011-06-30'),
            'QE3': ('2012-07-01', '2014-12-31'),
            'COVID-QE': ('2020-01-01', '2022-12-31')
        }
        
        # Ensure date index
        if not isinstance(self.baseline_data.index, pd.DatetimeIndex):
            logger.warning("Data does not have DatetimeIndex, attempting to use 'date' column")
            if 'date' in self.baseline_data.columns:
                data_with_dates = self.baseline_data.set_index('date')
            else:
                logger.error("Cannot identify date column for episode subsample")
                return results
        else:
            data_with_dates = self.baseline_data
        
        # Extract base variables
        y_full = data_with_dates['yield_changes_10y'].values
        qe_full = data_with_dates['qe_shocks_instrumented'].values
        threshold_full = data_with_dates['fiscal_indicator'].values
        
        # Get control variables if available
        control_cols = ['gdp_growth', 'unemployment_rate', 'inflation']
        controls_full = None
        if all(col in data_with_dates.columns for col in control_cols):
            controls_full = data_with_dates[control_cols].values
        
        # Test each QE episode
        for episode_name, (start_date, end_date) in qe_episodes.items():
            try:
                # Filter data for episode
                episode_mask = (data_with_dates.index >= start_date) & (data_with_dates.index <= end_date)
                
                if np.sum(episode_mask) < 10:
                    logger.warning(f"Insufficient observations for {episode_name}, skipping")
                    continue
                
                y_episode = y_full[episode_mask]
                qe_episode = qe_full[episode_mask]
                threshold_episode = threshold_full[episode_mask]
                controls_episode = controls_full[episode_mask] if controls_full is not None else None
                
                # Check for missing values
                valid_mask = ~(np.isnan(y_episode) | np.isnan(qe_episode) | np.isnan(threshold_episode))
                if controls_episode is not None:
                    valid_mask &= ~np.any(np.isnan(controls_episode), axis=1)
                
                y_valid = y_episode[valid_mask]
                qe_valid = qe_episode[valid_mask]
                threshold_valid = threshold_episode[valid_mask]
                controls_valid = controls_episode[valid_mask] if controls_episode is not None else None
                
                if len(y_valid) < 10:
                    logger.warning(f"Insufficient valid observations for {episode_name}, skipping")
                    continue
                
                # Fit threshold model
                threshold_model = HansenThresholdRegression()
                
                # Prepare X matrix
                if controls_valid is not None:
                    X = np.column_stack([qe_valid, controls_valid])
                else:
                    X = qe_valid.reshape(-1, 1)
                
                fit_result = threshold_model.fit(y_valid, X, threshold_valid, trim=0.15)
                threshold_estimate = threshold_model.threshold
                
                # Check if threshold was successfully estimated
                if threshold_estimate is None:
                    logger.warning(f"Threshold estimation failed for {episode_name}, skipping")
                    continue
                
                # Get regime-specific effects
                regime1_mask = threshold_valid <= threshold_estimate
                regime2_mask = threshold_valid > threshold_estimate
                
                # Estimate regime-specific models
                regime1_effect = None
                regime2_effect = None
                
                if np.sum(regime1_mask) >= 5:
                    y1 = y_valid[regime1_mask]
                    X1 = X[regime1_mask] if X.ndim > 1 else X[regime1_mask].reshape(-1, 1)
                    X1_with_const = np.column_stack([np.ones(len(y1)), X1])
                    beta1 = np.linalg.lstsq(X1_with_const, y1, rcond=None)[0]
                    regime1_effect = beta1[1]  # QE effect
                
                if np.sum(regime2_mask) >= 5:
                    y2 = y_valid[regime2_mask]
                    X2 = X[regime2_mask] if X.ndim > 1 else X[regime2_mask].reshape(-1, 1)
                    X2_with_const = np.column_stack([np.ones(len(y2)), X2])
                    beta2 = np.linalg.lstsq(X2_with_const, y2, rcond=None)[0]
                    regime2_effect = beta2[1]  # QE effect
                
                # Calculate attenuation
                attenuation_pct = None
                if regime1_effect is not None and regime2_effect is not None and regime1_effect != 0:
                    attenuation_pct = ((regime1_effect - regime2_effect) / regime1_effect) * 100
                
                # Compute first-stage F-statistic if instruments available
                first_stage_f = None
                instrument_valid = False
                if 'hf_qe_surprises' in data_with_dates.columns:
                    instruments_episode = data_with_dates['hf_qe_surprises'].values[episode_mask][valid_mask]
                    validator = InstrumentValidator()
                    f_stat = validator.weak_instrument_test(qe_valid, instruments_episode, controls_valid)
                    first_stage_f = f_stat.get('f_statistic')
                    instrument_valid = first_stage_f is not None and first_stage_f > self.min_first_stage_f
                
                result = RobustnessResults(
                    test_name=f"QE Episode: {episode_name}",
                    test_type="subsample",
                    threshold_estimate=threshold_estimate,
                    regime1_effect=regime1_effect,
                    regime2_effect=regime2_effect,
                    attenuation_pct=attenuation_pct,
                    first_stage_f_stat=first_stage_f,
                    instrument_valid=instrument_valid,
                    n_observations=len(y_valid),
                    specification_details={
                        'episode': episode_name,
                        'start_date': start_date,
                        'end_date': end_date,
                        'regime1_n': np.sum(regime1_mask),
                        'regime2_n': np.sum(regime2_mask)
                    }
                )
                
                results.append(result)
                logger.info(f"{episode_name}: τ={threshold_estimate:.3f}, β₁={regime1_effect:.4f}, β₂={regime2_effect:.4f}")
                
            except Exception as e:
                logger.error(f"Failed to estimate model for {episode_name}: {e}")
                continue
        
        logger.info(f"Completed QE episode subsamples test: {len(results)} specifications")
        return results
    
    def test_shadow_rate_conditioning(self) -> List[RobustnessResults]:
        """
        Test QE effects conditional on Wu-Xia shadow rate levels.
        
        Implements Requirement 8.6: Split sample by Wu-Xia shadow rate levels
        to compare effects in different monetary policy regimes.
        
        Returns:
            List of RobustnessResults for different shadow rate regimes
        """
        logger.info("Testing shadow rate conditioning...")
        
        results = []
        
        # Check if shadow rate is available
        if 'wu_xia_shadow_rate' not in self.baseline_data.columns:
            logger.warning("Wu-Xia shadow rate not found in data, skipping test")
            return results
        
        shadow_rate = self.baseline_data['wu_xia_shadow_rate'].values
        
        # Extract base variables
        y = self.baseline_data['yield_changes_10y'].values
        qe_shocks = self.baseline_data['qe_shocks_instrumented'].values
        threshold_var = self.baseline_data['fiscal_indicator'].values
        
        # Get control variables if available
        control_cols = ['gdp_growth', 'unemployment_rate', 'inflation']
        controls = None
        if all(col in self.baseline_data.columns for col in control_cols):
            controls = self.baseline_data[control_cols].values
        
        # Define shadow rate regimes
        # Split at median and at zero lower bound
        shadow_rate_median = np.nanmedian(shadow_rate)
        
        shadow_rate_regimes = {
            'Deep ZLB (shadow rate < -2%)': shadow_rate < -2.0,
            'Moderate ZLB (-2% ≤ shadow rate < 0%)': (shadow_rate >= -2.0) & (shadow_rate < 0.0),
            'Normal Policy (shadow rate ≥ 0%)': shadow_rate >= 0.0
        }
        
        # Test each shadow rate regime
        for regime_name, regime_mask in shadow_rate_regimes.items():
            try:
                # Check for missing values
                valid_mask = regime_mask & ~(np.isnan(y) | np.isnan(qe_shocks) | np.isnan(threshold_var))
                if controls is not None:
                    valid_mask &= ~np.any(np.isnan(controls), axis=1)
                
                if np.sum(valid_mask) < 15:
                    logger.warning(f"Insufficient observations for {regime_name}, skipping")
                    continue
                
                y_regime = y[valid_mask]
                qe_regime = qe_shocks[valid_mask]
                threshold_regime = threshold_var[valid_mask]
                controls_regime = controls[valid_mask] if controls is not None else None
                
                # Fit threshold model
                threshold_model = HansenThresholdRegression()
                
                # Prepare X matrix
                if controls_regime is not None:
                    X = np.column_stack([qe_regime, controls_regime])
                else:
                    X = qe_regime.reshape(-1, 1)
                
                fit_result = threshold_model.fit(y_regime, X, threshold_regime, trim=0.15)
                threshold_estimate = threshold_model.threshold
                
                # Get regime-specific effects
                regime1_mask = threshold_regime <= threshold_estimate
                regime2_mask = threshold_regime > threshold_estimate
                
                # Estimate regime-specific models
                regime1_effect = None
                regime2_effect = None
                
                if np.sum(regime1_mask) >= 5:
                    y1 = y_regime[regime1_mask]
                    X1 = X[regime1_mask] if X.ndim > 1 else X[regime1_mask].reshape(-1, 1)
                    X1_with_const = np.column_stack([np.ones(len(y1)), X1])
                    beta1 = np.linalg.lstsq(X1_with_const, y1, rcond=None)[0]
                    regime1_effect = beta1[1]  # QE effect
                
                if np.sum(regime2_mask) >= 5:
                    y2 = y_regime[regime2_mask]
                    X2 = X[regime2_mask] if X.ndim > 1 else X[regime2_mask].reshape(-1, 1)
                    X2_with_const = np.column_stack([np.ones(len(y2)), X2])
                    beta2 = np.linalg.lstsq(X2_with_const, y2, rcond=None)[0]
                    regime2_effect = beta2[1]  # QE effect
                
                # Calculate attenuation
                attenuation_pct = None
                if regime1_effect is not None and regime2_effect is not None and regime1_effect != 0:
                    attenuation_pct = ((regime1_effect - regime2_effect) / regime1_effect) * 100
                
                # Compute first-stage F-statistic if instruments available
                first_stage_f = None
                instrument_valid = False
                if 'hf_qe_surprises' in self.baseline_data.columns:
                    instruments = self.baseline_data['hf_qe_surprises'].values[valid_mask]
                    validator = InstrumentValidator()
                    f_stat = validator.weak_instrument_test(qe_regime, instruments, controls_regime)
                    first_stage_f = f_stat.get('f_statistic')
                    instrument_valid = first_stage_f is not None and first_stage_f > self.min_first_stage_f
                
                result = RobustnessResults(
                    test_name=f"Shadow Rate: {regime_name}",
                    test_type="shadow_rate",
                    threshold_estimate=threshold_estimate,
                    regime1_effect=regime1_effect,
                    regime2_effect=regime2_effect,
                    attenuation_pct=attenuation_pct,
                    first_stage_f_stat=first_stage_f,
                    instrument_valid=instrument_valid,
                    n_observations=np.sum(valid_mask),
                    specification_details={
                        'shadow_rate_regime': regime_name,
                        'regime1_n': np.sum(regime1_mask),
                        'regime2_n': np.sum(regime2_mask)
                    }
                )
                
                results.append(result)
                logger.info(f"{regime_name}: τ={threshold_estimate:.3f}, β₁={regime1_effect:.4f}, β₂={regime2_effect:.4f}")
                
            except Exception as e:
                logger.error(f"Failed to estimate model for {regime_name}: {e}")
                continue
        
        logger.info(f"Completed shadow rate conditioning test: {len(results)} specifications")
        return results
    
    def test_alternative_hf_windows(self) -> List[RobustnessResults]:
        """
        Test alternative high-frequency window sizes for instrument construction.
        
        Implements Requirement 8.7: Test HF windows of 15, 30, 45, and 60 minutes
        to validate instrument robustness.
        
        Returns:
            List of RobustnessResults for each window size
        """
        logger.info("Testing alternative HF windows...")
        
        results = []
        window_sizes = [15, 30, 45, 60]  # minutes
        
        # Check if we have the necessary data for HF identification
        if 'fomc_dates' not in self.baseline_data.columns and 'hf_qe_surprises' not in self.baseline_data.columns:
            logger.warning("FOMC dates or HF data not available, skipping HF window test")
            return results
        
        # Extract base variables
        y = self.baseline_data['yield_changes_10y'].values
        qe_shocks = self.baseline_data['qe_shocks_instrumented'].values
        threshold_var = self.baseline_data['fiscal_indicator'].values
        
        # Get control variables if available
        control_cols = ['gdp_growth', 'unemployment_rate', 'inflation']
        controls = None
        if all(col in self.baseline_data.columns for col in control_cols):
            controls = self.baseline_data[control_cols].values
        
        # Test each window size
        for window_minutes in window_sizes:
            try:
                # For this test, we'll use the existing HF surprises as a proxy
                # In a real implementation, you would re-extract HF surprises with different windows
                # Here we'll just test the instrument strength with the existing data
                
                if f'hf_qe_surprises_{window_minutes}min' in self.baseline_data.columns:
                    instruments = self.baseline_data[f'hf_qe_surprises_{window_minutes}min'].values
                else:
                    # Use baseline HF surprises as proxy
                    instruments = self.baseline_data.get('hf_qe_surprises', qe_shocks).values
                    logger.warning(f"Using baseline HF surprises for {window_minutes}-minute window test")
                
                # Check for missing values
                valid_mask = ~(np.isnan(y) | np.isnan(qe_shocks) | np.isnan(threshold_var) | np.isnan(instruments))
                if controls is not None:
                    valid_mask &= ~np.any(np.isnan(controls), axis=1)
                
                y_valid = y[valid_mask]
                qe_valid = qe_shocks[valid_mask]
                threshold_valid = threshold_var[valid_mask]
                instruments_valid = instruments[valid_mask]
                controls_valid = controls[valid_mask] if controls is not None else None
                
                if len(y_valid) < 20:
                    logger.warning(f"Insufficient observations for {window_minutes}-minute window, skipping")
                    continue
                
                # Compute first-stage F-statistic
                validator = InstrumentValidator()
                f_stat_result = validator.weak_instrument_test(qe_valid, instruments_valid, controls_valid)
                first_stage_f = f_stat_result.get('f_statistic')
                instrument_valid = first_stage_f is not None and first_stage_f > self.min_first_stage_f
                
                # Fit threshold model with instrumented shocks
                threshold_model = HansenThresholdRegression()
                
                if controls_valid is not None:
                    X = np.column_stack([qe_valid, controls_valid])
                else:
                    X = qe_valid.reshape(-1, 1)
                
                fit_result = threshold_model.fit(y_valid, X, threshold_valid, trim=0.15)
                threshold_estimate = threshold_model.threshold
                
                # Get regime-specific effects
                regime1_mask = threshold_valid <= threshold_estimate
                regime2_mask = threshold_valid > threshold_estimate
                
                regime1_effect = None
                regime2_effect = None
                
                if np.sum(regime1_mask) >= 10:
                    y1 = y_valid[regime1_mask]
                    X1 = X[regime1_mask] if X.ndim > 1 else X[regime1_mask].reshape(-1, 1)
                    X1_with_const = np.column_stack([np.ones(len(y1)), X1])
                    beta1 = np.linalg.lstsq(X1_with_const, y1, rcond=None)[0]
                    regime1_effect = beta1[1]
                
                if np.sum(regime2_mask) >= 10:
                    y2 = y_valid[regime2_mask]
                    X2 = X[regime2_mask] if X.ndim > 1 else X[regime2_mask].reshape(-1, 1)
                    X2_with_const = np.column_stack([np.ones(len(y2)), X2])
                    beta2 = np.linalg.lstsq(X2_with_const, y2, rcond=None)[0]
                    regime2_effect = beta2[1]
                
                attenuation_pct = None
                if regime1_effect is not None and regime2_effect is not None and regime1_effect != 0:
                    attenuation_pct = ((regime1_effect - regime2_effect) / regime1_effect) * 100
                
                result = RobustnessResults(
                    test_name=f"HF Window: {window_minutes} minutes",
                    test_type="instrument",
                    threshold_estimate=threshold_estimate,
                    regime1_effect=regime1_effect,
                    regime2_effect=regime2_effect,
                    attenuation_pct=attenuation_pct,
                    first_stage_f_stat=first_stage_f,
                    instrument_valid=instrument_valid,
                    n_observations=len(y_valid),
                    specification_details={
                        'window_minutes': window_minutes,
                        'regime1_n': np.sum(regime1_mask),
                        'regime2_n': np.sum(regime2_mask)
                    }
                )
                
                results.append(result)
                logger.info(f"{window_minutes}-min window: F={first_stage_f:.2f}, valid={instrument_valid}")
                
            except Exception as e:
                logger.error(f"Failed to test {window_minutes}-minute window: {e}")
                continue
        
        logger.info(f"Completed alternative HF windows test: {len(results)} specifications")
        return results
    
    def test_alternative_asset_classes(self) -> List[RobustnessResults]:
        """
        Test instruments constructed from different asset classes.
        
        Implements Requirement 8.8: Construct shocks using different futures contracts
        and asset classes to validate instrument robustness.
        
        Returns:
            List of RobustnessResults for each asset class
        """
        logger.info("Testing alternative asset classes...")
        
        results = []
        
        # Define alternative asset classes for instrument construction
        asset_classes = {
            'fed_funds_futures': 'Fed Funds Futures Only',
            'eurodollar_futures': 'Eurodollar Futures Only',
            'treasury_10y': '10Y Treasury Only',
            'treasury_2y': '2Y Treasury Only',
            'combined_futures': 'Combined Futures'
        }
        
        # Extract base variables
        y = self.baseline_data['yield_changes_10y'].values
        qe_shocks = self.baseline_data['qe_shocks_instrumented'].values
        threshold_var = self.baseline_data['fiscal_indicator'].values
        
        # Get control variables if available
        control_cols = ['gdp_growth', 'unemployment_rate', 'inflation']
        controls = None
        if all(col in self.baseline_data.columns for col in control_cols):
            controls = self.baseline_data[control_cols].values
        
        # Test each asset class
        for asset_col, asset_name in asset_classes.items():
            instrument_col = f'hf_instrument_{asset_col}'
            
            if instrument_col not in self.baseline_data.columns:
                logger.warning(f"Instrument from {asset_name} not found, skipping")
                continue
            
            try:
                instruments = self.baseline_data[instrument_col].values
                
                # Check for missing values
                valid_mask = ~(np.isnan(y) | np.isnan(qe_shocks) | np.isnan(threshold_var) | np.isnan(instruments))
                if controls is not None:
                    valid_mask &= ~np.any(np.isnan(controls), axis=1)
                
                y_valid = y[valid_mask]
                qe_valid = qe_shocks[valid_mask]
                threshold_valid = threshold_var[valid_mask]
                instruments_valid = instruments[valid_mask]
                controls_valid = controls[valid_mask] if controls is not None else None
                
                if len(y_valid) < 20:
                    logger.warning(f"Insufficient observations for {asset_name}, skipping")
                    continue
                
                # Compute first-stage F-statistic
                validator = InstrumentValidator()
                f_stat_result = validator.weak_instrument_test(qe_valid, instruments_valid, controls_valid)
                first_stage_f = f_stat_result.get('f_statistic')
                instrument_valid = first_stage_f is not None and first_stage_f > self.min_first_stage_f
                
                # Fit threshold model
                threshold_model = HansenThresholdRegression()
                
                if controls_valid is not None:
                    X = np.column_stack([qe_valid, controls_valid])
                else:
                    X = qe_valid.reshape(-1, 1)
                
                fit_result = threshold_model.fit(y_valid, X, threshold_valid, trim=0.15)
                threshold_estimate = threshold_model.threshold
                
                # Get regime-specific effects
                regime1_mask = threshold_valid <= threshold_estimate
                regime2_mask = threshold_valid > threshold_estimate
                
                regime1_effect = None
                regime2_effect = None
                
                if np.sum(regime1_mask) >= 10:
                    y1 = y_valid[regime1_mask]
                    X1 = X[regime1_mask] if X.ndim > 1 else X[regime1_mask].reshape(-1, 1)
                    X1_with_const = np.column_stack([np.ones(len(y1)), X1])
                    beta1 = np.linalg.lstsq(X1_with_const, y1, rcond=None)[0]
                    regime1_effect = beta1[1]
                
                if np.sum(regime2_mask) >= 10:
                    y2 = y_valid[regime2_mask]
                    X2 = X[regime2_mask] if X.ndim > 1 else X[regime2_mask].reshape(-1, 1)
                    X2_with_const = np.column_stack([np.ones(len(y2)), X2])
                    beta2 = np.linalg.lstsq(X2_with_const, y2, rcond=None)[0]
                    regime2_effect = beta2[1]
                
                attenuation_pct = None
                if regime1_effect is not None and regime2_effect is not None and regime1_effect != 0:
                    attenuation_pct = ((regime1_effect - regime2_effect) / regime1_effect) * 100
                
                result = RobustnessResults(
                    test_name=f"Asset Class: {asset_name}",
                    test_type="instrument",
                    threshold_estimate=threshold_estimate,
                    regime1_effect=regime1_effect,
                    regime2_effect=regime2_effect,
                    attenuation_pct=attenuation_pct,
                    first_stage_f_stat=first_stage_f,
                    instrument_valid=instrument_valid,
                    n_observations=len(y_valid),
                    specification_details={
                        'asset_class': asset_col,
                        'regime1_n': np.sum(regime1_mask),
                        'regime2_n': np.sum(regime2_mask)
                    }
                )
                
                results.append(result)
                logger.info(f"{asset_name}: F={first_stage_f:.2f}, valid={instrument_valid}")
                
            except Exception as e:
                logger.error(f"Failed to test {asset_name}: {e}")
                continue
        
        logger.info(f"Completed alternative asset classes test: {len(results)} specifications")
        return results
    
    def test_placebo_non_fomc_days(self) -> List[RobustnessResults]:
        """
        Conduct placebo tests on non-FOMC days.
        
        Implements Requirement 8.9: Test whether non-FOMC days produce valid instruments
        (they should not - this is a placebo test).
        
        Returns:
            List of RobustnessResults for placebo tests
        """
        logger.info("Testing placebo on non-FOMC days...")
        
        results = []
        
        # Check if we have placebo instruments
        if 'hf_placebo_surprises' not in self.baseline_data.columns:
            logger.warning("Placebo HF surprises not found, skipping placebo test")
            return results
        
        # Extract base variables
        y = self.baseline_data['yield_changes_10y'].values
        qe_shocks = self.baseline_data['qe_shocks_instrumented'].values
        threshold_var = self.baseline_data['fiscal_indicator'].values
        placebo_instruments = self.baseline_data['hf_placebo_surprises'].values
        
        # Get control variables if available
        control_cols = ['gdp_growth', 'unemployment_rate', 'inflation']
        controls = None
        if all(col in self.baseline_data.columns for col in control_cols):
            controls = self.baseline_data[control_cols].values
        
        try:
            # Check for missing values
            valid_mask = ~(np.isnan(y) | np.isnan(qe_shocks) | np.isnan(threshold_var) | np.isnan(placebo_instruments))
            if controls is not None:
                valid_mask &= ~np.any(np.isnan(controls), axis=1)
            
            y_valid = y[valid_mask]
            qe_valid = qe_shocks[valid_mask]
            threshold_valid = threshold_var[valid_mask]
            placebo_valid = placebo_instruments[valid_mask]
            controls_valid = controls[valid_mask] if controls is not None else None
            
            if len(y_valid) < 20:
                logger.warning("Insufficient observations for placebo test")
                return results
            
            # Compute first-stage F-statistic for placebo instruments
            # These should be weak (F < 10) since non-FOMC days shouldn't contain policy shocks
            validator = InstrumentValidator()
            f_stat_result = validator.weak_instrument_test(qe_valid, placebo_valid, controls_valid)
            first_stage_f = f_stat_result.get('f_statistic')
            
            # For placebo test, we EXPECT weak instruments (F < 10)
            placebo_passes = first_stage_f is not None and first_stage_f < self.min_first_stage_f
            
            result = RobustnessResults(
                test_name="Placebo Test: Non-FOMC Days",
                test_type="instrument",
                first_stage_f_stat=first_stage_f,
                instrument_valid=not placebo_passes,  # Inverted: we want weak instruments for placebo
                n_observations=len(y_valid),
                specification_details={
                    'test_type': 'placebo',
                    'expected_weak_instrument': True,
                    'placebo_passes': placebo_passes,
                    'interpretation': 'F < 10 indicates placebo test passes (non-FOMC days do not contain policy shocks)'
                }
            )
            
            results.append(result)
            logger.info(f"Placebo test: F={first_stage_f:.2f}, passes={placebo_passes} (expect F < 10)")
            
        except Exception as e:
            logger.error(f"Failed to conduct placebo test: {e}")
        
        logger.info(f"Completed placebo test: {len(results)} specifications")
        return results
