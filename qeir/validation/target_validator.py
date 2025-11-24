"""
Target Estimate Validator

This module provides validation methods to compare estimated values against
target values specified in the requirements document. It checks whether
estimates fall within acceptable tolerance ranges and generates detailed
reports for values outside tolerances.

Requirements: 10.1-10.7
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import logging

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Result of a single validation check."""
    
    name: str
    estimated_value: float
    target_value: float
    tolerance: float
    within_tolerance: bool
    deviation_pct: float
    message: str


@dataclass
class ValidationReport:
    """Comprehensive validation report."""
    
    results: List[ValidationResult]
    all_passed: bool
    n_passed: int
    n_failed: int
    
    def __str__(self) -> str:
        """Generate human-readable report."""
        lines = [
            "=" * 80,
            "TARGET ESTIMATE VALIDATION REPORT",
            "=" * 80,
            f"Total Checks: {len(self.results)}",
            f"Passed: {self.n_passed}",
            f"Failed: {self.n_failed}",
            f"Overall Status: {'PASS' if self.all_passed else 'FAIL'}",
            "=" * 80,
            ""
        ]
        
        for result in self.results:
            status = "[PASS]" if result.within_tolerance else "[FAIL]"
            lines.append(f"{status} | {result.name}")
            lines.append(f"  Estimated: {result.estimated_value:.6f}")
            lines.append(f"  Target:    {result.target_value:.6f} ± {result.tolerance:.6f}")
            lines.append(f"  Deviation: {result.deviation_pct:.2f}%")
            if not result.within_tolerance:
                lines.append(f"  WARNING: {result.message}")
            lines.append("")
        
        lines.append("=" * 80)
        return "\n".join(lines)


class TargetValidator:
    """
    Validates estimated values against target specifications.
    
    This class provides methods to validate:
    - Threshold estimates
    - Regime-specific effects
    - Investment effects
    - Channel shares
    
    Each validation method checks whether estimates fall within
    acceptable tolerance ranges and generates detailed reports.
    """
    
    def __init__(self, verbose: bool = True):
        """
        Initialize validator.
        
        Args:
            verbose: If True, log validation results
        """
        self.verbose = verbose
        self.validation_results: List[ValidationResult] = []
    
    def _check_value(
        self,
        name: str,
        estimated: float,
        target: float,
        tolerance: float
    ) -> ValidationResult:
        """
        Check if estimated value is within tolerance of target.
        
        Args:
            name: Name of the estimate being validated
            estimated: Estimated value
            target: Target value
            tolerance: Acceptable tolerance (absolute)
            
        Returns:
            ValidationResult with check details
        """
        deviation = abs(estimated - target)
        within_tolerance = deviation <= tolerance
        
        # Calculate percentage deviation
        if target != 0:
            deviation_pct = (deviation / abs(target)) * 100
        else:
            deviation_pct = float('inf') if deviation > 0 else 0.0
        
        message = ""
        if not within_tolerance:
            message = (
                f"Estimate deviates by {deviation:.6f} "
                f"(tolerance: ±{tolerance:.6f})"
            )
        
        result = ValidationResult(
            name=name,
            estimated_value=estimated,
            target_value=target,
            tolerance=tolerance,
            within_tolerance=within_tolerance,
            deviation_pct=deviation_pct,
            message=message
        )
        
        if self.verbose:
            if within_tolerance:
                logger.info(f"[PASS] {name}: {estimated:.6f} (target: {target:.6f})")
            else:
                logger.warning(f"[FAIL] {name}: {estimated:.6f} (target: {target:.6f}, tolerance: ±{tolerance:.6f})")
                logger.warning(f"  {message}")
        
        return result
    
    def _check_ci_contains_target(
        self,
        name: str,
        ci_lower: float,
        ci_upper: float,
        target: float
    ) -> ValidationResult:
        """
        Check if confidence interval contains target value.
        
        Args:
            name: Name of the estimate
            ci_lower: Lower bound of confidence interval
            ci_upper: Upper bound of confidence interval
            target: Target value
            
        Returns:
            ValidationResult with check details
        """
        contains_target = ci_lower <= target <= ci_upper
        
        # Use CI width as a proxy for tolerance
        ci_width = ci_upper - ci_lower
        ci_center = (ci_lower + ci_upper) / 2
        deviation = abs(ci_center - target)
        
        if target != 0:
            deviation_pct = (deviation / abs(target)) * 100
        else:
            deviation_pct = float('inf') if deviation > 0 else 0.0
        
        message = ""
        if not contains_target:
            message = (
                f"CI [{ci_lower:.6f}, {ci_upper:.6f}] does not contain "
                f"target {target:.6f}"
            )
        
        result = ValidationResult(
            name=name,
            estimated_value=ci_center,
            target_value=target,
            tolerance=ci_width / 2,
            within_tolerance=contains_target,
            deviation_pct=deviation_pct,
            message=message
        )
        
        if self.verbose:
            if contains_target:
                logger.info(f"[PASS] {name} CI contains target: [{ci_lower:.6f}, {ci_upper:.6f}]")
            else:
                logger.warning(f"[FAIL] {name} CI does not contain target {target:.6f}")
                logger.warning(f"  CI: [{ci_lower:.6f}, {ci_upper:.6f}]")
        
        return result
    
    def generate_report(self) -> ValidationReport:
        """
        Generate comprehensive validation report.
        
        Returns:
            ValidationReport summarizing all checks
        """
        n_passed = sum(1 for r in self.validation_results if r.within_tolerance)
        n_failed = len(self.validation_results) - n_passed
        all_passed = n_failed == 0
        
        report = ValidationReport(
            results=self.validation_results.copy(),
            all_passed=all_passed,
            n_passed=n_passed,
            n_failed=n_failed
        )
        
        if self.verbose:
            print(report)
        
        return report
    
    def reset(self):
        """Clear all validation results."""
        self.validation_results.clear()

    def validate_threshold_estimate(
        self,
        threshold_estimate: float,
        ci_lower: float,
        ci_upper: float,
        target_threshold: float = 0.285,
        tolerance: float = 0.015,
        target_ci_lower: float = 0.27,
        target_ci_upper: float = 0.30
    ) -> ValidationReport:
        """
        Validate threshold estimate against targets.
        
        Requirements: 10.1
        Target: τ̂ = 0.285 ± 0.015
        Target CI: [0.27, 0.30]
        
        Args:
            threshold_estimate: Estimated threshold value
            ci_lower: Lower bound of 95% confidence interval
            ci_upper: Upper bound of 95% confidence interval
            target_threshold: Target threshold value (default: 0.285)
            tolerance: Acceptable tolerance (default: 0.015)
            target_ci_lower: Target CI lower bound (default: 0.27)
            target_ci_upper: Target CI upper bound (default: 0.30)
            
        Returns:
            ValidationReport with threshold validation results
        """
        self.reset()
        
        # Check 1: Point estimate within tolerance
        result1 = self._check_value(
            name="Threshold Point Estimate",
            estimated=threshold_estimate,
            target=target_threshold,
            tolerance=tolerance
        )
        self.validation_results.append(result1)
        
        # Check 2: CI contains target
        result2 = self._check_ci_contains_target(
            name="Threshold CI Contains Target",
            ci_lower=ci_lower,
            ci_upper=ci_upper,
            target=target_threshold
        )
        self.validation_results.append(result2)
        
        # Check 3: CI lower bound within range
        result3 = self._check_value(
            name="Threshold CI Lower Bound",
            estimated=ci_lower,
            target=target_ci_lower,
            tolerance=0.02  # Allow some flexibility
        )
        self.validation_results.append(result3)
        
        # Check 4: CI upper bound within range
        result4 = self._check_value(
            name="Threshold CI Upper Bound",
            estimated=ci_upper,
            target=target_ci_upper,
            tolerance=0.02  # Allow some flexibility
        )
        self.validation_results.append(result4)
        
        return self.generate_report()

    def validate_regime_effects(
        self,
        low_regime_effect: float,
        high_regime_effect: float,
        attenuation_pct: float,
        target_low_regime: float = -9.4,
        tolerance_low_regime: float = 1.0,
        target_high_regime: float = -3.5,
        tolerance_high_regime: float = 0.5,
        target_attenuation: float = 63.0,
        tolerance_attenuation: float = 5.0
    ) -> ValidationReport:
        """
        Validate regime-specific effects against targets.
        
        Requirements: 10.2, 10.3, 10.4
        Targets:
        - Low regime: -9.4 bps ± 1.0 bps
        - High regime: -3.5 bps ± 0.5 bps
        - Attenuation: 63% ± 5%
        
        Args:
            low_regime_effect: QE effect in low-fiscal regime (basis points)
            high_regime_effect: QE effect in high-fiscal regime (basis points)
            attenuation_pct: Attenuation percentage: (β1 - β2) / β1 × 100
            target_low_regime: Target low regime effect (default: -9.4)
            tolerance_low_regime: Tolerance for low regime (default: 1.0)
            target_high_regime: Target high regime effect (default: -3.5)
            tolerance_high_regime: Tolerance for high regime (default: 0.5)
            target_attenuation: Target attenuation percentage (default: 63.0)
            tolerance_attenuation: Tolerance for attenuation (default: 5.0)
            
        Returns:
            ValidationReport with regime effect validation results
        """
        self.reset()
        
        # Check 1: Low regime effect
        result1 = self._check_value(
            name="Low Regime QE Effect (bps)",
            estimated=low_regime_effect,
            target=target_low_regime,
            tolerance=tolerance_low_regime
        )
        self.validation_results.append(result1)
        
        # Check 2: High regime effect
        result2 = self._check_value(
            name="High Regime QE Effect (bps)",
            estimated=high_regime_effect,
            target=target_high_regime,
            tolerance=tolerance_high_regime
        )
        self.validation_results.append(result2)
        
        # Check 3: Attenuation percentage
        result3 = self._check_value(
            name="Attenuation Percentage (%)",
            estimated=attenuation_pct,
            target=target_attenuation,
            tolerance=tolerance_attenuation
        )
        self.validation_results.append(result3)
        
        # Check 4: Verify attenuation calculation is consistent
        if low_regime_effect != 0:
            calculated_attenuation = ((low_regime_effect - high_regime_effect) / low_regime_effect) * 100
            result4 = self._check_value(
                name="Attenuation Calculation Consistency",
                estimated=attenuation_pct,
                target=calculated_attenuation,
                tolerance=0.5  # Allow some rounding tolerance
            )
            self.validation_results.append(result4)
        
        return self.generate_report()

    def validate_investment_effects(
        self,
        cumulative_effect: float,
        horizon: int = 12,
        target_cumulative: float = -2.7,
        tolerance: float = 0.3,
        horizon_effects: Optional[Dict[int, float]] = None
    ) -> ValidationReport:
        """
        Validate investment effects against targets.
        
        Requirements: 10.5
        Target: Cumulative effect of -2.7 pp ± 0.3 pp over 12 quarters
        
        Args:
            cumulative_effect: Cumulative investment effect (percentage points)
            horizon: Number of quarters for cumulative effect (default: 12)
            target_cumulative: Target cumulative effect (default: -2.7)
            tolerance: Acceptable tolerance (default: 0.3)
            horizon_effects: Optional dict of effects by horizon for detailed logging
            
        Returns:
            ValidationReport with investment effect validation results
        """
        self.reset()
        
        # Check 1: Cumulative effect over specified horizon
        result1 = self._check_value(
            name=f"Cumulative Investment Effect ({horizon}Q, pp)",
            estimated=cumulative_effect,
            target=target_cumulative,
            tolerance=tolerance
        )
        self.validation_results.append(result1)
        
        # Check 2: If horizon effects provided, verify cumulative calculation
        if horizon_effects is not None:
            # Calculate cumulative from horizon effects
            relevant_horizons = [h for h in range(horizon + 1) if h in horizon_effects]
            calculated_cumulative = sum(horizon_effects[h] for h in relevant_horizons)
            
            result2 = self._check_value(
                name="Cumulative Effect Calculation Consistency",
                estimated=cumulative_effect,
                target=calculated_cumulative,
                tolerance=0.01  # Very tight tolerance for internal consistency
            )
            self.validation_results.append(result2)
            
            # Log detailed horizon effects if verbose
            if self.verbose:
                logger.info(f"Horizon-by-horizon effects:")
                for h in sorted(relevant_horizons):
                    logger.info(f"  h={h}: {horizon_effects[h]:.4f} pp")
        
        return self.generate_report()

    def validate_channel_shares(
        self,
        distortion_share: float,
        rate_share: float,
        target_distortion_share: float = 0.65,
        tolerance_distortion: float = 0.05,
        target_rate_share: float = 0.35,
        tolerance_rate: float = 0.05,
        rate_contributions: Optional[Dict[int, float]] = None,
        distortion_contributions: Optional[Dict[int, float]] = None,
        total_effects: Optional[Dict[int, float]] = None
    ) -> ValidationReport:
        """
        Validate channel shares against targets.
        
        Requirements: 10.6, 10.7
        Targets:
        - Distortion share: 65% ± 5%
        - Rate share: 35% ± 5%
        - Shares must sum to 100%
        
        Args:
            distortion_share: Distortion channel share (0-1 scale)
            rate_share: Rate channel share (0-1 scale)
            target_distortion_share: Target distortion share (default: 0.65)
            tolerance_distortion: Tolerance for distortion share (default: 0.05)
            target_rate_share: Target rate share (default: 0.35)
            tolerance_rate: Tolerance for rate share (default: 0.05)
            rate_contributions: Optional dict of rate contributions by horizon
            distortion_contributions: Optional dict of distortion contributions by horizon
            total_effects: Optional dict of total effects by horizon
            
        Returns:
            ValidationReport with channel share validation results
        """
        self.reset()
        
        # Check 1: Distortion share
        result1 = self._check_value(
            name="Distortion Channel Share",
            estimated=distortion_share,
            target=target_distortion_share,
            tolerance=tolerance_distortion
        )
        self.validation_results.append(result1)
        
        # Check 2: Rate share
        result2 = self._check_value(
            name="Rate Channel Share",
            estimated=rate_share,
            target=target_rate_share,
            tolerance=tolerance_rate
        )
        self.validation_results.append(result2)
        
        # Check 3: Shares sum to 1.0
        shares_sum = distortion_share + rate_share
        result3 = self._check_value(
            name="Channel Shares Sum to 1.0",
            estimated=shares_sum,
            target=1.0,
            tolerance=0.01  # Tight tolerance for sum
        )
        self.validation_results.append(result3)
        
        # Check 4: Distortion dominates (distortion > rate)
        distortion_dominates = distortion_share > rate_share
        
        result4 = ValidationResult(
            name="Distortion Channel Dominates",
            estimated_value=distortion_share,
            target_value=rate_share,
            tolerance=0.0,
            within_tolerance=distortion_dominates,
            deviation_pct=0.0,
            message="" if distortion_dominates else "Distortion share should exceed rate share"
        )
        self.validation_results.append(result4)
        
        if self.verbose:
            if distortion_dominates:
                logger.info(f"[PASS] Distortion channel dominates: {distortion_share:.2%} > {rate_share:.2%}")
            else:
                logger.warning(f"[FAIL] Distortion channel does not dominate: {distortion_share:.2%} <= {rate_share:.2%}")
        
        # Check 5: If detailed contributions provided, verify share calculation
        if (rate_contributions is not None and 
            distortion_contributions is not None and 
            total_effects is not None):
            
            # Calculate shares from contributions
            horizons = sorted(set(rate_contributions.keys()) & 
                            set(distortion_contributions.keys()) & 
                            set(total_effects.keys()))
            
            cumulative_rate = sum(rate_contributions[h] for h in horizons)
            cumulative_distortion = sum(distortion_contributions[h] for h in horizons)
            cumulative_total = sum(total_effects[h] for h in horizons)
            
            if cumulative_total != 0:
                calculated_distortion_share = cumulative_distortion / cumulative_total
                calculated_rate_share = cumulative_rate / cumulative_total
                
                result5 = self._check_value(
                    name="Distortion Share Calculation Consistency",
                    estimated=distortion_share,
                    target=calculated_distortion_share,
                    tolerance=0.01
                )
                self.validation_results.append(result5)
                
                result6 = self._check_value(
                    name="Rate Share Calculation Consistency",
                    estimated=rate_share,
                    target=calculated_rate_share,
                    tolerance=0.01
                )
                self.validation_results.append(result6)
                
                # Log detailed decomposition if verbose
                if self.verbose:
                    logger.info(f"Channel decomposition details:")
                    logger.info(f"  Cumulative rate contribution: {cumulative_rate:.4f}")
                    logger.info(f"  Cumulative distortion contribution: {cumulative_distortion:.4f}")
                    logger.info(f"  Cumulative total effect: {cumulative_total:.4f}")
        
        return self.generate_report()
