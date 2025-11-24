"""
Example: Target Estimate Validation

This example demonstrates how to use the TargetValidator to validate
estimated values against target specifications from Requirements 10.1-10.7.

The validator checks:
1. Threshold estimates (τ̂ = 0.285 ± 0.015)
2. Regime-specific effects (low: -9.4 bps, high: -3.5 bps)
3. Investment effects (-2.7 pp over 12 quarters)
4. Channel shares (distortion: 65%, rate: 35%)
"""

from qeir.validation.target_validator import TargetValidator


def example_threshold_validation():
    """Example: Validate threshold estimates."""
    print("=" * 80)
    print("EXAMPLE 1: Threshold Estimate Validation")
    print("=" * 80)
    
    validator = TargetValidator(verbose=True)
    
    # Simulate threshold regression results
    threshold_estimate = 0.285
    ci_lower = 0.27
    ci_upper = 0.30
    
    print(f"\nEstimated threshold: {threshold_estimate:.3f}")
    print(f"95% CI: [{ci_lower:.3f}, {ci_upper:.3f}]")
    print("\nValidating against targets...")
    print()
    
    report = validator.validate_threshold_estimate(
        threshold_estimate=threshold_estimate,
        ci_lower=ci_lower,
        ci_upper=ci_upper
    )
    
    print(f"\nValidation Result: {'PASS' if report.all_passed else 'FAIL'}")
    print(f"Checks passed: {report.n_passed}/{len(report.results)}")


def example_regime_effects_validation():
    """Example: Validate regime-specific effects."""
    print("\n" + "=" * 80)
    print("EXAMPLE 2: Regime Effects Validation")
    print("=" * 80)
    
    validator = TargetValidator(verbose=True)
    
    # Simulate threshold regression results
    low_regime_effect = -9.4  # basis points
    high_regime_effect = -3.5  # basis points
    attenuation_pct = 63.0  # percentage
    
    print(f"\nLow regime effect: {low_regime_effect:.1f} bps")
    print(f"High regime effect: {high_regime_effect:.1f} bps")
    print(f"Attenuation: {attenuation_pct:.1f}%")
    print("\nValidating against targets...")
    print()
    
    report = validator.validate_regime_effects(
        low_regime_effect=low_regime_effect,
        high_regime_effect=high_regime_effect,
        attenuation_pct=attenuation_pct
    )
    
    print(f"\nValidation Result: {'PASS' if report.all_passed else 'FAIL'}")
    print(f"Checks passed: {report.n_passed}/{len(report.results)}")


def example_investment_effects_validation():
    """Example: Validate investment effects."""
    print("\n" + "=" * 80)
    print("EXAMPLE 3: Investment Effects Validation")
    print("=" * 80)
    
    validator = TargetValidator(verbose=True)
    
    # Simulate local projections results
    horizon_effects = {
        0: -0.1, 1: -0.2, 2: -0.3, 3: -0.3,
        4: -0.3, 5: -0.3, 6: -0.2, 7: -0.2,
        8: -0.2, 9: -0.2, 10: -0.2, 11: -0.2, 12: -0.2
    }
    cumulative_effect = sum(horizon_effects.values())
    
    print(f"\nCumulative effect over 12 quarters: {cumulative_effect:.2f} pp")
    print("\nValidating against targets...")
    print()
    
    report = validator.validate_investment_effects(
        cumulative_effect=cumulative_effect,
        horizon=12,
        horizon_effects=horizon_effects
    )
    
    print(f"\nValidation Result: {'PASS' if report.all_passed else 'FAIL'}")
    print(f"Checks passed: {report.n_passed}/{len(report.results)}")


def example_channel_shares_validation():
    """Example: Validate channel shares."""
    print("\n" + "=" * 80)
    print("EXAMPLE 4: Channel Shares Validation")
    print("=" * 80)
    
    validator = TargetValidator(verbose=True)
    
    # Simulate channel decomposition results
    distortion_share = 0.65
    rate_share = 0.35
    
    # Detailed contributions by horizon
    rate_contributions = {h: -0.1 for h in range(13)}  # Sum = -1.3
    distortion_contributions = {h: -0.2 for h in range(13)}  # Sum = -2.6
    total_effects = {h: -0.3 for h in range(13)}  # Sum = -3.9
    
    print(f"\nDistortion channel share: {distortion_share:.1%}")
    print(f"Rate channel share: {rate_share:.1%}")
    print(f"Total: {distortion_share + rate_share:.1%}")
    print("\nValidating against targets...")
    print()
    
    report = validator.validate_channel_shares(
        distortion_share=distortion_share,
        rate_share=rate_share,
        rate_contributions=rate_contributions,
        distortion_contributions=distortion_contributions,
        total_effects=total_effects
    )
    
    print(f"\nValidation Result: {'PASS' if report.all_passed else 'FAIL'}")
    print(f"Checks passed: {report.n_passed}/{len(report.results)}")


def example_out_of_tolerance():
    """Example: Validation with out-of-tolerance estimates."""
    print("\n" + "=" * 80)
    print("EXAMPLE 5: Out-of-Tolerance Estimates")
    print("=" * 80)
    
    validator = TargetValidator(verbose=True)
    
    # Simulate results that are outside tolerance
    threshold_estimate = 0.350  # Too high
    ci_lower = 0.32
    ci_upper = 0.38
    
    print(f"\nEstimated threshold: {threshold_estimate:.3f} (target: 0.285)")
    print(f"95% CI: [{ci_lower:.3f}, {ci_upper:.3f}]")
    print("\nValidating against targets...")
    print()
    
    report = validator.validate_threshold_estimate(
        threshold_estimate=threshold_estimate,
        ci_lower=ci_lower,
        ci_upper=ci_upper
    )
    
    print(f"\nValidation Result: {'PASS' if report.all_passed else 'FAIL'}")
    print(f"Checks passed: {report.n_passed}/{len(report.results)}")
    print(f"Checks failed: {report.n_failed}/{len(report.results)}")


def example_comprehensive_validation():
    """Example: Comprehensive validation of all estimates."""
    print("\n" + "=" * 80)
    print("EXAMPLE 6: Comprehensive Validation")
    print("=" * 80)
    
    validator = TargetValidator(verbose=False)
    
    # Simulate complete analysis results
    results = {
        'threshold': {
            'threshold_estimate': 0.285,
            'ci_lower': 0.27,
            'ci_upper': 0.30
        },
        'regime_effects': {
            'low_regime_effect': -9.4,
            'high_regime_effect': -3.5,
            'attenuation_pct': 63.0
        },
        'investment': {
            'cumulative_effect': -2.7
        },
        'channels': {
            'distortion_share': 0.65,
            'rate_share': 0.35
        }
    }
    
    print("\nRunning comprehensive validation...")
    
    # Validate all components
    reports = []
    
    print("\n1. Threshold estimates...")
    report1 = validator.validate_threshold_estimate(**results['threshold'])
    reports.append(('Threshold', report1))
    
    print("2. Regime effects...")
    report2 = validator.validate_regime_effects(**results['regime_effects'])
    reports.append(('Regime Effects', report2))
    
    print("3. Investment effects...")
    report3 = validator.validate_investment_effects(**results['investment'])
    reports.append(('Investment', report3))
    
    print("4. Channel shares...")
    report4 = validator.validate_channel_shares(**results['channels'])
    reports.append(('Channel Shares', report4))
    
    # Summary
    print("\n" + "=" * 80)
    print("COMPREHENSIVE VALIDATION SUMMARY")
    print("=" * 80)
    
    total_checks = sum(len(r.results) for _, r in reports)
    total_passed = sum(r.n_passed for _, r in reports)
    total_failed = sum(r.n_failed for _, r in reports)
    all_passed = all(r.all_passed for _, r in reports)
    
    for name, report in reports:
        status = "[PASS]" if report.all_passed else "[FAIL]"
        print(f"{status} | {name}: {report.n_passed}/{len(report.results)} checks passed")
    
    print("=" * 80)
    print(f"Overall: {total_passed}/{total_checks} checks passed")
    print(f"Status: {'ALL VALIDATIONS PASSED' if all_passed else 'SOME VALIDATIONS FAILED'}")
    print("=" * 80)


if __name__ == "__main__":
    # Run all examples
    example_threshold_validation()
    example_regime_effects_validation()
    example_investment_effects_validation()
    example_channel_shares_validation()
    example_out_of_tolerance()
    example_comprehensive_validation()
    
    print("\n" + "=" * 80)
    print("Examples complete!")
    print("=" * 80)
