"""
Example usage of the RobustnessTestSuite for QE methodology revision.

This script demonstrates how to use the robustness testing suite to validate
the main results across different specifications, sample splits, and instruments.
"""

import numpy as np
import pandas as pd
from qeir.robustness.robustness_suite import RobustnessTestSuite, RobustnessResults


def create_example_data():
    """Create example quarterly data for demonstration."""
    np.random.seed(42)
    
    # Create quarterly date range (2008Q1-2023Q4)
    dates = pd.date_range('2008-01-01', '2023-12-31', freq='Q')
    n_obs = len(dates)
    
    # Simulate data with realistic patterns
    fiscal_indicator = np.random.uniform(0.2, 0.4, n_obs)
    qe_shocks = np.random.randn(n_obs) * 0.5
    
    # Create regime-dependent yield changes
    regime_mask = fiscal_indicator > 0.285
    yield_changes = np.where(
        regime_mask,
        -0.035 * qe_shocks + np.random.randn(n_obs) * 0.05,  # High fiscal regime
        -0.094 * qe_shocks + np.random.randn(n_obs) * 0.05   # Low fiscal regime
    )
    
    data = pd.DataFrame({
        'yield_changes_10y': yield_changes,
        'qe_shocks_instrumented': qe_shocks,
        'fiscal_indicator': fiscal_indicator,
        'gdp_growth': np.random.randn(n_obs) * 0.02 + 0.02,
        'unemployment_rate': np.random.uniform(4, 10, n_obs),
        'inflation': np.random.randn(n_obs) * 0.01 + 0.02,
        'hf_qe_surprises': qe_shocks + np.random.randn(n_obs) * 0.1,
        'distortion_index': np.random.randn(n_obs) * 0.5,
        
        # Alternative fiscal indicators
        'gross_debt_gdp': np.random.uniform(0.5, 1.0, n_obs),
        'primary_deficit_gdp': np.random.uniform(-0.05, 0.05, n_obs),
        
        # Alternative distortion measures
        'fails_to_deliver': np.random.randn(n_obs) * 0.5,
        'repo_specialness': np.random.randn(n_obs) * 0.3,
        
        # Shadow rate
        'wu_xia_shadow_rate': np.random.uniform(-3, 2, n_obs),
    }, index=dates)
    
    return data


def main():
    """Run robustness testing suite example."""
    print("=" * 80)
    print("QE Methodology Revision - Robustness Testing Suite Example")
    print("=" * 80)
    
    # Create example data
    print("\n1. Creating example quarterly data (2008Q1-2023Q4)...")
    data = create_example_data()
    print(f"   Data shape: {data.shape}")
    print(f"   Date range: {data.index[0]} to {data.index[-1]}")
    
    # Define baseline results for comparison
    print("\n2. Defining baseline results...")
    baseline_results = RobustnessResults(
        test_name="Baseline Specification",
        test_type="threshold",
        threshold_estimate=0.285,
        regime1_effect=-0.094,
        regime2_effect=-0.035,
        attenuation_pct=63.0,
        first_stage_f_stat=15.2,
        instrument_valid=True,
        n_observations=64
    )
    print(f"   Baseline threshold: {baseline_results.threshold_estimate}")
    print(f"   Baseline attenuation: {baseline_results.attenuation_pct}%")
    
    # Initialize robustness test suite
    print("\n3. Initializing RobustnessTestSuite...")
    suite = RobustnessTestSuite(
        baseline_data=data,
        baseline_results=baseline_results,
        min_first_stage_f=10.0
    )
    
    # Run individual tests
    print("\n4. Running individual robustness tests...")
    
    print("\n   a) Double-threshold model (3 regimes)...")
    result_double = suite.test_double_threshold()
    if result_double.threshold_estimate is not None:
        print(f"      tau1 = {result_double.specification_details.get('threshold1', 'N/A'):.3f}")
        print(f"      tau2 = {result_double.specification_details.get('threshold2', 'N/A'):.3f}")
        print(f"      Regime effects: beta1={result_double.regime1_effect:.4f}, "
              f"beta2={result_double.regime2_effect:.4f}, beta3={result_double.regime3_effect:.4f}")
    
    print("\n   b) Smooth transition regression...")
    result_str = suite.test_smooth_transition()
    if result_str.transition_location is not None:
        print(f"      Transition location: c = {result_str.transition_location:.3f}")
        print(f"      Transition speed: gamma = {result_str.transition_speed:.3f}")
    
    print("\n   c) Alternative fiscal indicators...")
    results_fiscal = suite.test_alternative_fiscal_indicators()
    print(f"      Tested {len(results_fiscal)} alternative fiscal indicators")
    for result in results_fiscal:
        print(f"      - {result.test_name}: tau = {result.threshold_estimate:.3f}")
    
    print("\n   d) Alternative distortion measures...")
    results_distortion = suite.test_alternative_distortion_measures()
    print(f"      Tested {len(results_distortion)} alternative distortion measures")
    
    print("\n   e) QE episode subsamples...")
    results_episodes = suite.test_qe_episode_subsamples()
    print(f"      Tested {len(results_episodes)} QE episodes")
    for result in results_episodes:
        print(f"      - {result.test_name}: n = {result.n_observations}")
    
    print("\n   f) Shadow rate conditioning...")
    results_shadow = suite.test_shadow_rate_conditioning()
    print(f"      Tested {len(results_shadow)} shadow rate regimes")
    
    print("\n   g) Alternative HF windows...")
    results_windows = suite.test_alternative_hf_windows()
    print(f"      Tested {len(results_windows)} HF window sizes")
    
    # Generate summary table
    print("\n5. Generating summary table...")
    summary = suite.get_summary_table()
    print(f"\n{summary.to_string()}")
    
    # Check consistency with baseline
    print("\n6. Checking consistency with baseline...")
    consistency = suite.check_consistency(
        tolerance_threshold=0.05,
        tolerance_effect=0.3
    )
    print(f"   Consistent specifications: {consistency['consistent_count']}/{consistency['total_count']}")
    print(f"   Consistency rate: {consistency['consistency_rate']:.1%}")
    
    # Run all tests at once
    print("\n7. Running all tests together...")
    all_results = suite.run_all_tests()
    
    print("\n   Results by category:")
    for category, results in all_results.items():
        print(f"   - {category}: {len(results)} specifications")
    
    print(f"\n   Total specifications tested: {len(suite.results)}")
    
    # Final summary
    print("\n" + "=" * 80)
    print("Robustness Testing Complete!")
    print("=" * 80)
    print(f"\nKey findings:")
    print(f"- Total robustness checks: {len(suite.results)}")
    print(f"- Specifications consistent with baseline: {consistency['consistent_count']}")
    print(f"- Consistency rate: {consistency['consistency_rate']:.1%}")
    
    # Check instrument validity across all tests
    valid_instruments = sum(1 for r in suite.results if r.instrument_valid)
    total_with_instruments = sum(1 for r in suite.results if r.first_stage_f_stat is not None)
    if total_with_instruments > 0:
        print(f"- Valid instruments (F > 10): {valid_instruments}/{total_with_instruments}")
    
    print("\n[OK] Robustness testing suite example completed successfully!")


if __name__ == '__main__':
    main()
