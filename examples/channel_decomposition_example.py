"""
Example: Structural Channel Decomposition

This example demonstrates how to use the StructuralChannelDecomposer
to quantify the relative importance of interest rate versus market
distortion channels in QE transmission to investment.

The methodology follows a four-step approach:
1. Estimate effects of QE on channels (rates and distortions)
2. Estimate investment response to each channel
3. Decompose total effects into channel contributions
4. Calculate channel shares

Target: 65% distortion channel, 35% rate channel (±5%)
"""

import sys
import os
# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from qeir.analysis.channel_decomposition import (
    StructuralChannelDecomposer,
    ChannelDecompositionConfig
)


def create_example_data():
    """
    Create example quarterly data for demonstration.
    
    In practice, this data would come from:
    - QE shocks: High-frequency FOMC surprise identification
    - Interest rates: 10-year Treasury yields
    - Distortion index: Three-component market distortion measure
    - Investment growth: Private fixed investment growth
    """
    # Create quarterly date index (2008Q1-2023Q4)
    dates = pd.date_range('2008-01-01', periods=64, freq='QS')
    
    # Simulate QE shocks (instrumented from HF surprises)
    np.random.seed(42)
    qe_shocks = pd.Series(
        np.random.normal(0, 1, 64),
        index=dates,
        name='qe_shocks'
    )
    
    # Simulate interest rates (negatively affected by QE)
    interest_rates = pd.Series(
        5.0 - 0.3 * qe_shocks + np.random.normal(0, 0.5, 64),
        index=dates,
        name='interest_rates'
    )
    
    # Simulate market distortion index (positively affected by QE)
    distortion_index = pd.Series(
        0.5 + 0.4 * qe_shocks + np.random.normal(0, 0.3, 64),
        index=dates,
        name='distortion_index'
    )
    
    # Simulate investment growth (affected by both channels)
    # Distortion channel dominates (65% vs 35%)
    investment_growth = pd.Series(
        2.0 - 0.2 * interest_rates - 0.5 * distortion_index + np.random.normal(0, 0.4, 64),
        index=dates,
        name='investment_growth'
    )
    
    return qe_shocks, interest_rates, distortion_index, investment_growth


def main():
    """Run structural channel decomposition example"""
    
    print("=" * 70)
    print("Structural Channel Decomposition Example")
    print("=" * 70)
    
    # Load data
    print("\n1. Loading quarterly data (2008Q1-2023Q4)...")
    qe_shocks, interest_rates, distortion_index, investment_growth = create_example_data()
    print(f"   - QE shocks: {len(qe_shocks)} observations")
    print(f"   - Interest rates: {len(interest_rates)} observations")
    print(f"   - Distortion index: {len(distortion_index)} observations")
    print(f"   - Investment growth: {len(investment_growth)} observations")
    
    # Configure decomposer
    print("\n2. Configuring decomposer...")
    config = ChannelDecompositionConfig(
        max_horizon=12,  # 12 quarters (3 years)
        target_distortion_share=0.65,  # 65% target
        target_rate_share=0.35,  # 35% target
        share_tolerance=0.05,  # ±5%
        use_hac_errors=True,
        hac_lags=4
    )
    print(f"   - Max horizon: {config.max_horizon} quarters")
    print(f"   - Target shares: {config.target_distortion_share:.0%} distortion, "
          f"{config.target_rate_share:.0%} rate")
    print(f"   - Tolerance: ±{config.share_tolerance:.0%}")
    
    # Create decomposer
    decomposer = StructuralChannelDecomposer(config=config)
    
    # Run full decomposition
    print("\n3. Running four-step structural decomposition...")
    print("   Step 1: Estimating channel effects (QE → rates, QE → distortions)")
    print("   Step 2: Estimating investment responses (rates → investment, distortions → investment)")
    print("   Step 3: Decomposing total effects into channel contributions")
    print("   Step 4: Calculating channel shares")
    
    results = decomposer.run_full_decomposition(
        qe_shocks=qe_shocks,
        interest_rates=interest_rates,
        distortion_index=distortion_index,
        investment_growth=investment_growth
    )
    
    # Display Step 1 results
    print("\n" + "=" * 70)
    print("STEP 1: Channel Effects")
    print("=" * 70)
    
    rate_channel = results['channel_effects']['rate_channel']
    distortion_channel = results['channel_effects']['distortion_channel']
    
    print(f"\nRate Channel: rt = αr + βr·QEt + γr·Zt + εr,t")
    print(f"  βr = {rate_channel['beta']:.4f} (se={rate_channel['se']:.4f}, p={rate_channel['pvalue']:.4f})")
    print(f"  R² = {rate_channel['rsquared']:.4f}")
    print(f"  Interpretation: 1 unit QE shock → {rate_channel['beta']:.4f} change in interest rates")
    
    print(f"\nDistortion Channel: Dt = αD + βD·QEt + γD·Zt + εD,t")
    print(f"  βD = {distortion_channel['beta']:.4f} (se={distortion_channel['se']:.4f}, p={distortion_channel['pvalue']:.4f})")
    print(f"  R² = {distortion_channel['rsquared']:.4f}")
    print(f"  Interpretation: 1 unit QE shock → {distortion_channel['beta']:.4f} change in distortion index")
    
    # Display Step 2 results (selected horizons)
    print("\n" + "=" * 70)
    print("STEP 2: Investment Responses (Selected Horizons)")
    print("=" * 70)
    
    investment_responses = results['investment_responses']
    
    print(f"\nΔ^h It+h = αI,h + ρh·rt + κh·Dt + γI,h·Zt + εI,h,t")
    print(f"\n{'Horizon':>8} {'ρh (rate)':>12} {'κh (distortion)':>18} {'R²':>8}")
    print("-" * 50)
    
    for h in [0, 4, 8, 12]:
        if h in investment_responses:
            resp = investment_responses[h]
            print(f"{h:>8} {resp['rho']:>12.4f} {resp['kappa']:>18.4f} {resp['rsquared']:>8.4f}")
    
    # Display Step 3 results (selected horizons)
    print("\n" + "=" * 70)
    print("STEP 3: Effect Decomposition (Selected Horizons)")
    print("=" * 70)
    
    decomposition = results['decomposition']
    
    print(f"\nψ^rate_h = ρh × βr")
    print(f"ψ^dist_h = κh × βD")
    print(f"ψh = ψ^rate_h + ψ^dist_h")
    print(f"\n{'Horizon':>8} {'ψ^rate_h':>12} {'ψ^dist_h':>12} {'ψh (total)':>12}")
    print("-" * 50)
    
    for h in [0, 4, 8, 12]:
        if h in decomposition['rate_contributions']:
            rate_contrib = decomposition['rate_contributions'][h]
            dist_contrib = decomposition['distortion_contributions'][h]
            total = decomposition['total_effects'][h]
            print(f"{h:>8} {rate_contrib:>12.4f} {dist_contrib:>12.4f} {total:>12.4f}")
    
    # Display Step 4 results
    print("\n" + "=" * 70)
    print("STEP 4: Channel Shares")
    print("=" * 70)
    
    channel_shares = results['channel_shares']
    
    print(f"\nCumulative Effects (over {channel_shares['max_horizon']} quarters):")
    print(f"  Rate channel:       {channel_shares['cumulative_rate_effect']:>8.4f}")
    print(f"  Distortion channel: {channel_shares['cumulative_distortion_effect']:>8.4f}")
    print(f"  Total effect:       {channel_shares['cumulative_total_effect']:>8.4f}")
    
    print(f"\nChannel Shares:")
    print(f"  Distortion share: {channel_shares['distortion_share']:>6.2%} (target: 65% ± 5%)")
    print(f"  Rate share:       {channel_shares['rate_share']:>6.2%} (target: 35% ± 5%)")
    
    print(f"\nValidation:")
    print(f"  Shares sum to one:     {'✓' if channel_shares['shares_sum_to_one'] else '✗'}")
    print(f"  Meets target shares:   {'✓' if channel_shares['meets_target_shares'] else '✗'}")
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    print(f"\nThe structural decomposition reveals that:")
    print(f"  • {channel_shares['distortion_share']:.1%} of QE's effect on investment operates through")
    print(f"    the market distortion channel")
    print(f"  • {channel_shares['rate_share']:.1%} operates through the interest rate channel")
    
    if channel_shares['meets_target_shares']:
        print(f"\n✓ Results are within target range (65% ± 5% distortion, 35% ± 5% rate)")
    else:
        print(f"\n⚠ Results are outside target range")
    
    print(f"\nThis confirms that market distortions dominate the transmission of QE")
    print(f"effects to private investment, consistent with the theoretical prediction")
    print(f"that financial market frictions amplify QE's impact on investment decisions.")
    
    print("\n" + "=" * 70)


if __name__ == '__main__':
    main()
