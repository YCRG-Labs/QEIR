"""
Example: Publication Output Generation

This example demonstrates how to use the PublicationOutputGenerator to create
publication-ready tables and figures for the revised QE methodology analysis.

Author: QE Research Team
Date: 2025
"""

import numpy as np
import pandas as pd
from pathlib import Path

from qeir.output.publication_generator import PublicationOutputGenerator


def example_main_results_tables():
    """
    Example: Generate main results tables (Tables 2-4).
    """
    print("=" * 70)
    print("Example: Generating Main Results Tables")
    print("=" * 70)
    
    # Initialize generator
    gen = PublicationOutputGenerator(output_dir="output/publication")
    
    # Example 1: Table 2 - Threshold Regression
    print("\n1. Generating Table 2: Threshold Regression Estimates...")
    
    threshold_results = {
        'threshold_estimate': 0.285,
        'threshold_ci_lower': 0.27,
        'threshold_ci_upper': 0.30,
        'regime1_qe_effect': -9.4,
        'regime1_qe_se': 2.1,
        'regime1_qe_pvalue': 0.001,
        'regime2_qe_effect': -3.5,
        'regime2_qe_se': 1.8,
        'regime2_qe_pvalue': 0.05,
        'first_stage_f_stat': 15.3,
        'observations_total': 64,
        'regime1_observations': 32,
        'regime2_observations': 32
    }
    
    table2_path = gen.generate_table2_threshold_regression(threshold_results)
    print(f"   ✓ Table 2 saved to: {table2_path}")
    
    # Example 2: Table 3 - Regime Effects
    print("\n2. Generating Table 3: Regime-Specific Effects...")
    
    regime_results = {
        'regime1_qe_effect': -9.4,
        'regime1_qe_se': 2.1,
        'regime1_qe_pvalue': 0.001,
        'regime2_qe_effect': -3.5,
        'regime2_qe_se': 1.8,
        'regime2_qe_pvalue': 0.05,
        'attenuation_pct': 63.0,
        'attenuation_significant': True,
        'regime1_r_squared': 0.45,
        'regime2_r_squared': 0.38
    }
    
    table3_path = gen.generate_table3_regime_effects(regime_results)
    print(f"   ✓ Table 3 saved to: {table3_path}")
    
    # Example 3: Table 4 - Channel Decomposition
    print("\n3. Generating Table 4: Channel Decomposition...")
    
    channel_results = {
        'rate_channel_beta': -0.15,
        'rate_channel_se': 0.04,
        'rate_channel_pvalue': 0.001,
        'distortion_channel_beta': -0.28,
        'distortion_channel_se': 0.06,
        'distortion_channel_pvalue': 0.001,
        'distortion_share': 0.65,
        'rate_share': 0.35,
        'cumulative_effect_12q': -2.7,
        'meets_target_shares': True
    }
    
    table4_path = gen.generate_table4_channel_decomposition(channel_results)
    print(f"   ✓ Table 4 saved to: {table4_path}")
    
    print("\n" + "=" * 70)
    print("Main results tables generated successfully!")
    print("=" * 70)


def example_robustness_tables():
    """
    Example: Generate robustness tables (Tables 5-8).
    """
    print("\n" + "=" * 70)
    print("Example: Generating Robustness Tables")
    print("=" * 70)
    
    gen = PublicationOutputGenerator(output_dir="output/publication")
    
    # Example 1: Table 5 - Double Threshold
    print("\n1. Generating Table 5: Double-Threshold Model...")
    
    double_threshold_results = {
        'threshold1_estimate': 0.25,
        'threshold2_estimate': 0.32,
        'regime1_qe_effect': -10.2,
        'regime1_qe_se': 2.5,
        'regime1_qe_pvalue': 0.001,
        'regime2_qe_effect': -6.8,
        'regime2_qe_se': 2.0,
        'regime2_qe_pvalue': 0.01,
        'regime3_qe_effect': -2.9,
        'regime3_qe_se': 1.9,
        'regime3_qe_pvalue': 0.12,
        'regime1_observations': 20,
        'regime2_observations': 24,
        'regime3_observations': 20
    }
    
    table5_path = gen.generate_table5_double_threshold(double_threshold_results)
    print(f"   ✓ Table 5 saved to: {table5_path}")
    
    # Example 2: Table 6 - Alternative Fiscal Indicators
    print("\n2. Generating Table 6: Alternative Fiscal Indicators...")
    
    fiscal_results = {
        'debt_to_gdp': {
            'threshold_estimate': 0.285,
            'regime1_qe_effect': -9.4,
            'regime1_qe_se': 2.1,
            'regime2_qe_effect': -3.5,
            'regime2_qe_se': 1.8,
            'attenuation_pct': 63.0
        },
        'gross_debt': {
            'threshold_estimate': 0.92,
            'regime1_qe_effect': -9.1,
            'regime1_qe_se': 2.0,
            'regime2_qe_effect': -3.8,
            'regime2_qe_se': 1.9,
            'attenuation_pct': 58.2
        },
        'primary_deficit': {
            'threshold_estimate': 0.035,
            'regime1_qe_effect': -8.9,
            'regime1_qe_se': 2.2,
            'regime2_qe_effect': -3.9,
            'regime2_qe_se': 1.7,
            'attenuation_pct': 56.2
        }
    }
    
    table6_path = gen.generate_table6_alternative_fiscal(fiscal_results)
    print(f"   ✓ Table 6 saved to: {table6_path}")
    
    # Example 3: Table 7 - Alternative Distortion Measures
    print("\n3. Generating Table 7: Alternative Distortion Measures...")
    
    distortion_results = {
        'baseline': {
            'distortion_share': 0.65,
            'rate_share': 0.35,
            'cumulative_effect_12q': -2.7
        },
        'fails_to_deliver': {
            'distortion_share': 0.62,
            'rate_share': 0.38,
            'cumulative_effect_12q': -2.6
        },
        'repo_specialness': {
            'distortion_share': 0.68,
            'rate_share': 0.32,
            'cumulative_effect_12q': -2.8
        }
    }
    
    table7_path = gen.generate_table7_alternative_distortion(distortion_results)
    print(f"   ✓ Table 7 saved to: {table7_path}")
    
    print("\n" + "=" * 70)
    print("Robustness tables generated successfully!")
    print("=" * 70)


def example_main_figures():
    """
    Example: Generate main figures (Figures 1-6).
    """
    print("\n" + "=" * 70)
    print("Example: Generating Main Figures")
    print("=" * 70)
    
    gen = PublicationOutputGenerator(output_dir="output/publication")
    
    # Example 1: Figure 3 - SSR Minimization
    print("\n1. Generating Figure 3: SSR Minimization...")
    
    threshold_grid = np.linspace(0.2, 0.4, 100)
    ssr_values = (threshold_grid - 0.285)**2 * 1000 + 500
    threshold_estimate = 0.285
    confidence_interval = (0.27, 0.30)
    
    fig3_path = gen.generate_figure3_ssr_minimization(
        threshold_grid, ssr_values, threshold_estimate, confidence_interval
    )
    print(f"   ✓ Figure 3 saved to: {fig3_path}")
    
    # Example 2: Figure 5 - Local Projections
    print("\n2. Generating Figure 5: Local Projection IRFs...")
    
    horizons = np.arange(0, 13)
    effects = -2.5 * np.exp(-horizons / 5)
    ci_lower = effects - 0.6
    ci_upper = effects + 0.6
    
    fig5_path = gen.generate_figure5_local_projections(
        horizons, effects, ci_lower, ci_upper
    )
    print(f"   ✓ Figure 5 saved to: {fig5_path}")
    
    # Example 3: Figure 6 - Channel Decomposition
    print("\n3. Generating Figure 6: Channel Decomposition...")
    
    rate_contributions = -0.9 * np.exp(-horizons / 5)
    distortion_contributions = -1.6 * np.exp(-horizons / 5)
    
    channel_data = {
        'horizons': horizons,
        'rate_contributions': rate_contributions,
        'distortion_contributions': distortion_contributions,
        'total_effects': rate_contributions + distortion_contributions,
        'rate_share': 0.35,
        'distortion_share': 0.65
    }
    
    fig6_path = gen.generate_figure6_channel_decomposition(channel_data)
    print(f"   ✓ Figure 6 saved to: {fig6_path}")
    
    print("\n" + "=" * 70)
    print("Main figures generated successfully!")
    print("=" * 70)


def example_data_sources_table():
    """
    Example: Generate data sources table (Table 9).
    """
    print("\n" + "=" * 70)
    print("Example: Generating Data Sources Table")
    print("=" * 70)
    
    gen = PublicationOutputGenerator(output_dir="output/publication")
    
    # Define data dictionary
    data_dict = {
        '10Y Yields': {
            'description': '10-year Treasury constant maturity yield',
            'source': 'FRED (DGS10)',
            'frequency': 'Quarterly',
            'category': 'Dependent Variables'
        },
        'Investment': {
            'description': 'Real private fixed investment growth',
            'source': 'FRED (GPDIC1)',
            'frequency': 'Quarterly',
            'category': 'Dependent Variables'
        },
        'QE Shocks': {
            'description': 'Instrumented QE shocks from HF identification',
            'source': 'Authors calculation (CME futures)',
            'frequency': 'Quarterly',
            'category': 'Treatment Variables'
        },
        'Debt-to-GDP': {
            'description': 'Federal debt held by public as share of GDP',
            'source': 'FRED (GFDEGDQ188S)',
            'frequency': 'Quarterly',
            'category': 'Threshold Variables'
        },
        'Distortion Index': {
            'description': 'Three-component market distortion measure',
            'source': 'Authors calculation (Bloomberg, FRBNY, SIFMA)',
            'frequency': 'Quarterly',
            'category': 'Channel Variables'
        },
        'GDP Growth': {
            'description': 'Real GDP growth rate',
            'source': 'FRED (A191RL1Q225SBEA)',
            'frequency': 'Quarterly',
            'category': 'Control Variables'
        },
        'HF Surprises': {
            'description': 'High-frequency FOMC surprises (30-min window)',
            'source': 'Authors calculation (CME futures)',
            'frequency': 'Event-based',
            'category': 'Instruments'
        }
    }
    
    table9_path = gen.generate_table9_data_sources(data_dict)
    print(f"   ✓ Table 9 saved to: {table9_path}")
    
    print("\n" + "=" * 70)
    print("Data sources table generated successfully!")
    print("=" * 70)


def main():
    """
    Run all examples.
    """
    print("\n" + "=" * 70)
    print("PUBLICATION OUTPUT GENERATION EXAMPLES")
    print("=" * 70)
    print("\nThis script demonstrates how to generate publication-ready")
    print("tables and figures for the revised QE methodology analysis.")
    print("\nOutput will be saved to: output/publication/")
    print("=" * 70)
    
    # Generate all outputs
    example_main_results_tables()
    example_robustness_tables()
    example_main_figures()
    example_data_sources_table()
    
    print("\n" + "=" * 70)
    print("ALL EXAMPLES COMPLETED SUCCESSFULLY!")
    print("=" * 70)
    print("\nGenerated files:")
    print("  - Tables: output/publication/tables/*.tex")
    print("  - Figures: output/publication/figures/*.pdf")
    print("\nThese files are ready for inclusion in your LaTeX manuscript.")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
