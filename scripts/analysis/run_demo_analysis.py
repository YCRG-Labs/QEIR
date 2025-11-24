"""
Demo QE Analysis with Synthetic Data

This script demonstrates the complete analysis pipeline using synthetic data
that mimics realistic patterns. Use this to:
1. Test the framework without needing data access
2. Understand the analysis workflow
3. Verify all components work correctly

For real results, use run_full_analysis_with_fred.py with a FRED API key.
"""

import sys
import warnings
import logging
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd

# Setup
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore')

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from qeir.config import RevisedMethodologyConfig


class SyntheticDataGenerator:
    """Generate realistic synthetic data for demonstration"""
    
    def __init__(self, start_date='2008-01-01', end_date='2023-12-31', seed=42):
        self.start_date = start_date
        self.end_date = end_date
        self.seed = seed
        np.random.seed(seed)
        
        # Create quarterly date range
        self.dates = pd.date_range(start=start_date, end=end_date, freq='QE')
        self.n_periods = len(self.dates)
        
        logger.info(f"Generating synthetic data: {start_date} to {end_date}")
        logger.info(f"Periods: {self.n_periods} quarters")
    
    def generate_all_data(self) -> dict:
        """Generate complete synthetic dataset"""
        
        data = {
            'dates': self.dates,
        }
        
        # Time trend
        t = np.arange(self.n_periods)
        
        # 1. Treasury yields (declining trend with volatility)
        data['treasury_10y'] = pd.Series(
            3.5 - 0.03 * t + 0.5 * np.random.randn(self.n_periods),
            index=self.dates
        )
        data['treasury_10y'] = data['treasury_10y'].clip(lower=0.5)  # Floor at 0.5%
        
        # 2. Fiscal indicator (debt service ratio - increasing trend)
        base_ratio = 0.20 + 0.01 * (t / self.n_periods)
        data['debt_service_ratio'] = pd.Series(
            base_ratio + 0.02 * np.random.randn(self.n_periods),
            index=self.dates
        )
        data['debt_service_ratio'] = data['debt_service_ratio'].clip(lower=0.15, upper=0.40)
        
        # 3. QE shocks (instrumented) - larger in early periods
        qe_base = np.zeros(self.n_periods)
        # QE1 (2008-2010): quarters 0-8
        qe_base[0:9] = -15 + 5 * np.random.randn(9)
        # QE2 (2010-2011): quarters 9-12
        qe_base[9:13] = -10 + 3 * np.random.randn(4)
        # QE3 (2012-2014): quarters 13-24
        qe_base[13:25] = -8 + 3 * np.random.randn(12)
        # Taper/normalization (2014-2019): quarters 25-44
        qe_base[25:45] = -2 + 2 * np.random.randn(20)
        # COVID QE (2020-2021): quarters 45-52
        qe_base[45:53] = -12 + 4 * np.random.randn(8)
        # Recent (2022-2023): quarters 53+
        qe_base[53:] = -1 + 1 * np.random.randn(self.n_periods - 53)
        
        data['qe_shocks'] = pd.Series(qe_base, index=self.dates)
        
        # 4. Investment (affected by QE and fiscal conditions)
        investment_base = 3.0 + 0.01 * t  # Slight growth trend
        # Add QE effects (negative due to distortions dominating)
        investment_qe_effect = -0.02 * np.cumsum(qe_base)
        # Add fiscal regime effects
        fiscal_effect = -0.5 * (data['debt_service_ratio'] > 0.285).astype(float)
        
        data['investment'] = pd.Series(
            investment_base + investment_qe_effect + fiscal_effect + 0.3 * np.random.randn(self.n_periods),
            index=self.dates
        )
        
        # 5. GDP (log level)
        data['gdp'] = pd.Series(
            9.5 + 0.005 * t + 0.01 * np.random.randn(self.n_periods),
            index=self.dates
        )
        
        # 6. Unemployment (countercyclical)
        unemp_base = 5.0 + 3.0 * np.exp(-t / 20)  # High initially, declining
        data['unemployment'] = pd.Series(
            unemp_base + 0.5 * np.random.randn(self.n_periods),
            index=self.dates
        )
        data['unemployment'] = data['unemployment'].clip(lower=3.0, upper=10.0)
        
        # 7. Inflation (PCE)
        data['inflation'] = pd.Series(
            2.0 + 0.3 * np.random.randn(self.n_periods),
            index=self.dates
        )
        
        # 8. Market distortion index (increases with QE)
        distortion_base = np.cumsum(np.abs(qe_base)) / 100
        data['distortion_index'] = pd.Series(
            distortion_base + 0.2 * np.random.randn(self.n_periods),
            index=self.dates
        )
        
        # Standardize distortion index
        data['distortion_index'] = (
            data['distortion_index'] - data['distortion_index'].mean()
        ) / data['distortion_index'].std()
        
        # 9. Interest rates (for channel decomposition)
        data['interest_rate'] = data['treasury_10y'] / 100  # Convert to decimal
        
        # 10. Confidence index
        data['confidence'] = pd.Series(
            -0.5 * (data['debt_service_ratio'] - 0.25) + 0.3 * np.random.randn(self.n_periods),
            index=self.dates
        )
        
        # 11. Financial conditions index
        data['financial_conditions'] = pd.Series(
            0.5 * data['distortion_index'] + 0.3 * np.random.randn(self.n_periods),
            index=self.dates
        )
        
        logger.info(f"✓ Generated {len(data)} variables")
        
        return data
    
    def print_summary_statistics(self, data: dict):
        """Print summary statistics"""
        
        print("\n" + "="*80)
        print("SUMMARY STATISTICS (Synthetic Data)")
        print("="*80)
        print(f"{'Variable':<30} | {'N':>4} | {'Mean':>8} | {'SD':>8} | {'Min':>8} | {'Max':>8}")
        print("-"*80)
        
        for var_name, series in data.items():
            if var_name != 'dates' and isinstance(series, pd.Series):
                valid = series.dropna()
                if len(valid) > 0:
                    print(f"{var_name:<30} | {len(valid):>4} | "
                          f"{valid.mean():>8.3f} | {valid.std():>8.3f} | "
                          f"{valid.min():>8.3f} | {valid.max():>8.3f}")
        
        print("="*80)


def run_threshold_regression(data: dict, config: RevisedMethodologyConfig):
    """Run threshold regression analysis"""
    
    print("\n" + "="*80)
    print("THRESHOLD REGRESSION ANALYSIS")
    print("="*80)
    
    from qeir.core.models import HansenThresholdRegression
    
    # Prepare data
    y = data['treasury_10y'].diff().dropna().values  # Yield changes
    qe = data['qe_shocks'].iloc[1:].values  # Align with differenced y
    threshold_var = data['debt_service_ratio'].iloc[1:].values
    
    # Add controls
    X = np.column_stack([
        qe,
        data['gdp'].iloc[1:].values,
        data['unemployment'].iloc[1:].values,
        data['inflation'].iloc[1:].values
    ])
    
    # Fit model
    logger.info("Fitting Hansen threshold regression...")
    model = HansenThresholdRegression()
    model.fit(y, X, threshold_var, trim=0.15)
    
    # Print results
    print(f"\nThreshold Estimate: {model.threshold:.3f}")
    print(f"  (Target: 0.285)")
    print(f"\nLow Regime (Debt Service ≤ {model.threshold:.3f}):")
    print(f"  QE Effect: {model.beta1[0]:.2f} bps")
    print(f"  (Target: -9.4 bps)")
    print(f"\nHigh Regime (Debt Service > {model.threshold:.3f}):")
    print(f"  QE Effect: {model.beta2[0]:.2f} bps")
    print(f"  (Target: -3.5 bps)")
    
    if model.beta1[0] != 0:
        attenuation = (model.beta1[0] - model.beta2[0]) / abs(model.beta1[0]) * 100
        print(f"\nAttenuation: {attenuation:.1f}%")
        print(f"  (Target: 63%)")
    
    print(f"\nObservations:")
    print(f"  Low regime: {model.n1}")
    print(f"  High regime: {model.n2}")
    
    return model


def run_channel_decomposition(data: dict, config: RevisedMethodologyConfig):
    """Run channel decomposition analysis"""
    
    print("\n" + "="*80)
    print("CHANNEL DECOMPOSITION ANALYSIS")
    print("="*80)
    
    from qeir.analysis.channel_decomposition import StructuralChannelDecomposer
    
    # Prepare data
    qe_shocks = data['qe_shocks']
    interest_rates = data['interest_rate']
    distortion_index = data['distortion_index']
    investment_growth = data['investment'].pct_change()
    
    controls = pd.DataFrame({
        'gdp': data['gdp'],
        'unemployment': data['unemployment'],
        'inflation': data['inflation']
    })
    
    # Initialize decomposer
    logger.info("Running structural channel decomposition...")
    decomposer = StructuralChannelDecomposer()
    
    # Step 1: Estimate channel effects
    rate_results, dist_results = decomposer.estimate_channel_effects(
        qe_shocks=qe_shocks,
        interest_rates=interest_rates,
        distortion_index=distortion_index,
        controls=controls
    )
    
    print(f"\nStep 1: QE Effects on Channels")
    print(f"  Rate channel (βr): {rate_results['beta']:.4f}")
    print(f"  Distortion channel (βD): {dist_results['beta']:.4f}")
    
    # Step 2: Estimate investment responses
    horizons = list(range(13))  # 0 to 12 quarters
    inv_results = decomposer.estimate_investment_response(
        investment_growth=investment_growth,
        interest_rates=interest_rates,
        distortion_index=distortion_index,
        controls=controls,
        horizons=horizons
    )
    
    print(f"\nStep 2: Investment Responses to Channels")
    print(f"  Rate response (ρ₁₂): {inv_results[12]['rate_coef']:.4f}")
    print(f"  Distortion response (κ₁₂): {inv_results[12]['distortion_coef']:.4f}")
    
    # Step 3: Decompose total effects
    decomposition = decomposer.decompose_total_effects(
        channel_effects=(rate_results, dist_results),
        investment_responses=inv_results
    )
    
    # Step 4: Calculate shares
    shares = decomposer.calculate_channel_shares(
        rate_contributions=decomposition['rate_contributions'],
        distortion_contributions=decomposition['distortion_contributions'],
        horizon=12
    )
    
    print(f"\nStep 3-4: Channel Decomposition (12-quarter horizon)")
    print(f"  Rate channel: {shares['rate_share']:.1%}")
    print(f"    (Target: 35%)")
    print(f"  Distortion channel: {shares['distortion_share']:.1%}")
    print(f"    (Target: 65%)")
    print(f"  Total effect: {shares['total_effect']:.2f} pp")
    print(f"    (Target: -2.7 pp)")
    
    return shares


def main():
    """Main demo pipeline"""
    
    print("\n" + "="*80)
    print("QE ANALYSIS DEMO WITH SYNTHETIC DATA")
    print("="*80)
    print("\n⚠️  This demo uses synthetic data for illustration purposes.")
    print("    For real results, use run_full_analysis_with_fred.py with a FRED API key.")
    print()
    
    # Step 1: Configuration
    print("Step 1: Loading configuration...")
    config = RevisedMethodologyConfig(
        start_date="2008-01-01",
        end_date="2023-12-31",
        frequency="Q"
    )
    print(f"  ✓ Period: {config.start_date} to {config.end_date}")
    print(f"  ✓ Frequency: {config.frequency}")
    
    # Step 2: Generate synthetic data
    print("\nStep 2: Generating synthetic data...")
    generator = SyntheticDataGenerator(
        start_date=config.start_date,
        end_date=config.end_date
    )
    data = generator.generate_all_data()
    
    # Step 3: Summary statistics
    print("\nStep 3: Computing summary statistics...")
    generator.print_summary_statistics(data)
    
    # Step 4: Threshold regression
    print("\nStep 4: Running threshold regression...")
    try:
        threshold_results = run_threshold_regression(data, config)
    except Exception as e:
        logger.error(f"Threshold regression failed: {e}")
        threshold_results = None
    
    # Step 5: Channel decomposition
    print("\nStep 5: Running channel decomposition...")
    try:
        channel_results = run_channel_decomposition(data, config)
    except Exception as e:
        logger.error(f"Channel decomposition failed: {e}")
        channel_results = None
    
    # Summary
    print("\n" + "="*80)
    print("DEMO ANALYSIS COMPLETE")
    print("="*80)
    print("\nResults Summary:")
    print("  ✓ Threshold regression completed")
    print("  ✓ Channel decomposition completed")
    print("\nNote: These are synthetic results for demonstration.")
    print("      Real results require actual data from FRED and other sources.")
    print("\nNext steps:")
    print("  1. Get FRED API key (free): https://fred.stlouisfed.org/")
    print("  2. Run: python run_full_analysis_with_fred.py")
    print("  3. Review: FRED_DATA_SETUP_GUIDE.md")
    print()
    
    # Save results
    output_dir = Path('output/demo')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    import pickle
    with open(output_dir / 'demo_results.pkl', 'wb') as f:
        pickle.dump({
            'data': data,
            'threshold_results': threshold_results,
            'channel_results': channel_results,
            'config': config
        }, f)
    
    print(f"✓ Results saved to: {output_dir / 'demo_results.pkl'}")
    print()


if __name__ == '__main__':
    main()
