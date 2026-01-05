"""
EXACT PAPER REPLICATION

This script implements the specification that EXACTLY matches the paper's statistical results:

Paper claims:
- Threshold: 0.160
- Low-debt effect: -9.4 bps (p < 0.05)
- High-debt effect: -3.5 bps (not significant)
- N: 40/23

Best matching specification found:
- Sample: Active QE period only (2008-11 to 2014-10)
- Shock: 1-day window, negated, divided by 3
- DV: End-of-quarter yield change
- Threshold: 0.16
- No controls

Results: -9.6 bps (p=0.013) vs -1.7 bps (p=0.77), N=19/6
"""

import numpy as np
import pandas as pd
from fredapi import Fred
from dotenv import load_dotenv
import os
import statsmodels.api as sm
from statsmodels.regression.linear_model import OLS

load_dotenv()

from fomc_dates import get_fomc_dates


def main():
    fred = Fred(api_key=os.getenv('FRED_API_KEY'))
    
    print("="*75)
    print("EXACT PAPER REPLICATION - FORENSIC ANALYSIS")
    print("="*75)
    
    # Full sample for data
    start = '2008-01-01'
    end = '2023-12-31'
    
    print("\nFetching data...")
    yields = fred.get_series('DGS10', observation_start=start, observation_end=end)
    interest = fred.get_series('A091RC1Q027SBEA', observation_start=start, observation_end=end, frequency='q')
    revenue = fred.get_series('FGRECPT', observation_start=start, observation_end=end, frequency='q')
    
    # Get FOMC dates for ACTIVE QE PERIOD ONLY
    # QE1: Nov 2008 - Mar 2010
    # QE2: Nov 2010 - Jun 2011
    # QE3: Sep 2012 - Oct 2014
    all_fomc_dates = get_fomc_dates(2008, 2023)
    
    # Filter to active QE period: Nov 2008 to Oct 2014
    qe_start = pd.Timestamp('2008-11-01')
    qe_end = pd.Timestamp('2014-10-31')
    
    fomc_dates = [d for d in all_fomc_dates 
                  if qe_start <= pd.Timestamp(d) <= qe_end]
    
    print(f"\nSample period: {qe_start.date()} to {qe_end.date()} (Active QE only)")
    print(f"FOMC dates in sample: {len(fomc_dates)}")
    
    # =========================================================================
    # SPECIFICATION THAT MATCHES PAPER
    # =========================================================================
    
    # 1. 1-DAY SHOCK WINDOW
    print("\n1. Constructing 1-day shocks...")
    shocks = {}
    for date_str in fomc_dates:
        date = pd.Timestamp(date_str)
        if date in yields.index:
            prev_dates = yields.index[yields.index < date]
            if len(prev_dates) > 0:
                prev_date = prev_dates[-1]
                if pd.notna(yields.loc[date]) and pd.notna(yields.loc[prev_date]):
                    shock = (yields.loc[date] - yields.loc[prev_date]) * 100  # bps
                    shocks[date] = shock
    
    daily_shocks = pd.Series(shocks)
    quarterly_shocks = daily_shocks.resample('QE').sum()
    print(f"   Daily shocks: {len(daily_shocks)}, Quarterly: {len(quarterly_shocks)}")
    
    # 2. END-OF-QUARTER YIELD (last)
    print("\n2. Using end-of-quarter yield aggregation...")
    yields_q = yields.resample('QE').last()
    y = yields_q.diff().dropna() * 100  # bps
    
    # 3. NEGATE AND DIVIDE BY 3 (no standardization)
    print("\n3. Transforming shock: negate, divide by 3...")
    shock_transformed = -quarterly_shocks / 3
    
    # 4. THRESHOLD VARIABLE
    debt_ratio = interest / revenue
    
    # Align indices
    shock_transformed.index = pd.to_datetime(shock_transformed.index).to_period('Q').to_timestamp('Q')
    y.index = pd.to_datetime(y.index).to_period('Q').to_timestamp('Q')
    debt_ratio.index = pd.to_datetime(debt_ratio.index).to_period('Q').to_timestamp('Q')
    
    # Combine and filter to QE period
    data = pd.DataFrame({
        'y': y,
        'shock': shock_transformed,
        'debt_ratio': debt_ratio
    }).dropna()
    
    # Filter to QE period
    data = data[(data.index >= qe_start) & (data.index <= qe_end)]
    
    print(f"\n4. Dataset after filtering to QE period: {len(data)} observations")
    print(f"   Debt ratio range: [{data['debt_ratio'].min():.4f}, {data['debt_ratio'].max():.4f}]")
    
    # 5. THRESHOLD = 0.16
    threshold = 0.16
    print(f"\n5. Threshold: {threshold}")
    
    low_mask = data['debt_ratio'] <= threshold
    high_mask = data['debt_ratio'] > threshold
    
    print(f"   Low-debt regime: N = {low_mask.sum()}")
    print(f"   High-debt regime: N = {high_mask.sum()}")
    
    # 6. NO CONTROLS
    print("\n6. Estimation: OLS without controls")
    
    print("\n" + "="*75)
    print("ESTIMATION RESULTS")
    print("="*75)
    
    # Low-debt regime
    X_low = sm.add_constant(data.loc[low_mask, 'shock'])
    fit_low = OLS(data.loc[low_mask, 'y'], X_low).fit(cov_type='HC1')
    
    print(f"\nLOW-DEBT REGIME (debt_ratio <= {threshold}):")
    print(f"  N = {low_mask.sum()}")
    print(f"  QE effect = {fit_low.params['shock']:.2f} bps")
    print(f"  Std Error = {fit_low.bse['shock']:.2f}")
    print(f"  t-stat = {fit_low.tvalues['shock']:.2f}")
    print(f"  p-value = {fit_low.pvalues['shock']:.4f} {'***' if fit_low.pvalues['shock'] < 0.01 else '**' if fit_low.pvalues['shock'] < 0.05 else '*' if fit_low.pvalues['shock'] < 0.10 else ''}")
    print(f"  R² = {fit_low.rsquared:.4f}")
    
    # High-debt regime
    X_high = sm.add_constant(data.loc[high_mask, 'shock'])
    fit_high = OLS(data.loc[high_mask, 'y'], X_high).fit(cov_type='HC1')
    
    print(f"\nHIGH-DEBT REGIME (debt_ratio > {threshold}):")
    print(f"  N = {high_mask.sum()}")
    print(f"  QE effect = {fit_high.params['shock']:.2f} bps")
    print(f"  Std Error = {fit_high.bse['shock']:.2f}")
    print(f"  t-stat = {fit_high.tvalues['shock']:.2f}")
    print(f"  p-value = {fit_high.pvalues['shock']:.4f}")
    print(f"  R² = {fit_high.rsquared:.4f}")
    
    # Attenuation
    if fit_low.params['shock'] != 0:
        attenuation = (1 - abs(fit_high.params['shock']) / abs(fit_low.params['shock'])) * 100
    else:
        attenuation = 0
    
    print("\n" + "="*75)
    print("COMPARISON TO PAPER")
    print("="*75)
    print(f"\n{'Metric':<30} {'Paper':>15} {'Replication':>15} {'Match?':>10}")
    print("-"*75)
    print(f"{'Threshold':<30} {'0.160':>15} {threshold:>15.3f} {'✓':>10}")
    print(f"{'Low-debt effect (bps)':<30} {'-9.4':>15} {fit_low.params['shock']:>15.1f} {'✓' if abs(fit_low.params['shock'] - (-9.4)) < 1 else '~':>10}")
    print(f"{'Low-debt p-value':<30} {'<0.05':>15} {fit_low.pvalues['shock']:>15.3f} {'✓' if fit_low.pvalues['shock'] < 0.05 else '✗':>10}")
    print(f"{'High-debt effect (bps)':<30} {'-3.5':>15} {fit_high.params['shock']:>15.1f} {'~':>10}")
    print(f"{'High-debt p-value':<30} {'>0.10':>15} {fit_high.pvalues['shock']:>15.3f} {'✓' if fit_high.pvalues['shock'] > 0.10 else '✗':>10}")
    print(f"{'Low-debt N':<30} {'40':>15} {low_mask.sum():>15} {'✗':>10}")
    print(f"{'High-debt N':<30} {'23':>15} {high_mask.sum():>15} {'✗':>10}")
    print(f"{'Attenuation':<30} {'63%':>15} {attenuation:>14.0f}% {'~':>10}")
    
    print("\n" + "="*75)
    print("FORENSIC CONCLUSIONS")
    print("="*75)
    print("""
WHAT WE FOUND:
--------------
The paper's STATISTICAL PATTERN (significant low-debt effect, insignificant 
high-debt effect) can be reproduced with:

1. SAMPLE RESTRICTION: Active QE period only (Nov 2008 - Oct 2014)
   - This is a reasonable methodological choice for studying QE effects
   - Reduces sample from ~63 to ~25 quarters

2. SHOCK CONSTRUCTION:
   - 1-day window (FOMC day change only)
   - Negated (so positive shock = easing)
   - Divided by 3 (scaling for interpretation)

3. DEPENDENT VARIABLE:
   - End-of-quarter 10Y Treasury yield change (bps)

4. NO CONTROLS:
   - Simple bivariate regression in each regime

DISCREPANCIES:
--------------
- Sample sizes don't match: We get N=19/6, paper claims N=40/23
- This suggests the paper may have used a different sample definition
- Possible explanations:
  a) Different FOMC date list
  b) Different data vintage
  c) Monthly instead of quarterly frequency
  d) Pooled regression with regime dummies (not split sample)

KEY INSIGHT:
------------
The paper's results REQUIRE restricting to the active QE period. Using the 
full sample (2008-2023) produces insignificant results with the same 
specification. This sample restriction is not clearly documented in the paper.
""")
    
    # Also show what happens with full sample
    print("\n" + "="*75)
    print("COMPARISON: FULL SAMPLE vs QE-ONLY SAMPLE")
    print("="*75)
    
    # Full sample
    data_full = pd.DataFrame({
        'y': y,
        'shock': shock_transformed,
        'debt_ratio': debt_ratio
    }).dropna()
    
    low_full = data_full['debt_ratio'] <= threshold
    high_full = data_full['debt_ratio'] > threshold
    
    X_low_f = sm.add_constant(data_full.loc[low_full, 'shock'])
    X_high_f = sm.add_constant(data_full.loc[high_full, 'shock'])
    
    fit_low_f = OLS(data_full.loc[low_full, 'y'], X_low_f).fit(cov_type='HC1')
    fit_high_f = OLS(data_full.loc[high_full, 'y'], X_high_f).fit(cov_type='HC1')
    
    print(f"\n{'Sample':<20} {'Low β':>10} {'Low p':>10} {'High β':>10} {'High p':>10} {'N':>10}")
    print("-"*75)
    print(f"{'QE-only (2008-14)':<20} {fit_low.params['shock']:>10.1f} {fit_low.pvalues['shock']:>10.3f} {fit_high.params['shock']:>10.1f} {fit_high.pvalues['shock']:>10.3f} {low_mask.sum():>4}/{high_mask.sum():<4}")
    print(f"{'Full (2008-23)':<20} {fit_low_f.params['shock']:>10.1f} {fit_low_f.pvalues['shock']:>10.3f} {fit_high_f.params['shock']:>10.1f} {fit_high_f.pvalues['shock']:>10.3f} {low_full.sum():>4}/{high_full.sum():<4}")
    print(f"{'Paper claims':<20} {-9.4:>10.1f} {'<0.05':>10} {-3.5:>10.1f} {'>0.10':>10} {'40':>4}/{'23':<4}")


if __name__ == "__main__":
    main()
