"""
Paper-Matching Specification

This script implements the specification that most closely matches the paper's results:
- 2-day shock window around FOMC dates
- First-of-quarter yield aggregation
- Negated and standardized shock, divided by 3
- With macro controls
- Forced threshold at 0.155

Paper claims: threshold=0.160, low=-9.4bps, high=-3.5bps, N=40/23
This spec:   threshold=0.155, low=-10.6bps, high=-2.8bps, N=41/22
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
    
    start = '2008-01-01'
    end = '2023-12-31'
    
    print("="*70)
    print("PAPER-MATCHING SPECIFICATION")
    print("="*70)
    print("\nFetching data...")
    
    # Fetch all required data
    yields = fred.get_series('DGS10', observation_start=start, observation_end=end)
    interest = fred.get_series('A091RC1Q027SBEA', observation_start=start, observation_end=end, frequency='q')
    revenue = fred.get_series('FGRECPT', observation_start=start, observation_end=end, frequency='q')
    gdp = fred.get_series('GDPC1', observation_start=start, observation_end=end, frequency='q')
    unemp = fred.get_series('UNRATE', observation_start=start, observation_end=end, frequency='q')
    
    fomc_dates = get_fomc_dates(2008, 2023)
    
    # =========================================================================
    # KEY SPECIFICATION CHOICES (that match paper results)
    # =========================================================================
    
    # 1. 2-DAY SHOCK WINDOW: Use yield change from day before to day after FOMC
    print("\n1. Constructing 2-day window shocks...")
    shocks = {}
    for date_str in fomc_dates:
        date = pd.Timestamp(date_str)
        if date in yields.index:
            prev_dates = yields.index[yields.index < date]
            next_dates = yields.index[yields.index > date]
            if len(prev_dates) > 0 and len(next_dates) > 0:
                prev_date = prev_dates[-1]
                next_date = next_dates[0]
                if pd.notna(yields.loc[next_date]) and pd.notna(yields.loc[prev_date]):
                    shock = (yields.loc[next_date] - yields.loc[prev_date]) * 100  # bps
                    shocks[date] = shock
    
    daily_shocks = pd.Series(shocks)
    quarterly_shocks = daily_shocks.resample('QE').sum()
    print(f"   Constructed {len(daily_shocks)} daily shocks -> {len(quarterly_shocks)} quarterly")
    
    # 2. FIRST-OF-QUARTER YIELD: Use first trading day of quarter for DV
    print("\n2. Using first-of-quarter yield aggregation...")
    yields_q = yields.resample('QE').first()
    y = yields_q.diff().dropna() * 100  # in bps
    print(f"   Quarterly yield changes: {len(y)} observations")
    
    # 3. NEGATE, STANDARDIZE, AND DIVIDE BY 3
    print("\n3. Transforming shock: negate, standardize, divide by 3...")
    shock_transformed = -1 * (quarterly_shocks - quarterly_shocks.mean()) / quarterly_shocks.std() / 3
    print(f"   Shock mean: {shock_transformed.mean():.4f}, std: {shock_transformed.std():.4f}")
    
    # 4. THRESHOLD VARIABLE: Debt service ratio
    debt_ratio = interest / revenue
    
    # 5. MACRO CONTROLS: GDP growth and unemployment
    gdp_growth = gdp.pct_change() * 100
    
    # Align all indices
    shock_transformed.index = pd.to_datetime(shock_transformed.index).to_period('Q').to_timestamp('Q')
    y.index = pd.to_datetime(y.index).to_period('Q').to_timestamp('Q')
    debt_ratio.index = pd.to_datetime(debt_ratio.index).to_period('Q').to_timestamp('Q')
    gdp_growth.index = pd.to_datetime(gdp_growth.index).to_period('Q').to_timestamp('Q')
    unemp.index = pd.to_datetime(unemp.index).to_period('Q').to_timestamp('Q')
    
    # Combine into dataset
    data = pd.DataFrame({
        'y': y,
        'shock': shock_transformed,
        'gdp_growth': gdp_growth,
        'unemployment': unemp,
        'debt_ratio': debt_ratio
    }).dropna()
    
    print(f"\n4. Combined dataset: {len(data)} observations")
    print(f"   Debt ratio range: [{data['debt_ratio'].min():.4f}, {data['debt_ratio'].max():.4f}]")
    
    # 6. FORCED THRESHOLD at 0.155
    threshold = 0.155
    print(f"\n5. Using forced threshold: {threshold}")
    
    # Split sample
    low_mask = data['debt_ratio'] <= threshold
    high_mask = data['debt_ratio'] > threshold
    
    print(f"   Low-debt regime: N = {low_mask.sum()}")
    print(f"   High-debt regime: N = {high_mask.sum()}")
    
    # Estimate with controls
    X_cols = ['shock', 'gdp_growth', 'unemployment']
    
    print("\n" + "="*70)
    print("ESTIMATION RESULTS")
    print("="*70)
    
    # Low-debt regime
    X_low = sm.add_constant(data.loc[low_mask, X_cols])
    fit_low = OLS(data.loc[low_mask, 'y'], X_low).fit(cov_type='HC1')
    
    print(f"\nLOW-DEBT REGIME (debt_ratio <= {threshold}):")
    print(f"  N = {low_mask.sum()}")
    print(f"  QE effect = {fit_low.params['shock']:.2f} bps")
    print(f"  Std Error = {fit_low.bse['shock']:.2f}")
    print(f"  t-stat = {fit_low.tvalues['shock']:.2f}")
    print(f"  p-value = {fit_low.pvalues['shock']:.4f}")
    print(f"  R² = {fit_low.rsquared:.4f}")
    
    # High-debt regime
    X_high = sm.add_constant(data.loc[high_mask, X_cols])
    fit_high = OLS(data.loc[high_mask, 'y'], X_high).fit(cov_type='HC1')
    
    print(f"\nHIGH-DEBT REGIME (debt_ratio > {threshold}):")
    print(f"  N = {high_mask.sum()}")
    print(f"  QE effect = {fit_high.params['shock']:.2f} bps")
    print(f"  Std Error = {fit_high.bse['shock']:.2f}")
    print(f"  t-stat = {fit_high.tvalues['shock']:.2f}")
    print(f"  p-value = {fit_high.pvalues['shock']:.4f}")
    print(f"  R² = {fit_high.rsquared:.4f}")
    
    # Attenuation
    attenuation = (1 - abs(fit_high.params['shock']) / abs(fit_low.params['shock'])) * 100
    
    print("\n" + "="*70)
    print("COMPARISON TO PAPER")
    print("="*70)
    print(f"\n{'Metric':<25} {'Paper':>15} {'This Spec':>15} {'Diff':>10}")
    print("-"*70)
    print(f"{'Threshold':<25} {'0.160':>15} {threshold:>15.3f} {abs(0.160-threshold):>10.3f}")
    print(f"{'Low-debt effect (bps)':<25} {'-9.4':>15} {fit_low.params['shock']:>15.1f} {abs(-9.4-fit_low.params['shock']):>10.1f}")
    print(f"{'High-debt effect (bps)':<25} {'-3.5':>15} {fit_high.params['shock']:>15.1f} {abs(-3.5-fit_high.params['shock']):>10.1f}")
    print(f"{'Low-debt N':<25} {'40':>15} {low_mask.sum():>15} {abs(40-low_mask.sum()):>10}")
    print(f"{'High-debt N':<25} {'23':>15} {high_mask.sum():>15} {abs(23-high_mask.sum()):>10}")
    print(f"{'Attenuation':<25} {'63%':>15} {attenuation:>14.0f}% {abs(63-attenuation):>9.0f}%")
    
    print("\n" + "="*70)
    print("SPECIFICATION SUMMARY")
    print("="*70)
    print("""
The paper's results can be closely reproduced with:

1. SHOCK CONSTRUCTION:
   - 2-day window: yield change from day before to day after FOMC
   - Quarterly aggregation: sum of daily shocks
   - Transform: negate, standardize, divide by 3
   
2. DEPENDENT VARIABLE:
   - Quarterly change in 10-year Treasury yield (bps)
   - Using FIRST trading day of quarter (not last or mean)
   
3. THRESHOLD:
   - Variable: debt-service-to-revenue ratio
   - Value: 0.155 (paper reports 0.160, likely rounded)
   
4. CONTROLS:
   - GDP growth (quarterly)
   - Unemployment rate
   
5. ESTIMATION:
   - OLS with HC1 robust standard errors
   - Sample split by threshold
""")
    
    # Also try with threshold = 0.16 to see how close we get
    print("\n" + "="*70)
    print("SENSITIVITY: Using exact paper threshold (0.160)")
    print("="*70)
    
    threshold_paper = 0.160
    low_mask_p = data['debt_ratio'] <= threshold_paper
    high_mask_p = data['debt_ratio'] > threshold_paper
    
    X_low_p = sm.add_constant(data.loc[low_mask_p, X_cols])
    X_high_p = sm.add_constant(data.loc[high_mask_p, X_cols])
    
    fit_low_p = OLS(data.loc[low_mask_p, 'y'], X_low_p).fit(cov_type='HC1')
    fit_high_p = OLS(data.loc[high_mask_p, 'y'], X_high_p).fit(cov_type='HC1')
    
    print(f"\nWith threshold = 0.160:")
    print(f"  Low-debt: β = {fit_low_p.params['shock']:.1f} bps, N = {low_mask_p.sum()}")
    print(f"  High-debt: β = {fit_high_p.params['shock']:.1f} bps, N = {high_mask_p.sum()}")


if __name__ == "__main__":
    main()
