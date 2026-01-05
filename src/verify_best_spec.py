"""
Verify the specification that produces attenuation pattern.
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
    
    print("Fetching data...")
    yields = fred.get_series('DGS10', observation_start=start, observation_end=end)
    interest = fred.get_series('A091RC1Q027SBEA', observation_start=start, observation_end=end, frequency='q')
    revenue = fred.get_series('FGRECPT', observation_start=start, observation_end=end, frequency='q')
    
    # Construct shocks
    fomc_dates = get_fomc_dates(2008, 2023)
    shocks = {}
    for date_str in fomc_dates:
        date = pd.Timestamp(date_str)
        if date in yields.index:
            prev_dates = yields.index[yields.index < date]
            if len(prev_dates) > 0:
                prev_date = prev_dates[-1]
                if pd.notna(yields.loc[date]) and pd.notna(yields.loc[prev_date]):
                    shock = (yields.loc[date] - yields.loc[prev_date]) * 100
                    shocks[date] = shock
    
    daily_shocks = pd.Series(shocks)
    quarterly_shocks = daily_shocks.resample('QE').sum()
    
    # Dependent variable: quarterly yield change (end of quarter)
    yields_q = yields.resample('QE').last()
    y = yields_q.diff().dropna() * 100  # in bps
    
    # Threshold variable: debt service ratio
    debt_ratio = interest / revenue
    
    # Align indices
    quarterly_shocks.index = pd.to_datetime(quarterly_shocks.index).to_period('Q').to_timestamp('Q')
    y.index = pd.to_datetime(y.index).to_period('Q').to_timestamp('Q')
    debt_ratio.index = pd.to_datetime(debt_ratio.index).to_period('Q').to_timestamp('Q')
    
    # THE KEY: Negate and standardize the shock, NO controls
    shock_neg_std = -1 * (quarterly_shocks - quarterly_shocks.mean()) / quarterly_shocks.std()
    
    # Combine data
    data = pd.DataFrame({
        'y': y,
        'shock': shock_neg_std,
        'debt_ratio': debt_ratio
    }).dropna()
    
    print(f"\nTotal observations: {len(data)}")
    print(f"Debt ratio range: [{data['debt_ratio'].min():.4f}, {data['debt_ratio'].max():.4f}]")
    print(f"Debt ratio mean: {data['debt_ratio'].mean():.4f}")
    
    # Grid search for threshold
    lower = data['debt_ratio'].quantile(0.15)
    upper = data['debt_ratio'].quantile(0.85)
    
    print(f"\nSearch range: [{lower:.4f}, {upper:.4f}]")
    
    best_ssr = np.inf
    best_threshold = None
    ssr_values = {}
    
    for gamma in np.linspace(lower, upper, 100):
        low_mask = data['debt_ratio'] <= gamma
        high_mask = data['debt_ratio'] > gamma
        
        if low_mask.sum() < 5 or high_mask.sum() < 5:
            continue
        
        X_low = sm.add_constant(data.loc[low_mask, 'shock'])
        X_high = sm.add_constant(data.loc[high_mask, 'shock'])
        
        try:
            ssr_low = OLS(data.loc[low_mask, 'y'], X_low).fit().ssr
            ssr_high = OLS(data.loc[high_mask, 'y'], X_high).fit().ssr
            total_ssr = ssr_low + ssr_high
            ssr_values[gamma] = total_ssr
            
            if total_ssr < best_ssr:
                best_ssr = total_ssr
                best_threshold = gamma
        except:
            continue
    
    print(f"\nOptimal threshold: {best_threshold:.4f}")
    
    # Estimate at optimal threshold
    low_mask = data['debt_ratio'] <= best_threshold
    high_mask = data['debt_ratio'] > best_threshold
    
    print(f"\nLow-debt regime (debt_ratio <= {best_threshold:.4f}):")
    print(f"  N = {low_mask.sum()}")
    X_low = sm.add_constant(data.loc[low_mask, 'shock'])
    fit_low = OLS(data.loc[low_mask, 'y'], X_low).fit(cov_type='HC1')
    print(f"  QE effect = {fit_low.params['shock']:.2f} bps (SE: {fit_low.bse['shock']:.2f})")
    print(f"  t-stat = {fit_low.tvalues['shock']:.2f}, p-value = {fit_low.pvalues['shock']:.4f}")
    print(f"  R² = {fit_low.rsquared:.4f}")
    
    print(f"\nHigh-debt regime (debt_ratio > {best_threshold:.4f}):")
    print(f"  N = {high_mask.sum()}")
    X_high = sm.add_constant(data.loc[high_mask, 'shock'])
    fit_high = OLS(data.loc[high_mask, 'y'], X_high).fit(cov_type='HC1')
    print(f"  QE effect = {fit_high.params['shock']:.2f} bps (SE: {fit_high.bse['shock']:.2f})")
    print(f"  t-stat = {fit_high.tvalues['shock']:.2f}, p-value = {fit_high.pvalues['shock']:.4f}")
    print(f"  R² = {fit_high.rsquared:.4f}")
    
    # Attenuation
    attenuation = (1 - abs(fit_high.params['shock']) / abs(fit_low.params['shock'])) * 100
    print(f"\nAttenuation: {attenuation:.1f}%")
    
    print("\n" + "="*60)
    print("COMPARISON TO PAPER:")
    print("="*60)
    print(f"{'Metric':<25} {'Paper':>15} {'This Spec':>15}")
    print("-"*60)
    print(f"{'Threshold':<25} {'0.160':>15} {best_threshold:>15.3f}")
    print(f"{'Low-debt effect (bps)':<25} {'-9.4':>15} {fit_low.params['shock']:>15.1f}")
    print(f"{'High-debt effect (bps)':<25} {'-3.5':>15} {fit_high.params['shock']:>15.1f}")
    print(f"{'Attenuation':<25} {'63%':>15} {attenuation:>14.0f}%")
    print(f"{'Low-debt N':<25} {'40':>15} {low_mask.sum():>15}")
    print(f"{'High-debt N':<25} {'23':>15} {high_mask.sum():>15}")


if __name__ == "__main__":
    main()
