"""
Test if monthly frequency explains the sample size discrepancy.

Paper claims N=40/23 (total 63)
Our quarterly spec gives N=18/6 (total 24)

If monthly: 24 quarters × ~3 = ~72 months, closer to 63
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
    print("MONTHLY FREQUENCY TEST")
    print("="*75)
    
    start = '2008-01-01'
    end = '2023-12-31'
    
    print("\nFetching data...")
    yields = fred.get_series('DGS10', observation_start=start, observation_end=end)
    interest = fred.get_series('A091RC1Q027SBEA', observation_start=start, observation_end=end, frequency='q')
    revenue = fred.get_series('FGRECPT', observation_start=start, observation_end=end, frequency='q')
    
    all_fomc_dates = get_fomc_dates(2008, 2023)
    
    # Active QE period
    qe_start = pd.Timestamp('2008-11-01')
    qe_end = pd.Timestamp('2014-10-31')
    
    fomc_dates = [d for d in all_fomc_dates 
                  if qe_start <= pd.Timestamp(d) <= qe_end]
    
    print(f"\nSample period: {qe_start.date()} to {qe_end.date()}")
    print(f"FOMC dates: {len(fomc_dates)}")
    
    # Construct daily shocks
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
    
    # MONTHLY aggregation
    monthly_shocks = daily_shocks.resample('ME').sum()
    
    # Monthly yield changes
    yields_m = yields.resample('ME').last()
    y_monthly = yields_m.diff().dropna() * 100
    
    # Transform shock
    shock_m = -monthly_shocks / 3
    
    # Debt ratio - need to interpolate quarterly to monthly
    debt_ratio_q = interest / revenue
    debt_ratio_m = debt_ratio_q.resample('ME').ffill()  # Forward fill quarterly to monthly
    
    # Align indices
    shock_m.index = pd.to_datetime(shock_m.index).to_period('M').to_timestamp('M')
    y_monthly.index = pd.to_datetime(y_monthly.index).to_period('M').to_timestamp('M')
    debt_ratio_m.index = pd.to_datetime(debt_ratio_m.index).to_period('M').to_timestamp('M')
    
    # Combine
    data = pd.DataFrame({
        'y': y_monthly,
        'shock': shock_m,
        'debt_ratio': debt_ratio_m
    }).dropna()
    
    # Filter to QE period
    data = data[(data.index >= qe_start) & (data.index <= qe_end)]
    
    print(f"\nMonthly dataset: {len(data)} observations")
    print(f"Debt ratio range: [{data['debt_ratio'].min():.4f}, {data['debt_ratio'].max():.4f}]")
    
    # Threshold
    threshold = 0.16
    
    low_mask = data['debt_ratio'] <= threshold
    high_mask = data['debt_ratio'] > threshold
    
    print(f"\nWith threshold = {threshold}:")
    print(f"  Low-debt N = {low_mask.sum()}")
    print(f"  High-debt N = {high_mask.sum()}")
    print(f"  Total N = {len(data)}")
    
    # Estimate
    X_low = sm.add_constant(data.loc[low_mask, 'shock'])
    X_high = sm.add_constant(data.loc[high_mask, 'shock'])
    
    fit_low = OLS(data.loc[low_mask, 'y'], X_low).fit(cov_type='HC1')
    fit_high = OLS(data.loc[high_mask, 'y'], X_high).fit(cov_type='HC1')
    
    print("\n" + "="*75)
    print("MONTHLY FREQUENCY RESULTS")
    print("="*75)
    
    print(f"\nLow-debt regime:")
    print(f"  β = {fit_low.params['shock']:.2f} bps, p = {fit_low.pvalues['shock']:.4f}")
    
    print(f"\nHigh-debt regime:")
    print(f"  β = {fit_high.params['shock']:.2f} bps, p = {fit_high.pvalues['shock']:.4f}")
    
    print("\n" + "="*75)
    print("COMPARISON")
    print("="*75)
    print(f"\n{'Frequency':<15} {'Low β':>10} {'Low p':>10} {'High β':>10} {'High p':>10} {'N':>12}")
    print("-"*75)
    print(f"{'Monthly':<15} {fit_low.params['shock']:>10.1f} {fit_low.pvalues['shock']:>10.3f} {fit_high.params['shock']:>10.1f} {fit_high.pvalues['shock']:>10.3f} {low_mask.sum():>5}/{high_mask.sum():<5}")
    print(f"{'Paper':<15} {-9.4:>10.1f} {'<0.05':>10} {-3.5:>10.1f} {'>0.10':>10} {'40':>5}/{'23':<5}")
    
    # Try different thresholds to match N
    print("\n" + "="*75)
    print("THRESHOLD SEARCH TO MATCH N=40/23")
    print("="*75)
    
    for thresh in np.arange(0.13, 0.17, 0.005):
        low = data['debt_ratio'] <= thresh
        high = data['debt_ratio'] > thresh
        
        if low.sum() < 5 or high.sum() < 5:
            continue
        
        X_l = sm.add_constant(data.loc[low, 'shock'])
        X_h = sm.add_constant(data.loc[high, 'shock'])
        
        try:
            f_l = OLS(data.loc[low, 'y'], X_l).fit(cov_type='HC1')
            f_h = OLS(data.loc[high, 'y'], X_h).fit(cov_type='HC1')
            
            sig_l = '*' if f_l.pvalues['shock'] < 0.05 else ''
            sig_h = '*' if f_h.pvalues['shock'] < 0.05 else ''
            
            print(f"thresh={thresh:.3f}: β_low={f_l.params['shock']:>6.1f}{sig_l:<1} β_high={f_h.params['shock']:>6.1f}{sig_h:<1} N={low.sum():>2}/{high.sum():<2}")
        except:
            continue
    
    # Try extending sample period
    print("\n" + "="*75)
    print("EXTENDED SAMPLE PERIOD TEST")
    print("="*75)
    
    # Try 2008-2019 (pre-COVID)
    data_ext = pd.DataFrame({
        'y': y_monthly,
        'shock': shock_m,
        'debt_ratio': debt_ratio_m
    }).dropna()
    
    ext_end = pd.Timestamp('2019-12-31')
    data_ext = data_ext[(data_ext.index >= qe_start) & (data_ext.index <= ext_end)]
    
    print(f"\nExtended sample (2008-11 to 2019-12): {len(data_ext)} observations")
    
    low_ext = data_ext['debt_ratio'] <= 0.16
    high_ext = data_ext['debt_ratio'] > 0.16
    
    print(f"N = {low_ext.sum()}/{high_ext.sum()}")
    
    if low_ext.sum() >= 5 and high_ext.sum() >= 5:
        X_l = sm.add_constant(data_ext.loc[low_ext, 'shock'])
        X_h = sm.add_constant(data_ext.loc[high_ext, 'shock'])
        
        f_l = OLS(data_ext.loc[low_ext, 'y'], X_l).fit(cov_type='HC1')
        f_h = OLS(data_ext.loc[high_ext, 'y'], X_h).fit(cov_type='HC1')
        
        print(f"β_low = {f_l.params['shock']:.1f} (p={f_l.pvalues['shock']:.3f})")
        print(f"β_high = {f_h.params['shock']:.1f} (p={f_h.pvalues['shock']:.3f})")


if __name__ == "__main__":
    main()
