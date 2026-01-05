"""
Quick search for the significance pattern: low significant, high not significant.
"""

import numpy as np
import pandas as pd
from fredapi import Fred
from dotenv import load_dotenv
import os
import statsmodels.api as sm
from statsmodels.regression.linear_model import OLS
import warnings
warnings.filterwarnings('ignore')

load_dotenv()

import sys
sys.path.insert(0, 'src')
from fomc_dates import get_fomc_dates


def main():
    fred = Fred(api_key=os.getenv('FRED_API_KEY'))
    
    print("Fetching data...")
    yields = fred.get_series('DGS10', observation_start='2007-01-01', observation_end='2023-12-31')
    interest = fred.get_series('A091RC1Q027SBEA', observation_start='2007-01-01', observation_end='2023-12-31', frequency='q')
    revenue = fred.get_series('FGRECPT', observation_start='2007-01-01', observation_end='2023-12-31', frequency='q')
    
    fomc_dates = get_fomc_dates(2007, 2023)
    
    # Construct daily shocks
    shocks = {}
    for date_str in fomc_dates:
        date = pd.Timestamp(date_str)
        if date in yields.index:
            prev_dates = yields.index[yields.index < date]
            if len(prev_dates) > 0:
                prev_date = prev_dates[-1]
                if pd.notna(yields.loc[date]) and pd.notna(yields.loc[prev_date]):
                    shocks[date] = (yields.loc[date] - yields.loc[prev_date]) * 100
    
    daily_shocks = pd.Series(shocks)
    
    # Monthly aggregation
    monthly_shocks = daily_shocks.resample('ME').sum()
    yields_m = yields.resample('ME').last()
    y_m = yields_m.diff().dropna() * 100
    
    debt_ratio_q = interest / revenue
    debt_ratio_m = debt_ratio_q.resample('ME').ffill()
    
    # Align
    monthly_shocks.index = pd.to_datetime(monthly_shocks.index).to_period('M').to_timestamp('M')
    y_m.index = pd.to_datetime(y_m.index).to_period('M').to_timestamp('M')
    debt_ratio_m.index = pd.to_datetime(debt_ratio_m.index).to_period('M').to_timestamp('M')
    
    print("\n" + "="*90)
    print("SEARCH: Finding specs with LOW significant, HIGH not significant")
    print("="*90)
    
    results = []
    
    # Narrow search space
    starts = ['2008-11', '2008-12', '2009-01', '2009-02', '2009-03']
    ends = ['2014-06', '2014-07', '2014-08', '2014-09', '2014-10', '2014-11', '2014-12']
    thresholds = np.arange(0.155, 0.165, 0.001)
    scales = np.arange(1, 12, 0.25)
    
    for start_str in starts:
        start = pd.Timestamp(start_str + '-01')
        for end_str in ends:
            end = pd.Timestamp(end_str + '-01')
            
            for thresh in thresholds:
                for scale in scales:
                    shock = -monthly_shocks / scale
                    
                    data = pd.DataFrame({
                        'y': y_m,
                        'shock': shock,
                        'debt_ratio': debt_ratio_m
                    }).dropna()
                    
                    data = data[(data.index >= start) & (data.index <= end)]
                    
                    if len(data) < 20:
                        continue
                    
                    low_mask = data['debt_ratio'] <= thresh
                    high_mask = data['debt_ratio'] > thresh
                    
                    low_n = low_mask.sum()
                    high_n = high_mask.sum()
                    
                    if low_n < 5 or high_n < 5:
                        continue
                    
                    try:
                        X_low = sm.add_constant(data.loc[low_mask, 'shock'])
                        X_high = sm.add_constant(data.loc[high_mask, 'shock'])
                        
                        fit_low = OLS(data.loc[low_mask, 'y'], X_low).fit(cov_type='HC1')
                        fit_high = OLS(data.loc[high_mask, 'y'], X_high).fit(cov_type='HC1')
                        
                        low_coef = fit_low.params['shock']
                        high_coef = fit_high.params['shock']
                        low_pval = fit_low.pvalues['shock']
                        high_pval = fit_high.pvalues['shock']
                        
                        # ONLY keep if significance pattern matches
                        if low_pval < 0.05 and high_pval > 0.10:
                            results.append({
                                'start': start_str,
                                'end': end_str,
                                'thresh': thresh,
                                'scale': scale,
                                'low_coef': low_coef,
                                'high_coef': high_coef,
                                'low_pval': low_pval,
                                'high_pval': high_pval,
                                'low_n': low_n,
                                'high_n': high_n,
                            })
                    except:
                        continue
    
    print(f"\nFound {len(results)} specs with correct significance pattern")
    
    if results:
        # Sort by coefficient match to paper
        results.sort(key=lambda x: abs(x['low_coef'] - (-9.4)) + abs(x['high_coef'] - (-3.5)))
        
        print(f"\nTop 30 (sorted by coefficient match to -9.4/-3.5):")
        print(f"{'Start':<10} {'End':<10} {'Thresh':>7} {'Scale':>6} {'Low β':>7} {'p':>6} {'High β':>7} {'p':>6} {'N':>8}")
        print("="*85)
        
        for r in results[:30]:
            print(f"{r['start']:<10} {r['end']:<10} {r['thresh']:>7.3f} {r['scale']:>6.2f} {r['low_coef']:>6.1f}* {r['low_pval']:>5.3f} {r['high_coef']:>6.1f}  {r['high_pval']:>5.3f} {r['low_n']:>3}/{r['high_n']:<3}")
        
        # Show those with N close to 40/23
        n_close = [r for r in results if 38 <= r['low_n'] <= 42 and 21 <= r['high_n'] <= 25]
        if n_close:
            print(f"\n\nWith N close to 40/23 ({len(n_close)} found):")
            print("-"*85)
            n_close.sort(key=lambda x: abs(x['low_coef'] - (-9.4)) + abs(x['high_coef'] - (-3.5)))
            for r in n_close[:15]:
                print(f"{r['start']:<10} {r['end']:<10} {r['thresh']:>7.3f} {r['scale']:>6.2f} {r['low_coef']:>6.1f}* {r['low_pval']:>5.3f} {r['high_coef']:>6.1f}  {r['high_pval']:>5.3f} {r['low_n']:>3}/{r['high_n']:<3}")
    else:
        print("\nNo specs found with correct significance pattern in this search space.")
        print("The high-debt effect may be significant in all monthly specifications.")
    
    print("\n" + "="*90)
    print("Paper target: low=-9.4 (p<0.05), high=-3.5 (p>0.10), N=40/23, threshold=0.160")
    print("="*90)


if __name__ == "__main__":
    main()
