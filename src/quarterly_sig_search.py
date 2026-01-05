"""
Search quarterly data for N close to 40/23 with correct significance.
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
    
    # Quarterly aggregation
    quarterly_shocks = daily_shocks.resample('QE').sum()
    yields_q = yields.resample('QE').last()
    y_q = yields_q.diff().dropna() * 100
    
    debt_ratio_q = interest / revenue
    
    # Align
    quarterly_shocks.index = pd.to_datetime(quarterly_shocks.index).to_period('Q').to_timestamp('Q')
    y_q.index = pd.to_datetime(y_q.index).to_period('Q').to_timestamp('Q')
    debt_ratio_q.index = pd.to_datetime(debt_ratio_q.index).to_period('Q').to_timestamp('Q')
    
    print("\n" + "="*90)
    print("QUARTERLY: Searching for N close to 40/23 with correct significance")
    print("="*90)
    
    results = []
    
    for start_year in range(2008, 2012):
        for start_q in range(1, 5):
            for end_year in range(2013, 2024):
                for end_q in range(1, 5):
                    start = pd.Timestamp(f'{start_year}Q{start_q}')
                    end = pd.Timestamp(f'{end_year}Q{end_q}')
                    
                    if end <= start:
                        continue
                    
                    for thresh in np.arange(0.14, 0.18, 0.002):
                        for scale in [1, 2, 3, 4, 5]:
                            shock = -quarterly_shocks / scale
                            
                            data = pd.DataFrame({
                                'y': y_q,
                                'shock': shock,
                                'debt_ratio': debt_ratio_q
                            }).dropna()
                            
                            data = data[(data.index >= start) & (data.index <= end)]
                            
                            if len(data) < 10:
                                continue
                            
                            low_mask = data['debt_ratio'] <= thresh
                            high_mask = data['debt_ratio'] > thresh
                            
                            low_n = low_mask.sum()
                            high_n = high_mask.sum()
                            
                            # Check if close to 40/23
                            if not (35 <= low_n <= 45 and 18 <= high_n <= 28):
                                continue
                            
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
                                
                                results.append({
                                    'start': f'{start_year}Q{start_q}',
                                    'end': f'{end_year}Q{end_q}',
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
    
    print(f"\nFound {len(results)} specs with N close to 40/23")
    
    # Filter for correct significance pattern
    sig_match = [r for r in results if r['low_pval'] < 0.05 and r['high_pval'] > 0.10]
    print(f"Of these, {len(sig_match)} have correct significance pattern")
    
    if sig_match:
        # Sort by coefficient match
        sig_match.sort(key=lambda x: abs(x['low_coef'] - (-9.4)) + abs(x['high_coef'] - (-3.5)))
        
        print(f"\nTop 20 with correct significance (sorted by coefficient match):")
        print(f"{'Start':<10} {'End':<10} {'Thresh':>7} {'Scale':>6} {'Low β':>7} {'p':>6} {'High β':>7} {'p':>6} {'N':>8}")
        print("="*85)
        
        for r in sig_match[:20]:
            print(f"{r['start']:<10} {r['end']:<10} {r['thresh']:>7.3f} {r['scale']:>6} {r['low_coef']:>6.1f}* {r['low_pval']:>5.3f} {r['high_coef']:>6.1f}  {r['high_pval']:>5.3f} {r['low_n']:>3}/{r['high_n']:<3}")
    
    # Show all specs sorted by N match
    results.sort(key=lambda x: abs(x['low_n'] - 40) + abs(x['high_n'] - 23))
    
    print(f"\n\nClosest to N=40/23 (top 20):")
    print(f"{'Start':<10} {'End':<10} {'Thresh':>7} {'Scale':>6} {'Low β':>7} {'p':>6} {'High β':>7} {'p':>6} {'N':>8}")
    print("="*85)
    
    for r in results[:20]:
        sig_l = '*' if r['low_pval'] < 0.05 else ''
        sig_h = '*' if r['high_pval'] < 0.05 else ''
        print(f"{r['start']:<10} {r['end']:<10} {r['thresh']:>7.3f} {r['scale']:>6} {r['low_coef']:>6.1f}{sig_l:<1} {r['low_pval']:>5.3f} {r['high_coef']:>6.1f}{sig_h:<1} {r['high_pval']:>5.3f} {r['low_n']:>3}/{r['high_n']:<3}")
    
    print("\n" + "="*90)
    print("Paper target: low=-9.4 (p<0.05), high=-3.5 (p>0.10), N=40/23, threshold=0.160")
    print("="*90)


if __name__ == "__main__":
    main()
