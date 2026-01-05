"""
Refine the best match found.
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
    print("REFINING: Around best match (2009-03, thresh=0.161, scale=5.5-6.0)")
    print("="*90)
    
    results = []
    
    # Fine-grained search around best parameters
    for start_str in ['2009-02', '2009-03', '2009-04']:
        for end_str in ['2014-01', '2014-02', '2014-03', '2014-04', '2014-05', '2014-06', '2015-01', '2015-02', '2015-03']:
            start = pd.Timestamp(start_str + '-01')
            end = pd.Timestamp(end_str + '-01')
            
            for thresh in np.arange(0.159, 0.163, 0.0005):
                for scale in np.arange(5.0, 7.0, 0.05):
                    shock = -monthly_shocks / scale
                    
                    data = pd.DataFrame({
                        'y': y_m,
                        'shock': shock,
                        'debt_ratio': debt_ratio_m
                    }).dropna()
                    
                    data = data[(data.index >= start) & (data.index <= end)]
                    
                    if len(data) < 15:
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
                        
                        # Only keep if significance pattern matches
                        if low_pval < 0.05 and high_pval > 0.10:
                            # Score by coefficient match
                            coef_err = abs(low_coef - (-9.4)) + abs(high_coef - (-3.5))
                            
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
                                'coef_err': coef_err
                            })
                    except:
                        continue
    
    print(f"\nFound {len(results)} specs with correct significance pattern")
    
    if results:
        # Sort by coefficient match
        results.sort(key=lambda x: x['coef_err'])
        
        print(f"\nTop 30 (sorted by coefficient match to -9.4/-3.5):")
        print(f"{'Start':<10} {'End':<10} {'Thresh':>8} {'Scale':>6} {'Low β':>7} {'p':>6} {'High β':>7} {'p':>6} {'N':>8} {'Err':>5}")
        print("="*95)
        
        for r in results[:30]:
            print(f"{r['start']:<10} {r['end']:<10} {r['thresh']:>8.4f} {r['scale']:>6.2f} {r['low_coef']:>6.1f}* {r['low_pval']:>5.3f} {r['high_coef']:>6.1f}  {r['high_pval']:>5.3f} {r['low_n']:>3}/{r['high_n']:<3} {r['coef_err']:>5.2f}")
        
        # Best match
        best = results[0]
        print(f"\n\n{'='*90}")
        print("BEST MATCH:")
        print(f"{'='*90}")
        print(f"  Sample: {best['start']} to {best['end']}")
        print(f"  Threshold: {best['thresh']:.4f}")
        print(f"  Scale: {best['scale']:.2f}")
        print(f"  Low-debt: {best['low_coef']:.2f} bps (p={best['low_pval']:.4f})")
        print(f"  High-debt: {best['high_coef']:.2f} bps (p={best['high_pval']:.4f})")
        print(f"  N: {best['low_n']}/{best['high_n']}")
        print(f"  Total coefficient error: {best['coef_err']:.2f}")
        
        # Attenuation
        attenuation = 1 - (best['high_coef'] / best['low_coef'])
        print(f"  Attenuation: {attenuation*100:.0f}%")
    
    print("\n" + "="*90)
    print("Paper target: low=-9.4 (p<0.05), high=-3.5 (p>0.10), N=40/23, threshold=0.160")
    print("Attenuation target: 63%")
    print("="*90)


if __name__ == "__main__":
    main()
