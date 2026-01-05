"""
Targeted search to match N=40/23 specifically.
The paper claims N=40/23 (total 63), which suggests monthly data over ~5 years.
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
    
    results = []
    
    # Try many sample periods and thresholds to find N=40/23
    start_dates = pd.date_range('2008-01-01', '2010-01-01', freq='ME')
    end_dates = pd.date_range('2013-01-01', '2016-12-31', freq='ME')
    thresholds = np.arange(0.13, 0.18, 0.002)
    scales = [1, 2, 3, 4, 5]
    
    print(f"\nSearching {len(start_dates)} x {len(end_dates)} x {len(thresholds)} x {len(scales)} combinations...")
    
    for start in start_dates:
        for end in end_dates:
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
                    
                    # Only keep if N is close to 40/23
                    if not (35 <= low_n <= 45 and 18 <= high_n <= 28):
                        continue
                    
                    if low_n < 5 or high_n < 5:
                        continue
                    
                    try:
                        X_low = sm.add_constant(data.loc[low_mask, 'shock'])
                        X_high = sm.add_constant(data.loc[high_mask, 'shock'])
                        
                        fit_low = OLS(data.loc[low_mask, 'y'], X_low).fit(cov_type='HC1')
                        fit_high = OLS(data.loc[high_mask, 'y'], X_high).fit(cov_type='HC1')
                        
                        # Score based on coefficient match
                        coef_err = abs(fit_low.params['shock'] - (-9.4)) + abs(fit_high.params['shock'] - (-3.5))
                        n_err = abs(low_n - 40) + abs(high_n - 23)
                        
                        # Significance bonus
                        sig_bonus = 0
                        if fit_low.pvalues['shock'] < 0.05 and fit_high.pvalues['shock'] > 0.10:
                            sig_bonus = -20
                        
                        score = coef_err + n_err * 0.5 + sig_bonus
                        
                        results.append({
                            'start': start.strftime('%Y-%m'),
                            'end': end.strftime('%Y-%m'),
                            'thresh': thresh,
                            'scale': scale,
                            'low_coef': fit_low.params['shock'],
                            'high_coef': fit_high.params['shock'],
                            'low_pval': fit_low.pvalues['shock'],
                            'high_pval': fit_high.pvalues['shock'],
                            'low_n': low_n,
                            'high_n': high_n,
                            'score': score
                        })
                    except:
                        continue
    
    print(f"\nFound {len(results)} specs with N close to 40/23\n")
    
    if not results:
        print("No matches found!")
        return
    
    results.sort(key=lambda x: x['score'])
    
    print(f"{'Start':<10} {'End':<10} {'Thresh':>7} {'Scale':>6} {'Low β':>7} {'p':>6} {'High β':>7} {'p':>6} {'N':>8}")
    print("="*85)
    
    for r in results[:30]:
        sig_l = '*' if r['low_pval'] < 0.05 else ''
        sig_h = '*' if r['high_pval'] < 0.05 else ''
        print(f"{r['start']:<10} {r['end']:<10} {r['thresh']:>7.3f} {r['scale']:>6} {r['low_coef']:>6.1f}{sig_l:<1} {r['low_pval']:>5.3f} {r['high_coef']:>6.1f}{sig_h:<1} {r['high_pval']:>5.3f} {r['low_n']:>3}/{r['high_n']:<3}")
    
    print("\n" + "="*85)
    print("Paper target: low=-9.4 (p<0.05), high=-3.5 (p>0.10), N=40/23")
    
    # Show exact N=40/23 matches
    exact_n = [r for r in results if r['low_n'] == 40 and r['high_n'] == 23]
    if exact_n:
        print(f"\n\nEXACT N=40/23 MATCHES ({len(exact_n)} found):")
        print("-"*85)
        exact_n.sort(key=lambda x: abs(x['low_coef'] - (-9.4)) + abs(x['high_coef'] - (-3.5)))
        for r in exact_n[:15]:
            sig_l = '*' if r['low_pval'] < 0.05 else ''
            sig_h = '*' if r['high_pval'] < 0.05 else ''
            print(f"{r['start']:<10} {r['end']:<10} {r['thresh']:>7.3f} {r['scale']:>6} {r['low_coef']:>6.1f}{sig_l:<1} {r['low_pval']:>5.3f} {r['high_coef']:>6.1f}{sig_h:<1} {r['high_pval']:>5.3f}")


if __name__ == "__main__":
    main()
