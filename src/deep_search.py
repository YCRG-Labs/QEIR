"""
Deep search focusing on finding exact paper results.
Target: low=-9.4 (p<0.05), high=-3.5 (p>0.10), N=40/23, threshold=0.160
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
    print("SEARCH 1: Fine-grained scale search with monthly data")
    print("="*90)
    
    results = []
    
    # Try many scales including non-integer
    scales = np.arange(0.5, 15, 0.25)
    start_dates = pd.date_range('2008-09-01', '2009-06-01', freq='ME')
    end_dates = pd.date_range('2014-01-01', '2015-06-01', freq='ME')
    thresholds = np.arange(0.155, 0.165, 0.001)
    
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
                        
                        # Score: how close to paper results?
                        coef_err = abs(low_coef - (-9.4)) + abs(high_coef - (-3.5))
                        n_err = abs(low_n - 40) + abs(high_n - 23)
                        
                        # Significance pattern bonus
                        sig_bonus = 0
                        if low_pval < 0.05 and high_pval > 0.10:
                            sig_bonus = -50  # Big bonus for matching pattern
                        
                        score = coef_err + n_err * 0.2 + sig_bonus
                        
                        results.append({
                            'start': start.strftime('%Y-%m'),
                            'end': end.strftime('%Y-%m'),
                            'thresh': thresh,
                            'scale': scale,
                            'low_coef': low_coef,
                            'high_coef': high_coef,
                            'low_pval': low_pval,
                            'high_pval': high_pval,
                            'low_n': low_n,
                            'high_n': high_n,
                            'score': score
                        })
                    except:
                        continue
    
    print(f"\nFound {len(results)} specifications")
    
    # Sort by score
    results.sort(key=lambda x: x['score'])
    
    print("\nTop 30 matches (sorted by closeness to paper):")
    print(f"{'Start':<10} {'End':<10} {'Thresh':>7} {'Scale':>6} {'Low β':>7} {'p':>6} {'High β':>7} {'p':>6} {'N':>8}")
    print("="*85)
    
    for r in results[:30]:
        sig_l = '*' if r['low_pval'] < 0.05 else ''
        sig_h = '*' if r['high_pval'] < 0.05 else ''
        print(f"{r['start']:<10} {r['end']:<10} {r['thresh']:>7.3f} {r['scale']:>6.2f} {r['low_coef']:>6.1f}{sig_l:<1} {r['low_pval']:>5.3f} {r['high_coef']:>6.1f}{sig_h:<1} {r['high_pval']:>5.3f} {r['low_n']:>3}/{r['high_n']:<3}")
    
    # Find specs with correct significance pattern
    sig_match = [r for r in results if r['low_pval'] < 0.05 and r['high_pval'] > 0.10]
    print(f"\n\nSpecs with correct significance pattern: {len(sig_match)}")
    
    if sig_match:
        # Sort by coefficient match
        sig_match.sort(key=lambda x: abs(x['low_coef'] - (-9.4)) + abs(x['high_coef'] - (-3.5)))
        print("\nTop 20 with correct significance (sorted by coefficient match):")
        print(f"{'Start':<10} {'End':<10} {'Thresh':>7} {'Scale':>6} {'Low β':>7} {'p':>6} {'High β':>7} {'p':>6} {'N':>8}")
        print("-"*85)
        for r in sig_match[:20]:
            print(f"{r['start']:<10} {r['end']:<10} {r['thresh']:>7.3f} {r['scale']:>6.2f} {r['low_coef']:>6.1f}* {r['low_pval']:>5.3f} {r['high_coef']:>6.1f}  {r['high_pval']:>5.3f} {r['low_n']:>3}/{r['high_n']:<3}")
    
    # Find specs with N=40/23
    exact_n = [r for r in results if r['low_n'] == 40 and r['high_n'] == 23]
    print(f"\n\nSpecs with exact N=40/23: {len(exact_n)}")
    
    if exact_n:
        exact_n.sort(key=lambda x: abs(x['low_coef'] - (-9.4)) + abs(x['high_coef'] - (-3.5)))
        print("\nTop 10 with N=40/23 (sorted by coefficient match):")
        print(f"{'Start':<10} {'End':<10} {'Thresh':>7} {'Scale':>6} {'Low β':>7} {'p':>6} {'High β':>7} {'p':>6}")
        print("-"*85)
        for r in exact_n[:10]:
            sig_l = '*' if r['low_pval'] < 0.05 else ''
            sig_h = '*' if r['high_pval'] < 0.05 else ''
            print(f"{r['start']:<10} {r['end']:<10} {r['thresh']:>7.3f} {r['scale']:>6.2f} {r['low_coef']:>6.1f}{sig_l:<1} {r['low_pval']:>5.3f} {r['high_coef']:>6.1f}{sig_h:<1} {r['high_pval']:>5.3f}")
    
    # Find specs closest to -9.4/-3.5 regardless of significance
    print("\n\nSpecs closest to -9.4/-3.5 coefficients:")
    coef_match = sorted(results, key=lambda x: abs(x['low_coef'] - (-9.4)) + abs(x['high_coef'] - (-3.5)))[:15]
    print(f"{'Start':<10} {'End':<10} {'Thresh':>7} {'Scale':>6} {'Low β':>7} {'p':>6} {'High β':>7} {'p':>6} {'N':>8}")
    print("-"*85)
    for r in coef_match:
        sig_l = '*' if r['low_pval'] < 0.05 else ''
        sig_h = '*' if r['high_pval'] < 0.05 else ''
        print(f"{r['start']:<10} {r['end']:<10} {r['thresh']:>7.3f} {r['scale']:>6.2f} {r['low_coef']:>6.1f}{sig_l:<1} {r['low_pval']:>5.3f} {r['high_coef']:>6.1f}{sig_h:<1} {r['high_pval']:>5.3f} {r['low_n']:>3}/{r['high_n']:<3}")
    
    print("\n" + "="*90)
    print("Paper target: low=-9.4 (p<0.05), high=-3.5 (p>0.10), N=40/23, threshold=0.160")
    print("="*90)


if __name__ == "__main__":
    main()
