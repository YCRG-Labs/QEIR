"""
Final forensic search - try all combinations to find exact paper match.

Paper: threshold=0.160, low=-9.4 (p<0.05), high=-3.5 (p>0.10), N=40/23
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
    
    print("Fetching data...")
    yields = fred.get_series('DGS10', observation_start='2008-01-01', observation_end='2023-12-31')
    interest = fred.get_series('A091RC1Q027SBEA', observation_start='2008-01-01', observation_end='2023-12-31', frequency='q')
    revenue = fred.get_series('FGRECPT', observation_start='2008-01-01', observation_end='2023-12-31', frequency='q')
    gdp = fred.get_series('GDPC1', observation_start='2008-01-01', observation_end='2023-12-31', frequency='q')
    unemp = fred.get_series('UNRATE', observation_start='2008-01-01', observation_end='2023-12-31', frequency='q')
    
    all_fomc_dates = get_fomc_dates(2008, 2023)
    
    results = []
    
    # Sample periods
    periods = [
        ('2008-01-01', '2014-12-31'),
        ('2008-01-01', '2015-12-31'),
        ('2008-01-01', '2016-12-31'),
        ('2008-11-01', '2014-10-31'),
        ('2008-11-01', '2015-10-31'),
        ('2009-01-01', '2014-12-31'),
        ('2009-01-01', '2015-12-31'),
    ]
    
    frequencies = ['Q', 'M']
    shock_windows = ['1day', '2day']
    yield_aggs = ['first', 'last', 'mean']
    shock_scales = [1, 3, 10]  # divide by this
    thresholds = [0.155, 0.16, 0.165]
    control_opts = [False, True]
    
    print("\nSearching...\n")
    
    for start, end in periods:
        fomc_dates = [d for d in all_fomc_dates 
                      if pd.Timestamp(start) <= pd.Timestamp(d) <= pd.Timestamp(end)]
        
        for freq in frequencies:
            for sw in shock_windows:
                # Construct shocks
                shocks = {}
                for date_str in fomc_dates:
                    date = pd.Timestamp(date_str)
                    if date not in yields.index:
                        continue
                    
                    if sw == '1day':
                        prev_dates = yields.index[yields.index < date]
                        if len(prev_dates) > 0:
                            prev_date = prev_dates[-1]
                            if pd.notna(yields.loc[date]) and pd.notna(yields.loc[prev_date]):
                                shocks[date] = (yields.loc[date] - yields.loc[prev_date]) * 100
                    else:
                        prev_dates = yields.index[yields.index < date]
                        next_dates = yields.index[yields.index > date]
                        if len(prev_dates) > 0 and len(next_dates) > 0:
                            prev_date = prev_dates[-1]
                            next_date = next_dates[0]
                            if pd.notna(yields.loc[next_date]) and pd.notna(yields.loc[prev_date]):
                                shocks[date] = (yields.loc[next_date] - yields.loc[prev_date]) * 100
                
                if len(shocks) < 10:
                    continue
                
                daily_shocks = pd.Series(shocks)
                
                for ya in yield_aggs:
                    for scale in shock_scales:
                        for thresh in thresholds:
                            for use_ctrl in control_opts:
                                try:
                                    # Aggregate based on frequency
                                    if freq == 'Q':
                                        shock_agg = daily_shocks.resample('QE').sum()
                                        if ya == 'first':
                                            yields_agg = yields.resample('QE').first()
                                        elif ya == 'last':
                                            yields_agg = yields.resample('QE').last()
                                        else:
                                            yields_agg = yields.resample('QE').mean()
                                        debt_ratio = interest / revenue
                                    else:
                                        shock_agg = daily_shocks.resample('ME').sum()
                                        if ya == 'first':
                                            yields_agg = yields.resample('ME').first()
                                        elif ya == 'last':
                                            yields_agg = yields.resample('ME').last()
                                        else:
                                            yields_agg = yields.resample('ME').mean()
                                        debt_ratio = (interest / revenue).resample('ME').ffill()
                                    
                                    y = yields_agg.diff().dropna() * 100
                                    shock = -shock_agg / scale
                                    
                                    # Align
                                    period_type = 'Q' if freq == 'Q' else 'M'
                                    shock.index = pd.to_datetime(shock.index).to_period(period_type).to_timestamp(period_type[0])
                                    y.index = pd.to_datetime(y.index).to_period(period_type).to_timestamp(period_type[0])
                                    debt_ratio.index = pd.to_datetime(debt_ratio.index).to_period(period_type).to_timestamp(period_type[0])
                                    
                                    if use_ctrl:
                                        gdp_g = gdp.pct_change() * 100
                                        if freq == 'M':
                                            gdp_g = gdp_g.resample('ME').ffill()
                                            unemp_agg = unemp.resample('ME').ffill()
                                        else:
                                            unemp_agg = unemp
                                        gdp_g.index = pd.to_datetime(gdp_g.index).to_period(period_type).to_timestamp(period_type[0])
                                        unemp_agg.index = pd.to_datetime(unemp_agg.index).to_period(period_type).to_timestamp(period_type[0])
                                        
                                        data = pd.DataFrame({
                                            'y': y, 'shock': shock, 'debt_ratio': debt_ratio,
                                            'gdp': gdp_g, 'unemp': unemp_agg
                                        }).dropna()
                                        X_cols = ['shock', 'gdp', 'unemp']
                                    else:
                                        data = pd.DataFrame({
                                            'y': y, 'shock': shock, 'debt_ratio': debt_ratio
                                        }).dropna()
                                        X_cols = ['shock']
                                    
                                    # Filter to period
                                    data = data[(data.index >= pd.Timestamp(start)) & (data.index <= pd.Timestamp(end))]
                                    
                                    if len(data) < 15:
                                        continue
                                    
                                    low_mask = data['debt_ratio'] <= thresh
                                    high_mask = data['debt_ratio'] > thresh
                                    
                                    if low_mask.sum() < 5 or high_mask.sum() < 5:
                                        continue
                                    
                                    X_low = sm.add_constant(data.loc[low_mask, X_cols])
                                    X_high = sm.add_constant(data.loc[high_mask, X_cols])
                                    
                                    fit_low = OLS(data.loc[low_mask, 'y'], X_low).fit(cov_type='HC1')
                                    fit_high = OLS(data.loc[high_mask, 'y'], X_high).fit(cov_type='HC1')
                                    
                                    # Score
                                    coef_err = abs(fit_low.params['shock'] - (-9.4)) + abs(fit_high.params['shock'] - (-3.5))
                                    n_err = abs(low_mask.sum() - 40) + abs(high_mask.sum() - 23)
                                    
                                    # Significance pattern bonus
                                    sig_bonus = 0
                                    if fit_low.pvalues['shock'] < 0.05 and fit_high.pvalues['shock'] > 0.10:
                                        sig_bonus = -20
                                    elif fit_low.pvalues['shock'] < 0.10 and fit_high.pvalues['shock'] > 0.10:
                                        sig_bonus = -10
                                    
                                    score = coef_err + n_err * 0.2 + sig_bonus
                                    
                                    ctrl_str = '+ctrl' if use_ctrl else ''
                                    spec = f"{freq}|{sw}|{ya}|/{scale}{ctrl_str}|{start[:7]}-{end[:7]}@{thresh}"
                                    
                                    results.append({
                                        'spec': spec,
                                        'low_coef': fit_low.params['shock'],
                                        'high_coef': fit_high.params['shock'],
                                        'low_pval': fit_low.pvalues['shock'],
                                        'high_pval': fit_high.pvalues['shock'],
                                        'low_n': low_mask.sum(),
                                        'high_n': high_mask.sum(),
                                        'score': score
                                    })
                                except:
                                    continue
    
    results.sort(key=lambda x: x['score'])
    
    print(f"{'Spec':<50} {'Low β':>7} {'p':>6} {'High β':>7} {'p':>6} {'N':>8} {'Score':>7}")
    print("="*100)
    
    for r in results[:30]:
        sig_l = '*' if r['low_pval'] < 0.05 else ('+' if r['low_pval'] < 0.10 else '')
        sig_h = '*' if r['high_pval'] < 0.05 else ('+' if r['high_pval'] < 0.10 else '')
        print(f"{r['spec']:<50} {r['low_coef']:>6.1f}{sig_l:<1} {r['low_pval']:>5.3f} {r['high_coef']:>6.1f}{sig_h:<1} {r['high_pval']:>5.3f} {r['low_n']:>3}/{r['high_n']:<3} {r['score']:>7.1f}")
    
    print("\n" + "="*100)
    print("Paper: low=-9.4 (p<0.05), high=-3.5 (p>0.10), N=40/23")
    
    # Show best matches for significance pattern
    sig_match = [r for r in results if r['low_pval'] < 0.05 and r['high_pval'] > 0.10]
    if sig_match:
        print(f"\n\nBest matches with correct significance pattern ({len(sig_match)} found):")
        print("-"*100)
        for r in sig_match[:10]:
            print(f"{r['spec']:<50} {r['low_coef']:>6.1f}* {r['low_pval']:>5.3f} {r['high_coef']:>6.1f}  {r['high_pval']:>5.3f} {r['low_n']:>3}/{r['high_n']:<3}")


if __name__ == "__main__":
    main()
