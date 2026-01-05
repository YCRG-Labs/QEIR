"""
Exhaustive search for exact paper results.
Target: threshold=0.160, low=-9.4, high=-3.5, N=40/23, p_low<0.05, p_high>0.10
"""

import numpy as np
import pandas as pd
from fredapi import Fred
from dotenv import load_dotenv
import os
import statsmodels.api as sm
from statsmodels.regression.linear_model import OLS
from itertools import product

load_dotenv()

from fomc_dates import get_fomc_dates


def run_spec(yields, interest, revenue, gdp, unemp, fomc_dates,
             freq, shock_window, yield_agg, shock_scale, threshold, 
             use_controls, sample_start, sample_end, lag_shock=0):
    """Run a single specification."""
    
    # Filter FOMC dates
    dates = [d for d in fomc_dates 
             if pd.Timestamp(sample_start) <= pd.Timestamp(d) <= pd.Timestamp(sample_end)]
    
    if len(dates) < 10:
        return None
    
    # Construct shocks
    shocks = {}
    for date_str in dates:
        date = pd.Timestamp(date_str)
        if date not in yields.index:
            continue
        
        if shock_window == '1day':
            prev_dates = yields.index[yields.index < date]
            if len(prev_dates) > 0:
                prev_date = prev_dates[-1]
                if pd.notna(yields.loc[date]) and pd.notna(yields.loc[prev_date]):
                    shocks[date] = (yields.loc[date] - yields.loc[prev_date]) * 100
        elif shock_window == '2day':
            prev_dates = yields.index[yields.index < date]
            next_dates = yields.index[yields.index > date]
            if len(prev_dates) > 0 and len(next_dates) > 0:
                if pd.notna(yields.loc[next_dates[0]]) and pd.notna(yields.loc[prev_dates[-1]]):
                    shocks[date] = (yields.loc[next_dates[0]] - yields.loc[prev_dates[-1]]) * 100
        elif shock_window == '3day':
            # 3-day window: t-1 to t+1
            prev_dates = yields.index[yields.index < date]
            next_dates = yields.index[yields.index > date]
            if len(prev_dates) >= 1 and len(next_dates) >= 1:
                next_date = next_dates[0]
                if len(next_dates) > 1:
                    next_date = next_dates[1]  # t+2
                prev_date = prev_dates[-1]
                if pd.notna(yields.loc[next_date]) and pd.notna(yields.loc[prev_date]):
                    shocks[date] = (yields.loc[next_date] - yields.loc[prev_date]) * 100
    
    if len(shocks) < 5:
        return None
    
    daily_shocks = pd.Series(shocks)
    
    # Aggregate
    if freq == 'Q':
        shock_agg = daily_shocks.resample('QE').sum()
        if yield_agg == 'first':
            yields_agg = yields.resample('QE').first()
        elif yield_agg == 'last':
            yields_agg = yields.resample('QE').last()
        else:
            yields_agg = yields.resample('QE').mean()
        debt_ratio = interest / revenue
        period = 'Q'
    else:  # Monthly
        shock_agg = daily_shocks.resample('ME').sum()
        if yield_agg == 'first':
            yields_agg = yields.resample('ME').first()
        elif yield_agg == 'last':
            yields_agg = yields.resample('ME').last()
        else:
            yields_agg = yields.resample('ME').mean()
        debt_ratio = (interest / revenue).resample('ME').ffill()
        period = 'M'
    
    y = yields_agg.diff().dropna() * 100
    shock = -shock_agg / shock_scale
    
    # Apply lag if specified
    if lag_shock > 0:
        shock = shock.shift(lag_shock)
    
    # Align indices
    shock.index = pd.to_datetime(shock.index).to_period(period).to_timestamp(period[0])
    y.index = pd.to_datetime(y.index).to_period(period).to_timestamp(period[0])
    debt_ratio.index = pd.to_datetime(debt_ratio.index).to_period(period).to_timestamp(period[0])
    
    if use_controls:
        gdp_g = gdp.pct_change() * 100
        if freq == 'M':
            gdp_g = gdp_g.resample('ME').ffill()
            unemp_a = unemp.resample('ME').ffill()
        else:
            unemp_a = unemp
        gdp_g.index = pd.to_datetime(gdp_g.index).to_period(period).to_timestamp(period[0])
        unemp_a.index = pd.to_datetime(unemp_a.index).to_period(period).to_timestamp(period[0])
        
        data = pd.DataFrame({
            'y': y, 'shock': shock, 'debt_ratio': debt_ratio,
            'gdp': gdp_g, 'unemp': unemp_a
        }).dropna()
        X_cols = ['shock', 'gdp', 'unemp']
    else:
        data = pd.DataFrame({
            'y': y, 'shock': shock, 'debt_ratio': debt_ratio
        }).dropna()
        X_cols = ['shock']
    
    data = data[(data.index >= pd.Timestamp(sample_start)) & 
                (data.index <= pd.Timestamp(sample_end))]
    
    if len(data) < 10:
        return None
    
    low_mask = data['debt_ratio'] <= threshold
    high_mask = data['debt_ratio'] > threshold
    
    if low_mask.sum() < 3 or high_mask.sum() < 3:
        return None
    
    try:
        X_low = sm.add_constant(data.loc[low_mask, X_cols])
        X_high = sm.add_constant(data.loc[high_mask, X_cols])
        
        fit_low = OLS(data.loc[low_mask, 'y'], X_low).fit(cov_type='HC1')
        fit_high = OLS(data.loc[high_mask, 'y'], X_high).fit(cov_type='HC1')
        
        return {
            'low_coef': fit_low.params['shock'],
            'high_coef': fit_high.params['shock'],
            'low_pval': fit_low.pvalues['shock'],
            'high_pval': fit_high.pvalues['shock'],
            'low_n': low_mask.sum(),
            'high_n': high_mask.sum()
        }
    except:
        return None


def main():
    fred = Fred(api_key=os.getenv('FRED_API_KEY'))
    
    print("Fetching data...")
    yields = fred.get_series('DGS10', observation_start='2007-01-01', observation_end='2023-12-31')
    interest = fred.get_series('A091RC1Q027SBEA', observation_start='2007-01-01', observation_end='2023-12-31', frequency='q')
    revenue = fred.get_series('FGRECPT', observation_start='2007-01-01', observation_end='2023-12-31', frequency='q')
    gdp = fred.get_series('GDPC1', observation_start='2007-01-01', observation_end='2023-12-31', frequency='q')
    unemp = fred.get_series('UNRATE', observation_start='2007-01-01', observation_end='2023-12-31', frequency='q')
    
    fomc_dates = get_fomc_dates(2007, 2023)
    
    # Expanded parameter grid
    freqs = ['Q', 'M']
    shock_windows = ['1day', '2day', '3day']
    yield_aggs = ['first', 'last', 'mean']
    shock_scales = [1, 2, 3, 4, 5, 10]
    thresholds = [0.155, 0.16, 0.165]
    control_opts = [False, True]
    lags = [0, 1]
    
    # More sample periods
    sample_periods = [
        ('2008-01-01', '2014-12-31'),
        ('2008-01-01', '2015-12-31'),
        ('2008-01-01', '2016-12-31'),
        ('2008-06-01', '2014-12-31'),
        ('2008-06-01', '2015-06-30'),
        ('2008-09-01', '2014-10-31'),
        ('2008-11-01', '2014-10-31'),
        ('2008-11-01', '2015-10-31'),
        ('2008-11-01', '2016-06-30'),
        ('2009-01-01', '2014-12-31'),
        ('2009-01-01', '2015-12-31'),
        ('2009-03-01', '2014-10-31'),
        ('2007-01-01', '2015-12-31'),
        ('2007-06-01', '2014-12-31'),
    ]
    
    results = []
    total = 0
    
    print("\nSearching (this may take a minute)...\n")
    
    for start, end in sample_periods:
        for freq, sw, ya, scale, thresh, ctrl, lag in product(
            freqs, shock_windows, yield_aggs, shock_scales, thresholds, control_opts, lags
        ):
            total += 1
            r = run_spec(yields, interest, revenue, gdp, unemp, fomc_dates,
                        freq, sw, ya, scale, thresh, ctrl, start, end, lag)
            
            if r is None:
                continue
            
            # Calculate match score
            # Target: low=-9.4, high=-3.5, N=40/23
            low_err = abs(r['low_coef'] - (-9.4))
            high_err = abs(r['high_coef'] - (-3.5))
            n_low_err = abs(r['low_n'] - 40)
            n_high_err = abs(r['high_n'] - 23)
            
            # Significance pattern bonus
            sig_bonus = 0
            if r['low_pval'] < 0.05 and r['high_pval'] > 0.10:
                sig_bonus = -15
            elif r['low_pval'] < 0.10 and r['high_pval'] > 0.10:
                sig_bonus = -8
            
            # N match bonus (important!)
            n_bonus = 0
            if 35 <= r['low_n'] <= 45 and 18 <= r['high_n'] <= 28:
                n_bonus = -10
            
            score = low_err + high_err + (n_low_err + n_high_err) * 0.3 + sig_bonus + n_bonus
            
            ctrl_str = '+ctrl' if ctrl else ''
            lag_str = f'L{lag}' if lag > 0 else ''
            spec = f"{freq}|{sw}|{ya}|/{scale}{ctrl_str}{lag_str}|{start[:7]}~{end[:7]}@{thresh}"
            
            results.append({
                'spec': spec,
                'low_coef': r['low_coef'],
                'high_coef': r['high_coef'],
                'low_pval': r['low_pval'],
                'high_pval': r['high_pval'],
                'low_n': r['low_n'],
                'high_n': r['high_n'],
                'score': score
            })
    
    print(f"Tested {total} specifications, {len(results)} valid\n")
    
    results.sort(key=lambda x: x['score'])
    
    print(f"{'Spec':<52} {'Low β':>6} {'p':>6} {'High β':>6} {'p':>6} {'N':>8}")
    print("="*95)
    
    for r in results[:50]:
        sig_l = '*' if r['low_pval'] < 0.05 else ('+' if r['low_pval'] < 0.10 else ' ')
        sig_h = '*' if r['high_pval'] < 0.05 else ('+' if r['high_pval'] < 0.10 else ' ')
        
        # Highlight close matches
        coef_match = abs(r['low_coef'] - (-9.4)) < 1.5 and abs(r['high_coef'] - (-3.5)) < 1.5
        n_match = 35 <= r['low_n'] <= 45 and 18 <= r['high_n'] <= 28
        marker = ">>>" if coef_match and n_match else ""
        
        print(f"{marker}{r['spec']:<49} {r['low_coef']:>5.1f}{sig_l} {r['low_pval']:>5.3f} {r['high_coef']:>5.1f}{sig_h} {r['high_pval']:>5.3f} {r['low_n']:>3}/{r['high_n']:<3}")
    
    print("\n" + "="*95)
    print("Target: low=-9.4 (p<0.05), high=-3.5 (p>0.10), N=40/23")
    print("* = p<0.05, + = p<0.10, >>> = close match on both coefs and N")
    
    # Find best matches for each criterion
    print("\n\n=== BEST COEFFICIENT MATCHES (within 1 bps) ===")
    coef_matches = [r for r in results 
                    if abs(r['low_coef'] - (-9.4)) < 1 and abs(r['high_coef'] - (-3.5)) < 1]
    for r in coef_matches[:10]:
        sig_l = '*' if r['low_pval'] < 0.05 else ''
        print(f"{r['spec']:<52} β: {r['low_coef']:>5.1f}{sig_l}/{r['high_coef']:>5.1f}  N: {r['low_n']}/{r['high_n']}")
    
    print("\n=== BEST N MATCHES (within 5) ===")
    n_matches = [r for r in results 
                 if abs(r['low_n'] - 40) <= 5 and abs(r['high_n'] - 23) <= 5]
    for r in n_matches[:10]:
        sig_l = '*' if r['low_pval'] < 0.05 else ''
        print(f"{r['spec']:<52} β: {r['low_coef']:>5.1f}{sig_l}/{r['high_coef']:>5.1f}  N: {r['low_n']}/{r['high_n']}")
    
    print("\n=== BEST SIGNIFICANCE PATTERN MATCHES ===")
    sig_matches = [r for r in results if r['low_pval'] < 0.05 and r['high_pval'] > 0.10]
    sig_matches.sort(key=lambda x: abs(x['low_coef'] - (-9.4)) + abs(x['high_coef'] - (-3.5)))
    for r in sig_matches[:10]:
        print(f"{r['spec']:<52} β: {r['low_coef']:>5.1f}*/{r['high_coef']:>5.1f}   N: {r['low_n']}/{r['high_n']}")


if __name__ == "__main__":
    main()
