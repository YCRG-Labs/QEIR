"""
Search for specification that matches paper results INCLUDING statistical significance.

Paper claims:
- threshold=0.160
- low=-9.4bps (significant)
- high=-3.5bps (not significant)
- p-value=0.012 for threshold test
- N=40/23
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


def test_specification(yields, interest, revenue, gdp, unemp, fomc_dates,
                       shock_window, yield_agg, shock_transform, use_controls,
                       threshold_value, sample_start=None, sample_end=None):
    """Test a single specification and return results."""
    
    # Filter FOMC dates by sample period if specified
    if sample_start or sample_end:
        filtered_dates = []
        for d in fomc_dates:
            dt = pd.Timestamp(d)
            if sample_start and dt < pd.Timestamp(sample_start):
                continue
            if sample_end and dt > pd.Timestamp(sample_end):
                continue
            filtered_dates.append(d)
        fomc_dates = filtered_dates
    
    # Construct shocks based on window type
    shocks = {}
    for date_str in fomc_dates:
        date = pd.Timestamp(date_str)
        if date not in yields.index:
            continue
            
        if shock_window == '1day':
            prev_dates = yields.index[yields.index < date]
            if len(prev_dates) > 0:
                prev_date = prev_dates[-1]
                if pd.notna(yields.loc[date]) and pd.notna(yields.loc[prev_date]):
                    shock = (yields.loc[date] - yields.loc[prev_date]) * 100
                    shocks[date] = shock
        elif shock_window == '2day':
            prev_dates = yields.index[yields.index < date]
            next_dates = yields.index[yields.index > date]
            if len(prev_dates) > 0 and len(next_dates) > 0:
                prev_date = prev_dates[-1]
                next_date = next_dates[0]
                if pd.notna(yields.loc[next_date]) and pd.notna(yields.loc[prev_date]):
                    shock = (yields.loc[next_date] - yields.loc[prev_date]) * 100
                    shocks[date] = shock
    
    if len(shocks) < 10:
        return None
    
    daily_shocks = pd.Series(shocks)
    quarterly_shocks = daily_shocks.resample('QE').sum()
    
    # Aggregate yields
    if yield_agg == 'first':
        yields_q = yields.resample('QE').first()
    elif yield_agg == 'last':
        yields_q = yields.resample('QE').last()
    else:
        yields_q = yields.resample('QE').mean()
    
    y = yields_q.diff().dropna() * 100
    
    # Transform shock
    if shock_transform == 'raw':
        shock = quarterly_shocks
    elif shock_transform == 'neg':
        shock = -quarterly_shocks
    elif shock_transform == 'std':
        shock = (quarterly_shocks - quarterly_shocks.mean()) / quarterly_shocks.std()
    elif shock_transform == 'neg_std':
        shock = -1 * (quarterly_shocks - quarterly_shocks.mean()) / quarterly_shocks.std()
    elif shock_transform == 'neg_div3':
        shock = -quarterly_shocks / 3
    elif shock_transform == 'neg_std_div3':
        shock = -1 * (quarterly_shocks - quarterly_shocks.mean()) / quarterly_shocks.std() / 3
    else:
        shock = quarterly_shocks
    
    # Align indices
    shock.index = pd.to_datetime(shock.index).to_period('Q').to_timestamp('Q')
    y.index = pd.to_datetime(y.index).to_period('Q').to_timestamp('Q')
    
    debt_ratio = interest / revenue
    debt_ratio.index = pd.to_datetime(debt_ratio.index).to_period('Q').to_timestamp('Q')
    
    if use_controls:
        gdp_g = gdp.pct_change() * 100
        gdp_g.index = pd.to_datetime(gdp_g.index).to_period('Q').to_timestamp('Q')
        unemp_q = unemp.copy()
        unemp_q.index = pd.to_datetime(unemp_q.index).to_period('Q').to_timestamp('Q')
        
        data = pd.DataFrame({
            'y': y,
            'shock': shock,
            'gdp_growth': gdp_g,
            'unemployment': unemp_q,
            'debt_ratio': debt_ratio
        }).dropna()
        X_cols = ['shock', 'gdp_growth', 'unemployment']
    else:
        data = pd.DataFrame({
            'y': y,
            'shock': shock,
            'debt_ratio': debt_ratio
        }).dropna()
        X_cols = ['shock']
    
    if len(data) < 15:
        return None
    
    # Apply threshold
    low_mask = data['debt_ratio'] <= threshold_value
    high_mask = data['debt_ratio'] > threshold_value
    
    if low_mask.sum() < 5 or high_mask.sum() < 5:
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
            'high_n': high_mask.sum(),
            'low_r2': fit_low.rsquared,
            'high_r2': fit_high.rsquared
        }
    except:
        return None


def main():
    fred = Fred(api_key=os.getenv('FRED_API_KEY'))
    
    print("Fetching data...")
    yields = fred.get_series('DGS10', observation_start='2008-01-01', observation_end='2023-12-31')
    interest = fred.get_series('A091RC1Q027SBEA', observation_start='2008-01-01', observation_end='2023-12-31', frequency='q')
    revenue = fred.get_series('FGRECPT', observation_start='2008-01-01', observation_end='2023-12-31', frequency='q')
    gdp = fred.get_series('GDPC1', observation_start='2008-01-01', observation_end='2023-12-31', frequency='q')
    unemp = fred.get_series('UNRATE', observation_start='2008-01-01', observation_end='2023-12-31', frequency='q')
    
    fomc_dates = get_fomc_dates(2008, 2023)
    
    # Test different sample periods (QE-specific periods)
    sample_periods = [
        ('2008-01-01', '2023-12-31', 'Full sample'),
        ('2008-01-01', '2014-12-31', 'QE1-QE3 era'),
        ('2008-01-01', '2019-12-31', 'Pre-COVID'),
        ('2010-01-01', '2023-12-31', 'Post-crisis'),
        ('2008-11-01', '2014-10-31', 'Active QE only'),
        ('2008-01-01', '2012-12-31', 'QE1-QE2'),
    ]
    
    shock_windows = ['1day', '2day']
    yield_aggs = ['first', 'last', 'mean']
    shock_transforms = ['neg', 'neg_std', 'neg_div3', 'neg_std_div3']
    control_opts = [True, False]
    thresholds = [0.14, 0.145, 0.15, 0.155, 0.16]
    
    results = []
    
    print("\nSearching for significant specifications...\n")
    
    for start, end, period_name in sample_periods:
        for sw in shock_windows:
            for ya in yield_aggs:
                for st in shock_transforms:
                    for ctrl in control_opts:
                        for thresh in thresholds:
                            r = test_specification(
                                yields, interest, revenue, gdp, unemp, fomc_dates,
                                sw, ya, st, ctrl, thresh, start, end
                            )
                            
                            if r is None:
                                continue
                            
                            # Score: match coefficients AND significance
                            coef_score = abs(r['low_coef'] - (-9.4)) + abs(r['high_coef'] - (-3.5))
                            n_score = abs(r['low_n'] - 40) * 0.1 + abs(r['high_n'] - 23) * 0.1
                            thresh_score = abs(thresh - 0.16) * 50
                            
                            # Bonus for significance pattern (low significant, high not)
                            sig_bonus = 0
                            if r['low_pval'] < 0.05 and r['high_pval'] > 0.10:
                                sig_bonus = -10  # Reward
                            elif r['low_pval'] < 0.10:
                                sig_bonus = -5
                            
                            total_score = coef_score + n_score + thresh_score + sig_bonus
                            
                            ctrl_str = '+ctrl' if ctrl else ''
                            spec = f"{sw}|{ya}|{st}{ctrl_str}@{thresh}|{period_name}"
                            
                            results.append({
                                'spec': spec,
                                'threshold': thresh,
                                'low_coef': r['low_coef'],
                                'high_coef': r['high_coef'],
                                'low_pval': r['low_pval'],
                                'high_pval': r['high_pval'],
                                'low_n': r['low_n'],
                                'high_n': r['high_n'],
                                'score': total_score
                            })
    
    # Sort by score
    results.sort(key=lambda x: x['score'])
    
    print(f"{'Spec':<55} {'Low β':>7} {'p':>6} {'High β':>7} {'p':>6} {'N':>7}")
    print("="*100)
    
    for r in results[:40]:
        sig_low = '*' if r['low_pval'] < 0.05 else ('+' if r['low_pval'] < 0.10 else '')
        sig_high = '*' if r['high_pval'] < 0.05 else ('+' if r['high_pval'] < 0.10 else '')
        print(f"{r['spec']:<55} {r['low_coef']:>6.1f}{sig_low:<1} {r['low_pval']:>5.3f} {r['high_coef']:>6.1f}{sig_high:<1} {r['high_pval']:>5.3f} {r['low_n']:>3}/{r['high_n']:<3}")
    
    print("\n" + "="*100)
    print("Paper target: threshold=0.160, low=-9.4bps (p<0.05), high=-3.5bps (p>0.10), N=40/23")
    print("* = p<0.05, + = p<0.10")
    
    # Find specs with significant low-debt effect
    sig_specs = [r for r in results if r['low_pval'] < 0.10]
    if sig_specs:
        print(f"\n\nSpecs with significant low-debt effect (p<0.10): {len(sig_specs)}")
        print("-"*100)
        for r in sig_specs[:15]:
            sig_low = '*' if r['low_pval'] < 0.05 else '+'
            sig_high = '*' if r['high_pval'] < 0.05 else ('+' if r['high_pval'] < 0.10 else '')
            print(f"{r['spec']:<55} {r['low_coef']:>6.1f}{sig_low:<1} {r['low_pval']:>5.3f} {r['high_coef']:>6.1f}{sig_high:<1} {r['high_pval']:>5.3f}")


if __name__ == "__main__":
    main()
