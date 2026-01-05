"""
Find specification that produces results closest to paper claims:
- Threshold: ~0.160
- Low-debt: -9.4 bps
- High-debt: -3.5 bps
- N: 40/23
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
    gdp = fred.get_series('GDPC1', observation_start=start, observation_end=end, frequency='q')
    unemp = fred.get_series('UNRATE', observation_start=start, observation_end=end, frequency='q')
    inflation = fred.get_series('PCEPILFE', observation_start=start, observation_end=end, frequency='q')
    vix = fred.get_series('VIXCLS', observation_start=start, observation_end=end)
    
    fomc_dates = get_fomc_dates(2008, 2023)
    
    # Try different shock constructions
    def construct_shocks_daily_change(yields, fomc_dates):
        """Standard: daily yield change on FOMC dates"""
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
        return pd.Series(shocks)
    
    def construct_shocks_2day_window(yields, fomc_dates):
        """2-day window around FOMC"""
        shocks = {}
        for date_str in fomc_dates:
            date = pd.Timestamp(date_str)
            if date in yields.index:
                # Find day before and day after
                prev_dates = yields.index[yields.index < date]
                next_dates = yields.index[yields.index > date]
                if len(prev_dates) > 0 and len(next_dates) > 0:
                    prev_date = prev_dates[-1]
                    next_date = next_dates[0]
                    if pd.notna(yields.loc[next_date]) and pd.notna(yields.loc[prev_date]):
                        shock = (yields.loc[next_date] - yields.loc[prev_date]) * 100
                        shocks[date] = shock
        return pd.Series(shocks)
    
    results = []
    
    # Test different combinations
    shock_methods = [
        ('1day', construct_shocks_daily_change),
        ('2day', construct_shocks_2day_window),
    ]
    
    yield_agg_methods = ['last', 'mean', 'first']
    
    shock_transforms = [
        ('raw', lambda s: s),
        ('neg', lambda s: -s),
        ('std', lambda s: (s - s.mean()) / s.std()),
        ('neg_std', lambda s: -1 * (s - s.mean()) / s.std()),
        # Try different scaling
        ('neg_div3', lambda s: -s / 3),
        ('neg_std_div3', lambda s: -1 * (s - s.mean()) / s.std() / 3),
    ]
    
    # Different threshold values to force
    forced_thresholds = [None, 0.16, 0.155, 0.15, 0.145, 0.14]
    
    control_options = [False, True]
    
    print("\nSearching for best specification...\n")
    
    for shock_name, shock_fn in shock_methods:
        daily_shocks = shock_fn(yields, fomc_dates)
        quarterly_shocks = daily_shocks.resample('QE').sum()
        
        for yield_agg in yield_agg_methods:
            if yield_agg == 'last':
                yields_q = yields.resample('QE').last()
            elif yield_agg == 'mean':
                yields_q = yields.resample('QE').mean()
            else:
                yields_q = yields.resample('QE').first()
            
            y = yields_q.diff().dropna() * 100  # bps
            
            for trans_name, trans_fn in shock_transforms:
                shock = trans_fn(quarterly_shocks)
                
                for use_controls in control_options:
                    for forced_thresh in forced_thresholds:
                        # Align indices
                        shock_aligned = shock.copy()
                        shock_aligned.index = pd.to_datetime(shock_aligned.index).to_period('Q').to_timestamp('Q')
                        
                        y_aligned = y.copy()
                        y_aligned.index = pd.to_datetime(y_aligned.index).to_period('Q').to_timestamp('Q')
                        
                        debt_ratio = interest / revenue
                        debt_ratio.index = pd.to_datetime(debt_ratio.index).to_period('Q').to_timestamp('Q')
                        
                        if use_controls:
                            gdp_g = gdp.pct_change() * 100
                            gdp_g.index = pd.to_datetime(gdp_g.index).to_period('Q').to_timestamp('Q')
                            unemp_q = unemp.copy()
                            unemp_q.index = pd.to_datetime(unemp_q.index).to_period('Q').to_timestamp('Q')
                            
                            data = pd.DataFrame({
                                'y': y_aligned,
                                'shock': shock_aligned,
                                'gdp_growth': gdp_g,
                                'unemployment': unemp_q,
                                'debt_ratio': debt_ratio
                            }).dropna()
                            
                            X_cols = ['shock', 'gdp_growth', 'unemployment']
                        else:
                            data = pd.DataFrame({
                                'y': y_aligned,
                                'shock': shock_aligned,
                                'debt_ratio': debt_ratio
                            }).dropna()
                            
                            X_cols = ['shock']
                        
                        if len(data) < 20:
                            continue
                        
                        # Find threshold
                        if forced_thresh:
                            best_threshold = forced_thresh
                        else:
                            lower = data['debt_ratio'].quantile(0.15)
                            upper = data['debt_ratio'].quantile(0.85)
                            
                            best_ssr = np.inf
                            best_threshold = None
                            
                            for gamma in np.linspace(lower, upper, 50):
                                low_mask = data['debt_ratio'] <= gamma
                                high_mask = data['debt_ratio'] > gamma
                                
                                if low_mask.sum() < 5 or high_mask.sum() < 5:
                                    continue
                                
                                try:
                                    X_low = sm.add_constant(data.loc[low_mask, X_cols])
                                    X_high = sm.add_constant(data.loc[high_mask, X_cols])
                                    
                                    ssr_low = OLS(data.loc[low_mask, 'y'], X_low).fit().ssr
                                    ssr_high = OLS(data.loc[high_mask, 'y'], X_high).fit().ssr
                                    
                                    if ssr_low + ssr_high < best_ssr:
                                        best_ssr = ssr_low + ssr_high
                                        best_threshold = gamma
                                except:
                                    continue
                        
                        if best_threshold is None:
                            continue
                        
                        # Estimate
                        low_mask = data['debt_ratio'] <= best_threshold
                        high_mask = data['debt_ratio'] > best_threshold
                        
                        if low_mask.sum() < 5 or high_mask.sum() < 5:
                            continue
                        
                        try:
                            X_low = sm.add_constant(data.loc[low_mask, X_cols])
                            X_high = sm.add_constant(data.loc[high_mask, X_cols])
                            
                            fit_low = OLS(data.loc[low_mask, 'y'], X_low).fit(cov_type='HC1')
                            fit_high = OLS(data.loc[high_mask, 'y'], X_high).fit(cov_type='HC1')
                            
                            low_coef = fit_low.params['shock']
                            high_coef = fit_high.params['shock']
                            low_n = low_mask.sum()
                            high_n = high_mask.sum()
                            
                            # Score: how close to paper?
                            # Paper: threshold=0.16, low=-9.4, high=-3.5, n=40/23
                            thresh_err = abs(best_threshold - 0.16)
                            low_err = abs(low_coef - (-9.4))
                            high_err = abs(high_coef - (-3.5))
                            n_low_err = abs(low_n - 40)
                            n_high_err = abs(high_n - 23)
                            
                            # Combined score (lower is better)
                            score = thresh_err * 100 + low_err + high_err + n_low_err * 0.1 + n_high_err * 0.1
                            
                            ctrl_str = '+ctrl' if use_controls else ''
                            thresh_str = f'@{forced_thresh}' if forced_thresh else ''
                            spec = f"{shock_name}|{yield_agg}|{trans_name}{ctrl_str}{thresh_str}"
                            
                            results.append({
                                'spec': spec,
                                'threshold': best_threshold,
                                'low_coef': low_coef,
                                'high_coef': high_coef,
                                'low_n': low_n,
                                'high_n': high_n,
                                'score': score
                            })
                        except:
                            continue
    
    # Sort by score
    results.sort(key=lambda x: x['score'])
    
    print(f"{'Spec':<45} {'Thresh':>8} {'Low β':>8} {'High β':>8} {'Low N':>6} {'High N':>6} {'Score':>8}")
    print("="*95)
    
    for r in results[:30]:
        print(f"{r['spec']:<45} {r['threshold']:>8.3f} {r['low_coef']:>8.1f} {r['high_coef']:>8.1f} {r['low_n']:>6} {r['high_n']:>6} {r['score']:>8.1f}")
    
    print("\n" + "="*95)
    print("Paper target: threshold=0.160, low=-9.4, high=-3.5, N=40/23")
    print("\nBest match:")
    best = results[0]
    print(f"  Spec: {best['spec']}")
    print(f"  Threshold: {best['threshold']:.3f}")
    print(f"  Low-debt effect: {best['low_coef']:.1f} bps")
    print(f"  High-debt effect: {best['high_coef']:.1f} bps")
    print(f"  N: {best['low_n']}/{best['high_n']}")


if __name__ == "__main__":
    main()
