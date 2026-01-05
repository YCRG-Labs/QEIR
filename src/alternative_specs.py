"""
Try alternative specifications to match paper results:
- Different threshold definitions
- Different shock aggregation
- Pooled regression with interaction
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
    
    # Also try debt-to-GDP
    debt = fred.get_series('GFDEBTN', observation_start='2007-01-01', observation_end='2023-12-31', frequency='q')
    gdp = fred.get_series('GDP', observation_start='2007-01-01', observation_end='2023-12-31', frequency='q')
    
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
    
    # Monthly aggregation - try different methods
    monthly_shocks_sum = daily_shocks.resample('ME').sum()
    monthly_shocks_mean = daily_shocks.resample('ME').mean()
    monthly_shocks_last = daily_shocks.resample('ME').last()
    
    yields_m = yields.resample('ME').last()
    y_m = yields_m.diff().dropna() * 100
    
    # Different threshold variables
    debt_ratio_q = interest / revenue
    debt_gdp_q = debt / gdp
    
    debt_ratio_m = debt_ratio_q.resample('ME').ffill()
    debt_gdp_m = debt_gdp_q.resample('ME').ffill()
    
    # Align indices
    for s in [monthly_shocks_sum, monthly_shocks_mean, monthly_shocks_last, y_m, debt_ratio_m, debt_gdp_m]:
        s.index = pd.to_datetime(s.index).to_period('M').to_timestamp('M')
    
    print("\n" + "="*90)
    print("APPROACH 1: Try different threshold variables")
    print("="*90)
    
    # Test with debt-to-GDP ratio
    start = pd.Timestamp('2009-01-01')
    end = pd.Timestamp('2014-10-01')
    
    for thresh_name, thresh_var in [('interest/revenue', debt_ratio_m), ('debt/GDP', debt_gdp_m)]:
        print(f"\n--- Threshold: {thresh_name} ---")
        
        for shock_name, shock_var in [('sum', monthly_shocks_sum), ('mean', monthly_shocks_mean)]:
            for scale in [1, 3, 5, 10]:
                shock = -shock_var / scale
                
                data = pd.DataFrame({
                    'y': y_m,
                    'shock': shock,
                    'thresh_var': thresh_var
                }).dropna()
                
                data = data[(data.index >= start) & (data.index <= end)]
                
                # Find threshold that gives ~40/23 split
                for thresh in np.arange(0.10, 0.25, 0.005):
                    low_mask = data['thresh_var'] <= thresh
                    high_mask = data['thresh_var'] > thresh
                    
                    low_n = low_mask.sum()
                    high_n = high_mask.sum()
                    
                    if low_n == 40 and high_n == 23:
                        try:
                            X_low = sm.add_constant(data.loc[low_mask, 'shock'])
                            X_high = sm.add_constant(data.loc[high_mask, 'shock'])
                            
                            fit_low = OLS(data.loc[low_mask, 'y'], X_low).fit(cov_type='HC1')
                            fit_high = OLS(data.loc[high_mask, 'y'], X_high).fit(cov_type='HC1')
                            
                            sig_l = '*' if fit_low.pvalues['shock'] < 0.05 else ''
                            sig_h = '*' if fit_high.pvalues['shock'] < 0.05 else ''
                            
                            print(f"  {shock_name}/scale={scale}: thresh={thresh:.3f} | Low: {fit_low.params['shock']:.1f}{sig_l} (p={fit_low.pvalues['shock']:.3f}) | High: {fit_high.params['shock']:.1f}{sig_h} (p={fit_high.pvalues['shock']:.3f})")
                        except:
                            pass
    
    print("\n" + "="*90)
    print("APPROACH 2: Pooled regression with interaction term")
    print("="*90)
    
    # Pooled: y = a + b*shock + c*high_debt + d*shock*high_debt + e
    # Low-debt effect = b
    # High-debt effect = b + d
    
    for scale in [1, 2, 3, 4, 5]:
        shock = -monthly_shocks_sum / scale
        
        data = pd.DataFrame({
            'y': y_m,
            'shock': shock,
            'debt_ratio': debt_ratio_m
        }).dropna()
        
        data = data[(data.index >= start) & (data.index <= end)]
        
        for thresh in [0.155, 0.158, 0.160, 0.162, 0.165]:
            data['high_debt'] = (data['debt_ratio'] > thresh).astype(int)
            data['shock_x_high'] = data['shock'] * data['high_debt']
            
            X = sm.add_constant(data[['shock', 'high_debt', 'shock_x_high']])
            fit = OLS(data['y'], X).fit(cov_type='HC1')
            
            low_effect = fit.params['shock']
            high_effect = fit.params['shock'] + fit.params['shock_x_high']
            
            low_n = (data['high_debt'] == 0).sum()
            high_n = (data['high_debt'] == 1).sum()
            
            sig_l = '*' if fit.pvalues['shock'] < 0.05 else ''
            
            print(f"Scale={scale}, thresh={thresh:.3f}: Low={low_effect:.1f}{sig_l} (p={fit.pvalues['shock']:.3f}), High={high_effect:.1f}, N={low_n}/{high_n}")
    
    print("\n" + "="*90)
    print("APPROACH 3: Try FOMC-only observations (not monthly)")
    print("="*90)
    
    # Use only FOMC dates, not monthly aggregation
    # This could give different N
    
    fomc_shocks = daily_shocks.copy()
    
    # Get yield change on FOMC day
    fomc_y = pd.Series(index=fomc_shocks.index, dtype=float)
    for date in fomc_shocks.index:
        if date in yields.index:
            prev_dates = yields.index[yields.index < date]
            if len(prev_dates) > 0:
                prev_date = prev_dates[-1]
                if pd.notna(yields.loc[date]) and pd.notna(yields.loc[prev_date]):
                    fomc_y.loc[date] = (yields.loc[date] - yields.loc[prev_date]) * 100
    
    # Get debt ratio for each FOMC date
    fomc_debt = pd.Series(index=fomc_shocks.index, dtype=float)
    for date in fomc_shocks.index:
        # Find most recent quarterly debt ratio
        prev_q = debt_ratio_q.index[debt_ratio_q.index <= date]
        if len(prev_q) > 0:
            fomc_debt.loc[date] = debt_ratio_q.loc[prev_q[-1]]
    
    data = pd.DataFrame({
        'y': fomc_y,
        'shock': -fomc_shocks,
        'debt_ratio': fomc_debt
    }).dropna()
    
    # Filter to QE period
    data = data[(data.index >= '2008-11-01') & (data.index <= '2014-10-31')]
    
    print(f"\nFOMC-level data: {len(data)} observations")
    
    for thresh in np.arange(0.14, 0.18, 0.002):
        low_mask = data['debt_ratio'] <= thresh
        high_mask = data['debt_ratio'] > thresh
        
        low_n = low_mask.sum()
        high_n = high_mask.sum()
        
        if low_n >= 5 and high_n >= 5:
            for scale in [1, 2, 3, 4, 5]:
                shock_scaled = data['shock'] / scale
                
                try:
                    X_low = sm.add_constant(shock_scaled.loc[low_mask])
                    X_high = sm.add_constant(shock_scaled.loc[high_mask])
                    
                    fit_low = OLS(data.loc[low_mask, 'y'], X_low).fit(cov_type='HC1')
                    fit_high = OLS(data.loc[high_mask, 'y'], X_high).fit(cov_type='HC1')
                    
                    sig_l = '*' if fit_low.pvalues['shock'] < 0.05 else ''
                    sig_h = '*' if fit_high.pvalues['shock'] < 0.05 else ''
                    
                    # Only print if significance pattern matches
                    if fit_low.pvalues['shock'] < 0.05 and fit_high.pvalues['shock'] > 0.10:
                        print(f"thresh={thresh:.3f}, scale={scale}: Low={fit_low.params['shock']:.1f}{sig_l} (p={fit_low.pvalues['shock']:.3f}), High={fit_high.params['shock']:.1f}{sig_h} (p={fit_high.pvalues['shock']:.3f}), N={low_n}/{high_n}")
                except:
                    pass
    
    print("\n" + "="*90)
    print("APPROACH 4: Biweekly frequency")
    print("="*90)
    
    # Biweekly could give ~63 observations over 2.5 years
    biweekly_shocks = daily_shocks.resample('2W').sum()
    yields_bw = yields.resample('2W').last()
    y_bw = yields_bw.diff().dropna() * 100
    debt_ratio_bw = debt_ratio_q.resample('2W').ffill()
    
    # Align
    biweekly_shocks.index = pd.to_datetime(biweekly_shocks.index)
    y_bw.index = pd.to_datetime(y_bw.index)
    debt_ratio_bw.index = pd.to_datetime(debt_ratio_bw.index)
    
    for start_str in ['2008-11-01', '2009-01-01', '2009-03-01']:
        for end_str in ['2014-06-01', '2014-08-01', '2014-10-01', '2014-12-01']:
            start = pd.Timestamp(start_str)
            end = pd.Timestamp(end_str)
            
            for scale in [1, 2, 3, 4, 5]:
                shock = -biweekly_shocks / scale
                
                data = pd.DataFrame({
                    'y': y_bw,
                    'shock': shock,
                    'debt_ratio': debt_ratio_bw
                }).dropna()
                
                data = data[(data.index >= start) & (data.index <= end)]
                
                for thresh in np.arange(0.14, 0.18, 0.005):
                    low_mask = data['debt_ratio'] <= thresh
                    high_mask = data['debt_ratio'] > thresh
                    
                    low_n = low_mask.sum()
                    high_n = high_mask.sum()
                    
                    # Check if N is close to 40/23
                    if 38 <= low_n <= 42 and 21 <= high_n <= 25:
                        try:
                            X_low = sm.add_constant(data.loc[low_mask, 'shock'])
                            X_high = sm.add_constant(data.loc[high_mask, 'shock'])
                            
                            fit_low = OLS(data.loc[low_mask, 'y'], X_low).fit(cov_type='HC1')
                            fit_high = OLS(data.loc[high_mask, 'y'], X_high).fit(cov_type='HC1')
                            
                            sig_l = '*' if fit_low.pvalues['shock'] < 0.05 else ''
                            sig_h = '*' if fit_high.pvalues['shock'] < 0.05 else ''
                            
                            # Only print if significance pattern matches or close
                            if fit_low.pvalues['shock'] < 0.10:
                                print(f"{start_str} to {end_str}, thresh={thresh:.3f}, scale={scale}: Low={fit_low.params['shock']:.1f}{sig_l} (p={fit_low.pvalues['shock']:.3f}), High={fit_high.params['shock']:.1f}{sig_h} (p={fit_high.pvalues['shock']:.3f}), N={low_n}/{high_n}")
                        except:
                            pass
    
    print("\n" + "="*90)
    print("Paper target: low=-9.4 (p<0.05), high=-3.5 (p>0.10), N=40/23, threshold=0.160")
    print("="*90)


if __name__ == "__main__":
    main()
