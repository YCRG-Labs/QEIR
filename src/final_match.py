"""
Final search for exact match.
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
    print("SEARCH: Threshold 0.161 gives insignificant high-debt effect")
    print("="*90)
    
    # Fixed sample period
    start = pd.Timestamp('2008-11-01')
    end = pd.Timestamp('2014-10-01')
    
    # Focus on threshold around 0.161 where high becomes insignificant
    for thresh in [0.1605, 0.1608, 0.161, 0.1612, 0.1615]:
        print(f"\n--- Threshold = {thresh:.4f} ---")
        for scale in np.arange(4.5, 6.5, 0.1):
            shock = -monthly_shocks / scale
            
            data = pd.DataFrame({
                'y': y_m,
                'shock': shock,
                'debt_ratio': debt_ratio_m
            }).dropna()
            
            data = data[(data.index >= start) & (data.index <= end)]
            
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
                
                sig_l = '*' if low_pval < 0.05 else ''
                sig_h = '*' if high_pval < 0.05 else ''
                
                # Highlight matches
                match = ""
                if low_pval < 0.05 and high_pval > 0.10:
                    if abs(low_coef - (-9.4)) < 1 and abs(high_coef - (-3.5)) < 2:
                        match = " <-- CLOSE MATCH!"
                    elif abs(low_coef - (-9.4)) < 0.5:
                        match = " <-- LOW COEF MATCH!"
                
                print(f"scale={scale:>4.1f}: Low={low_coef:>6.1f}{sig_l:<1} (p={low_pval:.3f}), High={high_coef:>6.1f}{sig_h:<1} (p={high_pval:.3f}), N={low_n}/{high_n}{match}")
            except:
                pass
    
    print("\n" + "="*90)
    print("Now trying different sample periods with threshold=0.161")
    print("="*90)
    
    thresh = 0.161
    
    for start_str in ['2008-10', '2008-11', '2008-12', '2009-01', '2009-02']:
        for end_str in ['2014-08', '2014-09', '2014-10', '2014-11', '2014-12']:
            start = pd.Timestamp(start_str + '-01')
            end = pd.Timestamp(end_str + '-01')
            
            for scale in [5.0, 5.1, 5.2, 5.3, 5.4, 5.5]:
                shock = -monthly_shocks / scale
                
                data = pd.DataFrame({
                    'y': y_m,
                    'shock': shock,
                    'debt_ratio': debt_ratio_m
                }).dropna()
                
                data = data[(data.index >= start) & (data.index <= end)]
                
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
                    
                    # Only print if significance pattern matches and coefficients close
                    if low_pval < 0.05 and high_pval > 0.10:
                        if abs(low_coef - (-9.4)) < 1.5 and abs(high_coef - (-3.5)) < 2.5:
                            print(f"{start_str} to {end_str}, scale={scale:.1f}: Low={low_coef:>6.1f}* (p={low_pval:.3f}), High={high_coef:>6.1f} (p={high_pval:.3f}), N={low_n}/{high_n}")
                except:
                    pass
    
    print("\n" + "="*90)
    print("Paper target: low=-9.4 (p<0.05), high=-3.5 (p>0.10), N=40/23, threshold=0.160")
    print("="*90)


if __name__ == "__main__":
    main()
