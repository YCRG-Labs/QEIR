"""
Search for N=40/23 with quarterly data.
N=40+23=63 quarters = ~15.75 years, which is too long.
But maybe they used a different definition of "observations"?
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
    print("QUARTERLY DATA: Checking what N values are possible")
    print("="*90)
    
    # Check all possible sample periods
    for start_year in range(2008, 2012):
        for start_q in range(1, 5):
            for end_year in range(2013, 2024):
                for end_q in range(1, 5):
                    start = pd.Timestamp(f'{start_year}Q{start_q}')
                    end = pd.Timestamp(f'{end_year}Q{end_q}')
                    
                    if end <= start:
                        continue
                    
                    for thresh in np.arange(0.14, 0.18, 0.005):
                        data = pd.DataFrame({
                            'y': y_q,
                            'shock': -quarterly_shocks,
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
                        if 35 <= low_n <= 45 and 18 <= high_n <= 28:
                            print(f"{start_year}Q{start_q} to {end_year}Q{end_q}, thresh={thresh:.3f}: N={low_n}/{high_n}")
    
    print("\n" + "="*90)
    print("FOMC-LEVEL DATA: Each FOMC meeting as one observation")
    print("="*90)
    
    # Use FOMC dates directly
    fomc_y = pd.Series(index=daily_shocks.index, dtype=float)
    for date in daily_shocks.index:
        if date in yields.index:
            prev_dates = yields.index[yields.index < date]
            if len(prev_dates) > 0:
                prev_date = prev_dates[-1]
                if pd.notna(yields.loc[date]) and pd.notna(yields.loc[prev_date]):
                    fomc_y.loc[date] = (yields.loc[date] - yields.loc[prev_date]) * 100
    
    fomc_debt = pd.Series(index=daily_shocks.index, dtype=float)
    for date in daily_shocks.index:
        prev_q = debt_ratio_q.index[debt_ratio_q.index <= date]
        if len(prev_q) > 0:
            fomc_debt.loc[date] = debt_ratio_q.loc[prev_q[-1]]
    
    data = pd.DataFrame({
        'y': fomc_y,
        'shock': -daily_shocks,
        'debt_ratio': fomc_debt
    }).dropna()
    
    print(f"\nTotal FOMC observations: {len(data)}")
    
    # Check different sample periods
    for start_str in ['2008-01', '2008-06', '2008-11', '2009-01', '2009-06']:
        for end_str in ['2013-12', '2014-06', '2014-12', '2015-06', '2015-12', '2016-06']:
            start = pd.Timestamp(start_str + '-01')
            end = pd.Timestamp(end_str + '-01')
            
            subset = data[(data.index >= start) & (data.index <= end)]
            
            for thresh in np.arange(0.14, 0.18, 0.005):
                low_mask = subset['debt_ratio'] <= thresh
                high_mask = subset['debt_ratio'] > thresh
                
                low_n = low_mask.sum()
                high_n = high_mask.sum()
                
                if 35 <= low_n <= 45 and 18 <= high_n <= 28:
                    print(f"{start_str} to {end_str}, thresh={thresh:.3f}: N={low_n}/{high_n}")
    
    print("\n" + "="*90)
    print("Paper target: N=40/23")
    print("="*90)


if __name__ == "__main__":
    main()
