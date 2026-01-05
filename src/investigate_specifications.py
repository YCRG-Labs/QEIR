"""
Investigation Script: Systematically test different specifications
to understand what could have produced the paper's claimed results.

Paper claims:
- Threshold: ~0.160
- Low-debt QE effect: -9.4 bps
- High-debt QE effect: -3.5 bps  
- Hansen p-value: 0.012
- Attenuation: ~63%
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

def fetch_data():
    """Fetch all required FRED data."""
    fred = Fred(api_key=os.getenv('FRED_API_KEY'))
    
    start = '2008-01-01'
    end = '2023-12-31'
    
    data = {
        'yields': fred.get_series('DGS10', observation_start=start, observation_end=end),
        'investment': fred.get_series('GPDIC1', observation_start=start, observation_end=end, frequency='q'),
        'gdp': fred.get_series('GDPC1', observation_start=start, observation_end=end, frequency='q'),
        'unemployment': fred.get_series('UNRATE', observation_start=start, observation_end=end, frequency='q'),
        'inflation': fred.get_series('PCEPILFE', observation_start=start, observation_end=end, frequency='q'),
        'interest_payments': fred.get_series('A091RC1Q027SBEA', observation_start=start, observation_end=end, frequency='q'),
        'revenue': fred.get_series('FGRECPT', observation_start=start, observation_end=end, frequency='q'),
        'vix': fred.get_series('VIXCLS', observation_start=start, observation_end=end),
    }
    
    return data

def construct_shocks(yields, fomc_dates):
    """Construct QE shocks from yield changes on FOMC dates."""
    shocks = {}
    for date_str in fomc_dates:
        date = pd.Timestamp(date_str)
        if date in yields.index:
            prev_dates = yields.index[yields.index < date]
            if len(prev_dates) > 0:
                prev_date = prev_dates[-1]
                if pd.notna(yields.loc[date]) and pd.notna(yields.loc[prev_date]):
                    shock = (yields.loc[date] - yields.loc[prev_date]) * 100  # bps
                    shocks[date] = shock
    return pd.Series(shocks, name='qe_shock')

def simple_threshold_regression(y, X, threshold_var, threshold):
    """Simple threshold regression at a given threshold."""
    data = pd.concat([y, X, threshold_var], axis=1).dropna()
    y_clean = data.iloc[:, 0]
    X_clean = data.iloc[:, 1:-1]
    thresh_clean = data.iloc[:, -1]
    
    low_mask = thresh_clean <= threshold
    high_mask = thresh_clean > threshold
    
    X_with_const = sm.add_constant(X_clean)
    
    results = {}
    for regime, mask in [('low', low_mask), ('high', high_mask)]:
        if mask.sum() >= X_clean.shape[1] + 2:
            model = OLS(y_clean[mask], X_with_const[mask])
            fit = model.fit(cov_type='HC1')
            results[regime] = {
                'coef': fit.params.get('qe_shock', fit.params.iloc[1]),
                'se': fit.bse.get('qe_shock', fit.bse.iloc[1]),
                'n': int(fit.nobs),
                'r2': fit.rsquared
            }
    return results

def test_specification(data, spec_name, y_transform, shock_transform, threshold_var_transform, 
                       use_controls=True, threshold_override=None):
    """Test a specific specification."""
    
    fomc_dates = get_fomc_dates(2008, 2023)
    
    # Construct shocks
    daily_shocks = construct_shocks(data['yields'], fomc_dates)
    quarterly_shocks = daily_shocks.resample('QE').sum()
    
    # Apply shock transform
    shock = shock_transform(quarterly_shocks)
    
    # Construct dependent variable
    y = y_transform(data)
    
    # Construct threshold variable
    thresh_var = threshold_var_transform(data)
    
    # Align indices
    shock.index = pd.to_datetime(shock.index).to_period('Q').to_timestamp('Q')
    y.index = pd.to_datetime(y.index).to_period('Q').to_timestamp('Q')
    thresh_var.index = pd.to_datetime(thresh_var.index).to_period('Q').to_timestamp('Q')
    
    # Build X matrix
    if use_controls:
        gdp_growth = data['gdp'].pct_change() * 100
        gdp_growth.index = pd.to_datetime(gdp_growth.index).to_period('Q').to_timestamp('Q')
        unemp = data['unemployment'].copy()
        unemp.index = pd.to_datetime(unemp.index).to_period('Q').to_timestamp('Q')
        
        X = pd.DataFrame({
            'qe_shock': shock,
            'gdp_growth': gdp_growth,
            'unemployment': unemp
        })
    else:
        X = pd.DataFrame({'qe_shock': shock})
    
    # Find optimal threshold via grid search
    data_aligned = pd.concat([y, X, thresh_var], axis=1).dropna()
    thresh_values = data_aligned.iloc[:, -1]
    
    lower = thresh_values.quantile(0.15)
    upper = thresh_values.quantile(0.85)
    
    best_ssr = np.inf
    best_threshold = None
    
    for gamma in np.linspace(lower, upper, 50):
        low_mask = thresh_values <= gamma
        high_mask = thresh_values > gamma
        
        if low_mask.sum() < 10 or high_mask.sum() < 10:
            continue
            
        try:
            y_clean = data_aligned.iloc[:, 0]
            X_clean = sm.add_constant(data_aligned.iloc[:, 1:-1])
            
            ssr_low = OLS(y_clean[low_mask], X_clean[low_mask]).fit().ssr
            ssr_high = OLS(y_clean[high_mask], X_clean[high_mask]).fit().ssr
            total_ssr = ssr_low + ssr_high
            
            if total_ssr < best_ssr:
                best_ssr = total_ssr
                best_threshold = gamma
        except:
            continue
    
    if threshold_override:
        best_threshold = threshold_override
    
    if best_threshold is None:
        return None
        
    # Get results at best threshold
    results = simple_threshold_regression(y, X, thresh_var, best_threshold)
    
    return {
        'spec': spec_name,
        'threshold': best_threshold,
        'low_coef': results.get('low', {}).get('coef', np.nan),
        'low_se': results.get('low', {}).get('se', np.nan),
        'low_n': results.get('low', {}).get('n', 0),
        'high_coef': results.get('high', {}).get('coef', np.nan),
        'high_se': results.get('high', {}).get('se', np.nan),
        'high_n': results.get('high', {}).get('n', 0),
    }


def main():
    print("Fetching data...")
    data = fetch_data()
    
    # Define different transforms to test
    specifications = []
    
    # Y transforms (dependent variable)
    def y_yield_change(d):
        q = d['yields'].resample('QE').last()
        return q.diff().dropna() * 100  # bps
    
    def y_yield_change_mean(d):
        q = d['yields'].resample('QE').mean()
        return q.diff().dropna() * 100
    
    def y_yield_level(d):
        return d['yields'].resample('QE').last()
    
    def y_investment_growth(d):
        return np.log(d['investment']).diff().dropna() * 100
    
    # Shock transforms
    def shock_raw(s): return s
    def shock_std(s): return (s - s.mean()) / s.std()
    def shock_neg(s): return -s
    def shock_neg_std(s): return -1 * (s - s.mean()) / s.std()
    
    # Threshold variable transforms
    def thresh_debt_ratio(d):
        return d['interest_payments'] / d['revenue']
    
    def thresh_debt_ratio_lag(d):
        return (d['interest_payments'] / d['revenue']).shift(1)
    
    # Test many combinations
    y_transforms = [
        ('Δyield_last', y_yield_change),
        ('Δyield_mean', y_yield_change_mean),
        ('yield_level', y_yield_level),
        ('Δlog_inv', y_investment_growth),
    ]
    
    shock_transforms = [
        ('raw', shock_raw),
        ('std', shock_std),
        ('neg', shock_neg),
        ('neg_std', shock_neg_std),
    ]
    
    thresh_transforms = [
        ('debt_ratio', thresh_debt_ratio),
        ('debt_ratio_lag', thresh_debt_ratio_lag),
    ]
    
    control_options = [True, False]
    
    print("\nTesting specifications...\n")
    print(f"{'Specification':<50} {'Threshold':>10} {'Low β':>10} {'High β':>10} {'Low n':>7} {'High n':>7} {'Attenuates?':>12}")
    print("=" * 110)
    
    results = []
    
    for y_name, y_fn in y_transforms:
        for s_name, s_fn in shock_transforms:
            for t_name, t_fn in thresh_transforms:
                for use_ctrl in control_options:
                    ctrl_str = '+ctrl' if use_ctrl else ''
                    spec_name = f"{y_name}|{s_name}|{t_name}{ctrl_str}"
                    
                    try:
                        result = test_specification(
                            data, spec_name, y_fn, s_fn, t_fn, use_ctrl
                        )
                        
                        if result and not np.isnan(result['low_coef']):
                            results.append(result)
                            
                            # Check if this shows attenuation (smaller magnitude in high regime)
                            attenuates = abs(result['high_coef']) < abs(result['low_coef'])
                            att_str = "YES" if attenuates else "no"
                            
                            # Check if signs are both negative (as paper claims)
                            both_neg = result['low_coef'] < 0 and result['high_coef'] < 0
                            
                            # Highlight promising results
                            marker = ""
                            if attenuates and both_neg:
                                marker = " ***"
                            elif attenuates:
                                marker = " *"
                            
                            print(f"{spec_name:<50} {result['threshold']:>10.4f} {result['low_coef']:>10.2f} {result['high_coef']:>10.2f} {result['low_n']:>7} {result['high_n']:>7} {att_str:>12}{marker}")
                    except Exception as e:
                        pass
    
    # Summary
    print("\n" + "=" * 110)
    print("\nPaper claims: threshold=0.160, low=-9.4bps, high=-3.5bps (63% attenuation)")
    print("\nSpecs with attenuation AND both negative coefficients (marked ***):")
    
    for r in results:
        if r['low_coef'] < 0 and r['high_coef'] < 0:
            if abs(r['high_coef']) < abs(r['low_coef']):
                attenuation = (1 - abs(r['high_coef']) / abs(r['low_coef'])) * 100
                print(f"  {r['spec']}: τ={r['threshold']:.3f}, low={r['low_coef']:.1f}, high={r['high_coef']:.1f}, atten={attenuation:.0f}%")


if __name__ == "__main__":
    main()
