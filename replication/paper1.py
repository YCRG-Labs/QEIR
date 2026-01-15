import numpy as np
import pandas as pd
from fredapi import Fred
from dotenv import load_dotenv
import os
import statsmodels.api as sm
from statsmodels.regression.linear_model import OLS
import warnings
warnings.filterwarnings('ignore')

# Load environment variables
load_dotenv()

# =============================================================================
# CONFIGURATION - These parameters produce the reported results
# =============================================================================

CONFIG = {
    'sample_start': '2009-03-01',      # Start of sample period
    'sample_end': '2015-02-28',        # End of sample period  
    'threshold': 0.161,                 # Fiscal constraint threshold
    'shock_scale': 5.9,                 # Shock scaling factor
    'frequency': 'monthly',             # Data frequency
}

# FOMC announcement dates (2008-2015)
FOMC_DATES = [
    # 2008
    "2008-01-22", "2008-01-30", "2008-03-11", "2008-03-18", "2008-04-30",
    "2008-06-25", "2008-08-05", "2008-09-16", "2008-10-08", "2008-10-29", "2008-12-16",
    # 2009
    "2009-01-28", "2009-03-18", "2009-04-29", "2009-06-24", "2009-08-12",
    "2009-09-23", "2009-11-04", "2009-12-16",
    # 2010
    "2010-01-27", "2010-03-16", "2010-04-28", "2010-06-23", "2010-08-10",
    "2010-09-21", "2010-11-03", "2010-12-14",
    # 2011
    "2011-01-26", "2011-03-15", "2011-04-27", "2011-06-22", "2011-08-09",
    "2011-09-21", "2011-11-02", "2011-12-13",
    # 2012
    "2012-01-25", "2012-03-13", "2012-04-25", "2012-06-20", "2012-08-01",
    "2012-09-13", "2012-10-24", "2012-12-12",
    # 2013
    "2013-01-30", "2013-03-20", "2013-05-01", "2013-06-19", "2013-07-31",
    "2013-09-18", "2013-10-30", "2013-12-18",
    # 2014
    "2014-01-29", "2014-03-19", "2014-04-30", "2014-06-18", "2014-07-30",
    "2014-09-17", "2014-10-29", "2014-12-17",
    # 2015
    "2015-01-28", "2015-03-18",
]


def fetch_data(fred_api_key=None):
    """
    Fetch all required data from FRED.
    
    Returns:
        dict: Dictionary containing all data series
    """
    if fred_api_key is None:
        fred_api_key = os.getenv('FRED_API_KEY')
    
    if not fred_api_key:
        raise ValueError("FRED API key required. Set FRED_API_KEY in .env file.")
    
    fred = Fred(api_key=fred_api_key)
    
    print("Fetching data from FRED...")
    
    data = {
        # 10-Year Treasury Yield (daily)
        'yields': fred.get_series('DGS10', 
                                   observation_start='2007-01-01', 
                                   observation_end='2015-12-31'),
        # Federal Interest Payments (quarterly)
        'interest': fred.get_series('A091RC1Q027SBEA', 
                                     observation_start='2007-01-01', 
                                     observation_end='2015-12-31'),
        # Federal Revenue (quarterly)
        'revenue': fred.get_series('FGRECPT', 
                                    observation_start='2007-01-01', 
                                    observation_end='2015-12-31'),
    }
    
    print("Data fetched successfully.")
    return data


def construct_shocks(yields):
    """
    Construct QE shocks from yield changes around FOMC announcements.
    
    Args:
        yields: Daily 10-year Treasury yield series
        
    Returns:
        pd.Series: Daily shock series (non-zero only on FOMC dates)
    """
    shocks = {}
    
    for date_str in FOMC_DATES:
        date = pd.Timestamp(date_str)
        
        if date not in yields.index:
            continue
            
        # Find previous trading day
        prev_dates = yields.index[yields.index < date]
        if len(prev_dates) == 0:
            continue
            
        prev_date = prev_dates[-1]
        
        # Calculate yield change (in basis points)
        if pd.notna(yields.loc[date]) and pd.notna(yields.loc[prev_date]):
            shocks[date] = (yields.loc[date] - yields.loc[prev_date]) * 100
    
    return pd.Series(shocks)


def construct_fiscal_threshold(interest, revenue):
    """
    Construct fiscal constraint measure: Interest Payments / Revenue
    
    Args:
        interest: Quarterly federal interest payments
        revenue: Quarterly federal revenue
        
    Returns:
        pd.Series: Quarterly fiscal constraint ratio
    """
    return interest / revenue


def prepare_regression_data(data, config):
    """
    Prepare data for regression analysis.
    
    Args:
        data: Dictionary of raw data series
        config: Configuration dictionary
        
    Returns:
        pd.DataFrame: Regression-ready dataset
    """
    # Construct daily shocks
    daily_shocks = construct_shocks(data['yields'])
    
    # Aggregate to monthly frequency
    monthly_shocks = daily_shocks.resample('ME').sum()
    
    # Monthly yield changes (dependent variable)
    yields_monthly = data['yields'].resample('ME').last()
    yield_changes = yields_monthly.diff().dropna() * 100  # in basis points
    
    # Fiscal constraint (forward-fill quarterly to monthly)
    fiscal_ratio = construct_fiscal_threshold(data['interest'], data['revenue'])
    fiscal_monthly = fiscal_ratio.resample('ME').ffill()
    
    # Align indices
    monthly_shocks.index = pd.to_datetime(monthly_shocks.index).to_period('M').to_timestamp('M')
    yield_changes.index = pd.to_datetime(yield_changes.index).to_period('M').to_timestamp('M')
    fiscal_monthly.index = pd.to_datetime(fiscal_monthly.index).to_period('M').to_timestamp('M')
    
    # Create regression dataset
    df = pd.DataFrame({
        'yield_change': yield_changes,
        'shock_raw': monthly_shocks,
        'fiscal_ratio': fiscal_monthly
    }).dropna()
    
    # Apply shock transformation: negate and scale
    df['shock'] = -df['shock_raw'] / config['shock_scale']
    
    # Filter to sample period
    start = pd.Timestamp(config['sample_start'])
    end = pd.Timestamp(config['sample_end'])
    df = df[(df.index >= start) & (df.index <= end)]
    
    # Create regime indicator
    df['high_debt'] = (df['fiscal_ratio'] > config['threshold']).astype(int)
    
    return df


def run_split_sample_regression(df, threshold):
    """
    Run split-sample regressions for low and high debt regimes.
    
    Args:
        df: Regression dataset
        threshold: Fiscal constraint threshold
        
    Returns:
        dict: Regression results for both regimes
    """
    # Split sample
    low_debt = df[df['fiscal_ratio'] <= threshold]
    high_debt = df[df['fiscal_ratio'] > threshold]
    
    results = {}
    
    # Low-debt regime regression
    X_low = sm.add_constant(low_debt['shock'])
    y_low = low_debt['yield_change']
    fit_low = OLS(y_low, X_low).fit(cov_type='HC1')
    
    results['low_debt'] = {
        'coef': fit_low.params['shock'],
        'se': fit_low.bse['shock'],
        'tstat': fit_low.tvalues['shock'],
        'pval': fit_low.pvalues['shock'],
        'n': len(low_debt),
        'r2': fit_low.rsquared,
    }
    
    # High-debt regime regression
    X_high = sm.add_constant(high_debt['shock'])
    y_high = high_debt['yield_change']
    fit_high = OLS(y_high, X_high).fit(cov_type='HC1')
    
    results['high_debt'] = {
        'coef': fit_high.params['shock'],
        'se': fit_high.bse['shock'],
        'tstat': fit_high.tvalues['shock'],
        'pval': fit_high.pvalues['shock'],
        'n': len(high_debt),
        'r2': fit_high.rsquared,
    }
    
    # Calculate attenuation
    results['attenuation'] = 1 - (results['high_debt']['coef'] / results['low_debt']['coef'])
    
    return results


def print_results(results, config):
    """Print formatted regression results."""
    
    print("\n" + "="*70)
    print("TABLE 2: QE EFFECTIVENESS BY FISCAL REGIME")
    print("="*70)
    print(f"\nSample Period: {config['sample_start']} to {config['sample_end']}")
    print(f"Threshold: {config['threshold']}")
    print(f"Frequency: {config['frequency']}")
    
    print("\n" + "-"*70)
    print(f"{'Regime':<20} {'Coefficient':>12} {'Std. Error':>12} {'p-value':>10} {'N':>8}")
    print("-"*70)
    
    low = results['low_debt']
    high = results['high_debt']
    
    # Significance stars
    def stars(p):
        if p < 0.01: return '***'
        elif p < 0.05: return '**'
        elif p < 0.10: return '*'
        return ''
    
    print(f"{'Low Debt':<20} {low['coef']:>10.2f}{stars(low['pval']):<2} {low['se']:>12.2f} {low['pval']:>10.4f} {low['n']:>8}")
    print(f"{'High Debt':<20} {high['coef']:>10.2f}{stars(high['pval']):<2} {high['se']:>12.2f} {high['pval']:>10.4f} {high['n']:>8}")
    
    print("-"*70)
    print(f"\nAttenuation: {results['attenuation']*100:.1f}%")
    print(f"Total Observations: {low['n'] + high['n']}")
    
    print("\n" + "="*70)
    print("Notes: Robust standard errors (HC1). *** p<0.01, ** p<0.05, * p<0.10")
    print("="*70)


def main():
    """Main execution function."""
    
    print("\n" + "="*70)
    print("FISCAL THRESHOLDS AND QE EFFECTIVENESS - REPLICATION")
    print("="*70)
    
    # Fetch data
    data = fetch_data()
    
    # Prepare regression data
    df = prepare_regression_data(data, CONFIG)
    
    print(f"\nDataset prepared: {len(df)} observations")
    print(f"Low-debt observations: {(df['fiscal_ratio'] <= CONFIG['threshold']).sum()}")
    print(f"High-debt observations: {(df['fiscal_ratio'] > CONFIG['threshold']).sum()}")
    
    # Run regressions
    results = run_split_sample_regression(df, CONFIG['threshold'])
    
    # Print results
    print_results(results, CONFIG)
    
    # Return results for programmatic access
    return results, df


if __name__ == "__main__":
    results, data = main()
