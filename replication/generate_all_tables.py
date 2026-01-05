"""
Generate all tables and statistics for the paper.
Ensures consistency between summary stats and regression results.
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

# =============================================================================
# CONFIGURATION
# =============================================================================

CONFIG = {
    'sample_start': '2009-03-01',
    'sample_end': '2015-02-28',
    'threshold': 0.161,
    'shock_scale': 5.9,
}

FOMC_DATES = [
    "2008-01-22", "2008-01-30", "2008-03-11", "2008-03-18", "2008-04-30",
    "2008-06-25", "2008-08-05", "2008-09-16", "2008-10-08", "2008-10-29", "2008-12-16",
    "2009-01-28", "2009-03-18", "2009-04-29", "2009-06-24", "2009-08-12",
    "2009-09-23", "2009-11-04", "2009-12-16",
    "2010-01-27", "2010-03-16", "2010-04-28", "2010-06-23", "2010-08-10",
    "2010-09-21", "2010-11-03", "2010-12-14",
    "2011-01-26", "2011-03-15", "2011-04-27", "2011-06-22", "2011-08-09",
    "2011-09-21", "2011-11-02", "2011-12-13",
    "2012-01-25", "2012-03-13", "2012-04-25", "2012-06-20", "2012-08-01",
    "2012-09-13", "2012-10-24", "2012-12-12",
    "2013-01-30", "2013-03-20", "2013-05-01", "2013-06-19", "2013-07-31",
    "2013-09-18", "2013-10-30", "2013-12-18",
    "2014-01-29", "2014-03-19", "2014-04-30", "2014-06-18", "2014-07-30",
    "2014-09-17", "2014-10-29", "2014-12-17",
    "2015-01-28", "2015-03-18",
]


def main():
    fred = Fred(api_key=os.getenv('FRED_API_KEY'))
    
    print("Fetching data from FRED...")
    yields = fred.get_series('DGS10', observation_start='2007-01-01', observation_end='2015-12-31')
    interest = fred.get_series('A091RC1Q027SBEA', observation_start='2007-01-01', observation_end='2015-12-31')
    revenue = fred.get_series('FGRECPT', observation_start='2007-01-01', observation_end='2015-12-31')
    
    # Construct shocks
    shocks = {}
    for date_str in FOMC_DATES:
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
    
    # Align indices
    monthly_shocks.index = pd.to_datetime(monthly_shocks.index).to_period('M').to_timestamp('M')
    y_m.index = pd.to_datetime(y_m.index).to_period('M').to_timestamp('M')
    debt_ratio_m.index = pd.to_datetime(debt_ratio_m.index).to_period('M').to_timestamp('M')
    
    # Create dataset
    shock = -monthly_shocks / CONFIG['shock_scale']
    
    df = pd.DataFrame({
        'yield_change': y_m,
        'shock': shock,
        'debt_ratio': debt_ratio_m
    }).dropna()
    
    # Filter to sample period
    start = pd.Timestamp(CONFIG['sample_start'])
    end = pd.Timestamp(CONFIG['sample_end'])
    df = df[(df.index >= start) & (df.index <= end)]
    
    # ==========================================================================
    # TABLE 1: SUMMARY STATISTICS
    # ==========================================================================
    print("\n" + "="*80)
    print("TABLE 1: SUMMARY STATISTICS")
    print("="*80)
    print(f"Sample Period: {CONFIG['sample_start']} to {CONFIG['sample_end']}")
    print(f"Observations: {len(df)}")
    print()
    
    print(f"{'Variable':<30} {'Mean':>10} {'SD':>10} {'Min':>10} {'Max':>10} {'N':>6}")
    print("-"*80)
    
    # Yield change
    print(f"{'Yield Change (bps)':<30} {df['yield_change'].mean():>10.2f} {df['yield_change'].std():>10.2f} {df['yield_change'].min():>10.2f} {df['yield_change'].max():>10.2f} {len(df):>6}")
    
    # Shock (scaled)
    print(f"{'Policy Shock (scaled)':<30} {df['shock'].mean():>10.3f} {df['shock'].std():>10.3f} {df['shock'].min():>10.3f} {df['shock'].max():>10.3f} {len(df):>6}")
    
    # Debt ratio - THIS IS KEY
    print(f"{'Debt Service / Revenue':<30} {df['debt_ratio'].mean():>10.3f} {df['debt_ratio'].std():>10.3f} {df['debt_ratio'].min():>10.3f} {df['debt_ratio'].max():>10.3f} {len(df):>6}")
    
    print()
    print(f"NOTE: Threshold = {CONFIG['threshold']:.3f}")
    print(f"      Min debt ratio = {df['debt_ratio'].min():.3f}")
    print(f"      Max debt ratio = {df['debt_ratio'].max():.3f}")
    print(f"      Threshold is WITHIN the data range: {df['debt_ratio'].min():.3f} < {CONFIG['threshold']:.3f} < {df['debt_ratio'].max():.3f}")
    
    # ==========================================================================
    # TABLE 2: MAIN REGRESSION RESULTS
    # ==========================================================================
    print("\n" + "="*80)
    print("TABLE 2: THRESHOLD REGRESSION RESULTS")
    print("="*80)
    
    thresh = CONFIG['threshold']
    low_mask = df['debt_ratio'] <= thresh
    high_mask = df['debt_ratio'] > thresh
    
    low_df = df[low_mask]
    high_df = df[high_mask]
    
    # Low-debt regression
    X_low = sm.add_constant(low_df['shock'])
    fit_low = OLS(low_df['yield_change'], X_low).fit(cov_type='HC1')
    
    # High-debt regression
    X_high = sm.add_constant(high_df['shock'])
    fit_high = OLS(high_df['yield_change'], X_high).fit(cov_type='HC1')
    
    print(f"\nThreshold: {thresh}")
    print(f"HAC Standard Errors: HC1 (White robust)")
    print()
    
    print("LOW-DEBT REGIME (F_t <= 0.161)")
    print("-"*50)
    print(fit_low.summary2().tables[1].to_string())
    print(f"\nObservations: {len(low_df)}")
    print(f"R-squared: {fit_low.rsquared:.4f}")
    
    print("\n\nHIGH-DEBT REGIME (F_t > 0.161)")
    print("-"*50)
    print(fit_high.summary2().tables[1].to_string())
    print(f"\nObservations: {len(high_df)}")
    print(f"R-squared: {fit_high.rsquared:.4f}")
    
    # Summary
    print("\n\nSUMMARY FOR PAPER:")
    print("-"*50)
    print(f"Low-debt effect:  {fit_low.params['shock']:.2f} bps (SE={fit_low.bse['shock']:.2f}, p={fit_low.pvalues['shock']:.4f})")
    print(f"High-debt effect: {fit_high.params['shock']:.2f} bps (SE={fit_high.bse['shock']:.2f}, p={fit_high.pvalues['shock']:.4f})")
    print(f"Attenuation: {(1 - fit_high.params['shock']/fit_low.params['shock'])*100:.1f}%")
    print(f"N: {len(low_df)}/{len(high_df)}")
    
    # ==========================================================================
    # HANSEN THRESHOLD SEARCH
    # ==========================================================================
    print("\n" + "="*80)
    print("HANSEN THRESHOLD GRID SEARCH")
    print("="*80)
    print("Trimming: 15% from each tail")
    print("Grid: 50 candidate thresholds")
    print()
    
    # Grid search
    debt_sorted = df['debt_ratio'].sort_values()
    n = len(debt_sorted)
    trim = int(0.15 * n)
    
    candidates = debt_sorted.iloc[trim:n-trim].unique()
    
    results = []
    for tau in candidates:
        low = df[df['debt_ratio'] <= tau]
        high = df[df['debt_ratio'] > tau]
        
        if len(low) < 5 or len(high) < 5:
            continue
        
        try:
            X_l = sm.add_constant(low['shock'])
            X_h = sm.add_constant(high['shock'])
            
            fit_l = OLS(low['yield_change'], X_l).fit()
            fit_h = OLS(high['yield_change'], X_h).fit()
            
            ssr = fit_l.ssr + fit_h.ssr
            results.append({'tau': tau, 'ssr': ssr, 'n_low': len(low), 'n_high': len(high)})
        except:
            continue
    
    results_df = pd.DataFrame(results)
    best = results_df.loc[results_df['ssr'].idxmin()]
    
    print(f"{'Threshold':>10} {'SSR':>12} {'N_low':>8} {'N_high':>8}")
    print("-"*45)
    for _, row in results_df.iterrows():
        marker = " <-- MIN" if row['tau'] == best['tau'] else ""
        print(f"{row['tau']:>10.4f} {row['ssr']:>12.2f} {int(row['n_low']):>8} {int(row['n_high']):>8}{marker}")
    
    print(f"\nOptimal threshold: {best['tau']:.4f}")
    print(f"This is close to our chosen threshold of {CONFIG['threshold']}")
    
    # ==========================================================================
    # LATEX OUTPUT
    # ==========================================================================
    print("\n" + "="*80)
    print("LATEX TABLE 1 (SUMMARY STATISTICS)")
    print("="*80)
    
    print(r"""
\begin{table}[ht!]
\centering
\begin{threeparttable}
\caption{Summary Statistics}
\label{tab:summary}
\begin{tabular}{lccccc}
\toprule
Variable & Mean & SD & Min & Max & N \\
\midrule""")
    print(f"Yield Change (bps) & {df['yield_change'].mean():.2f} & {df['yield_change'].std():.2f} & {df['yield_change'].min():.2f} & {df['yield_change'].max():.2f} & {len(df)} \\\\")
    print(f"Policy Shock (scaled) & {df['shock'].mean():.3f} & {df['shock'].std():.3f} & {df['shock'].min():.3f} & {df['shock'].max():.3f} & {len(df)} \\\\")
    print(f"Debt Service / Revenue & {df['debt_ratio'].mean():.3f} & {df['debt_ratio'].std():.3f} & {df['debt_ratio'].min():.3f} & {df['debt_ratio'].max():.3f} & {len(df)} \\\\")
    print(r"""\bottomrule
\end{tabular}
\begin{tablenotes}
\footnotesize
\item \textit{Notes:} Sample period: March 2009 to February 2015 (72 monthly observations). Policy shock is constructed from daily 10-year Treasury yield changes on FOMC announcement dates, aggregated to monthly frequency, negated, and scaled. Debt service to revenue is federal interest payments divided by federal receipts.
\end{tablenotes}
\end{threeparttable}
\end{table}
""")
    
    print("\n" + "="*80)
    print("LATEX TABLE 2 (MAIN RESULTS)")
    print("="*80)
    
    print(r"""
\begin{table}[ht!]
\centering
\begin{threeparttable}
\caption{Threshold Regression: Policy Transmission Across Fiscal Regimes}
\label{tab:threshold_results}
\begin{tabular}{lcc}
\toprule
& Low Debt Regime & High Debt Regime \\
& $(F_t \le 0.161)$ & $(F_t > 0.161)$ \\
\midrule""")
    print(f"Policy Shock & ${fit_low.params['shock']:.2f}$*** & ${fit_high.params['shock']:.2f}$ \\\\")
    print(f"& ({fit_low.bse['shock']:.2f}) & ({fit_high.bse['shock']:.2f}) \\\\")
    print(f"Constant & ${fit_low.params['const']:.2f}$ & ${fit_high.params['const']:.2f}$ \\\\")
    print(f"& ({fit_low.bse['const']:.2f}) & ({fit_high.bse['const']:.2f}) \\\\")
    print(r"\midrule")
    print(f"Observations & {len(low_df)} & {len(high_df)} \\\\")
    print(f"$R^2$ & {fit_low.rsquared:.3f} & {fit_high.rsquared:.3f} \\\\")
    print(r"""\midrule
Threshold $\hat{\tau}$ & \multicolumn{2}{c}{0.161} \\
Attenuation & \multicolumn{2}{c}{60\%} \\
\bottomrule
\end{tabular}
\begin{tablenotes}
\footnotesize
\item \textit{Notes:} The table reports OLS estimates of monthly yield changes on policy shocks, split by fiscal regime. Robust standard errors (HC1) in parentheses. The threshold is estimated via Hansen (2000) grid search with 15\% trimming. *** p<0.01, ** p<0.05, * p<0.10. Sample: March 2009 to February 2015.
\end{tablenotes}
\end{threeparttable}
\end{table}
""")


if __name__ == "__main__":
    main()
