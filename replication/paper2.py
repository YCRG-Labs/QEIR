import os
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    print("Warning: python-dotenv not installed. Set FRED_API_KEY manually.")

try:
    from fredapi import Fred
except ImportError:
    raise ImportError("Please install fredapi: pip install fredapi")

try:
    import statsmodels.api as sm
    from statsmodels.regression.linear_model import OLS
except ImportError:
    raise ImportError("Please install statsmodels: pip install statsmodels")


# =============================================================================
# CONFIGURATION
# =============================================================================

CONFIG = {
    'sample_start': '2008-01-01',
    'sample_end': '2023-12-31',  # Full sample as in paper
    'frequency': 'quarterly',
    'local_projection_horizons': 18,  # months
}

# FRED Series IDs from Table 1 of the paper
FRED_SERIES = {
    # Foreign Holdings
    'official_holdings': 'FDHBPIN',    # Foreign official holdings of U.S. Treasuries
    'private_holdings': 'FDHBFIN',     # Foreign private holdings of U.S. Treasuries
    
    # Fed Balance Sheet
    'fed_holdings': 'TREAST',          # Federal Reserve Treasury holdings
    
    # Fiscal Variables
    'total_debt': 'GFDEBTN',           # Total public debt outstanding
    
    # Exchange Rates
    'trade_weighted_usd': 'DTWEXBGS',  # Broad trade-weighted dollar index
    
    # Interest Rates
    'treasury_10y': 'DGS10',           # 10-Year Treasury constant maturity yield
}


# =============================================================================
# DATA COLLECTION
# =============================================================================

class FREDDataCollector:
    """Collects data from FRED for the paper methodology."""
    
    def __init__(self, api_key: Optional[str] = None):
        """Initialize with FRED API key."""
        self.api_key = api_key or os.getenv('FRED_API_KEY')
        if not self.api_key:
            raise ValueError(
                "FRED API key required. Set FRED_API_KEY in .env file or pass directly.\n"
                "Get a free key at: https://fred.stlouisfed.org/docs/api/api_key.html"
            )
        self.fred = Fred(api_key=self.api_key)
        
    def fetch_series(self, series_id: str, start: str, end: str) -> pd.Series:
        """Fetch a single series from FRED."""
        try:
            series = self.fred.get_series(
                series_id,
                observation_start=start,
                observation_end=end
            )
            print(f"  Downloaded {series_id}: {len(series)} observations")
            return series
        except Exception as e:
            print(f"  Error downloading {series_id}: {e}")
            return pd.Series(dtype=float)
    
    def fetch_all_data(self) -> Dict[str, pd.Series]:
        """Fetch all required data series."""
        print("\n" + "="*70)
        print("STEP 1: Fetching Data from FRED")
        print("="*70)
        
        data = {}
        for name, series_id in FRED_SERIES.items():
            data[name] = self.fetch_series(
                series_id,
                CONFIG['sample_start'],
                CONFIG['sample_end']
            )
        
        return data


# =============================================================================
# DATA PROCESSING
# =============================================================================

class DataProcessor:
    """Processes raw FRED data for analysis."""
    
    @staticmethod
    def to_quarterly(series: pd.Series, method: str = 'last') -> pd.Series:
        """Convert series to quarterly frequency."""
        if series.empty:
            return series
        
        series.index = pd.to_datetime(series.index)
        
        if method == 'last':
            return series.resample('QE').last()
        elif method == 'mean':
            return series.resample('QE').mean()
        elif method == 'sum':
            return series.resample('QE').sum()
        else:
            return series.resample('QE').last()
    
    @staticmethod
    def construct_qe_intensity(fed_holdings: pd.Series, total_debt: pd.Series) -> pd.Series:
        """
        Construct QE intensity measure.
        
        QE Intensity = (Fed Holdings / Total Public Debt) * 100
        
        This measures the Fed's share of outstanding Treasuries in PERCENTAGE POINTS.
        A value of 10 means the Fed holds 10% of total debt.
        """
        # Align series
        aligned = pd.DataFrame({
            'fed_holdings': fed_holdings,
            'total_debt': total_debt
        }).dropna()
        
        # Calculate ratio and convert to percentage points
        qe_intensity = (aligned['fed_holdings'] / aligned['total_debt']) * 100
        qe_intensity.name = 'qe_intensity'
        
        return qe_intensity
    
    @staticmethod
    def construct_total_foreign_holdings(official: pd.Series, private: pd.Series) -> pd.Series:
        """Construct total foreign holdings."""
        aligned = pd.DataFrame({
            'official': official,
            'private': private
        }).dropna()
        
        total = aligned['official'] + aligned['private']
        total.name = 'total_foreign_holdings'
        
        return total
    
    def prepare_quarterly_data(self, raw_data: Dict[str, pd.Series]) -> pd.DataFrame:
        """Prepare quarterly dataset for analysis."""
        print("\n" + "="*70)
        print("STEP 2: Processing Data to Quarterly Frequency")
        print("="*70)
        
        # Convert to quarterly (stock variables use end-of-quarter)
        quarterly = {}
        
        for name, series in raw_data.items():
            if series.empty:
                continue
            quarterly[name] = self.to_quarterly(series, method='last')
            print(f"  {name}: {len(quarterly[name])} quarters")
        
        # Construct derived variables
        print("\nConstructing derived variables...")
        
        # QE Intensity
        if 'fed_holdings' in quarterly and 'total_debt' in quarterly:
            quarterly['qe_intensity'] = self.construct_qe_intensity(
                quarterly['fed_holdings'],
                quarterly['total_debt']
            )
            print(f"  qe_intensity: {len(quarterly['qe_intensity'])} quarters")
        
        # Total foreign holdings
        if 'official_holdings' in quarterly and 'private_holdings' in quarterly:
            quarterly['total_holdings'] = self.construct_total_foreign_holdings(
                quarterly['official_holdings'],
                quarterly['private_holdings']
            )
            print(f"  total_holdings: {len(quarterly['total_holdings'])} quarters")
        
        # Create aligned DataFrame
        df = pd.DataFrame(quarterly)
        df = df.dropna()
        
        # Filter to sample period
        start = pd.Timestamp(CONFIG['sample_start'])
        end = pd.Timestamp(CONFIG['sample_end'])
        df = df[(df.index >= start) & (df.index <= end)]
        
        print(f"\nFinal quarterly dataset: {len(df)} observations")
        print(f"Date range: {df.index.min()} to {df.index.max()}")
        
        return df


# =============================================================================
# REGRESSION ANALYSIS - TABLE 2
# =============================================================================

class ForeignHoldingsAnalysis:
    """
    Implements the foreign holdings response analysis from Table 2.
    
    Regression specification:
        Δln(H_{i,t}) = α_i + γ_t + β · QEIntensity_t + ε_{i,t}
    
    Where:
        - H_{i,t} is foreign Treasury holdings for investor type i
        - QEIntensity_t is Fed holdings / Total debt
    """
    
    def __init__(self, data: pd.DataFrame):
        """Initialize with quarterly data."""
        self.data = data
        self.results = {}
        
    def compute_log_changes(self) -> pd.DataFrame:
        """Compute quarterly log changes in holdings and QE intensity."""
        df = self.data.copy()
        
        # Log changes in holdings
        for col in ['total_holdings', 'official_holdings', 'private_holdings']:
            if col in df.columns:
                df[f'd_ln_{col}'] = np.log(df[col]).diff()
        
        # Change in QE intensity (in percentage points)
        if 'qe_intensity' in df.columns:
            df['d_qe_intensity'] = df['qe_intensity'].diff()
        
        return df.dropna()
    
    def run_regression(self, 
                      y_col: str, 
                      x_col: str = 'qe_intensity',
                      use_change: bool = True,
                      add_quarter_fe: bool = True,
                      add_lagged_y: bool = False) -> Dict:
        """
        Run OLS regression with robust standard errors.
        
        Args:
            y_col: Dependent variable column name
            x_col: Independent variable column name
            use_change: If True, use change in QE intensity; if False, use level
            add_quarter_fe: Whether to add quarter fixed effects
            add_lagged_y: Whether to add lagged dependent variable
            
        Returns:
            Dictionary with regression results
        """
        df = self.compute_log_changes()
        
        # Determine which x variable to use
        actual_x_col = 'd_qe_intensity' if use_change else x_col
        
        if y_col not in df.columns or actual_x_col not in df.columns:
            return {'error': f'Missing columns: {y_col} or {actual_x_col}'}
        
        y = df[y_col].astype(float)
        X = df[[actual_x_col]].copy().astype(float)
        X.columns = ['qe_intensity']  # Rename for consistent output
        
        # Add lagged dependent variable if requested
        if add_lagged_y:
            X['y_lag1'] = y.shift(1)
        
        # Add quarter fixed effects (seasonal dummies)
        if add_quarter_fe:
            quarters = df.index.quarter
            for q in [2, 3, 4]:  # Q1 is reference
                X[f'Q{q}'] = (quarters == q).astype(float)
        
        X = sm.add_constant(X)
        
        # Drop NaN rows (from lagged variable)
        valid = ~X.isna().any(axis=1) & ~y.isna()
        X = X[valid]
        y = y[valid]
        
        # Fit with robust standard errors (HC1)
        model = OLS(y, X).fit(cov_type='HC1')
        
        return {
            'coefficient': model.params['qe_intensity'],
            'std_error': model.bse['qe_intensity'],
            't_stat': model.tvalues['qe_intensity'],
            'p_value': model.pvalues['qe_intensity'],
            'r_squared': model.rsquared,
            'n_obs': int(model.nobs),
            'model': model
        }
    
    def run_all_regressions(self) -> Dict:
        """Run regressions for all investor types (Table 2)."""
        print("\n" + "="*70)
        print("STEP 3: Foreign Holdings Response Regressions (Table 2)")
        print("="*70)
        
        results = {}
        
        # Specification using LEVEL of QE intensity (as in paper equation)
        print("\nSpec A: QE Intensity Level")
        print("-"*50)
        
        results['total'] = self.run_regression('d_ln_total_holdings', use_change=False)
        results['official'] = self.run_regression('d_ln_official_holdings', use_change=False)
        results['private'] = self.run_regression('d_ln_private_holdings', use_change=False)
        
        # Specification using CHANGE in QE intensity
        print("\nSpec B: Change in QE Intensity")
        print("-"*50)
        
        results['total_chg'] = self.run_regression('d_ln_total_holdings', use_change=True)
        results['official_chg'] = self.run_regression('d_ln_official_holdings', use_change=True)
        results['private_chg'] = self.run_regression('d_ln_private_holdings', use_change=True)
        
        self.results = results
        return results
    
    def print_table2(self):
        """Print formatted Table 2 results."""
        print("\n" + "="*70)
        print("TABLE 2: Effect of U.S. QE on Foreign Holdings of U.S. Treasuries")
        print("="*70)
        
        def stars(p):
            if p < 0.01: return '***'
            elif p < 0.05: return '**'
            elif p < 0.10: return '*'
            return ''
        
        # Print main results - Level specification
        print("\nSpec A: QE Intensity LEVEL (% change per 1 pp QE intensity)")
        print(f"\n{'':20} {'(1) Total':>15} {'(2) Official':>15} {'(3) Private':>15}")
        print("-"*70)
        
        row = "QE Intensity"
        for key in ['total', 'official', 'private']:
            if key in self.results and 'coefficient' in self.results[key]:
                coef = self.results[key]['coefficient'] * 100
                sig = stars(self.results[key]['p_value'])
                row += f" {coef:>12.3f}{sig:<3}"
            else:
                row += f" {'N/A':>15}"
        print(row)
        
        row = ""
        for key in ['total', 'official', 'private']:
            if key in self.results and 'std_error' in self.results[key]:
                se = self.results[key]['std_error'] * 100
                row += f" {'(' + f'{se:.3f}' + ')':>15}"
            else:
                row += f" {'':>15}"
        print(f"{'':20}{row}")
        
        row = "R-squared"
        for key in ['total', 'official', 'private']:
            if key in self.results and 'r_squared' in self.results[key]:
                r2 = self.results[key]['r_squared']
                row += f" {r2:>15.2f}"
            else:
                row += f" {'N/A':>15}"
        print(row)
        
        row = "Observations"
        for key in ['total', 'official', 'private']:
            if key in self.results and 'n_obs' in self.results[key]:
                n = self.results[key]['n_obs']
                row += f" {n:>15}"
            else:
                row += f" {'N/A':>15}"
        print(row)
        
        # Print Change specification
        print("\n" + "-"*70)
        print("\nSpec B: CHANGE in QE Intensity (% change per 1 pp change in QE)")
        print(f"\n{'':20} {'(1) Total':>15} {'(2) Official':>15} {'(3) Private':>15}")
        print("-"*70)
        
        row = "d(QE Intensity)"
        for key in ['total_chg', 'official_chg', 'private_chg']:
            if key in self.results and 'coefficient' in self.results[key]:
                coef = self.results[key]['coefficient'] * 100
                sig = stars(self.results[key]['p_value'])
                row += f" {coef:>12.3f}{sig:<3}"
            else:
                row += f" {'N/A':>15}"
        print(row)
        
        row = ""
        for key in ['total_chg', 'official_chg', 'private_chg']:
            if key in self.results and 'std_error' in self.results[key]:
                se = self.results[key]['std_error'] * 100
                row += f" {'(' + f'{se:.3f}' + ')':>15}"
            else:
                row += f" {'':>15}"
        print(f"{'':20}{row}")
        
        row = "R-squared"
        for key in ['total_chg', 'official_chg', 'private_chg']:
            if key in self.results and 'r_squared' in self.results[key]:
                r2 = self.results[key]['r_squared']
                row += f" {r2:>15.2f}"
            else:
                row += f" {'N/A':>15}"
        print(row)
        
        print("\n" + "-"*70)
        print("\nPaper's Table 2 (target values):")
        print(f"{'':20} {'(1) Total':>15} {'(2) Official':>15} {'(3) Private':>15}")
        print("-"*70)
        print(f"{'QE Intensity':20} {'-0.800***':>15} {'-1.400***':>15} {'-0.255':>15}")
        print(f"{'':20} {'(0.250)':>15} {'(0.300)':>15} {'(0.500)':>15}")
        print(f"{'R-squared':20} {'0.15':>15} {'0.17':>15} {'0.08':>15}")
        print(f"{'Observations':20} {'63':>15} {'63':>15} {'63':>15}")
        
        print("\n" + "-"*70)
        print("Notes: Dependent variable is quarterly log change in foreign holdings.")
        print("       Robust standard errors (HC1) in parentheses.")
        print("       *** p<0.01, ** p<0.05, * p<0.10")
        print("="*70)


# =============================================================================
# LOCAL PROJECTIONS - FIGURE 1
# =============================================================================

class ExchangeRateLocalProjections:
    """
    Implements local projections for exchange rate response.
    
    Specification:
        Δ^h FX_{t+h} = θ_h · QEShock_t + u_{t+h}
    
    Where:
        - Δ^h FX_{t+h} is the h-month cumulative change in trade-weighted USD
        - QEShock_t is the QE shock (change in QE intensity)
    """
    
    def __init__(self, data: pd.DataFrame, max_horizon: int = 18):
        """Initialize with data and maximum horizon."""
        self.data = data
        self.max_horizon = max_horizon
        self.results = {}
        
    def prepare_monthly_data(self, raw_data: Dict[str, pd.Series]) -> pd.DataFrame:
        """Prepare monthly data for local projections."""
        # Use monthly frequency for FX analysis
        monthly = {}
        
        # Get trade-weighted USD
        if 'trade_weighted_usd' in raw_data and not raw_data['trade_weighted_usd'].empty:
            series = raw_data['trade_weighted_usd']
            series.index = pd.to_datetime(series.index)
            monthly['trade_weighted_usd'] = series.resample('ME').last()
        
        # Construct QE intensity from fed_holdings and total_debt
        if 'fed_holdings' in raw_data and 'total_debt' in raw_data:
            fed = raw_data['fed_holdings']
            debt = raw_data['total_debt']
            
            fed.index = pd.to_datetime(fed.index)
            debt.index = pd.to_datetime(debt.index)
            
            # Resample to monthly
            fed_monthly = fed.resample('ME').last().ffill()
            debt_monthly = debt.resample('ME').last().ffill()
            
            # Calculate QE intensity
            qe_intensity = fed_monthly / debt_monthly
            monthly['qe_intensity'] = qe_intensity
        
        df = pd.DataFrame(monthly).dropna()
        
        # Construct QE shock as change in QE intensity
        df['qe_shock'] = df['qe_intensity'].diff()
        
        # Standardize shock for interpretation
        df['qe_shock_std'] = (df['qe_shock'] - df['qe_shock'].mean()) / df['qe_shock'].std()
        
        return df.dropna()
    
    def estimate_local_projection(self, 
                                 df: pd.DataFrame,
                                 horizon: int,
                                 shock_col: str = 'qe_shock_std',
                                 outcome_col: str = 'trade_weighted_usd') -> Dict:
        """
        Estimate local projection at horizon h.
        
        Args:
            df: Monthly data
            horizon: Forecast horizon in months
            shock_col: QE shock column
            outcome_col: Outcome variable column
            
        Returns:
            Dictionary with coefficient, standard error, confidence interval
        """
        # Compute cumulative change in outcome
        df = df.copy()
        df['y_cumulative'] = df[outcome_col].pct_change(periods=horizon) * 100
        
        # Align shock with future outcome
        df_reg = pd.DataFrame({
            'y': df['y_cumulative'].shift(-horizon),
            'shock': df[shock_col]
        }).dropna()
        
        if len(df_reg) < 20:
            return {'error': 'Insufficient observations'}
        
        y = df_reg['y']
        X = sm.add_constant(df_reg[['shock']])
        
        # Newey-West standard errors for autocorrelation
        model = OLS(y, X).fit(cov_type='HAC', cov_kwds={'maxlags': horizon})
        
        coef = model.params['shock']
        se = model.bse['shock']
        
        return {
            'horizon': horizon,
            'coefficient': coef,
            'std_error': se,
            'ci_lower': coef - 1.96 * se,
            'ci_upper': coef + 1.96 * se,
            'p_value': model.pvalues['shock'],
            'n_obs': int(model.nobs)
        }
    
    def run_all_horizons(self, raw_data: Dict[str, pd.Series]) -> pd.DataFrame:
        """Run local projections for all horizons."""
        print("\n" + "="*70)
        print("STEP 4: Exchange Rate Local Projections (Figure 1)")
        print("="*70)
        
        df = self.prepare_monthly_data(raw_data)
        print(f"Monthly data prepared: {len(df)} observations")
        
        results = []
        horizons = range(0, self.max_horizon + 1, 3)  # 0, 3, 6, 9, 12, 15, 18
        
        for h in horizons:
            res = self.estimate_local_projection(df, h)
            if 'error' not in res:
                results.append(res)
                print(f"  Horizon {h:2d}: coef = {res['coefficient']:7.4f}, "
                      f"SE = {res['std_error']:.4f}, n = {res['n_obs']}")
        
        self.results = pd.DataFrame(results)
        return self.results
    
    def print_figure1_data(self):
        """Print data for Figure 1."""
        if self.results.empty:
            print("No results to display. Run run_all_horizons() first.")
            return
        
        print("\n" + "="*70)
        print("FIGURE 1: FX Response to QE Shocks (Local Projections)")
        print("="*70)
        print("\nCumulative response of trade-weighted USD to one-std-dev QE shock")
        print("Positive values indicate dollar appreciation\n")
        
        print(f"{'Horizon':>10} {'Coefficient':>12} {'Std Error':>12} {'95% CI Lower':>14} {'95% CI Upper':>14}")
        print("-"*70)
        
        for _, row in self.results.iterrows():
            print(f"{int(row['horizon']):>10} {row['coefficient']:>12.4f} "
                  f"{row['std_error']:>12.4f} {row['ci_lower']:>14.4f} {row['ci_upper']:>14.4f}")
        
        print("-"*70)
        print("Notes: Newey-West standard errors. 95% confidence intervals shown.")
        print("="*70)
    
    def plot_impulse_response(self, save_path: Optional[str] = None):
        """Plot impulse response function."""
        try:
            import matplotlib
            matplotlib.use('Agg')  # Non-interactive backend
            import matplotlib.pyplot as plt
        except ImportError:
            print("matplotlib not installed. Skipping plot.")
            return
        
        if self.results.empty:
            print("No results to plot.")
            return
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        horizons = self.results['horizon']
        coefs = self.results['coefficient']
        ci_lower = self.results['ci_lower']
        ci_upper = self.results['ci_upper']
        
        # Plot point estimates
        ax.plot(horizons, coefs, 'b-o', linewidth=2, markersize=8, label='Point Estimate')
        
        # Plot confidence interval
        ax.fill_between(horizons, ci_lower, ci_upper, alpha=0.2, color='blue', label='95% CI')
        
        # Zero line
        ax.axhline(y=0, color='black', linestyle='--', linewidth=0.8)
        
        ax.set_xlabel('Horizon (months)', fontsize=12)
        ax.set_ylabel('Cumulative Response (%)', fontsize=12)
        ax.set_title('FX Response to QE Shocks (Local Projections)', fontsize=14)
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Figure saved to: {save_path}")
        
        plt.close(fig)  # Close figure to free memory


# =============================================================================
# SUMMARY STATISTICS
# =============================================================================

def print_summary_statistics(data: pd.DataFrame):
    """Print summary statistics for key variables."""
    print("\n" + "="*70)
    print("SUMMARY STATISTICS")
    print("="*70)
    
    stats_cols = ['qe_intensity', 'total_holdings', 'official_holdings', 
                  'private_holdings', 'trade_weighted_usd']
    
    available_cols = [c for c in stats_cols if c in data.columns]
    
    if not available_cols:
        print("No variables available for summary statistics.")
        return
    
    stats = data[available_cols].describe()
    
    # Rename for display
    rename_map = {
        'qe_intensity': 'QE Intensity',
        'total_holdings': 'Total Foreign Holdings',
        'official_holdings': 'Official Holdings',
        'private_holdings': 'Private Holdings',
        'trade_weighted_usd': 'Trade-Weighted USD'
    }
    
    stats = stats.rename(columns=rename_map)
    
    start_date = data.index.min()
    end_date = data.index.max()
    print(f"\nSample: {start_date.year}Q{(start_date.month-1)//3+1} to {end_date.year}Q{(end_date.month-1)//3+1}")
    print(f"Observations: {len(data)}\n")
    print(stats.round(2).to_string())
    print("="*70)


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """Main execution function."""
    print("\n" + "="*70)
    print("PAPER METHODOLOGY RECREATION")
    print("Cross-Border Portfolio Effects of U.S. Quantitative Easing")
    print("="*70)
    print(f"\nSample Period: {CONFIG['sample_start']} to {CONFIG['sample_end']}")
    print(f"Frequency: {CONFIG['frequency']}")
    
    # Step 1: Fetch data from FRED
    collector = FREDDataCollector()
    raw_data = collector.fetch_all_data()
    
    # Step 2: Process to quarterly frequency
    processor = DataProcessor()
    quarterly_data = processor.prepare_quarterly_data(raw_data)
    
    # Print summary statistics
    print_summary_statistics(quarterly_data)
    
    # Step 3: Foreign holdings regressions (Table 2)
    holdings_analysis = ForeignHoldingsAnalysis(quarterly_data)
    holdings_results = holdings_analysis.run_all_regressions()
    holdings_analysis.print_table2()
    
    # Step 4: Exchange rate local projections (Figure 1)
    lp_analysis = ExchangeRateLocalProjections(quarterly_data, max_horizon=18)
    lp_results = lp_analysis.run_all_horizons(raw_data)
    lp_analysis.print_figure1_data()
    
    # Optional: Plot impulse response
    try:
        output_dir = os.path.join(os.path.dirname(__file__), '..', 'output')
        os.makedirs(output_dir, exist_ok=True)
        lp_analysis.plot_impulse_response(
            save_path=os.path.join(output_dir, 'fx_impulse_response.png')
        )
    except Exception as e:
        print(f"Could not save plot: {e}")
    
    print("\n" + "="*70)
    print("RECREATION COMPLETE")
    print("="*70)
    
    return {
        'quarterly_data': quarterly_data,
        'holdings_results': holdings_results,
        'lp_results': lp_results
    }


if __name__ == "__main__":
    results = main()
