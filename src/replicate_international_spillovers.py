"""
Replication Script: International Spillovers of U.S. Quantitative Easing

This script replicates the empirical analysis from the paper on international
spillovers of U.S. quantitative easing, focusing on the effects of QE on
foreign holdings of U.S. Treasury securities and exchange rates.

Requirements: 1.1, 1.2, 1.3, 12.1, 12.2, 12.3, 13.1, 13.3
"""

import argparse
import json
import logging
import os
import sys
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from fredapi import Fred
import statsmodels.api as sm
from statsmodels.regression.linear_model import OLS
from statsmodels.stats.sandwich_covariance import cov_hc0, cov_hac

from fomc_dates import FOMC_DATES_2008_2023, get_fomc_dates

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class RegressionResult:
    """Container for regression results."""
    coefficients: Dict[str, float]
    std_errors: Dict[str, float]
    t_stats: Dict[str, float]
    p_values: Dict[str, float]
    r_squared: float
    n_obs: int
    residuals: Optional[List[float]] = None


# =============================================================================
# FRED Client Helper with Retry Logic
# =============================================================================

class FREDClient:
    """
    FRED API client wrapper with retry logic and error handling.
    
    Requirements: 12.4 - Log errors and continue with available data
    """
    
    def __init__(self, api_key: Optional[str] = None, max_retries: int = 3, 
                 retry_delay: float = 1.0):
        """
        Initialize FRED client.
        
        Args:
            api_key: FRED API key. If None, reads from FRED_API_KEY env var.
            max_retries: Maximum number of retry attempts for failed requests.
            retry_delay: Base delay between retries (exponential backoff).
        """
        self.api_key = api_key or os.getenv('FRED_API_KEY')
        if not self.api_key:
            raise ValueError(
                "FRED API key required. Set FRED_API_KEY environment variable "
                "or pass api_key parameter."
            )
        self.fred = Fred(api_key=self.api_key)
        self.max_retries = max_retries
        self.retry_delay = retry_delay
    
    def fetch_series(self, series_id: str, start_date: str, end_date: str,
                     frequency: Optional[str] = None) -> Optional[pd.Series]:
        """
        Fetch a single series from FRED with retry logic.
        
        Args:
            series_id: FRED series identifier
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            frequency: Optional frequency aggregation ('d', 'm', 'q', 'a')
            
        Returns:
            pandas Series with DatetimeIndex, or None if fetch fails
        """
        for attempt in range(self.max_retries):
            try:
                kwargs = {
                    'observation_start': start_date,
                    'observation_end': end_date
                }
                if frequency:
                    kwargs['frequency'] = frequency
                
                data = self.fred.get_series(series_id, **kwargs)
                
                if data is not None and len(data) > 0:
                    data.name = series_id
                    logger.info(f"Successfully fetched {series_id}: {len(data)} observations")
                    return data
                else:
                    logger.warning(f"No data returned for {series_id}")
                    return None
                    
            except Exception as e:
                delay = self.retry_delay * (2 ** attempt)
                logger.warning(
                    f"Attempt {attempt + 1}/{self.max_retries} failed for {series_id}: {e}. "
                    f"Retrying in {delay:.1f}s..."
                )
                if attempt < self.max_retries - 1:
                    time.sleep(delay)
                else:
                    logger.error(f"Failed to fetch {series_id} after {self.max_retries} attempts")
                    return None
        
        return None


# =============================================================================
# Output Utility Functions
# =============================================================================

def create_output_dir(base_dir: str = "output") -> Path:
    """
    Create a timestamped output directory for results.
    
    Requirements: 12.1 - Create timestamped output directory for each run
    
    Args:
        base_dir: Base directory name for outputs
        
    Returns:
        Path to created output directory
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = Path(base_dir) / f"international_spillovers_{timestamp}"
    output_path.mkdir(parents=True, exist_ok=True)
    logger.info(f"Created output directory: {output_path}")
    return output_path


def save_dataframe(df: pd.DataFrame, filepath: Path, description: str = "") -> None:
    """
    Save DataFrame to CSV with consistent formatting.
    
    Requirements: 12.3 - Use consistent column naming conventions
    
    Args:
        df: DataFrame to save
        filepath: Output file path
        description: Optional description for logging
    """
    df.to_csv(filepath, index=True)
    logger.info(f"Saved {description or filepath.name}: {len(df)} rows to {filepath}")


def save_metadata(output_dir: Path, metadata: Dict) -> None:
    """
    Save metadata file documenting data sources and parameters.
    
    Requirements: 12.2 - Include metadata file documenting data sources
    
    Args:
        output_dir: Output directory path
        metadata: Dictionary of metadata to save
    """
    metadata_path = output_dir / "metadata.json"
    metadata['generated_at'] = datetime.now().isoformat()
    metadata['script'] = 'replicate_international_spillovers.py'
    
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2, default=str)
    
    logger.info(f"Saved metadata to {metadata_path}")


# =============================================================================
# Data Acquisition Functions
# =============================================================================

def fetch_holdings_data(client: FREDClient, start_date: str, 
                        end_date: str) -> Optional[pd.DataFrame]:
    """
    Fetch foreign holdings of U.S. Treasury securities.
    
    Requirements: 1.1 - Fetch FDHBPIN (official) and FDHBFIN (private)
    
    Args:
        client: FREDClient instance
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        
    Returns:
        DataFrame with official and private foreign holdings, or None if failed
    """
    official = client.fetch_series('FDHBPIN', start_date, end_date)
    private = client.fetch_series('FDHBFIN', start_date, end_date)
    
    if official is None and private is None:
        logger.error("Failed to fetch any holdings data")
        return None
    
    df = pd.DataFrame({
        'official_holdings': official,
        'private_holdings': private
    })
    
    # Compute total holdings
    df['total_holdings'] = df['official_holdings'].fillna(0) + df['private_holdings'].fillna(0)
    
    return df


def fetch_fed_balance_sheet(client: FREDClient, start_date: str,
                            end_date: str) -> Optional[pd.DataFrame]:
    """
    Fetch Federal Reserve Treasury holdings and total marketable debt.
    
    Requirements: 1.2 - Fetch TREAST and GFDEBTN (total public debt)
    
    Args:
        client: FREDClient instance
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        
    Returns:
        DataFrame with Fed holdings and total debt, or None if failed
    """
    fed_holdings = client.fetch_series('TREAST', start_date, end_date)
    # Use GFDEBTN (Total Public Debt) as proxy for marketable debt
    total_debt = client.fetch_series('GFDEBTN', start_date, end_date, frequency='q')
    
    if fed_holdings is None and total_debt is None:
        logger.error("Failed to fetch any balance sheet data")
        return None
    
    df = pd.DataFrame({
        'fed_treasury_holdings': fed_holdings,
        'total_marketable_debt': total_debt
    })
    
    return df


def fetch_fx_data(client: FREDClient, start_date: str,
                  end_date: str) -> Optional[pd.Series]:
    """
    Fetch trade-weighted U.S. dollar index.
    
    Requirements: 1.3 - Fetch DTWEXBGS
    
    Args:
        client: FREDClient instance
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        
    Returns:
        Series with dollar index, or None if failed
    """
    fx = client.fetch_series('DTWEXBGS', start_date, end_date, frequency='d')
    if fx is not None:
        fx.name = 'trade_weighted_dollar'
    return fx


def fetch_treasury_yields(client: FREDClient, start_date: str,
                          end_date: str) -> Optional[pd.Series]:
    """
    Fetch 10-year Treasury constant maturity yields at daily frequency.
    
    Requirements: 2.1 - Use daily changes in 10-year Treasury yields
    
    Args:
        client: FREDClient instance
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        
    Returns:
        Series with daily yields, or None if failed
    """
    yields = client.fetch_series('DGS10', start_date, end_date, frequency='d')
    if yields is not None:
        yields.name = 'treasury_10y_yield'
    return yields


def compute_qe_intensity(fed_holdings: pd.Series, 
                         total_debt: pd.Series) -> pd.Series:
    """
    Compute QE intensity as Fed share of outstanding Treasuries.
    
    Requirements: 1.4 - Compute QE intensity as ratio of Fed holdings to total debt
    
    Args:
        fed_holdings: Series of Fed Treasury holdings
        total_debt: Series of total marketable Treasury debt
        
    Returns:
        Series with QE intensity (0 to 1 range)
    """
    # Resample fed_holdings to quarterly (end of quarter average) to match total_debt
    fed_quarterly = fed_holdings.resample('QE').mean()
    
    # Align indices to quarter end dates
    fed_quarterly.index = pd.to_datetime(fed_quarterly.index).to_period('Q').to_timestamp('Q')
    total_debt_aligned = total_debt.copy()
    total_debt_aligned.index = pd.to_datetime(total_debt_aligned.index).to_period('Q').to_timestamp('Q')
    
    # Align series to common index
    aligned = pd.DataFrame({
        'fed': fed_quarterly,
        'total': total_debt_aligned
    }).dropna()
    
    # Convert to billions if needed (TREAST is in millions, GFDEBTN is in millions)
    # Both should be in same units now
    intensity = aligned['fed'] / aligned['total']
    intensity.name = 'qe_intensity'
    
    # Clip to valid range [0, 1]
    intensity = intensity.clip(0, 1)
    
    return intensity


# =============================================================================
# QE Shock Construction Functions
# =============================================================================

def construct_qe_shocks(yields: pd.Series, 
                        fomc_dates: List[str]) -> pd.Series:
    """
    Construct QE shocks from daily yield changes around FOMC dates.
    
    Requirements: 2.1, 2.2 - Use daily changes in 10-year Treasury yields 
                            around FOMC announcement dates
    
    Args:
        yields: Daily Treasury yield series
        fomc_dates: List of FOMC announcement dates (YYYY-MM-DD)
        
    Returns:
        Series of QE shocks indexed by FOMC date
    """
    shocks = {}
    
    for date_str in fomc_dates:
        date = pd.Timestamp(date_str)
        
        # Find the yield on FOMC date and previous trading day
        if date in yields.index:
            current_yield = yields.loc[date]
            
            # Get previous trading day (may not be exactly day before)
            prev_dates = yields.index[yields.index < date]
            if len(prev_dates) > 0:
                prev_date = prev_dates[-1]
                prev_yield = yields.loc[prev_date]
                
                # Shock is the change in yield (in basis points)
                if pd.notna(current_yield) and pd.notna(prev_yield):
                    shock = (current_yield - prev_yield) * 100  # Convert to bps
                    shocks[date] = shock
    
    shock_series = pd.Series(shocks, name='qe_shock')
    shock_series.index.name = 'date'
    
    logger.info(f"Constructed {len(shock_series)} QE shocks from {len(fomc_dates)} FOMC dates")
    
    return shock_series


def aggregate_shocks_monthly(daily_shocks: pd.Series) -> pd.Series:
    """
    Aggregate daily QE shocks to monthly frequency.
    
    Requirements: 2.3 - Aggregate daily shocks to monthly frequency
    
    Args:
        daily_shocks: Series of daily QE shocks
        
    Returns:
        Series of monthly aggregated shocks
    """
    # Resample to monthly, summing shocks within each month
    monthly_shocks = daily_shocks.resample('ME').sum()
    monthly_shocks.name = 'qe_shock_monthly'
    
    # Fill months with no FOMC meetings with 0
    monthly_shocks = monthly_shocks.fillna(0)
    
    return monthly_shocks


# =============================================================================
# Regression Analysis Functions
# =============================================================================

def ols_with_robust_se(y: pd.Series, X: pd.DataFrame, 
                       cov_type: str = 'HC1') -> RegressionResult:
    """
    Estimate OLS regression with heteroskedasticity-consistent standard errors.
    
    Requirements: 3.3 - Compute robust standard errors using HC estimators
    
    Args:
        y: Dependent variable
        X: Independent variables (should include constant if desired)
        cov_type: Type of robust covariance ('HC0', 'HC1', 'HC2', 'HC3')
        
    Returns:
        RegressionResult with coefficients, standard errors, etc.
    """
    # Align data
    data = pd.concat([y, X], axis=1).dropna()
    y_clean = data.iloc[:, 0]
    X_clean = data.iloc[:, 1:]
    
    # Fit OLS
    model = OLS(y_clean, X_clean)
    results = model.fit(cov_type=cov_type)
    
    return RegressionResult(
        coefficients=results.params.to_dict(),
        std_errors=results.bse.to_dict(),
        t_stats=results.tvalues.to_dict(),
        p_values=results.pvalues.to_dict(),
        r_squared=results.rsquared,
        n_obs=int(results.nobs),
        residuals=results.resid.tolist()
    )


def estimate_holdings_regression(holdings: pd.DataFrame,
                                  qe_intensity: pd.Series,
                                  holder_type: str = 'total') -> RegressionResult:
    """
    Estimate effect of QE on foreign holdings with fixed effects.
    
    Requirements: 3.1 - Regress monthly log changes in foreign Treasury holdings
                       on QE intensity with fixed effects
    
    Args:
        holdings: DataFrame with holdings data
        qe_intensity: Series of QE intensity
        holder_type: 'total', 'official', or 'private'
        
    Returns:
        RegressionResult for the holdings regression
    """
    # Select appropriate holdings column
    col_map = {
        'total': 'total_holdings',
        'official': 'official_holdings',
        'private': 'private_holdings'
    }
    holdings_col = col_map.get(holder_type, 'total_holdings')
    
    # Ensure numeric data and align indices to quarter end
    holdings_series = pd.to_numeric(holdings[holdings_col], errors='coerce')
    holdings_series.index = pd.to_datetime(holdings_series.index).to_period('Q').to_timestamp('Q')
    
    qe_intensity_numeric = pd.to_numeric(qe_intensity, errors='coerce')
    
    # Compute log changes
    y = np.log(holdings_series).diff()
    y.name = f'dlog_{holder_type}_holdings'
    
    # Align with QE intensity
    data = pd.DataFrame({
        'y': y,
        'qe_intensity': qe_intensity_numeric
    }).dropna()
    
    # Add constant
    X = sm.add_constant(data['qe_intensity'].astype(float))
    
    # Estimate with robust SE
    return ols_with_robust_se(data['y'].astype(float), X)


def local_projection(y: pd.Series, x: pd.Series, horizon: int,
                     controls: Optional[pd.DataFrame] = None,
                     nw_lags: Optional[int] = None) -> Dict:
    """
    Estimate local projection at a single horizon with Newey-West SE.
    
    Requirements: 3.2, 3.3 - Local projections with Newey-West standard errors
    
    Args:
        y: Dependent variable (level)
        x: Shock variable
        horizon: Forecast horizon
        controls: Optional control variables
        nw_lags: Newey-West lag truncation (default: horizon + 1)
        
    Returns:
        Dictionary with coefficient, SE, t-stat, p-value, CI bounds
    """
    # Compute cumulative change in y at horizon h
    y_h = y.shift(-horizon) - y
    y_h.name = f'y_h{horizon}'
    
    # Build regressor matrix
    if controls is not None:
        X = pd.concat([x, controls], axis=1)
    else:
        X = x.to_frame()
    
    X = sm.add_constant(X)
    
    # Align data
    data = pd.concat([y_h, X], axis=1).dropna()
    y_clean = data.iloc[:, 0]
    X_clean = data.iloc[:, 1:]
    
    if len(y_clean) < 10:
        return {
            'horizon': horizon,
            'coefficient': np.nan,
            'std_error': np.nan,
            't_stat': np.nan,
            'p_value': np.nan,
            'ci_lower': np.nan,
            'ci_upper': np.nan,
            'n_obs': len(y_clean)
        }
    
    # Fit with Newey-West HAC standard errors
    if nw_lags is None:
        nw_lags = horizon + 1
    
    model = OLS(y_clean, X_clean)
    results = model.fit(cov_type='HAC', cov_kwds={'maxlags': nw_lags})
    
    # Extract coefficient on shock variable (first non-constant regressor)
    shock_var = x.name if x.name else 'x'
    if shock_var in results.params.index:
        coef = results.params[shock_var]
        se = results.bse[shock_var]
        t_stat = results.tvalues[shock_var]
        p_val = results.pvalues[shock_var]
    else:
        # Fallback to second coefficient (after constant)
        coef = results.params.iloc[1]
        se = results.bse.iloc[1]
        t_stat = results.tvalues.iloc[1]
        p_val = results.pvalues.iloc[1]
    
    return {
        'horizon': horizon,
        'coefficient': coef,
        'std_error': se,
        't_stat': t_stat,
        'p_value': p_val,
        'ci_lower': coef - 1.96 * se,
        'ci_upper': coef + 1.96 * se,
        'n_obs': int(results.nobs)
    }


def estimate_fx_local_projections(fx: pd.Series, qe_shock: pd.Series,
                                   horizons: range = range(19)) -> pd.DataFrame:
    """
    Estimate FX response using local projections for multiple horizons.
    
    Requirements: 3.2 - Local projections for horizons 0 through 18 months
    
    Args:
        fx: Exchange rate series (log level)
        qe_shock: QE shock series
        horizons: Range of horizons to estimate (default 0-18)
        
    Returns:
        DataFrame with impulse response coefficients and confidence intervals
    """
    # Ensure fx is in log form
    log_fx = np.log(fx) if fx.min() > 0 else fx
    log_fx.name = 'log_fx'
    
    # Align shock series with fx
    qe_shock_aligned = qe_shock.reindex(log_fx.index).fillna(0)
    qe_shock_aligned.name = 'qe_shock'
    
    results = []
    for h in horizons:
        lp_result = local_projection(log_fx, qe_shock_aligned, horizon=h)
        results.append(lp_result)
    
    df = pd.DataFrame(results)
    logger.info(f"Estimated FX local projections for {len(horizons)} horizons")
    
    return df


# =============================================================================
# Results Output Functions
# =============================================================================

def generate_table1(holdings_results: Dict[str, RegressionResult],
                    sample_start: str, sample_end: str) -> pd.DataFrame:
    """
    Generate Table 1: Effect of U.S. QE on Foreign Holdings.
    
    Requirements: 4.1 - Generate CSV replicating Table 1
    
    Args:
        holdings_results: Dict mapping holder type to RegressionResult
        sample_start: Sample start date
        sample_end: Sample end date
        
    Returns:
        DataFrame formatted as Table 1
    """
    rows = []
    for holder_type, result in holdings_results.items():
        # Get QE intensity coefficient
        qe_coef = result.coefficients.get('qe_intensity', np.nan)
        qe_se = result.std_errors.get('qe_intensity', np.nan)
        
        rows.append({
            'variable': holder_type.capitalize(),
            'qe_intensity_coef': qe_coef,
            'std_error': qe_se,
            'r_squared': result.r_squared,
            'n_obs': result.n_obs,
            'sample_start': sample_start,
            'sample_end': sample_end
        })
    
    return pd.DataFrame(rows)


def generate_fx_results(lp_results: pd.DataFrame,
                        sample_start: str, sample_end: str) -> pd.DataFrame:
    """
    Generate FX local projection results table.
    
    Requirements: 4.2 - Generate CSV with local projection impulse response data
    
    Args:
        lp_results: DataFrame from estimate_fx_local_projections
        sample_start: Sample start date
        sample_end: Sample end date
        
    Returns:
        DataFrame with FX impulse response results
    """
    df = lp_results.copy()
    df['sample_start'] = sample_start
    df['sample_end'] = sample_end
    df['estimation_method'] = 'Local Projection with Newey-West SE'
    
    return df


# =============================================================================
# Main Execution Function
# =============================================================================

def main(output_dir: Optional[str] = None, 
         start_date: str = "2008-01-01",
         end_date: str = "2023-12-31") -> None:
    """
    Main execution function for international spillovers replication.
    
    Requirements: 13.1, 13.3 - Accept CLI arguments, print summary of outputs
    
    Args:
        output_dir: Base output directory (default: 'output')
        start_date: Sample start date
        end_date: Sample end date
    """
    logger.info("=" * 60)
    logger.info("International Spillovers Replication Script")
    logger.info("=" * 60)
    
    # Create output directory
    output_path = create_output_dir(output_dir or "output")
    
    # Initialize FRED client
    try:
        client = FREDClient()
    except ValueError as e:
        logger.error(f"Failed to initialize FRED client: {e}")
        print("\nError: FRED API key required. Please set FRED_API_KEY environment variable.")
        sys.exit(1)
    
    # Track metadata
    metadata = {
        'sample_start': start_date,
        'sample_end': end_date,
        'data_sources': {
            'foreign_holdings': ['FDHBPIN', 'FDHBFIN'],
            'fed_balance_sheet': ['TREAST', 'MVMTD027SBEA'],
            'exchange_rate': ['DTWEXBGS'],
            'treasury_yields': ['DGS10']
        },
        'fomc_dates_count': len(get_fomc_dates(
            int(start_date[:4]), int(end_date[:4])
        ))
    }
    
    generated_files = []
    
    # -------------------------------------------------------------------------
    # Step 1: Fetch Data
    # -------------------------------------------------------------------------
    logger.info("\n--- Step 1: Fetching Data ---")
    
    # Fetch holdings data
    holdings = fetch_holdings_data(client, start_date, end_date)
    if holdings is not None:
        save_dataframe(holdings, output_path / "raw_holdings.csv", "Holdings data")
        generated_files.append("raw_holdings.csv")
    
    # Fetch Fed balance sheet data
    balance_sheet = fetch_fed_balance_sheet(client, start_date, end_date)
    if balance_sheet is not None:
        save_dataframe(balance_sheet, output_path / "raw_balance_sheet.csv", 
                      "Balance sheet data")
        generated_files.append("raw_balance_sheet.csv")
    
    # Fetch FX data
    fx = fetch_fx_data(client, start_date, end_date)
    if fx is not None:
        fx.to_csv(output_path / "raw_fx.csv")
        logger.info(f"Saved FX data: {len(fx)} observations")
        generated_files.append("raw_fx.csv")
    
    # Fetch Treasury yields
    yields = fetch_treasury_yields(client, start_date, end_date)
    if yields is not None:
        yields.to_csv(output_path / "raw_yields.csv")
        logger.info(f"Saved yields data: {len(yields)} observations")
        generated_files.append("raw_yields.csv")

    
    # -------------------------------------------------------------------------
    # Step 2: Compute QE Intensity
    # -------------------------------------------------------------------------
    logger.info("\n--- Step 2: Computing QE Intensity ---")
    
    if balance_sheet is not None:
        qe_intensity = compute_qe_intensity(
            balance_sheet['fed_treasury_holdings'],
            balance_sheet['total_marketable_debt']
        )
        qe_intensity.to_csv(output_path / "qe_intensity.csv")
        logger.info(f"Computed QE intensity: {len(qe_intensity)} observations")
        generated_files.append("qe_intensity.csv")
    else:
        logger.warning("Skipping QE intensity computation - missing balance sheet data")
        qe_intensity = None
    
    # -------------------------------------------------------------------------
    # Step 3: Construct QE Shocks
    # -------------------------------------------------------------------------
    logger.info("\n--- Step 3: Constructing QE Shocks ---")
    
    fomc_dates = get_fomc_dates(int(start_date[:4]), int(end_date[:4]))
    
    if yields is not None:
        qe_shocks = construct_qe_shocks(yields, fomc_dates)
        qe_shocks.to_csv(output_path / "qe_shocks_daily.csv")
        generated_files.append("qe_shocks_daily.csv")
        
        monthly_shocks = aggregate_shocks_monthly(qe_shocks)
        monthly_shocks.to_csv(output_path / "qe_shocks_monthly.csv")
        generated_files.append("qe_shocks_monthly.csv")
    else:
        logger.warning("Skipping QE shock construction - missing yields data")
        qe_shocks = None
        monthly_shocks = None
    
    # -------------------------------------------------------------------------
    # Step 4: Holdings Regressions
    # -------------------------------------------------------------------------
    logger.info("\n--- Step 4: Estimating Holdings Regressions ---")
    
    holdings_results = {}
    if holdings is not None and qe_intensity is not None:
        for holder_type in ['total', 'official', 'private']:
            try:
                result = estimate_holdings_regression(
                    holdings, qe_intensity, holder_type
                )
                holdings_results[holder_type] = result
                logger.info(f"  {holder_type}: coef={result.coefficients.get('qe_intensity', np.nan):.4f}, "
                           f"R²={result.r_squared:.4f}, n={result.n_obs}")
            except Exception as e:
                logger.warning(f"  Failed to estimate {holder_type} regression: {e}")
        
        # Generate Table 1
        if holdings_results:
            table1 = generate_table1(holdings_results, start_date, end_date)
            save_dataframe(table1, output_path / "table1_holdings_effect.csv", "Table 1")
            generated_files.append("table1_holdings_effect.csv")
    else:
        logger.warning("Skipping holdings regressions - missing data")

    
    # -------------------------------------------------------------------------
    # Step 5: FX Local Projections
    # -------------------------------------------------------------------------
    logger.info("\n--- Step 5: Estimating FX Local Projections ---")
    
    if fx is not None and monthly_shocks is not None:
        # Resample FX to monthly for alignment with shocks
        fx_monthly = fx.resample('ME').last()
        
        lp_results = estimate_fx_local_projections(
            fx_monthly, monthly_shocks, horizons=range(19)
        )
        
        fx_results = generate_fx_results(lp_results, start_date, end_date)
        save_dataframe(fx_results, output_path / "fx_local_projections.csv", 
                      "FX local projections")
        generated_files.append("fx_local_projections.csv")
    else:
        logger.warning("Skipping FX local projections - missing data")
    
    # -------------------------------------------------------------------------
    # Step 6: Save Metadata and Summary
    # -------------------------------------------------------------------------
    logger.info("\n--- Step 6: Saving Metadata ---")
    
    metadata['generated_files'] = generated_files
    save_metadata(output_path, metadata)
    generated_files.append("metadata.json")
    
    # Print summary
    print("\n" + "=" * 60)
    print("EXECUTION COMPLETE")
    print("=" * 60)
    print(f"\nOutput directory: {output_path}")
    print(f"\nGenerated files ({len(generated_files)}):")
    for f in generated_files:
        print(f"  - {f}")
    print()


# =============================================================================
# CLI Entry Point
# =============================================================================

def parse_args():
    """
    Parse command-line arguments.
    
    Requirements: 13.1 - Accept command-line arguments for output directory and sample period
    """
    parser = argparse.ArgumentParser(
        description="Replicate International Spillovers of U.S. QE analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python replicate_international_spillovers.py
  python replicate_international_spillovers.py --output results
  python replicate_international_spillovers.py --start 2010-01-01 --end 2020-12-31
        """
    )
    
    parser.add_argument(
        '--output', '-o',
        type=str,
        default='output',
        help='Base output directory (default: output)'
    )
    
    parser.add_argument(
        '--start', '-s',
        type=str,
        default='2008-01-01',
        help='Sample start date YYYY-MM-DD (default: 2008-01-01)'
    )
    
    parser.add_argument(
        '--end', '-e',
        type=str,
        default='2023-12-31',
        help='Sample end date YYYY-MM-DD (default: 2023-12-31)'
    )
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(
        output_dir=args.output,
        start_date=args.start,
        end_date=args.end
    )
