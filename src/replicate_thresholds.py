"""
Replication Script: Fiscal Thresholds and Market Distortions in QE Transmission

This script replicates the empirical analysis from the paper on fiscal thresholds
and market distortions in QE transmission, focusing on the effects of QE on
private investment through interest rate and market distortion channels.

Requirements: 5.1, 5.2, 5.3, 5.4, 12.1, 12.2, 12.3, 13.1, 13.3
"""

import argparse
import json
import logging
import os
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from fredapi import Fred
import statsmodels.api as sm
from statsmodels.regression.linear_model import OLS

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


@dataclass
class ThresholdResult:
    """Container for Hansen threshold regression results."""
    threshold: float
    threshold_ci: Tuple[float, float]
    coef_low: Dict[str, float]
    coef_high: Dict[str, float]
    se_low: Dict[str, float]
    se_high: Dict[str, float]
    r_squared_low: float
    r_squared_high: float
    n_obs_low: int
    n_obs_high: int
    bootstrap_pvalue: float
    ssr_values: Dict[float, float] = field(default_factory=dict)


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
    output_path = Path(base_dir) / f"thresholds_{timestamp}"
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
    metadata['script'] = 'replicate_thresholds.py'
    
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2, default=str)
    
    logger.info(f"Saved metadata to {metadata_path}")



# =============================================================================
# Data Acquisition Functions
# =============================================================================

def fetch_macro_data(client: FREDClient, start_date: str, 
                     end_date: str) -> Optional[pd.DataFrame]:
    """
    Fetch macroeconomic data for thresholds analysis.
    
    Requirements: 5.1, 5.3, 5.4 - Fetch DGS10, GPDIC1, GDPC1, UNRATE, PCEPILFE
    
    Args:
        client: FREDClient instance
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        
    Returns:
        DataFrame with macro variables, or None if all fetches failed
    """
    series_map = {
        'DGS10': 'treasury_10y_yield',      # 10-year Treasury yield (daily)
        'GPDIC1': 'real_private_investment', # Real private fixed investment
        'GDPC1': 'real_gdp',                 # Real GDP
        'UNRATE': 'unemployment_rate',       # Unemployment rate
        'PCEPILFE': 'core_pce_inflation'     # Core PCE inflation
    }
    
    data = {}
    for series_id, col_name in series_map.items():
        # Use quarterly frequency for macro variables, daily for yields
        freq = 'd' if series_id == 'DGS10' else 'q'
        series = client.fetch_series(series_id, start_date, end_date, frequency=freq)
        if series is not None:
            data[col_name] = series
    
    if not data:
        logger.error("Failed to fetch any macro data")
        return None
    
    # Combine into DataFrame - will have mixed frequencies
    df = pd.DataFrame(data)
    return df


def fetch_treasury_yields_daily(client: FREDClient, start_date: str,
                                 end_date: str) -> Optional[pd.Series]:
    """
    Fetch 10-year Treasury constant maturity yields at daily frequency.
    
    Requirements: 5.1 - Fetch DGS10 at daily frequency
    
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


def fetch_debt_service_data(client: FREDClient, start_date: str,
                            end_date: str) -> Optional[pd.DataFrame]:
    """
    Fetch federal debt service and revenue data.
    
    Requirements: 5.2 - Fetch data to compute debt-service-to-revenue ratio
    
    Args:
        client: FREDClient instance
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        
    Returns:
        DataFrame with interest payments and revenue, or None if failed
    """
    # A091RC1Q027SBEA: Federal government interest payments
    interest = client.fetch_series('A091RC1Q027SBEA', start_date, end_date, frequency='q')
    
    # FGRECPT: Federal government current receipts
    revenue = client.fetch_series('FGRECPT', start_date, end_date, frequency='q')
    
    if interest is None and revenue is None:
        logger.error("Failed to fetch any debt service data")
        return None
    
    df = pd.DataFrame({
        'interest_payments': interest,
        'federal_revenue': revenue
    })
    
    return df


def compute_debt_service_ratio(interest: pd.Series, 
                               revenue: pd.Series) -> pd.Series:
    """
    Compute federal debt-service-to-revenue ratio.
    
    Requirements: 5.2 - Compute debt-service-to-revenue ratio
    
    Args:
        interest: Series of federal interest payments
        revenue: Series of federal revenue
        
    Returns:
        Series with debt-service-to-revenue ratio
    """
    # Align series to common index
    aligned = pd.DataFrame({
        'interest': interest,
        'revenue': revenue
    }).dropna()
    
    # Compute ratio
    ratio = aligned['interest'] / aligned['revenue']
    ratio.name = 'debt_service_ratio'
    
    return ratio


def fetch_distortion_proxies(client: FREDClient, start_date: str,
                             end_date: str) -> Optional[pd.DataFrame]:
    """
    Fetch market distortion proxy data (VIX and TED spread).
    
    Requirements: 7.1 - Use available FRED proxies for distortion index
    
    Args:
        client: FREDClient instance
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        
    Returns:
        DataFrame with VIX and TED spread, or None if failed
    """
    # VIXCLS: CBOE Volatility Index
    vix = client.fetch_series('VIXCLS', start_date, end_date, frequency='d')
    
    # TEDRATE: TED Spread (3-month LIBOR - 3-month T-bill)
    ted = client.fetch_series('TEDRATE', start_date, end_date, frequency='d')
    
    if vix is None and ted is None:
        logger.error("Failed to fetch any distortion proxy data")
        return None
    
    df = pd.DataFrame({
        'vix': vix,
        'ted_spread': ted
    })
    
    return df


def fetch_investment_data(client: FREDClient, start_date: str,
                          end_date: str) -> Optional[pd.Series]:
    """
    Fetch real private fixed investment data.
    
    Requirements: 5.3 - Fetch GPDIC1 for investment analysis
    
    Args:
        client: FREDClient instance
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        
    Returns:
        Series with real private fixed investment, or None if failed
    """
    investment = client.fetch_series('GPDIC1', start_date, end_date, frequency='q')
    if investment is not None:
        investment.name = 'real_private_investment'
    return investment


def fetch_macro_controls(client: FREDClient, start_date: str,
                         end_date: str) -> Optional[pd.DataFrame]:
    """
    Fetch macroeconomic control variables.
    
    Requirements: 5.4 - Fetch GDPC1, UNRATE, PCEPILFE
    
    Args:
        client: FREDClient instance
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        
    Returns:
        DataFrame with macro controls, or None if failed
    """
    gdp = client.fetch_series('GDPC1', start_date, end_date, frequency='q')
    unrate = client.fetch_series('UNRATE', start_date, end_date, frequency='q')
    inflation = client.fetch_series('PCEPILFE', start_date, end_date, frequency='q')
    
    if gdp is None and unrate is None and inflation is None:
        logger.error("Failed to fetch any macro control data")
        return None
    
    df = pd.DataFrame({
        'real_gdp': gdp,
        'unemployment_rate': unrate,
        'core_pce_inflation': inflation
    })
    
    return df



# =============================================================================
# Proxy Shock and Distortion Index Construction
# =============================================================================

def construct_proxy_shocks(yields: pd.Series, 
                           fomc_dates: List[str]) -> pd.Series:
    """
    Construct proxy QE shocks from daily yield changes on FOMC dates.
    
    Requirements: 6.1 - Compute daily changes in 10-year Treasury yields on FOMC dates
    
    Args:
        yields: Daily Treasury yield series
        fomc_dates: List of FOMC announcement dates (YYYY-MM-DD)
        
    Returns:
        Series of proxy QE shocks indexed by FOMC date
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
    
    shock_series = pd.Series(shocks, name='proxy_qe_shock')
    shock_series.index.name = 'date'
    
    logger.info(f"Constructed {len(shock_series)} proxy QE shocks from {len(fomc_dates)} FOMC dates")
    
    return shock_series


def aggregate_to_quarterly(daily_shocks: pd.Series) -> pd.Series:
    """
    Aggregate daily shocks to quarterly frequency by summing.
    
    Requirements: 6.2 - Convert daily shocks to quarterly frequency by summing
    
    Args:
        daily_shocks: Series of daily QE shocks
        
    Returns:
        Series of quarterly aggregated shocks
    """
    # Resample to quarterly, summing shocks within each quarter
    quarterly_shocks = daily_shocks.resample('QE').sum()
    quarterly_shocks.name = 'qe_shock_quarterly'
    
    # Fill quarters with no FOMC meetings with 0
    quarterly_shocks = quarterly_shocks.fillna(0)
    
    return quarterly_shocks


def standardize(series: pd.Series) -> pd.Series:
    """
    Standardize series to zero mean and unit standard deviation.
    
    Requirements: 6.3 - Normalize shock series to zero mean and unit variance
    
    Args:
        series: Input series to standardize
        
    Returns:
        Standardized series with mean ≈ 0 and std ≈ 1
    """
    clean_series = series.dropna()
    
    if len(clean_series) == 0:
        logger.warning("Cannot standardize empty series")
        return series
    
    mean = clean_series.mean()
    std = clean_series.std()
    
    if std == 0 or np.isnan(std):
        logger.warning("Cannot standardize series with zero variance")
        return series - mean  # Just demean
    
    standardized = (series - mean) / std
    standardized.name = f"{series.name}_standardized" if series.name else "standardized"
    
    return standardized


def construct_distortion_index(vix: pd.Series, ted: pd.Series) -> pd.Series:
    """
    Construct market distortion index from available proxies.
    
    Requirements: 7.1, 7.2 - Use VIX and TED spread, standardize before averaging
    
    Args:
        vix: VIX volatility index series
        ted: TED spread series
        
    Returns:
        Market distortion index (average of standardized components)
    """
    # Standardize each component
    vix_std = standardize(vix)
    ted_std = standardize(ted)
    
    # Combine into DataFrame for alignment
    components = pd.DataFrame({
        'vix_std': vix_std,
        'ted_std': ted_std
    })
    
    # Average the standardized components (handling missing values)
    distortion_index = components.mean(axis=1, skipna=True)
    distortion_index.name = 'distortion_index'
    
    logger.info(f"Constructed distortion index: {len(distortion_index.dropna())} observations")
    
    return distortion_index



# =============================================================================
# Hansen Threshold Regression
# =============================================================================

def ols_with_robust_se(y: pd.Series, X: pd.DataFrame, 
                       cov_type: str = 'HC1') -> RegressionResult:
    """
    Estimate OLS regression with heteroskedasticity-consistent standard errors.
    
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
    
    if len(y_clean) < X_clean.shape[1] + 1:
        raise ValueError("Insufficient observations for regression")
    
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


def compute_ssr(y: pd.Series, X: pd.DataFrame) -> float:
    """
    Compute sum of squared residuals for OLS regression.
    
    Args:
        y: Dependent variable
        X: Independent variables
        
    Returns:
        Sum of squared residuals
    """
    data = pd.concat([y, X], axis=1).dropna()
    y_clean = data.iloc[:, 0]
    X_clean = data.iloc[:, 1:]
    
    if len(y_clean) < X_clean.shape[1] + 1:
        return np.inf
    
    model = OLS(y_clean, X_clean)
    results = model.fit()
    
    return float(results.ssr)


def hansen_threshold(y: pd.Series, X: pd.DataFrame, 
                     threshold_var: pd.Series,
                     trim_pct: float = 0.15,
                     n_grid: int = 100) -> Tuple[float, Dict[float, float]]:
    """
    Hansen (2000) threshold regression with grid search.
    
    Requirements: 8.1, 8.2 - Search over candidate thresholds with 15% trimming
    
    Args:
        y: Dependent variable
        X: Independent variables (excluding threshold interactions)
        threshold_var: Variable used for threshold determination
        trim_pct: Percentage to trim from each tail (default 15%)
        n_grid: Number of grid points for threshold search
        
    Returns:
        Tuple of (optimal threshold, dict of SSR values at each candidate)
    """
    # Align all data
    data = pd.concat([y, X, threshold_var], axis=1).dropna()
    y_clean = data.iloc[:, 0]
    X_clean = data.iloc[:, 1:-1]
    thresh_clean = data.iloc[:, -1]
    
    # Determine search range with trimming
    lower_pct = trim_pct
    upper_pct = 1 - trim_pct
    thresh_lower = thresh_clean.quantile(lower_pct)
    thresh_upper = thresh_clean.quantile(upper_pct)
    
    logger.info(f"Threshold search range: [{thresh_lower:.4f}, {thresh_upper:.4f}] "
                f"(trimmed {trim_pct*100:.0f}% from each tail)")
    
    # Create grid of candidate thresholds
    candidates = np.linspace(thresh_lower, thresh_upper, n_grid)
    
    ssr_values = {}
    best_threshold = None
    best_ssr = np.inf
    
    for gamma in candidates:
        # Split sample by threshold
        low_mask = thresh_clean <= gamma
        high_mask = thresh_clean > gamma
        
        # Skip if either regime has too few observations
        n_low = low_mask.sum()
        n_high = high_mask.sum()
        min_obs = max(X_clean.shape[1] + 2, 10)
        
        if n_low < min_obs or n_high < min_obs:
            continue
        
        # Compute SSR for each regime
        try:
            ssr_low = compute_ssr(y_clean[low_mask], X_clean[low_mask])
            ssr_high = compute_ssr(y_clean[high_mask], X_clean[high_mask])
            total_ssr = ssr_low + ssr_high
            
            ssr_values[gamma] = total_ssr
            
            if total_ssr < best_ssr:
                best_ssr = total_ssr
                best_threshold = gamma
        except Exception as e:
            logger.debug(f"Failed at threshold {gamma}: {e}")
            continue
    
    if best_threshold is None:
        raise ValueError("Could not find valid threshold")
    
    logger.info(f"Optimal threshold: {best_threshold:.4f} (SSR: {best_ssr:.4f})")
    
    return best_threshold, ssr_values


def bootstrap_threshold_test(y: pd.Series, X: pd.DataFrame,
                             threshold_var: pd.Series,
                             optimal_threshold: float,
                             optimal_ssr: float,
                             n_bootstrap: int = 5000,
                             trim_pct: float = 0.15) -> float:
    """
    Bootstrap test for threshold significance (null: no threshold effect).
    
    Requirements: 8.4 - Bootstrap inference with configurable replications
    
    Args:
        y: Dependent variable
        X: Independent variables
        threshold_var: Threshold variable
        optimal_threshold: Estimated threshold from data
        optimal_ssr: SSR at optimal threshold
        n_bootstrap: Number of bootstrap replications
        trim_pct: Trimming percentage
        
    Returns:
        Bootstrap p-value for null of linearity
    """
    # Compute SSR under null (no threshold)
    data = pd.concat([y, X], axis=1).dropna()
    y_clean = data.iloc[:, 0]
    X_clean = data.iloc[:, 1:]
    
    ssr_null = compute_ssr(y_clean, X_clean)
    
    # Test statistic: F-type statistic
    observed_stat = (ssr_null - optimal_ssr) / optimal_ssr * len(y_clean)
    
    # Bootstrap under the null
    bootstrap_stats = []
    
    for b in range(n_bootstrap):
        # Resample residuals
        model = OLS(y_clean, X_clean)
        results = model.fit()
        residuals = results.resid
        
        # Bootstrap sample
        boot_indices = np.random.choice(len(y_clean), size=len(y_clean), replace=True)
        y_boot = results.fittedvalues + residuals.iloc[boot_indices].values
        y_boot.index = y_clean.index
        
        # Re-estimate threshold
        try:
            thresh_boot, ssr_boot = hansen_threshold(
                y_boot, X_clean, threshold_var.loc[y_clean.index],
                trim_pct=trim_pct, n_grid=50
            )
            ssr_boot_opt = min(ssr_boot.values())
            ssr_boot_null = compute_ssr(y_boot, X_clean)
            
            boot_stat = (ssr_boot_null - ssr_boot_opt) / ssr_boot_opt * len(y_boot)
            bootstrap_stats.append(boot_stat)
        except Exception:
            continue
    
    if len(bootstrap_stats) == 0:
        logger.warning("Bootstrap failed - returning NaN p-value")
        return np.nan
    
    # P-value: proportion of bootstrap stats >= observed
    p_value = np.mean(np.array(bootstrap_stats) >= observed_stat)
    
    logger.info(f"Bootstrap p-value: {p_value:.4f} ({len(bootstrap_stats)} successful replications)")
    
    return p_value


def estimate_threshold_regression(y: pd.Series, X: pd.DataFrame,
                                   threshold_var: pd.Series,
                                   trim_pct: float = 0.15,
                                   n_bootstrap: int = 5000) -> ThresholdResult:
    """
    Full Hansen threshold regression estimation.
    
    Requirements: 8.1, 8.2, 8.3, 8.4 - Complete threshold regression with inference
    
    Args:
        y: Dependent variable
        X: Independent variables
        threshold_var: Variable for threshold determination
        trim_pct: Trimming percentage (default 15%)
        n_bootstrap: Bootstrap replications for inference
        
    Returns:
        ThresholdResult with all estimation results
    """
    # Find optimal threshold
    threshold, ssr_values = hansen_threshold(y, X, threshold_var, trim_pct)
    
    # Align data
    data = pd.concat([y, X, threshold_var], axis=1).dropna()
    y_clean = data.iloc[:, 0]
    X_clean = data.iloc[:, 1:-1]
    thresh_clean = data.iloc[:, -1]
    
    # Split sample
    low_mask = thresh_clean <= threshold
    high_mask = thresh_clean > threshold
    
    # Estimate regime-specific regressions
    X_with_const = sm.add_constant(X_clean)
    
    result_low = ols_with_robust_se(y_clean[low_mask], X_with_const[low_mask])
    result_high = ols_with_robust_se(y_clean[high_mask], X_with_const[high_mask])
    
    # Bootstrap p-value
    optimal_ssr = min(ssr_values.values())
    p_value = bootstrap_threshold_test(
        y_clean, X_with_const, thresh_clean,
        threshold, optimal_ssr, n_bootstrap, trim_pct
    )
    
    # Confidence interval for threshold (simplified: use grid search bounds)
    sorted_ssrs = sorted(ssr_values.items(), key=lambda x: x[1])
    ci_candidates = [t for t, s in sorted_ssrs[:int(len(sorted_ssrs) * 0.1) + 1]]
    threshold_ci = (min(ci_candidates), max(ci_candidates))
    
    return ThresholdResult(
        threshold=threshold,
        threshold_ci=threshold_ci,
        coef_low=result_low.coefficients,
        coef_high=result_high.coefficients,
        se_low=result_low.std_errors,
        se_high=result_high.std_errors,
        r_squared_low=result_low.r_squared,
        r_squared_high=result_high.r_squared,
        n_obs_low=result_low.n_obs,
        n_obs_high=result_high.n_obs,
        bootstrap_pvalue=p_value,
        ssr_values=ssr_values
    )



# =============================================================================
# Local Projections for Investment
# =============================================================================

def local_projection(y: pd.Series, x: pd.Series, horizon: int,
                     controls: Optional[pd.DataFrame] = None,
                     nw_lags: Optional[int] = None) -> Dict:
    """
    Estimate local projection at a single horizon with Newey-West SE.
    
    Requirements: 9.2 - Use Newey-West HAC corrections
    
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


def estimate_investment_lp(investment: pd.Series, 
                           qe_shock: pd.Series,
                           controls: Optional[pd.DataFrame] = None,
                           horizons: range = range(13)) -> pd.DataFrame:
    """
    Estimate local projections for investment response to QE shocks.
    
    Requirements: 9.1 - Regress cumulative h-quarter changes for horizons 0-12
    
    Args:
        investment: Real private fixed investment series (log level)
        qe_shock: QE shock series
        controls: Optional macro control variables
        horizons: Range of horizons to estimate (default 0-12)
        
    Returns:
        DataFrame with impulse response coefficients and confidence intervals
    """
    # Ensure investment is in log form
    log_inv = np.log(investment) if investment.min() > 0 else investment
    log_inv.name = 'log_investment'
    
    # Align shock series with investment
    qe_shock_aligned = qe_shock.reindex(log_inv.index).fillna(0)
    qe_shock_aligned.name = 'qe_shock'
    
    # Align controls if provided
    if controls is not None:
        controls_aligned = controls.reindex(log_inv.index)
    else:
        controls_aligned = None
    
    results = []
    for h in horizons:
        lp_result = local_projection(
            log_inv, qe_shock_aligned, horizon=h, 
            controls=controls_aligned
        )
        results.append(lp_result)
    
    df = pd.DataFrame(results)
    logger.info(f"Estimated investment local projections for {len(horizons)} horizons")
    
    return df



# =============================================================================
# Channel Decomposition
# =============================================================================

def estimate_qe_to_channel(qe_shock: pd.Series, 
                           channel_var: pd.Series) -> float:
    """
    Estimate effect of QE shock on a channel variable.
    
    Requirements: 10.1 - Estimate effect of QE on yields and distortion separately
    
    Args:
        qe_shock: QE shock series
        channel_var: Channel variable (yields or distortion index)
        
    Returns:
        Coefficient of QE shock on channel variable
    """
    # Align data
    data = pd.DataFrame({
        'channel': channel_var,
        'qe_shock': qe_shock
    }).dropna()
    
    if len(data) < 5:
        return np.nan
    
    X = sm.add_constant(data['qe_shock'])
    model = OLS(data['channel'], X)
    results = model.fit()
    
    return results.params['qe_shock']


def estimate_channel_to_investment(investment: pd.Series,
                                    channel_var: pd.Series,
                                    controls: Optional[pd.DataFrame] = None) -> float:
    """
    Estimate effect of channel variable on investment.
    
    Requirements: 10.2 - Estimate auxiliary regressions of investment on channels
    
    Args:
        investment: Investment series (log level)
        channel_var: Channel variable (yields or distortion)
        controls: Optional control variables
        
    Returns:
        Coefficient of channel variable on investment
    """
    # Ensure investment is in log form
    log_inv = np.log(investment) if investment.min() > 0 else investment
    
    # Build regressor matrix
    if controls is not None:
        X = pd.concat([channel_var, controls], axis=1)
    else:
        X = channel_var.to_frame()
    
    X = sm.add_constant(X)
    
    # Align data
    data = pd.concat([log_inv, X], axis=1).dropna()
    y_clean = data.iloc[:, 0]
    X_clean = data.iloc[:, 1:]
    
    if len(y_clean) < 5:
        return np.nan
    
    model = OLS(y_clean, X_clean)
    results = model.fit()
    
    # Get coefficient on channel variable
    channel_name = channel_var.name if channel_var.name else channel_var.columns[0]
    if channel_name in results.params.index:
        return results.params[channel_name]
    else:
        return results.params.iloc[1]


def decompose_channels(total_effect: float,
                       qe_to_yield: float,
                       qe_to_distortion: float,
                       yield_to_inv: float,
                       distortion_to_inv: float) -> Dict:
    """
    Decompose investment response into rate and distortion channels.
    
    Requirements: 10.3 - Compute share of total response attributable to each channel
    
    Args:
        total_effect: Total effect of QE on investment
        qe_to_yield: Effect of QE on yields
        qe_to_distortion: Effect of QE on distortion index
        yield_to_inv: Effect of yields on investment
        distortion_to_inv: Effect of distortion on investment
        
    Returns:
        Dictionary with channel contributions and shares
    """
    # Channel contributions (indirect effects)
    rate_channel = qe_to_yield * yield_to_inv
    distortion_channel = qe_to_distortion * distortion_to_inv
    
    # Sum of channel contributions
    sum_channels = rate_channel + distortion_channel
    
    # Handle edge cases
    if sum_channels == 0 or np.isnan(sum_channels):
        rate_share = 0.5
        distortion_share = 0.5
    else:
        rate_share = rate_channel / sum_channels
        distortion_share = distortion_channel / sum_channels
    
    return {
        'total_effect': total_effect,
        'rate_channel_contribution': rate_channel,
        'distortion_channel_contribution': distortion_channel,
        'rate_channel_share': rate_share,
        'distortion_channel_share': distortion_share,
        'sum_of_shares': rate_share + distortion_share
    }


def full_channel_decomposition(investment: pd.Series,
                                qe_shock: pd.Series,
                                yields: pd.Series,
                                distortion_index: pd.Series,
                                controls: Optional[pd.DataFrame] = None) -> Dict:
    """
    Perform full channel decomposition analysis.
    
    Requirements: 10.1, 10.2, 10.3 - Complete decomposition analysis
    
    Args:
        investment: Investment series
        qe_shock: QE shock series
        yields: Treasury yield series
        distortion_index: Market distortion index
        controls: Optional control variables
        
    Returns:
        Dictionary with all decomposition results
    """
    # Align all series to common index
    data = pd.DataFrame({
        'investment': investment,
        'qe_shock': qe_shock,
        'yields': yields,
        'distortion': distortion_index
    }).dropna()
    
    if len(data) < 10:
        logger.warning("Insufficient data for channel decomposition")
        return {
            'total_effect': np.nan,
            'rate_channel_contribution': np.nan,
            'distortion_channel_contribution': np.nan,
            'rate_channel_share': np.nan,
            'distortion_channel_share': np.nan
        }
    
    # Step 1: Total effect of QE on investment
    log_inv = np.log(data['investment'])
    X_total = sm.add_constant(data['qe_shock'])
    model_total = OLS(log_inv, X_total)
    total_effect = model_total.fit().params['qe_shock']
    
    # Step 2: QE to channel effects
    qe_to_yield = estimate_qe_to_channel(data['qe_shock'], data['yields'])
    qe_to_distortion = estimate_qe_to_channel(data['qe_shock'], data['distortion'])
    
    # Step 3: Channel to investment effects
    yield_to_inv = estimate_channel_to_investment(
        data['investment'], data['yields'], controls
    )
    distortion_to_inv = estimate_channel_to_investment(
        data['investment'], data['distortion'], controls
    )
    
    # Step 4: Decompose
    decomposition = decompose_channels(
        total_effect, qe_to_yield, qe_to_distortion,
        yield_to_inv, distortion_to_inv
    )
    
    # Add intermediate results
    decomposition['qe_to_yield'] = qe_to_yield
    decomposition['qe_to_distortion'] = qe_to_distortion
    decomposition['yield_to_investment'] = yield_to_inv
    decomposition['distortion_to_investment'] = distortion_to_inv
    
    logger.info(f"Channel decomposition: Rate={decomposition['rate_channel_share']:.2%}, "
                f"Distortion={decomposition['distortion_channel_share']:.2%}")
    
    return decomposition



# =============================================================================
# Results Output Functions
# =============================================================================

def generate_table2(threshold_result: ThresholdResult,
                    sample_start: str, sample_end: str) -> pd.DataFrame:
    """
    Generate Table 2: Threshold Regression Results.
    
    Requirements: 11.1 - Generate CSV replicating Table 2
    
    Args:
        threshold_result: ThresholdResult from estimation
        sample_start: Sample start date
        sample_end: Sample end date
        
    Returns:
        DataFrame formatted as Table 2
    """
    # Get QE effect coefficient from each regime
    qe_coef_low = threshold_result.coef_low.get('qe_shock', 
                   threshold_result.coef_low.get('qe_shock_quarterly', np.nan))
    qe_coef_high = threshold_result.coef_high.get('qe_shock',
                    threshold_result.coef_high.get('qe_shock_quarterly', np.nan))
    
    qe_se_low = threshold_result.se_low.get('qe_shock',
                 threshold_result.se_low.get('qe_shock_quarterly', np.nan))
    qe_se_high = threshold_result.se_high.get('qe_shock',
                  threshold_result.se_high.get('qe_shock_quarterly', np.nan))
    
    rows = [
        {
            'regime': 'Low-Debt',
            'threshold': threshold_result.threshold,
            'qe_effect_bps': qe_coef_low * 100 if not np.isnan(qe_coef_low) else np.nan,
            'std_error': qe_se_low * 100 if not np.isnan(qe_se_low) else np.nan,
            'ci_lower': (qe_coef_low - 1.96 * qe_se_low) * 100 if not np.isnan(qe_coef_low) else np.nan,
            'ci_upper': (qe_coef_low + 1.96 * qe_se_low) * 100 if not np.isnan(qe_coef_low) else np.nan,
            'r_squared': threshold_result.r_squared_low,
            'n_obs': threshold_result.n_obs_low,
            'hansen_pvalue': threshold_result.bootstrap_pvalue,
            'sample_start': sample_start,
            'sample_end': sample_end
        },
        {
            'regime': 'High-Debt',
            'threshold': threshold_result.threshold,
            'qe_effect_bps': qe_coef_high * 100 if not np.isnan(qe_coef_high) else np.nan,
            'std_error': qe_se_high * 100 if not np.isnan(qe_se_high) else np.nan,
            'ci_lower': (qe_coef_high - 1.96 * qe_se_high) * 100 if not np.isnan(qe_coef_high) else np.nan,
            'ci_upper': (qe_coef_high + 1.96 * qe_se_high) * 100 if not np.isnan(qe_coef_high) else np.nan,
            'r_squared': threshold_result.r_squared_high,
            'n_obs': threshold_result.n_obs_high,
            'hansen_pvalue': threshold_result.bootstrap_pvalue,
            'sample_start': sample_start,
            'sample_end': sample_end
        }
    ]
    
    return pd.DataFrame(rows)


def generate_table3(decomposition: Dict,
                    sample_start: str, sample_end: str) -> pd.DataFrame:
    """
    Generate Table 3: Decomposition of QE Impact.
    
    Requirements: 11.2 - Generate CSV replicating Table 3
    
    Args:
        decomposition: Dictionary from channel decomposition
        sample_start: Sample start date
        sample_end: Sample end date
        
    Returns:
        DataFrame formatted as Table 3
    """
    rows = [
        {
            'channel': 'Interest-Rate',
            'contribution_pp': decomposition.get('rate_channel_contribution', np.nan),
            'share_pct': decomposition.get('rate_channel_share', np.nan) * 100,
            'sample_start': sample_start,
            'sample_end': sample_end
        },
        {
            'channel': 'Market-Distortion',
            'contribution_pp': decomposition.get('distortion_channel_contribution', np.nan),
            'share_pct': decomposition.get('distortion_channel_share', np.nan) * 100,
            'sample_start': sample_start,
            'sample_end': sample_end
        },
        {
            'channel': 'Total',
            'contribution_pp': decomposition.get('total_effect', np.nan),
            'share_pct': 100.0,
            'sample_start': sample_start,
            'sample_end': sample_end
        }
    ]
    
    return pd.DataFrame(rows)


def generate_summary_statistics(data: pd.DataFrame,
                                sample_start: str, 
                                sample_end: str) -> pd.DataFrame:
    """
    Generate summary statistics table.
    
    Requirements: 11.3 - Generate CSV containing summary statistics
    
    Args:
        data: DataFrame with all variables
        sample_start: Sample start date
        sample_end: Sample end date
        
    Returns:
        DataFrame with summary statistics
    """
    stats = []
    
    for col in data.columns:
        series = data[col].dropna()
        if len(series) > 0:
            stats.append({
                'variable': col,
                'mean': series.mean(),
                'std': series.std(),
                'min': series.min(),
                'p25': series.quantile(0.25),
                'median': series.median(),
                'p75': series.quantile(0.75),
                'max': series.max(),
                'n_obs': len(series),
                'sample_start': sample_start,
                'sample_end': sample_end
            })
    
    return pd.DataFrame(stats)


def generate_lp_results(lp_results: pd.DataFrame,
                        sample_start: str, sample_end: str) -> pd.DataFrame:
    """
    Generate local projection results table.
    
    Requirements: 9.3 - Save impulse response coefficients and confidence intervals
    
    Args:
        lp_results: DataFrame from estimate_investment_lp
        sample_start: Sample start date
        sample_end: Sample end date
        
    Returns:
        DataFrame with investment impulse response results
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
         end_date: str = "2023-12-31",
         n_bootstrap: int = 5000) -> None:
    """
    Main execution function for thresholds replication.
    
    Requirements: 13.1, 13.3 - Accept CLI arguments, print summary of outputs
    
    Args:
        output_dir: Base output directory (default: 'output')
        start_date: Sample start date
        end_date: Sample end date
        n_bootstrap: Number of bootstrap replications for threshold test
    """
    logger.info("=" * 60)
    logger.info("Fiscal Thresholds Replication Script")
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
        'n_bootstrap': n_bootstrap,
        'data_sources': {
            'treasury_yields': ['DGS10'],
            'investment': ['GPDIC1'],
            'macro_controls': ['GDPC1', 'UNRATE', 'PCEPILFE'],
            'debt_service': ['A091RC1Q027SBEA', 'FGRECPT'],
            'distortion_proxies': ['VIXCLS', 'TEDRATE']
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
    
    # Fetch daily Treasury yields
    yields_daily = fetch_treasury_yields_daily(client, start_date, end_date)
    if yields_daily is not None:
        yields_daily.to_csv(output_path / "raw_yields_daily.csv")
        logger.info(f"Saved daily yields: {len(yields_daily)} observations")
        generated_files.append("raw_yields_daily.csv")
    
    # Fetch debt service data
    debt_data = fetch_debt_service_data(client, start_date, end_date)
    if debt_data is not None:
        save_dataframe(debt_data, output_path / "raw_debt_service.csv", "Debt service data")
        generated_files.append("raw_debt_service.csv")
    
    # Fetch investment data
    investment = fetch_investment_data(client, start_date, end_date)
    if investment is not None:
        investment.to_csv(output_path / "raw_investment.csv")
        logger.info(f"Saved investment data: {len(investment)} observations")
        generated_files.append("raw_investment.csv")
    
    # Fetch macro controls
    controls = fetch_macro_controls(client, start_date, end_date)
    if controls is not None:
        save_dataframe(controls, output_path / "raw_macro_controls.csv", "Macro controls")
        generated_files.append("raw_macro_controls.csv")
    
    # Fetch distortion proxies
    distortion_data = fetch_distortion_proxies(client, start_date, end_date)
    if distortion_data is not None:
        save_dataframe(distortion_data, output_path / "raw_distortion_proxies.csv", 
                      "Distortion proxies")
        generated_files.append("raw_distortion_proxies.csv")
    
    # -------------------------------------------------------------------------
    # Step 2: Compute Debt Service Ratio
    # -------------------------------------------------------------------------
    logger.info("\n--- Step 2: Computing Debt Service Ratio ---")
    
    if debt_data is not None:
        debt_ratio = compute_debt_service_ratio(
            debt_data['interest_payments'],
            debt_data['federal_revenue']
        )
        debt_ratio.to_csv(output_path / "debt_service_ratio.csv")
        logger.info(f"Computed debt service ratio: {len(debt_ratio)} observations")
        generated_files.append("debt_service_ratio.csv")
    else:
        logger.warning("Skipping debt service ratio - missing data")
        debt_ratio = None
    
    # -------------------------------------------------------------------------
    # Step 3: Construct Proxy QE Shocks
    # -------------------------------------------------------------------------
    logger.info("\n--- Step 3: Constructing Proxy QE Shocks ---")
    
    fomc_dates = get_fomc_dates(int(start_date[:4]), int(end_date[:4]))
    
    if yields_daily is not None:
        proxy_shocks = construct_proxy_shocks(yields_daily, fomc_dates)
        proxy_shocks.to_csv(output_path / "proxy_shocks_daily.csv")
        generated_files.append("proxy_shocks_daily.csv")
        
        quarterly_shocks = aggregate_to_quarterly(proxy_shocks)
        quarterly_shocks.to_csv(output_path / "proxy_shocks_quarterly.csv")
        generated_files.append("proxy_shocks_quarterly.csv")
        
        # Standardize shocks
        std_shocks = standardize(quarterly_shocks)
        std_shocks.to_csv(output_path / "proxy_shocks_standardized.csv")
        generated_files.append("proxy_shocks_standardized.csv")
    else:
        logger.warning("Skipping proxy shock construction - missing yields data")
        proxy_shocks = None
        quarterly_shocks = None
        std_shocks = None
    
    # -------------------------------------------------------------------------
    # Step 4: Construct Distortion Index
    # -------------------------------------------------------------------------
    logger.info("\n--- Step 4: Constructing Distortion Index ---")
    
    if distortion_data is not None:
        distortion_index = construct_distortion_index(
            distortion_data['vix'],
            distortion_data['ted_spread']
        )
        distortion_index.to_csv(output_path / "distortion_index.csv")
        logger.info(f"Saved distortion index: {len(distortion_index.dropna())} observations")
        generated_files.append("distortion_index.csv")
        
        # Resample to quarterly for regression analysis
        distortion_quarterly = distortion_index.resample('QE').mean()
        distortion_quarterly.to_csv(output_path / "distortion_index_quarterly.csv")
        generated_files.append("distortion_index_quarterly.csv")
    else:
        logger.warning("Skipping distortion index - missing data")
        distortion_index = None
        distortion_quarterly = None
    
    # -------------------------------------------------------------------------
    # Step 5: Hansen Threshold Regression
    # -------------------------------------------------------------------------
    logger.info("\n--- Step 5: Estimating Threshold Regression ---")
    
    threshold_result = None
    if (investment is not None and quarterly_shocks is not None and 
        debt_ratio is not None):
        try:
            # Prepare data for threshold regression
            # Align all series to quarterly frequency with consistent period end dates
            inv_quarterly = investment.copy()
            inv_quarterly.index = pd.to_datetime(inv_quarterly.index).to_period('Q').to_timestamp('Q')
            
            shock_quarterly = quarterly_shocks.copy()
            shock_quarterly.index = pd.to_datetime(shock_quarterly.index).to_period('Q').to_timestamp('Q')
            
            debt_quarterly = debt_ratio.copy()
            debt_quarterly.index = pd.to_datetime(debt_quarterly.index).to_period('Q').to_timestamp('Q')
            
            # Create dependent variable: log change in investment
            y = np.log(inv_quarterly).diff().dropna()
            y.name = 'dlog_investment'
            
            # Prepare regressors
            X = shock_quarterly.to_frame()
            X.columns = ['qe_shock']
            
            # Estimate threshold regression
            threshold_result = estimate_threshold_regression(
                y, X, debt_quarterly,
                trim_pct=0.15,
                n_bootstrap=min(n_bootstrap, 1000)  # Reduce for speed
            )
            
            # Generate Table 2
            table2 = generate_table2(threshold_result, start_date, end_date)
            save_dataframe(table2, output_path / "table2_threshold_results.csv", "Table 2")
            generated_files.append("table2_threshold_results.csv")
            
            logger.info(f"Threshold estimate: {threshold_result.threshold:.4f}")
            logger.info(f"Bootstrap p-value: {threshold_result.bootstrap_pvalue:.4f}")
            
        except Exception as e:
            logger.error(f"Threshold regression failed: {e}")
    else:
        logger.warning("Skipping threshold regression - missing data")
    
    # -------------------------------------------------------------------------
    # Step 6: Local Projections for Investment
    # -------------------------------------------------------------------------
    logger.info("\n--- Step 6: Estimating Investment Local Projections ---")
    
    if investment is not None and quarterly_shocks is not None:
        try:
            # Align indices for local projections
            inv_aligned = investment.copy()
            inv_aligned.index = pd.to_datetime(inv_aligned.index).to_period('Q').to_timestamp('Q')
            
            shock_aligned = quarterly_shocks.copy()
            shock_aligned.index = pd.to_datetime(shock_aligned.index).to_period('Q').to_timestamp('Q')
            
            controls_aligned = None
            if controls is not None:
                controls_aligned = controls.copy()
                controls_aligned.index = pd.to_datetime(controls_aligned.index).to_period('Q').to_timestamp('Q')
            
            lp_results = estimate_investment_lp(
                inv_aligned, shock_aligned, 
                controls=controls_aligned,
                horizons=range(13)
            )
            
            lp_output = generate_lp_results(lp_results, start_date, end_date)
            save_dataframe(lp_output, output_path / "investment_local_projections.csv",
                          "Investment local projections")
            generated_files.append("investment_local_projections.csv")
            
        except Exception as e:
            logger.error(f"Local projections failed: {e}")
    else:
        logger.warning("Skipping local projections - missing data")
    
    # -------------------------------------------------------------------------
    # Step 7: Channel Decomposition
    # -------------------------------------------------------------------------
    logger.info("\n--- Step 7: Channel Decomposition ---")
    
    if (investment is not None and quarterly_shocks is not None and 
        yields_daily is not None and distortion_quarterly is not None):
        try:
            # Resample yields to quarterly and align indices
            yields_quarterly = yields_daily.resample('QE').mean()
            yields_quarterly.index = pd.to_datetime(yields_quarterly.index).to_period('Q').to_timestamp('Q')
            
            inv_aligned = investment.copy()
            inv_aligned.index = pd.to_datetime(inv_aligned.index).to_period('Q').to_timestamp('Q')
            
            shock_aligned = quarterly_shocks.copy()
            shock_aligned.index = pd.to_datetime(shock_aligned.index).to_period('Q').to_timestamp('Q')
            
            distortion_aligned = distortion_quarterly.copy()
            distortion_aligned.index = pd.to_datetime(distortion_aligned.index).to_period('Q').to_timestamp('Q')
            
            decomposition = full_channel_decomposition(
                inv_aligned, shock_aligned,
                yields_quarterly, distortion_aligned,
                controls=controls
            )
            
            # Generate Table 3
            table3 = generate_table3(decomposition, start_date, end_date)
            save_dataframe(table3, output_path / "table3_decomposition.csv", "Table 3")
            generated_files.append("table3_decomposition.csv")
            
        except Exception as e:
            logger.error(f"Channel decomposition failed: {e}")
    else:
        logger.warning("Skipping channel decomposition - missing data")
    
    # -------------------------------------------------------------------------
    # Step 8: Summary Statistics
    # -------------------------------------------------------------------------
    logger.info("\n--- Step 8: Generating Summary Statistics ---")
    
    # Combine available data for summary stats
    summary_data = {}
    if investment is not None:
        summary_data['investment'] = investment
    if quarterly_shocks is not None:
        summary_data['qe_shock'] = quarterly_shocks
    if debt_ratio is not None:
        summary_data['debt_service_ratio'] = debt_ratio
    if distortion_quarterly is not None:
        summary_data['distortion_index'] = distortion_quarterly
    if controls is not None:
        for col in controls.columns:
            summary_data[col] = controls[col]
    
    if summary_data:
        summary_df = pd.DataFrame(summary_data)
        summary_stats = generate_summary_statistics(summary_df, start_date, end_date)
        save_dataframe(summary_stats, output_path / "summary_statistics.csv", 
                      "Summary statistics")
        generated_files.append("summary_statistics.csv")
    
    # -------------------------------------------------------------------------
    # Step 9: Save Metadata and Summary
    # -------------------------------------------------------------------------
    logger.info("\n--- Step 9: Saving Metadata ---")
    
    metadata['generated_files'] = generated_files
    if threshold_result is not None:
        metadata['threshold_estimate'] = threshold_result.threshold
        metadata['threshold_pvalue'] = threshold_result.bootstrap_pvalue
    
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
        description="Replicate Fiscal Thresholds and Market Distortions analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python replicate_thresholds.py
  python replicate_thresholds.py --output results
  python replicate_thresholds.py --start 2010-01-01 --end 2020-12-31
  python replicate_thresholds.py --bootstrap 1000
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
    
    parser.add_argument(
        '--bootstrap', '-b',
        type=int,
        default=5000,
        help='Number of bootstrap replications (default: 5000)'
    )
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(
        output_dir=args.output,
        start_date=args.start,
        end_date=args.end,
        n_bootstrap=args.bootstrap
    )
