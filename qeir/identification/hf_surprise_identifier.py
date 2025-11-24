"""
High-Frequency FOMC Surprise Identification

This module implements high-frequency identification of FOMC policy surprises
following Swanson (2021) methodology. It extracts asset price changes within
narrow windows around FOMC announcements and decomposes them into target rate,
forward guidance, and QE factors using PCA.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
import json
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import warnings


class HFSurpriseIdentifier:
    """
    High-frequency FOMC surprise identifier.
    
    Implements Swanson (2021) methodology for identifying exogenous QE shocks
    using narrow time windows around FOMC announcements.
    
    Attributes:
        window_minutes: Width of window around announcements (default 30)
        fomc_dates: List of FOMC announcement timestamps
        hf_surprises: DataFrame of extracted high-frequency surprises
        policy_factors: Decomposed policy factors (target, FG, QE)
    """
    
    def __init__(self, window_minutes: int = 30, fomc_config_path: Optional[str] = None):
        """
        Initialize HF surprise identifier.
        
        Args:
            window_minutes: Width of window around FOMC announcements (default 30)
            fomc_config_path: Path to FOMC announcement configuration file
        """
        self.window_minutes = window_minutes
        self.fomc_dates = self._load_fomc_dates(fomc_config_path)
        self.hf_surprises = None
        self.policy_factors = None
        
    def _load_fomc_dates(self, config_path: Optional[str] = None) -> List[datetime]:
        """
        Load FOMC announcement dates from configuration file.
        
        Args:
            config_path: Path to configuration file with FOMC dates
            
        Returns:
            List of FOMC announcement timestamps
        """
        if config_path is None:
            # Try default location
            config_path = Path(__file__).parent.parent / 'config' / 'fomc_announcements.json'
        
        config_path = Path(config_path)
        
        if config_path.exists():
            try:
                with open(config_path, 'r') as f:
                    config = json.load(f)
                    
                # Parse dates from config
                dates = []
                for date_str in config.get('announcement_dates', []):
                    try:
                        # Try parsing with time
                        dt = pd.to_datetime(date_str)
                        dates.append(dt)
                    except Exception as e:
                        warnings.warn(f"Could not parse date {date_str}: {e}")
                        
                return sorted(dates)
                
            except Exception as e:
                warnings.warn(f"Could not load FOMC dates from {config_path}: {e}")
                return self._get_default_fomc_dates()
        else:
            warnings.warn(f"FOMC config file not found at {config_path}, using defaults")
            return self._get_default_fomc_dates()
    
    def _get_default_fomc_dates(self) -> List[datetime]:
        """
        Get default FOMC announcement dates for 2008-2023 period.
        
        Returns:
            List of FOMC announcement timestamps (approximate)
        """
        # Default FOMC dates (typically 8 meetings per year)
        # These are approximate - should be replaced with actual announcement times
        dates = []
        
        # Generate approximate FOMC dates (every 6 weeks, 8 per year)
        for year in range(2008, 2024):
            # Typical FOMC meeting months: Jan/Feb, Mar, Apr/May, Jun, Jul, Sep, Oct/Nov, Dec
            months = [1, 3, 5, 6, 7, 9, 11, 12]
            for month in months:
                # Typically announced at 2:00 PM ET (14:00)
                dt = datetime(year, month, 15, 14, 0, 0)
                dates.append(dt)
        
        return sorted(dates)
    
    def validate_announcement_timestamps(
        self,
        timestamps: List[datetime],
        min_spacing_hours: int = 24
    ) -> Dict[str, Any]:
        """
        Validate FOMC announcement timestamps.
        
        Checks for:
        - Sufficient spacing between announcements
        - Reasonable time of day (business hours)
        - Chronological ordering
        
        Args:
            timestamps: List of announcement timestamps to validate
            min_spacing_hours: Minimum hours between announcements
            
        Returns:
            Dictionary with validation results
        """
        validation_results = {
            'valid': True,
            'warnings': [],
            'errors': [],
            'n_announcements': len(timestamps)
        }
        
        if len(timestamps) == 0:
            validation_results['valid'] = False
            validation_results['errors'].append("No announcement timestamps provided")
            return validation_results
        
        # Check chronological ordering
        sorted_timestamps = sorted(timestamps)
        if timestamps != sorted_timestamps:
            validation_results['warnings'].append(
                "Timestamps not in chronological order - will be sorted"
            )
        
        # Check spacing between announcements
        for i in range(1, len(sorted_timestamps)):
            spacing = (sorted_timestamps[i] - sorted_timestamps[i-1]).total_seconds() / 3600
            if spacing < min_spacing_hours:
                validation_results['warnings'].append(
                    f"Announcements at {sorted_timestamps[i-1]} and {sorted_timestamps[i]} "
                    f"are only {spacing:.1f} hours apart"
                )
        
        # Check time of day (should be during business hours)
        for ts in sorted_timestamps:
            hour = ts.hour
            if hour < 8 or hour > 18:
                validation_results['warnings'].append(
                    f"Announcement at {ts} is outside typical business hours"
                )
        
        # Check for weekends
        for ts in sorted_timestamps:
            if ts.weekday() >= 5:  # Saturday=5, Sunday=6
                validation_results['warnings'].append(
                    f"Announcement at {ts} is on a weekend"
                )
        
        return validation_results

    def extract_hf_surprises(
        self,
        asset_prices: Dict[str, pd.Series],
        fomc_dates: Optional[List[datetime]] = None
    ) -> pd.DataFrame:
        """
        Extract high-frequency surprises around FOMC announcements.
        
        Computes asset price changes within narrow windows around each
        FOMC announcement. The window is centered on the announcement time.
        
        Args:
            asset_prices: Dictionary mapping asset names to intraday price series
                         Each series should have DatetimeIndex with minute frequency
            fomc_dates: Optional list of FOMC dates (uses self.fomc_dates if None)
            
        Returns:
            DataFrame with surprises by announcement and asset
            Rows: FOMC announcement dates
            Columns: Asset names
            Values: Price changes within window
        """
        if fomc_dates is None:
            fomc_dates = self.fomc_dates
        
        if len(fomc_dates) == 0:
            raise ValueError("No FOMC announcement dates provided")
        
        # Validate asset prices
        if not asset_prices:
            raise ValueError("No asset prices provided")
        
        # Initialize results DataFrame
        surprises = pd.DataFrame(
            index=pd.DatetimeIndex(fomc_dates),
            columns=list(asset_prices.keys())
        )
        
        # Half-window in minutes
        half_window = self.window_minutes // 2
        
        # Extract surprises for each announcement
        for fomc_date in fomc_dates:
            # Define window boundaries
            window_start = fomc_date - timedelta(minutes=half_window)
            window_end = fomc_date + timedelta(minutes=half_window)
            
            # Extract price changes for each asset
            for asset_name, price_series in asset_prices.items():
                try:
                    # Filter to window
                    window_prices = price_series[
                        (price_series.index >= window_start) &
                        (price_series.index <= window_end)
                    ]
                    
                    if len(window_prices) < 2:
                        # Insufficient data in window
                        surprises.loc[fomc_date, asset_name] = np.nan
                        warnings.warn(
                            f"Insufficient data for {asset_name} around {fomc_date}"
                        )
                        continue
                    
                    # Compute price change (last - first in window)
                    price_change = window_prices.iloc[-1] - window_prices.iloc[0]
                    surprises.loc[fomc_date, asset_name] = price_change
                    
                except Exception as e:
                    warnings.warn(
                        f"Error extracting surprise for {asset_name} at {fomc_date}: {e}"
                    )
                    surprises.loc[fomc_date, asset_name] = np.nan
        
        # Store results
        self.hf_surprises = surprises
        
        # Report missing data
        missing_count = surprises.isna().sum().sum()
        total_count = surprises.size
        if missing_count > 0:
            warnings.warn(
                f"Missing {missing_count}/{total_count} surprise observations "
                f"({100*missing_count/total_count:.1f}%)"
            )
        
        return surprises
    
    def _validate_window_boundaries(
        self,
        asset_prices: Dict[str, pd.Series],
        fomc_date: datetime
    ) -> Dict[str, Any]:
        """
        Validate that price data exists within window boundaries.
        
        Args:
            asset_prices: Dictionary of asset price series
            fomc_date: FOMC announcement timestamp
            
        Returns:
            Dictionary with validation results
        """
        half_window = self.window_minutes // 2
        window_start = fomc_date - timedelta(minutes=half_window)
        window_end = fomc_date + timedelta(minutes=half_window)
        
        validation = {
            'fomc_date': fomc_date,
            'window_start': window_start,
            'window_end': window_end,
            'assets_with_data': [],
            'assets_missing_data': []
        }
        
        for asset_name, price_series in asset_prices.items():
            window_prices = price_series[
                (price_series.index >= window_start) &
                (price_series.index <= window_end)
            ]
            
            if len(window_prices) >= 2:
                validation['assets_with_data'].append(asset_name)
            else:
                validation['assets_missing_data'].append(asset_name)
        
        return validation

    def decompose_policy_surprises(
        self,
        hf_surprises: Optional[pd.DataFrame] = None,
        n_components: int = 3
    ) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        Decompose policy surprises into target, forward guidance, and QE factors.
        
        Uses PCA following Swanson (2021) to separate:
        - Target rate factor (Fed funds futures)
        - Forward guidance factor (Eurodollar futures)
        - QE/LSAP factor (10Y Treasury yields)
        
        Args:
            hf_surprises: DataFrame of high-frequency surprises (uses self.hf_surprises if None)
            n_components: Number of principal components (default 3)
            
        Returns:
            Tuple of (target_factor, fg_factor, qe_factor) as Series
        """
        if hf_surprises is None:
            if self.hf_surprises is None:
                raise ValueError("No HF surprises available. Run extract_hf_surprises first.")
            hf_surprises = self.hf_surprises
        
        # Remove rows with any missing values
        surprises_clean = hf_surprises.dropna()
        
        if len(surprises_clean) < n_components:
            raise ValueError(
                f"Insufficient observations for PCA: {len(surprises_clean)} < {n_components}"
            )
        
        # Standardize surprises
        scaler = StandardScaler()
        surprises_scaled = scaler.fit_transform(surprises_clean)
        
        # Apply PCA
        pca = PCA(n_components=n_components)
        factors = pca.fit_transform(surprises_scaled)
        
        # Create factor series with original index
        factor_df = pd.DataFrame(
            factors,
            index=surprises_clean.index,
            columns=['PC1', 'PC2', 'PC3']
        )
        
        # Identify which PC corresponds to which factor based on loadings
        loadings = pd.DataFrame(
            pca.components_.T,
            columns=['PC1', 'PC2', 'PC3'],
            index=surprises_clean.columns
        )
        
        # Heuristic assignment based on typical patterns:
        # - Target factor: highest loading on short-term futures (FF)
        # - FG factor: highest loading on medium-term futures (ED)
        # - QE factor: highest loading on long-term yields (GS10)
        
        # Store loadings for inspection
        self.pca_loadings = loadings
        self.pca_explained_variance = pca.explained_variance_ratio_
        
        # For now, assign factors in order of explained variance
        # In practice, should verify loadings match expected pattern
        target_factor = factor_df['PC1']
        fg_factor = factor_df['PC2']
        qe_factor = factor_df['PC3']
        
        # Validate orthogonality
        correlation_matrix = factor_df.corr()
        max_correlation = correlation_matrix.abs().values[np.triu_indices_from(correlation_matrix.values, k=1)].max()
        
        if max_correlation > 0.01:
            warnings.warn(
                f"Factors not perfectly orthogonal: max correlation = {max_correlation:.4f}"
            )
        
        # Store results
        self.policy_factors = {
            'target': target_factor,
            'forward_guidance': fg_factor,
            'qe': qe_factor,
            'loadings': loadings,
            'explained_variance': pca.explained_variance_ratio_,
            'correlation_matrix': correlation_matrix
        }
        
        return target_factor, fg_factor, qe_factor
    
    def validate_factor_orthogonality(
        self,
        tolerance: float = 0.01
    ) -> Dict[str, Any]:
        """
        Validate that decomposed factors are orthogonal.
        
        Args:
            tolerance: Maximum acceptable correlation between factors
            
        Returns:
            Dictionary with validation results
        """
        if self.policy_factors is None:
            raise ValueError("No policy factors available. Run decompose_policy_surprises first.")
        
        correlation_matrix = self.policy_factors['correlation_matrix']
        
        # Get off-diagonal correlations
        n = len(correlation_matrix)
        off_diag_corr = []
        for i in range(n):
            for j in range(i+1, n):
                off_diag_corr.append(abs(correlation_matrix.iloc[i, j]))
        
        max_correlation = max(off_diag_corr)
        
        validation = {
            'orthogonal': max_correlation < tolerance,
            'max_correlation': max_correlation,
            'tolerance': tolerance,
            'correlation_matrix': correlation_matrix,
            'all_correlations': off_diag_corr
        }
        
        if not validation['orthogonal']:
            validation['warning'] = (
                f"Factors not orthogonal: max correlation {max_correlation:.4f} "
                f"exceeds tolerance {tolerance}"
            )
        
        return validation

    def construct_qe_instruments(
        self,
        qe_factor: Optional[pd.Series] = None,
        quarterly_dates: Optional[pd.DatetimeIndex] = None,
        aggregation_method: str = 'sum'
    ) -> pd.Series:
        """
        Aggregate QE shocks to quarterly frequency for use as instruments.
        
        Handles quarters with multiple FOMC announcements by aggregating
        the high-frequency QE factor within each quarter.
        
        Args:
            qe_factor: High-frequency QE component (uses self.policy_factors['qe'] if None)
            quarterly_dates: Target quarterly date index (if None, infers from data)
            aggregation_method: How to aggregate multiple announcements per quarter
                               Options: 'sum', 'mean', 'last'
            
        Returns:
            Quarterly QE shock series
        """
        if qe_factor is None:
            if self.policy_factors is None:
                raise ValueError("No policy factors available. Run decompose_policy_surprises first.")
            qe_factor = self.policy_factors['qe']
        
        # Convert to DataFrame for easier manipulation
        qe_df = pd.DataFrame({'qe_shock': qe_factor})
        
        # Add quarter column
        qe_df['quarter'] = qe_df.index.to_period('Q')
        
        # Aggregate by quarter
        if aggregation_method == 'sum':
            quarterly_shocks = qe_df.groupby('quarter')['qe_shock'].sum()
        elif aggregation_method == 'mean':
            quarterly_shocks = qe_df.groupby('quarter')['qe_shock'].mean()
        elif aggregation_method == 'last':
            quarterly_shocks = qe_df.groupby('quarter')['qe_shock'].last()
        else:
            raise ValueError(f"Unknown aggregation method: {aggregation_method}")
        
        # Convert period index to timestamp
        quarterly_shocks.index = quarterly_shocks.index.to_timestamp()
        
        # If target dates provided, reindex to match
        if quarterly_dates is not None:
            quarterly_shocks = quarterly_shocks.reindex(quarterly_dates, fill_value=0.0)
        
        # Store metadata about aggregation
        announcements_per_quarter = qe_df.groupby('quarter').size()
        self.quarterly_metadata = {
            'aggregation_method': aggregation_method,
            'announcements_per_quarter': announcements_per_quarter,
            'quarters_with_multiple_announcements': announcements_per_quarter[announcements_per_quarter > 1],
            'quarters_with_no_announcements': len(quarterly_dates) - len(announcements_per_quarter) if quarterly_dates is not None else 0
        }
        
        return quarterly_shocks
    
    def get_instrument_summary(self) -> Dict[str, Any]:
        """
        Get summary statistics for constructed instruments.
        
        Returns:
            Dictionary with instrument summary statistics
        """
        if self.policy_factors is None:
            raise ValueError("No policy factors available. Run decompose_policy_surprises first.")
        
        summary = {
            'n_announcements': len(self.policy_factors['qe']),
            'explained_variance': self.policy_factors['explained_variance'].tolist(),
            'factor_correlations': self.policy_factors['correlation_matrix'].to_dict()
        }
        
        # Add quarterly metadata if available
        if hasattr(self, 'quarterly_metadata'):
            summary['quarterly_aggregation'] = self.quarterly_metadata
        
        # Add factor statistics
        for factor_name in ['target', 'forward_guidance', 'qe']:
            factor = self.policy_factors[factor_name]
            summary[f'{factor_name}_stats'] = {
                'mean': float(factor.mean()),
                'std': float(factor.std()),
                'min': float(factor.min()),
                'max': float(factor.max()),
                'n_positive': int((factor > 0).sum()),
                'n_negative': int((factor < 0).sum())
            }
        
        return summary
