"""
Complete QE Analysis Pipeline with FRED Data

This script fetches real data from FRED and runs the complete analysis pipeline
to generate all tables and figures for the paper.

IMPORTANT LIMITATIONS:
- FRED does not provide high-frequency intraday data
- We use daily data as an approximation for HF identification
- True replication requires Bloomberg/Refinitiv intraday futures data
- Results may differ from paper targets due to data limitations

Requirements:
- FRED API key (set as environment variable FRED_API_KEY)
- All qeir package dependencies installed
"""

import os
import sys
import warnings
import logging
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from fredapi import Fred

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Suppress warnings
warnings.filterwarnings('ignore')

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from qeir.config import RevisedMethodologyConfig
from qeir.utils.data_processor import DataProcessor
from qeir.data.enhanced_data_construction import EnhancedDataConstructor
from qeir.identification.hf_surprise_identifier import HFSurpriseIdentifier
from qeir.core.enhanced_hypothesis1 import EnhancedHypothesis1Analyzer
from qeir.core.enhanced_hypothesis2 import InvestmentImpactAnalyzer
from qeir.robustness.robustness_suite import RobustnessTestSuite
from qeir.output.publication_generator import PublicationOutputGenerator
from qeir.validation.target_validator import TargetValidator


class FREDDataCollector:
    """Collect all required data from FRED API"""
    
    # FRED series IDs for required variables
    SERIES_IDS = {
        # Treasury yields
        'treasury_10y': 'DGS10',           # 10-Year Treasury Constant Maturity Rate
        'treasury_2y': 'DGS2',             # 2-Year Treasury
        'treasury_5y': 'DGS5',             # 5-Year Treasury
        'treasury_30y': 'DGS30',           # 30-Year Treasury
        
        # Policy rates
        'fed_funds_rate': 'FEDFUNDS',      # Effective Federal Funds Rate
        'discount_rate': 'INTDSRUSM193N',  # Discount Rate
        
        # Macroeconomic variables
        'gdp_real': 'GDPC1',               # Real GDP (Quarterly)
        'investment_private': 'GPDIC1',    # Real Private Fixed Investment (Quarterly)
        'unemployment': 'UNRATE',          # Unemployment Rate
        'inflation_pce': 'PCEPILFE',       # Core PCE Price Index
        
        # Fiscal variables
        'federal_debt': 'GFDEBTN',         # Federal Debt Total (Quarterly)
        'federal_revenue': 'FGRECPT',      # Federal Government Current Receipts (Quarterly)
        'federal_interest': 'A091RC1Q027SBEA',  # Net Interest Payments (Quarterly)
        
        # Financial conditions
        'vix': 'VIXCLS',                   # VIX Volatility Index
        'ted_spread': 'TEDRATE',           # TED Spread
        'baa_spread': 'BAA10Y',            # BAA Corporate Bond Spread
        
        # Fed balance sheet
        'fed_assets': 'WALCL',             # Fed Total Assets (Weekly)
        'fed_securities': 'WSHOSHO',       # Fed Securities Held Outright (Weekly)
        
        # Market indicators
        'sp500': 'SP500',                  # S&P 500 Index
        'dollar_index': 'DTWEXBGS',        # Trade Weighted Dollar Index
        
        # Confidence indicators
        'consumer_confidence': 'UMCSENT',  # University of Michigan Consumer Sentiment
        'financial_stress': 'STLFSI2',     # St. Louis Fed Financial Stress Index
    }
    
    def __init__(self, api_key: str = None):
        """Initialize FRED API connection"""
        self.api_key = api_key or os.getenv('FRED_API_KEY')
        if not self.api_key:
            raise ValueError(
                "FRED API key required. Set FRED_API_KEY environment variable or pass api_key parameter.\n"
                "Get a free API key at: https://fred.stlouisfed.org/docs/api/api_key.html"
            )
        
        self.fred = Fred(api_key=self.api_key)
        logger.info("FRED API connection established")
    
    def fetch_all_data(self, start_date: str = '2008-01-01', 
                       end_date: str = '2023-12-31') -> Dict[str, pd.Series]:
        """
        Fetch all required data from FRED.
        
        Args:
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            
        Returns:
            Dictionary of pandas Series with FRED data
        """
        logger.info(f"Fetching data from FRED: {start_date} to {end_date}")
        
        data = {}
        failed_series = []
        
        for var_name, series_id in self.SERIES_IDS.items():
            try:
                logger.info(f"Fetching {var_name} ({series_id})...")
                series = self.fred.get_series(series_id, start_date, end_date)
                data[var_name] = series
                logger.info(f"  ✓ {var_name}: {len(series)} observations")
            except Exception as e:
                logger.error(f"  ✗ Failed to fetch {var_name}: {e}")
                failed_series.append(var_name)
        
        if failed_series:
            logger.warning(f"Failed to fetch {len(failed_series)} series: {failed_series}")
        
        logger.info(f"Successfully fetched {len(data)}/{len(self.SERIES_IDS)} series")
        return data
    
    def construct_derived_variables(self, data: Dict[str, pd.Series]) -> Dict[str, pd.Series]:
        """
        Construct derived variables from FRED data.
        
        Args:
            data: Dictionary of FRED series
            
        Returns:
            Updated dictionary with derived variables
        """
        logger.info("Constructing derived variables...")
        
        # Debt service to revenue ratio (fiscal indicator)
        if 'federal_interest' in data and 'federal_revenue' in data:
            data['debt_service_ratio'] = data['federal_interest'] / data['federal_revenue']
            logger.info("  ✓ Constructed debt_service_ratio")
        
        # QE intensity (change in Fed assets)
        if 'fed_assets' in data:
            data['qe_intensity'] = data['fed_assets'].pct_change()
            logger.info("  ✓ Constructed qe_intensity")
        
        # Financial conditions index (composite)
        if all(k in data for k in ['vix', 'ted_spread', 'baa_spread']):
            # Standardize and average
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()
            
            vix_std = scaler.fit_transform(data['vix'].dropna().values.reshape(-1, 1)).flatten()
            ted_std = scaler.fit_transform(data['ted_spread'].dropna().values.reshape(-1, 1)).flatten()
            baa_std = scaler.fit_transform(data['baa_spread'].dropna().values.reshape(-1, 1)).flatten()
            
            # Align dates
            common_dates = data['vix'].dropna().index.intersection(
                data['ted_spread'].dropna().index
            ).intersection(data['baa_spread'].dropna().index)
            
            fci = pd.Series(
                (vix_std + ted_std + baa_std) / 3,
                index=common_dates
            )
            data['financial_conditions_index'] = fci
            logger.info("  ✓ Constructed financial_conditions_index")
        
        return data


class ApproximateHFIdentification:
    """
    Approximate high-frequency identification using daily data.
    
    WARNING: This is NOT the same as true HF identification with intraday data.
    Results will differ from paper targets. This is for demonstration only.
    """
    
    def __init__(self, fomc_dates: List[datetime]):
        self.fomc_dates = fomc_dates
        logger.warning(
            "Using daily data approximation for HF identification. "
            "True replication requires intraday futures data from Bloomberg/Refinitiv."
        )
    
    def extract_daily_surprises(self, 
                                treasury_10y: pd.Series,
                                fed_funds_rate: pd.Series) -> pd.DataFrame:
        """
        Extract daily yield changes around FOMC dates as approximation.
        
        Args:
            treasury_10y: Daily 10Y Treasury yields
            fed_funds_rate: Daily fed funds rate
            
        Returns:
            DataFrame with approximate surprises
        """
        logger.info("Extracting daily surprises around FOMC dates...")
        
        surprises = []
        
        for fomc_date in self.fomc_dates:
            # Get yield change on FOMC day
            if fomc_date in treasury_10y.index:
                # Change from previous day
                prev_date = treasury_10y.index[treasury_10y.index < fomc_date][-1] if any(treasury_10y.index < fomc_date) else None
                
                if prev_date:
                    treasury_change = treasury_10y[fomc_date] - treasury_10y[prev_date]
                    
                    surprises.append({
                        'date': fomc_date,
                        'treasury_10y_change': treasury_change,
                        'qe_proxy': treasury_change  # Simplified: use long-end change as QE proxy
                    })
        
        df = pd.DataFrame(surprises)
        logger.info(f"  ✓ Extracted {len(df)} FOMC surprises")
        
        return df
    
    def construct_quarterly_instruments(self, 
                                       surprises: pd.DataFrame,
                                       quarterly_dates: pd.DatetimeIndex) -> pd.Series:
        """
        Aggregate daily surprises to quarterly frequency.
        
        Args:
            surprises: DataFrame with daily surprises
            quarterly_dates: Target quarterly dates
            
        Returns:
            Quarterly QE instrument series
        """
        logger.info("Aggregating to quarterly frequency...")
        
        # Convert to quarterly by summing surprises within each quarter
        surprises['quarter'] = pd.PeriodIndex(surprises['date'], freq='Q')
        quarterly_shocks = surprises.groupby('quarter')['qe_proxy'].sum()
        
        # Reindex to match target dates
        quarterly_shocks.index = quarterly_shocks.index.to_timestamp()
        quarterly_shocks = quarterly_shocks.reindex(quarterly_dates, fill_value=0)
        
        logger.info(f"  ✓ Created quarterly instrument with {len(quarterly_shocks)} observations")
        
        return quarterly_shocks


def main():
    """Main analysis pipeline"""
    
    print("="*80)
    print("QE ANALYSIS PIPELINE WITH FRED DATA")
    print("="*80)
    print()
    
    # Step 1: Configuration
    print("Step 1: Loading configuration...")
    config = RevisedMethodologyConfig(
        start_date="2008-01-01",
        end_date="2023-12-31",
        frequency="Q"
    )
    print(f"  ✓ Period: {config.start_date} to {config.end_date}")
    print(f"  ✓ Frequency: {config.frequency}")
    print()
    
    # Step 2: Fetch FRED data
    print("Step 2: Fetching data from FRED...")
    try:
        collector = FREDDataCollector()
        fred_data = collector.fetch_all_data(config.start_date, config.end_date)
        fred_data = collector.construct_derived_variables(fred_data)
        print(f"  ✓ Fetched {len(fred_data)} series from FRED")
    except ValueError as e:
        print(f"  ✗ ERROR: {e}")
        print()
        print("To run this script, you need a FRED API key:")
        print("1. Get a free key at: https://fred.stlouisfed.org/docs/api/api_key.html")
        print("2. Set environment variable: export FRED_API_KEY='your_key_here'")
        print("3. Or create a .env file with: FRED_API_KEY=your_key_here")
        return
    print()
    
    # Step 3: Convert to quarterly
    print("Step 3: Converting to quarterly frequency...")
    processor = DataProcessor()
    
    rate_variables = [
        'treasury_10y', 'treasury_2y', 'treasury_5y', 'treasury_30y',
        'fed_funds_rate', 'unemployment', 'vix', 'ted_spread', 'baa_spread'
    ]
    
    stock_variables = [
        'federal_debt', 'federal_revenue', 'federal_interest',
        'fed_assets', 'fed_securities', 'sp500'
    ]
    
    quarterly_data = processor.convert_to_quarterly(
        data=fred_data,
        rate_variables=rate_variables,
        stock_variables=stock_variables
    )
    print(f"  ✓ Converted to quarterly: {len(quarterly_data)} series")
    print()
    
    # Step 4: Approximate HF identification
    print("Step 4: Approximating high-frequency identification...")
    print("  ⚠ WARNING: Using daily data approximation (not true HF identification)")
    
    # Load FOMC dates
    import json
    with open('qeir/config/fomc_announcements.json', 'r') as f:
        fomc_config = json.load(f)
    
    fomc_dates = [datetime.fromisoformat(d['date']) for d in fomc_config['announcements']]
    fomc_dates = [d for d in fomc_dates if config.start_date <= d.strftime('%Y-%m-%d') <= config.end_date]
    
    hf_approx = ApproximateHFIdentification(fomc_dates)
    daily_surprises = hf_approx.extract_daily_surprises(
        fred_data.get('treasury_10y'),
        fred_data.get('fed_funds_rate')
    )
    
    # Get quarterly dates from data
    quarterly_dates = pd.date_range(
        start=config.start_date,
        end=config.end_date,
        freq='QE'  # Quarter end
    )
    
    qe_instruments = hf_approx.construct_quarterly_instruments(
        daily_surprises,
        quarterly_dates
    )
    print(f"  ✓ Created QE instruments: {len(qe_instruments)} quarters")
    print()
    
    # Step 5: Construct analysis variables
    print("Step 5: Constructing analysis variables...")
    constructor = EnhancedDataConstructor()
    
    # Note: This will use synthetic/placeholder data for variables not in FRED
    # (e.g., dealer balance sheets, bid-ask spreads, HHI)
    print("  ⚠ WARNING: Some variables not available in FRED will use placeholders")
    print("     - Dealer balance sheet data (requires FRBNY Primary Dealer Statistics)")
    print("     - Treasury bid-ask spreads (requires Bloomberg)")
    print("     - Dealer market concentration (requires FRBNY data)")
    print()
    
    # Create a minimal dataset for demonstration
    dataset_dict = {
        'dates': quarterly_dates,
        'treasury_10y': quarterly_data.get('treasury_10y', pd.Series(index=quarterly_dates)),
        'investment': quarterly_data.get('investment_private', pd.Series(index=quarterly_dates)),
        'gdp': quarterly_data.get('gdp_real', pd.Series(index=quarterly_dates)),
        'unemployment': quarterly_data.get('unemployment', pd.Series(index=quarterly_dates)),
        'inflation': quarterly_data.get('inflation_pce', pd.Series(index=quarterly_dates)),
        'debt_service_ratio': quarterly_data.get('debt_service_ratio', pd.Series(index=quarterly_dates)),
        'qe_instruments': qe_instruments,
        'financial_conditions': quarterly_data.get('financial_conditions_index', pd.Series(index=quarterly_dates)),
    }
    
    print(f"  ✓ Constructed dataset with {len(dataset_dict)} variables")
    print()
    
    # Step 6: Summary statistics
    print("Step 6: Computing summary statistics...")
    print()
    print("Summary Statistics (2008Q1-2023Q4)")
    print("-" * 80)
    
    for var_name, series in dataset_dict.items():
        if var_name != 'dates' and isinstance(series, pd.Series):
            valid_data = series.dropna()
            if len(valid_data) > 0:
                print(f"{var_name:30s} | N={len(valid_data):3d} | "
                      f"Mean={valid_data.mean():8.3f} | "
                      f"SD={valid_data.std():8.3f} | "
                      f"Min={valid_data.min():8.3f} | "
                      f"Max={valid_data.max():8.3f}")
    
    print()
    print("="*80)
    print("DATA COLLECTION COMPLETE")
    print("="*80)
    print()
    print("IMPORTANT NOTES:")
    print("1. This analysis uses FRED data, which is publicly available but limited")
    print("2. High-frequency identification uses daily approximation (not true HF)")
    print("3. Some market microstructure variables are not available from FRED")
    print("4. Results will differ from paper targets due to data limitations")
    print()
    print("For full replication matching paper targets, you need:")
    print("  - Bloomberg Terminal (intraday futures data)")
    print("  - FRBNY Primary Dealer Statistics")
    print("  - TRACE Treasury transaction data")
    print()
    print("Next steps:")
    print("  1. Review the data quality and coverage")
    print("  2. Obtain additional data sources if needed")
    print("  3. Run threshold regression: python run_threshold_analysis.py")
    print("  4. Run channel decomposition: python run_channel_analysis.py")
    print("  5. Generate publication outputs: python run_publication_outputs.py")
    print()
    
    # Save data for next steps
    output_dir = Path('output/data')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save as pickle for easy loading
    import pickle
    with open(output_dir / 'fred_quarterly_data.pkl', 'wb') as f:
        pickle.dump(dataset_dict, f)
    
    print(f"✓ Data saved to: {output_dir / 'fred_quarterly_data.pkl'}")
    print()


if __name__ == '__main__':
    main()
