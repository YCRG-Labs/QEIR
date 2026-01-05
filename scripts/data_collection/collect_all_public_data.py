"""
Public Data Collection Script - FREE APIs Only

This script collects all data available from FREE public APIs.
No paid subscriptions required - only free API keys.

Required (Free) API Key:
- FRED_API_KEY: https://fred.stlouisfed.org/docs/api/api_key.html

What This Collects:
- 60+ economic series from FRED (Treasury yields, GDP, investment, etc.)
- Daily Treasury yields from Treasury.gov (no API key needed)
- FRBNY data (manual download instructions provided)

What This Does NOT Collect (requires paid subscriptions):
- High-frequency intraday data (Bloomberg/Refinitiv)
- Treasury bid-ask spreads (Bloomberg/TRACE)
- Some dealer data (requires manual FRBNY download)

Usage:
    # Set API key
    export FRED_API_KEY="your_key_here"
    
    # Run collection
    python collect_all_public_data.py
    
    # Or specify date range
    python collect_all_public_data.py --start 2008-01-01 --end 2023-12-31
"""

import sys
import os
import argparse
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any
import json

import pandas as pd
import numpy as np

# Try to load .env file if it exists
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # dotenv not installed, will use environment variables only

# Add project to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from qeir.data.public_data_sources import FRED_SERIES_IDS, PUBLIC_DATA_SOURCES
from qeir.data.collectors.fred_collector import FREDCollector
from qeir.data.collectors.treasury_collector import TreasuryCollector
from qeir.data.collectors.frbny_collector import FRBNYCollector

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('data_collection.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def check_api_keys():
    """Check which API keys are available - FRED only (free)"""
    available = {}
    missing = {}
    
    # Only check FRED (free API)
    fred_key = os.getenv('FRED_API_KEY')
    if fred_key:
        available['fred'] = 'FRED (Federal Reserve Economic Data)'
    else:
        missing['fred'] = {
            'name': 'FRED (Federal Reserve Economic Data)',
            'env_var': 'FRED_API_KEY',
            'signup_url': 'https://fred.stlouisfed.org/docs/api/api_key.html'
        }
    
    # No-API-key sources (always available)
    available['treasury'] = 'US Treasury (no API key needed)'
    available['frbny'] = 'FRBNY (manual download)'
    
    return available, missing


def print_api_key_status():
    """Print status of API keys"""
    available, missing = check_api_keys()
    
    print("\n" + "="*80)
    print("FREE PUBLIC DATA SOURCES")
    print("="*80)
    
    if available:
        print("\n✓ Available:")
        for source_id, name in available.items():
            print(f"  • {name}")
    
    if missing:
        print("\n✗ Required (FREE):")
        for source_id, info in missing.items():
            print(f"\n  • {info['name']}")
            print(f"    Set: {info['env_var']}")
            print(f"    Get FREE key: {info['signup_url']}")
    
    print("\n⚠️  NOT Collected (requires paid subscriptions):")
    print("  • High-frequency intraday data (Bloomberg/Refinitiv)")
    print("  • Treasury bid-ask spreads (Bloomberg/TRACE)")
    print("  • Real-time dealer positions (requires manual FRBNY download)")
    
    print("\n" + "="*80 + "\n")
    
    return len(available), len(missing)


def collect_fred_data(start_date: str, end_date: str) -> Dict[str, pd.Series]:
    """Collect all FRED data"""
    logger.info("="*80)
    logger.info("COLLECTING FRED DATA")
    logger.info("="*80)
    
    try:
        collector = FREDCollector()
        data = collector.fetch_all_series(FRED_SERIES_IDS, start_date, end_date)
        logger.info(f"✓ Collected {len(data)}/{len(FRED_SERIES_IDS)} FRED series")
        return data
    except ValueError as e:
        logger.error(f"✗ FRED collection failed: {e}")
        return {}


def collect_treasury_data(start_date: str, end_date: str) -> pd.DataFrame:
    """Collect Treasury data"""
    logger.info("\n" + "="*80)
    logger.info("COLLECTING TREASURY DATA")
    logger.info("="*80)
    
    try:
        collector = TreasuryCollector()
        start_year = int(start_date[:4])
        end_year = int(end_date[:4])
        data = collector.fetch_daily_yields(start_year, end_year)
        logger.info(f"✓ Collected {len(data)} days of Treasury yields")
        return data
    except Exception as e:
        logger.error(f"✗ Treasury collection failed: {e}")
        return pd.DataFrame()


def collect_frbny_data() -> Dict[str, pd.DataFrame]:
    """Collect FRBNY data"""
    logger.info("\n" + "="*80)
    logger.info("COLLECTING FRBNY DATA")
    logger.info("="*80)
    
    collector = FRBNYCollector()
    
    data = {
        'primary_dealers': collector.fetch_primary_dealer_data(),
        'soma_holdings': collector.fetch_soma_holdings()
    }
    
    logger.info("⚠ FRBNY data requires manual download from:")
    logger.info("  https://www.newyorkfed.org/markets/primarydealers")
    logger.info("  https://www.newyorkfed.org/markets/soma/sysopen_accholdings.html")
    
    return data


def save_collected_data(fred_data: Dict, treasury_data: pd.DataFrame,
                       frbny_data: Dict, output_dir: Path):
    """Save all collected data"""
    logger.info("\n" + "="*80)
    logger.info("SAVING COLLECTED DATA")
    logger.info("="*80)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save FRED data
    if fred_data:
        fred_df = pd.DataFrame(fred_data)
        fred_path = output_dir / 'fred_data.csv'
        fred_df.to_csv(fred_path)
        logger.info(f"✓ Saved FRED data: {fred_path}")
        
        # Also save as pickle for easy loading
        fred_pkl = output_dir / 'fred_data.pkl'
        fred_df.to_pickle(fred_pkl)
        logger.info(f"✓ Saved FRED pickle: {fred_pkl}")
    
    # Save Treasury data
    if not treasury_data.empty:
        treasury_path = output_dir / 'treasury_daily_yields.csv'
        treasury_data.to_csv(treasury_path, index=False)
        logger.info(f"✓ Saved Treasury data: {treasury_path}")
    
    # Save metadata
    metadata = {
        'collection_date': datetime.now().isoformat(),
        'fred_series_count': len(fred_data),
        'treasury_days': len(treasury_data),
        'sources': list(PUBLIC_DATA_SOURCES.keys())
    }
    
    metadata_path = output_dir / 'collection_metadata.json'
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"✓ Saved metadata: {metadata_path}")


def print_summary(fred_data: Dict, treasury_data: pd.DataFrame):
    """Print collection summary"""
    print("\n" + "="*80)
    print("DATA COLLECTION SUMMARY")
    print("="*80)
    
    print(f"\nFRED Series Collected: {len(fred_data)}/{len(FRED_SERIES_IDS)}")
    if fred_data:
        print("\nSample FRED series:")
        for i, (series_id, series) in enumerate(list(fred_data.items())[:5]):
            print(f"  • {series_id}: {len(series)} observations")
    
    print(f"\nTreasury Data: {len(treasury_data)} daily observations")
    
    print("\n" + "="*80)
    print("NEXT STEPS")
    print("="*80)
    print("\n1. Review collected data in: output/public_data/")
    print("2. Check data_collection.log for any errors")
    print("3. For missing data, see manual download instructions above")
    print("4. Run quarterly conversion: python convert_to_quarterly.py")
    print("5. Run analysis: python run_analysis_pipeline.py")
    print()


def main():
    parser = argparse.ArgumentParser(description='Collect all public QE data')
    parser.add_argument('--start', default='2008-01-01', help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end', default='2023-12-31', help='End date (YYYY-MM-DD)')
    parser.add_argument('--output', default='output/public_data', help='Output directory')
    args = parser.parse_args()
    
    print("\n" + "="*80)
    print("COMPREHENSIVE PUBLIC DATA COLLECTION")
    print("="*80)
    print(f"\nPeriod: {args.start} to {args.end}")
    print(f"Output: {args.output}")
    
    # Check API keys
    n_available, n_missing = print_api_key_status()
    
    if 'fred' not in [k for k, v in check_api_keys()[0].items() if k == 'fred']:
        print("ERROR: FRED API key required (FREE).")
        print("\nGet your FREE API key:")
        print("1. Visit: https://fred.stlouisfed.org/docs/api/api_key.html")
        print("2. Create account (2 minutes)")
        print("3. Copy API key")
        print("4. Set: $env:FRED_API_KEY=\"your_key_here\"")
        print("5. Run this script again")
        return 1
    
    # Collect data
    fred_data = collect_fred_data(args.start, args.end)
    treasury_data = collect_treasury_data(args.start, args.end)
    frbny_data = collect_frbny_data()
    
    # Save data
    output_dir = Path(args.output)
    save_collected_data(fred_data, treasury_data, frbny_data, output_dir)
    
    # Print summary
    print_summary(fred_data, treasury_data)
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
