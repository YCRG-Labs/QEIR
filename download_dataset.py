#!/usr/bin/env python3
"""
Comprehensive Data Collection Script for QE Research
Quantitative Easing and Bond Market Dynamics Analysis

REFINEMENTS BASED ON TESTING:
- Fixed ECB API endpoints and series codes
- Added fallback mechanisms for failed series
- Improved error handling and data validation
- Enhanced Yahoo Finance data collection with alternatives
- Added data quality checks and outlier cleaning
- Better frequency alignment and data processing
- Comprehensive logging and validation reports

This script downloads all necessary data for the research paper:
- Central bank balance sheet data
- Government bond yields
- Fiscal variables
- Investment data
- International portfolio flows
- Market microstructure indicators

Author: Research Team
Date: 2025
Version: 2.0 (Refined)
"""

import pandas as pd
import numpy as np
import requests
import time
import logging
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

# Load environment variables from .env
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    print("Missing required library: python-dotenv. Please install with: pip install python-dotenv")
    exit(1)

# Required libraries (install with: pip install fredapi yfinance oecd eurostat requests pandas numpy)
try:
    from fredapi import Fred
    import yfinance as yf
except ImportError as e:
    print(f"Missing required library: {e}")
    print("Please install with: pip install fredapi yfinance")
    exit(1)

# Configuration
class Config:
    # API Keys (set your own)
    FRED_API_KEY = os.getenv("FRED_API_KEY")
    
    # Date ranges
    START_DATE = "2000-01-01"
    END_DATE = "2024-12-31"
    
    # Countries
    COUNTRIES = {
        'US': 'United States',
        'EA': 'Euro Area', 
        'GB': 'United Kingdom',
        'JP': 'Japan',
        'CH': 'Switzerland',
        'SE': 'Sweden',
        'CA': 'Canada',
        'AU': 'Australia'
    }
    
    # Data directories
    DATA_DIR = "data"
    RAW_DIR = os.path.join(DATA_DIR, "raw")
    PROCESSED_DIR = os.path.join(DATA_DIR, "processed")

# Setup logging
def setup_logging():
    os.makedirs("logs", exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('logs/data_collection.log'),
            logging.StreamHandler()
        ]
    )

def create_directories():
    """Create necessary directories for data storage"""
    dirs = [
        Config.DATA_DIR,
        Config.RAW_DIR,
        Config.PROCESSED_DIR,
        os.path.join(Config.RAW_DIR, "central_banks"),
        os.path.join(Config.RAW_DIR, "bond_yields"),
        os.path.join(Config.RAW_DIR, "fiscal"),
        os.path.join(Config.RAW_DIR, "investment"),
        os.path.join(Config.RAW_DIR, "flows"),
        os.path.join(Config.RAW_DIR, "microstructure")
    ]
    
    for dir_path in dirs:
        os.makedirs(dir_path, exist_ok=True)
    
    logging.info("Created directory structure")

class FREDDataCollector:
    """Collector for Federal Reserve Economic Data"""
    
    def __init__(self, api_key: str):
        if api_key == "YOUR_FRED_API_KEY_HERE":
            logging.warning("Please set your FRED API key in Config.FRED_API_KEY")
            self.fred = None
        else:
            self.fred = Fred(api_key=api_key)
    
    def get_us_central_bank_data(self) -> Dict[str, pd.Series]:
        """Download US Federal Reserve balance sheet data"""
        if not self.fred:
            logging.error("FRED API key not set")
            return {}
        
        logging.info("Downloading US Federal Reserve balance sheet data...")
        
        series_map = {
            'fed_total_assets': 'WALCL',  # Total Assets
            'fed_treasury_holdings': 'WTREGEN',  # Treasury Securities
            'fed_mbs_holdings': 'WSHOMCB',  # Mortgage-backed Securities
            'fed_reserves': 'BOGMBASE',  # Monetary Base
            'fed_currency_circulation': 'BOGMBBCM',  # Currency in Circulation - try original
            'fed_bank_reserves': 'WRESBAL',  # Reserve Balances with Federal Reserve Banks
            'fed_repo_operations': 'WORAL',  # Repurchase Agreements
        }
        
        # Alternative series if primary ones fail
        alternative_series = {
            'fed_currency_circulation': ['BOGMBBCM', 'CURRSL', 'BOGMBBAM'],  # Multiple alternatives
            'fed_bank_reserves': ['WRESBAL', 'RESBALNS', 'TOTRESNS'],
            'fed_repo_operations': ['WORAL', 'RPONTSYD', 'RRPONTSYD']
        }
        
        data = {}
        for name, series_id in series_map.items():
            success = False
            
            # Try primary series first
            try:
                series = self.fred.get_series(series_id, start=Config.START_DATE, end=Config.END_DATE)
                if not series.empty:
                    data[name] = series
                    logging.info(f"Downloaded {name}: {len(series)} observations")
                    success = True
                time.sleep(0.1)  # Respect rate limits
            except Exception as e:
                logging.warning(f"Primary series {series_id} failed: {e}")
            
            # Try alternatives if primary failed
            if not success and name in alternative_series:
                for alt_series_id in alternative_series[name]:
                    try:
                        series = self.fred.get_series(alt_series_id, start=Config.START_DATE, end=Config.END_DATE)
                        if not series.empty:
                            data[f"{name}_alt"] = series
                            logging.info(f"Downloaded {name} (alternative {alt_series_id}): {len(series)} observations")
                            success = True
                            break
                        time.sleep(0.1)
                    except Exception as e:
                        logging.warning(f"Alternative series {alt_series_id} failed: {e}")
                        continue
            
            if not success:
                logging.error(f"All attempts failed for {name}")
        
        return data
    
    def get_us_bond_yields(self) -> Dict[str, pd.Series]:
        """Download US Treasury yield data"""
        if not self.fred:
            logging.error("FRED API key not set")
            return {}
        
        logging.info("Downloading US Treasury yields...")
        
        yield_series = {
            'us_2y': 'DGS2',    # 2-Year Treasury
            'us_5y': 'DGS5',    # 5-Year Treasury
            'us_10y': 'DGS10',  # 10-Year Treasury
            'us_30y': 'DGS30',  # 30-Year Treasury
            'us_3m': 'DGS3MO',  # 3-Month Treasury
        }
        
        data = {}
        for name, series_id in yield_series.items():
            try:
                series = self.fred.get_series(series_id, start=Config.START_DATE, end=Config.END_DATE)
                data[name] = series
                logging.info(f"Downloaded {name}: {len(series)} observations")
                time.sleep(0.1)
            except Exception as e:
                logging.error(f"Failed to download {name}: {e}")
        
        return data
    
    def get_us_fiscal_data(self) -> Dict[str, pd.Series]:
        """Download US fiscal variables"""
        if not self.fred:
            return {}
        
        logging.info("Downloading US fiscal data...")
        
        fiscal_series = {
            'us_federal_debt': 'GFDEBTN',     # Federal Debt Total
            'us_gdp': 'GDP',                  # GDP
            'us_interest_payments': 'A091RC1Q027SBEA',  # Federal Interest Payments
            'us_primary_balance': 'FYFSGDA188S',  # Federal Surplus/Deficit
        }
        
        data = {}
        for name, series_id in fiscal_series.items():
            try:
                series = self.fred.get_series(series_id, start=Config.START_DATE, end=Config.END_DATE)
                data[name] = series
                logging.info(f"Downloaded {name}: {len(series)} observations")
                time.sleep(0.1)
            except Exception as e:
                logging.error(f"Failed to download {name}: {e}")
        
        return data
    
    def get_us_investment_data(self) -> Dict[str, pd.Series]:
        """Download US investment data"""
        if not self.fred:
            return {}
        
        logging.info("Downloading US investment data...")
        
        investment_series = {
            'us_gross_investment': 'GPDI',          # Gross Private Domestic Investment
            'us_business_investment': 'PNFI',       # Private Nonresidential Fixed Investment
            'us_equipment_investment': 'PNFIC1',    # Equipment Investment
        }
        
        data = {}
        for name, series_id in investment_series.items():
            try:
                series = self.fred.get_series(series_id, start=Config.START_DATE, end=Config.END_DATE)
                data[name] = series
                logging.info(f"Downloaded {name}: {len(series)} observations")
                time.sleep(0.1)
            except Exception as e:
                logging.error(f"Failed to download {name}: {e}")
        
        return data
    
    def get_tic_data(self) -> Dict[str, pd.Series]:
        """Treasury International Capital data"""
        if not self.fred:
            return {}
        
        logging.info("Downloading Treasury International Capital data...")
        
        # TIC data from FRED
        tic_series = {
            'foreign_treasury_holdings': 'FDHBFIN',  # Foreign holdings of US Treasuries
            'foreign_agency_holdings': 'FDHBATN',   # Foreign holdings of Agency debt
            'china_treasury_holdings': 'FDHBFCH',   # China holdings
            'japan_treasury_holdings': 'FDHBFJP',   # Japan holdings
        }
        
        data = {}
        for name, series_id in tic_series.items():
            try:
                series = self.fred.get_series(series_id, start=Config.START_DATE, end=Config.END_DATE)
                if not series.empty:
                    data[name] = series
                    logging.info(f"Downloaded {name}: {len(series)} observations")
                time.sleep(0.1)
            except Exception as e:
                logging.warning(f"TIC series {series_id} failed: {e}")
        
        return data
    
    def get_boj_data(self) -> Dict[str, pd.Series]:
        """Bank of Japan data from FRED"""
        if not self.fred:
            return {}
        
        logging.info("Downloading Bank of Japan data...")
        
        boj_series = {
            'boj_total_assets': 'JPNASSETS',
            'japan_10y_yield': 'IRLTLT01JPM156N',  # Japan 10Y yield
            'japan_policy_rate': 'INTDSRJPM193N',  # Japan policy rate
        }
        
        data = {}
        for name, series_id in boj_series.items():
            try:
                series = self.fred.get_series(series_id, start=Config.START_DATE, end=Config.END_DATE)
                if not series.empty:
                    data[name] = series
                    logging.info(f"Downloaded {name}: {len(series)} observations")
                time.sleep(0.1)
            except Exception as e:
                logging.warning(f"BoJ series {series_id} failed: {e}")
        
        return data

class ECBDataCollector:
    """Collector for European Central Bank data"""
    
    def __init__(self):
        self.base_url = "https://data-api.ecb.europa.eu/service/data"
    
    def get_ecb_data(self, dataset: str, key: str, params: Dict = None) -> pd.DataFrame:
        """Generic ECB data fetcher with improved error handling"""
        url = f"{self.base_url}/{dataset}/{key}"
        
        default_params = {
            'format': 'csvdata',
            'startPeriod': '2000-01',
            'endPeriod': '2024-12'
        }
        
        if params:
            default_params.update(params)
        
        try:
            response = requests.get(url, params=default_params, timeout=30)
            response.raise_for_status()
            
            # Parse CSV response
            from io import StringIO
            df = pd.read_csv(StringIO(response.text), sep=',')
            
            # Process ECB data format
            if not df.empty and 'TIME_PERIOD' in df.columns:
                df['TIME_PERIOD'] = pd.to_datetime(df['TIME_PERIOD'])
                df = df.set_index('TIME_PERIOD')
                
                # Keep only the observation value column
                value_cols = [col for col in df.columns if 'OBS_VALUE' in col or col == 'OBS_VALUE']
                if value_cols:
                    df = df[value_cols].copy()
                    df.columns = ['value']
                    df = df.dropna()
            
            return df
            
        except Exception as e:
            logging.error(f"Failed to fetch ECB data {dataset}/{key}: {e}")
            return pd.DataFrame()
    
    def get_eurozone_yields(self) -> Dict[str, pd.DataFrame]:
        """Download Eurozone government bond yields using correct series codes"""
        logging.info("Downloading Eurozone bond yields...")
        
        data = {}
        
        # Try alternative yield series (government bond indices)
        yield_series = {
            'ea_2y': 'IRS.M.U2.L.L40.CI.0000.EUR.N.Z',
            'ea_5y': 'IRS.M.U2.L.L40.CI.0000.EUR.N.Z', 
            'ea_10y': 'IRS.M.U2.L.L40.CI.0000.EUR.N.Z',
        }
        
        # Try simpler approach with government bond data
        try:
            # Use government bond statistics
            response = requests.get(
                "https://data-api.ecb.europa.eu/service/data/IRS/M.U2.L.L40.CI.0000.EUR.N.Z",
                params={'format': 'csvdata', 'startPeriod': '2000-01', 'endPeriod': '2024-12'},
                timeout=30
            )
            
            if response.status_code == 200:
                from io import StringIO
                df = pd.read_csv(StringIO(response.text))
                if not df.empty:
                    data['ea_rates'] = df
                    logging.info(f"Downloaded EA interest rates: {len(df)} observations")
        except Exception as e:
            logging.warning(f"ECB yield data not available via API: {e}")
        
        # Fallback: Use FRED for Eurozone data
        try:
            logging.info("Trying FRED for Eurozone yield data...")
            if hasattr(self, 'fred_fallback'):
                # This would be set from the main script if FRED key is available
                euro_yields = {
                    'ea_10y_fred': 'IRLTLT01EZM156N'  # Euro area 10-year yield from FRED
                }
                
                for name, series_id in euro_yields.items():
                    try:
                        series = self.fred_fallback.get_series(series_id, start=Config.START_DATE, end=Config.END_DATE)
                        if not series.empty:
                            data[name] = pd.DataFrame({'value': series})
                            logging.info(f"Downloaded {name} from FRED: {len(series)} observations")
                    except Exception as e:
                        logging.warning(f"Failed to download {name} from FRED: {e}")
                        
        except Exception as e:
            logging.warning(f"FRED fallback failed: {e}")
        
        return data
    
    def get_ecb_balance_sheet(self) -> Dict[str, pd.DataFrame]:
        """Download ECB balance sheet data with fallbacks"""
        logging.info("Downloading ECB balance sheet data...")
        
        data = {}
        
        # Try with simpler balance sheet series
        try:
            # ECB total assets (consolidated balance sheet)
            response = requests.get(
                "https://data-api.ecb.europa.eu/service/data/BSI/M.U2.N.V.M10.X.1.U2.2300.Z01.E",
                params={'format': 'csvdata', 'startPeriod': '2000-01', 'endPeriod': '2024-12'},
                timeout=30
            )
            
            if response.status_code == 200:
                from io import StringIO
                df = pd.read_csv(StringIO(response.text))
                if not df.empty:
                    data['ecb_total_assets'] = df
                    logging.info(f"Downloaded ECB total assets: {len(df)} observations")
                    
        except Exception as e:
            logging.warning(f"ECB balance sheet data not available: {e}")
            
        # Alternative: Get from national central banks or FRED
        if not data and hasattr(self, 'fred_fallback'):
            try:
                # ECB assets from FRED
                ecb_assets = self.fred_fallback.get_series('ECBASSETS', start=Config.START_DATE, end=Config.END_DATE)
                if not ecb_assets.empty:
                    data['ecb_assets_fred'] = pd.DataFrame({'value': ecb_assets})
                    logging.info(f"Downloaded ECB assets from FRED: {len(ecb_assets)} observations")
            except Exception as e:
                logging.warning(f"ECB assets from FRED failed: {e}")
        
        return data

class YFinanceCollector:
    """Collector using Yahoo Finance for additional data"""
    
    def get_bond_etf_data(self) -> Dict[str, pd.DataFrame]:
        """Download bond ETF data as proxies for market conditions"""
        logging.info("Downloading bond ETF data...")
        
        etf_tickers = {
            'us_long_treasury': 'TLT',      # 20+ Year Treasury Bond ETF
            'us_inter_treasury': 'IEF',     # 7-10 Year Treasury Bond ETF
            'us_short_treasury': 'SHY',     # 1-3 Year Treasury Bond ETF
            'us_tips': 'SCHP',              # TIPS ETF
            'eu_government_bonds': 'VGEA.L', # European Government Bond ETF
            'eu_govt_bonds_alt': 'IEAG.L',  # Alternative European Government Bond ETF
            'jp_government_bonds': '1482.T', # Japanese Government Bond ETF
            'global_bonds': 'BNDW',         # Global Bond ETF
            'emerging_bonds': 'EMB',        # Emerging Market Bonds
        }
        
        # Alternative tickers if primary ones fail
        etf_alternatives = {
            'eu_government_bonds': ['VGEA.L', 'IEAG.L', 'AGGH', 'VGIT'],
            'jp_government_bonds': ['1482.T', 'FLJP', 'DXJ'],
        }
        
        data = {}
        for name, ticker in etf_tickers.items():
            success = False
            
            # Try primary ticker
            try:
                etf = yf.Ticker(ticker)
                hist = etf.history(start=Config.START_DATE, end=Config.END_DATE, interval="1d")
                if not hist.empty:
                    # Add basic technical indicators
                    hist['returns'] = hist['Close'].pct_change()
                    hist['volatility'] = hist['returns'].rolling(window=20).std() * np.sqrt(252)
                    
                    data[name] = hist
                    logging.info(f"Downloaded {name}: {len(hist)} observations")
                    success = True
                time.sleep(0.5)
            except Exception as e:
                logging.warning(f"Failed to download {ticker}: {e}")
            
            # Try alternatives if primary failed
            if not success and name in etf_alternatives:
                for alt_ticker in etf_alternatives[name]:
                    if alt_ticker == ticker:  # Skip if same as primary
                        continue
                    try:
                        etf = yf.Ticker(alt_ticker)
                        hist = etf.history(start=Config.START_DATE, end=Config.END_DATE, interval="1d")
                        if not hist.empty:
                            hist['returns'] = hist['Close'].pct_change()
                            hist['volatility'] = hist['returns'].rolling(window=20).std() * np.sqrt(252)
                            
                            data[f"{name}_alt"] = hist
                            logging.info(f"Downloaded {name} (alternative {alt_ticker}): {len(hist)} observations")
                            success = True
                            break
                        time.sleep(0.5)
                    except Exception as e:
                        logging.warning(f"Alternative ticker {alt_ticker} failed: {e}")
                        continue
            
            if not success:
                logging.error(f"All attempts failed for {name}")
        
        return data
    
    def get_exchange_rates(self) -> Dict[str, pd.DataFrame]:
        """Download exchange rate data with improved error handling"""
        logging.info("Downloading exchange rate data...")
        
        fx_pairs = {
            'eur_usd': 'EURUSD=X',
            'gbp_usd': 'GBPUSD=X',
            'jpy_usd': 'JPYUSD=X',
            'chf_usd': 'CHFUSD=X',
            'sek_usd': 'SEKUSD=X',
            'cad_usd': 'CADUSD=X',
            'aud_usd': 'AUDUSD=X',
            'dxy': 'DX-Y.NYB',  # Dollar Index
        }
        
        # Alternative sources for FX data
        fx_alternatives = {
            'eur_usd': ['EURUSD=X', 'EUR=X'],
            'gbp_usd': ['GBPUSD=X', 'GBP=X'],
            'jpy_usd': ['JPYUSD=X', 'JPY=X', 'USDJPY=X'],
        }
        
        data = {}
        for name, ticker in fx_pairs.items():
            success = False
            
            # Try primary ticker
            try:
                fx = yf.Ticker(ticker)
                hist = fx.history(start=Config.START_DATE, end=Config.END_DATE, interval="1d")
                if not hist.empty:
                    # Add volatility measure
                    hist['returns'] = hist['Close'].pct_change()
                    hist['volatility'] = hist['returns'].rolling(window=20).std() * np.sqrt(252)
                    
                    data[name] = hist
                    logging.info(f"Downloaded {name}: {len(hist)} observations")
                    success = True
                time.sleep(0.5)
            except Exception as e:
                logging.warning(f"Failed to download {ticker}: {e}")
            
            # Try alternatives if available
            if not success and name in fx_alternatives:
                for alt_ticker in fx_alternatives[name]:
                    if alt_ticker == ticker:
                        continue
                    try:
                        fx = yf.Ticker(alt_ticker)
                        hist = fx.history(start=Config.START_DATE, end=Config.END_DATE, interval="1d")
                        if not hist.empty:
                            hist['returns'] = hist['Close'].pct_change()
                            hist['volatility'] = hist['returns'].rolling(window=20).std() * np.sqrt(252)
                            
                            data[f"{name}_alt"] = hist
                            logging.info(f"Downloaded {name} (alternative {alt_ticker}): {len(hist)} observations")
                            success = True
                            break
                        time.sleep(0.5)
                    except Exception as e:
                        logging.warning(f"Alternative FX ticker {alt_ticker} failed: {e}")
                        continue
            
            if not success:
                logging.error(f"All FX attempts failed for {name}")
        
        return data
    
    def get_market_indicators(self) -> Dict[str, pd.DataFrame]:
        """Download additional market indicators"""
        logging.info("Downloading market indicators...")
        
        indicators = {
            'vix': '^VIX',          # Volatility Index
            'sp500': '^GSPC',       # S&P 500
            'nasdaq': '^IXIC',      # NASDAQ
            'gold': 'GC=F',         # Gold futures
            'oil': 'CL=F',          # Crude oil futures
            'copper': 'HG=F',       # Copper futures
        }
        
        data = {}
        for name, ticker in indicators.items():
            try:
                instrument = yf.Ticker(ticker)
                hist = instrument.history(start=Config.START_DATE, end=Config.END_DATE, interval="1d")
                if not hist.empty:
                    hist['returns'] = hist['Close'].pct_change()
                    hist['volatility'] = hist['returns'].rolling(window=20).std() * np.sqrt(252)
                    
                    data[name] = hist
                    logging.info(f"Downloaded {name}: {len(hist)} observations")
                time.sleep(0.5)
            except Exception as e:
                logging.warning(f"Failed to download {ticker}: {e}")
        
        return data
    
    def calculate_liquidity_proxies(self, etf_data):
        """Calculate liquidity proxies from ETF data"""
        
        liquidity_measures = {}
        
        for etf_name, df in etf_data.items():
            if 'High' in df.columns and 'Low' in df.columns and 'Volume' in df.columns:
                # Bid-ask spread proxy (High-Low)/Close
                spread_proxy = (df['High'] - df['Low']) / df['Close']
                
                # Amihud illiquidity measure
                returns = df['Close'].pct_change().abs()
                dollar_volume = df['Close'] * df['Volume']
                amihud = returns / (dollar_volume / 1e6)  # Scale by millions
                
                liquidity_measures[f'{etf_name}_spread_proxy'] = spread_proxy
                liquidity_measures[f'{etf_name}_amihud'] = amihud
        
        return liquidity_measures

class OECDCollector:
    """Collector for OECD data"""
    
    def __init__(self):
        self.base_url = "https://stats.oecd.org/SDMX-JSON/data"
    
    def get_oecd_fiscal_data(self) -> Dict[str, pd.DataFrame]:
        """Download OECD fiscal indicators"""
        logging.info("Downloading OECD fiscal data...")
        
        # Note: This is a simplified version. Full OECD API integration would require more setup
        # For now, we'll provide the framework and key identifiers
        
        fiscal_indicators = {
            'government_debt': 'EO/GGFL',           # Government gross financial liabilities
            'primary_balance': 'EO/NLGXQ',          # Primary balance
            'interest_payments': 'EO/GGFLQ',        # Government interest payments
        }
        
        countries = ['USA', 'DEU', 'GBR', 'JPN', 'CHE', 'SWE', 'CAN', 'AUS']
        
        data = {}
        # Placeholder for OECD data collection
        # In practice, you would implement the OECD SDMX API calls here
        logging.info("OECD data collection framework ready (implement API calls)")
        
        return data

class DataProcessor:
    """Process and clean collected data"""
    
    def __init__(self):
        self.processed_data = {}
    
    def calculate_qe_intensity(self, cb_holdings: pd.Series, total_outstanding: pd.Series) -> pd.Series:
        """Calculate QE intensity as CB holdings / total outstanding"""
        # Align series by date
        aligned_df = pd.DataFrame({'holdings': cb_holdings, 'outstanding': total_outstanding})
        aligned_df = aligned_df.interpolate(method='linear', limit=5)  # Fill small gaps
        
        # Calculate ratio
        intensity = aligned_df['holdings'] / aligned_df['outstanding']
        
        # Clean outliers (cap at reasonable levels)
        intensity = intensity.clip(lower=0, upper=1.0)
        
        return intensity
    
    def calculate_dcr(self, interest_payments: pd.Series, gdp: pd.Series) -> pd.Series:
        """Calculate debt service coverage ratio"""
        # Align series (both should be quarterly)
        aligned_df = pd.DataFrame({'interest': interest_payments, 'gdp': gdp})
        aligned_df = aligned_df.interpolate(method='linear', limit=2)
        
        # Calculate ratio
        dcr = aligned_df['interest'] / aligned_df['gdp']
        
        # Clean outliers
        dcr = dcr.clip(lower=0, upper=0.2)  # Cap at 20% of GDP
        
        return dcr
    
    def calculate_term_premium(self, long_yield: pd.Series, short_yield: pd.Series) -> pd.Series:
        """Calculate term premium (long - short yield)"""
        # Align series by date
        aligned_df = pd.DataFrame({'long': long_yield, 'short': short_yield})
        
        # Calculate spread
        spread = aligned_df['long'] - aligned_df['short']
        
        # Convert to basis points for easier interpretation
        spread_bp = spread * 100
        
        return spread_bp
    
    def align_frequencies(self, series_dict: Dict[str, pd.Series], target_freq: str = 'D') -> pd.DataFrame:
        """Align series of different frequencies to target frequency"""
        
        # Convert all series to target frequency
        aligned_series = {}
        
        for name, series in series_dict.items():
            if series.empty:
                continue
                
            try:
                # Ensure datetime index
                if not isinstance(series.index, pd.DatetimeIndex):
                    series.index = pd.to_datetime(series.index)
                
                # Remove timezone info to avoid conflicts
                if hasattr(series.index, 'tz') and series.index.tz is not None:
                    series.index = series.index.tz_localize(None)
                
                # Resample to target frequency
                if target_freq == 'D':
                    # For daily: forward fill for lower frequency data
                    resampled = series.resample('D').ffill()
                elif target_freq == 'W':
                    # For weekly: take last observation of week
                    resampled = series.resample('W').last()
                elif target_freq == 'M':
                    # For monthly: take last observation of month
                    resampled = series.resample('M').last()
                else:
                    resampled = series
                
                aligned_series[name] = resampled
                
            except Exception as e:
                logging.warning(f"Failed to align {name}: {e}")
                continue
        
        # Combine into DataFrame
        if aligned_series:
            df = pd.DataFrame(aligned_series)
            return df
        else:
            return pd.DataFrame()
    
    def clean_outliers(self, series: pd.Series, method: str = 'iqr', factor: float = 1.5) -> pd.Series:
        """Clean outliers from series"""
        
        if series.empty or series.dtype not in ['float64', 'int64']:
            return series
        
        if method == 'iqr':
            Q1 = series.quantile(0.25)
            Q3 = series.quantile(0.75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - factor * IQR
            upper_bound = Q3 + factor * IQR
            
            # Cap outliers rather than remove them
            cleaned = series.clip(lower=lower_bound, upper=upper_bound)
            
        elif method == 'zscore':
            z_scores = np.abs((series - series.mean()) / series.std())
            cleaned = series[z_scores < factor]
            
        else:
            cleaned = series
        
        outliers_removed = len(series) - len(cleaned)
        if outliers_removed > 0:
            logging.info(f"Capped {outliers_removed} outliers in series")
        
        return cleaned
    
    def standardize_timezones(self, df_dict):
        """Standardize all dataframes to remove timezone info"""
        standardized = {}
        
        for name, df in df_dict.items():
            if hasattr(df.index, 'tz') and df.index.tz is not None:
                df.index = df.index.tz_localize(None)
            standardized[name] = df
        
        return standardized
    
    def apply_quality_filters(self, df, min_obs_per_var=100, max_missing_pct=0.8):
        """Apply data quality filters"""
        
        # Remove variables with too few observations
        sufficient_data = df.count() >= min_obs_per_var
        df_filtered = df.loc[:, sufficient_data]
        
        # Remove time periods with excessive missing data
        row_missing_pct = df_filtered.isnull().sum(axis=1) / len(df_filtered.columns)
        clean_rows = row_missing_pct <= max_missing_pct
        df_final = df_filtered.loc[clean_rows]
        
        logging.info(f"Quality filters applied:")
        logging.info(f"  Variables: {len(df.columns)} → {len(df_final.columns)}")
        logging.info(f"  Observations: {len(df)} → {len(df_final)}")
        
        return df_final
    
    def process_all_data(self, raw_data: Dict) -> Dict[str, pd.DataFrame]:
        """Process all collected raw data with enhanced error handling"""
        logging.info("Processing collected data...")
        
        processed = {}
        
        # Process US data
        if 'us_data' in raw_data:
            us_data = raw_data['us_data']
            
            try:
                # Collect all US series
                us_series = {}
                
                # Add bond yields
                for key, series in us_data.get('yields', {}).items():
                    if isinstance(series, pd.Series) and not series.empty:
                        # Clean yield data (remove negative values, extreme outliers)
                        cleaned = self.clean_outliers(series.clip(lower=0, upper=20))
                        us_series[key] = cleaned
                
                # Add central bank data
                for key, series in us_data.get('central_bank', {}).items():
                    if isinstance(series, pd.Series) and not series.empty:
                        # Convert to billions for easier reading
                        if 'assets' in key.lower() or 'holdings' in key.lower():
                            series = series / 1000  # Convert millions to billions
                        us_series[key] = series
                
                # Add fiscal data
                for key, series in us_data.get('fiscal', {}).items():
                    if isinstance(series, pd.Series) and not series.empty:
                        # Convert debt to trillions, GDP to trillions
                        if 'debt' in key.lower():
                            series = series / 1_000_000  # Convert millions to trillions
                        elif 'gdp' in key.lower():
                            series = series / 1000  # Convert billions to trillions
                        us_series[key] = series
                
                # Add investment data
                for key, series in us_data.get('investment', {}).items():
                    if isinstance(series, pd.Series) and not series.empty:
                        us_series[key] = series
                
                # Add TIC data
                for key, series in us_data.get('tic', {}).items():
                    if isinstance(series, pd.Series) and not series.empty:
                        us_series[key] = series
                
                # Add BoJ data  
                for key, series in us_data.get('boj', {}).items():
                    if isinstance(series, pd.Series) and not series.empty:
                        us_series[key] = series
                
                # Align all series to daily frequency
                if us_series:
                    us_df = self.align_frequencies(us_series, target_freq='D')
                    
                    # Calculate derived variables
                    if 'fed_treasury_holdings' in us_df.columns and 'us_federal_debt' in us_df.columns:
                        us_df['us_qe_intensity'] = self.calculate_qe_intensity(
                            us_df['fed_treasury_holdings'], 
                            us_df['us_federal_debt']
                        )
                    
                    if 'us_interest_payments' in us_df.columns and 'us_gdp' in us_df.columns:
                        us_df['us_dcr'] = self.calculate_dcr(
                            us_df['us_interest_payments'], 
                            us_df['us_gdp']
                        )
                    
                    if 'us_10y' in us_df.columns and 'us_3m' in us_df.columns:
                        us_df['us_term_premium'] = self.calculate_term_premium(
                            us_df['us_10y'], 
                            us_df['us_3m']
                        )
                    
                    # Add moving averages for key variables
                    key_vars = ['us_10y', 'us_qe_intensity', 'fed_total_assets']
                    for var in key_vars:
                        if var in us_df.columns:
                            us_df[f'{var}_ma30'] = us_df[var].rolling(window=30, min_periods=15).mean()
                            us_df[f'{var}_ma90'] = us_df[var].rolling(window=90, min_periods=45).mean()
                    
                    # Remove rows with all NaN values
                    us_df = us_df.dropna(how='all')
                    
                    processed['us_panel'] = us_df
                    logging.info(f"US panel processed: {us_df.shape}")
                
            except Exception as e:
                logging.error(f"Error processing US data: {e}")
        
        # Process ECB data
        if 'ecb_data' in raw_data:
            try:
                ecb_series = {}
                
                for category, data_dict in raw_data['ecb_data'].items():
                    for name, df in data_dict.items():
                        if isinstance(df, pd.DataFrame) and not df.empty:
                            # Extract value column if it exists
                            if 'value' in df.columns:
                                series = df['value']
                                series.name = name
                                ecb_series[name] = series
                
                if ecb_series:
                    ecb_df = self.align_frequencies(ecb_series, target_freq='D')
                    processed['ecb_panel'] = ecb_df
                    logging.info(f"ECB panel processed: {ecb_df.shape}")
                
            except Exception as e:
                logging.error(f"Error processing ECB data: {e}")
        
        # Process market data
        if 'market_data' in raw_data:
            try:
                market_series = {}
                
                for category, data_dict in raw_data['market_data'].items():
                    for name, df in data_dict.items():
                        if isinstance(df, pd.DataFrame) and not df.empty:
                            # Remove timezone info
                            if hasattr(df.index, 'tz') and df.index.tz is not None:
                                df.index = df.index.tz_localize(None)
                            
                            # Use closing prices for ETFs and FX
                            if 'Close' in df.columns:
                                series = df['Close']
                                series.name = name
                                market_series[name] = series
                            
                            # Add volatility if available
                            if 'volatility' in df.columns:
                                vol_series = df['volatility']
                                vol_series.name = f"{name}_volatility"
                                market_series[f"{name}_volatility"] = vol_series
                
                if market_series:
                    market_df = self.align_frequencies(market_series, target_freq='D')
                    processed['market_panel'] = market_df
                    logging.info(f"Market panel processed: {market_df.shape}")
                
            except Exception as e:
                logging.error(f"Error processing market data: {e}")
        
        # Create combined panel if multiple datasets exist
        try:
            if len(processed) > 1:
                # Standardize timezones first
                processed = self.standardize_timezones(processed)
                
                # Combine all panels
                combined_data = []
                for name, df in processed.items():
                    if not df.empty:
                        combined_data.append(df)
                
                if combined_data:
                    combined_df = pd.concat(combined_data, axis=1, sort=True)
                    
                    # Remove duplicate columns
                    combined_df = combined_df.loc[:, ~combined_df.columns.duplicated()]
                    
                    processed['combined_panel'] = combined_df
                    logging.info(f"Combined panel created: {combined_df.shape}")
        
        except Exception as e:
            logging.error(f"Error creating combined panel: {e}")
        
        return processed

def main():
    """Main data collection orchestrator"""
    setup_logging()
    create_directories()
    
    logging.info("Starting comprehensive data collection for QE research...")
    
    # Initialize collectors
    fred_collector = FREDDataCollector(Config.FRED_API_KEY)
    ecb_collector = ECBDataCollector()
    yf_collector = YFinanceCollector()
    oecd_collector = OECDCollector()
    processor = DataProcessor()
    
    # Pass FRED collector to ECB for fallback
    if fred_collector.fred:
        ecb_collector.fred_fallback = fred_collector.fred
    
    # Collect all data
    all_data = {}
    
    # 1. US Federal Reserve Data
    logging.info("=== Collecting US Federal Reserve Data ===")
    us_data = {
        'central_bank': fred_collector.get_us_central_bank_data(),
        'yields': fred_collector.get_us_bond_yields(),
        'fiscal': fred_collector.get_us_fiscal_data(),
        'investment': fred_collector.get_us_investment_data(),
        'tic': fred_collector.get_tic_data(),
        'boj': fred_collector.get_boj_data(),
    }
    all_data['us_data'] = us_data
    
    # Save US data with validation
    for category, data_dict in us_data.items():
        category_dir = os.path.join(Config.RAW_DIR, f"us_{category}")
        os.makedirs(category_dir, exist_ok=True)
        
        for name, series in data_dict.items():
            if isinstance(series, pd.Series) and not series.empty:
                filepath = os.path.join(category_dir, f"{name}.csv")
                series.to_csv(filepath)
                
                # Basic validation
                if len(series) < 10:
                    logging.warning(f"Short series for {name}: {len(series)} observations")
                if series.isnull().sum() / len(series) > 0.5:
                    logging.warning(f"High missing data for {name}: {series.isnull().sum()/len(series)*100:.1f}%")
    
    # 2. European Central Bank Data
    logging.info("=== Collecting European Central Bank Data ===")
    ecb_data = {
        'yields': ecb_collector.get_eurozone_yields(),
        'balance_sheet': ecb_collector.get_ecb_balance_sheet(),
    }
    all_data['ecb_data'] = ecb_data
    
    # Save ECB data
    for category, data_dict in ecb_data.items():
        category_dir = os.path.join(Config.RAW_DIR, f"ecb_{category}")
        os.makedirs(category_dir, exist_ok=True)
        
        for name, df in data_dict.items():
            if isinstance(df, pd.DataFrame) and not df.empty:
                filepath = os.path.join(category_dir, f"{name}.csv")
                df.to_csv(filepath, index=True)
                logging.info(f"Saved {name}: {len(df)} observations")
    
    # 3. Market Data via Yahoo Finance
    logging.info("=== Collecting Market Data ===")
    market_data = {
        'bond_etfs': yf_collector.get_bond_etf_data(),
        'exchange_rates': yf_collector.get_exchange_rates(),
        'market_indicators': yf_collector.get_market_indicators(),
    }
    all_data['market_data'] = market_data
    
    # Add liquidity proxies
    if market_data['bond_etfs']:
        liquidity_proxies = yf_collector.calculate_liquidity_proxies(market_data['bond_etfs'])
        market_data['liquidity_proxies'] = liquidity_proxies
    
    # Save market data
    for category, data_dict in market_data.items():
        if category == 'liquidity_proxies':
            continue  # Skip liquidity proxies for now
            
        category_dir = os.path.join(Config.RAW_DIR, f"market_{category}")
        os.makedirs(category_dir, exist_ok=True)
        
        for name, df in data_dict.items():
            if isinstance(df, pd.DataFrame) and not df.empty:
                filepath = os.path.join(category_dir, f"{name}.csv")
                df.to_csv(filepath, index=True)
    
    # 4. Enhanced Data Processing
    logging.info("=== Processing Data ===")
    processed_data = processor.process_all_data(all_data)
    
    # Save processed data with validation
    for name, df in processed_data.items():
        if isinstance(df, pd.DataFrame) and not df.empty:
            filepath = os.path.join(Config.PROCESSED_DIR, f"{name}.csv")
            df.to_csv(filepath)
            
            # Data quality report
            missing_pct = df.isnull().sum().sum() / (len(df) * len(df.columns)) * 100
            date_range = f"{df.index.min()} to {df.index.max()}" if hasattr(df.index, 'min') else "No date index"
            
            logging.info(f"Saved processed dataset: {name}")
            logging.info(f"  Shape: {df.shape}")
            logging.info(f"  Date range: {date_range}")
            logging.info(f"  Missing data: {missing_pct:.1f}%")
            
            # Flag potential issues
            if missing_pct > 30:
                logging.warning(f"High missing data in {name}: {missing_pct:.1f}%")
            if len(df) < 100:
                logging.warning(f"Small dataset {name}: {len(df)} observations")
    
    # 5. Enhanced Summary Report
    logging.info("=== Data Collection Summary ===")
    total_series = 0
    total_observations = 0
    successful_downloads = []
    failed_downloads = []
    
    for category, data in all_data.items():
        if isinstance(data, dict):
            for subcategory, subdata in data.items():
                if isinstance(subdata, dict):
                    for name, series_or_df in subdata.items():
                        if isinstance(series_or_df, (pd.Series, pd.DataFrame)) and not series_or_df.empty:
                            total_series += 1
                            total_observations += len(series_or_df)
                            successful_downloads.append(f"{category}/{subcategory}/{name}")
                        else:
                            failed_downloads.append(f"{category}/{subcategory}/{name}")
    
    logging.info(f"Successful downloads: {len(successful_downloads)}")
    logging.info(f"Failed downloads: {len(failed_downloads)}")
    logging.info(f"Total series collected: {total_series}")
    logging.info(f"Total observations: {total_observations:,}")
    logging.info(f"Data saved in: {Config.DATA_DIR}")
    
    # Log failures for debugging
    if failed_downloads:
        logging.warning("Failed downloads:")
        for failure in failed_downloads[:10]:  # Show first 10
            logging.warning(f"  - {failure}")
        if len(failed_downloads) > 10:
            logging.warning(f"  ... and {len(failed_downloads) - 10} more")
    
    # 6. Create enhanced data dictionary
    create_enhanced_data_dictionary(all_data, processed_data)
    
    # 7. Data validation report
    create_data_validation_report(processed_data)
    
    logging.info("Data collection completed successfully!")
    
    return all_data, processed_data

def create_enhanced_data_dictionary(raw_data, processed_data):
    """Create enhanced data dictionary with metadata"""
    
    data_dict = {
        'Variable': [],
        'Description': [],
        'Source': [],
        'Frequency': [],
        'Units': [],
        'Start_Date': [],
        'End_Date': [],
        'Observations': [],
        'Missing_Pct': []
    }
    
    # Add US variables with detailed metadata
    us_vars = [
        ('fed_total_assets', 'Federal Reserve Total Assets', 'FRED/WALCL', 'Weekly', 'Billions USD'),
        ('fed_treasury_holdings', 'Fed Treasury Securities Holdings', 'FRED/WTREGEN', 'Weekly', 'Billions USD'),
        ('fed_mbs_holdings', 'Fed Mortgage-Backed Securities Holdings', 'FRED/WSHOMCB', 'Weekly', 'Billions USD'),
        ('us_10y', 'US 10-Year Treasury Constant Maturity Rate', 'FRED/DGS10', 'Daily', 'Percent'),
        ('us_5y', 'US 5-Year Treasury Constant Maturity Rate', 'FRED/DGS5', 'Daily', 'Percent'),
        ('us_2y', 'US 2-Year Treasury Constant Maturity Rate', 'FRED/DGS2', 'Daily', 'Percent'),
        ('us_federal_debt', 'Federal Debt: Total Public Debt', 'FRED/GFDEBTN', 'Monthly', 'Trillions USD'),
        ('us_gdp', 'Gross Domestic Product', 'FRED/GDP', 'Quarterly', 'Trillions USD'),
        ('us_qe_intensity', 'QE Intensity (CB Holdings/Total Debt)', 'Calculated', 'Daily', 'Ratio'),
        ('us_dcr', 'Debt Service Coverage Ratio', 'Calculated', 'Daily', 'Ratio'),
        ('us_term_premium', 'Term Premium (10Y-3M)', 'Calculated', 'Daily', 'Basis Points'),
        ('foreign_treasury_holdings', 'Foreign Holdings of US Treasuries', 'FRED/TIC', 'Monthly', 'Billions USD'),
        ('japan_10y_yield', 'Japan 10-Year Government Bond Yield', 'FRED', 'Monthly', 'Percent'),
    ]
    
    # Check if variables exist in processed data
    main_dataset = None
    for name, df in processed_data.items():
        if 'us' in name.lower() and 'panel' in name.lower():
            main_dataset = df
            break
    
    for var, desc, source, freq, units in us_vars:
        data_dict['Variable'].append(var)
        data_dict['Description'].append(desc)
        data_dict['Source'].append(source)
        data_dict['Frequency'].append(freq)
        data_dict['Units'].append(units)
        
        # Add metadata if variable exists in processed data
        if main_dataset is not None and var in main_dataset.columns:
            series = main_dataset[var].dropna()
            if not series.empty:
                data_dict['Start_Date'].append(series.index.min())
                data_dict['End_Date'].append(series.index.max())
                data_dict['Observations'].append(len(series))
                missing_pct = main_dataset[var].isnull().sum() / len(main_dataset) * 100
                data_dict['Missing_Pct'].append(f"{missing_pct:.1f}%")
            else:
                data_dict['Start_Date'].append('N/A')
                data_dict['End_Date'].append('N/A')
                data_dict['Observations'].append(0)
                data_dict['Missing_Pct'].append('100.0%')
        else:
            data_dict['Start_Date'].append('N/A')
            data_dict['End_Date'].append('N/A')
            data_dict['Observations'].append(0)
            data_dict['Missing_Pct'].append('N/A')
    
    # Save enhanced data dictionary
    dict_df = pd.DataFrame(data_dict)
    dict_path = os.path.join(Config.PROCESSED_DIR, 'enhanced_data_dictionary.csv')
    dict_df.to_csv(dict_path, index=False)
    
    logging.info(f"Enhanced data dictionary saved to: {dict_path}")

def create_data_validation_report(processed_data):
    """Create comprehensive data validation report"""
    
    report = {
        'Dataset': [],
        'Shape': [],
        'Date_Range': [],
        'Missing_Data_Pct': [],
        'Key_Variables_Available': [],
        'Data_Quality_Score': []
    }
    
    key_variables = ['us_10y', 'us_qe_intensity', 'us_dcr', 'fed_total_assets', 'fed_treasury_holdings']
    
    for name, df in processed_data.items():
        if isinstance(df, pd.DataFrame) and not df.empty:
            report['Dataset'].append(name)
            report['Shape'].append(f"{df.shape[0]} x {df.shape[1]}")
            
            # Date range
            if hasattr(df.index, 'min') and hasattr(df.index, 'max'):
                date_range = f"{df.index.min().strftime('%Y-%m-%d')} to {df.index.max().strftime('%Y-%m-%d')}"
            else:
                date_range = "No date index"
            report['Date_Range'].append(date_range)
            
            # Missing data percentage
            missing_pct = df.isnull().sum().sum() / (len(df) * len(df.columns)) * 100
            report['Missing_Data_Pct'].append(f"{missing_pct:.1f}%")
            
            # Key variables available
            available_key_vars = [var for var in key_variables if var in df.columns]
            report['Key_Variables_Available'].append(f"{len(available_key_vars)}/{len(key_variables)}")
            
            # Data quality score (0-100)
            quality_score = 100
            if missing_pct > 10:
                quality_score -= 20
            if missing_pct > 30:
                quality_score -= 30
            if len(df) < 100:
                quality_score -= 20
            if len(available_key_vars) < len(key_variables) * 0.8:
                quality_score -= 15
            
            report['Data_Quality_Score'].append(f"{max(0, quality_score)}/100")
    
    # Save validation report
    report_df = pd.DataFrame(report)
    report_path = os.path.join(Config.PROCESSED_DIR, 'data_validation_report.csv')
    report_df.to_csv(report_path, index=False)
    
    logging.info(f"Data validation report saved to: {report_path}")
    
    return report_df

def create_data_dictionary():
    """Create a basic data dictionary documenting all variables"""
    
    data_dict = {
        'Variable': [],
        'Description': [],
        'Source': [],
        'Frequency': [],
        'Units': []
    }
    
    # US Variables
    us_vars = [
        ('fed_total_assets', 'Federal Reserve Total Assets', 'FRED', 'Weekly', 'Billions USD'),
        ('fed_treasury_holdings', 'Fed Treasury Securities Holdings', 'FRED', 'Weekly', 'Billions USD'),
        ('us_10y', 'US 10-Year Treasury Yield', 'FRED', 'Daily', 'Percent'),
        ('us_qe_intensity', 'QE Intensity (CB Holdings/Total Debt)', 'Calculated', 'Daily', 'Ratio'),
        ('us_dcr', 'Debt Service Coverage Ratio', 'Calculated', 'Daily', 'Ratio'),
        ('us_term_premium', 'Term Premium (10Y-3M)', 'Calculated', 'Daily', 'Basis Points'),
    ]
    
    for var, desc, source, freq, units in us_vars:
        data_dict['Variable'].append(var)
        data_dict['Description'].append(desc)
        data_dict['Source'].append(source)
        data_dict['Frequency'].append(freq)
        data_dict['Units'].append(units)
    
    # Save data dictionary
    dict_df = pd.DataFrame(data_dict)
    dict_path = os.path.join(Config.PROCESSED_DIR, 'data_dictionary.csv')
    dict_df.to_csv(dict_path, index=False)
    
    logging.info(f"Data dictionary saved to: {dict_path}")

if __name__ == "__main__":
    try:
        all_data, processed_data = main()
        
        print("\n" + "="*60)
        print("DATA COLLECTION COMPLETED SUCCESSFULLY!")
        print("="*60)
        
        # Print quick summary
        if processed_data:
            for name, df in processed_data.items():
                if isinstance(df, pd.DataFrame) and not df.empty:
                    print(f"\n📊 {name.upper()}:")
                    print(f"   Shape: {df.shape}")
                    print(f"   Columns: {', '.join(df.columns[:5])}{'...' if len(df.columns) > 5 else ''}")
                    if hasattr(df.index, 'min'):
                        print(f"   Date range: {df.index.min().strftime('%Y-%m-%d')} to {df.index.max().strftime('%Y-%m-%d')}")
        
        print(f"\n📁 All data saved in: {Config.DATA_DIR}")
        print(f"📋 Check logs/data_collection.log for detailed information")
        print(f"🚀 Ready to run analysis notebook!")
        
    except KeyboardInterrupt:
        print("\n\n❌ Data collection interrupted by user")
        logging.info("Data collection interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Data collection failed: {e}")
        logging.error(f"Data collection failed: {e}")
        raise