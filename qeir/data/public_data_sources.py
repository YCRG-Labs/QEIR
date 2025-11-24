"""
Public Data Sources Configuration

This module defines all free public data sources available for the QE analysis.
Each source requires only a free API key or is completely open.

Free API Keys Required:
1. FRED (Federal Reserve Economic Data) - https://fred.stlouisfed.org/
2. Alpha Vantage (Financial data) - https://www.alphavantage.co/
3. NASDAQ Data Link (formerly Quandl) - https://data.nasdaq.com/
4. World Bank - https://data.worldbank.org/
5. BEA (Bureau of Economic Analysis) - https://apps.bea.gov/api/signup/

No API Key Required:
1. FRBNY (Federal Reserve Bank of New York) - Direct downloads
2. Treasury.gov - Direct downloads
3. CBO (Congressional Budget Office) - Direct downloads
"""

from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass
class DataSource:
    """Configuration for a public data source"""
    name: str
    requires_api_key: bool
    api_key_env_var: Optional[str]
    signup_url: Optional[str]
    description: str
    available_series: List[str]


# Define all public data sources
PUBLIC_DATA_SOURCES = {
    'fred': DataSource(
        name='FRED (Federal Reserve Economic Data)',
        requires_api_key=True,
        api_key_env_var='FRED_API_KEY',
        signup_url='https://fred.stlouisfed.org/docs/api/api_key.html',
        description='Comprehensive economic data from Federal Reserve',
        available_series=[
            'Treasury yields (all maturities)',
            'Federal funds rate',
            'GDP and components',
            'Unemployment rate',
            'Inflation (CPI, PCE)',
            'Federal debt and revenue',
            'Fed balance sheet',
            'Financial conditions indices',
            'Consumer confidence',
            'VIX and market indicators'
        ]
    ),
    
    'alphavantage': DataSource(
        name='Alpha Vantage',
        requires_api_key=True,
        api_key_env_var='ALPHAVANTAGE_API_KEY',
        signup_url='https://www.alphavantage.co/support/#api-key',
        description='Financial market data including daily Treasury yields',
        available_series=[
            'Daily Treasury yields',
            'Federal funds rate (daily)',
            'Stock market indices',
            'Currency exchange rates'
        ]
    ),
    
    'nasdaq': DataSource(
        name='NASDAQ Data Link (Quandl)',
        requires_api_key=True,
        api_key_env_var='NASDAQ_DATA_LINK_API_KEY',
        signup_url='https://data.nasdaq.com/sign-up',
        description='Financial and economic datasets',
        available_series=[
            'Treasury yields (historical)',
            'Fed funds futures',
            'Economic indicators',
            'Financial ratios'
        ]
    ),
    
    'bea': DataSource(
        name='BEA (Bureau of Economic Analysis)',
        requires_api_key=True,
        api_key_env_var='BEA_API_KEY',
        signup_url='https://apps.bea.gov/api/signup/',
        description='Official US economic statistics',
        available_series=[
            'GDP by component',
            'Private fixed investment',
            'Personal income and outlays',
            'International transactions'
        ]
    ),
    
    'frbny': DataSource(
        name='FRBNY (Federal Reserve Bank of New York)',
        requires_api_key=False,
        api_key_env_var=None,
        signup_url=None,
        description='NY Fed data including primary dealer statistics',
        available_series=[
            'Primary dealer positions',
            'Primary dealer financing',
            'SOMA holdings',
            'Reference rates (SOFR, etc.)'
        ]
    ),
    
    'treasury': DataSource(
        name='US Treasury',
        requires_api_key=False,
        api_key_env_var=None,
        signup_url=None,
        description='Treasury data including fiscal accounts',
        available_series=[
            'Daily Treasury yields',
            'Monthly Treasury Statement',
            'Debt to the penny',
            'Interest expense'
        ]
    ),
    
    'cbo': DataSource(
        name='CBO (Congressional Budget Office)',
        requires_api_key=False,
        api_key_env_var=None,
        signup_url=None,
        description='Budget and economic projections',
        available_series=[
            'Budget projections',
            'Economic projections',
            'Debt projections',
            'Historical budget data'
        ]
    )
}


# FRED Series IDs for all required variables
FRED_SERIES_IDS = {
    # Treasury Yields
    'DGS1MO': '1-Month Treasury',
    'DGS3MO': '3-Month Treasury',
    'DGS6MO': '6-Month Treasury',
    'DGS1': '1-Year Treasury',
    'DGS2': '2-Year Treasury',
    'DGS3': '3-Year Treasury',
    'DGS5': '5-Year Treasury',
    'DGS7': '7-Year Treasury',
    'DGS10': '10-Year Treasury',
    'DGS20': '20-Year Treasury',
    'DGS30': '30-Year Treasury',
    
    # Policy Rates
    'FEDFUNDS': 'Effective Federal Funds Rate',
    'DFEDTARU': 'Federal Funds Target Rate - Upper Limit',
    'DFEDTARL': 'Federal Funds Target Rate - Lower Limit',
    'INTDSRUSM193N': 'Discount Rate',
    
    # GDP and Components
    'GDPC1': 'Real GDP',
    'GPDIC1': 'Real Private Fixed Investment',
    'PCECC96': 'Real Personal Consumption Expenditures',
    'GCEC1': 'Real Government Consumption',
    'NETEXP': 'Net Exports',
    
    # Labor Market
    'UNRATE': 'Unemployment Rate',
    'PAYEMS': 'Nonfarm Payrolls',
    'CIVPART': 'Labor Force Participation Rate',
    'U6RATE': 'U-6 Unemployment Rate',
    
    # Inflation
    'CPIAUCSL': 'CPI All Items',
    'CPILFESL': 'CPI Less Food and Energy',
    'PCEPI': 'PCE Price Index',
    'PCEPILFE': 'Core PCE Price Index',
    
    # Fiscal Variables
    'GFDEBTN': 'Federal Debt Total',
    'FGRECPT': 'Federal Government Current Receipts',
    'FGEXPND': 'Federal Government Current Expenditures',
    'A091RC1Q027SBEA': 'Net Interest Payments',
    
    # Fed Balance Sheet
    'WALCL': 'Fed Total Assets',
    'WSHOSHO': 'Fed Securities Held Outright',
    'WTREGEN': 'Fed Treasury Securities',
    'WSHOMCB': 'Fed Mortgage-Backed Securities',
    'WORAL': 'Fed Other Assets',
    
    # Financial Conditions
    'VIXCLS': 'VIX Volatility Index',
    'TEDRATE': 'TED Spread',
    'T10Y2Y': '10Y-2Y Treasury Spread',
    'T10Y3M': '10Y-3M Treasury Spread',
    'BAA10Y': 'BAA Corporate Spread',
    'AAA10Y': 'AAA Corporate Spread',
    
    # Market Indicators
    'SP500': 'S&P 500',
    'DTWEXBGS': 'Trade Weighted Dollar Index',
    'DEXUSEU': 'USD/EUR Exchange Rate',
    
    # Confidence and Sentiment
    'UMCSENT': 'University of Michigan Consumer Sentiment',
    'CSCICP03USM665S': 'Consumer Confidence Index',
    
    # Financial Stress
    'STLFSI2': 'St. Louis Fed Financial Stress Index',
    'NFCI': 'Chicago Fed National Financial Conditions Index',
    
    # Money Supply
    'M1SL': 'M1 Money Stock',
    'M2SL': 'M2 Money Stock',
    'BOGMBASE': 'Monetary Base',
    
    # Credit Markets
    'TOTRESNS': 'Total Reserves',
    'EXCSRESNS': 'Excess Reserves',
    'DRTSCILM': 'Total Consumer Credit',
}
