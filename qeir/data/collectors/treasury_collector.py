"""US Treasury Data Collector - No API key required"""

import logging
import pandas as pd
import requests
from io import StringIO

logger = logging.getLogger(__name__)


class TreasuryCollector:
    """Collect data from US Treasury (no API key required)"""
    
    DAILY_YIELDS_URL = "https://home.treasury.gov/resource-center/data-chart-center/interest-rates/daily-treasury-rates.csv/all/{year}?type=daily_treasury_yield_curve&field_tdr_date_value={year}&page&_format=csv"
    
    def fetch_daily_yields(self, start_year: int, end_year: int) -> pd.DataFrame:
        """Fetch daily Treasury yield curve data"""
        all_data = []
        
        for year in range(start_year, end_year + 1):
            try:
                url = self.DAILY_YIELDS_URL.format(year=year)
                logger.info(f"Fetching Treasury yields for {year}")
                response = requests.get(url, timeout=30)
                
                if response.status_code == 200:
                    df = pd.read_csv(StringIO(response.text))
                    all_data.append(df)
                else:
                    logger.warning(f"Failed to fetch {year}: {response.status_code}")
            except Exception as e:
                logger.error(f"Error fetching {year}: {e}")
        
        if all_data:
            return pd.concat(all_data, ignore_index=True)
        return pd.DataFrame()
