"""FRED Data Collector - Comprehensive"""

import os
import logging
from typing import Dict, List, Optional
import pandas as pd
from fredapi import Fred

logger = logging.getLogger(__name__)


class FREDCollector:
    """Collect all required data from FRED API"""
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv('FRED_API_KEY')
        if not self.api_key:
            raise ValueError(
                "FRED API key required. Set FRED_API_KEY environment variable.\n"
                "Get free key: https://fred.stlouisfed.org/docs/api/api_key.html"
            )
        self.fred = Fred(api_key=self.api_key)
        logger.info("FRED API initialized")
    
    def fetch_series(self, series_id: str, start_date: str, end_date: str) -> pd.Series:
        """Fetch single series from FRED"""
        try:
            return self.fred.get_series(series_id, start_date, end_date)
        except Exception as e:
            logger.error(f"Failed to fetch {series_id}: {e}")
            return pd.Series()
    
    def fetch_all_series(self, series_ids: Dict[str, str], 
                        start_date: str, end_date: str) -> Dict[str, pd.Series]:
        """Fetch multiple series from FRED"""
        data = {}
        for series_id, description in series_ids.items():
            logger.info(f"Fetching {series_id}: {description}")
            series = self.fetch_series(series_id, start_date, end_date)
            if not series.empty:
                data[series_id] = series
        return data
