"""
Robust API Error Handling and Retry Mechanisms for FRED API

This module provides comprehensive error handling, retry mechanisms with exponential backoff,
fallback data sources, and detailed logging for FRED API interactions.

Author: QE Research Team
Date: 2025
Version: 1.0
"""

import time
import logging
import random
from typing import Dict, List, Optional, Any, Callable, Union
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
import pandas as pd
import numpy as np

try:
    from fredapi import Fred
    import requests
except ImportError as e:
    raise ImportError(f"Required dependencies not installed: {e}")

# Optional dependency for fallback data sources
try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False


class ErrorType(Enum):
    """Classification of API errors"""
    NETWORK_ERROR = "network_error"
    RATE_LIMIT_ERROR = "rate_limit_error"
    AUTHENTICATION_ERROR = "authentication_error"
    DATA_NOT_FOUND = "data_not_found"
    SERVER_ERROR = "server_error"
    TIMEOUT_ERROR = "timeout_error"
    UNKNOWN_ERROR = "unknown_error"


@dataclass
class RetryConfig:
    """Configuration for retry mechanisms"""
    max_retries: int = 5
    base_delay: float = 1.0
    max_delay: float = 60.0
    exponential_base: float = 2.0
    jitter: bool = True
    backoff_factor: float = 1.5
    timeout: float = 30.0


@dataclass
class APICallResult:
    """Result of an API call attempt"""
    success: bool
    data: Optional[pd.Series]
    error_type: Optional[ErrorType]
    error_message: Optional[str]
    attempt_number: int
    response_time: float
    timestamp: datetime


class FREDAPIErrorHandler:
    """
    Robust error handling and retry mechanisms for FRED API calls
    """
    
    def __init__(self, fred_api_key: str, retry_config: Optional[RetryConfig] = None):
        """
        Initialize the error handler
        
        Args:
            fred_api_key: FRED API key
            retry_config: Configuration for retry mechanisms
        """
        self.fred_api_key = fred_api_key
        self.retry_config = retry_config or RetryConfig()
        self.fred = Fred(api_key=fred_api_key)
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        
        # Track API call statistics
        self.call_stats = {
            'total_calls': 0,
            'successful_calls': 0,
            'failed_calls': 0,
            'retry_calls': 0,
            'error_types': {},
            'average_response_time': 0.0
        }
        
        # Fallback data sources
        self.fallback_sources = FallbackDataSources()
    
    def classify_error(self, exception: Exception) -> ErrorType:
        """
        Classify the type of error for appropriate handling
        
        Args:
            exception: The exception that occurred
            
        Returns:
            ErrorType classification
        """
        error_message = str(exception).lower()
        
        # Network-related errors
        if any(keyword in error_message for keyword in ['connection', 'network', 'dns', 'socket']):
            return ErrorType.NETWORK_ERROR
        
        # Rate limiting errors
        if any(keyword in error_message for keyword in ['rate limit', 'too many requests', '429']):
            return ErrorType.RATE_LIMIT_ERROR
        
        # Authentication errors
        if any(keyword in error_message for keyword in ['unauthorized', 'invalid api key', '401', '403']):
            return ErrorType.AUTHENTICATION_ERROR
        
        # Data not found errors
        if any(keyword in error_message for keyword in ['not found', '404', 'series does not exist']):
            return ErrorType.DATA_NOT_FOUND
        
        # Server errors
        if any(keyword in error_message for keyword in ['server error', '500', '502', '503', '504']):
            return ErrorType.SERVER_ERROR
        
        # Timeout errors
        if any(keyword in error_message for keyword in ['timeout', 'timed out']):
            return ErrorType.TIMEOUT_ERROR
        
        return ErrorType.UNKNOWN_ERROR
    
    def calculate_delay(self, attempt: int) -> float:
        """
        Calculate delay for exponential backoff with jitter
        
        Args:
            attempt: Current attempt number (0-based)
            
        Returns:
            Delay in seconds
        """
        # Exponential backoff
        delay = self.retry_config.base_delay * (self.retry_config.exponential_base ** attempt)
        
        # Apply backoff factor
        delay *= self.retry_config.backoff_factor
        
        # Cap at maximum delay
        delay = min(delay, self.retry_config.max_delay)
        
        # Add jitter to prevent thundering herd
        if self.retry_config.jitter:
            jitter_range = delay * 0.1  # 10% jitter
            delay += random.uniform(-jitter_range, jitter_range)
        
        return max(delay, 0.1)  # Minimum 0.1 second delay
    
    def should_retry(self, error_type: ErrorType, attempt: int) -> bool:
        """
        Determine if an error should trigger a retry
        
        Args:
            error_type: Type of error that occurred
            attempt: Current attempt number
            
        Returns:
            True if should retry, False otherwise
        """
        # Don't retry if max attempts reached
        if attempt >= self.retry_config.max_retries:
            return False
        
        # Retry for transient errors
        retryable_errors = {
            ErrorType.NETWORK_ERROR,
            ErrorType.RATE_LIMIT_ERROR,
            ErrorType.SERVER_ERROR,
            ErrorType.TIMEOUT_ERROR,
            ErrorType.UNKNOWN_ERROR
        }
        
        return error_type in retryable_errors
    
    def make_api_call_with_retry(self, series_id: str, start_date: str, end_date: str) -> APICallResult:
        """
        Make FRED API call with robust retry mechanism
        
        Args:
            series_id: FRED series ID
            start_date: Start date for data
            end_date: End date for data
            
        Returns:
            APICallResult with success status and data or error information
        """
        self.call_stats['total_calls'] += 1
        
        for attempt in range(self.retry_config.max_retries + 1):
            start_time = time.time()
            
            try:
                # Make the API call
                data = self.fred.get_series(
                    series_id, 
                    start=start_date, 
                    end=end_date,
                    timeout=self.retry_config.timeout
                )
                
                response_time = time.time() - start_time
                
                # Update statistics
                self.call_stats['successful_calls'] += 1
                if attempt > 0:
                    self.call_stats['retry_calls'] += 1
                
                # Update average response time
                total_successful = self.call_stats['successful_calls']
                current_avg = self.call_stats['average_response_time']
                self.call_stats['average_response_time'] = (
                    (current_avg * (total_successful - 1) + response_time) / total_successful
                )
                
                self.logger.info(f"Successfully retrieved {series_id} in {response_time:.2f}s (attempt {attempt + 1})")
                
                return APICallResult(
                    success=True,
                    data=data,
                    error_type=None,
                    error_message=None,
                    attempt_number=attempt + 1,
                    response_time=response_time,
                    timestamp=datetime.now()
                )
                
            except Exception as e:
                response_time = time.time() - start_time
                error_type = self.classify_error(e)
                error_message = str(e)
                
                # Update error statistics
                self.call_stats['failed_calls'] += 1
                if error_type.value not in self.call_stats['error_types']:
                    self.call_stats['error_types'][error_type.value] = 0
                self.call_stats['error_types'][error_type.value] += 1
                
                self.logger.warning(
                    f"API call failed for {series_id} (attempt {attempt + 1}): "
                    f"{error_type.value} - {error_message}"
                )
                
                # Check if we should retry
                if self.should_retry(error_type, attempt):
                    delay = self.calculate_delay(attempt)
                    self.logger.info(f"Retrying in {delay:.2f} seconds...")
                    time.sleep(delay)
                    continue
                else:
                    # No more retries, return failure
                    self.logger.error(f"All retry attempts exhausted for {series_id}")
                    
                    return APICallResult(
                        success=False,
                        data=None,
                        error_type=error_type,
                        error_message=error_message,
                        attempt_number=attempt + 1,
                        response_time=response_time,
                        timestamp=datetime.now()
                    )
        
        # This should never be reached, but just in case
        return APICallResult(
            success=False,
            data=None,
            error_type=ErrorType.UNKNOWN_ERROR,
            error_message="Maximum retries exceeded",
            attempt_number=self.retry_config.max_retries + 1,
            response_time=0.0,
            timestamp=datetime.now()
        )
    
    def get_series_with_fallback(self, series_id: str, start_date: str, end_date: str, 
                                fallback_ids: Optional[List[str]] = None) -> APICallResult:
        """
        Get series data with fallback to alternative series if primary fails
        
        Args:
            series_id: Primary FRED series ID
            start_date: Start date for data
            end_date: End date for data
            fallback_ids: List of alternative series IDs to try
            
        Returns:
            APICallResult with data from primary or fallback source
        """
        # Try primary series first
        result = self.make_api_call_with_retry(series_id, start_date, end_date)
        
        if result.success and result.data is not None and not result.data.empty:
            return result
        
        # Try fallback series if primary failed
        if fallback_ids:
            self.logger.info(f"Primary series {series_id} failed, trying fallback sources...")
            
            for fallback_id in fallback_ids:
                self.logger.info(f"Trying fallback series: {fallback_id}")
                fallback_result = self.make_api_call_with_retry(fallback_id, start_date, end_date)
                
                if fallback_result.success and fallback_result.data is not None and not fallback_result.data.empty:
                    self.logger.info(f"Successfully retrieved data from fallback series: {fallback_id}")
                    return fallback_result
        
        # Try external fallback sources
        external_result = self.fallback_sources.get_external_data(series_id, start_date, end_date)
        if external_result.success:
            self.logger.info(f"Successfully retrieved data from external fallback source")
            return external_result
        
        # All attempts failed
        self.logger.error(f"All data sources failed for series {series_id}")
        return result  # Return the original failure result
    
    def get_call_statistics(self) -> Dict[str, Any]:
        """
        Get comprehensive API call statistics
        
        Returns:
            Dictionary with call statistics and performance metrics
        """
        total_calls = self.call_stats['total_calls']
        success_rate = (self.call_stats['successful_calls'] / total_calls * 100) if total_calls > 0 else 0
        retry_rate = (self.call_stats['retry_calls'] / total_calls * 100) if total_calls > 0 else 0
        
        return {
            'total_calls': total_calls,
            'successful_calls': self.call_stats['successful_calls'],
            'failed_calls': self.call_stats['failed_calls'],
            'retry_calls': self.call_stats['retry_calls'],
            'success_rate_percent': round(success_rate, 2),
            'retry_rate_percent': round(retry_rate, 2),
            'average_response_time': round(self.call_stats['average_response_time'], 3),
            'error_breakdown': self.call_stats['error_types'].copy(),
            'last_updated': datetime.now().isoformat()
        }
    
    def reset_statistics(self):
        """Reset call statistics"""
        self.call_stats = {
            'total_calls': 0,
            'successful_calls': 0,
            'failed_calls': 0,
            'retry_calls': 0,
            'error_types': {},
            'average_response_time': 0.0
        }
        self.logger.info("API call statistics reset")


class FallbackDataSources:
    """
    Fallback data sources when FRED API fails
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Mapping of FRED series to alternative sources
        self.external_mappings = {
            # Stock market indices that can be retrieved from Yahoo Finance
            'VIXCLS': {'source': 'yahoo', 'symbol': '^VIX', 'name': 'VIX'},
            'SP500': {'source': 'yahoo', 'symbol': '^GSPC', 'name': 'S&P 500'},
            
            # Exchange rates
            'DEXUSEU': {'source': 'yahoo', 'symbol': 'EURUSD=X', 'name': 'EUR/USD'},
            'DEXJPUS': {'source': 'yahoo', 'symbol': 'USDJPY=X', 'name': 'USD/JPY'},
            
            # Treasury rates (limited availability)
            'DGS10': {'source': 'yahoo', 'symbol': '^TNX', 'name': '10-Year Treasury'},
        }
    
    def get_external_data(self, series_id: str, start_date: str, end_date: str) -> APICallResult:
        """
        Attempt to retrieve data from external sources
        
        Args:
            series_id: FRED series ID
            start_date: Start date
            end_date: End date
            
        Returns:
            APICallResult with external data if available
        """
        if series_id not in self.external_mappings:
            return APICallResult(
                success=False,
                data=None,
                error_type=ErrorType.DATA_NOT_FOUND,
                error_message=f"No external mapping available for {series_id}",
                attempt_number=1,
                response_time=0.0,
                timestamp=datetime.now()
            )
        
        mapping = self.external_mappings[series_id]
        
        if mapping['source'] == 'yahoo':
            return self._get_yahoo_data(mapping['symbol'], start_date, end_date)
        
        return APICallResult(
            success=False,
            data=None,
            error_type=ErrorType.DATA_NOT_FOUND,
            error_message=f"Unsupported external source: {mapping['source']}",
            attempt_number=1,
            response_time=0.0,
            timestamp=datetime.now()
        )
    
    def _get_yahoo_data(self, symbol: str, start_date: str, end_date: str) -> APICallResult:
        """
        Retrieve data from Yahoo Finance
        
        Args:
            symbol: Yahoo Finance symbol
            start_date: Start date
            end_date: End date
            
        Returns:
            APICallResult with Yahoo Finance data
        """
        if not YFINANCE_AVAILABLE:
            return APICallResult(
                success=False,
                data=None,
                error_type=ErrorType.DATA_NOT_FOUND,
                error_message="yfinance package not available",
                attempt_number=1,
                response_time=0.0,
                timestamp=datetime.now()
            )
        
        start_time = time.time()
        
        try:
            ticker = yf.Ticker(symbol)
            data = ticker.history(start=start_date, end=end_date)
            
            if data.empty:
                raise ValueError(f"No data returned for {symbol}")
            
            # Use closing price as the series value
            series = data['Close']
            series.name = symbol
            
            response_time = time.time() - start_time
            
            self.logger.info(f"Successfully retrieved {symbol} from Yahoo Finance")
            
            return APICallResult(
                success=True,
                data=series,
                error_type=None,
                error_message=None,
                attempt_number=1,
                response_time=response_time,
                timestamp=datetime.now()
            )
            
        except Exception as e:
            response_time = time.time() - start_time
            error_message = f"Yahoo Finance error for {symbol}: {str(e)}"
            
            self.logger.warning(error_message)
            
            return APICallResult(
                success=False,
                data=None,
                error_type=ErrorType.UNKNOWN_ERROR,
                error_message=error_message,
                attempt_number=1,
                response_time=response_time,
                timestamp=datetime.now()
            )


class ComprehensiveLogger:
    """
    Comprehensive logging system for API operations
    """
    
    def __init__(self, log_level: str = "INFO"):
        """
        Initialize comprehensive logging
        
        Args:
            log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        """
        self.logger = logging.getLogger('qeir.api_handler')
        self.logger.setLevel(getattr(logging, log_level.upper()))
        
        # Create formatters
        detailed_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
        )
        
        simple_formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s'
        )
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(simple_formatter)
        
        # File handler for detailed logs
        file_handler = logging.FileHandler('qeir_api_operations.log')
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(detailed_formatter)
        
        # Error file handler
        error_handler = logging.FileHandler('qeir_api_errors.log')
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(detailed_formatter)
        
        # Add handlers if not already added
        if not self.logger.handlers:
            self.logger.addHandler(console_handler)
            self.logger.addHandler(file_handler)
            self.logger.addHandler(error_handler)
    
    def log_api_call(self, series_id: str, result: APICallResult):
        """
        Log API call results with appropriate level
        
        Args:
            series_id: Series ID that was called
            result: Result of the API call
        """
        if result.success:
            self.logger.info(
                f"API SUCCESS: {series_id} - {len(result.data) if result.data is not None else 0} points - "
                f"{result.response_time:.2f}s - Attempt {result.attempt_number}"
            )
        else:
            self.logger.error(
                f"API FAILURE: {series_id} - {result.error_type.value if result.error_type else 'unknown'} - "
                f"{result.error_message} - {result.response_time:.2f}s - Attempt {result.attempt_number}"
            )
    
    def log_statistics_summary(self, stats: Dict[str, Any]):
        """
        Log API statistics summary
        
        Args:
            stats: Statistics dictionary from error handler
        """
        self.logger.info(
            f"API STATS: {stats['total_calls']} calls, "
            f"{stats['success_rate_percent']}% success rate, "
            f"{stats['retry_rate_percent']}% retry rate, "
            f"{stats['average_response_time']}s avg response time"
        )
        
        if stats['error_breakdown']:
            error_summary = ", ".join([f"{k}: {v}" for k, v in stats['error_breakdown'].items()])
            self.logger.info(f"ERROR BREAKDOWN: {error_summary}")