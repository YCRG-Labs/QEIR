"""
Data alignment utilities for QE hypothesis testing

This module provides robust data alignment functions to handle different
frequencies and date ranges in economic time series data.

Author: Kiro AI Assistant
Date: 2025-09-02
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional, Tuple, Any

logger = logging.getLogger(__name__)

def align_series_to_common_frequency(data_dict: Dict[str, pd.Series], 
                                   target_frequency: str = 'Q') -> Dict[str, pd.Series]:
    """
    Align multiple time series to a common frequency
    
    Args:
        data_dict: Dictionary of pandas Series with different frequencies
        target_frequency: Target frequency ('D', 'M', 'Q', 'A')
        
    Returns:
        Dictionary of aligned pandas Series
    """
    if not data_dict:
        return {}
    
    # Filter out None values
    valid_data = {k: v for k, v in data_dict.items() if v is not None and not v.empty}
    
    if not valid_data:
        return {}
    
    logger.info(f"Aligning {len(valid_data)} series to {target_frequency} frequency")
    
    aligned_data = {}
    
    for series_name, series in valid_data.items():
        try:
            # Determine current frequency
            freq = pd.infer_freq(series.index)
            
            if freq is None:
                # Try to infer from index differences
                if len(series) > 1:
                    diff = series.index[1] - series.index[0]
                    if diff.days <= 1:
                        current_freq = 'D'
                    elif diff.days <= 31:
                        current_freq = 'M'
                    elif diff.days <= 92:
                        current_freq = 'Q'
                    else:
                        current_freq = 'A'
                else:
                    current_freq = target_frequency
            else:
                current_freq = freq[0] if freq else target_frequency
            
            # Resample to target frequency
            if target_frequency == 'Q':
                if current_freq in ['D', 'B']:
                    # Daily to quarterly - take last value of quarter
                    resampled = series.resample('Q').last()
                elif current_freq == 'M':
                    # Monthly to quarterly - take last value of quarter
                    resampled = series.resample('Q').last()
                elif current_freq == 'A':
                    # Annual to quarterly - forward fill
                    resampled = series.resample('Q').ffill()
                else:
                    # Already quarterly or unknown
                    resampled = series
            elif target_frequency == 'M':
                if current_freq in ['D', 'B']:
                    # Daily to monthly - take last value of month
                    resampled = series.resample('M').last()
                elif current_freq == 'Q':
                    # Quarterly to monthly - forward fill
                    resampled = series.resample('M').ffill()
                elif current_freq == 'A':
                    # Annual to monthly - forward fill
                    resampled = series.resample('M').ffill()
                else:
                    # Already monthly or unknown
                    resampled = series
            else:
                # Default - keep original
                resampled = series
            
            # Remove NaN values
            resampled = resampled.dropna()
            
            if len(resampled) > 0:
                aligned_data[series_name] = resampled
                logger.info(f"Aligned {series_name}: {len(series)} -> {len(resampled)} observations")
            else:
                logger.warning(f"No data after alignment for {series_name}")
                
        except Exception as e:
            logger.error(f"Failed to align {series_name}: {e}")
            # Keep original if alignment fails
            aligned_data[series_name] = series
    
    return aligned_data

def find_common_date_range(data_dict: Dict[str, pd.Series], 
                          min_overlap_pct: float = 0.5) -> Tuple[pd.Timestamp, pd.Timestamp]:
    """
    Find the optimal common date range for multiple series
    
    Args:
        data_dict: Dictionary of pandas Series
        min_overlap_pct: Minimum percentage of series that must have data in the range
        
    Returns:
        Tuple of (start_date, end_date)
    """
    if not data_dict:
        raise ValueError("No data provided")
    
    valid_data = {k: v for k, v in data_dict.items() if v is not None and not v.empty}
    
    if not valid_data:
        raise ValueError("No valid data series")
    
    # Get all start and end dates
    start_dates = []
    end_dates = []
    
    for series in valid_data.values():
        start_dates.append(series.index.min())
        end_dates.append(series.index.max())
    
    # Find the range that maximizes overlap
    overall_start = min(start_dates)
    overall_end = max(end_dates)
    
    # Try different start dates to find optimal overlap
    best_start = overall_start
    best_end = overall_end
    best_count = 0
    
    for start_candidate in start_dates:
        for end_candidate in end_dates:
            if start_candidate >= end_candidate:
                continue
            
            # Count how many series have data in this range
            count = 0
            total_obs = 0
            
            for series in valid_data.values():
                series_in_range = series[(series.index >= start_candidate) & 
                                       (series.index <= end_candidate)]
                if len(series_in_range) > 0:
                    count += 1
                    total_obs += len(series_in_range)
            
            # Check if this range meets minimum overlap requirement
            overlap_pct = count / len(valid_data)
            
            if overlap_pct >= min_overlap_pct and total_obs > best_count:
                best_start = start_candidate
                best_end = end_candidate
                best_count = total_obs
    
    logger.info(f"Optimal date range: {best_start} to {best_end}")
    return best_start, best_end

def align_to_common_dates(data_dict: Dict[str, pd.Series], 
                         start_date: Optional[pd.Timestamp] = None,
                         end_date: Optional[pd.Timestamp] = None) -> Dict[str, pd.Series]:
    """
    Align all series to a common date range
    
    Args:
        data_dict: Dictionary of pandas Series
        start_date: Start date for alignment (if None, will be determined automatically)
        end_date: End date for alignment (if None, will be determined automatically)
        
    Returns:
        Dictionary of aligned pandas Series
    """
    if not data_dict:
        return {}
    
    valid_data = {k: v for k, v in data_dict.items() if v is not None and not v.empty}
    
    if not valid_data:
        return {}
    
    # Determine date range if not provided
    if start_date is None or end_date is None:
        auto_start, auto_end = find_common_date_range(valid_data)
        start_date = start_date or auto_start
        end_date = end_date or auto_end
    
    logger.info(f"Aligning to date range: {start_date} to {end_date}")
    
    aligned_data = {}
    
    for series_name, series in valid_data.items():
        try:
            # Filter to common date range
            mask = (series.index >= start_date) & (series.index <= end_date)
            filtered_series = series[mask]
            
            if len(filtered_series) > 0:
                aligned_data[series_name] = filtered_series
                logger.info(f"Aligned {series_name}: {len(filtered_series)} observations in range")
            else:
                logger.warning(f"No data in range for {series_name}")
                
        except Exception as e:
            logger.error(f"Failed to align {series_name} to date range: {e}")
    
    return aligned_data

def robust_data_alignment(data_dict: Dict[str, pd.Series], 
                         target_frequency: str = 'Q',
                         min_observations: int = 10) -> Dict[str, pd.Series]:
    """
    Perform robust data alignment with frequency conversion and date alignment
    
    Args:
        data_dict: Dictionary of pandas Series
        target_frequency: Target frequency for alignment
        min_observations: Minimum number of observations required
        
    Returns:
        Dictionary of robustly aligned pandas Series
    """
    logger.info("Starting robust data alignment...")
    
    # Step 1: Filter out None/empty series
    valid_data = {k: v for k, v in data_dict.items() if v is not None and not v.empty}
    
    if not valid_data:
        logger.warning("No valid data for alignment")
        return {}
    
    logger.info(f"Starting with {len(valid_data)} valid series")
    
    # Step 2: Find common date range first
    start_dates = [s.index.min() for s in valid_data.values()]
    end_dates = [s.index.max() for s in valid_data.values()]
    
    common_start = max(start_dates)
    common_end = min(end_dates)
    
    logger.info(f"Common date range: {common_start} to {common_end}")
    
    # Step 3: Filter each series to common range and resample
    resampled_data = {}
    
    for name, series in valid_data.items():
        try:
            # Filter to common range
            mask = (series.index >= common_start) & (series.index <= common_end)
            filtered = series[mask]
            
            if len(filtered) == 0:
                logger.warning(f"No data in common range for {name}")
                continue
            
            # Resample to target frequency
            if target_frequency == 'Q':
                resampled = filtered.resample('Q').last().dropna()
            elif target_frequency == 'M':
                resampled = filtered.resample('M').last().dropna()
            elif target_frequency == 'A':
                resampled = filtered.resample('A').last().dropna()
            else:
                resampled = filtered.dropna()
            
            if len(resampled) > 0:
                resampled_data[name] = resampled
                logger.info(f"Resampled {name}: {len(filtered)} -> {len(resampled)} observations")
            else:
                logger.warning(f"No data after resampling for {name}")
                
        except Exception as e:
            logger.error(f"Failed to process {name}: {e}")
    
    if not resampled_data:
        logger.warning("No data after resampling")
        return {}
    
    # Step 4: Find final common dates
    common_dates = None
    for series in resampled_data.values():
        if common_dates is None:
            common_dates = series.index
        else:
            common_dates = common_dates.intersection(series.index)
    
    if len(common_dates) < min_observations:
        logger.warning(f"Insufficient common observations: {len(common_dates)} < {min_observations}")
        # Try with monthly frequency if quarterly failed
        if target_frequency == 'Q':
            logger.info("Retrying with monthly frequency...")
            return robust_data_alignment(data_dict, target_frequency='M', min_observations=min_observations)
        else:
            return {}
    
    # Step 5: Extract final aligned series
    final_aligned = {}
    for name, series in resampled_data.items():
        try:
            aligned_series = series.loc[common_dates]
            if len(aligned_series) >= min_observations:
                final_aligned[name] = aligned_series
                logger.info(f"Final {name}: {len(aligned_series)} observations")
        except Exception as e:
            logger.error(f"Failed to align {name}: {e}")
    
    logger.info(f"Final alignment: {len(final_aligned)} series with {len(common_dates)} common observations")
    
    return final_aligned