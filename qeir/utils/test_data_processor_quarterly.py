"""
Test quarterly frequency conversion for DataProcessor

This test verifies the implementation of quarterly aggregation for rate and stock variables.
"""

import pandas as pd
import numpy as np
from datetime import datetime
from qeir.utils.data_processor import DataProcessor, ProcessingConfig


def test_quarterly_rate_aggregation():
    """Test that rate variables are aggregated using mean"""
    # Create monthly data for 2008-2009 (2 years = 24 months)
    dates = pd.date_range('2008-01-31', periods=24, freq='ME')
    
    # Create a rate variable (e.g., interest rate)
    # Q1 2008: Jan=2.0, Feb=2.5, Mar=3.0 -> mean should be 2.5
    # Q2 2008: Apr=3.5, May=4.0, Jun=4.5 -> mean should be 4.0
    monthly_values = [2.0, 2.5, 3.0,  # Q1 2008
                     3.5, 4.0, 4.5,  # Q2 2008
                     5.0, 5.5, 6.0,  # Q3 2008
                     6.5, 7.0, 7.5,  # Q4 2008
                     8.0, 8.5, 9.0,  # Q1 2009
                     9.5, 10.0, 10.5,  # Q2 2009
                     11.0, 11.5, 12.0,  # Q3 2009
                     12.5, 13.0, 13.5]  # Q4 2009
    
    monthly_series = pd.Series(monthly_values, index=dates, name='interest_rate')
    
    # Create DataProcessor with quarterly frequency
    config = ProcessingConfig(target_frequency='QE')
    processor = DataProcessor(config)
    
    # Convert to quarterly
    data = {'interest_rate': monthly_series}
    quarterly_data = processor.align_frequencies(
        data, 
        target_freq='QE',
        rate_variables=['interest_rate']
    )
    
    # Verify quarterly data
    quarterly_series = quarterly_data['interest_rate']
    
    # Check that we have 8 quarters
    assert len(quarterly_series) == 8, f"Expected 8 quarters, got {len(quarterly_series)}"
    
    # Check Q1 2008 mean: (2.0 + 2.5 + 3.0) / 3 = 2.5
    q1_2008 = quarterly_series.loc['2008-03-31']
    expected_q1 = 2.5
    assert np.isclose(q1_2008, expected_q1), f"Q1 2008: expected {expected_q1}, got {q1_2008}"
    
    # Check Q2 2008 mean: (3.5 + 4.0 + 4.5) / 3 = 4.0
    q2_2008 = quarterly_series.loc['2008-06-30']
    expected_q2 = 4.0
    assert np.isclose(q2_2008, expected_q2), f"Q2 2008: expected {expected_q2}, got {q2_2008}"
    
    print("✓ Rate variable aggregation test passed")


def test_quarterly_stock_aggregation():
    """Test that stock variables use end-of-quarter values"""
    # Create monthly data
    dates = pd.date_range('2008-01-31', periods=24, freq='ME')
    
    # Create a stock variable (e.g., debt level)
    # Q1 2008: Jan=100, Feb=105, Mar=110 -> should use 110 (last value)
    # Q2 2008: Apr=115, May=120, Jun=125 -> should use 125 (last value)
    monthly_values = [100, 105, 110,  # Q1 2008
                     115, 120, 125,  # Q2 2008
                     130, 135, 140,  # Q3 2008
                     145, 150, 155,  # Q4 2008
                     160, 165, 170,  # Q1 2009
                     175, 180, 185,  # Q2 2009
                     190, 195, 200,  # Q3 2009
                     205, 210, 215]  # Q4 2009
    
    monthly_series = pd.Series(monthly_values, index=dates, name='debt_level')
    
    # Create DataProcessor with quarterly frequency
    config = ProcessingConfig(target_frequency='QE')
    processor = DataProcessor(config)
    
    # Convert to quarterly
    data = {'debt_level': monthly_series}
    quarterly_data = processor.align_frequencies(
        data,
        target_freq='QE',
        stock_variables=['debt_level']
    )
    
    # Verify quarterly data
    quarterly_series = quarterly_data['debt_level']
    
    # Check that we have 8 quarters
    assert len(quarterly_series) == 8, f"Expected 8 quarters, got {len(quarterly_series)}"
    
    # Check Q1 2008: should be 110 (last value in quarter)
    q1_2008 = quarterly_series.loc['2008-03-31']
    expected_q1 = 110
    assert q1_2008 == expected_q1, f"Q1 2008: expected {expected_q1}, got {q1_2008}"
    
    # Check Q2 2008: should be 125 (last value in quarter)
    q2_2008 = quarterly_series.loc['2008-06-30']
    expected_q2 = 125
    assert q2_2008 == expected_q2, f"Q2 2008: expected {expected_q2}, got {q2_2008}"
    
    print("✓ Stock variable aggregation test passed")


def test_quarterly_validation():
    """Test validation of quarterly conversion"""
    # Create monthly data
    dates = pd.date_range('2008-01-31', periods=12, freq='ME')
    monthly_rate = pd.Series([2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5], 
                            index=dates, name='rate')
    monthly_stock = pd.Series([100, 105, 110, 115, 120, 125, 130, 135, 140, 145, 150, 155],
                             index=dates, name='stock')
    
    # Create processor
    config = ProcessingConfig(target_frequency='QE')
    processor = DataProcessor(config)
    
    # Convert to quarterly
    monthly_data = {'rate': monthly_rate, 'stock': monthly_stock}
    quarterly_data = processor.align_frequencies(
        monthly_data,
        target_freq='QE',
        rate_variables=['rate'],
        stock_variables=['stock']
    )
    
    # Validate conversion
    validation_results = processor.validate_quarterly_conversion(
        monthly_data,
        quarterly_data,
        rate_variables=['rate'],
        stock_variables=['stock']
    )
    
    # Check validation results
    assert validation_results['summary']['total_validated'] == 2
    assert validation_results['series_results']['rate']['is_valid']
    assert validation_results['series_results']['stock']['is_valid']
    assert validation_results['series_results']['rate']['variable_type'] == 'rate'
    assert validation_results['series_results']['stock']['variable_type'] == 'stock'
    
    print("✓ Quarterly validation test passed")


def test_date_range_filter():
    """Test filtering to target date range"""
    # Create data spanning 2007-2024
    dates = pd.date_range('2007-03-31', '2024-12-31', freq='QE')
    series = pd.Series(range(len(dates)), index=dates, name='test_series')
    
    # Create processor
    processor = DataProcessor()
    
    # Filter to target range (2008Q1-2023Q4)
    data = {'test_series': series}
    filtered_data = processor.filter_to_date_range(data)
    
    # Check filtered range
    filtered_series = filtered_data['test_series']
    assert filtered_series.index.min() >= pd.Timestamp('2008-03-31')
    assert filtered_series.index.max() <= pd.Timestamp('2023-12-31')
    
    print("✓ Date range filter test passed")


def test_cubic_spline_with_missing():
    """Test cubic spline interpolation with missing values"""
    # Create data with missing values
    dates = pd.date_range('2008-01-31', periods=24, freq='ME')
    values = [1.0, 2.0, np.nan, 4.0, 5.0, np.nan, np.nan, 8.0, 9.0, 10.0,
             11.0, np.nan, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0,
             21.0, 22.0, 23.0, 24.0]
    series = pd.Series(values, index=dates, name='test_series')
    
    # Create processor with cubic interpolation
    config = ProcessingConfig(interpolation_method='cubic')
    processor = DataProcessor(config)
    
    # Handle missing values
    data = {'test_series': series}
    processed_data = processor.handle_missing_values(data)
    
    # Check that missing values are filled
    processed_series = processed_data['test_series']
    assert processed_series.isna().sum() == 0, "Missing values should be filled"
    
    print("✓ Cubic spline interpolation test passed")


if __name__ == '__main__':
    print("Running DataProcessor quarterly conversion tests...\n")
    
    test_quarterly_rate_aggregation()
    test_quarterly_stock_aggregation()
    test_quarterly_validation()
    test_date_range_filter()
    test_cubic_spline_with_missing()
    
    print("\n✓ All tests passed!")
