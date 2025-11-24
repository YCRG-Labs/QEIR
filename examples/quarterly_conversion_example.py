"""
Example: Using DataProcessor for Quarterly Frequency Conversion

This example demonstrates how to use the extended DataProcessor to convert
monthly data to quarterly frequency with proper handling of rate and stock variables.
"""

import pandas as pd
import numpy as np
from qeir.utils.data_processor import DataProcessor, ProcessingConfig


def example_quarterly_conversion():
    """
    Example of converting monthly data to quarterly frequency
    """
    print("=" * 70)
    print("Example: Quarterly Frequency Conversion")
    print("=" * 70)
    print()
    
    # Step 1: Create sample monthly data (2008-2023)
    print("Step 1: Creating sample monthly data...")
    dates = pd.date_range('2008-01-31', '2023-12-31', freq='ME')
    
    # Rate variables (should use averaging)
    treasury_10y = pd.Series(
        np.random.uniform(1.5, 4.5, len(dates)),
        index=dates,
        name='treasury_10y'
    )
    
    fed_funds_rate = pd.Series(
        np.random.uniform(0.0, 2.5, len(dates)),
        index=dates,
        name='fed_funds_rate'
    )
    
    # Stock variables (should use end-of-quarter)
    federal_debt = pd.Series(
        np.linspace(10000, 30000, len(dates)),  # Growing debt
        index=dates,
        name='federal_debt'
    )
    
    fed_holdings = pd.Series(
        np.linspace(800, 8000, len(dates)),  # Growing Fed holdings
        index=dates,
        name='fed_holdings'
    )
    
    monthly_data = {
        'treasury_10y': treasury_10y,
        'fed_funds_rate': fed_funds_rate,
        'federal_debt': federal_debt,
        'fed_holdings': fed_holdings
    }
    
    print(f"  Created {len(monthly_data)} monthly series")
    print(f"  Date range: {dates[0]} to {dates[-1]}")
    print(f"  Total observations per series: {len(dates)}")
    print()
    
    # Step 2: Configure DataProcessor for quarterly frequency
    print("Step 2: Configuring DataProcessor...")
    config = ProcessingConfig(
        target_frequency='QE',  # Quarterly end
        interpolation_method='cubic'
    )
    processor = DataProcessor(config)
    print("  Target frequency: QE (Quarterly End)")
    print()
    
    # Step 3: Convert to quarterly frequency
    print("Step 3: Converting to quarterly frequency...")
    quarterly_data = processor.align_frequencies(
        monthly_data,
        target_freq='QE',
        rate_variables=['treasury_10y', 'fed_funds_rate'],
        stock_variables=['federal_debt', 'fed_holdings']
    )
    
    print(f"  Converted {len(quarterly_data)} series to quarterly")
    for name, series in quarterly_data.items():
        print(f"    {name}: {len(series)} quarters")
    print()
    
    # Step 4: Validate the conversion
    print("Step 4: Validating quarterly conversion...")
    validation_results = processor.validate_quarterly_conversion(
        monthly_data,
        quarterly_data,
        rate_variables=['treasury_10y', 'fed_funds_rate'],
        stock_variables=['federal_debt', 'fed_holdings']
    )
    
    print(f"  Validation summary:")
    print(f"    Total validated: {validation_results['summary']['total_validated']}")
    print(f"    Valid series: {validation_results['summary']['valid_count']}")
    print(f"    Validation rate: {validation_results['summary']['validation_rate']:.1%}")
    print()
    
    # Show details for each series
    print("  Series validation details:")
    for series_name, result in validation_results['series_results'].items():
        if result.get('is_valid'):
            var_type = result.get('variable_type', 'unknown')
            method = result.get('aggregation_method', 'unknown')
            mae = result.get('mean_absolute_error', 0)
            print(f"    {series_name}:")
            print(f"      Type: {var_type}")
            print(f"      Method: {method}")
            print(f"      MAE: {mae:.2e}")
            print(f"      Status: ✓ Valid")
    print()
    
    # Step 5: Filter to target date range (2008Q1-2023Q4)
    print("Step 5: Filtering to target date range (2008Q1-2023Q4)...")
    filtered_data = processor.filter_to_date_range(quarterly_data)
    
    for name, series in filtered_data.items():
        print(f"  {name}:")
        print(f"    Start: {series.index.min()}")
        print(f"    End: {series.index.max()}")
        print(f"    Quarters: {len(series)}")
    print()
    
    # Step 6: Validate processed data quality
    print("Step 6: Validating data quality...")
    quality_results = processor.validate_processed_data(filtered_data)
    
    print(f"  Overall quality score: {quality_results['summary']['overall_quality_score']:.1f}/100")
    print(f"  Valid series: {quality_results['summary']['valid_series_count']}/{quality_results['summary']['total_series']}")
    print()
    
    print("=" * 70)
    print("✓ Example completed successfully!")
    print("=" * 70)
    
    return filtered_data


def example_with_missing_values():
    """
    Example of handling missing values with cubic spline interpolation
    """
    print("\n")
    print("=" * 70)
    print("Example: Handling Missing Values with Cubic Spline")
    print("=" * 70)
    print()
    
    # Create data with missing values
    dates = pd.date_range('2008-01-31', periods=48, freq='ME')
    values = np.random.uniform(2.0, 4.0, len(dates))
    
    # Introduce missing values
    missing_indices = [5, 10, 15, 20, 25, 30]
    for idx in missing_indices:
        values[idx] = np.nan
    
    series = pd.Series(values, index=dates, name='treasury_yield')
    
    print(f"Created series with {series.isna().sum()} missing values")
    print()
    
    # Configure processor with cubic interpolation
    config = ProcessingConfig(
        interpolation_method='cubic',
        max_consecutive_missing=6
    )
    processor = DataProcessor(config)
    
    # Handle missing values
    print("Applying cubic spline interpolation...")
    processed_data = processor.handle_missing_values({'treasury_yield': series})
    
    processed_series = processed_data['treasury_yield']
    print(f"  Missing values after interpolation: {processed_series.isna().sum()}")
    print(f"  Status: ✓ All missing values filled")
    print()
    
    print("=" * 70)
    print("✓ Example completed successfully!")
    print("=" * 70)


if __name__ == '__main__':
    # Run examples
    quarterly_data = example_quarterly_conversion()
    example_with_missing_values()
    
    print("\n")
    print("=" * 70)
    print("Key Takeaways:")
    print("=" * 70)
    print("1. Use rate_variables parameter for yields and interest rates (averaging)")
    print("2. Use stock_variables parameter for debt and holdings (end-of-quarter)")
    print("3. Validate conversion with validate_quarterly_conversion()")
    print("4. Filter to target range with filter_to_date_range()")
    print("5. Use cubic interpolation for smooth missing value handling")
    print("=" * 70)
