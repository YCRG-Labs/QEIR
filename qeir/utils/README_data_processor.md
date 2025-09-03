# Data Processing and Alignment Pipeline

This module implements a comprehensive data processing and alignment pipeline for QE hypothesis testing data. It handles different frequencies, missing values, variable construction for hypothesis-specific measures (γ₁, λ₂, μ₂), and data quality validation with outlier detection.

## Overview

The data processing pipeline consists of several key components:

1. **Frequency Alignment**: Align time series with different frequencies to a common frequency
2. **Missing Value Handling**: Handle missing values using various interpolation methods
3. **Outlier Detection and Treatment**: Detect and handle outliers using statistical methods
4. **Variable Construction**: Construct hypothesis-specific measures from raw data
5. **Data Quality Validation**: Validate processed data quality and coverage

## Key Classes

### ProcessingConfig

Configuration class for the data processing pipeline:

```python
from qeir.utils.data_processor import ProcessingConfig

config = ProcessingConfig(
    target_frequency='M',           # Target frequency ('D', 'W', 'M', 'Q', 'Y')
    interpolation_method='linear',  # Missing value interpolation method
    outlier_method='iqr',          # Outlier detection method
    outlier_treatment='winsorize',  # Outlier treatment method
    max_missing_pct=20.0,          # Maximum allowed missing percentage
    min_observations=24            # Minimum observations required
)
```

### DataProcessor

Main class for data processing operations:

```python
from qeir.utils.data_processor import DataProcessor, ProcessingConfig
from qeir.utils.data_structures import HypothesisData

# Initialize processor
processor = DataProcessor(config)

# Process complete HypothesisData object
processed_data, report = processor.process_hypothesis_data(hypothesis_data)
```

## Processing Steps

### 1. Frequency Alignment

Aligns all time series to a common frequency:

- **Daily to Monthly**: Downsamples using last/first/mean values
- **Monthly to Quarterly**: Aggregates using specified method
- **Mixed Frequencies**: Handles series with different frequencies automatically

```python
# Align to monthly frequency
aligned_data = processor.align_frequencies(data_dict, target_freq='M')
```

### 2. Missing Value Handling

Handles missing values using various methods:

- **Linear Interpolation**: Linear interpolation between known values
- **Cubic Interpolation**: Cubic spline interpolation
- **Forward/Backward Fill**: Propagate last/next valid observation
- **KNN Imputation**: K-nearest neighbors imputation for complex patterns

```python
# Handle missing values
processed_data = processor.handle_missing_values(data_dict)
```

### 3. Outlier Detection and Treatment

Detects outliers using statistical methods:

- **IQR Method**: Interquartile range-based detection
- **Z-Score**: Standard deviation-based detection
- **Modified Z-Score**: Median absolute deviation-based detection

Treatment options:
- **Winsorization**: Cap outliers at percentile limits
- **Capping**: Cap at reasonable bounds
- **Interpolation**: Replace outliers with interpolated values
- **Removal**: Remove outlier observations

```python
# Detect and handle outliers
cleaned_data = processor.detect_and_handle_outliers(data_dict)
```

### 4. Variable Construction

Constructs hypothesis-specific measures:

#### γ₁ (Central Bank Reaction Strength)
- Fed balance sheet growth rate
- Policy rate change intensity
- Standardized composite measure

#### λ₂ (Confidence Effects)
- Consumer confidence (inverted for negative effects)
- Financial stress index
- VIX volatility index
- Standardized composite measure

#### μ₂ (Market Distortions)
- Credit spreads
- Liquidity premiums
- Term structure measures
- Standardized composite measure

```python
# Construct hypothesis variables
final_data = processor.construct_hypothesis_variables(data_dict)
```

### 5. Data Quality Validation

Validates processed data quality:

- **Coverage Analysis**: Date range and observation count validation
- **Missing Data Assessment**: Remaining missing value analysis
- **Quality Scoring**: Overall quality score (0-100)
- **Validation Reporting**: Comprehensive quality metrics

```python
# Validate data quality
validation_results = processor.validate_processed_data(data_dict)
```

## Usage Examples

### Basic Usage

```python
from qeir.utils.data_processor import DataProcessor
from qeir.utils.data_structures import HypothesisData

# Create sample data
hypothesis_data = HypothesisData()
hypothesis_data.central_bank_reaction = pd.Series(...)
hypothesis_data.confidence_effects = pd.Series(...)

# Process data
processor = DataProcessor()
processed_data, report = processor.process_hypothesis_data(hypothesis_data)

print(f"Overall quality: {report['summary']['overall_quality']:.1f}")
print(f"Valid series: {report['summary']['valid_series']}")
```

### Custom Configuration

```python
from qeir.utils.data_processor import DataProcessor, ProcessingConfig

# Custom configuration
config = ProcessingConfig(
    target_frequency='Q',           # Quarterly frequency
    interpolation_method='cubic',   # Cubic interpolation
    outlier_method='zscore',       # Z-score outlier detection
    outlier_treatment='cap',       # Cap outliers
    max_missing_pct=15.0,         # Stricter missing data threshold
    outlier_threshold=2.5         # More sensitive outlier detection
)

processor = DataProcessor(config)
processed_data, report = processor.process_hypothesis_data(hypothesis_data)
```

### Integration with Data Collection

```python
from qeir.utils.hypothesis_data_collector import HypothesisDataCollector

# Collect and process in one step
collector = HypothesisDataCollector(os.getenv("FRED_API_KEY"))

processing_config = {
    'target_frequency': 'M',
    'interpolation_method': 'linear',
    'outlier_treatment': 'winsorize'
}

processed_data, report = collector.collect_and_process_all_data(
    "2020-01-01", "2022-12-31", processing_config
)
```

## Processing Report

The processing pipeline generates a comprehensive report:

```python
{
    'summary': {
        'input_series': 10,
        'output_series': 15,
        'constructed_variables': 5,
        'overall_quality': 85.2,
        'valid_series': 14
    },
    'processing_steps': [
        {
            'step': 'frequency_alignment',
            'target_frequency': 'M',
            'series_processed': 10,
            'timestamp': '2025-01-01T12:00:00'
        },
        # ... more steps
    ],
    'validation_results': {
        'summary': {
            'overall_quality_score': 85.2,
            'total_series': 15,
            'valid_series_count': 14
        },
        'series_results': {
            'central_bank_reaction': {
                'is_valid': True,
                'quality_score': 92.1,
                'missing_percentage': 0.0,
                'outlier_percentage': 2.1
            },
            # ... more series
        }
    }
}
```

## Quality Metrics

The pipeline provides comprehensive quality metrics:

- **Overall Quality Score**: Composite score (0-100) based on:
  - Missing data percentage (penalty up to 50 points)
  - Outlier percentage (penalty up to 25 points)
  - Date coverage (penalty up to 25 points)
  - Minimum observations requirement

- **Series-Level Metrics**:
  - Total observations
  - Missing observations and percentage
  - Outlier count and percentage
  - Date coverage ratio
  - Quality score

## Error Handling

The pipeline includes robust error handling:

- **API Failures**: Graceful handling of data collection failures
- **Processing Errors**: Continue processing other series if one fails
- **Configuration Errors**: Validation of processing parameters
- **Data Quality Issues**: Automatic handling of common data problems

## Performance Considerations

- **Memory Efficiency**: Processes series individually to minimize memory usage
- **Computational Efficiency**: Optimized algorithms for large datasets
- **Parallel Processing**: Can be extended for parallel processing of multiple series
- **Caching**: Supports caching of intermediate results

## Extensibility

The pipeline is designed for extensibility:

- **Custom Interpolation Methods**: Add new missing value handling methods
- **Custom Outlier Detection**: Implement new outlier detection algorithms
- **Custom Variable Construction**: Add new hypothesis-specific variables
- **Custom Validation Rules**: Implement domain-specific validation logic

## Testing

Comprehensive test suite covers:

- Unit tests for individual components
- Integration tests for complete pipeline
- Error handling and edge cases
- Performance benchmarks
- Data quality validation

Run tests with:
```bash
python -m pytest tests/test_data_processor.py -v
python -m pytest tests/test_data_processing_integration.py -v
```

## Dependencies

- pandas: Time series data manipulation
- numpy: Numerical computations
- scipy: Statistical functions
- scikit-learn: Machine learning utilities (KNN imputation, scaling)

## Future Enhancements

Planned enhancements include:

- **Advanced Imputation**: More sophisticated missing value imputation methods
- **Seasonal Adjustment**: Automatic seasonal adjustment for economic time series
- **Regime Detection**: Automatic detection of structural breaks
- **Real-time Processing**: Support for real-time data processing
- **Distributed Processing**: Support for distributed processing of large datasets