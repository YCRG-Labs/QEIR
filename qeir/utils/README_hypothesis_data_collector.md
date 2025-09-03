# QE Hypothesis Data Collection Infrastructure

This document describes the enhanced FRED API data collection infrastructure for testing three specific quantitative easing (QE) hypotheses.

## Overview

The `HypothesisDataCollector` extends the existing QEIR data collection capabilities with hypothesis-specific data series mapping and collection methods. It provides automated data collection, quality validation, and error handling for comprehensive QE research.

## Features

### 🔄 Automated Data Collection
- **FRED API Integration**: Seamless integration with Federal Reserve Economic Data API
- **Hypothesis-Specific Series**: Pre-configured mappings for all three QE hypotheses
- **Fallback Mechanisms**: Alternative data series when primary sources fail
- **Rate Limiting**: Built-in API rate limiting and retry mechanisms

### 📊 Data Quality Assurance
- **Quality Validation**: Comprehensive data quality scoring (0-100)
- **Outlier Detection**: Statistical outlier identification using IQR method
- **Missing Data Analysis**: Detection and reporting of data gaps
- **Quality Recommendations**: Automated suggestions for data improvement

### 🛡️ Error Handling
- **Robust API Calls**: Exponential backoff and retry mechanisms
- **Graceful Degradation**: Continues operation when some series fail
- **Comprehensive Logging**: Detailed logging for debugging and monitoring
- **Validation Reports**: Structured quality and error reporting

## Supported Hypotheses

### Hypothesis 1: Central Bank Reaction and Confidence Effects
Tests whether strong central bank reactions combined with negative confidence effects create thresholds beyond which QE increases long-term yields.

**Key Data Series:**
- Central Bank Reaction (γ₁): Fed assets, Treasury holdings, MBS holdings
- Confidence Effects (λ₂): Consumer confidence, business confidence, financial stress
- Debt Service Burden: Federal interest payments, debt-to-GDP ratios
- Long-term Yields: 10Y/30Y Treasury yields, TIPS yields

### Hypothesis 2: QE Impact on Private Investment
Analyzes how intensive QE reduces long-term private investment when market distortions dominate interest rate effects.

**Key Data Series:**
- QE Intensity: Fed securities held, Treasury outstanding, MBS outstanding
- Private Investment: Fixed investment, equipment, structures, intellectual property
- Market Distortions (μ₂): Bid-ask spreads, corporate spreads, liquidity measures
- Interest Rate Channel: Fed funds rate, prime rate, corporate rates

### Hypothesis 3: International QE Effects and Currency
Tests whether QE reduces foreign demand for domestic bonds, leading to currency depreciation and inflationary pressures.

**Key Data Series:**
- Foreign Bond Holdings: TIC data, China/Japan holdings, agency debt
- Exchange Rates: Trade-weighted dollar, major currency pairs
- Inflation Measures: CPI, PCE, import prices, breakeven inflation
- Capital Flows: Portfolio flows, direct investment, reserve assets

## Usage Examples

### Basic Usage

```python
from qeir.utils.hypothesis_data_collector import HypothesisDataCollector
import os

# Initialize collector
fred_api_key = os.getenv('FRED_API_KEY')
collector = HypothesisDataCollector(fred_api_key)

# Collect data for specific hypothesis
h1_data = collector.collect_hypothesis1_data("2008-01-01", "2024-12-31")

# Validate data quality
validation = collector.validate_data_quality(h1_data)
print(f"Overall Quality Score: {validation['summary']['overall_quality_score']:.1f}/100")
```

### Structured Data Collection

```python
# Collect all hypothesis data in structured format
hypothesis_data = collector.collect_all_hypothesis_data("2008-01-01", "2024-12-31")

# Access specific data fields
central_bank_reaction = hypothesis_data.central_bank_reaction
confidence_effects = hypothesis_data.confidence_effects
private_investment = hypothesis_data.private_investment

# Check metadata
print(f"Total series collected: {hypothesis_data.metadata['total_series_count']}")
```

### Data Quality Analysis

```python
# Detailed quality validation
validation_result = collector.validate_data_quality(h1_data)

# Review individual series quality
for series_name, result in validation_result['series_results'].items():
    if result['is_valid']:
        print(f"{series_name}: {result['quality_score']:.1f}/100")
        if result['warnings']:
            print(f"  Warnings: {', '.join(result['warnings'])}")
```

## Data Structure

### HypothesisData Class

The `HypothesisData` dataclass provides a structured container for all hypothesis-related data:

```python
@dataclass
class HypothesisData:
    # Hypothesis 1: Threshold Effects
    central_bank_reaction: Optional[pd.Series] = None
    confidence_effects: Optional[pd.Series] = None
    debt_service_burden: Optional[pd.Series] = None
    long_term_yields: Optional[pd.Series] = None
    
    # Hypothesis 2: Investment Effects
    qe_intensity: Optional[pd.Series] = None
    private_investment: Optional[pd.Series] = None
    market_distortions: Optional[pd.Series] = None
    interest_rate_channel: Optional[pd.Series] = None
    
    # Hypothesis 3: International Effects
    foreign_bond_holdings: Optional[pd.Series] = None
    exchange_rate: Optional[pd.Series] = None
    inflation_measures: Optional[pd.Series] = None
    capital_flows: Optional[pd.Series] = None
    
    # Common fields
    dates: Optional[pd.DatetimeIndex] = None
    metadata: Optional[Dict[str, Any]] = None
```

### Quality Validation Results

```python
{
    'summary': {
        'overall_quality_score': 85.2,
        'total_series': 15,
        'valid_series_count': 14,
        'high_quality_series': 10,
        'medium_quality_series': 4,
        'low_quality_series': 0
    },
    'series_results': {
        'fed_total_assets': {
            'is_valid': True,
            'quality_score': 92.5,
            'missing_percentage': 0.0,
            'outlier_percentage': 2.1,
            'warnings': [],
            'recommendations': []
        }
        # ... more series results
    }
}
```

## Configuration

### Environment Variables

```bash
# Required: FRED API key
FRED_API_KEY=f143d39a9af034a09ad3074875af9aed

# Optional: Custom configuration
QEIR_CONFIG_PATH=path/to/config.json
```

### QEIR Configuration Integration

```python
from qeir.config import QEIRConfig

# Custom configuration
config = QEIRConfig({
    'qe_start_date': '2008-01-01',
    'qe_end_date': '2024-12-31',
    'bootstrap_iterations': 1000
})

# Initialize collector with config
collector = HypothesisDataCollector(fred_api_key, config)
```

## Error Handling

### Common Issues and Solutions

1. **Invalid API Key**
   ```python
   # Error: ValueError: Valid FRED API key is required
   # Solution: Set valid FRED_API_KEY environment variable
   ```

2. **API Rate Limiting**
   ```python
   # Automatic handling with exponential backoff
   # No user action required - collector handles retries
   ```

3. **Missing Data Series**
   ```python
   # Automatic fallback to alternative series
   # Check logs for failed series information
   ```

4. **Data Quality Issues**
   ```python
   # Review validation results and warnings
   validation = collector.validate_data_quality(data)
   for warning in validation['summary']['warnings']:
       print(f"Warning: {warning}")
   ```

## Performance Considerations

### Memory Usage
- Data is stored efficiently using pandas Series
- Large datasets are handled with streaming where possible
- Memory usage is monitored and reported

### API Efficiency
- Built-in rate limiting (0.1s between calls)
- Caching mechanisms to avoid duplicate requests
- Batch processing for multiple series

### Processing Speed
- Parallel processing where applicable
- Optimized data validation algorithms
- Efficient data structure operations

## Testing

### Unit Tests
```bash
# Run data collector tests
python -m pytest tests/test_hypothesis_data_collector.py -v

# Run integration tests
python -m pytest tests/test_hypothesis_integration.py -v
```

### Example Usage
```bash
# Run example data collection
python examples/example_hypothesis_data_collection.py
```

## Integration with QEIR Framework

The `HypothesisDataCollector` integrates seamlessly with existing QEIR components:

- **Configuration**: Uses `QEIRConfig` for consistent settings
- **Logging**: Integrates with QEIR logging system
- **Data Processing**: Compatible with existing analysis modules
- **Visualization**: Works with `PublicationVisualizationSuite`

## Future Enhancements

### Planned Features
- [ ] Additional data sources (ECB, BoJ, BoE APIs)
- [ ] Real-time data streaming capabilities
- [ ] Enhanced caching and persistence
- [ ] Machine learning-based data quality scoring
- [ ] Automated data cleaning and preprocessing

### Extension Points
- Custom data validators
- Additional hypothesis configurations
- Alternative data source integrations
- Custom quality metrics

## Support and Documentation

### Getting Help
1. Check the example scripts in `examples/`
2. Review test cases for usage patterns
3. Consult QEIR documentation for framework integration
4. Check logs for detailed error information

### Contributing
1. Follow existing code patterns and documentation standards
2. Add comprehensive tests for new features
3. Update documentation for any API changes
4. Ensure backward compatibility with existing code

## License

This module is part of the QEIR framework and follows the same licensing terms.