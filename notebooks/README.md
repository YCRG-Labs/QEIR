# QEIR Framework - Jupyter Notebooks

This directory contains comprehensive Jupyter notebooks demonstrating the QEIR (Quantitative Easing Investment Response) framework for hypothesis testing and econometric analysis.

## Overview

The QEIR framework tests three specific hypotheses about quantitative easing effects:

1. **Hypothesis 1:** Central Bank Reaction and Confidence Effects Testing
2. **Hypothesis 2:** QE Impact on Private Investment Analysis  
3. **Hypothesis 3:** International QE Effects and Currency Analysis

## Notebook Contents

### 📚 Tutorial Notebooks

#### `01_basic_tutorial.ipynb` - **START HERE**
**Recommended for:** First-time users, quick overview
- Complete step-by-step introduction to the QEIR framework
- Basic setup and configuration
- Simple analysis workflow for all three hypotheses
- Data collection and validation
- Results interpretation
- **Time to complete:** 30-45 minutes

#### `02_hypothesis1_detailed.ipynb` - Threshold Effects Analysis
**Recommended for:** Detailed Hypothesis 1 analysis
- In-depth analysis of central bank reaction and confidence effects
- Hansen threshold regression methodology
- Regime detection and visualization
- Economic interpretation of threshold behavior
- Robustness testing and sensitivity analysis
- **Time to complete:** 60-90 minutes

#### `03_hypothesis2_detailed.ipynb` - Investment Impact Analysis
**Recommended for:** Detailed Hypothesis 2 analysis
- QE impact on private investment mechanisms
- Local projections methodology
- Market distortion vs interest rate channel analysis
- Investment response visualization
- Policy implications
- **Time to complete:** 45-60 minutes

#### `04_hypothesis3_detailed.ipynb` - International Spillovers
**Recommended for:** Detailed Hypothesis 3 analysis
- International QE effects and currency analysis
- Foreign bond demand and exchange rate dynamics
- Inflation offset mechanisms
- Cross-country spillover effects
- International transmission channels
- **Time to complete:** 45-60 minutes

### 🔧 Advanced Notebooks

#### `05_advanced_configuration.ipynb` - Configuration and Customization
**Recommended for:** Power users, researchers
- Advanced configuration options and parameter tuning
- Custom time periods and QE episode analysis
- Alternative data series and proxy variables
- Performance optimization techniques
- Robustness testing configurations
- Parameter sensitivity analysis
- **Time to complete:** 60-90 minutes

#### `06_custom_analysis.ipynb` - Custom Analysis Examples
**Recommended for:** Advanced users, custom workflows
- Custom data integration techniques
- Alternative model specifications
- Custom visualization and reporting
- Export formats for external tools
- Integration with other analysis frameworks
- **Time to complete:** 45-75 minutes

## Getting Started

### Prerequisites

1. **Python Environment:** Python 3.8+ with Jupyter installed
2. **FRED API Key:** Free API key from [FRED](https://fred.stlouisfed.org/docs/api/api_key.html)
3. **Required Packages:** Install via `pip install -r requirements.txt`
4. **Internet Connection:** Required for data download

### Quick Start

1. **Start with the basic tutorial:**
   ```bash
   jupyter notebook 01_basic_tutorial.ipynb
   ```

2. **Validate your setup:**
   ```bash
   python validate_setup.py
   ```

3. **API key is loaded automatically:**
   ```python
   FRED_API_KEY = os.getenv('FRED_API_KEY')  # Load from environment variable
   ```

4. **Run all cells** to complete the basic analysis

4. **Proceed to detailed notebooks** for specific hypotheses

### Recommended Learning Path

```
01_basic_tutorial.ipynb
         ↓
Choose your focus area:
         ↓
┌─ 02_hypothesis1_detailed.ipynb (Threshold Effects)
├─ 03_hypothesis2_detailed.ipynb (Investment Impact)  
└─ 04_hypothesis3_detailed.ipynb (International Effects)
         ↓
Advanced topics:
         ↓
┌─ 05_advanced_configuration.ipynb (Configuration)
└─ 06_custom_analysis.ipynb (Custom Workflows)
```

## Notebook Features

### 📊 Interactive Analysis
- Step-by-step code execution
- Real-time data visualization
- Interactive parameter adjustment
- Immediate results feedback

### 📈 Comprehensive Visualizations
- Time series plots with QE period highlighting
- Threshold analysis scatter plots and regime visualization
- Correlation matrices and heatmaps
- Publication-quality figures
- Custom dashboard creation

### 🔍 Detailed Explanations
- Economic theory background
- Methodology explanations
- Statistical interpretation
- Policy implications
- Troubleshooting guidance

### 💾 Export Capabilities
- High-resolution figure export
- LaTeX table generation
- CSV data export
- JSON results export
- Publication-ready outputs

## Configuration Examples

### Research Configuration (High Precision)
```python
research_config = HypothesisTestingConfig(
    start_date="2000-01-01",
    end_date="2023-12-31",
    confidence_level=0.99,
    bootstrap_iterations=2000,
    enable_robustness_tests=True
)
```

### Development Configuration (Fast)
```python
dev_config = HypothesisTestingConfig(
    start_date="2018-01-01",
    end_date="2020-12-31",
    bootstrap_iterations=50,
    enable_robustness_tests=False
)
```

### QE Episode Analysis
```python
qe1_config = HypothesisTestingConfig(
    start_date="2008-11-25",
    end_date="2010-06-30",
    bootstrap_iterations=500
)
```

## Common Use Cases

### 🎓 Academic Research
- **Notebooks:** All notebooks, focus on detailed analysis
- **Configuration:** Research configuration with high precision
- **Outputs:** Publication-ready tables and figures
- **Time investment:** 4-6 hours for complete analysis

### 🏛️ Policy Analysis
- **Notebooks:** Basic tutorial + specific hypothesis notebooks
- **Configuration:** Standard configuration with robustness tests
- **Focus:** Economic interpretation and policy implications
- **Time investment:** 2-3 hours for targeted analysis

### 📚 Learning and Education
- **Notebooks:** Basic tutorial + advanced configuration
- **Configuration:** Development configuration for speed
- **Focus:** Understanding methodology and interpretation
- **Time investment:** 1-2 hours for overview

### 🔬 Method Development
- **Notebooks:** Custom analysis + advanced configuration
- **Configuration:** Custom configurations for testing
- **Focus:** Framework extension and customization
- **Time investment:** 3-5 hours for development

## Troubleshooting

### Common Issues

#### API Key Problems
```python
# Test your API key
from fredapi import Fred
fred = Fred(api_key=os.getenv('FRED_API_KEY'))
test_data = fred.get_series('GDP', limit=10)
print("API key works!" if len(test_data) > 0 else "API key failed")
```

#### Memory Issues
```python
# Use memory-optimized configuration
config = HypothesisTestingConfig(
    bootstrap_iterations=100,  # Reduce iterations
    save_intermediate_results=False,  # Save memory
    enable_robustness_tests=False  # Skip heavy computations
)
```

#### Slow Performance
```python
# Use development configuration
config = HypothesisTestingConfig(
    start_date="2015-01-01",  # Shorter period
    bootstrap_iterations=50,  # Fewer iterations
    h2_max_horizon=8  # Shorter horizon
)
```

### Getting Help

1. **Check the FAQ:** See `docs/faq.md` for common questions
2. **Troubleshooting Guide:** See `docs/troubleshooting_guide.md` for detailed solutions
3. **API Documentation:** See `docs/api_documentation.md` for technical details
4. **Enable Logging:** Add `logging.basicConfig(level=logging.DEBUG)` for detailed output

## Output Examples

### Analysis Results
```
QEIR FRAMEWORK - ANALYSIS SUMMARY
==================================================

Analysis Period: 2008-01-01 to 2023-12-31
Bootstrap Iterations: 1000
Confidence Level: 0.95

Hypothesis Testing Results:
------------------------------------------------------------

Hypothesis 1: Central Bank Reaction and Confidence Effects
  Status: ✓ COMPLETED
  Observations: 192
  Key Finding: Threshold detected
  Threshold Value: 0.2847

Hypothesis 2: QE Impact on Private Investment
  Status: ✓ COMPLETED  
  Observations: 192
  Key Finding: Investment response detected

Hypothesis 3: International QE Effects and Currency
  Status: ✓ COMPLETED
  Observations: 192
  Key Finding: Spillover effects detected
```

### Generated Files
```
analysis_outputs/
├── hypothesis_results_table.tex
├── threshold_analysis_plot.png
├── impulse_response_functions.png
├── international_spillovers.png
├── results_summary.json
├── hypothesis_data.csv
└── analysis_report.txt
```

## Performance Guidelines

### Execution Times (Approximate)

| Configuration | Basic Tutorial | Detailed Analysis | Full Robustness |
|---------------|----------------|-------------------|-----------------|
| Development   | 5-10 minutes   | 15-30 minutes     | 30-60 minutes   |
| Standard      | 10-20 minutes  | 30-60 minutes     | 1-2 hours       |
| Research      | 20-40 minutes  | 1-2 hours         | 2-4 hours       |

### Memory Requirements

| Dataset Size | Minimum RAM | Recommended RAM |
|--------------|-------------|-----------------|
| 5 years      | 2 GB        | 4 GB            |
| 10 years     | 4 GB        | 8 GB            |
| 20+ years    | 8 GB        | 16 GB           |

## Best Practices

### 🚀 Getting Started
1. Always start with `01_basic_tutorial.ipynb`
2. Use development configuration for initial exploration
3. Test with shorter time periods first
4. Verify API key before running analysis

### 📊 Analysis Workflow
1. Load and validate data quality first
2. Start with basic configurations
3. Examine results before increasing complexity
4. Save intermediate results for long analyses
5. Use robustness tests for final analysis

### 💾 Results Management
1. Create separate output directories for different analyses
2. Export results in multiple formats
3. Document configuration settings used
4. Save notebook outputs for reproducibility

### 🔧 Performance Optimization
1. Use appropriate configuration for your needs
2. Monitor memory usage during execution
3. Consider running analyses separately for large datasets
4. Cache data locally to avoid repeated API calls

## Contributing

If you create useful notebook examples or find improvements:

1. **Share Examples:** Create additional notebook examples
2. **Report Issues:** Document any problems encountered
3. **Suggest Improvements:** Propose enhancements to existing notebooks
4. **Add Documentation:** Improve explanations and comments

## License

These notebooks are part of the QEIR framework and follow the same license terms. See the main LICENSE file for details.

---

**Happy Analyzing!** 🎉

For additional support, refer to the documentation in the `docs/` directory or check the other notebooks for more examples.