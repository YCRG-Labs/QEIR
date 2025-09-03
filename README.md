# QEIR: Quantitative Easing Investment Response Framework

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

A comprehensive econometric framework for analyzing the investment response to quantitative easing policies. QEIR provides advanced statistical methods, machine learning models, and publication-ready tools for monetary policy research.

## Features

### 🔬 **Advanced Econometric Methods**
- **Threshold Effects Analysis**: Enhanced Hansen tests with credibility theory
- **Local Projections**: Impulse response analysis with confidence bands  
- **Spillover Analysis**: International transmission mechanisms
- **Robustness Testing**: Comprehensive sensitivity analysis

### 🤖 **Machine Learning Integration**
- **Neural Network Ensembles**: Deep learning for policy effect estimation
- **Gradient Boosting**: Advanced tree-based models
- **Model Comparison**: Automated benchmarking and validation
- **Uncertainty Quantification**: Bayesian and bootstrap methods

### 📊 **Publication-Ready Tools**
- **LaTeX Tables**: Automated regression table generation
- **High-Quality Figures**: Publication-standard visualizations
- **Interactive Dashboards**: Real-time analysis and exploration
- **Batch Processing**: Automated analysis workflows

### 🔧 **Developer-Friendly**
- **CLI Interface**: Command-line tools for all functions
- **Type Hints**: Full type annotation throughout
- **Comprehensive Tests**: >90% test coverage
- **Modern Packaging**: pip/conda installable

## Quick Start

### Installation

```bash
# Install from PyPI (recommended)
pip install qeir

# Or install from source
git clone https://github.com/your-username/qeir.git
cd qeir
pip install -e .
```

### Setup

1. **Get a free FRED API key**: https://fred.stlouisfed.org/docs/api/api_key.html
2. **Configure environment**:
   ```bash
   # The .env file is already configured with a FRED API key
   # If you need to use your own key, edit the .env file:
   # FRED_API_KEY=your_key_here
   ```

3. **Validate setup**:
   ```bash
   python validate_setup.py
   ```

### Basic Usage

```python
from qeir import QEHypothesisTester
from qeir.utils import HypothesisDataCollector

# Initialize data collector
import os
collector = HypothesisDataCollector(fred_api_key=os.getenv("FRED_API_KEY"))

# Collect data for analysis
data = collector.collect_hypothesis1_data(
    start_date="2008-01-01",
    end_date="2024-12-31"
)

# Run hypothesis test
tester = QEHypothesisTester(data_collector=collector)
results = tester.test_hypothesis1(data)

print(f"Threshold: {results['threshold']:.3f}")
print(f"P-value: {results['p_value']:.3f}")
```

### Command Line Interface

```bash
# Test individual hypothesis
qeir hypothesis test 1 --start-date 2008-01-01 --end-date 2024-12-31

# Test all hypotheses
qeir hypothesis test-all --config config.json

# Run validation suite
qeir validation run --quick-validation

# Generate publication figures
qeir publication generate-figures --output-dir ./figures/
```

## Documentation

- **[API Documentation](docs/api_documentation.md)**: Complete API reference
- **[User Guide](docs/)**: Step-by-step tutorials
- **[Notebooks](notebooks/)**: Interactive tutorials and examples
- **[FAQ](docs/faq.md)**: Common questions and troubleshooting

## Research Framework

### Hypothesis 1: Threshold Effects in Investment Response
Enhanced analysis of non-linear investment responses to QE policies using:
- Smooth transition regression (STR) models
- Hansen threshold tests with credibility theory
- Temporal scope correction (2008-2024)

### Hypothesis 2: Market Distortion and Confidence Effects  
Investigation of QE impact on market functioning through:
- Confidence effects analysis
- Market distortion metrics
- Cross-sectional heterogeneity

### Hypothesis 3: International Spillover Effects
Analysis of cross-border QE transmission via:
- International spillover mechanisms
- Multi-country panel analysis
- Network effects modeling

## Examples

### Basic Analysis
```python
# Simple threshold analysis
from qeir.examples import run_basic_analysis
results = run_basic_analysis()
```

### Advanced Machine Learning
```python
# Neural network ensemble
from qeir.core import NeuralNetworkEnsemble
ensemble = NeuralNetworkEnsemble()
predictions = ensemble.fit_predict(X_train, y_train, X_test)
```

### Publication Workflow
```python
# Generate publication-ready outputs
from qeir.publication import PublicationExportSystem
exporter = PublicationExportSystem()
exporter.generate_complete_publication_package()
```

## Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=qeir --cov-report=html

# Run specific test suite
pytest tests/test_hypothesis_testing.py -v
```

## Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

### Development Setup
```bash
# Clone repository
git clone https://github.com/your-username/qeir.git
cd qeir

# Install development dependencies
pip install -e ".[dev]"

# Set up pre-commit hooks
pre-commit install

# Run quality checks
make check
```

## Citation

If you use QEIR in your research, please cite:

```bibtex
@software{qeir2025,
  title={QEIR: Quantitative Easing Investment Response Framework},
  author={Your Name},
  year={2025},
  url={https://github.com/your-username/qeir},
  version={1.0.0}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Federal Reserve Economic Data (FRED) for providing economic time series
- The econometrics and machine learning communities for methodological foundations
- Contributors and users who help improve the framework

## Support

- **Issues**: [GitHub Issues](https://github.com/your-username/qeir/issues)
- **Discussions**: [GitHub Discussions](https://github.com/your-username/qeir/discussions)
- **Email**: [your.email@example.com](mailto:your.email@example.com)

---

**QEIR** - Advancing quantitative easing research through rigorous econometric analysis and modern computational methods.