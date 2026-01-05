"""
Configuration settings for QEIR analysis framework.
"""

from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from datetime import datetime
import os

# Default configuration
DEFAULT_CONFIG = {
    # Analysis settings
    "qe_start_date": "2008-01-01",
    "qe_end_date": "2024-12-31",
    "threshold_confidence_level": 0.95,
    "bootstrap_iterations": 1000,
    
    # Model settings
    "hansen_grid_size": 100,
    "local_projections_horizons": 12,
    "iv_weak_instrument_threshold": 10.0,
    
    # Visualization settings
    "figure_dpi": 300,
    "figure_format": "png",
    "journal_style": "jme",
    
    # Output settings
    "results_dir": "results",
    "figures_dir": "figures", 
    "publication_dir": "publication",
    
    # Performance settings
    "n_jobs": -1,
    "memory_limit": "8GB",
    "cache_results": True,
}

# Journal-specific styling
JOURNAL_STYLES = {
    "jme": {
        "figure_size": (8, 6),
        "font_family": "Times New Roman",
        "font_size": 12,
        "line_width": 1.5,
        "color_palette": ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"],
    },
    "aej": {
        "figure_size": (7, 5),
        "font_family": "Arial",
        "font_size": 11,
        "line_width": 1.2,
        "color_palette": ["#2E86AB", "#A23B72", "#F18F01", "#C73E1D"],
    },
    "jimf": {
        "figure_size": (8, 5.5),
        "font_family": "Helvetica",
        "font_size": 10,
        "line_width": 1.0,
        "color_palette": ["#264653", "#2A9D8F", "#E9C46A", "#F4A261"],
    }
}


class QEIRConfig:
    """Configuration manager for QEIR analysis."""
    
    def __init__(self, config_dict: Dict[str, Any] = None):
        """Initialize configuration."""
        self.config = DEFAULT_CONFIG.copy()
        if config_dict:
            self.config.update(config_dict)
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value."""
        return self.config.get(key, default)
    
    def set(self, key: str, value: Any) -> None:
        """Set configuration value."""
        self.config[key] = value
    
    def update(self, config_dict: Dict[str, Any]) -> None:
        """Update configuration with dictionary."""
        self.config.update(config_dict)
    
    def get_journal_style(self, journal: str) -> Dict[str, Any]:
        """Get journal-specific styling configuration."""
        return JOURNAL_STYLES.get(journal, JOURNAL_STYLES["jme"])
    
    def get_output_dir(self, output_type: str) -> Path:
        """Get output directory path."""
        base_dir = Path(self.get("results_dir", "results"))
        if output_type == "figures":
            return base_dir / self.get("figures_dir", "figures")
        elif output_type == "publication":
            return base_dir / self.get("publication_dir", "publication")
        else:
            return base_dir
    
    def ensure_output_dirs(self) -> None:
        """Ensure output directories exist."""
        for output_type in ["results", "figures", "publication"]:
            output_dir = self.get_output_dir(output_type)
            output_dir.mkdir(parents=True, exist_ok=True)


# Global configuration instance
config = QEIRConfig()


def load_config_from_file(config_path: str) -> QEIRConfig:
    """Load configuration from file."""
    import json
    
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config_dict = json.load(f)
    
    return QEIRConfig(config_dict)


def save_config_to_file(config_obj: QEIRConfig, config_path: str) -> None:
    """Save configuration to file."""
    import json
    
    config_path = Path(config_path)
    config_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(config_path, 'w') as f:
        json.dump(config_obj.config, f, indent=2)


def validate_environment() -> bool:
    """
    Validate that the environment is properly configured.
    
    Returns:
        bool: True if environment is valid, False otherwise
    """
    import logging
    
    # Check for FRED API key
    fred_api_key = os.getenv('FRED_API_KEY')
    if not fred_api_key:
        logging.error("FRED_API_KEY environment variable is not set")
        logging.error("Please set FRED_API_KEY in your .env file")
        logging.error("You can get a free API key from: https://fred.stlouisfed.org/docs/api/api_key.html")
        return False
    
    # Validate API key format (should be 32 characters)
    if len(fred_api_key) != 32:
        logging.warning(f"FRED API key format may be incorrect (got {len(fred_api_key)} characters, expected 32)")
    
    # Test API connection
    try:
        from fredapi import Fred
        fred = Fred(api_key=fred_api_key)
        # Try to fetch a small amount of data to test the connection
        test_data = fred.get_series('GDP', limit=1)
        if test_data is not None and len(test_data) > 0:
            logging.info("✓ FRED API connection successful")
            return True
        else:
            logging.error("✗ FRED API connection failed - no data returned")
            return False
    except Exception as e:
        logging.error(f"✗ FRED API connection failed: {e}")
        return False


def get_fred_api_key() -> str:
    """
    Get FRED API key from environment with validation.
    
    Returns:
        str: FRED API key
        
    Raises:
        ValueError: If API key is not found or invalid
    """
    fred_api_key = os.getenv('FRED_API_KEY')
    if not fred_api_key:
        raise ValueError(
            "FRED_API_KEY environment variable is required. "
            "Please set it in your .env file. "
            "You can get a free API key from: https://fred.stlouisfed.org/docs/api/api_key.html"
        )
    
    if fred_api_key == "YOUR_FRED_API_KEY_HERE":
        raise ValueError("Please replace the placeholder FRED API key with your actual key")
    
    return fred_api_key


@dataclass
class RevisedMethodologyConfig:
    """
    Configuration for revised QE methodology analysis.
    
    This configuration supports the quarterly frequency framework with
    high-frequency FOMC identification and structural channel decomposition.
    
    Attributes:
        # Data Settings
        start_date: Analysis start date (default: 2008Q1)
        end_date: Analysis end date (default: 2023Q4)
        frequency: Data frequency ('Q' for quarterly, 'M' for monthly)
        
        # High-Frequency Identification
        hf_window_minutes: Window around FOMC announcements for HF identification (default: 30)
        fomc_config_path: Path to FOMC announcement dates configuration
        hf_asset_classes: Asset classes for HF surprise extraction
        
        # Threshold Regression
        threshold_trim: Trim parameter for threshold search (default: 0.15)
        threshold_grid_size: Number of grid points for threshold search (default: 100)
        bootstrap_iterations: Bootstrap iterations for CI (default: 1000)
        confidence_level: Confidence level for intervals (default: 0.95)
        
        # Local Projections
        lp_max_horizon: Maximum horizon for local projections (default: 20 quarters)
        lp_hac_lags: Lags for HAC standard errors (default: 4)
        
        # Market Distortion Index
        distortion_components: Components to include in distortion index
        normalization_method: Method for normalizing components (default: 'standard')
        
        # Channel Decomposition
        decomposition_horizon: Horizon for channel share calculation (default: 12)
        target_distortion_share: Target distortion channel share (default: 0.65)
        target_rate_share: Target rate channel share (default: 0.35)
        share_tolerance: Tolerance for channel share validation (default: 0.05)
        
        # Instrument Validation
        weak_instrument_threshold: Minimum F-statistic for valid instruments (default: 10.0)
        
        # Robustness Testing
        robustness_specifications: List of robustness tests to run
        alternative_fiscal_indicators: Alternative fiscal indicators to test
        alternative_distortion_measures: Alternative distortion measures to test
        qe_episodes: QE episodes for subsample analysis
        
        # Target Estimates (for validation)
        target_threshold: Target threshold estimate (default: 0.285)
        target_threshold_ci: Target threshold confidence interval
        target_low_regime_effect: Target low regime effect (default: -9.4 bps)
        target_high_regime_effect: Target high regime effect (default: -3.5 bps)
        target_attenuation_pct: Target attenuation percentage (default: 63%)
        target_cumulative_investment: Target cumulative investment effect (default: -2.7 pp)
        
        # Hypothesis 3 Deprecation
        include_hypothesis3: Whether to include Hypothesis 3 (default: False)
        
        # Output Settings
        output_dir: Base output directory
        generate_latex_tables: Generate LaTeX tables (default: True)
        generate_publication_figures: Generate publication figures (default: True)
        figure_format: Figure format (default: 'pdf')
        figure_dpi: Figure DPI (default: 300)
    """
    
    # Data Settings
    start_date: str = "2008-01-01"
    end_date: str = "2023-12-31"
    frequency: str = "Q"  # Quarterly
    
    # High-Frequency Identification
    hf_window_minutes: int = 30
    fomc_config_path: str = "qeir/config/fomc_announcements.json"
    hf_asset_classes: List[str] = field(default_factory=lambda: [
        "fed_funds_futures",
        "eurodollar_futures",
        "treasury_10y"
    ])
    
    # Threshold Regression
    threshold_trim: float = 0.15
    threshold_grid_size: int = 100
    bootstrap_iterations: int = 1000
    confidence_level: float = 0.95
    
    # Local Projections
    lp_max_horizon: int = 20
    lp_hac_lags: int = 4
    
    # Market Distortion Index
    distortion_components: List[str] = field(default_factory=lambda: [
        "liquidity",
        "balance_sheet",
        "concentration"
    ])
    normalization_method: str = "standard"
    
    # Channel Decomposition
    decomposition_horizon: int = 12
    target_distortion_share: float = 0.65
    target_rate_share: float = 0.35
    share_tolerance: float = 0.05
    
    # Instrument Validation
    weak_instrument_threshold: float = 10.0
    
    # Robustness Testing
    robustness_specifications: List[str] = field(default_factory=lambda: [
        "double_threshold",
        "smooth_transition",
        "alternative_fiscal",
        "alternative_distortion",
        "qe_episodes",
        "shadow_rate",
        "alternative_hf_windows"
    ])
    alternative_fiscal_indicators: List[str] = field(default_factory=lambda: [
        "gross_debt_gdp",
        "primary_deficit_gdp",
        "r_minus_g",
        "cbo_fiscal_gap"
    ])
    alternative_distortion_measures: List[str] = field(default_factory=lambda: [
        "fails_to_deliver",
        "repo_specialness",
        "clearing_volumes",
        "dealer_capital_ratios"
    ])
    qe_episodes: List[str] = field(default_factory=lambda: [
        "QE1",
        "QE2",
        "QE3",
        "COVID_QE"
    ])
    
    # Target Estimates (for validation)
    target_threshold: float = 0.285
    target_threshold_ci: tuple = (0.27, 0.30)
    target_low_regime_effect: float = -9.4  # basis points
    target_high_regime_effect: float = -3.5  # basis points
    target_attenuation_pct: float = 63.0  # percentage
    target_cumulative_investment: float = -2.7  # percentage points
    
    # Hypothesis 3 Deprecation
    include_hypothesis3: bool = False
    
    # Output Settings
    output_dir: str = "output"
    generate_latex_tables: bool = True
    generate_publication_figures: bool = True
    figure_format: str = "pdf"
    figure_dpi: int = 300
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            "start_date": self.start_date,
            "end_date": self.end_date,
            "frequency": self.frequency,
            "hf_window_minutes": self.hf_window_minutes,
            "fomc_config_path": self.fomc_config_path,
            "hf_asset_classes": self.hf_asset_classes,
            "threshold_trim": self.threshold_trim,
            "threshold_grid_size": self.threshold_grid_size,
            "bootstrap_iterations": self.bootstrap_iterations,
            "confidence_level": self.confidence_level,
            "lp_max_horizon": self.lp_max_horizon,
            "lp_hac_lags": self.lp_hac_lags,
            "distortion_components": self.distortion_components,
            "normalization_method": self.normalization_method,
            "decomposition_horizon": self.decomposition_horizon,
            "target_distortion_share": self.target_distortion_share,
            "target_rate_share": self.target_rate_share,
            "share_tolerance": self.share_tolerance,
            "weak_instrument_threshold": self.weak_instrument_threshold,
            "robustness_specifications": self.robustness_specifications,
            "alternative_fiscal_indicators": self.alternative_fiscal_indicators,
            "alternative_distortion_measures": self.alternative_distortion_measures,
            "qe_episodes": self.qe_episodes,
            "target_threshold": self.target_threshold,
            "target_threshold_ci": self.target_threshold_ci,
            "target_low_regime_effect": self.target_low_regime_effect,
            "target_high_regime_effect": self.target_high_regime_effect,
            "target_attenuation_pct": self.target_attenuation_pct,
            "target_cumulative_investment": self.target_cumulative_investment,
            "include_hypothesis3": self.include_hypothesis3,
            "output_dir": self.output_dir,
            "generate_latex_tables": self.generate_latex_tables,
            "generate_publication_figures": self.generate_publication_figures,
            "figure_format": self.figure_format,
            "figure_dpi": self.figure_dpi,
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'RevisedMethodologyConfig':
        """Create configuration from dictionary."""
        return cls(**config_dict)
    
    def validate(self) -> List[str]:
        """
        Validate configuration parameters.
        
        Returns:
            List of validation error messages (empty if valid)
        """
        errors = []
        
        # Validate dates
        try:
            datetime.strptime(self.start_date, "%Y-%m-%d")
        except ValueError:
            errors.append(f"Invalid start_date format: {self.start_date}")
        
        try:
            datetime.strptime(self.end_date, "%Y-%m-%d")
        except ValueError:
            errors.append(f"Invalid end_date format: {self.end_date}")
        
        # Validate frequency
        if self.frequency not in ["Q", "M"]:
            errors.append(f"Invalid frequency: {self.frequency}. Must be 'Q' or 'M'")
        
        # Validate numeric ranges
        if self.threshold_trim <= 0 or self.threshold_trim >= 0.5:
            errors.append(f"threshold_trim must be in (0, 0.5), got {self.threshold_trim}")
        
        if self.confidence_level <= 0 or self.confidence_level >= 1:
            errors.append(f"confidence_level must be in (0, 1), got {self.confidence_level}")
        
        if self.weak_instrument_threshold <= 0:
            errors.append(f"weak_instrument_threshold must be positive, got {self.weak_instrument_threshold}")
        
        # Validate channel shares
        if abs(self.target_distortion_share + self.target_rate_share - 1.0) > 0.01:
            errors.append(f"Channel shares must sum to 1.0, got {self.target_distortion_share + self.target_rate_share}")
        
        return errors