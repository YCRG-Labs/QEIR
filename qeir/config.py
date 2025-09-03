"""
Configuration settings for QEIR analysis framework.
"""

from pathlib import Path
from typing import Dict, Any
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