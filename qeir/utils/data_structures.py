"""
Data structures for QE hypothesis testing

This module contains shared data structures used across the hypothesis testing framework.

Author: QE Research Team
Date: 2025
Version: 1.0
"""

import pandas as pd
from typing import Dict, Optional, Any
from dataclasses import dataclass


@dataclass
class HypothesisData:
    """
    Data structure for hypothesis-specific economic indicators.
    
    Note: Hypothesis 3 fields are deprecated as of the methodology revision.
    The analysis now focuses exclusively on domestic fiscal and investment
    channels (Hypotheses 1 and 2).
    """
    
    # Hypothesis 1: Threshold Effects
    central_bank_reaction: Optional[pd.Series] = None      # γ₁ proxy
    confidence_effects: Optional[pd.Series] = None         # λ₂ proxy  
    debt_service_burden: Optional[pd.Series] = None        # Debt service/GDP
    long_term_yields: Optional[pd.Series] = None           # 10Y Treasury yields
    
    # Hypothesis 2: Investment Effects
    qe_intensity: Optional[pd.Series] = None               # CB holdings/total outstanding
    private_investment: Optional[pd.Series] = None         # Private fixed investment
    market_distortions: Optional[pd.Series] = None         # μ₂ proxy (bid-ask spreads, etc.)
    interest_rate_channel: Optional[pd.Series] = None      # Policy rate transmission
    
    # Hypothesis 3: International Effects (DEPRECATED)
    # These fields are maintained for backward compatibility but should not be used
    # in new analyses. They will be removed in a future version.
    foreign_bond_holdings: Optional[pd.Series] = None      # DEPRECATED: Foreign holdings of domestic bonds
    exchange_rate: Optional[pd.Series] = None              # DEPRECATED: Trade-weighted exchange rate
    inflation_measures: Optional[pd.Series] = None         # DEPRECATED: CPI, PCE, import prices
    capital_flows: Optional[pd.Series] = None              # DEPRECATED: International capital flows
    
    # Common variables
    dates: Optional[pd.DatetimeIndex] = None
    metadata: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        """
        Validate data structure and warn about deprecated fields.
        
        This method provides backward compatibility checks and warnings
        when deprecated Hypothesis 3 fields are used.
        """
        import warnings
        
        # Check if any deprecated Hypothesis 3 fields are being used
        deprecated_fields = {
            'foreign_bond_holdings': self.foreign_bond_holdings,
            'exchange_rate': self.exchange_rate,
            'inflation_measures': self.inflation_measures,
            'capital_flows': self.capital_flows
        }
        
        used_deprecated_fields = [
            field_name for field_name, field_value in deprecated_fields.items()
            if field_value is not None
        ]
        
        if used_deprecated_fields:
            warnings.warn(
                f"The following Hypothesis 3 fields are deprecated and should not be used: "
                f"{', '.join(used_deprecated_fields)}. Hypothesis 3 (International Spillovers) "
                f"has been removed from the main analysis pipeline. These fields are maintained "
                f"for backward compatibility only and will be removed in a future version.",
                DeprecationWarning,
                stacklevel=2
            )