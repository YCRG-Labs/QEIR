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
    """Data structure for hypothesis-specific economic indicators"""
    
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
    
    # Hypothesis 3: International Effects
    foreign_bond_holdings: Optional[pd.Series] = None      # Foreign holdings of domestic bonds
    exchange_rate: Optional[pd.Series] = None              # Trade-weighted exchange rate
    inflation_measures: Optional[pd.Series] = None         # CPI, PCE, import prices
    capital_flows: Optional[pd.Series] = None              # International capital flows
    
    # Common variables
    dates: Optional[pd.DatetimeIndex] = None
    metadata: Optional[Dict[str, Any]] = None