"""
Minimal Market Distortion Analyzer for testing
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Optional, Dict, Any

@dataclass
class MarketDistortionConfig:
    """Configuration for market distortion analysis"""
    bid_ask_weight: float = 0.4
    liquidity_weight: float = 0.3
    volatility_weight: float = 0.3

class MarketDistortionProxyBuilder:
    """Test class"""
    def __init__(self, config: Optional[MarketDistortionConfig] = None):
        self.config = config or MarketDistortionConfig()
    
    def test_method(self):
        return "Test successful"

print("Minimal module loaded successfully")