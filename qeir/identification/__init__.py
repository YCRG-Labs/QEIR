"""
High-frequency identification module for QE analysis.

This module implements high-frequency FOMC surprise identification
following Swanson (2021) methodology.
"""

from .hf_surprise_identifier import HFSurpriseIdentifier
from .instrument_validator import InstrumentValidator

__all__ = ['HFSurpriseIdentifier', 'InstrumentValidator']
