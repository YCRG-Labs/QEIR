"""
QEIR: Quantitative Easing Investment Response Analysis Framework

A comprehensive econometric analysis framework for studying the effects of 
quantitative easing on investment flows and financial markets.
"""

__version__ = "1.0.0"
__author__ = "QE Research Team"

# Core imports
from .core.models import HansenThresholdRegression, LocalProjections
from .analysis.revised_qe_analyzer import RevisedQEAnalyzer
from .analysis.temporal_correction import TemporalScopeCorrector
from .analysis.identification import InstrumentValidator
from .analysis.theoretical_foundation import ThresholdTheoryBuilder, ChannelDecomposer
from .analysis.international_analysis import InternationalAnalyzer, FlowDecomposer
from .analysis.publication_strategy import PublicationAnalyzer
from .analysis.robustness_testing import RobustnessTestSuite

# Visualization imports
from .visualization.revision_visualization import RevisionVisualizationSuite
from .visualization.publication_visualization import PublicationVisualizationSuite
from .visualization.interactive_dashboard import InteractiveAnalysisDashboard

# Hypothesis testing imports
from .core.hypothesis_testing import QEHypothesisTester, HypothesisTestingConfig, HypothesisTestResults
from .core.enhanced_hypothesis2 import EnhancedHypothesis2Tester
from .core.enhanced_hypothesis3 import EnhancedHypothesis3Tester

# Utility imports
from .utils.model_diagnostics import ModelDiagnostics
from .utils.publication_model_diagnostics import PublicationModelDiagnostics
from .utils.model_specification_enhancer import ModelSpecificationEnhancer
from .utils.hypothesis_data_collector import HypothesisDataCollector
from .utils.data_structures import HypothesisData
from .utils.data_processor import DataProcessor, ProcessingConfig
from .utils.publication_export_system import PublicationExportSystem

__all__ = [
    # Core models
    "HansenThresholdRegression",
    "LocalProjections",
    
    # Main analyzer
    "RevisedQEAnalyzer",
    
    # Hypothesis testing framework
    "QEHypothesisTester",
    "HypothesisTestingConfig", 
    "HypothesisTestResults",
    "EnhancedHypothesis2Tester", 
    "EnhancedHypothesis3Tester",
    
    # Analysis components
    "TemporalScopeCorrector",
    "InstrumentValidator", 
    "ThresholdTheoryBuilder",
    "ChannelDecomposer",
    "InternationalAnalyzer",
    "FlowDecomposer",
    "PublicationAnalyzer",
    "RobustnessTestSuite",
    
    # Visualization
    "RevisionVisualizationSuite",
    "PublicationVisualizationSuite", 
    "InteractiveAnalysisDashboard",
    
    # Utilities
    "ModelDiagnostics",
    "PublicationModelDiagnostics",
    "ModelSpecificationEnhancer",
    "HypothesisDataCollector",
    "HypothesisData",
    "DataProcessor",
    "ProcessingConfig",
    "PublicationExportSystem",
]