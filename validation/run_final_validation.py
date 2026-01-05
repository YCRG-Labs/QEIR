#!/usr/bin/env python3
"""
Final Validation Runner for QE Hypothesis Testing Framework

This script runs the comprehensive final validation suite, including:
- Economic theory validation
- Literature benchmark comparison
- Robustness testing
- Publication-ready output generation
"""

import sys
import os
import pandas as pd
import numpy as np
from pathlib import Path
import argparse
import logging
from datetime import datetime

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from qeir.validation.final_validation_suite import FinalValidationSuite
from qeir.utils.hypothesis_data_collector import HypothesisDataCollector
from qeir.core.hypothesis_testing import QEHypothesisTester
from qeir.utils.data_processor import DataProcessor

def setup_logging(output_dir: Path):
    """Setup logging for validation run."""
    log_file = output_dir / f"final_validation_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    return logging.getLogger(__name__)

def load_validation_data(data_path: str = None) -> pd.DataFrame:
    """
    Load data for validation.
    
    Parameters:
    -----------
    data_path : str, optional
        Path to validation data file. If None, collects fresh data.
        
    Returns:
    --------
    pd.DataFrame
        Complete dataset for validation
    """
    if data_path and Path(data_path).exists():
        logger.info(f"Loading validation data from {data_path}")
        return pd.read_csv(data_path, index_col=0, parse_dates=True)
    
    logger.info("Collecting fresh validation data from FRED API")
    
    # Initialize data collector
    data_collector = HypothesisDataCollector()
    
    # Collect data for all hypotheses
    start_date = "2008-01-01"
    end_date = "2023-12-31"
    
    # Collect hypothesis-specific data
    h1_data = data_collector.collect_hypothesis1_data(start_date, end_date)
    h2_data = data_collector.collect_hypothesis2_data(start_date, end_date)
    h3_data = data_collector.collect_hypothesis3_data(start_date, end_date)
    
    # Combine all data
    all_data = {**h1_data, **h2_data, **h3_data}
    
    # Process and align data
    processor = DataProcessor()
    processed_data = processor.process_and_align_data(all_data)
    
    return processed_data

def run_validation_suite(data: pd.DataFrame, output_dir: str) -> dict:
    """
    Run the comprehensive validation suite.
    
    Parameters:
    -----------
    data : pd.DataFrame
        Complete dataset for validation
    output_dir : str
        Output directory for validation results
        
    Returns:
    --------
    dict
        Validation results
    """
    logger.info("Initializing final validation suite")
    
    # Initialize validation suite
    validation_suite = FinalValidationSuite(output_dir=output_dir)
    
    # Run comprehensive validation
    logger.info("Running comprehensive validation")
    validation_results = validation_suite.run_comprehensive_validation(data)
    
    return validation_results

def generate_validation_summary(validation_results: dict, output_dir: Path):
    """Generate validation summary report."""
    summary_path = output_dir / "validation_summary.txt"
    
    with open(summary_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("FINAL VALIDATION SUMMARY\n")
        f.write("QE Hypothesis Testing Framework\n")
        f.write("=" * 80 + "\n\n")
        
        f.write(f"Validation Date: {validation_results['timestamp']}\n")
        f.write(f"Data Period: {validation_results['data_period']['start']} to "
               f"{validation_results['data_period']['end']}\n")
        f.write(f"Total Observations: {validation_results['data_period']['observations']}\n\n")
        
        # Overall scores
        final_assessment = validation_results['final_assessment']
        f.write("OVERALL ASSESSMENT\n")
        f.write("-" * 40 + "\n")
        f.write(f"Overall Validity Score: {final_assessment['overall_validity_score']:.3f}\n")
        f.write(f"Robustness Score: {final_assessment['robustness_score']:.3f}\n")
        f.write(f"Literature Consistency: {final_assessment['literature_consistency_score']:.3f}\n")
        f.write(f"Publication Ready: {final_assessment['publication_readiness']['ready_for_submission']}\n\n")
        
        # Hypothesis-specific scores
        f.write("HYPOTHESIS VALIDATION SCORES\n")
        f.write("-" * 40 + "\n")
        for hypothesis, score in final_assessment['hypothesis_validity'].items():
            f.write(f"{hypothesis.title()}: {score:.3f}\n")
        f.write("\n")
        
        # Publication readiness criteria
        f.write("PUBLICATION READINESS CRITERIA\n")
        f.write("-" * 40 + "\n")
        pub_ready = final_assessment['publication_readiness']
        for criterion, status in pub_ready.items():
            if criterion != 'ready_for_submission':
                status_str = "✓" if status else "✗"
                f.write(f"{criterion.replace('_', ' ').title()}: {status_str}\n")
        f.write("\n")
        
        # Recommendations
        f.write("RECOMMENDATIONS\n")
        f.write("-" * 40 + "\n")
        for i, recommendation in enumerate(final_assessment['recommendations'], 1):
            f.write(f"{i}. {recommendation}\n")
        f.write("\n")
        
        # Key findings
        f.write("KEY VALIDATION FINDINGS\n")
        f.write("-" * 40 + "\n")
        
        # Hypothesis 1 findings
        h1_validation = validation_results['hypothesis_validation']['hypothesis1']
        f.write("Hypothesis 1 (Threshold Effects):\n")
        theory_consistent = sum(h1_validation['economic_theory_consistency'].values())
        f.write(f"  - Economic theory consistency: {theory_consistent}/4 criteria met\n")
        stats_valid = sum(h1_validation['statistical_validity'].values())
        f.write(f"  - Statistical validity: {stats_valid}/4 criteria met\n")
        
        # Hypothesis 2 findings
        h2_validation = validation_results['hypothesis_validation']['hypothesis2']
        f.write("Hypothesis 2 (Investment Distortions):\n")
        theory_consistent = sum(h2_validation['economic_theory_consistency'].values())
        f.write(f"  - Economic theory consistency: {theory_consistent}/4 criteria met\n")
        stats_valid = sum(h2_validation['statistical_validity'].values())
        f.write(f"  - Statistical validity: {stats_valid}/4 criteria met\n")
        
        # Hypothesis 3 findings
        h3_validation = validation_results['hypothesis_validation']['hypothesis3']
        f.write("Hypothesis 3 (International Spillovers):\n")
        theory_consistent = sum(h3_validation['economic_theory_consistency'].values())
        f.write(f"  - Economic theory consistency: {theory_consistent}/4 criteria met\n")
        stats_valid = sum(h3_validation['statistical_validity'].values())
        f.write(f"  - Statistical validity: {stats_valid}/4 criteria met\n")
        
        f.write("\n" + "=" * 80 + "\n")
        f.write("Validation completed successfully.\n")
        f.write("See detailed reports in the validation output directory.\n")
        f.write("=" * 80 + "\n")
    
    logger.info(f"Validation summary saved to {summary_path}")

def main():
    """Main validation runner."""
    parser = argparse.ArgumentParser(
        description="Run final validation for QE Hypothesis Testing Framework"
    )
    parser.add_argument(
        "--data-path", 
        type=str, 
        help="Path to validation data file (optional, will collect fresh data if not provided)"
    )
    parser.add_argument(
        "--output-dir", 
        type=str, 
        default="final_validation_results",
        help="Output directory for validation results"
    )
    parser.add_argument(
        "--save-data", 
        action="store_true",
        help="Save collected data for future validation runs"
    )
    parser.add_argument(
        "--quick-validation", 
        action="store_true",
        help="Run quick validation with reduced robustness testing"
    )
    
    args = parser.parse_args()
    
    # Setup output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup logging
    global logger
    logger = setup_logging(output_dir)
    
    try:
        logger.info("Starting final validation run")
        logger.info(f"Output directory: {output_dir.absolute()}")
        
        # Load validation data
        data = load_validation_data(args.data_path)
        logger.info(f"Loaded data with {len(data)} observations from "
                   f"{data.index.min()} to {data.index.max()}")
        
        # Save data if requested
        if args.save_data:
            data_path = output_dir / "validation_data.csv"
            data.to_csv(data_path)
            logger.info(f"Validation data saved to {data_path}")
        
        # Run validation suite
        validation_results = run_validation_suite(data, str(output_dir))
        
        # Generate summary
        generate_validation_summary(validation_results, output_dir)
        
        # Print final status
        final_assessment = validation_results['final_assessment']
        print("\n" + "=" * 80)
        print("FINAL VALIDATION COMPLETED")
        print("=" * 80)
        print(f"Overall Validity Score: {final_assessment['overall_validity_score']:.3f}")
        print(f"Publication Ready: {final_assessment['publication_readiness']['ready_for_submission']}")
        print(f"Results saved to: {output_dir.absolute()}")
        print("=" * 80)
        
        # Exit with appropriate code
        if final_assessment['publication_readiness']['ready_for_submission']:
            logger.info("Validation completed successfully - framework is publication ready")
            sys.exit(0)
        else:
            logger.warning("Validation completed with issues - see recommendations")
            sys.exit(1)
            
    except Exception as e:
        logger.error(f"Validation failed with error: {str(e)}")
        logger.exception("Full traceback:")
        sys.exit(2)

if __name__ == "__main__":
    main()