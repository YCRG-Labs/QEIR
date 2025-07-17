#!/usr/bin/env python3
"""
Main Pipeline for Quantitative Easing and Bond Market Dynamics Analysis
Complete end-to-end analysis testing all three hypotheses from the paper

This script:
1. Downloads comprehensive dataset 
2. Processes and validates data quality
3. Tests Hypothesis 1: Threshold effects on long-term yields
4. Tests Hypothesis 2: QE impact on long-term private investment  
5. Tests Hypothesis 3: International spillover effects
6. Generates comprehensive results report

Author: Research Team
Date: 2025
"""

import os
import sys
import pandas as pd
import numpy as np
import logging
import warnings
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

# Add src directory to path
sys.path.append('src')

# Import our modules
from download_dataset import main as download_data
from src.models import (SmoothTransitionRegression, HansenThresholdRegression, 
                       InstrumentalVariablesRegression, LocalProjections, PanelVAR)
from src.analysis import (DataProcessor, QEAnalyzer, EventStudyAnalyzer, 
                         ForeignFlowAnalyzer, StatisticalTests, DataValidation)

# Import hypothesis testing modules
import testhyp1
import testhyp2  
import testhyp3

warnings.filterwarnings('ignore')

def setup_logging():
    """Setup logging for the main pipeline"""
    os.makedirs("logs", exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('logs/main_pipeline.log'),
            logging.StreamHandler()
        ]
    )

def load_processed_data():
    """Load processed data from download step"""
    data_files = {
        'us_panel': 'data/processed/us_panel.csv',
        'combined_panel': 'data/processed/combined_panel.csv',
        'market_panel': 'data/processed/market_panel.csv'
    }
    
    loaded_data = {}
    for name, filepath in data_files.items():
        if os.path.exists(filepath):
            try:
                df = pd.read_csv(filepath, index_col=0, parse_dates=True)
                loaded_data[name] = df
                logging.info(f"Loaded {name}: {df.shape}")
            except Exception as e:
                logging.warning(f"Failed to load {filepath}: {e}")
        else:
            logging.warning(f"File not found: {filepath}")
    
    return loaded_data

def prepare_analysis_data(raw_data):
    """Prepare and validate data for econometric analysis"""
    logging.info("Preparing data for analysis...")
    
    # Use the most comprehensive dataset available
    if 'combined_panel' in raw_data and not raw_data['combined_panel'].empty:
        main_data = raw_data['combined_panel'].copy()
    elif 'us_panel' in raw_data and not raw_data['us_panel'].empty:
        main_data = raw_data['us_panel'].copy()
    else:
        raise ValueError("No suitable dataset found for analysis")
    
    # Data validation
    validation_report = DataValidation.check_data_quality(main_data)
    logging.info(f"Data quality score: {validation_report['data_quality_score']}/100")
    
    # Calculate key variables if missing
    processor = DataProcessor()
    
    # QE intensity
    if 'us_qe_intensity' not in main_data.columns:
        if 'fed_treasury_holdings' in main_data.columns and 'us_federal_debt' in main_data.columns:
            main_data['us_qe_intensity'] = processor.calculate_qe_intensity(
                main_data['fed_treasury_holdings'], 
                main_data['us_federal_debt']
            )
    
    # Debt coverage ratio  
    if 'us_dcr' not in main_data.columns:
        if 'us_interest_payments' in main_data.columns and 'us_gdp' in main_data.columns:
            main_data['us_dcr'] = processor.calculate_dcr(
                main_data['us_interest_payments'],
                main_data['us_gdp'] 
            )
    
    # Term premium
    if 'us_term_premium' not in main_data.columns:
        if 'us_10y' in main_data.columns and 'us_3m' in main_data.columns:
            main_data['us_term_premium'] = processor.calculate_term_premium(
                main_data['us_10y'],
                main_data['us_3m']
            )
    
    # Investment growth (for Hypothesis 2)
    investment_cols = [col for col in main_data.columns if 'investment' in col.lower()]
    if investment_cols and 'investment_growth' not in main_data.columns:
        inv_col = investment_cols[0]
        main_data['investment_growth'] = main_data[inv_col].pct_change(periods=4) * 100
    
    # Foreign holdings changes (for Hypothesis 3)
    if 'foreign_treasury_holdings' in main_data.columns:
        main_data['foreign_holdings_change'] = main_data['foreign_treasury_holdings'].diff()
        main_data['foreign_holdings_growth'] = main_data['foreign_treasury_holdings'].pct_change() * 100
    
    # Clean data for analysis
    analysis_data = main_data.dropna(subset=['us_10y', 'us_qe_intensity'], how='any')
    
    # Winsorize extreme values
    numeric_cols = analysis_data.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if col in ['us_10y', 'us_5y', 'us_2y']:  # Yields
            analysis_data[col] = processor.winsorize(analysis_data[col], 0.01, 0.99)
        elif 'intensity' in col:  # QE intensity variables
            analysis_data[col] = np.clip(analysis_data[col], 0, 1)
        elif 'growth' in col or 'change' in col:  # Growth/change variables
            analysis_data[col] = processor.winsorize(analysis_data[col], 0.05, 0.95)
    
    # Create additional control variables
    analysis_data['vix'] = analysis_data.get('vix', np.nan)
    analysis_data['dxy'] = analysis_data.get('dxy', np.nan)  # Dollar index
    
    # Create QE episode dummy
    qe_analyzer = QEAnalyzer()
    if 'fed_total_assets' in analysis_data.columns:
        episodes, _ = qe_analyzer.identify_qe_episodes(analysis_data['fed_total_assets'])
        analysis_data['qe_episode'] = episodes > 0
    
    logging.info(f"Analysis dataset prepared: {analysis_data.shape}")
    logging.info(f"Date range: {analysis_data.index.min()} to {analysis_data.index.max()}")
    
    return analysis_data

def run_hypothesis_tests(data):
    """Run all three hypothesis tests"""
    results = {}
    
    logging.info("="*50)
    logging.info("RUNNING HYPOTHESIS TESTS")
    logging.info("="*50)
    
    # Hypothesis 1: Threshold Effects on Yields
    logging.info("\n--- Testing Hypothesis 1: Threshold Effects ---")
    try:
        hyp1_results = testhyp1.test_hypothesis_1(data)
        results['hypothesis_1'] = hyp1_results
        
        # Log key findings
        if 'str_results' in hyp1_results:
            str_res = hyp1_results['str_results']
            threshold = str_res.get('threshold')
            if isinstance(threshold, (int, float)):
                logging.info(f"STR threshold found at QE intensity: {threshold:.3f}")
            else:
                logging.info(f"STR threshold found at QE intensity: N/A")
            logging.info(f"STR model significant: {str_res.get('significant', False)}")
        
        if 'hansen_results' in hyp1_results:
            hansen_res = hyp1_results['hansen_results']
            threshold = hansen_res.get('threshold')
            if isinstance(threshold, (int, float)):
                logging.info(f"Hansen threshold: {threshold:.3f}")
            else:
                logging.info(f"Hansen threshold: N/A")
            
    except Exception as e:
        logging.error(f"Hypothesis 1 test failed: {e}")
        results['hypothesis_1'] = {'error': str(e)}
    
    # Hypothesis 2: Investment Effects
    logging.info("\n--- Testing Hypothesis 2: Investment Effects ---")
    try:
        hyp2_results = testhyp2.test_hypothesis_2(data)
        results['hypothesis_2'] = hyp2_results
        
        # Log key findings
        if 'iv_results' in hyp2_results:
            iv_res = hyp2_results['iv_results']
            logging.info(f"QE effect on investment: {iv_res.get('qe_coefficient', 'N/A')}")
            logging.info(f"Investment effect significant: {iv_res.get('significant', False)}")
            
    except Exception as e:
        logging.error(f"Hypothesis 2 test failed: {e}")
        results['hypothesis_2'] = {'error': str(e)}
    
    # Hypothesis 3: International Spillovers
    logging.info("\n--- Testing Hypothesis 3: International Spillovers ---")
    try:
        hyp3_results = testhyp3.test_hypothesis_3(data)
        results['hypothesis_3'] = hyp3_results
        
        # Log key findings
        if 'foreign_flow_results' in hyp3_results:
            flow_res = hyp3_results['foreign_flow_results']
            logging.info(f"QE effect on foreign flows: {flow_res.get('qe_coefficient', 'N/A')}")
            logging.info(f"Foreign flow effect significant: {flow_res.get('significant', False)}")
            
    except Exception as e:
        logging.error(f"Hypothesis 3 test failed: {e}")
        results['hypothesis_3'] = {'error': str(e)}
    
    return results

def generate_comprehensive_report(data, results):
    """Generate comprehensive analysis report"""
    logging.info("Generating comprehensive report...")
    
    report = {
        'executive_summary': {},
        'data_summary': {},
        'hypothesis_results': results,
        'policy_implications': {},
        'robustness_checks': {}
    }
    
    # Data summary
    report['data_summary'] = {
        'total_observations': len(data),
        'date_range': f"{data.index.min()} to {data.index.max()}",
        'key_variables': list(data.columns),
        'missing_data_pct': (data.isnull().sum().sum() / (len(data) * len(data.columns))) * 100,
        'qe_intensity_range': f"{data['us_qe_intensity'].min():.3f} to {data['us_qe_intensity'].max():.3f}",
        'yield_range': f"{data['us_10y'].min():.2f}% to {data['us_10y'].max():.2f}%"
    }
    
    # Executive summary based on results
    summary = []
    
    # Hypothesis 1 summary
    if 'hypothesis_1' in results and 'error' not in results['hypothesis_1']:
        h1 = results['hypothesis_1']
        if h1.get('str_results', {}).get('significant', False):
            threshold = h1['str_results'].get('threshold', 0)
            summary.append(f"Strong evidence of threshold effects at QE intensity of {threshold:.1%}")
        if h1.get('hansen_results', {}).get('threshold') is not None:
            hansen_thresh = h1['hansen_results']['threshold']
            summary.append(f"Hansen threshold regression confirms threshold at {hansen_thresh:.1%}")
    
    # Hypothesis 2 summary  
    if 'hypothesis_2' in results and 'error' not in results['hypothesis_2']:
        h2 = results['hypothesis_2']
        if h2.get('iv_results', {}).get('significant', False):
            coeff = h2['iv_results'].get('qe_coefficient', 0)
            summary.append(f"QE significantly affects investment with coefficient {coeff:.3f}")
    
    # Hypothesis 3 summary
    if 'hypothesis_3' in results and 'error' not in results['hypothesis_3']:
        h3 = results['hypothesis_3']
        if h3.get('foreign_flow_results', {}).get('significant', False):
            coeff = h3['foreign_flow_results'].get('qe_coefficient', 0)
            summary.append(f"QE significantly affects foreign flows with coefficient {coeff:.3f}")
    
    report['executive_summary']['key_findings'] = summary
    
    # Policy implications
    policy_implications = []
    
    if any('threshold' in finding for finding in summary):
        policy_implications.append("QE effectiveness has clear limits - policymakers should avoid excessive intervention")
    
    if any('investment' in finding for finding in summary):
        policy_implications.append("QE may have unintended consequences for long-term private investment")
        
    if any('foreign' in finding for finding in summary):
        policy_implications.append("International spillovers suggest need for coordination among major central banks")
    
    report['policy_implications']['recommendations'] = policy_implications
    
    # Save report
    os.makedirs('results', exist_ok=True)
    
    # Save detailed results
    pd.Series(report['data_summary']).to_csv('results/data_summary.csv')
    
    # Save executive summary
    with open('results/executive_summary.txt', 'w') as f:
        f.write("QUANTITATIVE EASING AND BOND MARKET DYNAMICS\n")
        f.write("=" * 50 + "\n\n")
        f.write("EXECUTIVE SUMMARY\n\n")
        for finding in summary:
            f.write(f"• {finding}\n")
        f.write("\nPOLICY IMPLICATIONS\n\n")
        for implication in policy_implications:
            f.write(f"• {implication}\n")
            
    logging.info("Comprehensive report saved to results/")
    
    return report

def create_summary_visualizations(data, results):
    """Create key visualizations summarizing findings"""
    logging.info("Creating summary visualizations...")
    
    plt.style.use('default')
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('QE and Bond Market Dynamics: Key Findings', fontsize=16, fontweight='bold')
    
    # Plot 1: QE Intensity vs 10Y Yields over time
    ax1 = axes[0, 0]
    ax1_twin = ax1.twinx()
    
    ax1.plot(data.index, data['us_qe_intensity'], color='blue', alpha=0.7, label='QE Intensity')
    ax1_twin.plot(data.index, data['us_10y'], color='red', alpha=0.7, label='10Y Yield')
    
    ax1.set_ylabel('QE Intensity', color='blue')
    ax1_twin.set_ylabel('10Y Yield (%)', color='red')
    ax1.set_title('QE Intensity vs 10Y Yields Over Time')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Threshold relationship (if available)
    ax2 = axes[0, 1]
    if 'hypothesis_1' in results and 'scatter_data' in results['hypothesis_1']:
        scatter_data = results['hypothesis_1']['scatter_data']
        ax2.scatter(scatter_data['qe_intensity'], scatter_data['yield_change'], alpha=0.5)
        
        # Add threshold line if available
        if 'str_results' in results['hypothesis_1']:
            threshold = results['hypothesis_1']['str_results'].get('threshold')
            if threshold:
                ax2.axvline(x=threshold, color='red', linestyle='--', label=f'Threshold: {threshold:.3f}')
                ax2.legend()
    
    ax2.set_xlabel('QE Intensity')
    ax2.set_ylabel('Yield Change (bps)')
    ax2.set_title('Threshold Effects: QE Intensity vs Yield Response')
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Investment response (if available)
    ax3 = axes[1, 0]
    if 'investment_growth' in data.columns:
        # Rolling correlation between QE and investment
        rolling_corr = data['us_qe_intensity'].rolling(90).corr(data['investment_growth'])
        ax3.plot(data.index, rolling_corr, color='green', alpha=0.7)
        ax3.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        ax3.set_ylabel('Rolling Correlation')
        ax3.set_title('QE-Investment Correlation (90-day rolling)')
        ax3.grid(True, alpha=0.3)
    
    # Plot 4: Foreign holdings response (if available)
    ax4 = axes[1, 1]
    if 'foreign_holdings_growth' in data.columns:
        # Scatter plot of QE vs foreign holdings growth
        ax4.scatter(data['us_qe_intensity'], data['foreign_holdings_growth'], alpha=0.5, color='orange')
        
        # Add trend line
        valid_data = data[['us_qe_intensity', 'foreign_holdings_growth']].dropna()
        if len(valid_data) > 10:
            z = np.polyfit(valid_data['us_qe_intensity'], valid_data['foreign_holdings_growth'], 1)
            p = np.poly1d(z)
            ax4.plot(valid_data['us_qe_intensity'].sort_values(), 
                    p(valid_data['us_qe_intensity'].sort_values()), "r--", alpha=0.8)
    
    ax4.set_xlabel('QE Intensity')
    ax4.set_ylabel('Foreign Holdings Growth (%)')
    ax4.set_title('QE Impact on Foreign Treasury Holdings')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('results/summary_visualizations.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    logging.info("Summary visualizations saved to results/summary_visualizations.png")

def main():
    """Main pipeline execution"""
    start_time = datetime.now()
    setup_logging()
    
    logging.info("="*60)
    logging.info("QUANTITATIVE EASING AND BOND MARKET DYNAMICS")
    logging.info("Main Analysis Pipeline")
    logging.info("="*60)
    
    try:
        # Step 1: Download data
        logging.info("\n1. DOWNLOADING DATA")
        logging.info("-" * 30)
        
        if not os.path.exists('data/processed'):
            logging.info("No processed data found. Downloading fresh data...")
            download_data()
        else:
            logging.info("Using existing processed data. Delete data/processed to re-download.")
        
        # Step 2: Load and prepare data
        logging.info("\n2. LOADING AND PREPARING DATA")
        logging.info("-" * 30)
        
        raw_data = load_processed_data()
        if not raw_data:
            raise ValueError("No data available for analysis. Check data download.")
        
        analysis_data = prepare_analysis_data(raw_data)
        
        # Step 3: Run hypothesis tests
        logging.info("\n3. RUNNING HYPOTHESIS TESTS")
        logging.info("-" * 30)
        
        results = run_hypothesis_tests(analysis_data)
        
        # Step 4: Generate report
        logging.info("\n4. GENERATING COMPREHENSIVE REPORT")
        logging.info("-" * 30)
        
        report = generate_comprehensive_report(analysis_data, results)
        
        # Step 5: Create visualizations
        logging.info("\n5. CREATING VISUALIZATIONS")
        logging.info("-" * 30)
        
        create_summary_visualizations(analysis_data, results)
        
        # Final summary
        end_time = datetime.now()
        duration = end_time - start_time
        
        logging.info("\n" + "="*60)
        logging.info("ANALYSIS COMPLETED SUCCESSFULLY!")
        logging.info("="*60)
        logging.info(f"Total execution time: {duration}")
        logging.info(f"Results saved in: results/")
        logging.info(f"Analysis covered {len(analysis_data)} observations from {analysis_data.index.min().date()} to {analysis_data.index.max().date()}")
        
        # Print key findings
        if report['executive_summary'].get('key_findings'):
            logging.info("\nKEY FINDINGS:")
            for finding in report['executive_summary']['key_findings']:
                logging.info(f"• {finding}")
        
        if report['policy_implications'].get('recommendations'):
            logging.info("\nPOLICY IMPLICATIONS:")
            for implication in report['policy_implications']['recommendations']:
                logging.info(f"• {implication}")
        
        return True
        
    except Exception as e:
        logging.error(f"Pipeline failed: {e}")
        logging.error("Check logs/main_pipeline.log for detailed error information")
        return False

if __name__ == "__main__":
    success = main()
    
    if success:
        print("\n🎉 Analysis completed successfully!")
        print("📊 Check results/ folder for detailed findings")
        print("📈 Summary visualizations saved")
        print("📋 Check logs/main_pipeline.log for full details")
    else:
        print("\n❌ Analysis failed - check logs for details")
        sys.exit(1)