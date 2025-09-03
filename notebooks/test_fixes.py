#!/usr/bin/env python3
"""
Test script for the hypothesis testing fixes
"""

import sys
import os
sys.path.append('.')

# Set API key
os.environ['FRED_API_KEY'] = 'f143d39a9af034a09ad3074875af9aed'

from qeir.utils.hypothesis_data_collector import HypothesisDataCollector
from qeir.core.hypothesis_testing import HypothesisTestingConfig, QEHypothesisTester

def test_fixes():
    print('Testing fixed hypothesis framework...')
    
    try:
        # Initialize
        collector = HypothesisDataCollector(fred_api_key=os.environ['FRED_API_KEY'])
        config = HypothesisTestingConfig(
            start_date='2008-01-01',
            end_date='2023-12-31',
            bootstrap_iterations=10  # Reduced for testing
        )
        tester = QEHypothesisTester(data_collector=collector, config=config)
        
        # Test data loading
        print('Loading data...')
        data = tester.load_data()
        print(f'Data loaded successfully')
        
        # Check what data we have
        print(f'\nData availability:')
        print(f'  Central bank reaction: {"✓" if data.central_bank_reaction is not None else "✗"}')
        print(f'  Confidence effects: {"✓" if data.confidence_effects is not None else "✗"}')
        print(f'  Debt service burden: {"✓" if data.debt_service_burden is not None else "✗"}')
        print(f'  Long term yields: {"✓" if data.long_term_yields is not None else "✗"}')
        print(f'  QE intensity: {"✓" if data.qe_intensity is not None else "✗"}')
        print(f'  Private investment: {"✓" if data.private_investment is not None else "✗"}')
        print(f'  Market distortions: {"✓" if data.market_distortions is not None else "✗"}')
        print(f'  Interest rate channel: {"✓" if data.interest_rate_channel is not None else "✗"}')
        
        # Test each hypothesis
        print('\nTesting Hypothesis 1...')
        h1_results = tester.test_hypothesis1(data)
        if 'error' in h1_results.main_result:
            print(f'H1 Error: {h1_results.main_result["error"]}')
        else:
            print(f'H1 Success: {h1_results.main_result.get("fitted", False)}')
            print(f'H1 Results: {h1_results.main_result}')
        
        print('\nTesting Hypothesis 2...')
        h2_results = tester.test_hypothesis2(data)
        if 'error' in h2_results.main_result:
            print(f'H2 Error: {h2_results.main_result["error"]}')
        else:
            print(f'H2 Success: {h2_results.main_result.get("fitted", False)}')
            print(f'H2 Results: {h2_results.main_result}')
        
        print('\nTesting Hypothesis 3...')
        h3_results = tester.test_hypothesis3(data)
        if 'error' in h3_results.main_result:
            print(f'H3 Error: {h3_results.main_result["error"]}')
        else:
            print(f'H3 Success: {h3_results.main_result.get("fitted", False)}')
            print(f'H3 Results: {h3_results.main_result}')
        
        print('\nAll tests completed!')
        
    except Exception as e:
        print(f'Test failed with error: {e}')
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_fixes()