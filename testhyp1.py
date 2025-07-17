#!/usr/bin/env python3
"""
Test Hypothesis 1: Threshold Effects on Long-Term Yields

Hypothesis 1: When the central bank's reaction to debt service burdens is strong (high γ₁), 
but the negative confidence effect is significant (large |λ₂|), there exists a threshold 
beyond which further QE increases long-term yields.

Testing approach:
1. Smooth Transition Regression (STR) model
2. Hansen threshold regression for robustness  
3. Event study around threshold crossings
4. Robustness tests across different samples and specifications

Based on equations (10) and (11) from the paper.
"""

import pandas as pd
import numpy as np
import sys
import os
import logging
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

# Add src to path
sys.path.append('src')

from src.models import SmoothTransitionRegression, HansenThresholdRegression
from src.analysis import QEAnalyzer, EventStudyAnalyzer, StatisticalTests, DiagnosticTests, RobustnessTests

def prepare_hypothesis1_data(data):
    """Prepare data specifically for Hypothesis 1 testing"""
    
    # Required variables
    required_vars = ['us_10y', 'us_qe_intensity']
    missing_vars = [var for var in required_vars if var not in data.columns]
    
    if missing_vars:
        raise ValueError(f"Missing required variables: {missing_vars}")
    
    # Create analysis dataset
    h1_data = data[required_vars].copy()
    
    # Add control variables if available
    control_vars = ['us_dcr', 'us_term_premium', 'vix', 'fed_total_assets']
    for var in control_vars:
        if var in data.columns:
            h1_data[var] = data[var]
    
    # Create dependent variable: yield changes
    h1_data['yield_change'] = h1_data['us_10y'].diff() * 100  # Convert to basis points
    h1_data['yield_level'] = h1_data['us_10y']
    
    # Create lagged QE intensity to address endogeneity
    h1_data['qe_intensity_lag1'] = h1_data['us_qe_intensity'].shift(1)
    h1_data['qe_intensity_lag2'] = h1_data['us_qe_intensity'].shift(2)
    
    # Create QE intensity changes
    h1_data['qe_intensity_change'] = h1_data['us_qe_intensity'].diff()
    
    # Add time trend
    h1_data['time_trend'] = np.arange(len(h1_data))
    
    # Financial crisis dummy (2008-2009)
    crisis_start = pd.to_datetime('2008-01-01')
    crisis_end = pd.to_datetime('2009-12-31')
    h1_data['crisis_dummy'] = ((h1_data.index >= crisis_start) & (h1_data.index <= crisis_end)).astype(int)
    
    # COVID crisis dummy (2020-2021)
    covid_start = pd.to_datetime('2020-01-01') 
    covid_end = pd.to_datetime('2021-12-31')
    h1_data['covid_dummy'] = ((h1_data.index >= covid_start) & (h1_data.index <= covid_end)).astype(int)
    
    # Drop rows with missing key variables
    h1_data = h1_data.dropna(subset=['yield_change', 'us_qe_intensity', 'qe_intensity_lag1'])
    
    logging.info(f"Hypothesis 1 dataset prepared: {h1_data.shape}")
    logging.info(f"QE intensity range: {h1_data['us_qe_intensity'].min():.3f} to {h1_data['us_qe_intensity'].max():.3f}")
    
    return h1_data

def run_str_analysis(data):
    """Run Smooth Transition Regression analysis"""
    logging.info("Running Smooth Transition Regression (STR) analysis...")
    
    # Prepare variables
    y = data['yield_change'].values
    qe_intensity = data['us_qe_intensity'].values
    
    # Create control matrix
    controls = ['qe_intensity_lag1', 'time_trend', 'crisis_dummy', 'covid_dummy']
    available_controls = [col for col in controls if col in data.columns]
    
    if available_controls:
        X = data[available_controls].values
    else:
        X = np.ones((len(y), 1))  # Just constant
    
    # Remove any remaining NaN
    valid_idx = ~(np.isnan(y) | np.isnan(qe_intensity) | np.any(np.isnan(X), axis=1))
    y_clean = y[valid_idx]
    qe_clean = qe_intensity[valid_idx]
    X_clean = X[valid_idx]
    
    if len(y_clean) < 50:
        logging.warning("Insufficient data for STR analysis")
        return {'error': 'Insufficient data'}
    
    try:
        # Fit STR model
        str_model = SmoothTransitionRegression()
        
        # Try multiple initial values for robustness
        best_ssr = np.inf
        best_result = None
        
        initial_thresholds = np.percentile(qe_clean, [25, 50, 75])
        
        for init_c in initial_thresholds:
            for init_gamma in [0.5, 1.0, 2.0, 5.0]:
                try:
                    result = str_model.fit(y_clean, X_clean, qe_clean, 
                                         initial_gamma=init_gamma, initial_c=init_c)
                    
                    if result.success and result.fun < best_ssr:
                        best_ssr = result.fun
                        best_result = str_model
                        
                except Exception:
                    continue
        
        if best_result is None:
            return {'error': 'STR optimization failed'}
        
        # Calculate fitted values and residuals
        fitted_values = best_result.predict(X_clean, qe_clean)
        residuals = y_clean - fitted_values
        
        # Calculate R-squared
        tss = np.sum((y_clean - np.mean(y_clean))**2)
        rss = np.sum(residuals**2)
        r_squared = 1 - rss/tss
        
        # Statistical tests
        n = len(y_clean)
        k = len(best_result.coeffs)
        
        # F-test for model significance
        f_stat = (r_squared / (k-1)) / ((1 - r_squared) / (n - k))
        f_pvalue = 1 - stats.f.cdf(f_stat, k-1, n-k)
        
        # Test threshold significance (gamma parameter)
        gamma_tstat = best_result.gamma / (best_result.std_errors[1] if len(best_result.std_errors) > 1 else 1)
        gamma_pvalue = 2 * (1 - stats.norm.cdf(abs(gamma_tstat)))
        
        results = {
            'threshold': best_result.c,
            'gamma': best_result.gamma,
            'coefficients': best_result.coeffs,
            'std_errors': best_result.std_errors,
            'r_squared': r_squared,
            'n_observations': n,
            'f_statistic': f_stat,
            'f_pvalue': f_pvalue,
            'gamma_tstat': gamma_tstat,
            'gamma_pvalue': gamma_pvalue,
            'significant': gamma_pvalue < 0.05,
            'fitted_values': fitted_values,
            'residuals': residuals,
            'threshold_percentile': stats.percentileofscore(qe_clean, best_result.c)
        }
        
        logging.info(f"STR Results: Threshold = {best_result.c:.3f} (γ = {best_result.gamma:.2f})")
        logging.info(f"Model R² = {r_squared:.3f}, Threshold significant: {results['significant']}")
        
        return results
        
    except Exception as e:
        logging.error(f"STR analysis failed: {e}")
        return {'error': str(e)}

def run_hansen_analysis(data):
    """Run Hansen threshold regression analysis"""
    logging.info("Running Hansen threshold regression analysis...")
    
    # Prepare variables
    y = data['yield_change'].values
    threshold_var = data['us_qe_intensity'].values
    
    # Create regressor matrix (QE intensity + controls)
    regressors = ['qe_intensity_lag1', 'time_trend', 'crisis_dummy', 'covid_dummy']
    available_regressors = [col for col in regressors if col in data.columns]
    
    if available_regressors:
        X = data[available_regressors].values
    else:
        X = data['us_qe_intensity'].values.reshape(-1, 1)
    
    # Remove NaN
    valid_idx = ~(np.isnan(y) | np.isnan(threshold_var) | np.any(np.isnan(X), axis=1))
    y_clean = y[valid_idx]
    threshold_clean = threshold_var[valid_idx]
    X_clean = X[valid_idx]
    
    if len(y_clean) < 50:
        logging.warning("Insufficient data for Hansen analysis")
        return {'error': 'Insufficient data'}
    
    try:
        # Fit Hansen threshold model
        hansen_model = HansenThresholdRegression()
        hansen_model.fit(y_clean, X_clean, threshold_clean, trim=0.15)
        
        if hansen_model.threshold is None:
            return {'error': 'No threshold found'}
        
        # Calculate regime statistics
        regime1_mask = threshold_clean <= hansen_model.threshold
        regime2_mask = threshold_clean > hansen_model.threshold
        
        n1 = np.sum(regime1_mask)
        n2 = np.sum(regime2_mask)
        
        # Calculate fitted values and overall R-squared
        fitted_values = hansen_model.predict(X_clean, threshold_clean)
        residuals = y_clean - fitted_values
        
        tss = np.sum((y_clean - np.mean(y_clean))**2)
        rss = np.sum(residuals**2)
        r_squared = 1 - rss/tss
        
        # Test for threshold significance (sup-Wald test approximation)
        # Compare with linear model
        from sklearn.linear_model import LinearRegression
        linear_model = LinearRegression().fit(X_clean, y_clean)
        linear_residuals = y_clean - linear_model.predict(X_clean)
        linear_rss = np.sum(linear_residuals**2)
        
        # F-type statistic for threshold
        wald_stat = n1 * n2 / len(y_clean) * (linear_rss - rss) / rss
        
        results = {
            'threshold': hansen_model.threshold,
            'regime1_n': n1,
            'regime2_n': n2,
            'regime1_coeffs': hansen_model.beta1,
            'regime2_coeffs': hansen_model.beta2,
            'regime1_se': hansen_model.se1,
            'regime2_se': hansen_model.se2,
            'r_squared': r_squared,
            'wald_statistic': wald_stat,
            'threshold_percentile': stats.percentileofscore(threshold_clean, hansen_model.threshold),
            'fitted_values': fitted_values,
            'residuals': residuals
        }
        
        logging.info(f"Hansen Results: Threshold = {hansen_model.threshold:.3f}")
        logging.info(f"Regime 1: {n1} obs, Regime 2: {n2} obs, R² = {r_squared:.3f}")
        
        return results
        
    except Exception as e:
        logging.error(f"Hansen analysis failed: {e}")
        return {'error': str(e)}

def test_threshold_stability(data, str_results):
    """Test stability of threshold estimates across subsamples"""
    logging.info("Testing threshold stability across subsamples...")
    
    if 'error' in str_results:
        return {'error': 'Cannot test stability without valid STR results'}
    
    # Split sample tests
    n = len(data)
    split_point = n // 2
    
    first_half = data.iloc[:split_point]
    second_half = data.iloc[split_point:]
    
    stability_results = {}
    
    # Test both subsamples
    for sample_name, sample_data in [('first_half', first_half), ('second_half', second_half)]:
        try:
            subsample_str = run_str_analysis(sample_data)
            if 'threshold' in subsample_str:
                stability_results[sample_name] = {
                    'threshold': subsample_str['threshold'],
                    'significant': subsample_str.get('significant', False),
                    'n_obs': subsample_str['n_observations']
                }
        except Exception:
            stability_results[sample_name] = {'error': 'Subsample analysis failed'}
    
    # Rolling window estimation
    window_size = max(200, n // 4)
    rolling_thresholds = []
    
    for i in range(window_size, n - 50):
        window_data = data.iloc[i-window_size:i]
        try:
            window_str = run_str_analysis(window_data)
            if 'threshold' in window_str and window_str.get('significant', False):
                rolling_thresholds.append({
                    'date': data.index[i],
                    'threshold': window_str['threshold'],
                    'gamma': window_str.get('gamma', np.nan)
                })
        except Exception:
            continue
    
    stability_results['rolling_estimates'] = rolling_thresholds
    
    # Calculate stability metrics
    if rolling_thresholds:
        thresholds = [r['threshold'] for r in rolling_thresholds]
        stability_results['threshold_mean'] = np.mean(thresholds)
        stability_results['threshold_std'] = np.std(thresholds)
        stability_results['threshold_range'] = (np.min(thresholds), np.max(thresholds))
        stability_results['coefficient_variation'] = np.std(thresholds) / np.mean(thresholds)
    
    return stability_results

def analyze_threshold_mechanisms(data, str_results, hansen_results):
    """Analyze the economic mechanisms behind threshold effects"""
    logging.info("Analyzing threshold mechanisms...")
    
    if 'error' in str_results:
        return {'error': 'Cannot analyze mechanisms without valid threshold results'}
    
    threshold = str_results['threshold']
    
    # Create high/low QE intensity regimes
    data_copy = data.copy()
    data_copy['high_qe_regime'] = data_copy['us_qe_intensity'] > threshold
    
    # Analyze differences between regimes
    regime_analysis = {}
    
    # Basic statistics by regime
    for regime_name, regime_data in data_copy.groupby('high_qe_regime'):
        regime_label = 'high_qe' if regime_name else 'low_qe'
        
        regime_stats = {
            'n_observations': len(regime_data),
            'mean_qe_intensity': regime_data['us_qe_intensity'].mean(),
            'mean_yield_level': regime_data['us_10y'].mean(),
            'mean_yield_change': regime_data['yield_change'].mean(),
            'yield_volatility': regime_data['yield_change'].std()
        }
        
        # Add other variables if available
        for var in ['us_dcr', 'vix', 'fed_total_assets']:
            if var in regime_data.columns:
                regime_stats[f'mean_{var}'] = regime_data[var].mean()
        
        regime_analysis[regime_label] = regime_stats
    
    # Test for significant differences between regimes
    low_qe_data = data_copy[~data_copy['high_qe_regime']]
    high_qe_data = data_copy[data_copy['high_qe_regime']]
    
    if len(low_qe_data) > 10 and len(high_qe_data) > 10:
        # T-test for yield changes
        yield_ttest = stats.ttest_ind(
            low_qe_data['yield_change'].dropna(),
            high_qe_data['yield_change'].dropna()
        )
        
        regime_analysis['yield_change_ttest'] = {
            'statistic': yield_ttest.statistic,
            'p_value': yield_ttest.pvalue,
            'significant_difference': yield_ttest.pvalue < 0.05
        }
        
        # Test for volatility differences (F-test)
        low_var = low_qe_data['yield_change'].var()
        high_var = high_qe_data['yield_change'].var()
        f_stat = high_var / low_var if low_var > 0 else np.nan
        
        regime_analysis['volatility_test'] = {
            'low_regime_variance': low_var,
            'high_regime_variance': high_var,
            'f_statistic': f_stat
        }
    
    return regime_analysis

def create_hypothesis1_visualizations(data, results):
    """Create visualizations for Hypothesis 1 results"""
    logging.info("Creating Hypothesis 1 visualizations...")
    
    plt.style.use('default')
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Hypothesis 1: Threshold Effects on Long-Term Yields', fontsize=16, fontweight='bold')
    
    # Plot 1: QE Intensity vs Yield Changes (scatter)
    ax1 = axes[0, 0]
    ax1.scatter(data['us_qe_intensity'], data['yield_change'], alpha=0.5, s=20)
    
    # Add threshold line if STR was successful
    if 'str_results' in results and 'threshold' in results['str_results']:
        threshold = results['str_results']['threshold']
        ax1.axvline(x=threshold, color='red', linestyle='--', linewidth=2, 
                   label=f'STR Threshold: {threshold:.3f}')
        ax1.legend()
    
    ax1.set_xlabel('QE Intensity')
    ax1.set_ylabel('Yield Change (bps)')
    ax1.set_title('QE Intensity vs Yield Changes')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Time series of QE intensity and yields
    ax2 = axes[0, 1]
    ax2_twin = ax2.twinx()
    
    ax2.plot(data.index, data['us_qe_intensity'], color='blue', alpha=0.7, label='QE Intensity')
    ax2_twin.plot(data.index, data['us_10y'], color='red', alpha=0.7, label='10Y Yield')
    
    # Highlight threshold crossings
    if 'str_results' in results and 'threshold' in results['str_results']:
        threshold = results['str_results']['threshold']
        above_threshold = data['us_qe_intensity'] > threshold
        ax2.fill_between(data.index, 0, 1, where=above_threshold, alpha=0.2, color='red', 
                        transform=ax2.get_xaxis_transform(), label='Above Threshold')
    
    ax2.set_ylabel('QE Intensity', color='blue')
    ax2_twin.set_ylabel('10Y Yield (%)', color='red')
    ax2.set_title('QE Intensity and Yields Over Time')
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Regime comparison (if analysis available)
    ax3 = axes[1, 0]
    if 'mechanism_analysis' in results and 'low_qe' in results['mechanism_analysis']:
        regime_data = results['mechanism_analysis']
        
        regimes = ['low_qe', 'high_qe']
        yield_changes = [regime_data[r]['mean_yield_change'] for r in regimes if r in regime_data]
        yield_vols = [regime_data[r]['yield_volatility'] for r in regimes if r in regime_data]
        
        x = np.arange(len(regimes))
        width = 0.35
        
        ax3.bar(x - width/2, yield_changes, width, label='Mean Yield Change', alpha=0.7)
        ax3.bar(x + width/2, yield_vols, width, label='Yield Volatility', alpha=0.7)
        
        ax3.set_xlabel('QE Regime')
        ax3.set_ylabel('Basis Points')
        ax3.set_title('Yield Behavior by QE Regime')
        ax3.set_xticks(x)
        ax3.set_xticklabels(['Low QE', 'High QE'])
        ax3.legend()
        ax3.grid(True, alpha=0.3)
    
    # Plot 4: Model fit (if STR results available)
    ax4 = axes[1, 1]
    if 'str_results' in results and 'fitted_values' in results['str_results']:
        actual = data['yield_change'].dropna()
        fitted = results['str_results']['fitted_values']
        
        # Align lengths
        min_len = min(len(actual), len(fitted))
        actual = actual.iloc[:min_len]
        fitted = fitted[:min_len]
        
        ax4.scatter(fitted, actual, alpha=0.5, s=20)
        
        # Add 45-degree line
        min_val = min(actual.min(), fitted.min())
        max_val = max(actual.max(), fitted.max())
        ax4.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8)
        
        # Add R-squared
        r_squared = results['str_results'].get('r_squared', 0)
        ax4.text(0.05, 0.95, f'R² = {r_squared:.3f}', transform=ax4.transAxes, 
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    ax4.set_xlabel('Fitted Values')
    ax4.set_ylabel('Actual Values')
    ax4.set_title('Model Fit: Actual vs Fitted')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    os.makedirs('results', exist_ok=True)
    plt.savefig('results/hypothesis1_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create additional scatter plot with model predictions
    if 'str_results' in results and 'threshold' in results['str_results']:
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        
        # Create smooth QE intensity range for prediction
        qe_range = np.linspace(data['us_qe_intensity'].min(), data['us_qe_intensity'].max(), 100)
        
        # Create dummy control variables for prediction
        if 'str_results' in results and hasattr(results['str_results'], 'fitted_values'):
            # Plot actual data points
            ax.scatter(data['us_qe_intensity'], data['yield_change'], alpha=0.4, s=30, 
                      color='blue', label='Actual Data')
            
            # Add threshold line
            threshold = results['str_results']['threshold']
            ax.axvline(x=threshold, color='red', linestyle='--', linewidth=2, 
                      label=f'Threshold: {threshold:.3f}')
            
            ax.set_xlabel('QE Intensity')
            ax.set_ylabel('Yield Change (basis points)')
            ax.set_title('Threshold Model: QE Intensity vs Yield Response')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            plt.savefig('results/hypothesis1_threshold_detail.png', dpi=300, bbox_inches='tight')
            plt.close()

def test_hypothesis_1(data):
    """Main function to test Hypothesis 1"""
    logging.info("="*50)
    logging.info("TESTING HYPOTHESIS 1: THRESHOLD EFFECTS ON YIELDS")
    logging.info("="*50)
    
    results = {}
    
    try:
        # Prepare data
        h1_data = prepare_hypothesis1_data(data)
        results['data_summary'] = {
            'n_observations': len(h1_data),
            'date_range': f"{h1_data.index.min()} to {h1_data.index.max()}",
            'qe_intensity_range': f"{h1_data['us_qe_intensity'].min():.3f} to {h1_data['us_qe_intensity'].max():.3f}"
        }
        
        # Store scatter data for main visualization
        results['scatter_data'] = {
            'qe_intensity': h1_data['us_qe_intensity'],
            'yield_change': h1_data['yield_change']
        }
        
        # Run STR analysis
        str_results = run_str_analysis(h1_data)
        results['str_results'] = str_results
        
        # Run Hansen analysis
        hansen_results = run_hansen_analysis(h1_data)
        results['hansen_results'] = hansen_results
        
        # Test threshold stability
        if 'threshold' in str_results:
            stability_results = test_threshold_stability(h1_data, str_results)
            results['stability_results'] = stability_results
        
        # Analyze threshold mechanisms
        mechanism_results = analyze_threshold_mechanisms(h1_data, str_results, hansen_results)
        results['mechanism_analysis'] = mechanism_results
        
        # Create visualizations
        create_hypothesis1_visualizations(h1_data, results)
        
        # Summary conclusion
        hypothesis_supported = False
        
        if ('str_results' in results and results['str_results'].get('significant', False)) or \
           ('hansen_results' in results and 'threshold' in results['hansen_results']):
            hypothesis_supported = True
        
        results['hypothesis_supported'] = hypothesis_supported
        results['conclusion'] = {
            'supported': hypothesis_supported,
            'str_significant': results.get('str_results', {}).get('significant', False),
            'hansen_threshold_found': 'threshold' in results.get('hansen_results', {}),
            'evidence_strength': 'strong' if hypothesis_supported else 'weak'
        }
        
        logging.info(f"Hypothesis 1 supported: {hypothesis_supported}")
        
        return results
        
    except Exception as e:
        logging.error(f"Hypothesis 1 testing failed: {e}")
        return {'error': str(e)}

if __name__ == "__main__":
    # For standalone testing
    logging.basicConfig(level=logging.INFO)
    
    # Load test data (assumes main pipeline has run)
    try:
        test_data = pd.read_csv('data/processed/us_panel.csv', index_col=0, parse_dates=True)
        results = test_hypothesis_1(test_data)
        
        print("\nHypothesis 1 Test Results:")
        print("-" * 30)
        if 'error' not in results:
            print(f"Hypothesis supported: {results['hypothesis_supported']}")
            if 'str_results' in results and 'threshold' in results['str_results']:
                print(f"STR threshold: {results['str_results']['threshold']:.3f}")
            if 'hansen_results' in results and 'threshold' in results['hansen_results']:
                print(f"Hansen threshold: {results['hansen_results']['threshold']:.3f}")
        else:
            print(f"Error: {results['error']}")
            
    except Exception as e:
        print(f"Standalone test failed: {e}")
        print("Run main.py first to generate processed data")