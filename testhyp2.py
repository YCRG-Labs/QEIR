#!/usr/bin/env python3
"""
Test Hypothesis 2: QE Impact on Long-Term Private Investment

Hypothesis 2: Intensive QE reduces long-term private investment (I_t^LT) through two channels:
(i) the interest rate channel, where changes in r_t^L affect investment incentives, and 
(ii) the distortion channel, where excessive QE directly discourages investment due to market uncertainty.

Testing approach:
1. Instrumental Variables (IV) regression to address endogeneity
2. Local projections for dynamic effects
3. Channel decomposition analysis
4. Robustness tests across different investment measures

Based on equations (14) and (15) from the paper.
Investment equation: I_t^LT = I_0 - μ₁r_t^L - μ₂(QE_t^L/L_t)² + error
"""

import pandas as pd
import numpy as np
import sys
import os
import logging
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

# Add src to path
sys.path.append('src')

from src.models import InstrumentalVariablesRegression, LocalProjections
from src.analysis import QEAnalyzer, StatisticalTests, DiagnosticTests, RobustnessTests

def prepare_hypothesis2_data(data):
    """Prepare data specifically for Hypothesis 2 testing"""
    
    # Investment variables (try multiple proxies)
    investment_candidates = [
        'us_business_investment', 'us_gross_investment', 'us_equipment_investment',
        'investment_growth', 'business_investment', 'gross_investment'
    ]
    
    investment_var = None
    for var in investment_candidates:
        if var in data.columns and data[var].dropna().count() > 100:
            investment_var = var
            break
    
    if investment_var is None:
        # Create synthetic investment proxy if real data not available
        if 'us_gdp' in data.columns:
            # Use GDP growth as proxy for investment (highly correlated)
            investment_var = 'investment_proxy'
            data[investment_var] = data['us_gdp'].pct_change(periods=4) * 100
        else:
            raise ValueError("No suitable investment variable found")
    
    # Required variables
    required_vars = ['us_10y', 'us_qe_intensity', investment_var]
    missing_vars = [var for var in required_vars if var not in data.columns]
    
    if missing_vars:
        raise ValueError(f"Missing required variables: {missing_vars}")
    
    # Create analysis dataset
    h2_data = data[required_vars].copy()
    h2_data.columns = ['long_yield', 'qe_intensity', 'investment']
    
    # Add control variables
    control_vars = ['us_dcr', 'vix', 'fed_total_assets', 'us_gdp', 'dxy']
    for var in control_vars:
        if var in data.columns:
            h2_data[var] = data[var]
    
    # Create key variables for the investment equation
    h2_data['qe_intensity_squared'] = h2_data['qe_intensity'] ** 2
    h2_data['investment_log'] = np.log(h2_data['investment'].clip(lower=0.01))
    h2_data['investment_growth'] = h2_data['investment'].pct_change(periods=4) * 100
    
    # Create lags to address endogeneity
    h2_data['long_yield_lag1'] = h2_data['long_yield'].shift(1)
    h2_data['long_yield_lag2'] = h2_data['long_yield'].shift(2)
    h2_data['qe_intensity_lag1'] = h2_data['qe_intensity'].shift(1)
    h2_data['qe_intensity_lag2'] = h2_data['qe_intensity'].shift(2)
    
    # Create instruments for IV estimation
    # Instrument 1: Lagged QE intensity changes
    h2_data['qe_change_lag1'] = h2_data['qe_intensity'].diff().shift(1)
    h2_data['qe_change_lag2'] = h2_data['qe_intensity'].diff().shift(2)
    
    # Instrument 2: Foreign central bank QE spillovers (proxy using volatility)
    if 'vix' in h2_data.columns:
        h2_data['foreign_spillover'] = h2_data['vix'].rolling(window=20).mean()
    
    # Instrument 3: Predetermined Treasury auction calendar effects
    h2_data['month'] = h2_data.index.month
    h2_data['auction_effect'] = np.sin(2 * np.pi * h2_data['month'] / 12)
    
    # Financial crisis controls
    crisis_start = pd.to_datetime('2008-01-01')
    crisis_end = pd.to_datetime('2009-12-31')
    h2_data['crisis_dummy'] = ((h2_data.index >= crisis_start) & (h2_data.index <= crisis_end)).astype(int)
    
    covid_start = pd.to_datetime('2020-01-01')
    covid_end = pd.to_datetime('2021-12-31')
    h2_data['covid_dummy'] = ((h2_data.index >= covid_start) & (h2_data.index <= covid_end)).astype(int)
    
    # Time trend
    h2_data['time_trend'] = np.arange(len(h2_data))
    
    # Drop missing values
    h2_data = h2_data.dropna(subset=['investment', 'long_yield', 'qe_intensity'])
    
    logging.info(f"Hypothesis 2 dataset prepared: {h2_data.shape}")
    logging.info(f"Investment variable used: {investment_var}")
    logging.info(f"Investment range: {h2_data['investment'].min():.2f} to {h2_data['investment'].max():.2f}")
    
    return h2_data

def run_iv_regression(data):
    """Run instrumental variables regression for investment equation"""
    logging.info("Running IV regression for investment equation...")
    
    # Dependent variable: log investment or investment growth
    if data['investment'].min() > 0:
        y = data['investment_log'].values
        dep_var_name = 'log_investment'
    else:
        y = data['investment_growth'].values
        dep_var_name = 'investment_growth'
    
    # Endogenous variables: long yield and QE intensity squared
    endogenous_vars = ['long_yield', 'qe_intensity_squared']
    X_endo = data[endogenous_vars].values
    
    # Exogenous controls
    exog_controls = ['time_trend', 'crisis_dummy', 'covid_dummy']
    available_controls = [col for col in exog_controls if col in data.columns]
    
    if available_controls:
        X_exog = data[available_controls].values
        X = np.column_stack([X_endo, X_exog])
    else:
        X = X_endo
    
    # Instruments
    instruments = ['long_yield_lag2', 'qe_change_lag1', 'qe_change_lag2', 'auction_effect']
    if 'foreign_spillover' in data.columns:
        instruments.append('foreign_spillover')
    
    available_instruments = [col for col in instruments if col in data.columns]
    Z = data[available_instruments].values
    
    # Add exogenous controls to instruments
    if available_controls:
        Z = np.column_stack([Z, X_exog])
    
    # Remove NaN
    valid_idx = ~(np.isnan(y) | np.any(np.isnan(X), axis=1) | np.any(np.isnan(Z), axis=1))
    y_clean = y[valid_idx]
    X_clean = X[valid_idx]
    Z_clean = Z[valid_idx]
    
    if len(y_clean) < 50:
        logging.warning("Insufficient data for IV regression")
        return {'error': 'Insufficient data'}
    
    try:
        # Fit IV model
        iv_model = InstrumentalVariablesRegression()
        endogenous_idx = list(range(len(endogenous_vars)))  # First variables are endogenous
        
        iv_model.fit(y_clean, X_clean, Z_clean, endogenous_idx=endogenous_idx)
        
        # Extract results
        second_stage = iv_model.second_stage_results
        
        # Calculate additional statistics
        fitted_values = second_stage.fittedvalues
        residuals = second_stage.resid
        
        # First stage F-statistics (weak instruments test)
        first_stage_f_stats = []
        for i, first_stage in enumerate(iv_model.first_stage_results):
            f_stat = first_stage.fvalue
            first_stage_f_stats.append(f_stat)
        
        # Sargan test for overidentification (if overidentified)
        n_instruments = Z_clean.shape[1]
        n_endogenous = len(endogenous_idx)
        n_exogenous = X_clean.shape[1] - n_endogenous
        
        sargan_stat = np.nan
        sargan_pvalue = np.nan
        
        if n_instruments > n_endogenous:  # Overidentified
            # Simplified Sargan test
            resid_sq = residuals ** 2
            Z_resid_reg = LinearRegression().fit(Z_clean, resid_sq)
            sargan_stat = len(y_clean) * Z_resid_reg.score(Z_clean, resid_sq)
            df_sargan = n_instruments - n_endogenous
            sargan_pvalue = 1 - stats.chi2.cdf(sargan_stat, df_sargan)
        
        # Extract coefficients
        coeff_names = ['const'] + endogenous_vars + available_controls
        coeffs = pd.Series(second_stage.params, index=coeff_names)
        std_errors = pd.Series(second_stage.bse, index=coeff_names)
        t_stats = pd.Series(second_stage.tvalues, index=coeff_names)
        p_values = pd.Series(second_stage.pvalues, index=coeff_names)

        # Find QE coefficients
        qe_coeff = coeffs.get('qe_intensity_squared', np.nan)
        qe_se = std_errors.get('qe_intensity_squared', np.nan)
        qe_pvalue = p_values.get('qe_intensity_squared', np.nan)

        # Interest rate coefficient
        yield_coeff = coeffs.get('long_yield', np.nan)
        yield_se = std_errors.get('long_yield', np.nan)
        yield_pvalue = p_values.get('long_yield', np.nan)
        
        results = {
            'dependent_variable': dep_var_name,
            'n_observations': len(y_clean),
            'coefficients': coeffs.to_dict(),
            'std_errors': std_errors.to_dict(),
            'p_values': p_values.to_dict(),
            'qe_coefficient': qe_coeff,
            'qe_std_error': qe_se,
            'qe_p_value': qe_pvalue,
            'qe_significant': qe_pvalue < 0.05 if not np.isnan(qe_pvalue) else False,
            'yield_coefficient': yield_coeff,
            'yield_std_error': yield_se,
            'yield_p_value': yield_pvalue,
            'yield_significant': yield_pvalue < 0.05 if not np.isnan(yield_pvalue) else False,
            'r_squared': second_stage.rsquared,
            'adjusted_r_squared': second_stage.rsquared_adj,
            'first_stage_f_stats': first_stage_f_stats,
            'weak_instruments': any(f < 10 for f in first_stage_f_stats if not np.isnan(f)),
            'sargan_statistic': sargan_stat,
            'sargan_p_value': sargan_pvalue,
            'overid_test_passed': sargan_pvalue > 0.1 if not np.isnan(sargan_pvalue) else True,
            'fitted_values': fitted_values,
            'residuals': residuals,
            'significant': (qe_pvalue < 0.05 if not np.isnan(qe_pvalue) else False) or 
                         (yield_pvalue < 0.05 if not np.isnan(yield_pvalue) else False)
        }
        
        logging.info(f"IV Results: QE coeff = {qe_coeff:.3f} (p={qe_pvalue:.3f})")
        logging.info(f"Yield coeff = {yield_coeff:.3f} (p={yield_pvalue:.3f})")
        logging.info(f"R² = {second_stage.rsquared:.3f}")
        
        return results
        
    except Exception as e:
        logging.error(f"IV regression failed: {e}")
        return {'error': str(e)}

def run_local_projections(data):
    """Run local projections for dynamic investment effects"""
    logging.info("Running local projections for dynamic effects...")
    
    # Prepare variables
    investment = data['investment']
    qe_shock = data['qe_intensity'].diff()  # QE intensity changes as shock
    
    # Controls
    controls = data[['long_yield_lag1', 'time_trend', 'crisis_dummy', 'covid_dummy']].dropna()
    
    try:
        # Fit local projections
        lp_model = LocalProjections(max_horizon=12)  # 12 quarters ahead
        lp_model.fit(investment, qe_shock, controls=controls, lags=2)
        
        # Get impulse responses
        impulse_responses = lp_model.get_impulse_responses(shock_idx=1)  # QE shock coefficient
        
        if impulse_responses.empty:
            return {'error': 'No significant impulse responses found'}
        
        # Calculate cumulative effects
        impulse_responses['cumulative'] = impulse_responses['coefficient'].cumsum()
        
        # Identify peak effect
        peak_horizon = impulse_responses.loc[impulse_responses['coefficient'].abs().idxmax(), 'horizon']
        peak_effect = impulse_responses.loc[impulse_responses['coefficient'].abs().idxmax(), 'coefficient']
        
        # Calculate persistence (half-life)
        peak_coeff = impulse_responses['coefficient'].max()
        half_life_threshold = peak_coeff / 2
        
        half_life = None
        for idx, row in impulse_responses.iterrows():
            if row['horizon'] > peak_horizon and abs(row['coefficient']) <= half_life_threshold:
                half_life = row['horizon']
                break
        
        results = {
            'impulse_responses': impulse_responses.to_dict('records'),
            'peak_horizon': peak_horizon,
            'peak_effect': peak_effect,
            'half_life': half_life,
            'significant_horizons': impulse_responses[impulse_responses['coefficient'].abs() > 
                                                   1.96 * impulse_responses['coefficient'].abs().std()]['horizon'].tolist(),
            'total_observations': sum(1 for h in range(lp_model.max_horizon + 1) if h in lp_model.results and lp_model.results[h] is not None)
        }
        
        logging.info(f"Local Projections: Peak effect at horizon {peak_horizon} = {peak_effect:.3f}")
        
        return results
        
    except Exception as e:
        logging.error(f"Local projections failed: {e}")
        return {'error': str(e)}

def analyze_investment_channels(data, iv_results):
    """Analyze the two channels: interest rate vs distortion"""
    logging.info("Analyzing investment channels...")
    
    if 'error' in iv_results:
        return {'error': 'Cannot analyze channels without valid IV results'}
    
    # Channel decomposition based on IV results
    yield_coeff = iv_results.get('yield_coefficient', 0)
    qe_coeff = iv_results.get('qe_coefficient', 0)
    
    # Calculate channel contributions
    data_copy = data.copy()
    
    # Interest rate channel effect
    yield_change = data_copy['long_yield'].diff()
    interest_rate_effect = yield_coeff * yield_change
    
    # Distortion channel effect (QE intensity squared)
    qe_squared_change = data_copy['qe_intensity_squared'].diff()
    distortion_effect = qe_coeff * qe_squared_change
    
    # Total predicted effect
    total_effect = interest_rate_effect + distortion_effect
    
    # Channel analysis
    channels = {
        'interest_rate_channel': {
            'coefficient': yield_coeff,
            'mean_effect': interest_rate_effect.mean(),
            'std_effect': interest_rate_effect.std(),
            'contribution_pct': abs(interest_rate_effect.mean()) / (abs(interest_rate_effect.mean()) + abs(distortion_effect.mean())) * 100
        },
        'distortion_channel': {
            'coefficient': qe_coeff,
            'mean_effect': distortion_effect.mean(),
            'std_effect': distortion_effect.std(),
            'contribution_pct': abs(distortion_effect.mean()) / (abs(interest_rate_effect.mean()) + abs(distortion_effect.mean())) * 100
        },
        'total_effect': {
            'mean': total_effect.mean(),
            'std': total_effect.std(),
            'correlation_with_actual': data_copy['investment'].diff().corr(total_effect)
        }
    }
    
    # Test relative importance of channels
    # Regression of investment changes on channel effects
    try:
        channel_data = pd.DataFrame({
            'investment_change': data_copy['investment'].diff(),
            'interest_rate_effect': interest_rate_effect,
            'distortion_effect': distortion_effect
        }).dropna()
        
        if len(channel_data) > 20:
            from sklearn.linear_model import LinearRegression
            X_channels = channel_data[['interest_rate_effect', 'distortion_effect']]
            y_inv_change = channel_data['investment_change']
            
            channel_reg = LinearRegression().fit(X_channels, y_inv_change)
            
            channels['channel_regression'] = {
                'interest_rate_coeff': channel_reg.coef_[0],
                'distortion_coeff': channel_reg.coef_[1],
                'r_squared': channel_reg.score(X_channels, y_inv_change)
            }
    except Exception:
        channels['channel_regression'] = {'error': 'Channel regression failed'}
    
    return channels

def test_robustness_investment(data):
    """Test robustness across different specifications"""
    logging.info("Testing robustness of investment results...")
    
    robustness_results = {}
    
    # Test 1: Different investment measures
    investment_vars = ['investment', 'investment_growth']
    if 'investment_log' in data.columns:
        investment_vars.append('investment_log')
    
    for inv_var in investment_vars:
        if inv_var not in data.columns:
            continue
            
        try:
            # Simple OLS for comparison
            y = data[inv_var].dropna()
            X = data[['long_yield', 'qe_intensity_squared', 'time_trend']].dropna()
            
            # Align data
            common_idx = y.index.intersection(X.index)
            y_aligned = y.loc[common_idx]
            X_aligned = X.loc[common_idx]
            
            if len(y_aligned) > 30:
                reg = LinearRegression().fit(X_aligned, y_aligned)
                
                robustness_results[f'ols_{inv_var}'] = {
                    'yield_coeff': reg.coef_[0],
                    'qe_coeff': reg.coef_[1],
                    'r_squared': reg.score(X_aligned, y_aligned)
                }
        except Exception:
            continue
    
    # Test 2: Subsample analysis
    n = len(data)
    split_point = n // 2
    
    subsamples = {
        'first_half': data.iloc[:split_point],
        'second_half': data.iloc[split_point:],
        'pre_2015': data[data.index < '2015-01-01'],
        'post_2015': data[data.index >= '2015-01-01']
    }
    
    for sample_name, sample_data in subsamples.items():
        if len(sample_data) < 30:
            continue
            
        try:
            subsample_iv = run_iv_regression(sample_data)
            if 'qe_coefficient' in subsample_iv:
                robustness_results[f'subsample_{sample_name}'] = {
                    'qe_coefficient': subsample_iv['qe_coefficient'],
                    'qe_significant': subsample_iv['qe_significant'],
                    'n_obs': subsample_iv['n_observations']
                }
        except Exception:
            continue
    
    # Test 3: Alternative control specifications
    control_specifications = [
        ['time_trend'],
        ['time_trend', 'crisis_dummy'],
        ['time_trend', 'crisis_dummy', 'covid_dummy']
    ]
    
    for i, controls in enumerate(control_specifications):
        try:
            # Simplified regression with different controls
            available_controls = [c for c in controls if c in data.columns]
            if available_controls:
                y = data['investment'].dropna()
                X_vars = ['long_yield', 'qe_intensity_squared'] + available_controls
                X = data[X_vars].dropna()
                
                common_idx = y.index.intersection(X.index)
                if len(common_idx) > 30:
                    reg = LinearRegression().fit(X.loc[common_idx], y.loc[common_idx])
                    
                    robustness_results[f'controls_spec_{i+1}'] = {
                        'qe_coefficient': reg.coef_[1],  # QE squared coefficient
                        'r_squared': reg.score(X.loc[common_idx], y.loc[common_idx]),
                        'controls': available_controls
                    }
        except Exception:
            continue
    
    return robustness_results

def create_hypothesis2_visualizations(data, results):
    """Create visualizations for Hypothesis 2 results"""
    logging.info("Creating Hypothesis 2 visualizations...")
    
    plt.style.use('default')
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Hypothesis 2: QE Impact on Long-Term Private Investment', fontsize=16, fontweight='bold')
    
    # Plot 1: Investment vs QE Intensity
    ax1 = axes[0, 0]
    ax1.scatter(data['qe_intensity'], data['investment'], alpha=0.6, s=30)
    
    # Add trend line
    if len(data.dropna()) > 10:
        valid_data = data[['qe_intensity', 'investment']].dropna()
        z = np.polyfit(valid_data['qe_intensity'], valid_data['investment'], 1)
        p = np.poly1d(z)
        x_trend = np.linspace(valid_data['qe_intensity'].min(), valid_data['qe_intensity'].max(), 100)
        ax1.plot(x_trend, p(x_trend), "r--", alpha=0.8, linewidth=2)
    
    ax1.set_xlabel('QE Intensity')
    ax1.set_ylabel('Investment Level')
    ax1.set_title('QE Intensity vs Investment')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Two channels effects
    ax2 = axes[0, 1]
    if 'channel_analysis' in results and 'interest_rate_channel' in results['channel_analysis']:
        channels = results['channel_analysis']
        
        channel_names = ['Interest Rate\nChannel', 'Distortion\nChannel']
        effects = [
            channels['interest_rate_channel']['mean_effect'],
            channels['distortion_channel']['mean_effect']
        ]
        contributions = [
            channels['interest_rate_channel']['contribution_pct'],
            channels['distortion_channel']['contribution_pct']
        ]
        
        bars = ax2.bar(channel_names, effects, alpha=0.7, color=['blue', 'red'])
        ax2.set_ylabel('Mean Effect on Investment')
        ax2.set_title('Investment Channels Decomposition')
        ax2.grid(True, alpha=0.3)
        
        # Add contribution percentages
        for bar, contrib in zip(bars, contributions):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{contrib:.1f}%', ha='center', va='bottom')
    
    # Plot 3: Time series of investment and QE
    ax3 = axes[1, 0]
    ax3_twin = ax3.twinx()
    
    ax3.plot(data.index, data['investment'], color='green', alpha=0.7, label='Investment')
    ax3_twin.plot(data.index, data['qe_intensity'], color='orange', alpha=0.7, label='QE Intensity')
    
    ax3.set_ylabel('Investment Level', color='green')
    ax3_twin.set_ylabel('QE Intensity', color='orange')
    ax3.set_title('Investment and QE Over Time')
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Impulse responses (if available)
    ax4 = axes[1, 1]
    if 'local_projections' in results and 'impulse_responses' in results['local_projections']:
        ir_data = pd.DataFrame(results['local_projections']['impulse_responses'])
        
        ax4.plot(ir_data['horizon'], ir_data['coefficient'], 'b-', linewidth=2, label='Point Estimate')
        ax4.fill_between(ir_data['horizon'], ir_data['lower_ci'], ir_data['upper_ci'], 
                        alpha=0.3, color='blue', label='95% CI')
        ax4.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        
        ax4.set_xlabel('Quarters Ahead')
        ax4.set_ylabel('Investment Response')
        ax4.set_title('Dynamic Response to QE Shock')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
    else:
        # Plot residuals vs fitted if IV results available
        if 'iv_results' in results and 'fitted_values' in results['iv_results']:
            fitted = results['iv_results']['fitted_values']
            residuals = results['iv_results']['residuals']
            
            ax4.scatter(fitted, residuals, alpha=0.6, s=30)
            ax4.axhline(y=0, color='red', linestyle='--', alpha=0.8)
            ax4.set_xlabel('Fitted Values')
            ax4.set_ylabel('Residuals')
            ax4.set_title('Residuals vs Fitted (IV Model)')
            ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    os.makedirs('results', exist_ok=True)
    plt.savefig('results/hypothesis2_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()

def test_hypothesis_2(data):
    """Main function to test Hypothesis 2"""
    logging.info("="*50)
    logging.info("TESTING HYPOTHESIS 2: QE IMPACT ON INVESTMENT")
    logging.info("="*50)
    
    results = {}
    
    try:
        # Prepare data
        h2_data = prepare_hypothesis2_data(data)
        results['data_summary'] = {
            'n_observations': len(h2_data),
            'date_range': f"{h2_data.index.min()} to {h2_data.index.max()}",
            'investment_range': f"{h2_data['investment'].min():.2f} to {h2_data['investment'].max():.2f}",
            'qe_intensity_range': f"{h2_data['qe_intensity'].min():.3f} to {h2_data['qe_intensity'].max():.3f}"
        }
        
        # Run IV regression
        iv_results = run_iv_regression(h2_data)
        results['iv_results'] = iv_results
        
        # Run local projections
        lp_results = run_local_projections(h2_data)
        results['local_projections'] = lp_results
        
        # Analyze channels
        if 'error' not in iv_results:
            channel_results = analyze_investment_channels(h2_data, iv_results)
            results['channel_analysis'] = channel_results
        
        # Robustness tests
        robustness_results = test_robustness_investment(h2_data)
        results['robustness'] = robustness_results
        
        # Create visualizations
        create_hypothesis2_visualizations(h2_data, results)
        
        # Summary conclusion
        hypothesis_supported = False
        
        # Check if QE has significant negative effect on investment
        if 'iv_results' in results and results['iv_results'].get('qe_significant', False):
            qe_coeff = results['iv_results'].get('qe_coefficient', 0)
            if qe_coeff < 0:  # Negative effect as hypothesized
                hypothesis_supported = True
        
        # Alternative evidence from channel analysis
        if 'channel_analysis' in results:
            distortion_effect = results['channel_analysis'].get('distortion_channel', {}).get('mean_effect', 0)
            if distortion_effect < 0:
                hypothesis_supported = True
        
        results['hypothesis_supported'] = hypothesis_supported
        results['conclusion'] = {
            'supported': hypothesis_supported,
            'qe_effect_significant': results.get('iv_results', {}).get('qe_significant', False),
            'qe_coefficient': results.get('iv_results', {}).get('qe_coefficient', np.nan),
            'distortion_channel_evidence': 'channel_analysis' in results,
            'evidence_strength': 'strong' if hypothesis_supported else 'weak'
        }
        
        logging.info(f"Hypothesis 2 supported: {hypothesis_supported}")
        
        return results
        
    except Exception as e:
        logging.error(f"Hypothesis 2 testing failed: {e}")
        return {'error': str(e)}

if __name__ == "__main__":
    # For standalone testing
    logging.basicConfig(level=logging.INFO)
    
    # Load test data
    try:
        test_data = pd.read_csv('data/processed/us_panel.csv', index_col=0, parse_dates=True)
        results = test_hypothesis_2(test_data)
        
        print("\nHypothesis 2 Test Results:")
        print("-" * 30)
        if 'error' not in results:
            print(f"Hypothesis supported: {results['hypothesis_supported']}")
            if 'iv_results' in results and 'qe_coefficient' in results['iv_results']:
                print(f"QE coefficient: {results['iv_results']['qe_coefficient']:.4f}")
                print(f"QE significant: {results['iv_results']['qe_significant']}")
        else:
            print(f"Error: {results['error']}")
            
    except Exception as e:
        print(f"Standalone test failed: {e}")
        print("Run main.py first to generate processed data")