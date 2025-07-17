#!/usr/bin/env python3
"""
Test Hypothesis 3: International Spillover Effects

Hypothesis 3: QE reduces foreign demand for domestic bonds, leading to currency depreciation 
and potential inflationary pressures, offsetting intended benefits.

Testing approach:
1. Panel VAR for international spillovers
2. High-frequency identification around QE announcements  
3. TIC data analysis for foreign investor behavior
4. Exchange rate and portfolio rebalancing effects
5. Official vs private foreign investor decomposition

Based on equations (16) and (17) from the paper.
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

from src.models import PanelVAR, HighFrequencyIdentification
from src.analysis import ForeignFlowAnalyzer, EventStudyAnalyzer, StatisticalTests, DiagnosticTests

def prepare_hypothesis3_data(data):
    """Prepare data specifically for Hypothesis 3 testing"""
    
    # Core variables for international spillovers
    core_vars = ['us_qe_intensity', 'us_10y']
    
    # Foreign holdings variables (TIC data)
    foreign_vars = [
        'foreign_treasury_holdings', 'foreign_holdings_change', 'foreign_holdings_growth',
        'china_treasury_holdings', 'japan_treasury_holdings'
    ]
    
    # Exchange rate variables
    fx_vars = ['eur_usd', 'gbp_usd', 'jpy_usd', 'dxy']
    
    # Market variables
    market_vars = ['vix', 'sp500', 'gold']
    
    # Collect available variables
    available_vars = core_vars.copy()
    
    for var_list in [foreign_vars, fx_vars, market_vars]:
        available_vars.extend([var for var in var_list if var in data.columns])
    
    # Remove duplicates
    available_vars = list(set(available_vars))
    missing_core = [var for var in core_vars if var not in data.columns]
    
    if missing_core:
        raise ValueError(f"Missing core variables: {missing_core}")
    
    # Create analysis dataset
    h3_data = data[available_vars].copy()
    
    # Calculate foreign holdings variables if not present
    if 'foreign_treasury_holdings' in h3_data.columns and 'foreign_holdings_change' not in h3_data.columns:
        h3_data['foreign_holdings_change'] = h3_data['foreign_treasury_holdings'].diff()
        h3_data['foreign_holdings_growth'] = h3_data['foreign_treasury_holdings'].pct_change() * 100
    
    # Calculate exchange rate changes and volatilities
    fx_columns = [col for col in h3_data.columns if any(fx in col for fx in ['eur_usd', 'gbp_usd', 'jpy_usd', 'dxy'])]
    for fx_col in fx_columns:
        h3_data[f'{fx_col}_change'] = h3_data[fx_col].pct_change() * 100
        h3_data[f'{fx_col}_vol'] = h3_data[fx_col].pct_change().rolling(20).std() * np.sqrt(252) * 100
    
    # Create QE intensity changes and lags
    h3_data['qe_intensity_change'] = h3_data['us_qe_intensity'].diff()
    h3_data['qe_intensity_lag1'] = h3_data['us_qe_intensity'].shift(1)
    h3_data['qe_intensity_lag2'] = h3_data['us_qe_intensity'].shift(2)
    
    # Yield changes
    h3_data['yield_change'] = h3_data['us_10y'].diff() * 100
    h3_data['yield_lag1'] = h3_data['us_10y'].shift(1)
    
    # Create crisis and regime dummies
    crisis_start = pd.to_datetime('2008-01-01')
    crisis_end = pd.to_datetime('2009-12-31')
    h3_data['crisis_dummy'] = ((h3_data.index >= crisis_start) & (h3_data.index <= crisis_end)).astype(int)
    
    covid_start = pd.to_datetime('2020-01-01')
    covid_end = pd.to_datetime('2021-12-31')
    h3_data['covid_dummy'] = ((h3_data.index >= covid_start) & (h3_data.index <= covid_end)).astype(int)
    
    # QE episode dummy
    if 'fed_total_assets' in data.columns:
        from src.analysis import QEAnalyzer
        qe_analyzer = QEAnalyzer()
        episodes, _ = qe_analyzer.identify_qe_episodes(data['fed_total_assets'])
        h3_data['qe_episode'] = episodes > 0
    
    # Time trend
    h3_data['time_trend'] = np.arange(len(h3_data))
    
    # Monthly/quarterly dummies for seasonality
    h3_data['month'] = h3_data.index.month
    h3_data['quarter'] = h3_data.index.quarter
    
    logging.info(f"Hypothesis 3 dataset prepared: {h3_data.shape}")
    logging.info(f"Available foreign variables: {[col for col in h3_data.columns if 'foreign' in col]}")
    logging.info(f"Available FX variables: {[col for col in h3_data.columns if any(fx in col for fx in ['eur', 'gbp', 'jpy', 'dxy'])]}")
    
    return h3_data

def analyze_foreign_holdings_response(data):
    """Analyze foreign investor response to QE using TIC data"""
    logging.info("Analyzing foreign holdings response to QE...")
    
    # Find available foreign holdings variables
    foreign_vars = [col for col in data.columns if 'foreign' in col and 'holdings' in col]
    
    if not foreign_vars:
        logging.warning("No foreign holdings data available")
        return {'error': 'No foreign holdings data'}
    
    results = {}
    
    # Analyze each foreign holdings series
    for foreign_var in foreign_vars:
        if foreign_var.endswith('_change') or foreign_var.endswith('_growth'):
            # This is a flow variable
            flow_var = foreign_var
            dependent_var = data[flow_var]
        else:
            # Create flow from level
            flow_var = f'{foreign_var}_flow'
            dependent_var = data[foreign_var].diff()
        
        # Regression analysis: Foreign flows ~ QE intensity + controls
        reg_data = pd.DataFrame({
            'foreign_flow': dependent_var,
            'qe_intensity': data['us_qe_intensity'],
            'qe_intensity_lag1': data['qe_intensity_lag1'],
            'qe_change': data['qe_intensity_change'],
            'yield_level': data['us_10y'],
            'yield_change': data['yield_change'],
            'crisis': data['crisis_dummy'],
            'covid': data['covid_dummy'],
            'time_trend': data['time_trend']
        }).dropna()
        
        if len(reg_data) < 30:
            continue
        
        # Linear regression
        try:
            X_vars = ['qe_intensity_lag1', 'qe_change', 'yield_change', 'crisis', 'covid', 'time_trend']
            X = reg_data[X_vars]
            y = reg_data['foreign_flow']
            
            reg = LinearRegression().fit(X, y)
            
            # Statistical tests
            y_pred = reg.predict(X)
            residuals = y - y_pred
            
            # R-squared
            r_squared = reg.score(X, y)
            
            # T-tests for coefficients (approximate)
            n = len(y)
            k = len(X_vars)
            mse = np.sum(residuals**2) / (n - k - 1)
            
            # Coefficient standard errors (simplified)
            X_with_const = np.column_stack([np.ones(len(X)), X])
            try:
                cov_matrix = mse * np.linalg.inv(X_with_const.T @ X_with_const)
                se = np.sqrt(np.diag(cov_matrix))
                
                # QE coefficient (first regressor after constant)
                qe_coeff = reg.coef_[0]  # qe_intensity_lag1
                qe_se = se[1] if len(se) > 1 else np.nan
                qe_tstat = qe_coeff / qe_se if qe_se > 0 else np.nan
                qe_pvalue = 2 * (1 - stats.norm.cdf(abs(qe_tstat))) if not np.isnan(qe_tstat) else np.nan
                
            except:
                qe_coeff = reg.coef_[0]
                qe_se = np.nan
                qe_tstat = np.nan
                qe_pvalue = np.nan
            
            results[foreign_var] = {
                'qe_coefficient': qe_coeff,
                'qe_std_error': qe_se,
                'qe_t_statistic': qe_tstat,
                'qe_p_value': qe_pvalue,
                'qe_significant': qe_pvalue < 0.05 if not np.isnan(qe_pvalue) else False,
                'r_squared': r_squared,
                'n_observations': len(reg_data),
                'all_coefficients': dict(zip(X_vars, reg.coef_)),
                'mean_foreign_flow': y.mean(),
                'std_foreign_flow': y.std()
            }
            
            logging.info(f"{foreign_var}: QE coeff = {qe_coeff:.3f} (p={qe_pvalue:.3f})")
            
        except Exception as e:
            logging.warning(f"Regression failed for {foreign_var}: {e}")
            continue
    
    # Aggregate analysis
    if results:
        # Count significant results
        significant_negative = sum(1 for r in results.values() 
                                 if r.get('qe_significant', False) and r.get('qe_coefficient', 0) < 0)
        total_tests = len(results)
        
        results['summary'] = {
            'total_foreign_series_tested': total_tests,
            'significant_negative_effects': significant_negative,
            'proportion_negative': significant_negative / total_tests if total_tests > 0 else 0,
            'evidence_strength': 'strong' if significant_negative >= total_tests * 0.5 else 'weak'
        }
    
    return results

def analyze_exchange_rate_effects(data):
    """Analyze QE effects on exchange rates"""
    logging.info("Analyzing exchange rate effects...")
    
    # Find FX variables
    fx_vars = [col for col in data.columns if any(fx in col for fx in ['eur_usd', 'gbp_usd', 'jpy_usd']) 
               and not col.endswith('_change') and not col.endswith('_vol')]
    
    if 'dxy' in data.columns:
        fx_vars.append('dxy')
    
    if not fx_vars:
        logging.warning("No exchange rate data available")
        return {'error': 'No FX data'}
    
    results = {}
    
    for fx_var in fx_vars:
        # Use FX changes as dependent variable
        fx_change_var = f'{fx_var}_change'
        
        if fx_change_var in data.columns:
            fx_changes = data[fx_change_var]
        else:
            fx_changes = data[fx_var].pct_change() * 100
        
        # Regression: FX changes ~ QE intensity + controls
        reg_data = pd.DataFrame({
            'fx_change': fx_changes,
            'qe_intensity': data['us_qe_intensity'],
            'qe_change': data['qe_intensity_change'],
            'yield_change': data['yield_change'],
            'vix': data.get('vix', np.nan),
            'crisis': data['crisis_dummy'],
            'covid': data['covid_dummy'],
            'time_trend': data['time_trend']
        }).dropna()
        
        if len(reg_data) < 30:
            continue
        
        try:
            # Select available regressors
            X_vars = ['qe_change', 'yield_change', 'crisis', 'covid']
            if 'vix' in reg_data.columns and not reg_data['vix'].isna().all():
                X_vars.append('vix')
            
            X = reg_data[X_vars]
            y = reg_data['fx_change']
            
            reg = LinearRegression().fit(X, y)
            
            # Calculate statistics
            y_pred = reg.predict(X)
            r_squared = reg.score(X, y)
            
            # QE coefficient
            qe_coeff = reg.coef_[0]  # qe_change coefficient
            
            results[fx_var] = {
                'qe_coefficient': qe_coeff,
                'r_squared': r_squared,
                'n_observations': len(reg_data),
                'fx_volatility': y.std(),
                'mean_fx_change': y.mean()
            }
            
            # Dollar index interpretation (inverse relationship)
            if fx_var == 'dxy':
                expected_sign = 'negative'  # QE should weaken dollar (reduce DXY)
                result_interpretation = 'weakening' if qe_coeff < 0 else 'strengthening'
            else:
                expected_sign = 'positive'  # QE should weaken dollar (increase EUR/USD, etc.)
                result_interpretation = 'weakening' if qe_coeff > 0 else 'strengthening'
            
            results[fx_var]['expected_sign'] = expected_sign
            results[fx_var]['result_interpretation'] = result_interpretation
            results[fx_var]['consistent_with_theory'] = (
                (expected_sign == 'negative' and qe_coeff < 0) or 
                (expected_sign == 'positive' and qe_coeff > 0)
            )
            
            logging.info(f"{fx_var}: QE coeff = {qe_coeff:.3f}, implies dollar {result_interpretation}")
            
        except Exception as e:
            logging.warning(f"FX regression failed for {fx_var}: {e}")
            continue
    
    return results

def run_panel_var_analysis(data):
    """Run Panel VAR for international spillovers"""
    logging.info("Running Panel VAR analysis...")
    
    # Create pseudo-panel by splitting US data into regimes/periods
    # Since we don't have true cross-country panel, we'll create synthetic panel
    
    try:
        # Method 1: Split by QE intensity regimes
        qe_median = data['us_qe_intensity'].median()
        data_copy = data.copy()
        data_copy['regime'] = (data_copy['us_qe_intensity'] > qe_median).astype(int)
        
        # Method 2: Split by time periods
        n = len(data_copy)
        data_copy['period'] = pd.cut(range(n), bins=3, labels=['early', 'middle', 'late'])
        
        # Select variables for VAR
        var_variables = ['us_qe_intensity', 'us_10y']
        
        # Add available international variables
        if 'foreign_holdings_growth' in data_copy.columns:
            var_variables.append('foreign_holdings_growth')
        if 'dxy_change' in data_copy.columns:
            var_variables.append('dxy_change')
        elif 'dxy' in data_copy.columns:
            data_copy['dxy_change'] = data_copy['dxy'].pct_change() * 100
            var_variables.append('dxy_change')
        
        # Prepare panel data
        panel_data = []
        
        for regime in [0, 1]:
            regime_data = data_copy[data_copy['regime'] == regime][var_variables].dropna()
            if len(regime_data) > 50:  # Minimum observations for VAR
                regime_data['country'] = f'regime_{regime}'
                panel_data.append(regime_data)
        
        if len(panel_data) < 2:
            return {'error': 'Insufficient data for Panel VAR'}
        
        # Combine panel data
        combined_panel = pd.concat(panel_data, ignore_index=False)
        
        # Fit Panel VAR
        panel_var = PanelVAR(lags=2)
        panel_var.fit(combined_panel, country_col='country')
        
        # Calculate impulse responses
        impulse_responses = {}
        
        for country in ['regime_0', 'regime_1']:
            if country in panel_var.models and panel_var.models[country] is not None:
                # QE shock to foreign holdings (if available)
                if len(var_variables) >= 3:
                    ir = panel_var.impulse_response(country, periods=10, shock_var=0, response_var=2)
                    if ir is not None:
                        impulse_responses[country] = ir
        
        results = {
            'panel_structure': {
                'regimes': list(combined_panel['country'].unique()),
                'variables': var_variables,
                'total_observations': len(combined_panel)
            },
            'impulse_responses': impulse_responses,
            'model_success': len(impulse_responses) > 0
        }
        
        return results
        
    except Exception as e:
        logging.error(f"Panel VAR analysis failed: {e}")
        return {'error': str(e)}

def run_high_frequency_identification(data):
    """Run high-frequency identification around QE announcements"""
    logging.info("Running high-frequency identification...")
    
    # Create synthetic QE announcement dates (major QE programs)
    qe_announcements = [
        '2008-11-25',  # QE1 announcement
        '2010-11-03',  # QE2 announcement  
        '2012-09-13',  # QE3 announcement
        '2020-03-15',  # COVID QE announcement
        '2020-03-23'   # Additional COVID measures
    ]
    
    qe_dates = [pd.to_datetime(date) for date in qe_announcements if pd.to_datetime(date) in data.index]
    
    if len(qe_dates) < 2:
        return {'error': 'Insufficient QE announcement dates in data range'}
    
    try:
        # Create event windows around announcements
        event_data = []
        
        for announcement in qe_dates:
            # 5-day window around announcement
            start_date = announcement - pd.Timedelta(days=2)
            end_date = announcement + pd.Timedelta(days=2)
            
            window_data = data.loc[start_date:end_date]
            
            if len(window_data) > 2:
                # Calculate announcement day surprise
                if announcement in window_data.index:
                    qe_surprise = window_data.loc[announcement, 'qe_intensity_change']
                    yield_change = window_data.loc[announcement, 'yield_change']
                    
                    # Foreign holdings response (if available)
                    foreign_response = np.nan
                    if 'foreign_holdings_change' in window_data.columns:
                        # Use next day response
                        next_day = announcement + pd.Timedelta(days=1)
                        if next_day in window_data.index:
                            foreign_response = window_data.loc[next_day, 'foreign_holdings_change']
                    
                    # FX response
                    fx_response = np.nan
                    if 'dxy_change' in window_data.columns:
                        fx_response = window_data.loc[announcement, 'dxy_change']
                    
                    event_data.append({
                        'date': announcement,
                        'qe_surprise': qe_surprise,
                        'yield_response': yield_change,
                        'foreign_response': foreign_response,
                        'fx_response': fx_response
                    })
        
        if len(event_data) < 2:
            return {'error': 'Insufficient event data'}
        
        event_df = pd.DataFrame(event_data).dropna(subset=['qe_surprise', 'yield_response'])
        
        # High-frequency regressions
        results = {}
        
        # Yield response to QE surprise
        if len(event_df) >= 2:
            X = event_df[['qe_surprise']].values
            y = event_df['yield_response'].values
            
            reg = LinearRegression().fit(X, y)
            results['yield_response'] = {
                'coefficient': reg.coef_[0],
                'r_squared': reg.score(X, y),
                'n_events': len(event_df)
            }
        
        # Foreign holdings response (if available)
        foreign_data = event_df.dropna(subset=['foreign_response'])
        if len(foreign_data) >= 2:
            X = foreign_data[['qe_surprise']].values
            y = foreign_data['foreign_response'].values
            
            reg = LinearRegression().fit(X, y)
            results['foreign_response'] = {
                'coefficient': reg.coef_[0],
                'r_squared': reg.score(X, y),
                'n_events': len(foreign_data)
            }
        
        # FX response (if available)
        fx_data = event_df.dropna(subset=['fx_response'])
        if len(fx_data) >= 2:
            X = fx_data[['qe_surprise']].values
            y = fx_data['fx_response'].values
            
            reg = LinearRegression().fit(X, y)
            results['fx_response'] = {
                'coefficient': reg.coef_[0],
                'r_squared': reg.score(X, y),
                'n_events': len(fx_data)
            }
        
        results['event_data'] = event_df.to_dict('records')
        results['qe_announcement_dates'] = [d.strftime('%Y-%m-%d') for d in qe_dates]
        
        return results
        
    except Exception as e:
        logging.error(f"High-frequency identification failed: {e}")
        return {'error': str(e)}

def analyze_official_vs_private_flows(data):
    """Analyze differences between official and private foreign investor behavior"""
    logging.info("Analyzing official vs private foreign flows...")
    
    # Look for official vs private breakdown
    china_var = 'china_treasury_holdings' if 'china_treasury_holdings' in data.columns else None
    japan_var = 'japan_treasury_holdings' if 'japan_treasury_holdings' in data.columns else None
    total_var = 'foreign_treasury_holdings' if 'foreign_treasury_holdings' in data.columns else None
    
    if not any([china_var, japan_var, total_var]):
        return {'error': 'No foreign holdings breakdown available'}
    
    results = {}
    
    # Analyze major official holders (China, Japan)
    for country, var in [('China', china_var), ('Japan', japan_var)]:
        if var is None:
            continue
            
        # Calculate flows
        flows = data[var].diff()
        
        # Regression analysis
        reg_data = pd.DataFrame({
            'flows': flows,
            'qe_intensity': data['us_qe_intensity'],
            'qe_change': data['qe_intensity_change'],
            'yield_change': data['yield_change'],
            'crisis': data['crisis_dummy'],
            'covid': data['covid_dummy']
        }).dropna()
        
        if len(reg_data) < 20:
            continue
            
        try:
            X = reg_data[['qe_change', 'yield_change', 'crisis', 'covid']]
            y = reg_data['flows']
            
            reg = LinearRegression().fit(X, y)
            
            results[country] = {
                'qe_coefficient': reg.coef_[0],
                'r_squared': reg.score(X, y),
                'mean_flows': y.mean(),
                'flow_volatility': y.std(),
                'n_observations': len(reg_data)
            }
            
        except Exception:
            continue
    
    # Calculate private flows (if total and major official holders available)
    if total_var and china_var and japan_var:
        try:
            official_flows = data[china_var].diff() + data[japan_var].diff()
            total_flows = data[total_var].diff()
            private_flows = total_flows - official_flows
            
            # Regression for private flows
            reg_data = pd.DataFrame({
                'private_flows': private_flows,
                'qe_change': data['qe_intensity_change'],
                'yield_change': data['yield_change'],
                'crisis': data['crisis_dummy'],
                'covid': data['covid_dummy']
            }).dropna()
            
            if len(reg_data) >= 20:
                X = reg_data[['qe_change', 'yield_change', 'crisis', 'covid']]
                y = reg_data['private_flows']
                
                reg = LinearRegression().fit(X, y)
                
                results['Private'] = {
                    'qe_coefficient': reg.coef_[0],
                    'r_squared': reg.score(X, y),
                    'mean_flows': y.mean(),
                    'flow_volatility': y.std(),
                    'n_observations': len(reg_data)
                }
                
                # Compare official vs private responses
                if 'China' in results and 'Japan' in results:
                    avg_official_response = np.mean([results['China']['qe_coefficient'], 
                                                   results['Japan']['qe_coefficient']])
                    private_response = results['Private']['qe_coefficient']
                    
                    results['comparison'] = {
                        'average_official_response': avg_official_response,
                        'private_response': private_response,
                        'difference': private_response - avg_official_response,
                        'private_more_sensitive': abs(private_response) > abs(avg_official_response)
                    }
        except Exception as e:
            logging.warning(f"Private flows calculation failed: {e}")
    
    return results

def create_hypothesis3_visualizations(data, results):
    """Create visualizations for Hypothesis 3 results"""
    logging.info("Creating Hypothesis 3 visualizations...")
    
    plt.style.use('default')
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Hypothesis 3: International Spillover Effects', fontsize=16, fontweight='bold')
    
    # Plot 1: Foreign holdings vs QE intensity
    ax1 = axes[0, 0]
    if 'foreign_treasury_holdings' in data.columns:
        # Normalize for better visualization
        qe_norm = (data['us_qe_intensity'] - data['us_qe_intensity'].min()) / (data['us_qe_intensity'].max() - data['us_qe_intensity'].min())
        foreign_norm = (data['foreign_treasury_holdings'] - data['foreign_treasury_holdings'].min()) / (data['foreign_treasury_holdings'].max() - data['foreign_treasury_holdings'].min())
        
        ax1.plot(data.index, qe_norm, label='QE Intensity (normalized)', alpha=0.7)
        ax1.plot(data.index, foreign_norm, label='Foreign Holdings (normalized)', alpha=0.7)
        ax1.set_ylabel('Normalized Values')
        ax1.set_title('QE Intensity vs Foreign Treasury Holdings')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
    
    # Plot 2: Exchange rate effects
    ax2 = axes[0, 1]
    if 'fx_effects' in results and results['fx_effects']:
        fx_results = results['fx_effects']
        
        # Plot QE coefficients for different currencies
        currencies = []
        coefficients = []
        interpretations = []
        
        for fx_var, fx_data in fx_results.items():
            if isinstance(fx_data, dict) and 'qe_coefficient' in fx_data:
                currencies.append(fx_var.upper().replace('_', '/'))
                coefficients.append(fx_data['qe_coefficient'])
                interpretations.append(fx_data.get('result_interpretation', 'unknown'))
        
        if currencies:
            colors = ['green' if interp == 'weakening' else 'red' for interp in interpretations]
            bars = ax2.bar(currencies, coefficients, color=colors, alpha=0.7)
            ax2.axhline(y=0, color='black', linestyle='-', alpha=0.5)
            ax2.set_ylabel('QE Coefficient')
            ax2.set_title('QE Effects on Exchange Rates')
            ax2.tick_params(axis='x', rotation=45)
            
            # Add interpretation legend
            from matplotlib.patches import Patch
            legend_elements = [Patch(facecolor='green', alpha=0.7, label='Dollar Weakening'),
                             Patch(facecolor='red', alpha=0.7, label='Dollar Strengthening')]
            ax2.legend(handles=legend_elements)
    
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Foreign flow responses by country/type
    ax3 = axes[1, 0]
    if 'official_vs_private' in results and results['official_vs_private']:
        flow_results = results['official_vs_private']
        
        countries = []
        responses = []
        
        for country, country_data in flow_results.items():
            if isinstance(country_data, dict) and 'qe_coefficient' in country_data:
                countries.append(country)
                responses.append(country_data['qe_coefficient'])
        
        if countries:
            colors = ['blue' if country in ['China', 'Japan'] else 'orange' for country in countries]
            bars = ax3.bar(countries, responses, color=colors, alpha=0.7)
            ax3.axhline(y=0, color='black', linestyle='-', alpha=0.5)
            ax3.set_ylabel('QE Response Coefficient')
            ax3.set_title('Foreign Flow Responses by Investor Type')
            
            # Add legend
            from matplotlib.patches import Patch
            legend_elements = [Patch(facecolor='blue', alpha=0.7, label='Official Investors'),
                             Patch(facecolor='orange', alpha=0.7, label='Private Investors')]
            ax3.legend(handles=legend_elements)
    
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: High-frequency event responses
    ax4 = axes[1, 1]
    if 'high_frequency' in results and 'event_data' in results['high_frequency']:
        event_data = pd.DataFrame(results['high_frequency']['event_data'])
        
        if not event_data.empty and 'qe_surprise' in event_data.columns:
            # Plot yield response to QE surprises
            if 'yield_response' in event_data.columns:
                ax4.scatter(event_data['qe_surprise'], event_data['yield_response'], 
                           alpha=0.7, s=100, label='Yield Response')
                
                # Add trend line
                if len(event_data) > 1:
                    z = np.polyfit(event_data['qe_surprise'], event_data['yield_response'], 1)
                    p = np.poly1d(z)
                    x_trend = np.linspace(event_data['qe_surprise'].min(), event_data['qe_surprise'].max(), 100)
                    ax4.plot(x_trend, p(x_trend), "r--", alpha=0.8)
            
            ax4.set_xlabel('QE Surprise')
            ax4.set_ylabel('Market Response')
            ax4.set_title('High-Frequency Response to QE Announcements')
            ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    os.makedirs('results', exist_ok=True)
    plt.savefig('results/hypothesis3_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()

def test_hypothesis_3(data):
    """Main function to test Hypothesis 3"""
    logging.info("="*50)
    logging.info("TESTING HYPOTHESIS 3: INTERNATIONAL SPILLOVERS")
    logging.info("="*50)
    
    results = {}
    
    try:
        # Prepare data
        h3_data = prepare_hypothesis3_data(data)
        results['data_summary'] = {
            'n_observations': len(h3_data),
            'date_range': f"{h3_data.index.min()} to {h3_data.index.max()}",
            'foreign_vars_available': [col for col in h3_data.columns if 'foreign' in col],
            'fx_vars_available': [col for col in h3_data.columns if any(fx in col for fx in ['eur', 'gbp', 'jpy', 'dxy'])]
        }
        
        # Analyze foreign holdings response
        foreign_results = analyze_foreign_holdings_response(h3_data)
        results['foreign_flow_results'] = foreign_results
        
        # Analyze exchange rate effects
        fx_results = analyze_exchange_rate_effects(h3_data)
        results['fx_effects'] = fx_results
        
        # Run Panel VAR analysis
        panel_var_results = run_panel_var_analysis(h3_data)
        results['panel_var'] = panel_var_results
        
        # High-frequency identification
        hf_results = run_high_frequency_identification(h3_data)
        results['high_frequency'] = hf_results
        
        # Official vs private analysis
        official_private_results = analyze_official_vs_private_flows(h3_data)
        results['official_vs_private'] = official_private_results
        
        # Create visualizations
        create_hypothesis3_visualizations(h3_data, results)
        
        # Summary conclusion
        hypothesis_supported = False
        evidence_count = 0
        
        # Check foreign holdings evidence
        if 'foreign_flow_results' in results and 'summary' in results['foreign_flow_results']:
            summary = results['foreign_flow_results']['summary']
            if summary.get('evidence_strength') == 'strong':
                hypothesis_supported = True
                evidence_count += 1
        
        # Check FX evidence
        if 'fx_effects' in results and isinstance(results['fx_effects'], dict):
            fx_consistent = sum(1 for fx_data in results['fx_effects'].values() 
                               if isinstance(fx_data, dict) and fx_data.get('consistent_with_theory', False))
            total_fx = len([fx_data for fx_data in results['fx_effects'].values() if isinstance(fx_data, dict)])
            
            if total_fx > 0 and fx_consistent / total_fx >= 0.5:
                evidence_count += 1
        
        # Check high-frequency evidence
        if 'high_frequency' in results and 'foreign_response' in results['high_frequency']:
            foreign_response = results['high_frequency']['foreign_response']
            if foreign_response.get('coefficient', 0) < 0:  # Negative response as expected
                evidence_count += 1
        
        if evidence_count >= 2:
            hypothesis_supported = True
        
        results['hypothesis_supported'] = hypothesis_supported
        results['conclusion'] = {
            'supported': hypothesis_supported,
            'evidence_count': evidence_count,
            'foreign_evidence': 'foreign_flow_results' in results and 'error' not in results['foreign_flow_results'],
            'fx_evidence': 'fx_effects' in results and 'error' not in results['fx_effects'],
            'high_freq_evidence': 'high_frequency' in results and 'error' not in results['high_frequency'],
            'evidence_strength': 'strong' if evidence_count >= 2 else 'moderate' if evidence_count == 1 else 'weak'
        }
        
        logging.info(f"Hypothesis 3 supported: {hypothesis_supported}")
        logging.info(f"Evidence count: {evidence_count}")
        
        return results
        
    except Exception as e:
        logging.error(f"Hypothesis 3 testing failed: {e}")
        return {'error': str(e)}

if __name__ == "__main__":
    # For standalone testing
    logging.basicConfig(level=logging.INFO)
    
    # Load test data
    try:
        test_data = pd.read_csv('data/processed/combined_panel.csv', index_col=0, parse_dates=True)
        results = test_hypothesis_3(test_data)
        
        print("\nHypothesis 3 Test Results:")
        print("-" * 30)
        if 'error' not in results:
            print(f"Hypothesis supported: {results['hypothesis_supported']}")
            print(f"Evidence strength: {results['conclusion']['evidence_strength']}")
            
            if 'foreign_flow_results' in results and 'summary' in results['foreign_flow_results']:
                summary = results['foreign_flow_results']['summary']
                print(f"Foreign flows evidence: {summary['evidence_strength']}")
                
        else:
            print(f"Error: {results['error']}")
            
    except Exception as e:
        print(f"Standalone test failed: {e}")
        print("Run main.py first to generate processed data")