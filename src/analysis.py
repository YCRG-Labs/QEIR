import pandas as pd
import numpy as np
from scipy import stats
from scipy.optimize import minimize_scalar
import statsmodels.api as sm
from statsmodels.stats.diagnostic import acorr_breusch_godfrey
from statsmodels.stats.stattools import durbin_watson
from statsmodels.tsa.stattools import adfuller, kpss
import warnings
warnings.filterwarnings('ignore')

class DataProcessor:
    """Core data processing functions for QE analysis"""
    
    @staticmethod
    def calculate_qe_intensity(cb_holdings, total_outstanding):
        """Calculate QE intensity as CB holdings / total outstanding securities"""
        intensity = cb_holdings / total_outstanding
        return np.clip(intensity, 0, 1.0)  # Cap at 100%
    
    @staticmethod
    def calculate_dcr(interest_payments, gdp):
        """Calculate debt service coverage ratio"""
        dcr = interest_payments / gdp
        return np.clip(dcr, 0, 0.3)  # Cap at 30% of GDP
    
    @staticmethod
    def calculate_term_premium(long_yield, short_yield):
        """Calculate term premium (long - short yield) in basis points"""
        return (long_yield - short_yield) * 100
    
    @staticmethod
    def winsorize(series, lower=0.01, upper=0.99):
        """Winsorize series at specified percentiles"""
        lower_val = series.quantile(lower)
        upper_val = series.quantile(upper)
        return np.clip(series, lower_val, upper_val)
    
    @staticmethod
    def align_frequencies(data_dict, target_freq='D'):
        """Align multiple series to target frequency"""
        aligned = {}
        
        for name, series in data_dict.items():
            if isinstance(series, pd.Series) and not series.empty:
                series.index = pd.to_datetime(series.index)
                
                if target_freq == 'D':
                    aligned[name] = series.resample('D').ffill()
                elif target_freq == 'W':
                    aligned[name] = series.resample('W').last()
                elif target_freq == 'M':
                    aligned[name] = series.resample('M').last()
                else:
                    aligned[name] = series
        
        return pd.DataFrame(aligned) if aligned else pd.DataFrame()

class QEAnalyzer:
    """Main analysis functions for QE research"""
    
    @staticmethod
    def identify_qe_episodes(fed_assets, threshold_pct=5):
        """Identify QE episodes based on Fed balance sheet expansion"""
        asset_growth = fed_assets.pct_change(periods=90) * 100  # 90-day growth rate
        qe_episodes = asset_growth > threshold_pct
        
        # Create episode labels
        episodes = pd.Series(0, index=fed_assets.index)
        episode_num = 1
        in_episode = False
        
        for date, is_qe in qe_episodes.items():
            if is_qe and not in_episode:
                in_episode = True
                episode_start = date
            elif not is_qe and in_episode:
                in_episode = False
                episodes.loc[episode_start:date] = episode_num
                episode_num += 1
                
        return episodes, qe_episodes
    
    @staticmethod
    def detect_threshold_candidates(qe_intensity, n_candidates=20):
        """Detect candidate threshold values for regime switching"""
        clean_intensity = qe_intensity.dropna()
        return np.linspace(clean_intensity.quantile(0.1), 
                          clean_intensity.quantile(0.9), 
                          n_candidates)
    
    @staticmethod
    def calculate_rolling_volatility(returns, window=20):
        """Calculate rolling volatility (annualized)"""
        return returns.rolling(window=window).std() * np.sqrt(252)
    
    @staticmethod
    def construct_investment_proxy(business_inv, equipment_inv):
        """Construct long-term investment proxy"""
        # Use business investment as primary, equipment as secondary
        investment = business_inv.fillna(equipment_inv)
        return investment.rolling(window=4).mean()  # Quarterly smoothing

class EventStudyAnalyzer:
    """Event study analysis for QE announcements"""
    
    def __init__(self, event_window=(-10, 20)):
        self.event_window = event_window
        self.results = {}
        
    def run_event_study(self, returns, event_dates, market_returns=None):
        """
        Run event study analysis
        returns: asset returns series
        event_dates: list of event dates
        market_returns: market benchmark returns (optional)
        """
        event_effects = []
        
        for event_date in event_dates:
            try:
                event_date = pd.to_datetime(event_date)
                
                # Define event window
                start_date = event_date + pd.Timedelta(days=self.event_window[0])
                end_date = event_date + pd.Timedelta(days=self.event_window[1])
                
                # Extract returns in event window
                event_returns = returns.loc[start_date:end_date]
                
                if len(event_returns) > 5:  # Minimum observations
                    # Calculate cumulative abnormal returns
                    if market_returns is not None:
                        market_event = market_returns.loc[start_date:end_date]
                        abnormal_returns = event_returns - market_event
                    else:
                        abnormal_returns = event_returns
                        
                    car = abnormal_returns.cumsum()
                    
                    event_effects.append({
                        'event_date': event_date,
                        'car_total': car.iloc[-1],
                        'car_announcement': car.iloc[abs(self.event_window[0]):abs(self.event_window[0])+3].sum(),
                        'max_effect': car.max(),
                        'min_effect': car.min()
                    })
                    
            except Exception:
                continue
                
        return pd.DataFrame(event_effects)
    
    def test_significance(self, event_effects, alpha=0.05):
        """Test statistical significance of event effects"""
        if event_effects.empty:
            return {}
            
        # T-test for cumulative abnormal returns
        car_stats = stats.ttest_1samp(event_effects['car_total'], 0)
        announcement_stats = stats.ttest_1samp(event_effects['car_announcement'], 0)
        
        return {
            'car_mean': event_effects['car_total'].mean(),
            'car_tstat': car_stats.statistic,
            'car_pvalue': car_stats.pvalue,
            'car_significant': car_stats.pvalue < alpha,
            'announcement_mean': event_effects['car_announcement'].mean(),
            'announcement_tstat': announcement_stats.statistic,
            'announcement_pvalue': announcement_stats.pvalue,
            'announcement_significant': announcement_stats.pvalue < alpha
        }

class ForeignFlowAnalyzer:
    """Analyze international capital flows and foreign investor behavior"""
    
    @staticmethod
    def calculate_flow_changes(foreign_holdings):
        """Calculate monthly changes in foreign holdings"""
        return foreign_holdings.diff()
    
    @staticmethod
    def decompose_flows(total_flows, official_flows):
        """Decompose total flows into official vs private components"""
        private_flows = total_flows - official_flows
        
        return {
            'total_flows': total_flows,
            'official_flows': official_flows,
            'private_flows': private_flows,
            'official_share': official_flows / total_flows
        }
    
    @staticmethod
    def calculate_flow_volatility(flows, window=12):
        """Calculate rolling volatility of capital flows"""
        return flows.rolling(window=window).std()
    
    @staticmethod
    def identify_flow_reversals(flows, threshold_std=2):
        """Identify significant flow reversals"""
        flow_z_scores = (flows - flows.rolling(12).mean()) / flows.rolling(12).std()
        return np.abs(flow_z_scores) > threshold_std

class MarketMicrostructureAnalyzer:
    """Analyze market microstructure and liquidity effects"""
    
    @staticmethod
    def calculate_bid_ask_proxy(high, low, close):
        """Calculate bid-ask spread proxy from OHLC data"""
        return (high - low) / close
    
    @staticmethod
    def calculate_amihud_illiquidity(returns, dollar_volume):
        """Calculate Amihud illiquidity measure"""
        abs_returns = np.abs(returns)
        illiquidity = abs_returns / (dollar_volume / 1e6)  # Scale by millions
        return illiquidity.replace([np.inf, -np.inf], np.nan)
    
    @staticmethod
    def calculate_price_impact(returns, volume, window=20):
        """Calculate price impact measure"""
        vol_scaled_returns = returns / np.log(volume + 1)
        return vol_scaled_returns.rolling(window=window).std()
    
    @staticmethod
    def liquidity_beta(asset_liquidity, market_liquidity, window=90):
        """Calculate liquidity beta"""
        cov_matrix = pd.concat([asset_liquidity, market_liquidity], axis=1).rolling(window=window).cov()
        market_var = market_liquidity.rolling(window=window).var()
        
        # Extract covariance and calculate beta
        beta_series = []
        for i in range(len(asset_liquidity)):
            if i >= window:
                cov_val = cov_matrix.iloc[i*2+1, 0]  # Covariance element
                var_val = market_var.iloc[i]
                beta = cov_val / var_val if var_val != 0 else np.nan
                beta_series.append(beta)
            else:
                beta_series.append(np.nan)
                
        return pd.Series(beta_series, index=asset_liquidity.index)

class StatisticalTests:
    """Statistical testing functions"""
    
    @staticmethod
    def stationarity_test(series, test='adf'):
        """Test for stationarity"""
        clean_series = series.dropna()
        
        if test == 'adf':
            stat, pvalue, _, _, _, _ = adfuller(clean_series, autolag='AIC')
            return {'test': 'ADF', 'statistic': stat, 'p_value': pvalue, 'stationary': pvalue < 0.05}
        elif test == 'kpss':
            stat, pvalue, _, _ = kpss(clean_series, regression='c')
            return {'test': 'KPSS', 'statistic': stat, 'p_value': pvalue, 'stationary': pvalue > 0.05}
    
    @staticmethod
    def cointegration_test(y, x):
        """Engle-Granger cointegration test"""
        # First stage regression
        X_reg = sm.add_constant(x)
        model = sm.OLS(y, X_reg).fit()
        residuals = model.resid
        
        # Test residuals for stationarity
        adf_result = StatisticalTests.stationarity_test(residuals, test='adf')
        
        return {
            'first_stage_r2': model.rsquared,
            'residual_adf_stat': adf_result['statistic'],
            'residual_adf_pvalue': adf_result['p_value'],
            'cointegrated': adf_result['stationary']
        }
    
    @staticmethod
    def granger_causality_test(y, x, maxlag=4):
        """Granger causality test"""
        # Create lagged variables
        data = pd.concat([y, x], axis=1).dropna()
        data.columns = ['y', 'x']
        
        # Restricted model (y on its own lags)
        y_lags = pd.concat([data['y'].shift(i) for i in range(1, maxlag+1)], axis=1)
        y_lags.columns = [f'y_lag_{i}' for i in range(1, maxlag+1)]
        
        restricted_data = pd.concat([data['y'], y_lags], axis=1).dropna()
        X_restricted = sm.add_constant(restricted_data.iloc[:, 1:])
        y_restricted = restricted_data.iloc[:, 0]
        
        model_restricted = sm.OLS(y_restricted, X_restricted).fit()
        
        # Unrestricted model (y on its own lags + x lags)
        x_lags = pd.concat([data['x'].shift(i) for i in range(1, maxlag+1)], axis=1)
        x_lags.columns = [f'x_lag_{i}' for i in range(1, maxlag+1)]
        
        unrestricted_data = pd.concat([data['y'], y_lags, x_lags], axis=1).dropna()
        X_unrestricted = sm.add_constant(unrestricted_data.iloc[:, 1:])
        y_unrestricted = unrestricted_data.iloc[:, 0]
        
        model_unrestricted = sm.OLS(y_unrestricted, X_unrestricted).fit()
        
        # F-test for restriction
        f_stat = ((model_restricted.ssr - model_unrestricted.ssr) / maxlag) / \
                (model_unrestricted.ssr / model_unrestricted.df_resid)
        
        p_value = 1 - stats.f.cdf(f_stat, maxlag, model_unrestricted.df_resid)
        
        return {
            'f_statistic': f_stat,
            'p_value': p_value,
            'granger_causes': p_value < 0.05,
            'restricted_r2': model_restricted.rsquared,
            'unrestricted_r2': model_unrestricted.rsquared
        }

class DiagnosticTests:
    """Regression diagnostic tests"""
    
    @staticmethod
    def autocorrelation_test(residuals, lags=4):
        """Breusch-Godfrey test for autocorrelation"""
        try:
            lm_stat, lm_pvalue, f_stat, f_pvalue = acorr_breusch_godfrey(residuals, nlags=lags)
            dw_stat = durbin_watson(residuals)
            
            return {
                'lm_statistic': lm_stat,
                'lm_p_value': lm_pvalue,
                'f_statistic': f_stat,
                'f_p_value': f_pvalue,
                'durbin_watson': dw_stat,
                'autocorrelation_present': lm_pvalue < 0.05
            }
        except:
            return {'error': 'Autocorrelation test failed'}
    
    @staticmethod
    def heteroskedasticity_test(residuals, fitted_values):
        """Breusch-Pagan test for heteroskedasticity"""
        # Regress squared residuals on fitted values
        residuals_sq = residuals ** 2
        X = sm.add_constant(fitted_values)
        
        try:
            model = sm.OLS(residuals_sq, X).fit()
            n = len(residuals)
            lm_stat = n * model.rsquared
            p_value = 1 - stats.chi2.cdf(lm_stat, df=1)
            
            return {
                'lm_statistic': lm_stat,
                'p_value': p_value,
                'heteroskedasticity_present': p_value < 0.05
            }
        except:
            return {'error': 'Heteroskedasticity test failed'}
    
    @staticmethod
    def normality_test(residuals):
        """Jarque-Bera test for normality"""
        jb_stat, jb_pvalue = stats.jarque_bera(residuals.dropna())
        sw_stat, sw_pvalue = stats.shapiro(residuals.dropna()[:5000])  # Shapiro limited to 5000 obs
        
        return {
            'jarque_bera_stat': jb_stat,
            'jarque_bera_p_value': jb_pvalue,
            'shapiro_stat': sw_stat,
            'shapiro_p_value': sw_pvalue,
            'normal_distribution': jb_pvalue > 0.05 and sw_pvalue > 0.05
        }

class RobustnessTests:
    """Robustness testing functions"""
    
    @staticmethod
    def bootstrap_confidence_intervals(data, statistic_func, n_bootstrap=1000, alpha=0.05):
        """Bootstrap confidence intervals for any statistic"""
        n = len(data)
        bootstrap_stats = []
        
        for _ in range(n_bootstrap):
            sample = np.random.choice(data, size=n, replace=True)
            stat = statistic_func(sample)
            bootstrap_stats.append(stat)
            
        bootstrap_stats = np.array(bootstrap_stats)
        lower = np.percentile(bootstrap_stats, 100 * alpha / 2)
        upper = np.percentile(bootstrap_stats, 100 * (1 - alpha / 2))
        
        return {
            'mean': np.mean(bootstrap_stats),
            'std': np.std(bootstrap_stats),
            'lower_ci': lower,
            'upper_ci': upper,
            'bootstrap_distribution': bootstrap_stats
        }
    
    @staticmethod
    def jackknife_estimation(data, estimator_func):
        """Jackknife estimation for bias correction"""
        n = len(data)
        jackknife_estimates = []
        
        for i in range(n):
            # Remove observation i
            jackknife_sample = np.concatenate([data[:i], data[i+1:]])
            estimate = estimator_func(jackknife_sample)
            jackknife_estimates.append(estimate)
            
        jackknife_estimates = np.array(jackknife_estimates)
        
        # Original estimate
        original_estimate = estimator_func(data)
        
        # Bias-corrected estimate
        bias = (n - 1) * (np.mean(jackknife_estimates) - original_estimate)
        bias_corrected = original_estimate - bias
        
        # Standard error
        se = np.sqrt((n - 1) / n * np.sum((jackknife_estimates - np.mean(jackknife_estimates))**2))
        
        return {
            'original_estimate': original_estimate,
            'bias_corrected_estimate': bias_corrected,
            'bias': bias,
            'standard_error': se,
            'jackknife_estimates': jackknife_estimates
        }
    
    @staticmethod
    def subsample_stability(data, model_func, subsample_fractions=[0.7, 0.8, 0.9]):
        """Test model stability across subsamples"""
        n = len(data)
        results = {}
        
        for frac in subsample_fractions:
            subsample_size = int(n * frac)
            subsample_results = []
            
            # Multiple random subsamples
            for _ in range(50):
                indices = np.random.choice(n, size=subsample_size, replace=False)
                subsample = data.iloc[indices] if hasattr(data, 'iloc') else data[indices]
                
                try:
                    result = model_func(subsample)
                    subsample_results.append(result)
                except:
                    continue
                    
            if subsample_results:
                results[frac] = {
                    'mean': np.mean(subsample_results),
                    'std': np.std(subsample_results),
                    'min': np.min(subsample_results),
                    'max': np.max(subsample_results),
                    'estimates': subsample_results
                }
                
        return results

class DataValidation:
    """Data quality validation functions"""
    
    @staticmethod
    def check_data_quality(df, min_obs=100, max_missing_pct=50):
        """Comprehensive data quality check"""
        report = {
            'total_observations': len(df),
            'total_variables': len(df.columns),
            'date_range': (df.index.min(), df.index.max()) if hasattr(df.index, 'min') else (None, None),
            'missing_data_pct': (df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100,
            'variables_with_sufficient_data': [],
            'variables_with_excessive_missing': [],
            'potential_outliers': {},
            'data_quality_score': 100
        }
        
        # Check individual variables
        for col in df.columns:
            missing_pct = (df[col].isnull().sum() / len(df)) * 100
            
            if missing_pct <= max_missing_pct and df[col].count() >= min_obs:
                report['variables_with_sufficient_data'].append(col)
            else:
                report['variables_with_excessive_missing'].append(col)
                
            # Check for outliers (values beyond 3 standard deviations)
            if df[col].dtype in ['float64', 'int64']:
                z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
                outlier_count = (z_scores > 3).sum()
                if outlier_count > 0:
                    report['potential_outliers'][col] = outlier_count
                    
        # Calculate overall quality score
        if report['missing_data_pct'] > 20:
            report['data_quality_score'] -= 30
        if len(report['variables_with_excessive_missing']) > len(df.columns) * 0.3:
            report['data_quality_score'] -= 25
        if report['total_observations'] < min_obs * 2:
            report['data_quality_score'] -= 20
            
        return report
    
    @staticmethod
    def validate_economic_relationships(data):
        """Validate economic relationships in the data"""
        validations = {}
        
        # Check if yields are reasonable (0-20%)
        yield_cols = [col for col in data.columns if 'yield' in col.lower() or col.endswith('y')]
        for col in yield_cols:
            if col in data.columns:
                valid_range = (data[col] >= 0) & (data[col] <= 20)
                validations[f'{col}_in_reasonable_range'] = valid_range.mean()
                
        # Check if QE intensity is between 0 and 1
        if 'us_qe_intensity' in data.columns:
            qe_valid = (data['us_qe_intensity'] >= 0) & (data['us_qe_intensity'] <= 1)
            validations['qe_intensity_valid_range'] = qe_valid.mean()
            
        # Check term structure (long yields > short yields on average)
        if 'us_10y' in data.columns and 'us_2y' in data.columns:
            term_structure_normal = data['us_10y'] > data['us_2y']
            validations['normal_term_structure_pct'] = term_structure_normal.mean()
            
        return validations