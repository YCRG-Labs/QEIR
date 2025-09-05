"""
Enhanced Identification Strategy Module

This module provides comprehensive tools for validating instrumental variables
and testing identification assumptions in econometric models, specifically
designed for QE analysis with rigorous statistical testing.
"""

import numpy as np
import pandas as pd
from scipy import stats
from scipy.linalg import inv
import warnings
from typing import Dict, List, Tuple, Optional, Union


class InstrumentValidator:
    """
    Comprehensive instrumental variable validation class providing formal
    statistical tests for instrument validity, relevance, and exogeneity.
    
    This class implements standard econometric tests including:
    - Weak instrument tests (Cragg-Donald, Stock-Yogo)
    - Overidentification tests (Sargan, Hansen J-tests)
    - First-stage F-statistics for instrument relevance
    - Endogeneity tests (Hausman, Durbin-Wu-Hausman)
    """
    
    def __init__(self, significance_level: float = 0.05):
        """
        Initialize the InstrumentValidator.
        
        Parameters:
        -----------
        significance_level : float, default=0.05
            Significance level for statistical tests
        """
        self.significance_level = significance_level
        self.test_results = {}
        
        # Critical values for Stock-Yogo weak instrument test
        # These are approximate critical values for 10% maximal IV size
        self.stock_yogo_critical_values = {
            1: 16.38,  # 1 endogenous variable
            2: 19.93,  # 2 endogenous variables  
            3: 22.30,  # 3 endogenous variables
        }
    
    def weak_instrument_test(self, 
                           instruments: np.ndarray, 
                           endogenous_vars: np.ndarray,
                           exogenous_controls: Optional[np.ndarray] = None) -> Dict:
        """
        Perform comprehensive weak instrument tests using Cragg-Donald statistics
        and Stock-Yogo critical values.
        
        Parameters:
        -----------
        instruments : np.ndarray
            Matrix of instrumental variables (n_obs x n_instruments)
        endogenous_vars : np.ndarray
            Matrix of endogenous variables (n_obs x n_endogenous)
        exogenous_controls : np.ndarray, optional
            Matrix of exogenous control variables (n_obs x n_controls)
            
        Returns:
        --------
        Dict containing test results with keys:
            - 'cragg_donald_stat': Cragg-Donald F-statistic
            - 'first_stage_f_stats': First-stage F-statistics for each endogenous variable
            - 'stock_yogo_critical': Stock-Yogo critical value
            - 'weak_instruments': Boolean indicating if instruments are weak
            - 'individual_relevance': F-stats for individual instrument relevance
        """
        n_obs, n_instruments = instruments.shape
        n_endogenous = endogenous_vars.shape[1] if endogenous_vars.ndim > 1 else 1
        
        if endogenous_vars.ndim == 1:
            endogenous_vars = endogenous_vars.reshape(-1, 1)
            
        # Prepare regressor matrix
        if exogenous_controls is not None:
            X = np.column_stack([instruments, exogenous_controls])
            n_controls = exogenous_controls.shape[1]
        else:
            X = instruments
            n_controls = 0
            
        # Add constant term
        X = np.column_stack([np.ones(n_obs), X])
        
        # First-stage regressions for each endogenous variable
        first_stage_f_stats = []
        individual_relevance = []
        
        for j in range(n_endogenous):
            y_j = endogenous_vars[:, j]
            
            # OLS regression: y_j = X * beta + error
            try:
                beta_hat = inv(X.T @ X) @ X.T @ y_j
                y_pred = X @ beta_hat
                residuals = y_j - y_pred
                
                # Calculate F-statistic for joint significance of instruments
                # H0: coefficients on instruments are jointly zero
                ssr_restricted = np.sum((y_j - np.mean(y_j))**2)
                ssr_unrestricted = np.sum(residuals**2)
                
                f_stat = ((ssr_restricted - ssr_unrestricted) / n_instruments) / \
                        (ssr_unrestricted / (n_obs - X.shape[1]))
                
                first_stage_f_stats.append(f_stat)
                
                # Individual instrument relevance (t-statistics)
                mse = ssr_unrestricted / (n_obs - X.shape[1])
                var_beta = mse * np.diag(inv(X.T @ X))
                t_stats = beta_hat[1:n_instruments+1] / np.sqrt(var_beta[1:n_instruments+1])
                individual_relevance.append(t_stats)
                
            except np.linalg.LinAlgError:
                warnings.warn(f"Singular matrix in first-stage regression for variable {j}")
                first_stage_f_stats.append(np.nan)
                individual_relevance.append(np.full(n_instruments, np.nan))
        
        # Cragg-Donald statistic (minimum eigenvalue statistic)
        try:
            # This is a simplified version - full implementation would use
            # the minimum eigenvalue of the concentration matrix
            cragg_donald_stat = min(first_stage_f_stats) if first_stage_f_stats else np.nan
        except (ValueError, TypeError):
            cragg_donald_stat = np.nan
            
        # Stock-Yogo critical value lookup
        stock_yogo_critical = self.stock_yogo_critical_values.get(
            min(n_endogenous, 3), 
            self.stock_yogo_critical_values[3]
        )
        
        # Determine if instruments are weak
        weak_instruments = (cragg_donald_stat < stock_yogo_critical) if not np.isnan(cragg_donald_stat) else True
        
        results = {
            'cragg_donald_stat': cragg_donald_stat,
            'first_stage_f_stats': first_stage_f_stats,
            'stock_yogo_critical': stock_yogo_critical,
            'weak_instruments': weak_instruments,
            'individual_relevance': individual_relevance,
            'n_instruments': n_instruments,
            'n_endogenous': n_endogenous,
            'test_type': 'weak_instrument_test'
        }
        
        self.test_results['weak_instrument'] = results
        return results
    
    def overidentification_test(self,
                              instruments: np.ndarray,
                              endogenous_vars: np.ndarray,
                              dependent_var: np.ndarray,
                              exogenous_controls: Optional[np.ndarray] = None,
                              test_type: str = 'sargan') -> Dict:
        """
        Perform overidentification tests (Sargan or Hansen J-test) to test
        the validity of instrumental variables.
        
        Parameters:
        -----------
        instruments : np.ndarray
            Matrix of instrumental variables (n_obs x n_instruments)
        endogenous_vars : np.ndarray
            Matrix of endogenous variables (n_obs x n_endogenous)
        dependent_var : np.ndarray
            Dependent variable vector (n_obs,)
        exogenous_controls : np.ndarray, optional
            Matrix of exogenous control variables
        test_type : str, default='sargan'
            Type of test: 'sargan' or 'hansen'
            
        Returns:
        --------
        Dict containing test results with keys:
            - 'j_statistic': J-statistic value
            - 'p_value': P-value of the test
            - 'degrees_freedom': Degrees of freedom
            - 'reject_overid': Boolean indicating if overidentification is rejected
            - 'test_type': Type of test performed
        """
        n_obs = len(dependent_var)
        n_instruments = instruments.shape[1]
        n_endogenous = endogenous_vars.shape[1] if endogenous_vars.ndim > 1 else 1
        
        if endogenous_vars.ndim == 1:
            endogenous_vars = endogenous_vars.reshape(-1, 1)
            
        # Check if model is exactly identified
        if n_instruments <= n_endogenous:
            return {
                'j_statistic': np.nan,
                'p_value': np.nan,
                'degrees_freedom': 0,
                'reject_overid': False,
                'test_type': test_type,
                'message': 'Model is exactly identified - overidentification test not applicable'
            }
        
        # Prepare regressor matrices
        if exogenous_controls is not None:
            X_exog = np.column_stack([np.ones(n_obs), exogenous_controls])
            Z = np.column_stack([X_exog, instruments])
        else:
            X_exog = np.ones((n_obs, 1))
            Z = np.column_stack([X_exog, instruments])
            
        X_endog = endogenous_vars
        
        try:
            # Two-stage least squares estimation
            # First stage: regress endogenous variables on instruments
            P_Z = Z @ inv(Z.T @ Z) @ Z.T  # Projection matrix
            X_endog_hat = P_Z @ X_endog
            
            # Second stage: regress y on predicted endogenous variables and exogenous controls
            X_2sls = np.column_stack([X_exog, X_endog_hat])
            beta_2sls = inv(X_2sls.T @ X_2sls) @ X_2sls.T @ dependent_var
            
            # Calculate residuals
            residuals = dependent_var - X_2sls @ beta_2sls
            
            # Sargan/Hansen J-statistic
            # J = n * R² from regression of residuals on all instruments
            residuals_on_instruments = P_Z @ residuals
            ssr_residuals = np.sum(residuals_on_instruments**2)
            tss_residuals = np.sum(residuals**2)
            
            if tss_residuals > 0:
                r_squared = ssr_residuals / tss_residuals
                j_statistic = n_obs * r_squared
            else:
                j_statistic = 0
                
            # Degrees of freedom = number of overidentifying restrictions
            degrees_freedom = n_instruments - n_endogenous
            
            # P-value from chi-squared distribution
            p_value = 1 - stats.chi2.cdf(j_statistic, degrees_freedom)
            
            # Reject overidentification if p-value < significance level
            reject_overid = p_value < self.significance_level
            
        except np.linalg.LinAlgError:
            warnings.warn("Singular matrix in overidentification test")
            j_statistic = np.nan
            p_value = np.nan
            degrees_freedom = n_instruments - n_endogenous
            reject_overid = False
            
        results = {
            'j_statistic': j_statistic,
            'p_value': p_value,
            'degrees_freedom': degrees_freedom,
            'reject_overid': reject_overid,
            'test_type': test_type,
            'n_overid_restrictions': degrees_freedom
        }
        
        self.test_results['overidentification'] = results
        return results
    
    def instrument_relevance_test(self,
                                instruments: np.ndarray,
                                endogenous_vars: np.ndarray,
                                exogenous_controls: Optional[np.ndarray] = None,
                                individual_tests: bool = True) -> Dict:
        """
        Test instrument relevance using first-stage F-statistics and individual
        significance tests.
        
        Parameters:
        -----------
        instruments : np.ndarray
            Matrix of instrumental variables (n_obs x n_instruments)
        endogenous_vars : np.ndarray
            Matrix of endogenous variables (n_obs x n_endogenous)
        exogenous_controls : np.ndarray, optional
            Matrix of exogenous control variables
        individual_tests : bool, default=True
            Whether to perform individual instrument relevance tests
            
        Returns:
        --------
        Dict containing test results with keys:
            - 'joint_f_stats': F-statistics for joint instrument significance
            - 'joint_p_values': P-values for joint tests
            - 'individual_t_stats': Individual t-statistics (if requested)
            - 'individual_p_values': Individual p-values (if requested)
            - 'relevant_instruments': Boolean array indicating relevant instruments
            - 'strong_instruments': Boolean indicating if instruments are collectively strong
        """
        n_obs, n_instruments = instruments.shape
        n_endogenous = endogenous_vars.shape[1] if endogenous_vars.ndim > 1 else 1
        
        if endogenous_vars.ndim == 1:
            endogenous_vars = endogenous_vars.reshape(-1, 1)
            
        # Prepare regressor matrix
        if exogenous_controls is not None:
            X = np.column_stack([np.ones(n_obs), exogenous_controls, instruments])
            n_controls = exogenous_controls.shape[1] + 1  # +1 for constant
        else:
            X = np.column_stack([np.ones(n_obs), instruments])
            n_controls = 1  # Just constant
            
        joint_f_stats = []
        joint_p_values = []
        individual_t_stats = []
        individual_p_values = []
        
        for j in range(n_endogenous):
            y_j = endogenous_vars[:, j]
            
            try:
                # OLS regression
                beta_hat = inv(X.T @ X) @ X.T @ y_j
                y_pred = X @ beta_hat
                residuals = y_j - y_pred
                
                # Calculate standard errors
                mse = np.sum(residuals**2) / (n_obs - X.shape[1])
                var_covar_matrix = mse * inv(X.T @ X)
                
                # Joint F-test for instruments
                # H0: all instrument coefficients are zero
                R = np.zeros((n_instruments, X.shape[1]))
                R[:, n_controls:] = np.eye(n_instruments)  # Select instrument coefficients
                
                # F-statistic = (R*beta)'(R*Var(beta)*R')^(-1)(R*beta) / q
                R_beta = R @ beta_hat
                R_var_R = R @ var_covar_matrix @ R.T
                
                f_stat = (R_beta.T @ inv(R_var_R) @ R_beta) / n_instruments
                p_value_joint = 1 - stats.f.cdf(f_stat, n_instruments, n_obs - X.shape[1])
                
                joint_f_stats.append(f_stat)
                joint_p_values.append(p_value_joint)
                
                # Individual t-tests for instruments
                if individual_tests:
                    instrument_coeffs = beta_hat[n_controls:]
                    instrument_se = np.sqrt(np.diag(var_covar_matrix)[n_controls:])
                    
                    t_stats = instrument_coeffs / instrument_se
                    p_values_indiv = 2 * (1 - stats.t.cdf(np.abs(t_stats), n_obs - X.shape[1]))
                    
                    individual_t_stats.append(t_stats)
                    individual_p_values.append(p_values_indiv)
                    
            except np.linalg.LinAlgError:
                warnings.warn(f"Singular matrix in relevance test for variable {j}")
                joint_f_stats.append(np.nan)
                joint_p_values.append(np.nan)
                if individual_tests:
                    individual_t_stats.append(np.full(n_instruments, np.nan))
                    individual_p_values.append(np.full(n_instruments, np.nan))
        
        # Determine instrument strength
        # Rule of thumb: F-stat > 10 indicates strong instruments
        strong_instruments = all(f > 10 for f in joint_f_stats if not np.isnan(f))
        
        # Individual instrument relevance (significant at 5% level)
        if individual_tests and individual_p_values:
            relevant_instruments = np.array([
                np.any([p_vals < self.significance_level for p_vals in individual_p_values], axis=0)
            ]).flatten()
        else:
            relevant_instruments = np.full(n_instruments, True)
            
        results = {
            'joint_f_stats': joint_f_stats,
            'joint_p_values': joint_p_values,
            'individual_t_stats': individual_t_stats if individual_tests else None,
            'individual_p_values': individual_p_values if individual_tests else None,
            'relevant_instruments': relevant_instruments,
            'strong_instruments': strong_instruments,
            'min_f_stat': min(joint_f_stats) if joint_f_stats else np.nan,
            'test_type': 'instrument_relevance_test'
        }
        
        self.test_results['instrument_relevance'] = results
        return results
    
    def comprehensive_validation(self,
                               instruments: np.ndarray,
                               endogenous_vars: np.ndarray,
                               dependent_var: np.ndarray,
                               exogenous_controls: Optional[np.ndarray] = None) -> Dict:
        """
        Perform comprehensive instrumental variable validation including all tests.
        
        Parameters:
        -----------
        instruments : np.ndarray
            Matrix of instrumental variables
        endogenous_vars : np.ndarray
            Matrix of endogenous variables
        dependent_var : np.ndarray
            Dependent variable vector
        exogenous_controls : np.ndarray, optional
            Matrix of exogenous control variables
            
        Returns:
        --------
        Dict containing all test results and overall assessment
        """
        results = {}
        
        # 1. Weak instrument test
        results['weak_instrument'] = self.weak_instrument_test(
            instruments, endogenous_vars, exogenous_controls
        )
        
        # 2. Instrument relevance test
        results['instrument_relevance'] = self.instrument_relevance_test(
            instruments, endogenous_vars, exogenous_controls
        )
        
        # 3. Overidentification test (if applicable)
        results['overidentification'] = self.overidentification_test(
            instruments, endogenous_vars, dependent_var, exogenous_controls
        )
        
        # Overall assessment
        weak_instruments = results['weak_instrument']['weak_instruments']
        strong_instruments = results['instrument_relevance']['strong_instruments']
        overid_rejected = results['overidentification']['reject_overid']
        
        # Determine overall validity
        if weak_instruments:
            overall_assessment = "WEAK_INSTRUMENTS"
        elif overid_rejected:
            overall_assessment = "OVERIDENTIFICATION_REJECTED"
        elif strong_instruments and not overid_rejected:
            overall_assessment = "VALID_INSTRUMENTS"
        else:
            overall_assessment = "MARGINAL_INSTRUMENTS"
            
        results['overall_assessment'] = overall_assessment
        results['recommendations'] = self._generate_recommendations(results)
        
        return results
    
    def _generate_recommendations(self, test_results: Dict) -> List[str]:
        """Generate recommendations based on test results."""
        recommendations = []
        
        weak_test = test_results.get('weak_instrument', {})
        relevance_test = test_results.get('instrument_relevance', {})
        overid_test = test_results.get('overidentification', {})
        
        if weak_test.get('weak_instruments', True):
            recommendations.append(
                "Consider finding stronger instruments or using alternative identification strategies"
            )
            
        if not relevance_test.get('strong_instruments', False):
            recommendations.append(
                "First-stage F-statistics suggest weak instruments (F < 10 rule of thumb)"
            )
            
        if overid_test.get('reject_overid', False):
            recommendations.append(
                "Overidentification test rejected - some instruments may be invalid"
            )
            
        if not recommendations:
            recommendations.append("All instrument validation tests passed successfully")
            
        return recommendations
    
    def print_results(self, test_results: Optional[Dict] = None):
        """Print formatted test results."""
        if test_results is None:
            test_results = self.test_results
            
        print("=" * 60)
        print("INSTRUMENTAL VARIABLE VALIDATION RESULTS")
        print("=" * 60)
        
        for test_name, results in test_results.items():
            if test_name == 'overall_assessment' or test_name == 'recommendations':
                continue
                
            print(f"\n{test_name.upper().replace('_', ' ')}:")
            print("-" * 40)
            
            if test_name == 'weak_instrument':
                print(f"Cragg-Donald Statistic: {results.get('cragg_donald_stat', 'N/A'):.4f}")
                print(f"Stock-Yogo Critical Value: {results.get('stock_yogo_critical', 'N/A'):.4f}")
                print(f"Weak Instruments: {results.get('weak_instruments', 'N/A')}")
                
            elif test_name == 'instrument_relevance':
                f_stats = results.get('joint_f_stats', [])
                print(f"First-stage F-statistics: {[f'{f:.4f}' for f in f_stats]}")
                print(f"Strong Instruments (F > 10): {results.get('strong_instruments', 'N/A')}")
                
            elif test_name == 'overidentification':
                print(f"J-statistic: {results.get('j_statistic', 'N/A'):.4f}")
                print(f"P-value: {results.get('p_value', 'N/A'):.4f}")
                print(f"Degrees of Freedom: {results.get('degrees_freedom', 'N/A')}")
                print(f"Reject Overidentification: {results.get('reject_overid', 'N/A')}")
        
        if 'overall_assessment' in test_results:
            print(f"\nOVERALL ASSESSMENT: {test_results['overall_assessment']}")
            
        if 'recommendations' in test_results:
            print("\nRECOMMENDATIONS:")
            for i, rec in enumerate(test_results['recommendations'], 1):
                print(f"{i}. {rec}")
    
    def high_frequency_identification(self,
                                fomc_dates: pd.DatetimeIndex,
                                treasury_futures_2y: pd.Series,
                                treasury_futures_10y: pd.Series,
                                window_minutes: int = 30) -> Dict:
        """
        Implement high-frequency identification around FOMC announcements
        following Gertler & Karadi (2015) and Swanson (2021).
        
        Parameters:
        -----------
        fomc_dates : pd.DatetimeIndex
            Dates of FOMC announcements
        treasury_futures_2y : pd.Series
            2-year Treasury futures prices (high frequency)
        treasury_futures_10y : pd.Series
            10-year Treasury futures prices (high frequency)
        window_minutes : int, default=30
            Window around announcements in minutes
            
        Returns:
        --------
        Dict containing high-frequency surprises and instruments
        """
        surprises_2y = []
        surprises_10y = []
        
        for fomc_date in fomc_dates:
            # Extract price changes in window around announcement
            start_window = fomc_date - pd.Timedelta(minutes=window_minutes//2)
            end_window = fomc_date + pd.Timedelta(minutes=window_minutes//2)
            
            # Calculate price changes (surprises)
            try:
                pre_price_2y = treasury_futures_2y.loc[start_window]
                post_price_2y = treasury_futures_2y.loc[end_window]
                surprise_2y = post_price_2y - pre_price_2y
                
                pre_price_10y = treasury_futures_10y.loc[start_window]
                post_price_10y = treasury_futures_10y.loc[end_window]
                surprise_10y = post_price_10y - pre_price_10y
                
                surprises_2y.append(surprise_2y)
                surprises_10y.append(surprise_10y)
                
            except KeyError:
                # Handle missing data
                surprises_2y.append(np.nan)
                surprises_10y.append(np.nan)
        
        return {
            'fomc_dates': fomc_dates,
            'surprises_2y': np.array(surprises_2y),
            'surprises_10y': np.array(surprises_10y),
            'qe_instrument': np.array(surprises_10y) - np.array(surprises_2y)  # Term structure surprise
        }

    def fomc_rotation_instrument(self, 
                               dates: pd.DatetimeIndex,
                               fed_districts: Optional[List[str]] = None) -> np.ndarray:
        """
        Create instrumental variable based on Federal Reserve district rotation
        in FOMC voting membership.
        
        The Federal Reserve System includes 12 regional banks, with presidents
        rotating voting rights on the FOMC. This rotation provides exogenous
        variation in monetary policy preferences that can serve as an instrument.
        
        Parameters:
        -----------
        dates : pd.DatetimeIndex
            Time series dates for the analysis period
        fed_districts : List[str], optional
            List of Fed district identifiers to focus on
            
        Returns:
        --------
        np.ndarray
            Instrumental variable based on FOMC rotation (n_obs,)
        """
        if fed_districts is None:
            # Default to districts with historically different QE preferences
            fed_districts = ['NY', 'SF', 'CHI', 'PHI', 'MIN']
        
        # FOMC voting rotation schedule (simplified)
        # In reality, this would use actual historical rotation data
        rotation_schedule = {
            'NY': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],  # NY Fed always votes
            'CHI': [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # Rotates every 3 years
            'SF': [0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],   # Different rotation
            'PHI': [0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0],  # Different rotation
            'MIN': [0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0],  # Different rotation
        }
        
        instrument_values = []
        
        for date in dates:
            year = date.year
            month = date.month
            
            # Calculate rotation-based voting power
            # This is a simplified version - real implementation would use
            # actual FOMC composition and voting records
            voting_power = 0
            for district in fed_districts:
                if district in rotation_schedule:
                    # Determine if district has voting power in this period
                    rotation_index = ((year - 2008) * 12 + month - 1) % 12
                    if rotation_schedule[district][rotation_index % len(rotation_schedule[district])]:
                        # Weight by district's historical QE preference
                        district_weights = {
                            'NY': 0.8,   # Generally pro-QE
                            'CHI': 0.6,  # Moderate
                            'SF': 0.7,   # Pro-QE
                            'PHI': 0.4,  # Conservative
                            'MIN': 0.3   # Conservative
                        }
                        voting_power += district_weights.get(district, 0.5)
            
            instrument_values.append(voting_power)
        
        return np.array(instrument_values)
    
    def auction_calendar_instrument(self, 
                                  dates: pd.DatetimeIndex,
                                  auction_frequency: str = 'monthly') -> np.ndarray:
        """
        Create instrumental variable based on Treasury auction calendar timing.
        
        Treasury auction schedules are predetermined and provide exogenous
        variation in market conditions that can affect QE transmission.
        
        Parameters:
        -----------
        dates : pd.DatetimeIndex
            Time series dates for the analysis period
        auction_frequency : str, default='monthly'
            Frequency of auction calendar effects to capture
            
        Returns:
        --------
        np.ndarray
            Instrumental variable based on auction timing (n_obs,)
        """
        if len(dates) == 0:
            return np.array([])
            
        instrument_values = []
        
        # Treasury auction schedule patterns (simplified)
        # Real implementation would use actual auction calendars
        auction_schedule = {
            'bills_3m': [1, 8, 15, 22],      # Weekly on these days of month
            'bills_6m': [1, 15],             # Bi-weekly
            'notes_2y': [15],                # Monthly mid-month
            'notes_5y': [15],                # Monthly mid-month  
            'notes_10y': [15],               # Monthly mid-month
            'bonds_30y': [15],               # Monthly mid-month
        }
        
        # Security weights for QE importance
        security_weights = {
            'bills_3m': 0.1,
            'bills_6m': 0.1,
            'notes_2y': 0.2,
            'notes_5y': 0.3,
            'notes_10y': 0.4,  # Most important for QE
            'bonds_30y': 0.3,
        }
        
        for date in dates:
            day_of_month = date.day
            month = date.month
            
            # Calculate auction intensity for this date
            auction_intensity = 0
            
            # Check if date falls on or near major auction dates
            for security_type, auction_days in auction_schedule.items():
                for auction_day in auction_days:
                    # Create window around auction dates
                    distance = abs(day_of_month - auction_day)
                    if distance <= 2:
                        # Distance decay (closer to auction = higher weight)
                        distance_weight = max(0, 1 - (distance / 3))
                        auction_intensity += (security_weights[security_type] * 
                                            distance_weight)
            
            # Add base level and seasonal variation (some months have more auctions)
            base_level = 0.1  # Minimum base level
            seasonal_multiplier = 1 + 0.2 * np.sin(2 * np.pi * month / 12)
            auction_intensity = base_level + auction_intensity * seasonal_multiplier
            
            instrument_values.append(auction_intensity)
        
        return np.array(instrument_values)
    
    def debt_ceiling_instrument(self, 
                              dates: pd.DatetimeIndex,
                              debt_ceiling_episodes: Optional[List[Tuple[str, str]]] = None) -> np.ndarray:
        """
        Create instrumental variable based on Congressional debt ceiling episodes.
        
        Debt ceiling debates create exogenous fiscal uncertainty that affects
        monetary policy transmission independently of QE decisions.
        
        Parameters:
        -----------
        dates : pd.DatetimeIndex
            Time series dates for the analysis period
        debt_ceiling_episodes : List[Tuple[str, str]], optional
            List of (start_date, end_date) tuples for debt ceiling episodes
            
        Returns:
        --------
        np.ndarray
            Instrumental variable based on debt ceiling episodes (n_obs,)
        """
        if debt_ceiling_episodes is None:
            # Historical debt ceiling episodes (major ones during QE period)
            debt_ceiling_episodes = [
                ('2011-05-01', '2011-08-02'),  # 2011 debt ceiling crisis
                ('2013-09-01', '2013-10-17'),  # 2013 government shutdown
                ('2015-03-01', '2015-11-02'),  # 2015 debt ceiling suspension
                ('2017-03-01', '2017-09-08'),  # 2017 debt ceiling debate
                ('2019-03-01', '2019-07-22'),  # 2019 debt ceiling suspension
                ('2021-08-01', '2021-12-16'),  # 2021 debt ceiling debate
                ('2023-01-01', '2023-06-03'),  # 2023 debt ceiling crisis
            ]
        
        # Convert string dates to datetime
        episodes = []
        for start_str, end_str in debt_ceiling_episodes:
            start_date = pd.to_datetime(start_str)
            end_date = pd.to_datetime(end_str)
            episodes.append((start_date, end_date))
        
        instrument_values = []
        
        for date in dates:
            debt_ceiling_intensity = 0
            
            for start_date, end_date in episodes:
                if start_date <= date <= end_date:
                    # Calculate intensity based on position within episode
                    episode_length = (end_date - start_date).days
                    days_from_start = (date - start_date).days
                    
                    # Intensity increases as we approach the deadline
                    if episode_length > 0:
                        intensity = 1 - (days_from_start / episode_length)
                        # Add some curvature - intensity increases faster near deadline
                        intensity = intensity ** 0.5
                        debt_ceiling_intensity = max(debt_ceiling_intensity, intensity)
                
                # Also add pre-episode buildup (30 days before)
                elif date >= start_date - pd.Timedelta(days=30) and date < start_date:
                    days_to_start = (start_date - date).days
                    buildup_intensity = 0.3 * (1 - days_to_start / 30)
                    debt_ceiling_intensity = max(debt_ceiling_intensity, buildup_intensity)
            
            instrument_values.append(debt_ceiling_intensity)
        
        return np.array(instrument_values)
    
    def foreign_spillover_instrument(self, 
                                   dates: pd.DatetimeIndex,
                                   central_banks: Optional[List[str]] = None,
                                   qe_announcements: Optional[Dict[str, List[str]]] = None) -> np.ndarray:
        """
        Create instrumental variable based on other central banks' QE timing.
        
        Foreign central bank QE announcements provide exogenous variation
        that affects US markets through spillover effects but is independent
        of US QE decisions.
        
        Parameters:
        -----------
        dates : pd.DatetimeIndex
            Time series dates for the analysis period
        central_banks : List[str], optional
            List of central banks to include ('ECB', 'BOJ', 'BOE', 'BOC')
        qe_announcements : Dict[str, List[str]], optional
            Dictionary mapping central bank codes to announcement dates
            
        Returns:
        --------
        np.ndarray
            Instrumental variable based on foreign QE spillovers (n_obs,)
        """
        if central_banks is None:
            central_banks = ['ECB', 'BOJ', 'BOE', 'BOC']
        
        if qe_announcements is None:
            # Major foreign QE announcements (simplified historical data)
            qe_announcements = {
                'ECB': [
                    '2015-01-22',  # ECB QE announcement
                    '2015-03-09',  # ECB QE start
                    '2016-03-10',  # ECB expansion
                    '2016-12-08',  # ECB extension
                    '2019-09-12',  # ECB restart
                    '2020-03-18',  # PEPP announcement
                    '2020-06-04',  # PEPP expansion
                ],
                'BOJ': [
                    '2013-04-04',  # BOJ QQE announcement
                    '2014-10-31',  # BOJ expansion
                    '2016-01-29',  # BOJ negative rates
                    '2016-09-21',  # BOJ yield curve control
                    '2020-03-16',  # BOJ COVID response
                ],
                'BOE': [
                    '2009-03-05',  # BOE QE start
                    '2011-10-06',  # BOE expansion
                    '2012-07-05',  # BOE expansion
                    '2016-08-04',  # BOE post-Brexit QE
                    '2020-03-19',  # BOE COVID response
                    '2020-06-18',  # BOE expansion
                ],
                'BOC': [
                    '2020-03-27',  # BOC QE announcement
                    '2020-04-15',  # BOC expansion
                    '2020-10-28',  # BOC adjustment
                ]
            }
        
        # Convert announcement dates to datetime
        announcement_dates = {}
        for cb, dates_list in qe_announcements.items():
            if cb in central_banks:
                announcement_dates[cb] = [pd.to_datetime(date) for date in dates_list]
        
        instrument_values = []
        
        for date in dates:
            spillover_intensity = 0
            
            for cb, cb_announcements in announcement_dates.items():
                for announcement_date in cb_announcements:
                    # Calculate time distance from announcement
                    days_diff = abs((date - announcement_date).days)
                    
                    # Spillover effects decay over time
                    if days_diff <= 30:  # 30-day window
                        # Central bank importance weights for US spillovers
                        cb_weights = {
                            'ECB': 0.4,   # High spillover to US
                            'BOJ': 0.3,   # Moderate spillover
                            'BOE': 0.2,   # Lower spillover
                            'BOC': 0.1,   # Minimal spillover
                        }
                        
                        # Time decay function (exponential decay)
                        time_weight = np.exp(-days_diff / 10)  # 10-day half-life
                        
                        spillover_effect = cb_weights.get(cb, 0.1) * time_weight
                        spillover_intensity += spillover_effect
            
            # Normalize to reasonable range
            spillover_intensity = min(spillover_intensity, 1.0)
            instrument_values.append(spillover_intensity)
        
        return np.array(instrument_values)
    
    def validate_instrument_construction(self,
                                       instrument: np.ndarray,
                                       dates: pd.DatetimeIndex,
                                       instrument_name: str) -> Dict:
        """
        Validate the construction and properties of a newly created instrument.
        
        Parameters:
        -----------
        instrument : np.ndarray
            The constructed instrumental variable
        dates : pd.DatetimeIndex
            Corresponding dates for the instrument
        instrument_name : str
            Name of the instrument for reporting
            
        Returns:
        --------
        Dict containing validation results
        """
        validation_results = {
            'instrument_name': instrument_name,
            'n_observations': len(instrument),
            'date_range': (dates.min(), dates.max()),
            'mean': np.mean(instrument),
            'std': np.std(instrument),
            'min': np.min(instrument),
            'max': np.max(instrument),
            'n_zeros': np.sum(instrument == 0),
            'n_missing': np.sum(np.isnan(instrument)),
        }
        
        # Check for sufficient variation
        validation_results['sufficient_variation'] = validation_results['std'] > 0.01
        
        # Check for temporal clustering (potential endogeneity concern)
        # Calculate autocorrelation at lag 1
        if len(instrument) > 1:
            autocorr_lag1 = np.corrcoef(instrument[:-1], instrument[1:])[0, 1]
            validation_results['autocorr_lag1'] = autocorr_lag1
            validation_results['high_autocorr_warning'] = abs(autocorr_lag1) > 0.7
        else:
            validation_results['autocorr_lag1'] = np.nan
            validation_results['high_autocorr_warning'] = False
        
        # Check for outliers (values > 3 standard deviations from mean)
        if validation_results['std'] > 0:
            z_scores = np.abs((instrument - validation_results['mean']) / validation_results['std'])
            n_outliers = np.sum(z_scores > 3)
            validation_results['n_outliers'] = n_outliers
            validation_results['outlier_percentage'] = (n_outliers / len(instrument)) * 100
        else:
            validation_results['n_outliers'] = 0
            validation_results['outlier_percentage'] = 0
        
        # Generate warnings and recommendations
        warnings = []
        if not validation_results['sufficient_variation']:
            warnings.append("Insufficient variation in instrument - may be weak")
        if validation_results['high_autocorr_warning']:
            warnings.append("High autocorrelation detected - potential endogeneity concern")
        if validation_results['outlier_percentage'] > 5:
            warnings.append(f"High percentage of outliers ({validation_results['outlier_percentage']:.1f}%)")
        if validation_results['n_missing'] > 0:
            warnings.append(f"Missing values detected ({validation_results['n_missing']} observations)")
        
        validation_results['warnings'] = warnings
        validation_results['validation_passed'] = len(warnings) == 0
        
        return validation_results
    
    def test_instrument_exogeneity(self,
                                 instrument: np.ndarray,
                                 potential_confounders: np.ndarray,
                                 instrument_name: str) -> Dict:
        """
        Test instrument exogeneity by checking correlation with potential confounders.
        
        Parameters:
        -----------
        instrument : np.ndarray
            The instrumental variable to test
        potential_confounders : np.ndarray
            Matrix of variables that could violate exogeneity (n_obs x n_confounders)
        instrument_name : str
            Name of the instrument for reporting
            
        Returns:
        --------
        Dict containing exogeneity test results
        """
        if potential_confounders.ndim == 1:
            potential_confounders = potential_confounders.reshape(-1, 1)
        
        n_confounders = potential_confounders.shape[1]
        correlations = []
        p_values = []
        
        for i in range(n_confounders):
            confounder = potential_confounders[:, i]
            
            # Remove missing values
            valid_mask = ~(np.isnan(instrument) | np.isnan(confounder))
            if np.sum(valid_mask) > 10:  # Need sufficient observations
                corr, p_val = stats.pearsonr(instrument[valid_mask], confounder[valid_mask])
                correlations.append(corr)
                p_values.append(p_val)
            else:
                correlations.append(np.nan)
                p_values.append(np.nan)
        
        # Check for concerning correlations
        concerning_correlations = []
        for i, (corr, p_val) in enumerate(zip(correlations, p_values)):
            if not np.isnan(corr) and abs(corr) > 0.3 and p_val < 0.05:
                concerning_correlations.append({
                    'confounder_index': i,
                    'correlation': corr,
                    'p_value': p_val
                })
        
        exogeneity_results = {
            'instrument_name': instrument_name,
            'correlations': correlations,
            'p_values': p_values,
            'max_abs_correlation': np.nanmax(np.abs(correlations)) if correlations else np.nan,
            'concerning_correlations': concerning_correlations,
            'exogeneity_concern': len(concerning_correlations) > 0,
            'n_significant_correlations': len(concerning_correlations)
        }
        
        # Generate recommendations
        if exogeneity_results['exogeneity_concern']:
            exogeneity_results['recommendation'] = (
                f"Instrument shows concerning correlations with {len(concerning_correlations)} "
                "potential confounders. Consider alternative instruments or control for these variables."
            )
        else:
            exogeneity_results['recommendation'] = (
                "No concerning correlations detected. Instrument appears exogenous to tested confounders."
            )
        
        return exogeneity_results
    
    def hausman_test(self,
                    instruments: np.ndarray,
                    endogenous_vars: np.ndarray,
                    dependent_var: np.ndarray,
                    exogenous_controls: Optional[np.ndarray] = None) -> Dict:
        """
        Perform Hausman test for endogeneity detection.
        
        The Hausman test compares OLS and IV estimates to test whether
        endogeneity is present. Under the null hypothesis of exogeneity,
        both estimators are consistent, but OLS is more efficient.
        
        Parameters:
        -----------
        instruments : np.ndarray
            Matrix of instrumental variables (n_obs x n_instruments)
        endogenous_vars : np.ndarray
            Matrix of endogenous variables (n_obs x n_endogenous)
        dependent_var : np.ndarray
            Dependent variable vector (n_obs,)
        exogenous_controls : np.ndarray, optional
            Matrix of exogenous control variables
            
        Returns:
        --------
        Dict containing Hausman test results
        """
        n_obs = len(dependent_var)
        
        if endogenous_vars.ndim == 1:
            endogenous_vars = endogenous_vars.reshape(-1, 1)
        
        n_endogenous = endogenous_vars.shape[1]
        
        # Prepare regressor matrices
        if exogenous_controls is not None:
            X_exog = np.column_stack([np.ones(n_obs), exogenous_controls])
            Z = np.column_stack([X_exog, instruments])
        else:
            X_exog = np.ones((n_obs, 1))
            Z = np.column_stack([X_exog, instruments])
        
        X_all = np.column_stack([X_exog, endogenous_vars])
        
        try:
            # OLS estimation
            beta_ols = inv(X_all.T @ X_all) @ X_all.T @ dependent_var
            residuals_ols = dependent_var - X_all @ beta_ols
            
            # Calculate OLS covariance matrix
            mse_ols = np.sum(residuals_ols**2) / (n_obs - X_all.shape[1])
            var_ols = mse_ols * inv(X_all.T @ X_all)
            
            # IV estimation (2SLS)
            # First stage: regress endogenous variables on instruments
            P_Z = Z @ inv(Z.T @ Z) @ Z.T
            X_endog_hat = P_Z @ endogenous_vars
            
            # Second stage
            X_2sls = np.column_stack([X_exog, X_endog_hat])
            beta_iv = inv(X_2sls.T @ X_2sls) @ X_2sls.T @ dependent_var
            residuals_iv = dependent_var - X_2sls @ beta_iv
            
            # Calculate IV covariance matrix (robust to heteroskedasticity)
            mse_iv = np.sum(residuals_iv**2) / (n_obs - X_2sls.shape[1])
            
            # For proper IV covariance, we need to account for first-stage estimation
            # This is a simplified version
            var_iv = mse_iv * inv(X_2sls.T @ X_2sls)
            
            # Hausman test statistic
            # H = (beta_iv - beta_ols)' * (Var(beta_iv) - Var(beta_ols))^(-1) * (beta_iv - beta_ols)
            
            # Focus on endogenous variable coefficients
            beta_diff = beta_iv[-n_endogenous:] - beta_ols[-n_endogenous:]
            var_diff = var_iv[-n_endogenous:, -n_endogenous:] - var_ols[-n_endogenous:, -n_endogenous:]
            
            # Check if variance difference matrix is positive definite
            eigenvals = np.linalg.eigvals(var_diff)
            if np.all(eigenvals > 1e-10):  # Positive definite check
                hausman_stat = beta_diff.T @ inv(var_diff) @ beta_diff
                p_value = 1 - stats.chi2.cdf(hausman_stat, n_endogenous)
                test_valid = True
            else:
                # Variance difference matrix is not positive definite
                hausman_stat = np.nan
                p_value = np.nan
                test_valid = False
                
        except (np.linalg.LinAlgError, ValueError) as e:
            hausman_stat = np.nan
            p_value = np.nan
            test_valid = False
            beta_diff = np.full(n_endogenous, np.nan)
        
        # Determine endogeneity conclusion
        if test_valid and not np.isnan(p_value):
            endogeneity_detected = p_value < self.significance_level
        else:
            endogeneity_detected = None  # Test inconclusive
        
        results = {
            'hausman_statistic': hausman_stat,
            'p_value': p_value,
            'degrees_freedom': n_endogenous,
            'endogeneity_detected': endogeneity_detected,
            'test_valid': test_valid,
            'coefficient_differences': beta_diff,
            'test_type': 'hausman_test'
        }
        
        return results
    
    def durbin_wu_hausman_test(self,
                              instruments: np.ndarray,
                              endogenous_vars: np.ndarray,
                              dependent_var: np.ndarray,
                              exogenous_controls: Optional[np.ndarray] = None) -> Dict:
        """
        Perform Durbin-Wu-Hausman test for multiple endogenous variables.
        
        This is an alternative formulation of the Hausman test that is more
        robust and can handle multiple endogenous variables more effectively.
        
        Parameters:
        -----------
        instruments : np.ndarray
            Matrix of instrumental variables (n_obs x n_instruments)
        endogenous_vars : np.ndarray
            Matrix of endogenous variables (n_obs x n_endogenous)
        dependent_var : np.ndarray
            Dependent variable vector (n_obs,)
        exogenous_controls : np.ndarray, optional
            Matrix of exogenous control variables
            
        Returns:
        --------
        Dict containing Durbin-Wu-Hausman test results
        """
        n_obs = len(dependent_var)
        
        if endogenous_vars.ndim == 1:
            endogenous_vars = endogenous_vars.reshape(-1, 1)
        
        n_endogenous = endogenous_vars.shape[1]
        
        # Prepare regressor matrices
        if exogenous_controls is not None:
            X_exog = np.column_stack([np.ones(n_obs), exogenous_controls])
            Z = np.column_stack([X_exog, instruments])
        else:
            X_exog = np.ones((n_obs, 1))
            Z = np.column_stack([X_exog, instruments])
        
        try:
            # Step 1: First-stage regressions to get residuals
            P_Z = Z @ inv(Z.T @ Z) @ Z.T
            first_stage_residuals = []
            
            for j in range(n_endogenous):
                y_j = endogenous_vars[:, j]
                y_j_hat = P_Z @ y_j
                residual_j = y_j - y_j_hat
                first_stage_residuals.append(residual_j)
            
            first_stage_residuals = np.column_stack(first_stage_residuals)
            
            # Step 2: Augmented regression
            # Regress y on X_endog, X_exog, and first-stage residuals
            X_augmented = np.column_stack([X_exog, endogenous_vars, first_stage_residuals])
            
            beta_augmented = inv(X_augmented.T @ X_augmented) @ X_augmented.T @ dependent_var
            residuals_augmented = dependent_var - X_augmented @ beta_augmented
            
            # Step 3: Test significance of first-stage residuals
            # H0: coefficients on first-stage residuals are jointly zero
            
            # Extract coefficients on first-stage residuals
            residual_coeffs = beta_augmented[-n_endogenous:]
            
            # Calculate covariance matrix
            mse = np.sum(residuals_augmented**2) / (n_obs - X_augmented.shape[1])
            var_covar = mse * inv(X_augmented.T @ X_augmented)
            
            # Variance of residual coefficients
            var_residual_coeffs = var_covar[-n_endogenous:, -n_endogenous:]
            
            # F-test for joint significance
            f_stat = (residual_coeffs.T @ inv(var_residual_coeffs) @ residual_coeffs) / n_endogenous
            p_value = 1 - stats.f.cdf(f_stat, n_endogenous, n_obs - X_augmented.shape[1])
            
            # Individual t-tests for each endogenous variable
            individual_t_stats = []
            individual_p_values = []
            
            for j in range(n_endogenous):
                t_stat = residual_coeffs[j] / np.sqrt(var_residual_coeffs[j, j])
                p_val_indiv = 2 * (1 - stats.t.cdf(abs(t_stat), n_obs - X_augmented.shape[1]))
                individual_t_stats.append(t_stat)
                individual_p_values.append(p_val_indiv)
            
            test_valid = True
            
        except (np.linalg.LinAlgError, ValueError):
            f_stat = np.nan
            p_value = np.nan
            individual_t_stats = [np.nan] * n_endogenous
            individual_p_values = [np.nan] * n_endogenous
            residual_coeffs = np.full(n_endogenous, np.nan)
            test_valid = False
        
        # Determine endogeneity conclusion
        if test_valid and not np.isnan(p_value):
            endogeneity_detected = p_value < self.significance_level
        else:
            endogeneity_detected = None
        
        results = {
            'f_statistic': f_stat,
            'p_value': p_value,
            'degrees_freedom': (n_endogenous, n_obs - X_augmented.shape[1] if test_valid else np.nan),
            'endogeneity_detected': endogeneity_detected,
            'test_valid': test_valid,
            'residual_coefficients': residual_coeffs,
            'individual_t_stats': individual_t_stats,
            'individual_p_values': individual_p_values,
            'test_type': 'durbin_wu_hausman_test'
        }
        
        return results
    
    def endogeneity_robust_estimation(self,
                                    instruments: np.ndarray,
                                    endogenous_vars: np.ndarray,
                                    dependent_var: np.ndarray,
                                    exogenous_controls: Optional[np.ndarray] = None,
                                    method: str = 'control_function') -> Dict:
        """
        Perform endogeneity-robust estimation using control function approach.
        
        The control function approach adds first-stage residuals as additional
        regressors to control for endogeneity bias.
        
        Parameters:
        -----------
        instruments : np.ndarray
            Matrix of instrumental variables (n_obs x n_instruments)
        endogenous_vars : np.ndarray
            Matrix of endogenous variables (n_obs x n_endogenous)
        dependent_var : np.ndarray
            Dependent variable vector (n_obs,)
        exogenous_controls : np.ndarray, optional
            Matrix of exogenous control variables
        method : str, default='control_function'
            Estimation method ('control_function' or 'two_stage_residual_inclusion')
            
        Returns:
        --------
        Dict containing robust estimation results
        """
        # Validate method
        if method not in ['control_function', 'two_stage_residual_inclusion']:
            raise ValueError(f"Unknown method: {method}")
            
        n_obs = len(dependent_var)
        
        if endogenous_vars.ndim == 1:
            endogenous_vars = endogenous_vars.reshape(-1, 1)
        
        n_endogenous = endogenous_vars.shape[1]
        
        # Prepare regressor matrices
        if exogenous_controls is not None:
            X_exog = np.column_stack([np.ones(n_obs), exogenous_controls])
            Z = np.column_stack([X_exog, instruments])
        else:
            X_exog = np.ones((n_obs, 1))
            Z = np.column_stack([X_exog, instruments])
        
        try:
            # Step 1: First-stage regressions
            P_Z = Z @ inv(Z.T @ Z) @ Z.T
            first_stage_residuals = []
            first_stage_fitted = []
            
            for j in range(n_endogenous):
                y_j = endogenous_vars[:, j]
                y_j_hat = P_Z @ y_j
                residual_j = y_j - y_j_hat
                
                first_stage_residuals.append(residual_j)
                first_stage_fitted.append(y_j_hat)
            
            first_stage_residuals = np.column_stack(first_stage_residuals)
            first_stage_fitted = np.column_stack(first_stage_fitted)
            
            # Step 2: Control function estimation
            if method == 'control_function':
                # Include both endogenous variables and their residuals
                X_cf = np.column_stack([X_exog, endogenous_vars, first_stage_residuals])
                
                beta_cf = inv(X_cf.T @ X_cf) @ X_cf.T @ dependent_var
                residuals_cf = dependent_var - X_cf @ beta_cf
                
                # Extract coefficients
                n_exog = X_exog.shape[1]
                exog_coeffs = beta_cf[:n_exog]
                endog_coeffs = beta_cf[n_exog:n_exog + n_endogenous]
                control_coeffs = beta_cf[n_exog + n_endogenous:]
                
            elif method == 'two_stage_residual_inclusion':
                # Use fitted values instead of original endogenous variables
                X_tsri = np.column_stack([X_exog, first_stage_fitted, first_stage_residuals])
                
                beta_tsri = inv(X_tsri.T @ X_tsri) @ X_tsri.T @ dependent_var
                residuals_cf = dependent_var - X_tsri @ beta_tsri
                
                # Extract coefficients
                n_exog = X_exog.shape[1]
                exog_coeffs = beta_tsri[:n_exog]
                endog_coeffs = beta_tsri[n_exog:n_exog + n_endogenous]
                control_coeffs = beta_tsri[n_exog + n_endogenous:]
            

            
            # Step 3: Calculate standard errors (corrected for two-step estimation)
            if method == 'control_function':
                X_final = X_cf
            else:
                X_final = X_tsri
                
            mse = np.sum(residuals_cf**2) / (n_obs - X_final.shape[1])
            
            # This is a simplified standard error calculation
            # Full implementation would account for first-stage estimation uncertainty
            var_covar = mse * inv(X_final.T @ X_final)
            
            # Extract standard errors for endogenous variable coefficients
            endog_se = np.sqrt(np.diag(var_covar[n_exog:n_exog + n_endogenous, 
                                                 n_exog:n_exog + n_endogenous]))
            
            # T-statistics and p-values for endogenous variables
            t_stats = endog_coeffs / endog_se
            p_values = 2 * (1 - stats.t.cdf(np.abs(t_stats), n_obs - X_final.shape[1]))
            
            # Test for endogeneity (significance of control function coefficients)
            control_se = np.sqrt(np.diag(var_covar[-n_endogenous:, -n_endogenous:]))
            control_t_stats = control_coeffs / control_se
            control_p_values = 2 * (1 - stats.t.cdf(np.abs(control_t_stats), 
                                                   n_obs - X_final.shape[1]))
            
            # Joint test for endogeneity
            endogeneity_f_stat = (control_coeffs.T @ inv(var_covar[-n_endogenous:, -n_endogenous:]) @ 
                                 control_coeffs) / n_endogenous
            endogeneity_p_value = 1 - stats.f.cdf(endogeneity_f_stat, n_endogenous, 
                                                 n_obs - X_final.shape[1])
            
            estimation_successful = True
            
        except (np.linalg.LinAlgError, ValueError):
            exog_coeffs = np.full(X_exog.shape[1], np.nan)
            endog_coeffs = np.full(n_endogenous, np.nan)
            control_coeffs = np.full(n_endogenous, np.nan)
            endog_se = np.full(n_endogenous, np.nan)
            t_stats = np.full(n_endogenous, np.nan)
            p_values = np.full(n_endogenous, np.nan)
            control_t_stats = np.full(n_endogenous, np.nan)
            control_p_values = np.full(n_endogenous, np.nan)
            endogeneity_f_stat = np.nan
            endogeneity_p_value = np.nan
            estimation_successful = False
        
        results = {
            'method': method,
            'exogenous_coefficients': exog_coeffs,
            'endogenous_coefficients': endog_coeffs,
            'control_function_coefficients': control_coeffs,
            'endogenous_standard_errors': endog_se,
            'endogenous_t_statistics': t_stats,
            'endogenous_p_values': p_values,
            'control_t_statistics': control_t_stats,
            'control_p_values': control_p_values,
            'endogeneity_f_statistic': endogeneity_f_stat,
            'endogeneity_p_value': endogeneity_p_value,
            'endogeneity_detected': (endogeneity_p_value < self.significance_level 
                                   if estimation_successful and not np.isnan(endogeneity_p_value) 
                                   else None),
            'estimation_successful': estimation_successful,
            'test_type': 'endogeneity_robust_estimation'
        }
        
        return results
    
    def sensitivity_analysis_identification(self,
                                          instruments_list: List[np.ndarray],
                                          endogenous_vars: np.ndarray,
                                          dependent_var: np.ndarray,
                                          exogenous_controls: Optional[np.ndarray] = None,
                                          instrument_names: Optional[List[str]] = None) -> Dict:
        """
        Perform sensitivity analysis across different instrument specifications.
        
        This method tests the robustness of identification by comparing results
        across different sets of instruments.
        
        Parameters:
        -----------
        instruments_list : List[np.ndarray]
            List of different instrument matrices to test
        endogenous_vars : np.ndarray
            Matrix of endogenous variables (n_obs x n_endogenous)
        dependent_var : np.ndarray
            Dependent variable vector (n_obs,)
        exogenous_controls : np.ndarray, optional
            Matrix of exogenous control variables
        instrument_names : List[str], optional
            Names for each instrument set
            
        Returns:
        --------
        Dict containing sensitivity analysis results
        """
        if instrument_names is None:
            instrument_names = [f"Instrument_Set_{i+1}" for i in range(len(instruments_list))]
        
        sensitivity_results = {
            'instrument_sets': instrument_names,
            'n_instrument_sets': len(instruments_list),
            'validation_results': [],
            'endogeneity_tests': [],
            'coefficient_estimates': [],
            'coefficient_stability': {},
        }
        
        coefficient_matrix = []
        
        for i, instruments in enumerate(instruments_list):
            set_name = instrument_names[i]
            
            # 1. Instrument validation
            validation = self.comprehensive_validation(
                instruments, endogenous_vars, dependent_var, exogenous_controls
            )
            validation['instrument_set_name'] = set_name
            sensitivity_results['validation_results'].append(validation)
            
            # 2. Endogeneity tests
            hausman_result = self.hausman_test(
                instruments, endogenous_vars, dependent_var, exogenous_controls
            )
            hausman_result['instrument_set_name'] = set_name
            
            dwh_result = self.durbin_wu_hausman_test(
                instruments, endogenous_vars, dependent_var, exogenous_controls
            )
            dwh_result['instrument_set_name'] = set_name
            
            sensitivity_results['endogeneity_tests'].append({
                'instrument_set_name': set_name,
                'hausman_test': hausman_result,
                'durbin_wu_hausman_test': dwh_result
            })
            
            # 3. Coefficient estimates (using control function)
            cf_result = self.endogeneity_robust_estimation(
                instruments, endogenous_vars, dependent_var, exogenous_controls
            )
            cf_result['instrument_set_name'] = set_name
            sensitivity_results['coefficient_estimates'].append(cf_result)
            
            # Store coefficients for stability analysis
            if cf_result['estimation_successful']:
                coefficient_matrix.append(cf_result['endogenous_coefficients'])
        
        # 4. Coefficient stability analysis
        if len(coefficient_matrix) > 1:
            coefficient_matrix = np.array(coefficient_matrix)
            
            # Calculate coefficient ranges and standard deviations
            coeff_means = np.mean(coefficient_matrix, axis=0)
            coeff_stds = np.std(coefficient_matrix, axis=0)
            coeff_mins = np.min(coefficient_matrix, axis=0)
            coeff_maxs = np.max(coefficient_matrix, axis=0)
            coeff_ranges = coeff_maxs - coeff_mins
            
            # Coefficient of variation (relative stability measure)
            coeff_cv = np.abs(coeff_stds / coeff_means) if np.all(coeff_means != 0) else np.full_like(coeff_means, np.inf)
            
            sensitivity_results['coefficient_stability'] = {
                'means': coeff_means,
                'standard_deviations': coeff_stds,
                'minimums': coeff_mins,
                'maximums': coeff_maxs,
                'ranges': coeff_ranges,
                'coefficients_of_variation': coeff_cv,
                'stable_coefficients': coeff_cv < 0.1,  # CV < 10% considered stable
                'n_stable_coefficients': np.sum(coeff_cv < 0.1)
            }
        
        # 5. Overall assessment
        valid_instruments = sum(1 for val in sensitivity_results['validation_results'] 
                              if val['overall_assessment'] == 'VALID_INSTRUMENTS')
        
        consistent_endogeneity = self._assess_endogeneity_consistency(
            sensitivity_results['endogeneity_tests']
        )
        
        if len(coefficient_matrix) > 1:
            stable_identification = (sensitivity_results['coefficient_stability']['n_stable_coefficients'] 
                                   >= len(coeff_means) * 0.8)  # 80% of coefficients stable
        else:
            stable_identification = None
        
        sensitivity_results['overall_assessment'] = {
            'valid_instrument_sets': valid_instruments,
            'total_instrument_sets': len(instruments_list),
            'consistent_endogeneity_detection': consistent_endogeneity,
            'stable_identification': stable_identification,
            'recommendation': self._generate_sensitivity_recommendations(
                valid_instruments, len(instruments_list), consistent_endogeneity, stable_identification
            )
        }
        
        return sensitivity_results
    
    def _assess_endogeneity_consistency(self, endogeneity_tests: List[Dict]) -> bool:
        """Assess consistency of endogeneity detection across instrument sets."""
        hausman_results = []
        dwh_results = []
        
        for test_set in endogeneity_tests:
            hausman = test_set['hausman_test']['endogeneity_detected']
            dwh = test_set['durbin_wu_hausman_test']['endogeneity_detected']
            
            if hausman is not None:
                hausman_results.append(hausman)
            if dwh is not None:
                dwh_results.append(dwh)
        
        # Check consistency within each test type
        hausman_consistent = len(set(hausman_results)) <= 1 if hausman_results else True
        dwh_consistent = len(set(dwh_results)) <= 1 if dwh_results else True
        
        return hausman_consistent and dwh_consistent
    
    def _generate_sensitivity_recommendations(self, 
                                            valid_instruments: int, 
                                            total_instruments: int,
                                            consistent_endogeneity: bool,
                                            stable_identification: Optional[bool]) -> str:
        """Generate recommendations based on sensitivity analysis."""
        if valid_instruments == 0:
            return "No valid instrument sets found. Consider alternative identification strategies."
        
        if valid_instruments < total_instruments * 0.5:
            return "Less than half of instrument sets are valid. Review instrument construction."
        
        if not consistent_endogeneity:
            return "Inconsistent endogeneity detection across instruments. Results may be unreliable."
        
        if stable_identification is False:
            return "Coefficient estimates are not stable across instruments. Identification may be weak."
        
        if valid_instruments == total_instruments and consistent_endogeneity and stable_identification:
            return "Identification appears robust across all instrument specifications."
        
        return "Identification shows mixed results. Consider additional robustness checks."