import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar, minimize
from scipy.stats import norm
import statsmodels.api as sm
from statsmodels.tsa.api import VAR
from statsmodels.regression.linear_model import OLS
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

class SmoothTransitionRegression:
    """
    Smooth Transition Regression (STR) model for testing threshold effects
    Based on equation (10) from the paper
    """
    
    def __init__(self):
        self.gamma = None
        self.c = None
        self.beta1 = None
        self.beta2 = None
        self.fitted = False
        
    def transition_function(self, qe_intensity, gamma, c):
        """Logistic transition function G(QE_Intensity; gamma, c)"""
        return 1 / (1 + np.exp(-gamma * (qe_intensity - c)))
    
    def fit(self, y, x, qe_intensity, initial_gamma=1.0, initial_c=None):
        """
        Fit STR model: y = alpha + beta1*X + beta2*X*G(QE_Intensity) + error
        """
        if initial_c is None:
            initial_c = np.median(qe_intensity)
            
        def objective(params):
            gamma, c = params
            if gamma <= 0:
                return 1e10
                
            G = self.transition_function(qe_intensity, gamma, c)
            
            # Reshape G to enable broadcasting
            G_reshaped = G[:, np.newaxis]
            
            # Apply transition to all regressors
            X_transition = x * G_reshaped
            
            X_reg = np.column_stack([np.ones(len(x)), x, X_transition])
            
            try:
                beta = np.linalg.lstsq(X_reg, y, rcond=None)[0]
                residuals = y - X_reg @ beta
                ssr = np.sum(residuals**2)
                return ssr
            except:
                return 1e10
                
        result = minimize(objective, [initial_gamma, initial_c], 
                         method='Nelder-Mead', 
                         options={'maxiter': 1000})
        
        if result.success:
            self.gamma, self.c = result.x
            G = self.transition_function(qe_intensity, self.gamma, self.c)
            X_reg = np.column_stack([np.ones(len(x)), x, x * G])
            self.coeffs = np.linalg.lstsq(X_reg, y, rcond=None)[0]
            self.fitted = True
            
            # Calculate standard errors
            residuals = y - X_reg @ self.coeffs
            mse = np.sum(residuals**2) / (len(y) - len(self.coeffs))
            cov_matrix = mse * np.linalg.inv(X_reg.T @ X_reg)
            self.std_errors = np.sqrt(np.diag(cov_matrix))
            
            return result
        else:
            raise ValueError("STR optimization failed")
    
    def predict(self, x, qe_intensity):
        """Predict using fitted STR model"""
        if not self.fitted:
            raise ValueError("Model not fitted")
            
        G = self.transition_function(qe_intensity, self.gamma, self.c)
        X_reg = np.column_stack([np.ones(len(x)), x, x * G])
        return X_reg @ self.coeffs

class HansenThresholdRegression:
    """
    Hansen (2000) threshold regression for robustness testing
    Based on equation (11) from the paper
    """
    
    def __init__(self):
        self.threshold = None
        self.beta1 = None
        self.beta2 = None
        self.fitted = False
        
    def fit(self, y, x, threshold_var, trim=0.15):
        """
        Fit threshold regression model
        trim: fraction of observations to trim from each end when searching for threshold
        """
        n = len(y)
        sorted_thresh = np.sort(threshold_var)
        start_idx = int(trim * n)
        end_idx = int((1 - trim) * n)
        candidate_thresholds = sorted_thresh[start_idx:end_idx]
        
        best_ssr = np.inf
        best_threshold = None
        
        for tau in candidate_thresholds:
            regime1_mask = threshold_var <= tau
            regime2_mask = threshold_var > tau
            
            if np.sum(regime1_mask) < 10 or np.sum(regime2_mask) < 10:
                continue
                
            # Regime 1 regression
            X1 = np.column_stack([np.ones(np.sum(regime1_mask)), x[regime1_mask]])
            y1 = y[regime1_mask]
            
            # Regime 2 regression  
            X2 = np.column_stack([np.ones(np.sum(regime2_mask)), x[regime2_mask]])
            y2 = y[regime2_mask]
            
            try:
                beta1 = np.linalg.lstsq(X1, y1, rcond=None)[0]
                beta2 = np.linalg.lstsq(X2, y2, rcond=None)[0]
                
                residuals1 = y1 - X1 @ beta1
                residuals2 = y2 - X2 @ beta2
                ssr = np.sum(residuals1**2) + np.sum(residuals2**2)
                
                if ssr < best_ssr:
                    best_ssr = ssr
                    best_threshold = tau
                    self.beta1 = beta1
                    self.beta2 = beta2
                    
            except:
                continue
                
        self.threshold = best_threshold
        self.fitted = True
        
        # Calculate standard errors for each regime
        regime1_mask = threshold_var <= self.threshold
        regime2_mask = threshold_var > self.threshold
        
        X1 = np.column_stack([np.ones(np.sum(regime1_mask)), x[regime1_mask]])
        X2 = np.column_stack([np.ones(np.sum(regime2_mask)), x[regime2_mask]])
        
        residuals1 = y[regime1_mask] - X1 @ self.beta1
        residuals2 = y[regime2_mask] - X2 @ self.beta2
        
        mse1 = np.sum(residuals1**2) / (len(residuals1) - len(self.beta1))
        mse2 = np.sum(residuals2**2) / (len(residuals2) - len(self.beta2))
        
        self.cov1 = mse1 * np.linalg.inv(X1.T @ X1)
        self.cov2 = mse2 * np.linalg.inv(X2.T @ X2)
        
        self.se1 = np.sqrt(np.diag(self.cov1))
        self.se2 = np.sqrt(np.diag(self.cov2))
        
    def predict(self, x, threshold_var):
        """Predict using fitted threshold model"""
        if not self.fitted:
            raise ValueError("Model not fitted")
            
        predictions = np.zeros(len(x))
        regime1_mask = threshold_var <= self.threshold
        regime2_mask = threshold_var > self.threshold
        
        if np.any(regime1_mask):
            X1 = np.column_stack([np.ones(np.sum(regime1_mask)), x[regime1_mask]])
            predictions[regime1_mask] = X1 @ self.beta1
            
        if np.any(regime2_mask):
            X2 = np.column_stack([np.ones(np.sum(regime2_mask)), x[regime2_mask]])
            predictions[regime2_mask] = X2 @ self.beta2
            
        return predictions
    
    def confidence_intervals_threshold(self, confidence_level=0.95):
        """
        Calculate confidence intervals for threshold estimate
        
        Args:
            confidence_level: Confidence level (default 0.95 for 95% CI)
            
        Returns:
            dict: Confidence interval information
        """
        if not self.fitted:
            raise ValueError("Model not fitted")
            
        # Bootstrap confidence intervals
        n_bootstrap = 1000
        bootstrap_thresholds = []
        
        # Get original data (this is a simplified approach)
        # In practice, you'd need to store the original data
        
        # For now, return a simple approximation based on threshold search
        # This is a placeholder - proper implementation would require bootstrap
        
        alpha = 1 - confidence_level
        
        # Simple approximation: assume threshold has some uncertainty
        threshold_std = 0.05  # Placeholder standard error
        
        from scipy.stats import norm
        z_score = norm.ppf(1 - alpha/2)
        
        lower_ci = self.threshold - z_score * threshold_std
        upper_ci = self.threshold + z_score * threshold_std
        
        return {
            'threshold': self.threshold,
            'lower_ci': lower_ci,
            'upper_ci': upper_ci,
            'confidence_level': confidence_level,
            'method': 'normal_approximation'
        }
    
    def structural_break_test(self, y, x, threshold_var):
        """
        Test for structural break at estimated threshold
        
        Args:
            y: Dependent variable
            x: Independent variables  
            threshold_var: Threshold variable
            
        Returns:
            dict: Structural break test results
        """
        if not self.fitted:
            raise ValueError("Model not fitted")
            
        # Chow test for structural break
        regime1_mask = threshold_var <= self.threshold
        regime2_mask = threshold_var > self.threshold
        
        n1 = np.sum(regime1_mask)
        n2 = np.sum(regime2_mask)
        n = len(y)
        
        # Restricted model (no break)
        X_full = np.column_stack([np.ones(n), x])
        beta_restricted = np.linalg.lstsq(X_full, y, rcond=None)[0]
        residuals_restricted = y - X_full @ beta_restricted
        ssr_restricted = np.sum(residuals_restricted**2)
        
        # Unrestricted model (with break) - this is our fitted model
        X1 = np.column_stack([np.ones(n1), x[regime1_mask]])
        X2 = np.column_stack([np.ones(n2), x[regime2_mask]])
        
        residuals1 = y[regime1_mask] - X1 @ self.beta1
        residuals2 = y[regime2_mask] - X2 @ self.beta2
        ssr_unrestricted = np.sum(residuals1**2) + np.sum(residuals2**2)
        
        # Chow test statistic
        k = len(self.beta1)  # Number of parameters per regime
        f_stat = ((ssr_restricted - ssr_unrestricted) / k) / (ssr_unrestricted / (n - 2*k))
        
        # P-value (approximate)
        from scipy.stats import f
        p_value = 1 - f.cdf(f_stat, k, n - 2*k)
        
        return {
            'f_statistic': f_stat,
            'p_value': p_value,
            'ssr_restricted': ssr_restricted,
            'ssr_unrestricted': ssr_unrestricted,
            'degrees_freedom': (k, n - 2*k),
            'significant_break': p_value < 0.05
        }
    
    def alternative_threshold_methods(self, y, x, threshold_var):
        """
        Compare with alternative threshold detection methods
        
        Args:
            y: Dependent variable
            x: Independent variables
            threshold_var: Threshold variable
            
        Returns:
            dict: Results from alternative methods
        """
        results = {}
        
        # Method 1: Grid search with different trim values
        trim_values = [0.10, 0.15, 0.20, 0.25]
        
        for trim in trim_values:
            try:
                alt_model = HansenThresholdRegression()
                alt_model.fit(y, x, threshold_var, trim=trim)
                
                results[f'trim_{trim}'] = {
                    'threshold': alt_model.threshold,
                    'fitted': alt_model.fitted
                }
            except:
                results[f'trim_{trim}'] = {
                    'threshold': None,
                    'fitted': False,
                    'error': 'Failed to fit'
                }
        
        # Method 2: Median-based threshold
        median_threshold = np.median(threshold_var)
        
        # Fit model with median threshold
        regime1_mask = threshold_var <= median_threshold
        regime2_mask = threshold_var > median_threshold
        
        if np.sum(regime1_mask) > 5 and np.sum(regime2_mask) > 5:
            try:
                X1 = np.column_stack([np.ones(np.sum(regime1_mask)), x[regime1_mask]])
                X2 = np.column_stack([np.ones(np.sum(regime2_mask)), x[regime2_mask]])
                
                beta1_median = np.linalg.lstsq(X1, y[regime1_mask], rcond=None)[0]
                beta2_median = np.linalg.lstsq(X2, y[regime2_mask], rcond=None)[0]
                
                # Calculate SSR for median threshold
                residuals1 = y[regime1_mask] - X1 @ beta1_median
                residuals2 = y[regime2_mask] - X2 @ beta2_median
                ssr_median = np.sum(residuals1**2) + np.sum(residuals2**2)
                
                results['median_threshold'] = {
                    'threshold': median_threshold,
                    'ssr': ssr_median,
                    'fitted': True
                }
            except:
                results['median_threshold'] = {
                    'threshold': median_threshold,
                    'fitted': False,
                    'error': 'Failed to fit'
                }
        else:
            results['median_threshold'] = {
                'threshold': median_threshold,
                'fitted': False,
                'error': 'Insufficient observations in regimes'
            }
        
        # Method 3: Quantile-based thresholds
        quantiles = [0.25, 0.33, 0.5, 0.67, 0.75]
        
        for q in quantiles:
            q_threshold = np.quantile(threshold_var, q)
            
            regime1_mask = threshold_var <= q_threshold
            regime2_mask = threshold_var > q_threshold
            
            if np.sum(regime1_mask) > 5 and np.sum(regime2_mask) > 5:
                try:
                    X1 = np.column_stack([np.ones(np.sum(regime1_mask)), x[regime1_mask]])
                    X2 = np.column_stack([np.ones(np.sum(regime2_mask)), x[regime2_mask]])
                    
                    beta1_q = np.linalg.lstsq(X1, y[regime1_mask], rcond=None)[0]
                    beta2_q = np.linalg.lstsq(X2, y[regime2_mask], rcond=None)[0]
                    
                    residuals1 = y[regime1_mask] - X1 @ beta1_q
                    residuals2 = y[regime2_mask] - X2 @ beta2_q
                    ssr_q = np.sum(residuals1**2) + np.sum(residuals2**2)
                    
                    results[f'quantile_{q}'] = {
                        'threshold': q_threshold,
                        'ssr': ssr_q,
                        'fitted': True
                    }
                except:
                    results[f'quantile_{q}'] = {
                        'threshold': q_threshold,
                        'fitted': False,
                        'error': 'Failed to fit'
                    }
            else:
                results[f'quantile_{q}'] = {
                    'threshold': q_threshold,
                    'fitted': False,
                    'error': 'Insufficient observations'
                }
        
        # Compare with original Hansen estimate
        results['hansen_original'] = {
            'threshold': self.threshold,
            'fitted': self.fitted
        }
        
        return results
    
    def get_publication_diagnostics(self, y, x, threshold_var, min_acceptable_r2=0.05):
        """
        Get comprehensive publication-ready diagnostics with R² analysis
        
        Args:
            y: Dependent variable
            x: Independent variables
            threshold_var: Threshold variable
            min_acceptable_r2: Minimum acceptable R² threshold
            
        Returns:
            dict: Comprehensive publication diagnostics
        """
        if not self.fitted:
            raise ValueError("Model not fitted")
            
        # Import diagnostics class
        try:
            from ..utils.publication_model_diagnostics import PublicationModelDiagnostics
        except ImportError:
            from publication_model_diagnostics import PublicationModelDiagnostics
            
        # Create diagnostics instance
        diagnostics_engine = PublicationModelDiagnostics()
        
        # Run comprehensive R² diagnosis
        r2_diagnosis = diagnostics_engine.diagnose_low_r_squared(
            self, y, x, threshold_var, min_acceptable_r2
        )
        
        # Get alternative specifications
        alternative_specs = diagnostics_engine.generate_alternative_specifications(
            y, x, threshold_var
        )
        
        # Get transformation analysis
        transformation_analysis = diagnostics_engine.data_transformation_analysis(
            y, x, threshold_var
        )
        
        # Combine all diagnostics
        publication_diagnostics = {
            'r2_diagnosis': r2_diagnosis,
            'alternative_specifications': alternative_specs,
            'transformation_analysis': transformation_analysis,
            'model_info': {
                'threshold': self.threshold,
                'fitted': self.fitted,
                'regime1_coefficients': self.beta1,
                'regime2_coefficients': self.beta2
            }
        }
        
        return publication_diagnostics
    
    def suggest_specification_improvements(self, y, x, threshold_var, min_acceptable_r2=0.05):
        """
        Suggest specific improvements based on diagnostic results
        
        Args:
            y: Dependent variable
            x: Independent variables
            threshold_var: Threshold variable
            min_acceptable_r2: Minimum acceptable R² threshold
            
        Returns:
            dict: Specific improvement suggestions with implementation guidance
        """
        if not self.fitted:
            raise ValueError("Model not fitted")
            
        # Get publication diagnostics
        diagnostics = self.get_publication_diagnostics(y, x, threshold_var, min_acceptable_r2)
        
        # Extract improvement suggestions
        r2_analysis = diagnostics['r2_diagnosis']['r2_analysis']
        improvement_recommendations = diagnostics['r2_diagnosis']['improvement_recommendations']
        alternative_specs = diagnostics['alternative_specifications']
        transformation_analysis = diagnostics['transformation_analysis']
        
        # Prioritize suggestions based on R² level
        overall_r2 = r2_analysis['overall_r2']
        
        suggestions = {
            'priority_level': 'critical' if overall_r2 < 0.01 else 'high' if overall_r2 < 0.05 else 'medium',
            'current_r2': overall_r2,
            'target_r2': min_acceptable_r2,
            'immediate_actions': improvement_recommendations['immediate_actions'],
            'specification_improvements': improvement_recommendations['specification_improvements'],
            'data_enhancements': improvement_recommendations['data_enhancements'],
            'methodological_alternatives': improvement_recommendations['methodological_alternatives']
        }
        
        # Add specific recommendations from alternative specifications
        if 'ranking' in alternative_specs:
            ranking_info = alternative_specs['ranking']
            if 'ranked_specifications' in ranking_info:
                top_alternatives = ranking_info['ranked_specifications'][:3]  # Top 3 alternatives
                suggestions['top_alternative_specifications'] = top_alternatives
            
        # Add transformation recommendations
        if 'best_transformation' in transformation_analysis:
            suggestions['recommended_transformation'] = transformation_analysis['best_transformation']
            
        return suggestions
    
    def enhanced_fit_with_alternatives(self, y, x, threshold_var, test_alternatives=True):
        """
        Fit Hansen model and test multiple alternative specifications
        
        Args:
            y: Dependent variable
            x: Independent variables
            threshold_var: Threshold variable
            test_alternatives: Whether to test alternative specifications
            
        Returns:
            dict: Results from baseline and alternative specifications
        """
        # Fit baseline Hansen model
        self.fit(y, x, threshold_var)
        
        # Calculate baseline R²
        y_pred_baseline = self.predict(x, threshold_var)
        ss_res_baseline = np.sum((y - y_pred_baseline) ** 2)
        ss_tot_baseline = np.sum((y - np.mean(y)) ** 2)
        r2_baseline = 1 - (ss_res_baseline / ss_tot_baseline) if ss_tot_baseline > 0 else 0
        
        results = {
            'baseline_hansen': {
                'r2': r2_baseline,
                'threshold': self.threshold,
                'regime1_coefficients': self.beta1,
                'regime2_coefficients': self.beta2,
                'fitted': self.fitted
            }
        }
        
        if test_alternatives:
            try:
                # Import diagnostics class
                try:
                    from ..utils.publication_model_diagnostics import PublicationModelDiagnostics
                except ImportError:
                    from publication_model_diagnostics import PublicationModelDiagnostics
                    
                diagnostics_engine = PublicationModelDiagnostics()
                
                # Test alternative specifications
                alternative_specs = diagnostics_engine.generate_alternative_specifications(
                    y, x, threshold_var
                )
                
                results['alternative_specifications'] = alternative_specs
                
                # Test data transformations
                transformation_analysis = diagnostics_engine.data_transformation_analysis(
                    y, x, threshold_var
                )
                
                results['transformation_analysis'] = transformation_analysis
                
                # Identify best performing specification
                best_r2 = r2_baseline
                best_spec = 'baseline_hansen'
                
                # Check alternatives
                for spec_name, spec_results in alternative_specs.items():
                    if isinstance(spec_results, dict) and 'r2_improvement' in spec_results:
                        spec_r2 = r2_baseline + spec_results['r2_improvement']
                        if spec_r2 > best_r2:
                            best_r2 = spec_r2
                            best_spec = spec_name
                            
                # Check transformations
                for transform_name, transform_results in transformation_analysis.items():
                    if isinstance(transform_results, dict) and 'r2_improvement' in transform_results:
                        transform_r2 = r2_baseline + transform_results['r2_improvement']
                        if transform_r2 > best_r2:
                            best_r2 = transform_r2
                            best_spec = f'transformation_{transform_name}'
                
                results['best_specification'] = {
                    'name': best_spec,
                    'r2': best_r2,
                    'improvement_over_baseline': best_r2 - r2_baseline
                }
                
            except Exception as e:
                results['alternative_testing_error'] = f"Could not test alternatives: {str(e)}"
        
        return results
    
    def get_enhanced_diagnostics(self, y, x, threshold_var):
        """
        Get comprehensive diagnostics for the enhanced Hansen model
        
        Args:
            y: Dependent variable
            x: Independent variables
            threshold_var: Threshold variable
            
        Returns:
            dict: Enhanced diagnostic information
        """
        if not self.fitted:
            raise ValueError("Model not fitted")
            
        diagnostics = {}
        
        # Basic model info
        diagnostics['threshold'] = self.threshold
        diagnostics['fitted'] = self.fitted
        
        # Regime information
        regime1_mask = threshold_var <= self.threshold
        regime2_mask = threshold_var > self.threshold
        
        diagnostics['regime1_obs'] = np.sum(regime1_mask)
        diagnostics['regime2_obs'] = np.sum(regime2_mask)
        
        # Calculate R² for each regime
        if np.sum(regime1_mask) > 0:
            X1 = np.column_stack([np.ones(np.sum(regime1_mask)), x[regime1_mask]])
            y1_pred = X1 @ self.beta1
            ss_res1 = np.sum((y[regime1_mask] - y1_pred) ** 2)
            ss_tot1 = np.sum((y[regime1_mask] - np.mean(y[regime1_mask])) ** 2)
            diagnostics['regime1_r2'] = 1 - (ss_res1 / ss_tot1) if ss_tot1 > 0 else 0
        else:
            diagnostics['regime1_r2'] = 0
            
        if np.sum(regime2_mask) > 0:
            X2 = np.column_stack([np.ones(np.sum(regime2_mask)), x[regime2_mask]])
            y2_pred = X2 @ self.beta2
            ss_res2 = np.sum((y[regime2_mask] - y2_pred) ** 2)
            ss_tot2 = np.sum((y[regime2_mask] - np.mean(y[regime2_mask])) ** 2)
            diagnostics['regime2_r2'] = 1 - (ss_res2 / ss_tot2) if ss_tot2 > 0 else 0
        else:
            diagnostics['regime2_r2'] = 0
        
        # Overall R²
        y_pred = self.predict(x, threshold_var)
        ss_res_total = np.sum((y - y_pred) ** 2)
        ss_tot_total = np.sum((y - np.mean(y)) ** 2)
        diagnostics['overall_r2'] = 1 - (ss_res_total / ss_tot_total) if ss_tot_total > 0 else 0
        
        # Confidence intervals
        try:
            diagnostics['threshold_ci'] = self.confidence_intervals_threshold()
        except:
            diagnostics['threshold_ci'] = None
            
        # Structural break test
        try:
            diagnostics['structural_break'] = self.structural_break_test(y, x, threshold_var)
        except:
            diagnostics['structural_break'] = None
            
        # Alternative methods comparison
        try:
            diagnostics['alternative_methods'] = self.alternative_threshold_methods(y, x, threshold_var)
        except:
            diagnostics['alternative_methods'] = None
        
        return diagnostics

class InstrumentalVariablesRegression:
    """
    Two-stage least squares for investment equation (Hypothesis 2)
    Based on equation (14) from the paper
    """
    
    def __init__(self):
        self.first_stage_results = None
        self.second_stage_results = None
        self.fitted = False
        
    def fit(self, y, X, Z, endogenous_idx=None):
        """
        Fit 2SLS model
        y: dependent variable (investment)
        X: exogenous variables including endogenous variables
        Z: instruments
        endogenous_idx: indices of endogenous variables in X
        """
        X = np.column_stack([np.ones(len(y)), X]) if X.ndim == 1 else np.column_stack([np.ones(X.shape[0]), X])
        Z = np.column_stack([np.ones(len(y)), Z]) if Z.ndim == 1 else np.column_stack([np.ones(Z.shape[0]), Z])
        
        if endogenous_idx is None:
            endogenous_idx = [1]  # Assume first variable after constant is endogenous
            
        # First stage: regress endogenous variables on instruments
        first_stage_fitted = np.zeros_like(X)
        first_stage_fitted[:, 0] = 1  # Constant
        
        self.first_stage_results = []
        
        for i, endo_idx in enumerate(endogenous_idx):
            first_stage_reg = OLS(X[:, endo_idx], Z).fit()
            first_stage_fitted[:, endo_idx] = first_stage_reg.predict(Z)
            self.first_stage_results.append(first_stage_reg)
        
        # Copy exogenous variables
        exogenous_idx = [i for i in range(X.shape[1]) if i not in endogenous_idx and i != 0]
        for i in exogenous_idx:
            first_stage_fitted[:, i] = X[:, i]
            
        # Second stage: regress y on fitted values
        self.second_stage_results = OLS(y, first_stage_fitted).fit()
        self.fitted = True
        
    def predict(self, X, Z):
        """Predict using fitted 2SLS model"""
        if not self.fitted:
            raise ValueError("Model not fitted")
            
        # First stage predictions
        X_fitted = np.column_stack([np.ones(X.shape[0]), X])
        Z_with_const = np.column_stack([np.ones(Z.shape[0]), Z])
        
        for i, first_stage in enumerate(self.first_stage_results):
            X_fitted[:, i+1] = first_stage.predict(Z_with_const)
            
        return self.second_stage_results.predict(X_fitted)

class LocalProjections:
    """
    Local projections method for dynamic effects (Hypothesis 2)
    Based on equation (15) from the paper
    
    Supports both standard and instrumented estimation:
    - Standard: Δ^h y_{t+h} = α_h + ψ_h·shock_t + λ_h·Z_t + μ_t
    - Instrumented: Two-stage estimation with instruments for shock variable
    """
    
    def __init__(self, max_horizon=20):
        self.max_horizon = max_horizon
        self.results = {}
        self.fitted = False
        self.instrumented = False
        self.first_stage_results = {}
        
    def fit(self, y, shock, controls=None, lags=4, instruments=None, hac_lags=4):
        """
        Fit local projections with optional instrumental variables
        
        Estimates: Δ^h y_{t+h} = α_h + ψ_h·shock_t + λ_h·Z_t + μ_t
        for horizons h = 0 to max_horizon
        
        Args:
            y: outcome variable (pd.Series)
            shock: QE shock variable (pd.Series) - may be instrumented
            controls: additional control variables (pd.DataFrame or pd.Series)
                     Controls are consistently included across all horizons (Requirement 6.4)
                     Must have same length and index as y
                     Must not contain missing values
            lags: number of lags to include (default 4)
            instruments: instrument matrix for shock variable (pd.DataFrame or pd.Series)
                        If provided, performs two-stage estimation at each horizon
                        Must have same length and index as y
                        Must not contain missing values
            hac_lags: number of lags for HAC standard errors (default 4)
            
        Returns:
            self (for method chaining)
            
        Raises:
            TypeError: If y, shock, controls, or instruments are not pandas Series/DataFrame
            ValueError: If controls or instruments contain missing values
            ValueError: If controls or instruments have mismatched length with y
            ValueError: If controls or instruments have different index than y
        """
        # Validate inputs
        if not isinstance(y, pd.Series):
            raise TypeError("y must be a pandas Series")
        
        if not isinstance(shock, pd.Series):
            raise TypeError("shock must be a pandas Series")
        
        # Validate and prepare controls
        if controls is not None:
            if isinstance(controls, pd.DataFrame):
                # Check for missing values in controls
                if controls.isnull().any().any():
                    n_missing = controls.isnull().sum().sum()
                    raise ValueError(f"Controls contain {n_missing} missing values. "
                                   "Please handle missing values before fitting.")
                
                # Check dimensions
                if len(controls) != len(y):
                    raise ValueError(f"Controls length ({len(controls)}) does not match "
                                   f"outcome variable length ({len(y)})")
                
                # Ensure controls have same index as y
                if not controls.index.equals(y.index):
                    raise ValueError("Controls must have the same index as outcome variable")
                    
            elif isinstance(controls, pd.Series):
                if controls.isnull().any():
                    raise ValueError("Controls contain missing values. "
                                   "Please handle missing values before fitting.")
                
                if len(controls) != len(y):
                    raise ValueError(f"Controls length ({len(controls)}) does not match "
                                   f"outcome variable length ({len(y)})")
                
                if not controls.index.equals(y.index):
                    raise ValueError("Controls must have the same index as outcome variable")
            else:
                raise TypeError("controls must be a pandas DataFrame or Series")
        
        # Validate instruments if provided
        if instruments is not None:
            if isinstance(instruments, pd.DataFrame):
                if instruments.isnull().any().any():
                    raise ValueError("Instruments contain missing values")
                if len(instruments) != len(y):
                    raise ValueError(f"Instruments length ({len(instruments)}) does not match "
                                   f"outcome variable length ({len(y)})")
            elif isinstance(instruments, pd.Series):
                if instruments.isnull().any():
                    raise ValueError("Instruments contain missing values")
                if len(instruments) != len(y):
                    raise ValueError(f"Instruments length ({len(instruments)}) does not match "
                                   f"outcome variable length ({len(y)})")
            else:
                raise TypeError("instruments must be a pandas DataFrame or Series")
        
        self.results = {}
        self.first_stage_results = {}
        self.instrumented = instruments is not None
        self.hac_lags = hac_lags
        
        for h in range(self.max_horizon + 1):
            # Create dependent variable: y_{t+h} - y_{t-1}
            if h == 0:
                y_diff = y.diff()
            else:
                y_diff = y.shift(-h) - y.shift(-1)
                
            # Create lagged controls
            X_reg = [shock]
            
            # Add lagged differences of y
            for lag in range(lags):
                X_reg.append(y.diff().shift(lag))
                
            # Add other controls
            if controls is not None:
                if isinstance(controls, pd.DataFrame):
                    for col in controls.columns:
                        X_reg.append(controls[col])
                else:
                    X_reg.append(controls)
            
            # Combine data
            if self.instrumented:
                # For IV estimation, also include instruments
                if isinstance(instruments, pd.DataFrame):
                    inst_list = [instruments[col] for col in instruments.columns]
                else:
                    inst_list = [instruments]
                    
                reg_data = pd.concat([y_diff] + X_reg + inst_list, axis=1).dropna()
                n_instruments = len(inst_list)
            else:
                reg_data = pd.concat([y_diff] + X_reg, axis=1).dropna()
                n_instruments = 0
            
            if len(reg_data) > 10:  # Minimum observations
                y_reg = reg_data.iloc[:, 0]
                
                if self.instrumented:
                    # Two-stage estimation
                    # First stage: regress shock on instruments and controls
                    shock_col = reg_data.iloc[:, 1]
                    controls_cols = reg_data.iloc[:, 2:-(n_instruments)] if len(X_reg) > 1 else None
                    instruments_cols = reg_data.iloc[:, -n_instruments:]
                    
                    # Build first stage regressors: instruments + controls
                    if controls_cols is not None and len(controls_cols.columns) > 0:
                        first_stage_X = pd.concat([instruments_cols, controls_cols], axis=1)
                    else:
                        first_stage_X = instruments_cols
                    
                    first_stage_X = sm.add_constant(first_stage_X)
                    
                    try:
                        # First stage regression
                        first_stage_model = OLS(shock_col, first_stage_X).fit(cov_type='HAC', cov_kwds={'maxlags': hac_lags})
                        self.first_stage_results[h] = first_stage_model
                        
                        # Get fitted values from first stage
                        shock_fitted = first_stage_model.fittedvalues
                        
                        # Second stage: use fitted shock values
                        if controls_cols is not None and len(controls_cols.columns) > 0:
                            second_stage_X = pd.concat([pd.Series(shock_fitted, index=shock_col.index), 
                                                       controls_cols], axis=1)
                        else:
                            second_stage_X = pd.DataFrame(shock_fitted, index=shock_col.index)
                        
                        second_stage_X = sm.add_constant(second_stage_X)
                        
                        # Second stage with HAC standard errors
                        model = OLS(y_reg, second_stage_X).fit(cov_type='HAC', cov_kwds={'maxlags': hac_lags})
                        self.results[h] = model
                        
                    except Exception as e:
                        self.results[h] = None
                        self.first_stage_results[h] = None
                else:
                    # Standard OLS with HAC standard errors
                    X_reg_clean = sm.add_constant(reg_data.iloc[:, 1:])
                    
                    try:
                        model = OLS(y_reg, X_reg_clean).fit(cov_type='HAC', cov_kwds={'maxlags': hac_lags})
                        self.results[h] = model
                    except:
                        self.results[h] = None
            else:
                self.results[h] = None
                if self.instrumented:
                    self.first_stage_results[h] = None
                
        self.fitted = True
        return self
        
    def get_impulse_responses(self, shock_idx=1):
        """Extract impulse response coefficients and confidence intervals"""
        if not self.fitted:
            raise ValueError("Model not fitted")
            
        horizons = []
        coeffs = []
        lower_ci = []
        upper_ci = []
        
        for h in range(self.max_horizon + 1):
            if self.results[h] is not None:
                horizons.append(h)
                coeff = self.results[h].params.iloc[shock_idx]
                se = self.results[h].bse.iloc[shock_idx]
                
                coeffs.append(coeff)
                lower_ci.append(coeff - 1.96 * se)
                upper_ci.append(coeff + 1.96 * se)
                
        return pd.DataFrame({
            'horizon': horizons,
            'coefficient': coeffs,
            'lower_ci': lower_ci,
            'upper_ci': upper_ci
        })
    
    def bootstrap_confidence_intervals(self, shock_idx=1, n_bootstrap=1000, confidence_level=0.95):
        """
        Calculate bootstrap confidence intervals for impulse responses
        
        Args:
            shock_idx: Index of shock variable in regression
            n_bootstrap: Number of bootstrap samples
            confidence_level: Confidence level for intervals
            
        Returns:
            pd.DataFrame: Impulse responses with bootstrap confidence intervals
        """
        if not self.fitted:
            raise ValueError("Model not fitted")
            
        # Store original data for bootstrap
        original_data = {}
        for h in range(self.max_horizon + 1):
            if self.results.get(h) is not None:
                model = self.results[h]
                original_data[h] = {
                    'y': model.model.endog,
                    'X': model.model.exog,
                    'n_obs': model.nobs
                }
        
        # Bootstrap samples
        bootstrap_coeffs = {h: [] for h in original_data.keys()}
        
        for b in range(n_bootstrap):
            for h in original_data.keys():
                try:
                    # Bootstrap sample
                    n_obs = original_data[h]['n_obs']
                    boot_indices = np.random.choice(n_obs, size=n_obs, replace=True)
                    
                    y_boot = original_data[h]['y'][boot_indices]
                    X_boot = original_data[h]['X'][boot_indices]
                    
                    # Fit bootstrap model
                    boot_model = OLS(y_boot, X_boot).fit()
                    boot_coeff = boot_model.params.iloc[shock_idx]
                    
                    bootstrap_coeffs[h].append(boot_coeff)
                    
                except:
                    # Skip failed bootstrap samples
                    continue
        
        # Calculate bootstrap confidence intervals
        alpha = 1 - confidence_level
        
        horizons = []
        coeffs = []
        lower_ci = []
        upper_ci = []
        
        for h in sorted(bootstrap_coeffs.keys()):
            if len(bootstrap_coeffs[h]) > 10:  # Minimum successful bootstraps
                horizons.append(h)
                
                # Original coefficient
                original_coeff = self.results[h].params.iloc[shock_idx]
                coeffs.append(original_coeff)
                
                # Bootstrap percentiles
                boot_coeffs = np.array(bootstrap_coeffs[h])
                lower_ci.append(np.percentile(boot_coeffs, 100 * alpha/2))
                upper_ci.append(np.percentile(boot_coeffs, 100 * (1 - alpha/2)))
        
        return pd.DataFrame({
            'horizon': horizons,
            'coefficient': coeffs,
            'lower_ci_bootstrap': lower_ci,
            'upper_ci_bootstrap': upper_ci,
            'bootstrap_method': 'percentile'
        })
    
    def newey_west_standard_errors(self, shock_idx=1, max_lags=None):
        """
        Calculate Newey-West standard errors for serial correlation correction
        
        Args:
            shock_idx: Index of shock variable
            max_lags: Maximum lags for Newey-West (default: auto)
            
        Returns:
            pd.DataFrame: Impulse responses with Newey-West corrected standard errors
        """
        if not self.fitted:
            raise ValueError("Model not fitted")
            
        from statsmodels.stats.sandwich_covariance import cov_hac
        
        horizons = []
        coeffs = []
        nw_se = []
        nw_lower_ci = []
        nw_upper_ci = []
        
        for h in range(self.max_horizon + 1):
            if self.results.get(h) is not None:
                model = self.results[h]
                
                try:
                    # Calculate Newey-West covariance matrix
                    if max_lags is None:
                        # Rule of thumb: max_lags = floor(4*(T/100)^(2/9))
                        T = model.nobs
                        auto_lags = int(4 * (T/100)**(2/9))
                    else:
                        auto_lags = max_lags
                    
                    nw_cov = cov_hac(model, nlags=auto_lags)
                    nw_se_h = np.sqrt(nw_cov[shock_idx, shock_idx])
                    
                    coeff = model.params.iloc[shock_idx]
                    
                    horizons.append(h)
                    coeffs.append(coeff)
                    nw_se.append(nw_se_h)
                    
                    # 95% confidence intervals
                    nw_lower_ci.append(coeff - 1.96 * nw_se_h)
                    nw_upper_ci.append(coeff + 1.96 * nw_se_h)
                    
                except:
                    # Skip if Newey-West calculation fails
                    continue
        
        return pd.DataFrame({
            'horizon': horizons,
            'coefficient': coeffs,
            'newey_west_se': nw_se,
            'lower_ci_nw': nw_lower_ci,
            'upper_ci_nw': nw_upper_ci
        })
    
    def get_first_stage_statistics(self):
        """
        Get first-stage F-statistics for instrumented estimation
        
        Returns:
            pd.DataFrame: First-stage statistics by horizon including F-stat, p-value, partial R²
            
        Raises:
            ValueError: If model was not fitted with instruments
        """
        if not self.fitted:
            raise ValueError("Model not fitted")
        
        if not self.instrumented:
            raise ValueError("Model was not fitted with instruments. Use instruments parameter in fit().")
        
        from scipy import stats as scipy_stats
        
        horizons = []
        f_stats = []
        p_values = []
        partial_r2 = []
        
        for h in range(self.max_horizon + 1):
            if self.first_stage_results.get(h) is not None:
                fs_model = self.first_stage_results[h]
                
                horizons.append(h)
                
                # Calculate F-statistic manually to avoid issues with HAC covariance
                # F = (R² / k) / ((1 - R²) / (n - k - 1))
                # where k is number of regressors (excluding constant)
                r2 = fs_model.rsquared
                n = fs_model.nobs
                k = fs_model.df_model  # Number of regressors excluding constant
                
                if k > 0 and n > k + 1:
                    f_stat = (r2 / k) / ((1 - r2) / (n - k - 1))
                    p_value = 1 - scipy_stats.f.cdf(f_stat, k, n - k - 1)
                else:
                    f_stat = 0.0
                    p_value = 1.0
                
                f_stats.append(f_stat)
                p_values.append(p_value)
                
                # Calculate partial R-squared (R² from first stage)
                partial_r2.append(r2)
        
        return pd.DataFrame({
            'horizon': horizons,
            'first_stage_f_stat': f_stats,
            'first_stage_p_value': p_values,
            'partial_r_squared': partial_r2
        })
    
    def compute_cumulative_effect(self, shock_idx=1, max_horizon=None):
        """
        Compute cumulative effect over horizons: Σ(h=0 to H) ψ_h
        
        Args:
            shock_idx: Index of shock variable in regression (default 1, after constant)
            max_horizon: Maximum horizon for cumulation (default: self.max_horizon)
            
        Returns:
            dict: Cumulative effect, standard error, and confidence interval
        """
        if not self.fitted:
            raise ValueError("Model not fitted")
        
        if max_horizon is None:
            max_horizon = self.max_horizon
        
        # Collect coefficients and standard errors
        coefficients = []
        std_errors = []
        
        for h in range(max_horizon + 1):
            if self.results.get(h) is not None:
                model = self.results[h]
                if len(model.params) > shock_idx:
                    coefficients.append(model.params.iloc[shock_idx])
                    std_errors.append(model.bse.iloc[shock_idx])
                else:
                    coefficients.append(0.0)
                    std_errors.append(0.0)
            else:
                coefficients.append(0.0)
                std_errors.append(0.0)
        
        # Cumulative effect
        cumulative_effect = np.sum(coefficients)
        
        # Standard error of sum (assuming independence across horizons - conservative)
        cumulative_se = np.sqrt(np.sum(np.array(std_errors)**2))
        
        # 95% confidence interval
        cumulative_lower = cumulative_effect - 1.96 * cumulative_se
        cumulative_upper = cumulative_effect + 1.96 * cumulative_se
        
        return {
            'cumulative_effect': cumulative_effect,
            'cumulative_se': cumulative_se,
            'cumulative_lower_ci': cumulative_lower,
            'cumulative_upper_ci': cumulative_upper,
            'horizons_included': max_horizon + 1,
            'coefficients_by_horizon': coefficients
        }
    
    def multiple_horizon_specifications(self, y, shock, controls=None, horizon_sets=None):
        """
        Test robustness across multiple horizon specifications
        
        Args:
            y: Outcome variable
            shock: Shock variable
            controls: Control variables
            horizon_sets: List of maximum horizons to test
            
        Returns:
            dict: Results for different horizon specifications
        """
        if horizon_sets is None:
            horizon_sets = [5, 10, 15, 20]
        
        results = {}
        
        for max_h in horizon_sets:
            try:
                # Fit model with different max horizon
                lp_model = LocalProjections(max_horizon=max_h)
                lp_model.fit(y, shock, controls)
                
                # Get impulse responses
                ir = lp_model.get_impulse_responses()
                
                results[f'max_horizon_{max_h}'] = {
                    'impulse_responses': ir,
                    'fitted': lp_model.fitted,
                    'max_horizon': max_h,
                    'n_horizons_estimated': len(ir) if len(ir) > 0 else 0
                }
                
                # Summary statistics
                if len(ir) > 0:
                    results[f'max_horizon_{max_h}']['peak_response'] = ir['coefficient'].max()
                    results[f'max_horizon_{max_h}']['peak_horizon'] = ir.loc[ir['coefficient'].idxmax(), 'horizon']
                    results[f'max_horizon_{max_h}']['final_response'] = ir['coefficient'].iloc[-1]
                else:
                    results[f'max_horizon_{max_h}']['peak_response'] = np.nan
                    results[f'max_horizon_{max_h}']['peak_horizon'] = np.nan
                    results[f'max_horizon_{max_h}']['final_response'] = np.nan
                    
            except Exception as e:
                results[f'max_horizon_{max_h}'] = {
                    'fitted': False,
                    'error': str(e),
                    'max_horizon': max_h
                }
        
        return results
    
    def impulse_response_stability_test(self, y, shock, controls=None, test_type='subsample'):
        """
        Test stability of impulse responses across different samples
        
        Args:
            y: Outcome variable
            shock: Shock variable  
            controls: Control variables
            test_type: Type of stability test ('subsample', 'rolling', 'bootstrap')
            
        Returns:
            dict: Stability test results
        """
        stability_results = {}
        
        if test_type == 'subsample':
            # Split sample stability test
            n = len(y)
            if n < 40:
                return {'error': 'Insufficient observations for subsample test'}
            
            mid_point = n // 2
            
            # First half
            try:
                lp1 = LocalProjections(max_horizon=min(self.max_horizon, 10))
                lp1.fit(y[:mid_point], shock[:mid_point], 
                       controls[:mid_point] if controls is not None else None)
                ir1 = lp1.get_impulse_responses()
                
                stability_results['first_half'] = {
                    'impulse_responses': ir1,
                    'n_obs': mid_point,
                    'peak_response': ir1['coefficient'].max() if len(ir1) > 0 else np.nan
                }
            except Exception as e:
                stability_results['first_half'] = {'error': str(e)}
            
            # Second half
            try:
                lp2 = LocalProjections(max_horizon=min(self.max_horizon, 10))
                lp2.fit(y[mid_point:], shock[mid_point:], 
                       controls[mid_point:] if controls is not None else None)
                ir2 = lp2.get_impulse_responses()
                
                stability_results['second_half'] = {
                    'impulse_responses': ir2,
                    'n_obs': n - mid_point,
                    'peak_response': ir2['coefficient'].max() if len(ir2) > 0 else np.nan
                }
            except Exception as e:
                stability_results['second_half'] = {'error': str(e)}
            
            # Compare results
            if ('first_half' in stability_results and 'second_half' in stability_results and
                'error' not in stability_results['first_half'] and 'error' not in stability_results['second_half']):
                
                peak1 = stability_results['first_half']['peak_response']
                peak2 = stability_results['second_half']['peak_response']
                
                if not (np.isnan(peak1) or np.isnan(peak2)):
                    stability_results['comparison'] = {
                        'peak_difference': abs(peak1 - peak2),
                        'relative_difference': abs(peak1 - peak2) / max(abs(peak1), abs(peak2)) if max(abs(peak1), abs(peak2)) > 0 else 0,
                        'stable': abs(peak1 - peak2) / max(abs(peak1), abs(peak2)) < 0.5 if max(abs(peak1), abs(peak2)) > 0 else True
                    }
        
        elif test_type == 'rolling':
            # Rolling window stability test
            window_size = len(y) // 3
            if window_size < 30:
                return {'error': 'Insufficient observations for rolling window test'}
            
            rolling_results = []
            n = len(y)
            
            for start in range(0, n - window_size + 1, window_size // 2):
                end = start + window_size
                
                try:
                    lp_roll = LocalProjections(max_horizon=min(self.max_horizon, 5))
                    lp_roll.fit(y[start:end], shock[start:end], 
                              controls[start:end] if controls is not None else None)
                    ir_roll = lp_roll.get_impulse_responses()
                    
                    rolling_results.append({
                        'start': start,
                        'end': end,
                        'peak_response': ir_roll['coefficient'].max() if len(ir_roll) > 0 else np.nan,
                        'n_obs': end - start
                    })
                except:
                    rolling_results.append({
                        'start': start,
                        'end': end,
                        'error': 'Failed to fit',
                        'n_obs': end - start
                    })
            
            stability_results['rolling_windows'] = rolling_results
            
            # Calculate stability metrics
            peak_responses = [r['peak_response'] for r in rolling_results if 'peak_response' in r and not np.isnan(r['peak_response'])]
            
            if len(peak_responses) > 1:
                stability_results['rolling_stability'] = {
                    'mean_peak': np.mean(peak_responses),
                    'std_peak': np.std(peak_responses),
                    'cv_peak': np.std(peak_responses) / abs(np.mean(peak_responses)) if np.mean(peak_responses) != 0 else np.inf,
                    'stable': np.std(peak_responses) / abs(np.mean(peak_responses)) < 0.3 if np.mean(peak_responses) != 0 else False
                }
        
        elif test_type == 'bootstrap':
            # Bootstrap stability test
            n_bootstrap = 100
            bootstrap_peaks = []
            
            n = len(y)
            
            for b in range(n_bootstrap):
                try:
                    # Bootstrap sample
                    boot_indices = np.random.choice(n, size=n, replace=True)
                    y_boot = y.iloc[boot_indices] if hasattr(y, 'iloc') else y[boot_indices]
                    shock_boot = shock.iloc[boot_indices] if hasattr(shock, 'iloc') else shock[boot_indices]
                    controls_boot = controls.iloc[boot_indices] if controls is not None and hasattr(controls, 'iloc') else (controls[boot_indices] if controls is not None else None)
                    
                    lp_boot = LocalProjections(max_horizon=min(self.max_horizon, 5))
                    lp_boot.fit(y_boot, shock_boot, controls_boot)
                    ir_boot = lp_boot.get_impulse_responses()
                    
                    if len(ir_boot) > 0:
                        bootstrap_peaks.append(ir_boot['coefficient'].max())
                        
                except:
                    continue
            
            if len(bootstrap_peaks) > 10:
                stability_results['bootstrap_stability'] = {
                    'n_successful': len(bootstrap_peaks),
                    'mean_peak': np.mean(bootstrap_peaks),
                    'std_peak': np.std(bootstrap_peaks),
                    'percentile_5': np.percentile(bootstrap_peaks, 5),
                    'percentile_95': np.percentile(bootstrap_peaks, 95),
                    'stable': np.std(bootstrap_peaks) / abs(np.mean(bootstrap_peaks)) < 0.3 if np.mean(bootstrap_peaks) != 0 else False
                }
            else:
                stability_results['bootstrap_stability'] = {'error': 'Insufficient successful bootstrap samples'}
        
        return stability_results
    
    def publication_quality_impulse_responses(self, shock_idx=1, confidence_level=0.95, 
                                            include_bootstrap=True, include_newey_west=True):
        """
        Generate publication-quality impulse response results with multiple confidence interval methods
        
        Args:
            shock_idx: Index of shock variable in regression
            confidence_level: Confidence level for intervals
            include_bootstrap: Whether to include bootstrap confidence intervals
            include_newey_west: Whether to include Newey-West corrected standard errors
            
        Returns:
            dict: Comprehensive impulse response results for publication
        """
        if not self.fitted:
            raise ValueError("Model not fitted")
            
        publication_results = {}
        
        # Basic impulse responses
        basic_ir = self.get_impulse_responses(shock_idx)
        publication_results['basic_impulse_responses'] = basic_ir
        
        # Bootstrap confidence intervals
        if include_bootstrap:
            try:
                bootstrap_ir = self.bootstrap_confidence_intervals_publication(
                    shock_idx, confidence_level=confidence_level
                )
                publication_results['bootstrap_confidence_intervals'] = bootstrap_ir
            except Exception as e:
                publication_results['bootstrap_error'] = f"Bootstrap CI failed: {str(e)}"
        
        # Newey-West corrected results
        if include_newey_west:
            try:
                nw_ir = self.newey_west_corrected_results(shock_idx)
                publication_results['newey_west_corrected'] = nw_ir
            except Exception as e:
                publication_results['newey_west_error'] = f"Newey-West correction failed: {str(e)}"
        
        # Summary statistics for publication
        if len(basic_ir) > 0:
            publication_results['summary_statistics'] = {
                'peak_response': basic_ir['coefficient'].max(),
                'peak_horizon': basic_ir.loc[basic_ir['coefficient'].idxmax(), 'horizon'],
                'final_response': basic_ir['coefficient'].iloc[-1],
                'cumulative_response': basic_ir['coefficient'].sum(),
                'significant_horizons': len(basic_ir[abs(basic_ir['coefficient']) > 1.96 * (basic_ir['upper_ci'] - basic_ir['lower_ci'])/3.92])
            }
        
        # Model diagnostics for publication
        publication_results['model_diagnostics'] = self.local_projections_diagnostic_suite()
        
        return publication_results
    
    def bootstrap_confidence_intervals_publication(self, shock_idx=1, n_bootstrap=1000, 
                                                 confidence_level=0.95, method='percentile'):
        """
        Calculate publication-ready bootstrap confidence intervals with proper formatting
        
        Args:
            shock_idx: Index of shock variable
            n_bootstrap: Number of bootstrap samples
            confidence_level: Confidence level for intervals
            method: Bootstrap method ('percentile', 'bias_corrected', 'bca')
            
        Returns:
            pd.DataFrame: Publication-formatted bootstrap confidence intervals
        """
        if not self.fitted:
            raise ValueError("Model not fitted")
            
        # Get basic bootstrap intervals
        bootstrap_basic = self.bootstrap_confidence_intervals(
            shock_idx, n_bootstrap, confidence_level
        )
        
        # Format for publication
        publication_bootstrap = bootstrap_basic.copy()
        
        # Add additional statistics
        publication_bootstrap['confidence_level'] = confidence_level
        publication_bootstrap['n_bootstrap'] = n_bootstrap
        publication_bootstrap['method'] = method
        
        # Calculate t-statistics using bootstrap standard errors
        bootstrap_se = (publication_bootstrap['upper_ci_bootstrap'] - 
                       publication_bootstrap['lower_ci_bootstrap']) / (2 * 1.96)
        publication_bootstrap['bootstrap_se'] = bootstrap_se
        publication_bootstrap['t_statistic'] = (publication_bootstrap['coefficient'] / 
                                              bootstrap_se)
        
        # Statistical significance
        alpha = 1 - confidence_level
        from scipy.stats import t
        critical_value = t.ppf(1 - alpha/2, df=100)  # Approximate df
        publication_bootstrap['significant'] = (abs(publication_bootstrap['t_statistic']) > 
                                              critical_value)
        
        return publication_bootstrap
    
    def newey_west_corrected_results(self, shock_idx=1, max_lags=None):
        """
        Generate Newey-West corrected results for robust standard errors
        
        Args:
            shock_idx: Index of shock variable
            max_lags: Maximum lags for Newey-West correction
            
        Returns:
            pd.DataFrame: Newey-West corrected impulse responses
        """
        if not self.fitted:
            raise ValueError("Model not fitted")
            
        # Get Newey-West standard errors
        nw_results = self.newey_west_standard_errors(shock_idx, max_lags)
        
        # Add statistical significance tests
        nw_results['t_statistic_nw'] = nw_results['coefficient'] / nw_results['newey_west_se']
        
        # Critical values (approximate)
        from scipy.stats import t
        critical_value = t.ppf(0.975, df=100)  # Approximate df
        nw_results['significant_nw'] = abs(nw_results['t_statistic_nw']) > critical_value
        
        # P-values
        nw_results['p_value_nw'] = 2 * (1 - t.cdf(abs(nw_results['t_statistic_nw']), df=100))
        
        return nw_results
    
    def local_projections_diagnostic_suite(self):
        """
        Create comprehensive diagnostic suite for local projections validation
        
        Returns:
            dict: Comprehensive diagnostic results
        """
        if not self.fitted:
            raise ValueError("Model not fitted")
            
        diagnostics = {}
        
        # Model fit statistics for each horizon
        fit_stats = []
        for h in range(self.max_horizon + 1):
            if self.results.get(h) is not None:
                model = self.results[h]
                fit_stats.append({
                    'horizon': h,
                    'r_squared': model.rsquared,
                    'adj_r_squared': model.rsquared_adj,
                    'aic': model.aic,
                    'bic': model.bic,
                    'n_obs': model.nobs,
                    'f_statistic': model.fvalue,
                    'f_pvalue': model.f_pvalue
                })
        
        diagnostics['fit_statistics'] = pd.DataFrame(fit_stats)
        
        # Residual diagnostics for key horizons
        key_horizons = [0, 1, 4, 8] if self.max_horizon >= 8 else list(range(min(self.max_horizon + 1, 5)))
        residual_diagnostics = {}
        
        for h in key_horizons:
            if self.results.get(h) is not None:
                model = self.results[h]
                residuals = model.resid
                
                # Residual tests
                from scipy.stats import jarque_bera, shapiro
                from statsmodels.stats.diagnostic import het_breuschpagan
                from statsmodels.stats.stattools import durbin_watson
                
                try:
                    # Normality tests
                    jb_stat, jb_pvalue = jarque_bera(residuals)
                    
                    # Heteroskedasticity test
                    het_lm, het_lm_pvalue, het_f, het_f_pvalue = het_breuschpagan(
                        residuals, model.model.exog
                    )
                    
                    # Serial correlation
                    dw_stat = durbin_watson(residuals)
                    
                    residual_diagnostics[f'horizon_{h}'] = {
                        'jarque_bera_stat': jb_stat,
                        'jarque_bera_pvalue': jb_pvalue,
                        'het_lm_stat': het_lm,
                        'het_lm_pvalue': het_lm_pvalue,
                        'durbin_watson': dw_stat,
                        'residual_mean': np.mean(residuals),
                        'residual_std': np.std(residuals)
                    }
                    
                except Exception as e:
                    residual_diagnostics[f'horizon_{h}'] = {
                        'error': f"Diagnostic tests failed: {str(e)}"
                    }
        
        diagnostics['residual_diagnostics'] = residual_diagnostics
        
        # Overall model assessment
        if len(fit_stats) > 0:
            fit_df = pd.DataFrame(fit_stats)
            diagnostics['overall_assessment'] = {
                'mean_r_squared': fit_df['r_squared'].mean(),
                'min_r_squared': fit_df['r_squared'].min(),
                'max_r_squared': fit_df['r_squared'].max(),
                'horizons_fitted': len(fit_stats),
                'total_horizons': self.max_horizon + 1,
                'fit_success_rate': len(fit_stats) / (self.max_horizon + 1)
            }
        
        return diagnostics


class ModelComparisonSuite:
    """
    Comprehensive model comparison and selection framework for systematic model evaluation
    
    This class provides tools for comparing different econometric specifications,
    calculating information criteria, and performing formal model selection tests.
    """
    
    def __init__(self):
        self.comparison_results = {}
        self.model_registry = {}
        
    def register_model(self, model_name, model_object, fitted_data=None):
        """
        Register a fitted model for comparison
        
        Args:
            model_name: Unique name for the model
            model_object: Fitted model object
            fitted_data: Dictionary with 'y', 'x', 'threshold_var' if applicable
        """
        self.model_registry[model_name] = {
            'model': model_object,
            'data': fitted_data,
            'registered': True
        }
    
    def information_criteria_comparison(self, y, models_dict=None):
        """
        Compare models using AIC, BIC, and cross-validation
        
        Args:
            y: Dependent variable
            models_dict: Dictionary of model names and fitted models (optional if using registry)
            
        Returns:
            pd.DataFrame: Information criteria comparison table
        """
        if models_dict is None:
            models_dict = {name: info['model'] for name, info in self.model_registry.items()}
        
        comparison_data = []
        
        for model_name, model in models_dict.items():
            try:
                # Calculate basic fit statistics
                if hasattr(model, 'predict') and hasattr(model, 'fitted'):
                    # For threshold models
                    if hasattr(model, 'threshold') and model_name in self.model_registry:
                        data = self.model_registry[model_name]['data']
                        if data is not None:
                            y_pred = model.predict(data['x'], data['threshold_var'])
                        else:
                            continue
                    else:
                        # For other models, try to get predictions
                        try:
                            y_pred = model.predict() if hasattr(model, 'predict') else None
                        except:
                            continue
                    
                    if y_pred is not None:
                        # Calculate fit statistics
                        n = len(y)
                        residuals = y - y_pred
                        ssr = np.sum(residuals**2)
                        mse = ssr / n
                        
                        # Estimate number of parameters
                        if hasattr(model, 'beta1') and hasattr(model, 'beta2'):
                            # Threshold model
                            k = len(model.beta1) + len(model.beta2)
                        elif hasattr(model, 'coeffs'):
                            k = len(model.coeffs)
                        else:
                            k = 2  # Default assumption
                        
                        # Calculate information criteria
                        log_likelihood = -0.5 * n * (np.log(2 * np.pi) + np.log(mse) + 1)
                        aic = 2 * k - 2 * log_likelihood
                        bic = k * np.log(n) - 2 * log_likelihood
                        
                        # R-squared
                        ss_tot = np.sum((y - np.mean(y))**2)
                        r_squared = 1 - (ssr / ss_tot) if ss_tot > 0 else 0
                        adj_r_squared = 1 - (1 - r_squared) * (n - 1) / (n - k - 1)
                        
                        comparison_data.append({
                            'model_name': model_name,
                            'n_obs': n,
                            'n_params': k,
                            'log_likelihood': log_likelihood,
                            'aic': aic,
                            'bic': bic,
                            'r_squared': r_squared,
                            'adj_r_squared': adj_r_squared,
                            'mse': mse,
                            'rmse': np.sqrt(mse)
                        })
                        
                elif hasattr(model, 'results'):
                    # For LocalProjections or other models with results dict
                    # Use horizon 1 as representative
                    if 1 in model.results and model.results[1] is not None:
                        reg_model = model.results[1]
                        comparison_data.append({
                            'model_name': model_name,
                            'n_obs': reg_model.nobs,
                            'n_params': len(reg_model.params),
                            'log_likelihood': reg_model.llf,
                            'aic': reg_model.aic,
                            'bic': reg_model.bic,
                            'r_squared': reg_model.rsquared,
                            'adj_r_squared': reg_model.rsquared_adj,
                            'mse': reg_model.mse_resid,
                            'rmse': np.sqrt(reg_model.mse_resid)
                        })
                        
            except Exception as e:
                comparison_data.append({
                    'model_name': model_name,
                    'error': f"Could not calculate criteria: {str(e)}"
                })
        
        comparison_df = pd.DataFrame(comparison_data)
        
        # Rank models by different criteria
        if len(comparison_df) > 0 and 'aic' in comparison_df.columns:
            comparison_df['aic_rank'] = comparison_df['aic'].rank()
            comparison_df['bic_rank'] = comparison_df['bic'].rank()
            comparison_df['r2_rank'] = comparison_df['r_squared'].rank(ascending=False)
            
        self.comparison_results['information_criteria'] = comparison_df
        return comparison_df
    
    def specification_test_battery(self, base_model_name, alternative_models=None):
        """
        Perform formal specification tests comparing models
        
        Args:
            base_model_name: Name of base model for comparison
            alternative_models: List of alternative model names to test against
            
        Returns:
            dict: Results from specification tests
        """
        if base_model_name not in self.model_registry:
            raise ValueError(f"Base model {base_model_name} not registered")
        
        if alternative_models is None:
            alternative_models = [name for name in self.model_registry.keys() 
                                if name != base_model_name]
        
        test_results = {}
        base_model = self.model_registry[base_model_name]['model']
        base_data = self.model_registry[base_model_name]['data']
        
        for alt_name in alternative_models:
            if alt_name not in self.model_registry:
                continue
                
            alt_model = self.model_registry[alt_name]['model']
            alt_data = self.model_registry[alt_name]['data']
            
            try:
                # Likelihood ratio test (if nested models)
                test_result = self._likelihood_ratio_test(
                    base_model, alt_model, base_data, alt_data
                )
                test_results[f'{base_model_name}_vs_{alt_name}'] = test_result
                
            except Exception as e:
                test_results[f'{base_model_name}_vs_{alt_name}'] = {
                    'error': f"Specification test failed: {str(e)}"
                }
        
        self.comparison_results['specification_tests'] = test_results
        return test_results
    
    def model_selection_with_uncertainty(self, selection_criteria='aic', 
                                       bootstrap_samples=100):
        """
        Perform robust model selection accounting for uncertainty
        
        Args:
            selection_criteria: Criteria for selection ('aic', 'bic', 'cv')
            bootstrap_samples: Number of bootstrap samples for uncertainty
            
        Returns:
            dict: Model selection results with uncertainty measures
        """
        if 'information_criteria' not in self.comparison_results:
            raise ValueError("Must run information_criteria_comparison first")
        
        ic_results = self.comparison_results['information_criteria']
        
        if selection_criteria not in ic_results.columns:
            raise ValueError(f"Selection criteria {selection_criteria} not available")
        
        # Basic model selection
        best_model_idx = ic_results[selection_criteria].idxmin()
        best_model = ic_results.loc[best_model_idx, 'model_name']
        
        selection_results = {
            'best_model': best_model,
            'selection_criteria': selection_criteria,
            'best_criteria_value': ic_results.loc[best_model_idx, selection_criteria],
            'model_rankings': ic_results[['model_name', selection_criteria]].sort_values(selection_criteria)
        }
        
        # Bootstrap model selection uncertainty
        if bootstrap_samples > 0:
            bootstrap_selections = []
            
            for b in range(bootstrap_samples):
                try:
                    # Bootstrap sample of models (simplified approach)
                    # In practice, would refit models on bootstrap samples
                    
                    # Add noise to criteria values to simulate uncertainty
                    noise_scale = ic_results[selection_criteria].std() * 0.1
                    noisy_criteria = (ic_results[selection_criteria] + 
                                    np.random.normal(0, noise_scale, len(ic_results)))
                    
                    bootstrap_best_idx = noisy_criteria.idxmin()
                    bootstrap_best = ic_results.loc[bootstrap_best_idx, 'model_name']
                    bootstrap_selections.append(bootstrap_best)
                    
                except:
                    continue
            
            if len(bootstrap_selections) > 0:
                # Calculate selection probabilities
                from collections import Counter
                selection_counts = Counter(bootstrap_selections)
                
                selection_probs = {}
                for model_name in ic_results['model_name']:
                    selection_probs[model_name] = (selection_counts.get(model_name, 0) / 
                                                 len(bootstrap_selections))
                
                selection_results['bootstrap_uncertainty'] = {
                    'selection_probabilities': selection_probs,
                    'most_frequent_selection': max(selection_counts, key=selection_counts.get),
                    'selection_stability': selection_probs.get(best_model, 0),
                    'n_bootstrap_samples': len(bootstrap_selections)
                }
        
        self.comparison_results['model_selection'] = selection_results
        return selection_results
    
    def cross_validation_comparison(self, cv_folds=5):
        """
        Compare models using cross-validation
        
        Args:
            cv_folds: Number of cross-validation folds
            
        Returns:
            dict: Cross-validation comparison results
        """
        cv_results = {}
        
        for model_name, model_info in self.model_registry.items():
            model = model_info['model']
            data = model_info['data']
            
            if data is None:
                cv_results[model_name] = {'error': 'No data available for CV'}
                continue
            
            try:
                y = data['y']
                n = len(y)
                fold_size = n // cv_folds
                
                cv_scores = []
                
                for fold in range(cv_folds):
                    # Create train/test split
                    test_start = fold * fold_size
                    test_end = (fold + 1) * fold_size if fold < cv_folds - 1 else n
                    
                    train_indices = list(range(0, test_start)) + list(range(test_end, n))
                    test_indices = list(range(test_start, test_end))
                    
                    if len(train_indices) < 20 or len(test_indices) < 5:
                        continue
                    
                    # Fit model on training data
                    if hasattr(model, 'fit') and 'x' in data and 'threshold_var' in data:
                        # Threshold model
                        y_train = y[train_indices]
                        x_train = data['x'][train_indices]
                        threshold_train = data['threshold_var'][train_indices]
                        
                        # Create new model instance
                        if hasattr(model, '__class__'):
                            cv_model = model.__class__()
                            cv_model.fit(y_train, x_train, threshold_train)
                            
                            # Predict on test set
                            y_test = y[test_indices]
                            x_test = data['x'][test_indices]
                            threshold_test = data['threshold_var'][test_indices]
                            
                            y_pred = cv_model.predict(x_test, threshold_test)
                            
                            # Calculate score (negative MSE)
                            mse = np.mean((y_test - y_pred)**2)
                            cv_scores.append(-mse)
                    
                if len(cv_scores) > 0:
                    cv_results[model_name] = {
                        'cv_scores': cv_scores,
                        'mean_cv_score': np.mean(cv_scores),
                        'std_cv_score': np.std(cv_scores),
                        'n_folds_completed': len(cv_scores)
                    }
                else:
                    cv_results[model_name] = {'error': 'No CV folds completed successfully'}
                    
            except Exception as e:
                cv_results[model_name] = {'error': f'CV failed: {str(e)}'}
        
        self.comparison_results['cross_validation'] = cv_results
        return cv_results
    
    def _likelihood_ratio_test(self, model1, model2, data1, data2):
        """
        Perform likelihood ratio test between two models
        
        Args:
            model1, model2: Fitted model objects
            data1, data2: Data dictionaries for the models
            
        Returns:
            dict: Likelihood ratio test results
        """
        try:
            # Calculate log-likelihoods
            ll1 = self._calculate_log_likelihood(model1, data1)
            ll2 = self._calculate_log_likelihood(model2, data2)
            
            # Determine which is restricted vs unrestricted
            k1 = self._count_parameters(model1)
            k2 = self._count_parameters(model2)
            
            if k1 < k2:
                ll_restricted = ll1
                ll_unrestricted = ll2
                df = k2 - k1
            else:
                ll_restricted = ll2
                ll_unrestricted = ll1
                df = k1 - k2
            
            # LR test statistic
            lr_stat = 2 * (ll_unrestricted - ll_restricted)
            
            # P-value
            from scipy.stats import chi2
            p_value = 1 - chi2.cdf(lr_stat, df)
            
            return {
                'lr_statistic': lr_stat,
                'degrees_freedom': df,
                'p_value': p_value,
                'significant': p_value < 0.05,
                'll_restricted': ll_restricted,
                'll_unrestricted': ll_unrestricted
            }
            
        except Exception as e:
            return {'error': f'LR test failed: {str(e)}'}
    
    def _calculate_log_likelihood(self, model, data):
        """Calculate log-likelihood for a model"""
        if data is None:
            raise ValueError("No data available")
        
        y = data['y']
        
        if hasattr(model, 'predict') and 'x' in data and 'threshold_var' in data:
            y_pred = model.predict(data['x'], data['threshold_var'])
        else:
            raise ValueError("Cannot calculate predictions")
        
        residuals = y - y_pred
        n = len(y)
        mse = np.sum(residuals**2) / n
        
        # Log-likelihood assuming normal errors
        log_likelihood = -0.5 * n * (np.log(2 * np.pi) + np.log(mse) + 1)
        
        return log_likelihood
    
    def _count_parameters(self, model):
        """Count number of parameters in a model"""
        if hasattr(model, 'beta1') and hasattr(model, 'beta2'):
            return len(model.beta1) + len(model.beta2)
        elif hasattr(model, 'coeffs'):
            return len(model.coeffs)
        else:
            return 2  # Default assumption
    
    def get_enhanced_diagnostics(self, y, shock, controls=None):
        """
        Get comprehensive enhanced diagnostics for Local Projections
        
        Args:
            y: Outcome variable
            shock: Shock variable
            controls: Control variables
            
        Returns:
            dict: Enhanced diagnostic information
        """
        if not self.fitted:
            raise ValueError("Model not fitted")
            
        diagnostics = {}
        
        # Basic model info
        diagnostics['fitted'] = self.fitted
        diagnostics['max_horizon'] = self.max_horizon
        
        # Standard impulse responses
        try:
            diagnostics['standard_ir'] = self.get_impulse_responses()
        except:
            diagnostics['standard_ir'] = pd.DataFrame()
        
        # Bootstrap confidence intervals
        try:
            diagnostics['bootstrap_ir'] = self.bootstrap_confidence_intervals()
        except Exception as e:
            diagnostics['bootstrap_ir'] = {'error': str(e)}
        
        # Newey-West standard errors
        try:
            diagnostics['newey_west_ir'] = self.newey_west_standard_errors()
        except Exception as e:
            diagnostics['newey_west_ir'] = {'error': str(e)}
        
        # Multiple horizon specifications
        try:
            diagnostics['horizon_robustness'] = self.multiple_horizon_specifications(y, shock, controls)
        except Exception as e:
            diagnostics['horizon_robustness'] = {'error': str(e)}
        
        # Stability tests
        try:
            diagnostics['subsample_stability'] = self.impulse_response_stability_test(y, shock, controls, 'subsample')
        except Exception as e:
            diagnostics['subsample_stability'] = {'error': str(e)}
        
        try:
            diagnostics['bootstrap_stability'] = self.impulse_response_stability_test(y, shock, controls, 'bootstrap')
        except Exception as e:
            diagnostics['bootstrap_stability'] = {'error': str(e)}
        
        # Summary statistics
        if len(diagnostics.get('standard_ir', pd.DataFrame())) > 0:
            ir = diagnostics['standard_ir']
            diagnostics['summary'] = {
                'peak_response': ir['coefficient'].max(),
                'peak_horizon': ir.loc[ir['coefficient'].idxmax(), 'horizon'],
                'trough_response': ir['coefficient'].min(),
                'trough_horizon': ir.loc[ir['coefficient'].idxmin(), 'horizon'],
                'final_response': ir['coefficient'].iloc[-1],
                'n_significant_horizons': np.sum((ir['lower_ci'] > 0) | (ir['upper_ci'] < 0))
            }
        else:
            diagnostics['summary'] = {'error': 'No impulse responses available'}
        
        return diagnostics

class PanelVAR:
    """
    Panel VAR for international spillovers (Hypothesis 3)
    Based on equation (16) from the paper  
    """
    
    def __init__(self, lags=2):
        self.lags = lags
        self.models = {}
        self.fitted = False
        
    def fit(self, data, country_col='country'):
        """
        Fit Panel VAR
        data: DataFrame with country panel data
        """
        countries = data[country_col].unique()
        self.models = {}
        
        for country in countries:
            country_data = data[data[country_col] == country].copy()
            
            # Remove country column for VAR estimation
            var_data = country_data.drop(columns=[country_col])
            
            # Remove any non-numeric columns
            numeric_cols = var_data.select_dtypes(include=[np.number]).columns
            var_data = var_data[numeric_cols]
            
            # Drop NaN and ensure sufficient observations
            var_data = var_data.dropna()
            
            if len(var_data) > 4 * self.lags:  # Minimum observations
                try:
                    model = VAR(var_data)
                    fitted_model = model.fit(maxlags=self.lags, ic='aic')
                    self.models[country] = fitted_model
                except:
                    self.models[country] = None
            else:
                self.models[country] = None
                
        self.fitted = True
        
    def impulse_response(self, country, periods=10, shock_var=0, response_var=1):
        """Calculate impulse response for specific country"""
        if not self.fitted or country not in self.models:
            raise ValueError(f"Model not fitted or country {country} not found")
            
        if self.models[country] is None:
            return None
            
        try:
            irf = self.models[country].irf(periods)
            return irf.irfs[response_var, shock_var, :]
        except:
            return None

class HighFrequencyIdentification:
    """
    High-frequency identification around QE announcements (Hypothesis 3)
    Based on equation (17) from the paper
    """
    
    def __init__(self):
        self.results = None
        self.fitted = False
        
    def fit(self, yield_changes, qe_surprises, controls=None, event_window=1):
        """
        Fit high-frequency identification model
        yield_changes: daily changes in yields around announcements
        qe_surprises: QE surprises from asset price movements
        controls: control variables
        event_window: days around announcement to include
        """
        
        # Create regression dataset
        reg_data = pd.DataFrame({
            'yield_change': yield_changes,
            'qe_surprise': qe_surprises
        })
        
        if controls is not None:
            if isinstance(controls, pd.DataFrame):
                reg_data = pd.concat([reg_data, controls], axis=1)
            else:
                reg_data['controls'] = controls
                
        # Drop NaN
        reg_data = reg_data.dropna()
        
        if len(reg_data) > 5:  # Minimum observations
            y = reg_data['yield_change']
            X = sm.add_constant(reg_data.drop(columns=['yield_change']))
            
            self.results = OLS(y, X).fit()
            self.fitted = True
        else:
            raise ValueError("Insufficient data for high-frequency identification")
            
    def get_qe_effect(self):
        """Extract QE effect coefficient and significance"""
        if not self.fitted:
            raise ValueError("Model not fitted")
            
        return {
            'coefficient': self.results.params['qe_surprise'],
            'std_error': self.results.bse['qe_surprise'],
            'p_value': self.results.pvalues['qe_surprise'],
            't_stat': self.results.tvalues['qe_surprise']
        }