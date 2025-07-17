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
    """
    
    def __init__(self, max_horizon=20):
        self.max_horizon = max_horizon
        self.results = {}
        self.fitted = False
        
    def fit(self, y, shock, controls=None, lags=4):
        """
        Fit local projections
        y: outcome variable
        shock: QE shock variable
        controls: additional control variables
        """
        self.results = {}
        
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
                    
            # Combine and drop NaN
            reg_data = pd.concat([y_diff] + X_reg, axis=1).dropna()
            
            if len(reg_data) > 10:  # Minimum observations
                y_reg = reg_data.iloc[:, 0]
                X_reg_clean = sm.add_constant(reg_data.iloc[:, 1:])
                
                try:
                    model = OLS(y_reg, X_reg_clean).fit()
                    self.results[h] = model
                except:
                    self.results[h] = None
            else:
                self.results[h] = None
                
        self.fitted = True
        
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