"""
Machine Learning Models for QE Hypothesis Testing Robustness

This module implements Random Forest, Gradient Boosting, and Neural Network models
for non-parametric analysis and robustness testing of QE hypotheses.
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, field
import warnings
warnings.filterwarnings('ignore')

# ML libraries
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit, cross_val_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.inspection import partial_dependence, PartialDependenceDisplay
import matplotlib.pyplot as plt
import seaborn as sns

# Deep learning
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logging.warning("PyTorch not available. Neural network models will be disabled.")

# SHAP for interpretability
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    logging.warning("SHAP not available. Model interpretability features will be limited.")


@dataclass
class MLModelConfig:
    """Configuration for ML models"""
    
    # Random Forest settings
    rf_n_estimators: int = 100
    rf_max_depth: Optional[int] = None
    rf_min_samples_split: int = 2
    rf_min_samples_leaf: int = 1
    rf_random_state: int = 42
    rf_n_jobs: int = -1
    
    # Gradient Boosting settings
    gb_n_estimators: int = 100
    gb_learning_rate: float = 0.1
    gb_max_depth: int = 3
    gb_min_samples_split: int = 2
    gb_min_samples_leaf: int = 1
    gb_random_state: int = 42
    
    # Neural Network settings
    nn_hidden_layers: List[int] = field(default_factory=lambda: [64, 32, 16])
    nn_dropout_rate: float = 0.2
    nn_learning_rate: float = 0.001
    nn_batch_size: int = 32
    nn_epochs: int = 100
    nn_early_stopping_patience: int = 10
    nn_random_state: int = 42
    
    # Cross-validation settings
    cv_folds: int = 5
    cv_scoring: str = 'neg_mean_squared_error'
    
    # Feature importance settings
    feature_importance_threshold: float = 0.01
    partial_dependence_features: Optional[List[str]] = None
    
    # Ensemble settings
    ensemble_weights: Optional[Dict[str, float]] = None
    uncertainty_quantiles: List[float] = field(default_factory=lambda: [0.05, 0.95])


@dataclass
class MLModelResults:
    """Results from ML model fitting"""
    
    model_name: str
    fitted: bool = False
    
    # Model performance
    train_score: Optional[float] = None
    test_score: Optional[float] = None
    cv_scores: Optional[List[float]] = None
    cv_mean: Optional[float] = None
    cv_std: Optional[float] = None
    
    # Predictions
    train_predictions: Optional[np.ndarray] = None
    test_predictions: Optional[np.ndarray] = None
    prediction_intervals: Optional[Dict[str, np.ndarray]] = None
    
    # Feature importance
    feature_importance: Optional[pd.DataFrame] = None
    feature_names: Optional[List[str]] = None
    
    # Model-specific results
    model_specific: Dict[str, Any] = field(default_factory=dict)
    
    # Diagnostics
    residuals: Optional[np.ndarray] = None
    fitted_values: Optional[np.ndarray] = None
    
    # Metadata
    training_time: Optional[float] = None
    hyperparameters: Optional[Dict[str, Any]] = None


class RandomForestThresholdDetector:
    """
    Random Forest model for non-parametric threshold detection and analysis.
    
    Implements Requirements 5.1, 5.2, 5.6 for Random Forest models with
    feature importance analysis and partial dependence plots.
    """
    
    def __init__(self, config: Optional[MLModelConfig] = None):
        """
        Initialize Random Forest threshold detector.
        
        Args:
            config: MLModelConfig for model parameters
        """
        self.config = config or MLModelConfig()
        self.logger = logging.getLogger(__name__)
        
        # Initialize model
        self.model = RandomForestRegressor(
            n_estimators=self.config.rf_n_estimators,
            max_depth=self.config.rf_max_depth,
            min_samples_split=self.config.rf_min_samples_split,
            min_samples_leaf=self.config.rf_min_samples_leaf,
            random_state=self.config.rf_random_state,
            n_jobs=self.config.rf_n_jobs
        )
        
        # Storage
        self.scaler = StandardScaler()
        self.fitted = False
        self.feature_names = None
        self.results = MLModelResults(model_name="RandomForest")
        
    def fit(self, 
            X: Union[np.ndarray, pd.DataFrame], 
            y: Union[np.ndarray, pd.Series],
            feature_names: Optional[List[str]] = None,
            test_size: float = 0.2,
            hyperparameter_tuning: bool = True) -> MLModelResults:
        """
        Fit Random Forest model for threshold detection.
        
        Args:
            X: Feature matrix
            y: Target variable
            feature_names: Names of features
            test_size: Proportion of data for testing
            hyperparameter_tuning: Whether to perform hyperparameter optimization
            
        Returns:
            MLModelResults object with fitting results
        """
        import time
        start_time = time.time()
        
        self.logger.info("Fitting Random Forest threshold detector")
        
        # Convert to numpy arrays if needed
        if isinstance(X, pd.DataFrame):
            self.feature_names = X.columns.tolist()
            X = X.values
        else:
            self.feature_names = feature_names or [f"feature_{i}" for i in range(X.shape[1])]
        
        if isinstance(y, pd.Series):
            y = y.values
        
        # Split data (time series aware)
        split_idx = int(len(X) * (1 - test_size))
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Hyperparameter tuning if requested
        if hyperparameter_tuning:
            self.logger.info("Performing hyperparameter tuning")
            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [None, 10, 20, 30],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            }
            
            # Use TimeSeriesSplit for time series data
            tscv = TimeSeriesSplit(n_splits=self.config.cv_folds)
            
            grid_search = GridSearchCV(
                self.model, 
                param_grid, 
                cv=tscv,
                scoring=self.config.cv_scoring,
                n_jobs=self.config.rf_n_jobs,
                verbose=0
            )
            
            grid_search.fit(X_train_scaled, y_train)
            self.model = grid_search.best_estimator_
            self.results.hyperparameters = grid_search.best_params_
            
            self.logger.info(f"Best parameters: {grid_search.best_params_}")
        
        # Fit final model
        self.model.fit(X_train_scaled, y_train)
        
        # Make predictions
        train_pred = self.model.predict(X_train_scaled)
        test_pred = self.model.predict(X_test_scaled)
        
        # Calculate performance metrics
        train_score = r2_score(y_train, train_pred)
        test_score = r2_score(y_test, test_pred)
        
        # Cross-validation scores
        cv_scores = cross_val_score(
            self.model, X_train_scaled, y_train,
            cv=TimeSeriesSplit(n_splits=self.config.cv_folds),
            scoring=self.config.cv_scoring
        )
        
        # Feature importance analysis
        feature_importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        # Store results
        self.results.fitted = True
        self.results.train_score = train_score
        self.results.test_score = test_score
        self.results.cv_scores = cv_scores.tolist()
        self.results.cv_mean = cv_scores.mean()
        self.results.cv_std = cv_scores.std()
        self.results.train_predictions = train_pred
        self.results.test_predictions = test_pred
        self.results.feature_importance = feature_importance_df
        self.results.feature_names = self.feature_names
        self.results.residuals = np.concatenate([y_train - train_pred, y_test - test_pred])
        self.results.fitted_values = np.concatenate([train_pred, test_pred])
        self.results.training_time = time.time() - start_time
        
        # Model-specific results
        self.results.model_specific = {
            'n_estimators': self.model.n_estimators,
            'max_depth': self.model.max_depth,
            'oob_score': getattr(self.model, 'oob_score_', None),
            'feature_importances': self.model.feature_importances_.tolist()
        }
        
        self.fitted = True
        self.logger.info(f"Random Forest fitting completed. Train R²: {train_score:.4f}, Test R²: {test_score:.4f}")
        
        return self.results
    
    def get_feature_importance_analysis(self, top_n: int = 10) -> Dict[str, Any]:
        """
        Get detailed feature importance analysis.
        
        Args:
            top_n: Number of top features to analyze
            
        Returns:
            Dictionary with feature importance analysis
        """
        if not self.fitted:
            raise ValueError("Model must be fitted before feature importance analysis")
        
        # Get top features
        top_features = self.results.feature_importance.head(top_n)
        
        # Calculate cumulative importance
        cumulative_importance = top_features['importance'].cumsum()
        
        analysis = {
            'top_features': top_features.to_dict('records'),
            'cumulative_importance': cumulative_importance.tolist(),
            'features_for_80_percent': len(cumulative_importance[cumulative_importance <= 0.8]) + 1,
            'importance_threshold_features': len(
                self.results.feature_importance[
                    self.results.feature_importance['importance'] >= self.config.feature_importance_threshold
                ]
            )
        }
        
        return analysis
    
    def plot_feature_importance(self, top_n: int = 15, figsize: Tuple[int, int] = (10, 8)) -> plt.Figure:
        """
        Plot feature importance.
        
        Args:
            top_n: Number of top features to plot
            figsize: Figure size
            
        Returns:
            Matplotlib figure
        """
        if not self.fitted:
            raise ValueError("Model must be fitted before plotting")
        
        fig, ax = plt.subplots(figsize=figsize)
        
        top_features = self.results.feature_importance.head(top_n)
        
        bars = ax.barh(range(len(top_features)), top_features['importance'])
        ax.set_yticks(range(len(top_features)))
        ax.set_yticklabels(top_features['feature'])
        ax.set_xlabel('Feature Importance')
        ax.set_title('Random Forest Feature Importance')
        ax.invert_yaxis()
        
        # Add value labels on bars
        for i, bar in enumerate(bars):
            width = bar.get_width()
            ax.text(width + 0.001, bar.get_y() + bar.get_height()/2, 
                   f'{width:.3f}', ha='left', va='center')
        
        plt.tight_layout()
        return fig
    
    def plot_partial_dependence(self, 
                               X: Union[np.ndarray, pd.DataFrame],
                               features: Optional[List[str]] = None,
                               figsize: Tuple[int, int] = (15, 10)) -> plt.Figure:
        """
        Plot partial dependence plots for relationship visualization.
        
        Args:
            X: Feature matrix used for fitting
            features: Features to plot (if None, uses top important features)
            figsize: Figure size
            
        Returns:
            Matplotlib figure
        """
        if not self.fitted:
            raise ValueError("Model must be fitted before plotting partial dependence")
        
        # Convert to numpy if needed
        if isinstance(X, pd.DataFrame):
            X_array = X.values
        else:
            X_array = X
        
        # Scale features
        X_scaled = self.scaler.transform(X_array)
        
        # Select features to plot
        if features is None:
            # Use top 6 most important features
            top_features = self.results.feature_importance.head(6)['feature'].tolist()
            feature_indices = [self.feature_names.index(f) for f in top_features]
        else:
            feature_indices = [self.feature_names.index(f) for f in features if f in self.feature_names]
            top_features = features
        
        # Create partial dependence plots
        fig, axes = plt.subplots(2, 3, figsize=figsize)
        axes = axes.ravel()
        
        for i, (feature_idx, feature_name) in enumerate(zip(feature_indices[:6], top_features[:6])):
            if i < len(axes):
                # Calculate partial dependence
                pd_result = partial_dependence(
                    self.model, X_scaled, [feature_idx], 
                    grid_resolution=50
                )
                
                axes[i].plot(pd_result[1][0], pd_result[0][0])
                axes[i].set_xlabel(feature_name)
                axes[i].set_ylabel('Partial Dependence')
                axes[i].set_title(f'Partial Dependence: {feature_name}')
                axes[i].grid(True, alpha=0.3)
        
        # Remove empty subplots
        for i in range(len(feature_indices), len(axes)):
            fig.delaxes(axes[i])
        
        plt.tight_layout()
        return fig
    
    def detect_thresholds(self, 
                         X: Union[np.ndarray, pd.DataFrame],
                         threshold_variable: str,
                         n_quantiles: int = 20) -> Dict[str, Any]:
        """
        Detect potential thresholds using Random Forest predictions.
        
        Args:
            X: Feature matrix
            threshold_variable: Name of variable to analyze for thresholds
            n_quantiles: Number of quantiles to analyze
            
        Returns:
            Dictionary with threshold analysis
        """
        if not self.fitted:
            raise ValueError("Model must be fitted before threshold detection")
        
        if isinstance(X, pd.DataFrame):
            X_array = X.values
            if threshold_variable not in X.columns:
                raise ValueError(f"Threshold variable '{threshold_variable}' not found in features")
            threshold_idx = X.columns.get_loc(threshold_variable)
        else:
            X_array = X
            if threshold_variable not in self.feature_names:
                raise ValueError(f"Threshold variable '{threshold_variable}' not found in features")
            threshold_idx = self.feature_names.index(threshold_variable)
        
        # Scale features
        X_scaled = self.scaler.transform(X_array)
        
        # Get threshold variable values
        threshold_values = X_array[:, threshold_idx]
        
        # Calculate quantiles
        quantiles = np.linspace(0, 1, n_quantiles + 1)[1:-1]  # Exclude 0 and 1
        threshold_points = np.quantile(threshold_values, quantiles)
        
        # Analyze predictions at different threshold levels
        threshold_analysis = []
        
        for i, threshold_point in enumerate(threshold_points):
            # Split data at threshold
            below_threshold = threshold_values <= threshold_point
            above_threshold = threshold_values > threshold_point
            
            if np.sum(below_threshold) > 5 and np.sum(above_threshold) > 5:
                # Make predictions for both regimes
                pred_below = self.model.predict(X_scaled[below_threshold])
                pred_above = self.model.predict(X_scaled[above_threshold])
                
                # Calculate regime statistics
                mean_pred_below = np.mean(pred_below)
                mean_pred_above = np.mean(pred_above)
                std_pred_below = np.std(pred_below)
                std_pred_above = np.std(pred_above)
                
                # Calculate difference in means
                mean_difference = mean_pred_above - mean_pred_below
                
                threshold_analysis.append({
                    'threshold_value': threshold_point,
                    'quantile': quantiles[i],
                    'n_below': np.sum(below_threshold),
                    'n_above': np.sum(above_threshold),
                    'mean_pred_below': mean_pred_below,
                    'mean_pred_above': mean_pred_above,
                    'std_pred_below': std_pred_below,
                    'std_pred_above': std_pred_above,
                    'mean_difference': mean_difference,
                    'abs_mean_difference': abs(mean_difference)
                })
        
        # Find threshold with maximum difference
        if threshold_analysis:
            max_diff_idx = np.argmax([t['abs_mean_difference'] for t in threshold_analysis])
            optimal_threshold = threshold_analysis[max_diff_idx]
        else:
            optimal_threshold = None
        
        return {
            'threshold_variable': threshold_variable,
            'threshold_analysis': threshold_analysis,
            'optimal_threshold': optimal_threshold,
            'n_quantiles_analyzed': len(threshold_analysis)
        }
    
    def predict(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """
        Make predictions using fitted Random Forest model.
        
        Args:
            X: Feature matrix
            
        Returns:
            Predictions array
        """
        if not self.fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        if isinstance(X, pd.DataFrame):
            X = X.values
        
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)
    
    def get_prediction_intervals(self, 
                               X: Union[np.ndarray, pd.DataFrame],
                               quantiles: List[float] = None) -> Dict[str, np.ndarray]:
        """
        Get prediction intervals using quantile regression forest approach.
        
        Args:
            X: Feature matrix
            quantiles: Quantiles for prediction intervals
            
        Returns:
            Dictionary with prediction intervals
        """
        if not self.fitted:
            raise ValueError("Model must be fitted before getting prediction intervals")
        
        if quantiles is None:
            quantiles = self.config.uncertainty_quantiles
        
        if isinstance(X, pd.DataFrame):
            X = X.values
        
        X_scaled = self.scaler.transform(X)
        
        # Get predictions from all trees
        all_predictions = np.array([tree.predict(X_scaled) for tree in self.model.estimators_])
        
        # Calculate quantiles across trees
        intervals = {}
        for q in quantiles:
            intervals[f'quantile_{q}'] = np.quantile(all_predictions, q, axis=0)
        
        # Add mean prediction
        intervals['mean'] = np.mean(all_predictions, axis=0)
        intervals['std'] = np.std(all_predictions, axis=0)
        
        return intervals


class GradientBoostingComplexRelationships:
    """
    Gradient Boosting model for capturing complex non-linear QE relationships.
    
    Implements Requirements 5.1, 5.2, 5.3 for Gradient Boosting models with
    SHAP value analysis and hyperparameter optimization.
    """
    
    def __init__(self, config: Optional[MLModelConfig] = None):
        """
        Initialize Gradient Boosting model.
        
        Args:
            config: MLModelConfig for model parameters
        """
        self.config = config or MLModelConfig()
        self.logger = logging.getLogger(__name__)
        
        # Initialize model
        self.model = GradientBoostingRegressor(
            n_estimators=self.config.gb_n_estimators,
            learning_rate=self.config.gb_learning_rate,
            max_depth=self.config.gb_max_depth,
            min_samples_split=self.config.gb_min_samples_split,
            min_samples_leaf=self.config.gb_min_samples_leaf,
            random_state=self.config.gb_random_state
        )
        
        # Storage
        self.scaler = StandardScaler()
        self.fitted = False
        self.feature_names = None
        self.results = MLModelResults(model_name="GradientBoosting")
        self.shap_explainer = None
        
    def fit(self, 
            X: Union[np.ndarray, pd.DataFrame], 
            y: Union[np.ndarray, pd.Series],
            feature_names: Optional[List[str]] = None,
            test_size: float = 0.2,
            hyperparameter_tuning: bool = True) -> MLModelResults:
        """
        Fit Gradient Boosting model for complex relationship detection.
        
        Args:
            X: Feature matrix
            y: Target variable
            feature_names: Names of features
            test_size: Proportion of data for testing
            hyperparameter_tuning: Whether to perform hyperparameter optimization
            
        Returns:
            MLModelResults object with fitting results
        """
        import time
        start_time = time.time()
        
        self.logger.info("Fitting Gradient Boosting model for complex relationships")
        
        # Convert to numpy arrays if needed
        if isinstance(X, pd.DataFrame):
            self.feature_names = X.columns.tolist()
            X = X.values
        else:
            self.feature_names = feature_names or [f"feature_{i}" for i in range(X.shape[1])]
        
        if isinstance(y, pd.Series):
            y = y.values
        
        # Split data (time series aware)
        split_idx = int(len(X) * (1 - test_size))
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Hyperparameter tuning if requested
        if hyperparameter_tuning:
            self.logger.info("Performing hyperparameter tuning for Gradient Boosting")
            param_grid = {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 5, 7],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            }
            
            # Use TimeSeriesSplit for time series data
            tscv = TimeSeriesSplit(n_splits=self.config.cv_folds)
            
            grid_search = GridSearchCV(
                self.model, 
                param_grid, 
                cv=tscv,
                scoring=self.config.cv_scoring,
                n_jobs=-1,
                verbose=0
            )
            
            grid_search.fit(X_train_scaled, y_train)
            self.model = grid_search.best_estimator_
            self.results.hyperparameters = grid_search.best_params_
            
            self.logger.info(f"Best parameters: {grid_search.best_params_}")
        
        # Fit final model
        self.model.fit(X_train_scaled, y_train)
        
        # Make predictions
        train_pred = self.model.predict(X_train_scaled)
        test_pred = self.model.predict(X_test_scaled)
        
        # Calculate performance metrics
        train_score = r2_score(y_train, train_pred)
        test_score = r2_score(y_test, test_pred)
        
        # Cross-validation scores
        cv_scores = cross_val_score(
            self.model, X_train_scaled, y_train,
            cv=TimeSeriesSplit(n_splits=self.config.cv_folds),
            scoring=self.config.cv_scoring
        )
        
        # Feature importance analysis
        feature_importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        # Initialize SHAP explainer if available
        if SHAP_AVAILABLE:
            try:
                self.shap_explainer = shap.TreeExplainer(self.model)
                self.logger.info("SHAP explainer initialized successfully")
            except Exception as e:
                self.logger.warning(f"Failed to initialize SHAP explainer: {e}")
                self.shap_explainer = None
        
        # Store results
        self.results.fitted = True
        self.results.train_score = train_score
        self.results.test_score = test_score
        self.results.cv_scores = cv_scores.tolist()
        self.results.cv_mean = cv_scores.mean()
        self.results.cv_std = cv_scores.std()
        self.results.train_predictions = train_pred
        self.results.test_predictions = test_pred
        self.results.feature_importance = feature_importance_df
        self.results.feature_names = self.feature_names
        self.results.residuals = np.concatenate([y_train - train_pred, y_test - test_pred])
        self.results.fitted_values = np.concatenate([train_pred, test_pred])
        self.results.training_time = time.time() - start_time
        
        # Model-specific results
        self.results.model_specific = {
            'n_estimators': self.model.n_estimators,
            'learning_rate': self.model.learning_rate,
            'max_depth': self.model.max_depth,
            'train_score_stages': self.model.train_score_.tolist(),
            'feature_importances': self.model.feature_importances_.tolist()
        }
        
        self.fitted = True
        self.logger.info(f"Gradient Boosting fitting completed. Train R²: {train_score:.4f}, Test R²: {test_score:.4f}")
        
        return self.results
    
    def get_shap_analysis(self, 
                         X: Union[np.ndarray, pd.DataFrame],
                         max_samples: int = 1000) -> Dict[str, Any]:
        """
        Perform SHAP value analysis for model interpretability.
        
        Args:
            X: Feature matrix for SHAP analysis
            max_samples: Maximum number of samples to analyze (for performance)
            
        Returns:
            Dictionary with SHAP analysis results
        """
        if not self.fitted:
            raise ValueError("Model must be fitted before SHAP analysis")
        
        if not SHAP_AVAILABLE:
            self.logger.warning("SHAP not available. Returning feature importance instead.")
            return {
                'shap_available': False,
                'feature_importance': self.results.feature_importance.to_dict('records'),
                'message': 'SHAP not available, using feature importance'
            }
        
        if self.shap_explainer is None:
            self.logger.warning("SHAP explainer not initialized. Cannot perform SHAP analysis.")
            return {
                'shap_available': False,
                'feature_importance': self.results.feature_importance.to_dict('records'),
                'message': 'SHAP explainer not initialized'
            }
        
        self.logger.info("Performing SHAP value analysis")
        
        # Convert to numpy if needed and scale
        if isinstance(X, pd.DataFrame):
            X_array = X.values
        else:
            X_array = X
        
        X_scaled = self.scaler.transform(X_array)
        
        # Limit samples for performance
        if len(X_scaled) > max_samples:
            sample_indices = np.random.choice(len(X_scaled), max_samples, replace=False)
            X_sample = X_scaled[sample_indices]
        else:
            X_sample = X_scaled
            sample_indices = np.arange(len(X_scaled))
        
        try:
            # Calculate SHAP values
            shap_values = self.shap_explainer.shap_values(X_sample)
            
            # Calculate SHAP feature importance (mean absolute SHAP values)
            shap_importance = np.abs(shap_values).mean(axis=0)
            
            # Create SHAP importance DataFrame
            shap_importance_df = pd.DataFrame({
                'feature': self.feature_names,
                'shap_importance': shap_importance
            }).sort_values('shap_importance', ascending=False)
            
            # Calculate SHAP interaction values if possible
            try:
                shap_interaction_values = self.shap_explainer.shap_interaction_values(X_sample[:100])  # Limit for performance
                interaction_available = True
            except Exception as e:
                self.logger.warning(f"SHAP interaction values not available: {e}")
                shap_interaction_values = None
                interaction_available = False
            
            analysis = {
                'shap_available': True,
                'shap_values': shap_values,
                'shap_importance': shap_importance_df.to_dict('records'),
                'shap_interaction_values': shap_interaction_values,
                'interaction_available': interaction_available,
                'sample_indices': sample_indices.tolist(),
                'n_samples_analyzed': len(X_sample),
                'feature_names': self.feature_names
            }
            
            self.logger.info(f"SHAP analysis completed for {len(X_sample)} samples")
            return analysis
            
        except Exception as e:
            self.logger.error(f"SHAP analysis failed: {e}")
            return {
                'shap_available': False,
                'error': str(e),
                'feature_importance': self.results.feature_importance.to_dict('records'),
                'message': 'SHAP analysis failed, using feature importance'
            }
    
    def plot_shap_summary(self, 
                         X: Union[np.ndarray, pd.DataFrame],
                         max_samples: int = 1000,
                         plot_type: str = 'dot',
                         figsize: Tuple[int, int] = (10, 8)) -> plt.Figure:
        """
        Create SHAP summary plot for model interpretability.
        
        Args:
            X: Feature matrix for SHAP analysis
            max_samples: Maximum number of samples to analyze
            plot_type: Type of SHAP plot ('dot', 'bar', 'violin')
            figsize: Figure size
            
        Returns:
            Matplotlib figure
        """
        if not SHAP_AVAILABLE or self.shap_explainer is None:
            # Create alternative plot using feature importance
            fig, ax = plt.subplots(figsize=figsize)
            top_features = self.results.feature_importance.head(15)
            bars = ax.barh(range(len(top_features)), top_features['importance'])
            ax.set_yticks(range(len(top_features)))
            ax.set_yticklabels(top_features['feature'])
            ax.set_xlabel('Feature Importance')
            ax.set_title('Gradient Boosting Feature Importance (SHAP not available)')
            ax.invert_yaxis()
            plt.tight_layout()
            return fig
        
        # Get SHAP analysis
        shap_analysis = self.get_shap_analysis(X, max_samples)
        
        if not shap_analysis['shap_available']:
            # Fallback to feature importance plot
            fig, ax = plt.subplots(figsize=figsize)
            importance_data = shap_analysis['feature_importance'][:15]  # Top 15
            features = [item['feature'] for item in importance_data]
            importances = [item['importance'] for item in importance_data]
            
            bars = ax.barh(range(len(features)), importances)
            ax.set_yticks(range(len(features)))
            ax.set_yticklabels(features)
            ax.set_xlabel('Feature Importance')
            ax.set_title('Gradient Boosting Feature Importance')
            ax.invert_yaxis()
            plt.tight_layout()
            return fig
        
        # Create SHAP plot
        fig = plt.figure(figsize=figsize)
        
        # Convert X to scaled format for SHAP
        if isinstance(X, pd.DataFrame):
            X_array = X.values
        else:
            X_array = X
        
        X_scaled = self.scaler.transform(X_array)
        
        # Limit samples
        sample_indices = shap_analysis['sample_indices']
        X_sample = X_scaled[sample_indices]
        shap_values = shap_analysis['shap_values']
        
        try:
            if plot_type == 'dot':
                shap.summary_plot(shap_values, X_sample, feature_names=self.feature_names, show=False)
            elif plot_type == 'bar':
                shap.summary_plot(shap_values, X_sample, feature_names=self.feature_names, 
                                plot_type='bar', show=False)
            elif plot_type == 'violin':
                shap.summary_plot(shap_values, X_sample, feature_names=self.feature_names, 
                                plot_type='violin', show=False)
            else:
                shap.summary_plot(shap_values, X_sample, feature_names=self.feature_names, show=False)
            
            plt.title(f'SHAP Summary Plot - {plot_type.title()}')
            plt.tight_layout()
            
        except Exception as e:
            self.logger.error(f"SHAP plotting failed: {e}")
            # Create fallback plot
            plt.clf()
            ax = plt.gca()
            importance_data = shap_analysis['shap_importance'][:15]
            features = [item['feature'] for item in importance_data]
            importances = [item['shap_importance'] for item in importance_data]
            
            bars = ax.barh(range(len(features)), importances)
            ax.set_yticks(range(len(features)))
            ax.set_yticklabels(features)
            ax.set_xlabel('Mean |SHAP Value|')
            ax.set_title('SHAP Feature Importance')
            ax.invert_yaxis()
            plt.tight_layout()
        
        return fig
    
    def plot_shap_dependence(self, 
                           X: Union[np.ndarray, pd.DataFrame],
                           feature_name: str,
                           interaction_feature: Optional[str] = None,
                           max_samples: int = 1000,
                           figsize: Tuple[int, int] = (10, 6)) -> plt.Figure:
        """
        Create SHAP dependence plot for a specific feature.
        
        Args:
            X: Feature matrix for SHAP analysis
            feature_name: Name of feature to plot
            interaction_feature: Name of interaction feature (auto-detected if None)
            max_samples: Maximum number of samples to analyze
            figsize: Figure size
            
        Returns:
            Matplotlib figure
        """
        if not SHAP_AVAILABLE or self.shap_explainer is None:
            raise ValueError("SHAP not available or explainer not initialized")
        
        if feature_name not in self.feature_names:
            raise ValueError(f"Feature '{feature_name}' not found in model features")
        
        # Get SHAP analysis
        shap_analysis = self.get_shap_analysis(X, max_samples)
        
        if not shap_analysis['shap_available']:
            raise ValueError("SHAP analysis failed")
        
        # Get feature index
        feature_idx = self.feature_names.index(feature_name)
        
        # Convert X to scaled format
        if isinstance(X, pd.DataFrame):
            X_array = X.values
        else:
            X_array = X
        
        X_scaled = self.scaler.transform(X_array)
        sample_indices = shap_analysis['sample_indices']
        X_sample = X_scaled[sample_indices]
        shap_values = shap_analysis['shap_values']
        
        # Create dependence plot
        fig = plt.figure(figsize=figsize)
        
        try:
            if interaction_feature is not None:
                if interaction_feature in self.feature_names:
                    interaction_idx = self.feature_names.index(interaction_feature)
                    shap.dependence_plot(feature_idx, shap_values, X_sample, 
                                       feature_names=self.feature_names,
                                       interaction_index=interaction_idx, show=False)
                else:
                    shap.dependence_plot(feature_idx, shap_values, X_sample, 
                                       feature_names=self.feature_names, show=False)
            else:
                shap.dependence_plot(feature_idx, shap_values, X_sample, 
                                   feature_names=self.feature_names, show=False)
            
            plt.title(f'SHAP Dependence Plot - {feature_name}')
            plt.tight_layout()
            
        except Exception as e:
            self.logger.error(f"SHAP dependence plot failed: {e}")
            # Create fallback scatter plot
            plt.clf()
            ax = plt.gca()
            ax.scatter(X_sample[:, feature_idx], shap_values[:, feature_idx], alpha=0.6)
            ax.set_xlabel(feature_name)
            ax.set_ylabel(f'SHAP value for {feature_name}')
            ax.set_title(f'SHAP Dependence Plot - {feature_name}')
            plt.tight_layout()
        
        return fig
    
    def get_learning_curve_analysis(self) -> Dict[str, Any]:
        """
        Analyze learning curves and training progression.
        
        Returns:
            Dictionary with learning curve analysis
        """
        if not self.fitted:
            raise ValueError("Model must be fitted before learning curve analysis")
        
        # Get training scores by stage
        train_scores = self.model.train_score_
        
        # Calculate validation scores if available
        try:
            # Use staged_predict to get validation scores
            # This requires validation data, so we'll use the training approach
            n_estimators = len(train_scores)
            stages = np.arange(1, n_estimators + 1)
            
            analysis = {
                'n_estimators': n_estimators,
                'stages': stages.tolist(),
                'train_scores': train_scores.tolist(),
                'final_train_score': train_scores[-1],
                'best_iteration': np.argmax(train_scores) + 1,
                'best_score': np.max(train_scores),
                'learning_rate': self.model.learning_rate,
                'max_depth': self.model.max_depth
            }
            
            # Calculate score improvements
            score_improvements = np.diff(train_scores)
            analysis['score_improvements'] = score_improvements.tolist()
            analysis['avg_improvement'] = np.mean(score_improvements)
            analysis['improvement_std'] = np.std(score_improvements)
            
            # Detect potential overfitting (if scores start decreasing)
            if len(train_scores) > 10:
                recent_trend = np.polyfit(stages[-10:], train_scores[-10:], 1)[0]
                analysis['recent_trend'] = recent_trend
                analysis['potential_overfitting'] = recent_trend < -0.001
            
            return analysis
            
        except Exception as e:
            self.logger.error(f"Learning curve analysis failed: {e}")
            return {
                'error': str(e),
                'n_estimators': len(train_scores),
                'train_scores': train_scores.tolist()
            }
    
    def plot_learning_curves(self, figsize: Tuple[int, int] = (12, 5)) -> plt.Figure:
        """
        Plot learning curves showing training progression.
        
        Args:
            figsize: Figure size
            
        Returns:
            Matplotlib figure
        """
        if not self.fitted:
            raise ValueError("Model must be fitted before plotting learning curves")
        
        learning_analysis = self.get_learning_curve_analysis()
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        # Plot 1: Training score progression
        stages = learning_analysis['stages']
        train_scores = learning_analysis['train_scores']
        
        ax1.plot(stages, train_scores, 'b-', linewidth=2, label='Training Score')
        ax1.set_xlabel('Boosting Iterations')
        ax1.set_ylabel('Training Score')
        ax1.set_title('Training Score Progression')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # Add best iteration marker
        best_iter = learning_analysis['best_iteration']
        best_score = learning_analysis['best_score']
        ax1.axvline(x=best_iter, color='r', linestyle='--', alpha=0.7, label=f'Best: {best_iter}')
        ax1.plot(best_iter, best_score, 'ro', markersize=8)
        
        # Plot 2: Score improvements
        if 'score_improvements' in learning_analysis:
            improvements = learning_analysis['score_improvements']
            ax2.plot(stages[1:], improvements, 'g-', linewidth=2, alpha=0.7)
            ax2.axhline(y=0, color='k', linestyle='-', alpha=0.3)
            ax2.set_xlabel('Boosting Iterations')
            ax2.set_ylabel('Score Improvement')
            ax2.set_title('Score Improvement per Iteration')
            ax2.grid(True, alpha=0.3)
            
            # Add trend line for recent improvements
            if len(improvements) > 10:
                recent_improvements = improvements[-10:]
                recent_stages = stages[-10:]
                z = np.polyfit(recent_stages, recent_improvements, 1)
                p = np.poly1d(z)
                ax2.plot(recent_stages, p(recent_stages), 'r--', alpha=0.8, 
                        label=f'Recent trend: {z[0]:.6f}')
                ax2.legend()
        
        plt.tight_layout()
        return fig
    
    def optimize_hyperparameters(self, 
                                X: Union[np.ndarray, pd.DataFrame], 
                                y: Union[np.ndarray, pd.Series],
                                param_grid: Optional[Dict[str, List]] = None,
                                cv_folds: int = 5,
                                scoring: str = 'neg_mean_squared_error',
                                n_jobs: int = -1) -> Dict[str, Any]:
        """
        Perform comprehensive hyperparameter optimization.
        
        Args:
            X: Feature matrix
            y: Target variable
            param_grid: Custom parameter grid (uses default if None)
            cv_folds: Number of cross-validation folds
            scoring: Scoring metric for optimization
            n_jobs: Number of parallel jobs
            
        Returns:
            Dictionary with optimization results
        """
        self.logger.info("Starting comprehensive hyperparameter optimization")
        
        # Convert to numpy arrays if needed
        if isinstance(X, pd.DataFrame):
            X = X.values
        if isinstance(y, pd.Series):
            y = y.values
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Default parameter grid if not provided
        if param_grid is None:
            param_grid = {
                'n_estimators': [50, 100, 200, 300],
                'learning_rate': [0.01, 0.05, 0.1, 0.2],
                'max_depth': [3, 4, 5, 6, 7],
                'min_samples_split': [2, 5, 10, 20],
                'min_samples_leaf': [1, 2, 4, 8],
                'subsample': [0.8, 0.9, 1.0],
                'max_features': ['sqrt', 'log2', None]
            }
        
        # Use TimeSeriesSplit for time series data
        tscv = TimeSeriesSplit(n_splits=cv_folds)
        
        # Create base model
        base_model = GradientBoostingRegressor(random_state=self.config.gb_random_state)
        
        # Perform grid search
        grid_search = GridSearchCV(
            base_model,
            param_grid,
            cv=tscv,
            scoring=scoring,
            n_jobs=n_jobs,
            verbose=1,
            return_train_score=True
        )
        
        import time
        start_time = time.time()
        grid_search.fit(X_scaled, y)
        optimization_time = time.time() - start_time
        
        # Extract results
        results = {
            'best_params': grid_search.best_params_,
            'best_score': grid_search.best_score_,
            'best_estimator': grid_search.best_estimator_,
            'cv_results': grid_search.cv_results_,
            'optimization_time': optimization_time,
            'n_parameter_combinations': len(grid_search.cv_results_['params']),
            'scoring_metric': scoring
        }
        
        # Analyze parameter importance
        cv_results_df = pd.DataFrame(grid_search.cv_results_)
        
        # Calculate parameter impact analysis
        param_impact = {}
        for param in param_grid.keys():
            param_values = [params[param] for params in cv_results_df['params']]
            param_scores = cv_results_df['mean_test_score'].values
            
            # Group by parameter value and calculate mean score
            param_df = pd.DataFrame({'param_value': param_values, 'score': param_scores})
            param_grouped = param_df.groupby('param_value')['score'].agg(['mean', 'std', 'count'])
            
            param_impact[param] = {
                'values': param_grouped.index.tolist(),
                'mean_scores': param_grouped['mean'].tolist(),
                'std_scores': param_grouped['std'].tolist(),
                'counts': param_grouped['count'].tolist(),
                'score_range': param_grouped['mean'].max() - param_grouped['mean'].min()
            }
        
        results['parameter_impact'] = param_impact
        
        # Update model with best parameters
        self.model = grid_search.best_estimator_
        
        self.logger.info(f"Hyperparameter optimization completed in {optimization_time:.2f} seconds")
        self.logger.info(f"Best score: {grid_search.best_score_:.4f}")
        self.logger.info(f"Best parameters: {grid_search.best_params_}")
        
        return results
    
    def perform_cross_validation_analysis(self, 
                                        X: Union[np.ndarray, pd.DataFrame], 
                                        y: Union[np.ndarray, pd.Series],
                                        cv_folds: int = 5,
                                        scoring_metrics: List[str] = None) -> Dict[str, Any]:
        """
        Perform comprehensive cross-validation analysis.
        
        Args:
            X: Feature matrix
            y: Target variable
            cv_folds: Number of cross-validation folds
            scoring_metrics: List of scoring metrics to evaluate
            
        Returns:
            Dictionary with cross-validation analysis
        """
        if not self.fitted:
            raise ValueError("Model must be fitted before cross-validation analysis")
        
        self.logger.info("Performing comprehensive cross-validation analysis")
        
        # Default scoring metrics
        if scoring_metrics is None:
            scoring_metrics = [
                'neg_mean_squared_error',
                'neg_mean_absolute_error', 
                'r2',
                'neg_root_mean_squared_error'
            ]
        
        # Convert to numpy arrays if needed
        if isinstance(X, pd.DataFrame):
            X = X.values
        if isinstance(y, pd.Series):
            y = y.values
        
        # Scale features
        X_scaled = self.scaler.transform(X)
        
        # Use TimeSeriesSplit for time series data
        tscv = TimeSeriesSplit(n_splits=cv_folds)
        
        # Perform cross-validation for each metric
        cv_results = {}
        
        for metric in scoring_metrics:
            try:
                scores = cross_val_score(
                    self.model, X_scaled, y,
                    cv=tscv,
                    scoring=metric,
                    n_jobs=-1
                )
                
                cv_results[metric] = {
                    'scores': scores.tolist(),
                    'mean': scores.mean(),
                    'std': scores.std(),
                    'min': scores.min(),
                    'max': scores.max(),
                    'cv_folds': cv_folds
                }
                
            except Exception as e:
                self.logger.warning(f"Cross-validation failed for metric {metric}: {e}")
                cv_results[metric] = {
                    'error': str(e),
                    'scores': None
                }
        
        # Perform learning curve analysis
        try:
            from sklearn.model_selection import learning_curve
            
            train_sizes = np.linspace(0.1, 1.0, 10)
            train_sizes_abs, train_scores, val_scores = learning_curve(
                self.model, X_scaled, y,
                train_sizes=train_sizes,
                cv=tscv,
                scoring='neg_mean_squared_error',
                n_jobs=-1
            )
            
            learning_curve_results = {
                'train_sizes': train_sizes_abs.tolist(),
                'train_scores_mean': train_scores.mean(axis=1).tolist(),
                'train_scores_std': train_scores.std(axis=1).tolist(),
                'val_scores_mean': val_scores.mean(axis=1).tolist(),
                'val_scores_std': val_scores.std(axis=1).tolist()
            }
            
        except Exception as e:
            self.logger.warning(f"Learning curve analysis failed: {e}")
            learning_curve_results = {'error': str(e)}
        
        # Compile final results
        analysis_results = {
            'cv_results': cv_results,
            'learning_curve': learning_curve_results,
            'model_params': self.model.get_params(),
            'cv_folds': cv_folds,
            'n_samples': len(X),
            'n_features': X.shape[1]
        }
        
        self.logger.info("Cross-validation analysis completed")
        return analysis_results
    
    def predict(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """
        Make predictions using fitted Gradient Boosting model.
        
        Args:
            X: Feature matrix
            
        Returns:
            Predictions array
        """
        if not self.fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        if isinstance(X, pd.DataFrame):
            X = X.values
        
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)
    
    def predict_staged(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """
        Make staged predictions showing progression through boosting iterations.
        
        Args:
            X: Feature matrix
            
        Returns:
            Array of predictions for each boosting stage
        """
        if not self.fitted:
            raise ValueError("Model must be fitted before making staged predictions")
        
        if isinstance(X, pd.DataFrame):
            X = X.values
        
        X_scaled = self.scaler.transform(X)
        
        # Get staged predictions
        staged_predictions = list(self.model.staged_predict(X_scaled))
        return np.array(staged_predictions)
    
    def get_prediction_intervals(self, 
                               X: Union[np.ndarray, pd.DataFrame],
                               quantiles: List[float] = None,
                               n_bootstrap: int = 100) -> Dict[str, np.ndarray]:
        """
        Get prediction intervals using bootstrap approach.
        
        Args:
            X: Feature matrix
            quantiles: Quantiles for prediction intervals
            n_bootstrap: Number of bootstrap samples
            
        Returns:
            Dictionary with prediction intervals
        """
        if not self.fitted:
            raise ValueError("Model must be fitted before getting prediction intervals")
        
        if quantiles is None:
            quantiles = self.config.uncertainty_quantiles
        
        if isinstance(X, pd.DataFrame):
            X = X.values
        
        X_scaled = self.scaler.transform(X)
        
        # Get base predictions
        base_predictions = self.model.predict(X_scaled)
        
        # Bootstrap approach for uncertainty estimation
        # Note: This is a simplified approach. For production, consider using
        # quantile regression or more sophisticated uncertainty quantification
        
        # Use residuals from training to estimate prediction uncertainty
        if self.results.residuals is not None:
            residual_std = np.std(self.results.residuals)
            
            # Generate prediction intervals assuming normal distribution of residuals
            intervals = {}
            for q in quantiles:
                from scipy import stats
                z_score = stats.norm.ppf(q)
                intervals[f'quantile_{q}'] = base_predictions + z_score * residual_std
            
            intervals['mean'] = base_predictions
            intervals['std'] = np.full_like(base_predictions, residual_std)
            
            return intervals
        
        else:
            # Fallback: return just the predictions
            return {
                'mean': base_predictions,
                'std': np.zeros_like(base_predictions)
            }


class NeuralNetworkEnsemble:
    """
    Neural Network ensemble for deep learning analysis of QE effects.
    
    Implements Requirements 5.1, 5.2, 5.4, 5.6 for Neural Network models with
    attention mechanisms and uncertainty estimation.
    """
    
    def __init__(self, config: Optional[MLModelConfig] = None):
        """
        Initialize Neural Network ensemble.
        
        Args:
            config: MLModelConfig for model parameters
        """
        self.config = config or MLModelConfig()
        self.logger = logging.getLogger(__name__)
        
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for Neural Network models")
        
        # Storage
        self.scaler = StandardScaler()
        self.fitted = False
        self.feature_names = None
        self.results = MLModelResults(model_name="NeuralNetwork")
        self.models = []  # Ensemble of models
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def _create_model(self, input_dim: int) -> nn.Module:
        """
        Create a neural network model with attention mechanism.
        
        Args:
            input_dim: Number of input features
            
        Returns:
            PyTorch neural network model
        """
        class AttentionNN(nn.Module):
            def __init__(self, input_dim, hidden_layers, dropout_rate):
                super(AttentionNN, self).__init__()
                
                # Build layers
                layers = []
                prev_dim = input_dim
                
                for hidden_dim in hidden_layers:
                    layers.extend([
                        nn.Linear(prev_dim, hidden_dim),
                        nn.ReLU(),
                        nn.Dropout(dropout_rate)
                    ])
                    prev_dim = hidden_dim
                
                # Attention mechanism
                self.attention = nn.Sequential(
                    nn.Linear(prev_dim, prev_dim // 2),
                    nn.Tanh(),
                    nn.Linear(prev_dim // 2, prev_dim),
                    nn.Softmax(dim=1)
                )
                
                # Main network
                self.network = nn.Sequential(*layers)
                
                # Output layer
                self.output = nn.Linear(prev_dim, 1)
                
            def forward(self, x):
                # Pass through main network
                features = self.network(x)
                
                # Apply attention
                attention_weights = self.attention(features)
                attended_features = features * attention_weights
                
                # Output
                output = self.output(attended_features)
                return output.squeeze()
        
        return AttentionNN(
            input_dim, 
            self.config.nn_hidden_layers, 
            self.config.nn_dropout_rate
        )
    
    def fit(self, 
            X: Union[np.ndarray, pd.DataFrame], 
            y: Union[np.ndarray, pd.Series],
            feature_names: Optional[List[str]] = None,
            test_size: float = 0.2,
            n_ensemble: int = 5) -> MLModelResults:
        """
        Fit Neural Network ensemble.
        
        Args:
            X: Feature matrix
            y: Target variable
            feature_names: Names of features
            test_size: Proportion of data for testing
            n_ensemble: Number of models in ensemble
            
        Returns:
            MLModelResults object with fitting results
        """
        import time
        start_time = time.time()
        
        self.logger.info(f"Fitting Neural Network ensemble with {n_ensemble} models")
        
        # Convert to numpy arrays if needed
        if isinstance(X, pd.DataFrame):
            self.feature_names = X.columns.tolist()
            X = X.values
        else:
            self.feature_names = feature_names or [f"feature_{i}" for i in range(X.shape[1])]
        
        if isinstance(y, pd.Series):
            y = y.values
        
        # Split data (time series aware)
        split_idx = int(len(X) * (1 - test_size))
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Convert to tensors
        X_train_tensor = torch.FloatTensor(X_train_scaled).to(self.device)
        y_train_tensor = torch.FloatTensor(y_train).to(self.device)
        X_test_tensor = torch.FloatTensor(X_test_scaled).to(self.device)
        y_test_tensor = torch.FloatTensor(y_test).to(self.device)
        
        # Train ensemble of models
        ensemble_predictions_train = []
        ensemble_predictions_test = []
        
        for i in range(n_ensemble):
            self.logger.info(f"Training model {i+1}/{n_ensemble}")
            
            # Create model
            model = self._create_model(X_train_scaled.shape[1]).to(self.device)
            
            # Loss and optimizer
            criterion = nn.MSELoss()
            optimizer = optim.Adam(model.parameters(), lr=self.config.nn_learning_rate)
            
            # Training loop with early stopping
            best_val_loss = float('inf')
            patience_counter = 0
            train_losses = []
            val_losses = []
            
            # Create data loader
            train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
            train_loader = DataLoader(train_dataset, batch_size=self.config.nn_batch_size, shuffle=False)
            
            for epoch in range(self.config.nn_epochs):
                model.train()
                epoch_train_loss = 0
                
                for batch_X, batch_y in train_loader:
                    optimizer.zero_grad()
                    outputs = model(batch_X)
                    loss = criterion(outputs, batch_y)
                    loss.backward()
                    optimizer.step()
                    epoch_train_loss += loss.item()
                
                # Validation
                model.eval()
                with torch.no_grad():
                    val_outputs = model(X_test_tensor)
                    val_loss = criterion(val_outputs, y_test_tensor).item()
                
                train_losses.append(epoch_train_loss / len(train_loader))
                val_losses.append(val_loss)
                
                # Early stopping
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    # Save best model state
                    best_model_state = model.state_dict().copy()
                else:
                    patience_counter += 1
                    if patience_counter >= self.config.nn_early_stopping_patience:
                        self.logger.info(f"Early stopping at epoch {epoch+1}")
                        break
            
            # Load best model state
            model.load_state_dict(best_model_state)
            
            # Store model
            self.models.append(model)
            
            # Get predictions
            model.eval()
            with torch.no_grad():
                train_pred = model(X_train_tensor).cpu().numpy()
                test_pred = model(X_test_tensor).cpu().numpy()
                
            ensemble_predictions_train.append(train_pred)
            ensemble_predictions_test.append(test_pred)
        
        # Ensemble predictions (mean)
        ensemble_train_pred = np.mean(ensemble_predictions_train, axis=0)
        ensemble_test_pred = np.mean(ensemble_predictions_test, axis=0)
        
        # Calculate performance metrics
        train_score = r2_score(y_train, ensemble_train_pred)
        test_score = r2_score(y_test, ensemble_test_pred)
        
        # Calculate prediction intervals using ensemble variance
        train_pred_std = np.std(ensemble_predictions_train, axis=0)
        test_pred_std = np.std(ensemble_predictions_test, axis=0)
        
        # Store results
        self.results.fitted = True
        self.results.train_score = train_score
        self.results.test_score = test_score
        self.results.train_predictions = ensemble_train_pred
        self.results.test_predictions = ensemble_test_pred
        self.results.feature_names = self.feature_names
        self.results.residuals = np.concatenate([y_train - ensemble_train_pred, y_test - ensemble_test_pred])
        self.results.fitted_values = np.concatenate([ensemble_train_pred, ensemble_test_pred])
        self.results.training_time = time.time() - start_time
        
        # Prediction intervals
        prediction_intervals = {}
        for q in self.config.uncertainty_quantiles:
            from scipy import stats
            z_score = stats.norm.ppf(q)
            train_interval = ensemble_train_pred + z_score * train_pred_std
            test_interval = ensemble_test_pred + z_score * test_pred_std
            prediction_intervals[f'quantile_{q}'] = np.concatenate([train_interval, test_interval])
        
        self.results.prediction_intervals = prediction_intervals
        
        # Model-specific results
        self.results.model_specific = {
            'n_ensemble': n_ensemble,
            'hidden_layers': self.config.nn_hidden_layers,
            'dropout_rate': self.config.nn_dropout_rate,
            'learning_rate': self.config.nn_learning_rate,
            'device': str(self.device),
            'ensemble_train_std': train_pred_std.tolist(),
            'ensemble_test_std': test_pred_std.tolist()
        }
        
        self.fitted = True
        self.logger.info(f"Neural Network ensemble fitting completed. Train R²: {train_score:.4f}, Test R²: {test_score:.4f}")
        
        return self.results
    
    def predict(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """
        Make predictions using fitted Neural Network ensemble.
        
        Args:
            X: Feature matrix
            
        Returns:
            Ensemble predictions array
        """
        if not self.fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        if isinstance(X, pd.DataFrame):
            X = X.values
        
        X_scaled = self.scaler.transform(X)
        X_tensor = torch.FloatTensor(X_scaled).to(self.device)
        
        # Get predictions from all models
        predictions = []
        for model in self.models:
            model.eval()
            with torch.no_grad():
                pred = model(X_tensor).cpu().numpy()
                predictions.append(pred)
        
        # Return ensemble mean
        return np.mean(predictions, axis=0)
    
    def get_prediction_intervals(self, 
                               X: Union[np.ndarray, pd.DataFrame],
                               quantiles: List[float] = None) -> Dict[str, np.ndarray]:
        """
        Get prediction intervals using ensemble variance.
        
        Args:
            X: Feature matrix
            quantiles: Quantiles for prediction intervals
            
        Returns:
            Dictionary with prediction intervals
        """
        if not self.fitted:
            raise ValueError("Model must be fitted before getting prediction intervals")
        
        if quantiles is None:
            quantiles = self.config.uncertainty_quantiles
        
        if isinstance(X, pd.DataFrame):
            X = X.values
        
        X_scaled = self.scaler.transform(X)
        X_tensor = torch.FloatTensor(X_scaled).to(self.device)
        
        # Get predictions from all models
        predictions = []
        for model in self.models:
            model.eval()
            with torch.no_grad():
                pred = model(X_tensor).cpu().numpy()
                predictions.append(pred)
        
        predictions = np.array(predictions)
        
        # Calculate ensemble statistics
        mean_pred = np.mean(predictions, axis=0)
        std_pred = np.std(predictions, axis=0)
        
        # Calculate quantiles
        intervals = {}
        for q in quantiles:
            from scipy import stats
            z_score = stats.norm.ppf(q)
            intervals[f'quantile_{q}'] = mean_pred + z_score * std_pred
        
        intervals['mean'] = mean_pred
        intervals['std'] = std_pred
        
        return intervals
    
    def get_attention_analysis(self, 
                             X: Union[np.ndarray, pd.DataFrame],
                             sample_indices: Optional[List[int]] = None) -> Dict[str, Any]:
        """
        Analyze attention weights to understand feature importance.
        
        Args:
            X: Feature matrix
            sample_indices: Specific samples to analyze (if None, uses first 100)
            
        Returns:
            Dictionary with attention analysis
        """
        if not self.fitted:
            raise ValueError("Model must be fitted before attention analysis")
        
        if isinstance(X, pd.DataFrame):
            X = X.values
        
        X_scaled = self.scaler.transform(X)
        
        # Select samples to analyze
        if sample_indices is None:
            sample_indices = list(range(min(100, len(X_scaled))))
        
        X_sample = X_scaled[sample_indices]
        X_tensor = torch.FloatTensor(X_sample).to(self.device)
        
        # Collect attention weights from all models
        all_attention_weights = []
        all_input_gradients = []
        
        for i, model in enumerate(self.models):
            model.eval()
            
            # Method 1: Use attention weights (but these are for hidden features, not input features)
            with torch.no_grad():
                features = model.network(X_tensor)
                attention_weights = model.attention(features).cpu().numpy()
                all_attention_weights.append(attention_weights)
            
            # Method 2: Use input gradients for true feature importance
            X_tensor_grad = X_tensor.clone().detach().requires_grad_(True)
            output = model(X_tensor_grad)
            
            # Calculate gradients with respect to input features
            gradients = []
            for j in range(output.shape[0]):  # For each sample
                if output[j].requires_grad:
                    grad = torch.autograd.grad(
                        outputs=output[j], 
                        inputs=X_tensor_grad,
                        retain_graph=True,
                        create_graph=False
                    )[0]
                    gradients.append(grad[j].detach().cpu().numpy())
            
            if gradients:
                input_gradients = np.array(gradients)
                all_input_gradients.append(input_gradients)
        
        # Use input gradients for feature importance (more meaningful for input features)
        if all_input_gradients:
            # Average gradients across ensemble and samples
            avg_input_gradients = np.mean(all_input_gradients, axis=0)  # Average across models
            feature_importance = np.mean(np.abs(avg_input_gradients), axis=0)  # Average across samples, take absolute value
            feature_importance_std = np.std(np.abs(avg_input_gradients), axis=0)
            
            # Use attention weights for sample-level analysis
            avg_attention_weights = np.mean(all_attention_weights, axis=0)
            std_attention_weights = np.std(all_attention_weights, axis=0)
        else:
            # Fallback: use uniform importance if gradient calculation fails
            feature_importance = np.ones(len(self.feature_names)) / len(self.feature_names)
            feature_importance_std = np.zeros(len(self.feature_names))
            avg_attention_weights = np.ones((len(sample_indices), len(self.feature_names))) / len(self.feature_names)
            std_attention_weights = np.zeros((len(sample_indices), len(self.feature_names)))
        
        # Ensure arrays have the same length as feature names
        if len(feature_importance) != len(self.feature_names):
            self.logger.warning(f"Feature importance length mismatch: {len(feature_importance)} vs {len(self.feature_names)}")
            # Truncate or pad to match feature names
            if len(feature_importance) > len(self.feature_names):
                feature_importance = feature_importance[:len(self.feature_names)]
                feature_importance_std = feature_importance_std[:len(self.feature_names)]
            else:
                # Pad with zeros if needed
                pad_length = len(self.feature_names) - len(feature_importance)
                feature_importance = np.pad(feature_importance, (0, pad_length), 'constant')
                feature_importance_std = np.pad(feature_importance_std, (0, pad_length), 'constant')
        
        # Create feature importance DataFrame
        attention_importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'attention_importance': feature_importance.tolist(),
            'attention_std': feature_importance_std.tolist()
        }).sort_values('attention_importance', ascending=False)
        
        # For return values, create input-level attention weights based on feature importance
        # Normalize feature importance to sum to 1 (like attention weights should)
        normalized_importance = feature_importance / np.sum(feature_importance) if np.sum(feature_importance) > 0 else feature_importance
        normalized_std = feature_importance_std / np.sum(feature_importance) if np.sum(feature_importance) > 0 else feature_importance_std
        
        # This gives a more interpretable result for the input features
        input_attention_weights = np.tile(normalized_importance, (len(sample_indices), 1))
        input_attention_std = np.tile(normalized_std, (len(sample_indices), 1))
        
        return {
            'attention_weights': input_attention_weights.tolist(),
            'attention_std': input_attention_std.tolist(),
            'feature_importance': attention_importance_df.to_dict('records'),
            'sample_indices': sample_indices,
            'n_models': len(self.models),
            'feature_names': self.feature_names
        }
    
    def plot_attention_analysis(self, 
                              X: Union[np.ndarray, pd.DataFrame],
                              sample_indices: Optional[List[int]] = None,
                              figsize: Tuple[int, int] = (15, 10)) -> plt.Figure:
        """
        Plot attention analysis results.
        
        Args:
            X: Feature matrix
            sample_indices: Specific samples to analyze
            figsize: Figure size
            
        Returns:
            Matplotlib figure with attention analysis
        """
        attention_analysis = self.get_attention_analysis(X, sample_indices)
        
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        
        # Plot 1: Feature importance from attention
        ax = axes[0, 0]
        feature_importance = attention_analysis['feature_importance'][:10]  # Top 10
        features = [f['feature'] for f in feature_importance]
        importances = [f['attention_importance'] for f in feature_importance]
        
        bars = ax.barh(range(len(features)), importances)
        ax.set_yticks(range(len(features)))
        ax.set_yticklabels(features)
        ax.set_xlabel('Attention Importance')
        ax.set_title('Feature Importance from Attention Weights')
        ax.invert_yaxis()
        
        # Add value labels
        for i, bar in enumerate(bars):
            width = bar.get_width()
            ax.text(width + 0.001, bar.get_y() + bar.get_height()/2, 
                   f'{width:.3f}', ha='left', va='center')
        
        # Plot 2: Attention weights heatmap
        ax = axes[0, 1]
        attention_weights = np.array(attention_analysis['attention_weights'])
        
        # Show first 20 samples and top 10 features for clarity
        n_samples = min(20, attention_weights.shape[0])
        n_features = min(10, attention_weights.shape[1])
        
        im = ax.imshow(attention_weights[:n_samples, :n_features].T, 
                      cmap='viridis', aspect='auto')
        ax.set_xlabel('Sample Index')
        ax.set_ylabel('Feature Index')
        ax.set_title('Attention Weights Heatmap')
        
        # Add feature labels
        top_features = [f['feature'] for f in feature_importance[:n_features]]
        ax.set_yticks(range(n_features))
        ax.set_yticklabels(top_features)
        
        plt.colorbar(im, ax=ax)
        
        # Plot 3: Attention weight distribution
        ax = axes[1, 0]
        attention_flat = attention_weights.flatten()
        ax.hist(attention_flat, bins=50, alpha=0.7, edgecolor='black')
        ax.set_xlabel('Attention Weight')
        ax.set_ylabel('Frequency')
        ax.set_title('Distribution of Attention Weights')
        ax.grid(True, alpha=0.3)
        
        # Plot 4: Feature attention consistency across ensemble
        ax = axes[1, 1]
        feature_stds = [f['attention_std'] for f in feature_importance[:10]]
        
        ax.bar(range(len(features)), feature_stds)
        ax.set_xticks(range(len(features)))
        ax.set_xticklabels(features, rotation=45, ha='right')
        ax.set_ylabel('Attention Standard Deviation')
        ax.set_title('Attention Consistency Across Ensemble')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def get_uncertainty_analysis(self, 
                               X: Union[np.ndarray, pd.DataFrame]) -> Dict[str, Any]:
        """
        Comprehensive uncertainty analysis using ensemble variance.
        
        Args:
            X: Feature matrix
            
        Returns:
            Dictionary with uncertainty analysis
        """
        if not self.fitted:
            raise ValueError("Model must be fitted before uncertainty analysis")
        
        # Get prediction intervals
        intervals = self.get_prediction_intervals(X)
        
        # Calculate uncertainty metrics
        mean_pred = intervals['mean']
        std_pred = intervals['std']
        
        # Uncertainty statistics
        uncertainty_stats = {
            'mean_uncertainty': float(np.mean(std_pred)),
            'std_uncertainty': float(np.std(std_pred)),
            'min_uncertainty': float(np.min(std_pred)),
            'max_uncertainty': float(np.max(std_pred)),
            'uncertainty_range': float(np.max(std_pred) - np.min(std_pred))
        }
        
        # High uncertainty regions (top 10%)
        high_uncertainty_threshold = np.percentile(std_pred, 90)
        high_uncertainty_indices = np.where(std_pred >= high_uncertainty_threshold)[0]
        
        # Low uncertainty regions (bottom 10%)
        low_uncertainty_threshold = np.percentile(std_pred, 10)
        low_uncertainty_indices = np.where(std_pred <= low_uncertainty_threshold)[0]
        
        return {
            'uncertainty_stats': uncertainty_stats,
            'prediction_intervals': intervals,
            'high_uncertainty_indices': high_uncertainty_indices.tolist(),
            'low_uncertainty_indices': low_uncertainty_indices.tolist(),
            'high_uncertainty_threshold': high_uncertainty_threshold,
            'low_uncertainty_threshold': low_uncertainty_threshold,
            'n_samples': len(X)
        }
    
    def plot_uncertainty_analysis(self, 
                                X: Union[np.ndarray, pd.DataFrame],
                                y: Optional[Union[np.ndarray, pd.Series]] = None,
                                figsize: Tuple[int, int] = (15, 10)) -> plt.Figure:
        """
        Plot comprehensive uncertainty analysis.
        
        Args:
            X: Feature matrix
            y: True values (optional)
            figsize: Figure size
            
        Returns:
            Matplotlib figure with uncertainty analysis
        """
        uncertainty_analysis = self.get_uncertainty_analysis(X)
        intervals = uncertainty_analysis['prediction_intervals']
        
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        
        # Plot 1: Predictions with uncertainty bands
        ax = axes[0, 0]
        mean_pred = intervals['mean']
        std_pred = intervals['std']
        
        # Show first 100 points for clarity
        n_show = min(100, len(mean_pred))
        x_range = range(n_show)
        
        ax.plot(x_range, mean_pred[:n_show], 'b-', label='Ensemble Mean', linewidth=2)
        ax.fill_between(x_range, 
                       mean_pred[:n_show] - 2*std_pred[:n_show],
                       mean_pred[:n_show] + 2*std_pred[:n_show],
                       alpha=0.3, color='blue', label='±2σ Uncertainty')
        
        if y is not None:
            if isinstance(y, pd.Series):
                y = y.values
            ax.plot(x_range, y[:n_show], 'r--', label='True Values', alpha=0.7)
        
        ax.set_xlabel('Sample Index')
        ax.set_ylabel('Value')
        ax.set_title('Predictions with Uncertainty Bands')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 2: Uncertainty distribution
        ax = axes[0, 1]
        ax.hist(std_pred, bins=50, alpha=0.7, edgecolor='black')
        ax.axvline(uncertainty_analysis['uncertainty_stats']['mean_uncertainty'], 
                  color='red', linestyle='--', label='Mean Uncertainty')
        ax.set_xlabel('Prediction Standard Deviation')
        ax.set_ylabel('Frequency')
        ax.set_title('Distribution of Prediction Uncertainty')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 3: Uncertainty vs Prediction scatter
        ax = axes[1, 0]
        ax.scatter(mean_pred, std_pred, alpha=0.6)
        ax.set_xlabel('Predicted Value')
        ax.set_ylabel('Prediction Uncertainty (σ)')
        ax.set_title('Uncertainty vs Prediction Magnitude')
        ax.grid(True, alpha=0.3)
        
        # Add trend line
        z = np.polyfit(mean_pred, std_pred, 1)
        p = np.poly1d(z)
        ax.plot(mean_pred, p(mean_pred), 'r--', alpha=0.8, 
               label=f'Trend: {z[0]:.4f}x + {z[1]:.4f}')
        ax.legend()
        
        # Plot 4: High vs Low uncertainty regions
        ax = axes[1, 1]
        high_unc_idx = uncertainty_analysis['high_uncertainty_indices'][:20]  # First 20
        low_unc_idx = uncertainty_analysis['low_uncertainty_indices'][:20]   # First 20
        
        if len(high_unc_idx) > 0 and len(low_unc_idx) > 0:
            high_unc_values = std_pred[high_unc_idx]
            low_unc_values = std_pred[low_unc_idx]
            
            ax.boxplot([low_unc_values, high_unc_values], 
                      tick_labels=['Low Uncertainty\n(Bottom 10%)', 'High Uncertainty\n(Top 10%)'])
            ax.set_ylabel('Prediction Uncertainty (σ)')
            ax.set_title('High vs Low Uncertainty Regions')
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, 'Insufficient data\nfor comparison', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Uncertainty Comparison (N/A)')
        
        plt.tight_layout()
        return fig
    
    def save_models(self, filepath: str) -> None:
        """
        Save the ensemble models to disk.
        
        Args:
            filepath: Path to save the models
        """
        if not self.fitted:
            raise ValueError("Models must be fitted before saving")
        
        import pickle
        
        save_data = {
            'models': [model.state_dict() for model in self.models],
            'scaler': self.scaler,
            'config': self.config,
            'feature_names': self.feature_names,
            'results': self.results,
            'device': str(self.device)
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(save_data, f)
        
        self.logger.info(f"Neural Network ensemble saved to {filepath}")
    
    def load_models(self, filepath: str) -> None:
        """
        Load ensemble models from disk.
        
        Args:
            filepath: Path to load the models from
        """
        import pickle
        
        with open(filepath, 'rb') as f:
            save_data = pickle.load(f)
        
        # Restore configuration and data
        self.config = save_data['config']
        self.scaler = save_data['scaler']
        self.feature_names = save_data['feature_names']
        self.results = save_data['results']
        
        # Recreate models
        self.models = []
        input_dim = len(self.feature_names)
        
        for model_state in save_data['models']:
            model = self._create_model(input_dim).to(self.device)
            model.load_state_dict(model_state)
            self.models.append(model)
        
        self.fitted = True
        self.logger.info(f"Neural Network ensemble loaded from {filepath}")
    
    def get_model_diagnostics(self) -> Dict[str, Any]:
        """
        Get comprehensive model diagnostics.
        
        Returns:
            Dictionary with model diagnostics
        """
        if not self.fitted:
            raise ValueError("Models must be fitted before diagnostics")
        
        diagnostics = {
            'ensemble_size': len(self.models),
            'model_architecture': {
                'hidden_layers': self.config.nn_hidden_layers,
                'dropout_rate': self.config.nn_dropout_rate,
                'learning_rate': self.config.nn_learning_rate
            },
            'training_info': {
                'device': str(self.device),
                'training_time': self.results.training_time,
                'early_stopping_patience': self.config.nn_early_stopping_patience
            },
            'performance_metrics': {
                'train_r2': self.results.train_score,
                'test_r2': self.results.test_score,
                'train_rmse': np.sqrt(mean_squared_error(
                    self.results.fitted_values[:len(self.results.train_predictions)],
                    self.results.train_predictions
                )) if self.results.train_predictions is not None else None,
                'test_rmse': np.sqrt(mean_squared_error(
                    self.results.fitted_values[len(self.results.train_predictions):],
                    self.results.test_predictions
                )) if self.results.test_predictions is not None else None
            },
            'ensemble_statistics': {
                'mean_train_std': np.mean(self.results.model_specific.get('ensemble_train_std', [])),
                'mean_test_std': np.mean(self.results.model_specific.get('ensemble_test_std', [])),
                'prediction_consistency': np.corrcoef([
                    model(torch.FloatTensor(self.scaler.transform(
                        np.random.randn(100, len(self.feature_names))
                    )).to(self.device)).detach().cpu().numpy()
                    for model in self.models
                ]).mean() if len(self.models) > 1 else 1.0
            }
        }
        
        return diagnostics


class MLEnsembleIntegrator:
    """
    Integrates multiple ML models for robust QE hypothesis testing.
    
    Implements Requirements 5.3, 5.4, 5.6 for ensemble methods combining
    statistical and ML predictions with uncertainty quantification.
    """
    
    def __init__(self, config: Optional[MLModelConfig] = None):
        """
        Initialize ML ensemble integrator.
        
        Args:
            config: MLModelConfig for ensemble parameters
        """
        self.config = config or MLModelConfig()
        self.logger = logging.getLogger(__name__)
        
        # Model storage
        self.rf_model = None
        self.gb_model = None
        self.nn_model = None
        
        # Results storage
        self.ensemble_results = None
        self.fitted = False
        
    def fit_all_models(self, 
                      X: Union[np.ndarray, pd.DataFrame], 
                      y: Union[np.ndarray, pd.Series],
                      feature_names: Optional[List[str]] = None,
                      test_size: float = 0.2) -> Dict[str, MLModelResults]:
        """
        Fit all ML models in the ensemble.
        
        Args:
            X: Feature matrix
            y: Target variable
            feature_names: Names of features
            test_size: Proportion of data for testing
            
        Returns:
            Dictionary with results from all models
        """
        self.logger.info("Fitting all ML models in ensemble")
        
        results = {}
        
        # Fit Random Forest
        try:
            self.logger.info("Fitting Random Forest model...")
            self.rf_model = RandomForestThresholdDetector(self.config)
            results['RandomForest'] = self.rf_model.fit(X, y, feature_names, test_size)
        except Exception as e:
            self.logger.error(f"Random Forest fitting failed: {e}")
            results['RandomForest'] = None
        
        # Fit Gradient Boosting
        try:
            self.logger.info("Fitting Gradient Boosting model...")
            self.gb_model = GradientBoostingComplexRelationships(self.config)
            results['GradientBoosting'] = self.gb_model.fit(X, y, feature_names, test_size)
        except Exception as e:
            self.logger.error(f"Gradient Boosting fitting failed: {e}")
            results['GradientBoosting'] = None
        
        # Fit Neural Network
        try:
            if TORCH_AVAILABLE:
                self.logger.info("Fitting Neural Network ensemble...")
                self.nn_model = NeuralNetworkEnsemble(self.config)
                results['NeuralNetwork'] = self.nn_model.fit(X, y, feature_names, test_size, n_ensemble=3)
            else:
                self.logger.warning("PyTorch not available, skipping Neural Network")
                results['NeuralNetwork'] = None
        except Exception as e:
            self.logger.error(f"Neural Network fitting failed: {e}")
            results['NeuralNetwork'] = None
        
        self.fitted = True
        self.logger.info("All ML models fitted successfully")
        
        return results
    
    def create_ensemble_predictions(self, X: Union[np.ndarray, pd.DataFrame]) -> Dict[str, Any]:
        """
        Create ensemble predictions combining all models.
        
        Args:
            X: Feature matrix
            
        Returns:
            Dictionary with ensemble predictions and analysis
        """
        if not self.fitted:
            raise ValueError("Models must be fitted before creating ensemble predictions")
        
        predictions = {}
        
        # Get predictions from each model
        if self.rf_model and self.rf_model.fitted:
            predictions['RandomForest'] = self.rf_model.predict(X)
        
        if self.gb_model and self.gb_model.fitted:
            predictions['GradientBoosting'] = self.gb_model.predict(X)
        
        if self.nn_model and self.nn_model.fitted:
            predictions['NeuralNetwork'] = self.nn_model.predict(X)
        
        if not predictions:
            raise ValueError("No fitted models available for ensemble predictions")
        
        # Create ensemble prediction (weighted average)
        weights = self.config.ensemble_weights or {name: 1.0 for name in predictions.keys()}
        
        # Normalize weights
        total_weight = sum(weights.get(name, 1.0) for name in predictions.keys())
        normalized_weights = {name: weights.get(name, 1.0) / total_weight for name in predictions.keys()}
        
        # Calculate weighted ensemble prediction
        ensemble_pred = np.zeros_like(list(predictions.values())[0])
        for name, pred in predictions.items():
            ensemble_pred += normalized_weights[name] * pred
        
        # Calculate ensemble uncertainty (standard deviation across models)
        pred_array = np.array(list(predictions.values()))
        ensemble_std = np.std(pred_array, axis=0)
        
        # Model agreement analysis
        model_agreement = {}
        model_names = list(predictions.keys())
        for i, name1 in enumerate(model_names):
            for j, name2 in enumerate(model_names[i+1:], i+1):
                correlation = np.corrcoef(predictions[name1], predictions[name2])[0, 1]
                model_agreement[f"{name1}_vs_{name2}"] = correlation
        
        self.ensemble_results = {
            'individual_predictions': predictions,
            'ensemble_prediction': ensemble_pred,
            'ensemble_std': ensemble_std,
            'weights': normalized_weights,
            'model_agreement': model_agreement,
            'n_models': len(predictions)
        }
        
        return self.ensemble_results
    
    def get_model_comparison_analysis(self) -> Dict[str, Any]:
        """
        Compare performance across all models.
        
        Returns:
            Dictionary with model comparison analysis
        """
        if not self.fitted:
            raise ValueError("Models must be fitted before comparison analysis")
        
        comparison = {
            'model_performance': {},
            'feature_importance_comparison': {},
            'prediction_correlation_matrix': {}
        }
        
        # Performance comparison
        models = {
            'RandomForest': self.rf_model,
            'GradientBoosting': self.gb_model,
            'NeuralNetwork': self.nn_model
        }
        
        for name, model in models.items():
            if model and model.fitted:
                results = model.results
                comparison['model_performance'][name] = {
                    'train_score': results.train_score,
                    'test_score': results.test_score,
                    'training_time': results.training_time,
                    'cv_mean': getattr(results, 'cv_mean', None),
                    'cv_std': getattr(results, 'cv_std', None)
                }
                
                # Feature importance comparison
                if hasattr(model, 'results') and model.results.feature_importance is not None:
                    top_features = model.results.feature_importance.head(10)
                    comparison['feature_importance_comparison'][name] = {
                        'top_features': top_features['feature'].tolist(),
                        'importance_values': top_features.iloc[:, 1].tolist()  # Second column is importance
                    }
        
        return comparison
    
    def plot_ensemble_analysis(self, 
                             X: Union[np.ndarray, pd.DataFrame],
                             y: Optional[Union[np.ndarray, pd.Series]] = None,
                             figsize: Tuple[int, int] = (18, 12)) -> plt.Figure:
        """
        Plot comprehensive ensemble analysis.
        
        Args:
            X: Feature matrix
            y: True values (optional)
            figsize: Figure size
            
        Returns:
            Matplotlib figure with ensemble analysis
        """
        if not self.fitted:
            raise ValueError("Models must be fitted before plotting")
        
        # Get ensemble predictions
        if self.ensemble_results is None:
            self.create_ensemble_predictions(X)
        
        fig, axes = plt.subplots(2, 3, figsize=figsize)
        axes = axes.ravel()
        
        # Plot 1: Individual model predictions
        predictions = self.ensemble_results['individual_predictions']
        ensemble_pred = self.ensemble_results['ensemble_prediction']
        
        ax = axes[0]
        for name, pred in predictions.items():
            ax.plot(pred[:100], label=name, alpha=0.7)  # Limit to first 100 points for clarity
        ax.plot(ensemble_pred[:100], label='Ensemble', linewidth=2, color='black')
        ax.set_title('Model Predictions Comparison')
        ax.set_xlabel('Sample Index')
        ax.set_ylabel('Prediction')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 2: Model agreement correlation matrix
        ax = axes[1]
        agreement = self.ensemble_results['model_agreement']
        if agreement:
            model_names = list(predictions.keys())
            n_models = len(model_names)
            corr_matrix = np.eye(n_models)
            
            for i, name1 in enumerate(model_names):
                for j, name2 in enumerate(model_names):
                    if i != j:
                        key = f"{name1}_vs_{name2}" if f"{name1}_vs_{name2}" in agreement else f"{name2}_vs_{name1}"
                        if key in agreement:
                            corr_matrix[i, j] = agreement[key]
            
            im = ax.imshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1)
            ax.set_xticks(range(n_models))
            ax.set_yticks(range(n_models))
            ax.set_xticklabels(model_names, rotation=45)
            ax.set_yticklabels(model_names)
            ax.set_title('Model Agreement Correlation')
            plt.colorbar(im, ax=ax)
        
        # Plot 3: Ensemble uncertainty
        ax = axes[2]
        ensemble_std = self.ensemble_results['ensemble_std']
        ax.plot(ensemble_std[:100])
        ax.set_title('Ensemble Prediction Uncertainty')
        ax.set_xlabel('Sample Index')
        ax.set_ylabel('Standard Deviation')
        ax.grid(True, alpha=0.3)
        
        # Plot 4: Model performance comparison
        ax = axes[3]
        comparison = self.get_model_comparison_analysis()
        performance = comparison['model_performance']
        
        model_names = list(performance.keys())
        test_scores = [performance[name]['test_score'] for name in model_names]
        
        bars = ax.bar(model_names, test_scores)
        ax.set_title('Model Performance Comparison (Test R²)')
        ax.set_ylabel('R² Score')
        ax.tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar, score in zip(bars, test_scores):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{score:.3f}', ha='center', va='bottom')
        
        # Plot 5: Residual analysis (if true values provided)
        ax = axes[4]
        if y is not None:
            if isinstance(y, pd.Series):
                y = y.values
            residuals = y - ensemble_pred
            ax.scatter(ensemble_pred, residuals, alpha=0.6)
            ax.axhline(y=0, color='r', linestyle='--')
            ax.set_xlabel('Predicted Values')
            ax.set_ylabel('Residuals')
            ax.set_title('Ensemble Residual Plot')
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, 'True values not provided\nfor residual analysis', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Residual Analysis (N/A)')
        
        # Plot 6: Feature importance comparison (if available)
        ax = axes[5]
        feature_importance = comparison.get('feature_importance_comparison', {})
        
        if feature_importance:
            # Plot top features from each model
            colors = ['blue', 'green', 'red', 'orange']
            for i, (model_name, importance_data) in enumerate(feature_importance.items()):
                features = importance_data['top_features'][:5]  # Top 5 features
                importances = importance_data['importance_values'][:5]
                
                y_pos = np.arange(len(features)) + i * 0.25
                ax.barh(y_pos, importances, height=0.2, 
                       label=model_name, color=colors[i % len(colors)], alpha=0.7)
            
            ax.set_yticks(np.arange(len(features)) + 0.25)
            ax.set_yticklabels(features)
            ax.set_xlabel('Feature Importance')
            ax.set_title('Top Feature Importance Comparison')
            ax.legend()
        else:
            ax.text(0.5, 0.5, 'Feature importance\nnot available', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Feature Importance (N/A)')
        
        plt.tight_layout()
        return fig
        
        # Fit Random Forest
        try:
            self.logger.info("Fitting Random Forest model")
            self.rf_model = RandomForestThresholdDetector(self.config)
            rf_results = self.rf_model.fit(X, y, feature_names, test_size, hyperparameter_tuning=True)
            results['random_forest'] = rf_results
            self.logger.info("Random Forest fitting completed successfully")
        except Exception as e:
            self.logger.error(f"Random Forest fitting failed: {e}")
            results['random_forest'] = None
        
        # Fit Gradient Boosting
        try:
            self.logger.info("Fitting Gradient Boosting model")
            self.gb_model = GradientBoostingComplexRelationships(self.config)
            gb_results = self.gb_model.fit(X, y, feature_names, test_size, hyperparameter_tuning=True)
            results['gradient_boosting'] = gb_results
            self.logger.info("Gradient Boosting fitting completed successfully")
        except Exception as e:
            self.logger.error(f"Gradient Boosting fitting failed: {e}")
            results['gradient_boosting'] = None
        
        # Fit Neural Network (if PyTorch available)
        if TORCH_AVAILABLE:
            try:
                self.logger.info("Fitting Neural Network ensemble")
                self.nn_model = NeuralNetworkEnsemble(self.config)
                nn_results = self.nn_model.fit(X, y, feature_names, test_size, n_ensemble=3)
                results['neural_network'] = nn_results
                self.logger.info("Neural Network fitting completed successfully")
            except Exception as e:
                self.logger.error(f"Neural Network fitting failed: {e}")
                results['neural_network'] = None
        else:
            self.logger.warning("PyTorch not available. Skipping Neural Network model.")
            results['neural_network'] = None
        
        self.fitted = True
        return results
    
    def create_ensemble_predictions(self, 
                                  X: Union[np.ndarray, pd.DataFrame],
                                  weights: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
        """
        Create ensemble predictions combining all fitted models.
        
        Args:
            X: Feature matrix for predictions
            weights: Weights for each model (auto-calculated if None)
            
        Returns:
            Dictionary with ensemble predictions and analysis
        """
        if not self.fitted:
            raise ValueError("Models must be fitted before creating ensemble predictions")
        
        self.logger.info("Creating ensemble predictions")
        
        # Get predictions from each model
        predictions = {}
        model_weights = {}
        
        if self.rf_model and self.rf_model.fitted:
            try:
                predictions['random_forest'] = self.rf_model.predict(X)
                model_weights['random_forest'] = self.rf_model.results.test_score or 0.0
            except Exception as e:
                self.logger.warning(f"Random Forest prediction failed: {e}")
        
        if self.gb_model and self.gb_model.fitted:
            try:
                predictions['gradient_boosting'] = self.gb_model.predict(X)
                model_weights['gradient_boosting'] = self.gb_model.results.test_score or 0.0
            except Exception as e:
                self.logger.warning(f"Gradient Boosting prediction failed: {e}")
        
        if self.nn_model and self.nn_model.fitted:
            try:
                predictions['neural_network'] = self.nn_model.predict(X)
                model_weights['neural_network'] = self.nn_model.results.test_score or 0.0
            except Exception as e:
                self.logger.warning(f"Neural Network prediction failed: {e}")
        
        if not predictions:
            raise ValueError("No models available for ensemble predictions")
        
        # Calculate weights
        if weights is None:
            # Use performance-based weighting
            total_weight = sum(max(0, w) for w in model_weights.values())
            if total_weight > 0:
                weights = {k: max(0, v) / total_weight for k, v in model_weights.items()}
            else:
                # Equal weights if no valid performance scores
                weights = {k: 1.0 / len(predictions) for k in predictions.keys()}
        
        # Create weighted ensemble prediction
        ensemble_pred = np.zeros_like(list(predictions.values())[0])
        
        for model_name, pred in predictions.items():
            weight = weights.get(model_name, 0.0)
            ensemble_pred += weight * pred
        
        # Calculate ensemble statistics
        pred_array = np.array(list(predictions.values()))
        ensemble_std = np.std(pred_array, axis=0)
        ensemble_min = np.min(pred_array, axis=0)
        ensemble_max = np.max(pred_array, axis=0)
        
        # Model agreement analysis
        model_agreement = {}
        for i, (name1, pred1) in enumerate(predictions.items()):
            for j, (name2, pred2) in enumerate(predictions.items()):
                if i < j:
                    correlation = np.corrcoef(pred1, pred2)[0, 1]
                    model_agreement[f"{name1}_vs_{name2}"] = correlation
        
        ensemble_results = {
            'ensemble_prediction': ensemble_pred,
            'individual_predictions': predictions,
            'weights': weights,
            'ensemble_std': ensemble_std,
            'ensemble_min': ensemble_min,
            'ensemble_max': ensemble_max,
            'model_agreement': model_agreement,
            'n_models': len(predictions),
            'model_names': list(predictions.keys())
        }
        
        self.ensemble_results = ensemble_results
        self.logger.info(f"Ensemble predictions created using {len(predictions)} models")
        
        return ensemble_results
    
    def get_model_comparison_analysis(self) -> Dict[str, Any]:
        """
        Get comprehensive comparison analysis of all models.
        
        Returns:
            Dictionary with model comparison analysis
        """
        if not self.fitted:
            raise ValueError("Models must be fitted before comparison analysis")
        
        comparison = {
            'model_performance': {},
            'feature_importance_comparison': {},
            'prediction_correlation': {},
            'model_characteristics': {}
        }
        
        # Performance comparison
        models = {
            'random_forest': self.rf_model,
            'gradient_boosting': self.gb_model,
            'neural_network': self.nn_model
        }
        
        for name, model in models.items():
            if model and model.fitted:
                results = model.results
                comparison['model_performance'][name] = {
                    'train_score': results.train_score,
                    'test_score': results.test_score,
                    'cv_mean': results.cv_mean,
                    'cv_std': results.cv_std,
                    'training_time': results.training_time
                }
                
                # Feature importance (if available)
                if results.feature_importance is not None:
                    top_features = results.feature_importance.head(10)
                    comparison['feature_importance_comparison'][name] = {
                        'top_features': top_features['feature'].tolist(),
                        'importance_values': top_features['importance'].tolist()
                    }
                
                # Model characteristics
                comparison['model_characteristics'][name] = results.model_specific
        
        return comparison
    
    def plot_ensemble_analysis(self, 
                             X: Union[np.ndarray, pd.DataFrame],
                             y: Optional[Union[np.ndarray, pd.Series]] = None,
                             figsize: Tuple[int, int] = (15, 12)) -> plt.Figure:
        """
        Create comprehensive ensemble analysis plots.
        
        Args:
            X: Feature matrix
            y: True values (optional, for residual analysis)
            figsize: Figure size
            
        Returns:
            Matplotlib figure with ensemble analysis
        """
        if not self.fitted:
            raise ValueError("Models must be fitted before plotting")
        
        # Get ensemble predictions
        if self.ensemble_results is None:
            self.create_ensemble_predictions(X)
        
        fig, axes = plt.subplots(2, 3, figsize=figsize)
        axes = axes.ravel()
        
        # Plot 1: Individual model predictions
        predictions = self.ensemble_results['individual_predictions']
        ensemble_pred = self.ensemble_results['ensemble_prediction']
        
        ax = axes[0]
        for name, pred in predictions.items():
            ax.plot(pred[:100], label=name, alpha=0.7)  # Limit to first 100 points for clarity
        ax.plot(ensemble_pred[:100], label='Ensemble', linewidth=2, color='black')
        ax.set_title('Model Predictions Comparison')
        ax.set_xlabel('Sample Index')
        ax.set_ylabel('Prediction')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 2: Model agreement correlation matrix
        ax = axes[1]
        agreement = self.ensemble_results['model_agreement']
        if agreement:
            model_names = list(predictions.keys())
            n_models = len(model_names)
            corr_matrix = np.eye(n_models)
            
            for i, name1 in enumerate(model_names):
                for j, name2 in enumerate(model_names):
                    if i != j:
                        key = f"{name1}_vs_{name2}" if f"{name1}_vs_{name2}" in agreement else f"{name2}_vs_{name1}"
                        if key in agreement:
                            corr_matrix[i, j] = agreement[key]
            
            im = ax.imshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1)
            ax.set_xticks(range(n_models))
            ax.set_yticks(range(n_models))
            ax.set_xticklabels(model_names, rotation=45)
            ax.set_yticklabels(model_names)
            ax.set_title('Model Agreement Correlation')
            plt.colorbar(im, ax=ax)
        
        # Plot 3: Ensemble uncertainty
        ax = axes[2]
        ensemble_std = self.ensemble_results['ensemble_std']
        ax.plot(ensemble_std[:100])
        ax.set_title('Ensemble Prediction Uncertainty')
        ax.set_xlabel('Sample Index')
        ax.set_ylabel('Standard Deviation')
        ax.grid(True, alpha=0.3)
        
        # Plot 4: Model performance comparison
        ax = axes[3]
        comparison = self.get_model_comparison_analysis()
        performance = comparison['model_performance']
        
        model_names = list(performance.keys())
        test_scores = [performance[name]['test_score'] for name in model_names]
        
        bars = ax.bar(model_names, test_scores)
        ax.set_title('Model Performance Comparison (Test R²)')
        ax.set_ylabel('R² Score')
        ax.tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar, score in zip(bars, test_scores):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{score:.3f}', ha='center', va='bottom')
        
        # Plot 5: Residual analysis (if true values provided)
        ax = axes[4]
        if y is not None:
            if isinstance(y, pd.Series):
                y = y.values
            residuals = y - ensemble_pred
            ax.scatter(ensemble_pred, residuals, alpha=0.6)
            ax.axhline(y=0, color='r', linestyle='--')
            ax.set_xlabel('Predicted Values')
            ax.set_ylabel('Residuals')
            ax.set_title('Ensemble Residual Plot')
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, 'True values not provided\nfor residual analysis', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Residual Analysis (N/A)')
        
        # Plot 6: Feature importance comparison (if available)
        ax = axes[5]
        feature_importance = comparison.get('feature_importance_comparison', {})
        
        if feature_importance:
            # Plot top features from each model
            colors = ['blue', 'green', 'red', 'orange']
            for i, (model_name, importance_data) in enumerate(feature_importance.items()):
                features = importance_data['top_features'][:5]  # Top 5 features
                importances = importance_data['importance_values'][:5]
                
                y_pos = np.arange(len(features)) + i * 0.25
                ax.barh(y_pos, importances, height=0.2, 
                       label=model_name, color=colors[i % len(colors)], alpha=0.7)
            
            ax.set_yticks(np.arange(len(features)) + 0.25)
            ax.set_yticklabels(features)
            ax.set_xlabel('Feature Importance')
            ax.set_title('Top Feature Importance Comparison')
            ax.legend()
        else:
            ax.text(0.5, 0.5, 'Feature importance\nnot available', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Feature Importance (N/A)')
        
        plt.tight_layout()
        return fig