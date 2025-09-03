"""
Model Comparison and Ensemble Integration Framework

This module implements comprehensive model comparison capabilities and ensemble methods
for combining statistical and machine learning models in QE hypothesis testing.
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Optional, Any, Tuple, Union, Callable
from dataclasses import dataclass, field
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Statistical libraries
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    mean_absolute_percentage_error
)
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

# Import model classes
from .ml_models import MLModelResults, RandomForestThresholdDetector, GradientBoostingComplexRelationships
from .models import HansenThresholdRegression, LocalProjections, InstrumentalVariablesRegression


@dataclass
class ModelComparisonConfig:
    """Configuration for model comparison and ensemble methods"""
    
    # Cross-validation settings
    cv_folds: int = 5
    cv_scoring: List[str] = field(default_factory=lambda: ['neg_mean_squared_error', 'r2', 'neg_mean_absolute_error'])
    test_size: float = 0.2
    
    # Information criteria settings
    use_aic: bool = True
    use_bic: bool = True
    use_hqic: bool = True
    
    # Ensemble settings
    ensemble_methods: List[str] = field(default_factory=lambda: ['simple_average', 'weighted_average', 'stacking'])
    ensemble_weights: Optional[Dict[str, float]] = None
    
    # Uncertainty quantification
    uncertainty_method: str = 'bootstrap'  # Options: bootstrap, quantile_regression, ensemble_variance
    bootstrap_samples: int = 1000
    confidence_levels: List[float] = field(default_factory=lambda: [0.05, 0.95])
    
    # Performance metrics
    primary_metric: str = 'r2'
    secondary_metrics: List[str] = field(default_factory=lambda: ['mse', 'mae', 'mape'])
    
    # Model selection criteria
    selection_criterion: str = 'cv_score'  # Options: cv_score, information_criteria, ensemble
    
    # Sensitivity analysis
    enable_sensitivity_analysis: bool = True
    sensitivity_parameters: List[str] = field(default_factory=lambda: ['sample_period', 'feature_selection', 'hyperparameters'])


@dataclass
class ModelPerformanceMetrics:
    """Performance metrics for a single model"""
    
    model_name: str
    model_type: str  # 'statistical' or 'ml'
    
    # Cross-validation scores
    cv_scores: Dict[str, List[float]] = field(default_factory=dict)
    cv_mean: Dict[str, float] = field(default_factory=dict)
    cv_std: Dict[str, float] = field(default_factory=dict)
    
    # Out-of-sample performance
    test_r2: Optional[float] = None
    test_mse: Optional[float] = None
    test_mae: Optional[float] = None
    test_mape: Optional[float] = None
    
    # Information criteria (for statistical models)
    aic: Optional[float] = None
    bic: Optional[float] = None
    hqic: Optional[float] = None
    
    # Model-specific metrics
    model_specific_metrics: Dict[str, Any] = field(default_factory=dict)
    
    # Computational metrics
    training_time: Optional[float] = None
    prediction_time: Optional[float] = None
    
    # Metadata
    n_parameters: Optional[int] = None
    n_observations: Optional[int] = None
    fitted_successfully: bool = False


@dataclass
class EnsembleResults:
    """Results from ensemble model combination"""
    
    ensemble_method: str
    models_included: List[str]
    
    # Ensemble predictions
    ensemble_predictions: np.ndarray
    individual_predictions: Dict[str, np.ndarray]
    
    # Ensemble weights
    model_weights: Dict[str, float]
    
    # Performance metrics
    ensemble_r2: float
    ensemble_mse: float
    ensemble_mae: float
    
    # Uncertainty quantification
    prediction_intervals: Dict[str, np.ndarray] = field(default_factory=dict)
    prediction_variance: Optional[np.ndarray] = None
    
    # Comparison with individual models
    performance_improvement: Dict[str, float] = field(default_factory=dict)
    
    # Metadata
    n_models: int = 0
    ensemble_timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


class ModelComparisonFramework:
    """
    Framework for comparing statistical and ML models with cross-validation scoring
    and information criteria.
    
    Implements Requirements 5.3, 5.4 for model comparison and selection.
    """
    
    def __init__(self, config: Optional[ModelComparisonConfig] = None):
        """
        Initialize model comparison framework.
        
        Args:
            config: ModelComparisonConfig for comparison parameters
        """
        self.config = config or ModelComparisonConfig()
        self.logger = logging.getLogger(__name__)
        
        # Storage for models and results
        self.models: Dict[str, Any] = {}
        self.model_results: Dict[str, Any] = {}
        self.performance_metrics: Dict[str, ModelPerformanceMetrics] = {}
        
        # Comparison results
        self.comparison_results: Optional[Dict[str, Any]] = None
        self.best_model: Optional[str] = None
        
        self.logger.info("ModelComparisonFramework initialized")
    
    def add_model(self, 
                  name: str, 
                  model: Any, 
                  model_type: str,
                  fitted_results: Optional[Any] = None) -> None:
        """
        Add a model to the comparison framework.
        
        Args:
            name: Unique name for the model
            model: Model instance (fitted or unfitted)
            model_type: Type of model ('statistical' or 'ml')
            fitted_results: Pre-computed results if model is already fitted
        """
        self.models[name] = {
            'model': model,
            'type': model_type,
            'fitted_results': fitted_results
        }
        
        self.logger.info(f"Added {model_type} model: {name}")
    
    def run_cross_validation(self, 
                           X: Union[np.ndarray, pd.DataFrame],
                           y: Union[np.ndarray, pd.Series],
                           model_names: Optional[List[str]] = None) -> Dict[str, ModelPerformanceMetrics]:
        """
        Run cross-validation for all models or specified models.
        
        Args:
            X: Feature matrix
            y: Target variable
            model_names: List of model names to evaluate (if None, evaluates all)
            
        Returns:
            Dictionary of performance metrics for each model
        """
        self.logger.info("Running cross-validation for model comparison")
        
        if model_names is None:
            model_names = list(self.models.keys())
        
        # Convert to numpy arrays if needed
        if isinstance(X, pd.DataFrame):
            X_array = X.values
            feature_names = X.columns.tolist()
        else:
            X_array = X
            feature_names = [f"feature_{i}" for i in range(X.shape[1])]
        
        if isinstance(y, pd.Series):
            y_array = y.values
        else:
            y_array = y
        
        # Setup time series cross-validation
        tscv = TimeSeriesSplit(n_splits=self.config.cv_folds)
        
        cv_results = {}
        
        for model_name in model_names:
            if model_name not in self.models:
                self.logger.warning(f"Model {model_name} not found, skipping")
                continue
            
            self.logger.info(f"Cross-validating model: {model_name}")
            
            model_info = self.models[model_name]
            model = model_info['model']
            model_type = model_info['type']
            
            # Initialize performance metrics
            metrics = ModelPerformanceMetrics(
                model_name=model_name,
                model_type=model_type,
                n_observations=len(y_array)
            )
            
            try:
                import time
                start_time = time.time()
                
                if model_type == 'ml':
                    # Handle ML models
                    cv_scores = self._cross_validate_ml_model(model, X_array, y_array, tscv, feature_names)
                    
                elif model_type == 'statistical':
                    # Handle statistical models
                    cv_scores = self._cross_validate_statistical_model(model, X_array, y_array, tscv)
                
                else:
                    raise ValueError(f"Unknown model type: {model_type}")
                
                # Store cross-validation results
                metrics.cv_scores = cv_scores
                metrics.cv_mean = {metric: np.mean(scores) for metric, scores in cv_scores.items()}
                metrics.cv_std = {metric: np.std(scores) for metric, scores in cv_scores.items()}
                metrics.training_time = time.time() - start_time
                metrics.fitted_successfully = True
                
                # Calculate out-of-sample performance
                test_metrics = self._calculate_out_of_sample_performance(
                    model, X_array, y_array, model_type, feature_names
                )
                
                metrics.test_r2 = test_metrics.get('r2')
                metrics.test_mse = test_metrics.get('mse')
                metrics.test_mae = test_metrics.get('mae')
                metrics.test_mape = test_metrics.get('mape')
                
                # Calculate information criteria for statistical models
                if model_type == 'statistical':
                    ic_metrics = self._calculate_information_criteria(model, X_array, y_array)
                    metrics.aic = ic_metrics.get('aic')
                    metrics.bic = ic_metrics.get('bic')
                    metrics.hqic = ic_metrics.get('hqic')
                
                self.logger.info(f"Cross-validation completed for {model_name}")
                
            except Exception as e:
                self.logger.error(f"Error in cross-validation for {model_name}: {str(e)}")
                metrics.fitted_successfully = False
                metrics.model_specific_metrics['error'] = str(e)
            
            cv_results[model_name] = metrics
            self.performance_metrics[model_name] = metrics
        
        self.logger.info("Cross-validation completed for all models")
        return cv_results
    
    def _cross_validate_ml_model(self, 
                                model: Any, 
                                X: np.ndarray, 
                                y: np.ndarray, 
                                cv: TimeSeriesSplit,
                                feature_names: List[str]) -> Dict[str, List[float]]:
        """Cross-validate ML models."""
        cv_scores = {metric: [] for metric in self.config.cv_scoring}
        
        for train_idx, val_idx in cv.split(X):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            # Create a copy of the model for this fold
            if hasattr(model, '__class__'):
                if 'RandomForest' in model.__class__.__name__:
                    fold_model = RandomForestThresholdDetector(model.config)
                elif 'GradientBoosting' in model.__class__.__name__:
                    fold_model = GradientBoostingComplexRelationships(model.config)
                else:
                    # Generic ML model handling
                    fold_model = model.__class__()
            else:
                fold_model = model
            
            # Fit model on training fold
            if hasattr(fold_model, 'fit'):
                fold_model.fit(X_train, y_train, feature_names=feature_names, hyperparameter_tuning=False)
                
                # Make predictions
                if hasattr(fold_model, 'predict'):
                    y_pred = fold_model.predict(X_val)
                else:
                    continue
            else:
                continue
            
            # Calculate metrics for this fold
            if 'neg_mean_squared_error' in self.config.cv_scoring:
                cv_scores['neg_mean_squared_error'].append(-mean_squared_error(y_val, y_pred))
            if 'r2' in self.config.cv_scoring:
                cv_scores['r2'].append(r2_score(y_val, y_pred))
            if 'neg_mean_absolute_error' in self.config.cv_scoring:
                cv_scores['neg_mean_absolute_error'].append(-mean_absolute_error(y_val, y_pred))
        
        return cv_scores
    
    def _cross_validate_statistical_model(self, 
                                        model: Any, 
                                        X: np.ndarray, 
                                        y: np.ndarray, 
                                        cv: TimeSeriesSplit) -> Dict[str, List[float]]:
        """Cross-validate statistical models."""
        cv_scores = {metric: [] for metric in self.config.cv_scoring}
        
        for train_idx, val_idx in cv.split(X):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            try:
                # Fit statistical model
                if hasattr(model, 'fit'):
                    if 'Hansen' in model.__class__.__name__:
                        # Hansen threshold model needs threshold variable
                        # Use first column as threshold variable for simplicity
                        threshold_var = X_train[:, 0] if X_train.shape[1] > 0 else y_train
                        model.fit(y_train, X_train, threshold_var)
                        
                        # Predict using fitted model
                        if hasattr(model, 'predict'):
                            y_pred = model.predict(X_val)
                        else:
                            # Simple prediction for Hansen model
                            y_pred = np.full(len(y_val), np.mean(y_train))
                    
                    elif 'LocalProjections' in model.__class__.__name__:
                        # Local projections model
                        shock = X_train[:, 0] if X_train.shape[1] > 0 else np.zeros(len(y_train))
                        controls = pd.DataFrame(X_train[:, 1:]) if X_train.shape[1] > 1 else None
                        
                        model.fit(pd.Series(y_train), pd.Series(shock), controls)
                        
                        # Simple prediction
                        y_pred = np.full(len(y_val), np.mean(y_train))
                    
                    else:
                        # Generic statistical model
                        model.fit(X_train, y_train)
                        y_pred = model.predict(X_val) if hasattr(model, 'predict') else np.full(len(y_val), np.mean(y_train))
                
                else:
                    # Model doesn't have fit method, skip
                    continue
                
                # Calculate metrics for this fold
                if 'neg_mean_squared_error' in self.config.cv_scoring:
                    cv_scores['neg_mean_squared_error'].append(-mean_squared_error(y_val, y_pred))
                if 'r2' in self.config.cv_scoring:
                    cv_scores['r2'].append(r2_score(y_val, y_pred))
                if 'neg_mean_absolute_error' in self.config.cv_scoring:
                    cv_scores['neg_mean_absolute_error'].append(-mean_absolute_error(y_val, y_pred))
                    
            except Exception as e:
                self.logger.warning(f"Error in statistical model cross-validation fold: {str(e)}")
                # Add NaN for failed folds
                for metric in cv_scores:
                    cv_scores[metric].append(np.nan)
        
        # Remove NaN values
        for metric in cv_scores:
            cv_scores[metric] = [score for score in cv_scores[metric] if not np.isnan(score)]
        
        return cv_scores
    
    def _calculate_out_of_sample_performance(self, 
                                           model: Any, 
                                           X: np.ndarray, 
                                           y: np.ndarray,
                                           model_type: str,
                                           feature_names: List[str]) -> Dict[str, float]:
        """Calculate out-of-sample performance metrics."""
        # Split data
        split_idx = int(len(X) * (1 - self.config.test_size))
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        try:
            if model_type == 'ml':
                # Fit ML model
                if hasattr(model, 'fit'):
                    model.fit(X_train, y_train, feature_names=feature_names, hyperparameter_tuning=False)
                    y_pred = model.predict(X_test)
                else:
                    return {}
            
            elif model_type == 'statistical':
                # Fit statistical model
                if 'Hansen' in model.__class__.__name__:
                    threshold_var = X_train[:, 0] if X_train.shape[1] > 0 else y_train
                    model.fit(y_train, X_train, threshold_var)
                    y_pred = np.full(len(y_test), np.mean(y_train))  # Simplified prediction
                
                elif 'LocalProjections' in model.__class__.__name__:
                    shock = X_train[:, 0] if X_train.shape[1] > 0 else np.zeros(len(y_train))
                    controls = pd.DataFrame(X_train[:, 1:]) if X_train.shape[1] > 1 else None
                    model.fit(pd.Series(y_train), pd.Series(shock), controls)
                    y_pred = np.full(len(y_test), np.mean(y_train))  # Simplified prediction
                
                else:
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test) if hasattr(model, 'predict') else np.full(len(y_test), np.mean(y_train))
            
            # Calculate metrics
            metrics = {
                'r2': r2_score(y_test, y_pred),
                'mse': mean_squared_error(y_test, y_pred),
                'mae': mean_absolute_error(y_test, y_pred),
                'mape': mean_absolute_percentage_error(y_test, y_pred) if np.all(y_test != 0) else np.nan
            }
            
            return metrics
            
        except Exception as e:
            self.logger.warning(f"Error calculating out-of-sample performance: {str(e)}")
            return {}
    
    def _calculate_information_criteria(self, 
                                      model: Any, 
                                      X: np.ndarray, 
                                      y: np.ndarray) -> Dict[str, float]:
        """Calculate information criteria for statistical models."""
        try:
            # This is a simplified implementation
            # In practice, would need model-specific implementations
            
            n = len(y)
            
            # Estimate number of parameters (simplified)
            if hasattr(model, 'n_params'):
                k = model.n_params
            elif hasattr(model, 'beta1') and hasattr(model, 'beta2'):
                # Hansen model
                k = len(model.beta1) + len(model.beta2) + 1  # +1 for threshold
            else:
                k = X.shape[1] + 1  # Features + intercept
            
            # Calculate residual sum of squares (simplified)
            if hasattr(model, 'residuals'):
                rss = np.sum(model.residuals ** 2)
            else:
                # Estimate RSS
                y_pred = np.full(len(y), np.mean(y))  # Simplified prediction
                rss = np.sum((y - y_pred) ** 2)
            
            # Calculate information criteria
            aic = n * np.log(rss / n) + 2 * k
            bic = n * np.log(rss / n) + k * np.log(n)
            hqic = n * np.log(rss / n) + 2 * k * np.log(np.log(n))
            
            return {
                'aic': aic,
                'bic': bic,
                'hqic': hqic
            }
            
        except Exception as e:
            self.logger.warning(f"Error calculating information criteria: {str(e)}")
            return {}
    
    def compare_models(self) -> Dict[str, Any]:
        """
        Compare all models and select the best performing model.
        
        Returns:
            Dictionary with comprehensive model comparison results
        """
        if not self.performance_metrics:
            raise ValueError("No performance metrics available. Run cross_validation first.")
        
        self.logger.info("Comparing models based on performance metrics")
        
        # Create comparison DataFrame
        comparison_data = []
        
        for model_name, metrics in self.performance_metrics.items():
            if not metrics.fitted_successfully:
                continue
            
            row = {
                'model_name': model_name,
                'model_type': metrics.model_type,
                'cv_r2_mean': metrics.cv_mean.get('r2', np.nan),
                'cv_r2_std': metrics.cv_std.get('r2', np.nan),
                'cv_mse_mean': -metrics.cv_mean.get('neg_mean_squared_error', np.nan),
                'cv_mse_std': metrics.cv_std.get('neg_mean_squared_error', np.nan),
                'test_r2': metrics.test_r2,
                'test_mse': metrics.test_mse,
                'test_mae': metrics.test_mae,
                'aic': metrics.aic,
                'bic': metrics.bic,
                'training_time': metrics.training_time,
                'n_observations': metrics.n_observations
            }
            
            comparison_data.append(row)
        
        comparison_df = pd.DataFrame(comparison_data)
        
        if comparison_df.empty:
            raise ValueError("No successfully fitted models to compare")
        
        # Model selection based on criterion
        if self.config.selection_criterion == 'cv_score':
            # Select based on cross-validation R²
            best_idx = comparison_df['cv_r2_mean'].idxmax()
            selection_metric = 'cv_r2_mean'
            
        elif self.config.selection_criterion == 'information_criteria':
            # Select based on BIC (lower is better)
            bic_available = comparison_df['bic'].notna()
            if bic_available.any():
                best_idx = comparison_df.loc[bic_available, 'bic'].idxmin()
                selection_metric = 'bic'
            else:
                # Fallback to CV score
                best_idx = comparison_df['cv_r2_mean'].idxmax()
                selection_metric = 'cv_r2_mean (fallback)'
        
        else:
            # Default to CV score
            best_idx = comparison_df['cv_r2_mean'].idxmax()
            selection_metric = 'cv_r2_mean'
        
        self.best_model = comparison_df.loc[best_idx, 'model_name']
        
        # Calculate rankings
        comparison_df['r2_rank'] = comparison_df['cv_r2_mean'].rank(ascending=False)
        comparison_df['mse_rank'] = comparison_df['cv_mse_mean'].rank(ascending=True)
        
        # Statistical significance tests between models
        significance_tests = self._perform_model_significance_tests()
        
        # Compile results
        self.comparison_results = {
            'comparison_table': comparison_df,
            'best_model': self.best_model,
            'selection_criterion': selection_metric,
            'selection_metric_value': comparison_df.loc[best_idx, selection_metric.split(' ')[0]],
            'model_rankings': {
                'by_r2': comparison_df.nlargest(len(comparison_df), 'cv_r2_mean')[['model_name', 'cv_r2_mean']].to_dict('records'),
                'by_mse': comparison_df.nsmallest(len(comparison_df), 'cv_mse_mean')[['model_name', 'cv_mse_mean']].to_dict('records')
            },
            'significance_tests': significance_tests,
            'summary_statistics': {
                'n_models_compared': len(comparison_df),
                'n_statistical_models': len(comparison_df[comparison_df['model_type'] == 'statistical']),
                'n_ml_models': len(comparison_df[comparison_df['model_type'] == 'ml']),
                'best_r2': comparison_df['cv_r2_mean'].max(),
                'worst_r2': comparison_df['cv_r2_mean'].min(),
                'r2_range': comparison_df['cv_r2_mean'].max() - comparison_df['cv_r2_mean'].min()
            }
        }
        
        self.logger.info(f"Model comparison completed. Best model: {self.best_model}")
        return self.comparison_results
    
    def _perform_model_significance_tests(self) -> Dict[str, Any]:
        """Perform statistical significance tests between models."""
        significance_tests = {}
        
        # Get models with CV scores
        models_with_scores = {}
        for model_name, metrics in self.performance_metrics.items():
            if metrics.fitted_successfully and 'r2' in metrics.cv_scores:
                models_with_scores[model_name] = metrics.cv_scores['r2']
        
        if len(models_with_scores) < 2:
            return {'message': 'Insufficient models for significance testing'}
        
        # Pairwise t-tests
        model_names = list(models_with_scores.keys())
        pairwise_tests = {}
        
        for i, model1 in enumerate(model_names):
            for j, model2 in enumerate(model_names[i+1:], i+1):
                scores1 = models_with_scores[model1]
                scores2 = models_with_scores[model2]
                
                if len(scores1) > 1 and len(scores2) > 1:
                    try:
                        t_stat, p_value = stats.ttest_rel(scores1, scores2)
                        pairwise_tests[f"{model1}_vs_{model2}"] = {
                            't_statistic': t_stat,
                            'p_value': p_value,
                            'significant': p_value < 0.05,
                            'mean_diff': np.mean(scores1) - np.mean(scores2)
                        }
                    except Exception as e:
                        pairwise_tests[f"{model1}_vs_{model2}"] = {'error': str(e)}
        
        significance_tests['pairwise_tests'] = pairwise_tests
        
        # Overall ANOVA test
        if len(models_with_scores) > 2:
            try:
                all_scores = [scores for scores in models_with_scores.values()]
                f_stat, p_value = stats.f_oneway(*all_scores)
                significance_tests['anova_test'] = {
                    'f_statistic': f_stat,
                    'p_value': p_value,
                    'significant': p_value < 0.05
                }
            except Exception as e:
                significance_tests['anova_test'] = {'error': str(e)}
        
        return significance_tests
    
    def plot_model_comparison(self, figsize: Tuple[int, int] = (12, 8)) -> plt.Figure:
        """
        Create visualization of model comparison results.
        
        Args:
            figsize: Figure size
            
        Returns:
            Matplotlib figure
        """
        if self.comparison_results is None:
            raise ValueError("No comparison results available. Run compare_models first.")
        
        comparison_df = self.comparison_results['comparison_table']
        
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        
        # R² comparison
        axes[0, 0].bar(comparison_df['model_name'], comparison_df['cv_r2_mean'], 
                       yerr=comparison_df['cv_r2_std'], capsize=5)
        axes[0, 0].set_title('Cross-Validation R² Scores')
        axes[0, 0].set_ylabel('R² Score')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # MSE comparison
        axes[0, 1].bar(comparison_df['model_name'], comparison_df['cv_mse_mean'], 
                       yerr=comparison_df['cv_mse_std'], capsize=5)
        axes[0, 1].set_title('Cross-Validation MSE Scores')
        axes[0, 1].set_ylabel('MSE')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # Training time comparison
        if comparison_df['training_time'].notna().any():
            axes[1, 0].bar(comparison_df['model_name'], comparison_df['training_time'])
            axes[1, 0].set_title('Training Time Comparison')
            axes[1, 0].set_ylabel('Time (seconds)')
            axes[1, 0].tick_params(axis='x', rotation=45)
        
        # Information criteria (if available)
        ic_available = comparison_df[['aic', 'bic']].notna().any(axis=1)
        if ic_available.any():
            ic_df = comparison_df.loc[ic_available, ['model_name', 'aic', 'bic']]
            x_pos = np.arange(len(ic_df))
            width = 0.35
            
            axes[1, 1].bar(x_pos - width/2, ic_df['aic'], width, label='AIC')
            axes[1, 1].bar(x_pos + width/2, ic_df['bic'], width, label='BIC')
            axes[1, 1].set_title('Information Criteria')
            axes[1, 1].set_ylabel('IC Value')
            axes[1, 1].set_xticks(x_pos)
            axes[1, 1].set_xticklabels(ic_df['model_name'], rotation=45)
            axes[1, 1].legend()
        
        plt.tight_layout()
        return fig
    
    def get_model_selection_summary(self) -> Dict[str, Any]:
        """
        Get a summary of model selection results.
        
        Returns:
            Dictionary with model selection summary
        """
        if self.comparison_results is None:
            raise ValueError("No comparison results available. Run compare_models first.")
        
        comparison_df = self.comparison_results['comparison_table']
        best_model_row = comparison_df[comparison_df['model_name'] == self.best_model].iloc[0]
        
        summary = {
            'best_model': {
                'name': self.best_model,
                'type': best_model_row['model_type'],
                'cv_r2_mean': best_model_row['cv_r2_mean'],
                'cv_r2_std': best_model_row['cv_r2_std'],
                'test_r2': best_model_row['test_r2'],
                'test_mse': best_model_row['test_mse']
            },
            'selection_criterion': self.comparison_results['selection_criterion'],
            'performance_gap': {
                'r2_advantage': best_model_row['cv_r2_mean'] - comparison_df['cv_r2_mean'].median(),
                'rank_position': int(best_model_row['r2_rank'])
            },
            'model_diversity': {
                'statistical_models': self.comparison_results['summary_statistics']['n_statistical_models'],
                'ml_models': self.comparison_results['summary_statistics']['n_ml_models'],
                'total_models': self.comparison_results['summary_statistics']['n_models_compared']
            },
            'robustness_indicators': {
                'cv_stability': best_model_row['cv_r2_std'],
                'generalization_gap': abs(best_model_row['cv_r2_mean'] - best_model_row['test_r2']) if pd.notna(best_model_row['test_r2']) else None
            }
        }
        
        return summary


class EnsemblePredictionFramework:
    """
    Framework for ensemble prediction methods combining multiple models.
    
    Implements Requirements 5.4, 5.6 for ensemble methods and uncertainty quantification.
    """
    
    def __init__(self, config: Optional[ModelComparisonConfig] = None):
        """
        Initialize ensemble prediction framework.
        
        Args:
            config: ModelComparisonConfig for ensemble parameters
        """
        self.config = config or ModelComparisonConfig()
        self.logger = logging.getLogger(__name__)
        
        # Storage for models and predictions
        self.models: Dict[str, Any] = {}
        self.model_predictions: Dict[str, np.ndarray] = {}
        self.ensemble_results: Dict[str, EnsembleResults] = {}
        
        self.logger.info("EnsemblePredictionFramework initialized")
    
    def add_model_predictions(self, 
                            model_name: str, 
                            predictions: np.ndarray,
                            model_instance: Optional[Any] = None) -> None:
        """
        Add model predictions to the ensemble.
        
        Args:
            model_name: Name of the model
            predictions: Model predictions array
            model_instance: Optional model instance for additional methods
        """
        self.model_predictions[model_name] = predictions
        if model_instance is not None:
            self.models[model_name] = model_instance
        
        self.logger.info(f"Added predictions for model: {model_name}")
    
    def create_simple_average_ensemble(self, 
                                     model_names: Optional[List[str]] = None) -> EnsembleResults:
        """
        Create simple average ensemble of model predictions.
        
        Args:
            model_names: List of model names to include (if None, uses all)
            
        Returns:
            EnsembleResults object
        """
        if model_names is None:
            model_names = list(self.model_predictions.keys())
        
        if len(model_names) < 2:
            raise ValueError("Need at least 2 models for ensemble")
        
        self.logger.info(f"Creating simple average ensemble with {len(model_names)} models")
        
        # Get predictions for selected models
        predictions_matrix = np.column_stack([
            self.model_predictions[name] for name in model_names
        ])
        
        # Calculate simple average
        ensemble_predictions = np.mean(predictions_matrix, axis=1)
        
        # Equal weights for all models
        equal_weight = 1.0 / len(model_names)
        model_weights = {name: equal_weight for name in model_names}
        
        # Create individual predictions dict
        individual_predictions = {name: self.model_predictions[name] for name in model_names}
        
        # Calculate ensemble variance for uncertainty quantification
        prediction_variance = np.var(predictions_matrix, axis=1)
        
        # Create results object
        results = EnsembleResults(
            ensemble_method='simple_average',
            models_included=model_names,
            ensemble_predictions=ensemble_predictions,
            individual_predictions=individual_predictions,
            model_weights=model_weights,
            ensemble_r2=0.0,  # Will be calculated when true values are provided
            ensemble_mse=0.0,
            ensemble_mae=0.0,
            prediction_variance=prediction_variance,
            n_models=len(model_names)
        )
        
        self.ensemble_results['simple_average'] = results
        return results
    
    def create_weighted_average_ensemble(self, 
                                       model_names: Optional[List[str]] = None,
                                       weights: Optional[Dict[str, float]] = None,
                                       weight_method: str = 'performance') -> EnsembleResults:
        """
        Create weighted average ensemble with performance-based or custom weights.
        
        Args:
            model_names: List of model names to include
            weights: Custom weights dict (if None, calculates based on weight_method)
            weight_method: Method for calculating weights ('performance', 'inverse_variance', 'custom')
            
        Returns:
            EnsembleResults object
        """
        if model_names is None:
            model_names = list(self.model_predictions.keys())
        
        if len(model_names) < 2:
            raise ValueError("Need at least 2 models for ensemble")
        
        self.logger.info(f"Creating weighted average ensemble with {len(model_names)} models")
        
        # Get predictions for selected models
        predictions_matrix = np.column_stack([
            self.model_predictions[name] for name in model_names
        ])
        
        # Calculate weights
        if weights is None:
            if weight_method == 'performance':
                weights = self._calculate_performance_weights(model_names)
            elif weight_method == 'inverse_variance':
                weights = self._calculate_inverse_variance_weights(model_names, predictions_matrix)
            else:
                # Default to equal weights
                equal_weight = 1.0 / len(model_names)
                weights = {name: equal_weight for name in model_names}
        
        # Normalize weights to sum to 1
        total_weight = sum(weights.values())
        normalized_weights = {name: w / total_weight for name, w in weights.items()}
        
        # Calculate weighted average
        weight_array = np.array([normalized_weights[name] for name in model_names])
        ensemble_predictions = np.average(predictions_matrix, axis=1, weights=weight_array)
        
        # Create individual predictions dict
        individual_predictions = {name: self.model_predictions[name] for name in model_names}
        
        # Calculate weighted variance for uncertainty quantification
        weighted_mean = ensemble_predictions
        weighted_variance = np.average(
            (predictions_matrix - weighted_mean.reshape(-1, 1)) ** 2,
            axis=1, weights=weight_array
        )
        
        # Create results object
        results = EnsembleResults(
            ensemble_method='weighted_average',
            models_included=model_names,
            ensemble_predictions=ensemble_predictions,
            individual_predictions=individual_predictions,
            model_weights=normalized_weights,
            ensemble_r2=0.0,  # Will be calculated when true values are provided
            ensemble_mse=0.0,
            ensemble_mae=0.0,
            prediction_variance=weighted_variance,
            n_models=len(model_names)
        )
        
        self.ensemble_results['weighted_average'] = results
        return results
    
    def create_stacking_ensemble(self, 
                               model_names: Optional[List[str]] = None,
                               meta_learner: str = 'linear',
                               X_meta: Optional[np.ndarray] = None,
                               y_true: Optional[np.ndarray] = None) -> EnsembleResults:
        """
        Create stacking ensemble with meta-learner.
        
        Args:
            model_names: List of model names to include
            meta_learner: Type of meta-learner ('linear', 'ridge', 'lasso')
            X_meta: Features for meta-learner training
            y_true: True values for meta-learner training
            
        Returns:
            EnsembleResults object
        """
        if model_names is None:
            model_names = list(self.model_predictions.keys())
        
        if len(model_names) < 2:
            raise ValueError("Need at least 2 models for ensemble")
        
        if X_meta is None or y_true is None:
            self.logger.warning("Stacking ensemble requires training data. Using simple average instead.")
            return self.create_simple_average_ensemble(model_names)
        
        self.logger.info(f"Creating stacking ensemble with {len(model_names)} models")
        
        from sklearn.linear_model import LinearRegression, Ridge, Lasso
        from sklearn.model_selection import cross_val_predict
        
        # Get predictions for selected models
        predictions_matrix = np.column_stack([
            self.model_predictions[name] for name in model_names
        ])
        
        # Initialize meta-learner
        if meta_learner == 'ridge':
            meta_model = Ridge(alpha=1.0)
        elif meta_learner == 'lasso':
            meta_model = Lasso(alpha=1.0)
        else:
            meta_model = LinearRegression()
        
        try:
            # Fit meta-learner using cross-validation to avoid overfitting
            meta_predictions = cross_val_predict(
                meta_model, predictions_matrix, y_true, cv=5
            )
            
            # Fit final meta-learner on all data
            meta_model.fit(predictions_matrix, y_true)
            
            # Get ensemble predictions
            ensemble_predictions = meta_model.predict(predictions_matrix)
            
            # Extract model weights (coefficients)
            if hasattr(meta_model, 'coef_'):
                coefficients = meta_model.coef_
                # Normalize coefficients to get weights
                total_coef = np.sum(np.abs(coefficients))
                if total_coef > 0:
                    model_weights = {
                        name: abs(coef) / total_coef 
                        for name, coef in zip(model_names, coefficients)
                    }
                else:
                    equal_weight = 1.0 / len(model_names)
                    model_weights = {name: equal_weight for name in model_names}
            else:
                equal_weight = 1.0 / len(model_names)
                model_weights = {name: equal_weight for name in model_names}
            
        except Exception as e:
            self.logger.warning(f"Error in stacking ensemble: {str(e)}. Using simple average.")
            return self.create_simple_average_ensemble(model_names)
        
        # Create individual predictions dict
        individual_predictions = {name: self.model_predictions[name] for name in model_names}
        
        # Calculate prediction variance
        prediction_variance = np.var(predictions_matrix, axis=1)
        
        # Create results object
        results = EnsembleResults(
            ensemble_method='stacking',
            models_included=model_names,
            ensemble_predictions=ensemble_predictions,
            individual_predictions=individual_predictions,
            model_weights=model_weights,
            ensemble_r2=0.0,  # Will be calculated when true values are provided
            ensemble_mse=0.0,
            ensemble_mae=0.0,
            prediction_variance=prediction_variance,
            n_models=len(model_names)
        )
        
        # Store meta-learner for future predictions
        results.model_specific_metrics = {'meta_learner': meta_model}
        
        self.ensemble_results['stacking'] = results
        return results
    
    def _calculate_performance_weights(self, model_names: List[str]) -> Dict[str, float]:
        """Calculate weights based on model performance (requires external performance metrics)."""
        # This is a placeholder - in practice, would use performance metrics from ModelComparisonFramework
        # For now, return equal weights
        equal_weight = 1.0 / len(model_names)
        return {name: equal_weight for name in model_names}
    
    def _calculate_inverse_variance_weights(self, 
                                          model_names: List[str], 
                                          predictions_matrix: np.ndarray) -> Dict[str, float]:
        """Calculate weights based on inverse variance of predictions."""
        weights = {}
        
        for i, name in enumerate(model_names):
            pred_variance = np.var(predictions_matrix[:, i])
            # Inverse variance weight (higher variance = lower weight)
            weights[name] = 1.0 / (pred_variance + 1e-8)  # Add small constant to avoid division by zero
        
        return weights
    
    def quantify_uncertainty(self, 
                           ensemble_results: EnsembleResults,
                           y_true: Optional[np.ndarray] = None,
                           method: str = 'ensemble_variance') -> Dict[str, np.ndarray]:
        """
        Quantify uncertainty in ensemble predictions.
        
        Args:
            ensemble_results: EnsembleResults object
            y_true: True values for calibration (optional)
            method: Uncertainty quantification method
            
        Returns:
            Dictionary with uncertainty measures
        """
        self.logger.info(f"Quantifying uncertainty using method: {method}")
        
        uncertainty_measures = {}
        
        if method == 'ensemble_variance':
            # Use variance across model predictions
            predictions_matrix = np.column_stack([
                pred for pred in ensemble_results.individual_predictions.values()
            ])
            
            uncertainty_measures['prediction_std'] = np.std(predictions_matrix, axis=1)
            uncertainty_measures['prediction_variance'] = np.var(predictions_matrix, axis=1)
            
            # Calculate prediction intervals using ensemble variance
            std_dev = uncertainty_measures['prediction_std']
            for confidence_level in self.config.confidence_levels:
                alpha = 1 - confidence_level
                z_score = stats.norm.ppf(1 - alpha/2)
                
                lower_bound = ensemble_results.ensemble_predictions - z_score * std_dev
                upper_bound = ensemble_results.ensemble_predictions + z_score * std_dev
                
                uncertainty_measures[f'lower_bound_{confidence_level}'] = lower_bound
                uncertainty_measures[f'upper_bound_{confidence_level}'] = upper_bound
        
        elif method == 'bootstrap' and y_true is not None:
            # Bootstrap uncertainty quantification
            n_samples = len(ensemble_results.ensemble_predictions)
            bootstrap_predictions = []
            
            for _ in range(self.config.bootstrap_samples):
                # Bootstrap sample indices
                bootstrap_indices = np.random.choice(n_samples, n_samples, replace=True)
                
                # Get bootstrap predictions for each model
                bootstrap_ensemble = []
                for model_name in ensemble_results.models_included:
                    model_pred = ensemble_results.individual_predictions[model_name]
                    bootstrap_pred = model_pred[bootstrap_indices]
                    bootstrap_ensemble.append(bootstrap_pred)
                
                # Calculate ensemble prediction for bootstrap sample
                bootstrap_matrix = np.column_stack(bootstrap_ensemble)
                weights = np.array([ensemble_results.model_weights[name] for name in ensemble_results.models_included])
                bootstrap_ensemble_pred = np.average(bootstrap_matrix, axis=1, weights=weights)
                bootstrap_predictions.append(bootstrap_ensemble_pred)
            
            # Calculate bootstrap statistics
            bootstrap_array = np.array(bootstrap_predictions)
            uncertainty_measures['bootstrap_std'] = np.std(bootstrap_array, axis=0)
            uncertainty_measures['bootstrap_mean'] = np.mean(bootstrap_array, axis=0)
            
            # Bootstrap confidence intervals
            for confidence_level in self.config.confidence_levels:
                alpha = 1 - confidence_level
                lower_percentile = (alpha/2) * 100
                upper_percentile = (1 - alpha/2) * 100
                
                uncertainty_measures[f'bootstrap_lower_{confidence_level}'] = np.percentile(
                    bootstrap_array, lower_percentile, axis=0
                )
                uncertainty_measures[f'bootstrap_upper_{confidence_level}'] = np.percentile(
                    bootstrap_array, upper_percentile, axis=0
                )
        
        elif method == 'quantile_regression':
            # Placeholder for quantile regression uncertainty
            # Would require fitting quantile regression models
            self.logger.warning("Quantile regression uncertainty not implemented. Using ensemble variance.")
            return self.quantify_uncertainty(ensemble_results, y_true, 'ensemble_variance')
        
        return uncertainty_measures
    
    def evaluate_ensemble_performance(self, 
                                    ensemble_results: EnsembleResults,
                                    y_true: np.ndarray) -> EnsembleResults:
        """
        Evaluate ensemble performance against true values.
        
        Args:
            ensemble_results: EnsembleResults object
            y_true: True values for evaluation
            
        Returns:
            Updated EnsembleResults with performance metrics
        """
        self.logger.info("Evaluating ensemble performance")
        
        # Calculate ensemble performance metrics
        ensemble_results.ensemble_r2 = r2_score(y_true, ensemble_results.ensemble_predictions)
        ensemble_results.ensemble_mse = mean_squared_error(y_true, ensemble_results.ensemble_predictions)
        ensemble_results.ensemble_mae = mean_absolute_error(y_true, ensemble_results.ensemble_predictions)
        
        # Calculate individual model performance for comparison
        individual_performance = {}
        for model_name, predictions in ensemble_results.individual_predictions.items():
            individual_performance[model_name] = {
                'r2': r2_score(y_true, predictions),
                'mse': mean_squared_error(y_true, predictions),
                'mae': mean_absolute_error(y_true, predictions)
            }
        
        # Calculate performance improvement over individual models
        individual_r2_scores = [perf['r2'] for perf in individual_performance.values()]
        individual_mse_scores = [perf['mse'] for perf in individual_performance.values()]
        
        ensemble_results.performance_improvement = {
            'r2_improvement_over_best': ensemble_results.ensemble_r2 - max(individual_r2_scores),
            'r2_improvement_over_average': ensemble_results.ensemble_r2 - np.mean(individual_r2_scores),
            'mse_improvement_over_best': min(individual_mse_scores) - ensemble_results.ensemble_mse,
            'mse_improvement_over_average': np.mean(individual_mse_scores) - ensemble_results.ensemble_mse,
            'individual_performance': individual_performance
        }
        
        self.logger.info(f"Ensemble R²: {ensemble_results.ensemble_r2:.4f}, "
                        f"Best individual R²: {max(individual_r2_scores):.4f}")
        
        return ensemble_results
    
    def perform_sensitivity_analysis(self, 
                                   model_names: Optional[List[str]] = None,
                                   sensitivity_params: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Perform sensitivity analysis for key model assumptions.
        
        Args:
            model_names: List of model names to analyze
            sensitivity_params: List of parameters to test sensitivity for
            
        Returns:
            Dictionary with sensitivity analysis results
        """
        if model_names is None:
            model_names = list(self.model_predictions.keys())
        
        if sensitivity_params is None:
            sensitivity_params = self.config.sensitivity_parameters
        
        self.logger.info(f"Performing sensitivity analysis for {len(sensitivity_params)} parameters")
        
        sensitivity_results = {}
        
        for param in sensitivity_params:
            if param == 'sample_period':
                # Test sensitivity to different sample periods
                sensitivity_results[param] = self._test_sample_period_sensitivity(model_names)
            
            elif param == 'feature_selection':
                # Test sensitivity to feature selection
                sensitivity_results[param] = self._test_feature_selection_sensitivity(model_names)
            
            elif param == 'hyperparameters':
                # Test sensitivity to hyperparameter changes
                sensitivity_results[param] = self._test_hyperparameter_sensitivity(model_names)
            
            else:
                self.logger.warning(f"Unknown sensitivity parameter: {param}")
        
        return sensitivity_results
    
    def _test_sample_period_sensitivity(self, model_names: List[str]) -> Dict[str, Any]:
        """Test sensitivity to different sample periods."""
        # Placeholder implementation
        return {
            'parameter': 'sample_period',
            'test_periods': ['2008-2015', '2016-2023', '2010-2020'],
            'message': 'Sample period sensitivity analysis requires retraining models with different periods'
        }
    
    def _test_feature_selection_sensitivity(self, model_names: List[str]) -> Dict[str, Any]:
        """Test sensitivity to feature selection."""
        # Placeholder implementation
        return {
            'parameter': 'feature_selection',
            'message': 'Feature selection sensitivity analysis requires access to original features'
        }
    
    def _test_hyperparameter_sensitivity(self, model_names: List[str]) -> Dict[str, Any]:
        """Test sensitivity to hyperparameter changes."""
        # Placeholder implementation
        return {
            'parameter': 'hyperparameters',
            'message': 'Hyperparameter sensitivity analysis requires retraining models with different parameters'
        }
    
    def plot_ensemble_results(self, 
                            ensemble_results: EnsembleResults,
                            y_true: Optional[np.ndarray] = None,
                            uncertainty_measures: Optional[Dict[str, np.ndarray]] = None,
                            figsize: Tuple[int, int] = (15, 10)) -> plt.Figure:
        """
        Plot ensemble results with individual model predictions and uncertainty.
        
        Args:
            ensemble_results: EnsembleResults object
            y_true: True values for comparison
            uncertainty_measures: Uncertainty quantification results
            figsize: Figure size
            
        Returns:
            Matplotlib figure
        """
        n_plots = 3 if y_true is not None else 2
        fig, axes = plt.subplots(1, n_plots, figsize=figsize)
        
        if n_plots == 2:
            axes = [axes[0], axes[1]]
        
        # Plot 1: Individual model predictions vs ensemble
        x_axis = np.arange(len(ensemble_results.ensemble_predictions))
        
        # Plot individual model predictions
        for model_name, predictions in ensemble_results.individual_predictions.items():
            axes[0].plot(x_axis, predictions, alpha=0.6, label=model_name, linewidth=1)
        
        # Plot ensemble prediction
        axes[0].plot(x_axis, ensemble_results.ensemble_predictions, 
                    color='black', linewidth=2, label='Ensemble')
        
        # Add uncertainty bands if available
        if uncertainty_measures and 'prediction_std' in uncertainty_measures:
            std_dev = uncertainty_measures['prediction_std']
            axes[0].fill_between(
                x_axis,
                ensemble_results.ensemble_predictions - 2*std_dev,
                ensemble_results.ensemble_predictions + 2*std_dev,
                alpha=0.2, color='black', label='±2σ Uncertainty'
            )
        
        axes[0].set_title('Individual Models vs Ensemble Predictions')
        axes[0].set_xlabel('Observation')
        axes[0].set_ylabel('Prediction')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Plot 2: Model weights
        model_names = list(ensemble_results.model_weights.keys())
        weights = list(ensemble_results.model_weights.values())
        
        bars = axes[1].bar(model_names, weights)
        axes[1].set_title('Ensemble Model Weights')
        axes[1].set_xlabel('Model')
        axes[1].set_ylabel('Weight')
        axes[1].tick_params(axis='x', rotation=45)
        
        # Add weight values on bars
        for bar, weight in zip(bars, weights):
            height = bar.get_height()
            axes[1].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                        f'{weight:.3f}', ha='center', va='bottom')
        
        # Plot 3: Predictions vs True values (if available)
        if y_true is not None:
            # Scatter plot of predictions vs true values
            axes[2].scatter(y_true, ensemble_results.ensemble_predictions, 
                           alpha=0.6, label='Ensemble')
            
            # Plot individual models with different colors
            colors = plt.cm.tab10(np.linspace(0, 1, len(ensemble_results.individual_predictions)))
            for i, (model_name, predictions) in enumerate(ensemble_results.individual_predictions.items()):
                axes[2].scatter(y_true, predictions, alpha=0.4, s=20, 
                               color=colors[i], label=model_name)
            
            # Perfect prediction line
            min_val = min(y_true.min(), ensemble_results.ensemble_predictions.min())
            max_val = max(y_true.max(), ensemble_results.ensemble_predictions.max())
            axes[2].plot([min_val, max_val], [min_val, max_val], 
                        'r--', alpha=0.8, label='Perfect Prediction')
            
            axes[2].set_title('Predictions vs True Values')
            axes[2].set_xlabel('True Values')
            axes[2].set_ylabel('Predictions')
            axes[2].legend()
            axes[2].grid(True, alpha=0.3)
            
            # Add R² annotation
            axes[2].text(0.05, 0.95, f'Ensemble R² = {ensemble_results.ensemble_r2:.3f}',
                        transform=axes[2].transAxes, verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        return fig
    
    def get_ensemble_summary(self, ensemble_results: EnsembleResults) -> Dict[str, Any]:
        """
        Get comprehensive summary of ensemble results.
        
        Args:
            ensemble_results: EnsembleResults object
            
        Returns:
            Dictionary with ensemble summary
        """
        summary = {
            'ensemble_method': ensemble_results.ensemble_method,
            'models_included': ensemble_results.models_included,
            'n_models': ensemble_results.n_models,
            'model_weights': ensemble_results.model_weights,
            'performance_metrics': {
                'r2': ensemble_results.ensemble_r2,
                'mse': ensemble_results.ensemble_mse,
                'mae': ensemble_results.ensemble_mae
            },
            'uncertainty_info': {
                'has_prediction_variance': ensemble_results.prediction_variance is not None,
                'has_prediction_intervals': len(ensemble_results.prediction_intervals) > 0,
                'mean_prediction_std': np.mean(np.sqrt(ensemble_results.prediction_variance)) if ensemble_results.prediction_variance is not None else None
            },
            'performance_improvement': ensemble_results.performance_improvement,
            'ensemble_timestamp': ensemble_results.ensemble_timestamp
        }
        
        return summary