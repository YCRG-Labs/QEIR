"""
Cross-Validation and Sensitivity Analysis Framework

This module implements time series cross-validation for temporal data,
sensitivity analysis for different data periods and specifications,
and robustness tables comparing results across models.
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Optional, Any, Tuple, Union, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Statistical libraries
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

# Import existing components
from .model_diagnostics import DiagnosticConfig, DiagnosticResults
from .model_comparison import ModelComparisonFramework, ModelPerformanceMetrics


@dataclass
class CrossValidationConfig:
    """Configuration for time series cross-validation"""
    
    # Cross-validation settings
    cv_method: str = 'time_series_split'  # Options: time_series_split, expanding_window, rolling_window
    n_splits: int = 5
    test_size: Optional[int] = None  # Number of observations in test set
    gap: int = 0  # Gap between train and test sets
    
    # Expanding window settings
    min_train_size: Optional[int] = None
    
    # Rolling window settings
    window_size: Optional[int] = None
    
    # Performance metrics
    scoring_metrics: List[str] = field(default_factory=lambda: [
        'r2', 'mse', 'mae', 'mape', 'directional_accuracy'
    ])
    
    # Validation settings
    validate_stationarity: bool = True
    validate_autocorrelation: bool = True
    
    # Parallel processing
    n_jobs: int = 1


@dataclass
class SensitivityConfig:
    """Configuration for sensitivity analysis"""
    
    # Data period sensitivity
    period_analysis: bool = True
    period_splits: List[str] = field(default_factory=lambda: [
        'pre_crisis', 'crisis', 'post_crisis', 'full_sample'
    ])
    crisis_start: str = '2007-01-01'
    crisis_end: str = '2009-12-31'
    
    # Sample size sensitivity
    sample_size_analysis: bool = True
    sample_size_fractions: List[float] = field(default_factory=lambda: [0.5, 0.7, 0.8, 0.9, 1.0])
    
    # Feature sensitivity
    feature_analysis: bool = True
    feature_subsets: List[str] = field(default_factory=lambda: [
        'core_features', 'extended_features', 'all_features'
    ])
    
    # Hyperparameter sensitivity
    hyperparameter_analysis: bool = True
    hyperparameter_ranges: Dict[str, List[Any]] = field(default_factory=dict)
    
    # Outlier sensitivity
    outlier_analysis: bool = True
    outlier_removal_methods: List[str] = field(default_factory=lambda: [
        'none', 'z_score', 'iqr', 'isolation_forest'
    ])
    outlier_thresholds: List[float] = field(default_factory=lambda: [2.0, 2.5, 3.0])
    
    # Specification sensitivity
    specification_analysis: bool = True
    alternative_specifications: List[str] = field(default_factory=lambda: [
        'linear', 'polynomial', 'interaction_terms', 'log_transform'
    ])


@dataclass
class CrossValidationResults:
    """Results from time series cross-validation"""
    
    cv_method: str
    n_splits: int
    
    # Cross-validation scores
    cv_scores: Dict[str, List[float]] = field(default_factory=dict)
    cv_mean: Dict[str, float] = field(default_factory=dict)
    cv_std: Dict[str, float] = field(default_factory=dict)
    
    # Fold-specific results
    fold_results: List[Dict[str, Any]] = field(default_factory=list)
    
    # Predictions
    out_of_sample_predictions: np.ndarray = field(default_factory=lambda: np.array([]))
    prediction_intervals: Dict[str, np.ndarray] = field(default_factory=dict)
    
    # Validation metrics
    temporal_consistency: Dict[str, float] = field(default_factory=dict)
    prediction_stability: float = 0.0
    
    # Metadata
    total_observations: int = 0
    average_train_size: float = 0.0
    average_test_size: float = 0.0
    cv_timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class SensitivityResults:
    """Results from sensitivity analysis"""
    
    analysis_type: str
    
    # Period sensitivity
    period_results: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    
    # Sample size sensitivity
    sample_size_results: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    
    # Feature sensitivity
    feature_results: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    
    # Hyperparameter sensitivity
    hyperparameter_results: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    
    # Outlier sensitivity
    outlier_results: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    
    # Specification sensitivity
    specification_results: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    
    # Overall sensitivity metrics
    sensitivity_score: float = 0.0
    most_sensitive_factor: str = ""
    least_sensitive_factor: str = ""
    
    # Metadata
    sensitivity_timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


class TimeSeriesCrossValidation:
    """
    Time series cross-validation framework for temporal data.
    
    Implements Requirements 5.3, 5.6 for cross-validation and temporal validation.
    """
    
    def __init__(self, config: Optional[CrossValidationConfig] = None):
        """
        Initialize time series cross-validation framework.
        
        Args:
            config: CrossValidationConfig for CV parameters
        """
        self.config = config or CrossValidationConfig()
        self.logger = logging.getLogger(__name__)
        
        # Storage for CV results
        self.cv_results: Dict[str, CrossValidationResults] = {}
        
        self.logger.info("TimeSeriesCrossValidation initialized")
    
    def cross_validate_model(self,
                           model: Any,
                           X: Union[np.ndarray, pd.DataFrame],
                           y: Union[np.ndarray, pd.Series],
                           model_name: str,
                           dates: Optional[pd.DatetimeIndex] = None) -> CrossValidationResults:
        """
        Perform time series cross-validation for a model.
        
        Args:
            model: Model to cross-validate
            X: Feature matrix
            y: Target variable
            model_name: Name identifier for the model
            dates: Optional datetime index for temporal analysis
            
        Returns:
            CrossValidationResults with comprehensive CV metrics
        """
        self.logger.info(f"Running time series cross-validation for model: {model_name}")
        
        # Convert to numpy arrays
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
        
        # Setup cross-validation splitter
        cv_splitter = self._setup_cv_splitter(len(y_array))
        
        # Initialize results
        results = CrossValidationResults(
            cv_method=self.config.cv_method,
            n_splits=self.config.n_splits,
            total_observations=len(y_array)
        )
        
        # Initialize score storage
        for metric in self.config.scoring_metrics:
            results.cv_scores[metric] = []
        
        # Perform cross-validation
        fold_predictions = []
        fold_actuals = []
        train_sizes = []
        test_sizes = []
        
        for fold_idx, (train_idx, test_idx) in enumerate(cv_splitter.split(X_array)):
            self.logger.info(f"Processing fold {fold_idx + 1}/{self.config.n_splits}")
            
            # Split data
            X_train, X_test = X_array[train_idx], X_array[test_idx]
            y_train, y_test = y_array[train_idx], y_array[test_idx]
            
            train_sizes.append(len(X_train))
            test_sizes.append(len(X_test))
            
            try:
                # Fit model on training fold
                fold_model = self._fit_fold_model(model, X_train, y_train, feature_names)
                
                # Make predictions
                y_pred = self._predict_fold_model(fold_model, X_test, model)
                
                # Calculate fold metrics
                fold_metrics = self._calculate_fold_metrics(y_test, y_pred)
                
                # Store fold results
                fold_result = {
                    'fold': fold_idx,
                    'train_size': len(X_train),
                    'test_size': len(X_test),
                    'train_indices': train_idx.tolist(),
                    'test_indices': test_idx.tolist(),
                    'metrics': fold_metrics,
                    'predictions': y_pred.tolist(),
                    'actuals': y_test.tolist()
                }
                
                if dates is not None:
                    fold_result['train_dates'] = dates[train_idx].tolist()
                    fold_result['test_dates'] = dates[test_idx].tolist()
                
                results.fold_results.append(fold_result)
                
                # Accumulate scores
                for metric, score in fold_metrics.items():
                    if metric in results.cv_scores:
                        results.cv_scores[metric].append(score)
                
                # Store predictions for out-of-sample analysis
                fold_predictions.extend(y_pred)
                fold_actuals.extend(y_test)
                
            except Exception as e:
                self.logger.warning(f"Fold {fold_idx} failed: {str(e)}")
                # Add NaN scores for failed fold
                for metric in self.config.scoring_metrics:
                    results.cv_scores[metric].append(np.nan)
        
        # Calculate summary statistics
        for metric in self.config.scoring_metrics:
            scores = [s for s in results.cv_scores[metric] if not np.isnan(s)]
            if scores:
                results.cv_mean[metric] = np.mean(scores)
                results.cv_std[metric] = np.std(scores)
            else:
                results.cv_mean[metric] = np.nan
                results.cv_std[metric] = np.nan
        
        # Calculate additional metrics
        results.average_train_size = np.mean(train_sizes)
        results.average_test_size = np.mean(test_sizes)
        
        if fold_predictions and fold_actuals:
            results.out_of_sample_predictions = np.array(fold_predictions)
            
            # Temporal consistency analysis
            if dates is not None:
                results.temporal_consistency = self._analyze_temporal_consistency(
                    results.fold_results, dates
                )
            
            # Prediction stability
            results.prediction_stability = self._calculate_prediction_stability(
                results.fold_results
            )
        
        # Store results
        self.cv_results[model_name] = results
        
        self.logger.info(f"Time series cross-validation completed for {model_name}")
        return results
    
    def _setup_cv_splitter(self, n_samples: int):
        """Setup cross-validation splitter based on configuration."""
        if self.config.cv_method == 'time_series_split':
            return TimeSeriesSplit(
                n_splits=self.config.n_splits,
                test_size=self.config.test_size,
                gap=self.config.gap
            )
        
        elif self.config.cv_method == 'expanding_window':
            return self._create_expanding_window_splitter(n_samples)
        
        elif self.config.cv_method == 'rolling_window':
            return self._create_rolling_window_splitter(n_samples)
        
        else:
            raise ValueError(f"Unknown CV method: {self.config.cv_method}")
    
    def _create_expanding_window_splitter(self, n_samples: int):
        """Create expanding window cross-validation splitter."""
        class ExpandingWindowSplit:
            def __init__(self, n_splits, min_train_size, test_size):
                self.n_splits = n_splits
                self.min_train_size = min_train_size or n_samples // 3
                self.test_size = test_size or n_samples // 10
            
            def split(self, X):
                splits = []
                for i in range(self.n_splits):
                    # Calculate split points
                    test_end = n_samples - i * self.test_size
                    test_start = test_end - self.test_size
                    train_end = test_start
                    train_start = max(0, train_end - self.min_train_size - i * (n_samples // self.n_splits))
                    
                    if train_start < train_end and test_start < test_end:
                        train_idx = np.arange(train_start, train_end)
                        test_idx = np.arange(test_start, test_end)
                        splits.append((train_idx, test_idx))
                
                return reversed(splits)  # Return in chronological order
        
        return ExpandingWindowSplit(
            self.config.n_splits,
            self.config.min_train_size,
            self.config.test_size
        )
    
    def _create_rolling_window_splitter(self, n_samples: int):
        """Create rolling window cross-validation splitter."""
        class RollingWindowSplit:
            def __init__(self, n_splits, window_size, test_size):
                self.n_splits = n_splits
                self.window_size = window_size or n_samples // 2
                self.test_size = test_size or n_samples // 10
            
            def split(self, X):
                splits = []
                step_size = (n_samples - self.window_size - self.test_size) // (self.n_splits - 1)
                
                for i in range(self.n_splits):
                    train_start = i * step_size
                    train_end = train_start + self.window_size
                    test_start = train_end
                    test_end = test_start + self.test_size
                    
                    if test_end <= n_samples:
                        train_idx = np.arange(train_start, train_end)
                        test_idx = np.arange(test_start, test_end)
                        splits.append((train_idx, test_idx))
                
                return splits
        
        return RollingWindowSplit(
            self.config.n_splits,
            self.config.window_size,
            self.config.test_size
        )
    
    def _fit_fold_model(self, original_model: Any, X_train: np.ndarray, y_train: np.ndarray, feature_names: List[str]) -> Any:
        """Fit model on training fold."""
        # Create a copy of the original model
        model_class = original_model.__class__
        
        try:
            # Create new instance
            if hasattr(original_model, 'get_params'):
                params = original_model.get_params()
                fold_model = model_class(**params)
            else:
                fold_model = model_class()
            
            # Fit the model based on type
            if 'Hansen' in model_class.__name__:
                # Hansen threshold model
                threshold_var = X_train[:, 0] if X_train.shape[1] > 0 else y_train
                fold_model.fit(y_train, X_train, threshold_var)
            elif 'LocalProjections' in model_class.__name__:
                # Local projections model
                shock = X_train[:, 0] if X_train.shape[1] > 0 else np.zeros(len(y_train))
                controls = pd.DataFrame(X_train[:, 1:]) if X_train.shape[1] > 1 else None
                fold_model.fit(pd.Series(y_train), pd.Series(shock), controls)
            elif hasattr(fold_model, 'fit'):
                # ML models or other models with fit method
                if 'RandomForest' in model_class.__name__ or 'GradientBoosting' in model_class.__name__:
                    fold_model.fit(X_train, y_train, feature_names=feature_names, hyperparameter_tuning=False)
                else:
                    fold_model.fit(X_train, y_train)
            else:
                raise ValueError(f"Don't know how to fit model of type {model_class.__name__}")
            
            return fold_model
            
        except Exception as e:
            self.logger.warning(f"Error fitting fold model: {str(e)}")
            raise
    
    def _predict_fold_model(self, fold_model: Any, X_test: np.ndarray, original_model: Any) -> np.ndarray:
        """Make predictions using fold model."""
        try:
            if hasattr(fold_model, 'predict'):
                return fold_model.predict(X_test)
            else:
                # Fallback: return mean prediction
                if hasattr(original_model, 'predict'):
                    # Try to use original model structure
                    return np.full(len(X_test), np.mean(X_test))
                else:
                    return np.zeros(len(X_test))
        except Exception as e:
            self.logger.warning(f"Error making fold predictions: {str(e)}")
            return np.zeros(len(X_test))
    
    def _calculate_fold_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Calculate metrics for a single fold."""
        metrics = {}
        
        try:
            if 'r2' in self.config.scoring_metrics:
                metrics['r2'] = r2_score(y_true, y_pred)
            
            if 'mse' in self.config.scoring_metrics:
                metrics['mse'] = mean_squared_error(y_true, y_pred)
            
            if 'mae' in self.config.scoring_metrics:
                metrics['mae'] = mean_absolute_error(y_true, y_pred)
            
            if 'mape' in self.config.scoring_metrics:
                # Avoid division by zero
                non_zero_mask = y_true != 0
                if np.any(non_zero_mask):
                    mape = np.mean(np.abs((y_true[non_zero_mask] - y_pred[non_zero_mask]) / y_true[non_zero_mask])) * 100
                    metrics['mape'] = mape
                else:
                    metrics['mape'] = np.nan
            
            if 'directional_accuracy' in self.config.scoring_metrics:
                # Calculate directional accuracy (for time series)
                if len(y_true) > 1:
                    true_direction = np.diff(y_true) > 0
                    pred_direction = np.diff(y_pred) > 0
                    directional_acc = np.mean(true_direction == pred_direction)
                    metrics['directional_accuracy'] = directional_acc
                else:
                    metrics['directional_accuracy'] = np.nan
        
        except Exception as e:
            self.logger.warning(f"Error calculating fold metrics: {str(e)}")
            for metric in self.config.scoring_metrics:
                metrics[metric] = np.nan
        
        return metrics
    
    def _analyze_temporal_consistency(self, fold_results: List[Dict[str, Any]], dates: pd.DatetimeIndex) -> Dict[str, float]:
        """Analyze temporal consistency of cross-validation results."""
        consistency_metrics = {}
        
        try:
            # Extract performance over time
            fold_r2_scores = []
            fold_dates = []
            
            for fold_result in fold_results:
                if 'metrics' in fold_result and 'r2' in fold_result['metrics']:
                    fold_r2_scores.append(fold_result['metrics']['r2'])
                    # Use middle date of test period
                    test_dates = fold_result.get('test_dates', [])
                    if test_dates:
                        middle_date = pd.to_datetime(test_dates[len(test_dates)//2])
                        fold_dates.append(middle_date)
            
            if len(fold_r2_scores) > 1:
                # Temporal stability (coefficient of variation)
                consistency_metrics['r2_temporal_stability'] = np.std(fold_r2_scores) / np.mean(fold_r2_scores)
                
                # Trend in performance over time
                if len(fold_dates) == len(fold_r2_scores):
                    # Convert dates to numeric for correlation
                    date_numeric = [(d - fold_dates[0]).days for d in fold_dates]
                    correlation, p_value = stats.pearsonr(date_numeric, fold_r2_scores)
                    consistency_metrics['performance_time_correlation'] = correlation
                    consistency_metrics['performance_time_correlation_pvalue'] = p_value
        
        except Exception as e:
            self.logger.warning(f"Error analyzing temporal consistency: {str(e)}")
        
        return consistency_metrics
    
    def _calculate_prediction_stability(self, fold_results: List[Dict[str, Any]]) -> float:
        """Calculate stability of predictions across folds."""
        try:
            # For overlapping test periods, calculate prediction consistency
            all_predictions = []
            all_indices = []
            
            for fold_result in fold_results:
                predictions = fold_result.get('predictions', [])
                test_indices = fold_result.get('test_indices', [])
                
                if predictions and test_indices:
                    all_predictions.extend(predictions)
                    all_indices.extend(test_indices)
            
            if len(all_predictions) > 1:
                # Calculate coefficient of variation as stability measure
                return np.std(all_predictions) / np.mean(np.abs(all_predictions))
            else:
                return 0.0
        
        except Exception as e:
            self.logger.warning(f"Error calculating prediction stability: {str(e)}")
            return 0.0
    
    def compare_cv_results(self, model_names: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Compare cross-validation results across models.
        
        Args:
            model_names: List of model names to compare (if None, compares all)
            
        Returns:
            DataFrame with comparison results
        """
        if model_names is None:
            model_names = list(self.cv_results.keys())
        
        comparison_data = []
        
        for model_name in model_names:
            if model_name not in self.cv_results:
                continue
            
            results = self.cv_results[model_name]
            
            row = {
                'model_name': model_name,
                'cv_method': results.cv_method,
                'n_splits': results.n_splits,
                'avg_train_size': results.average_train_size,
                'avg_test_size': results.average_test_size,
                'prediction_stability': results.prediction_stability
            }
            
            # Add CV metrics
            for metric in self.config.scoring_metrics:
                row[f'{metric}_mean'] = results.cv_mean.get(metric, np.nan)
                row[f'{metric}_std'] = results.cv_std.get(metric, np.nan)
            
            # Add temporal consistency metrics
            for metric, value in results.temporal_consistency.items():
                row[f'temporal_{metric}'] = value
            
            comparison_data.append(row)
        
        return pd.DataFrame(comparison_data)
    
    def plot_cv_results(self, model_name: str, figsize: Tuple[int, int] = (15, 10)) -> plt.Figure:
        """
        Create comprehensive cross-validation results visualization.
        
        Args:
            model_name: Name of model to plot
            figsize: Figure size
            
        Returns:
            Matplotlib figure
        """
        if model_name not in self.cv_results:
            raise ValueError(f"No CV results found for model: {model_name}")
        
        results = self.cv_results[model_name]
        
        fig, axes = plt.subplots(2, 3, figsize=figsize)
        
        # CV scores across folds
        metrics_to_plot = ['r2', 'mse', 'mae']
        for i, metric in enumerate(metrics_to_plot):
            if metric in results.cv_scores:
                scores = [s for s in results.cv_scores[metric] if not np.isnan(s)]
                if scores:
                    axes[0, i].plot(range(len(scores)), scores, 'o-')
                    axes[0, i].axhline(y=np.mean(scores), color='red', linestyle='--', alpha=0.7)
                    axes[0, i].set_title(f'{metric.upper()} Across Folds')
                    axes[0, i].set_xlabel('Fold')
                    axes[0, i].set_ylabel(metric.upper())
        
        # Prediction vs actual for last fold
        if results.fold_results:
            last_fold = results.fold_results[-1]
            predictions = last_fold.get('predictions', [])
            actuals = last_fold.get('actuals', [])
            
            if predictions and actuals:
                axes[1, 0].scatter(actuals, predictions, alpha=0.6)
                min_val = min(min(actuals), min(predictions))
                max_val = max(max(actuals), max(predictions))
                axes[1, 0].plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.7)
                axes[1, 0].set_xlabel('Actual')
                axes[1, 0].set_ylabel('Predicted')
                axes[1, 0].set_title('Predictions vs Actuals (Last Fold)')
        
        # Residuals for last fold
        if results.fold_results:
            last_fold = results.fold_results[-1]
            predictions = last_fold.get('predictions', [])
            actuals = last_fold.get('actuals', [])
            
            if predictions and actuals:
                residuals = np.array(actuals) - np.array(predictions)
                axes[1, 1].scatter(predictions, residuals, alpha=0.6)
                axes[1, 1].axhline(y=0, color='red', linestyle='--', alpha=0.7)
                axes[1, 1].set_xlabel('Predicted')
                axes[1, 1].set_ylabel('Residuals')
                axes[1, 1].set_title('Residual Plot (Last Fold)')
        
        # Performance summary
        summary_text = f"Model: {model_name}\n"
        summary_text += f"CV Method: {results.cv_method}\n"
        summary_text += f"N Splits: {results.n_splits}\n"
        summary_text += f"Avg Train Size: {results.average_train_size:.0f}\n"
        summary_text += f"Avg Test Size: {results.average_test_size:.0f}\n"
        summary_text += f"Prediction Stability: {results.prediction_stability:.3f}\n"
        
        for metric in ['r2', 'mse', 'mae']:
            if metric in results.cv_mean:
                mean_val = results.cv_mean[metric]
                std_val = results.cv_std[metric]
                summary_text += f"{metric.upper()}: {mean_val:.3f} ± {std_val:.3f}\n"
        
        axes[1, 2].text(0.1, 0.5, summary_text, transform=axes[1, 2].transAxes,
                       fontsize=10, verticalalignment='center')
        axes[1, 2].set_xticks([])
        axes[1, 2].set_yticks([])
        axes[1, 2].set_title('CV Summary')
        
        plt.tight_layout()
        return fig


class SensitivityAnalysis:
    """
    Comprehensive sensitivity analysis framework for robustness testing.
    
    Implements Requirements 5.3, 5.6 for sensitivity analysis and robustness testing.
    """
    
    def __init__(self, config: Optional[SensitivityConfig] = None):
        """
        Initialize sensitivity analysis framework.
        
        Args:
            config: SensitivityConfig for sensitivity parameters
        """
        self.config = config or SensitivityConfig()
        self.logger = logging.getLogger(__name__)
        
        # Storage for sensitivity results
        self.sensitivity_results: Dict[str, SensitivityResults] = {}
        
        self.logger.info("SensitivityAnalysis initialized")
    
    def run_comprehensive_sensitivity_analysis(self,
                                             model: Any,
                                             X: Union[np.ndarray, pd.DataFrame],
                                             y: Union[np.ndarray, pd.Series],
                                             model_name: str,
                                             dates: Optional[pd.DatetimeIndex] = None) -> SensitivityResults:
        """
        Run comprehensive sensitivity analysis for a model.
        
        Args:
            model: Model to analyze
            X: Feature matrix
            y: Target variable
            model_name: Name identifier for the model
            dates: Optional datetime index for temporal analysis
            
        Returns:
            SensitivityResults with comprehensive sensitivity metrics
        """
        self.logger.info(f"Running comprehensive sensitivity analysis for model: {model_name}")
        
        # Convert to pandas for easier manipulation
        if isinstance(X, np.ndarray):
            X_df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(X.shape[1])])
        else:
            X_df = X.copy()
        
        if isinstance(y, np.ndarray):
            y_series = pd.Series(y)
        else:
            y_series = y.copy()
        
        if dates is not None:
            X_df.index = dates
            y_series.index = dates
        
        # Initialize results
        results = SensitivityResults(analysis_type='comprehensive')
        
        # Period sensitivity analysis
        if self.config.period_analysis and dates is not None:
            self.logger.info("Running period sensitivity analysis")
            results.period_results = self._analyze_period_sensitivity(
                model, X_df, y_series, model_name, dates
            )
        
        # Sample size sensitivity analysis
        if self.config.sample_size_analysis:
            self.logger.info("Running sample size sensitivity analysis")
            results.sample_size_results = self._analyze_sample_size_sensitivity(
                model, X_df, y_series, model_name
            )
        
        # Feature sensitivity analysis
        if self.config.feature_analysis:
            self.logger.info("Running feature sensitivity analysis")
            results.feature_results = self._analyze_feature_sensitivity(
                model, X_df, y_series, model_name
            )
        
        # Hyperparameter sensitivity analysis
        if self.config.hyperparameter_analysis:
            self.logger.info("Running hyperparameter sensitivity analysis")
            results.hyperparameter_results = self._analyze_hyperparameter_sensitivity(
                model, X_df, y_series, model_name
            )
        
        # Outlier sensitivity analysis
        if self.config.outlier_analysis:
            self.logger.info("Running outlier sensitivity analysis")
            results.outlier_results = self._analyze_outlier_sensitivity(
                model, X_df, y_series, model_name
            )
        
        # Specification sensitivity analysis
        if self.config.specification_analysis:
            self.logger.info("Running specification sensitivity analysis")
            results.specification_results = self._analyze_specification_sensitivity(
                model, X_df, y_series, model_name
            )
        
        # Calculate overall sensitivity metrics
        results.sensitivity_score, results.most_sensitive_factor, results.least_sensitive_factor = \
            self._calculate_overall_sensitivity(results)
        
        # Store results
        self.sensitivity_results[model_name] = results
        
        self.logger.info(f"Comprehensive sensitivity analysis completed for {model_name}")
        return results
    
    def _analyze_period_sensitivity(self,
                                  model: Any,
                                  X: pd.DataFrame,
                                  y: pd.Series,
                                  model_name: str,
                                  dates: pd.DatetimeIndex) -> Dict[str, Dict[str, Any]]:
        """Analyze sensitivity to different time periods."""
        period_results = {}
        
        # Define periods
        crisis_start = pd.to_datetime(self.config.crisis_start)
        crisis_end = pd.to_datetime(self.config.crisis_end)
        
        periods = {
            'full_sample': (dates.min(), dates.max()),
            'pre_crisis': (dates.min(), crisis_start),
            'crisis': (crisis_start, crisis_end),
            'post_crisis': (crisis_end, dates.max())
        }
        
        baseline_performance = None
        
        for period_name, (start_date, end_date) in periods.items():
            try:
                # Filter data for period
                period_mask = (dates >= start_date) & (dates <= end_date)
                
                if period_mask.sum() < 10:  # Skip periods with too few observations
                    period_results[period_name] = {'error': 'Insufficient observations'}
                    continue
                
                X_period = X[period_mask]
                y_period = y[period_mask]
                
                # Fit model on period data
                period_model = self._fit_sensitivity_model(model, X_period, y_period)
                
                # Calculate performance metrics
                performance = self._calculate_sensitivity_performance(period_model, X_period, y_period)
                
                # Store baseline (full sample) for comparison
                if period_name == 'full_sample':
                    baseline_performance = performance
                
                period_results[period_name] = {
                    'performance': performance,
                    'n_observations': len(X_period),
                    'period_start': start_date.isoformat(),
                    'period_end': end_date.isoformat()
                }
                
            except Exception as e:
                period_results[period_name] = {'error': str(e)}
        
        # Calculate sensitivity metrics
        if baseline_performance:
            for period_name, period_data in period_results.items():
                if 'performance' in period_data and period_name != 'full_sample':
                    period_perf = period_data['performance']
                    sensitivity_metrics = {}
                    
                    for metric in ['r2', 'mse', 'mae']:
                        if metric in baseline_performance and metric in period_perf:
                            baseline_val = baseline_performance[metric]
                            period_val = period_perf[metric]
                            
                            if baseline_val != 0:
                                pct_change = (period_val - baseline_val) / abs(baseline_val) * 100
                                sensitivity_metrics[f'{metric}_pct_change'] = pct_change
                    
                    period_data['sensitivity_metrics'] = sensitivity_metrics
        
        return period_results
    
    def _analyze_sample_size_sensitivity(self,
                                       model: Any,
                                       X: pd.DataFrame,
                                       y: pd.Series,
                                       model_name: str) -> Dict[str, Dict[str, Any]]:
        """Analyze sensitivity to different sample sizes."""
        sample_size_results = {}
        
        n_total = len(X)
        baseline_performance = None
        
        for fraction in self.config.sample_size_fractions:
            try:
                n_sample = int(n_total * fraction)
                
                if n_sample < 10:  # Skip very small samples
                    sample_size_results[f'fraction_{fraction}'] = {'error': 'Sample too small'}
                    continue
                
                # Random sample (with fixed seed for reproducibility)
                np.random.seed(42)
                sample_indices = np.random.choice(n_total, size=n_sample, replace=False)
                
                X_sample = X.iloc[sample_indices]
                y_sample = y.iloc[sample_indices]
                
                # Fit model on sample
                sample_model = self._fit_sensitivity_model(model, X_sample, y_sample)
                
                # Calculate performance
                performance = self._calculate_sensitivity_performance(sample_model, X_sample, y_sample)
                
                # Store baseline (full sample) for comparison
                if fraction == 1.0:
                    baseline_performance = performance
                
                sample_size_results[f'fraction_{fraction}'] = {
                    'performance': performance,
                    'n_observations': n_sample,
                    'fraction': fraction
                }
                
            except Exception as e:
                sample_size_results[f'fraction_{fraction}'] = {'error': str(e)}
        
        # Calculate sensitivity metrics
        if baseline_performance:
            for fraction_key, sample_data in sample_size_results.items():
                if 'performance' in sample_data and fraction_key != 'fraction_1.0':
                    sample_perf = sample_data['performance']
                    sensitivity_metrics = {}
                    
                    for metric in ['r2', 'mse', 'mae']:
                        if metric in baseline_performance and metric in sample_perf:
                            baseline_val = baseline_performance[metric]
                            sample_val = sample_perf[metric]
                            
                            if baseline_val != 0:
                                pct_change = (sample_val - baseline_val) / abs(baseline_val) * 100
                                sensitivity_metrics[f'{metric}_pct_change'] = pct_change
                    
                    sample_data['sensitivity_metrics'] = sensitivity_metrics
        
        return sample_size_results
    
    def _analyze_feature_sensitivity(self,
                                   model: Any,
                                   X: pd.DataFrame,
                                   y: pd.Series,
                                   model_name: str) -> Dict[str, Dict[str, Any]]:
        """Analyze sensitivity to different feature subsets."""
        feature_results = {}
        
        n_features = X.shape[1]
        feature_names = X.columns.tolist()
        
        # Define feature subsets
        feature_subsets = {
            'all_features': feature_names,
            'core_features': feature_names[:max(1, n_features//2)],  # First half
            'extended_features': feature_names[max(1, n_features//2):],  # Second half
        }
        
        # Add single feature analysis
        for i, feature in enumerate(feature_names[:min(5, n_features)]):  # Limit to first 5 features
            feature_subsets[f'single_{feature}'] = [feature]
        
        baseline_performance = None
        
        for subset_name, features in feature_subsets.items():
            try:
                if not features:
                    continue
                
                X_subset = X[features]
                
                # Fit model on feature subset
                subset_model = self._fit_sensitivity_model(model, X_subset, y)
                
                # Calculate performance
                performance = self._calculate_sensitivity_performance(subset_model, X_subset, y)
                
                # Store baseline (all features) for comparison
                if subset_name == 'all_features':
                    baseline_performance = performance
                
                feature_results[subset_name] = {
                    'performance': performance,
                    'features': features,
                    'n_features': len(features)
                }
                
            except Exception as e:
                feature_results[subset_name] = {'error': str(e)}
        
        # Calculate sensitivity metrics
        if baseline_performance:
            for subset_name, subset_data in feature_results.items():
                if 'performance' in subset_data and subset_name != 'all_features':
                    subset_perf = subset_data['performance']
                    sensitivity_metrics = {}
                    
                    for metric in ['r2', 'mse', 'mae']:
                        if metric in baseline_performance and metric in subset_perf:
                            baseline_val = baseline_performance[metric]
                            subset_val = subset_perf[metric]
                            
                            if baseline_val != 0:
                                pct_change = (subset_val - baseline_val) / abs(baseline_val) * 100
                                sensitivity_metrics[f'{metric}_pct_change'] = pct_change
                    
                    subset_data['sensitivity_metrics'] = sensitivity_metrics
        
        return feature_results
    
    def _analyze_hyperparameter_sensitivity(self,
                                          model: Any,
                                          X: pd.DataFrame,
                                          y: pd.Series,
                                          model_name: str) -> Dict[str, Dict[str, Any]]:
        """Analyze sensitivity to hyperparameter changes."""
        hyperparameter_results = {}
        
        # This is a simplified implementation
        # In practice, would need model-specific hyperparameter ranges
        
        try:
            # Get baseline performance with default parameters
            baseline_model = self._fit_sensitivity_model(model, X, y)
            baseline_performance = self._calculate_sensitivity_performance(baseline_model, X, y)
            
            hyperparameter_results['baseline'] = {
                'performance': baseline_performance,
                'parameters': 'default'
            }
            
            # For ML models, try different hyperparameters
            if hasattr(model, 'get_params'):
                params = model.get_params()
                
                # Example: vary regularization parameter if available
                if 'alpha' in params:
                    for alpha in [0.01, 0.1, 1.0, 10.0]:
                        try:
                            modified_model = model.__class__(**{**params, 'alpha': alpha})
                            modified_model = self._fit_sensitivity_model(modified_model, X, y)
                            performance = self._calculate_sensitivity_performance(modified_model, X, y)
                            
                            hyperparameter_results[f'alpha_{alpha}'] = {
                                'performance': performance,
                                'parameters': {'alpha': alpha}
                            }
                        except Exception as e:
                            hyperparameter_results[f'alpha_{alpha}'] = {'error': str(e)}
        
        except Exception as e:
            hyperparameter_results['error'] = str(e)
        
        return hyperparameter_results
    
    def _analyze_outlier_sensitivity(self,
                                   model: Any,
                                   X: pd.DataFrame,
                                   y: pd.Series,
                                   model_name: str) -> Dict[str, Dict[str, Any]]:
        """Analyze sensitivity to outlier removal."""
        outlier_results = {}
        
        # Baseline: no outlier removal
        try:
            baseline_model = self._fit_sensitivity_model(model, X, y)
            baseline_performance = self._calculate_sensitivity_performance(baseline_model, X, y)
            
            outlier_results['no_removal'] = {
                'performance': baseline_performance,
                'n_observations': len(X),
                'outliers_removed': 0
            }
        except Exception as e:
            outlier_results['no_removal'] = {'error': str(e)}
        
        # Try different outlier removal methods
        for method in self.config.outlier_removal_methods:
            if method == 'none':
                continue
            
            for threshold in self.config.outlier_thresholds:
                try:
                    # Detect outliers
                    outlier_mask = self._detect_outliers_for_sensitivity(y, method, threshold)
                    
                    if outlier_mask.sum() == 0:  # No outliers detected
                        continue
                    
                    # Remove outliers
                    clean_mask = ~outlier_mask
                    X_clean = X[clean_mask]
                    y_clean = y[clean_mask]
                    
                    if len(X_clean) < 10:  # Skip if too few observations remain
                        continue
                    
                    # Fit model on clean data
                    clean_model = self._fit_sensitivity_model(model, X_clean, y_clean)
                    performance = self._calculate_sensitivity_performance(clean_model, X_clean, y_clean)
                    
                    outlier_results[f'{method}_threshold_{threshold}'] = {
                        'performance': performance,
                        'n_observations': len(X_clean),
                        'outliers_removed': outlier_mask.sum(),
                        'outlier_percentage': outlier_mask.sum() / len(X) * 100
                    }
                    
                except Exception as e:
                    outlier_results[f'{method}_threshold_{threshold}'] = {'error': str(e)}
        
        return outlier_results
    
    def _analyze_specification_sensitivity(self,
                                         model: Any,
                                         X: pd.DataFrame,
                                         y: pd.Series,
                                         model_name: str) -> Dict[str, Dict[str, Any]]:
        """Analyze sensitivity to different model specifications."""
        specification_results = {}
        
        # Baseline: original specification
        try:
            baseline_model = self._fit_sensitivity_model(model, X, y)
            baseline_performance = self._calculate_sensitivity_performance(baseline_model, X, y)
            
            specification_results['original'] = {
                'performance': baseline_performance,
                'specification': 'original'
            }
        except Exception as e:
            specification_results['original'] = {'error': str(e)}
        
        # Try alternative specifications
        for spec in self.config.alternative_specifications:
            try:
                if spec == 'linear':
                    # Use linear regression as alternative
                    from sklearn.linear_model import LinearRegression
                    alt_model = LinearRegression()
                    alt_model.fit(X, y)
                    y_pred = alt_model.predict(X)
                    
                elif spec == 'polynomial':
                    # Add polynomial features
                    from sklearn.preprocessing import PolynomialFeatures
                    from sklearn.linear_model import LinearRegression
                    from sklearn.pipeline import Pipeline
                    
                    poly_model = Pipeline([
                        ('poly', PolynomialFeatures(degree=2, include_bias=False)),
                        ('linear', LinearRegression())
                    ])
                    poly_model.fit(X, y)
                    y_pred = poly_model.predict(X)
                    
                elif spec == 'log_transform':
                    # Log transform target (if positive)
                    if (y > 0).all():
                        y_log = np.log(y)
                        log_model = self._fit_sensitivity_model(model, X, y_log)
                        y_pred_log = self._predict_sensitivity_model(log_model, X, model)
                        y_pred = np.exp(y_pred_log)  # Transform back
                    else:
                        continue
                
                else:
                    continue  # Skip unknown specifications
                
                # Calculate performance
                performance = {
                    'r2': r2_score(y, y_pred),
                    'mse': mean_squared_error(y, y_pred),
                    'mae': mean_absolute_error(y, y_pred)
                }
                
                specification_results[spec] = {
                    'performance': performance,
                    'specification': spec
                }
                
            except Exception as e:
                specification_results[spec] = {'error': str(e)}
        
        return specification_results
    
    def _fit_sensitivity_model(self, original_model: Any, X: pd.DataFrame, y: pd.Series) -> Any:
        """Fit model for sensitivity analysis."""
        # Convert to numpy arrays
        X_array = X.values
        y_array = y.values
        feature_names = X.columns.tolist()
        
        # Create a copy of the original model
        model_class = original_model.__class__
        
        try:
            if hasattr(original_model, 'get_params'):
                params = original_model.get_params()
                sensitivity_model = model_class(**params)
            else:
                sensitivity_model = model_class()
            
            # Fit based on model type
            if 'Hansen' in model_class.__name__:
                threshold_var = X_array[:, 0] if X_array.shape[1] > 0 else y_array
                sensitivity_model.fit(y_array, X_array, threshold_var)
            elif 'LocalProjections' in model_class.__name__:
                shock = X_array[:, 0] if X_array.shape[1] > 0 else np.zeros(len(y_array))
                controls = pd.DataFrame(X_array[:, 1:]) if X_array.shape[1] > 1 else None
                sensitivity_model.fit(pd.Series(y_array), pd.Series(shock), controls)
            elif hasattr(sensitivity_model, 'fit'):
                if 'RandomForest' in model_class.__name__ or 'GradientBoosting' in model_class.__name__:
                    sensitivity_model.fit(X_array, y_array, feature_names=feature_names, hyperparameter_tuning=False)
                else:
                    sensitivity_model.fit(X_array, y_array)
            else:
                raise ValueError(f"Don't know how to fit model of type {model_class.__name__}")
            
            return sensitivity_model
            
        except Exception as e:
            self.logger.warning(f"Error fitting sensitivity model: {str(e)}")
            raise
    
    def _predict_sensitivity_model(self, model: Any, X: pd.DataFrame, original_model: Any) -> np.ndarray:
        """Make predictions for sensitivity analysis."""
        X_array = X.values
        
        try:
            if hasattr(model, 'predict'):
                return model.predict(X_array)
            else:
                # Fallback
                return np.full(len(X_array), np.mean(X_array))
        except Exception as e:
            self.logger.warning(f"Error making sensitivity predictions: {str(e)}")
            return np.zeros(len(X_array))
    
    def _calculate_sensitivity_performance(self, model: Any, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        """Calculate performance metrics for sensitivity analysis."""
        try:
            y_pred = self._predict_sensitivity_model(model, X, model)
            
            performance = {
                'r2': r2_score(y, y_pred),
                'mse': mean_squared_error(y, y_pred),
                'mae': mean_absolute_error(y, y_pred)
            }
            
            # Add MAPE if no zero values
            if (y != 0).all():
                mape = np.mean(np.abs((y - y_pred) / y)) * 100
                performance['mape'] = mape
            
            return performance
            
        except Exception as e:
            self.logger.warning(f"Error calculating sensitivity performance: {str(e)}")
            return {'r2': np.nan, 'mse': np.nan, 'mae': np.nan}
    
    def _detect_outliers_for_sensitivity(self, y: pd.Series, method: str, threshold: float) -> np.ndarray:
        """Detect outliers for sensitivity analysis."""
        if method == 'z_score':
            z_scores = np.abs(stats.zscore(y))
            return z_scores > threshold
        
        elif method == 'iqr':
            Q1 = y.quantile(0.25)
            Q3 = y.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR
            return (y < lower_bound) | (y > upper_bound)
        
        elif method == 'isolation_forest':
            try:
                from sklearn.ensemble import IsolationForest
                iso_forest = IsolationForest(contamination=0.1, random_state=42)
                outlier_labels = iso_forest.fit_predict(y.values.reshape(-1, 1))
                return outlier_labels == -1
            except:
                return np.zeros(len(y), dtype=bool)
        
        else:
            return np.zeros(len(y), dtype=bool)
    
    def _calculate_overall_sensitivity(self, results: SensitivityResults) -> Tuple[float, str, str]:
        """Calculate overall sensitivity metrics."""
        sensitivity_scores = {}
        
        # Calculate sensitivity score for each analysis type
        for analysis_type in ['period', 'sample_size', 'feature', 'hyperparameter', 'outlier', 'specification']:
            analysis_results = getattr(results, f'{analysis_type}_results')
            
            if not analysis_results:
                continue
            
            # Calculate coefficient of variation of R² scores across conditions
            r2_scores = []
            for condition_name, condition_data in analysis_results.items():
                if 'performance' in condition_data and 'r2' in condition_data['performance']:
                    r2_score = condition_data['performance']['r2']
                    if not np.isnan(r2_score):
                        r2_scores.append(r2_score)
            
            if len(r2_scores) > 1:
                cv = np.std(r2_scores) / np.mean(r2_scores) if np.mean(r2_scores) != 0 else 0
                sensitivity_scores[analysis_type] = cv
        
        if not sensitivity_scores:
            return 0.0, "unknown", "unknown"
        
        # Overall sensitivity score (average CV)
        overall_score = np.mean(list(sensitivity_scores.values()))
        
        # Most and least sensitive factors
        most_sensitive = max(sensitivity_scores.keys(), key=lambda k: sensitivity_scores[k])
        least_sensitive = min(sensitivity_scores.keys(), key=lambda k: sensitivity_scores[k])
        
        return overall_score, most_sensitive, least_sensitive
    
    def create_sensitivity_summary_table(self, model_name: str) -> pd.DataFrame:
        """
        Create summary table of sensitivity analysis results.
        
        Args:
            model_name: Name of model to summarize
            
        Returns:
            DataFrame with sensitivity summary
        """
        if model_name not in self.sensitivity_results:
            raise ValueError(f"No sensitivity results found for model: {model_name}")
        
        results = self.sensitivity_results[model_name]
        summary_data = []
        
        # Process each analysis type
        analysis_types = [
            ('period', results.period_results),
            ('sample_size', results.sample_size_results),
            ('feature', results.feature_results),
            ('hyperparameter', results.hyperparameter_results),
            ('outlier', results.outlier_results),
            ('specification', results.specification_results)
        ]
        
        for analysis_type, analysis_results in analysis_types:
            if not analysis_results:
                continue
            
            for condition_name, condition_data in analysis_results.items():
                if 'performance' not in condition_data:
                    continue
                
                performance = condition_data['performance']
                
                row = {
                    'analysis_type': analysis_type,
                    'condition': condition_name,
                    'r2': performance.get('r2', np.nan),
                    'mse': performance.get('mse', np.nan),
                    'mae': performance.get('mae', np.nan),
                    'n_observations': condition_data.get('n_observations', np.nan)
                }
                
                # Add sensitivity metrics if available
                if 'sensitivity_metrics' in condition_data:
                    sens_metrics = condition_data['sensitivity_metrics']
                    for metric, value in sens_metrics.items():
                        row[f'sensitivity_{metric}'] = value
                
                summary_data.append(row)
        
        return pd.DataFrame(summary_data)
    
    def plot_sensitivity_analysis(self, model_name: str, figsize: Tuple[int, int] = (15, 12)) -> plt.Figure:
        """
        Create comprehensive sensitivity analysis visualization.
        
        Args:
            model_name: Name of model to plot
            figsize: Figure size
            
        Returns:
            Matplotlib figure
        """
        if model_name not in self.sensitivity_results:
            raise ValueError(f"No sensitivity results found for model: {model_name}")
        
        results = self.sensitivity_results[model_name]
        
        fig, axes = plt.subplots(3, 2, figsize=figsize)
        
        # Period sensitivity
        if results.period_results:
            periods = []
            r2_scores = []
            
            for period_name, period_data in results.period_results.items():
                if 'performance' in period_data and 'r2' in period_data['performance']:
                    periods.append(period_name)
                    r2_scores.append(period_data['performance']['r2'])
            
            if periods and r2_scores:
                axes[0, 0].bar(periods, r2_scores)
                axes[0, 0].set_title('Period Sensitivity (R²)')
                axes[0, 0].set_ylabel('R² Score')
                axes[0, 0].tick_params(axis='x', rotation=45)
        
        # Sample size sensitivity
        if results.sample_size_results:
            fractions = []
            r2_scores = []
            
            for fraction_key, fraction_data in results.sample_size_results.items():
                if 'performance' in fraction_data and 'r2' in fraction_data['performance']:
                    fraction = fraction_data.get('fraction', 0)
                    fractions.append(fraction)
                    r2_scores.append(fraction_data['performance']['r2'])
            
            if fractions and r2_scores:
                axes[0, 1].plot(fractions, r2_scores, 'o-')
                axes[0, 1].set_title('Sample Size Sensitivity (R²)')
                axes[0, 1].set_xlabel('Sample Size Fraction')
                axes[0, 1].set_ylabel('R² Score')
        
        # Feature sensitivity
        if results.feature_results:
            features = []
            r2_scores = []
            
            for feature_key, feature_data in results.feature_results.items():
                if 'performance' in feature_data and 'r2' in feature_data['performance']:
                    features.append(feature_key)
                    r2_scores.append(feature_data['performance']['r2'])
            
            if features and r2_scores:
                axes[1, 0].bar(features, r2_scores)
                axes[1, 0].set_title('Feature Sensitivity (R²)')
                axes[1, 0].set_ylabel('R² Score')
                axes[1, 0].tick_params(axis='x', rotation=45)
        
        # Outlier sensitivity
        if results.outlier_results:
            methods = []
            r2_scores = []
            
            for method_key, method_data in results.outlier_results.items():
                if 'performance' in method_data and 'r2' in method_data['performance']:
                    methods.append(method_key)
                    r2_scores.append(method_data['performance']['r2'])
            
            if methods and r2_scores:
                axes[1, 1].bar(methods, r2_scores)
                axes[1, 1].set_title('Outlier Sensitivity (R²)')
                axes[1, 1].set_ylabel('R² Score')
                axes[1, 1].tick_params(axis='x', rotation=45)
        
        # Overall sensitivity summary
        summary_text = f"Model: {model_name}\n"
        summary_text += f"Overall Sensitivity Score: {results.sensitivity_score:.3f}\n"
        summary_text += f"Most Sensitive Factor: {results.most_sensitive_factor}\n"
        summary_text += f"Least Sensitive Factor: {results.least_sensitive_factor}\n"
        
        axes[2, 0].text(0.1, 0.5, summary_text, transform=axes[2, 0].transAxes,
                       fontsize=12, verticalalignment='center')
        axes[2, 0].set_xticks([])
        axes[2, 0].set_yticks([])
        axes[2, 0].set_title('Sensitivity Summary')
        
        # Sensitivity score by analysis type
        analysis_types = []
        sensitivity_scores = []
        
        for analysis_type in ['period', 'sample_size', 'feature', 'outlier']:
            analysis_results = getattr(results, f'{analysis_type}_results')
            if analysis_results:
                # Calculate CV for this analysis type
                r2_scores = []
                for condition_data in analysis_results.values():
                    if 'performance' in condition_data and 'r2' in condition_data['performance']:
                        r2_score = condition_data['performance']['r2']
                        if not np.isnan(r2_score):
                            r2_scores.append(r2_score)
                
                if len(r2_scores) > 1:
                    cv = np.std(r2_scores) / np.mean(r2_scores) if np.mean(r2_scores) != 0 else 0
                    analysis_types.append(analysis_type)
                    sensitivity_scores.append(cv)
        
        if analysis_types and sensitivity_scores:
            axes[2, 1].bar(analysis_types, sensitivity_scores)
            axes[2, 1].set_title('Sensitivity by Analysis Type')
            axes[2, 1].set_ylabel('Coefficient of Variation')
            axes[2, 1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        return fig


class RobustnessComparisonFramework:
    """
    Framework for creating robustness tables comparing results across models.
    
    Implements Requirements 5.3, 5.6 for robustness comparison and reporting.
    """
    
    def __init__(self):
        """Initialize robustness comparison framework."""
        self.logger = logging.getLogger(__name__)
        
        # Storage for comparison results
        self.comparison_results: Dict[str, Any] = {}
        
        self.logger.info("RobustnessComparisonFramework initialized")
    
    def create_robustness_comparison_table(self,
                                         cv_results: Dict[str, CrossValidationResults],
                                         sensitivity_results: Dict[str, SensitivityResults],
                                         diagnostic_results: Optional[Dict[str, DiagnosticResults]] = None) -> pd.DataFrame:
        """
        Create comprehensive robustness comparison table.
        
        Args:
            cv_results: Cross-validation results for multiple models
            sensitivity_results: Sensitivity analysis results for multiple models
            diagnostic_results: Optional diagnostic results for multiple models
            
        Returns:
            DataFrame with robustness comparison
        """
        self.logger.info("Creating robustness comparison table")
        
        comparison_data = []
        
        # Get all model names
        all_models = set(cv_results.keys()) | set(sensitivity_results.keys())
        if diagnostic_results:
            all_models |= set(diagnostic_results.keys())
        
        for model_name in all_models:
            row = {'model_name': model_name}
            
            # Cross-validation metrics
            if model_name in cv_results:
                cv_result = cv_results[model_name]
                row.update({
                    'cv_r2_mean': cv_result.cv_mean.get('r2', np.nan),
                    'cv_r2_std': cv_result.cv_std.get('r2', np.nan),
                    'cv_mse_mean': cv_result.cv_mean.get('mse', np.nan),
                    'cv_mse_std': cv_result.cv_std.get('mse', np.nan),
                    'cv_prediction_stability': cv_result.prediction_stability,
                    'cv_n_splits': cv_result.n_splits
                })
                
                # Temporal consistency
                for metric, value in cv_result.temporal_consistency.items():
                    row[f'cv_{metric}'] = value
            
            # Sensitivity metrics
            if model_name in sensitivity_results:
                sens_result = sensitivity_results[model_name]
                row.update({
                    'sensitivity_score': sens_result.sensitivity_score,
                    'most_sensitive_factor': sens_result.most_sensitive_factor,
                    'least_sensitive_factor': sens_result.least_sensitive_factor
                })
                
                # Period sensitivity
                if sens_result.period_results:
                    period_r2_scores = []
                    for period_data in sens_result.period_results.values():
                        if 'performance' in period_data and 'r2' in period_data['performance']:
                            r2_score = period_data['performance']['r2']
                            if not np.isnan(r2_score):
                                period_r2_scores.append(r2_score)
                    
                    if len(period_r2_scores) > 1:
                        row['period_sensitivity_cv'] = np.std(period_r2_scores) / np.mean(period_r2_scores)
                
                # Sample size sensitivity
                if sens_result.sample_size_results:
                    sample_r2_scores = []
                    for sample_data in sens_result.sample_size_results.values():
                        if 'performance' in sample_data and 'r2' in sample_data['performance']:
                            r2_score = sample_data['performance']['r2']
                            if not np.isnan(r2_score):
                                sample_r2_scores.append(r2_score)
                    
                    if len(sample_r2_scores) > 1:
                        row['sample_size_sensitivity_cv'] = np.std(sample_r2_scores) / np.mean(sample_r2_scores)
            
            # Diagnostic metrics
            if diagnostic_results and model_name in diagnostic_results:
                diag_result = diagnostic_results[model_name]
                row.update({
                    'diagnostic_score': diag_result.diagnostic_score,
                    'normality_passed': diag_result.normality_passed,
                    'homoskedasticity_passed': diag_result.homoskedasticity_passed,
                    'no_autocorrelation_passed': diag_result.no_autocorrelation_passed,
                    'linearity_passed': diag_result.linearity_passed,
                    'stationarity_passed': diag_result.stationarity_passed,
                    'n_outliers': len(diag_result.outliers_detected),
                    'n_influential': len(diag_result.influential_observations)
                })
            
            comparison_data.append(row)
        
        comparison_df = pd.DataFrame(comparison_data)
        
        # Calculate overall robustness score
        if not comparison_df.empty:
            robustness_components = []
            
            # CV stability (higher is better, so invert)
            if 'cv_prediction_stability' in comparison_df.columns:
                cv_stability = comparison_df['cv_prediction_stability'].fillna(1.0)
                robustness_components.append(1 / (1 + cv_stability))  # Invert so lower stability = lower robustness
            
            # Sensitivity score (lower is better, so invert)
            if 'sensitivity_score' in comparison_df.columns:
                sens_score = comparison_df['sensitivity_score'].fillna(1.0)
                robustness_components.append(1 / (1 + sens_score))
            
            # Diagnostic score (higher is better)
            if 'diagnostic_score' in comparison_df.columns:
                diag_score = comparison_df['diagnostic_score'].fillna(0.0)
                robustness_components.append(diag_score)
            
            if robustness_components:
                comparison_df['overall_robustness_score'] = np.mean(robustness_components, axis=0)
        
        # Store results
        self.comparison_results['robustness_table'] = comparison_df
        
        self.logger.info("Robustness comparison table created")
        return comparison_df
    
    def rank_models_by_robustness(self, comparison_df: pd.DataFrame) -> pd.DataFrame:
        """
        Rank models by overall robustness.
        
        Args:
            comparison_df: Robustness comparison DataFrame
            
        Returns:
            DataFrame with model rankings
        """
        if 'overall_robustness_score' not in comparison_df.columns:
            raise ValueError("Overall robustness score not available. Run create_robustness_comparison_table first.")
        
        # Rank by overall robustness score (higher is better)
        ranked_df = comparison_df.sort_values('overall_robustness_score', ascending=False).copy()
        ranked_df['robustness_rank'] = range(1, len(ranked_df) + 1)
        
        return ranked_df[['model_name', 'overall_robustness_score', 'robustness_rank']]
    
    def export_robustness_table_latex(self, comparison_df: pd.DataFrame, filename: str) -> str:
        """
        Export robustness comparison table to LaTeX format.
        
        Args:
            comparison_df: Robustness comparison DataFrame
            filename: Output filename
            
        Returns:
            LaTeX table string
        """
        # Select key columns for LaTeX table
        key_columns = ['model_name']
        
        if 'cv_r2_mean' in comparison_df.columns:
            key_columns.extend(['cv_r2_mean', 'cv_r2_std'])
        
        if 'sensitivity_score' in comparison_df.columns:
            key_columns.append('sensitivity_score')
        
        if 'diagnostic_score' in comparison_df.columns:
            key_columns.append('diagnostic_score')
        
        if 'overall_robustness_score' in comparison_df.columns:
            key_columns.append('overall_robustness_score')
        
        # Create LaTeX table
        latex_df = comparison_df[key_columns].copy()
        
        # Format numeric columns
        for col in latex_df.columns:
            if col != 'model_name' and latex_df[col].dtype in ['float64', 'float32']:
                latex_df[col] = latex_df[col].round(3)
        
        # Generate LaTeX
        latex_str = latex_df.to_latex(
            index=False,
            escape=False,
            column_format='l' + 'c' * (len(key_columns) - 1),
            caption='Model Robustness Comparison',
            label='tab:robustness_comparison'
        )
        
        # Save to file
        with open(filename, 'w') as f:
            f.write(latex_str)
        
        self.logger.info(f"Robustness table exported to {filename}")
        return latex_str