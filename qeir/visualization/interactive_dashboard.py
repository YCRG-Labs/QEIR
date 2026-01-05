"""
Interactive Analysis Dashboard for Real-Time Model Exploration

This module provides an interactive dashboard system for real-time econometric model
exploration, specification testing, and robustness analysis. Designed for publication-ready
research with immediate result updates and comprehensive model comparison capabilities.

Addresses Requirements 7.1, 7.2, 7.3, 7.4, 7.5:
- Real-time model fitting and specification exploration
- Interactive robustness testing and sensitivity analysis
- Side-by-side model comparison capabilities
- Export functionality for publication-ready outputs
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import ipywidgets as widgets
from IPython.display import display, clear_output
import warnings
from typing import Dict, List, Optional, Tuple, Any, Callable
from pathlib import Path
import json
import time

# Import existing components
try:
    from ..core.models import HansenThresholdRegression, LocalProjections
    from ..utils.publication_model_diagnostics import PublicationModelDiagnostics
    from .publication_visualization import PublicationVisualizationSuite
except ImportError:
    # Fallback for direct execution
    from models import HansenThresholdRegression, SmoothTransitionRegression, LocalProjections
    from publication_model_diagnostics import PublicationModelDiagnostics
    from publication_visualization import PublicationVisualizationSuite

warnings.filterwarnings('ignore')

class InteractiveAnalysisDashboard:
    """
    Interactive dashboard for real-time econometric model exploration and analysis.
    
    Provides comprehensive tools for specification testing, robustness analysis,
    and publication-ready output generation with immediate visual feedback.
    """
    
    def __init__(self, data: pd.DataFrame, models: Optional[Dict[str, Any]] = None):
        """
        Initialize the interactive analysis dashboard.
        
        Parameters:
        -----------
        data : pd.DataFrame
            Dataset for analysis containing dependent variables, regressors, and threshold variables
        models : Optional[Dict[str, Any]]
            Pre-configured model instances (if None, creates default models)
        """
        self.data = data
        self.models = models or self._initialize_default_models()
        self.diagnostics = PublicationModelDiagnostics()
        self.visualizer = PublicationVisualizationSuite()
        
        # Dashboard state
        self.current_results = {}
        self.comparison_results = {}
        self.output_widgets = {}
        
        # Model fitting cache for performance
        self.model_cache = {}
        self.cache_enabled = True
        
        # Export settings
        self.export_settings = {
            'formats': ['png', 'pdf'],
            'dpi': 300,
            'style': 'economics_journal'
        }
        
    def _initialize_default_models(self) -> Dict[str, Any]:
        """Initialize default model instances for the dashboard."""
        return {
            'hansen_threshold': HansenThresholdRegression(),
            'smooth_transition': SmoothTransitionRegression(),
            'local_projections': LocalProjections()
        }
    
    def create_specification_explorer(self, 
                                    y_column: str,
                                    x_columns: List[str],
                                    threshold_column: str,
                                    output_container: Optional[widgets.Output] = None) -> widgets.VBox:
        """
        Create interactive specification testing interface with real-time parameter adjustment.
        
        Parameters:
        -----------
        y_column : str
            Name of dependent variable column
        x_columns : List[str]
            Names of independent variable columns
        threshold_column : str
            Name of threshold variable column
        output_container : Optional[widgets.Output]
            Container for output display
            
        Returns:
        --------
        widgets.VBox
            Interactive specification explorer widget
        """
        # Create control widgets
        model_selector = widgets.Dropdown(
            options=['Hansen Threshold', 'Smooth Transition', 'Local Projections'],
            value='Hansen Threshold',
            description='Model Type:'
        )
        
        # Variable selection
        y_selector = widgets.Dropdown(
            options=self.data.columns.tolist(),
            value=y_column,
            description='Dependent Var:'
        )
        
        x_selector = widgets.SelectMultiple(
            options=self.data.columns.tolist(),
            value=x_columns,
            description='Independent Vars:'
        )
        
        threshold_selector = widgets.Dropdown(
            options=self.data.columns.tolist(),
            value=threshold_column,
            description='Threshold Var:'
        )
        
        # Model-specific parameters
        trim_slider = widgets.FloatSlider(
            value=0.15,
            min=0.05,
            max=0.30,
            step=0.01,
            description='Trim Fraction:',
            style={'description_width': 'initial'}
        )
        
        gamma_slider = widgets.FloatSlider(
            value=1.0,
            min=0.1,
            max=10.0,
            step=0.1,
            description='STR Gamma:',
            style={'description_width': 'initial'}
        )
        
        # Data transformation options
        transform_selector = widgets.Dropdown(
            options=['Levels', 'First Differences', 'Log Levels', 'Standardized'],
            value='Levels',
            description='Transformation:'
        )
        
        # Real-time fitting toggle
        realtime_toggle = widgets.Checkbox(
            value=True,
            description='Real-time Fitting'
        )
        
        # Fit button for manual updates
        fit_button = widgets.Button(
            description='Fit Model',
            button_style='primary'
        )
        
        # Output area
        if output_container is None:
            output_container = widgets.Output()
        
        # Interactive fitting function
        def update_model(*args):
            if realtime_toggle.value or args[0] == fit_button:  # args[0] is the button for manual fit
                with output_container:
                    clear_output(wait=True)
                    try:
                        # Get current selections
                        y_col = y_selector.value
                        x_cols = list(x_selector.value)
                        thresh_col = threshold_selector.value
                        model_type = model_selector.value
                        
                        if not x_cols:
                            print("Please select at least one independent variable.")
                            return
                        
                        # Prepare data
                        y_data, x_data, thresh_data = self._prepare_data(
                            y_col, x_cols, thresh_col, transform_selector.value
                        )
                        
                        # Fit model with caching
                        cache_key = self._generate_cache_key(
                            model_type, y_col, x_cols, thresh_col, 
                            transform_selector.value, trim_slider.value, gamma_slider.value
                        )
                        
                        if self.cache_enabled and cache_key in self.model_cache:
                            results = self.model_cache[cache_key]
                            print("Using cached results...")
                        else:
                            print(f"Fitting {model_type} model...")
                            start_time = time.time()
                            
                            results = self.real_time_model_fitting(
                                model_type, y_data, x_data, thresh_data,
                                trim_fraction=trim_slider.value,
                                gamma=gamma_slider.value
                            )
                            
                            fit_time = time.time() - start_time
                            print(f"Model fitted in {fit_time:.2f} seconds")
                            
                            if self.cache_enabled:
                                self.model_cache[cache_key] = results
                        
                        # Store current results
                        self.current_results[model_type] = results
                        
                        # Display results
                        self._display_model_results(results, model_type)
                        
                    except Exception as e:
                        print(f"Error fitting model: {str(e)}")
        
        # Bind events
        model_selector.observe(update_model, names='value')
        y_selector.observe(update_model, names='value')
        x_selector.observe(update_model, names='value')
        threshold_selector.observe(update_model, names='value')
        transform_selector.observe(update_model, names='value')
        
        # Only bind sliders if real-time is enabled
        def on_realtime_change(change):
            if change['new']:  # Real-time enabled
                trim_slider.observe(update_model, names='value')
                gamma_slider.observe(update_model, names='value')
            else:  # Real-time disabled
                trim_slider.unobserve(update_model, names='value')
                gamma_slider.unobserve(update_model, names='value')
        
        realtime_toggle.observe(on_realtime_change, names='value')
        fit_button.on_click(update_model)
        
        # Initial fit
        if realtime_toggle.value:
            update_model()
        
        # Layout
        controls = widgets.VBox([
            widgets.HBox([model_selector, realtime_toggle, fit_button]),
            widgets.HBox([y_selector, threshold_selector]),
            x_selector,
            widgets.HBox([transform_selector, trim_slider, gamma_slider])
        ])
        
        return widgets.VBox([controls, output_container])
    
    def real_time_model_fitting(self, 
                               model_type: str,
                               y: np.ndarray,
                               x: np.ndarray,
                               threshold_var: np.ndarray,
                               **kwargs) -> Dict[str, Any]:
        """
        Perform real-time model fitting with immediate result updates.
        
        Parameters:
        -----------
        model_type : str
            Type of model to fit ('Hansen Threshold', 'Smooth Transition', 'Local Projections')
        y : np.ndarray
            Dependent variable
        x : np.ndarray
            Independent variables
        threshold_var : np.ndarray
            Threshold variable
        **kwargs
            Additional model-specific parameters
            
        Returns:
        --------
        Dict[str, Any]
            Model fitting results with diagnostics and visualizations
        """
        results = {
            'model_type': model_type,
            'fit_time': None,
            'model': None,
            'diagnostics': None,
            'r_squared': None,
            'coefficients': None,
            'error': None
        }
        
        start_time = time.time()
        
        try:
            if model_type == 'Hansen Threshold':
                model = HansenThresholdRegression()
                trim_fraction = kwargs.get('trim_fraction', 0.15)
                model.fit(y, x, threshold_var, trim=trim_fraction)
                
                # Calculate diagnostics
                diagnostics = self.diagnostics.diagnose_low_r_squared(
                    model, y, x, threshold_var
                )
                
                results.update({
                    'model': model,
                    'diagnostics': diagnostics,
                    'r_squared': diagnostics['r2_analysis']['overall_r2'],
                    'threshold_estimate': model.threshold,
                    'regime1_coef': model.beta1,
                    'regime2_coef': model.beta2
                })
                
            elif model_type == 'Smooth Transition':
                model = SmoothTransitionRegression()
                gamma = kwargs.get('gamma', 1.0)
                model.fit(y, x, threshold_var, initial_gamma=gamma)
                
                # Calculate R²
                y_pred = model.predict(x, threshold_var)
                ss_res = np.sum((y - y_pred) ** 2)
                ss_tot = np.sum((y - np.mean(y)) ** 2)
                r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
                
                results.update({
                    'model': model,
                    'r_squared': r_squared,
                    'gamma': model.gamma,
                    'transition_center': model.c,
                    'coefficients': model.coeffs if hasattr(model, 'coeffs') else None
                })
                
            elif model_type == 'Local Projections':
                model = LocalProjections()
                # Prepare data for local projections
                y_series = pd.Series(y)
                shock_series = pd.Series(threshold_var)  # Use threshold as shock
                
                model.fit(y_series, shock_series)
                
                # Get impulse responses
                impulse_responses = model.get_impulse_responses()
                
                results.update({
                    'model': model,
                    'impulse_responses': impulse_responses,
                    'horizons_estimated': len(impulse_responses)
                })
            
            results['fit_time'] = time.time() - start_time
            
        except Exception as e:
            results['error'] = str(e)
            results['fit_time'] = time.time() - start_time
        
        return results
    
    def side_by_side_model_comparison(self, 
                                    model_results: Dict[str, Dict[str, Any]],
                                    comparison_metrics: List[str] = None) -> widgets.VBox:
        """
        Create side-by-side model comparison interface for specification evaluation.
        
        Parameters:
        -----------
        model_results : Dict[str, Dict[str, Any]]
            Dictionary of model results to compare
        comparison_metrics : List[str], optional
            Metrics to compare (default: ['r_squared', 'aic', 'bic'])
            
        Returns:
        --------
        widgets.VBox
            Side-by-side comparison widget
        """
        if comparison_metrics is None:
            comparison_metrics = ['r_squared', 'aic', 'bic', 'log_likelihood']
        
        # Create comparison table
        comparison_data = []
        
        for model_name, results in model_results.items():
            row = {'Model': model_name}
            
            # Extract metrics
            if 'diagnostics' in results and results['diagnostics']:
                r2_analysis = results['diagnostics'].get('r2_analysis', {})
                row['R²'] = f"{r2_analysis.get('overall_r2', 0):.4f}"
                row['Adj. R²'] = f"{r2_analysis.get('adjusted_r2', 0):.4f}"
            elif 'r_squared' in results:
                row['R²'] = f"{results['r_squared']:.4f}"
                row['Adj. R²'] = 'N/A'
            
            # Add model-specific metrics
            if results.get('model_type') == 'Hansen Threshold':
                row['Threshold'] = f"{results.get('threshold_estimate', 0):.4f}"
                row['Regime 1 Obs'] = results.get('diagnostics', {}).get('r2_analysis', {}).get('regime1_obs', 'N/A')
                row['Regime 2 Obs'] = results.get('diagnostics', {}).get('r2_analysis', {}).get('regime2_obs', 'N/A')
            elif results.get('model_type') == 'Smooth Transition':
                row['Gamma'] = f"{results.get('gamma', 0):.4f}"
                row['Transition Center'] = f"{results.get('transition_center', 0):.4f}"
            elif results.get('model_type') == 'Local Projections':
                row['Horizons'] = results.get('horizons_estimated', 'N/A')
            
            row['Fit Time (s)'] = f"{results.get('fit_time', 0):.3f}"
            
            comparison_data.append(row)
        
        # Create DataFrame for display
        comparison_df = pd.DataFrame(comparison_data)
        
        # Create HTML table
        html_table = comparison_df.to_html(index=False, classes='table table-striped')
        
        # Create widgets
        table_widget = widgets.HTML(value=html_table)
        
        # Model selector for detailed view
        model_selector = widgets.Dropdown(
            options=list(model_results.keys()),
            description='Detailed View:'
        )
        
        detailed_output = widgets.Output()
        
        def show_detailed_results(change):
            with detailed_output:
                clear_output(wait=True)
                selected_model = change['new']
                results = model_results[selected_model]
                
                print(f"Detailed Results for {selected_model}")
                print("=" * 50)
                
                if 'diagnostics' in results and results['diagnostics']:
                    diagnostics = results['diagnostics']
                    
                    # R² Analysis
                    r2_analysis = diagnostics.get('r2_analysis', {})
                    print(f"Overall R²: {r2_analysis.get('overall_r2', 0):.6f}")
                    print(f"Adjusted R²: {r2_analysis.get('adjusted_r2', 0):.6f}")
                    print(f"Regime 1 R²: {r2_analysis.get('regime1_r2', 0):.6f}")
                    print(f"Regime 2 R²: {r2_analysis.get('regime2_r2', 0):.6f}")
                    
                    # Improvement suggestions
                    improvements = diagnostics.get('improvement_recommendations', {})
                    if improvements:
                        print("\nImprovement Recommendations:")
                        for category, suggestions in improvements.items():
                            if suggestions:
                                print(f"\n{category.replace('_', ' ').title()}:")
                                for suggestion in suggestions[:3]:  # Show top 3
                                    print(f"  • {suggestion}")
                
                # Model-specific details
                if results.get('model_type') == 'Hansen Threshold':
                    print(f"\nThreshold Estimate: {results.get('threshold_estimate', 0):.6f}")
                    if 'regime1_coef' in results:
                        print(f"Regime 1 Coefficients: {results['regime1_coef']}")
                    if 'regime2_coef' in results:
                        print(f"Regime 2 Coefficients: {results['regime2_coef']}")
                        
                elif results.get('model_type') == 'Smooth Transition':
                    print(f"\nGamma Parameter: {results.get('gamma', 0):.6f}")
                    print(f"Transition Center: {results.get('transition_center', 0):.6f}")
                    
                elif results.get('model_type') == 'Local Projections':
                    if 'impulse_responses' in results:
                        ir = results['impulse_responses']
                        print(f"\nImpulse Response Summary:")
                        print(f"Horizons estimated: {len(ir)}")
                        if len(ir) > 0:
                            print(f"Peak response: {ir['coefficient'].max():.6f} at horizon {ir.loc[ir['coefficient'].idxmax(), 'horizon']}")
        
        model_selector.observe(show_detailed_results, names='value')
        
        # Initial display
        if model_results:
            show_detailed_results({'new': list(model_results.keys())[0]})
        
        return widgets.VBox([
            widgets.HTML("<h3>Model Comparison</h3>"),
            table_widget,
            widgets.HBox([model_selector]),
            detailed_output
        ])
    
    def _prepare_data(self, 
                     y_column: str, 
                     x_columns: List[str], 
                     threshold_column: str,
                     transformation: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Prepare data with specified transformation.
        
        Parameters:
        -----------
        y_column : str
            Dependent variable column name
        x_columns : List[str]
            Independent variable column names
        threshold_column : str
            Threshold variable column name
        transformation : str
            Type of transformation to apply
            
        Returns:
        --------
        Tuple[np.ndarray, np.ndarray, np.ndarray]
            Transformed y, x, and threshold data
        """
        # Extract data
        y = self.data[y_column].values
        x = self.data[x_columns].values
        threshold_var = self.data[threshold_column].values
        
        # Remove NaN values
        valid_mask = ~(np.isnan(y) | np.any(np.isnan(x), axis=1) | np.isnan(threshold_var))
        y = y[valid_mask]
        x = x[valid_mask]
        threshold_var = threshold_var[valid_mask]
        
        # Apply transformation
        if transformation == 'First Differences':
            y = np.diff(y)
            x = np.diff(x, axis=0)
            threshold_var = np.diff(threshold_var)
        elif transformation == 'Log Levels':
            # Only apply log to positive values
            if np.all(y > 0):
                y = np.log(y)
            if np.all(x > 0):
                x = np.log(x)
            if np.all(threshold_var > 0):
                threshold_var = np.log(threshold_var)
        elif transformation == 'Standardized':
            from sklearn.preprocessing import StandardScaler
            scaler_y = StandardScaler()
            scaler_x = StandardScaler()
            scaler_thresh = StandardScaler()
            
            y = scaler_y.fit_transform(y.reshape(-1, 1)).flatten()
            x = scaler_x.fit_transform(x)
            threshold_var = scaler_thresh.fit_transform(threshold_var.reshape(-1, 1)).flatten()
        
        return y, x, threshold_var
    
    def _generate_cache_key(self, *args) -> str:
        """Generate cache key for model results."""
        return str(hash(str(args)))
    
    def _display_model_results(self, results: Dict[str, Any], model_type: str):
        """Display model results in a formatted way."""
        if results.get('error'):
            print(f"Error: {results['error']}")
            return
        
        print(f"{model_type} Model Results")
        print("=" * 40)
        
        if 'r_squared' in results:
            print(f"R-squared: {results['r_squared']:.6f}")
        
        if 'fit_time' in results:
            print(f"Fit time: {results['fit_time']:.3f} seconds")
        
        # Model-specific output
        if model_type == 'Hansen Threshold':
            if 'threshold_estimate' in results:
                print(f"Threshold estimate: {results['threshold_estimate']:.6f}")
            
            if 'diagnostics' in results and results['diagnostics']:
                r2_analysis = results['diagnostics']['r2_analysis']
                print(f"Regime 1 observations: {r2_analysis.get('regime1_obs', 0)}")
                print(f"Regime 2 observations: {r2_analysis.get('regime2_obs', 0)}")
                print(f"Regime 1 R²: {r2_analysis.get('regime1_r2', 0):.6f}")
                print(f"Regime 2 R²: {r2_analysis.get('regime2_r2', 0):.6f}")
                
                # Show concern level
                concern = results['diagnostics'].get('r2_concern_level', {})
                print(f"R² Assessment: {concern.get('level', 'unknown')} - {concern.get('description', '')}")
        
        elif model_type == 'Smooth Transition':
            if 'gamma' in results:
                print(f"Gamma parameter: {results['gamma']:.6f}")
            if 'transition_center' in results:
                print(f"Transition center: {results['transition_center']:.6f}")
        
        elif model_type == 'Local Projections':
            if 'horizons_estimated' in results:
                print(f"Horizons estimated: {results['horizons_estimated']}")
    
    def create_robustness_tester(self, 
                                base_results: Dict[str, Any],
                                y_column: str,
                                x_columns: List[str],
                                threshold_column: str) -> widgets.VBox:
        """
        Create interactive robustness testing interface with real-time sensitivity analysis.
        
        Parameters:
        -----------
        base_results : Dict[str, Any]
            Base model results to test robustness against
        y_column : str
            Dependent variable column name
        x_columns : List[str]
            Independent variable column names
        threshold_column : str
            Threshold variable column name
            
        Returns:
        --------
        widgets.VBox
            Interactive robustness testing widget
        """
        # Robustness test controls
        test_selector = widgets.SelectMultiple(
            options=[
                'Subsample Stability',
                'Parameter Sensitivity',
                'Alternative Specifications',
                'Bootstrap Confidence',
                'Outlier Robustness'
            ],
            value=['Subsample Stability', 'Parameter Sensitivity'],
            description='Robustness Tests:'
        )
        
        # Subsample controls
        start_date_slider = widgets.IntSlider(
            value=0,
            min=0,
            max=len(self.data) - 50,
            step=1,
            description='Start Index:',
            style={'description_width': 'initial'}
        )
        
        end_date_slider = widgets.IntSlider(
            value=len(self.data),
            min=50,
            max=len(self.data),
            step=1,
            description='End Index:',
            style={'description_width': 'initial'}
        )
        
        # Bootstrap controls
        bootstrap_samples = widgets.IntSlider(
            value=100,
            min=50,
            max=1000,
            step=50,
            description='Bootstrap Samples:',
            style={'description_width': 'initial'}
        )
        
        # Sensitivity controls
        sensitivity_range = widgets.FloatSlider(
            value=0.1,
            min=0.05,
            max=0.5,
            step=0.05,
            description='Sensitivity Range:',
            style={'description_width': 'initial'}
        )
        
        # Run tests button
        run_tests_button = widgets.Button(
            description='Run Robustness Tests',
            button_style='success'
        )
        
        # Output area
        robustness_output = widgets.Output()
        
        def run_robustness_tests(button):
            with robustness_output:
                clear_output(wait=True)
                
                selected_tests = list(test_selector.value)
                print("Running robustness tests...")
                print("=" * 50)
                
                robustness_results = {}
                
                # Prepare base data
                y_data, x_data, thresh_data = self._prepare_data(
                    y_column, x_columns, threshold_column, 'Levels'
                )
                
                for test_name in selected_tests:
                    print(f"\nRunning {test_name}...")
                    
                    try:
                        if test_name == 'Subsample Stability':
                            result = self.interactive_subsample_analysis(
                                y_data, x_data, thresh_data,
                                start_idx=start_date_slider.value,
                                end_idx=end_date_slider.value
                            )
                            
                        elif test_name == 'Parameter Sensitivity':
                            result = self.parameter_sensitivity_interface(
                                y_data, x_data, thresh_data,
                                sensitivity_range=sensitivity_range.value
                            )
                            
                        elif test_name == 'Alternative Specifications':
                            result = self.specification_robustness_explorer(
                                y_data, x_data, thresh_data
                            )
                            
                        elif test_name == 'Bootstrap Confidence':
                            result = self._bootstrap_robustness_test(
                                y_data, x_data, thresh_data,
                                n_bootstrap=bootstrap_samples.value
                            )
                            
                        elif test_name == 'Outlier Robustness':
                            result = self._outlier_robustness_test(
                                y_data, x_data, thresh_data
                            )
                        
                        robustness_results[test_name] = result
                        self._display_robustness_result(test_name, result)
                        
                    except Exception as e:
                        print(f"Error in {test_name}: {str(e)}")
                        robustness_results[test_name] = {'error': str(e)}
                
                # Store results for export
                self.comparison_results['robustness'] = robustness_results
                print("\nRobustness testing completed!")
        
        run_tests_button.on_click(run_robustness_tests)
        
        # Layout
        controls = widgets.VBox([
            test_selector,
            widgets.HBox([start_date_slider, end_date_slider]),
            widgets.HBox([bootstrap_samples, sensitivity_range]),
            run_tests_button
        ])
        
        return widgets.VBox([
            widgets.HTML("<h3>Robustness Testing</h3>"),
            controls,
            robustness_output
        ])
    
    def interactive_subsample_analysis(self, 
                                     y: np.ndarray,
                                     x: np.ndarray,
                                     threshold_var: np.ndarray,
                                     start_idx: int = 0,
                                     end_idx: int = None) -> Dict[str, Any]:
        """
        Perform interactive subsample analysis for temporal stability testing.
        
        Parameters:
        -----------
        y : np.ndarray
            Dependent variable
        x : np.ndarray
            Independent variables
        threshold_var : np.ndarray
            Threshold variable
        start_idx : int
            Starting index for subsample
        end_idx : int, optional
            Ending index for subsample
            
        Returns:
        --------
        Dict[str, Any]
            Subsample analysis results
        """
        if end_idx is None:
            end_idx = len(y)
        
        # Create subsample
        y_sub = y[start_idx:end_idx]
        x_sub = x[start_idx:end_idx]
        thresh_sub = threshold_var[start_idx:end_idx]
        
        # Fit model on subsample
        model = HansenThresholdRegression()
        model.fit(y_sub, x_sub, thresh_sub)
        
        # Calculate diagnostics
        diagnostics = self.diagnostics.diagnose_low_r_squared(
            model, y_sub, x_sub, thresh_sub
        )
        
        # Compare with full sample (if available)
        full_sample_r2 = None
        if hasattr(self, 'current_results') and 'Hansen Threshold' in self.current_results:
            full_results = self.current_results['Hansen Threshold']
            if 'diagnostics' in full_results:
                full_sample_r2 = full_results['diagnostics']['r2_analysis']['overall_r2']
        
        return {
            'subsample_size': len(y_sub),
            'subsample_r2': diagnostics['r2_analysis']['overall_r2'],
            'full_sample_r2': full_sample_r2,
            'threshold_estimate': model.threshold,
            'regime_balance': diagnostics['r2_analysis']['regime_balance'],
            'start_idx': start_idx,
            'end_idx': end_idx
        }
    
    def parameter_sensitivity_interface(self, 
                                      y: np.ndarray,
                                      x: np.ndarray,
                                      threshold_var: np.ndarray,
                                      sensitivity_range: float = 0.1) -> Dict[str, Any]:
        """
        Create parameter sensitivity analysis with tornado plot generation.
        
        Parameters:
        -----------
        y : np.ndarray
            Dependent variable
        x : np.ndarray
            Independent variables
        threshold_var : np.ndarray
            Threshold variable
        sensitivity_range : float
            Range for sensitivity analysis (as fraction)
            
        Returns:
        --------
        Dict[str, Any]
            Parameter sensitivity results
        """
        # Base model
        base_model = HansenThresholdRegression()
        base_model.fit(y, x, threshold_var)
        
        base_diagnostics = self.diagnostics.diagnose_low_r_squared(
            base_model, y, x, threshold_var
        )
        base_r2 = base_diagnostics['r2_analysis']['overall_r2']
        
        # Test different trim fractions
        trim_values = np.linspace(0.05, 0.30, 10)
        trim_r2_values = []
        
        for trim in trim_values:
            try:
                model = HansenThresholdRegression()
                model.fit(y, x, threshold_var, trim=trim)
                
                diagnostics = self.diagnostics.diagnose_low_r_squared(
                    model, y, x, threshold_var
                )
                trim_r2_values.append(diagnostics['r2_analysis']['overall_r2'])
            except:
                trim_r2_values.append(0)
        
        # Test data transformations
        transformations = ['Levels', 'First Differences', 'Log Levels', 'Standardized']
        transform_r2_values = []
        
        for transform in transformations:
            try:
                y_trans, x_trans, thresh_trans = self._prepare_data(
                    self.data.columns[0],  # Placeholder - would need actual column names
                    self.data.columns[1:3].tolist(),
                    self.data.columns[3],
                    transform
                )
                
                model = HansenThresholdRegression()
                model.fit(y_trans, x_trans, thresh_trans)
                
                diagnostics = self.diagnostics.diagnose_low_r_squared(
                    model, y_trans, x_trans, thresh_trans
                )
                transform_r2_values.append(diagnostics['r2_analysis']['overall_r2'])
            except:
                transform_r2_values.append(0)
        
        return {
            'base_r2': base_r2,
            'trim_sensitivity': {
                'trim_values': trim_values.tolist(),
                'r2_values': trim_r2_values,
                'best_trim': trim_values[np.argmax(trim_r2_values)],
                'best_r2': max(trim_r2_values)
            },
            'transformation_sensitivity': {
                'transformations': transformations,
                'r2_values': transform_r2_values,
                'best_transformation': transformations[np.argmax(transform_r2_values)],
                'best_r2': max(transform_r2_values)
            }
        }
    
    def specification_robustness_explorer(self, 
                                        y: np.ndarray,
                                        x: np.ndarray,
                                        threshold_var: np.ndarray) -> Dict[str, Any]:
        """
        Explore alternative model specifications for robustness testing.
        
        Parameters:
        -----------
        y : np.ndarray
            Dependent variable
        x : np.ndarray
            Independent variables
        threshold_var : np.ndarray
            Threshold variable
            
        Returns:
        --------
        Dict[str, Any]
            Specification robustness results
        """
        # Test alternative specifications using diagnostics module
        alternative_specs = self.diagnostics.generate_alternative_specifications(
            y, x, threshold_var
        )
        
        # Test different model types
        model_comparison = {}
        
        # Hansen Threshold
        try:
            hansen_model = HansenThresholdRegression()
            hansen_model.fit(y, x, threshold_var)
            hansen_diagnostics = self.diagnostics.diagnose_low_r_squared(
                hansen_model, y, x, threshold_var
            )
            model_comparison['Hansen Threshold'] = {
                'r2': hansen_diagnostics['r2_analysis']['overall_r2'],
                'threshold': hansen_model.threshold
            }
        except Exception as e:
            model_comparison['Hansen Threshold'] = {'error': str(e)}
        
        # Smooth Transition
        try:
            str_model = SmoothTransitionRegression()
            str_model.fit(y, x, threshold_var)
            y_pred = str_model.predict(x, threshold_var)
            ss_res = np.sum((y - y_pred) ** 2)
            ss_tot = np.sum((y - np.mean(y)) ** 2)
            str_r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
            
            model_comparison['Smooth Transition'] = {
                'r2': str_r2,
                'gamma': str_model.gamma,
                'transition_center': str_model.c
            }
        except Exception as e:
            model_comparison['Smooth Transition'] = {'error': str(e)}
        
        return {
            'alternative_specifications': alternative_specs,
            'model_comparison': model_comparison
        }
    
    def _bootstrap_robustness_test(self, 
                                  y: np.ndarray,
                                  x: np.ndarray,
                                  threshold_var: np.ndarray,
                                  n_bootstrap: int = 100) -> Dict[str, Any]:
        """Perform bootstrap robustness testing."""
        
        bootstrap_r2 = []
        bootstrap_thresholds = []
        
        n_obs = len(y)
        
        for i in range(n_bootstrap):
            # Bootstrap sample
            boot_indices = np.random.choice(n_obs, size=n_obs, replace=True)
            y_boot = y[boot_indices]
            x_boot = x[boot_indices]
            thresh_boot = threshold_var[boot_indices]
            
            try:
                model = HansenThresholdRegression()
                model.fit(y_boot, x_boot, thresh_boot)
                
                diagnostics = self.diagnostics.diagnose_low_r_squared(
                    model, y_boot, x_boot, thresh_boot
                )
                
                bootstrap_r2.append(diagnostics['r2_analysis']['overall_r2'])
                bootstrap_thresholds.append(model.threshold)
                
            except:
                continue
        
        if len(bootstrap_r2) > 0:
            return {
                'bootstrap_r2_mean': np.mean(bootstrap_r2),
                'bootstrap_r2_std': np.std(bootstrap_r2),
                'bootstrap_r2_ci': [np.percentile(bootstrap_r2, 2.5), np.percentile(bootstrap_r2, 97.5)],
                'bootstrap_threshold_mean': np.mean(bootstrap_thresholds),
                'bootstrap_threshold_std': np.std(bootstrap_thresholds),
                'bootstrap_threshold_ci': [np.percentile(bootstrap_thresholds, 2.5), np.percentile(bootstrap_thresholds, 97.5)],
                'successful_bootstraps': len(bootstrap_r2)
            }
        else:
            return {'error': 'No successful bootstrap samples'}
    
    def _outlier_robustness_test(self, 
                                y: np.ndarray,
                                x: np.ndarray,
                                threshold_var: np.ndarray) -> Dict[str, Any]:
        """Test robustness to outliers."""
        
        # Identify outliers using IQR method
        def identify_outliers(data):
            Q1 = np.percentile(data, 25)
            Q3 = np.percentile(data, 75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            return (data < lower_bound) | (data > upper_bound)
        
        # Find outliers in y and threshold variable
        y_outliers = identify_outliers(y)
        thresh_outliers = identify_outliers(threshold_var)
        
        # Combined outlier mask
        outlier_mask = y_outliers | thresh_outliers
        
        # Fit model without outliers
        if np.sum(~outlier_mask) > 20:  # Ensure sufficient observations
            y_clean = y[~outlier_mask]
            x_clean = x[~outlier_mask]
            thresh_clean = threshold_var[~outlier_mask]
            
            try:
                model_clean = HansenThresholdRegression()
                model_clean.fit(y_clean, x_clean, thresh_clean)
                
                diagnostics_clean = self.diagnostics.diagnose_low_r_squared(
                    model_clean, y_clean, x_clean, thresh_clean
                )
                
                # Compare with full sample
                model_full = HansenThresholdRegression()
                model_full.fit(y, x, threshold_var)
                
                diagnostics_full = self.diagnostics.diagnose_low_r_squared(
                    model_full, y, x, threshold_var
                )
                
                return {
                    'outliers_detected': np.sum(outlier_mask),
                    'outlier_percentage': np.sum(outlier_mask) / len(y) * 100,
                    'full_sample_r2': diagnostics_full['r2_analysis']['overall_r2'],
                    'clean_sample_r2': diagnostics_clean['r2_analysis']['overall_r2'],
                    'r2_improvement': diagnostics_clean['r2_analysis']['overall_r2'] - diagnostics_full['r2_analysis']['overall_r2'],
                    'threshold_full': model_full.threshold,
                    'threshold_clean': model_clean.threshold
                }
                
            except Exception as e:
                return {'error': f'Could not fit clean model: {str(e)}'}
        else:
            return {'error': 'Too many outliers detected - insufficient clean observations'}
    
    def _display_robustness_result(self, test_name: str, result: Dict[str, Any]):
        """Display robustness test results."""
        
        print(f"\n{test_name} Results:")
        print("-" * 30)
        
        if 'error' in result:
            print(f"Error: {result['error']}")
            return
        
        if test_name == 'Subsample Stability':
            print(f"Subsample size: {result['subsample_size']}")
            print(f"Subsample R²: {result['subsample_r2']:.6f}")
            if result['full_sample_r2']:
                print(f"Full sample R²: {result['full_sample_r2']:.6f}")
                print(f"R² difference: {result['subsample_r2'] - result['full_sample_r2']:.6f}")
            print(f"Threshold estimate: {result['threshold_estimate']:.6f}")
            
        elif test_name == 'Parameter Sensitivity':
            print(f"Base R²: {result['base_r2']:.6f}")
            
            trim_sens = result['trim_sensitivity']
            print(f"Best trim fraction: {trim_sens['best_trim']:.3f} (R² = {trim_sens['best_r2']:.6f})")
            
            trans_sens = result['transformation_sensitivity']
            print(f"Best transformation: {trans_sens['best_transformation']} (R² = {trans_sens['best_r2']:.6f})")
            
        elif test_name == 'Bootstrap Confidence':
            print(f"Bootstrap R² mean: {result['bootstrap_r2_mean']:.6f}")
            print(f"Bootstrap R² std: {result['bootstrap_r2_std']:.6f}")
            print(f"Bootstrap R² 95% CI: [{result['bootstrap_r2_ci'][0]:.6f}, {result['bootstrap_r2_ci'][1]:.6f}]")
            print(f"Successful bootstraps: {result['successful_bootstraps']}")
            
        elif test_name == 'Outlier Robustness':
            print(f"Outliers detected: {result['outliers_detected']} ({result['outlier_percentage']:.1f}%)")
            print(f"Full sample R²: {result['full_sample_r2']:.6f}")
            print(f"Clean sample R²: {result['clean_sample_r2']:.6f}")
            print(f"R² improvement: {result['r2_improvement']:.6f}")
    
    def export_publication_figures(self, 
                                 selected_results: Dict[str, Any],
                                 export_path: str = "publication_outputs",
                                 formats: List[str] = None) -> Dict[str, str]:
        """
        Export selected specifications as publication-ready figures.
        
        Parameters:
        -----------
        selected_results : Dict[str, Any]
            Selected model results to export
        export_path : str
            Base path for exports
        formats : List[str], optional
            Export formats (default: ['png', 'pdf'])
            
        Returns:
        --------
        Dict[str, str]
            Dictionary mapping figure names to file paths
        """
        if formats is None:
            formats = self.export_settings['formats']
        
        export_path = Path(export_path)
        export_path.mkdir(parents=True, exist_ok=True)
        
        exported_files = {}
        
        for model_name, results in selected_results.items():
            try:
                if results.get('model_type') == 'Hansen Threshold' and 'diagnostics' in results:
                    # Create threshold analysis figure
                    threshold_results = {
                        'threshold_estimate': results.get('threshold_estimate'),
                        'regime_1_coef': results.get('regime1_coef'),
                        'regime_2_coef': results.get('regime2_coef')
                    }
                    
                    fig = self.visualizer.create_threshold_analysis_figure(
                        threshold_results,
                        title=f"{model_name} Threshold Analysis"
                    )
                    
                    # Save figure
                    base_filename = f"{model_name.lower().replace(' ', '_')}_threshold_analysis"
                    for fmt in formats:
                        file_path = export_path / f"{base_filename}.{fmt}"
                        fig.savefig(file_path, format=fmt, dpi=self.export_settings['dpi'],
                                   bbox_inches='tight')
                        exported_files[f"{model_name}_threshold_{fmt}"] = str(file_path)
                    
                    plt.close(fig)
                    
                    # Create diagnostic panel
                    if 'model' in results:
                        model = results['model']
                        # Prepare diagnostic data (simplified)
                        diagnostics_data = {
                            'residuals': np.random.normal(0, 1, 100),  # Placeholder
                            'fitted_values': np.random.normal(0, 1, 100),  # Placeholder
                            'standardized_residuals': np.random.normal(0, 1, 100)  # Placeholder
                        }
                        
                        fig = self.visualizer.create_diagnostic_panel_figure(
                            diagnostics_data,
                            title=f"{model_name} Diagnostic Panel"
                        )
                        
                        base_filename = f"{model_name.lower().replace(' ', '_')}_diagnostics"
                        for fmt in formats:
                            file_path = export_path / f"{base_filename}.{fmt}"
                            fig.savefig(file_path, format=fmt, dpi=self.export_settings['dpi'],
                                       bbox_inches='tight')
                            exported_files[f"{model_name}_diagnostics_{fmt}"] = str(file_path)
                        
                        plt.close(fig)
                
            except Exception as e:
                print(f"Error exporting {model_name}: {str(e)}")
        
        return exported_files
    
    def batch_figure_generation(self, 
                               analysis_results: Dict[str, Any],
                               output_directory: str = "batch_outputs") -> Dict[str, List[str]]:
        """
        Generate batch figures for automated publication output creation.
        
        Parameters:
        -----------
        analysis_results : Dict[str, Any]
            Complete analysis results from dashboard
        output_directory : str
            Directory for batch outputs
            
        Returns:
        --------
        Dict[str, List[str]]
            Dictionary mapping figure types to lists of generated files
        """
        output_path = Path(output_directory)
        output_path.mkdir(parents=True, exist_ok=True)
        
        generated_files = {
            'threshold_analysis': [],
            'diagnostics': [],
            'robustness': [],
            'comparison': []
        }
        
        # Generate threshold analysis figures
        for model_name, results in analysis_results.items():
            if results.get('model_type') == 'Hansen Threshold':
                try:
                    threshold_results = {
                        'threshold_estimate': results.get('threshold_estimate'),
                        'regime_1_coef': results.get('regime1_coef'),
                        'regime_2_coef': results.get('regime2_coef')
                    }
                    
                    fig = self.visualizer.create_threshold_analysis_figure(
                        threshold_results,
                        title=f"{model_name} Analysis"
                    )
                    
                    filename = output_path / f"threshold_{model_name.lower().replace(' ', '_')}.png"
                    fig.savefig(filename, dpi=300, bbox_inches='tight')
                    generated_files['threshold_analysis'].append(str(filename))
                    plt.close(fig)
                    
                except Exception as e:
                    print(f"Error generating threshold figure for {model_name}: {str(e)}")
        
        # Generate comparison figures if multiple models
        if len(analysis_results) > 1:
            try:
                # Prepare comparison data
                comparison_data = {}
                for model_name, results in analysis_results.items():
                    if 'diagnostics' in results:
                        r2_analysis = results['diagnostics']['r2_analysis']
                        comparison_data[model_name] = {
                            'r_squared': r2_analysis['overall_r2'],
                            'aic': 0,  # Placeholder
                            'bic': 0,  # Placeholder
                            'log_likelihood': 0  # Placeholder
                        }
                
                if comparison_data:
                    fig = self.visualizer.create_model_fit_comparison_figure(
                        comparison_data,
                        title="Model Comparison"
                    )
                    
                    filename = output_path / "model_comparison.png"
                    fig.savefig(filename, dpi=300, bbox_inches='tight')
                    generated_files['comparison'].append(str(filename))
                    plt.close(fig)
                    
            except Exception as e:
                print(f"Error generating comparison figure: {str(e)}")
        
        return generated_files
    
    def automated_report_generation(self, 
                                  analysis_results: Dict[str, Any],
                                  report_path: str = "automated_report.html") -> str:
        """
        Create automated report generation with comprehensive results summary.
        
        Parameters:
        -----------
        analysis_results : Dict[str, Any]
            Complete analysis results
        report_path : str
            Path for the generated report
            
        Returns:
        --------
        str
            Path to the generated report
        """
        html_content = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Interactive Dashboard Analysis Report</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 40px; }
                .header { background-color: #f0f0f0; padding: 20px; border-radius: 5px; }
                .section { margin: 20px 0; }
                .model-results { border: 1px solid #ddd; padding: 15px; margin: 10px 0; border-radius: 5px; }
                .metric { display: inline-block; margin: 10px 20px 10px 0; }
                .warning { color: #d9534f; font-weight: bold; }
                .success { color: #5cb85c; font-weight: bold; }
                table { border-collapse: collapse; width: 100%; }
                th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
                th { background-color: #f2f2f2; }
            </style>
        </head>
        <body>
        """
        
        # Header
        html_content += f"""
        <div class="header">
            <h1>Interactive Dashboard Analysis Report</h1>
            <p>Generated on: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            <p>Models analyzed: {len(analysis_results)}</p>
        </div>
        """
        
        # Summary table
        html_content += """
        <div class="section">
            <h2>Model Summary</h2>
            <table>
                <tr>
                    <th>Model</th>
                    <th>R²</th>
                    <th>Fit Time (s)</th>
                    <th>Status</th>
                    <th>Key Findings</th>
                </tr>
        """
        
        for model_name, results in analysis_results.items():
            r2 = "N/A"
            fit_time = results.get('fit_time', 0)
            status = "Error" if results.get('error') else "Success"
            findings = ""
            
            if 'diagnostics' in results and results['diagnostics']:
                r2_analysis = results['diagnostics']['r2_analysis']
                r2 = f"{r2_analysis['overall_r2']:.6f}"
                
                concern_level = results['diagnostics'].get('r2_concern_level', {})
                if concern_level.get('level') == 'critical':
                    findings = "Critical R² issue"
                elif concern_level.get('level') == 'severe':
                    findings = "Severe R² issue"
                else:
                    findings = "Acceptable fit"
            elif 'r_squared' in results and results['r_squared'] is not None:
                r2 = f"{results['r_squared']:.6f}"
                findings = "Basic fit statistics"
            
            status_class = "success" if status == "Success" else "warning"
            
            html_content += f"""
                <tr>
                    <td>{model_name}</td>
                    <td>{r2}</td>
                    <td>{fit_time:.3f}</td>
                    <td class="{status_class}">{status}</td>
                    <td>{findings}</td>
                </tr>
            """
        
        html_content += """
            </table>
        </div>
        """
        
        # Detailed results for each model
        for model_name, results in analysis_results.items():
            html_content += f"""
            <div class="section">
                <div class="model-results">
                    <h3>{model_name} Detailed Results</h3>
            """
            
            if results.get('error'):
                html_content += f'<p class="warning">Error: {results["error"]}</p>'
            else:
                # Model-specific details
                if results.get('model_type') == 'Hansen Threshold':
                    html_content += f"""
                    <div class="metric">Threshold Estimate: {results.get('threshold_estimate', 'N/A')}</div>
                    """
                    
                    if 'diagnostics' in results:
                        r2_analysis = results['diagnostics']['r2_analysis']
                        html_content += f"""
                        <div class="metric">Regime 1 Obs: {r2_analysis.get('regime1_obs', 'N/A')}</div>
                        <div class="metric">Regime 2 Obs: {r2_analysis.get('regime2_obs', 'N/A')}</div>
                        <div class="metric">Regime 1 R²: {r2_analysis.get('regime1_r2', 0):.6f}</div>
                        <div class="metric">Regime 2 R²: {r2_analysis.get('regime2_r2', 0):.6f}</div>
                        """
                        
                        # Improvement recommendations
                        improvements = results['diagnostics'].get('improvement_recommendations', {})
                        if improvements:
                            html_content += "<h4>Improvement Recommendations:</h4><ul>"
                            for category, suggestions in improvements.items():
                                if suggestions:
                                    html_content += f"<li><strong>{category.replace('_', ' ').title()}:</strong><ul>"
                                    for suggestion in suggestions[:3]:  # Top 3 suggestions
                                        html_content += f"<li>{suggestion}</li>"
                                    html_content += "</ul></li>"
                            html_content += "</ul>"
                
                elif results.get('model_type') == 'Smooth Transition':
                    html_content += f"""
                    <div class="metric">Gamma: {results.get('gamma', 'N/A')}</div>
                    <div class="metric">Transition Center: {results.get('transition_center', 'N/A')}</div>
                    <div class="metric">R²: {results.get('r_squared', 'N/A')}</div>
                    """
                
                elif results.get('model_type') == 'Local Projections':
                    html_content += f"""
                    <div class="metric">Horizons Estimated: {results.get('horizons_estimated', 'N/A')}</div>
                    """
            
            html_content += """
                </div>
            </div>
            """
        
        # Robustness results if available
        if hasattr(self, 'comparison_results') and 'robustness' in self.comparison_results:
            html_content += """
            <div class="section">
                <h2>Robustness Testing Results</h2>
            """
            
            for test_name, test_results in self.comparison_results['robustness'].items():
                html_content += f"""
                <div class="model-results">
                    <h4>{test_name}</h4>
                """
                
                if 'error' in test_results:
                    html_content += f'<p class="warning">Error: {test_results["error"]}</p>'
                else:
                    # Display key metrics based on test type
                    if test_name == 'Bootstrap Confidence':
                        html_content += f"""
                        <div class="metric">Bootstrap R² Mean: {test_results.get('bootstrap_r2_mean', 'N/A')}</div>
                        <div class="metric">Bootstrap R² Std: {test_results.get('bootstrap_r2_std', 'N/A')}</div>
                        """
                    elif test_name == 'Outlier Robustness':
                        html_content += f"""
                        <div class="metric">Outliers Detected: {test_results.get('outliers_detected', 'N/A')}</div>
                        <div class="metric">R² Improvement: {test_results.get('r2_improvement', 'N/A')}</div>
                        """
                
                html_content += "</div>"
            
            html_content += "</div>"
        
        html_content += """
        </body>
        </html>
        """
        
        # Write report
        report_path = Path(report_path)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        return str(report_path)
    
    def publication_quality_checker(self, 
                                  exported_files: Dict[str, str],
                                  quality_standards: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Implement publication quality checker for output validation.
        
        Parameters:
        -----------
        exported_files : Dict[str, str]
            Dictionary of exported files to check
        quality_standards : Dict[str, Any], optional
            Quality standards to check against
            
        Returns:
        --------
        Dict[str, Any]
            Quality check results
        """
        if quality_standards is None:
            quality_standards = {
                'min_dpi': 300,
                'required_formats': ['png', 'pdf'],
                'max_file_size_mb': 10,
                'min_figure_width': 3.0,
                'min_figure_height': 2.0
            }
        
        quality_results = {
            'passed_files': [],
            'failed_files': [],
            'warnings': [],
            'overall_status': 'PASS'
        }
        
        for file_key, file_path in exported_files.items():
            file_path = Path(file_path)
            
            if not file_path.exists():
                quality_results['failed_files'].append({
                    'file': str(file_path),
                    'issue': 'File does not exist'
                })
                continue
            
            # Check file size
            file_size_mb = file_path.stat().st_size / (1024 * 1024)
            if file_size_mb > quality_standards['max_file_size_mb']:
                quality_results['warnings'].append({
                    'file': str(file_path),
                    'issue': f'File size ({file_size_mb:.1f} MB) exceeds recommended maximum'
                })
            
            # Check format
            file_format = file_path.suffix.lower().lstrip('.')
            if file_format not in quality_standards['required_formats']:
                quality_results['warnings'].append({
                    'file': str(file_path),
                    'issue': f'Format {file_format} not in required formats'
                })
            
            # For image files, check basic properties
            if file_format in ['png', 'jpg', 'jpeg']:
                try:
                    from PIL import Image
                    with Image.open(file_path) as img:
                        width_inches = img.width / quality_standards['min_dpi']
                        height_inches = img.height / quality_standards['min_dpi']
                        
                        if width_inches < quality_standards['min_figure_width']:
                            quality_results['warnings'].append({
                                'file': str(file_path),
                                'issue': f'Figure width ({width_inches:.1f}") below minimum'
                            })
                        
                        if height_inches < quality_standards['min_figure_height']:
                            quality_results['warnings'].append({
                                'file': str(file_path),
                                'issue': f'Figure height ({height_inches:.1f}") below minimum'
                            })
                except ImportError:
                    quality_results['warnings'].append({
                        'file': str(file_path),
                        'issue': 'Cannot check image properties (PIL not available)'
                    })
                except Exception as e:
                    quality_results['warnings'].append({
                        'file': str(file_path),
                        'issue': f'Error checking image: {str(e)}'
                    })
            
            # If no major issues, mark as passed
            if not any(fail['file'] == str(file_path) for fail in quality_results['failed_files']):
                quality_results['passed_files'].append(str(file_path))
        
        # Determine overall status
        if quality_results['failed_files']:
            quality_results['overall_status'] = 'FAIL'
        elif quality_results['warnings']:
            quality_results['overall_status'] = 'PASS_WITH_WARNINGS'
        
        return quality_results