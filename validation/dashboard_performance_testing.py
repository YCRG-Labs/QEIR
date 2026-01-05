"""
Dashboard Performance Testing Module

This module provides specialized performance testing for the interactive
analysis dashboard, focusing on real-time responsiveness and user experience
metrics.
"""

import time
import asyncio
import threading
from typing import Dict, List, Any, Optional, Callable
import numpy as np
import pandas as pd
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed
import psutil
import matplotlib.pyplot as plt
from performance_benchmarks import PerformanceMetrics, MemoryOptimizer

@dataclass
class DashboardMetrics:
    """Dashboard-specific performance metrics"""
    response_time_ms: float
    render_time_ms: float
    update_latency_ms: float
    memory_usage_mb: float
    cpu_usage_percent: float
    concurrent_users: int
    operation_type: str
    timestamp: str
    success: bool
    error_message: Optional[str] = None

class DashboardPerformanceTester:
    """
    Comprehensive performance testing suite for interactive dashboard
    components with focus on real-time responsiveness
    """
    
    def __init__(self):
        self.metrics = []
        self.baseline_performance = {}
        self.stress_test_results = {}
        
    def test_real_time_model_fitting(self, data_sizes: List[int], 
                                   concurrent_users: List[int] = [1, 5, 10]) -> Dict[str, Any]:
        """Test real-time model fitting performance under different loads"""
        results = {}
        
        for size in data_sizes:
            for users in concurrent_users:
                test_name = f"model_fitting_size_{size}_users_{users}"
                
                # Generate test data
                data = self._generate_dashboard_test_data(size)
                
                # Run concurrent model fitting simulation
                metrics = self._run_concurrent_operation(
                    operation_func=self._simulate_model_fitting,
                    operation_args=(data,),
                    num_concurrent=users,
                    operation_name=test_name
                )
                
                results[test_name] = {
                    'mean_response_time': np.mean([m.response_time_ms for m in metrics]),
                    'p95_response_time': np.percentile([m.response_time_ms for m in metrics], 95),
                    'success_rate': sum(m.success for m in metrics) / len(metrics),
                    'memory_usage': np.mean([m.memory_usage_mb for m in metrics]),
                    'detailed_metrics': metrics
                }
                
        return results
        
    def test_interactive_parameter_adjustment(self, parameter_ranges: Dict[str, List],
                                            update_frequencies: List[float] = [0.1, 0.5, 1.0]) -> Dict[str, Any]:
        """Test performance of interactive parameter adjustment"""
        results = {}
        
        for freq in update_frequencies:
            test_name = f"parameter_adjustment_freq_{freq}Hz"
            
            # Simulate rapid parameter updates
            metrics = []
            start_time = time.time()
            
            while time.time() - start_time < 10:  # 10-second test
                for param_name, param_values in parameter_ranges.items():
                    new_value = np.random.choice(param_values)
                    
                    metric = self._simulate_parameter_update(param_name, new_value)
                    metrics.append(metric)
                    
                    # Wait for next update based on frequency
                    time.sleep(1.0 / freq)
                    
            results[test_name] = {
                'mean_update_latency': np.mean([m.update_latency_ms for m in metrics]),
                'max_update_latency': np.max([m.update_latency_ms for m in metrics]),
                'updates_per_second': len(metrics) / 10,
                'memory_stability': self._check_memory_stability(metrics),
                'detailed_metrics': metrics
            }
            
        return results
        
    def test_visualization_rendering(self, figure_complexities: List[str],
                                   concurrent_renders: List[int] = [1, 3, 5]) -> Dict[str, Any]:
        """Test visualization rendering performance"""
        results = {}
        
        complexity_configs = {
            'simple': {'n_series': 1, 'n_points': 100, 'n_subplots': 1},
            'medium': {'n_series': 3, 'n_points': 1000, 'n_subplots': 2},
            'complex': {'n_series': 5, 'n_points': 5000, 'n_subplots': 4}
        }
        
        for complexity in figure_complexities:
            if complexity not in complexity_configs:
                continue
                
            config = complexity_configs[complexity]
            
            for concurrent in concurrent_renders:
                test_name = f"visualization_{complexity}_concurrent_{concurrent}"
                
                # Run concurrent visualization rendering
                metrics = self._run_concurrent_operation(
                    operation_func=self._simulate_visualization_rendering,
                    operation_args=(config,),
                    num_concurrent=concurrent,
                    operation_name=test_name
                )
                
                results[test_name] = {
                    'mean_render_time': np.mean([m.render_time_ms for m in metrics]),
                    'p95_render_time': np.percentile([m.render_time_ms for m in metrics], 95),
                    'success_rate': sum(m.success for m in metrics) / len(metrics),
                    'memory_peak': np.max([m.memory_usage_mb for m in metrics]),
                    'detailed_metrics': metrics
                }
                
        return results
        
    def test_data_export_performance(self, export_formats: List[str],
                                   data_sizes: List[int]) -> Dict[str, Any]:
        """Test data export performance for different formats and sizes"""
        results = {}
        
        for format_type in export_formats:
            for size in data_sizes:
                test_name = f"export_{format_type}_size_{size}"
                
                # Generate export data
                export_data = self._generate_export_test_data(size)
                
                # Measure export performance
                start_time = time.time()
                memory_before = psutil.Process().memory_info().rss / 1024 / 1024
                
                try:
                    success = self._simulate_data_export(export_data, format_type)
                    response_time = (time.time() - start_time) * 1000  # ms
                    memory_after = psutil.Process().memory_info().rss / 1024 / 1024
                    
                    metric = DashboardMetrics(
                        response_time_ms=response_time,
                        render_time_ms=0,  # Not applicable for export
                        update_latency_ms=0,  # Not applicable for export
                        memory_usage_mb=memory_after - memory_before,
                        cpu_usage_percent=psutil.Process().cpu_percent(),
                        concurrent_users=1,
                        operation_type=test_name,
                        timestamp=pd.Timestamp.now().isoformat(),
                        success=success
                    )
                    
                except Exception as e:
                    metric = DashboardMetrics(
                        response_time_ms=float('inf'),
                        render_time_ms=0,
                        update_latency_ms=0,
                        memory_usage_mb=0,
                        cpu_usage_percent=0,
                        concurrent_users=1,
                        operation_type=test_name,
                        timestamp=pd.Timestamp.now().isoformat(),
                        success=False,
                        error_message=str(e)
                    )
                
                results[test_name] = {
                    'response_time_ms': metric.response_time_ms,
                    'memory_usage_mb': metric.memory_usage_mb,
                    'success': metric.success,
                    'error_message': metric.error_message
                }
                
        return results
        
    def stress_test_dashboard(self, duration_seconds: int = 60,
                            max_concurrent_users: int = 20) -> Dict[str, Any]:
        """Run comprehensive stress test on dashboard components"""
        print(f"Starting {duration_seconds}-second stress test with up to {max_concurrent_users} concurrent users...")
        
        stress_results = {
            'start_time': pd.Timestamp.now().isoformat(),
            'duration_seconds': duration_seconds,
            'max_concurrent_users': max_concurrent_users,
            'operations_completed': 0,
            'errors_encountered': 0,
            'performance_degradation': {},
            'resource_usage': []
        }
        
        # Define stress test operations
        operations = [
            ('model_fitting', self._simulate_model_fitting),
            ('parameter_update', self._simulate_parameter_update),
            ('visualization_render', self._simulate_visualization_rendering),
            ('data_export', self._simulate_data_export)
        ]
        
        start_time = time.time()
        
        with ThreadPoolExecutor(max_workers=max_concurrent_users) as executor:
            futures = []
            
            while time.time() - start_time < duration_seconds:
                # Gradually increase load
                current_load = min(
                    max_concurrent_users,
                    int((time.time() - start_time) / duration_seconds * max_concurrent_users) + 1
                )
                
                # Submit operations up to current load
                while len(futures) < current_load:
                    op_name, op_func = np.random.choice(operations, 1)[0]
                    
                    if op_name == 'model_fitting':
                        future = executor.submit(op_func, self._generate_dashboard_test_data(1000))
                    elif op_name == 'parameter_update':
                        future = executor.submit(op_func, 'threshold', 0.5)
                    elif op_name == 'visualization_render':
                        future = executor.submit(op_func, {'n_series': 2, 'n_points': 500, 'n_subplots': 1})
                    else:  # data_export
                        future = executor.submit(op_func, self._generate_export_test_data(100), 'csv')
                        
                    futures.append((future, op_name))
                
                # Check completed operations
                completed_futures = []
                for future, op_name in futures:
                    if future.done():
                        try:
                            result = future.result()
                            stress_results['operations_completed'] += 1
                        except Exception as e:
                            stress_results['errors_encountered'] += 1
                            print(f"Error in {op_name}: {e}")
                        completed_futures.append((future, op_name))
                
                # Remove completed futures
                for completed in completed_futures:
                    futures.remove(completed)
                
                # Record resource usage
                memory_usage = psutil.Process().memory_info().rss / 1024 / 1024
                cpu_usage = psutil.Process().cpu_percent()
                
                stress_results['resource_usage'].append({
                    'timestamp': time.time() - start_time,
                    'memory_mb': memory_usage,
                    'cpu_percent': cpu_usage,
                    'active_operations': len(futures)
                })
                
                time.sleep(0.1)  # Brief pause
                
        # Calculate performance degradation
        if stress_results['resource_usage']:
            resource_df = pd.DataFrame(stress_results['resource_usage'])
            
            stress_results['performance_degradation'] = {
                'memory_growth_rate': self._calculate_growth_rate(resource_df['memory_mb']),
                'cpu_utilization_mean': resource_df['cpu_percent'].mean(),
                'cpu_utilization_max': resource_df['cpu_percent'].max(),
                'peak_concurrent_operations': resource_df['active_operations'].max()
            }
            
        return stress_results
        
    def _run_concurrent_operation(self, operation_func: Callable, operation_args: tuple,
                                num_concurrent: int, operation_name: str) -> List[DashboardMetrics]:
        """Run operation concurrently and collect metrics"""
        metrics = []
        
        with ThreadPoolExecutor(max_workers=num_concurrent) as executor:
            # Submit all operations
            futures = [
                executor.submit(self._measure_operation, operation_func, operation_args, 
                              operation_name, i)
                for i in range(num_concurrent)
            ]
            
            # Collect results
            for future in as_completed(futures):
                try:
                    metric = future.result()
                    metrics.append(metric)
                except Exception as e:
                    # Create error metric
                    error_metric = DashboardMetrics(
                        response_time_ms=float('inf'),
                        render_time_ms=0,
                        update_latency_ms=0,
                        memory_usage_mb=0,
                        cpu_usage_percent=0,
                        concurrent_users=num_concurrent,
                        operation_type=operation_name,
                        timestamp=pd.Timestamp.now().isoformat(),
                        success=False,
                        error_message=str(e)
                    )
                    metrics.append(error_metric)
                    
        return metrics
        
    def _measure_operation(self, operation_func: Callable, operation_args: tuple,
                          operation_name: str, user_id: int) -> DashboardMetrics:
        """Measure performance of a single operation"""
        start_time = time.time()
        memory_before = psutil.Process().memory_info().rss / 1024 / 1024
        
        try:
            # Execute operation
            result = operation_func(*operation_args)
            
            end_time = time.time()
            memory_after = psutil.Process().memory_info().rss / 1024 / 1024
            
            return DashboardMetrics(
                response_time_ms=(end_time - start_time) * 1000,
                render_time_ms=getattr(result, 'render_time_ms', 0),
                update_latency_ms=getattr(result, 'update_latency_ms', 0),
                memory_usage_mb=memory_after - memory_before,
                cpu_usage_percent=psutil.Process().cpu_percent(),
                concurrent_users=1,  # Will be updated by caller
                operation_type=operation_name,
                timestamp=pd.Timestamp.now().isoformat(),
                success=True
            )
            
        except Exception as e:
            return DashboardMetrics(
                response_time_ms=float('inf'),
                render_time_ms=0,
                update_latency_ms=0,
                memory_usage_mb=0,
                cpu_usage_percent=0,
                concurrent_users=1,
                operation_type=operation_name,
                timestamp=pd.Timestamp.now().isoformat(),
                success=False,
                error_message=str(e)
            )
            
    def _simulate_model_fitting(self, data: Dict[str, np.ndarray]) -> Any:
        """Simulate model fitting operation"""
        # Simulate computational work
        X = data['X']
        y = data['y']
        
        # Simple linear regression simulation
        XtX = X.T @ X
        Xty = X.T @ y
        
        # Add regularization to ensure invertibility
        reg_matrix = np.eye(X.shape[1]) * 0.01
        coefficients = np.linalg.solve(XtX + reg_matrix, Xty)
        
        # Simulate additional processing time
        time.sleep(np.random.uniform(0.01, 0.05))
        
        return {'coefficients': coefficients, 'fitted': True}
        
    def _simulate_parameter_update(self, param_name: str, param_value: Any) -> DashboardMetrics:
        """Simulate parameter update operation"""
        start_time = time.time()
        
        # Simulate parameter validation and update
        time.sleep(np.random.uniform(0.001, 0.01))
        
        # Simulate model re-fitting with new parameter
        data = self._generate_dashboard_test_data(500)
        self._simulate_model_fitting(data)
        
        update_time = (time.time() - start_time) * 1000
        
        return DashboardMetrics(
            response_time_ms=update_time,
            render_time_ms=0,
            update_latency_ms=update_time,
            memory_usage_mb=psutil.Process().memory_info().rss / 1024 / 1024,
            cpu_usage_percent=psutil.Process().cpu_percent(),
            concurrent_users=1,
            operation_type='parameter_update',
            timestamp=pd.Timestamp.now().isoformat(),
            success=True
        )
        
    def _simulate_visualization_rendering(self, config: Dict[str, int]) -> Any:
        """Simulate visualization rendering"""
        start_time = time.time()
        
        # Create figure based on configuration
        fig, axes = plt.subplots(1, config['n_subplots'], figsize=(12, 6))
        if config['n_subplots'] == 1:
            axes = [axes]
            
        for i, ax in enumerate(axes):
            for series in range(config['n_series']):
                x = np.linspace(0, 10, config['n_points'])
                y = np.sin(x + series) + np.random.randn(config['n_points']) * 0.1
                ax.plot(x, y, label=f'Series {series}')
            ax.legend()
            ax.set_title(f'Subplot {i+1}')
            
        plt.tight_layout()
        
        # Simulate rendering time
        render_time = (time.time() - start_time) * 1000
        
        plt.close(fig)  # Clean up
        
        return {'render_time_ms': render_time, 'rendered': True}
        
    def _simulate_data_export(self, data: pd.DataFrame, format_type: str) -> bool:
        """Simulate data export operation"""
        try:
            if format_type == 'csv':
                # Simulate CSV export
                csv_string = data.to_csv()
            elif format_type == 'excel':
                # Simulate Excel export (more computationally intensive)
                import io
                buffer = io.BytesIO()
                data.to_excel(buffer, index=False)
            elif format_type == 'json':
                # Simulate JSON export
                json_string = data.to_json()
            else:
                raise ValueError(f"Unsupported format: {format_type}")
                
            # Simulate file I/O delay
            time.sleep(np.random.uniform(0.01, 0.1))
            return True
            
        except Exception:
            return False
            
    def _generate_dashboard_test_data(self, size: int) -> Dict[str, np.ndarray]:
        """Generate test data for dashboard operations"""
        np.random.seed(42)
        
        n_features = min(10, max(3, size // 100))
        X = np.random.randn(size, n_features)
        y = np.random.randn(size)
        threshold_var = np.random.randn(size)
        
        return {'X': X, 'y': y, 'threshold_var': threshold_var}
        
    def _generate_export_test_data(self, size: int) -> pd.DataFrame:
        """Generate test data for export operations"""
        np.random.seed(42)
        
        return pd.DataFrame({
            'variable_1': np.random.randn(size),
            'variable_2': np.random.randn(size),
            'variable_3': np.random.randint(0, 100, size),
            'timestamp': pd.date_range('2020-01-01', periods=size, freq='D')
        })
        
    def _check_memory_stability(self, metrics: List[DashboardMetrics]) -> Dict[str, float]:
        """Check memory usage stability over time"""
        memory_values = [m.memory_usage_mb for m in metrics]
        
        return {
            'mean_memory': np.mean(memory_values),
            'std_memory': np.std(memory_values),
            'memory_trend': self._calculate_growth_rate(memory_values),
            'memory_stability_score': 1.0 / (1.0 + np.std(memory_values))
        }
        
    def _calculate_growth_rate(self, values: List[float]) -> float:
        """Calculate growth rate of a time series"""
        if len(values) < 2:
            return 0.0
            
        x = np.arange(len(values))
        coeffs = np.polyfit(x, values, 1)
        return coeffs[0]  # Slope represents growth rate
        
    def generate_dashboard_performance_report(self) -> str:
        """Generate comprehensive dashboard performance report"""
        if not self.metrics:
            return "No dashboard performance data available."
            
        report = ["Dashboard Performance Report", "=" * 50, ""]
        
        # Overall statistics
        response_times = [m.response_time_ms for m in self.metrics if m.success]
        success_rate = sum(m.success for m in self.metrics) / len(self.metrics)
        
        report.extend([
            f"Total Operations: {len(self.metrics)}",
            f"Success Rate: {success_rate:.2%}",
            f"Mean Response Time: {np.mean(response_times):.2f} ms",
            f"95th Percentile Response Time: {np.percentile(response_times, 95):.2f} ms",
            f"Max Response Time: {np.max(response_times):.2f} ms",
            ""
        ])
        
        # Performance by operation type
        operations = {}
        for metric in self.metrics:
            if metric.operation_type not in operations:
                operations[metric.operation_type] = []
            operations[metric.operation_type].append(metric)
            
        for op_name, op_metrics in operations.items():
            successful_ops = [m for m in op_metrics if m.success]
            if successful_ops:
                report.extend([
                    f"{op_name.upper()}:",
                    f"  Operations: {len(op_metrics)}",
                    f"  Success Rate: {len(successful_ops)/len(op_metrics):.2%}",
                    f"  Mean Response Time: {np.mean([m.response_time_ms for m in successful_ops]):.2f} ms",
                    f"  Memory Usage: {np.mean([m.memory_usage_mb for m in successful_ops]):.2f} MB",
                    ""
                ])
                
        return "\n".join(report)

if __name__ == "__main__":
    # Example usage
    dashboard_tester = DashboardPerformanceTester()
    
    print("Running dashboard performance tests...")
    
    # Test real-time model fitting
    model_results = dashboard_tester.test_real_time_model_fitting(
        data_sizes=[100, 500, 1000],
        concurrent_users=[1, 3, 5]
    )
    
    # Test interactive parameter adjustment
    param_results = dashboard_tester.test_interactive_parameter_adjustment(
        parameter_ranges={'threshold': [0.1, 0.5, 1.0], 'lambda': [0.01, 0.1, 1.0]},
        update_frequencies=[0.5, 1.0, 2.0]
    )
    
    # Test visualization rendering
    viz_results = dashboard_tester.test_visualization_rendering(
        figure_complexities=['simple', 'medium', 'complex'],
        concurrent_renders=[1, 2, 3]
    )
    
    # Run stress test
    stress_results = dashboard_tester.stress_test_dashboard(
        duration_seconds=30,
        max_concurrent_users=10
    )
    
    print("Dashboard performance testing completed.")
    print(f"Stress test completed {stress_results['operations_completed']} operations "
          f"with {stress_results['errors_encountered']} errors.")