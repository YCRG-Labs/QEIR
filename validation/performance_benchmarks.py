"""
Performance Benchmarking and Optimization Framework

This module provides comprehensive performance testing and optimization
capabilities for the publication technical outputs system, ensuring
scalable performance across different dataset sizes and computational
requirements.
"""

import time
import psutil
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional, Callable
import matplotlib.pyplot as plt
import seaborn as sns
from dataclasses import dataclass
from contextlib import contextmanager
import gc
import tracemalloc
import warnings
warnings.filterwarnings('ignore')

@dataclass
class PerformanceMetrics:
    """Container for performance measurement results"""
    execution_time: float
    memory_usage_mb: float
    peak_memory_mb: float
    cpu_usage_percent: float
    dataset_size: int
    operation_name: str
    timestamp: str
    
class PerformanceBenchmarkSuite:
    """
    Comprehensive performance benchmarking suite for econometric models
    and publication system components
    """
    
    def __init__(self):
        self.results = []
        self.baseline_metrics = {}
        
    @contextmanager
    def measure_performance(self, operation_name: str, dataset_size: int = 0):
        """Context manager for measuring performance metrics"""
        # Start monitoring
        tracemalloc.start()
        process = psutil.Process()
        start_memory = process.memory_info().rss / 1024 / 1024  # MB
        start_cpu = process.cpu_percent()
        start_time = time.time()
        
        try:
            yield
        finally:
            # Collect metrics
            end_time = time.time()
            end_memory = process.memory_info().rss / 1024 / 1024  # MB
            end_cpu = process.cpu_percent()
            
            current, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            
            metrics = PerformanceMetrics(
                execution_time=end_time - start_time,
                memory_usage_mb=end_memory - start_memory,
                peak_memory_mb=peak / 1024 / 1024,  # Convert to MB
                cpu_usage_percent=(start_cpu + end_cpu) / 2,
                dataset_size=dataset_size,
                operation_name=operation_name,
                timestamp=pd.Timestamp.now().isoformat()
            )
            
            self.results.append(metrics)
            
    def benchmark_model_fitting(self, model_class, data_sizes: List[int]) -> Dict[str, Any]:
        """Benchmark model fitting performance across different dataset sizes"""
        results = {}
        
        for size in data_sizes:
            # Generate synthetic data
            np.random.seed(42)
            data = self._generate_synthetic_data(size)
            
            with self.measure_performance(f"{model_class.__name__}_fit", size):
                try:
                    model = model_class()
                    if hasattr(model, 'fit'):
                        model.fit(data['y'], data['X'], data.get('threshold_var'))
                    elif hasattr(model, 'estimate'):
                        model.estimate(data['y'], data['X'], data.get('threshold_var'))
                except Exception as e:
                    print(f"Error fitting {model_class.__name__} with size {size}: {e}")
                    
        return self._aggregate_results_by_operation(f"{model_class.__name__}_fit")
        
    def benchmark_diagnostic_computation(self, diagnostic_class, data_sizes: List[int]) -> Dict[str, Any]:
        """Benchmark diagnostic computation performance"""
        results = {}
        
        for size in data_sizes:
            data = self._generate_synthetic_data(size)
            
            with self.measure_performance(f"{diagnostic_class.__name__}_compute", size):
                try:
                    diagnostics = diagnostic_class()
                    if hasattr(diagnostics, 'compute_diagnostics'):
                        diagnostics.compute_diagnostics(data['y'], data['X'])
                    elif hasattr(diagnostics, 'diagnose_low_r_squared'):
                        # Mock model for diagnostics
                        class MockModel:
                            def __init__(self):
                                self.r_squared = 0.001
                                self.coefficients = np.random.randn(data['X'].shape[1])
                                self.residuals = np.random.randn(len(data['y']))
                        
                        mock_model = MockModel()
                        diagnostics.diagnose_low_r_squared(mock_model, data, data.get('threshold_var'))
                except Exception as e:
                    print(f"Error computing diagnostics with size {size}: {e}")
                    
        return self._aggregate_results_by_operation(f"{diagnostic_class.__name__}_compute")
        
    def benchmark_visualization_generation(self, viz_class, data_sizes: List[int]) -> Dict[str, Any]:
        """Benchmark visualization generation performance"""
        results = {}
        
        for size in data_sizes:
            data = self._generate_synthetic_data(size)
            
            with self.measure_performance(f"{viz_class.__name__}_generate", size):
                try:
                    viz = viz_class()
                    if hasattr(viz, 'create_threshold_analysis_figure'):
                        # Mock model results for visualization
                        mock_results = {
                            'threshold_value': 0.5,
                            'coefficients': np.random.randn(2, data['X'].shape[1]),
                            'confidence_intervals': np.random.randn(2, data['X'].shape[1], 2),
                            'fitted_values': np.random.randn(len(data['y']))
                        }
                        viz.create_threshold_analysis_figure(mock_results, data)
                    plt.close('all')  # Clean up figures
                except Exception as e:
                    print(f"Error generating visualization with size {size}: {e}")
                    
        return self._aggregate_results_by_operation(f"{viz_class.__name__}_generate")
        
    def benchmark_batch_processing(self, batch_sizes: List[int], operations_per_batch: int = 10) -> Dict[str, Any]:
        """Benchmark batch processing performance for publication workflows"""
        results = {}
        
        for batch_size in batch_sizes:
            total_operations = batch_size * operations_per_batch
            
            with self.measure_performance(f"batch_processing_{batch_size}", total_operations):
                try:
                    # Simulate batch processing workflow
                    for i in range(batch_size):
                        # Simulate model fitting
                        data = self._generate_synthetic_data(1000)
                        
                        # Simulate multiple operations per batch item
                        for j in range(operations_per_batch):
                            # Simple computation to simulate work
                            result = np.linalg.inv(data['X'].T @ data['X'] + np.eye(data['X'].shape[1]) * 0.01)
                            
                        # Force garbage collection periodically
                        if i % 10 == 0:
                            gc.collect()
                            
                except Exception as e:
                    print(f"Error in batch processing with batch size {batch_size}: {e}")
                    
        return self._aggregate_results_by_operation("batch_processing")
        
    def _generate_synthetic_data(self, size: int) -> Dict[str, np.ndarray]:
        """Generate synthetic data for benchmarking"""
        np.random.seed(42)  # Ensure reproducibility
        
        # Generate features
        n_features = min(10, max(3, size // 100))  # Scale features with data size
        X = np.random.randn(size, n_features)
        
        # Generate threshold variable
        threshold_var = np.random.randn(size)
        
        # Generate dependent variable with some relationship
        beta = np.random.randn(n_features)
        y = X @ beta + 0.5 * threshold_var + np.random.randn(size) * 0.1
        
        return {
            'y': y,
            'X': X,
            'threshold_var': threshold_var
        }
        
    def _aggregate_results_by_operation(self, operation_prefix: str) -> Dict[str, Any]:
        """Aggregate performance results by operation type"""
        operation_results = [r for r in self.results if r.operation_name.startswith(operation_prefix)]
        
        if not operation_results:
            return {}
            
        df = pd.DataFrame([
            {
                'dataset_size': r.dataset_size,
                'execution_time': r.execution_time,
                'memory_usage_mb': r.memory_usage_mb,
                'peak_memory_mb': r.peak_memory_mb,
                'cpu_usage_percent': r.cpu_usage_percent
            }
            for r in operation_results
        ])
        
        return {
            'mean_execution_time': df['execution_time'].mean(),
            'std_execution_time': df['execution_time'].std(),
            'mean_memory_usage': df['memory_usage_mb'].mean(),
            'peak_memory_usage': df['peak_memory_mb'].max(),
            'scaling_factor': self._compute_scaling_factor(df),
            'efficiency_score': self._compute_efficiency_score(df),
            'detailed_results': df
        }
        
    def _compute_scaling_factor(self, df: pd.DataFrame) -> float:
        """Compute how execution time scales with dataset size"""
        if len(df) < 2:
            return 1.0
            
        # Fit log-log relationship: log(time) = a + b * log(size)
        log_size = np.log(df['dataset_size'].replace(0, 1))  # Avoid log(0)
        log_time = np.log(df['execution_time'].replace(0, 1e-6))  # Avoid log(0)
        
        if len(log_size.unique()) > 1:
            coeffs = np.polyfit(log_size, log_time, 1)
            return coeffs[0]  # Scaling exponent
        return 1.0
        
    def _compute_efficiency_score(self, df: pd.DataFrame) -> float:
        """Compute efficiency score (lower is better)"""
        if len(df) == 0:
            return float('inf')
            
        # Normalize by dataset size and compute composite score
        normalized_time = df['execution_time'] / (df['dataset_size'] + 1)
        normalized_memory = df['memory_usage_mb'] / (df['dataset_size'] + 1)
        
        return (normalized_time.mean() + normalized_memory.mean() / 100).item()
        
    def generate_performance_report(self) -> str:
        """Generate comprehensive performance report"""
        if not self.results:
            return "No performance data available."
            
        report = ["Performance Benchmarking Report", "=" * 40, ""]
        
        # Group results by operation
        operations = {}
        for result in self.results:
            op_base = result.operation_name.split('_')[0]
            if op_base not in operations:
                operations[op_base] = []
            operations[op_base].append(result)
            
        for op_name, op_results in operations.items():
            report.append(f"\n{op_name.upper()} Performance:")
            report.append("-" * 30)
            
            df = pd.DataFrame([
                {
                    'Size': r.dataset_size,
                    'Time (s)': f"{r.execution_time:.3f}",
                    'Memory (MB)': f"{r.memory_usage_mb:.1f}",
                    'Peak Memory (MB)': f"{r.peak_memory_mb:.1f}",
                    'CPU (%)': f"{r.cpu_usage_percent:.1f}"
                }
                for r in op_results
            ])
            
            report.append(df.to_string(index=False))
            
        return "\n".join(report)
        
    def create_performance_visualizations(self, save_path: Optional[str] = None):
        """Create performance visualization plots"""
        if not self.results:
            print("No performance data to visualize.")
            return
            
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Performance Benchmarking Results', fontsize=16)
        
        # Convert results to DataFrame
        df = pd.DataFrame([
            {
                'operation': r.operation_name,
                'dataset_size': r.dataset_size,
                'execution_time': r.execution_time,
                'memory_usage_mb': r.memory_usage_mb,
                'peak_memory_mb': r.peak_memory_mb,
                'cpu_usage_percent': r.cpu_usage_percent
            }
            for r in self.results
        ])
        
        # Execution time vs dataset size
        for op in df['operation'].unique():
            op_data = df[df['operation'] == op]
            if len(op_data) > 1:
                axes[0, 0].loglog(op_data['dataset_size'], op_data['execution_time'], 
                                 'o-', label=op, alpha=0.7)
        axes[0, 0].set_xlabel('Dataset Size')
        axes[0, 0].set_ylabel('Execution Time (s)')
        axes[0, 0].set_title('Execution Time Scaling')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Memory usage vs dataset size
        for op in df['operation'].unique():
            op_data = df[df['operation'] == op]
            if len(op_data) > 1:
                axes[0, 1].loglog(op_data['dataset_size'], op_data['memory_usage_mb'], 
                                 's-', label=op, alpha=0.7)
        axes[0, 1].set_xlabel('Dataset Size')
        axes[0, 1].set_ylabel('Memory Usage (MB)')
        axes[0, 1].set_title('Memory Usage Scaling')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Performance efficiency heatmap
        pivot_data = df.pivot_table(
            values='execution_time', 
            index='operation', 
            columns='dataset_size', 
            aggfunc='mean'
        )
        if not pivot_data.empty:
            sns.heatmap(pivot_data, ax=axes[1, 0], cmap='YlOrRd', 
                       cbar_kws={'label': 'Execution Time (s)'})
            axes[1, 0].set_title('Performance Heatmap')
        
        # CPU usage distribution
        df.boxplot(column='cpu_usage_percent', by='operation', ax=axes[1, 1])
        axes[1, 1].set_title('CPU Usage Distribution by Operation')
        axes[1, 1].set_xlabel('Operation')
        axes[1, 1].set_ylabel('CPU Usage (%)')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Performance visualizations saved to {save_path}")
        else:
            plt.show()
            
        plt.close()

class MemoryOptimizer:
    """Memory usage optimization utilities for large dataset handling"""
    
    @staticmethod
    def optimize_dataframe_memory(df: pd.DataFrame) -> pd.DataFrame:
        """Optimize DataFrame memory usage by downcasting numeric types"""
        optimized_df = df.copy()
        
        for col in optimized_df.columns:
            col_type = optimized_df[col].dtype
            
            if col_type != 'object':
                c_min = optimized_df[col].min()
                c_max = optimized_df[col].max()
                
                if str(col_type)[:3] == 'int':
                    if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                        optimized_df[col] = optimized_df[col].astype(np.int8)
                    elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                        optimized_df[col] = optimized_df[col].astype(np.int16)
                    elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                        optimized_df[col] = optimized_df[col].astype(np.int32)
                        
                elif str(col_type)[:5] == 'float':
                    if c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                        optimized_df[col] = optimized_df[col].astype(np.float32)
                        
        return optimized_df
        
    @staticmethod
    def chunked_processing(data: pd.DataFrame, chunk_size: int, 
                          processing_func: Callable, **kwargs) -> List[Any]:
        """Process large datasets in chunks to manage memory usage"""
        results = []
        n_chunks = len(data) // chunk_size + (1 if len(data) % chunk_size > 0 else 0)
        
        for i in range(n_chunks):
            start_idx = i * chunk_size
            end_idx = min((i + 1) * chunk_size, len(data))
            chunk = data.iloc[start_idx:end_idx]
            
            try:
                result = processing_func(chunk, **kwargs)
                results.append(result)
            except Exception as e:
                print(f"Error processing chunk {i+1}/{n_chunks}: {e}")
                
            # Force garbage collection after each chunk
            gc.collect()
            
        return results
        
    @staticmethod
    @contextmanager
    def memory_monitor(threshold_mb: float = 1000):
        """Context manager to monitor and warn about high memory usage"""
        process = psutil.Process()
        start_memory = process.memory_info().rss / 1024 / 1024
        
        try:
            yield
        finally:
            end_memory = process.memory_info().rss / 1024 / 1024
            memory_increase = end_memory - start_memory
            
            if memory_increase > threshold_mb:
                print(f"Warning: Memory usage increased by {memory_increase:.1f} MB "
                      f"(threshold: {threshold_mb} MB)")
                
    @staticmethod
    def get_memory_usage_report() -> Dict[str, float]:
        """Get current memory usage statistics"""
        process = psutil.Process()
        memory_info = process.memory_info()
        
        return {
            'rss_mb': memory_info.rss / 1024 / 1024,
            'vms_mb': memory_info.vms / 1024 / 1024,
            'percent': process.memory_percent(),
            'available_mb': psutil.virtual_memory().available / 1024 / 1024,
            'total_mb': psutil.virtual_memory().total / 1024 / 1024
        }

if __name__ == "__main__":
    # Example usage
    benchmark_suite = PerformanceBenchmarkSuite()
    
    # Test with different data sizes
    test_sizes = [100, 500, 1000, 5000]
    
    print("Running performance benchmarks...")
    
    # Benchmark batch processing
    batch_results = benchmark_suite.benchmark_batch_processing([1, 5, 10, 20])
    
    # Generate report
    report = benchmark_suite.generate_performance_report()
    print(report)
    
    # Create visualizations
    benchmark_suite.create_performance_visualizations('performance_benchmark_results.png')
    
    # Memory optimization example
    print("\nMemory optimization example:")
    memory_report = MemoryOptimizer.get_memory_usage_report()
    print(f"Current memory usage: {memory_report['rss_mb']:.1f} MB "
          f"({memory_report['percent']:.1f}% of total)")