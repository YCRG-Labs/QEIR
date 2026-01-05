"""
Comprehensive Data Validation and Quality Checking System

This module provides advanced data validation, outlier detection, temporal alignment checks,
gap handling, and comprehensive data quality reporting for economic time series data.

Author: QE Research Team
Date: 2025
Version: 1.0
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional, Tuple, Any, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import warnings
from scipy import stats

# Optional dependencies for advanced outlier detection
try:
    from sklearn.preprocessing import StandardScaler
    from sklearn.ensemble import IsolationForest
    from sklearn.cluster import DBSCAN
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


class ValidationSeverity(Enum):
    """Severity levels for validation issues"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class OutlierMethod(Enum):
    """Methods for outlier detection"""
    IQR = "iqr"
    Z_SCORE = "z_score"
    MODIFIED_Z_SCORE = "modified_z_score"
    ISOLATION_FOREST = "isolation_forest"
    DBSCAN = "dbscan"


@dataclass
class ValidationIssue:
    """Represents a data validation issue"""
    severity: ValidationSeverity
    category: str
    message: str
    affected_dates: Optional[List[datetime]] = None
    affected_values: Optional[List[float]] = None
    recommendation: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class QualityMetrics:
    """Comprehensive quality metrics for a time series"""
    series_name: str
    total_observations: int
    missing_count: int
    missing_percentage: float
    outlier_count: int
    outlier_percentage: float
    duplicate_count: int
    negative_count: int
    zero_count: int
    infinite_count: int
    quality_score: float
    completeness_score: float
    consistency_score: float
    accuracy_score: float
    date_range: Dict[str, datetime]
    frequency_analysis: Dict[str, Any]
    statistical_summary: Dict[str, float]
    issues: List[ValidationIssue]


@dataclass
class TemporalAlignment:
    """Results of temporal alignment analysis"""
    common_start_date: datetime
    common_end_date: datetime
    frequency: str
    alignment_issues: List[ValidationIssue]
    gap_analysis: Dict[str, Any]
    frequency_mismatches: List[str]
    recommended_alignment: Dict[str, Any]


class AdvancedDataValidator:
    """
    Advanced data validation and quality checking system
    """
    
    def __init__(self, 
                 outlier_threshold: float = 3.0,
                 missing_threshold: float = 10.0,
                 quality_threshold: float = 70.0):
        """
        Initialize the advanced data validator
        
        Args:
            outlier_threshold: Threshold for outlier detection (standard deviations)
            missing_threshold: Threshold for missing data percentage warnings
            quality_threshold: Minimum quality score threshold
        """
        self.outlier_threshold = outlier_threshold
        self.missing_threshold = missing_threshold
        self.quality_threshold = quality_threshold
        
        self.logger = logging.getLogger(__name__)
        
        # Initialize outlier detectors if sklearn is available
        if SKLEARN_AVAILABLE:
            self.isolation_forest = IsolationForest(contamination=0.1, random_state=42)
            self.dbscan = DBSCAN(eps=0.5, min_samples=5)
        else:
            self.isolation_forest = None
            self.dbscan = None
    
    def detect_outliers(self, series: pd.Series, method: OutlierMethod = OutlierMethod.IQR) -> Dict[str, Any]:
        """
        Detect outliers using various methods
        
        Args:
            series: Time series to analyze
            method: Outlier detection method
            
        Returns:
            Dictionary with outlier detection results
        """
        if series.empty or series.isna().all():
            return {
                'outlier_indices': [],
                'outlier_values': [],
                'outlier_dates': [],
                'method': method.value,
                'threshold_used': None,
                'outlier_count': 0
            }
        
        # Remove NaN values for analysis
        clean_series = series.dropna()
        
        if len(clean_series) < 10:  # Need minimum data for outlier detection
            return {
                'outlier_indices': [],
                'outlier_values': [],
                'outlier_dates': [],
                'method': method.value,
                'threshold_used': None,
                'outlier_count': 0,
                'warning': 'Insufficient data for reliable outlier detection'
            }
        
        outlier_mask = pd.Series(False, index=clean_series.index)
        threshold_used = None
        
        if method == OutlierMethod.IQR:
            Q1 = clean_series.quantile(0.25)
            Q3 = clean_series.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            outlier_mask = (clean_series < lower_bound) | (clean_series > upper_bound)
            threshold_used = {'lower': lower_bound, 'upper': upper_bound, 'IQR': IQR}
            
        elif method == OutlierMethod.Z_SCORE:
            z_scores = np.abs(stats.zscore(clean_series))
            outlier_mask = z_scores > self.outlier_threshold
            threshold_used = self.outlier_threshold
            
        elif method == OutlierMethod.MODIFIED_Z_SCORE:
            median = clean_series.median()
            mad = np.median(np.abs(clean_series - median))
            modified_z_scores = 0.6745 * (clean_series - median) / mad
            outlier_mask = np.abs(modified_z_scores) > self.outlier_threshold
            threshold_used = self.outlier_threshold
            
        elif method == OutlierMethod.ISOLATION_FOREST:
            if not SKLEARN_AVAILABLE or self.isolation_forest is None:
                self.logger.warning("sklearn not available, falling back to IQR method")
                return self.detect_outliers(series, OutlierMethod.IQR)
            try:
                # Reshape for sklearn
                X = clean_series.values.reshape(-1, 1)
                outlier_predictions = self.isolation_forest.fit_predict(X)
                outlier_mask = pd.Series(outlier_predictions == -1, index=clean_series.index)
                threshold_used = 'isolation_forest_contamination_0.1'
            except Exception as e:
                self.logger.warning(f"Isolation Forest failed: {e}")
                return self.detect_outliers(series, OutlierMethod.IQR)  # Fallback to IQR
                
        elif method == OutlierMethod.DBSCAN:
            if not SKLEARN_AVAILABLE or self.dbscan is None:
                self.logger.warning("sklearn not available, falling back to IQR method")
                return self.detect_outliers(series, OutlierMethod.IQR)
            try:
                # Reshape for sklearn
                X = clean_series.values.reshape(-1, 1)
                cluster_labels = self.dbscan.fit_predict(X)
                outlier_mask = pd.Series(cluster_labels == -1, index=clean_series.index)
                threshold_used = 'dbscan_eps_0.5_min_samples_5'
            except Exception as e:
                self.logger.warning(f"DBSCAN failed: {e}")
                return self.detect_outliers(series, OutlierMethod.IQR)  # Fallback to IQR
        
        # Extract outlier information
        outlier_indices = outlier_mask[outlier_mask].index.tolist()
        outlier_values = clean_series[outlier_mask].tolist()
        outlier_dates = outlier_indices if outlier_indices and hasattr(outlier_indices[0], 'date') else []
        
        return {
            'outlier_indices': outlier_indices,
            'outlier_values': outlier_values,
            'outlier_dates': outlier_dates,
            'method': method.value,
            'threshold_used': threshold_used,
            'outlier_count': len(outlier_indices),
            'outlier_percentage': (len(outlier_indices) / len(clean_series)) * 100
        }
    
    def analyze_missing_data(self, series: pd.Series) -> Dict[str, Any]:
        """
        Comprehensive missing data analysis
        
        Args:
            series: Time series to analyze
            
        Returns:
            Dictionary with missing data analysis results
        """
        total_obs = len(series)
        missing_count = series.isna().sum()
        missing_percentage = (missing_count / total_obs) * 100 if total_obs > 0 else 0
        
        # Analyze missing data patterns
        missing_mask = series.isna()
        
        # Find consecutive missing periods
        missing_runs = []
        if missing_mask.any():
            # Group consecutive missing values
            missing_groups = (missing_mask != missing_mask.shift()).cumsum()
            for group_id, group in missing_mask.groupby(missing_groups):
                if group.iloc[0]:  # This group represents missing values
                    start_idx = group.index[0]
                    end_idx = group.index[-1]
                    missing_runs.append({
                        'start': start_idx,
                        'end': end_idx,
                        'length': len(group),
                        'start_date': start_idx if hasattr(start_idx, 'date') else None,
                        'end_date': end_idx if hasattr(end_idx, 'date') else None
                    })
        
        # Analyze missing data distribution
        missing_at_start = 0
        missing_at_end = 0
        missing_in_middle = 0
        
        if missing_mask.any():
            # Count missing at start
            for val in missing_mask:
                if val:
                    missing_at_start += 1
                else:
                    break
            
            # Count missing at end
            for val in reversed(missing_mask):
                if val:
                    missing_at_end += 1
                else:
                    break
            
            # Missing in middle
            missing_in_middle = missing_count - missing_at_start - missing_at_end
        
        return {
            'total_observations': total_obs,
            'missing_count': missing_count,
            'missing_percentage': missing_percentage,
            'missing_runs': missing_runs,
            'longest_missing_run': max([run['length'] for run in missing_runs]) if missing_runs else 0,
            'missing_at_start': missing_at_start,
            'missing_at_end': missing_at_end,
            'missing_in_middle': missing_in_middle,
            'has_missing_data': missing_count > 0,
            'missing_pattern': self._classify_missing_pattern(missing_at_start, missing_at_end, missing_in_middle, total_obs)
        }
    
    def _classify_missing_pattern(self, start: int, end: int, middle: int, total: int) -> str:
        """Classify the pattern of missing data"""
        if start + end + middle == 0:
            return 'complete'
        elif middle == 0 and start > 0 and end > 0:
            return 'edge_missing'
        elif start == 0 and end == 0:
            return 'middle_gaps'
        elif start > 0 and end == 0 and middle == 0:
            return 'start_missing'
        elif start == 0 and end > 0 and middle == 0:
            return 'end_missing'
        else:
            return 'mixed_pattern'
    
    def analyze_temporal_consistency(self, series: pd.Series) -> Dict[str, Any]:
        """
        Analyze temporal consistency and frequency patterns
        
        Args:
            series: Time series to analyze
            
        Returns:
            Dictionary with temporal consistency analysis
        """
        if not hasattr(series.index, 'freq') and not isinstance(series.index, pd.DatetimeIndex):
            return {
                'has_datetime_index': False,
                'error': 'Series does not have a datetime index'
            }
        
        # Analyze frequency
        try:
            inferred_freq = pd.infer_freq(series.index)
        except:
            inferred_freq = None
        
        # Calculate time differences
        time_diffs = series.index.to_series().diff().dropna()
        
        # Analyze gaps
        if len(time_diffs) > 0:
            median_diff = time_diffs.median()
            mode_diff = time_diffs.mode().iloc[0] if not time_diffs.mode().empty else median_diff
            
            # Find irregular gaps (more than 2x the typical interval)
            irregular_gaps = time_diffs[time_diffs > 2 * mode_diff]
            
            # Find duplicated timestamps
            duplicated_timestamps = series.index.duplicated().sum()
            
            return {
                'has_datetime_index': True,
                'inferred_frequency': inferred_freq,
                'median_time_diff': median_diff,
                'mode_time_diff': mode_diff,
                'irregular_gaps': len(irregular_gaps),
                'irregular_gap_dates': irregular_gaps.index.tolist(),
                'duplicated_timestamps': duplicated_timestamps,
                'is_regular_frequency': len(irregular_gaps) == 0 and duplicated_timestamps == 0,
                'frequency_consistency_score': max(0, 100 - (len(irregular_gaps) / len(time_diffs)) * 100)
            }
        else:
            return {
                'has_datetime_index': True,
                'inferred_frequency': None,
                'error': 'Insufficient data for temporal analysis'
            }
    
    def validate_series_comprehensive(self, series: pd.Series, series_name: str) -> QualityMetrics:
        """
        Comprehensive validation of a single time series
        
        Args:
            series: Time series to validate
            series_name: Name of the series
            
        Returns:
            QualityMetrics object with comprehensive validation results
        """
        issues = []
        
        if series is None or series.empty:
            return QualityMetrics(
                series_name=series_name,
                total_observations=0,
                missing_count=0,
                missing_percentage=100.0,
                outlier_count=0,
                outlier_percentage=0.0,
                duplicate_count=0,
                negative_count=0,
                zero_count=0,
                infinite_count=0,
                quality_score=0.0,
                completeness_score=0.0,
                consistency_score=0.0,
                accuracy_score=0.0,
                date_range={},
                frequency_analysis={},
                statistical_summary={},
                issues=[ValidationIssue(
                    ValidationSeverity.CRITICAL,
                    "data_availability",
                    "Series is empty or None",
                    recommendation="Check data source and collection process"
                )]
            )
        
        # Basic statistics
        total_obs = len(series)
        missing_count = series.isna().sum()
        missing_percentage = (missing_count / total_obs) * 100 if total_obs > 0 else 100
        
        # Count special values
        numeric_series = pd.to_numeric(series, errors='coerce')
        negative_count = (numeric_series < 0).sum() if not numeric_series.isna().all() else 0
        zero_count = (numeric_series == 0).sum() if not numeric_series.isna().all() else 0
        infinite_count = np.isinf(numeric_series).sum() if not numeric_series.isna().all() else 0
        duplicate_count = series.duplicated().sum()
        
        # Missing data analysis
        missing_analysis = self.analyze_missing_data(series)
        
        # Outlier detection
        outlier_analysis = self.detect_outliers(series, OutlierMethod.IQR)
        outlier_count = outlier_analysis['outlier_count']
        outlier_percentage = outlier_analysis.get('outlier_percentage', 0.0)
        
        # Temporal consistency
        temporal_analysis = self.analyze_temporal_consistency(series)
        
        # Statistical summary
        try:
            clean_series = numeric_series.dropna()
            if len(clean_series) > 0:
                statistical_summary = {
                    'mean': float(clean_series.mean()),
                    'median': float(clean_series.median()),
                    'std': float(clean_series.std()),
                    'min': float(clean_series.min()),
                    'max': float(clean_series.max()),
                    'skewness': float(clean_series.skew()),
                    'kurtosis': float(clean_series.kurtosis())
                }
            else:
                statistical_summary = {}
        except:
            statistical_summary = {}
        
        # Date range analysis
        date_range = {}
        if hasattr(series.index, 'min') and hasattr(series.index, 'max'):
            try:
                date_range = {
                    'start': series.index.min(),
                    'end': series.index.max(),
                    'span_days': (series.index.max() - series.index.min()).days
                }
            except:
                pass
        
        # Generate validation issues
        if missing_percentage > self.missing_threshold:
            issues.append(ValidationIssue(
                ValidationSeverity.WARNING if missing_percentage < 25 else ValidationSeverity.ERROR,
                "missing_data",
                f"High missing data percentage: {missing_percentage:.1f}%",
                recommendation="Consider interpolation, alternative data sources, or exclude from analysis"
            ))
        
        if outlier_percentage > 5.0:
            issues.append(ValidationIssue(
                ValidationSeverity.WARNING,
                "outliers",
                f"High outlier percentage: {outlier_percentage:.1f}%",
                affected_dates=outlier_analysis.get('outlier_dates', []),
                affected_values=outlier_analysis.get('outlier_values', []),
                recommendation="Review outliers and consider winsorization or robust methods"
            ))
        
        if infinite_count > 0:
            issues.append(ValidationIssue(
                ValidationSeverity.ERROR,
                "infinite_values",
                f"Infinite values detected: {infinite_count}",
                recommendation="Replace infinite values with NaN or appropriate bounds"
            ))
        
        if duplicate_count > 0:
            issues.append(ValidationIssue(
                ValidationSeverity.WARNING,
                "duplicates",
                f"Duplicate values detected: {duplicate_count}",
                recommendation="Review for data collection errors or natural duplicates"
            ))
        
        if not temporal_analysis.get('is_regular_frequency', True):
            issues.append(ValidationIssue(
                ValidationSeverity.WARNING,
                "temporal_irregularity",
                f"Irregular time frequency detected: {temporal_analysis.get('irregular_gaps', 0)} gaps",
                recommendation="Consider resampling or interpolation for regular frequency"
            ))
        
        # Calculate quality scores
        completeness_score = max(0, 100 - missing_percentage)
        consistency_score = temporal_analysis.get('frequency_consistency_score', 100)
        accuracy_score = max(0, 100 - outlier_percentage - (infinite_count / total_obs * 100))
        
        # Overall quality score (weighted average)
        quality_score = (
            completeness_score * 0.4 +
            consistency_score * 0.3 +
            accuracy_score * 0.3
        )
        
        return QualityMetrics(
            series_name=series_name,
            total_observations=total_obs,
            missing_count=missing_count,
            missing_percentage=missing_percentage,
            outlier_count=outlier_count,
            outlier_percentage=outlier_percentage,
            duplicate_count=duplicate_count,
            negative_count=negative_count,
            zero_count=zero_count,
            infinite_count=infinite_count,
            quality_score=quality_score,
            completeness_score=completeness_score,
            consistency_score=consistency_score,
            accuracy_score=accuracy_score,
            date_range=date_range,
            frequency_analysis=temporal_analysis,
            statistical_summary=statistical_summary,
            issues=issues
        )
    
    def analyze_temporal_alignment(self, data: Dict[str, pd.Series]) -> TemporalAlignment:
        """
        Analyze temporal alignment across multiple series
        
        Args:
            data: Dictionary of time series
            
        Returns:
            TemporalAlignment object with alignment analysis
        """
        if not data:
            return TemporalAlignment(
                common_start_date=datetime.now(),
                common_end_date=datetime.now(),
                frequency="unknown",
                alignment_issues=[],
                gap_analysis={},
                frequency_mismatches=[],
                recommended_alignment={}
            )
        
        issues = []
        
        # Collect all date ranges and frequencies
        date_ranges = {}
        frequencies = {}
        
        for series_name, series in data.items():
            if hasattr(series.index, 'min') and hasattr(series.index, 'max'):
                try:
                    date_ranges[series_name] = {
                        'start': series.index.min(),
                        'end': series.index.max()
                    }
                    frequencies[series_name] = pd.infer_freq(series.index)
                except:
                    pass
        
        if not date_ranges:
            issues.append(ValidationIssue(
                ValidationSeverity.ERROR,
                "temporal_alignment",
                "No valid datetime indices found in any series",
                recommendation="Ensure all series have proper datetime indices"
            ))
            
            return TemporalAlignment(
                common_start_date=datetime.now(),
                common_end_date=datetime.now(),
                frequency="unknown",
                alignment_issues=issues,
                gap_analysis={},
                frequency_mismatches=[],
                recommended_alignment={}
            )
        
        # Find common date range
        all_starts = [dr['start'] for dr in date_ranges.values()]
        all_ends = [dr['end'] for dr in date_ranges.values()]
        
        common_start = max(all_starts)
        common_end = min(all_ends)
        
        # Check for frequency mismatches
        unique_frequencies = set(freq for freq in frequencies.values() if freq is not None)
        frequency_mismatches = []
        
        if len(unique_frequencies) > 1:
            frequency_mismatches = [
                f"{name}: {freq}" for name, freq in frequencies.items() if freq is not None
            ]
            issues.append(ValidationIssue(
                ValidationSeverity.WARNING,
                "frequency_mismatch",
                f"Multiple frequencies detected: {unique_frequencies}",
                recommendation="Resample series to common frequency before analysis"
            ))
        
        # Determine most common frequency
        freq_counts = {}
        for freq in frequencies.values():
            if freq is not None:
                freq_counts[freq] = freq_counts.get(freq, 0) + 1
        
        most_common_freq = max(freq_counts.keys(), key=freq_counts.get) if freq_counts else "unknown"
        
        # Analyze gaps in common period
        gap_analysis = {}
        if common_start <= common_end:
            for series_name, series in data.items():
                # Filter to common period
                common_period_series = series[(series.index >= common_start) & (series.index <= common_end)]
                missing_analysis = self.analyze_missing_data(common_period_series)
                gap_analysis[series_name] = missing_analysis
        
        # Generate recommendations
        recommended_alignment = {
            'start_date': common_start,
            'end_date': common_end,
            'frequency': most_common_freq,
            'resampling_needed': len(unique_frequencies) > 1,
            'interpolation_needed': any(
                gap['missing_percentage'] > 0 for gap in gap_analysis.values()
            )
        }
        
        return TemporalAlignment(
            common_start_date=common_start,
            common_end_date=common_end,
            frequency=most_common_freq,
            alignment_issues=issues,
            gap_analysis=gap_analysis,
            frequency_mismatches=frequency_mismatches,
            recommended_alignment=recommended_alignment
        )
    
    def generate_quality_report(self, data: Dict[str, pd.Series]) -> Dict[str, Any]:
        """
        Generate comprehensive data quality report
        
        Args:
            data: Dictionary of time series to analyze
            
        Returns:
            Comprehensive quality report dictionary
        """
        self.logger.info("Generating comprehensive data quality report...")
        
        # Validate each series
        series_metrics = {}
        for series_name, series in data.items():
            metrics = self.validate_series_comprehensive(series, series_name)
            series_metrics[series_name] = metrics
        
        # Analyze temporal alignment
        alignment_analysis = self.analyze_temporal_alignment(data)
        
        # Calculate overall statistics
        total_series = len(data)
        high_quality_series = sum(1 for m in series_metrics.values() if m.quality_score >= 80)
        medium_quality_series = sum(1 for m in series_metrics.values() if 50 <= m.quality_score < 80)
        low_quality_series = sum(1 for m in series_metrics.values() if m.quality_score < 50)
        
        overall_quality = np.mean([m.quality_score for m in series_metrics.values()]) if series_metrics else 0
        
        # Collect all issues by severity
        all_issues = []
        for metrics in series_metrics.values():
            all_issues.extend(metrics.issues)
        all_issues.extend(alignment_analysis.alignment_issues)
        
        issues_by_severity = {
            'critical': [i for i in all_issues if i.severity == ValidationSeverity.CRITICAL],
            'error': [i for i in all_issues if i.severity == ValidationSeverity.ERROR],
            'warning': [i for i in all_issues if i.severity == ValidationSeverity.WARNING],
            'info': [i for i in all_issues if i.severity == ValidationSeverity.INFO]
        }
        
        # Generate summary
        summary = {
            'total_series': total_series,
            'high_quality_series': high_quality_series,
            'medium_quality_series': medium_quality_series,
            'low_quality_series': low_quality_series,
            'overall_quality_score': overall_quality,
            'total_issues': len(all_issues),
            'critical_issues': len(issues_by_severity['critical']),
            'error_issues': len(issues_by_severity['error']),
            'warning_issues': len(issues_by_severity['warning']),
            'info_issues': len(issues_by_severity['info']),
            'report_timestamp': datetime.now().isoformat()
        }
        
        self.logger.info(f"Quality report generated: {total_series} series, {overall_quality:.1f} overall quality score")
        
        return {
            'summary': summary,
            'series_metrics': {name: metrics.__dict__ for name, metrics in series_metrics.items()},
            'temporal_alignment': alignment_analysis.__dict__,
            'issues_by_severity': {
                severity: [issue.__dict__ for issue in issues] 
                for severity, issues in issues_by_severity.items()
            },
            'recommendations': self._generate_recommendations(series_metrics, alignment_analysis)
        }
    
    def _generate_recommendations(self, series_metrics: Dict[str, QualityMetrics], 
                                alignment: TemporalAlignment) -> List[str]:
        """Generate actionable recommendations based on quality analysis"""
        recommendations = []
        
        # Series-specific recommendations
        low_quality_series = [name for name, metrics in series_metrics.items() if metrics.quality_score < 50]
        if low_quality_series:
            recommendations.append(
                f"Review low quality series: {', '.join(low_quality_series[:5])}{'...' if len(low_quality_series) > 5 else ''}"
            )
        
        # Missing data recommendations
        high_missing_series = [name for name, metrics in series_metrics.items() if metrics.missing_percentage > 20]
        if high_missing_series:
            recommendations.append(
                f"Consider interpolation or alternative sources for series with high missing data: {', '.join(high_missing_series[:3])}"
            )
        
        # Outlier recommendations
        high_outlier_series = [name for name, metrics in series_metrics.items() if metrics.outlier_percentage > 10]
        if high_outlier_series:
            recommendations.append(
                f"Review outliers in: {', '.join(high_outlier_series[:3])} - consider winsorization or robust methods"
            )
        
        # Temporal alignment recommendations
        if alignment.frequency_mismatches:
            recommendations.append(
                f"Resample series to common frequency ({alignment.frequency}) for consistent analysis"
            )
        
        if alignment.recommended_alignment.get('interpolation_needed', False):
            recommendations.append(
                "Apply interpolation methods to fill gaps in common analysis period"
            )
        
        return recommendations