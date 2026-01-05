"""
Batch Processing Monitoring and Reporting Utilities

This module provides monitoring, reporting, and notification capabilities
for batch processing operations.

Author: QE Research Team
Date: 2025
Version: 1.0
"""

import json
import logging
import time
import smtplib
import requests
import psutil
import threading
from pathlib import Path
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import sqlite3
import pandas as pd


@dataclass
class SystemMetrics:
    """System resource metrics."""
    
    timestamp: str
    cpu_percent: float
    memory_percent: float
    disk_usage_percent: float
    available_memory_gb: float
    disk_free_gb: float
    active_processes: int


@dataclass
class JobMetrics:
    """Job execution metrics."""
    
    job_id: str
    job_name: str
    status: str
    start_time: Optional[str]
    end_time: Optional[str]
    duration_seconds: Optional[float]
    cpu_time_seconds: Optional[float]
    memory_peak_mb: Optional[float]
    disk_io_mb: Optional[float]
    error_count: int
    warning_count: int


@dataclass
class BatchMetrics:
    """Overall batch processing metrics."""
    
    batch_id: str
    start_time: str
    end_time: Optional[str]
    total_jobs: int
    completed_jobs: int
    failed_jobs: int
    cancelled_jobs: int
    average_job_duration: Optional[float]
    total_processing_time: Optional[float]
    throughput_jobs_per_hour: Optional[float]
    system_metrics: List[SystemMetrics] = field(default_factory=list)
    job_metrics: List[JobMetrics] = field(default_factory=list)


class BatchMonitor:
    """
    Monitoring system for batch processing operations.
    
    Tracks system resources, job performance, and provides
    real-time monitoring and alerting capabilities.
    """
    
    def __init__(self, 
                 monitoring_interval: int = 30,
                 enable_system_monitoring: bool = True,
                 enable_job_monitoring: bool = True,
                 enable_notifications: bool = False,
                 database_path: Optional[str] = None):
        """
        Initialize batch monitor.
        
        Args:
            monitoring_interval: Seconds between monitoring checks
            enable_system_monitoring: Enable system resource monitoring
            enable_job_monitoring: Enable individual job monitoring
            enable_notifications: Enable alert notifications
            database_path: Optional SQLite database path for metrics storage
        """
        self.monitoring_interval = monitoring_interval
        self.enable_system_monitoring = enable_system_monitoring
        self.enable_job_monitoring = enable_job_monitoring
        self.enable_notifications = enable_notifications
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        
        # Monitoring state
        self.is_monitoring = False
        self.monitoring_thread: Optional[threading.Thread] = None
        
        # Metrics storage
        self.current_batch_metrics: Optional[BatchMetrics] = None
        self.system_metrics_history: List[SystemMetrics] = []
        self.job_metrics_history: List[JobMetrics] = []
        
        # Database setup
        self.database_path = database_path
        if self.database_path:
            self._setup_database()
        
        # Notification settings
        self.notification_handlers: List[Callable] = []
        
        # Alert thresholds
        self.alert_thresholds = {
            'cpu_percent': 90.0,
            'memory_percent': 85.0,
            'disk_usage_percent': 90.0,
            'job_failure_rate': 0.2,
            'job_timeout_minutes': 180
        }
        
        self.logger.info("BatchMonitor initialized")
    
    def start_monitoring(self, batch_id: str):
        """Start monitoring for a batch processing session."""
        if self.is_monitoring:
            self.logger.warning("Monitoring already active")
            return
        
        self.is_monitoring = True
        self.current_batch_metrics = BatchMetrics(
            batch_id=batch_id,
            start_time=datetime.now().isoformat(),
            end_time=None,
            total_jobs=0,
            completed_jobs=0,
            failed_jobs=0,
            cancelled_jobs=0,
            average_job_duration=None,
            total_processing_time=None,
            throughput_jobs_per_hour=None
        )
        
        # Start monitoring thread
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitoring_thread.start()
        
        self.logger.info(f"Started monitoring for batch: {batch_id}")
    
    def stop_monitoring(self):
        """Stop monitoring and finalize metrics."""
        if not self.is_monitoring:
            return
        
        self.is_monitoring = False
        
        if self.current_batch_metrics:
            self.current_batch_metrics.end_time = datetime.now().isoformat()
            self._finalize_batch_metrics()
        
        self.logger.info("Stopped batch monitoring")
    
    def update_job_status(self, 
                         job_id: str, 
                         job_name: str, 
                         status: str,
                         metrics: Optional[Dict[str, Any]] = None):
        """Update job status and metrics."""
        if not self.enable_job_monitoring or not self.current_batch_metrics:
            return
        
        # Find existing job metrics or create new
        job_metric = None
        for jm in self.job_metrics_history:
            if jm.job_id == job_id:
                job_metric = jm
                break
        
        if not job_metric:
            job_metric = JobMetrics(
                job_id=job_id,
                job_name=job_name,
                status=status,
                start_time=None,
                end_time=None,
                duration_seconds=None,
                cpu_time_seconds=None,
                memory_peak_mb=None,
                disk_io_mb=None,
                error_count=0,
                warning_count=0
            )
            self.job_metrics_history.append(job_metric)
        
        # Update job metrics
        job_metric.status = status
        
        if status == "running" and not job_metric.start_time:
            job_metric.start_time = datetime.now().isoformat()
            self.current_batch_metrics.total_jobs += 1
        
        elif status in ["completed", "failed", "cancelled"]:
            if not job_metric.end_time:
                job_metric.end_time = datetime.now().isoformat()
                
                # Calculate duration
                if job_metric.start_time:
                    start_dt = datetime.fromisoformat(job_metric.start_time)
                    end_dt = datetime.fromisoformat(job_metric.end_time)
                    job_metric.duration_seconds = (end_dt - start_dt).total_seconds()
                
                # Update batch counters
                if status == "completed":
                    self.current_batch_metrics.completed_jobs += 1
                elif status == "failed":
                    self.current_batch_metrics.failed_jobs += 1
                elif status == "cancelled":
                    self.current_batch_metrics.cancelled_jobs += 1
        
        # Update additional metrics if provided
        if metrics:
            if 'cpu_time_seconds' in metrics:
                job_metric.cpu_time_seconds = metrics['cpu_time_seconds']
            if 'memory_peak_mb' in metrics:
                job_metric.memory_peak_mb = metrics['memory_peak_mb']
            if 'disk_io_mb' in metrics:
                job_metric.disk_io_mb = metrics['disk_io_mb']
            if 'error_count' in metrics:
                job_metric.error_count = metrics['error_count']
            if 'warning_count' in metrics:
                job_metric.warning_count = metrics['warning_count']
        
        # Store in database if enabled
        if self.database_path:
            self._store_job_metrics(job_metric)
        
        # Check for alerts
        self._check_job_alerts(job_metric)
        
        self.logger.debug(f"Updated job {job_id} status to {status}")
    
    def _monitoring_loop(self):
        """Main monitoring loop."""
        while self.is_monitoring:
            try:
                if self.enable_system_monitoring:
                    self._collect_system_metrics()
                
                # Check for system alerts
                self._check_system_alerts()
                
                time.sleep(self.monitoring_interval)
                
            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {e}")
                time.sleep(self.monitoring_interval)
    
    def _collect_system_metrics(self):
        """Collect current system metrics."""
        try:
            # Get system metrics
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            metrics = SystemMetrics(
                timestamp=datetime.now().isoformat(),
                cpu_percent=cpu_percent,
                memory_percent=memory.percent,
                disk_usage_percent=disk.percent,
                available_memory_gb=memory.available / (1024**3),
                disk_free_gb=disk.free / (1024**3),
                active_processes=len(psutil.pids())
            )
            
            self.system_metrics_history.append(metrics)
            
            # Store in database if enabled
            if self.database_path:
                self._store_system_metrics(metrics)
            
            # Limit history size
            if len(self.system_metrics_history) > 1000:
                self.system_metrics_history = self.system_metrics_history[-500:]
            
        except Exception as e:
            self.logger.error(f"Error collecting system metrics: {e}")
    
    def _check_system_alerts(self):
        """Check for system resource alerts."""
        if not self.system_metrics_history:
            return
        
        latest_metrics = self.system_metrics_history[-1]
        alerts = []
        
        # CPU alert
        if latest_metrics.cpu_percent > self.alert_thresholds['cpu_percent']:
            alerts.append(f"High CPU usage: {latest_metrics.cpu_percent:.1f}%")
        
        # Memory alert
        if latest_metrics.memory_percent > self.alert_thresholds['memory_percent']:
            alerts.append(f"High memory usage: {latest_metrics.memory_percent:.1f}%")
        
        # Disk alert
        if latest_metrics.disk_usage_percent > self.alert_thresholds['disk_usage_percent']:
            alerts.append(f"High disk usage: {latest_metrics.disk_usage_percent:.1f}%")
        
        # Send alerts if any
        if alerts:
            alert_message = f"System Resource Alert:\n" + "\n".join(alerts)
            self._send_notification("System Alert", alert_message)
    
    def _check_job_alerts(self, job_metric: JobMetrics):
        """Check for job-specific alerts."""
        alerts = []
        
        # Job timeout alert
        if (job_metric.status == "running" and 
            job_metric.start_time and 
            job_metric.duration_seconds and
            job_metric.duration_seconds > self.alert_thresholds['job_timeout_minutes'] * 60):
            alerts.append(f"Job {job_metric.job_id} has been running for {job_metric.duration_seconds/60:.1f} minutes")
        
        # Job failure rate alert
        if self.current_batch_metrics:
            total_finished = (self.current_batch_metrics.completed_jobs + 
                            self.current_batch_metrics.failed_jobs)
            if total_finished > 5:  # Only check after some jobs are done
                failure_rate = self.current_batch_metrics.failed_jobs / total_finished
                if failure_rate > self.alert_thresholds['job_failure_rate']:
                    alerts.append(f"High job failure rate: {failure_rate:.1%}")
        
        # Send alerts if any
        if alerts:
            alert_message = f"Job Alert:\n" + "\n".join(alerts)
            self._send_notification("Job Alert", alert_message)
    
    def _finalize_batch_metrics(self):
        """Finalize batch metrics when monitoring stops."""
        if not self.current_batch_metrics:
            return
        
        # Calculate summary statistics
        completed_jobs = [jm for jm in self.job_metrics_history if jm.status == "completed"]
        
        if completed_jobs:
            durations = [jm.duration_seconds for jm in completed_jobs if jm.duration_seconds]
            if durations:
                self.current_batch_metrics.average_job_duration = sum(durations) / len(durations)
        
        # Calculate total processing time
        if self.current_batch_metrics.start_time and self.current_batch_metrics.end_time:
            start_dt = datetime.fromisoformat(self.current_batch_metrics.start_time)
            end_dt = datetime.fromisoformat(self.current_batch_metrics.end_time)
            self.current_batch_metrics.total_processing_time = (end_dt - start_dt).total_seconds()
            
            # Calculate throughput
            if self.current_batch_metrics.total_processing_time > 0:
                hours = self.current_batch_metrics.total_processing_time / 3600
                self.current_batch_metrics.throughput_jobs_per_hour = (
                    self.current_batch_metrics.completed_jobs / hours
                )
        
        # Store final metrics
        if self.database_path:
            self._store_batch_metrics(self.current_batch_metrics)
        
        self.logger.info("Finalized batch metrics")
    
    def _setup_database(self):
        """Setup SQLite database for metrics storage."""
        try:
            conn = sqlite3.connect(self.database_path)
            cursor = conn.cursor()
            
            # Create tables
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS batch_metrics (
                    batch_id TEXT PRIMARY KEY,
                    start_time TEXT,
                    end_time TEXT,
                    total_jobs INTEGER,
                    completed_jobs INTEGER,
                    failed_jobs INTEGER,
                    cancelled_jobs INTEGER,
                    average_job_duration REAL,
                    total_processing_time REAL,
                    throughput_jobs_per_hour REAL
                )
            ''')
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS job_metrics (
                    job_id TEXT,
                    batch_id TEXT,
                    job_name TEXT,
                    status TEXT,
                    start_time TEXT,
                    end_time TEXT,
                    duration_seconds REAL,
                    cpu_time_seconds REAL,
                    memory_peak_mb REAL,
                    disk_io_mb REAL,
                    error_count INTEGER,
                    warning_count INTEGER,
                    PRIMARY KEY (job_id, batch_id)
                )
            ''')
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS system_metrics (
                    timestamp TEXT,
                    batch_id TEXT,
                    cpu_percent REAL,
                    memory_percent REAL,
                    disk_usage_percent REAL,
                    available_memory_gb REAL,
                    disk_free_gb REAL,
                    active_processes INTEGER,
                    PRIMARY KEY (timestamp, batch_id)
                )
            ''')
            
            conn.commit()
            conn.close()
            
            self.logger.info(f"Database setup complete: {self.database_path}")
            
        except Exception as e:
            self.logger.error(f"Error setting up database: {e}")
    
    def _store_batch_metrics(self, metrics: BatchMetrics):
        """Store batch metrics in database."""
        try:
            conn = sqlite3.connect(self.database_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT OR REPLACE INTO batch_metrics VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                metrics.batch_id,
                metrics.start_time,
                metrics.end_time,
                metrics.total_jobs,
                metrics.completed_jobs,
                metrics.failed_jobs,
                metrics.cancelled_jobs,
                metrics.average_job_duration,
                metrics.total_processing_time,
                metrics.throughput_jobs_per_hour
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            self.logger.error(f"Error storing batch metrics: {e}")
    
    def _store_job_metrics(self, metrics: JobMetrics):
        """Store job metrics in database."""
        try:
            conn = sqlite3.connect(self.database_path)
            cursor = conn.cursor()
            
            batch_id = self.current_batch_metrics.batch_id if self.current_batch_metrics else "unknown"
            
            cursor.execute('''
                INSERT OR REPLACE INTO job_metrics VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                metrics.job_id,
                batch_id,
                metrics.job_name,
                metrics.status,
                metrics.start_time,
                metrics.end_time,
                metrics.duration_seconds,
                metrics.cpu_time_seconds,
                metrics.memory_peak_mb,
                metrics.disk_io_mb,
                metrics.error_count,
                metrics.warning_count
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            self.logger.error(f"Error storing job metrics: {e}")
    
    def _store_system_metrics(self, metrics: SystemMetrics):
        """Store system metrics in database."""
        try:
            conn = sqlite3.connect(self.database_path)
            cursor = conn.cursor()
            
            batch_id = self.current_batch_metrics.batch_id if self.current_batch_metrics else "unknown"
            
            cursor.execute('''
                INSERT OR REPLACE INTO system_metrics VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                metrics.timestamp,
                batch_id,
                metrics.cpu_percent,
                metrics.memory_percent,
                metrics.disk_usage_percent,
                metrics.available_memory_gb,
                metrics.disk_free_gb,
                metrics.active_processes
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            self.logger.error(f"Error storing system metrics: {e}")
    
    def add_notification_handler(self, handler: Callable[[str, str], None]):
        """Add a notification handler function."""
        self.notification_handlers.append(handler)
    
    def _send_notification(self, subject: str, message: str):
        """Send notification using registered handlers."""
        if not self.enable_notifications:
            return
        
        for handler in self.notification_handlers:
            try:
                handler(subject, message)
            except Exception as e:
                self.logger.error(f"Error sending notification: {e}")
    
    def get_current_status(self) -> Dict[str, Any]:
        """Get current monitoring status."""
        status = {
            'is_monitoring': self.is_monitoring,
            'batch_metrics': None,
            'latest_system_metrics': None,
            'active_jobs': 0,
            'recent_job_metrics': []
        }
        
        if self.current_batch_metrics:
            status['batch_metrics'] = {
                'batch_id': self.current_batch_metrics.batch_id,
                'start_time': self.current_batch_metrics.start_time,
                'total_jobs': self.current_batch_metrics.total_jobs,
                'completed_jobs': self.current_batch_metrics.completed_jobs,
                'failed_jobs': self.current_batch_metrics.failed_jobs,
                'cancelled_jobs': self.current_batch_metrics.cancelled_jobs
            }
        
        if self.system_metrics_history:
            latest = self.system_metrics_history[-1]
            status['latest_system_metrics'] = {
                'timestamp': latest.timestamp,
                'cpu_percent': latest.cpu_percent,
                'memory_percent': latest.memory_percent,
                'disk_usage_percent': latest.disk_usage_percent,
                'available_memory_gb': latest.available_memory_gb
            }
        
        # Count active jobs
        active_jobs = [jm for jm in self.job_metrics_history if jm.status == "running"]
        status['active_jobs'] = len(active_jobs)
        
        # Recent job metrics (last 10)
        status['recent_job_metrics'] = [
            {
                'job_id': jm.job_id,
                'job_name': jm.job_name,
                'status': jm.status,
                'duration_seconds': jm.duration_seconds
            }
            for jm in self.job_metrics_history[-10:]
        ]
        
        return status
    
    def generate_report(self, output_file: str):
        """Generate comprehensive monitoring report."""
        if not self.current_batch_metrics:
            self.logger.warning("No batch metrics available for report")
            return
        
        report_data = {
            'batch_summary': {
                'batch_id': self.current_batch_metrics.batch_id,
                'start_time': self.current_batch_metrics.start_time,
                'end_time': self.current_batch_metrics.end_time,
                'total_jobs': self.current_batch_metrics.total_jobs,
                'completed_jobs': self.current_batch_metrics.completed_jobs,
                'failed_jobs': self.current_batch_metrics.failed_jobs,
                'cancelled_jobs': self.current_batch_metrics.cancelled_jobs,
                'success_rate': (self.current_batch_metrics.completed_jobs / 
                               max(1, self.current_batch_metrics.total_jobs)),
                'average_job_duration': self.current_batch_metrics.average_job_duration,
                'total_processing_time': self.current_batch_metrics.total_processing_time,
                'throughput_jobs_per_hour': self.current_batch_metrics.throughput_jobs_per_hour
            },
            'job_details': [
                {
                    'job_id': jm.job_id,
                    'job_name': jm.job_name,
                    'status': jm.status,
                    'start_time': jm.start_time,
                    'end_time': jm.end_time,
                    'duration_seconds': jm.duration_seconds,
                    'cpu_time_seconds': jm.cpu_time_seconds,
                    'memory_peak_mb': jm.memory_peak_mb,
                    'error_count': jm.error_count,
                    'warning_count': jm.warning_count
                }
                for jm in self.job_metrics_history
            ],
            'system_metrics_summary': self._summarize_system_metrics(),
            'report_generated': datetime.now().isoformat()
        }
        
        # Save report
        with open(output_file, 'w') as f:
            json.dump(report_data, f, indent=2, default=str)
        
        self.logger.info(f"Monitoring report generated: {output_file}")
    
    def _summarize_system_metrics(self) -> Dict[str, Any]:
        """Summarize system metrics over the monitoring period."""
        if not self.system_metrics_history:
            return {}
        
        cpu_values = [sm.cpu_percent for sm in self.system_metrics_history]
        memory_values = [sm.memory_percent for sm in self.system_metrics_history]
        disk_values = [sm.disk_usage_percent for sm in self.system_metrics_history]
        
        return {
            'cpu_percent': {
                'min': min(cpu_values),
                'max': max(cpu_values),
                'avg': sum(cpu_values) / len(cpu_values)
            },
            'memory_percent': {
                'min': min(memory_values),
                'max': max(memory_values),
                'avg': sum(memory_values) / len(memory_values)
            },
            'disk_usage_percent': {
                'min': min(disk_values),
                'max': max(disk_values),
                'avg': sum(disk_values) / len(disk_values)
            },
            'monitoring_duration_minutes': len(self.system_metrics_history) * (self.monitoring_interval / 60),
            'data_points': len(self.system_metrics_history)
        }


# Notification handlers
def email_notification_handler(smtp_server: str, 
                              smtp_port: int,
                              username: str, 
                              password: str,
                              to_email: str,
                              from_email: str) -> Callable[[str, str], None]:
    """Create email notification handler."""
    
    def send_email(subject: str, message: str):
        try:
            msg = MIMEMultipart()
            msg['From'] = from_email
            msg['To'] = to_email
            msg['Subject'] = f"QEIR Batch Processing: {subject}"
            
            msg.attach(MIMEText(message, 'plain'))
            
            server = smtplib.SMTP(smtp_server, smtp_port)
            server.starttls()
            server.login(username, password)
            server.send_message(msg)
            server.quit()
            
        except Exception as e:
            logging.error(f"Error sending email notification: {e}")
    
    return send_email


def webhook_notification_handler(webhook_url: str) -> Callable[[str, str], None]:
    """Create webhook notification handler."""
    
    def send_webhook(subject: str, message: str):
        try:
            payload = {
                'subject': subject,
                'message': message,
                'timestamp': datetime.now().isoformat(),
                'source': 'QEIR Batch Processing'
            }
            
            response = requests.post(webhook_url, json=payload, timeout=10)
            response.raise_for_status()
            
        except Exception as e:
            logging.error(f"Error sending webhook notification: {e}")
    
    return send_webhook


def console_notification_handler(subject: str, message: str):
    """Simple console notification handler."""
    print(f"\n{'='*50}")
    print(f"NOTIFICATION: {subject}")
    print(f"{'='*50}")
    print(message)
    print(f"{'='*50}\n")