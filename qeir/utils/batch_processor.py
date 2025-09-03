"""
Batch Processing and Automation Utilities for QEIR Hypothesis Testing

This module provides batch processing capabilities for running multiple hypothesis
tests across different time periods, specifications, and configurations.

Author: QE Research Team
Date: 2025
Version: 1.0
"""

import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Callable
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
import threading
import queue
import schedule
import os

from ..core.hypothesis_testing import QEHypothesisTester, HypothesisTestingConfig
from ..utils.hypothesis_data_collector import HypothesisDataCollector


@dataclass
class BatchJob:
    """Represents a single batch job configuration."""
    
    job_id: str
    job_name: str
    config: HypothesisTestingConfig
    hypotheses_to_test: List[int] = field(default_factory=lambda: [1, 2, 3])
    output_directory: Optional[str] = None
    priority: int = 1  # Higher number = higher priority
    dependencies: List[str] = field(default_factory=list)  # Job IDs this job depends on
    retry_count: int = 0
    max_retries: int = 3
    status: str = "pending"  # pending, running, completed, failed, cancelled
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    error_message: Optional[str] = None
    results: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BatchProcessingConfig:
    """Configuration for batch processing operations."""
    
    # Execution settings
    max_concurrent_jobs: int = 4
    max_concurrent_hypotheses: int = 2
    job_timeout_minutes: int = 120
    retry_delay_minutes: int = 5
    
    # Resource management
    memory_limit_gb: float = 8.0
    cpu_limit_percent: float = 80.0
    disk_space_limit_gb: float = 10.0
    
    # Output management
    base_output_directory: str = "batch_results"
    compress_results: bool = True
    cleanup_intermediate_files: bool = True
    
    # Monitoring and logging
    enable_progress_monitoring: bool = True
    log_level: str = "INFO"
    save_job_logs: bool = True
    
    # Scheduling
    enable_scheduling: bool = False
    schedule_check_interval_minutes: int = 1
    
    # Notification settings
    enable_notifications: bool = False
    notification_email: Optional[str] = None
    notification_webhook: Optional[str] = None


class BatchProcessor:
    """
    Main batch processing engine for QEIR hypothesis testing.
    
    Handles job queuing, execution, monitoring, and result aggregation.
    """
    
    def __init__(self, 
                 config: BatchProcessingConfig,
                 fred_api_key: str,
                 qeir_config: Optional[Any] = None):
        """
        Initialize batch processor.
        
        Args:
            config: BatchProcessingConfig for processing settings
            fred_api_key: FRED API key for data collection
            qeir_config: Optional QEIR configuration
        """
        self.config = config
        self.fred_api_key = fred_api_key
        self.qeir_config = qeir_config
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(getattr(logging, config.log_level))
        
        # Job management
        self.job_queue: queue.PriorityQueue = queue.PriorityQueue()
        self.jobs: Dict[str, BatchJob] = {}
        self.completed_jobs: Dict[str, BatchJob] = {}
        self.failed_jobs: Dict[str, BatchJob] = {}
        
        # Execution state
        self.is_running = False
        self.executor: Optional[ThreadPoolExecutor] = None
        self.active_jobs: Dict[str, threading.Thread] = {}
        
        # Monitoring
        self.start_time: Optional[datetime] = None
        self.stats = {
            'jobs_submitted': 0,
            'jobs_completed': 0,
            'jobs_failed': 0,
            'total_processing_time': 0.0
        }
        
        # Setup output directory
        self.base_output_dir = Path(config.base_output_directory)
        self.base_output_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger.info("BatchProcessor initialized")
    
    def add_job(self, job: BatchJob) -> str:
        """
        Add a job to the processing queue.
        
        Args:
            job: BatchJob to add
            
        Returns:
            Job ID
        """
        # Validate job
        if job.job_id in self.jobs:
            raise ValueError(f"Job ID already exists: {job.job_id}")
        
        # Set output directory if not specified
        if not job.output_directory:
            job.output_directory = str(self.base_output_dir / job.job_id)
        
        # Create output directory
        Path(job.output_directory).mkdir(parents=True, exist_ok=True)
        
        # Add to job tracking
        self.jobs[job.job_id] = job
        
        # Add to queue (priority queue uses negative priority for max-heap behavior)
        self.job_queue.put((-job.priority, job.created_at, job.job_id))
        
        self.stats['jobs_submitted'] += 1
        self.logger.info(f"Added job {job.job_id} to queue")
        
        return job.job_id
    
    def create_job_from_config(self, 
                              job_config: Dict[str, Any],
                              job_id: Optional[str] = None) -> BatchJob:
        """
        Create a BatchJob from configuration dictionary.
        
        Args:
            job_config: Job configuration dictionary
            job_id: Optional job ID (auto-generated if not provided)
            
        Returns:
            BatchJob instance
        """
        if not job_id:
            job_id = f"job_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{len(self.jobs)}"
        
        # Extract hypothesis testing config
        hypothesis_config_dict = {k: v for k, v in job_config.items() 
                                 if k not in ['job_name', 'hypotheses_to_test', 'priority', 'dependencies']}
        hypothesis_config = HypothesisTestingConfig(**hypothesis_config_dict)
        
        # Create job
        job = BatchJob(
            job_id=job_id,
            job_name=job_config.get('job_name', job_id),
            config=hypothesis_config,
            hypotheses_to_test=job_config.get('hypotheses_to_test', [1, 2, 3]),
            priority=job_config.get('priority', 1),
            dependencies=job_config.get('dependencies', [])
        )
        
        return job
    
    def create_period_jobs(self, 
                          base_config: HypothesisTestingConfig,
                          periods: List[Tuple[str, str]],
                          job_name_prefix: str = "period_job") -> List[str]:
        """
        Create multiple jobs for different time periods.
        
        Args:
            base_config: Base configuration to use for all jobs
            periods: List of (start_date, end_date) tuples
            job_name_prefix: Prefix for job names
            
        Returns:
            List of created job IDs
        """
        job_ids = []
        
        for i, (start_date, end_date) in enumerate(periods):
            # Create period-specific config
            period_config = HypothesisTestingConfig(**base_config.__dict__)
            period_config.start_date = start_date
            period_config.end_date = end_date
            
            # Create job
            job_id = f"{job_name_prefix}_{start_date}_{end_date}"
            job = BatchJob(
                job_id=job_id,
                job_name=f"{job_name_prefix} ({start_date} to {end_date})",
                config=period_config
            )
            
            job_ids.append(self.add_job(job))
        
        return job_ids
    
    def create_specification_jobs(self,
                                 base_config: HypothesisTestingConfig,
                                 specifications: Dict[str, Dict[str, Any]],
                                 job_name_prefix: str = "spec_job") -> List[str]:
        """
        Create multiple jobs for different specifications.
        
        Args:
            base_config: Base configuration to use for all jobs
            specifications: Dictionary of specification name -> config overrides
            job_name_prefix: Prefix for job names
            
        Returns:
            List of created job IDs
        """
        job_ids = []
        
        for spec_name, spec_overrides in specifications.items():
            # Create specification-specific config
            spec_config = HypothesisTestingConfig(**base_config.__dict__)
            
            # Apply overrides
            for key, value in spec_overrides.items():
                if hasattr(spec_config, key):
                    setattr(spec_config, key, value)
            
            # Create job
            job_id = f"{job_name_prefix}_{spec_name}"
            job = BatchJob(
                job_id=job_id,
                job_name=f"{job_name_prefix} ({spec_name})",
                config=spec_config
            )
            
            job_ids.append(self.add_job(job))
        
        return job_ids
    
    def start_processing(self):
        """Start the batch processing engine."""
        if self.is_running:
            self.logger.warning("Batch processor already running")
            return
        
        self.is_running = True
        self.start_time = datetime.now()
        self.executor = ThreadPoolExecutor(max_workers=self.config.max_concurrent_jobs)
        
        self.logger.info("Starting batch processing engine")
        
        # Start main processing loop in separate thread
        processing_thread = threading.Thread(target=self._processing_loop, daemon=True)
        processing_thread.start()
        
        # Start monitoring thread if enabled
        if self.config.enable_progress_monitoring:
            monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
            monitoring_thread.start()
    
    def stop_processing(self, wait_for_completion: bool = True):
        """Stop the batch processing engine."""
        self.logger.info("Stopping batch processing engine")
        self.is_running = False
        
        if self.executor and wait_for_completion:
            self.executor.shutdown(wait=True)
        elif self.executor:
            self.executor.shutdown(wait=False)
    
    def _processing_loop(self):
        """Main processing loop."""
        while self.is_running:
            try:
                # Get next job from queue (with timeout)
                try:
                    priority, created_at, job_id = self.job_queue.get(timeout=1.0)
                except queue.Empty:
                    continue
                
                job = self.jobs[job_id]
                
                # Check dependencies
                if not self._check_dependencies(job):
                    # Re-queue job with lower priority
                    self.job_queue.put((priority - 0.1, created_at, job_id))
                    time.sleep(1)
                    continue
                
                # Check resource availability
                if not self._check_resources():
                    # Re-queue job
                    self.job_queue.put((priority, created_at, job_id))
                    time.sleep(5)
                    continue
                
                # Submit job for execution
                future = self.executor.submit(self._execute_job, job)
                self.active_jobs[job_id] = future
                
                self.job_queue.task_done()
                
            except Exception as e:
                self.logger.error(f"Error in processing loop: {e}")
                time.sleep(1)
    
    def _execute_job(self, job: BatchJob) -> BatchJob:
        """
        Execute a single batch job.
        
        Args:
            job: BatchJob to execute
            
        Returns:
            Updated BatchJob with results
        """
        job.status = "running"
        job.started_at = datetime.now().isoformat()
        
        self.logger.info(f"Starting execution of job {job.job_id}")
        
        try:
            # Setup job-specific logging
            job_log_file = Path(job.output_directory) / "job.log"
            job_handler = logging.FileHandler(job_log_file)
            job_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
            
            job_logger = logging.getLogger(f"job_{job.job_id}")
            job_logger.addHandler(job_handler)
            job_logger.setLevel(logging.INFO)
            
            # Initialize data collector and tester
            data_collector = HypothesisDataCollector(fred_api_key=self.fred_api_key)
            tester = QEHypothesisTester(
                data_collector=data_collector,
                config=job.config,
                qeir_config=self.qeir_config
            )
            
            # Load data
            job_logger.info("Loading data from FRED API")
            data = tester.load_data()
            
            # Save data if requested
            if job.config.save_intermediate_results:
                data_file = Path(job.output_directory) / "data.json"
                self._save_data(data, data_file)
            
            # Execute hypothesis tests
            results = {}
            
            for hypothesis_num in job.hypotheses_to_test:
                job_logger.info(f"Testing Hypothesis {hypothesis_num}")
                
                if hypothesis_num == 1:
                    result = tester.test_hypothesis1(data)
                elif hypothesis_num == 2:
                    result = tester.test_hypothesis2(data)
                elif hypothesis_num == 3:
                    result = tester.test_hypothesis3(data)
                else:
                    job_logger.warning(f"Unknown hypothesis number: {hypothesis_num}")
                    continue
                
                results[f'hypothesis{hypothesis_num}'] = result
                
                # Save individual results
                result_file = Path(job.output_directory) / f"hypothesis_{hypothesis_num}_results.json"
                self._save_results(result, result_file)
            
            # Run robustness tests if enabled
            if job.config.enable_robustness_tests and results:
                job_logger.info("Running robustness tests")
                robustness_results = tester.run_robustness_tests(results)
                
                # Add robustness results to each hypothesis
                for hypothesis_key in results:
                    results[hypothesis_key].robustness_results = robustness_results
            
            # Generate summary report
            summary_file = Path(job.output_directory) / "job_summary.txt"
            self._generate_job_summary(job, results, summary_file)
            
            # Cleanup if requested
            if self.config.cleanup_intermediate_files:
                self._cleanup_intermediate_files(Path(job.output_directory))
            
            # Compress results if requested
            if self.config.compress_results:
                self._compress_results(Path(job.output_directory))
            
            # Update job status
            job.status = "completed"
            job.completed_at = datetime.now().isoformat()
            job.results = {
                'hypotheses_completed': list(results.keys()),
                'output_directory': job.output_directory,
                'summary_file': str(summary_file)
            }
            
            self.completed_jobs[job.job_id] = job
            self.stats['jobs_completed'] += 1
            
            job_logger.info(f"Job {job.job_id} completed successfully")
            
        except Exception as e:
            # Handle job failure
            job.status = "failed"
            job.completed_at = datetime.now().isoformat()
            job.error_message = str(e)
            
            self.failed_jobs[job.job_id] = job
            self.stats['jobs_failed'] += 1
            
            self.logger.error(f"Job {job.job_id} failed: {e}")
            
            # Retry logic
            if job.retry_count < job.max_retries:
                job.retry_count += 1
                job.status = "pending"
                
                # Re-queue with delay
                retry_time = datetime.now() + timedelta(minutes=self.config.retry_delay_minutes)
                self.job_queue.put((-job.priority, retry_time.isoformat(), job.job_id))
                
                self.logger.info(f"Re-queuing job {job.job_id} for retry {job.retry_count}/{job.max_retries}")
        
        finally:
            # Remove from active jobs
            if job.job_id in self.active_jobs:
                del self.active_jobs[job.job_id]
        
        return job
    
    def _check_dependencies(self, job: BatchJob) -> bool:
        """Check if job dependencies are satisfied."""
        for dep_job_id in job.dependencies:
            if dep_job_id not in self.completed_jobs:
                return False
        return True
    
    def _check_resources(self) -> bool:
        """Check if system resources are available for new job."""
        # Simple resource check (could be enhanced with actual system monitoring)
        active_job_count = len(self.active_jobs)
        return active_job_count < self.config.max_concurrent_jobs
    
    def _monitoring_loop(self):
        """Monitoring loop for progress tracking."""
        while self.is_running:
            try:
                self._log_progress()
                time.sleep(60)  # Log progress every minute
            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {e}")
    
    def _log_progress(self):
        """Log current processing progress."""
        total_jobs = len(self.jobs)
        completed = self.stats['jobs_completed']
        failed = self.stats['jobs_failed']
        active = len(self.active_jobs)
        pending = total_jobs - completed - failed - active
        
        if self.start_time:
            elapsed = datetime.now() - self.start_time
            elapsed_str = str(elapsed).split('.')[0]  # Remove microseconds
        else:
            elapsed_str = "Unknown"
        
        self.logger.info(
            f"Progress: {completed}/{total_jobs} completed, "
            f"{active} active, {pending} pending, {failed} failed. "
            f"Elapsed: {elapsed_str}"
        )
    
    def _save_data(self, data, file_path: Path):
        """Save hypothesis data to file."""
        # Simplified data saving (would need proper implementation)
        data_dict = {
            'timestamp': datetime.now().isoformat(),
            'data_summary': 'Data saved successfully'
        }
        
        with open(file_path, 'w') as f:
            json.dump(data_dict, f, indent=2, default=str)
    
    def _save_results(self, results, file_path: Path):
        """Save hypothesis results to file."""
        # Convert results to serializable format
        results_dict = {
            'hypothesis_name': results.hypothesis_name,
            'hypothesis_number': results.hypothesis_number,
            'main_result': results.main_result,
            'statistical_significance': results.statistical_significance,
            'economic_significance': results.economic_significance,
            'data_period': results.data_period,
            'test_timestamp': results.test_timestamp
        }
        
        with open(file_path, 'w') as f:
            json.dump(results_dict, f, indent=2, default=str)
    
    def _generate_job_summary(self, job: BatchJob, results: Dict, summary_file: Path):
        """Generate summary report for completed job."""
        with open(summary_file, 'w') as f:
            f.write(f"QEIR Batch Job Summary\n")
            f.write(f"=" * 30 + "\n\n")
            
            f.write(f"Job ID: {job.job_id}\n")
            f.write(f"Job Name: {job.job_name}\n")
            f.write(f"Status: {job.status}\n")
            f.write(f"Started: {job.started_at}\n")
            f.write(f"Completed: {job.completed_at}\n\n")
            
            f.write(f"Configuration:\n")
            f.write(f"  Start Date: {job.config.start_date}\n")
            f.write(f"  End Date: {job.config.end_date}\n")
            f.write(f"  Hypotheses Tested: {job.hypotheses_to_test}\n\n")
            
            f.write(f"Results Summary:\n")
            for hypothesis_key, result in results.items():
                f.write(f"  {hypothesis_key}: {result.hypothesis_name}\n")
                f.write(f"    Status: {'Completed' if result.main_result else 'Failed'}\n")
                if result.data_period:
                    f.write(f"    Observations: {result.data_period.get('observations', 'N/A')}\n")
    
    def _cleanup_intermediate_files(self, job_dir: Path):
        """Clean up intermediate files to save space."""
        # Remove large intermediate files (implementation would depend on specific files)
        patterns_to_remove = ['*.tmp', '*.cache', '*_intermediate.json']
        
        for pattern in patterns_to_remove:
            for file_path in job_dir.glob(pattern):
                try:
                    file_path.unlink()
                    self.logger.debug(f"Removed intermediate file: {file_path}")
                except Exception as e:
                    self.logger.warning(f"Could not remove {file_path}: {e}")
    
    def _compress_results(self, job_dir: Path):
        """Compress job results to save space."""
        import zipfile
        
        zip_file = job_dir / "results.zip"
        
        try:
            with zipfile.ZipFile(zip_file, 'w', zipfile.ZIP_DEFLATED) as zf:
                for file_path in job_dir.glob("*.json"):
                    zf.write(file_path, file_path.name)
                for file_path in job_dir.glob("*.txt"):
                    zf.write(file_path, file_path.name)
            
            self.logger.info(f"Compressed results to {zip_file}")
            
        except Exception as e:
            self.logger.warning(f"Could not compress results: {e}")
    
    def get_status(self) -> Dict[str, Any]:
        """Get current batch processing status."""
        return {
            'is_running': self.is_running,
            'start_time': self.start_time.isoformat() if self.start_time else None,
            'stats': self.stats.copy(),
            'active_jobs': list(self.active_jobs.keys()),
            'queue_size': self.job_queue.qsize(),
            'total_jobs': len(self.jobs)
        }
    
    def get_job_status(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Get status of specific job."""
        if job_id in self.jobs:
            job = self.jobs[job_id]
            return {
                'job_id': job.job_id,
                'job_name': job.job_name,
                'status': job.status,
                'created_at': job.created_at,
                'started_at': job.started_at,
                'completed_at': job.completed_at,
                'retry_count': job.retry_count,
                'error_message': job.error_message,
                'results': job.results
            }
        return None
    
    def cancel_job(self, job_id: str) -> bool:
        """Cancel a pending or running job."""
        if job_id in self.jobs:
            job = self.jobs[job_id]
            if job.status in ['pending', 'running']:
                job.status = 'cancelled'
                job.completed_at = datetime.now().isoformat()
                
                # Remove from active jobs if running
                if job_id in self.active_jobs:
                    # Note: This is a simplified cancellation
                    # In practice, you'd need more sophisticated cancellation logic
                    del self.active_jobs[job_id]
                
                self.logger.info(f"Cancelled job {job_id}")
                return True
        
        return False


class ScheduledBatchProcessor(BatchProcessor):
    """
    Extended batch processor with scheduling capabilities.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.scheduled_jobs: Dict[str, Any] = {}
        self.scheduler_running = False
    
    def schedule_job(self, 
                    job_config: Dict[str, Any],
                    schedule_spec: str,
                    job_id: Optional[str] = None) -> str:
        """
        Schedule a job to run at specified times.
        
        Args:
            job_config: Job configuration dictionary
            schedule_spec: Schedule specification (e.g., "daily", "weekly", "monthly")
            job_id: Optional job ID
            
        Returns:
            Scheduled job ID
        """
        if not job_id:
            job_id = f"scheduled_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Parse schedule specification
        if schedule_spec == "daily":
            schedule.every().day.at("02:00").do(self._create_and_queue_job, job_config, job_id)
        elif schedule_spec == "weekly":
            schedule.every().monday.at("02:00").do(self._create_and_queue_job, job_config, job_id)
        elif schedule_spec == "monthly":
            schedule.every().month.do(self._create_and_queue_job, job_config, job_id)
        else:
            # Custom schedule (simplified - would need more sophisticated parsing)
            schedule.every().day.at(schedule_spec).do(self._create_and_queue_job, job_config, job_id)
        
        self.scheduled_jobs[job_id] = {
            'job_config': job_config,
            'schedule_spec': schedule_spec,
            'created_at': datetime.now().isoformat()
        }
        
        # Start scheduler if not running
        if not self.scheduler_running:
            self._start_scheduler()
        
        self.logger.info(f"Scheduled job {job_id} with schedule: {schedule_spec}")
        return job_id
    
    def _create_and_queue_job(self, job_config: Dict[str, Any], base_job_id: str):
        """Create and queue a scheduled job."""
        # Create unique job ID for this execution
        execution_time = datetime.now().strftime('%Y%m%d_%H%M%S')
        job_id = f"{base_job_id}_{execution_time}"
        
        # Create and add job
        job = self.create_job_from_config(job_config, job_id)
        self.add_job(job)
        
        self.logger.info(f"Created scheduled job execution: {job_id}")
    
    def _start_scheduler(self):
        """Start the job scheduler."""
        self.scheduler_running = True
        
        def scheduler_loop():
            while self.scheduler_running:
                schedule.run_pending()
                time.sleep(self.config.schedule_check_interval_minutes * 60)
        
        scheduler_thread = threading.Thread(target=scheduler_loop, daemon=True)
        scheduler_thread.start()
        
        self.logger.info("Job scheduler started")
    
    def stop_scheduler(self):
        """Stop the job scheduler."""
        self.scheduler_running = False
        schedule.clear()
        self.logger.info("Job scheduler stopped")


def create_batch_config_from_file(config_file: str) -> BatchProcessingConfig:
    """Create BatchProcessingConfig from JSON file."""
    with open(config_file, 'r') as f:
        config_dict = json.load(f)
    
    return BatchProcessingConfig(**config_dict)


def save_batch_config_to_file(config: BatchProcessingConfig, config_file: str):
    """Save BatchProcessingConfig to JSON file."""
    config_dict = config.__dict__
    
    with open(config_file, 'w') as f:
        json.dump(config_dict, f, indent=2)