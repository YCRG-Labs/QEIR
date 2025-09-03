# QEIR Batch Processing Automation

This document describes the batch processing and automation capabilities of the QEIR hypothesis testing framework.

## Overview

The batch processing automation system allows you to:

- Run multiple hypothesis tests across different time periods and specifications
- Monitor system resources and job progress in real-time
- Schedule recurring analysis jobs
- Automatically retry failed jobs
- Generate comprehensive reports and notifications
- Manage large-scale analysis workflows efficiently

## Components

### 1. BatchProcessor

The main batch processing engine that manages job queues, execution, and resource allocation.

**Key Features:**
- Concurrent job execution with configurable limits
- Job prioritization and dependency management
- Automatic retry mechanisms for failed jobs
- Result compression and cleanup
- Progress monitoring and logging

**Usage:**
```python
from qeir.utils.batch_processor import BatchProcessor, BatchProcessingConfig

# Create configuration
config = BatchProcessingConfig(
    max_concurrent_jobs=4,
    base_output_directory="batch_results",
    compress_results=True,
    cleanup_intermediate_files=True
)

# Initialize processor
processor = BatchProcessor(config=config, fred_api_key=os.getenv("FRED_API_KEY"))

# Create and add jobs
job_ids = processor.create_period_jobs(
    base_config=hypothesis_config,
    periods=[("2008-01-01", "2015-12-31"), ("2016-01-01", "2023-12-31")],
    job_name_prefix="period_analysis"
)

# Start processing
processor.start_processing()
```

### 2. ScheduledBatchProcessor

Extended processor with job scheduling capabilities for recurring analysis.

**Scheduling Options:**
- Daily execution at specified time
- Weekly execution on specific day
- Monthly execution on specific date
- Custom schedule specifications

**Usage:**
```python
from qeir.utils.batch_processor import ScheduledBatchProcessor

processor = ScheduledBatchProcessor(config=config, fred_api_key=os.getenv("FRED_API_KEY"))

# Schedule daily job
job_id = processor.schedule_job(
    job_config=job_configuration,
    schedule_spec="daily",  # Run daily at 2 AM
    job_id="daily_hypothesis_test"
)
```

### 3. BatchMonitor

Real-time monitoring system for tracking system resources, job performance, and generating alerts.

**Monitoring Features:**
- System resource tracking (CPU, memory, disk)
- Individual job performance metrics
- Real-time alerts and notifications
- SQLite database storage for historical data
- Comprehensive reporting

**Usage:**
```python
from qeir.utils.batch_monitor import BatchMonitor

monitor = BatchMonitor(
    monitoring_interval=30,
    enable_system_monitoring=True,
    enable_job_monitoring=True,
    enable_notifications=True,
    database_path="monitoring.db"
)

# Start monitoring
monitor.start_monitoring("batch_id_123")

# Add notification handlers
monitor.add_notification_handler(email_handler)
monitor.add_notification_handler(webhook_handler)
```

## Command Line Interface

### Basic Batch Processing

```bash
# Run batch processing with configuration file
qeir hypothesis batch --config batch_config.json --output batch_results

# Run with specific periods and specifications
qeir hypothesis batch --config batch_config.json \
    --periods "2008-2015,2016-2023" \
    --specifications "default,alternative" \
    --max-workers 4
```

### Advanced Batch Processing with Automation

```bash
# Enable monitoring and automation features
qeir hypothesis batch --config batch_config.json \
    --enable-monitoring \
    --monitoring-interval 30 \
    --auto-retry \
    --max-retries 3 \
    --compress-results \
    --cleanup-intermediate \
    --notification-webhook "https://your-webhook.com/notify"
```

### Scheduled Batch Processing

```bash
# Schedule daily batch processing
qeir hypothesis batch --config batch_config.json \
    --enable-scheduling \
    --schedule "daily" \
    --enable-monitoring
```

### Batch Management Commands

```bash
# List batch jobs
qeir hypothesis batch-manage list --output batch_results

# Monitor batch processing in real-time
qeir hypothesis batch-manage monitor --output batch_results --refresh 10

# Generate comprehensive report
qeir hypothesis batch-manage report --output batch_results --format json

# Cancel specific job
qeir hypothesis batch-manage cancel job_id_123 --output batch_results
```

## Configuration

### Batch Processing Configuration

```json
{
  "batch_processing_config": {
    "max_concurrent_jobs": 4,
    "job_timeout_minutes": 120,
    "retry_delay_minutes": 5,
    "memory_limit_gb": 8.0,
    "cpu_limit_percent": 80.0,
    "base_output_directory": "batch_results",
    "compress_results": true,
    "cleanup_intermediate_files": true,
    "enable_progress_monitoring": true,
    "log_level": "INFO"
  },
  "base_config": {
    "confidence_level": 0.95,
    "bootstrap_iterations": 1000,
    "enable_robustness_tests": true
  },
  "periods": [
    ["2008-01-01", "2015-12-31"],
    ["2016-01-01", "2023-12-31"]
  ],
  "specifications": ["default", "alternative"],
  "specifications_config": {
    "default": {
      "h1_confidence_proxy": "consumer_confidence"
    },
    "alternative": {
      "h1_confidence_proxy": "financial_stress_index"
    }
  }
}
```

### Monitoring Configuration

```json
{
  "resource_monitoring": {
    "enable_system_monitoring": true,
    "monitoring_interval_seconds": 30,
    "cpu_alert_threshold": 90.0,
    "memory_alert_threshold": 85.0,
    "disk_alert_threshold": 90.0,
    "job_timeout_alert_minutes": 180,
    "failure_rate_alert_threshold": 0.2
  }
}
```

## Notification System

### Email Notifications

```python
from qeir.utils.batch_monitor import email_notification_handler

# Create email handler
email_handler = email_notification_handler(
    smtp_server="smtp.gmail.com",
    smtp_port=587,
    username="your_email@gmail.com",
    password="your_password",
    to_email="recipient@example.com",
    from_email="your_email@gmail.com"
)

monitor.add_notification_handler(email_handler)
```

### Webhook Notifications

```python
from qeir.utils.batch_monitor import webhook_notification_handler

# Create webhook handler
webhook_handler = webhook_notification_handler("https://your-webhook.com/notify")
monitor.add_notification_handler(webhook_handler)
```

### Custom Notification Handlers

```python
def custom_notification_handler(subject: str, message: str):
    """Custom notification handler."""
    # Your custom notification logic here
    print(f"ALERT: {subject} - {message}")

monitor.add_notification_handler(custom_notification_handler)
```

## Monitoring and Reporting

### Real-time Monitoring

The monitoring system tracks:

- **System Resources**: CPU usage, memory consumption, disk space
- **Job Performance**: Execution time, resource usage, success/failure rates
- **Queue Status**: Pending jobs, active jobs, completion rates
- **Error Tracking**: Failed jobs, retry attempts, error messages

### Database Storage

All metrics are stored in SQLite database with tables:

- `batch_metrics`: Overall batch processing statistics
- `job_metrics`: Individual job performance data
- `system_metrics`: System resource usage over time

### Report Generation

Generate comprehensive reports in multiple formats:

```python
# JSON report
monitor.generate_report("batch_report.json")

# HTML report (via CLI)
qeir hypothesis batch-manage report --format html --output batch_results

# CSV export (via CLI)
qeir hypothesis batch-manage report --format csv --output batch_results
```

## Best Practices

### Resource Management

1. **Set appropriate concurrency limits** based on your system capabilities
2. **Monitor system resources** to avoid overloading
3. **Use compression and cleanup** to manage disk space
4. **Set reasonable timeouts** for long-running jobs

### Job Organization

1. **Use descriptive job names** and IDs for easy identification
2. **Group related jobs** using consistent naming conventions
3. **Set appropriate priorities** for time-sensitive analyses
4. **Use dependencies** to ensure proper execution order

### Monitoring and Alerts

1. **Enable monitoring** for production batch processing
2. **Set appropriate alert thresholds** to avoid false alarms
3. **Use multiple notification channels** for critical alerts
4. **Review monitoring reports** regularly to optimize performance

### Error Handling

1. **Enable automatic retries** for transient failures
2. **Set reasonable retry limits** to avoid infinite loops
3. **Monitor failure rates** to identify systematic issues
4. **Keep detailed logs** for troubleshooting

## Troubleshooting

### Common Issues

**High Memory Usage:**
- Reduce `max_concurrent_jobs`
- Enable `cleanup_intermediate_files`
- Increase `memory_limit_gb` threshold

**Job Timeouts:**
- Increase `job_timeout_minutes`
- Reduce `bootstrap_iterations` for faster execution
- Check system resource availability

**Database Errors:**
- Ensure write permissions for database directory
- Check disk space availability
- Verify SQLite installation

**Notification Failures:**
- Verify SMTP settings for email notifications
- Check webhook URL accessibility
- Test notification handlers independently

### Performance Optimization

1. **Adjust concurrency** based on system capabilities
2. **Use SSD storage** for better I/O performance
3. **Enable result compression** to save disk space
4. **Monitor resource usage** and adjust limits accordingly
5. **Use appropriate bootstrap iterations** for speed vs. accuracy trade-off

## Examples

See the following files for complete examples:

- `examples/example_batch_automation.py`: Comprehensive batch processing examples
- `qeir/config/batch_automation_template.json`: Configuration template
- `tests/test_batch_automation.py`: Test cases and usage patterns

## API Reference

For detailed API documentation, see the docstrings in:

- `qeir.utils.batch_processor`
- `qeir.utils.batch_monitor`
- `qeir.cli` (batch-related commands)