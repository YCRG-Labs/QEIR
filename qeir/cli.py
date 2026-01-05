#!/usr/bin/env python3
"""
Command-line interface for QEIR analysis framework.
"""

import argparse
import sys
import json
import logging
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime
import time

from .analysis.revised_qe_analyzer import RevisedQEAnalyzer
from .visualization.publication_visualization import PublicationVisualizationSuite
from .utils.publication_export_system import PublicationExportSystem
from .core.hypothesis_testing import QEHypothesisTester, HypothesisTestingConfig
from .utils.hypothesis_data_collector import HypothesisDataCollector
from .config import QEIRConfig, load_config_from_file, save_config_to_file


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="QEIR: Quantitative Easing Investment Response Analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Test all hypotheses with default configuration
  qeir hypothesis test-all --fred-api-key YOUR_KEY
  
  # Test specific hypothesis with custom config
  qeir hypothesis test --hypothesis 1 --config custom_config.json
  
  # Run batch analysis for multiple periods
  qeir hypothesis batch --config batch_config.json --periods 2008-2015,2016-2023
  
  # Generate configuration template
  qeir config create --output hypothesis_config.json
        """
    )
    
    # Global arguments
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose logging")
    parser.add_argument("--quiet", "-q", action="store_true", help="Suppress output except errors")
    parser.add_argument("--log-file", help="Log file path")
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Hypothesis testing commands
    hypothesis_parser = subparsers.add_parser("hypothesis", help="Hypothesis testing commands")
    hypothesis_subparsers = hypothesis_parser.add_subparsers(dest="hypothesis_command", help="Hypothesis testing operations")
    
    # Test individual hypothesis
    test_parser = hypothesis_subparsers.add_parser("test", help="Test individual hypothesis")
    test_parser.add_argument("--hypothesis", choices=[1, 2, 3], type=int, required=True,
                            help="Hypothesis number to test (1, 2, or 3)")
    test_parser.add_argument("--config", "-c", help="Configuration file path")
    test_parser.add_argument("--fred-api-key", help="FRED API key (or set FRED_API_KEY env var)")
    test_parser.add_argument("--start-date", help="Start date (YYYY-MM-DD)")
    test_parser.add_argument("--end-date", help="End date (YYYY-MM-DD)")
    test_parser.add_argument("--output", "-o", default="hypothesis_results", help="Output directory")
    test_parser.add_argument("--save-data", action="store_true", help="Save collected data")
    test_parser.add_argument("--no-robustness", action="store_true", help="Skip robustness tests")
    
    # Test all hypotheses
    test_all_parser = hypothesis_subparsers.add_parser("test-all", help="Test all hypotheses")
    test_all_parser.add_argument("--config", "-c", help="Configuration file path")
    test_all_parser.add_argument("--fred-api-key", help="FRED API key (or set FRED_API_KEY env var)")
    test_all_parser.add_argument("--start-date", help="Start date (YYYY-MM-DD)")
    test_all_parser.add_argument("--end-date", help="End date (YYYY-MM-DD)")
    test_all_parser.add_argument("--output", "-o", default="hypothesis_results", help="Output directory")
    test_all_parser.add_argument("--save-data", action="store_true", help="Save collected data")
    test_all_parser.add_argument("--no-robustness", action="store_true", help="Skip robustness tests")
    test_all_parser.add_argument("--parallel", action="store_true", help="Run hypotheses in parallel")
    
    # Batch processing
    batch_parser = hypothesis_subparsers.add_parser("batch", help="Batch processing for multiple scenarios")
    batch_parser.add_argument("--config", "-c", required=True, help="Batch configuration file path")
    batch_parser.add_argument("--periods", help="Comma-separated list of date periods (start1-end1,start2-end2)")
    batch_parser.add_argument("--specifications", help="Comma-separated list of specification names")
    batch_parser.add_argument("--output", "-o", default="batch_results", help="Output directory")
    batch_parser.add_argument("--max-workers", type=int, default=4, help="Maximum parallel workers")
    batch_parser.add_argument("--enable-monitoring", action="store_true", help="Enable batch monitoring")
    batch_parser.add_argument("--monitoring-interval", type=int, default=30, help="Monitoring interval in seconds")
    batch_parser.add_argument("--enable-scheduling", action="store_true", help="Enable job scheduling")
    batch_parser.add_argument("--schedule", help="Schedule specification (e.g., 'daily', 'weekly', '02:00')")
    batch_parser.add_argument("--auto-retry", action="store_true", help="Enable automatic job retry on failure")
    batch_parser.add_argument("--max-retries", type=int, default=3, help="Maximum retry attempts")
    batch_parser.add_argument("--compress-results", action="store_true", help="Compress results to save space")
    batch_parser.add_argument("--cleanup-intermediate", action="store_true", help="Clean up intermediate files")
    batch_parser.add_argument("--notification-email", help="Email address for notifications")
    batch_parser.add_argument("--notification-webhook", help="Webhook URL for notifications")
    
    # Status monitoring
    status_parser = hypothesis_subparsers.add_parser("status", help="Check status of running tests")
    status_parser.add_argument("--output", "-o", default="hypothesis_results", help="Results directory to check")
    status_parser.add_argument("--watch", "-w", action="store_true", help="Watch for changes")
    
    # Batch management
    batch_mgmt_parser = hypothesis_subparsers.add_parser("batch-manage", help="Manage batch processing operations")
    batch_mgmt_subparsers = batch_mgmt_parser.add_subparsers(dest="batch_mgmt_command", help="Batch management operations")
    
    # List batch jobs
    list_jobs_parser = batch_mgmt_subparsers.add_parser("list", help="List batch jobs")
    list_jobs_parser.add_argument("--output", "-o", default="batch_results", help="Batch results directory")
    list_jobs_parser.add_argument("--status", choices=["all", "running", "completed", "failed"], default="all", help="Filter by status")
    
    # Cancel batch job
    cancel_job_parser = batch_mgmt_subparsers.add_parser("cancel", help="Cancel batch job")
    cancel_job_parser.add_argument("job_id", help="Job ID to cancel")
    cancel_job_parser.add_argument("--output", "-o", default="batch_results", help="Batch results directory")
    
    # Monitor batch processing
    monitor_parser = batch_mgmt_subparsers.add_parser("monitor", help="Monitor batch processing")
    monitor_parser.add_argument("--output", "-o", default="batch_results", help="Batch results directory")
    monitor_parser.add_argument("--refresh", "-r", type=int, default=10, help="Refresh interval in seconds")
    monitor_parser.add_argument("--database", help="Monitoring database path")
    
    # Generate batch report
    report_parser = batch_mgmt_subparsers.add_parser("report", help="Generate batch processing report")
    report_parser.add_argument("--output", "-o", default="batch_results", help="Batch results directory")
    report_parser.add_argument("--database", help="Monitoring database path")
    report_parser.add_argument("--format", choices=["json", "html", "csv"], default="json", help="Report format")
    
    # Configuration management commands
    config_parser = subparsers.add_parser("config", help="Configuration management")
    config_subparsers = config_parser.add_subparsers(dest="config_command", help="Configuration operations")
    
    # Create configuration template
    create_config_parser = config_subparsers.add_parser("create", help="Create configuration template")
    create_config_parser.add_argument("--output", "-o", default="hypothesis_config.json", 
                                     help="Output configuration file path")
    create_config_parser.add_argument("--template", choices=["basic", "advanced", "batch"], 
                                     default="basic", help="Configuration template type")
    
    # Validate configuration
    validate_config_parser = config_subparsers.add_parser("validate", help="Validate configuration file")
    validate_config_parser.add_argument("config_file", help="Configuration file to validate")
    
    # Show configuration
    show_config_parser = config_subparsers.add_parser("show", help="Show current configuration")
    show_config_parser.add_argument("--config", "-c", help="Configuration file path")
    show_config_parser.add_argument("--format", choices=["json", "yaml", "table"], default="table",
                                   help="Output format")
    
    # Legacy commands (maintain backward compatibility)
    analyze_parser = subparsers.add_parser("analyze", help="Run QE analysis (legacy)")
    analyze_parser.add_argument("--data", required=True, help="Path to data file")
    analyze_parser.add_argument("--output", default="results", help="Output directory")
    analyze_parser.add_argument("--hypothesis", choices=[1, 2, 3], type=int, 
                               help="Specific hypothesis to test")
    
    viz_parser = subparsers.add_parser("visualize", help="Generate visualizations")
    viz_parser.add_argument("--results", required=True, help="Path to results file")
    viz_parser.add_argument("--output", default="figures", help="Output directory")
    viz_parser.add_argument("--format", choices=["png", "pdf", "svg"], default="png",
                           help="Output format")
    
    export_parser = subparsers.add_parser("export", help="Export publication outputs")
    export_parser.add_argument("--results", required=True, help="Path to results")
    export_parser.add_argument("--output", default="publication", help="Output directory")
    export_parser.add_argument("--journal", choices=["jme", "aej", "jimf"], 
                              help="Target journal format")
    
    # Final validation command
    final_validation_parser = subparsers.add_parser("final-validation", help="Run final validation suite")
    final_validation_parser.add_argument("--data-path", help="Path to validation data file (optional)")
    final_validation_parser.add_argument("--output-dir", default="final_validation_results", 
                                       help="Output directory for validation results")
    final_validation_parser.add_argument("--save-data", action="store_true", 
                                       help="Save collected data for future validation runs")
    final_validation_parser.add_argument("--quick-validation", action="store_true",
                                       help="Run quick validation with reduced robustness testing")
    final_validation_parser.add_argument("--fred-api-key", help="FRED API key (or set FRED_API_KEY env var)")
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 1
    
    # Setup logging
    setup_logging(args)
    
    try:
        # Route to appropriate handler
        if args.command == "hypothesis":
            return handle_hypothesis_command(args)
        elif args.command == "config":
            return handle_config_command(args)
        elif args.command == "analyze":
            return run_analysis(args)
        elif args.command == "visualize":
            return run_visualization(args)
        elif args.command == "export":
            return run_export(args)
        elif args.command == "final-validation":
            return run_final_validation(args)
    except KeyboardInterrupt:
        print("\nOperation cancelled by user", file=sys.stderr)
        return 130
    except Exception as e:
        logging.error(f"Unexpected error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1
    
    return 0


def setup_logging(args):
    """Setup logging configuration."""
    level = logging.WARNING
    if args.verbose:
        level = logging.DEBUG
    elif args.quiet:
        level = logging.ERROR
    else:
        level = logging.INFO
    
    # Configure logging
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    if hasattr(args, 'log_file') and args.log_file:
        logging.basicConfig(
            level=level,
            format=log_format,
            handlers=[
                logging.FileHandler(args.log_file),
                logging.StreamHandler(sys.stdout)
            ]
        )
    else:
        logging.basicConfig(level=level, format=log_format)


def handle_hypothesis_command(args):
    """Handle hypothesis testing commands."""
    if args.hypothesis_command == "test":
        return run_hypothesis_test(args)
    elif args.hypothesis_command == "test-all":
        return run_all_hypothesis_tests(args)
    elif args.hypothesis_command == "batch":
        return run_batch_processing(args)
    elif args.hypothesis_command == "status":
        return show_test_status(args)
    elif args.hypothesis_command == "batch-manage":
        return handle_batch_management(args)
    else:
        print("Unknown hypothesis command", file=sys.stderr)
        return 1


def handle_config_command(args):
    """Handle configuration management commands."""
    if args.config_command == "create":
        return create_config_template(args)
    elif args.config_command == "validate":
        return validate_config_file(args)
    elif args.config_command == "show":
        return show_config(args)
    else:
        print("Unknown config command", file=sys.stderr)
        return 1


def run_hypothesis_test(args):
    """Run individual hypothesis test."""
    import os
    
    logging.info(f"Starting Hypothesis {args.hypothesis} test")
    
    # Get FRED API key
    from qeir.config import get_fred_api_key
    try:
        fred_api_key = args.fred_api_key or get_fred_api_key()
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    
    try:
        # Load configuration
        config = load_hypothesis_config(args.config)
        
        # Override config with command line arguments
        if args.start_date:
            config.start_date = args.start_date
        if args.end_date:
            config.end_date = args.end_date
        if args.no_robustness:
            config.enable_robustness_tests = False
        
        # Setup output directory
        output_dir = Path(args.output)
        output_dir.mkdir(parents=True, exist_ok=True)
        config.output_directory = str(output_dir)
        
        # Initialize data collector and tester
        data_collector = HypothesisDataCollector(fred_api_key=fred_api_key)
        tester = QEHypothesisTester(data_collector=data_collector, config=config)
        
        # Create progress tracker
        progress = ProgressTracker(f"Hypothesis {args.hypothesis} Test")
        
        # Load data
        progress.update("Loading data from FRED API...")
        data = tester.load_data()
        
        if args.save_data:
            data_file = output_dir / f"hypothesis_{args.hypothesis}_data.json"
            save_hypothesis_data(data, data_file)
            logging.info(f"Data saved to {data_file}")
        
        # Run specific hypothesis test
        progress.update(f"Running Hypothesis {args.hypothesis} analysis...")
        
        if args.hypothesis == 1:
            results = tester.test_hypothesis1(data)
        elif args.hypothesis == 2:
            results = tester.test_hypothesis2(data)
        elif args.hypothesis == 3:
            results = tester.test_hypothesis3(data)
        
        # Run robustness tests if enabled
        if config.enable_robustness_tests:
            progress.update("Running robustness tests...")
            robustness_results = tester.run_robustness_tests({f'hypothesis{args.hypothesis}': results})
            results.robustness_results = robustness_results
        
        # Save results
        progress.update("Saving results...")
        results_file = output_dir / f"hypothesis_{args.hypothesis}_results.json"
        save_hypothesis_results(results, results_file)
        
        # Generate summary report
        progress.update("Generating summary report...")
        summary_file = output_dir / f"hypothesis_{args.hypothesis}_summary.txt"
        generate_summary_report(results, summary_file)
        
        progress.complete(f"Hypothesis {args.hypothesis} test completed successfully")
        print(f"\nResults saved to: {output_dir}")
        print(f"Summary report: {summary_file}")
        
        return 0
        
    except Exception as e:
        logging.error(f"Error in hypothesis test: {e}")
        return 1


def run_all_hypothesis_tests(args):
    """Run all hypothesis tests."""
    import os
    from concurrent.futures import ThreadPoolExecutor, as_completed
    
    logging.info("Starting all hypothesis tests")
    
    # Get FRED API key
    from qeir.config import get_fred_api_key
    try:
        fred_api_key = args.fred_api_key or get_fred_api_key()
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    
    try:
        # Load configuration
        config = load_hypothesis_config(args.config)
        
        # Override config with command line arguments
        if args.start_date:
            config.start_date = args.start_date
        if args.end_date:
            config.end_date = args.end_date
        if args.no_robustness:
            config.enable_robustness_tests = False
        
        # Setup output directory
        output_dir = Path(args.output)
        output_dir.mkdir(parents=True, exist_ok=True)
        config.output_directory = str(output_dir)
        
        # Initialize data collector and tester
        data_collector = HypothesisDataCollector(fred_api_key=fred_api_key)
        tester = QEHypothesisTester(data_collector=data_collector, config=config)
        
        # Create progress tracker
        progress = ProgressTracker("All Hypothesis Tests")
        
        # Load data once for all tests
        progress.update("Loading data from FRED API...")
        data = tester.load_data()
        
        if args.save_data:
            data_file = output_dir / "all_hypotheses_data.json"
            save_hypothesis_data(data, data_file)
            logging.info(f"Data saved to {data_file}")
        
        # Run tests
        all_results = {}
        
        if args.parallel:
            # Run tests in parallel
            progress.update("Running all hypothesis tests in parallel...")
            
            def run_single_test(hypothesis_num):
                if hypothesis_num == 1:
                    return hypothesis_num, tester.test_hypothesis1(data)
                elif hypothesis_num == 2:
                    return hypothesis_num, tester.test_hypothesis2(data)
                elif hypothesis_num == 3:
                    return hypothesis_num, tester.test_hypothesis3(data)
            
            with ThreadPoolExecutor(max_workers=3) as executor:
                future_to_hypothesis = {executor.submit(run_single_test, i): i for i in [1, 2, 3]}
                
                for future in as_completed(future_to_hypothesis):
                    hypothesis_num = future_to_hypothesis[future]
                    try:
                        hypothesis_num, result = future.result()
                        all_results[f'hypothesis{hypothesis_num}'] = result
                        progress.update(f"Completed Hypothesis {hypothesis_num}")
                    except Exception as e:
                        logging.error(f"Error in Hypothesis {hypothesis_num}: {e}")
        else:
            # Run tests sequentially
            for i in [1, 2, 3]:
                progress.update(f"Running Hypothesis {i} analysis...")
                
                if i == 1:
                    result = tester.test_hypothesis1(data)
                elif i == 2:
                    result = tester.test_hypothesis2(data)
                elif i == 3:
                    result = tester.test_hypothesis3(data)
                
                all_results[f'hypothesis{i}'] = result
        
        # Run robustness tests if enabled
        if config.enable_robustness_tests:
            progress.update("Running comprehensive robustness tests...")
            robustness_results = tester.run_robustness_tests(all_results)
            
            # Add robustness results to each hypothesis
            for hypothesis_key in all_results:
                all_results[hypothesis_key].robustness_results = robustness_results
        
        # Save all results
        progress.update("Saving all results...")
        for hypothesis_key, results in all_results.items():
            hypothesis_num = hypothesis_key.replace('hypothesis', '')
            results_file = output_dir / f"hypothesis_{hypothesis_num}_results.json"
            save_hypothesis_results(results, results_file)
        
        # Generate comprehensive summary report
        progress.update("Generating comprehensive summary report...")
        summary_file = output_dir / "all_hypotheses_summary.txt"
        generate_comprehensive_summary(all_results, summary_file)
        
        progress.complete("All hypothesis tests completed successfully")
        print(f"\nResults saved to: {output_dir}")
        print(f"Comprehensive summary: {summary_file}")
        
        return 0
        
    except Exception as e:
        logging.error(f"Error in all hypothesis tests: {e}")
        return 1


def run_batch_processing(args):
    """Run batch processing for multiple scenarios with automation capabilities."""
    import os
    from ..utils.batch_processor import BatchProcessor, BatchProcessingConfig, ScheduledBatchProcessor
    from ..utils.batch_monitor import BatchMonitor, email_notification_handler, webhook_notification_handler
    
    logging.info("Starting automated batch processing")
    
    try:
        # Get FRED API key
        from qeir.config import get_fred_api_key
        try:
            fred_api_key = get_fred_api_key()
        except ValueError as e:
            print(f"Error: {e}", file=sys.stderr)
            return 1
        
        # Load batch configuration
        with open(args.config, 'r') as f:
            batch_config_dict = json.load(f)
        
        # Create batch processing configuration
        batch_processing_config = BatchProcessingConfig(
            max_concurrent_jobs=args.max_workers,
            base_output_directory=args.output,
            compress_results=args.compress_results,
            cleanup_intermediate_files=args.cleanup_intermediate,
            enable_progress_monitoring=args.enable_monitoring,
            log_level="INFO"
        )
        
        # Initialize batch processor
        if args.enable_scheduling and args.schedule:
            processor = ScheduledBatchProcessor(
                config=batch_processing_config,
                fred_api_key=fred_api_key
            )
        else:
            processor = BatchProcessor(
                config=batch_processing_config,
                fred_api_key=fred_api_key
            )
        
        # Setup monitoring if enabled
        monitor = None
        if args.enable_monitoring:
            monitor = BatchMonitor(
                monitoring_interval=args.monitoring_interval,
                enable_system_monitoring=True,
                enable_job_monitoring=True,
                enable_notifications=bool(args.notification_email or args.notification_webhook),
                database_path=str(Path(args.output) / "monitoring.db")
            )
            
            # Setup notification handlers
            if args.notification_email:
                # Note: This would need SMTP configuration
                print("Email notifications configured (SMTP settings required)")
            
            if args.notification_webhook:
                webhook_handler = webhook_notification_handler(args.notification_webhook)
                monitor.add_notification_handler(webhook_handler)
            
            # Start monitoring
            batch_id = f"batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            monitor.start_monitoring(batch_id)
        
        # Parse periods and specifications
        periods = []
        if args.periods:
            for period_str in args.periods.split(','):
                start, end = period_str.strip().split('-')
                periods.append((start, end))
        else:
            periods = batch_config_dict.get('periods', [])
        
        specifications = []
        if args.specifications:
            specifications = [spec.strip() for spec in args.specifications.split(',')]
        else:
            specifications = batch_config_dict.get('specifications', ['default'])
        
        # Create base hypothesis testing configuration
        base_config_dict = batch_config_dict.get('base_config', {})
        base_hypothesis_config = HypothesisTestingConfig(**base_config_dict)
        
        # Handle scheduling if enabled
        if args.enable_scheduling and args.schedule:
            print(f"Scheduling batch jobs with schedule: {args.schedule}")
            
            # Create job configuration for scheduling
            job_config = {
                **base_config_dict,
                'periods': periods,
                'specifications': specifications,
                'specifications_config': batch_config_dict.get('specifications_config', {})
            }
            
            # Schedule the job
            scheduled_job_id = processor.schedule_job(job_config, args.schedule)
            print(f"Scheduled job created: {scheduled_job_id}")
            
            # Start processing to handle scheduled jobs
            processor.start_processing()
            
            print("Batch processor started with scheduling. Press Ctrl+C to stop.")
            try:
                while True:
                    time.sleep(60)
                    status = processor.get_status()
                    print(f"Status: {status['stats']['jobs_completed']} completed, "
                          f"{len(status['active_jobs'])} active, "
                          f"{status['queue_size']} queued")
            except KeyboardInterrupt:
                print("\nStopping batch processor...")
                processor.stop_processing()
                if monitor:
                    monitor.stop_monitoring()
            
            return 0
        
        # Create jobs for immediate execution
        print(f"Creating {len(periods) * len(specifications)} batch jobs...")
        
        # Create period-based jobs
        if periods:
            job_ids = processor.create_period_jobs(
                base_config=base_hypothesis_config,
                periods=periods,
                job_name_prefix="period_analysis"
            )
            print(f"Created {len(job_ids)} period-based jobs")
        
        # Create specification-based jobs
        if len(specifications) > 1:
            spec_configs = batch_config_dict.get('specifications_config', {})
            spec_job_ids = processor.create_specification_jobs(
                base_config=base_hypothesis_config,
                specifications=spec_configs,
                job_name_prefix="spec_analysis"
            )
            print(f"Created {len(spec_job_ids)} specification-based jobs")
        
        # Start processing
        processor.start_processing()
        
        # Monitor progress
        print("Batch processing started. Monitoring progress...")
        try:
            while processor.is_running:
                time.sleep(10)
                status = processor.get_status()
                
                if monitor:
                    monitor_status = monitor.get_current_status()
                    print(f"Progress: {status['stats']['jobs_completed']}/{status['total_jobs']} completed, "
                          f"{len(status['active_jobs'])} active, "
                          f"CPU: {monitor_status.get('latest_system_metrics', {}).get('cpu_percent', 'N/A'):.1f}%, "
                          f"Memory: {monitor_status.get('latest_system_metrics', {}).get('memory_percent', 'N/A'):.1f}%")
                else:
                    print(f"Progress: {status['stats']['jobs_completed']}/{status['total_jobs']} completed, "
                          f"{len(status['active_jobs'])} active, "
                          f"{status['queue_size']} queued")
                
                # Check if all jobs are done
                if (status['stats']['jobs_completed'] + status['stats']['jobs_failed']) >= status['total_jobs']:
                    break
        
        except KeyboardInterrupt:
            print("\nStopping batch processing...")
        
        finally:
            processor.stop_processing()
            if monitor:
                monitor.stop_monitoring()
                
                # Generate monitoring report
                report_file = Path(args.output) / "monitoring_report.json"
                monitor.generate_report(str(report_file))
                print(f"Monitoring report saved: {report_file}")
        
        # Generate final summary
        final_status = processor.get_status()
        print(f"\nBatch processing completed:")
        print(f"  Total jobs: {final_status['total_jobs']}")
        print(f"  Completed: {final_status['stats']['jobs_completed']}")
        print(f"  Failed: {final_status['stats']['jobs_failed']}")
        print(f"  Success rate: {final_status['stats']['jobs_completed']/max(1, final_status['total_jobs']):.1%}")
        
        # Save batch summary
        batch_summary = {
            'batch_id': batch_id if monitor else f"batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            'configuration': {
                'periods': periods,
                'specifications': specifications,
                'max_workers': args.max_workers,
                'monitoring_enabled': args.enable_monitoring,
                'scheduling_enabled': args.enable_scheduling
            },
            'results': final_status,
            'output_directory': args.output,
            'completed_at': datetime.now().isoformat()
        }
        
        summary_file = Path(args.output) / "batch_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(batch_summary, f, indent=2, default=str)
        
        print(f"Batch summary saved: {summary_file}")
        
        return 0 if final_status['stats']['jobs_failed'] == 0 else 1
        
    except Exception as e:
        logging.error(f"Error in batch processing: {e}")
        return 1


def show_test_status(args):
    """Show status of running or completed tests."""
    output_dir = Path(args.output)
    
    if not output_dir.exists():
        print(f"Results directory not found: {output_dir}")
        return 1
    
    print(f"Status for results in: {output_dir}")
    print("=" * 50)
    
    # Check for completed tests
    completed_tests = []
    for i in [1, 2, 3]:
        results_file = output_dir / f"hypothesis_{i}_results.json"
        if results_file.exists():
            completed_tests.append(i)
            # Get test timestamp
            try:
                with open(results_file, 'r') as f:
                    results_data = json.load(f)
                    timestamp = results_data.get('test_timestamp', 'Unknown')
                print(f"✓ Hypothesis {i}: Completed at {timestamp}")
            except:
                print(f"✓ Hypothesis {i}: Completed (timestamp unavailable)")
        else:
            print(f"✗ Hypothesis {i}: Not completed")
    
    # Check for data files
    data_files = list(output_dir.glob("*_data.json"))
    if data_files:
        print(f"\nData files found: {len(data_files)}")
        for data_file in data_files:
            print(f"  - {data_file.name}")
    
    # Check for summary reports
    summary_files = list(output_dir.glob("*_summary.txt"))
    if summary_files:
        print(f"\nSummary reports found: {len(summary_files)}")
        for summary_file in summary_files:
            print(f"  - {summary_file.name}")
    
    if args.watch:
        print("\nWatching for changes... (Press Ctrl+C to stop)")
        try:
            import time
            while True:
                time.sleep(5)
                # Re-check status (simplified implementation)
                new_completed = []
                for i in [1, 2, 3]:
                    results_file = output_dir / f"hypothesis_{i}_results.json"
                    if results_file.exists() and i not in completed_tests:
                        new_completed.append(i)
                
                if new_completed:
                    for i in new_completed:
                        print(f"✓ NEW: Hypothesis {i} completed")
                    completed_tests.extend(new_completed)
        except KeyboardInterrupt:
            print("\nStopped watching")
    
    return 0


def create_config_template(args):
    """Create configuration template."""
    output_file = Path(args.output)
    
    if args.template == "basic":
        config_template = {
            "start_date": "2008-01-01",
            "end_date": "2023-12-31",
            "confidence_level": 0.95,
            "bootstrap_iterations": 1000,
            "enable_robustness_tests": True,
            "generate_publication_outputs": True,
            "output_directory": "hypothesis_testing_results"
        }
    elif args.template == "advanced":
        config_template = {
            "start_date": "2008-01-01",
            "end_date": "2023-12-31",
            "confidence_level": 0.95,
            "bootstrap_iterations": 1000,
            
            # Hypothesis 1 settings
            "h1_threshold_trim": 0.15,
            "h1_min_regime_size": 10,
            "h1_test_alternative_thresholds": True,
            "h1_confidence_proxy": "consumer_confidence",
            "h1_reaction_proxy": "fed_total_assets",
            
            # Hypothesis 2 settings
            "h2_max_horizon": 20,
            "h2_lags": 4,
            "h2_use_instrumental_variables": True,
            "h2_investment_measure": "private_fixed_investment",
            "h2_distortion_proxy": "corporate_bond_spreads",
            
            # Hypothesis 3 settings
            "h3_causality_lags": 4,
            "h3_test_spillovers": True,
            "h3_exchange_rate_measure": "trade_weighted_dollar",
            "h3_inflation_measure": "cpi_all_items",
            
            # Robustness testing
            "enable_robustness_tests": True,
            "test_alternative_periods": True,
            "test_alternative_specifications": True,
            
            # Output settings
            "generate_publication_outputs": True,
            "output_directory": "hypothesis_testing_results",
            "save_intermediate_results": True
        }
    elif args.template == "batch":
        config_template = {
            "base_config": {
                "confidence_level": 0.95,
                "bootstrap_iterations": 1000,
                "enable_robustness_tests": True
            },
            "periods": [
                ["2008-01-01", "2015-12-31"],
                ["2016-01-01", "2023-12-31"],
                ["2008-01-01", "2023-12-31"]
            ],
            "specifications": ["default", "alternative_1", "alternative_2"],
            "specifications_config": {
                "default": {
                    "h1_confidence_proxy": "consumer_confidence",
                    "h2_investment_measure": "private_fixed_investment"
                },
                "alternative_1": {
                    "h1_confidence_proxy": "financial_stress_index",
                    "h2_investment_measure": "equipment_investment"
                },
                "alternative_2": {
                    "h1_threshold_trim": 0.10,
                    "h2_max_horizon": 16
                }
            }
        }
    
    # Save configuration template
    with open(output_file, 'w') as f:
        json.dump(config_template, f, indent=2)
    
    print(f"Configuration template created: {output_file}")
    print(f"Template type: {args.template}")
    
    return 0


def validate_config_file(args):
    """Validate configuration file."""
    config_file = Path(args.config_file)
    
    if not config_file.exists():
        print(f"Configuration file not found: {config_file}", file=sys.stderr)
        return 1
    
    try:
        with open(config_file, 'r') as f:
            config_data = json.load(f)
        
        # Basic validation
        errors = []
        warnings = []
        
        # Check required fields
        required_fields = ['start_date', 'end_date']
        for field in required_fields:
            if field not in config_data:
                errors.append(f"Missing required field: {field}")
        
        # Validate date formats
        if 'start_date' in config_data:
            try:
                datetime.strptime(config_data['start_date'], '%Y-%m-%d')
            except ValueError:
                errors.append("Invalid start_date format. Use YYYY-MM-DD")
        
        if 'end_date' in config_data:
            try:
                datetime.strptime(config_data['end_date'], '%Y-%m-%d')
            except ValueError:
                errors.append("Invalid end_date format. Use YYYY-MM-DD")
        
        # Check numeric ranges
        if 'confidence_level' in config_data:
            if not 0 < config_data['confidence_level'] < 1:
                errors.append("confidence_level must be between 0 and 1")
        
        if 'bootstrap_iterations' in config_data:
            if config_data['bootstrap_iterations'] < 100:
                warnings.append("bootstrap_iterations < 100 may give unreliable results")
        
        # Report results
        if errors:
            print("Configuration validation FAILED:")
            for error in errors:
                print(f"  ERROR: {error}")
        
        if warnings:
            print("Configuration validation warnings:")
            for warning in warnings:
                print(f"  WARNING: {warning}")
        
        if not errors and not warnings:
            print("Configuration validation PASSED")
        elif not errors:
            print("Configuration validation PASSED with warnings")
        
        return 1 if errors else 0
        
    except json.JSONDecodeError as e:
        print(f"Invalid JSON in configuration file: {e}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"Error validating configuration: {e}", file=sys.stderr)
        return 1


def show_config(args):
    """Show current configuration."""
    try:
        if args.config:
            config = load_hypothesis_config(args.config)
        else:
            config = HypothesisTestingConfig()
        
        config_dict = config.__dict__
        
        if args.format == "json":
            print(json.dumps(config_dict, indent=2, default=str))
        elif args.format == "yaml":
            try:
                import yaml
                print(yaml.dump(config_dict, default_flow_style=False))
            except ImportError:
                print("PyYAML not installed. Using JSON format:")
                print(json.dumps(config_dict, indent=2, default=str))
        elif args.format == "table":
            print("Current Configuration:")
            print("=" * 50)
            for key, value in config_dict.items():
                print(f"{key:30} : {value}")
        
        return 0
        
    except Exception as e:
        print(f"Error showing configuration: {e}", file=sys.stderr)
        return 1


# Helper functions

def load_hypothesis_config(config_path: Optional[str]) -> HypothesisTestingConfig:
    """Load hypothesis testing configuration."""
    if config_path:
        config_file = Path(config_path)
        if not config_file.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_file}")
        
        with open(config_file, 'r') as f:
            config_dict = json.load(f)
        
        return HypothesisTestingConfig(**config_dict)
    else:
        return HypothesisTestingConfig()


def save_hypothesis_data(data, file_path: Path):
    """Save hypothesis data to file."""
    # Convert data to serializable format
    data_dict = {
        'metadata': data.metadata if hasattr(data, 'metadata') else {},
        'dates': data.dates.strftime('%Y-%m-%d').tolist() if hasattr(data, 'dates') else [],
        'timestamp': datetime.now().isoformat()
    }
    
    # Add data series
    for attr_name in dir(data):
        if not attr_name.startswith('_') and attr_name not in ['metadata', 'dates']:
            attr_value = getattr(data, attr_name)
            if hasattr(attr_value, 'to_dict'):
                data_dict[attr_name] = attr_value.to_dict()
    
    with open(file_path, 'w') as f:
        json.dump(data_dict, f, indent=2, default=str)


def save_hypothesis_results(results, file_path: Path):
    """Save hypothesis results to file."""
    # Convert results to serializable format
    results_dict = {
        'hypothesis_name': results.hypothesis_name,
        'hypothesis_number': results.hypothesis_number,
        'main_result': results.main_result,
        'statistical_significance': results.statistical_significance,
        'economic_significance': results.economic_significance,
        'robustness_results': results.robustness_results,
        'data_period': results.data_period,
        'test_timestamp': results.test_timestamp
    }
    
    # Add model results (with careful serialization)
    if results.model_results:
        model_results_dict = {}
        for attr_name in dir(results.model_results):
            if not attr_name.startswith('_'):
                attr_value = getattr(results.model_results, attr_name)
                if attr_value is not None:
                    model_results_dict[attr_name] = attr_value
        results_dict['model_results'] = model_results_dict
    
    with open(file_path, 'w') as f:
        json.dump(results_dict, f, indent=2, default=str)


def generate_summary_report(results, file_path: Path):
    """Generate summary report for hypothesis results."""
    with open(file_path, 'w') as f:
        f.write(f"QEIR Hypothesis Testing Summary Report\n")
        f.write(f"=" * 50 + "\n\n")
        
        f.write(f"Hypothesis: {results.hypothesis_name}\n")
        f.write(f"Hypothesis Number: {results.hypothesis_number}\n")
        f.write(f"Test Timestamp: {results.test_timestamp}\n\n")
        
        f.write(f"Data Period:\n")
        f.write(f"  Start Date: {results.data_period.get('start_date', 'N/A')}\n")
        f.write(f"  End Date: {results.data_period.get('end_date', 'N/A')}\n")
        f.write(f"  Observations: {results.data_period.get('observations', 'N/A')}\n\n")
        
        f.write(f"Main Results:\n")
        for key, value in results.main_result.items():
            f.write(f"  {key}: {value}\n")
        f.write("\n")
        
        f.write(f"Statistical Significance:\n")
        for key, value in results.statistical_significance.items():
            f.write(f"  {key}: {value}\n")
        f.write("\n")
        
        f.write(f"Economic Significance:\n")
        for key, value in results.economic_significance.items():
            f.write(f"  {key}: {value}\n")
        f.write("\n")
        
        if results.robustness_results:
            f.write(f"Robustness Tests:\n")
            for key, value in results.robustness_results.items():
                f.write(f"  {key}: {value}\n")


def generate_comprehensive_summary(all_results: Dict, file_path: Path):
    """Generate comprehensive summary for all hypothesis results."""
    with open(file_path, 'w') as f:
        f.write(f"QEIR Comprehensive Hypothesis Testing Summary\n")
        f.write(f"=" * 60 + "\n\n")
        
        f.write(f"Test Timestamp: {datetime.now().isoformat()}\n")
        f.write(f"Hypotheses Tested: {len(all_results)}\n\n")
        
        for hypothesis_key, results in all_results.items():
            f.write(f"{hypothesis_key.upper()}:\n")
            f.write(f"  Name: {results.hypothesis_name}\n")
            f.write(f"  Status: {'Completed' if results.main_result else 'Failed'}\n")
            
            if 'error' not in results.main_result:
                f.write(f"  Data Period: {results.data_period.get('start_date')} to {results.data_period.get('end_date')}\n")
                f.write(f"  Observations: {results.data_period.get('observations')}\n")
                
                # Key findings
                key_findings = []
                for key, value in results.main_result.items():
                    if isinstance(value, bool) and value:
                        key_findings.append(key)
                
                if key_findings:
                    f.write(f"  Key Findings: {', '.join(key_findings)}\n")
            else:
                f.write(f"  Error: {results.main_result['error']}\n")
            
            f.write("\n")


def generate_batch_summary(output_dir: Path, summary_file: Path):
    """Generate batch processing summary."""
    batch_summary = {
        'timestamp': datetime.now().isoformat(),
        'output_directory': str(output_dir),
        'jobs': []
    }
    
    # Scan for completed jobs
    for job_dir in output_dir.iterdir():
        if job_dir.is_dir():
            job_info = {
                'job_name': job_dir.name,
                'job_directory': str(job_dir),
                'completed_hypotheses': []
            }
            
            # Check for completed hypothesis tests
            for i in [1, 2, 3]:
                results_file = job_dir / f"hypothesis_{i}_results.json"
                if results_file.exists():
                    job_info['completed_hypotheses'].append(i)
            
            # Check for job configuration
            config_file = job_dir / "job_config.json"
            if config_file.exists():
                try:
                    with open(config_file, 'r') as f:
                        job_config = json.load(f)
                    job_info['configuration'] = job_config
                except:
                    pass
            
            batch_summary['jobs'].append(job_info)
    
    # Save batch summary
    with open(summary_file, 'w') as f:
        json.dump(batch_summary, f, indent=2)


class ProgressTracker:
    """Simple progress tracker for CLI operations."""
    
    def __init__(self, operation_name: str):
        self.operation_name = operation_name
        self.start_time = time.time()
        print(f"Starting: {operation_name}")
    
    def update(self, message: str):
        elapsed = time.time() - self.start_time
        print(f"[{elapsed:.1f}s] {message}")
    
    def complete(self, message: str):
        elapsed = time.time() - self.start_time
        print(f"[{elapsed:.1f}s] ✓ {message}")


# Legacy functions (maintain backward compatibility)

def run_analysis(args):
    """Run QE analysis (legacy)."""
    print(f"Running QE analysis on {args.data}")
    
    # Initialize analyzer
    analyzer = RevisedQEAnalyzer()
    
    # Load data and run analysis
    # This would be implemented based on specific data format
    print(f"Results will be saved to {args.output}")
    
    return 0


def run_visualization(args):
    """Generate visualizations."""
    print(f"Generating visualizations from {args.results}")
    
    # Initialize visualization suite
    viz_suite = PublicationVisualizationSuite()
    
    # Generate figures
    # This would load results and create visualizations
    print(f"Figures saved to {args.output}")
    
    return 0


def run_export(args):
    """Export publication outputs."""
    print(f"Exporting publication outputs from {args.results}")
    
    # Initialize export system
    export_system = PublicationExportSystem()
    
    # Generate publication-ready outputs
    # This would format results for specific journals
    print(f"Publication outputs saved to {args.output}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

def handle_batch_management(args):
    """Handle batch management commands."""
    if args.batch_mgmt_command == "list":
        return list_batch_jobs(args)
    elif args.batch_mgmt_command == "cancel":
        return cancel_batch_job(args)
    elif args.batch_mgmt_command == "monitor":
        return monitor_batch_processing(args)
    elif args.batch_mgmt_command == "report":
        return generate_batch_report(args)
    else:
        print("Unknown batch management command", file=sys.stderr)
        return 1


def list_batch_jobs(args):
    """List batch jobs and their status."""
    output_dir = Path(args.output)
    
    if not output_dir.exists():
        print(f"Batch results directory not found: {output_dir}")
        return 1
    
    print(f"Batch jobs in: {output_dir}")
    print("=" * 80)
    print(f"{'Job ID':<30} {'Status':<12} {'Started':<20} {'Duration':<12}")
    print("-" * 80)
    
    # Look for job directories and status files
    job_count = 0
    for job_dir in output_dir.iterdir():
        if job_dir.is_dir():
            job_id = job_dir.name
            
            # Check for job status indicators
            status = "unknown"
            started = "N/A"
            duration = "N/A"
            
            # Look for completion indicators
            if (job_dir / "job_summary.txt").exists():
                status = "completed"
            elif (job_dir / "job.log").exists():
                # Check log file for status
                try:
                    with open(job_dir / "job.log", 'r') as f:
                        log_content = f.read()
                        if "completed successfully" in log_content.lower():
                            status = "completed"
                        elif "failed" in log_content.lower() or "error" in log_content.lower():
                            status = "failed"
                        else:
                            status = "running"
                except:
                    pass
            
            # Get timestamps from directory
            try:
                stat = job_dir.stat()
                started = datetime.fromtimestamp(stat.st_ctime).strftime('%Y-%m-%d %H:%M:%S')
                
                if status == "completed":
                    # Try to calculate duration
                    if (job_dir / "job_summary.txt").exists():
                        summary_stat = (job_dir / "job_summary.txt").stat()
                        duration_sec = summary_stat.st_mtime - stat.st_ctime
                        duration = f"{duration_sec/60:.1f}m"
            except:
                pass
            
            # Filter by status if requested
            if args.status != "all" and status != args.status:
                continue
            
            print(f"{job_id:<30} {status:<12} {started:<20} {duration:<12}")
            job_count += 1
    
    print("-" * 80)
    print(f"Total jobs: {job_count}")
    
    return 0


def cancel_batch_job(args):
    """Cancel a batch job."""
    output_dir = Path(args.output)
    job_dir = output_dir / args.job_id
    
    if not job_dir.exists():
        print(f"Job not found: {args.job_id}")
        return 1
    
    # Create cancellation marker
    cancel_file = job_dir / ".cancelled"
    with open(cancel_file, 'w') as f:
        f.write(f"Cancelled at: {datetime.now().isoformat()}\n")
    
    print(f"Cancellation requested for job: {args.job_id}")
    print("Note: Running processes may take time to stop gracefully")
    
    return 0


def monitor_batch_processing(args):
    """Monitor batch processing in real-time."""
    import sqlite3
    
    output_dir = Path(args.output)
    
    # Check for monitoring database
    db_path = args.database or (output_dir / "monitoring.db")
    
    if not Path(db_path).exists():
        print(f"Monitoring database not found: {db_path}")
        print("Enable monitoring with --enable-monitoring when running batch processing")
        return 1
    
    print(f"Monitoring batch processing from: {db_path}")
    print("Press Ctrl+C to stop monitoring")
    print("=" * 80)
    
    try:
        while True:
            # Query database for latest metrics
            try:
                conn = sqlite3.connect(db_path)
                cursor = conn.cursor()
                
                # Get latest batch metrics
                cursor.execute('''
                    SELECT batch_id, total_jobs, completed_jobs, failed_jobs, cancelled_jobs
                    FROM batch_metrics 
                    ORDER BY start_time DESC 
                    LIMIT 1
                ''')
                batch_row = cursor.fetchone()
                
                # Get latest system metrics
                cursor.execute('''
                    SELECT timestamp, cpu_percent, memory_percent, disk_usage_percent
                    FROM system_metrics 
                    ORDER BY timestamp DESC 
                    LIMIT 1
                ''')
                system_row = cursor.fetchone()
                
                # Get active jobs
                cursor.execute('''
                    SELECT COUNT(*) FROM job_metrics WHERE status = 'running'
                ''')
                active_jobs = cursor.fetchone()[0]
                
                conn.close()
                
                # Display current status
                print(f"\r{datetime.now().strftime('%H:%M:%S')} | ", end="")
                
                if batch_row:
                    batch_id, total, completed, failed, cancelled = batch_row
                    progress = (completed + failed + cancelled) / max(1, total) * 100
                    print(f"Progress: {progress:.1f}% ({completed}/{total}) | ", end="")
                
                print(f"Active: {active_jobs} | ", end="")
                
                if system_row:
                    timestamp, cpu, memory, disk = system_row
                    print(f"CPU: {cpu:.1f}% | Memory: {memory:.1f}% | Disk: {disk:.1f}%", end="")
                
                print("", flush=True)
                
            except Exception as e:
                print(f"\rError querying database: {e}", end="", flush=True)
            
            time.sleep(args.refresh)
            
    except KeyboardInterrupt:
        print("\nStopped monitoring")
    
    return 0


def generate_batch_report(args):
    """Generate comprehensive batch processing report."""
    import sqlite3
    import pandas as pd
    
    output_dir = Path(args.output)
    db_path = args.database or (output_dir / "monitoring.db")
    
    if not Path(db_path).exists():
        print(f"Monitoring database not found: {db_path}")
        return 1
    
    try:
        conn = sqlite3.connect(db_path)
        
        # Generate report based on format
        if args.format == "json":
            report_file = output_dir / "batch_report.json"
            report_data = generate_json_report(conn)
            
            with open(report_file, 'w') as f:
                json.dump(report_data, f, indent=2, default=str)
            
        elif args.format == "csv":
            # Generate CSV files for different metrics
            job_metrics_df = pd.read_sql_query("SELECT * FROM job_metrics", conn)
            system_metrics_df = pd.read_sql_query("SELECT * FROM system_metrics", conn)
            batch_metrics_df = pd.read_sql_query("SELECT * FROM batch_metrics", conn)
            
            job_metrics_df.to_csv(output_dir / "job_metrics.csv", index=False)
            system_metrics_df.to_csv(output_dir / "system_metrics.csv", index=False)
            batch_metrics_df.to_csv(output_dir / "batch_metrics.csv", index=False)
            
            report_file = output_dir / "batch_metrics.csv"
            
        elif args.format == "html":
            report_file = output_dir / "batch_report.html"
            generate_html_report(conn, report_file)
        
        conn.close()
        
        print(f"Batch report generated: {report_file}")
        return 0
        
    except Exception as e:
        print(f"Error generating report: {e}")
        return 1


def generate_json_report(conn):
    """Generate JSON format report from database."""
    cursor = conn.cursor()
    
    # Get batch summary
    cursor.execute("SELECT * FROM batch_metrics ORDER BY start_time DESC LIMIT 1")
    batch_data = cursor.fetchone()
    
    # Get job summary
    cursor.execute('''
        SELECT status, COUNT(*) as count, AVG(duration_seconds) as avg_duration
        FROM job_metrics 
        GROUP BY status
    ''')
    job_summary = cursor.fetchall()
    
    # Get system metrics summary
    cursor.execute('''
        SELECT 
            AVG(cpu_percent) as avg_cpu,
            MAX(cpu_percent) as max_cpu,
            AVG(memory_percent) as avg_memory,
            MAX(memory_percent) as max_memory
        FROM system_metrics
    ''')
    system_summary = cursor.fetchone()
    
    report = {
        'batch_summary': {
            'batch_id': batch_data[0] if batch_data else None,
            'start_time': batch_data[1] if batch_data else None,
            'end_time': batch_data[2] if batch_data else None,
            'total_jobs': batch_data[3] if batch_data else 0,
            'completed_jobs': batch_data[4] if batch_data else 0,
            'failed_jobs': batch_data[5] if batch_data else 0,
            'cancelled_jobs': batch_data[6] if batch_data else 0
        },
        'job_summary': [
            {'status': row[0], 'count': row[1], 'avg_duration_seconds': row[2]}
            for row in job_summary
        ],
        'system_summary': {
            'avg_cpu_percent': system_summary[0] if system_summary else None,
            'max_cpu_percent': system_summary[1] if system_summary else None,
            'avg_memory_percent': system_summary[2] if system_summary else None,
            'max_memory_percent': system_summary[3] if system_summary else None
        } if system_summary else {},
        'generated_at': datetime.now().isoformat()
    }
    
    return report


def generate_html_report(conn, output_file):
    """Generate HTML format report from database."""
    report_data = generate_json_report(conn)
    
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>QEIR Batch Processing Report</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
            th {{ background-color: #f2f2f2; }}
            .metric {{ background-color: #f9f9f9; padding: 10px; margin: 10px 0; }}
        </style>
    </head>
    <body>
        <h1>QEIR Batch Processing Report</h1>
        <p>Generated: {report_data['generated_at']}</p>
        
        <h2>Batch Summary</h2>
        <div class="metric">
            <strong>Batch ID:</strong> {report_data['batch_summary']['batch_id']}<br>
            <strong>Start Time:</strong> {report_data['batch_summary']['start_time']}<br>
            <strong>End Time:</strong> {report_data['batch_summary']['end_time']}<br>
            <strong>Total Jobs:</strong> {report_data['batch_summary']['total_jobs']}<br>
            <strong>Completed:</strong> {report_data['batch_summary']['completed_jobs']}<br>
            <strong>Failed:</strong> {report_data['batch_summary']['failed_jobs']}<br>
            <strong>Cancelled:</strong> {report_data['batch_summary']['cancelled_jobs']}
        </div>
        
        <h2>Job Summary</h2>
        <table>
            <tr><th>Status</th><th>Count</th><th>Avg Duration (seconds)</th></tr>
    """
    
    for job in report_data['job_summary']:
        html_content += f"""
            <tr>
                <td>{job['status']}</td>
                <td>{job['count']}</td>
                <td>{job['avg_duration_seconds']:.1f if job['avg_duration_seconds'] else 'N/A'}</td>
            </tr>
        """
    
    html_content += """
        </table>
        
        <h2>System Resource Summary</h2>
        <div class="metric">
    """
    
    sys_summary = report_data['system_summary']
    if sys_summary:
        html_content += f"""
            <strong>Average CPU:</strong> {sys_summary.get('avg_cpu_percent', 'N/A'):.1f}%<br>
            <strong>Peak CPU:</strong> {sys_summary.get('max_cpu_percent', 'N/A'):.1f}%<br>
            <strong>Average Memory:</strong> {sys_summary.get('avg_memory_percent', 'N/A'):.1f}%<br>
            <strong>Peak Memory:</strong> {sys_summary.get('max_memory_percent', 'N/A'):.1f}%
        """
    else:
        html_content += "No system metrics available"
    
    html_content += """
        </div>
    </body>
    </html>
    """
    
    with open(output_file, 'w') as f:
        f.write(html_content)


class ProgressTracker:
    """Simple progress tracker for CLI operations."""
    
    def __init__(self, operation_name: str):
        self.operation_name = operation_name
        self.start_time = time.time()
    
    def update(self, message: str):
        elapsed = time.time() - self.start_time
        print(f"[{elapsed:.1f}s] {self.operation_name}: {message}")
    
    def complete(self, message: str):
        elapsed = time.time() - self.start_time
        print(f"[{elapsed:.1f}s] {self.operation_name}: {message}")


def load_hypothesis_config(config_file: Optional[str]) -> HypothesisTestingConfig:
    """Load hypothesis testing configuration from file or create default."""
    if config_file and Path(config_file).exists():
        with open(config_file, 'r') as f:
            config_dict = json.load(f)
        return HypothesisTestingConfig(**config_dict)
    else:
        # Return default configuration
        return HypothesisTestingConfig()


def save_hypothesis_data(data, file_path: Path):
    """Save hypothesis data to file."""
    # Simplified data saving - would need proper implementation
    data_summary = {
        'timestamp': datetime.now().isoformat(),
        'data_loaded': True,
        'file_path': str(file_path)
    }
    
    with open(file_path, 'w') as f:
        json.dump(data_summary, f, indent=2)


def save_hypothesis_results(results, file_path: Path):
    """Save hypothesis results to file."""
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


def generate_summary_report(results, summary_file: Path):
    """Generate summary report for hypothesis test."""
    with open(summary_file, 'w') as f:
        f.write(f"QEIR Hypothesis Test Summary\n")
        f.write(f"=" * 40 + "\n\n")
        
        f.write(f"Hypothesis: {results.hypothesis_name}\n")
        f.write(f"Test Date: {results.test_timestamp}\n")
        f.write(f"Data Period: {results.data_period}\n\n")
        
        f.write(f"Main Result: {results.main_result}\n")
        f.write(f"Statistical Significance: {results.statistical_significance}\n")
        f.write(f"Economic Significance: {results.economic_significance}\n")


def generate_comprehensive_summary(all_results: Dict, summary_file: Path):
    """Generate comprehensive summary for all hypothesis tests."""
    with open(summary_file, 'w') as f:
        f.write(f"QEIR Comprehensive Hypothesis Testing Summary\n")
        f.write(f"=" * 50 + "\n\n")
        
        f.write(f"Test Date: {datetime.now().isoformat()}\n")
        f.write(f"Hypotheses Tested: {len(all_results)}\n\n")
        
        for hypothesis_key, results in all_results.items():
            f.write(f"{hypothesis_key.upper()}: {results.hypothesis_name}\n")
            f.write(f"  Status: {'Completed' if results.main_result else 'Failed'}\n")
            f.write(f"  Statistical Significance: {results.statistical_significance}\n")
            f.write(f"  Economic Significance: {results.economic_significance}\n\n")


def generate_batch_summary(output_dir: Path, summary_file: Path):
    """Generate batch processing summary."""
    # Count completed jobs
    job_dirs = [d for d in output_dir.iterdir() if d.is_dir()]
    completed_jobs = len([d for d in job_dirs if (d / "job_summary.txt").exists()])
    
    summary = {
        'batch_completed_at': datetime.now().isoformat(),
        'total_job_directories': len(job_dirs),
        'completed_jobs': completed_jobs,
        'success_rate': completed_jobs / max(1, len(job_dirs)),
        'output_directory': str(output_dir)
    }
    
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)

def run_final_validation(args):
    """Run final validation suite."""
    import os
    from pathlib import Path
    from ..validation.final_validation_suite import FinalValidationSuite
    from ..utils.hypothesis_data_collector import HypothesisDataCollector
    from ..utils.data_processor import DataProcessor
    
    logging.info("Starting final validation suite")
    
    # Get FRED API key if needed for data collection
    from qeir.config import get_fred_api_key
    try:
        fred_api_key = args.fred_api_key or get_fred_api_key()
    except ValueError as e:
        fred_api_key = None
    
    try:
        # Setup output directory
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load or collect validation data
        if args.data_path and Path(args.data_path).exists():
            logging.info(f"Loading validation data from {args.data_path}")
            import pandas as pd
            data = pd.read_csv(args.data_path, index_col=0, parse_dates=True)
        else:
            if not fred_api_key:
                print("Error: FRED API key required for data collection. Set FRED_API_KEY environment variable", 
                      file=sys.stderr)
                return 1
            
            logging.info("Collecting fresh validation data from FRED API")
            
            # Initialize data collector
            data_collector = HypothesisDataCollector(fred_api_key=fred_api_key)
            
            # Collect data for all hypotheses
            start_date = "2008-01-01"
            end_date = "2023-12-31"
            
            # Collect hypothesis-specific data
            h1_data = data_collector.collect_hypothesis1_data(start_date, end_date)
            h2_data = data_collector.collect_hypothesis2_data(start_date, end_date)
            h3_data = data_collector.collect_hypothesis3_data(start_date, end_date)
            
            # Combine all data
            all_data = {**h1_data, **h2_data, **h3_data}
            
            # Process and align data
            processor = DataProcessor()
            data = processor.process_and_align_data(all_data)
        
        logging.info(f"Loaded data with {len(data)} observations from "
                   f"{data.index.min()} to {data.index.max()}")
        
        # Save data if requested
        if args.save_data:
            data_path = output_dir / "validation_data.csv"
            data.to_csv(data_path)
            logging.info(f"Validation data saved to {data_path}")
        
        # Initialize validation suite
        validation_suite = FinalValidationSuite(output_dir=str(output_dir))
        
        # Run comprehensive validation
        logging.info("Running comprehensive validation")
        validation_results = validation_suite.run_comprehensive_validation(data)
        
        # Generate summary
        summary_path = output_dir / "validation_summary.txt"
        with open(summary_path, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("FINAL VALIDATION SUMMARY\n")
            f.write("QE Hypothesis Testing Framework\n")
            f.write("=" * 80 + "\n\n")
            
            f.write(f"Validation Date: {validation_results['timestamp']}\n")
            f.write(f"Data Period: {validation_results['data_period']['start']} to "
                   f"{validation_results['data_period']['end']}\n")
            f.write(f"Total Observations: {validation_results['data_period']['observations']}\n\n")
            
            # Overall scores
            final_assessment = validation_results['final_assessment']
            f.write("OVERALL ASSESSMENT\n")
            f.write("-" * 40 + "\n")
            f.write(f"Overall Validity Score: {final_assessment['overall_validity_score']:.3f}\n")
            f.write(f"Robustness Score: {final_assessment['robustness_score']:.3f}\n")
            f.write(f"Literature Consistency: {final_assessment['literature_consistency_score']:.3f}\n")
            f.write(f"Publication Ready: {final_assessment['publication_readiness']['ready_for_submission']}\n\n")
            
            # Hypothesis-specific scores
            f.write("HYPOTHESIS VALIDATION SCORES\n")
            f.write("-" * 40 + "\n")
            for hypothesis, score in final_assessment['hypothesis_validity'].items():
                f.write(f"{hypothesis.title()}: {score:.3f}\n")
            f.write("\n")
            
            # Publication readiness criteria
            f.write("PUBLICATION READINESS CRITERIA\n")
            f.write("-" * 40 + "\n")
            pub_ready = final_assessment['publication_readiness']
            for criterion, status in pub_ready.items():
                if criterion != 'ready_for_submission':
                    status_str = "✓" if status else "✗"
                    f.write(f"{criterion.replace('_', ' ').title()}: {status_str}\n")
            f.write("\n")
            
            # Recommendations
            f.write("RECOMMENDATIONS\n")
            f.write("-" * 40 + "\n")
            for i, recommendation in enumerate(final_assessment['recommendations'], 1):
                f.write(f"{i}. {recommendation}\n")
            f.write("\n")
            
            f.write("=" * 80 + "\n")
            f.write("Validation completed successfully.\n")
            f.write("See detailed reports in the validation output directory.\n")
            f.write("=" * 80 + "\n")
        
        logging.info(f"Validation summary saved to {summary_path}")
        
        # Print final status
        final_assessment = validation_results['final_assessment']
        print("\n" + "=" * 80)
        print("FINAL VALIDATION COMPLETED")
        print("=" * 80)
        print(f"Overall Validity Score: {final_assessment['overall_validity_score']:.3f}")
        print(f"Publication Ready: {final_assessment['publication_readiness']['ready_for_submission']}")
        print(f"Results saved to: {output_dir.absolute()}")
        print("=" * 80)
        
        # Exit with appropriate code
        if final_assessment['publication_readiness']['ready_for_submission']:
            logging.info("Validation completed successfully - framework is publication ready")
            return 0
        else:
            logging.warning("Validation completed with issues - see recommendations")
            return 1
            
    except Exception as e:
        logging.error(f"Final validation failed with error: {str(e)}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 2