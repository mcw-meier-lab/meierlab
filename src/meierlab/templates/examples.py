"""
Example templates for meierlab data processing workflows.

This module provides example templates that demonstrate how to use
and customize the template system for different use cases.
"""

from typing import Any, Dict, List
from .base import BaseDownloadTemplate
from .download import XNATDownloadTemplate


class CustomXNATDownloadTemplate(XNATDownloadTemplate):
    """
    Custom XNAT download template with additional functionality.
    
    This example shows how to extend the base XNAT template with
    custom preprocessing steps and validation.
    
    Examples
    --------
    >>> from meierlab.templates.examples import CustomXNATDownloadTemplate
    >>> template = CustomXNATDownloadTemplate()
    >>> template.run()
    """
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration with custom settings."""
        config = super()._get_default_config()
        config.update({
            'preprocess_data': True,
            'quality_check': True,
            'backup_original': True,
            'max_file_size_gb': 10,
            'allowed_scan_types': ['T1w', 'T2w', 'dwi', 'bold']
        })
        return config
    
    def _validate_config(self) -> None:
        """Validate configuration with custom rules."""
        super()._validate_config()
        
        if self.config.get('max_file_size_gb', 0) <= 0:
            raise ValueError("max_file_size_gb must be positive")
        
        if not isinstance(self.config.get('allowed_scan_types', []), list):
            raise ValueError("allowed_scan_types must be a list")
    
    def _add_arguments(self, parser):
        """Add custom command line arguments."""
        super()._add_arguments(parser)
        
        parser.add_argument(
            '--preprocess',
            action='store_true',
            help='Enable custom preprocessing'
        )
        parser.add_argument(
            '--quality-check',
            action='store_true',
            help='Enable quality checks'
        )
        parser.add_argument(
            '--backup',
            action='store_true',
            help='Backup original files'
        )
    
    def _setup(self, download_dict: Dict[str, Any]) -> None:
        """Setup with custom preprocessing."""
        # Perform custom preprocessing if enabled
        if self.config.get('preprocess_data'):
            self._preprocess_data(download_dict)
        
        # Call parent setup method
        super()._setup(download_dict)
        
        # Perform quality checks if enabled
        if self.config.get('quality_check'):
            self._quality_check(download_dict)
    
    def _preprocess_data(self, download_dict: Dict[str, Any]) -> None:
        """Custom preprocessing step."""
        self.logger.info("Performing custom preprocessing...")
        # Add your custom preprocessing logic here
        for bids_label in download_dict.keys():
            self.logger.info(f"Preprocessing {bids_label}")
    
    def _quality_check(self, download_dict: Dict[str, Any]) -> None:
        """Custom quality check step."""
        self.logger.info("Performing quality checks...")
        # Add your custom quality check logic here
        for bids_label in download_dict.keys():
            self.logger.info(f"Quality checking {bids_label}")


class MinimalDownloadTemplate(BaseDownloadTemplate):
    """
    Minimal download template for simple use cases.
    
    This example shows how to create a minimal template for simple
    download scenarios without XNAT complexity.
    
    Examples
    --------
    >>> from meierlab.templates.examples import MinimalDownloadTemplate
    >>> template = MinimalDownloadTemplate()
    >>> template.run()
    """
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get minimal default configuration."""
        return {
            'source_url': '',
            'destination_dir': './downloads',
            'file_pattern': '*',
            'max_files': 100
        }
    
    def _validate_config(self) -> None:
        """Validate minimal configuration."""
        if not self.config.get('source_url'):
            raise ValueError("source_url is required")
        
        if not self.config.get('destination_dir'):
            raise ValueError("destination_dir is required")
    
    def _add_arguments(self, parser):
        """Add minimal command line arguments."""
        parser.add_argument(
            '--source',
            help='Source URL or path',
            required=True
        )
        parser.add_argument(
            '--dest',
            help='Destination directory',
            default=self.config['destination_dir']
        )
        parser.add_argument(
            '--pattern',
            help='File pattern to match',
            default=self.config['file_pattern']
        )
    
    def run(self) -> None:
        """Run minimal download template."""
        args = self.parse_arguments()
        
        self.config.update({
            'source_url': args.source,
            'destination_dir': args.dest,
            'file_pattern': args.pattern
        })
        
        self.logger.info(f"Downloading from {self.config['source_url']}")
        self.logger.info(f"Destination: {self.config['destination_dir']}")
        
        # Create destination directory
        self.create_folder(self.config['destination_dir'])
        
        # Add your download logic here
        self.logger.info("Download completed")


class BatchProcessingTemplate(BaseDownloadTemplate):
    """
    Batch processing template for multiple data sources.
    
    This example shows how to create a template for batch processing
    multiple data sources with different configurations.
    
    Examples
    --------
    >>> from meierlab.templates.examples import BatchProcessingTemplate
    >>> template = BatchProcessingTemplate()
    >>> template.run()
    """
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get batch processing configuration."""
        return {
            'batch_config_file': '',
            'parallel_processing': False,
            'max_workers': 4,
            'retry_failed': True,
            'max_retries': 3
        }
    
    def _validate_config(self) -> None:
        """Validate batch configuration."""
        if self.config.get('max_workers', 0) <= 0:
            raise ValueError("max_workers must be positive")
        
        if self.config.get('max_retries', 0) < 0:
            raise ValueError("max_retries must be non-negative")
    
    def _add_arguments(self, parser):
        """Add batch processing arguments."""
        parser.add_argument(
            '--batch-config',
            help='Batch configuration file',
            required=True
        )
        parser.add_argument(
            '--parallel',
            action='store_true',
            help='Enable parallel processing'
        )
        parser.add_argument(
            '--workers',
            type=int,
            help='Number of parallel workers',
            default=self.config['max_workers']
        )
    
    def run(self) -> None:
        """Run batch processing template."""
        args = self.parse_arguments()
        
        self.config.update({
            'batch_config_file': args.batch_config,
            'parallel_processing': args.parallel,
            'max_workers': args.workers
        })
        
        # Load batch configuration
        batch_config = self.load_config_from_file(self.config['batch_config_file'])
        
        self.logger.info(f"Processing {len(batch_config)} batch items")
        
        if self.config['parallel_processing']:
            self._process_batch_parallel(batch_config)
        else:
            self._process_batch_sequential(batch_config)
    
    def _process_batch_sequential(self, batch_config: List[Dict[str, Any]]) -> None:
        """Process batch items sequentially."""
        for i, item_config in enumerate(batch_config):
            self.logger.info(f"Processing item {i+1}/{len(batch_config)}")
            try:
                self._process_single_item(item_config)
            except Exception as e:
                self.logger.error(f"Failed to process item {i+1}: {e}")
                if self.config['retry_failed']:
                    self._retry_item(item_config, i+1)
    
    def _process_batch_parallel(self, batch_config: List[Dict[str, Any]]) -> None:
        """Process batch items in parallel."""
        import concurrent.futures
        
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=self.config['max_workers']
        ) as executor:
            futures = []
            for i, item_config in enumerate(batch_config):
                future = executor.submit(self._process_single_item, item_config)
                futures.append((future, i+1))
            
            for future, item_num in futures:
                try:
                    future.result()
                    self.logger.info(f"Completed item {item_num}")
                except Exception as e:
                    self.logger.error(f"Failed to process item {item_num}: {e}")
    
    def _process_single_item(self, item_config: Dict[str, Any]) -> None:
        """Process a single batch item."""
        self.logger.info(f"Processing: {item_config.get('name', 'Unknown')}")
        # Add your item processing logic here
    
    def _retry_item(self, item_config: Dict[str, Any], item_num: int) -> None:
        """Retry processing a failed item."""
        max_retries = self.config['max_retries']
        for attempt in range(max_retries):
            try:
                self.logger.info(f"Retrying item {item_num} (attempt {attempt + 1})")
                self._process_single_item(item_config)
                return
            except Exception as e:
                self.logger.error(f"Retry {attempt + 1} failed: {e}")
        
        self.logger.error(f"Item {item_num} failed after {max_retries} retries") 