"""
Base template class for meierlab data processing workflows.

This module provides the foundation for creating customizable
data processing scripts that users can modify or use as-is.
"""

import argparse
import logging
import os
import subprocess
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional


class BaseDownloadTemplate(ABC):
    """
    Base class for download templates.
    
    This class provides common functionality for data download scripts
    and serves as a foundation for creating customizable templates.
    Users can inherit from this class to create their own download scripts
    or use the provided XNATDownloadTemplate as-is.
    
    Attributes
    ----------
    config : dict
        Configuration dictionary containing all template parameters
    logger : logging.Logger
        Logger instance for the template
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the base template.
        
        Parameters
        ----------
        config : dict, optional
            Configuration dictionary. If None, default configuration is used.
        """
        self.config = self._get_default_config()
        if config:
            self.config.update(config)
        
        self.logger = self._setup_logging()
        self._validate_config()
    
    @abstractmethod
    def _get_default_config(self) -> Dict[str, Any]:
        """
        Get default configuration for the template.
        
        Returns
        -------
        dict
            Default configuration dictionary
        """
        pass
    
    @abstractmethod
    def _validate_config(self) -> None:
        """
        Validate the configuration.
        
        Raises
        ------
        ValueError
            If configuration is invalid
        """
        pass
    
    def _setup_logging(self) -> logging.Logger:
        """
        Setup logging for the template.
        
        Returns
        -------
        logging.Logger
            Configured logger instance
        """
        logger = logging.getLogger(self.__class__.__name__)
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger
    
    def create_folder(self, folder: str) -> None:
        """
        Create a folder if it does not exist.
        
        Parameters
        ----------
        folder : str
            Path to the folder to create.
        """
        if not os.path.isdir(folder):
            os.makedirs(folder)
            self.logger.info(f"Created directory: {folder}")
    
    def update_permissions(self, directory: str) -> None:
        """
        Update permissions for a directory.
        
        Parameters
        ----------
        directory : str
            Path to the directory to update permissions for.
        """
        try:
            # Set directory permissions to 770
            cmd = f"find {directory} -type d -exec chmod 770 {{}} \\;"
            subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
            
            # Set file permissions to 664
            cmd = f"find {directory} -type f -exec chmod 664 {{}} \\;"
            subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
            
            self.logger.info(f"Updated permissions for: {directory}")
        except subprocess.CalledProcessError as e:
            self.logger.warning(f"Could not update permissions for {directory}: {e}")
    
    def parse_arguments(self) -> argparse.Namespace:
        """
        Parse command line arguments.
        
        Returns
        -------
        argparse.Namespace
            Parsed arguments
        """
        parser = argparse.ArgumentParser(
            description='Template-based data download script.',
            formatter_class=argparse.ArgumentDefaultsHelpFormatter
        )
        
        # Add common arguments
        parser.add_argument(
            '--config', '-c',
            help='Path to configuration file (JSON/YAML)',
            default=None
        )
        parser.add_argument(
            '--verbose', '-v',
            action='store_true',
            help='Enable verbose logging'
        )
        parser.add_argument(
            '--dry-run',
            action='store_true',
            help='Perform a dry run without downloading'
        )
        
        # Add template-specific arguments
        self._add_arguments(parser)
        
        args = parser.parse_args()
        
        # Update logging level if verbose
        if args.verbose:
            self.logger.setLevel(logging.DEBUG)
        
        return args
    
    @abstractmethod
    def _add_arguments(self, parser: argparse.ArgumentParser) -> None:
        """
        Add template-specific command line arguments.
        
        Parameters
        ----------
        parser : argparse.ArgumentParser
            Argument parser to add arguments to
        """
        pass
    
    @abstractmethod
    def run(self) -> None:
        """
        Run the template's main functionality.
        """
        pass
    
    def load_config_from_file(self, config_path: str) -> Dict[str, Any]:
        """
        Load configuration from a file.
        
        Parameters
        ----------
        config_path : str
            Path to the configuration file
            
        Returns
        -------
        dict
            Loaded configuration
        """
        import json
        import yaml
        
        config_path = Path(config_path)
        
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        with open(config_path, 'r') as f:
            if config_path.suffix.lower() in ['.yaml', '.yml']:
                config = yaml.safe_load(f)
            elif config_path.suffix.lower() == '.json':
                config = json.load(f)
            else:
                raise ValueError(f"Unsupported configuration file format: {config_path.suffix}")
        
        return config
    
    def save_config_to_file(self, config_path: str) -> None:
        """
        Save current configuration to a file.
        
        Parameters
        ----------
        config_path : str
            Path to save the configuration file
        """
        import json
        import yaml
        
        config_path = Path(config_path)
        
        # Create parent directories if they don't exist
        config_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(config_path, 'w') as f:
            if config_path.suffix.lower() in ['.yaml', '.yml']:
                yaml.dump(self.config, f, default_flow_style=False, indent=2)
            elif config_path.suffix.lower() == '.json':
                json.dump(self.config, f, indent=2)
            else:
                raise ValueError(f"Unsupported configuration file format: {config_path.suffix}")
        
        self.logger.info(f"Configuration saved to: {config_path}")
    
    def get_template_info(self) -> Dict[str, Any]:
        """
        Get information about the template.
        
        Returns
        -------
        dict
            Template information including name, description, and version
        """
        return {
            'name': self.__class__.__name__,
            'description': self.__doc__ or 'No description available',
            'version': getattr(self, '__version__', '1.0.0'),
            'config_keys': list(self.config.keys())
        } 