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
from typing import Any


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

    def __init__(self, config: dict[str, Any] | None = None):
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

        # Setup logging first
        self.logger = self._setup_logging()

        # Load sensitive information from environment variables
        self._load_env_vars()

        self._validate_config()

    def _load_env_vars(self) -> None:
        """
        Load sensitive information from environment variables.

        This method looks for environment variables that match configuration
        keys with a '_ENV' suffix and loads them into the config.
        For example, XNAT_PASSWORD_ENV will be loaded into the 'password' config key.
        """
        env_mappings = self._get_env_mappings()

        for config_key, env_var in env_mappings.items():
            if env_var in os.environ:
                self.config[config_key] = os.environ[env_var]
                self.logger.debug(
                    f"Loaded {config_key} from environment variable {env_var}"
                )

    def _get_env_mappings(self) -> dict[str, str]:
        """
        Get mappings from configuration keys to environment variable names.

        Returns
        -------
        dict
            Mapping of config keys to environment variable names
        """
        return {
            "username": "XNAT_USERNAME",
            "password": "XNAT_PASSWORD",
        }

    @abstractmethod
    def _get_default_config(self) -> dict[str, Any]:
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
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
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
            description="Template-based data download script.",
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        )

        # Add common arguments
        parser.add_argument(
            "--config",
            "-c",
            help="Path to configuration file (JSON/YAML)",
            default=None,
        )
        parser.add_argument(
            "--verbose", "-v", action="store_true", help="Enable verbose logging"
        )
        parser.add_argument(
            "--dry-run",
            action="store_true",
            help="Perform a dry run without downloading",
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

    def load_config_from_file(self, config_path: str) -> dict[str, Any]:
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

        with open(config_path) as f:
            if config_path.suffix.lower() in [".yaml", ".yml"]:
                config = yaml.safe_load(f)
            elif config_path.suffix.lower() == ".json":
                config = json.load(f)
            else:
                raise ValueError(
                    f"Unsupported configuration file format: {config_path.suffix}"
                )

        return config

    def save_config_to_file(
        self, config_path: str, exclude_sensitive: bool = True
    ) -> None:
        """
        Save current configuration to a file.

        Parameters
        ----------
        config_path : str
            Path to save the configuration file
        exclude_sensitive : bool
            Whether to exclude sensitive information (passwords, tokens, etc.)
        """
        import json

        import yaml

        config_path = Path(config_path)

        # Create parent directories if they don't exist
        config_path.parent.mkdir(parents=True, exist_ok=True)

        # Create a copy of config for saving
        save_config = self.config.copy()

        if exclude_sensitive:
            sensitive_keys = ["password", "api_key", "token", "secret"]
            for key in sensitive_keys:
                if key in save_config:
                    save_config[key] = f"<{key.upper()}_FROM_ENV>"

        with open(config_path, "w") as f:
            if config_path.suffix.lower() in [".yaml", ".yml"]:
                yaml.dump(save_config, f, default_flow_style=False, indent=2)
            elif config_path.suffix.lower() == ".json":
                json.dump(save_config, f, indent=2)
            else:
                raise ValueError(
                    f"Unsupported configuration file format: {config_path.suffix}"
                )

        self.logger.info(f"Configuration saved to: {config_path}")

    def get_template_info(self) -> dict[str, Any]:
        """
        Get information about the template.

        Returns
        -------
        dict
            Template information including name, description, and version
        """
        return {
            "name": self.__class__.__name__,
            "description": self.__doc__ or "No description available",
            "version": getattr(self, "__version__", "1.0.0"),
            "config_keys": list(self.config.keys()),
        }
