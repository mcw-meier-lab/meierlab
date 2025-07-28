"""
Test configuration utilities for meierlab templates.

This module provides secure ways to handle test credentials and sensitive
information during testing and development.
"""

import os
import tempfile
from contextlib import contextmanager
from pathlib import Path

import yaml


class ConfigManager:
    """
    Manager for test configurations with secure credential handling.

    This class provides methods to create test configurations that can
    safely handle sensitive information like passwords and API keys.
    """

    def __init__(self, base_dir: str | None = None):
        """
        Initialize the test config manager.

        Parameters
        ----------
        base_dir : str, optional
            Base directory for test configurations. If None, uses a temporary directory.
        """
        if base_dir is None:
            self.base_dir = tempfile.mkdtemp(prefix="meierlab_test_")
        else:
            self.base_dir = Path(base_dir)
            self.base_dir.mkdir(parents=True, exist_ok=True)

    def create_env_file(self, filename: str = ".env.test") -> str:
        """
        Create a .env file for testing with environment variables.

        Parameters
        ----------
        filename : str
            Name of the environment file

        Returns
        -------
        str
            Path to the created environment file
        """
        env_file = Path(self.base_dir) / filename

        env_content = """# Test Environment Variables
# Replace these with your actual test credentials

# XNAT Credentials
XNAT_USERNAME=test_user
XNAT_PASSWORD=test_password

# Non-sensitive configuration
XNAT_ADDRESS=https://test.xnat.org
XNAT_PROJECT=TEST_PROJECT
"""

        with open(env_file, "w") as f:
            f.write(env_content)

        return str(env_file)

    def create_test_config(self, template_type: str = "xnat") -> str:
        """
        Create a test configuration file.

        Parameters
        ----------
        template_type : str
            Type of template configuration to create

        Returns
        -------
        str
            Path to the created configuration file
        """
        if template_type == "xnat":
            config = {
                "address": "https://test.xnat.org",
                "project": "TEST_PROJECT",
                "username": "test_user",
                "password": "test_password",
                "working_directory": str(Path(self.base_dir) / "downloads"),
                "dcm2nii": True,
                "bids": True,
                "dry_run": True,
                "verbose": True,
            }
        elif template_type == "custom-xnat":
            config = {
                "address": "https://test.xnat.org",
                "project": "TEST_PROJECT",
                "username": "test_user",
                "password": "test_password",
                "working_directory": str(Path(self.base_dir) / "downloads"),
                "dcm2nii": True,
                "bids": True,
                "preprocess_data": True,
                "quality_check": True,
                "max_file_size_gb": 5,
                "dry_run": True,
            }
        else:
            raise ValueError(f"Unknown template type: {template_type}")

        config_file = Path(self.base_dir) / f"{template_type}_test_config.yaml"

        with open(config_file, "w") as f:
            yaml.dump(config, f, default_flow_style=False, indent=2)

        return str(config_file)

    def create_env_config(self, template_type: str = "xnat") -> str:
        """
        Create a configuration file that uses environment variables.

        Parameters
        ----------
        template_type : str
            Type of template configuration to create

        Returns
        -------
        str
            Path to the created configuration file
        """
        if template_type == "xnat":
            config = {
                "address": "https://test.xnat.org",
                "project": "TEST_PROJECT",
                "username": "${XNAT_USERNAME}",
                "password": "${XNAT_PASSWORD}",
                "working_directory": str(Path(self.base_dir) / "downloads"),
                "dcm2nii": True,
                "bids": True,
                "dry_run": True,
                "verbose": True,
            }
        else:
            raise ValueError(f"Unknown template type: {template_type}")

        config_file = Path(self.base_dir) / f"{template_type}_env_config.yaml"

        with open(config_file, "w") as f:
            yaml.dump(config, f, default_flow_style=False, indent=2)

        return str(config_file)

    @contextmanager
    def test_environment(self, env_vars: dict[str, str]):
        """
        Context manager for setting up test environment variables.

        Parameters
        ----------
        env_vars : dict
            Environment variables to set during the test
        """
        # Store original environment variables
        original_env = {}
        for key in env_vars:
            original_env[key] = os.environ.get(key)

        try:
            # Set new environment variables
            for key, value in env_vars.items():
                os.environ[key] = value

            yield

        finally:
            # Restore original environment variables
            for key, value in original_env.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value

    def cleanup(self):
        """Clean up test configuration files."""
        import shutil

        if os.path.exists(self.base_dir):
            shutil.rmtree(self.base_dir)


def load_dotenv_file(env_file: str) -> dict[str, str]:
    """
    Load environment variables from a .env file.

    Parameters
    ----------
    env_file : str
        Path to the .env file

    Returns
    -------
    dict
        Dictionary of environment variables
    """
    env_vars = {}

    with open(env_file) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, value = line.split("=", 1)
                env_vars[key.strip()] = value.strip()

    return env_vars


def setup_test_credentials(env_file: str | None = None) -> dict[str, str]:
    """
    Set up test credentials from environment or file.

    Parameters
    ----------
    env_file : str, optional
        Path to .env file containing credentials

    Returns
    -------
    dict
        Dictionary of test credentials
    """
    # Default test credentials
    default_credentials = {
        "XNAT_USERNAME": "test_user",
        "XNAT_PASSWORD": "test_password",
    }

    # Load from .env file if provided
    if env_file and os.path.exists(env_file):
        file_credentials = load_dotenv_file(env_file)
        default_credentials.update(file_credentials)

    # Override with actual environment variables if they exist
    for key in default_credentials:
        if key in os.environ:
            default_credentials[key] = os.environ[key]

    return default_credentials


def create_test_config_with_env(
    template_type: str = "xnat", env_file: str | None = None
) -> str:
    """
    Create a test configuration that uses environment variables.

    Parameters
    ----------
    template_type : str
        Type of template configuration to create
    env_file : str, optional
        Path to .env file containing credentials

    Returns
    -------
    str
        Path to the created configuration file
    """
    manager = ConfigManager()

    # Set up test environment
    credentials = setup_test_credentials(env_file)

    with manager.test_environment(credentials):
        return manager.create_test_config(template_type)


# Convenience functions for common test scenarios
def get_xnat_test_config(use_env: bool = True, env_file: str | None = None) -> str:
    """
    Get XNAT test configuration.

    Parameters
    ----------
    use_env : bool
        Whether to use environment variables for sensitive data
    env_file : str, optional
        Path to .env file containing credentials

    Returns
    -------
    str
        Path to the test configuration file
    """
    if use_env:
        return create_test_config_with_env("xnat", env_file)
    else:
        manager = ConfigManager()
        return manager.create_test_config("xnat")


def get_custom_xnat_test_config(
    use_env: bool = True, env_file: str | None = None
) -> str:
    """
    Get custom XNAT test configuration.

    Parameters
    ----------
    use_env : bool
        Whether to use environment variables for sensitive data
    env_file : str, optional
        Path to .env file containing credentials

    Returns
    -------
    str
        Path to the test configuration file
    """
    if use_env:
        return create_test_config_with_env("custom-xnat", env_file)
    else:
        manager = ConfigManager()
        return manager.create_test_config("custom-xnat")
