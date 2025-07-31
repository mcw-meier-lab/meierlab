"""
Tests for the meierlab template system.
"""

import os
import tempfile

import pytest

from meierlab.templates import BaseDownloadTemplate, XNATDownloadTemplate
from meierlab.templates.examples import (
    BatchProcessingTemplate,
    CustomXNATDownloadTemplate,
    MinimalDownloadTemplate,
)
from meierlab.templates.test_config import (
    ConfigManager,
    setup_test_credentials,
)


class TestBaseDownloadTemplate:
    """Test the base download template functionality."""

    def test_template_creation(self):
        """Test that templates can be created."""
        # This should raise an error since BaseDownloadTemplate is abstract
        with pytest.raises(TypeError):
            BaseDownloadTemplate()

    def test_template_info(self):
        """Test template information retrieval."""
        # Create a concrete template for testing with dry run mode
        template = MinimalDownloadTemplate(
            {"dry_run": True, "source_url": "https://example.com"}
        )
        info = template.get_template_info()

        assert "name" in info
        assert "description" in info
        assert "version" in info
        assert "config_keys" in info
        assert info["name"] == "MinimalDownloadTemplate"

    def test_config_validation(self):
        """Test configuration validation."""
        # Test with valid config in dry run mode
        MinimalDownloadTemplate({"dry_run": True, "source_url": "https://example.com"})

        # Test with invalid config (should still fail even in dry run)
        with pytest.raises(ValueError):
            MinimalDownloadTemplate({"source_url": "", "dry_run": True})

    def test_folder_creation(self):
        """Test folder creation functionality."""
        template = MinimalDownloadTemplate(
            {"dry_run": True, "source_url": "https://example.com"}
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            test_folder = os.path.join(temp_dir, "test_folder")
            template.create_folder(test_folder)

            assert os.path.exists(test_folder)
            assert os.path.isdir(test_folder)


class TestXNATDownloadTemplate:
    """Test the XNAT download template."""

    def test_template_creation(self):
        """Test XNAT template creation."""
        template = XNATDownloadTemplate({"dry_run": True})
        assert template is not None
        assert hasattr(template, "config")
        assert hasattr(template, "logger")

    def test_default_config(self):
        """Test default configuration."""
        template = XNATDownloadTemplate({"dry_run": True})
        config = template.config

        assert "address" in config
        assert "project" in config
        assert "working_directory" in config
        assert config["address"] == "https://test.xnat.org"
        assert config["project"] == "TEST_PROJECT"

    def test_config_validation(self):
        """Test configuration validation."""
        # Test with missing required fields
        with pytest.raises(ValueError):
            XNATDownloadTemplate(
                {"address": "", "project": "", "username": None, "password": None}
            )

    def test_config_file_operations(self):
        """Test configuration file loading and saving."""
        template = XNATDownloadTemplate({"dry_run": True})

        # Use test config manager for secure credential handling
        manager = ConfigManager()

        try:
            # Create test config with environment variables
            config_file = manager.create_test_config("xnat")

            # Test loading config
            loaded_config = template.load_config_from_file(config_file)
            assert loaded_config["address"] == "https://test.xnat.org"
            assert loaded_config["project"] == "TEST_PROJECT"
            assert loaded_config["dcm2nii"] is True
            assert loaded_config["bids"] is True

            # Test saving config (should exclude sensitive data)
            save_file = config_file.replace(".yaml", "_saved.yaml")
            template.save_config_to_file(save_file, exclude_sensitive=True)

            assert os.path.exists(save_file)

            # Verify saved content (should not contain actual passwords)
            with open(save_file) as f:
                saved_content = f.read()
                assert "address" in saved_content
                assert "project" in saved_content
                assert "test_password" not in saved_content  # Password should be masked

        finally:
            # Cleanup
            manager.cleanup()


class TestCustomTemplates:
    """Test custom template examples."""

    def test_custom_xnat_template(self):
        """Test custom XNAT template."""
        template = CustomXNATDownloadTemplate({"dry_run": True})

        # Check custom config options
        assert "preprocess_data" in template.config
        assert "quality_check" in template.config
        assert "max_file_size_gb" in template.config
        assert template.config["preprocess_data"] is True
        assert template.config["quality_check"] is True
        assert template.config["max_file_size_gb"] == 10

    def test_minimal_template(self):
        """Test minimal download template."""
        template = MinimalDownloadTemplate(
            {
                "source_url": "https://example.com",
                "destination_dir": "/tmp/test",
                "dry_run": True,
            }
        )

        assert template.config["source_url"] == "https://example.com"
        assert template.config["destination_dir"] == "/tmp/test"

    def test_batch_template(self):
        """Test batch processing template."""
        template = BatchProcessingTemplate({"dry_run": True})

        assert "batch_config_file" in template.config
        assert "parallel_processing" in template.config
        assert "max_workers" in template.config
        assert template.config["max_workers"] == 4


class TestTemplateCLI:
    """Test template CLI functionality."""

    def test_template_imports(self):
        """Test that all templates can be imported."""
        from meierlab.templates.cli import create_template

        # Test template creation with dry run config
        template = create_template("xnat", {"dry_run": True})
        assert isinstance(template, XNATDownloadTemplate)

        template = create_template(
            "minimal", {"dry_run": True, "source_url": "https://example.com"}
        )
        assert isinstance(template, MinimalDownloadTemplate)

        # Test invalid template
        with pytest.raises(ValueError):
            create_template("invalid_template")


@pytest.mark.download
class TestTemplateIntegration:
    """Integration tests for templates (marked as download tests)."""

    def test_xnat_template_dry_run(self):
        """Test XNAT template in dry run mode."""
        # Use test credentials from environment
        credentials = setup_test_credentials()

        template = XNATDownloadTemplate(
            {
                "dry_run": True,
                "username": credentials.get("XNAT_USERNAME", "test"),
                "password": credentials.get("XNAT_PASSWORD", "test"),
            }
        )

        # In dry run mode, this should not try to parse command line arguments
        # and should not fail due to missing credentials
        assert template.config["dry_run"] is True
        assert template.config["username"] == credentials.get("XNAT_USERNAME", "test")
        assert template.config["password"] == credentials.get("XNAT_PASSWORD", "test")

    def test_template_logging(self):
        """Test template logging functionality."""
        template = MinimalDownloadTemplate(
            {
                "source_url": "https://example.com",
                "destination_dir": "/tmp/test",
                "dry_run": True,
            }
        )

        # Test that logger is properly configured
        assert template.logger is not None
        assert template.logger.level <= 20  # INFO or lower

    def test_environment_variable_loading(self):
        """Test loading credentials from environment variables."""
        manager = ConfigManager()

        try:
            # Set up test environment
            test_credentials = {
                "XNAT_USERNAME": "env_test_user",
                "XNAT_PASSWORD": "env_test_pass",
            }

            with manager.test_environment(test_credentials):
                # Create template - should load from environment
                template = XNATDownloadTemplate({"dry_run": True})

                # Check that environment variables were loaded
                assert template.config["username"] == "env_test_user"
                assert template.config["password"] == "env_test_pass"

        finally:
            manager.cleanup()

    def test_secure_config_saving(self):
        """Test that sensitive data is properly masked when saving config."""
        manager = ConfigManager()

        try:
            # Create template with test credentials
            template = XNATDownloadTemplate(
                {
                    "username": "test_user",
                    "password": "secret_password",
                    "api_key": "secret_key",
                    "dry_run": True,
                }
            )

            # Save config with sensitive data masking
            config_file = manager.create_test_config("xnat")
            save_file = config_file.replace(".yaml", "_secure.yaml")
            template.save_config_to_file(save_file, exclude_sensitive=True)

            # Verify sensitive data is masked
            with open(save_file) as f:
                content = f.read()
                assert "secret_password" not in content
                assert "secret_key" not in content
                assert "<PASSWORD_FROM_ENV>" in content
                assert "<API_KEY_FROM_ENV>" in content

        finally:
            manager.cleanup()
