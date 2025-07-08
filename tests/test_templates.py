"""
Tests for the meierlab template system.
"""

import pytest
import tempfile
import os
from pathlib import Path

from meierlab.templates import BaseDownloadTemplate, XNATDownloadTemplate
from meierlab.templates.examples import (
    CustomXNATDownloadTemplate,
    MinimalDownloadTemplate,
    BatchProcessingTemplate
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
        # Create a concrete template for testing
        template = MinimalDownloadTemplate()
        info = template.get_template_info()
        
        assert 'name' in info
        assert 'description' in info
        assert 'version' in info
        assert 'config_keys' in info
        assert info['name'] == 'MinimalDownloadTemplate'
    
    def test_config_validation(self):
        """Test configuration validation."""
        template = MinimalDownloadTemplate()
        
        # Test with invalid config
        with pytest.raises(ValueError):
            template = MinimalDownloadTemplate({'source_url': ''})
    
    def test_folder_creation(self):
        """Test folder creation functionality."""
        template = MinimalDownloadTemplate()
        
        with tempfile.TemporaryDirectory() as temp_dir:
            test_folder = os.path.join(temp_dir, 'test_folder')
            template.create_folder(test_folder)
            
            assert os.path.exists(test_folder)
            assert os.path.isdir(test_folder)


class TestXNATDownloadTemplate:
    """Test the XNAT download template."""
    
    def test_template_creation(self):
        """Test XNAT template creation."""
        template = XNATDownloadTemplate()
        assert template is not None
        assert hasattr(template, 'config')
        assert hasattr(template, 'logger')
    
    def test_default_config(self):
        """Test default configuration."""
        template = XNATDownloadTemplate()
        config = template.config
        
        assert 'address' in config
        assert 'project' in config
        assert 'working_directory' in config
        assert config['address'] == 'https://cirxnat2.rcc.mcw.edu'
        assert config['project'] == 'CSI_MRI_MCW'
    
    def test_config_validation(self):
        """Test configuration validation."""
        # Test with missing required fields
        with pytest.raises(ValueError):
            XNATDownloadTemplate({
                'address': '',
                'project': '',
                'username': None,
                'password': None
            })
    
    def test_config_file_operations(self):
        """Test configuration file loading and saving."""
        template = XNATDownloadTemplate()
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            config_content = """
address: "https://test.example.com"
project: "TEST_PROJECT"
username: "test_user"
password: "test_pass"
working_directory: "/tmp/test"
dcm2nii: true
bids: true
"""
            f.write(config_content)
            config_file = f.name
        
        try:
            # Test loading config
            loaded_config = template.load_config_from_file(config_file)
            assert loaded_config['address'] == 'https://test.example.com'
            assert loaded_config['project'] == 'TEST_PROJECT'
            assert loaded_config['dcm2nii'] is True
            assert loaded_config['bids'] is True
            
            # Test saving config
            save_file = config_file.replace('.yaml', '_saved.yaml')
            template.save_config_to_file(save_file)
            
            assert os.path.exists(save_file)
            
            # Verify saved content
            with open(save_file, 'r') as f:
                saved_content = f.read()
                assert 'address' in saved_content
                assert 'project' in saved_content
            
        finally:
            # Cleanup
            if os.path.exists(config_file):
                os.unlink(config_file)
            if os.path.exists(save_file):
                os.unlink(save_file)


class TestCustomTemplates:
    """Test custom template examples."""
    
    def test_custom_xnat_template(self):
        """Test custom XNAT template."""
        template = CustomXNATDownloadTemplate()
        
        # Check custom config options
        assert 'preprocess_data' in template.config
        assert 'quality_check' in template.config
        assert 'max_file_size_gb' in template.config
        assert template.config['preprocess_data'] is True
        assert template.config['quality_check'] is True
        assert template.config['max_file_size_gb'] == 10
    
    def test_minimal_template(self):
        """Test minimal download template."""
        template = MinimalDownloadTemplate({
            'source_url': 'https://example.com',
            'destination_dir': '/tmp/test'
        })
        
        assert template.config['source_url'] == 'https://example.com'
        assert template.config['destination_dir'] == '/tmp/test'
    
    def test_batch_template(self):
        """Test batch processing template."""
        template = BatchProcessingTemplate()
        
        assert 'batch_config_file' in template.config
        assert 'parallel_processing' in template.config
        assert 'max_workers' in template.config
        assert template.config['max_workers'] == 4


class TestTemplateCLI:
    """Test template CLI functionality."""
    
    def test_template_imports(self):
        """Test that all templates can be imported."""
        from meierlab.templates.cli import create_template, list_templates
        
        # Test template creation
        template = create_template('xnat')
        assert isinstance(template, XNATDownloadTemplate)
        
        template = create_template('minimal')
        assert isinstance(template, MinimalDownloadTemplate)
        
        # Test invalid template
        with pytest.raises(ValueError):
            create_template('invalid_template')


@pytest.mark.download
class TestTemplateIntegration:
    """Integration tests for templates (marked as download tests)."""
    
    def test_xnat_template_dry_run(self):
        """Test XNAT template in dry run mode."""
        template = XNATDownloadTemplate({
            'dry_run': True,
            'username': 'test',
            'password': 'test'
        })
        
        # This should not actually connect to XNAT in dry run mode
        # but should validate the configuration
        with pytest.raises(ValueError):
            template.run()  # Should fail due to invalid credentials
    
    def test_template_logging(self):
        """Test template logging functionality."""
        template = MinimalDownloadTemplate({
            'source_url': 'https://example.com',
            'destination_dir': '/tmp/test'
        })
        
        # Test that logger is properly configured
        assert template.logger is not None
        assert template.logger.level <= 20  # INFO or lower 