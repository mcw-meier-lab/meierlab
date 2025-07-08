# MeierLab Template System

The MeierLab template system provides a flexible and extensible framework for creating customizable data processing workflows. Users can use the provided templates as-is or create their own custom templates by inheriting from the base classes.

## Overview

The template system consists of:

- **BaseDownloadTemplate**: Abstract base class providing common functionality
- **XNATDownloadTemplate**: Ready-to-use template for XNAT data downloads
- **Example templates**: Demonstrations of customization patterns
- **Configuration files**: YAML/JSON configuration examples

## Quick Start

### Using the XNAT Download Template

The XNAT download template is ready to use out of the box:

```python
from meierlab.templates import XNATDownloadTemplate

# Create template with default configuration
template = XNATDownloadTemplate()

# Run the template
template.run()
```

### Command Line Usage

```bash
# Basic usage with command line arguments
python -m meierlab.templates.download \
    --username your_username \
    --password your_password \
    --working-directory /path/to/data \
    --dcm2nii \
    --bids

# Using a configuration file
python -m meierlab.templates.download \
    --config config.yaml \
    --verbose
```

### Configuration File Usage

Create a configuration file (YAML or JSON):

```yaml
# config.yaml
address: "https://cirxnat2.rcc.mcw.edu"
project: "CSI_MRI_MCW"
username: "your_username"
password: "your_password"
working_directory: "/path/to/data"
dcm2nii: true
bids: true
dry_run: false
```

Then use it:

```python
from meierlab.templates import XNATDownloadTemplate

# Load configuration from file
template = XNATDownloadTemplate()
config = template.load_config_from_file('config.yaml')

# Create template with loaded configuration
template = XNATDownloadTemplate(config)
template.run()
```

## Creating Custom Templates

### Basic Custom Template

```python
from meierlab.templates import BaseDownloadTemplate
from typing import Dict, Any

class MyCustomTemplate(BaseDownloadTemplate):
    """My custom download template."""
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Define default configuration."""
        return {
            'source_url': '',
            'destination': './downloads',
            'file_pattern': '*'
        }
    
    def _validate_config(self) -> None:
        """Validate configuration."""
        if not self.config.get('source_url'):
            raise ValueError("source_url is required")
    
    def _add_arguments(self, parser):
        """Add custom command line arguments."""
        parser.add_argument('--source', required=True, help='Source URL')
        parser.add_argument('--dest', default='./downloads', help='Destination')
    
    def run(self) -> None:
        """Main execution logic."""
        args = self.parse_arguments()
        
        # Update config with command line arguments
        self.config.update({
            'source_url': args.source,
            'destination': args.dest
        })
        
        # Your custom logic here
        self.logger.info(f"Downloading from {self.config['source_url']}")
        self.create_folder(self.config['destination'])
```

### Extending Existing Templates

```python
from meierlab.templates import XNATDownloadTemplate
from typing import Dict, Any

class CustomXNATTemplate(XNATDownloadTemplate):
    """Custom XNAT template with additional features."""
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Extend default configuration."""
        config = super()._get_default_config()
        config.update({
            'quality_check': True,
            'backup_files': True,
            'max_file_size_gb': 10
        })
        return config
    
    def _validate_config(self) -> None:
        """Add custom validation."""
        super()._validate_config()
        
        if self.config.get('max_file_size_gb', 0) <= 0:
            raise ValueError("max_file_size_gb must be positive")
    
    def _setup(self, download_dict):
        """Override setup with custom logic."""
        # Custom preprocessing
        if self.config.get('quality_check'):
            self._perform_quality_check(download_dict)
        
        # Call parent setup
        super()._setup(download_dict)
        
        # Custom post-processing
        if self.config.get('backup_files'):
            self._backup_files(download_dict)
    
    def _perform_quality_check(self, download_dict):
        """Custom quality check logic."""
        self.logger.info("Performing quality checks...")
        # Your quality check logic here
    
    def _backup_files(self, download_dict):
        """Custom backup logic."""
        self.logger.info("Backing up files...")
        # Your backup logic here
```

## Template Features

### Configuration Management

Templates support multiple configuration sources:

1. **Default configuration**: Built into the template
2. **Configuration files**: YAML or JSON files
3. **Command line arguments**: Override any configuration
4. **Programmatic configuration**: Pass dict to constructor

### Logging

All templates include comprehensive logging:

```python
template = XNATDownloadTemplate()
template.logger.info("Starting download...")
template.logger.warning("Large file detected")
template.logger.error("Download failed")
```

### Error Handling

Templates include robust error handling:

```python
try:
    template.run()
except ValueError as e:
    print(f"Configuration error: {e}")
except Exception as e:
    print(f"Unexpected error: {e}")
```

### Dry Run Mode

Test templates without making changes:

```python
template = XNATDownloadTemplate({'dry_run': True})
template.run()  # Will show what would be done without doing it
```

## Available Templates

### XNATDownloadTemplate

**Purpose**: Download data from XNAT servers with DICOM to NIfTI conversion and BIDS organization.

**Features**:
- XNAT server connection
- DICOM to NIfTI conversion
- BIDS format support
- Scan type filtering
- Subject/experiment selection

**Configuration Options**:
- `address`: XNAT server URL
- `project`: XNAT project name
- `username`/`password`: Authentication
- `working_directory`: Download location
- `dcm2nii`: Enable DICOM conversion
- `bids`: Enable BIDS organization
- `scan`: Specific scan types
- `subject`/`experiment`: Specific subjects/experiments

### CustomXNATDownloadTemplate

**Purpose**: Extended XNAT template with quality checks and preprocessing.

**Additional Features**:
- Custom preprocessing
- Quality checks
- File size limits
- Scan type filtering
- Backup functionality

### MinimalDownloadTemplate

**Purpose**: Simple template for basic download scenarios.

**Features**:
- Minimal configuration
- Simple file downloads
- Pattern matching
- Basic error handling

### BatchProcessingTemplate

**Purpose**: Process multiple data sources in parallel or sequentially.

**Features**:
- Parallel processing
- Retry logic
- Batch configuration files
- Progress tracking

## Configuration File Formats

### YAML Configuration

```yaml
# Example YAML configuration
address: "https://example.com"
project: "MY_PROJECT"
username: "user"
password: "pass"
working_directory: "/path/to/data"
dcm2nii: true
bids: true
dry_run: false
```

### JSON Configuration

```json
{
  "address": "https://example.com",
  "project": "MY_PROJECT",
  "username": "user",
  "password": "pass",
  "working_directory": "/path/to/data",
  "dcm2nii": true,
  "bids": true,
  "dry_run": false
}
```

## Best Practices

### 1. Configuration Management

- Use configuration files for complex setups
- Keep sensitive information (passwords) out of version control
- Use environment variables for secrets
- Validate configurations before use

### 2. Error Handling

- Always validate configuration
- Handle network errors gracefully
- Provide meaningful error messages
- Use dry run mode for testing

### 3. Logging

- Use appropriate log levels
- Include context in log messages
- Log progress for long-running operations
- Consider log file rotation

### 4. Customization

- Inherit from appropriate base class
- Override only necessary methods
- Maintain backward compatibility
- Document custom features

### 5. Testing

- Test with dry run mode
- Use small datasets for testing
- Test error conditions
- Validate output formats

## Examples

See the `config_examples/` directory for complete configuration examples:

- `xnat_config.yaml`: Basic XNAT configuration
- `custom_xnat_config.yaml`: Advanced XNAT configuration
- `batch_config.yaml`: Batch processing configuration

## Troubleshooting

### Common Issues

1. **Authentication Errors**
   - Verify username and password
   - Check XNAT server accessibility
   - Ensure proper permissions

2. **Configuration Errors**
   - Validate configuration file format
   - Check required fields
   - Use `--verbose` for detailed error messages

3. **Permission Errors**
   - Check directory permissions
   - Ensure write access to working directory
   - Verify file ownership

4. **Network Errors**
   - Check internet connectivity
   - Verify XNAT server status
   - Check firewall settings

### Getting Help

- Use `--help` for command line options
- Enable verbose logging with `--verbose`
- Check template information with `get_template_info()`
- Review example configurations

## Contributing

To contribute new templates:

1. Inherit from appropriate base class
2. Implement required abstract methods
3. Add comprehensive documentation
4. Include example configurations
5. Add tests for new functionality
6. Follow existing code style

## License

This template system is part of the MeierLab package and follows the same license terms. 