"""
Command line interface for meierlab templates.

This module provides a unified CLI for running templates with various options.
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, Any

from .base import BaseDownloadTemplate
from .download import XNATDownloadTemplate
from .examples import (
    CustomXNATDownloadTemplate,
    MinimalDownloadTemplate,
    BatchProcessingTemplate
)


def create_template(template_name: str, config: Dict[str, Any] = None) -> BaseDownloadTemplate:
    """
    Create a template instance by name.
    
    Parameters
    ----------
    template_name : str
        Name of the template to create
    config : dict, optional
        Configuration dictionary
        
    Returns
    -------
    BaseDownloadTemplate
        Template instance
        
    Raises
    ------
    ValueError
        If template name is not recognized
    """
    templates = {
        'xnat': XNATDownloadTemplate,
        'custom-xnat': CustomXNATDownloadTemplate,
        'minimal': MinimalDownloadTemplate,
        'batch': BatchProcessingTemplate
    }
    
    if template_name not in templates:
        available = ', '.join(templates.keys())
        raise ValueError(f"Unknown template '{template_name}'. Available: {available}")
    
    template_class = templates[template_name]
    return template_class(config)


def list_templates():
    """List available templates with descriptions."""
    templates = {
        'xnat': 'XNAT server download with DICOM to NIfTI conversion',
        'custom-xnat': 'Extended XNAT template with quality checks',
        'minimal': 'Simple download template for basic use cases',
        'batch': 'Batch processing template for multiple data sources'
    }
    
    print("Available templates:")
    print()
    for name, description in templates.items():
        print(f"  {name:<15} {description}")
    print()
    print("Use 'meierlab-template <template> --help' for template-specific options")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description='MeierLab Template System CLI',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        'template',
        nargs='?',
        help='Template name to run (use "list" to see available templates)'
    )
    
    parser.add_argument(
        '--config', '-c',
        help='Configuration file path (YAML or JSON)'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging'
    )
    
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Perform a dry run without making changes'
    )
    
    parser.add_argument(
        '--info',
        action='store_true',
        help='Show template information and exit'
    )
    
    # Parse arguments
    args, remaining = parser.parse_known_args()
    
    # Handle special cases
    if not args.template:
        parser.print_help()
        return 1
    
    if args.template == 'list':
        list_templates()
        return 0
    
    # Load configuration if provided
    config = {}
    if args.config:
        try:
            template = create_template('xnat')  # Use any template to access config loading
            config = template.load_config_from_file(args.config)
        except Exception as e:
            print(f"Error loading configuration file: {e}")
            return 1
    
    # Add CLI options to config
    if args.verbose:
        config['verbose'] = True
    if args.dry_run:
        config['dry_run'] = True
    
    try:
        # Create template
        template = create_template(args.template, config)
        
        # Show template info if requested
        if args.info:
            info = template.get_template_info()
            print(f"Template: {info['name']}")
            print(f"Description: {info['description']}")
            print(f"Version: {info['version']}")
            print(f"Configuration keys: {', '.join(info['config_keys'])}")
            return 0
        
        # Set up logging level
        if config.get('verbose'):
            template.logger.setLevel('DEBUG')
        
        # Run template with remaining arguments
        sys.argv = [sys.argv[0]] + remaining
        template.run()
        
    except ValueError as e:
        print(f"Configuration error: {e}")
        return 1
    except KeyboardInterrupt:
        print("\nOperation cancelled by user")
        return 1
    except Exception as e:
        print(f"Unexpected error: {e}")
        if config.get('verbose'):
            import traceback
            traceback.print_exc()
        return 1
    
    return 0


if __name__ == '__main__':
    sys.exit(main()) 