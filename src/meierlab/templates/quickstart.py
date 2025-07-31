#!/usr/bin/env python3
"""
Quickstart script for the MeierLab template system.

This script demonstrates how to use the template system with minimal setup.
Run this script to see examples of how to use the templates.
"""

import os
import sys
import tempfile
from pathlib import Path

# Add the parent directory to the path so we can import meierlab
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from meierlab.templates import BaseDownloadTemplate, XNATDownloadTemplate
from meierlab.templates.examples import MinimalDownloadTemplate


def demo_basic_usage():
    """Demonstrate basic template usage."""
    print("=== Basic Template Usage ===")

    # Create a minimal template
    template = MinimalDownloadTemplate(
        {"source_url": "https://example.com/data", "destination_dir": "./downloads"}
    )

    print(f"Template: {template.get_template_info()['name']}")
    print(f"Configuration: {template.config}")
    print()


def demo_configuration_management():
    """Demonstrate configuration management."""
    print("=== Configuration Management ===")

    # Create XNAT template
    template = XNATDownloadTemplate()

    # Show default configuration
    print("Default configuration keys:")
    for key in template.config.keys():
        print(f"  - {key}")
    print()

    # Create a temporary configuration file
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        config_content = """
address: "https://test.example.com"
project: "TEST_PROJECT"
username: "test_user"
password: "test_pass"
working_directory: "/tmp/test"
dcm2nii: true
bids: true
dry_run: true
"""
        f.write(config_content)
        config_file = f.name

    try:
        # Load configuration from file
        loaded_config = template.load_config_from_file(config_file)
        print("Loaded configuration:")
        for key, value in loaded_config.items():
            if key not in ["password"]:  # Don't show password
                print(f"  {key}: {value}")
        print()

        # Save current configuration
        save_file = config_file.replace(".yaml", "_saved.yaml")
        template.save_config_to_file(save_file)
        print(f"Configuration saved to: {save_file}")
        print()

    finally:
        # Cleanup
        if os.path.exists(config_file):
            os.unlink(config_file)
        if os.path.exists(save_file):
            os.unlink(save_file)


def demo_custom_template():
    """Demonstrate custom template creation."""
    print("=== Custom Template Creation ===")

    class MyCustomTemplate(BaseDownloadTemplate):
        """A simple custom template."""

        def _get_default_config(self):
            return {"input_file": "", "output_dir": "./output", "process_data": True}

        def _validate_config(self):
            if not self.config.get("input_file"):
                raise ValueError("input_file is required")

        def _add_arguments(self, parser):
            parser.add_argument("--input", required=True, help="Input file")
            parser.add_argument("--output", default="./output", help="Output directory")

        def run(self):
            print(f"Processing {self.config['input_file']}")
            print(f"Output directory: {self.config['output_dir']}")
            self.create_folder(self.config["output_dir"])
            print("Processing complete!")

    # Create and use custom template
    template = MyCustomTemplate(
        {"input_file": "data.txt", "output_dir": "./custom_output"}
    )

    print(f"Custom template: {template.get_template_info()['name']}")
    print(f"Configuration: {template.config}")

    # Run the template
    template.run()
    print()


def demo_error_handling():
    """Demonstrate error handling."""
    print("=== Error Handling ===")

    # Test configuration validation
    try:
        MinimalDownloadTemplate({"source_url": ""})
    except ValueError as e:
        print(f"Configuration validation error: {e}")

    # Test dry run mode
    try:
        XNATDownloadTemplate({"dry_run": True, "username": "test", "password": "test"})
        print("Dry run mode enabled - no actual downloads will occur")
    except ValueError as e:
        print(f"Expected validation error in dry run: {e}")

    print()


def main():
    """Main demonstration function."""
    print("MeierLab Template System - Quickstart Demo")
    print("=" * 50)
    print()

    try:
        demo_basic_usage()
        demo_configuration_management()
        demo_custom_template()
        demo_error_handling()

        print("=== Next Steps ===")
        print("1. Create your own configuration file")
        print("2. Use the XNATDownloadTemplate for XNAT data")
        print("3. Create custom templates by inheriting from BaseDownloadTemplate")
        print("4. Check the README.md for detailed documentation")
        print(
            "5. Run 'python -m meierlab.templates.cli list' to see available templates"
        )
        print()
        print("For more information, see: src/meierlab/templates/README.md")

    except Exception as e:
        print(f"Error during demonstration: {e}")
        print("Make sure you have all required dependencies installed.")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
