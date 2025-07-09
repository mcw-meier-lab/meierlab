#!/usr/bin/env python3
"""
Setup script for secure test environment.

This script helps users set up a secure test environment with proper
credential handling for the MeierLab template system.
"""

import os
import sys
from pathlib import Path

from meierlab.templates.test_config import ConfigManager, setup_test_credentials


def setup_test_environment():
    """Set up a complete test environment."""
    print("🔐 MeierLab Template System - Test Environment Setup")
    print("=" * 50)
    print()
    
    # Create test config manager
    manager = ConfigManager()
    
    try:
        print("1. Creating test configuration files...")
        
        # Create .env file
        env_file = manager.create_env_file()
        print(f"   ✅ Created environment file: {env_file}")
        
        # Create test configurations
        xnat_config = manager.create_test_config('xnat')
        print(f"   ✅ Created XNAT test config: {xnat_config}")
        
        custom_config = manager.create_test_config('custom-xnat')
        print(f"   ✅ Created custom XNAT test config: {custom_config}")
        
        print()
        print("2. Setting up environment variables...")
        
        # Load test credentials
        credentials = setup_test_credentials(env_file)
        
        # Set up test environment
        with manager.test_environment(credentials):
            print("   ✅ Test environment variables set")
            print(f"   📋 Username: {credentials.get('XNAT_USERNAME', 'test_user')}")
            print(f"   📋 Password: {'*' * len(credentials.get('XNAT_PASSWORD', 'test_pass'))}")
        
        print()
        print("3. Testing template system...")
        
        # Test template creation
        from .download import XNATDownloadTemplate
        
        template = XNATDownloadTemplate()
        print("   ✅ Template created successfully")
        
        info = template.get_template_info()
        print(f"   📋 Template: {info['name']}")
        print(f"   📋 Version: {info['version']}")
        
        print()
        print("4. Configuration examples...")
        
        # Show example usage
        print("   📝 Example usage:")
        print("   ```python")
        print("   from meierlab.templates import XNATDownloadTemplate")
        print("   from meierlab.templates.test_config import TestConfigManager")
        print("   ")
        print("   # Create test config")
        print("   manager = TestConfigManager()")
        print("   config_file = manager.create_test_config('xnat')")
        print("   ")
        print("   # Use template")
        print("   template = XNATDownloadTemplate()")
        print("   template.load_config_from_file(config_file)")
        print("   template.run()")
        print("   ```")
        
        print()
        print("5. Security notes...")
        print("   🔒 Test credentials are safe to use")
        print("   🔒 Environment variables are automatically loaded")
        print("   🔒 Sensitive data is masked when saving configs")
        print("   🔒 Add .env files to .gitignore")
        
        print()
        print("✅ Test environment setup complete!")
        print()
        print("Next steps:")
        print("1. Copy env_example.txt to .env and add your real credentials")
        print("2. Add .env to .gitignore")
        print("3. Run tests: pytest tests/test_templates.py")
        print("4. Try the quickstart: python src/meierlab/templates/quickstart.py")
        
        return True
        
    except Exception as e:
        print(f"❌ Error setting up test environment: {e}")
        return False
    
    finally:
        # Clean up temporary files
        manager.cleanup()


def create_env_file():
    """Create a .env file from the example."""
    print("📝 Creating .env file from example...")
    
    # Get the example file path
    current_dir = Path(__file__).parent
    example_file = current_dir / "config_examples" / "env_example.txt"
    env_file = Path.cwd() / ".env"
    
    if not example_file.exists():
        print(f"❌ Example file not found: {example_file}")
        return False
    
    if env_file.exists():
        print(f"⚠️  .env file already exists: {env_file}")
        response = input("Overwrite? (y/N): ")
        if response.lower() != 'y':
            print("Skipping .env file creation")
            return True
    
    try:
        # Copy example to .env
        with open(example_file, 'r') as f:
            content = f.read()
        
        with open(env_file, 'w') as f:
            f.write(content)
        
        print(f"✅ Created .env file: {env_file}")
        print("📝 Edit this file with your actual credentials")
        
        # Check if .gitignore exists and add .env
        gitignore_file = Path.cwd() / ".gitignore"
        if gitignore_file.exists():
            with open(gitignore_file, 'r') as f:
                gitignore_content = f.read()
            
            if ".env" not in gitignore_content:
                with open(gitignore_file, 'a') as f:
                    f.write("\n# Environment variables\n.env\n")
                print("✅ Added .env to .gitignore")
        else:
            with open(gitignore_file, 'w') as f:
                f.write("# Environment variables\n.env\n")
            print("✅ Created .gitignore with .env")
        
        return True
        
    except Exception as e:
        print(f"❌ Error creating .env file: {e}")
        return False


def main():
    """Main setup function."""
    if len(sys.argv) > 1:
        command = sys.argv[1]
        
        if command == "env":
            success = create_env_file()
        elif command == "test":
            success = setup_test_environment()
        else:
            print("Usage: python setup_test_env.py [env|test]")
            print("  env  - Create .env file from example")
            print("  test - Set up complete test environment")
            return 1
    else:
        print("🔐 MeierLab Template System - Setup")
        print("=" * 40)
        print()
        print("Choose an option:")
        print("1. Create .env file from example")
        print("2. Set up complete test environment")
        print()
        
        choice = input("Enter choice (1 or 2): ")
        
        if choice == "1":
            success = create_env_file()
        elif choice == "2":
            success = setup_test_environment()
        else:
            print("Invalid choice")
            return 1
    
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main()) 