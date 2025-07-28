"""
Template system for meierlab data processing workflows.

This module provides template classes and utilities for creating
customizable data processing scripts that users can modify or use as-is.
"""

from .base import BaseDownloadTemplate
from .download import XNATDownloadTemplate

__all__ = [
    "BaseDownloadTemplate",
    "XNATDownloadTemplate",
]

# Import example templates
try:
    from .examples import (
        BatchProcessingTemplate,
        CustomXNATDownloadTemplate,
        MinimalDownloadTemplate,
    )

    __all__.extend(
        [
            "BatchProcessingTemplate",
            "CustomXNATDownloadTemplate",
            "MinimalDownloadTemplate",
        ]
    )
except ImportError:
    # Examples might not be available in all environments
    pass
