# read version from installed package
from importlib.metadata import version

__version__ = version("meierlab")

# Import template system
from .templates import BaseDownloadTemplate, XNATDownloadTemplate

__all__ = [
    "BaseDownloadTemplate",
    "XNATDownloadTemplate",
    "__version__",
]
