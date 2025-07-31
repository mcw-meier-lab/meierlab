import importlib.resources as pkg_resources

from .atlas import load_updated_schaefer


# Use importlib.resources instead of pkg_resources
def get_resource_path(package, resource):
    """Get the path to a resource file."""
    return pkg_resources.files(package) / resource


MNI_NII = get_resource_path("meierlab.datasets", "data/mni305.cor.nii.gz")
MNI_MGZ = get_resource_path("meierlab.datasets", "data/mni305.cor.mgz")
__all__ = ["MNI_MGZ", "MNI_NII", "load_updated_schaefer"]
