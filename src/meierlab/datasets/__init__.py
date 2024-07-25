from pathlib import Path

from pkg_resources import resource_filename as pkgrf

from .atlas import load_updated_schaefer

MNI_NII = Path(pkgrf("meierlab", "datasets/data/mni305.cor.nii.gz"))
MNI_MGZ = Path(pkgrf("meierlab", "datasets/data/mni305.cor.mgz"))
__all__ = ["load_updated_schaefer", "MNI_NII", "MNI_MGZ"]
