import importlib.resources as pkg_resources

from sklearn.utils import Bunch


def load_updated_schaefer():
    # Use importlib.resources instead of pkg_resources
    def get_resource_path(package, resource):
        """Get the path to a resource file."""
        return pkg_resources.files(package) / resource

    atlas = get_resource_path(
        "meierlab.datasets",
        "data/schaefer2018/tpl-MNI152NLin2009cAsym_res-02_atlas-Schaefer2018_desc-400Parcels7Networks_dseg.nii.gz",
    )
    labels = get_resource_path(
        "meierlab.datasets",
        "data/schaefer2018/tpl-MNI152NLin2009cAsym_atlas-Schaefer2018UPDATED_desc-400Parcels7Networks_dseg_separateLabels.csv",
    )

    schaefer = Bunch(atlas=atlas, labels=labels)

    return schaefer
