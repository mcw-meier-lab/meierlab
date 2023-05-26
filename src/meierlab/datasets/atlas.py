from pathlib import Path
from pkg_resources import resource_filename as pkgrf
from sklearn.utils import Bunch


def load_updated_schaefer():
    atlas = Path(pkgrf('meierlab','datasets/data/schaefer2018/tpl-MNI152NLin2009cAsym_res-02_atlas-Schaefer2018_desc-400Parcels7Networks_dseg.nii.gz'))
    labels = Path(pkgrf('meierlab','datasets/data/schaefer2018/tpl-MNI152NLin2009cAsym_atlas-Schaefer2018UPDATED_desc-400Parcels7Networks_dseg_separateLabels.csv'))

    schaefer = Bunch(atlas=atlas,labels=labels)

    return schaefer