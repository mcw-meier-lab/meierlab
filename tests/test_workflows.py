import shutil

import pytest

from meierlab.quality.freesurfer import FreeSurfer
from meierlab.quality.workflows import fs_quality_wf


@pytest.mark.filterwarnings("ignore::UserWarning")
@pytest.mark.filterwarnings("ignore::DeprecationWarning")
def test_fs_quality_wf(
    fake_freesurfer_home,
    fake_subjects_dir,
    example_subject_id,
    fake_recon_all,
    fake_tlrc_data,
    surf_data,
    aparc_aseg_data,
    label_data,
    orig_mgz,
    wm_mgz,
    cmap,
):
    mri_dir = fake_subjects_dir / f"{example_subject_id}/mri"
    mri_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(orig_mgz, mri_dir)
    shutil.copy(wm_mgz, mri_dir)

    shutil.copy(cmap, fake_freesurfer_home)
    for mgz in aparc_aseg_data.glob("*mgz"):
        shutil.copy(mgz, mri_dir / mgz.name)

    surf_dir = fake_subjects_dir / f"{example_subject_id}/surf"
    surf_dir.mkdir(parents=True, exist_ok=True)
    for f1 in surf_data.glob("*"):
        shutil.copy(f1, surf_dir / f1.name)

    label_dir = fake_subjects_dir / f"{example_subject_id}/label"
    label_dir.mkdir()
    for f1 in label_data.glob("*"):
        shutil.copy(f1, label_dir / f1.name)

    fs_dir = FreeSurfer(fake_freesurfer_home, fake_subjects_dir, example_subject_id)

    fs_html = fs_quality_wf(fs_dir, fake_subjects_dir)
    assert fs_html.exists()
