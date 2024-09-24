import os
import shutil
from pathlib import Path

import matplotlib
import pytest

from meierlab.quality.freesurfer import FreeSurfer, get_FreeSurfer_colormap
from meierlab.stats.freesurfer import gen_group_report, get_aparc_stats, get_aseg_stats

pytestmark = pytest.mark.skipif(
    "SUBJECTS_DIR" not in os.environ, reason="No FreeSurfer"
)


def test_get_FreeSurfer_colormap(fake_freesurfer_home, cmap):
    shutil.copy(cmap, fake_freesurfer_home)
    assert isinstance(
        get_FreeSurfer_colormap(fake_freesurfer_home), matplotlib.colors.ListedColormap
    )


def test_get_stats(
    fake_freesurfer_home, fake_subjects_dir, fake_recon_all, example_subject_id
):
    stats_dir = fake_subjects_dir / f"{example_subject_id}/stats"
    stats_dir.mkdir(parents=True)
    stats_file = stats_dir / "aseg.stats"
    stats_file.write_text("stats file")

    fs_dir = FreeSurfer(fake_freesurfer_home, fake_subjects_dir, example_subject_id)

    assert fs_dir.get_stats("aseg.stats").exists()


def test_check_recon_all_success(
    fake_freesurfer_home, fake_subjects_dir, example_subject_id
):
    scripts_dir = fake_subjects_dir / f"{example_subject_id}/scripts"
    scripts_dir.mkdir(parents=True)
    recon_file = scripts_dir / "recon-all.log"
    recon_file.write_text("finished without error")

    fs_dir = FreeSurfer(fake_freesurfer_home, fake_subjects_dir, example_subject_id)

    assert fs_dir.check_recon_all()
    assert fs_dir.recon_success


def test_check_recon_all_failure(
    fake_freesurfer_home, fake_subjects_dir, example_subject_id
):
    scripts_dir = fake_subjects_dir / f"{example_subject_id}/scripts"
    scripts_dir.mkdir(parents=True)
    recon_file = scripts_dir / "recon-all.log"
    recon_file.write_text("ERROR")

    fs_dir = FreeSurfer(fake_freesurfer_home, fake_subjects_dir, example_subject_id)

    assert not fs_dir.check_recon_all()
    assert not fs_dir.recon_success


def test_gen_tlrc_data(
    fake_freesurfer_home,
    fake_subjects_dir,
    example_subject_id,
    fake_recon_all,
    fake_tlrc_data,
    orig_mgz,
):
    fs_dir = FreeSurfer(fake_freesurfer_home, fake_subjects_dir, example_subject_id)

    mri_dir = fake_subjects_dir / f"{example_subject_id}/mri"
    mri_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(orig_mgz, mri_dir / "orig.mgz")

    transforms_dir = fake_subjects_dir / f"{example_subject_id}/mri/transforms"

    fs_dir.gen_tlrc_data(transforms_dir)
    inv_xfm = transforms_dir / "inv.xfm"
    mni2orig = transforms_dir / "mni2orig.nii.gz"
    assert inv_xfm.exists()
    assert mni2orig.exists()


def test_gen_tlrc_report(
    fake_freesurfer_home,
    fake_subjects_dir,
    example_subject_id,
    fake_recon_all,
    fake_tlrc_data,
    orig_mgz,
    wm_mgz,
):
    fs_dir = FreeSurfer(fake_freesurfer_home, fake_subjects_dir, example_subject_id)

    mri_dir = fake_subjects_dir / f"{example_subject_id}/mri"
    shutil.copy(orig_mgz, mri_dir)
    shutil.copy(wm_mgz, mri_dir)
    transforms_dir = fake_subjects_dir / f"{example_subject_id}/mri/transforms"

    output = Path(
        fs_dir.gen_tlrc_report(tlrc_dir=transforms_dir, output_dir=fake_subjects_dir)
    )
    assert output.exists()


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_gen_aparcaseg_plots(
    cmap,
    fake_freesurfer_home,
    fake_subjects_dir,
    example_subject_id,
    fake_recon_all,
    aparc_aseg_data,
):
    shutil.copy(cmap, fake_freesurfer_home)
    fs_dir = FreeSurfer(fake_freesurfer_home, fake_subjects_dir, example_subject_id)
    mri_dir = fake_subjects_dir / f"{example_subject_id}/mri"
    mri_dir.mkdir(parents=True, exist_ok=True)
    for mgz in aparc_aseg_data.glob("*mgz"):
        shutil.copy(mgz, mri_dir / mgz.name)

    imgs = fs_dir.gen_aparcaseg_plots(mri_dir)
    assert len(imgs) > 0
    assert imgs[0].exists()


@pytest.mark.filterwarnings("ignore::DeprecationWarning")
def test_gen_surf_plots(
    cmap,
    fake_freesurfer_home,
    fake_subjects_dir,
    example_subject_id,
    fake_recon_all,
    surf_data,
    label_data,
):
    shutil.copy(cmap, fake_freesurfer_home)
    fs_dir = FreeSurfer(fake_freesurfer_home, fake_subjects_dir, example_subject_id)

    surf_dir = fake_subjects_dir / f"{example_subject_id}/surf"
    surf_dir.mkdir(parents=True, exist_ok=True)
    for f1 in surf_data.glob("*"):
        shutil.copy(f1, surf_dir / f1.name)

    label_dir = fake_subjects_dir / f"{example_subject_id}/label"
    label_dir.mkdir()
    for f1 in label_data.glob("*"):
        shutil.copy(f1, label_dir / f1.name)

    imgs = fs_dir.gen_surf_plots(surf_dir)
    assert len(imgs) > 0
    assert imgs[0].exists()


def test_gen_report(
    fake_freesurfer_home, fake_subjects_dir, example_subject_id, fake_recon_all
):
    fs_dir = FreeSurfer(fake_freesurfer_home, fake_subjects_dir, example_subject_id)

    html_file = fs_dir.gen_report("example.html", fake_subjects_dir)
    assert html_file.exists()


def test_get_aseg_stats(fake_subjects_dir):
    subjects = [
        os.path.basename(sub)
        for sub in Path(os.getenv("SUBJECTS_DIR")).iterdir()
        if sub.is_dir()
    ]
    aseg = get_aseg_stats(subjects, "aseg.csv", output_dir=fake_subjects_dir)
    assert Path(aseg).exists()


def test_get_aparc_stats(fake_subjects_dir):
    subjects = [
        os.path.basename(sub)
        for sub in Path(os.getenv("SUBJECTS_DIR")).iterdir()
        if sub.is_dir()
    ]
    aparc_list = get_aparc_stats(subjects, "aparc.csv", output_dir=fake_subjects_dir)
    assert len(aparc_list) > 0
    for aparc in aparc_list:
        assert Path(aparc).exists()


def test_gen_group_report(tests_data_path):
    html_file = gen_group_report(
        f"{tests_data_path}/aseg.csv",
        [f for f in list(Path(tests_data_path).glob("*aparc*csv")) if f.is_file()],
        tests_data_path,
        "example.html",
        tests_data_path,
    )
    assert html_file.exists()
