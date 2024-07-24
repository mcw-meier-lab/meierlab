import shutil

import matplotlib

from meierlab.datasets import MNI_MGZ
from meierlab.freesurfer import FreeSurfer, get_FreeSurfer_colormap


def test_get_FreeSurfer_colormap(fake_freesurfer_home):
    lut_file = fake_freesurfer_home / "FreeSurferColorLUT.txt"
    lut_file.write_text("""\n
#No. Label Name:                            R   G   B   A

0   Unknown                                 0   0   0   0
1   LCE                  70  130 180 0
2   LCWM              245 245 245 0
3   LCC                    205 62  78  0
""")
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
    fake_freesurfer_home, fake_subjects_dir, example_subject_id, fake_recon_all
):
    fs_dir = FreeSurfer(fake_freesurfer_home, fake_subjects_dir, example_subject_id)

    mri_dir = fake_subjects_dir / f"{example_subject_id}/mri"
    mri_dir.mkdir(parents=True)
    shutil.copy(MNI_MGZ, mri_dir / "orig.mgz")

    transforms_dir = fake_subjects_dir / f"{example_subject_id}/mri/transforms"
    transforms_dir.mkdir()
    lta_file = transforms_dir / "talairach.xfm.lta"
    lta_file.write_text("""type      = 0 # LINEAR_VOX_TO_VOX
nxforms   = 1
mean      = 0.0000 0.0000 0.0000
sigma     = 1.0000
1 4 4
1.036918997764587e+00 -1.005799975246191e-02 3.288500010967255e-02 -6.767410278320312e+00
-1.460499968379736e-02 1.093307971954346e+00 2.313899993896484e-01 -3.600976562500000e+01
-8.844000287353992e-03 -3.252570033073425e-01 9.822819828987122e-01 3.093771362304688e+01
0.000000000000000e+00 0.000000000000000e+00 0.000000000000000e+00 1.000000000000000e+00
src volume info
valid = 1  # volume info valid
filename = orig.mgz
volume = 256 256 256
voxelsize = 1.000000000000000e+00 1.000000000000000e+00 1.000000000000000e+00
xras   = -1.000000000000000e+00 0.000000000000000e+00 0.000000000000000e+00
yras   = 0.000000000000000e+00 0.000000000000000e+00 -1.000000000000000e+00
zras   = 0.000000000000000e+00 1.000000000000000e+00 0.000000000000000e+00
cras   = -2.718666076660156e+00 1.089466857910156e+01 -2.639067077636719e+01
dst volume info
valid = 1  # volume info valid
filename = mni305.cor.mgz
volume = 256 256 256
voxelsize = 1.000000000000000e+00 1.000000000000000e+00 1.000000000000000e+00
xras   = -1.000000000000000e+00 0.000000000000000e+00 0.000000000000000e+00
yras   = 0.000000000000000e+00 0.000000000000000e+00 -1.000000000000000e+00
zras   = 0.000000000000000e+00 1.000000000000000e+00 0.000000000000000e+00
cras   = 0.000000000000000e+00 0.000000000000000e+00 0.000000000000000e+00
subject subj
fscale 0.100000""")

    fs_dir.gen_tlrc_data(transforms_dir)
    inv_xfm = transforms_dir / "inv.xfm"
    mni2orig = transforms_dir / "mni2orig.nii.gz"
    assert inv_xfm.exists()
    assert mni2orig.exists()
