import os
import shutil
from pathlib import Path

import pytest

from meierlab.cirxnat import Cirxnat


@pytest.fixture
def user():
    return "lespana"


@pytest.fixture
def password():
    return os.getenv("CIR2")


@pytest.fixture
def project():
    return "Sandbox"


@pytest.fixture
def example_server(user, password, project):
    return Cirxnat(
        address="https://cirxnat2.rcc.mcw.edu",
        user=user,
        password=password,
        project=project,
    )


@pytest.fixture
def example_subject_id():
    return "subject0004"


@pytest.fixture
def example_experiment_id():
    return "exam0004"


@pytest.fixture
def example_scan_num():
    return "2"


@pytest.fixture
def tests_data_path():
    return Path(os.path.join(os.path.dirname(__file__), "data"))


@pytest.fixture
def orig_mgz(tests_data_path):
    return tests_data_path / "orig.mgz"


@pytest.fixture
def wm_mgz(tests_data_path):
    return tests_data_path / "wm.mgz"


@pytest.fixture
def xfm(tests_data_path):
    return tests_data_path / "talairach.xfm.lta"


@pytest.fixture
def cmap(tests_data_path):
    return tests_data_path / "FreeSurferColorLUT.txt"


@pytest.fixture
def aparc_aseg_data(tests_data_path):
    return tests_data_path / "aparc_aseg"


@pytest.fixture
def surf_data(tests_data_path):
    return tests_data_path / "surf"


@pytest.fixture
def label_data(tests_data_path):
    return tests_data_path / "label"


@pytest.fixture
def fake_freesurfer_home(tmp_path):
    fs_home = tmp_path / "freesurfer"
    fs_home.mkdir()

    return fs_home


@pytest.fixture
def fake_subjects_dir(tmp_path):
    subjects_dir = tmp_path / "subjects"
    subjects_dir.mkdir()

    return subjects_dir


@pytest.fixture
def fake_recon_all(tmp_path, example_subject_id):
    scripts_dir = tmp_path / "subjects" / example_subject_id / "scripts"
    scripts_dir.mkdir(parents=True)
    recon_file = scripts_dir / "recon-all.log"
    recon_file.write_text("finished without error")

    return


@pytest.fixture
def fake_tlrc_data(tmp_path, example_subject_id, xfm):
    transforms_dir = tmp_path / "subjects" / example_subject_id / "mri/transforms"
    transforms_dir.mkdir(parents=True)
    shutil.copy(xfm, transforms_dir)

    return
