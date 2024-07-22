import os

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
    return os.path.join(os.path.dirname(__file__), "data")
