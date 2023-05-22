import pytest
import os
from meierlab.cirxnat import Cirxnat


@pytest.fixture
def user():
    return "lespana"


@pytest.fixture
def password():
    return os.getenv("DEVX")


@pytest.fixture
def project():
    return "Sandbox"


@pytest.fixture
def example_server(user, password, project):
    return Cirxnat(
        address="https://devxnat.rcc.mcw.edu",
        user=user,
        password=password,
        project=project,
    )


@pytest.fixture
def example_subject_id():
    return "06"


@pytest.fixture
def example_experiment_id():
    return "06_121720"


@pytest.fixture
def example_scan_num():
    return "1"


@pytest.fixture
def tests_data_path():
    return os.path.join(os.path.dirname(__file__), "data")
