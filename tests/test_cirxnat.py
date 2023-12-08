import pytest
import os
import pandas as pd

@pytest.mark.skipif("CIR2" not in os.environ, reason="local testing only")
def test_get_user(example_server):
    assert example_server.get_user() == "lespana"


@pytest.mark.skipif("CIR2" not in os.environ, reason="local testing only")
def test_get_project(example_server):
    assert example_server.get_project() == "CIRXNAT2"


@pytest.mark.skipif("CIR2" not in os.environ, reason="local testing only")
def test_get_address(example_server):
    assert example_server.get_address() == "https://cirxnat2.rcc.mcw.edu"


@pytest.mark.skipif("CIR2" not in os.environ, reason="local testing only")
def test_get_subjects_json(example_server):
    assert example_server.get_subjects_json()[0] == {
        "insert_date": "2023-09-27 17:31:21.352",
        "project": "Sandbox",
        "ID": "CIRXNAT2_S11400",
        "label": "subject0004",
        "insert_user": "admin",
        "URI": "/data/subjects/CIRXNAT2_S11400"
    }


@pytest.mark.skipif("CIR2" not in os.environ, reason="local testing only")
def test_get_experiments_json(example_server, example_subject_id):
    assert example_server.get_experiments_json(example_subject_id) == [
        {
            "date": "2023-02-19",
            "xsiType": "xnat:mrSessionData",
            "xnat:subjectassessordata/id": "CIRXNAT2_E18718",
            "insert_date": "2023-09-27 17:31:21.352",
            "project": "Sandbox",
            "ID": "CIRXNAT2_E18718",
            "label": "exam0004",
            "URI": "/data/experiments/CIRXNAT2_E18718"
        }
    ]


@pytest.mark.skipif("CIR2" not in os.environ, reason="local testing only")
def test_get_scans_json(example_server, example_subject_id, example_experiment_id):
    assert example_server.get_scans_json(example_subject_id, example_experiment_id)[
        0
    ] == {
        "xsiType": "xnat:mrScanData",
        "xnat_imagescandata_id": "236583",
        "note": "",
        "series_description": "T1W",
        "ID": "2",
        "type": "T1W",
        "URI": "/data/experiments/CIRXNAT2_E18718/scans/2",
        "quality": "usable"
    }


@pytest.mark.skipif("CIR2" not in os.environ, reason="local testing only")
def test_get_scans_dictionary(
    example_server,
    example_subject_id,
    example_experiment_id,
):
    assert example_server.get_scans_dictionary(
        example_subject_id, example_experiment_id
    ) == {
        "2": "T1W",
        "13": "ASL 1025PLD",
        "14": "ManPreScan ASL 1525PLD",
        "15": "ManPreScan ASL 2025PLD",
        "16": "ManPreScan ASL 2525PLD",
        "17": "ManPreScan ASL 3025PLD",
        "1350": "CBF:Feb 16 2023 13-17-43 CST",
        "1450": "CBF:Feb 16 2023 13-19-14 CST",
        "1550": "CBF:Feb 16 2023 13-20-42 CST",
        "1650": "CBF:Feb 16 2023 13-22-11 CST",
        "1750": "CBF:Feb 16 2023 13-23-50 CST",
    }


@pytest.mark.skipif("CIR2" not in os.environ, reason="local testing only")
def test_print_all_experiments(example_server):
    assert example_server.print_all_experiments() == [
    {
        "experiment_label": "exam0004",
        "experiment_id": "CIRXNAT2_E18718",
        "subject_label": "subject0004",
        "subject_id": "CIRXNAT2_S11400",
        "date": "2023-02-19",
        "insert_date": "2023-09-27 17:31:21.352",
        "uri": "/data/experiments/CIRXNAT2_E18718" 
    }]


@pytest.mark.skipif("CIR2" not in os.environ, reason="local testing only")
def test_get_dicom_header(
    example_server,
    example_experiment_id,
    example_scan_num,
):
    assert (
        example_server.get_dicom_header(example_experiment_id, example_scan_num) != []
    )


@pytest.mark.skipif("CIR2" not in os.environ, reason="local testing only")
def test_get_dicom_tag(
    example_server,
    example_subject_id,
    example_experiment_id,
    example_scan_num,
):
    assert example_server.get_dicom_tag(
        example_subject_id, example_experiment_id, example_scan_num, "00180081"
    ) == [
        {
            "tag1": "(0018,0081)",
            "vr": "DS",
            "value": "2.008",
            "tag2": "",
            "desc": "Echo Time"
        }
    ]
    # if tag retrieval fails
    assert (
        example_server.get_dicom_tag(
            example_subject_id,
            example_experiment_id,
            example_scan_num,
            "00180018",
        )
        == []
    )


@pytest.mark.skipif("CIR2" not in os.environ, reason="local testing only")
def test_get_scans_usability(
    example_server,
    example_subject_id,
    example_experiment_id,
):
    assert example_server.get_scans_usability(
        example_subject_id, example_experiment_id
    ) == {
        "2": ["T1W", "usable", ""],
        "13": ["ASL 1025PLD", "usable", ""],
        "14": ["ManPreScan ASL 1525PLD", "usable", ""],
        "15": ["ManPreScan ASL 2025PLD", "usable", ""],
        "16": ["ManPreScan ASL 2525PLD", "usable", ""],
        "17": ["ManPreScan ASL 3025PLD", "usable", ""],
        "1350": ["CBF:Feb 16 2023 13-17-43 CST", "usable", ""],
        "1450": ["CBF:Feb 16 2023 13-19-14 CST", "usable", ""],
        "1550": ["CBF:Feb 16 2023 13-20-42 CST", "usable", ""],
        "1650": ["CBF:Feb 16 2023 13-22-11 CST", "usable", ""],
        "1750": ["CBF:Feb 16 2023 13-23-50 CST", "usable", ""]
        }


@pytest.mark.skipif("CIR2" not in os.environ, reason="local testing only")
def test_get_experiment_note_json(
    example_server,
    example_subject_id,
    example_experiment_id,
):
    assert example_server.get_experiment_note_json(
        example_subject_id, example_experiment_id
    ) == {
        "subject_ID": "CIRXNAT2_S11400",
        "date": "2023-02-19",
        "dcmPatientId": "exam0004",
        "modality": "MR",
        "prearchivePath": "/data/xnat/prearchive/Sandbox/20230927_173109073/exam0004",
        "scanner/model": "SIGNA Premier",
        "project": "Sandbox",
        "scanner/manufacturer": "GE MEDICAL SYSTEMS",
        "label": "exam0004",
        "dcmPatientName": "subject0004",
        "UID": "1.2.276.0.7230010.3.1.4.8323328.2070476540",
        "fieldStrength": "3.0",
        "id": "CIRXNAT2_E18718",
        "ID": "CIRXNAT2_E18718",
        "session_type": "GE_COMPARE"
        }


@pytest.mark.skipif("CIR2" not in os.environ, reason="local testing only")
@pytest.mark.download
def test_zip_scan_descriptions_to_file_success(
    example_server,
    example_subject_id,
    example_experiment_id,
):
    assert (
        example_server.zip_scan_descriptions_to_file(
            example_subject_id,
            example_experiment_id,
            descriptions=["ZTE"],
            out_file="test_file.zip",
        )
        == 0
    )
    os.remove("test_file.zip")


@pytest.mark.skipif("CIR2" not in os.environ, reason="local testing only")
@pytest.mark.download
def test_zip_scan_description_to_file_fail(
    example_server,
    example_subject_id,
    example_experiment_id,
):
    # if description doesn't exist
    assert (
        example_server.zip_scan_descriptions_to_file(
            example_subject_id,
            example_experiment_id,
            descriptions=["MPRAGE"],
            out_file="test_file.zip",
        )
        == 1
    )
    assert (
        example_server.zip_scan_descriptions_to_file(
            example_subject_id,
            example_experiment_id,
            descriptions=[],
            out_file="test_file.zip",
        )
        == 1
    )


@pytest.mark.skipif("CIR2" not in os.environ, reason="local testing only")
def test_get_project_dcm_params(example_server):
    assert type(example_server.get_project_dcm_params()) == pd.DataFrame
