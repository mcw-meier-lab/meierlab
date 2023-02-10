import pytest
import os


def test_get_user(example_server):
    assert example_server.get_user() == "lespana"


def test_get_project(example_server):
    assert example_server.get_project() == "Sandbox"


def test_get_address(example_server):
    assert example_server.get_address() == "https://devxnat.rcc.mcw.edu/apps"


def test_get_subjects_json(example_server):
    assert example_server.get_subjects_json()[0] == {
        "insert_date": "2021-03-18 11:36:15.236",
        "project": "Sandbox",
        "ID": "DEVX_S02749",
        "label": "25A",
        "insert_user": "bswearingen",
        "URI": "/data/subjects/DEVX_S02749",
    }


def test_get_experiments_json(example_server, example_subject_id):
    assert example_server.get_experiments_json(example_subject_id) == [
        {
            "date": "2020-12-17",
            "xsiType": "xnat:mrSessionData",
            "xnat:subjectassessordata/id": "DEVX_E07019",
            "insert_date": "2021-03-18 11:33:11.835",
            "project": "Sandbox",
            "ID": "DEVX_E07019",
            "label": "06_121720",
            "URI": "/data/experiments/DEVX_E07019",
        }
    ]


def test_get_scans_json(example_server, example_subject_id, example_experiment_id):
    assert example_server.get_scans_json(example_subject_id, example_experiment_id)[
        0
    ] == {
        "xsiType": "xnat:mrScanData",
        "xnat_imagescandata_id": "92755",
        "note": "",
        "series_description": "3 Plane Localizer",
        "ID": "1",
        "type": "3 Plane Localizer",
        "URI": "/data/experiments/DEVX_E07019/scans/1",
        "quality": "usable",
    }


def test_get_scans_list(
    example_server,
    example_subject_id,
    example_experiment_id,
):
    assert example_server.get_scans_list(example_subject_id, example_experiment_id) == [
        "1",
        "2",
        "3",
        "4",
        "5",
        "6",
        "7",
        "8",
        "9",
        "10",
        "11",
        "12",
        "500",
        "501",
        "502",
        "700",
        "701",
        "702",
        "900",
        "901",
        "902",
        "1100",
        "1101",
        "1102",
    ]


def test_get_scans_descriptions(
    example_server,
    example_subject_id,
    example_experiment_id,
):
    assert example_server.get_scans_descriptions(
        example_subject_id, example_experiment_id
    ) == [
        "3 Plane Localizer",
        "3 Plane Localizer",
        "HighRes_3Dspgr_2Nex",
        "ZTE -Pseudo-CT",
        "WATER: HighRes_LAVA_3Nex_0.8x0.8x1mm",
        "FOV_Check_P20_3Dspgr_R2_Mp1_ARC_HS",
        "WATER: UlnarDev_P20_LAVA_R2_Mp40_ARC_HS",
        "UlnarDev_P20_3Dspgr_R2_Mp40_ARC_HS",
        "WATER: WristFlex_P20_LAVA_R2_Mp40_ARC_HS",
        "WristFlex_P20_3Dspgr_R2_Mp40_ARC_HS",
        "WATER: Rotation_P20_LAVA_R2_Mp40_ARC_HS",
        "Rotation_P20_3Dspgr_R2_Mp40_ARC_HS",
        "FAT: HighRes_LAVA_3Nex_0.8x0.8x1mm",
        "InPhase: HighRes_LAVA_3Nex_0.8x0.8x1mm",
        "OutPhase: HighRes_LAVA_3Nex_0.8x0.8x1mm",
        "FAT: UlnarDev_P20_LAVA_R2_Mp40_ARC_HS",
        "InPhase: UlnarDev_P20_LAVA_R2_Mp40_ARC_HS",
        "OutPhase: UlnarDev_P20_LAVA_R2_Mp40_ARC_HS",
        "FAT: WristFlex_P20_LAVA_R2_Mp40_ARC_HS",
        "InPhase: WristFlex_P20_LAVA_R2_Mp40_ARC_HS",
        "OutPhase: WristFlex_P20_LAVA_R2_Mp40_ARC_HS",
        "FAT: Rotation_P20_LAVA_R2_Mp40_ARC_HS",
        "InPhase: Rotation_P20_LAVA_R2_Mp40_ARC_HS",
        "OutPhase: Rotation_P20_LAVA_R2_Mp40_ARC_HS",
    ]


def test_print_all_experiments(example_server):
    assert example_server.print_all_experiments() == [
        {
            "experiment_label": "25A",
            "experiment_id": "DEVX_E06530",
            "subject_label": "25A",
            "subject_id": "DEVX_S02749",
            "date": "2018-07-19",
            "insert_date": "2021-03-18 11:36:15.666",
            "uri": "/data/experiments/DEVX_E06530",
        },
        {
            "experiment_label": "06_121720",
            "experiment_id": "DEVX_E07019",
            "subject_label": "06",
            "subject_id": "DEVX_S03114",
            "date": "2020-12-17",
            "insert_date": "2021-03-18 11:33:11.835",
            "uri": "/data/experiments/DEVX_E07019",
        },
    ]


def test_get_dicom_header(
    example_server,
    example_experiment_id,
    example_scan_num,
):
    assert (
        example_server.get_dicom_header(example_experiment_id, example_scan_num) != []
    )


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
            "value": "78.72",
            "tag2": "",
            "desc": "Echo Time",
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


def test_get_scans_usability(
    example_server,
    example_subject_id,
    example_experiment_id,
):
    assert example_server.get_scans_usability(
        example_subject_id, example_experiment_id
    ) == {
        "1": ["3 Plane Localizer", "usable", ""],
        "2": ["3 Plane Localizer", "usable", ""],
        "3": ["HighRes_3Dspgr_2Nex", "usable", ""],
        "4": ["ZTE -Pseudo-CT", "usable", ""],
        "5": ["WATER: HighRes_LAVA_3Nex_0.8x0.8x1mm", "usable", ""],
        "6": ["FOV_Check_P20_3Dspgr_R2_Mp1_ARC_HS", "usable", ""],
        "7": ["WATER: UlnarDev_P20_LAVA_R2_Mp40_ARC_HS", "usable", ""],
        "8": ["UlnarDev_P20_3Dspgr_R2_Mp40_ARC_HS", "usable", ""],
        "9": ["WATER: WristFlex_P20_LAVA_R2_Mp40_ARC_HS", "usable", ""],
        "10": ["WristFlex_P20_3Dspgr_R2_Mp40_ARC_HS", "usable", ""],
        "11": ["WATER: Rotation_P20_LAVA_R2_Mp40_ARC_HS", "usable", ""],
        "12": ["Rotation_P20_3Dspgr_R2_Mp40_ARC_HS", "usable", ""],
        "500": ["FAT: HighRes_LAVA_3Nex_0.8x0.8x1mm", "usable", ""],
        "501": ["InPhase: HighRes_LAVA_3Nex_0.8x0.8x1mm", "usable", ""],
        "502": ["OutPhase: HighRes_LAVA_3Nex_0.8x0.8x1mm", "usable", ""],
        "700": ["FAT: UlnarDev_P20_LAVA_R2_Mp40_ARC_HS", "usable", ""],
        "701": ["InPhase: UlnarDev_P20_LAVA_R2_Mp40_ARC_HS", "usable", ""],
        "702": ["OutPhase: UlnarDev_P20_LAVA_R2_Mp40_ARC_HS", "usable", ""],
        "900": ["FAT: WristFlex_P20_LAVA_R2_Mp40_ARC_HS", "usable", ""],
        "901": ["InPhase: WristFlex_P20_LAVA_R2_Mp40_ARC_HS", "usable", ""],
        "902": ["OutPhase: WristFlex_P20_LAVA_R2_Mp40_ARC_HS", "usable", ""],
        "1100": ["FAT: Rotation_P20_LAVA_R2_Mp40_ARC_HS", "usable", ""],
        "1101": ["InPhase: Rotation_P20_LAVA_R2_Mp40_ARC_HS", "usable", ""],
        "1102": ["OutPhase: Rotation_P20_LAVA_R2_Mp40_ARC_HS", "usable", ""],
    }


def test_get_experiment_note_json(
    example_server,
    example_subject_id,
    example_experiment_id,
):
    assert example_server.get_experiment_note_json(
        example_subject_id, example_experiment_id
    ) == {
        "dcmPatientId": "06_121720",
        "dcmPatientWeight": 83.91,
        "subject_ID": "DEVX_S03114",
        "date": "2020-12-17",
        "modality": "MR",
        "project": "Sandbox",
        "scanner/model": "SIGNA Premier",
        "study_id": "2225",
        "label": "06_121720",
        "scanner/manufacturer": "GE MEDICAL SYSTEMS",
        "operator": "8",
        "acquisition_site": "MCW Premier",
        "dcmPatientName": "06",
        "UID": "1.2.840.113619.6.475.237616204123338346346467786375596601260",
        "scanner": "MCWPREM",
        "fieldStrength": "3.0",
        "session_type": "Kinematics",
        "time": "09:42:54",
        "ID": "DEVX_E07019",
        "id": "DEVX_E07019",
    }


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
