# -*- coding: utf-8 -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""Update Brad's XNAT base for python3"""
import subprocess
import json
import time
import os


class Cirxnat:
    """
    An instance of XNAT.
    """

    def __init__(self, address, project, user, password):
        self.address = address
        self.project = project
        self.user = user
        self.password = password
        self.cookie = address.replace("/", "_").replace(":", "_")
        self.proxy = os.getenv("http_proxy") if os.getenv("http_proxy") else "''"

    # Getters
    def get_user(self):
        """Return XNAT user"""
        return self.user

    def get_project(self):
        """Return XNAT project"""
        return self.project

    def _remove_doubles(self, curl_cmd):
        """Remove double slashes for URL"""
        return curl_cmd.replace("//data", "/data")

    def _get_curl_get_base(self, ending=""):
        """Provide common base curl string for retrieving data"""
        curl_base = (
            f"curl -k -c {self.cookie} -b {self.cookie}"
            f" -x {self.proxy}"
            f" -s -X GET -u {self.user}:{self.password}"
            f' -L "{self.address}/data/archive/projects/"'
            f'"{self.project}/subjects{ending}"'
        )
        return self._remove_doubles(curl_base)

    def get_address(self):
        """Return XNAT server address"""
        return self.address

    # Retrieval
    def get_subjects(self, mformat="csv"):
        """Returns a csv copy of the list of subjects"""
        curl_cmd = self._get_curl_get_base(f"?format={mformat}")
        curl_output = subprocess.run(
            curl_cmd, shell=True, check=True, capture_output=True, text=True
        ).stdout.rstrip()
        return curl_output

    def get_subjects_json(self):
        """Get the subject list JSON and split it up into individual subjects"""
        curl_output = self.get_subjects("json")
        return (json.loads(curl_output))["ResultSet"]["Result"]

    def get_experiments(self, subject_id, mformat="csv"):
        """Returns a csv of subject experiments"""
        curl_cmd = self._get_curl_get_base(
            f"/{subject_id}/experiments?format={mformat}"
        )
        curl_output = subprocess.run(
            curl_cmd, shell=True, check=True, capture_output=True, text=True
        ).stdout.rstrip()
        return curl_output

    def get_experiments_json(self, subject_id):
        """Get the experiments JSON and split them up"""
        curl_output = self.get_experiments(subject_id, "json")
        return (json.loads(curl_output))["ResultSet"]["Result"]

    def get_experiment_note_json(self, subject_id, experiment_id):
        """Get overall QA/scan note for experiment"""
        curl_cmd = self._get_curl_get_base(
            f"/{subject_id}/experiments/{experiment_id}?format=json"
        )
        curl_output = subprocess.run(
            curl_cmd, shell=True, check=True, capture_output=True, text=True
        ).stdout.rstrip()
        try:
            out_json = json.loads(curl_output)
            return out_json["items"][0]["data_fields"]
        except (RuntimeError, ValueError):
            return ""

    def get_scans(self, subject_id, experiment_id, mformat="csv"):
        """Returns a csv file of subject experiment scans"""
        curl_cmd = self._get_curl_get_base(
            f"/{subject_id}/experiments/{experiment_id}/scans?format={mformat}"
        )
        return subprocess.run(
            curl_cmd, shell=True, check=True, capture_output=True, text=True
        ).stdout.rstrip()

    def get_scans_json(self, subject_id, experiment_id):
        """Returns a list of scan dictionaries"""
        curl_output = self.get_scans(subject_id, experiment_id, "json")
        return (json.loads(curl_output))["ResultSet"]["Result"]

    def get_scans_list(self, subject_id, experiment_id):
        """Returns a list of subject experiment scans"""
        all_scans = self.get_scans_json(subject_id, experiment_id)
        scans_list = []
        for scan in all_scans:
            scans_list.append(scan["ID"])
        return scans_list

    def get_scans_descriptions(self, subject_id, experiment_id):
        """Returns a list of subject expeiments scan series descriptions"""
        curl_cmd = self._get_curl_get_base(
            f"/{subject_id}/experiments/{experiment_id}/scans?format=json"
        )
        curl_output = subprocess.run(
            curl_cmd, shell=True, check=True, capture_output=True, text=True
        ).stdout.rstrip()
        all_scans = (json.loads(curl_output))["ResultSet"]["Result"]
        scans_list = []
        for scan in all_scans:
            scans_list.append(scan["series_description"])
        return scans_list

    def print_all_experiments(self):
        """Returns a dictionary of all {exp, subj} on this server/project.
        Note python server instance contains one project!"""
        experiment_list = []
        # Get the subject list and split it up into individual subjects
        subjects = self.get_subjects_json()
        for subject in subjects:
            experiments = self.get_experiments_json(subject["ID"])
            for exp in experiments:
                exp_dict = {
                    "experiment_label": exp["label"],
                    "experiment_id": exp["ID"],
                    "subject_label": subject["label"],
                    "subject_id": subject["ID"],
                    "date": exp["date"],
                    "insert_date": exp["insert_date"],
                    "uri": exp["URI"],
                }
                experiment_list.append(exp_dict)
        return experiment_list

    def get_dicom_header(self, experiment_id, scan_num):
        """Returns a JSON formatted subject experiment scan DICOM header"""
        curl_cmd = (
            f"curl -k -c {self.cookie} -b {self.cookie}"
            f" -x {self.proxy}"
            f" -s -X GET -u {self.user}:{self.password}"
            f' -L "{self.address}/REST/services/dicomdump?src=/archive/projects/'
            f'{self.project}/experiments/{experiment_id}/scans/{scan_num}"'
        )
        curl_cmd = self._remove_doubles(curl_cmd)
        curl_output = subprocess.run(
            curl_cmd, shell=True, check=True, capture_output=True, text=True
        ).stdout.rstrip()
        return (json.loads(curl_output))["ResultSet"]["Result"]

    def get_dicom_tag(self, subject_id, experiment_id, scan_num, tag_id):
        """Pass in the DICOM tag id as an 8 digit string,
        or a valid text string, to get back the value"""

        curl_cmd = (
            f"curl -k -c {self.cookie} -b {self.cookie} -s"
            f" -x {self.proxy}"
            f" -u {self.user}:{self.password}"
            f' -L "{self.address}/data/services/dicomdump?src=/archive/projects/'
            f"{self.project}/subjects/{subject_id}/experiments/{experiment_id}/"
            f'scans/{scan_num}&field={tag_id}"'
        )
        curl_output = subprocess.run(
            curl_cmd, shell=True, check=True, capture_output=True, text=True
        ).stdout.rstrip()
        return (json.loads(curl_output))["ResultSet"]["Result"]

    def get_dicom_tags(self, experiment, scans_list):
        """Get common dicom tag info"""
        tags = {
            "(0008,0008)": "image_type",
            "(0018,0050)": "slice_thickness",
            "(0018,0080)": "repetition_time",
            "(0018,0081)": "echo_time",
            "(0018,0082)": "inversion_time",
            "(0018,0083)": "num_avgs",
            "(0018,0087)": "field_strength",
            "(0018,0091)": "echo_train_len",
            "(0018,0093)": "percent_sampling",
            "(0018,0094)": "percent_phase_fov",
            "(0018,0095)": "pixel_bandwidth",
            "(0018,1020)": "software_version",
            "(0018,1250)": "coil",
            "(0018,1310)": "acq_matrix",
            "(0018,1314)": "flip_angle",
            "(0018,1316)": "sar",
            "(0028,0010)": "rows",
            "(0028,0011)": "cols",
            "(0028,0030)": "pixel_spacing",
        }
        scans_dcm = {}
        for scan in scans_list:
            tag_vals = {}
            dcm_hdr = self.get_dicom_header(experiment_id=experiment, scan_num=scan)

            for dcm_tag in dcm_hdr:
                # pylint: disable=consider-iterating-dictionary
                if dcm_tag["tag1"] in tags.keys():
                    tag_vals[tags[dcm_tag["tag1"]]] = dcm_tag["value"]

            scans_dcm[scan] = tag_vals

        return scans_dcm

    def get_scans_usability(self, subject_id, experiment_id):
        """Returns a dictionary of subject scans and their usability"""
        curl_cmd = self._get_curl_get_base(
            f"/{subject_id}/experiments/{experiment_id}/scans?format=json"
        )
        curl_output = subprocess.run(
            curl_cmd, shell=True, check=True, capture_output=True, text=True
        ).stdout.rstrip()
        all_scans = (json.loads(curl_output))["ResultSet"]["Result"]
        scans_usability = {}
        for scan in all_scans:
            scans_usability[scan["ID"]] = [
                str(scan["series_description"]),
                str(scan["quality"]),
                str(scan["note"]).replace(",", ";"),
            ]

        return scans_usability

    def zip_scans_to_file(self, subject_id, experiment_id, out_file, scan_list="ALL"):
        """Returns a zip file with all of the resource files"""
        curl_cmd = (
            f"curl -k -c {self.cookie} -b {self.cookie}"
            f" -x {self.proxy}"
            f" -X GET -o {out_file} -s -u {self.user}:{self.password}"
            f" -L '{self.address}/data/archive/projects/{self.project}/"
            f"subjects/{subject_id}/experiments/{experiment_id}/"
            f"scans/{scan_list}/files?format=zip'"
        )
        curl_cmd = self._remove_doubles(curl_cmd)
        curl_output = subprocess.run(
            curl_cmd, shell=True, check=True, capture_output=True, text=True
        ).stdout.rstrip()
        return curl_output

    def _create_scan_ids(self, subject_id, experiment_id, scan_desc_list):
        """Converts a scan description to a scanID.
        More specialized version of that found in selectedDownload.py"""
        scan_ids = []
        scan_dict = self.get_scans_json(subject_id, experiment_id)
        # get the ID where series description matches scan desc
        for scan in scan_dict:
            if any(ss in scan["series_description"] for ss in scan_desc_list):
                scan_ids.append(scan["ID"])
        return ",".join(scan_ids)

    def zip_scan_descriptions_to_file(
        self, subject_id, experiment_id, descriptions, out_file
    ):
        """Zip scans by series description"""
        id_string = self._create_scan_ids(subject_id, experiment_id, descriptions)

        if id_string != "":
            self.zip_scans_to_file(
                subject_id, experiment_id, out_file, scan_list=id_string
            )
            time.sleep(30)
            error = 0

        else:
            # ID string found no matching scans
            error = 1

        return error
