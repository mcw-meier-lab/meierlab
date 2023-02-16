# -*- coding: utf-8 -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""XNAT base for python3 using requests"""
import json
import time
import os
import requests
import pandas as pd


class Cirxnat:
    """
    An instance of XNAT.
    """

    def __init__(self, address, project, user, password):
        self.address = address
        self.project = project
        self.user = user
        self.password = password
        self.cookie = {f"{project}_cookie": address.replace("/", "_").replace(":", "_")}
        self.proxy = {
            "http": os.getenv("http_proxy") if os.getenv("http_proxy") else None,
            "https": os.getenv("https_proxy") if os.getenv("https_proxy") else None,
        }
        self.auth = (self.user, self.password)

    # Getters
    def get_user(self):
        """Return XNAT user"""
        return self.user

    def get_project(self):
        """Return XNAT project"""
        return self.project

    def _remove_doubles(self, url):
        """Remove double slashes for URL"""
        return url.replace("//data", "/data")

    def _get_base_url(self, ending=""):
        """Provide base URL string for retrieving data"""
        base_url = (
            f"{self.address}/data/archive/projects/{self.project}/subjects{ending}"
        )
        return self._remove_doubles(base_url)

    def _get_dicom_url(self, ending=""):
        """Provide base URL for retrieving DICOM data"""
        dicom_url = (
            f"{self.address}/REST/services/dicomdump?src=/archive/projects/"
            f"{self.project}{ending}"
        )
        return self._remove_doubles(dicom_url)

    def get_address(self):
        """Return XNAT server address"""
        return self.address

    # Retrieval
    def get_subjects(self, mformat="csv"):
        """Get subjects associated with project.

        Parameters
        ----------
        mformat : str
            Output format. Default: 'csv'.

        Returns
        -------
        Text string request result
        """
        url = self._get_base_url()
        payload = {"format": mformat}
        with requests.Session() as session:
            session.auth = requests.auth.HTTPBasicAuth(self.user, self.password)
            request = session.get(
                url,
                params=payload,
            )
        return request.text.rstrip()

    def get_subjects_json(self):
        """Get the subject list JSON and split it up into individual subjects

        Returns
        -------
        JSON object with subjects.
        """
        request = self.get_subjects("json")
        return (json.loads(request))["ResultSet"]["Result"]

    def get_experiments(self, subject_id, mformat="csv"):
        """Get experiments associated with a subject

        Parameters
        ----------
        subject_id : str
            Subject label from XNAT.
        mformat : str
            Output format. Default: 'csv'.

        Returns
        -------
        Text string request result.
        """
        url = self._get_base_url(f"/{subject_id}/experiments")
        payload = {"format": mformat}
        request = requests.get(
            url, auth=self.auth, cookies=self.cookie, proxies=self.proxy, params=payload
        )
        return request.text.rstrip()

    def get_experiments_json(self, subject_id):
        """Get the experiments JSON and split them up

        Parameters
        ----------
        subject_id : str
            Subject label from XNAT.

        Returns
        -------
        JSON object with subject's experiments.
        """
        request = self.get_experiments(subject_id, "json")
        return (json.loads(request))["ResultSet"]["Result"]

    def get_experiment_note_json(self, subject_id, experiment_id):
        """Get overall QA/scan note for experiment.

        Parameters
        ----------
        subject_id : str
            Subject label from XNAT.
        experiment_id : str
            Experiment label from XNAT.

        Returns
        -------
        JSON object with note from XNAT.
        """
        url = self._get_base_url(f"/{subject_id}/experiments/{experiment_id}")
        payload = {"format": "json"}
        request = requests.get(
            url, auth=self.auth, cookies=self.cookie, proxies=self.proxy, params=payload
        )

        try:
            out_json = json.loads(request)
            return out_json["items"][0]["data_fields"]
        except (RuntimeError, ValueError):
            return ""

    def get_scans(self, subject_id, experiment_id, mformat="csv"):
        """Get scans associated with a subject's experiment

        Parameters
        ----------
        subject_id : str
            Subject label from XNAT
        experiment_id : str
            Experiment label from XNAT
        mformat : str
            Output format. Default: 'csv'.

        Returns
        -------
        Text string request result.
        """
        url = self._get_base_url(f"/{subject_id}/experiments/{experiment_id}/scans")
        payload = {"format": mformat}
        request = requests.get(
            url, auth=self.auth, cookies=self.cookie, proxies=self.proxy, params=payload
        )
        return request.text.rstrip()

    def get_scans_json(self, subject_id, experiment_id):
        """Get a subject's experiment's scans in JSON format

        Parameters
        ----------
        subject_id : str
            Subject label from XNAT.
        experiment_id : str
            Experiment label from XNAT.

        Returns
        -------
        JSON object with scans from experiment.
        """
        request = self.get_scans(subject_id, experiment_id, "json")
        return (json.loads(request))["ResultSet"]["Result"]

    def get_scans_dictionary(self, subject_id, experiment_id):
        """Get a dictionary of scan IDs and their descriptions from an experiment

        Parameters
        ----------
        subject_id : str
            Subject label from XNAT
        experiment_id : str
            Experiment label from XNAT

        Returns
        -------
        Dictionary of scan ID: series_description
        """
        all_scans = self.get_scans_json(subject_id, experiment_id)
        scans_dictionary = {}
        for scan in all_scans:
            scans_dictionary[scan["ID"]] = scan["series_description"]
        return scans_dictionary

    def print_all_experiments(self):
        """Get a dictionary of subjects with experiment data.

        Returns
        -------
        List of experiment dictionaries with label, IDs, upload date, XNAT URI.
        """
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
        """Get a JSON formatted subject experiment scan DICOM header

        Parameters
        ----------
        experiment_id : str
            Experiment label from XNAT.
        scan_num : int
            Scan number from XNAT.

        Returns
        -------
        JSON object containing scan header information.
        """
        url = self._get_dicom_url(f"/experiments/{experiment_id}/scans/{scan_num}")
        request = requests.get(
            url, auth=self.auth, cookies=self.cookie, proxies=self.proxy
        ).text.rstrip()
        return (json.loads(request))["ResultSet"]["Result"]

    def get_dicom_tag(self, subject_id, experiment_id, scan_num, tag_id):
        """Pass in the DICOM tag id as an 8 digit string,
        or a valid text string, to get back the value

        Parameters
        ----------
        subject_id : str
            Subject label from XNAT.
        experiment_id : str
            Experiment label from XNAT.
        scan_num : int
            Scan number from XNAT.
        tag_id : str
            DICOM tag.

        Returns
        -------
        JSON object with tag information.
        """
        url = self._get_dicom_url(
            f"/subjects/{subject_id}/experiments/{experiment_id}/scans/{scan_num}"
        )
        payload = {"field": tag_id}
        request = requests.get(
            url, auth=self.auth, cookies=self.cookie, proxies=self.proxy, params=payload
        ).text.rstrip()
        return (json.loads(request))["ResultSet"]["Result"]

    def get_dicom_tags(self, experiment, scans_list, extra_tags={}):
        """Get common dicom tag info

        Parameters
        ----------
        experiment : str
            Experiment label from XNAT.
        scans_list : list
            List of scans to get data from.

        Returns
        -------
        Dictionary of scans DICOM tags: values.
        """
        tags = {
            "(0008,0008)": "image_type",
            "(0008,0070)": "manufacturer",
            "(0008,1090)": "scanner",
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
        if extra_tags:
            for key, val in extra_tags.items():
                tags[key] = val

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
        """Get a dictionary of subject scans and their usability

        Parameters
        ----------
        subject_id : str
            Subject label from XNAT.
        experiment_id : str
            Experiment label from XNAT.

        Returns
        -------
        Scan usability dictionary with ID, series_description, and QA note.
        """
        url = self._get_base_url(f"/{subject_id}/experiments/{experiment_id}/scans")
        payload = {"format": "json"}
        request = requests.get(
            url, auth=self.auth, cookies=self.cookie, proxies=self.proxy, params=payload
        ).text.rstrip()

        all_scans = (json.loads(request))["ResultSet"]["Result"]
        scans_usability = {}
        for scan in all_scans:
            scans_usability[scan["ID"]] = [
                str(scan["series_description"]),
                str(scan["quality"]),
                str(scan["note"]).replace(",", ";"),
            ]

        return scans_usability

    def zip_scans_to_file(self, subject_id, experiment_id, out_file, scan_list="ALL"):
        """Returns a zip file with all of the resource files

        Parameters
        ----------
        subject_id : str
            Subject label from XNAT.
        experiment_id : str
            Experiment label from XNAT.
        out_file : str
            Path to output file
        scan_list : list
            List of scans to download. Default: 'ALL'

        """
        url = self._get_base_url(
            f"/{subject_id}/experiments/{experiment_id}/scans/{scan_list}/files"
        )
        payload = {"format": "zip"}
        request = requests.get(
            url,
            auth=self.auth,
            cookies=self.cookie,
            proxies=self.proxy,
            params=payload,
            stream=True,
        )
        with open(out_file, "wb") as zip:
            for chunk in request.iter_content(chunk_size=512):
                if chunk:
                    zip.write(chunk)

        return

    def _create_scan_ids(self, subject_id, experiment_id, scan_desc_list):
        """Converts a scan description to a scanID."""
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
        """Zip scans by series description

        Parameters
        ----------
        subject_id : str
            Subject label from XNAT.
        experiment_id : str
            Experiment label from XNAt.
        descriptions : list
            List of scan_description strings.
        out_file : str
            Path to output file.

        Returns
        -------
        0 if files downloaded successfully, 1 if no matching scans were found.
        """
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

    def get_project_dcm_params(self):
        """Get a pandas DataFrame containing DICOM parameter information
        for all experiments in a project.

        Returns
        -------
        DataFrame
        """
        proj_df = pd.DataFrame()
        experiments = self.print_all_experiments()
        for exp in experiments:
            exp_scans = {}
            scans = self.get_scans_dictionary(
                exp["subject_label"], exp["experiment_label"]
            )
            scans_dcm = self.get_dicom_tags(exp["experiment_label"], scans.keys())

            for scan, tag_vals in scans_dcm.items():
                for tag, val in tag_vals.items():
                    column = f"{scans[scan]}_{tag}"
                    exp_scans[column] = val

            exp_df = pd.DataFrame(exp_scans, index=[exp["experiment_label"]])
            proj_df = pd.concat([proj_df, exp_df])

        return proj_df
