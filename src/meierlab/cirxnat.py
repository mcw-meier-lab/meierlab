# vi: set ft=python sts=4 ts=4 sw=4 et:
"""XNAT base for python3 using requests."""

import json
import os
import time

import pandas as pd
import requests


class Cirxnat:
    """An instance of XNAT."""

    def __init__(self, address, project, user, password):
        self.address = address
        self.project = project
        self.user = str(user)
        self.password = str(password)
        self.proxy = {
            "http": os.getenv("http_proxy") if os.getenv("http_proxy") else None,
            "https": os.getenv("https_proxy") if os.getenv("https_proxy") else None,
        }
        self.session = requests.Session()
        self.session.verify = False
        self.session.auth = requests.auth.HTTPBasicAuth(self.user, self.password)
        self.session.proxies = self.proxy

    # Getters
    def get_user(self):
        """Get XNAT user.

        Returns
        -------
        str
            CIRXNAT user associated with current server.
        """
        return self.user

    def get_project(self):
        """Get XNAT project.

        Returns
        -------
        str
            CIRXNAT project associated with current server.
        """
        return self.project

    def _remove_doubles(self, url):
        """Remove double slashes for URL."""
        return url.replace("//data", "/data")

    def _get_base_url(self, ending=""):
        """Provide base URL string for retrieving data."""
        base_url = (
            f"{self.address}/data/archive/projects/" f"{self.project}/subjects{ending}"
        )
        return self._remove_doubles(base_url)

    def _get_dicom_url(self, ending=""):
        """Provide base URL for retrieving DICOM data."""
        dicom_url = (
            f"{self.address}/REST/services/dicomdump?src=/archive/projects/"
            f"{self.project}{ending}"
        )
        return self._remove_doubles(dicom_url)

    def get_address(self):
        """Get XNAT server address.

        Returns
        -------
        str
            CIRXNAT address associated with current server.
        """
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
        str
            Text string request result.
        """
        url = self._get_base_url()
        payload = {"format": mformat}
        response = self.session.get(url, params=payload)
        return response.text.rstrip()

    def get_subjects_json(self):
        """Get the subject list JSON and split it up into individual subjects.

        Returns
        -------
        list
            List of JSON dictionaries with subjects.

        Examples
        --------
        >>> import os
        >>> from meierlab import cirxnat
        >>> cir1_pass = os.getenv('CIR1')
        >>> cir1_addr = os.getenv('CIR1ADDR')
        >>> server = cirxnat.Cirxnat(address=cir1_addr,
        project='UWARC',user='lespana',password=cir1_pass)
        >>> subjects = server.get_subjects_json()
        >>> print(subjects[0])
        {'insert_date': '2016-07-20 12:51:48.678',
        'project': 'UWARC',
        'ID': 'XNAT_S00071',
        'label': '4_ASN_test_3D_TOF',
        'insert_user': 'admin',
        'URI': '/data/subjects/XNAT_S00071'}
        """
        response = self.get_subjects("json")
        return (json.loads(response))["ResultSet"]["Result"]

    def get_experiments(self, subject_id, mformat="csv"):
        """Get experiments associated with a subject.

        Parameters
        ----------
        subject_id : str
            Subject label from XNAT.
        mformat : str
            Output format. Default: 'csv'.

        Returns
        -------
        str
            Text string request result.
        """
        url = self._get_base_url(f"/{subject_id}/experiments")
        payload = {"format": mformat}
        response = self.session.get(url, params=payload)
        return response.text.rstrip()

    def get_experiments_json(self, subject_id):
        """Get the experiments JSON and split them up.

        Parameters
        ----------
        subject_id : str
            Subject label from XNAT.

        Returns
        -------
        list
            List of JSON dictionaries with subject's experiments.

        Examples
        --------
        >>> import os
        >>> from meierlab import cirxnat
        >>> cir1_pass = os.getenv('CIR1')
        >>> cir1_addr = os.getenv('CIR1ADDR')
        >>> server = cirxnat.Cirxnat(address=cir1_addr,
        project='UWARC',user='lespana',password=cir1_pass)
        >>> experiments = server.get_experiments_json('4_ASN_test_3D_TOF')
        >>> print(experiments[0])
        {'date': '2014-10-28',
        'xsiType': 'xnat:mrSessionData',
        'xnat:subjectassessordata/id': 'XNAT_E00130',
        'insert_date': '2016-07-20 13:07:19.808',
        'project': 'UWARC',
        'ID': 'XNAT_E00130',
        'label': '4_ASN_test_3D_TOF_28_10_2014_1',
        'URI': '/data/experiments/XNAT_E00130'}
        """
        response = self.get_experiments(subject_id, "json")
        return (json.loads(response))["ResultSet"]["Result"]

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
        str
            JSON string with note from XNAT.

        Examples
        --------
        >>> import os
        >>> from meierlab import cirxnat
        >>> cir1_pass = os.getenv('CIR1')
        >>> cir1_addr = os.getenv('CIR1ADDR')
        >>> server = cirxnat.Cirxnat(address=cir1_addr,
        project='UWARC',user='lespana',password=cir1_pass)
        >>> note = server.get_experiment_note_json(
            '4_QA_HUMAN','4_QA_HUMAN_11_11_2015_1')
        >>> print(note['note'])
        'Note:  There is nose wrap on all Sagittal Sequences.'
        """
        url = self._get_base_url(f"/{subject_id}/experiments/{experiment_id}")
        payload = {"format": "json"}
        response = self.session.get(url, params=payload)

        try:
            out_json = response.json()
            return out_json["items"][0]["data_fields"]
        except (RuntimeError, ValueError):
            return ""

    def get_scans(self, subject_id, experiment_id, mformat="csv"):
        """Get scans associated with a subject's experiment.

        Parameters
        ----------
        subject_id : str
            Subject label from XNAT.
        experiment_id : str
            Experiment label from XNAT.
        mformat : str
            Output format. Default: 'csv'.

        Returns
        -------
        str
            Text string request result.
        """
        url = self._get_base_url(f"/{subject_id}/experiments/{experiment_id}/scans")
        payload = {"format": mformat}
        response = self.session.get(url, params=payload)
        return response.text.rstrip()

    def get_scans_json(self, subject_id, experiment_id):
        """Get a subject's experiment's scans in JSON format.

        Parameters
        ----------
        subject_id : str
            Subject label from XNAT.
        experiment_id : str
            Experiment label from XNAT.

        Returns
        -------
        list
            List of scans from experiment.

        Examples
        --------
        >>> import os
        >>> from meierlab import cirxnat
        >>> cir1_pass = os.getenv('CIR1')
        >>> cir1_addr = os.getenv('CIR1ADDR')
        >>> server = cirxnat.Cirxnat(address=cir1_addr,
        project='UWARC',user='lespana',password=cir1_pass)
        >>> scans = server.get_scans_json(
        '4_QA_HUMAN','4_QA_HUMAN_11_11_2015_1')
        >>> print(scans[0])
        {'xsiType': 'xnat:mrScanData',
        'xnat_imagescandata_id': '1423',
        'note': '',
        'series_description': '3Plane Loc SSFSE',
        'ID': '1',
        'type': '3Plane Loc SSFSE',
        'URI': '/data/experiments/XNAT_E00131/scans/1',
        'quality': 'usable'}
        """
        response = self.get_scans(subject_id, experiment_id, "json")
        return (json.loads(response))["ResultSet"]["Result"]

    def get_scans_dictionary(self, subject_id, experiment_id):
        """
        Get a dictionary of scan IDs and their descriptions from an experiment.

        Parameters
        ----------
        subject_id : str
            Subject label from XNAT.
        experiment_id : str
            Experiment label from XNAT.

        Returns
        -------
        dict
            Dictionary of scan ID: series_description.

        Examples
        --------
        >>> import os
        >>> from meierlab import cirxnat
        >>> cir1_pass = os.getenv('CIR1')
        >>> cir1_addr = os.getenv('CIR1ADDR')
        >>> server = cirxnat.Cirxnat(address=cir1_addr,
        project='UWARC',user='lespana',password=cir1_pass)
        >>> scans = server.get_scans_dictionary(
            '4_QA_HUMAN','4_QA_HUMAN_11_11_2015_1')
        >>> print(scans)
        {'1': '3Plane Loc SSFSE',
        '2': '3Plane Loc SSFSE',
        ...,
        }
        """
        all_scans = self.get_scans_json(subject_id, experiment_id)
        scans_dictionary = {}
        for scan in all_scans:
            scans_dictionary[scan["ID"]] = scan["series_description"]
        return scans_dictionary

    def print_all_experiments(self):
        """Get a list of experiment dictionaries with experiment data.

        Returns
        -------
        list
            List of experiment dictionaries with labels,
            IDs, upload date, XNAT URI.

        Examples
        --------
        >>> import os
        >>> from meierlab import cirxnat
        >>> cir1_pass = os.getenv('CIR1')
        >>> cir1_addr = os.getenv('CIR1ADDR')
        >>> server = cirxnat.Cirxnat(address=cir1_addr,
        project='UWARC',user='lespana',password=cir1_pass)
        >>> all_experiments = server.print_all_experiments()
        >>> print(len(all_experiments))
        595
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

    def _parse_shadow_hdr(self, dcm_value):
        """Parse the Siemens shadow header to get head coil value.

        Parameters
        ----------
        dcm_value : list
            Shadow header value from `get_dicom_tag`.

        Returns
        -------
        str
            Number of connected coils corresponding to the head coil used.
        """
        vals = dcm_value.split("\n")
        try:
            value = str(len(list(filter(lambda x: "RxChannelConnected" in x, vals))))
        except Exception:
            value = ""

        return value

    def get_dicom_header(self, experiment_id, scan_num):
        r"""Get a JSON formatted subject experiment scan DICOM header.

        Parameters
        ----------
        experiment_id : str
            Experiment label from XNAT.
        scan_num : int
            Scan number from XNAT.

        Returns
        -------
        list
            JSON dictionary containing scan header information.

        Examples
        --------
        >>> import os
        >>> from meierlab import cirxnat
        >>> cir1_pass = os.getenv('CIR1')
        >>> cir1_addr = os.getenv('CIR1ADDR')
        >>> server = cirxnat.Cirxnat(address=cir1_addr,
        project='UWARC',user='lespana',password=cir1_pass)
        >>> hdr = server.get_dicom_header('4_ASN_test_3D_TOF_28_10_2014_1','1')
        >>> print(hdr[0])
        {'tag1': '(0002,0001)',
        'vr': 'OB',
        'value': '00\\01',
        'tag2': '',
        'desc': 'File Meta Information Version'}
        """
        url = self._get_dicom_url(f"/experiments/{experiment_id}/scans/{scan_num}")
        response = self.session.get(url).text.rstrip()
        return (json.loads(response))["ResultSet"]["Result"]

    def get_dicom_tag(self, subject_id, experiment_id, scan_num, tag_id):
        """Get DICOM tag value from id an 8 digit/text string.

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
        str
            JSON string with tag information.
        """
        url = self._get_dicom_url(
            f"/subjects/{subject_id}/experiments/{experiment_id}" f"/scans/{scan_num}"
        )
        payload = {"field": tag_id}
        response = self.session.get(url, params=payload).text.rstrip()
        return (json.loads(response))["ResultSet"]["Result"]

    def get_dicom_tags(self, experiment, scans_list, extra_tags=None):
        """Get common dicom tag info.

        Defaults include:
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
            "(0029,1020)": "channels",

        Parameters
        ----------
        experiment : str
            Experiment label from XNAT.
        scans_list : list
            List of scans to get data from.
        extra_tags : dict, optional
            Dictionary of tag (str) : label (str) to add to defaults.

        Returns
        -------
        dict
            Dictionary of scans DICOM tags: values.

        Examples
        --------
        >>> import os
        >>> from meierlab import cirxnat
        >>> cir1_pass = os.getenv('CIR1')
        >>> cir1_addr = os.getenv('CIR1ADDR')
        >>> server = cirxnat.Cirxnat(address=cir1_addr,
        project='UWARC',user='lespana',password=cir1_pass)
        >>> tags = server.get_dicom_tags('4_ASN_test_3D_TOF_28_10_2014_1',[1])
        >>> print(tags[1]['echo_time'])
        '1.448'
        """
        if extra_tags is None:
            extra_tags = {}
        tags = {
            "(0008,0008)": "image_type",
            "(0008,0022)": "scan_date",
            "(0008,0070)": "manufacturer",
            "(0008,103E)": "scan_desc",
            "(0008,1090)": "scanner",
            "(0010,0010)": "subject_id",
            "(0010,0020)": "session_id",
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
            "(0029,1020)": "channels",
        }
        if extra_tags:
            for key, val in extra_tags.items():
                tags[key] = val

        scans_dcm = {}
        for scan in scans_list:
            tag_vals = {}
            dcm_hdr = self.get_dicom_header(experiment_id=experiment, scan_num=scan)

            try:
                channel_hdr = next(
                    list(filter(lambda x: "RxChannelConnected" in x["value"], dcm_hdr))
                )["value"]
                tag_vals["channels"] = self._parse_shadow_hdr(channel_hdr)
            except Exception:
                tag_vals["channels"] = ""

            for dcm_tag in dcm_hdr:
                # pylint: disable=consider-iterating-dictionary
                if dcm_tag["tag1"] in tags.keys():
                    if tags[dcm_tag["tag1"]] != "channels":
                        tag_vals[tags[dcm_tag["tag1"]]] = dcm_tag["value"]

            scans_dcm[scan] = tag_vals

        return scans_dcm

    def get_scans_usability(self, subject_id, experiment_id, scan_list=None):
        """Get a dictionary of subject scans and their usability.

        Parameters
        ----------
        subject_id : str
            Subject label from XNAT.
        experiment_id : str
            Experiment label from XNAT.
        scan_list : list, optional
            List of scan descriptions to use.

        Returns
        -------
        dict
            Scan usability dictionary with ID, series_description, and QA note.

        Examples
        --------
        >>> import os
        >>> from meierlab import cirxnat
        >>> cir1_pass = os.getenv('CIR1')
        >>> cir1_addr = os.getenv('CIR1ADDR')
        >>> server = cirxnat.Cirxnat(address=cir1_addr,
        project='UWARC',user='lespana',password=cir1_pass)
        >>> usability = server.get_scans_usability(
            '4_ASN_test_3D_TOF','4_ASN_test_3D_TOF_28_10_2014_1',['2'])
        >>> print(usability)
        {'1': ['desc':'3Plane Loc  32ch', 'quality':'usable','note': '']}
        """
        if scan_list is None:
            scan_list = []
        url = self._get_base_url(f"/{subject_id}/experiments/{experiment_id}/scans")
        payload = {"format": "json"}
        response = self.session.get(url, params=payload).text.rstrip()

        all_scans = (json.loads(response))["ResultSet"]["Result"]
        scans_usability = {}
        for scan in all_scans:
            if not scan_list:
                scans_usability[scan["ID"]] = {
                    "desc": str(scan["series_description"]),
                    "quality": str(scan["quality"]),
                    "note": str(scan["note"]).replace(",", ";"),
                }
            else:
                for s in scan_list:
                    if s in scan["series_description"]:
                        scans_usability[scan["ID"]] = {
                            "desc": str(scan["series_description"]),
                            "quality": str(scan["quality"]),
                            "note": str(scan["note"]).replace(",", ";"),
                        }

        return scans_usability

    def zip_scans_to_file(self, subject_id, experiment_id, out_file, scan_list="ALL"):
        """Return a zip file with all of the resource files.

        Parameters
        ----------
        subject_id : str
            Subject label from XNAT.
        experiment_id : str
            Experiment label from XNAT.
        out_file : str
            Path to output file
        scan_list : list, optional
            List of scans to download. Default: 'ALL'
        """
        url = self._get_base_url(
            f"/{subject_id}/experiments/{experiment_id}/scans/{scan_list}/files"
        )
        payload = {"format": "zip"}
        response = self.session.get(
            url,
            params=payload,
            stream=True,
        )
        with open(out_file, "wb") as zip:
            for chunk in response.iter_content(chunk_size=512):
                if chunk:
                    zip.write(chunk)

        return

    def _create_scan_ids(self, subject_id, experiment_id, scan_desc_list):
        """Convert a scan description to a scanID."""
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
        """Zip scans by series description.

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
        int
            0 if files downloaded successfully,
            1 if no matching scans were found.

        Examples
        --------
        >>> import os
        >>> from meierlab import cirxnat
        >>> cir1_pass = os.getenv('CIR1')
        >>> cir1_addr = os.getenv('CIR1ADDR')
        >>> server = cirxnat.Cirxnat(address=cir1_addr,
        project='UWARC',user='lespana',password=cir1_pass)
        >>> err = server.zip_scans_descriptions_to_file(
            '4_ASN_test_3D_TOF','4_ASN_test_3D_TOF_28_10_2014_1',['BRAVO'])
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

    def get_project_dcm_params(self, scan_list=None, extra_tags=None):
        """Get DICOM parameter information for all experiments in a project.

        Parameters
        ----------
        scan_list : list, optional
            List of scan descriptions to get DICOM information for.
            Default: all scans.

        Returns
        -------
        :class:`~pandas.DataFrame`
            DataFrame containing DICOM parameter information.

        Examples
        --------
        >>> import os
        >>> from meierlab import cirxnat
        >>> cir1_pass = os.getenv('CIR1')
        >>> cir1_addr = os.getenv('CIR1ADDR')
        >>> server = cirxnat.Cirxnat(address=cir1_addr,
        project='UWARC',user='lespana',password=cir1_pass)
        >>> dcm_params = server.get_project_dcm_params(scan_list=['BRAVO'])
        >>> dcm_params.head()
        Ax FSPGR BRAVO_channels ... SB DTI MCW Axial R/L_note
        4_ASN_test_3D_TOF_28_10_2014_1 ... NaN
        4_QA_HUMAN_11_11_2015_1 NaN ... NaN
        WISC_IH_1746_20_11_2015_3 NaN ... NaN
        4_WISC_IH_1746_03_05_2016_4 NaN ... NaN
        4_WISC_IH_1556_24_11_2015_1 NaN ... NaN
        [5 rows x 79 columns]
        """
        if extra_tags is None:
            extra_tags = {}
        if scan_list is None:
            scan_list = []
        proj_df = pd.DataFrame()
        experiments = self.print_all_experiments()
        for exp in experiments:
            exp_scans = {}
            scans = self.get_scans_dictionary(
                exp["subject_label"], exp["experiment_label"]
            )
            if not scan_list:
                scans_dcm = self.get_dicom_tags(
                    exp["experiment_label"], scans.keys(), extra_tags
                )
                usability = self.get_scans_usability(
                    exp["subject_label"], exp["experiment_label"]
                )
            else:
                new_scans = {}
                new_use = {}
                for scan_num, scan_desc in scans.items():
                    for s in scan_list:
                        if s in scan_desc:
                            new_scans[scan_num] = scan_desc
                            new_use[scan_num] = scan_desc

                scans_dcm = self.get_dicom_tags(
                    exp["experiment_label"], new_scans.keys(), extra_tags
                )
                usability = self.get_scans_usability(
                    exp["subject_label"], exp["experiment_label"], new_use.values()
                )

            for scan, tag_vals in scans_dcm.items():
                for tag, val in tag_vals.items():
                    column = f"{scans[scan]}_{tag}"
                    exp_scans[column] = val
            for scan, tag_vals in usability.items():
                for tag, val in tag_vals.items():
                    column = f"{scans[scan]}_{tag}"
                    exp_scans[column] = val

            exp_df = pd.DataFrame(exp_scans, index=[exp["experiment_label"]])
            proj_df = pd.concat([proj_df, exp_df])

        return proj_df
