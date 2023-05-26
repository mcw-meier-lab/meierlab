"""
Basic Downloads
===============

See how to use download tools and verify the data.
"""

#############################
# Basic Setup
# -----------

import os
from pathlib import Path
from zipfile import ZipFile
from meierlab import cirxnat

# Have CIRXNAT info saved in your environment (e.g., .bashrc)
# NOTE: this won't work if you don't have access to the project
devx_addr = os.getenv("CIR1ADDR")
devx_pw = os.getenv("CIR1")

# You only need to set the server once per session/project
server = cirxnat.Cirxnat(address=devx_addr,
                         project="UNC_ARC",
                         user="lespana",
                         password=devx_pw)

# We'll set this up as if we're getting ready to run a BIDS-app pipeline
source_dir = Path("./sourcedata")
source_dir.mkdir(exist_ok=True,parents=True)

#############################
# Download
# --------
experiments = server.print_all_experiments()
scan_descs = ['MPRAGE'] # This is a T1w scan

# Just do one subject for now
experiment = [exp for exp in experiments if exp['experiment_label'] == '1_UNC_PH_0001_21_5_2016_2'][0]
bids_subj = f"sub-{experiment['experiment_label'].replace('_','')}"

zip_filename = f"{experiment['experiment_label']}.zip"

server.zip_scan_descriptions_to_file(experiment['subject_label'],
                                    experiment['experiment_label'],
                                    scan_descs,
                                    zip_filename)
        
# Unzip the file
with ZipFile(zip_filename, "r") as zip_file:
    zip_file.extractall(path=source_dir)

# Cleanup
os.remove(zip_filename)

################################
# Setup
# -----

# Now we've downloaded and can convert DICOMS
from nipype.interfaces.dcm2nii import Dcm2niix
import shutil

raw_dir = Path("./rawdata")
raw_dir.mkdir(exist_ok=True,parents=True)
bids_dir = Path(f"{raw_dir}/{bids_subj}")
dcm_dirs = list(source_dir.glob(f"{experiment['experiment_label']}/**/DICOM"))

for dcm_dir in dcm_dirs:
    converter = Dcm2niix(source_dir=dcm_dir,
                         bids_format=True,
                         output_dir=dcm_dir)
    outputs = converter.run().outputs

    nii_file = Path(outputs.converted_files)
    bids_file = Path(outputs.bids)

    data_dir = Path(f"{bids_dir}/anat")
    data_dir.mkdir(exist_ok=True,parents=True)
    nii_filename = f"{bids_subj}_T1w.nii.gz"
    bids_filename = f"{bids_subj}_T1w.json"

    # Move files to the new BIDS directory
    nii_file.rename(data_dir / nii_filename)
    bids_file.rename(data_dir / bids_filename)

# Cleanup
shutil.rmtree(source_dir / experiment['experiment_label'])


####################################
# Look at the data
# ----------------

from nilearn import plotting

anat_nii = data_dir / f"{bids_subj}_T1w.nii.gz"
anat = plotting.plot_anat(anat_img=anat_nii)
anat