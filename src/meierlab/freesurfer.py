from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import colors
from nipype.interfaces.freesurfer.preprocess import MRIConvert
from nipype.interfaces.fsl import FLIRT

from meierlab.datasets import MNI_NII


def get_FreeSurfer_colormap(freesurfer_home):
    """
    Generates matplotlib colormap from FreeSurfer LUT.
    Code from:
    https://github.com/Deep-MI/qatools-python/blob/freesurfer-module-releases/qatoolspython/createScreenshots.py

    Parameters
    ----------
    freesurfer_home : path or str representing a path to a directory
       Path corresponding to FREESURFER_HOME env var.

    Returns
    -------
    colormap : matplotlib.colors.ListedColormap
        A matplotlib compatible FreeSurfer colormap.
    """
    lut = pd.read_csv(
        freesurfer_home / "FreeSurferColorLUT.txt",
        sep=" ",
        comment="#",
        header=None,
        skipinitialspace=True,
        skip_blank_lines=True,
    )
    lut = np.array(lut)
    lut_tab = np.array(lut[:, (2, 3, 4, 5)] / 255, dtype="float32")
    lut_tab[:, 3] = 1
    colormap = colors.ListedColormap(lut_tab)

    return colormap


class FreeSurfer:
    """
    Base class to access FreeSurfer-specific files.
    """

    def __init__(self, home_dir, subjects_dir, subject_id):
        self.home_dir = Path(home_dir)
        self.subjects_dir = Path(subjects_dir)
        self.subject_id = subject_id
        self.recon_success = self.check_recon_all()

    def get_stats(self, file_name):
        """Utility function to retrieve a FreeSurfer stats file for processing.

        Parameters
        ----------
        file_name : str
            FreeSurfer stats file to retrieve. Usually ends in ".stats"

        Returns
        -------
        :class: `~pathlib.Path`
            Stats file path
        """
        stats_file = self.subjects_dir / self.subject_id / "stats" / file_name

        return stats_file

    def check_recon_all(self):
        """Verifies that the subject's FreeSurfer recon finished successfully.

        Returns
        -------
        bool
            True if FreeSurfer finished successfully.

        Raises
        ------
        Exception
            Errors if no recon-all file.
        """
        recon_file = self.subjects_dir / self.subject_id / "scripts/recon-all.log"

        if recon_file.exists():
            with open(recon_file) as reconall:
                line = reconall.readlines()[-1]
                if "finished without error" in line:
                    return True
                else:
                    return False
        else:
            raise Exception("Subject has no recon-all output.")

    def gen_tlrc_data(self, output_dir):
        if not isinstance(output_dir, Path):
            output_dir = Path(output_dir)

        # get inverse transform
        lta_file = (
            self.subjects_dir / self.subject_id / "mri/transforms/talairach.xfm.lta"
        )
        xfm = np.genfromtxt(lta_file, skip_header=5, max_rows=4)
        inverse_xfm = np.linalg.inv(xfm)
        np.savetxt(
            output_dir / "inv.xfm",
            inverse_xfm,
            fmt="%0.8f",
            delimiter=" ",
            newline="\n",
            encoding="utf-8",
        )

        # convert subject original T1 to nifti (for FSL)
        convert = MRIConvert(
            in_file=Path(self.subjects_dir, self.subject_id, "mri") / "orig.mgz",
            out_file=output_dir / "orig.nii.gz",
            out_type="niigz",
        )
        convert.run()

        # use FSL to convert template file to subject original space
        flirt = FLIRT(
            in_file=MNI_NII,
            reference=output_dir / "orig.nii.gz",
            out_file=output_dir / "mni2orig.nii.gz",
            in_matrix_file=output_dir / "inv.xfm",
            apply_xfm=True,
            out_matrix_file=output_dir / "out.mat",
        )
        flirt.run()

        return
