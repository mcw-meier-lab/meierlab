import datetime
import os
from pathlib import Path

import nest_asyncio
import numpy as np
import pandas as pd
from matplotlib import colors
from matplotlib import pyplot as plt
from nilearn import plotting
from nireports.interfaces.reporting.base import SimpleBeforeAfterRPT
from pydra import ShellCommandTask, Submitter
from pydra.engine.specs import ShellSpec, SpecInfo

from meierlab.datasets import MNI_NII
from meierlab.reports import Template

nest_asyncio.apply()


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
        sep=r"\s+",
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
        self.data_dir = self.subjects_dir / self.subject_id
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

        Examples
        --------
        >>> from meierlab.quality.freesurfer import FreeSurfer
        >>> fs_dir = FreeSurfer(home_dir='/opt/freesurfer', subjects_dir='/opt/data', subject_id='sub-001')
        >>> aseg = fs_dir.get_stats('aseg.stats')
        """
        stats_file = self.data_dir / "stats" / file_name

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
        recon_file = self.data_dir / "scripts/recon-all.log"

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
        """Generates inverse talairach data for report generation.

        Parameters
        ----------
        output_dir : :class: `~pathlib.Path` or str
            Path for intermediate file output.

        Examples
        --------
        >>> from meierlab.quality.freesurfer import FreeSurfer
        >>> fs_dir = FreeSurfer(home_dir='/opt/freesurfer', subjects_dir='/opt/data', subject_id='sub-001')
        >>> fs_dir.gen_tlrc_data('.')
        """

        if not isinstance(output_dir, Path):
            output_dir = Path(output_dir)

        # get inverse transform
        lta_file = self.data_dir / "mri/transforms/talairach.xfm.lta"
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
        mriconvert_info_spec = SpecInfo(
            name="Input",
            fields=[
                (
                    "in_file",
                    Path,
                    {
                        "help_string": "input file ...",
                        "position": 2,
                        "mandatory": True,
                        "argstr": "--input_volume",
                    },
                ),
                (
                    "out_file",
                    Path,
                    {
                        "help_string": "name of output...",
                        "position": 3,
                        "argstr": "--output_volume",
                    },
                ),
                (
                    "out_type",
                    str,
                    {
                        "help_string": "type of output...",
                        "position": 1,
                        "argstr": "--out_type",
                    },
                ),
            ],
            bases=(ShellSpec,),
        )
        convert = ShellCommandTask(
            name="convert",
            executable="mri_convert",
            in_file=Path(self.data_dir, "mri") / "orig.mgz",
            out_file=output_dir / "orig.nii.gz",
            out_type="nii",
            input_spec=mriconvert_info_spec,
        )
        print(convert.cmdline)

        with Submitter(plugin="cf") as sub:
            sub(convert)

        # use FSL to convert template file to subject original space
        flirt_info_spec = SpecInfo(
            name="Input",
            fields=[
                ("in_file", Path, {"help_string": "input file", "argstr": "-in"}),
                (
                    "reference",
                    Path,
                    {"help_string": "reference file", "argstr": "-ref"},
                ),
                ("out_file", Path, {"help_string": "output name", "argstr": "-out"}),
                (
                    "in_matrix_file",
                    Path,
                    {"help_string": "matrix file", "argstr": "-init"},
                ),
                (
                    "apply_xfm",
                    bool,
                    {"help_string": "transform to apply", "argstr": "-applyxfm"},
                ),
                (
                    "out_matrix_file",
                    Path,
                    {"help_string": "output matrix file", "argstr": "-omat"},
                ),
            ],
            bases=(ShellSpec,),
        )
        flirt = ShellCommandTask(
            name="flirt",
            executable="flirt",
            in_file=MNI_NII,
            reference=output_dir / "orig.nii.gz",
            out_file=output_dir / "mni2orig.nii.gz",
            in_matrix_file=output_dir / "inv.xfm",
            apply_xfm=True,
            out_matrix_file=output_dir / "out.mat",
            input_spec=flirt_info_spec,
        )

        with Submitter(plugin="cf") as sub:
            sub(flirt)

        return

    def gen_tlrc_report(self, output_dir, gen_data=True, tlrc_dir=None):
        """Generates a before and after report of Talairach registration. (Will also run file generation if needed.)

        Parameters
        ----------
        output_dir : :class: `~pathlib.Path` or str
            Path to SVG output.
        gen_data : bool, optional
            Generate inverse Talairach data, by default True
        tlrc_dir : :class: `~pathlib.Path` or str, optional
            Path to output of `gen_tlrc_data`. Default is the subject's mri/transforms directory.

        Returns
        -------
        svg
            SVG file generated from the niworkflows SimpleBeforeAfterRPT

        Examples
        --------
        >>> from meierlab.quality.freesurfer import FreeSurfer
        >>> fs_dir = FreeSurfer(home_dir='/opt/freesurfer', subjects_dir='/opt/data', subject_id='sub-001')
        >>> report = fs_dir.gen_tlrc_report(output_dir='.')
        """
        if tlrc_dir is None:
            tlrc_dir = self.data_dir / "mri/transforms"
        if not isinstance(output_dir, Path):
            output_dir = Path(output_dir)
        if not isinstance(tlrc_dir, Path):
            tlrc_dir = Path(tlrc_dir)

        mri_dir = self.data_dir / "mri"

        if gen_data:
            self.gen_tlrc_data(tlrc_dir)

        # use white matter segmentation to compare registrations
        report = SimpleBeforeAfterRPT(
            before=mri_dir / "orig.mgz",
            after=tlrc_dir / "mni2orig.nii.gz",
            wm_seg=mri_dir / "wm.mgz",
            before_label="Subject Orig",
            after_label="Template",
            out_report=output_dir / "tlrc.svg",
        )
        result = report.run()
        output = result.outputs.out_report

        return output

    def gen_aparcaseg_plots(self, output_dir, num_imgs=None):
        """Generate parcellation images (aparc & aseg).

        Parameters
        ----------
        output_dir : :class: `~pathlib.Path` or str
            Path or str representing path to output directory.
        num_imgs : int, optional
            Number of images/slices to make.

        Returns
        -------
        list
            List of image paths

        Examples
        --------
        >>> from meierlab.quality.freesurfer import FreeSurfer
        >>> fs_dir = FreeSurfer(home_dir='/opt/freesurfer', subjects_dir='/opt/data', subject_id='sub-001')
        >>> images = fs_dir.gen_aparcaseg_plots('.')
        """
        if num_imgs is None:
            num_imgs = 10

        mri_dir = self.data_dir / "mri"
        cmap = get_FreeSurfer_colormap(self.home_dir)

        # get parcellation and segmentation images
        plotting.plot_roi(
            mri_dir / "aparc+aseg.mgz",
            mri_dir / "T1.mgz",
            cmap=cmap,
            display_mode="mosaic",
            dim=-1,
            cut_coords=num_imgs,
            alpha=0.5,
            output_file=output_dir / "aseg.svg",
        )
        display = plotting.plot_anat(
            mri_dir / "brainmask.mgz",
            display_mode="mosaic",
            cut_coords=num_imgs,
            dim=-1,
        )
        display.add_contours(
            mri_dir / "lh.ribbon.mgz",
            colors="b",
            linewidths=0.5,
            levels=[0.5],
        )
        display.add_contours(
            mri_dir / "rh.ribbon.mgz",
            colors="r",
            linewidths=0.5,
            levels=[0.5],
        )
        display.savefig(output_dir / "aparc.svg")
        display.close()

        return [Path(output_dir / "aseg.svg"), Path(output_dir / "aparc.svg")]

    def gen_surf_plots(self, output_dir):
        """Generates pial, inflated, and sulcal images from various viewpoints

        Parameters
        ----------
        output_dir : path or str representing a path to a directory
            Surface plot output directory.

        Returns
        -------
        list
            List of generated SVG images

        Examples
        --------
        >>> from meierlab.quality.freesurfer import FreeSurfer
        >>> fs_dir = FreeSurfer(home_dir='/opt/freesurfer', subjects_dir='/opt/data', subject_id='sub-001')
        >>> images = fs_dir.gen_surf_plots('.')
        """

        surf_dir = self.data_dir / "surf"
        label_dir = self.data_dir / "label"
        cmap = get_FreeSurfer_colormap(self.home_dir)

        hemis = {"lh": "left", "rh": "right"}
        for key, val in hemis.items():
            pial = surf_dir / f"{key}.pial"
            inflated = surf_dir / f"{key}.inflated"
            sulc = surf_dir / f"{key}.sulc"
            white = surf_dir / f"{key}.white"
            annot = label_dir / f"{key}.aparc.annot"

            label_files = {pial: "pial", inflated: "infl", white: "white"}

            for surf, label in label_files.items():
                fig, axs = plt.subplots(2, 3, subplot_kw={"projection": "3d"})
                plotting.plot_surf_roi(
                    surf,
                    annot,
                    hemi=val,
                    view="lateral",
                    bg_map=sulc,
                    bg_on_data=True,
                    darkness=1,
                    cmap=cmap,
                    axes=axs[0, 0],
                    figure=fig,
                )
                plotting.plot_surf_roi(
                    surf,
                    annot,
                    hemi=val,
                    view="medial",
                    bg_map=sulc,
                    bg_on_data=True,
                    darkness=1,
                    cmap=cmap,
                    axes=axs[0, 1],
                    figure=fig,
                )
                plotting.plot_surf_roi(
                    surf,
                    annot,
                    hemi=val,
                    view="dorsal",
                    bg_map=sulc,
                    bg_on_data=True,
                    darkness=1,
                    cmap=cmap,
                    axes=axs[0, 2],
                    figure=fig,
                )
                plotting.plot_surf_roi(
                    surf,
                    annot,
                    hemi=val,
                    view="ventral",
                    bg_map=sulc,
                    bg_on_data=True,
                    darkness=1,
                    cmap=cmap,
                    axes=axs[1, 0],
                    figure=fig,
                )
                plotting.plot_surf_roi(
                    surf,
                    annot,
                    hemi=val,
                    view="anterior",
                    bg_map=sulc,
                    bg_on_data=True,
                    darkness=1,
                    cmap=cmap,
                    axes=axs[1, 1],
                    figure=fig,
                )
                plotting.plot_surf_roi(
                    surf,
                    annot,
                    hemi=val,
                    view="posterior",
                    bg_map=sulc,
                    bg_on_data=True,
                    darkness=1,
                    cmap=cmap,
                    axes=axs[1, 2],
                    figure=fig,
                )

                plt.savefig(output_dir / f"{key}_{label}.svg", dpi=300, format="svg")
                plt.close()

        imgs = sorted(Path(output_dir).glob("*svg"))
        return imgs

    def gen_report(self, out_name, output_dir, img_out=None, template=None):
        """Generate html report with FreeSurfer images.

        Parameters
        ----------
        out_name : str
            HTML file name
        output_dir : path or str representing path to a directory
            Location where html file will be output.
        img_out : path or str representing path to a directory, optional
            Location where SVG images are saved, default is subject's data directory.
        template : str, optional
            HTML template to use. Default is local freesurfer.html.

        Returns
        -------
        :class: `~pathlib.Path`
            Path to html file.

        Examples
        --------
        >>> from meierlab.quality.freesurfer import FreeSurfer
        >>> fs_dir = FreeSurfer(home_dir='/opt/freesurfer', subjects_dir='/opt/data', subject_id='sub-001')
        >>> report = fs_dir.gen_report(out_name='sub-001.html',output_dir='.')
        """
        if template is None:
            template = Path(
                os.path.join(os.path.dirname(__file__), "html/freesurfer.html")
            )
        if img_out is None:
            image_list = list(self.data_dir.glob("*/*svg"))
        else:
            image_list = list(Path(img_out).glob("*svg"))

        tlrc = []
        aseg = []
        surf = []

        for img in image_list:
            with open(img) as img_file:
                img_data = img_file.read()

            if "tlrc" in img.name:
                tlrc.append(img_data)
            elif "aseg" in img.name or "aparc" in img.name:
                aseg.append(img_data)
            else:
                labels = {
                    "lh_pial": "LH Pial",
                    "rh_pial": "RH Pial",
                    "lh_infl": "LH Inflated",
                    "rh_infl": "RH Inflated",
                    "lh_white": "LH White Matter",
                    "rh_white": "RH White Matter",
                }
                surf_tuple = (labels[img.stem], img_data)
                surf.append(surf_tuple)

        _config = {
            "timestamp": datetime.datetime.now().strftime("%Y-%m-%d, %H:%M"),
            "subject": self.subject_id,
            "tlrc": tlrc,
            "aseg": aseg,
            "surf": surf,
        }

        tpl = Template(str(template))
        tpl.generate_conf(_config, output_dir / out_name)

        return Path(output_dir / out_name)
