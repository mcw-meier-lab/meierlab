"""Custom nipype interfaces for FreeSurfer stats commands"""

import datetime
import math
import os
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from nipype.interfaces.base import (
    File,
    InputMultiObject,
    TraitedSpec,
    traits,
)
from nipype.interfaces.freesurfer.base import FSCommand, FSTraitedSpec

from meierlab.reports import Template


class AsegStatsInputSpec(FSTraitedSpec):
    # asegstats2table --subjects --meas volume --delimiter=comma --skip --tablefile
    subjects = InputMultiObject(
        traits.Str(),
        argstr="%s...",
        desc="subjects to pull stats from",
        mandatory=True,
        position=1,
    )
    meas = traits.Enum("volume", "mean", argstr="--meas %s", desc="measure to output")
    delim = traits.Enum(
        "comma",
        "tab",
        "space",
        "semicolon",
        argstr="--delimiter=%s",
    )
    skip = traits.Bool(argstr="--skip", desc="skip empty files")
    tablefile = File(
        argstr="--tablefile %s", exists=False, desc="Output file name", mandatory=True
    )
    transpose = traits.Bool(argstr="--transpose", desc="transpose table")
    segs = traits.Bool(argstr="--all-segs", desc="use all segs available")


class AsegStatsOutputSpec(TraitedSpec):
    out_table = File(desc="output file")


class AsegStats(FSCommand):
    _cmd = "asegstats2table --subjects"
    input_spec = AsegStatsInputSpec
    output_spec = AsegStatsOutputSpec

    def run(self, **inputs):
        return super().run(**inputs)

    def _list_outputs(self):
        outputs = self._outputs().get()
        outputs["out_table"] = self.inputs.tablefile
        return outputs


class AparcStatsInputSpec(FSTraitedSpec):
    # aparcstats2table --subjects --skip --delimiter=comma --meas area volume thickness --hemi --tablefile
    subjects = InputMultiObject(
        traits.Str(),
        argstr="%s...",
        mandatory=True,
        desc="subjects to pull aparc stats",
        position=1,
    )
    hemi = traits.Enum(
        "lh", "rh", argstr="--hemi %s", mandatory=True, desc="hemisphere to use"
    )
    meas = traits.Enum(
        "area",
        "volume",
        "thickness",
        "thicknessstd",
        "meancurv",
        "gauscurv",
        "foldind",
        "curvind",
        argstr="--meas %s",
        desc="measure",
    )
    delim = traits.Enum(
        "tab",
        "comma",
        "space",
        "semicolon",
        argstr="--delimiter=%s",
        desc="table delimiter",
    )
    skip = traits.Bool(argstr="--skip", desc="skip empty inputs")
    tablefile = File(
        argstr="--tablefile %s", mandatory=True, exists=False, desc="output file name"
    )
    transpose = traits.Bool(argstr="--transpose", desc="transpose table")


class AparcStatsOutputSpec(TraitedSpec):
    out_table = File(desc="output file")


class AparcStats(FSCommand):
    _cmd = "aparcstats2table --subjects"
    input_spec = AparcStatsInputSpec
    output_spec = AparcStatsOutputSpec

    def run(self, **inputs):
        return super().run(**inputs)

    def _list_outputs(self):
        outputs = self._outputs().get()
        outputs["out_table"] = self.inputs.tablefile
        return outputs


def get_aseg_stats(
    subjects,
    tablefile,
    meas="volume",
    delim="comma",
    skip=True,
    segs=True,
    output_dir=".",
):
    """Generate aseg table.

    Parameters
    ----------
    subjects : list
        List of subject IDs to use
    tablefile : str
        Name of output file
    meas : str, optional
        Choose from volume, area. By default "volume"
    delim : str, optional
        String delimiter to use, by default "comma"
    skip : bool, optional
        Skip rather than crash if missing data, by default True
    segs : bool, optional
        Use all-segs flag, by default True
    output_dir : str, optional
        Output directory, by default "."

    Returns
    -------
    Path
        Path to output tablefile.
    """
    aseg_cmd = AsegStats(
        subjects=subjects,
        meas=meas,
        delim=delim,
        skip=skip,
        tablefile=Path(output_dir, tablefile),
        segs=segs,
    )
    print(aseg_cmd.cmdline)
    aseg_file = aseg_cmd.run().outputs.out_table
    return Path(output_dir, aseg_file)


def get_aparc_stats(
    subjects,
    tablefile,
    measures=None,
    hemis=None,
    delim="comma",
    skip=True,
    output_dir=".",
):
    """Generate parcellation stats.

    Parameters
    ----------
    subjects : list
        List of subject IDs
    tablefile : str
        Name of output file
    measures : str, optional
        Choose one of , by default None
    hemis : str, optional
        Choose one of ['lh','rh'], will run both by default.
    delim : str, optional
        String delimiter, by default "comma"
    skip : bool, optional
        Skip rather than crash if missing data, by default True
    output_dir : str, optional
        Output directory, by default "."

    Returns
    -------
    list
        List of paths to output files
    """
    if measures is None:
        measures = ["area", "volume", "thickness"]
    if hemis is None:
        hemis = ["lh", "rh"]

    results = []

    for m in measures:
        for h in hemis:
            aparc_cmd = AparcStats(
                subjects=subjects,
                meas=m,
                hemi=h,
                delim=delim,
                skip=skip,
                tablefile=Path(output_dir, f"{h}_{m}_{tablefile}"),
            )
            print(aparc_cmd.cmdline)
            res = aparc_cmd.run()
            results.append(Path(output_dir, res.outputs.out_table))

    return results


def gen_fs_fig(df):
    """Utility function to generate figures

    Parameters
    ----------
    df : Pandas dataframe
        Dataframe from aseg/aparc files

    Returns
    -------
    matplotlib figure
        Matplotlib subplot figure with boxplots
    """
    fig, axs = plt.subplots(
        math.ceil(len(list(df.columns)[1:]) / 3), 3, figsize=(10, 50)
    )
    col = 0
    row = 0
    for roi in list(df.columns)[1:]:
        axs[row, col].boxplot(df[roi])
        axs[row, col].set_title(roi)

        if col == 2:
            col = 0
            row += 1
        else:
            col += 1

    fig.subplots_adjust(
        left=0.08, right=0.98, bottom=0.05, top=0.9, hspace=0.4, wspace=0.4
    )

    return fig


def gen_group_report(
    aseg_file, aparc_files, fig_dir, out_name, output_dir, template=None
):
    """Generate basic FreeSurfer stats report

    Parameters
    ----------
    aseg_file : str
        Name of aseg file
    aparc_files : list
        List of aparc file names
    fig_dir : str, path
        Output directory for figures
    out_name : str
        Output file name
    output_dir : str, path
        Output directory name
    template : str, optional
        HTML template file to use, by default freesurfer_group_stats.html

    Returns
    -------
    str, path
        Path to output html file
    """
    if template is None:
        template = Path(
            os.path.join(os.path.dirname(__file__), "html/freesurfer_group_stats.html")
        )

    aseg = pd.read_csv(aseg_file)
    fig = gen_fs_fig(aseg)
    fig.savefig(f"{fig_dir}/aseg.svg")

    measures = {}
    for a_file in aparc_files:
        aparc = pd.read_csv(a_file)
        meas = next(iter(aparc.columns))
        measures[f"aparc_{Path(a_file).name}.svg"] = (
            f"{meas.split('.')[0]}_{meas.split('.')[2]}"
        )
        fig = gen_fs_fig(aparc)
        fig.savefig(f"{fig_dir}/aparc_{Path(a_file).name}.svg")

    image_list = list(Path(fig_dir).glob("*svg"))
    aseg_list = []
    aparc_list = []
    for img in image_list:
        with open(img) as img_file:
            img_data = img_file.read()

        if "aseg" in img.name:
            aseg_list.append(img_data)
        elif "aparc" in img.name:
            aparc_tuple = (measures[img.name], img_data)
            aparc_list.append(aparc_tuple)

    _config = {
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d, %H:%M"),
        "aseg": aseg_list,
        "aparc": aparc_list,
    }

    tpl = Template(str(template))
    tpl.generate_conf(_config, output_dir / out_name)

    return Path(output_dir / out_name)
