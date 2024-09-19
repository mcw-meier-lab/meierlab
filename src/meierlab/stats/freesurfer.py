"""Custom nipype interfaces for FreeSurfer stats commands"""

from pathlib import Path

from nipype.interfaces.base import (
    File,
    InputMultiObject,
    TraitedSpec,
    traits,
)
from nipype.interfaces.freesurfer.base import FSCommand, FSTraitedSpec


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
    """_summary_

    Parameters
    ----------
    FSCommand : _type_
        _description_
    """

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
    """

    Parameters
    ----------
    FSCommand : _type_
        _description_
    """

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
