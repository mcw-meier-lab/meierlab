from pathlib import Path


def fs_quality_wf(fs_dir, output_dir, img_out=None):
    """Workflow to generate all images and run FreeSurfer quality report.

    Parameters
    ----------
    fs_dir : :class: `~FreeSurfer`
        FreeSurfer class representing subject's FreeSurfer directory.
    output_dir : path or str
        Path to output directory.

    Returns
    -------
    :class: `~pathlib.Path`

    Examples
    --------
    >>> from meierlab.quality.freesurfer import FreeSurfer
    >>> from meierlab.quality.workflows import fs_quality_wf
    >>> fs_dir = reeSurfer(home_dir='/opt/freesurfer', subjects_dir='/opt/data', subject_id='sub-001')
    >>> report = fs_quality_wf(fs_dir, output_dir='.')
    """
    img_dir = Path(output_dir) / fs_dir.subject_id
    img_dir.mkdir(parents=True, exist_ok=True)

    if fs_dir.recon_success:
        fs_dir.gen_tlrc_report(img_dir)
        fs_dir.gen_aparcaseg_plots(img_dir)
        fs_dir.gen_surf_plots(img_dir)
        html_report = fs_dir.gen_report(
            f"{fs_dir.subject_id}.html", img_dir, img_out=img_out
        )

    else:
        print(">> Subject failed recon-all, SKIPPING")

    return html_report
