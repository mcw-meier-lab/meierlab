from pathlib import Path


def fs_quality_wf(fs_dir, output_dir):
    """Workflow to generate all images and run FreeSurfer quality report.

    Parameters
    ----------
    fs_dir : :class: `~FreeSurfer`
        FreeSurfer class representing subject's FreeSurfer directory.
    output_dir : path or str
        Path to output directory.
    """
    img_dir = Path(output_dir / fs_dir.subject_id)
    img_dir.mkdir(parents=True, exist_ok=True)

    if fs_dir.recon_success:
        fs_dir.gen_tlrc_report(img_dir / "mri/transforms", img_dir)
        fs_dir.gen_aparcaseg_plots(img_dir)
        fs_dir.gen_surf_plots(img_dir)
        html_report = fs_dir.gen_report(f"{fs_dir.subject_id}.html", img_dir)

    else:
        print(">> Subject failed recon-all, SKIPPING")

    return html_report
