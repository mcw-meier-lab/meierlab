import gc
import logging
import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from datetime import datetime
from glob import glob
from pathlib import Path

import dipy.stats.analysis as dsa
import dipy.tracking.streamline as dts
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import psutil
import statsmodels.formula.api as smf
from dipy.io.image import load_nifti
from dipy.io.peaks import load_peaks
from dipy.io.streamline import load_tractogram, load_trk
from dipy.segment.bundles import (
    bundle_shape_similarity,
)
from dipy.segment.clustering import QuickBundles
from dipy.segment.featurespeed import ResampleFeature
from dipy.segment.metricspeed import AveragePointwiseEuclideanMetric


@dataclass
class LMMConfig:
    """Configuration for Linear Mixed Model analysis.

    Parameters
    ----------
    fixed_effects : list of str
        List of fixed effects terms (e.g., ['group', 'age', 'sex'])
    random_effects : list of str
        List of random effects terms (e.g., ['subject'])
    interaction_terms : list of str, optional
        List of interaction terms (e.g., ['group:age'])
    covariates : list of str, optional
        List of additional covariates
    family : str, optional
        Distribution family for generalized LMM (e.g., 'gaussian', 'binomial')
    link : str, optional
        Link function for generalized LMM
    formula : str, optional
        Custom formula string (overrides other settings if provided)
    """

    fixed_effects: list[str]
    random_effects: list[str]
    interaction_terms: list[str] | None = None
    covariates: list[str] | None = None
    family: str = "gaussian"
    link: str | None = None
    formula: str | None = None


def _create_html_report(profiles_data: dict, out_dir: Path, metrics: list[str]):
    """Create an HTML report with all profile plots and information.

    Parameters
    ----------
    profiles_data : dict
        Dictionary containing profile data for all subjects and bundles
    out_dir : Path
        Output directory
    metrics : list of str
        List of diffusion metrics
    """
    html_content = """<!DOCTYPE html>
<html>
<head>
    <title>AFQ Profile Report</title>
    <style>
        body {{font-family: Arial, sans-serif; margin: 20px;}}
        .bundle-section {{margin-bottom: 30px;}}
        .subject-section {{margin-bottom: 20px;}}
        img {{max-width: 100%; height: auto;}}
        table {{border-collapse: collapse; width: 100%;}}
        th, td {{border: 1px solid #ddd; padding: 8px; text-align: left;}}
        th {{background-color: #f2f2f2;}}
    </style>
</head>
<body>
    <h1>AFQ Profile Report</h1>
    <p>Generated on: {timestamp}</p>
"""

    html_content = html_content.format(
        timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    )

    for bundle_name, subjects_data in profiles_data.items():
        html_content += f"""
    <div class="bundle-section">
        <h2>Bundle: {bundle_name}</h2>
        <table>
            <tr>
                <th>Subject ID</th>
                {"".join(f"<th>{metric}</th>" for metric in metrics)}
            </tr>
"""

        for subject_id, metric_data in subjects_data.items():
            html_content += f"""
            <tr>
                <td>{subject_id}</td>
                {"".join(f"<td>{data['mean']:.4f} ± {data['std']:.4f}</td>" for data in metric_data.values())}
            </tr>
"""

        html_content += """
        </table>
    </div>
"""

    html_content += """
</body>
</html>
"""

    with open(out_dir / "afq_report.html", "w") as f:
        f.write(html_content)


def afq(atlas_dir: Path, data_dir: Path, out_dir: Path, metrics: list[str]):
    """Compute AFQ profiles for all subjects and bundles.

    Parameters
    ----------
    atlas_dir : Path
        Path to atlas directory
    data_dir : Path
        Path to data directory
    out_dir : Path
        Path to output directory
    metrics : list of str
        List of diffusion metrics to compute

    Notes
    -----
    This function processes diffusion metrics along white matter bundles using
    the AFQ (Automated Fiber Quantification) approach. It generates profiles
    for each subject and bundle, and creates an HTML report summarizing the results.
    """
    # Create output directory
    os.makedirs(out_dir, exist_ok=True)

    # Initialize data structures
    profiles_data = {}
    all_profiles_df = []

    # Load and process atlas bundles
    atlas_bundles = list(atlas_dir.glob("*.trk"))
    feature = ResampleFeature(nb_points=100)
    metric = AveragePointwiseEuclideanMetric(feature)
    qb = QuickBundles(threshold=np.inf, metric=metric)

    # Collect all subject paths using the same logic as buan_shape_similarity
    all_subjects = []
    if os.path.isdir(data_dir):
        groups = sorted(os.listdir(data_dir))
    else:
        raise ValueError("Not a directory")

    for group in groups:
        group_path = os.path.join(data_dir, group)
        if os.path.isdir(group_path):
            subjects = sorted(os.listdir(group_path))
            logging.info(
                f"First {len(subjects)} subjects in matrix belong to {group} group"
            )
            all_subjects.extend([os.path.join(group_path, sub) for sub in subjects])

    N = len(all_subjects)
    logging.info(f"Processing {N} subjects")

    for bundle in atlas_bundles:
        mb = load_trk(str(bundle), reference="same", bbox_valid_check=False).streamlines
        mb_name = os.path.basename(bundle).split(".trk")[0]
        logging.info(f"Processing bundle: {mb_name}")

        # Cluster and get standard bundle
        cluster_mb = qb.cluster(mb)
        standard_mb = cluster_mb.centroids[0]

        # Initialize bundle data
        profiles_data[mb_name] = {}

        # Process each subject
        for subj_path in all_subjects:
            subject_id = os.path.basename(subj_path)
            logging.info(f"Processing subject: {subject_id}")

            # Initialize subject data
            profiles_data[mb_name][subject_id] = {}

            # Find and load recognized bundle
            rec_bundle = list(
                Path(subj_path).glob(f"**/rec_bundles/*{mb_name}__recognized.trk")
            )
            if not rec_bundle:
                logging.warning(
                    f"No recognized bundle found for {subject_id} - {mb_name}"
                )
                continue

            rec_bundle = rec_bundle[0]
            rb = load_trk(
                str(rec_bundle), reference="same", bbox_valid_check=False
            ).streamlines
            oriented_rb = dts.orient_by_streamline(rb, standard_mb)
            w_diff = dsa.gaussian_weights(oriented_rb)

            # Process each metric
            for diff_metric in metrics:
                # Load diffusion data
                diff_data_path = list(
                    Path(subj_path).glob(f"**/anatomical_measures/{diff_metric}.nii.gz")
                )
                if not diff_data_path:
                    logging.warning(f"No {diff_metric} data found for {subject_id}")
                    continue

                diff_data, diff_affine = load_nifti(str(diff_data_path[0]))

                # Compute profile
                profile = dsa.afq_profile(
                    diff_data, oriented_rb, affine=diff_affine, weights=w_diff
                )

                # Store profile data
                profiles_data[mb_name][subject_id][diff_metric] = {
                    "profile": profile,
                    "mean": np.mean(profile),
                    "std": np.std(profile),
                }

                # Add to DataFrame
                for node_idx, value in enumerate(profile):
                    all_profiles_df.append(
                        {
                            "subject_id": subject_id,
                            "bundle": mb_name,
                            "metric": diff_metric,
                            "node": node_idx,
                            "value": value,
                        }
                    )

                # Create and save profile plot
                plt.figure(figsize=(8, 4))
                plt.plot(profile)
                plt.title(f"{subject_id} - {mb_name} - {diff_metric}")
                plt.xlabel("Node along bundle")
                plt.ylabel(diff_metric)
                plt.tight_layout()

                # Save plot
                plot_dir = out_dir / "plots" / mb_name
                os.makedirs(plot_dir, exist_ok=True)
                plt.savefig(plot_dir / f"{subject_id}_{diff_metric}.png")
                plt.close()

    # Save all profile data to CSV
    df = pd.DataFrame(all_profiles_df)
    df.to_csv(out_dir / "afq_profiles.csv", index=False)

    # Create HTML report
    _create_html_report(profiles_data, out_dir, metrics)

    logging.info("AFQ processing completed successfully")


def _get_bundle_size(bundle_path):
    """Get the size of a bundle file in bytes.

    Parameters
    ----------
    bundle_path : str
        Path to the bundle file

    Returns
    -------
    int
        Size of the bundle file in bytes
    """
    return os.path.getsize(bundle_path)


def _compute_similarity(args):
    """Helper function for parallel processing of bundle comparisons.

    Parameters
    ----------
    args : tuple
        Contains (i, j, bundle1, bundle2, rng, clust_thr, threshold)
        i : int
            First bundle index
        j : int
            Second bundle index
        bundle1 : Streamlines
            First bundle to compare
        bundle2 : Streamlines
            Second bundle to compare
        rng : numpy.random.Generator
            Random number generator
        clust_thr : float
            Clustering threshold
        threshold : float
            Similarity threshold

    Returns
    -------
    tuple
        (i, j, similarity_score)
    """
    i, j, bundle1, bundle2, rng, clust_thr, threshold = args
    return (
        i,
        j,
        bundle_shape_similarity(
            bundle1, bundle2, rng, clust_thr=clust_thr, threshold=threshold
        ),
    )


def _get_available_memory():
    """Get available system memory in bytes.

    Returns
    -------
    int
        Available system memory in bytes
    """
    return psutil.virtual_memory().available


def _estimate_bundle_memory_usage(bundle_path):
    """Estimate memory usage for a bundle based on its file size.

    Parameters
    ----------
    bundle_path : str
        Path to the bundle file

    Returns
    -------
    int
        Estimated memory usage in bytes (approximately 2-3x file size)
    """
    # Rough estimate: bundle in memory is about 2-3x its file size
    return _get_bundle_size(bundle_path) * 3


def _check_output_exists(out_dir: Path, bun: str) -> bool:
    """Check if output files for a bundle already exist.

    Parameters
    ----------
    out_dir : Path
        Output directory
    bun : str
        Bundle filename

    Returns
    -------
    bool
        True if both .npy and visualization files exist and are not empty
    """
    base_name = bun[:-4]  # Remove .trk extension
    npy_path = out_dir / f"{base_name}.npy"
    vis_path = out_dir / f"SM_{base_name}"

    # Check if both files exist and are not empty
    if npy_path.exists() and vis_path.exists():
        if npy_path.stat().st_size > 0:
            return True
    return False


def buan_shape_similarity(
    atlas_dir: Path,
    data_dir: Path,
    out_dir: Path,
    clust_thr=10,
    threshold=10,
    num_workers=None,
    batch_size=10,
    memory_threshold=0.8,
    force=False,
):
    """Compute bundle shape similarity matrix for all subjects in parallel.

    Parameters
    ----------
    atlas_dir : Path
        Path to atlas directory
    data_dir : Path
        Path to data directory
    out_dir : Path
        Path to output directory
    clust_thr : float, optional
        Clustering threshold for bundle shape similarity
        Default is 10
    threshold : float, optional
        Threshold for bundle shape similarity
        Default is 10
    num_workers : int, optional
        Number of parallel workers
        Default is None (uses all available CPUs)
    batch_size : int, optional
        Number of bundles to process in each batch
        Default is 10
    memory_threshold : float, optional
        Maximum fraction of available memory to use (0.0 to 1.0)
        Default is 0.8
    force : bool, optional
        If True, reprocess bundles even if output exists
        Default is False

    Notes
    -----
    This function computes shape similarity between bundles across all subjects
    using parallel processing and memory management to handle large datasets.
    """
    rng = np.random.default_rng()

    # Create output directory if it doesn't exist
    os.makedirs(out_dir, exist_ok=True)

    # Collect all subject paths
    all_subjects = []
    if os.path.isdir(data_dir):
        groups = sorted(os.listdir(data_dir))
    else:
        raise ValueError("Not a directory")

    for group in groups:
        group_path = os.path.join(data_dir, group)
        if os.path.isdir(group_path):
            subjects = sorted(os.listdir(group_path))
            logging.info(
                f"First {len(subjects)} subjects in matrix belong to {group} group"
            )
            all_subjects.extend([os.path.join(group_path, sub) for sub in subjects])

    N = len(all_subjects)
    logging.info(f"Processing {N} subjects")

    # Get all bundle files from first subject and sort by size
    bundles_dir = os.path.join(all_subjects[0], "rec_bundles")
    bundles = sorted(os.listdir(bundles_dir))

    # Sort bundles by size (smallest first)
    bundle_sizes = [
        (bun, _get_bundle_size(os.path.join(bundles_dir, bun))) for bun in bundles
    ]
    bundle_sizes.sort(key=lambda x: x[1])
    bundles = [bun for bun, _ in bundle_sizes]

    # Process bundles in batches
    for i in range(0, len(bundles), batch_size):
        batch = bundles[i : i + batch_size]
        logging.info(f"Processing batch of {len(batch)} bundles")

        for bun in batch:
            # Skip if output exists and force=False
            if not force and _check_output_exists(out_dir, bun):
                logging.info(f"Skipping {bun} - output already exists")
                continue

            logging.info(f"Processing bundle: {bun}")

            # Check available memory before loading bundles
            available_memory = _get_available_memory()
            estimated_memory_needed = N * _estimate_bundle_memory_usage(
                os.path.join(bundles_dir, bun)
            )

            if estimated_memory_needed > available_memory * memory_threshold:
                logging.warning(f"Not enough memory for bundle {bun}. Skipping...")
                continue

            # Load bundles in batches to manage memory
            all_bundles = []
            for sub in all_subjects:
                bundle_path = os.path.join(sub, "rec_bundles", bun)
                try:
                    bundle = load_tractogram(
                        bundle_path,
                        reference="same",
                        bbox_valid_check=False,
                    ).streamlines
                    all_bundles.append(bundle)
                except Exception as e:
                    logging.error(f"Error loading bundle {bundle_path}: {e}")
                    all_bundles.append(None)

            # Initialize similarity matrix
            ba_matrix = np.zeros((N, N))

            # Prepare arguments for parallel processing
            args = []
            for i in range(N):
                if all_bundles[i] is None:
                    continue
                for j in range(i, N):
                    if all_bundles[j] is None:
                        continue
                    args.append(
                        (
                            i,
                            j,
                            all_bundles[i],
                            all_bundles[j],
                            rng,
                            clust_thr,
                            threshold,
                        )
                    )

            # Process comparisons in parallel
            with ProcessPoolExecutor(max_workers=num_workers) as executor:
                for i, j, value in executor.map(_compute_similarity, args):
                    ba_matrix[i][j] = value
                    ba_matrix[j][i] = value  # Fill in symmetric part

            # Save results
            output_path = os.path.join(out_dir, bun[:-4] + ".npy")
            logging.info(f"Saving BA score matrix to {output_path}")
            np.save(output_path, ba_matrix)

            # Plot and save visualization
            plt.figure()
            plt.title(bun[:-4])
            plt.imshow(ba_matrix, cmap="Blues")
            plt.colorbar()
            plt.clim(0, 1)
            plt.savefig(os.path.join(out_dir, f"SM_{bun[:-4]}"))
            plt.close()

            # Clear memory
            del all_bundles
            gc.collect()


def _process_bundle_metrics(args):
    """Helper function for parallel processing of bundle metrics.

    Parameters
    ----------
    args : tuple
        Contains (mb_path, bd_path, org_bd_path, metric_files_dti,
                 metric_files_csa, subject, group_id, no_disks, out_dir)
        mb_path : str
            Path to model bundle
        bd_path : str
            Path to bundle
        org_bd_path : str
            Path to original bundle
        metric_files_dti : list of str
            List of DTI metric file paths
        metric_files_csa : list of str
            List of CSA metric file paths
        subject : str
            Subject identifier
        group_id : str
            Group identifier
        no_disks : int
            Number of disks for assignment map
        out_dir : str
            Output directory
    """
    (
        mb_path,
        bd_path,
        org_bd_path,
        metric_files_dti,
        metric_files_csa,
        subject,
        group_id,
        no_disks,
        out_dir,
    ) = args

    try:
        # Load bundles
        mbundles = load_tractogram(
            mb_path, reference="same", bbox_valid_check=False
        ).streamlines
        bundles = load_tractogram(
            bd_path, reference="same", bbox_valid_check=False
        ).streamlines
        orig_bundles = load_tractogram(
            org_bd_path, reference="same", bbox_valid_check=False
        ).streamlines

        if len(orig_bundles) <= 5:
            logging.warning(
                f"Skipping bundle {os.path.basename(mb_path)} - too few streamlines"
            )
            return

        # Compute assignment map
        indx = dsa.assignment_map(bundles, mbundles, no_disks)
        ind = np.array(indx)

        # Load and transform bundles
        _, affine = load_nifti(metric_files_dti[0])
        affine_r = np.linalg.inv(affine)
        transformed_orig_bundles = dts.transform_streamlines(orig_bundles, affine_r)

        # Process DTI metrics
        for metric_path in metric_files_dti:
            try:
                metric_name = os.path.basename(metric_path)[:-7]
                bm = os.path.basename(mb_path)[:-4]

                logging.info(f"Processing DTI metric {metric_name} for bundle {bm}")

                dt = {}
                metric, _ = load_nifti(metric_path)

                dsa.anatomical_measures(
                    transformed_orig_bundles,
                    metric,
                    dt,
                    metric_name,
                    bm,
                    subject,
                    group_id,
                    ind,
                    out_dir,
                )
            except Exception as e:
                logging.error(f"Error processing DTI metric {metric_path}: {e}")
                continue

        # Process CSA metrics
        for metric_path in metric_files_csa:
            try:
                metric_name = os.path.basename(metric_path)[:-5]
                bm = os.path.basename(mb_path)[:-4]

                logging.info(f"Processing CSA metric {metric_name} for bundle {bm}")

                dt = {}
                metric = load_peaks(metric_path)

                dsa.peak_values(
                    transformed_orig_bundles,
                    metric,
                    dt,
                    metric_name,
                    bm,
                    subject,
                    group_id,
                    ind,
                    out_dir,
                )
            except Exception as e:
                logging.error(f"Error processing CSA metric {metric_path}: {e}")
                continue

    except Exception as e:
        logging.error(f"Error processing bundle {os.path.basename(mb_path)}: {e}")
        return


def buan_profiles(
    model_bundle_folder: Path,
    bundle_folder: Path,
    orig_bundle_folder: Path,
    metric_folder: Path,
    group_id: str,
    subject: str,
    *,
    no_disks: int = 100,
    out_dir: str = "",
    num_workers: int | None = None,
    batch_size: int = 5,
    memory_threshold: float = 0.8,
):
    """Compute bundle profiles for a subject using parallel processing.

    Parameters
    ----------
    model_bundle_folder : Path
        Path to model bundle directory
    bundle_folder : Path
        Path to bundle directory
    orig_bundle_folder : Path
        Path to original bundle directory
    metric_folder : Path
        Path to metric files directory
    group_id : str
        Group identifier
    subject : str
        Subject identifier
    no_disks : int, optional
        Number of disks for assignment map
        Default is 100
    out_dir : str, optional
        Output directory
        Default is current directory
    num_workers : int, optional
        Number of parallel workers
        Default is None (uses all available cores)
    batch_size : int, optional
        Number of bundles to process in each batch
        Default is 5
    memory_threshold : float, optional
        Maximum fraction of available memory to use
        Default is 0.8

    Notes
    -----
    This function computes bundle profiles for a subject using parallel processing
    and memory management to handle large datasets efficiently.
    """
    # Create output directory
    os.makedirs(out_dir, exist_ok=True)

    # Get all bundle files
    mb = sorted(glob(os.path.join(model_bundle_folder, "*.trk")))
    bd = sorted(glob(os.path.join(bundle_folder, "*.trk")))
    org_bd = sorted(glob(os.path.join(orig_bundle_folder, "*.trk")))

    if not (len(mb) == len(bd) == len(org_bd)):
        raise ValueError(
            "Number of bundles in model, bundle, and original bundle folders must match"
        )

    # Get metric files
    metric_files_dti = sorted(glob(os.path.join(metric_folder, "*.nii.gz")))
    metric_files_csa = sorted(glob(os.path.join(metric_folder, "*.pam5")))

    if not (metric_files_dti or metric_files_csa):
        raise ValueError("No metric files found in metric folder")

    # Prepare arguments for parallel processing
    args_list = []
    for mb_path, bd_path, org_bd_path in zip(mb, bd, org_bd, strict=False):
        args_list.append(
            (
                mb_path,
                bd_path,
                org_bd_path,
                metric_files_dti,
                metric_files_csa,
                subject,
                group_id,
                no_disks,
                out_dir,
            )
        )

    # Process bundles in batches to manage memory
    for i in range(0, len(args_list), batch_size):
        batch = args_list[i : i + batch_size]
        logging.info(f"Processing batch of {len(batch)} bundles")

        # Check available memory
        available_memory = _get_available_memory()
        estimated_memory_needed = sum(
            _estimate_bundle_memory_usage(mb_path) for mb_path in batch[0]
        )

        if estimated_memory_needed > available_memory * memory_threshold:
            logging.warning("Not enough memory for batch. Reducing batch size...")
            batch = batch[: len(batch) // 2]
            if not batch:
                raise MemoryError("Not enough memory to process even a single bundle")

        # Process batch in parallel
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            list(executor.map(_process_bundle_metrics, batch))

        # Clear memory after each batch
        gc.collect()

    logging.info("Bundle profile processing completed successfully")


def _get_metric_name(path: str) -> tuple[str, str, str]:
    """Extract metric name, bundle name, and combined name from a file path.

    Parameters
    ----------
    path : str
        Path to the input metric file

    Returns
    -------
    tuple of str
        (metric_name, bundle_name, combined_name)
    """
    name = os.path.basename(path)
    ext_pos = name.rfind(".")
    if ext_pos == -1:
        return " ", " ", " "

    for i in range(len(name)):
        if name[i] == "_":
            if name[i + 1] not in ["L", "R"]:
                return name[i + 1 : ext_pos], name[:i], name[:ext_pos]

    return " ", " ", " "


def _save_lmm_plot(plot_file: str, title: str, bundle_name: str, x: list, y: list):
    """Save LMM plot with significance thresholds.

    Parameters
    ----------
    plot_file : str
        Path to save the plot
    title : str
        Plot title
    bundle_name : str
        Name of the bundle
    x : list
        List of segment/disk numbers
    y : list
        List of -log10(pvalues)
    """
    n = len(x)
    dotted = np.ones(n) * 2  # p < 0.01 threshold
    c1 = np.random.rand(1, 3)  # Random color for bars

    # Create figure with appropriate size
    plt.figure(figsize=(10, 6))

    # Plot significance thresholds
    (l1,) = plt.plot(
        x,
        dotted,
        color="red",
        marker=".",
        linestyle="solid",
        linewidth=0.6,
        markersize=0.7,
        label="p-value < 0.01",
    )

    (l2,) = plt.plot(
        x,
        dotted + 1,  # p < 0.001 threshold
        color="black",
        marker=".",
        linestyle="solid",
        linewidth=0.4,
        markersize=0.4,
        label="p-value < 0.001",
    )

    # Add first legend for thresholds
    first_legend = plt.legend(handles=[l1, l2], loc="upper right")

    # Plot the actual data
    axes = plt.gca()
    axes.add_artist(first_legend)
    axes.set_ylim([0, 6])

    l3 = plt.bar(x, y, color=c1, alpha=0.5, label=bundle_name)
    plt.legend(handles=[l3], loc="upper left")

    # Add labels and title
    plt.title(title.upper())
    plt.xlabel("Segment Number")
    plt.ylabel("-log10(Pvalues)")

    # Save and close
    plt.tight_layout()
    plt.savefig(plot_file)
    plt.close()


def _build_lmm_formula(config: LMMConfig, metric_name: str) -> str:
    """Build formula string for LMM based on configuration.

    Parameters
    ----------
    config : LMMConfig
        LMM configuration object
    metric_name : str
        Name of the metric being analyzed

    Returns
    -------
    str
        Formula string for statsmodels
    """
    if config.formula:
        return config.formula

    # Start with the response variable
    formula_parts = [f"{metric_name} ~"]

    # Add fixed effects
    formula_parts.append(" + ".join(config.fixed_effects))

    # Add interaction terms if specified
    if config.interaction_terms:
        formula_parts.append(" + " + " + ".join(config.interaction_terms))

    # Add covariates if specified
    if config.covariates:
        formula_parts.append(" + " + " + ".join(config.covariates))

    # Add random effects
    if config.random_effects:
        formula_parts.append(" + (" + " + ".join(config.random_effects) + ")")

    return " ".join(formula_parts).replace("  ", " ")


def _process_lmm_file(args):
    """Process a single HDF file for LMM analysis.

    Parameters
    ----------
    args : tuple
        Contains (file_path, no_disks, out_dir, config)
        file_path : str
            Path to HDF5 file
        no_disks : int
            Number of disks/segments
        out_dir : str
            Output directory
        config : LMMConfig
            LMM configuration
    """
    file_path, no_disks, out_dir, config = args

    try:
        logging.info(f"Processing file: {file_path}")

        # Get metric and bundle names
        metric_name, bundle_name, save_name = _get_metric_name(file_path)
        if metric_name == " ":
            logging.warning(f"Could not extract metric name from {file_path}")
            return

        logging.info(f"Processing {metric_name} for bundle {bundle_name}")

        # Initialize arrays for results
        pvalues = np.zeros((no_disks, len(config.fixed_effects)))
        coefficients = np.zeros((no_disks, len(config.fixed_effects)))
        std_errors = np.zeros((no_disks, len(config.fixed_effects)))

        # Process each disk
        for i in range(no_disks):
            disk_count = i + 1
            try:
                # Read data for current disk
                df = pd.read_hdf(file_path, where=f"disk={disk_count}")

                if len(df) < 10:
                    logging.warning(
                        f"Not enough data for disk {disk_count} in {file_path}"
                    )
                    pvalues[i] = 1.0  # Set to non-significant
                    continue

                # Build and fit model
                formula = _build_lmm_formula(config, metric_name)
                logging.info(f"Fitting model: {formula}")

                if config.family == "gaussian":
                    md = smf.mixedlm(formula, df, groups=df[config.random_effects[0]])
                else:
                    # Handle generalized LMM
                    md = smf.mixedlm(
                        formula,
                        df,
                        groups=df[config.random_effects[0]],
                        family=config.family,
                        link=config.link,
                    )

                mdf = md.fit()

                # Store results
                pvalues[i] = mdf.pvalues[: len(config.fixed_effects)]
                coefficients[i] = mdf.params[: len(config.fixed_effects)]
                std_errors[i] = mdf.bse[: len(config.fixed_effects)]

            except Exception as e:
                logging.error(f"Error processing disk {disk_count} in {file_path}: {e}")
                pvalues[i] = 1.0  # Set to non-significant
                continue

        # Save results
        save_base = os.path.join(out_dir, save_name)

        # Save p-values
        np.save(f"{save_base}_pvalues.npy", pvalues)

        # Save coefficients and standard errors
        np.save(f"{save_base}_coefficients.npy", coefficients)
        np.save(f"{save_base}_std_errors.npy", std_errors)

        # Create and save plots for each fixed effect
        for j, effect in enumerate(config.fixed_effects):
            y = -1 * np.log10(pvalues[:, j])
            x = list(range(1, len(pvalues) + 1))

            plot_file = f"{save_base}_{effect}.png"
            _save_lmm_plot(plot_file, f"{metric_name} - {effect}", bundle_name, x, y)

            # Save effect-specific results
            np.save(f"{save_base}_{effect}_pvalues.npy", pvalues[:, j])
            np.save(f"{save_base}_{effect}_coefficients.npy", coefficients[:, j])
            np.save(f"{save_base}_{effect}_std_errors.npy", std_errors[:, j])

    except Exception as e:
        logging.error(f"Error processing file {file_path}: {e}")
        return


def buan_lmm_plots(
    h5_files: list[str],
    config: LMMConfig,
    *,
    no_disks: int = 100,
    out_dir: str = "",
    num_workers: int | None = None,
    batch_size: int = 5,
    memory_threshold: float = 0.8,
):
    """Apply linear mixed models to bundle metrics and generate plots.

    Parameters
    ----------
    h5_files : list of str
        List of paths to HDF5 files containing metric data
    config : LMMConfig
        LMM configuration specifying model structure
    no_disks : int, optional
        Number of disks used for dividing bundle into segments
        Default is 100
    out_dir : str, optional
        Output directory for results
        Default is current directory
    num_workers : int, optional
        Number of parallel workers
        Default is None (uses all available cores)
    batch_size : int, optional
        Number of files to process in each batch
        Default is 5
    memory_threshold : float, optional
        Maximum fraction of available memory to use
        Default is 0.8

    Raises
    ------
    ValueError
        If no input files are provided or configuration is invalid
    MemoryError
        If not enough memory is available to process files
    """
    # Create output directory
    os.makedirs(out_dir, exist_ok=True)

    # Validate input files
    if not h5_files:
        raise ValueError("No input files provided")

    # Validate configuration
    if not config.fixed_effects:
        raise ValueError("At least one fixed effect must be specified")
    if not config.random_effects:
        raise ValueError("At least one random effect must be specified")

    # Process files in batches
    for i in range(0, len(h5_files), batch_size):
        batch = h5_files[i : i + batch_size]
        logging.info(f"Processing batch of {len(batch)} files")

        # Check memory usage
        available_memory = _get_available_memory()
        estimated_memory_needed = sum(_estimate_bundle_memory_usage(f) for f in batch)

        if estimated_memory_needed > available_memory * memory_threshold:
            logging.warning("Not enough memory for batch. Reducing batch size...")
            batch = batch[: len(batch) // 2]
            if not batch:
                raise MemoryError("Not enough memory to process even a single file")

        # Prepare arguments for parallel processing
        args_list = [(f, no_disks, out_dir, config) for f in batch]

        # Process batch in parallel
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            list(executor.map(_process_lmm_file, args_list))

        # Clear memory after each batch
        gc.collect()

    logging.info("LMM processing completed successfully")
