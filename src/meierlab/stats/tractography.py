import gc
import logging
import multiprocessing
import os
import shutil
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from datetime import datetime
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
    out_dir.mkdir(parents=True, exist_ok=True)

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
    if data_dir.is_dir():
        groups = sorted(data_dir.iterdir())
    else:
        raise ValueError("Not a directory")

    for group in groups:
        group_path = data_dir / group
        if group_path.is_dir():
            subjects = sorted(group_path.iterdir())
            logging.info(
                f"First {len(subjects)} subjects in matrix belong to {group} group"
            )
            all_subjects.extend([group_path / sub for sub in subjects])

    N = len(all_subjects)
    logging.info(f"Processing {N} subjects")

    for bundle in atlas_bundles:
        mb = load_trk(str(bundle), reference="same", bbox_valid_check=False).streamlines
        mb_name = bundle.stem
        logging.info(f"Processing bundle: {mb_name}")

        # Cluster and get standard bundle
        cluster_mb = qb.cluster(mb)
        standard_mb = cluster_mb.centroids[0]

        # Initialize bundle data
        profiles_data[mb_name] = {}

        # Process each subject
        for subj_path in all_subjects:
            subject_id = subj_path.stem
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


def _estimate_bundle_memory_usage(bundle_path: str) -> int:
    """Estimate memory usage for a bundle file based on its file size.

    Parameters
    ----------
    bundle_path : str
        Path to the bundle file (.trk or .h5)

    Returns
    -------
    int
        Estimated memory usage in bytes
    """
    try:
        # Get file size
        file_size = os.path.getsize(bundle_path)

        # Different expansion factors based on file type
        if bundle_path.endswith(".h5"):
            # For HDF5 files, we estimate memory usage based on:
            # 1. File size (compressed data)
            # 2. Expected expansion factor when loaded into memory
            # 3. Additional overhead for pandas DataFrame operations
            estimated_memory = file_size * 5  # Conservative estimate
        else:
            # For .trk files, use the original estimation
            estimated_memory = file_size * 3  # Original estimate for .trk files

        return estimated_memory
    except Exception as e:
        logging.warning(f"Error estimating memory for {bundle_path}: {e}")
        return 1024 * 1024 * 100  # Default to 100MB if estimation fails


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


def _collect_paths(data_dir: Path):
    """Collect all paths from data directory.

    Parameters
    ----------
    data_dir : Path
        Path to data directory

    Returns
    -------
    tuple of (list of str, list of Path)
        List of groups and list of paths from data directory
    """
    all_subjects = []
    if data_dir.is_dir():
        groups = sorted(d.name for d in data_dir.iterdir())
    else:
        raise ValueError("Not a directory")

    for group in groups:
        group_path = data_dir / group
        if group_path.is_dir():
            subjects = sorted(s.name for s in group_path.iterdir())
            logging.info(
                f"First {len(subjects)} subjects in matrix belong to {group} group"
            )
            all_subjects.extend([group_path / sub for sub in subjects])

    return groups, all_subjects


def buan_shape_similarity(
    data_dir: Path,
    out_dir: Path,
    clust_thr=(5, 3, 1.5),
    threshold=6,
    num_workers=None,
    batch_size=10,
    memory_threshold=0.8,
    force=False,
):
    """Compute bundle shape similarity matrix for all subjects in parallel.

    Parameters
    ----------
    data_dir : Path
        Path to data directory
    out_dir : Path
        Path to output directory
    clust_thr : tuple, optional
        Clustering thresholds for bundle shape similarity
        Default is (5, 3, 1.5)
    threshold : float, optional
        Threshold for bundle shape similarity
        Default is 6
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
    out_dir.mkdir(parents=True, exist_ok=True)

    # Collect all subject paths
    _, all_subjects = _collect_paths(data_dir)
    N = len(all_subjects)
    logging.info(f"Processing {N} subjects")

    # Get all bundle files from first subject and sort by size
    bundles_dir = all_subjects[0] / "rec_bundles"
    bundles = sorted(b.name for b in bundles_dir.iterdir())

    # Sort bundles by size (smallest first)
    bundle_sizes = [
        (str(bun), _get_bundle_size(str(bundles_dir / bun))) for bun in bundles
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
                str(bundles_dir / bun)
            )

            if estimated_memory_needed > available_memory * memory_threshold:
                logging.warning(f"Not enough memory for bundle {bun}. Skipping...")
                continue

            # Load bundles in batches to manage memory
            all_bundles = []
            for sub in all_subjects:
                bundle_path = str(sub / "rec_bundles" / bun)
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
            with ProcessPoolExecutor(
                max_workers=num_workers, mp_context=multiprocessing.get_context("fork")
            ) as executor:
                for i, j, value in executor.map(_compute_similarity, args):
                    ba_matrix[i][j] = value
                    ba_matrix[j][i] = value  # Fill in symmetric part

            # Save results
            output_path = str(out_dir / (bun[:-4] + ".npy"))
            logging.info(f"Saving BA score matrix to {output_path}")
            np.save(output_path, ba_matrix)

            # Plot and save visualization
            plt.figure()
            plt.title(bun[:-4])
            plt.imshow(ba_matrix, cmap="Blues")
            plt.colorbar()
            plt.clim(0, 1)
            plt.savefig(str(out_dir / f"SM_{bun[:-4]}"))
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
        metric_files : list of str
            List of diffusion metric file paths
        metric_files_pam : list of str
            List of pam5 metric file paths
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
        metric_files,
        metric_files_pam,
        subject,
        group_id,
        no_disks,
        out_dir,
    ) = args

    try:
        # Load bundles
        mbundles = load_trk(
            mb_path, reference="same", bbox_valid_check=False
        ).streamlines
        bundles = load_trk(
            bd_path, reference="same", bbox_valid_check=False
        ).streamlines
        orig_bundles = load_trk(
            org_bd_path, reference="same", bbox_valid_check=False
        ).streamlines
    except Exception as e:
        logging.error(f"Error loading bundle {mb_path}: {e}")
        return

    if len(orig_bundles) <= 5:
        logging.warning(
            f"Skipping bundle {os.path.basename(mb_path)} - too few streamlines"
        )
        return

    # Compute assignment map
    indx = dsa.assignment_map(bundles, mbundles, no_disks)
    ind = np.array(indx)

    # Load and transform bundles
    _, affine = load_nifti(metric_files[0])
    affine_r = np.linalg.inv(affine)
    transformed_orig_bundles = dts.transform_streamlines(orig_bundles, affine_r)

    # Create a unique output directory for this process
    process_id = os.getpid()
    process_out_dir = Path(out_dir) / f"process_{process_id}"
    process_out_dir.mkdir(parents=True, exist_ok=True)

    # Process DTI metrics
    for ii in range(len(metric_files)):
        metric_path = metric_files[ii]
        try:
            metric_name = os.path.split(metric_path)[1][:-7]
            bm = os.path.split(mb_path)[1][:-4]
            logging.info(f"Processing metric {metric_name} for bundle {bm}")

            dt = {}
            metric, _ = load_nifti(metric_path)

            # Create a unique output directory for this metric and bundle
            metric_out_dir = process_out_dir / f"{metric_name}_{bm}"
            metric_out_dir.mkdir(parents=True, exist_ok=True)

            dsa.anatomical_measures(
                transformed_orig_bundles,
                metric,
                dt,
                metric_name,
                bm,
                subject.name,
                group_id,
                ind,
                str(metric_out_dir),
            )
        except Exception as e:
            logging.error(f"Error processing DTI metric {metric_path}: {e}")
            continue

    # Process pam5 metrics
    for metric_path in metric_files_pam:
        try:
            metric_name = _get_metric_name(metric_path)[0]
            bm = os.path.basename(mb_path)[:-4]

            logging.info(f"Processing metric {metric_name} for bundle {bm}")

            dt = {}
            metric = load_peaks(str(metric_path))

            # Create a unique output directory for this metric and bundle
            metric_out_dir = process_out_dir / f"{metric_name}_{bm}"
            metric_out_dir.mkdir(parents=True, exist_ok=True)

            dsa.peak_values(
                transformed_orig_bundles,
                metric,
                dt,
                metric_name,
                bm,
                subject.name,
                group_id,
                ind,
                str(metric_out_dir),
            )
        except Exception as e:
            logging.error(f"Error processing metric {metric_path}: {e}")
            continue

    # After processing is complete, merge the results into the main output directory
    try:
        for metric_dir in process_out_dir.iterdir():
            if metric_dir.is_dir():
                target_dir = Path(out_dir) / metric_dir.name
                target_dir.mkdir(parents=True, exist_ok=True)

                # Move all files from the process-specific directory to the target directory
                for file in metric_dir.glob("*"):
                    if file.is_file():
                        target_file = target_dir / file.name
                        lock_file = target_file.with_suffix(".lock")

                        # Try to acquire lock
                        try:
                            # Create lock file
                            with open(lock_file, "x") as f:
                                f.write(str(process_id))

                            if target_file.exists():
                                # If file exists, append the data
                                try:
                                    with pd.HDFStore(
                                        str(target_file), mode="a"
                                    ) as target_store:
                                        with pd.HDFStore(
                                            str(file), mode="r"
                                        ) as source_store:
                                            for key in source_store.keys():
                                                df = source_store.get(key)
                                                target_store.append(
                                                    key, df, format="t", append=True
                                                )
                                except Exception as e:
                                    logging.error(
                                        f"Error appending to HDF5 file {target_file}: {e}"
                                    )
                                    # If append fails, try to create a new file
                                    try:
                                        with pd.HDFStore(
                                            str(target_file), mode="w"
                                        ) as target_store:
                                            with pd.HDFStore(
                                                str(file), mode="r"
                                            ) as source_store:
                                                for key in source_store.keys():
                                                    df = source_store.get(key)
                                                    target_store.put(
                                                        key, df, format="t"
                                                    )
                                    except Exception as e:
                                        logging.error(
                                            f"Error creating new HDF5 file {target_file}: {e}"
                                        )
                                        continue
                            else:
                                # If file doesn't exist, just move it
                                shutil.move(str(file), str(target_file))
                        except FileExistsError:
                            # If lock file exists, wait and retry
                            time.sleep(0.1)
                            continue
                        finally:
                            # Remove lock file
                            try:
                                lock_file.unlink()
                            except Exception:
                                pass
    except Exception as e:
        logging.error(f"Error merging results for process {process_id}: {e}")
    finally:
        # Clean up the process-specific directory
        try:
            shutil.rmtree(process_out_dir)
        except Exception as e:
            logging.error(f"Error cleaning up process directory {process_out_dir}: {e}")


def buan_profiles(
    model_bundle_folder: Path,
    data_dir: Path,
    *,
    no_disks: int = 100,
    out_dir: Path = Path("bundle_profiles"),
    num_workers: int | None = None,
    batch_size: int = 5,
    memory_threshold: float = 0.8,
):
    """Compute bundle profiles for a subject using parallel processing.

    Parameters
    ----------
    model_bundle_folder : Path
        Path to model bundle directory
    data_dir : Path
        Path to data directory
    no_disks : int, optional
        Number of disks for assignment map
        Default is 100
    out_dir : Path, optional
        Output directory
        Default is bundle_profiles in current directory
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
    out_dir.mkdir(parents=True, exist_ok=True)

    groups, all_subjects = _collect_paths(data_dir)
    N = len(all_subjects)
    logging.info(f"Processing {N} subjects")

    # Get all bundle files
    mb = sorted(model_bundle_folder.glob("*.trk"))
    bundles_dir = all_subjects[0] / "rec_bundles"
    bundles = sorted(b.name for b in bundles_dir.iterdir())

    # Sort bundles by size (smallest first)
    bundle_sizes = [
        (str(bun), _get_bundle_size(str(bundles_dir / bun))) for bun in bundles
    ]
    bundle_sizes.sort(key=lambda x: x[1])
    bundles = [bun for bun, _ in bundle_sizes]

    for i in range(0, len(bundles), batch_size):
        batch = bundles[i : i + batch_size]
        logging.info(f"Processing batch of {len(batch)} bundles")

        for bun in batch:
            logging.info(f"Processing bundle: {bun}")

            # Check available memory before loading bundles
            available_memory = _get_available_memory()
            estimated_memory_needed = N * _estimate_bundle_memory_usage(
                str(bundles_dir / bun)
            )

            if estimated_memory_needed > available_memory * memory_threshold:
                logging.warning(f"Not enough memory for bundle {bun}. Skipping...")
                continue

            # Load bundles in batches to manage memory
            for sub in all_subjects:
                for g in groups:
                    if g in str(sub):
                        group_id = g
                        break

                bd = sorted(sub.glob("rec_bundles/*.trk"))
                org_bd = sorted(sub.glob("org_bundles/*.trk"))

                if not (len(mb) == len(bd) == len(org_bd)):
                    raise ValueError(
                        "Number of bundles in model, bundle, and original bundle folders must match"
                    )

                # Get metric files
                metric_files = sorted(sub.glob("anatomical_measures/*.nii.gz"))
                metric_files_pam = sorted(sub.glob("anatomical_measures/*.pam5"))

                if not (metric_files or metric_files_pam):
                    raise ValueError("No metric files found in metric folder")

                # Prepare arguments for parallel processing
                args_list = []
                for mb_path, bd_path, org_bd_path in zip(mb, bd, org_bd, strict=False):
                    args_list.append(
                        (
                            str(mb_path),
                            str(bd_path),
                            str(org_bd_path),
                            metric_files,
                            metric_files_pam,
                            sub,
                            group_id,
                            no_disks,
                            out_dir,
                        )
                    )

                # Process batch in parallel
                with ProcessPoolExecutor(
                    max_workers=num_workers,
                    mp_context=multiprocessing.get_context("fork"),
                ) as executor:
                    list(executor.map(_process_bundle_metrics, args_list))

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

    # Find the last underscore before the extension
    last_underscore = name.rfind("_", 0, ext_pos)
    if last_underscore == -1:
        return " ", " ", " "

    # Check if the part after the last underscore is a metric name
    # (not a hemisphere indicator like L or R)
    metric_part = name[last_underscore + 1 : ext_pos]
    if metric_part in ["L", "R"]:
        # If it's a hemisphere indicator, look for the second-to-last underscore
        second_last_underscore = name.rfind("_", 0, last_underscore)
        if second_last_underscore == -1:
            return " ", " ", " "
        return (
            name[second_last_underscore + 1 : ext_pos],
            name[:second_last_underscore],
            name[:ext_pos],
        )

    return name[last_underscore + 1 : ext_pos], name[:last_underscore], name[:ext_pos]


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

        # Create a unique output directory for this process
        process_id = os.getpid()
        process_out_dir = Path(out_dir) / f"process_{process_id}"
        process_out_dir.mkdir(parents=True, exist_ok=True)

        # Initialize arrays for results
        pvalues = np.zeros((no_disks, len(config.fixed_effects)))
        coefficients = np.zeros((no_disks, len(config.fixed_effects)))
        std_errors = np.zeros((no_disks, len(config.fixed_effects)))

        # Read the entire HDF5 file once
        try:
            df = pd.read_hdf(file_path, key=Path(file_path).stem)
        except Exception as e:
            logging.error(f"Error reading HDF5 file {file_path}: {e}")
            return

        # Process each disk
        for i in range(no_disks):
            disk_count = i + 1
            try:
                # Filter data for current disk
                disk_data = df[df["disk"] == disk_count]

                if len(disk_data) < 10:
                    logging.warning(
                        f"Not enough data for disk {disk_count} in {file_path}"
                    )
                    pvalues[i] = 1.0  # Set to non-significant
                    continue

                # Build and fit model
                formula = _build_lmm_formula(config, metric_name)
                logging.info(f"Fitting model: {formula}")

                if config.family == "gaussian":
                    md = smf.mixedlm(
                        formula, disk_data, groups=disk_data[config.random_effects[0]]
                    )
                else:
                    # Handle generalized LMM
                    md = smf.mixedlm(
                        formula,
                        disk_data,
                        groups=disk_data[config.random_effects[0]],
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

        # Save results to process-specific directory
        save_base = process_out_dir / save_name

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

        # After processing is complete, merge the results into the main output directory
        try:
            for file in process_out_dir.glob("*"):
                if file.is_file():
                    target_file = Path(out_dir) / file.name
                    if target_file.exists():
                        # If file exists, merge the data
                        if file.suffix == ".npy":
                            # For numpy files, we need to handle them differently
                            source_data = np.load(str(file))
                            target_data = np.load(str(target_file))
                            # Combine the data appropriately based on the file type
                            if "pvalues" in file.name:
                                # For pvalues, take the minimum (most significant)
                                combined_data = np.minimum(source_data, target_data)
                            else:
                                # For other metrics, take the mean
                                combined_data = (source_data + target_data) / 2
                            np.save(str(target_file), combined_data)
                        else:
                            # For other files (like plots), just move them
                            shutil.move(str(file), str(target_file))
                    else:
                        # If file doesn't exist, just move it
                        shutil.move(str(file), str(target_file))
        except Exception as e:
            logging.error(f"Error merging results for process {process_id}: {e}")
        finally:
            # Clean up the process-specific directory
            try:
                shutil.rmtree(process_out_dir)
            except Exception as e:
                logging.error(
                    f"Error cleaning up process directory {process_out_dir}: {e}"
                )

    except Exception as e:
        logging.error(f"Error processing file {file_path}: {e}")
        return


def buan_lmm_plots(
    h5_files: list[str],
    config: LMMConfig,
    *,
    no_disks: int = 100,
    out_dir: Path = Path("lmm_plots"),
    num_workers: int | None = None,
    batch_size: int = 5,
    memory_threshold: float = 0.8,
):
    """Apply linear mixed models to bundle metrics and generate plots.

    Parameters
    ----------
    h5_files : list of Path
        List of paths to HDF5 files containing metric data
    config : LMMConfig
        LMM configuration specifying model structure
    no_disks : int, optional
        Number of disks used for dividing bundle into segments
        Default is 100
    out_dir : Path, optional
        Output directory for results
        Default is lmm_plots in current directory
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
    out_dir.mkdir(parents=True, exist_ok=True)

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
        estimated_memory_needed = sum(
            _estimate_bundle_memory_usage(str(f)) for f in batch
        )

        if estimated_memory_needed > available_memory * memory_threshold:
            logging.warning("Not enough memory for batch. Reducing batch size...")
            batch = batch[: len(batch) // 2]
            if not batch:
                raise MemoryError("Not enough memory to process even a single file")

        # Prepare arguments for parallel processing
        args_list = [(f, no_disks, str(out_dir), config) for f in batch]

        # Process batch in parallel
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            list(executor.map(_process_lmm_file, args_list))

        # Clear memory after each batch
        gc.collect()

    logging.info("LMM processing completed successfully")
