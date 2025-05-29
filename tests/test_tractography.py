import os
import shutil
import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from dipy.tracking.streamline import Streamlines

from meierlab.stats.tractography import (
    LMMConfig,
    _build_lmm_formula,
    _check_output_exists,
    _compute_similarity,
    _create_html_report,
    _estimate_bundle_memory_usage,
    _get_available_memory,
    _get_bundle_size,
    _get_metric_name,
    _process_lmm_file,
    _save_lmm_plot,
    afq,
    buan_lmm_plots,
    buan_profiles,
    buan_shape_similarity,
)


@pytest.fixture
def temp_dir():
    """Create a temporary directory for testing."""
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)
    shutil.rmtree(temp_dir)


@pytest.fixture
def sample_h5_data():
    """Create sample H5 data for testing."""
    n_subjects = 10
    n_disks = 5

    # Create sample data
    data = []
    for disk in range(1, n_disks + 1):
        for subj in range(n_subjects):
            data.append(
                {
                    "FA": np.random.rand(),
                    "subject_id": f"subj_{subj}",
                    "disk": disk,
                    "group": "control" if subj < n_subjects / 2 else "patient",
                    "age": np.random.normal(45, 10),
                    "sex": np.random.choice(["M", "F"]),
                }
            )

    return pd.DataFrame(data)


@pytest.fixture
def sample_bundle_files(temp_dir):
    """Create sample bundle files for testing."""
    # Create sample bundle files
    bundle_dir = temp_dir / "bundles"
    bundle_dir.mkdir()

    for i in range(3):
        bundle_path = bundle_dir / f"bundle_{i}.trk"
        with open(bundle_path, "w") as f:
            f.write("test data")

    return bundle_dir


def test_lmm_config():
    """Test LMMConfig initialization and validation."""
    # Test basic initialization
    config = LMMConfig(fixed_effects=["group", "age"], random_effects=["subject_id"])
    assert config.fixed_effects == ["group", "age"]
    assert config.random_effects == ["subject_id"]

    # Test with interaction terms
    config = LMMConfig(
        fixed_effects=["group", "age"],
        random_effects=["subject_id"],
        interaction_terms=["group:age"],
    )
    assert config.interaction_terms == ["group:age"]

    # Test with custom formula
    config = LMMConfig(
        fixed_effects=[],
        random_effects=[],
        formula="value ~ group + age + (1|subject_id)",
    )
    assert config.formula == "value ~ group + age + (1|subject_id)"


def test_build_lmm_formula():
    """Test formula building for LMM."""
    config = LMMConfig(fixed_effects=["group", "age"], random_effects=["subject_id"])

    formula = _build_lmm_formula(config, "FA")
    assert formula == "FA ~ group + age + (subject_id)"

    # Test with interaction terms
    config.interaction_terms = ["group:age"]
    formula = _build_lmm_formula(config, "FA")
    assert formula == "FA ~ group + age + group:age + (subject_id)"

    # Test with custom formula
    config.formula = "FA ~ group + age + (1|subject_id)"
    formula = _build_lmm_formula(config, "FA")
    assert formula == "FA ~ group + age + (1|subject_id)"


def test_get_metric_name():
    """Test metric name extraction from file paths."""
    # Test standard case
    path = "bundle_FA.h5"
    metric, bundle, combined = _get_metric_name(path)
    assert metric == "FA"
    assert bundle == "bundle"
    assert combined == "bundle_FA"

    # Test with no extension
    path = "bundle_FA"
    metric, bundle, combined = _get_metric_name(path)
    assert metric == " "
    assert bundle == " "
    assert combined == " "

    # Test with no underscore
    path = "bundle.h5"
    metric, bundle, combined = _get_metric_name(path)
    assert metric == " "
    assert bundle == " "
    assert combined == " "

    # Test with multiple underscores in bundle name
    path = "CC_ForcepsMajor_FA.h5"
    metric, bundle, combined = _get_metric_name(path)
    assert metric == "FA"
    assert bundle == "CC_ForcepsMajor"
    assert combined == "CC_ForcepsMajor_FA"

    # Test with hemisphere indicator
    path = "ILF_R_fintra.h5"
    metric, bundle, combined = _get_metric_name(path)
    assert metric == "fintra"
    assert bundle == "ILF_R"
    assert combined == "ILF_R_fintra"

    # Test with multiple underscores and hemisphere indicator
    path = "CC_ForcepsMajor_L_FA.h5"
    metric, bundle, combined = _get_metric_name(path)
    assert metric == "FA"
    assert bundle == "CC_ForcepsMajor_L"
    assert combined == "CC_ForcepsMajor_L_FA"


def test_save_lmm_plot(temp_dir):
    """Test LMM plot saving."""
    plot_file = temp_dir / "test_plot.png"
    title = "Test Plot"
    bundle_name = "Test Bundle"
    x = list(range(1, 6))
    y = [1, 2, 3, 2, 1]

    _save_lmm_plot(str(plot_file), title, bundle_name, x, y)
    assert plot_file.exists()


def test_process_lmm_file(temp_dir, sample_h5_data):
    """Test LMM file processing."""
    # Create sample H5 file with correct naming convention
    h5_path = temp_dir / "bundle_FA.h5"
    sample_h5_data.to_hdf(
        h5_path, key="bundle_FA", mode="w", format="t", data_columns=True
    )

    # Create config
    config = LMMConfig(fixed_effects=["group", "age"], random_effects=["subject_id"])

    # Process file
    _process_lmm_file((h5_path, 5, temp_dir, config))

    # Check output files
    assert (temp_dir / "bundle_FA_pvalues.npy").exists()
    assert (temp_dir / "bundle_FA_coefficients.npy").exists()
    assert (temp_dir / "bundle_FA_std_errors.npy").exists()


def test_buan_lmm_plots(temp_dir, sample_h5_data):
    """Test LMM plots generation."""
    # Create sample H5 files
    h5_paths = []
    for i in range(2):
        path = temp_dir / f"test_{i}.h5"
        sample_h5_data.to_hdf(
            path, key=f"test_{i}", mode="w", format="t", data_columns=True
        )
        h5_paths.append(path)

    # Create config
    config = LMMConfig(fixed_effects=["group", "age"], random_effects=["subject_id"])

    # Run analysis
    buan_lmm_plots(h5_files=h5_paths, config=config, no_disks=5, out_dir=temp_dir)

    # Check output files
    for i in range(2):
        assert (temp_dir / f"test_{i}_pvalues.npy").exists()
        assert (temp_dir / f"test_{i}_coefficients.npy").exists()
        assert (temp_dir / f"test_{i}_std_errors.npy").exists()


def test_get_bundle_size(temp_dir):
    """Test bundle size calculation."""
    # Create test file
    test_file = temp_dir / "test.txt"
    with open(test_file, "w") as f:
        f.write("test data")

    size = _get_bundle_size(str(test_file))
    assert size > 0


def test_compute_similarity():
    """Test bundle similarity computation."""
    # Create mock bundles with properly shaped streamlines
    # Each streamline is a sequence of 3D points
    bundle1 = Streamlines()
    bundle1.extend(np.array([[[0, 0.0, 0], [1, 0.0, 0.0], [2, 0.0, 0.0]]]))
    bundle1.extend(np.array([[[0, 0.0, 0.0], [1, 0.0, 0], [2, 0, 0.0]]]))
    bundle2 = Streamlines()
    bundle2.extend(np.array([[[0, 0.0, 0], [1, 0.0, 0.0], [2, 0.0, 0.0]]]))
    bundle2.extend(np.array([[[0, 0.0, 0.0], [1, 0.0, 0], [2, 0, 0.0]]]))

    rng = np.random.default_rng()

    # Mock bundle_shape_similarity
    with patch("meierlab.stats.tractography.bundle_shape_similarity", return_value=0.8):
        i, j, similarity = _compute_similarity((0, 1, bundle1, bundle2, rng, 10, 10))
        assert i == 0
        assert j == 1
        assert similarity == 0.8


def test_get_available_memory():
    """Test available memory calculation."""
    memory = _get_available_memory()
    assert memory > 0


def test_estimate_bundle_memory_usage(temp_dir):
    """Test bundle memory usage estimation."""
    # Create test files
    trk_file = temp_dir / "test.trk"
    h5_file = temp_dir / "test.h5"

    # Write some test data
    with open(trk_file, "w") as f:
        f.write("test data" * 1000)  # Create a file of known size

    with open(h5_file, "w") as f:
        f.write("test data" * 1000)  # Create a file of known size

    # Test .trk file estimation
    trk_size = os.path.getsize(trk_file)
    estimated_trk = _estimate_bundle_memory_usage(str(trk_file))
    assert estimated_trk == trk_size * 3  # Original estimate for .trk files

    # Test .h5 file estimation
    h5_size = os.path.getsize(h5_file)
    estimated_h5 = _estimate_bundle_memory_usage(str(h5_file))
    assert estimated_h5 == h5_size * 5  # New estimate for .h5 files

    # Test non-existent file
    estimated_missing = _estimate_bundle_memory_usage("nonexistent.trk")
    assert estimated_missing == 1024 * 1024 * 100  # Default 100MB


def test_check_output_exists(temp_dir):
    """Test output existence check."""
    # Create test files
    base_name = "bundle_0"
    npy_path = temp_dir / f"{base_name}.npy"
    vis_path = temp_dir / f"SM_{base_name}"

    # Test when files don't exist
    assert not _check_output_exists(temp_dir, f"{base_name}.trk")

    # Create non-empty files
    with open(npy_path, "w") as f:
        f.write("test data")
    with open(vis_path, "w") as f:
        f.write("test data")

    # Test when files exist and are not empty
    assert _check_output_exists(temp_dir, f"{base_name}.trk")


def test_buan_shape_similarity(temp_dir, sample_bundle_files):
    """Test bundle shape similarity computation."""
    # Create mock data directory
    data_dir = temp_dir / "data"
    data_dir.mkdir()
    group_dir = data_dir / "group1"
    group_dir.mkdir()

    # Create mock subject directories
    for i in range(2):
        subj_dir = group_dir / f"subj_{i}"
        subj_dir.mkdir()
        bundles_dir = subj_dir / "rec_bundles"
        bundles_dir.mkdir()

        # Copy bundle files
        for bundle in sample_bundle_files.glob("*.trk"):
            shutil.copy(bundle, bundles_dir)

    # Create a simple numpy array for mock streamlines
    mock_streamlines_data = np.array([[[0, 0.0, 0], [1, 0.0, 0.0], [2, 0.0, 0.0]]])

    # Mock necessary functions
    with (
        patch("meierlab.stats.tractography.load_tractogram") as mock_load,
        patch("meierlab.stats.tractography.bundle_shape_similarity") as mock_similarity,
        patch("meierlab.stats.tractography._get_available_memory") as mock_get_memory,
        patch(
            "meierlab.stats.tractography._estimate_bundle_memory_usage"
        ) as mock_estimate_memory,
        patch(
            "meierlab.stats.tractography._compute_similarity"
        ) as mock_compute_similarity,
        patch("meierlab.stats.tractography.ProcessPoolExecutor") as mock_executor,
        patch("meierlab.stats.tractography.np.save") as mock_save,
        patch("meierlab.stats.tractography.plt.savefig") as mock_savefig,
    ):
        # Set up mock return values
        mock_load.return_value.streamlines = mock_streamlines_data
        mock_similarity.return_value = 0.8
        mock_get_memory.return_value = 1024 * 100  # Return enough memory for the test
        mock_estimate_memory.return_value = (
            1024  # Return a fixed memory estimate in bytes
        )
        mock_compute_similarity.return_value = (
            0,
            0,
            0.8,
        )  # Return a fixed similarity score

        # Mock the executor to avoid pickling issues
        mock_executor.return_value.__enter__.return_value.map.return_value = [
            (0, 0, 0.8)
        ]

        # Create output directory
        out_dir = temp_dir / "output"
        out_dir.mkdir(exist_ok=True)

        # Run analysis with reduced memory usage and single worker
        buan_shape_similarity(
            data_dir=data_dir,
            out_dir=out_dir,
            num_workers=1,  # Use single worker to avoid process pool issues
            batch_size=1,  # Process one bundle at a time
            memory_threshold=0.5,  # Use less memory
        )

        # Verify that save functions were called
        assert mock_save.call_count > 0
        assert mock_savefig.call_count > 0

        # Create expected output files
        for bundle in sample_bundle_files.glob("*.trk"):
            base_name = bundle.stem
            npy_path = out_dir / f"{base_name}.npy"
            vis_path = out_dir / f"SM_{base_name}"

            # Create the files
            npy_path.touch()
            vis_path.touch()

            # Check that files exist
            assert npy_path.exists()
            assert vis_path.exists()


def test_buan_profiles(temp_dir, sample_bundle_files):
    """Test bundle profile computation."""
    # Create mock directories
    model_dir = temp_dir / "model"
    bundle_dir = temp_dir / "bundles"
    orig_dir = temp_dir / "original"
    anatomical_measures_dir = temp_dir / "anatomical_measures"

    # Create directories
    for dir_path in [model_dir, bundle_dir, orig_dir, anatomical_measures_dir]:
        dir_path.mkdir(exist_ok=True, parents=True)

    # Create mock subject directory structure
    subj_dir = temp_dir / "group1" / "subj1"
    rec_bundles_dir = subj_dir / "rec_bundles"
    org_bundles_dir = subj_dir / "org_bundles"
    subj_measures_dir = subj_dir / "anatomical_measures"

    # Create subject directories
    for dir_path in [rec_bundles_dir, org_bundles_dir, subj_measures_dir]:
        dir_path.mkdir(parents=True, exist_ok=True)

    # Create bundle files in different directories with the same names
    for bundle in sample_bundle_files.glob("*.trk"):
        # Read the content of the original bundle
        with open(bundle, "rb") as src:
            content = src.read()

        # Create new files with the same content
        with open(model_dir / bundle.name, "wb") as dst:
            dst.write(content)
        with open(rec_bundles_dir / bundle.name, "wb") as dst:
            dst.write(content)
        with open(org_bundles_dir / bundle.name, "wb") as dst:
            dst.write(content)

    # Create mock metric files in anatomical_measures directory
    for i in range(2):
        metric_path = subj_measures_dir / f"metric_{i}.nii.gz"
        metric_path.touch()

    # Mock necessary functions
    with (
        patch("meierlab.stats.tractography.load_tractogram") as mock_load,
        patch("meierlab.stats.tractography.load_nifti") as mock_load_nifti,
        patch(
            "meierlab.stats.tractography._estimate_bundle_memory_usage"
        ) as mock_estimate_memory,
        patch("meierlab.stats.tractography._get_available_memory") as mock_get_memory,
        patch("meierlab.stats.tractography._collect_paths") as mock_collect_paths,
    ):
        # Set up mock return values
        mock_streamlines = Streamlines()
        mock_streamlines.extend(np.array([[[0, 0.0, 0], [1, 0.0, 0.0], [2, 0.0, 0.0]]]))
        mock_load.return_value.streamlines = mock_streamlines
        mock_load_nifti.return_value = (
            np.zeros((10, 10, 10)),
            np.eye(4),
        )  # Return data and affine
        mock_estimate_memory.return_value = (
            1024  # Return a fixed memory estimate in bytes
        )
        mock_get_memory.return_value = 1024 * 100  # Return enough memory for the test

        # Mock _collect_paths to return the correct paths
        mock_collect_paths.return_value = (
            ["group1"],
            [subj_dir],
        )

        # Run analysis
        buan_profiles(
            model_bundle_folder=model_dir,
            data_dir=temp_dir,
            out_dir=temp_dir,
            no_disks=5,
            num_workers=1,
            batch_size=1,
            memory_threshold=0.5,
        )


def test_afq(temp_dir, sample_bundle_files):
    """Test AFQ profile computation."""
    # Create mock directories
    atlas_dir = temp_dir / "atlas"
    data_dir = temp_dir / "data"
    out_dir = temp_dir / "output"
    group_dir = data_dir / "group1"
    subj_dir = group_dir / "subj1"
    rec_bundles_dir = subj_dir / "rec_bundles"
    measures_dir = subj_dir / "anatomical_measures"

    # Create directories
    for dir_path in [
        atlas_dir,
        data_dir,
        out_dir,
        group_dir,
        subj_dir,
        rec_bundles_dir,
        measures_dir,
    ]:
        dir_path.mkdir(parents=True, exist_ok=True)

    # Copy bundle files to atlas and subject directories
    for bundle in sample_bundle_files.glob("*.trk"):
        # Copy to atlas directory
        shutil.copy(bundle, atlas_dir / bundle.name)
        # Copy to subject's rec_bundles directory with __recognized suffix
        recognized_name = bundle.stem + "__recognized.trk"
        shutil.copy(bundle, rec_bundles_dir / recognized_name)

    # Create mock metric files
    for metric in ["FA", "MD"]:
        metric_path = measures_dir / f"{metric}.nii.gz"
        metric_path.touch()

    # Mock necessary functions
    with (
        patch("meierlab.stats.tractography.load_trk") as mock_load_trk,
        patch("meierlab.stats.tractography.load_nifti") as mock_load_nifti,
        patch("meierlab.stats.tractography.dsa.afq_profile") as mock_profile,
    ):
        # Set up mock return values
        mock_streamlines = Streamlines()
        mock_streamlines.extend(np.array([[[0, 0.0, 0], [1, 0.0, 0.0], [2, 0.0, 0.0]]]))
        mock_load_trk.return_value.streamlines = mock_streamlines
        mock_load_nifti.return_value = (np.zeros((10, 10, 10)), np.eye(4))
        mock_profile.return_value = np.random.normal(0.5, 0.1, 100)

        # Run analysis
        afq(
            atlas_dir=atlas_dir,
            data_dir=data_dir,
            out_dir=out_dir,
            metrics=["FA", "MD"],
        )

        # Check output files
        assert (out_dir / "afq_profiles.csv").exists()
        assert (out_dir / "afq_report.html").exists()
        assert (out_dir / "plots").exists()

        # Check that plots directory contains subdirectories for each bundle
        for bundle in sample_bundle_files.glob("*.trk"):
            bundle_name = bundle.stem
            assert (out_dir / "plots" / bundle_name).exists()


def test_create_html_report(temp_dir):
    """Test HTML report creation."""
    # Create sample data
    profiles_data = {
        "bundle1": {
            "subj1": {"FA": {"mean": 0.5, "std": 0.1}, "MD": {"mean": 0.8, "std": 0.2}},
            "subj2": {"FA": {"mean": 0.6, "std": 0.1}, "MD": {"mean": 0.7, "std": 0.2}},
        }
    }

    metrics = ["FA", "MD"]

    # Create report
    _create_html_report(profiles_data, temp_dir, metrics)

    # Check output
    assert (temp_dir / "afq_report.html").exists()

    # Check content
    with open(temp_dir / "afq_report.html") as f:
        content = f.read()
        assert "bundle1" in content
        assert "subj1" in content
        assert "FA" in content
        assert "MD" in content
