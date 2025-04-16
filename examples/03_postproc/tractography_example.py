"""
Tractography Pipeline Examples
============================

This example demonstrates how to use different tractography pipeline configurations
for deterministic, probabilistic, and CMC tracking methods using DIPY's built-in datasets.

The example shows:
1. Loading configuration files
2. Loading example data from DIPY
3. Running different pipeline types
4. Comparing results
5. Optimizing parameters
6. Quality control and pipeline comparison
"""

from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from dipy.core.gradients import gradient_table
from dipy.data import get_fnames
from dipy.io.gradients import read_bvals_bvecs
from dipy.io.image import load_nifti
from dipy.segment.mask import median_otsu
from dipy.tracking.metrics import mean_curvature, mean_orientation_dispersion
from dipy.tracking.utils import length
from scipy import stats

from meierlab.postproc.quality.pipeline_config import load_config
from meierlab.postproc.quality.pipeline_examples import PipelineExamples
from meierlab.postproc.quality.tractography_viz import TractographyVisualizer


def load_example_data():
    """Load example data from DIPY's built-in datasets."""
    print("Loading example data from DIPY...")

    # Get the example data filenames
    hardi_fname, hardi_bval_fname, hardi_bvec_fname = get_fnames("stanford_hardi")

    # Load the data
    data, affine = load_nifti(hardi_fname)
    bvals, bvecs = read_bvals_bvecs(hardi_bval_fname, hardi_bvec_fname)
    gtab = gradient_table(bvals, bvecs)

    # Create a brain mask
    maskdata, mask = median_otsu(
        data, vol_idx=range(10, 50), median_radius=3, numpass=1
    )

    # Create synthetic tissue maps for CMC example
    # In a real scenario, these would come from segmentation
    wm_mask = mask.astype(float)
    gm_mask = np.zeros_like(mask, dtype=float)
    csf_mask = np.zeros_like(mask, dtype=float)

    # Add some noise to make it more realistic
    np.random.seed(42)
    wm_mask += np.random.normal(0, 0.1, wm_mask.shape)
    gm_mask += np.random.normal(0, 0.1, gm_mask.shape)
    csf_mask += np.random.normal(0, 0.1, csf_mask.shape)

    # Normalize
    wm_mask = np.clip(wm_mask, 0, 1)
    gm_mask = np.clip(gm_mask, 0, 1)
    csf_mask = np.clip(csf_mask, 0, 1)

    tissue_maps = {"wm": wm_mask, "gm": gm_mask, "csf": csf_mask}

    return {
        "data": data,
        "mask": mask,
        "affine": affine,
        "gtab": gtab,
        "tissue_maps": tissue_maps,
    }


def compute_streamline_metrics(streamlines, affine):
    """Compute various metrics for a set of streamlines."""
    metrics = {}

    # Basic metrics
    metrics["n_streamlines"] = len(streamlines)
    metrics["mean_length"] = np.mean([length(s) for s in streamlines])
    metrics["std_length"] = np.std([length(s) for s in streamlines])
    metrics["min_length"] = np.min([length(s) for s in streamlines])
    metrics["max_length"] = np.max([length(s) for s in streamlines])

    # Curvature metrics
    metrics["mean_curvature"] = np.mean([mean_curvature(s) for s in streamlines])
    metrics["std_curvature"] = np.std([mean_curvature(s) for s in streamlines])

    # Orientation dispersion
    metrics["mean_orientation_dispersion"] = mean_orientation_dispersion(streamlines)

    return metrics


def perform_statistical_tests(metrics_dict):
    """Perform statistical tests to compare pipeline metrics."""
    print("\nPerforming statistical tests...")

    # Convert metrics to DataFrame for easier analysis
    df = pd.DataFrame(metrics_dict).T

    # Initialize results dictionary
    test_results = {}

    # Perform ANOVA for each metric
    for metric in df.columns:
        if metric != "n_streamlines":  # Skip count metrics
            # Check normality using Shapiro-Wilk test
            normality_tests = {}
            for pipeline in df.index:
                _, p_value = stats.shapiro(df.loc[pipeline, metric])
                normality_tests[pipeline] = p_value

            # If all distributions are normal, use ANOVA
            if all(p > 0.05 for p in normality_tests.values()):
                _, p_value = stats.f_oneway(*[df.loc[p, metric] for p in df.index])
                test_type = "ANOVA"
            else:
                # Use Kruskal-Wallis test for non-normal distributions
                _, p_value = stats.kruskal(*[df.loc[p, metric] for p in df.index])
                test_type = "Kruskal-Wallis"

            # If significant, perform post-hoc tests
            if p_value < 0.05:
                post_hoc = {}
                for i, p1 in enumerate(df.index):
                    for p2 in df.index[i + 1 :]:
                        if test_type == "ANOVA":
                            _, p = stats.ttest_ind(
                                df.loc[p1, metric], df.loc[p2, metric]
                            )
                        else:
                            _, p = stats.mannwhitneyu(
                                df.loc[p1, metric], df.loc[p2, metric]
                            )
                        post_hoc[f"{p1}_vs_{p2}"] = p
            else:
                post_hoc = None

            test_results[metric] = {
                "test_type": test_type,
                "p_value": p_value,
                "significant": p_value < 0.05,
                "post_hoc": post_hoc,
            }

    return test_results


def compare_pipelines(pipelines, streamlines_dict, affine, output_dir):
    """Compare different pipeline results and generate comparison report."""
    print("\nComparing pipeline results...")

    # Initialize visualizer
    visualizer = TractographyVisualizer(output_dir=output_dir)

    # Compute metrics for each pipeline
    metrics_dict = {}
    for name, streamlines in streamlines_dict.items():
        print(f"Computing metrics for {name} pipeline...")
        metrics_dict[name] = compute_streamline_metrics(streamlines, affine)

    # Perform statistical tests
    test_results = perform_statistical_tests(metrics_dict)

    # Generate comparison visualizations
    print("Generating comparison visualizations...")
    visualizer.generate_comparison_report(
        subject_streamlines=streamlines_dict,
        subject_metrics=metrics_dict,
        atlas_streamlines=None,  # No atlas for this example
        atlas_metrics=None,
        output_file="pipeline_comparison.html",
    )

    # Save metrics and test results to files
    metrics_file = output_dir / "pipeline_metrics.yaml"
    with open(metrics_file, "w") as f:
        yaml.dump(metrics_dict, f)

    test_results_file = output_dir / "statistical_tests.yaml"
    with open(test_results_file, "w") as f:
        yaml.dump(test_results, f)

    # Print significant differences
    print("\nSignificant differences between pipelines:")
    for metric, results in test_results.items():
        if results["significant"]:
            print(f"\n{metric}:")
            print(f"Test type: {results['test_type']}")
            print(f"Overall p-value: {results['p_value']:.3f}")
            if results["post_hoc"]:
                print("Post-hoc comparisons:")
                for comparison, p in results["post_hoc"].items():
                    if p < 0.05:
                        print(f"  {comparison}: p = {p:.3f}")

    print(f"\nComparison results saved to {output_dir}")
    return metrics_dict, test_results


def run_tractography_example():
    """Run tractography pipeline examples with DIPY example data."""

    # Get the example directory
    example_dir = Path(__file__).parent
    config_dir = example_dir / "configs"
    output_dir = example_dir / "output"
    output_dir.mkdir(exist_ok=True)

    # Load example configurations
    det_config = load_config(config_dir / "deterministic_example.yaml")
    prob_config = load_config(config_dir / "probabilistic_example.yaml")
    cmc_config = load_config(config_dir / "cmc_example.yaml")

    # Initialize pipeline examples
    det_pipeline = PipelineExamples(det_config)
    prob_pipeline = PipelineExamples(prob_config)
    cmc_pipeline = PipelineExamples(cmc_config)

    try:
        # Load example data
        data_dict = load_example_data()

        # Run deterministic pipeline
        print("\nRunning deterministic pipeline...")
        det_streamlines = det_pipeline.run_deterministic_pipeline(
            dwi_data=data_dict["data"],
            mask=data_dict["mask"],
            affine=data_dict["affine"],
            output_prefix="example_det",
        )

        # Run probabilistic pipeline
        print("\nRunning probabilistic pipeline...")
        prob_streamlines = prob_pipeline.run_probabilistic_pipeline(
            dwi_data=data_dict["data"],
            mask=data_dict["mask"],
            affine=data_dict["affine"],
            output_prefix="example_prob",
        )

        # Run CMC pipeline
        print("\nRunning CMC pipeline...")
        cmc_streamlines = cmc_pipeline.run_cmc_pipeline(
            dwi_data=data_dict["data"],
            mask=data_dict["mask"],
            affine=data_dict["affine"],
            output_prefix="example_cmc",
            tissue_maps=data_dict["tissue_maps"],
        )

        # Run optimization if configured
        if det_config.optimize_params:
            print("\nOptimizing deterministic pipeline...")
            det_best_params = det_pipeline.optimize_pipeline(
                dwi_data=data_dict["data"],
                mask=data_dict["mask"],
                affine=data_dict["affine"],
                pipeline_type="deterministic",
            )
            print("Best parameters for deterministic pipeline:")
            print(det_best_params)

        if prob_config.optimize_params:
            print("\nOptimizing probabilistic pipeline...")
            prob_best_params = prob_pipeline.optimize_pipeline(
                dwi_data=data_dict["data"],
                mask=data_dict["mask"],
                affine=data_dict["affine"],
                pipeline_type="probabilistic",
            )
            print("Best parameters for probabilistic pipeline:")
            print(prob_best_params)

        if cmc_config.optimize_params:
            print("\nOptimizing CMC pipeline...")
            cmc_best_params = cmc_pipeline.optimize_pipeline(
                dwi_data=data_dict["data"],
                mask=data_dict["mask"],
                affine=data_dict["affine"],
                pipeline_type="cmc",
            )
            print("Best parameters for CMC pipeline:")
            print(cmc_best_params)

        # Compare pipeline results
        streamlines_dict = {
            "deterministic": det_streamlines,
            "probabilistic": prob_streamlines,
            "cmc": cmc_streamlines,
        }

        metrics_dict, test_results = compare_pipelines(
            pipelines=[det_pipeline, prob_pipeline, cmc_pipeline],
            streamlines_dict=streamlines_dict,
            affine=data_dict["affine"],
            output_dir=output_dir,
        )

        print("\nPipeline examples completed successfully!")
        print("Results saved in:")
        print(f"- Deterministic: {det_config.output_dir}")
        print(f"- Probabilistic: {prob_config.output_dir}")
        print(f"- CMC: {cmc_config.output_dir}")
        print(f"- Comparison: {output_dir}")

    except Exception as e:
        print(f"Error running pipeline examples: {e!s}")


if __name__ == "__main__":
    run_tractography_example()
