import gc
import logging
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import memory_profiler
import numpy as np
import optuna
import psutil
import yaml
from dipy.data import get_sphere
from dipy.direction import peaks_from_model
from dipy.io.stateful_tractogram import Space, StatefulTractogram
from dipy.io.streamline import save_tractogram
from dipy.reconst.csdeconv import ConstrainedSphericalDeconvModel
from dipy.reconst.dti import TensorModel
from dipy.tracking.local_tracking import LocalTracking
from dipy.tracking.stopping_criterion import ThresholdStoppingCriterion
from dipy.tracking.utils import random_seeds_from_mask

from .pipeline_config import (
    PipelineConfig,
    load_config,
)
from .tractography_viz import TractographyVisualizer


class PipelineExamples:
    """Class containing example tractography pipeline configurations."""

    def __init__(self, config: PipelineConfig = None):
        """
        Initialize the pipeline examples.

        Args:
            config: Pipeline configuration (optional)
        """
        self.config = config if config is not None else PipelineConfig()
        self.output_dir = Path(self.config.output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.visualizer = TractographyVisualizer(output_dir=str(self.output_dir))
        self.logger = logging.getLogger(__name__)
        self._memory_profiler = memory_profiler.LineProfiler()
        self._current_workers = None

        # Initialize multiprocessing pool
        self.pool = ProcessPoolExecutor(max_workers=self.config.num_workers)

    def __del__(self):
        """Cleanup multiprocessing pool."""
        self.pool.shutdown()

    def _get_optimal_workers(self) -> int:
        """Calculate optimal number of workers based on system resources."""
        cpu_count = psutil.cpu_count(logical=False)
        available_memory = psutil.virtual_memory().available
        memory_per_worker = 2 * 1024 * 1024 * 1024  # 2GB per worker

        # Calculate based on CPU and memory constraints
        cpu_workers = max(1, cpu_count - 1)
        memory_workers = max(1, available_memory // memory_per_worker)

        return min(cpu_workers, memory_workers)

    def _adjust_workers(self, required_memory: int) -> int:
        """Adjust number of workers based on required memory."""
        available_memory = psutil.virtual_memory().available
        memory_per_worker = required_memory

        # Calculate maximum workers based on memory
        max_workers = max(1, available_memory // memory_per_worker)

        # Use minimum of configured workers and memory-based workers
        optimal_workers = min(self.config.num_workers, max_workers)

        if optimal_workers != self._current_workers:
            self.logger.info(
                f"Adjusting workers from {self._current_workers} to {optimal_workers}"
            )
            self._current_workers = optimal_workers

        return optimal_workers

    @memory_profiler.profile
    def run_deterministic_pipeline(
        self, dwi_data, mask, affine, output_prefix, params=None
    ):
        """
        Run deterministic tractography pipeline with memory profiling.

        Args:
            dwi_data: DWI data array
            mask: Binary mask array
            affine: Affine transformation matrix
            output_prefix: Prefix for output files
            params: Optional dictionary of parameters to override config
        """
        print("Running deterministic pipeline...")

        # Use provided parameters or config defaults
        step_size = (
            params.get("step_size", self.config.step_size)
            if params
            else self.config.step_size
        )
        max_cross = (
            params.get("max_cross", self.config.max_cross)
            if params
            else self.config.max_cross
        )
        seeds_count = (
            params.get("seeds_count", self.config.seeds_count)
            if params
            else self.config.seeds_count
        )
        fa_threshold = (
            params.get("dti_fa_threshold", self.config.dti_fa_threshold)
            if params
            else self.config.dti_fa_threshold
        )

        # Fit DTI model
        dti_model = TensorModel(dwi_data.gtab)
        dti_fit = dti_model.fit(dwi_data.data, mask)

        # Get FA and eigenvectors
        fa = dti_fit.fa
        peaks = dti_fit.peaks

        # Create stopping criterion
        stopping_criterion = ThresholdStoppingCriterion(fa, fa_threshold)

        # Generate seeds
        seeds = random_seeds_from_mask(mask, seeds_count=seeds_count)

        # Run tracking
        streamlines = LocalTracking(
            peaks,
            stopping_criterion,
            seeds,
            affine,
            step_size=step_size,
            max_cross=max_cross,
        )

        # Save results
        sft = StatefulTractogram(streamlines, dwi_data, Space.RASMM)
        save_tractogram(
            sft, str(self.output_dir / f"{output_prefix}_deterministic.trk")
        )

        # Estimate required memory
        required_memory = dwi_data.nbytes * 3  # Rough estimate: 3x data size
        self._adjust_workers(required_memory)

        try:
            # Run pipeline with memory profiling
            streamlines = self._run_deterministic_tracking(
                dwi_data, mask, affine, params
            )

            # Clean up memory
            gc.collect()

            return streamlines
        except MemoryError:
            self.logger.error("Insufficient memory for deterministic pipeline")
            raise

    def run_probabilistic_pipeline(
        self, dwi_data, mask, affine, output_prefix, params=None
    ):
        """
        Run probabilistic tractography pipeline.

        Args:
            dwi_data: DWI data array
            mask: Binary mask array
            affine: Affine transformation matrix
            output_prefix: Prefix for output files
            params: Optional dictionary of parameters to override config
        """
        print("Running probabilistic pipeline...")

        # Use provided parameters or config defaults
        step_size = (
            params.get("step_size", self.config.step_size)
            if params
            else self.config.step_size
        )
        max_cross = (
            params.get("max_cross", self.config.max_cross)
            if params
            else self.config.max_cross
        )
        seeds_count = (
            params.get("seeds_count", self.config.seeds_count)
            if params
            else self.config.seeds_count
        )
        relative_peak_threshold = (
            params.get(
                "csd_relative_peak_threshold", self.config.csd_relative_peak_threshold
            )
            if params
            else self.config.csd_relative_peak_threshold
        )
        min_separation_angle = (
            params.get("csd_min_separation_angle", self.config.csd_min_separation_angle)
            if params
            else self.config.csd_min_separation_angle
        )
        gfa_threshold = (
            params.get("csd_gfa_threshold", self.config.csd_gfa_threshold)
            if params
            else self.config.csd_gfa_threshold
        )

        # Fit CSD model
        sphere = get_sphere(self.config.csd_sphere)
        csd_model = ConstrainedSphericalDeconvModel(dwi_data.gtab, None)
        csd_model.fit(dwi_data.data, mask)

        # Get peaks
        peaks = peaks_from_model(
            csd_model,
            dwi_data.data,
            sphere,
            relative_peak_threshold=relative_peak_threshold,
            min_separation_angle=min_separation_angle,
            mask=mask,
        )

        # Create stopping criterion
        stopping_criterion = ThresholdStoppingCriterion(peaks.gfa, gfa_threshold)

        # Generate seeds
        seeds = random_seeds_from_mask(mask, seeds_count=seeds_count)

        # Run tracking
        streamlines = LocalTracking(
            peaks,
            stopping_criterion,
            seeds,
            affine,
            step_size=step_size,
            max_cross=max_cross,
            return_all=True,
        )

        # Save results
        sft = StatefulTractogram(streamlines, dwi_data, Space.RASMM)
        save_tractogram(
            sft, str(self.output_dir / f"{output_prefix}_probabilistic.trk")
        )

        # Estimate required memory
        required_memory = dwi_data.nbytes * 4  # Rough estimate: 4x data size
        self._adjust_workers(required_memory)

        try:
            # Run pipeline with memory profiling
            streamlines = self._run_probabilistic_tracking(
                dwi_data, mask, affine, params
            )

            # Clean up memory
            gc.collect()

            return streamlines
        except MemoryError:
            self.logger.error("Insufficient memory for probabilistic pipeline")
            raise

    def _run_pipeline_parallel(
        self, pipeline_func, dwi_data, mask, affine, output_prefix, params
    ):
        """Run pipeline function in parallel."""
        future = self.pool.submit(
            pipeline_func, dwi_data, mask, affine, output_prefix, params
        )
        return future.result()

    def optimize_pipeline(
        self, pipeline_type: str, data: dict, config: PipelineConfig
    ) -> dict:
        """Optimize pipeline parameters using Bayesian optimization."""

        def objective(trial: optuna.Trial) -> float:
            # Suggest parameters based on ranges
            params = {}
            for param, range_dict in config.param_ranges.items():
                if range_dict["type"] == "float":
                    params[param] = trial.suggest_float(
                        param, range_dict["min"], range_dict["max"]
                    )
                elif range_dict["type"] == "int":
                    params[param] = trial.suggest_int(
                        param, range_dict["min"], range_dict["max"]
                    )

            try:
                # Run pipeline with suggested parameters
                if pipeline_type == "deterministic":
                    results = self.run_deterministic_pipeline(data, params)
                elif pipeline_type == "probabilistic":
                    results = self.run_probabilistic_pipeline(data, params)
                elif pipeline_type == "cmc":
                    results = self.run_cmc_pipeline(data, params)
                else:
                    raise ValueError(f"Unknown pipeline type: {pipeline_type}")

                # Calculate objective value
                if config.optimization_metric == "streamline_count":
                    return len(results["streamlines"])
                elif config.optimization_metric == "mean_length":
                    return np.mean([len(s) for s in results["streamlines"]])
                elif config.optimization_metric == "quality_score":
                    valid, _ = self._validate_streamlines(results["streamlines"])
                    if not valid:
                        return float("-inf")
                    return len(results["streamlines"]) * (
                        1
                        - np.mean(
                            self._calculate_streamline_curvatures(
                                results["streamlines"]
                            )
                        )
                    )
                else:
                    raise ValueError(
                        f"Unknown optimization metric: {config.optimization_metric}"
                    )

            except Exception as e:
                self.logger.error(f"Trial failed: {e!s}")
                return float("-inf")

        # Create study with early stopping
        study = optuna.create_study(
            direction="maximize",
            pruner=optuna.pruners.MedianPruner(
                n_startup_trials=5, n_warmup_steps=10, interval_steps=1
            ),
        )

        # Run optimization
        study.optimize(
            objective,
            n_trials=config.n_trials,
            callbacks=[self._early_stopping_callback],
        )

        # Save optimization results
        self._save_optimization_results(study, pipeline_type)

        return {
            "best_params": study.best_params,
            "best_value": study.best_value,
            "best_trial": study.best_trial.number,
        }

    def _early_stopping_callback(
        self, study: optuna.Study, trial: optuna.Trial
    ) -> None:
        """Early stopping callback for optimization."""
        if len(study.trials) < 10:
            return

        # Get last 10 trials
        recent_trials = study.trials[-10:]
        values = [t.value for t in recent_trials if t.value is not None]

        if len(values) < 5:
            return

        # Check if improvement is below threshold
        improvement = (max(values) - min(values)) / abs(min(values))
        if improvement < 0.01:  # 1% improvement threshold
            self.logger.info(
                "Early stopping: No significant improvement in last 10 trials"
            )
            study.stop()

    def _save_optimization_results(
        self, study: optuna.Study, pipeline_type: str
    ) -> None:
        """Save optimization results and visualizations."""
        output_dir = Path(self.config.output_dir) / f"{pipeline_type}_optimization"
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save best parameters
        with open(output_dir / "best_params.yaml", "w") as f:
            yaml.dump(study.best_params, f)

        # Save optimization history
        history = {
            "best_value": study.best_value,
            "best_trial": study.best_trial.number,
            "n_trials": len(study.trials),
            "params_importance": optuna.importance.get_param_importances(study),
        }
        with open(output_dir / "optimization_history.yaml", "w") as f:
            yaml.dump(history, f)

        # Generate and save visualizations
        fig1 = optuna.visualization.plot_optimization_history(study)
        fig1.write_image(str(output_dir / "optimization_history.png"))

        fig2 = optuna.visualization.plot_param_importances(study)
        fig2.write_image(str(output_dir / "param_importances.png"))

        fig3 = optuna.visualization.plot_parallel_coordinate(study)
        fig3.write_image(str(output_dir / "parallel_coordinate.png"))

    def _validate_streamlines(self, streamlines: list) -> tuple[bool, str]:
        """Validate streamlines for quality issues."""
        if not streamlines:
            return False, "No streamlines generated"

        # Check for self-intersections
        self_intersections = self._check_streamline_intersections(streamlines)
        if self_intersections:
            return (
                False,
                f"Found {len(self_intersections)} self-intersecting streamlines",
            )

        # Check for loops
        loops = self._check_streamline_loops(streamlines)
        if loops:
            return False, f"Found {len(loops)} streamlines with loops"

        # Check anatomical plausibility
        valid, message = self._check_anatomical_plausibility(streamlines)
        if not valid:
            return False, message

        return True, "All quality checks passed"

    def _check_streamline_intersections(self, streamlines: list) -> list:
        """Check for self-intersections in streamlines."""
        intersecting = []
        for i, streamline in enumerate(streamlines):
            points = np.array(streamline)
            # Check for duplicate points
            if len(np.unique(points, axis=0)) < len(points):
                intersecting.append(i)
        return intersecting

    def _check_streamline_loops(self, streamlines: list) -> list:
        """Check for loops in streamlines."""
        loops = []
        for i, streamline in enumerate(streamlines):
            points = np.array(streamline)
            # Check if start and end points are close
            if np.linalg.norm(points[0] - points[-1]) < 1.0:
                loops.append(i)
        return loops

    def _check_anatomical_plausibility(self, streamlines: list) -> tuple[bool, str]:
        """Check if streamlines are anatomically plausible."""
        # Get streamline statistics
        lengths = [len(s) for s in streamlines]
        mean_length = np.mean(lengths)
        std_length = np.std(lengths)

        # Check length distribution
        if mean_length < 10 or std_length > mean_length * 2:
            return False, "Unusual streamline length distribution"

        # Check curvature
        curvatures = self._calculate_streamline_curvatures(streamlines)
        if np.mean(curvatures) > 0.5:  # Arbitrary threshold
            return False, "High average curvature detected"

        return True, "Anatomically plausible"

    def _calculate_streamline_curvatures(self, streamlines: list) -> np.ndarray:
        """Calculate curvature for each streamline."""
        curvatures = []
        for streamline in streamlines:
            points = np.array(streamline)
            # Calculate tangent vectors
            tangents = np.diff(points, axis=0)
            # Normalize tangents
            tangents = tangents / np.linalg.norm(tangents, axis=1)[:, np.newaxis]
            # Calculate curvature as change in tangent direction
            curvature = np.arccos(
                np.clip(np.sum(tangents[:-1] * tangents[1:], axis=1), -1.0, 1.0)
            )
            curvatures.append(np.mean(curvature))
        return np.array(curvatures)

    def _compare_pipeline_runs(self, results1: dict, results2: dict) -> dict:
        """Compare results from two pipeline runs."""
        comparison = {
            "streamline_count": {
                "run1": len(results1["streamlines"]),
                "run2": len(results2["streamlines"]),
                "difference": len(results1["streamlines"])
                - len(results2["streamlines"]),
            },
            "mean_length": {
                "run1": np.mean([len(s) for s in results1["streamlines"]]),
                "run2": np.mean([len(s) for s in results2["streamlines"]]),
                "difference": np.mean([len(s) for s in results1["streamlines"]])
                - np.mean([len(s) for s in results2["streamlines"]]),
            },
            "quality_metrics": {
                "run1": self._validate_streamlines(results1["streamlines"]),
                "run2": self._validate_streamlines(results2["streamlines"]),
            },
        }
        return comparison


def run_example_pipelines(config_file: str | None = None):
    """Run example pipelines with different configurations."""
    # Load or create configuration
    if config_file:
        config = load_config(config_file)
    else:
        config = PipelineConfig()

    # Initialize pipeline examples
    examples = PipelineExamples(config)

    # Example: Compare pipelines for a single subject
    try:
        # Load example data (replace with actual data loading)
        dwi_data = None  # Replace with actual DWI data
        mask = None  # Replace with actual mask
        affine = None  # Replace with actual affine
        tissue_maps = None  # Replace with actual tissue maps if available

        # Run pipeline comparison
        examples.compare_pipelines(
            dwi_data=dwi_data,
            mask=mask,
            affine=affine,
            subject_id="example_subject",
            tissue_maps=tissue_maps,
        )

        # Run optimization if configured
        if config.optimize_params:
            print("\nRunning parameter optimization...")
            for pipeline_type in ["deterministic", "probabilistic"]:
                best_params = examples.optimize_pipeline(
                    pipeline_type=pipeline_type, data=dwi_data, config=config
                )
                print(f"Best parameters for {pipeline_type} pipeline:")
                print(best_params)

        print("\nPipeline comparison completed successfully!")
        print(f"Results saved in: {examples.output_dir}")

    except Exception as e:
        print(f"Error running example pipelines: {e!s}")


if __name__ == "__main__":
    run_example_pipelines()
