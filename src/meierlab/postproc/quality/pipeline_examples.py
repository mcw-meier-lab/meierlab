from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import optuna
from dipy.data import get_sphere
from dipy.direction import peaks_from_model
from dipy.io.stateful_tractogram import Space, StatefulTractogram
from dipy.io.streamline import save_tractogram
from dipy.reconst.csdeconv import ConstrainedSphericalDeconvModel
from dipy.reconst.dti import TensorModel
from dipy.tracking.local_tracking import LocalTracking
from dipy.tracking.stopping_criterion import ThresholdStoppingCriterion
from dipy.tracking.utils import random_seeds_from_mask
from optuna.visualization import plot_optimization_history, plot_param_importances

from .pipeline_config import (
    OptimizationConfig,
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

        # Initialize multiprocessing pool
        self.pool = ProcessPoolExecutor(max_workers=self.config.num_workers)

    def __del__(self):
        """Cleanup multiprocessing pool."""
        self.pool.shutdown()

    def run_deterministic_pipeline(
        self, dwi_data, mask, affine, output_prefix, params=None
    ):
        """
        Run deterministic tractography pipeline.

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

        return streamlines

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

        return streamlines

    def _run_pipeline_parallel(
        self, pipeline_func, dwi_data, mask, affine, output_prefix, params
    ):
        """Run pipeline function in parallel."""
        future = self.pool.submit(
            pipeline_func, dwi_data, mask, affine, output_prefix, params
        )
        return future.result()

    def optimize_pipeline(
        self,
        dwi_data,
        mask,
        affine,
        pipeline_type: str,
        opt_config: OptimizationConfig = None,
    ):
        """
        Optimize pipeline parameters using Optuna with parallel processing.

        Args:
            dwi_data: DWI data array
            mask: Binary mask array
            affine: Affine transformation matrix
            pipeline_type: Type of pipeline to optimize ('deterministic' or 'probabilistic')
            opt_config: Optimization configuration (optional)
        """
        if opt_config is None:
            opt_config = OptimizationConfig()

        def objective(trial):
            # Suggest parameters based on config ranges
            params = {}
            for param_name, param_range in self.config.param_ranges.items():
                if param_name.startswith(pipeline_type):
                    params[param_name] = trial.suggest_float(
                        param_name, param_range[0], param_range[1]
                    )

            # Run pipeline with suggested parameters in parallel
            if pipeline_type == "deterministic":
                streamlines = self._run_pipeline_parallel(
                    self.run_deterministic_pipeline,
                    dwi_data,
                    mask,
                    affine,
                    "optimization",
                    params,
                )
            else:
                streamlines = self._run_pipeline_parallel(
                    self.run_probabilistic_pipeline,
                    dwi_data,
                    mask,
                    affine,
                    "optimization",
                    params,
                )

            # Compute metrics
            metrics = self._compute_metrics(streamlines)

            # Return the metric to optimize
            return metrics.get(opt_config.metric, 0)

        # Create study with parallel processing support
        study = optuna.create_study(
            direction=opt_config.direction,
            pruner=optuna.pruners.MedianPruner() if opt_config.early_stopping else None,
            storage=f"sqlite:///{self.output_dir}/optimization.db",
            study_name=f"{pipeline_type}_optimization",
        )

        # Run optimization with parallel processing
        study.optimize(
            objective,
            n_trials=opt_config.n_trials,
            n_jobs=self.config.num_workers,
            show_progress_bar=True,
        )

        # Save optimization results
        self._save_optimization_results(study, pipeline_type)

        return study.best_params

    def _save_optimization_results(self, study, pipeline_type):
        """Save optimization results and visualizations."""
        # Save best parameters
        best_params = study.best_params
        with open(self.output_dir / f"{pipeline_type}_best_params.yaml", "w") as f:
            import yaml

            yaml.dump(best_params, f)

        # Save optimization history plot
        fig = plot_optimization_history(study)
        fig.write_image(
            str(self.output_dir / f"{pipeline_type}_optimization_history.png")
        )

        # Save parameter importance plot
        fig = plot_param_importances(study)
        fig.write_image(str(self.output_dir / f"{pipeline_type}_param_importance.png"))

        # Save study statistics
        stats = {
            "best_value": study.best_value,
            "best_params": study.best_params,
            "n_trials": len(study.trials),
            "best_trial": study.best_trial.number,
        }
        with open(
            self.output_dir / f"{pipeline_type}_optimization_stats.yaml", "w"
        ) as f:
            yaml.dump(stats, f)


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
                    dwi_data=dwi_data,
                    mask=mask,
                    affine=affine,
                    pipeline_type=pipeline_type,
                )
                print(f"Best parameters for {pipeline_type} pipeline:")
                print(best_params)

        print("\nPipeline comparison completed successfully!")
        print(f"Results saved in: {examples.output_dir}")

    except Exception as e:
        print(f"Error running example pipelines: {e!s}")


if __name__ == "__main__":
    run_example_pipelines()
