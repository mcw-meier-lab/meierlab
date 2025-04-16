from dataclasses import dataclass


@dataclass
class PipelineConfig:
    """Configuration for tractography pipeline parameters."""

    # General parameters
    output_dir: str = "pipeline_output"
    num_workers: int = 4

    # Tracking parameters
    step_size: float = 0.5
    max_cross: int = 1
    seeds_count: int = 10000

    # DTI parameters
    dti_fa_threshold: float = 0.2

    # CSD parameters
    csd_sphere: str = "repulsion724"
    csd_relative_peak_threshold: float = 0.5
    csd_min_separation_angle: float = 25.0
    csd_gfa_threshold: float = 0.1

    # CMC parameters
    wm_threshold: float = 0.2
    gm_threshold: float = 0.2
    csf_threshold: float = 0.2

    # Optimization parameters
    optimize_params: bool = False
    param_ranges: dict[str, list[float]] = None
    optimization_metric: str = "similarity_score"
    n_trials: int = 20

    def __post_init__(self):
        """Initialize default parameter ranges for optimization."""
        if self.param_ranges is None:
            self.param_ranges = {
                "step_size": [0.1, 1.0],
                "max_cross": [1, 3],
                "seeds_count": [5000, 20000],
                "dti_fa_threshold": [0.1, 0.3],
                "csd_relative_peak_threshold": [0.3, 0.7],
                "csd_min_separation_angle": [15.0, 35.0],
                "csd_gfa_threshold": [0.05, 0.2],
                "wm_threshold": [0.1, 0.3],
                "gm_threshold": [0.1, 0.3],
                "csf_threshold": [0.1, 0.3],
            }


@dataclass
class OptimizationConfig:
    """Configuration for parameter optimization."""

    # Optimization method
    method: str = "bayesian"  # Options: "bayesian", "grid", "random"

    # Bayesian optimization parameters
    n_initial_points: int = 5
    acq_func: str = "EI"  # Expected Improvement

    # Grid search parameters
    grid_size: int = 5

    # Random search parameters
    n_random_trials: int = 20

    # General optimization parameters
    n_trials: int = 20
    metric: str = "similarity_score"
    direction: str = "maximize"  # Options: "maximize", "minimize"

    # Early stopping
    early_stopping: bool = True
    patience: int = 5
    min_delta: float = 0.01


def load_config(config_file: str) -> PipelineConfig:
    """Load pipeline configuration from a YAML file."""
    import yaml

    with open(config_file) as f:
        config_dict = yaml.safe_load(f)

    return PipelineConfig(**config_dict)


def save_config(config: PipelineConfig, config_file: str):
    """Save pipeline configuration to a YAML file."""
    import yaml

    config_dict = config.__dict__
    with open(config_file, "w") as f:
        yaml.dump(config_dict, f, default_flow_style=False)


def get_default_config() -> PipelineConfig:
    """Get default pipeline configuration."""
    return PipelineConfig()


def get_optimization_config() -> OptimizationConfig:
    """Get default optimization configuration."""
    return OptimizationConfig()
