# Tractography Pipeline Examples

This directory contains examples demonstrating different tractography pipeline configurations
and their usage with the MeierLab processing tools. The examples use DIPY's built-in datasets
to demonstrate the functionality without requiring external data.

## Dependencies

The example scripts depend on the following modules from `src/meierlab/postproc/quality/`:

1. `pipeline_examples.py`: Core pipeline implementation
   - `PipelineExamples` class for running different tractography methods
   - Optimization and parameter tuning functionality
   - Parallel processing support

2. `pipeline_config.py`: Configuration management
   - `PipelineConfig` class for pipeline parameters
   - Configuration loading and saving
   - Parameter range definitions for optimization

3. `tractography_viz.py`: Visualization capabilities
   - `TractographyVisualizer` class for generating visualizations
   - HTML report generation
   - Interactive comparison plots

4. `tractography_metrics.py`: Quality metrics and analysis
   - `TractographyMetrics` class for computing metrics
   - Statistical analysis tools
   - Quality control visualizations

These modules provide the core functionality that the examples build upon to demonstrate:
- Different tractography methods
- Parameter optimization
- Quality control
- Statistical comparison
- Visualization and reporting

## Configuration Files

The `configs` directory contains example configuration files for different tractography methods:

1. `deterministic_example.yaml`: Configuration for deterministic tractography using DTI
   - Step size, max crossings, and seed count parameters
   - DTI-specific parameters like FA threshold
   - Optimization settings and parameter ranges

2. `probabilistic_example.yaml`: Configuration for probabilistic tractography using CSD
   - CSD-specific parameters like relative peak threshold
   - Minimum separation angle and GFA threshold
   - Optimization settings for probabilistic tracking

3. `cmc_example.yaml`: Configuration for CMC tractography using tissue probability maps
   - Tissue probability thresholds for WM, GM, and CSF
   - CMC-specific optimization parameters
   - Quality score computation settings

Each configuration file includes:
- Output settings
- Tracking parameters
- Model-specific parameters
- Optimization settings
- Parameter ranges for optimization

## Example Script

The `tractography_example.py` script demonstrates how to:
1. Load example data from DIPY's built-in datasets
2. Load different pipeline configurations
3. Run various tractography methods
4. Compare results
5. Optimize parameters
6. Perform quality control and statistical comparisons

### Data Source

The example uses DIPY's Stanford HARDI dataset, which includes:
- DWI data
- b-values and b-vectors
- Brain mask (automatically generated)
- Synthetic tissue probability maps (for CMC example)

### Quality Control and Comparison

The example includes comprehensive quality control and pipeline comparison features:

#### Metrics Computation
- Basic metrics:
  - Number of streamlines
  - Mean, standard deviation, minimum, and maximum streamline lengths
- Advanced metrics:
  - Mean curvature and standard deviation
  - Orientation dispersion
  - Density maps

#### Statistical Analysis
- Normality testing using Shapiro-Wilk test
- ANOVA or Kruskal-Wallis test for comparing pipeline metrics
- Post-hoc tests (t-test or Mann-Whitney U test) for significant differences
- Results saved in YAML format for further analysis

#### Visualization
- Interactive HTML reports comparing pipeline results
- Streamline density maps
- Metric distributions and comparisons
- Statistical test results visualization

### Usage

```python
from meierlab.examples.03_postproc.tractography_example import run_tractography_example

# Run the example
run_tractography_example()
```

### Output

The script will:
1. Download and load the example data from DIPY
2. Create output directories for each pipeline type
3. Generate tractography results
4. Create visualizations and comparison reports
5. Save optimization results if enabled
6. Generate statistical comparison results

Output files include:
- `pipeline_metrics.yaml`: Detailed metrics for each pipeline
- `statistical_tests.yaml`: Results of statistical comparisons
- `pipeline_comparison.html`: Interactive visualization report
- Individual pipeline results and visualizations

## Customization

To customize the examples:
1. Modify the configuration files in `configs/`
2. Replace the example data loading with your own data
3. Adjust optimization parameters as needed
4. Add custom metrics to the comparison pipeline

## Documentation

For more information, see:
- [Tractography Pipeline Documentation](../docs/tractography.md)
- [Configuration Guide](../docs/configuration.md)
- [Optimization Guide](../docs/optimization.md)
- [Quality Control Guide](../docs/quality_control.md)
- [DIPY Documentation](https://dipy.org/documentation/)
