# ExploreASL Quality Checker

A robust and configurable quality assessment tool for ExploreASL processed data. This tool provides comprehensive quality evaluation of structural MRI, ASL, and regional CBF metrics with customizable thresholds and visualization options.

## Features

- **Comprehensive Quality Assessment**: Evaluates structural, ASL, registration, and volume metrics
- **Configurable Thresholds**: Customizable quality thresholds for all metrics
- **Multiple Output Formats**: HTML reports and interactive Dash dashboards
- **Robust Error Handling**: Graceful handling of missing data and file errors
- **Flexible Configuration**: Support for YAML configuration files
- **Regional Analysis**: Automatic detection and analysis of regional CBF data
- **PDF Collection**: Automatic copying of individual subject PDF files to QA directory
- **Logging**: Detailed logging for debugging and monitoring
- **Statistical Outlier Detection**: Identifies outliers using ±4 standard deviation rule
- **Missing Subjects Detection**: Compares expected vs actual subjects to identify missing data

## Installation

The quality checker is part of the meierlab package. Ensure you have the required dependencies:

```bash
pip install pandas plotly dash pyyaml
```

## Quick Start

### Basic Usage

```python
from meierlab.quality.exploreasl import ExploreASLQualityChecker

# Create quality checker
qc = ExploreASLQualityChecker("/path/to/exploreasl/output")

# Load data and generate report
qc.load_data()
qc.assess_quality()
report_path = qc.generate_quality_report()
print(f"Report saved to: {report_path}")
```

### Using the Workflow Function

```python
from meierlab.quality.exploreasl import exploreasl_quality_wf

# Generate quality report and Dash app
report_path, dash_app = exploreasl_quality_wf(
    xasl_dir="/path/to/exploreasl/output",
    output_dir="quality_reports",
    create_dash=True
)

# Run the Dash app
dash_app.run(debug=True)
```

## Configuration

### Default Configuration

The quality checker comes with sensible defaults for common metrics:

- **Structural Metrics**: SNR, CNR, FBER, EFC, asymmetry indices
- **ASL Metrics**: CBF values, GM/WM ratios, error metrics
- **Registration Metrics**: Registration quality (e.g., TC_ASL2T1w_Perc)
- **Volume Metrics**: Tissue volumes and ratios

### Custom Configuration

You can customize thresholds and settings:

```python
custom_config = {
    "structural_metrics": {
        "T1w_SNR_GM_Ratio": {"min": 10, "max": None},
        "T1w_CNR_GM_WM_Ratio": {"min": 1.5, "max": None},
    },
    "asl_metrics": {
        "CBF_GM_Median_mL100gmin": {"min": 35, "max": 75},
    },
    "plot_settings": {
        "box_points": "outliers",
        "histogram_bins": 30,
    }
}

qc = QualityChecker("/path/to/exploreasl/output", config=custom_config)
```

### Configuration Files

You can use YAML configuration files:

```python
from meierlab.quality.exploreasl import load_config

config = load_config("exploreasl_config.yaml")
qc = QualityChecker("/path/to/exploreasl/output", config=config)
```

Example configuration file (`exploreasl_config.yaml`):

```yaml
structural_metrics:
  T1w_SNR_GM_Ratio:
    min: 8
    max: null
    description: "Signal-to-noise ratio for gray matter"
  T1w_CNR_GM_WM_Ratio:
    min: 1.2
    max: null
    description: "Contrast-to-noise ratio between GM and WM"
  registration_metrics:
    TC_ASL2T1w_Perc:
      min: 0.8
      max: 1.2
      description: "Tissue contrast (ASL to T1w registration percent)"

asl_metrics:
  CBF_GM_Median_mL100gmin:
    min: 30
    max: 80
    description: "Median CBF in gray matter"

plot_settings:
  box_points: "all"
  histogram_bins: 20

log_level: "INFO"
```

## Directory Structure

The quality checker expects the following ExploreASL directory structure:

```
exploreasl_output/
├── participants.tsv
├── Population/
│   └── Stats/
│       └── mean_qCBF_StandardSpace_MNI_Structural_*.tsv
└── sub-*/
    ├── QC_collection_*.json
    └── *.pdf (optional individual subject reports)
```

## Quality Metrics

The quality checker uses statistical outlier detection based on ±4 standard deviations from the mean for all metrics. This approach identifies extreme outliers that are statistically significant.

### Assessment Status

- **passed**: No outliers detected beyond ±4 SD
- **outliers_detected**: Found outliers beyond ±4 SD (with count and percentage)
- **no_data**: No valid data available for assessment
- **missing_subjects**: Analysis of expected vs actual subjects in the dataset

### Statistical Information

For each metric, the quality assessment provides:
- **Mean**: Average value of the metric
- **Standard Deviation**: Measure of data spread
- **Upper Bound**: Mean + 4×SD (outlier threshold)
- **Lower Bound**: Mean - 4×SD (outlier threshold)
- **Outlier Count**: Number of data points beyond ±4 SD
- **Outlier Percentage**: Percentage of data that are outliers
- **Outlier Subjects**: List of subject IDs and their outlier values (when outliers are detected)
- **Missing Subjects Analysis**:
- - **Expected Subjects**: List from participants.tsv
- - **Actual Subjects**: Subjects found in the data
- - **Missing Subjects**: Expected but not found in data
- - **Extra Subjects**: Found in data but not expected
- - **Completion Rate**: Percentage of expected subjects that are present

### Structural Metrics

- **T1w_SNR_GM_Ratio**: Signal-to-noise ratio for gray matter
- **T1w_CNR_GM_WM_Ratio**: Contrast-to-noise ratio between GM and WM
- **T1w_FBER_WMref_Ratio**: Foreground-background energy ratio
- **T1w_EFC_bits**: Entropy focus criterion
- **T1w_Mean_AI_Perc**: Mean asymmetry index percentage
- **T1w_SD_AI_Perc**: Standard deviation of asymmetry index
- **T1w_IQR_Perc**: Interquartile range percentage

### ASL Metrics

- **CBF_GM_Median_mL100gmin**: Median CBF in gray matter
- **CBF_GM_PVC2_mL100gmin**: Partial volume corrected CBF in gray matter
- **CBF_WM_PVC2_mL100gmin**: Partial volume corrected CBF in white matter
- **CBF_GM_WM_Ratio**: Ratio of GM to WM CBF
- **RMSE_Perc**: Root mean square error percentage
- **nRMSE_Perc**: Normalized RMSE percentage
- **Mean_SSIM_Perc**: Mean structural similarity index
- **PeakSNR_Ratio**: Peak signal-to-noise ratio
- **AI_Perc**: Asymmetry index percentage

### Registration Metrics

- **TC_ASL2T1w_Perc**: Tissue contrast (ASL to T1w registration percent)

### Volume Metrics

- **Overall_GM_vol**: Total gray matter volume
- **Overall_WM_vol**: Total white matter volume
- **Overall_CSF_vol**: Total CSF volume
- **Overall_GM_ICVRatio**: Gray matter to intracranial volume ratio
- **Overall_GMWM_ICVRatio**: Gray+white matter to intracranial volume ratio

## Output Formats

### HTML Reports

The quality checker generates comprehensive HTML reports with:

- Quality summary with pass/fail status for each metric (including registration)
- Interactive plots for all metrics (including registration)
- Regional CBF analysis (if available)
- Data summary statistics
- **Outlier Details**: Subject IDs and values for any detected outliers
- **Subject PDF Files**: Links to individual subject PDF files copied from ASL directories
- **Outlier Detection**: Standard deviation lines (±4 SD) on all plots to identify statistical outliers
- **Statistical Summary**: Mean, standard deviation, and outlier bounds for each metric
- **Missing Subjects Analysis**: Expected vs actual subject counts and list of missing/extra subjects

### Dash Dashboard

Interactive Dash application with:

- Grid layout of all quality plots (including registration)
- Hover information for data points
- Responsive design
- Real-time quality assessment
- **Statistical Outliers**: Visual indicators for data points beyond ±4 standard deviations

### QA Directory Structure

The quality checker creates the following output structure:

```
exploreasl_qc/
├── exploreasl_quality_report.html
├── subject_pdfs/
│   ├── sub-001_catreport_T1.pdf
│   ├── xASL_Report_sub-001.pdf
│   ├── sub-002_catreport_T1.pdf
│   └── xASL_Report_sub-002.pdf
└── (other QA files)
```

## Advanced Usage

### Custom Plots

```python
# Create specific metric plots
structural_figs = qc.create_metric_plots("structural_metrics")
asl_figs = qc.create_metric_plots("asl_metrics")
regional_figs = qc.create_regional_plots()

# Access individual figures
for fig in structural_figs:
    fig.show()
```

### Quality Assessment

```python
# Perform quality assessment
quality_summary = qc.assess_quality()

# Access results
for group_name, group_results in quality_summary.items():
    if group_name == "data_summary":
        continue
    print(f"\n{group_name}:")
    for metric, result in group_results.items():
        print(f"  {metric}: {result['status']} - {result['message']}")
```

### Data Access

```python
# Access raw data
main_df, stats_df = qc.load_data()

# Main data contains all metrics
print(main_df.columns)

# Stats data contains regional CBF (if available)
if not stats_df.empty:
    print(stats_df['roi'].unique())
```

## Error Handling

The quality checker includes robust error handling:

- **Missing Files**: Graceful handling of missing ExploreASL files
- **Invalid Data**: Automatic detection and reporting of data issues
- **Configuration Errors**: Clear error messages for invalid configurations
- **Logging**: Detailed logging for debugging

## Logging

Configure logging level:

```python
qc = QualityChecker("/path/to/exploreasl/output", log_level="DEBUG")
```

Available levels: `DEBUG`, `INFO`, `WARNING`, `ERROR`

## Command Line Usage

The quality checker can be used from the command line:

```bash
python -m meierlab.quality.exploreasl /path/to/exploreasl/output --output-dir reports
```

Options:
- `--output-dir`: Output directory for reports
- `--stats-file`: Path to specific stats file
- `--no-dash`: Don't create Dash app

## Examples

See `examples/02_preproc/exploreasl_quality_example.py` for comprehensive examples including:

- Basic quality checks
- Custom configurations
- Configuration file usage
- Workflow examples
- Advanced analysis
- Interactive dashboards

## Troubleshooting

### Common Issues

1. **FileNotFoundError**: Ensure ExploreASL directory path is correct
2. **No data found**: Check that participants.tsv and QC JSON files exist
3. **Missing metrics**: Some metrics may not be available in all datasets
4. **Configuration errors**: Validate YAML syntax in configuration files

### Debug Mode

Enable debug logging for detailed information:

```python
qc = ExploreASLQualityChecker("/path/to/exploreasl/output", log_level="DEBUG")
```

## Contributing

To extend the quality checker:

1. Add new metrics to the default configuration
2. Implement custom plot functions
3. Add new quality assessment methods
4. Update documentation

## License

This quality checker is part of the meierlab package and follows the same license terms.
