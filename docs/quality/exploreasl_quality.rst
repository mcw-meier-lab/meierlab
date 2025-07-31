ExploreASL Quality Assessment
============================

The ExploreASL quality checker is a comprehensive tool for assessing the quality of ExploreASL processed data. It evaluates structural MRI, ASL, and regional CBF metrics to identify potential issues in your data.

Quick Start
-----------

Basic quality check with default settings:

.. code-block:: python

    from meierlab.quality.exploreasl import ExploreASLQualityChecker

    # Create quality checker
    qc = ExploreASLQualityChecker("/path/to/exploreasl/output")

    # Generate quality report
    report_path = qc.generate_quality_report()
    print(f"Report saved to: {report_path}")

Using the workflow function (recommended):

.. code-block:: python

    from meierlab.quality.exploreasl import exploreasl_quality_wf

    # Generate report and interactive dashboard
    report_path, dash_app = exploreasl_quality_wf(
        xasl_dir="/path/to/exploreasl/output",
        output_dir="quality_reports"
    )

    # Run interactive dashboard
    dash_app.run(debug=True, port=8050)

What the Quality Checker Does
----------------------------

The quality checker evaluates your ExploreASL data across multiple dimensions:

**Structural Quality**
- Signal-to-noise ratio (SNR) for gray matter
- Contrast-to-noise ratio (CNR) between gray and white matter
- Foreground-background energy ratio
- Entropy focus criterion
- Asymmetry indices

**ASL Quality**
- Cerebral blood flow (CBF) values in gray and white matter
- GM/WM CBF ratios
- Error metrics (RMSE, nRMSE)
- Image quality metrics (SSIM, Peak SNR)

**Registration Quality**
- Tissue contrast between ASL and T1w images

**Regional Analysis**
- CBF values in specific brain regions (if available)

Understanding the Results
------------------------

The quality checker uses statistical outlier detection (±4 standard deviations) to identify potential issues:

**Status Meanings:**
- **passed**: No outliers detected - data looks good
- **outliers_detected**: Found statistical outliers - review these subjects
- **no_data**: No valid data available for this metric
- **missing_subjects**: Some expected subjects are missing from the data

**What to Look For:**
- Subjects with outlier values in multiple metrics
- Systematic issues across the dataset
- Missing subjects that might indicate processing failures

Customizing Quality Thresholds
-----------------------------

You can customize quality thresholds based on your specific requirements:

.. code-block:: python

    custom_config = {
        "structural_metrics": {
            "T1w_SNR_GM_Ratio": {"min": 10, "max": None},  # Stricter SNR requirement
        },
        "asl_metrics": {
            "CBF_GM_Median_mL100gmin": {"min": 35, "max": 75},  # Tighter CBF range
        }
    }

    qc = ExploreASLQualityChecker("/path/to/exploreasl/output", config=custom_config)

Using Configuration Files
------------------------

For complex configurations, use YAML files:

.. code-block:: python

    from meierlab.quality.exploreasl import load_config

    config = load_config("my_quality_config.yaml")
    qc = ExploreASLQualityChecker("/path/to/exploreasl/output", config=config)

Example configuration file (``my_quality_config.yaml``):

.. code-block:: yaml

    structural_metrics:
      T1w_SNR_GM_Ratio:
        min: 8
        max: null
        description: "Signal-to-noise ratio for gray matter"
    
    asl_metrics:
      CBF_GM_Median_mL100gmin:
        min: 30
        max: 80
        description: "Median CBF in gray matter"
    
    plot_settings:
      box_points: "outliers"
      histogram_bins: 30

Output Formats
-------------

**HTML Report**
- Comprehensive quality summary
- Interactive plots for all metrics
- Statistical summaries
- Links to individual subject PDFs

**Interactive Dashboard**
- Grid layout of all quality plots
- Hover information for data points
- Real-time quality assessment

**Subject PDFs**
- Automatically copied from ExploreASL directories
- Individual subject quality reports

Common Use Cases
----------------

**Initial Data Screening**
After running ExploreASL, quickly assess overall data quality:

.. code-block:: python

    qc = ExploreASLQualityChecker("/path/to/exploreasl/output")
    qc.generate_quality_report("initial_screening")

**Detailed Analysis**
For in-depth quality assessment with custom thresholds:

.. code-block:: python

    # Load custom configuration
    config = load_config("strict_quality_config.yaml")
    qc = ExploreASLQualityChecker("/path/to/exploreasl/output", config=config)
    
    # Generate detailed report
    qc.generate_quality_report("detailed_analysis")

**Interactive Exploration**
Use the Dash dashboard for interactive data exploration:

.. code-block:: python

    report_path, dash_app = exploreasl_quality_wf(
        xasl_dir="/path/to/exploreasl/output",
        create_dash=True
    )
    
    # Open in browser
    dash_app.run(debug=True, port=8050)

Troubleshooting
--------------

**Common Issues:**

1. **FileNotFoundError**: Check that the ExploreASL directory path is correct
2. **No data found**: Ensure participants.tsv and QC JSON files exist
3. **Missing metrics**: Some metrics may not be available in all datasets

**Debug Mode:**
Enable detailed logging for troubleshooting:

.. code-block:: python

    qc = ExploreASLQualityChecker("/path/to/exploreasl/output", log_level="DEBUG")

**Expected Directory Structure:**
Make sure your ExploreASL output follows this structure:

.. code-block:: text

    exploreasl_output/
    ├── participants.tsv
    ├── Population/
    │   └── Stats/
    │       └── mean_qCBF_StandardSpace_MNI_Structural_*.tsv
    └── sub-*/
        ├── QC_collection_*.json
        └── *.pdf (optional)

Next Steps
----------

- See :ref:`quality_ref` for detailed API documentation
- Check ``examples/02_preproc/exploreasl_quality_example.py`` for comprehensive examples
- Review the configuration example at ``src/meierlab/quality/config_examples/exploreasl_config.yaml`` 