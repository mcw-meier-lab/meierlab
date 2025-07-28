#!/usr/bin/env python3
"""
ExploreASL Quality Check Example

This script demonstrates how to use the robust ExploreASL quality checker
with different configurations and options.
"""

from meierlab.quality.exploreasl import (
    ExploreASLQualityChecker,
    exploreasl_quality_wf,
    load_config,
)


def basic_quality_check():
    """Basic quality check using default settings."""
    print("=== Basic Quality Check ===")

    # Example path - replace with your actual ExploreASL output directory
    xasl_dir = "/path/to/exploreasl/output"

    try:
        # Create quality checker with default settings
        qc = ExploreASLQualityChecker(xasl_dir)

        # Load data
        main_df, stats_df = qc.load_data()
        print(f"Loaded data for {len(main_df)} participants")

        # Assess quality
        qc.assess_quality()
        print("Quality assessment completed")

        # Generate report
        report_path = qc.generate_quality_report()
        print(f"Quality report saved to: {report_path}")

        return qc

    except FileNotFoundError as e:
        print(f"Error: {e}")
        print(
            "Please update the xasl_dir path to point to your ExploreASL output directory"
        )
        return None


def custom_config_quality_check():
    """Quality check with custom configuration."""
    print("\n=== Custom Configuration Quality Check ===")

    # Custom configuration
    custom_config = {
        "structural_metrics": {
            "T1w_SNR_GM_Ratio": {"min": 10, "max": None},  # Stricter SNR requirement
            "T1w_CNR_GM_WM_Ratio": {"min": 1.5, "max": None},  # Higher CNR requirement
        },
        "asl_metrics": {
            "CBF_GM_Median_mL100gmin": {"min": 35, "max": 75},  # Tighter CBF range
            "CBF_GM_WM_Ratio": {"min": 2.5, "max": 3.5},  # Stricter GM/WM ratio
        },
        "plot_settings": {
            "box_points": "outliers",  # Only show outliers
            "histogram_bins": 30,
        },
        "log_level": "DEBUG",
    }

    xasl_dir = "/path/to/exploreasl/output"

    try:
        qc = ExploreASLQualityChecker(xasl_dir, config=custom_config)
        qc.load_data()
        qc.assess_quality()
        report_path = qc.generate_quality_report("custom_quality_report")
        print(f"Custom quality report saved to: {report_path}")

        return qc

    except FileNotFoundError as e:
        print(f"Error: {e}")
        return None


def config_file_quality_check():
    """Quality check using configuration file."""
    print("\n=== Configuration File Quality Check ===")

    xasl_dir = "/path/to/exploreasl/output"
    config_file = "exploreasl_config.yaml"

    try:
        # Load configuration from file
        config = load_config(config_file)
        print(f"Loaded configuration from {config_file}")

        # Create quality checker with file-based config
        qc = ExploreASLQualityChecker(xasl_dir, config=config)
        qc.load_data()
        qc.assess_quality()
        report_path = qc.generate_quality_report("config_file_report")
        print(f"Config file quality report saved to: {report_path}")

        return qc

    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please ensure the config file exists or update the path")
        return None


def workflow_example():
    """Example using the workflow function."""
    print("\n=== Workflow Example ===")

    xasl_dir = "/path/to/exploreasl/output"
    output_dir = "quality_reports"

    try:
        # Use workflow function for simple quality check
        report_path, dash_app = exploreasl_quality_wf(
            xasl_dir=xasl_dir, output_dir=output_dir, create_dash=True
        )

        print(f"Workflow completed. Report: {report_path}")
        print("Dash app created. Run 'dash_app.run(debug=True)' to start it.")

        return dash_app

    except FileNotFoundError as e:
        print(f"Error: {e}")
        return None


def advanced_analysis():
    """Advanced analysis with custom plots and statistics."""
    print("\n=== Advanced Analysis ===")

    xasl_dir = "/path/to/exploreasl/output"

    try:
        qc = ExploreASLQualityChecker(xasl_dir)
        main_df, stats_df = qc.load_data()

        # Create custom plots
        structural_figs = qc.create_metric_plots("structural_metrics")
        asl_figs = qc.create_metric_plots("asl_metrics")
        regional_figs = qc.create_regional_plots()

        print(f"Created {len(structural_figs)} structural metric plots")
        print(f"Created {len(asl_figs)} ASL metric plots")
        print(f"Created {len(regional_figs)} regional CBF plots")

        # Access quality summary
        quality_summary = qc.assess_quality()

        # Print summary statistics
        print("\nQuality Summary:")
        for group_name, group_results in quality_summary.items():
            if group_name == "data_summary":
                continue
            print(f"\n{group_name.replace('_', ' ').title()}:")
            for metric, result in group_results.items():
                status = result["status"]
                message = result["message"]
                print(f"  {metric}: {status} - {message}")

        return qc

    except FileNotFoundError as e:
        print(f"Error: {e}")
        return None


def create_dash_dashboard():
    """Create and run interactive Dash dashboard."""
    print("\n=== Interactive Dashboard ===")

    xasl_dir = "/path/to/exploreasl/output"

    try:
        qc = ExploreASLQualityChecker(xasl_dir)
        qc.load_data()

        # Create Dash app
        app = qc.create_dash_app()

        print("Dash dashboard created successfully!")
        print(
            "To run the dashboard, use: app.run(debug=True, host='0.0.0.0', port=8050)"
        )

        return app

    except FileNotFoundError as e:
        print(f"Error: {e}")
        return None


def main():
    """Run all examples."""
    print("ExploreASL Quality Check Examples")
    print("=" * 40)

    # Note: These examples require actual ExploreASL data
    # Uncomment the functions you want to run and update the paths

    # Basic example
    # basic_quality_check()

    # Custom configuration example
    # custom_config_quality_check()

    # Configuration file example
    # config_file_quality_check()

    # Workflow example
    # workflow_example()

    # Advanced analysis
    # advanced_analysis()

    # Interactive dashboard
    # create_dash_dashboard()

    print("\nTo run these examples:")
    print("1. Update the xasl_dir paths to point to your ExploreASL output")
    print("2. Uncomment the function calls you want to run")
    print("3. Ensure you have the required dependencies installed")
    print("4. Run the script")


if __name__ == "__main__":
    main()
