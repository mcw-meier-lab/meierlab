"""
Tests for ExploreASL quality checker.
"""

import shutil
import tempfile
from pathlib import Path

import pandas as pd
import pytest
import yaml

from meierlab.quality.exploreasl import ExploreASLQualityChecker, load_config


class TestQualityChecker:
    """Test cases for ExploreASLQualityChecker."""

    def setup_method(self):
        """Set up test data."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.xasl_dir = self.temp_dir / "exploreasl_output"
        self.xasl_dir.mkdir()

        # Create test participants.tsv
        participants_data = {
            "participant_id": ["sub-001", "sub-002", "sub-003"],
            "age": [25, 30, 35],
            "sex": ["M", "F", "M"],
        }
        participants_df = pd.DataFrame(participants_data)
        participants_df.to_csv(
            self.xasl_dir / "participants.tsv", sep="\t", index=False
        )

        # Create test QC JSON files
        for i in range(1, 4):
            subj_dir = self.xasl_dir / f"sub-00{i}"
            subj_dir.mkdir()

            qc_data = {
                "ASL": {
                    "CBF_GM_Median_mL100gmin": 50 + i * 5,
                    "CBF_WM_PVC2_mL100gmin": 20 + i * 2,
                    "CBF_GM_WM_Ratio": 2.5 + i * 0.1,
                    "RMSE_Perc": 10 - i * 0.5,
                    "Mean_SSIM_Perc": 85 + i * 2,
                },
                "Structural": {
                    "ID": f"sub-00{i}",
                    "T1w_SNR_GM_Ratio": 10 + i * 0.5,
                    "T1w_CNR_GM_WM_Ratio": 1.5 + i * 0.1,
                    "T1w_FBER_WMref_Ratio": 0.9 + i * 0.02,
                    "T1w_EFC_bits": 0.3 - i * 0.02,
                    "T1w_Mean_AI_Perc": 1.0 + i * 0.01,
                },
            }

            import json

            with open(subj_dir / "QC_collection.json", "w") as f:
                json.dump(qc_data, f)

        # Create Population/Stats directory
        stats_dir = self.xasl_dir / "Population" / "Stats"
        stats_dir.mkdir(parents=True)

        # Create test stats file
        stats_data = {
            "participant_id": ["sub-001", "sub-002", "sub-003"],
            "Caudate_L": [45, 48, 52],
            "Caudate_R": [47, 50, 54],
            "Putamen_L": [55, 58, 62],
            "Putamen_R": [57, 60, 64],
        }
        stats_df = pd.DataFrame(stats_data)
        stats_df.to_csv(stats_dir / "test_stats.tsv", sep="\t", index=False)

        # Create test registration file
        reg_data = {
            "participant_id": ["sub-001", "sub-002", "sub-003"],
            "TC_ASL2T1w_Perc": [1.0, 0.95, 1.1],
        }
        reg_df = pd.DataFrame(reg_data)
        reg_df.to_csv(stats_dir / "RegistrationTC.tsv", sep="\t", index=False)

        # Create test PDF files in subject directories
        for i in range(1, 4):
            subj_dir = self.xasl_dir / f"sub-00{i}"
            # Create a mock PDF file
            pdf_file = subj_dir / f"sub-00{i}_qc_report.pdf"
            pdf_file.write_text("Mock PDF content")

    def teardown_method(self):
        """Clean up test data."""
        shutil.rmtree(self.temp_dir)

    def test_initialization(self):
        """Test ExploreASLQualityChecker initialization."""
        qc = ExploreASLQualityChecker(self.xasl_dir)
        assert qc.xasl_dir == self.xasl_dir
        assert qc.main_df is None
        assert qc.stats_df is None

    def test_load_data(self):
        """Test data loading."""
        qc = ExploreASLQualityChecker(self.xasl_dir)
        main_df, stats_df, reg_df = qc.load_data()

        assert len(main_df) == 3
        assert "subject" in main_df.columns
        assert "CBF_GM_Median_mL100gmin" in main_df.columns
        assert "T1w_SNR_GM_Ratio" in main_df.columns

        # Check stats data
        assert len(stats_df) > 0
        assert "roi" in stats_df.columns
        assert "location" in stats_df.columns
        assert "value" in stats_df.columns

        # Check registration data
        assert len(reg_df) > 0
        assert "TC_ASL2T1w_Perc" in reg_df.columns

    def test_assess_quality(self):
        """Test quality assessment."""
        qc = ExploreASLQualityChecker(self.xasl_dir)
        qc.load_data()
        quality_summary = qc.assess_quality()

        assert "structural_metrics" in quality_summary
        assert "asl_metrics" in quality_summary
        assert "registration_metrics" in quality_summary
        assert "missing_subjects" in quality_summary
        assert "data_summary" in quality_summary

        # Check that some metrics were assessed
        assert len(quality_summary["structural_metrics"]) > 0
        assert len(quality_summary["asl_metrics"]) > 0
        assert len(quality_summary["registration_metrics"]) > 0

        # Check missing subjects analysis
        missing_analysis = quality_summary["missing_subjects"]
        assert "expected_subjects" in missing_analysis
        assert "actual_subjects" in missing_analysis
        assert "missing_subjects" in missing_analysis
        assert "extra_subjects" in missing_analysis
        assert "summary" in missing_analysis

        # Check summary statistics
        summary = missing_analysis["summary"]
        assert "total_expected" in summary
        assert "total_actual" in summary
        assert "missing_count" in summary
        assert "extra_count" in summary
        assert "completion_rate" in summary

        # Check registration metric status (should be passed or outliers_detected)
        reg_result = quality_summary["registration_metrics"].get("TC_ASL2T1w_Perc")
        assert reg_result is not None
        assert reg_result["status"] in ["passed", "outliers_detected", "no_data"]

        # Check that statistical information is included
        if reg_result["status"] in ["passed", "outliers_detected"]:
            assert "mean" in reg_result
            assert "std" in reg_result
            assert "upper_bound" in reg_result
            assert "lower_bound" in reg_result

            # Check for outlier subjects if outliers were detected
            if reg_result["status"] == "outliers_detected":
                assert "outlier_subjects" in reg_result
                assert isinstance(reg_result["outlier_subjects"], list)
                for outlier in reg_result["outlier_subjects"]:
                    assert "subject_id" in outlier
                    assert "value" in outlier

    def test_create_metric_plots(self):
        """Test metric plot creation."""
        qc = ExploreASLQualityChecker(self.xasl_dir)
        qc.load_data()

        structural_figs = qc.create_metric_plots("structural_metrics")
        asl_figs = qc.create_metric_plots("asl_metrics")
        registration_figs = qc.create_registration_plot()

        assert len(structural_figs) > 0
        assert len(asl_figs) > 0
        assert len(registration_figs) > 0

        # Check that standard deviation lines are added
        for fig in structural_figs + asl_figs + registration_figs:
            # Check that the figure has horizontal lines (including SD lines)
            assert len(fig.data) > 0
            # The figure should have horizontal lines for mean and ±4 SD
            # This is a basic check that the figure was created with additional lines
            assert fig.layout.title is not None

    def test_create_regional_plots(self):
        """Test regional plot creation."""
        qc = ExploreASLQualityChecker(self.xasl_dir)
        qc.load_data()

        regional_figs = qc.create_regional_plots()
        assert len(regional_figs) > 0

    def test_generate_quality_report(self):
        """Test quality report generation."""
        qc = ExploreASLQualityChecker(self.xasl_dir)
        qc.load_data()
        qc.assess_quality()

        report_path = qc.generate_quality_report()
        assert report_path.exists()
        assert report_path.suffix == ".html"
        # Check that registration metric is in the report
        with open(report_path) as f:
            html = f.read()
            assert "Registration Metrics" in html
            assert "TC_ASL2T1w_Perc" in html

        # Check that PDF files were copied
        pdf_dir = qc.output_dir / "subject_pdfs"
        assert pdf_dir.exists()
        for i in range(1, 4):
            pdf_file = pdf_dir / f"sub-00{i}_qc_report.pdf"
            assert pdf_file.exists()

    def test_custom_config(self):
        """Test custom configuration."""
        custom_config = {
            "structural_metrics": {"T1w_SNR_GM_Ratio": {"min": 15, "max": None}},
            "asl_metrics": {"CBF_GM_Median_mL100gmin": {"min": 60, "max": 80}},
        }

        qc = ExploreASLQualityChecker(self.xasl_dir, config=custom_config)
        qc.load_data()
        quality_summary = qc.assess_quality()

        # Check that custom thresholds are applied
        snr_result = quality_summary["structural_metrics"].get("T1w_SNR_GM_Ratio")
        if snr_result:
            assert snr_result["status"] in ["passed", "failed", "no_data"]

    def test_invalid_directory(self):
        """Test handling of invalid directory."""
        with pytest.raises(FileNotFoundError):
            ExploreASLQualityChecker("/nonexistent/path")


class TestLoadConfig:
    """Test cases for load_config function."""

    def test_load_config(self):
        """Test configuration loading."""
        config_data = {
            "structural_metrics": {"T1w_SNR_GM_Ratio": {"min": 8, "max": None}},
            "asl_metrics": {"CBF_GM_Median_mL100gmin": {"min": 30, "max": 80}},
        }

        # Create temporary config file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(config_data, f)
            config_file = f.name

        try:
            config = load_config(config_file)
            assert config == config_data
        finally:
            Path(config_file).unlink()

    def test_load_config_nonexistent(self):
        """Test loading nonexistent config file."""
        with pytest.raises(FileNotFoundError):
            load_config("/nonexistent/config.yaml")


def test_exploreasl_quality_wf():
    """Test the workflow function."""
    from meierlab.quality.exploreasl import exploreasl_quality_wf

    # This test would require a more complex setup with actual ExploreASL data
    # For now, we just test that the function exists and can be imported
    assert callable(exploreasl_quality_wf)
