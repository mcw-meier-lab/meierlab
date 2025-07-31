import json
import logging
import shutil
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import yaml
from dash import Dash, dcc, html


class ExploreASLQualityChecker:
    """
    A robust quality checker for ExploreASL data with configurable metrics and visualization.

    This class provides comprehensive quality assessment for ExploreASL processed data,
    including structural, ASL, and regional CBF metrics with customizable thresholds
    and visualization options.

    Parameters
    ----------
    xasl_dir : str or Path
        Path to the ExploreASL output directory
    config : dict, optional
        Configuration dictionary with quality thresholds and plot settings
    output_dir : str or Path, optional
        Directory to save quality reports and plots
    log_level : str, optional
        Logging level ('DEBUG', 'INFO', 'WARNING', 'ERROR')

    Examples
    --------
    >>> from meierlab.quality.exploreasl import ExploreASLQualityChecker
    >>> qc = ExploreASLQualityChecker("/path/to/exploreasl/output")
    >>> qc.generate_quality_report()
    """

    def __init__(
        self,
        xasl_dir: str | Path,
        config: dict | None = None,
        output_dir: str | Path | None = None,
        log_level: str = "INFO",
    ):
        self.xasl_dir = Path(xasl_dir)
        self.output_dir = Path(output_dir) if output_dir else Path("exploreasl_qc")
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Setup logging
        logging.basicConfig(level=getattr(logging, log_level.upper()))
        self.logger = logging.getLogger(__name__)

        # Default configuration
        self.default_config = {
            "structural_metrics": {
                "T1w_SNR_GM_Ratio": {"min": 8, "max": None},
                "T1w_CNR_GM_WM_Ratio": {"min": 1.2, "max": None},
                "T1w_FBER_WMref_Ratio": {"min": 0.8, "max": None},
                "T1w_EFC_bits": {"min": None, "max": 0.4},
                "T1w_Mean_AI_Perc": {"min": 0.8, "max": 1.2},
                "T1w_SD_AI_Perc": {"min": None, "max": 0.1},
                "T1w_IQR_Perc": {"min": None, "max": 0.1},
            },
            "asl_metrics": {
                "CBF_GM_Median_mL100gmin": {"min": 30, "max": 80},
                "CBF_GM_PVC2_mL100gmin": {"min": 35, "max": 85},
                "CBF_WM_PVC2_mL100gmin": {"min": 15, "max": 35},
                "CBF_GM_WM_Ratio": {"min": 2.0, "max": 4.0},
                "RMSE_Perc": {"min": None, "max": 15},
                "nRMSE_Perc": {"min": None, "max": 10},
                "Mean_SSIM_Perc": {"min": 80, "max": None},
                "PeakSNR_Ratio": {"min": 20, "max": None},
                "AI_Perc": {"min": 0.8, "max": 1.2},
            },
            "registration_metrics": {
                "TC_ASL2T1w_Perc": {"min": 0.8, "max": 1.2},
            },
            "plot_settings": {
                "box_points": "all",
                "violin_points": "all",
                "histogram_bins": 20,
                "correlation_method": "pearson",
            },
            "output_formats": ["html", "png"],
            "dash_app": True,
        }

        # Update with user config
        self.config = self.default_config.copy()
        if config:
            self._update_config(config)

        # Data storage
        self.main_df = None
        self.stats_df = None
        self.quality_summary = {}

        # Validate directory structure
        self._validate_directory()

    def _update_config(self, user_config: dict):
        """Update configuration with user settings."""
        for key, value in user_config.items():
            if key in self.config:
                if isinstance(value, dict) and isinstance(self.config[key], dict):
                    self.config[key].update(value)
                else:
                    self.config[key] = value
            else:
                self.config[key] = value

    def _validate_directory(self):
        """Validate ExploreASL directory structure."""
        required_files = ["participants.tsv", "Population/Stats"]

        missing_files = []
        for file_path in required_files:
            if not (self.xasl_dir / file_path).exists():
                missing_files.append(file_path)

        if missing_files:
            raise FileNotFoundError(
                f"Missing required files in ExploreASL directory: {missing_files}"
            )

        # Check for QC JSON files
        json_files = list(self.xasl_dir.glob("sub-*/QC_collection*json"))
        if not json_files:
            self.logger.warning("No QC JSON files found in sub-* directories")

    def _restructure_stats_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Restructure regional statistics data for easier plotting.

        Parameters
        ----------
        df : pd.DataFrame
            Raw statistics dataframe

        Returns
        -------
        pd.DataFrame
            Restructured data with ROI and location columns
        """
        roi_list = [
            c for c in df.columns if any(suffix in c for suffix in ["_B", "_L", "_R"])
        ]

        if not roi_list:
            self.logger.warning("No regional columns found in stats data")
            return df

        melt = pd.melt(df, id_vars=["participant_id"], value_vars=roi_list)
        melt["location"] = [c.split("_")[1] for c in melt["variable"]]
        melt["roi"] = [c.split("_")[0] for c in melt["variable"]]
        melt["value"] = pd.to_numeric(melt["value"], errors="coerce")

        return melt

    def _read_participants_data(self) -> pd.DataFrame:
        """Read and process participants.tsv file."""
        summary_file = self.xasl_dir / "participants.tsv"

        try:
            summary = pd.read_csv(summary_file, delimiter="\t")
            # Add prefix to avoid column conflicts
            summary.columns = [f"Overall_{col}" for col in summary.columns]
            return summary
        except Exception as e:
            self.logger.error(f"Error reading participants.tsv: {e}")
            return pd.DataFrame()

    def _read_stats_data(self, stats_file: Path) -> pd.DataFrame:
        """Read and process statistics file."""
        try:
            stats = pd.read_csv(stats_file, delimiter="\t")

            # Remove common non-numeric columns
            cols_to_drop = ["session", "LongitudinalTimePoint", "SubjectNList", "Site"]
            stats = stats.drop(
                [col for col in cols_to_drop if col in stats.columns], axis=1
            )

            stats = self._restructure_stats_data(stats)
            return stats
        except Exception as e:
            self.logger.error(f"Error reading stats file {stats_file}: {e}")
            return pd.DataFrame()

    def _read_reg_data(self, reg_file: Path) -> pd.DataFrame:
        """Read and process registration data."""
        try:
            reg = pd.read_csv(reg_file, delimiter="\t")
            # Only drop 'session' column if it exists
            if "session" in reg.columns:
                reg = reg.drop(columns=["session"], axis=1)
            return reg
        except Exception as e:
            self.logger.error(f"Error reading registration file {reg_file}: {e}")
            return pd.DataFrame()

    def _read_qc_json_data(self) -> pd.DataFrame:
        """Read and process QC JSON files."""
        json_files = list(self.xasl_dir.glob("sub-*/QC_collection*json"))
        json_data = pd.DataFrame()

        for jfile in json_files:
            try:
                with open(jfile) as jf:
                    subj_data = json.load(jf)

                    # Extract ASL and Structural data
                    asl_data = pd.DataFrame(subj_data.get("ASL", {}), index=[0])
                    t1w_data = pd.DataFrame(subj_data.get("Structural", {}), index=[0])

                    # Combine data
                    sub_df = pd.concat([asl_data, t1w_data], axis=1)
                    sub_df["subject"] = t1w_data.get("ID", jfile.parent.name)
                    sub_df = sub_df.reset_index(drop=True)
                    json_data = pd.concat([json_data, sub_df], ignore_index=True)

            except Exception as e:
                self.logger.warning(f"Error processing {jfile}: {e}")
                continue

        return json_data

    def load_data(
        self, stats_file: str | Path | None = None
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Load and process all ExploreASL data.

        Parameters
        ----------
        stats_file : str or Path, optional
            Path to specific statistics file. If None, will search for common patterns.

        Returns
        -------
        tuple
            (main_df, stats_df, reg_df) - Combined data, regional statistics, and registration data
        """
        self.logger.info("Loading ExploreASL data...")

        # Read participants data
        summary_df = self._read_participants_data()

        # Read stats data
        if stats_file is None:
            # Try to find stats file automatically
            stats_patterns = [
                "Population/Stats/mean_qCBF_StandardSpace_MNI_Structural_*.tsv",
                "Population/Stats/mean*.tsv",
                "Population/Stats/median*.tsv",
                "Population/Stats/test_stats.tsv",  # Add test file pattern
            ]

            for pattern in stats_patterns:
                stats_files = list(self.xasl_dir.glob(pattern))
                if stats_files:
                    stats_file = stats_files[0]
                    break
            else:
                self.logger.warning("No stats file found, skipping regional analysis")
                stats_df = pd.DataFrame()
        else:
            stats_file = Path(stats_file)

        if stats_file and stats_file.exists():
            stats_df = self._read_stats_data(stats_file)
        else:
            stats_df = pd.DataFrame()

        # Read registration data
        reg_file = self.xasl_dir / "Population/Stats/RegistrationTC.tsv"
        reg_df = self._read_reg_data(reg_file)

        # Read QC JSON data
        json_data = self._read_qc_json_data()

        # Merge all data
        if not summary_df.empty and not json_data.empty:
            main_df = summary_df.merge(
                json_data,
                left_on="Overall_participant_id",
                right_on="subject",
                how="outer",
            )
        elif not json_data.empty:
            main_df = json_data
        elif not summary_df.empty:
            main_df = summary_df
        else:
            raise ValueError("No data found in ExploreASL directory")

        self.main_df = main_df
        self.stats_df = stats_df
        self.reg_df = reg_df
        self.logger.info(f"Loaded data for {len(main_df)} participants")
        return main_df, stats_df, reg_df

    def _check_metric_thresholds(self, df: pd.DataFrame, metric_group: str) -> dict:
        """
        Check metrics for statistical outliers using standard deviation.

        Parameters
        ----------
        df : pd.DataFrame
            Data to check
        metric_group : str
            Group of metrics to check ('structural_metrics', 'asl_metrics', etc.)

        Returns
        -------
        dict
            Quality assessment results based on statistical outliers
        """
        results = {}
        metrics = self.config.get(metric_group, {})

        for metric in metrics.keys():
            if metric not in df.columns:
                continue

            values = pd.to_numeric(df[metric], errors="coerce")
            valid_values = values.dropna()

            if len(valid_values) == 0:
                results[metric] = {"status": "no_data", "message": "No valid data"}
                continue

            # Calculate statistical bounds using ±4 SD
            mean_val = valid_values.mean()
            std_val = valid_values.std()
            upper_bound = mean_val + 4 * std_val
            lower_bound = mean_val - 4 * std_val

            # Find outliers beyond ±4 SD
            outliers = valid_values[
                (valid_values > upper_bound) | (valid_values < lower_bound)
            ]

            if len(outliers) > 0:
                outlier_percentage = (len(outliers) / len(valid_values)) * 100

                # Get subject IDs for outliers
                outlier_subjects = []
                for outlier_val in outliers:
                    # Find the subject(s) with this outlier value
                    outlier_indices = df[df[metric] == outlier_val].index
                    for idx in outlier_indices:
                        # Try to get subject ID from various possible column names
                        subject_id = None
                        for col in [
                            "subject",
                            "participant_id",
                            "Overall_participant_id",
                        ]:
                            if col in df.columns:
                                subject_id = df.loc[idx, col]
                                break
                        if subject_id:
                            outlier_subjects.append(
                                {
                                    "subject_id": str(subject_id),
                                    "value": float(outlier_val),
                                }
                            )

                # Remove duplicates based on subject_id and value combination
                unique_outliers = []
                seen = set()
                for outlier in outlier_subjects:
                    key = (outlier["subject_id"], outlier["value"])
                    if key not in seen:
                        unique_outliers.append(outlier)
                        seen.add(key)

                results[metric] = {
                    "status": "outliers_detected",
                    "message": f"Found {len(outliers)} outliers ({outlier_percentage:.1f}%) beyond ±4 SD",
                    "outlier_count": len(outliers),
                    "outlier_percentage": outlier_percentage,
                    "outlier_subjects": unique_outliers,
                    "mean": mean_val,
                    "std": std_val,
                    "upper_bound": upper_bound,
                    "lower_bound": lower_bound,
                }
            else:
                results[metric] = {
                    "status": "passed",
                    "message": f"No outliers detected (mean: {mean_val:.2f}, ±4 SD: {lower_bound:.2f} to {upper_bound:.2f})",
                    "mean": mean_val,
                    "std": std_val,
                    "upper_bound": upper_bound,
                    "lower_bound": lower_bound,
                }

        return results

    def _check_missing_subjects(self) -> dict:
        """
        Check for missing subjects by comparing expected vs actual subjects.

        Returns
        -------
        dict
            Missing subjects analysis results
        """
        missing_analysis = {
            "expected_subjects": [],
            "actual_subjects": [],
            "missing_subjects": [],
            "extra_subjects": [],
            "summary": {},
        }

        # Get expected subjects from participants.tsv
        participants_file = self.xasl_dir / "participants.tsv"
        if participants_file.exists():
            try:
                participants_df = pd.read_csv(participants_file, delimiter="\t")
                expected_subjects = participants_df["participant_id"].tolist()
                missing_analysis["expected_subjects"] = expected_subjects
            except Exception as e:
                self.logger.warning(f"Error reading participants.tsv: {e}")
                expected_subjects = []
        else:
            self.logger.warning("participants.tsv not found")
            expected_subjects = []

        # Get actual subjects from the data
        if self.main_df is not None:
            # Try different column names for subject IDs
            subject_cols = ["subject", "participant_id", "Overall_participant_id"]
            actual_subjects = []

            for col in subject_cols:
                if col in self.main_df.columns:
                    actual_subjects = self.main_df[col].dropna().unique().tolist()
                    break

            missing_analysis["actual_subjects"] = actual_subjects

            # Find missing and extra subjects
            expected_set = set(expected_subjects)
            actual_set = set(actual_subjects)

            missing_subjects = list(expected_set - actual_set)
            extra_subjects = list(actual_set - expected_set)

            missing_analysis["missing_subjects"] = missing_subjects
            missing_analysis["extra_subjects"] = extra_subjects

            # Create summary
            missing_analysis["summary"] = {
                "total_expected": len(expected_subjects),
                "total_actual": len(actual_subjects),
                "missing_count": len(missing_subjects),
                "extra_count": len(extra_subjects),
                "completion_rate": (len(actual_subjects) / len(expected_subjects) * 100)
                if expected_subjects
                else 0,
            }

        return missing_analysis

    def assess_quality(self) -> dict:
        """
        Perform comprehensive quality assessment.

        Returns
        -------
        dict
            Quality assessment results
        """
        if self.main_df is None:
            raise ValueError("Data not loaded. Call load_data() first.")

        self.quality_summary = {
            "structural_metrics": self._check_metric_thresholds(
                self.main_df, "structural_metrics"
            ),
            "asl_metrics": self._check_metric_thresholds(self.main_df, "asl_metrics"),
            "registration_metrics": self._check_metric_thresholds(
                self.reg_df, "registration_metrics"
            )
            if hasattr(self, "reg_df") and self.reg_df is not None
            else {},
            "missing_subjects": self._check_missing_subjects(),
            "data_summary": {
                "total_participants": len(self.main_df),
                "missing_data": self.main_df.isnull().sum().to_dict(),
                "regional_data": len(self.stats_df) if not self.stats_df.empty else 0,
            },
        }

        return self.quality_summary

    def create_metric_plots(self, metric_group: str) -> list[go.Figure]:
        """
        Create plots for a specific metric group.

        Parameters
        ----------
        metric_group : str
            Group of metrics to plot

        Returns
        -------
        list
            List of plotly figures
        """
        if self.main_df is None:
            raise ValueError("Data not loaded. Call load_data() first.")

        metrics = self.config.get(metric_group, {})
        figures = []

        for metric in metrics.keys():
            if metric not in self.main_df.columns:
                continue

            # Create box plot
            fig = px.box(
                self.main_df,
                y=metric,
                points=self.config["plot_settings"]["box_points"],
                hover_data=["subject"],
                title=f"{metric} Distribution",
            )

            # Add standard deviation lines for outlier detection
            values = pd.to_numeric(self.main_df[metric], errors="coerce").dropna()
            if len(values) > 0:
                mean_val = values.mean()
                std_val = values.std()

                # Add mean line
                fig.add_hline(
                    y=mean_val,
                    line_dash="dot",
                    line_color="blue",
                    annotation_text="Mean",
                )

                # Add ±4 SD lines
                upper_4sd = mean_val + 4 * std_val
                lower_4sd = mean_val - 4 * std_val

                fig.add_hline(
                    y=upper_4sd,
                    line_dash="dashdot",
                    line_color="orange",
                    annotation_text="+4 SD",
                )
                fig.add_hline(
                    y=lower_4sd,
                    line_dash="dashdot",
                    line_color="orange",
                    annotation_text="-4 SD",
                )

            figures.append(fig)

        return figures

    def create_regional_plots(self) -> list[go.Figure]:
        """
        Create regional CBF plots.

        Returns
        -------
        list
            List of plotly figures
        """
        if self.stats_df is None or self.stats_df.empty:
            return []

        figures = []
        unique_rois = self.stats_df["roi"].drop_duplicates()

        for roi in unique_rois:
            df = self.stats_df[self.stats_df["roi"] == roi]
            fig = px.box(
                df,
                x="roi",
                y="value",
                color="location",
                points=self.config["plot_settings"]["box_points"],
                hover_data=["participant_id"],
                title=f"{roi} Regional CBF",
            )

            # Add standard deviation lines for outlier detection
            values = pd.to_numeric(df["value"], errors="coerce").dropna()
            if len(values) > 0:
                mean_val = values.mean()
                std_val = values.std()

                # Add mean line
                fig.add_hline(
                    y=mean_val,
                    line_dash="dot",
                    line_color="blue",
                    annotation_text="Mean",
                )

                # Add ±4 SD lines
                upper_4sd = mean_val + 4 * std_val
                lower_4sd = mean_val - 4 * std_val

                fig.add_hline(
                    y=upper_4sd,
                    line_dash="dashdot",
                    line_color="orange",
                    annotation_text="+4 SD",
                )
                fig.add_hline(
                    y=lower_4sd,
                    line_dash="dashdot",
                    line_color="orange",
                    annotation_text="-4 SD",
                )

            figures.append(fig)

        return figures

    def create_registration_plot(self) -> list:
        """
        Create plot for registration metric(s).
        Returns
        -------
        list
            List of plotly figures (usually one)
        """
        if not hasattr(self, "reg_df") or self.reg_df is None or self.reg_df.empty:
            return []
        metrics = self.config.get("registration_metrics", {})
        figures = []
        for metric in metrics.keys():
            if metric not in self.reg_df.columns:
                continue
            fig = px.box(
                self.reg_df,
                y=metric,
                points=self.config["plot_settings"].get("box_points", "all"),
                title=f"{metric} Distribution (Registration)",
            )

            # Add standard deviation lines for outlier detection
            values = pd.to_numeric(self.reg_df[metric], errors="coerce").dropna()
            if len(values) > 0:
                mean_val = values.mean()
                std_val = values.std()

                # Add mean line
                fig.add_hline(
                    y=mean_val,
                    line_dash="dot",
                    line_color="blue",
                    annotation_text="Mean",
                )

                # Add ±4 SD lines
                upper_4sd = mean_val + 4 * std_val
                lower_4sd = mean_val - 4 * std_val

                fig.add_hline(
                    y=upper_4sd,
                    line_dash="dashdot",
                    line_color="orange",
                    annotation_text="+4 SD",
                )
                fig.add_hline(
                    y=lower_4sd,
                    line_dash="dashdot",
                    line_color="orange",
                    annotation_text="-4 SD",
                )

            figures.append(fig)
        return figures

    def _copy_subject_pdfs(self) -> list[Path]:
        """
        Copy PDF files from individual subject directories to QA directory.

        Returns
        -------
        List[Path]
            List of copied PDF file paths
        """
        copied_files = []
        pdf_dir = self.output_dir / "subject_pdfs"
        pdf_dir.mkdir(exist_ok=True)

        # Find all subject directories
        subject_dirs = [
            d
            for d in self.xasl_dir.iterdir()
            if d.is_dir() and d.name.startswith("sub-")
        ]

        for subject_dir in subject_dirs:
            subject_id = subject_dir.name

            # Look for PDF files in the subject directory and its subdirectories
            pdf_files = list(subject_dir.rglob("*.pdf"))

            if pdf_files:
                for pdf_file in pdf_files:
                    try:
                        # Copy PDF to QA directory with subject name
                        if subject_id in pdf_file.name:
                            dest_file = f"{pdf_dir}/{pdf_file.name}"
                        else:
                            dest_file = f"{pdf_dir}/{subject_id}_{pdf_file.name}"
                        shutil.copy2(pdf_file, dest_file)
                        copied_files.append(dest_file)
                        self.logger.info(f"Copied {pdf_file} to {dest_file}")
                    except Exception as e:
                        self.logger.warning(f"Failed to copy {pdf_file}: {e}")

        if copied_files:
            self.logger.info(f"Copied {len(copied_files)} PDF files to {pdf_dir}")
        else:
            self.logger.info("No PDF files found in subject directories")

        return copied_files

    def generate_quality_report(
        self, output_name: str = "exploreasl_quality_report"
    ) -> Path:
        """
        Generate comprehensive quality report.

        Parameters
        ----------
        output_name : str
            Base name for output files

        Returns
        -------
        Path
            Path to generated HTML report
        """
        if self.main_df is None:
            self.load_data()

        # Assess quality if not already done
        if not self.quality_summary:
            self.assess_quality()

        # Create plots
        structural_figs = self.create_metric_plots("structural_metrics")
        asl_figs = self.create_metric_plots("asl_metrics")
        regional_figs = self.create_regional_plots()
        registration_figs = self.create_registration_plot()

        # Copy subject PDF files
        self._copy_subject_pdfs()

        # Generate HTML report
        html_content = self._generate_html_report(
            structural_figs, asl_figs, regional_figs, registration_figs
        )

        report_path = self.output_dir / f"{output_name}.html"
        with open(report_path, "w") as f:
            f.write(html_content)

        self.logger.info(f"Quality report saved to {report_path}")
        return report_path

    def _generate_html_report(
        self, structural_figs, asl_figs, regional_figs, registration_figs=None
    ):
        """Generate HTML content for the quality report."""
        html_content = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>ExploreASL Quality Report</title>
            <script src=\"https://cdn.plot.ly/plotly-latest.min.js\"></script>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; }
                .section { margin: 20px 0; }
                .metric-group { margin: 15px 0; }
                .quality-summary { background: #f5f5f5; padding: 15px; border-radius: 5px; }
                .failed { color: red; }
                .passed { color: green; }
                .warning { color: orange; }
                .outliers_detected { color: #ff6600; font-weight: bold; }
                .no_data { color: #666666; font-style: italic; }
                .stats-info { font-size: 0.9em; color: #666; margin-left: 20px; }
            </style>
        </head>
        <body>
            <h1>ExploreASL Quality Report</h1>
            <div class=\"quality-summary\">
                <h2>Quality Summary</h2>
        """
        # Add quality summary
        for group_name, group_results in self.quality_summary.items():
            if group_name == "data_summary":
                continue
            html_content += f"<h3>{group_name.replace('_', ' ').title()}</h3>"

            # Special handling for missing subjects
            if group_name == "missing_subjects":
                summary = group_results.get("summary", {})
                html_content += f'<p class="stats-info">Expected: {summary.get("total_expected", 0)} subjects</p>'
                html_content += f'<p class="stats-info">Actual: {summary.get("total_actual", 0)} subjects</p>'
                html_content += f'<p class="stats-info">Completion Rate: {summary.get("completion_rate", 0):.1f}%</p>'

                if group_results.get("missing_subjects"):
                    html_content += f'<p class="stats-info"><strong>Missing Subjects ({len(group_results["missing_subjects"])}):</strong></p>'
                    html_content += '<ul class="stats-info">'
                    for subject in group_results["missing_subjects"]:
                        html_content += f"<li>{subject}</li>"
                    html_content += "</ul>"

                if group_results.get("extra_subjects"):
                    html_content += f'<p class="stats-info"><strong>Extra Subjects ({len(group_results["extra_subjects"])}):</strong></p>'
                    html_content += '<ul class="stats-info">'
                    for subject in group_results["extra_subjects"]:
                        html_content += f"<li>{subject}</li>"
                    html_content += "</ul>"
                continue

            # Handle regular metric groups
            for metric, result in group_results.items():
                status_class = result["status"]
                html_content += (
                    f'<p class="{status_class}">{metric}: {result["message"]}</p>'
                )

                # Add statistical information for passed and outlier cases
                if (
                    result["status"] in ["passed", "outliers_detected"]
                    and "mean" in result
                ):
                    stats_info = f"Mean: {result['mean']:.2f}, Std: {result['std']:.2f}, Range: {result['lower_bound']:.2f} to {result['upper_bound']:.2f}"
                    html_content += f'<p class="stats-info">{stats_info}</p>'

                # Add outlier subject details if available
                if (
                    result["status"] == "outliers_detected"
                    and "outlier_subjects" in result
                ):
                    html_content += (
                        '<p class="stats-info"><strong>Outlier Subjects:</strong></p>'
                    )
                    html_content += '<ul class="stats-info">'
                    for outlier in result["outlier_subjects"]:
                        html_content += (
                            f"<li>{outlier['subject_id']}: {outlier['value']:.2f}</li>"
                        )
                    html_content += "</ul>"

        # Add plots
        for fig_group, figs in [
            ("Structural Metrics", structural_figs),
            ("ASL Metrics", asl_figs),
            ("Regional CBF", regional_figs),
            (
                "Registration Metrics",
                registration_figs if registration_figs is not None else [],
            ),
        ]:
            if figs:
                html_content += f'<div class="section"><h2>{fig_group}</h2>'
                for fig in figs:
                    html_content += fig.to_html(full_html=False, include_plotlyjs=False)
                html_content += "</div>"

        html_content += """
            </div>
        </body>
        </html>
        """
        return html_content

    def create_dash_app(self) -> Dash:
        """
        Create interactive Dash application.

        Returns
        -------
        Dash
            Dash application instance
        """
        if self.main_df is None:
            self.load_data()

        # Create plots
        structural_figs = self.create_metric_plots("structural_metrics")
        asl_figs = self.create_metric_plots("asl_metrics")
        regional_figs = self.create_regional_plots()
        registration_figs = self.create_registration_plot()

        # Create Dash app
        app = Dash(__name__)

        app.layout = html.Div(
            [
                html.H1("ExploreASL Quality Dashboard"),
                html.Div(
                    [
                        html.H2("Structural Metrics"),
                        html.Div(
                            [dcc.Graph(figure=fig) for fig in structural_figs],
                            style={
                                "display": "grid",
                                "gridTemplateColumns": "repeat(auto-fit, minmax(400px, 1fr))",
                            },
                        ),
                    ]
                ),
                html.Div(
                    [
                        html.H2("ASL Metrics"),
                        html.Div(
                            [dcc.Graph(figure=fig) for fig in asl_figs],
                            style={
                                "display": "grid",
                                "gridTemplateColumns": "repeat(auto-fit, minmax(400px, 1fr))",
                            },
                        ),
                    ]
                ),
                html.Div(
                    [
                        html.H2("Regional CBF"),
                        html.Div(
                            [dcc.Graph(figure=fig) for fig in regional_figs],
                            style={
                                "display": "grid",
                                "gridTemplateColumns": "repeat(auto-fit, minmax(400px, 1fr))",
                            },
                        ),
                    ]
                ),
                html.Div(
                    [
                        html.H2("Registration Metrics"),
                        html.Div(
                            [dcc.Graph(figure=fig) for fig in registration_figs],
                            style={
                                "display": "grid",
                                "gridTemplateColumns": "repeat(auto-fit, minmax(400px, 1fr))",
                            },
                        ),
                    ]
                ),
            ]
        )

        return app


def load_config(config_file: str | Path) -> dict:
    """
    Load configuration from YAML file.

    Parameters
    ----------
    config_file : str or Path
        Path to YAML configuration file

    Returns
    -------
    dict
        Configuration dictionary

    Examples
    --------
    >>> from meierlab.quality.plots import load_config
    >>> config = load_config("exploreasl_config.yaml")
    """
    config_file = Path(config_file)

    if not config_file.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_file}")

    with open(config_file) as f:
        config = yaml.safe_load(f)

    return config


def exploreasl_quality_wf(
    xasl_dir: str | Path,
    output_dir: str | Path | None = None,
    config: dict | str | Path | None = None,
    create_dash: bool = True,
) -> tuple[Path, Dash | None]:
    """
    Workflow to generate ExploreASL quality assessment.

    Parameters
    ----------
    xasl_dir : str or Path
        Path to ExploreASL output directory
    output_dir : str or Path, optional
        Output directory for reports
    config : dict, str, or Path, optional
        Quality check configuration (dict or path to YAML file)
    create_dash : bool, optional
        Whether to create Dash app

    Returns
    -------
    tuple
        (report_path, dash_app) - Path to HTML report and optional Dash app

    Examples
    --------
    >>> from meierlab.quality.plots import exploreasl_quality_wf
    >>> report_path, app = exploreasl_quality_wf("/path/to/exploreasl/output")
    >>> # With custom config
    >>> report_path, app = exploreasl_quality_wf("/path/to/exploreasl/output",
    ...                                         config="my_config.yaml")
    """
    # Load config if it's a file path
    if isinstance(config | (str, Path)):
        config = load_config(config)

    qc = ExploreASLQualityChecker(xasl_dir, config=config, output_dir=output_dir)
    report_path = qc.generate_quality_report()

    dash_app = None
    if create_dash:
        dash_app = qc.create_dash_app()

    return report_path, dash_app


# Backward compatibility
def main():
    """Legacy main function for backward compatibility."""
    import argparse

    parser = argparse.ArgumentParser(description="ExploreASL Quality Check")
    parser.add_argument("xasl_dir", help="Path to ExploreASL output directory")
    parser.add_argument("--output-dir", help="Output directory for reports")
    parser.add_argument("--stats-file", help="Path to specific stats file")
    parser.add_argument("--no-dash", action="store_true", help="Don't create Dash app")

    args = parser.parse_args()

    qc = ExploreASLQualityChecker(args.xasl_dir, output_dir=args.output_dir)
    qc.load_data(args.stats_file)

    report_path = qc.generate_quality_report()
    print(f"Quality report saved to: {report_path}")

    if not args.no_dash:
        app = qc.create_dash_app()
        print("Starting Dash app...")
        app.run(debug=True)


if __name__ == "__main__":
    main()
