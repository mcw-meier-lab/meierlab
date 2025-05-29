"""
Quality metrics and visualization for tractography analysis.
"""

import os
from concurrent.futures import ProcessPoolExecutor
from typing import Any

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dipy import utils
from dipy.reconst.csdeconv import ConstrainedSphericalDeconvModel
from dipy.tracking.metrics import mean_curvature, mean_orientation
from dipy.tracking.streamlinespeed import Streamlines
from dipy.tracking.utils import density_map
from dipy.viz import actor, window
from plotly.subplots import make_subplots
from scipy import stats
from scipy.stats import f_oneway, ttest_ind
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import ParameterGrid
from statsmodels.stats.multitest import multipletests


class TractographyMetrics:
    """Class for computing and analyzing tractography quality metrics."""

    def __init__(
        self, affine: np.ndarray, mask_shape: tuple, cc_mask: np.ndarray | None = None
    ):
        """
        Initialize metrics computation.

        Args:
            affine: Affine transformation matrix
            mask_shape: Shape of the brain mask
            cc_mask: Optional corpus callosum mask
        """
        self.affine = affine
        self.mask_shape = mask_shape
        self.cc_mask = cc_mask

    def compute_streamline_metrics(
        self,
        streamlines: Streamlines,
        fod: dict,
        sift_weights: np.ndarray | None = None,
        act: bool = True,
        params: dict | None = None,
    ) -> dict[str, Any]:
        """Compute comprehensive quality metrics for filtered streamlines."""
        from dipy.tracking.metrics import mean_curvature, mean_orientation
        from dipy.tracking.streamlinespeed import length
        from dipy.tracking.utils import density_map

        metrics = {}

        # Basic metrics
        metrics["n_streamlines"] = len(streamlines)
        metrics["mean_length"] = np.mean(length(streamlines))
        metrics["std_length"] = np.std(length(streamlines))
        metrics["min_length"] = np.min(length(streamlines))
        metrics["max_length"] = np.max(length(streamlines))

        # Curvature metrics
        curvatures = mean_curvature(streamlines)
        metrics["mean_curvature"] = np.mean(curvatures)
        metrics["std_curvature"] = np.std(curvatures)
        metrics["max_curvature"] = np.max(curvatures)

        # Orientation metrics
        orientations = mean_orientation(streamlines)
        metrics["mean_orientation"] = np.mean(orientations)
        metrics["std_orientation"] = np.std(orientations)

        # Density metrics
        density = density_map(streamlines, self.affine, self.mask_shape)
        metrics["mean_density"] = np.mean(density[density > 0])
        metrics["std_density"] = np.std(density[density > 0])
        metrics["max_density"] = np.max(density)

        # SIFT-specific metrics
        if sift_weights is not None:
            metrics["mean_sift_weight"] = np.mean(sift_weights)
            metrics["std_sift_weight"] = np.std(sift_weights)
            metrics["min_sift_weight"] = np.min(sift_weights)
            metrics["max_sift_weight"] = np.max(sift_weights)

            # Compute weight distribution
            weight_bins = np.linspace(0, 1, 11)
            weight_hist, _ = np.histogram(sift_weights, bins=weight_bins)
            metrics["weight_distribution"] = weight_hist / len(sift_weights)

            # Compute weight-length correlation
            if len(streamlines) > 0:
                lengths = length(streamlines)
                metrics["weight_length_correlation"] = np.corrcoef(
                    lengths, sift_weights
                )[0, 1]

            # Compute weight-density correlation
            if density.size > 0:
                density_values = density[density > 0]
                metrics["weight_density_correlation"] = np.corrcoef(
                    density_values, sift_weights[: len(density_values)]
                )[0, 1]

        # ACT-specific metrics
        if (
            act
            and params is not None
            and all(
                pve is not None
                for pve in [
                    params.get("wm_pve"),
                    params.get("gm_pve"),
                    params.get("csf_pve"),
                ]
            )
        ):
            wm_density = density * params["wm_pve"]
            gm_density = density * params["gm_pve"]
            csf_density = density * params["csf_pve"]

            metrics["mean_wm_density"] = np.mean(wm_density[wm_density > 0])
            metrics["mean_gm_density"] = np.mean(gm_density[gm_density > 0])
            metrics["mean_csf_density"] = np.mean(csf_density[csf_density > 0])

            metrics["wm_fraction"] = np.sum(wm_density) / np.sum(density)
            metrics["gm_fraction"] = np.sum(gm_density) / np.sum(density)
            metrics["csf_fraction"] = np.sum(csf_density) / np.sum(density)

            # Compute SIFT weight correlations with tissue densities
            if sift_weights is not None:
                metrics["weight_wm_correlation"] = np.corrcoef(
                    wm_density[wm_density > 0],
                    sift_weights[: len(wm_density[wm_density > 0])],
                )[0, 1]
                metrics["weight_gm_correlation"] = np.corrcoef(
                    gm_density[gm_density > 0],
                    sift_weights[: len(gm_density[gm_density > 0])],
                )[0, 1]
                metrics["weight_csf_correlation"] = np.corrcoef(
                    csf_density[csf_density > 0],
                    sift_weights[: len(csf_density[csf_density > 0])],
                )[0, 1]

        # FOD agreement metrics
        if "wm_fod_norm" in fod:
            from dipy.tracking.metrics import fod_agreement

            agreement = fod_agreement(streamlines, fod["wm_fod_norm"])
            metrics["mean_fod_agreement"] = np.mean(agreement)
            metrics["std_fod_agreement"] = np.std(agreement)
            metrics["min_fod_agreement"] = np.min(agreement)
            metrics["max_fod_agreement"] = np.max(agreement)

            # Compute SIFT weight-FOD agreement correlation
            if sift_weights is not None:
                metrics["weight_fod_correlation"] = np.corrcoef(
                    agreement, sift_weights
                )[0, 1]

        # Corpus Callosum specific metrics
        if self.cc_mask is not None:
            cc_metrics = self._compute_cc_metrics(streamlines, density, sift_weights)
            metrics.update(cc_metrics)

        return metrics

    def _compute_cc_metrics(
        self,
        streamlines: Streamlines,
        density: np.ndarray,
        sift_weights: np.ndarray | None = None,
    ) -> dict[str, Any]:
        """Compute corpus callosum specific metrics."""
        metrics = {}

        # Get streamlines in CC
        cc_streamlines = []
        cc_indices = []
        for i, streamline in enumerate(streamlines):
            points = np.round(streamline).astype(int)
            if np.any(self.cc_mask[points[:, 0], points[:, 1], points[:, 2]]):
                cc_streamlines.append(streamline)
                cc_indices.append(i)

        if not cc_streamlines:
            return metrics

        cc_streamlines = Streamlines(cc_streamlines)

        # CC-specific metrics
        metrics["cc_n_streamlines"] = len(cc_streamlines)
        metrics["cc_fraction"] = len(cc_streamlines) / len(streamlines)

        # CC density metrics
        cc_density = density_map(cc_streamlines, self.affine, self.mask_shape)
        metrics["cc_mean_density"] = np.mean(cc_density[cc_density > 0])
        metrics["cc_std_density"] = np.std(cc_density[cc_density > 0])
        metrics["cc_max_density"] = np.max(cc_density)

        # CC shape metrics
        from dipy.tracking.metrics import mean_curvature, mean_orientation

        cc_curvatures = mean_curvature(cc_streamlines)
        metrics["cc_mean_curvature"] = np.mean(cc_curvatures)
        metrics["cc_std_curvature"] = np.std(cc_curvatures)

        cc_orientations = mean_orientation(cc_streamlines)
        metrics["cc_mean_orientation"] = np.mean(cc_orientations)
        metrics["cc_std_orientation"] = np.std(cc_orientations)

        # CC symmetry metrics
        left_density = cc_density[:, :, : cc_density.shape[2] // 2]
        right_density = cc_density[:, :, cc_density.shape[2] // 2 :]
        metrics["cc_symmetry_score"] = 1 - np.abs(
            np.sum(left_density) - np.sum(right_density)
        ) / np.sum(cc_density)

        # CC SIFT metrics
        if sift_weights is not None and cc_indices:
            cc_weights = sift_weights[cc_indices]
            metrics["cc_mean_sift_weight"] = np.mean(cc_weights)
            metrics["cc_std_sift_weight"] = np.std(cc_weights)
            metrics["cc_min_sift_weight"] = np.min(cc_weights)
            metrics["cc_max_sift_weight"] = np.max(cc_weights)

            # Compute weight distribution for CC streamlines
            weight_bins = np.linspace(0, 1, 11)
            weight_hist, _ = np.histogram(cc_weights, bins=weight_bins)
            metrics["cc_weight_distribution"] = weight_hist / len(cc_weights)

        return metrics

    def compare_filtering_runs(
        self,
        results: list[dict[str, Any]],
        output_dir: str,
        correction_method: str = "fdr_bh",
    ) -> dict[str, Any]:
        """Compare metrics between different filtering runs with multiple comparison correction."""
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)

        # Extract metrics from all runs
        all_metrics = []
        for i, result in enumerate(results):
            metrics = result["metrics"].copy()
            metrics["run"] = i
            all_metrics.append(metrics)

        # Convert to DataFrame
        df = pd.DataFrame(all_metrics)

        # Perform statistical tests
        stats_results = {}
        p_values = []
        metrics_list = []

        for metric in df.columns:
            if metric not in ["run", "weight_distribution"]:
                # ANOVA for multiple runs
                if len(results) > 2:
                    f_stat, p_value = stats.f_oneway(
                        *[df[df["run"] == i][metric] for i in range(len(results))]
                    )
                    stats_results[metric] = {
                        "test": "ANOVA",
                        "statistic": f_stat,
                        "p_value": p_value,
                    }
                # t-test for two runs
                else:
                    t_stat, p_value = stats.ttest_ind(
                        df[df["run"] == 0][metric], df[df["run"] == 1][metric]
                    )
                    stats_results[metric] = {
                        "test": "t-test",
                        "statistic": t_stat,
                        "p_value": p_value,
                    }
                p_values.append(p_value)
                metrics_list.append(metric)

        # Apply multiple comparison correction
        if p_values:
            reject, p_adjusted, _, _ = multipletests(
                p_values, alpha=0.05, method=correction_method
            )

            # Update stats results with corrected p-values
            for metric, p_adj, rej in zip(
                metrics_list, p_adjusted, reject, strict=False
            ):
                stats_results[metric]["p_value_corrected"] = p_adj
                stats_results[metric]["significant"] = rej

        # Generate interactive plots
        self._plot_interactive_comparison(df, stats_results, output_dir)

        # Save statistical results
        stats_df = pd.DataFrame(stats_results).T
        stats_df.to_csv(f"{output_dir}/statistical_comparison.csv")

        return stats_results

    def _plot_interactive_comparison(
        self, df: pd.DataFrame, stats_results: dict[str, Any], output_dir: str
    ) -> None:
        """Generate interactive comparison plots using Plotly."""
        # Create basic metrics comparison
        basic_metrics = [
            "n_streamlines",
            "mean_length",
            "mean_curvature",
            "mean_fod_agreement",
        ]
        fig = make_subplots(rows=2, cols=2, subplot_titles=basic_metrics)

        for i, metric in enumerate(basic_metrics):
            if metric in df.columns:
                row = i // 2 + 1
                col = i % 2 + 1

                # Add box plot
                fig.add_trace(
                    go.Box(
                        y=df[metric],
                        x=df["run"],
                        name=metric,
                        boxpoints="all",
                        jitter=0.3,
                        pointpos=-1.8,
                    ),
                    row=row,
                    col=col,
                )

                # Add significance annotation
                if metric in stats_results:
                    p_value = stats_results[metric].get(
                        "p_value_corrected", stats_results[metric]["p_value"]
                    )
                    significance = (
                        "***"
                        if p_value < 0.001
                        else "**"
                        if p_value < 0.01
                        else "*"
                        if p_value < 0.05
                        else "ns"
                    )
                    fig.add_annotation(
                        text=f"p = {p_value:.4f} {significance}",
                        xref=f"x{i + 1}",
                        yref=f"y{i + 1}",
                        x=0.5,
                        y=0.95,
                        showarrow=False,
                        font=dict(size=12),
                    )

        fig.update_layout(height=800, width=1200, title_text="Basic Metrics Comparison")
        fig.write_html(f"{output_dir}/basic_metrics_comparison.html")

        # Create CC metrics comparison if available
        cc_metrics = [col for col in df.columns if col.startswith("cc_")]
        if cc_metrics:
            fig = make_subplots(rows=2, cols=2, subplot_titles=cc_metrics[:4])

            for i, metric in enumerate(cc_metrics[:4]):
                row = i // 2 + 1
                col = i % 2 + 1

                fig.add_trace(
                    go.Box(
                        y=df[metric],
                        x=df["run"],
                        name=metric,
                        boxpoints="all",
                        jitter=0.3,
                        pointpos=-1.8,
                    ),
                    row=row,
                    col=col,
                )

                if metric in stats_results:
                    p_value = stats_results[metric].get(
                        "p_value_corrected", stats_results[metric]["p_value"]
                    )
                    significance = (
                        "***"
                        if p_value < 0.001
                        else "**"
                        if p_value < 0.01
                        else "*"
                        if p_value < 0.05
                        else "ns"
                    )
                    fig.add_annotation(
                        text=f"p = {p_value:.4f} {significance}",
                        xref=f"x{i + 1}",
                        yref=f"y{i + 1}",
                        x=0.5,
                        y=0.95,
                        showarrow=False,
                        font=dict(size=12),
                    )

            fig.update_layout(
                height=800, width=1200, title_text="CC Metrics Comparison"
            )
            fig.write_html(f"{output_dir}/cc_metrics_comparison.html")

        # Create correlation matrix
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 1:
            corr_matrix = df[numeric_cols].corr()

            fig = go.Figure(
                data=go.Heatmap(
                    z=corr_matrix,
                    x=corr_matrix.columns,
                    y=corr_matrix.columns,
                    colorscale="RdBu",
                    zmid=0,
                    text=np.round(corr_matrix, 2),
                    texttemplate="%{text}",
                    textfont={"size": 10},
                )
            )

            fig.update_layout(title="Metric Correlations", height=800, width=800)
            fig.write_html(f"{output_dir}/metric_correlations.html")

        # Create parallel coordinates plot
        fig = px.parallel_coordinates(
            df,
            color="run",
            dimensions=[col for col in df.columns if col != "run"],
            color_continuous_scale=px.colors.sequential.Viridis,
            title="Parallel Coordinates Plot of All Metrics",
        )
        fig.write_html(f"{output_dir}/parallel_coordinates.html")

        # Create scatter plot matrix
        fig = px.scatter_matrix(
            df,
            dimensions=[col for col in df.columns if col != "run"],
            color="run",
            title="Scatter Plot Matrix of Metrics",
        )
        fig.write_html(f"{output_dir}/scatter_matrix.html")

        # Create statistical summary table
        stats_df = pd.DataFrame(stats_results).T
        stats_df["significant"] = stats_df["significant"].map(
            {True: "Yes", False: "No"}
        )

        fig = go.Figure(
            data=[
                go.Table(
                    header=dict(
                        values=list(stats_df.columns),
                        fill_color="paleturquoise",
                        align="left",
                    ),
                    cells=dict(
                        values=[stats_df[col] for col in stats_df.columns],
                        fill_color="lavender",
                        align="left",
                    ),
                )
            ]
        )

        fig.update_layout(title="Statistical Summary", height=400, width=1200)
        fig.write_html(f"{output_dir}/statistical_summary.html")

    def _compute_cmc_metrics(
        self, streamlines: Streamlines, tissue_maps: dict[str, np.ndarray]
    ) -> dict:
        """Compute CMC-specific quality metrics.

        Args:
            streamlines: Generated streamlines
            tissue_maps: Dictionary containing tissue probability maps

        Returns:
            Dictionary containing CMC metrics
        """
        # Initialize metrics dictionary
        metrics = {
            "tissue_probs": {},
            "fractions": {},
            "entropy": {},
            "quality_score": 0.0,
        }

        # Compute tissue probabilities along streamlines
        wm_probs = []
        gm_probs = []
        csf_probs = []

        for streamline in streamlines:
            # Convert streamline points to voxel coordinates
            vox_coords = np.round(
                utils.apply_affine(np.linalg.inv(self.affine), streamline)
            ).astype(int)

            # Get tissue probabilities for each point
            wm_probs.extend(
                [tissue_maps["wm_pve"][tuple(coord)] for coord in vox_coords]
            )
            gm_probs.extend(
                [tissue_maps["gm_pve"][tuple(coord)] for coord in vox_coords]
            )
            csf_probs.extend(
                [tissue_maps["csf_pve"][tuple(coord)] for coord in vox_coords]
            )

        # Compute statistics for each tissue type
        for tissue, probs in [("wm", wm_probs), ("gm", gm_probs), ("csf", csf_probs)]:
            metrics["tissue_probs"][tissue] = {
                "mean": np.mean(probs),
                "std": np.std(probs),
                "min": np.min(probs),
                "max": np.max(probs),
                "median": np.median(probs),
            }

            # Compute fraction of points with probability > 0.5
            metrics["fractions"][tissue] = np.mean(np.array(probs) > 0.5)

        # Compute tissue entropy
        metrics["entropy"] = {
            "mean": self._compute_tissue_entropy(wm_probs, gm_probs, csf_probs),
            "per_point": self._compute_per_point_entropy(wm_probs, gm_probs, csf_probs),
        }

        # Compute quality score
        metrics["quality_score"] = self._compute_cmc_quality_score(metrics)

        return metrics

    def _compute_per_point_entropy(
        self, wm_probs: list[float], gm_probs: list[float], csf_probs: list[float]
    ) -> list[float]:
        """Compute entropy for each point along streamlines."""
        entropies = []
        for wm, gm, csf in zip(wm_probs, gm_probs, csf_probs, strict=False):
            # Normalize probabilities
            total = wm + gm + csf
            if total > 0:
                wm_norm = wm / total
                gm_norm = gm / total
                csf_norm = csf / total

                # Compute entropy
                entropy = 0
                for p in [wm_norm, gm_norm, csf_norm]:
                    if p > 0:
                        entropy -= p * np.log2(p)
                entropies.append(entropy)
            else:
                entropies.append(0)
        return entropies

    def _compute_cmc_quality_score(self, metrics: dict) -> float:
        """Compute overall quality score for CMC-based tracking."""
        # Weights for different metrics
        weights = {"wm_prob_mean": 0.4, "wm_fraction": 0.3, "entropy": 0.3}

        # Normalize metrics to [0, 1] range
        wm_prob_score = metrics["tissue_probs"]["wm"]["mean"]  # Already in [0, 1]
        wm_fraction_score = metrics["fractions"]["wm"]  # Already in [0, 1]

        # Normalize entropy (lower is better)
        max_entropy = 1.585  # Maximum entropy for 3 classes
        entropy_score = 1 - (metrics["entropy"]["mean"] / max_entropy)

        # Compute weighted score
        quality_score = (
            weights["wm_prob_mean"] * wm_prob_score
            + weights["wm_fraction"] * wm_fraction_score
            + weights["entropy"] * entropy_score
        )

        return quality_score

    def visualize_cmc_metrics(self, metrics: dict, output_dir: str) -> None:
        """Generate visualizations for CMC metrics.

        Args:
            metrics: Dictionary containing CMC metrics
            output_dir: Directory to save visualizations
        """
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots

        # Create output directory
        os.makedirs(output_dir, exist_ok=True)

        # Create subplots for tissue probabilities
        fig = make_subplots(
            rows=2,
            cols=2,
            subplot_titles=(
                "Tissue Probabilities",
                "Tissue Fractions",
                "Entropy Distribution",
                "Quality Score",
            ),
        )

        # Plot tissue probabilities
        tissues = ["wm", "gm", "csf"]
        prob_means = [metrics["tissue_probs"][t]["mean"] for t in tissues]
        prob_stds = [metrics["tissue_probs"][t]["std"] for t in tissues]

        fig.add_trace(
            go.Bar(
                x=tissues,
                y=prob_means,
                error_y=dict(type="data", array=prob_stds),
                name="Mean Probability",
            ),
            row=1,
            col=1,
        )

        # Plot tissue fractions
        fractions = [metrics["fractions"][t] for t in tissues]
        fig.add_trace(
            go.Bar(x=tissues, y=fractions, name="Fraction > 0.5"), row=1, col=2
        )

        # Plot entropy distribution
        if "per_point" in metrics["entropy"]:
            fig.add_trace(
                go.Histogram(
                    x=metrics["entropy"]["per_point"], name="Point-wise Entropy"
                ),
                row=2,
                col=1,
            )

        # Plot quality score
        fig.add_trace(
            go.Indicator(
                mode="gauge+number",
                value=metrics["quality_score"],
                title={"text": "Quality Score"},
                gauge={"axis": {"range": [0, 1]}},
                domain={"row": 1, "column": 2},
            ),
            row=2,
            col=2,
        )

        # Update layout
        fig.update_layout(
            title="CMC Metrics Visualization", showlegend=True, height=800
        )

        # Save plot
        fig.write_html(os.path.join(output_dir, "cmc_metrics.html"))

        # Create additional visualizations
        self._plot_tissue_probability_distributions(metrics, output_dir)
        self._plot_entropy_along_streamlines(metrics, output_dir)

    def _plot_tissue_probability_distributions(
        self, metrics: dict, output_dir: str
    ) -> None:
        """Plot probability distributions for each tissue type."""
        import plotly.graph_objects as go

        fig = go.Figure()

        tissues = ["wm", "gm", "csf"]
        colors = ["blue", "green", "red"]

        for tissue, color in zip(tissues, colors, strict=False):
            probs = metrics["tissue_probs"][tissue]
            fig.add_trace(
                go.Box(
                    y=[probs["mean"]],
                    name=tissue.upper(),
                    boxpoints="all",
                    jitter=0.3,
                    pointpos=-1.8,
                    marker_color=color,
                )
            )

        fig.update_layout(
            title="Tissue Probability Distributions",
            yaxis_title="Probability",
            showlegend=True,
        )

        fig.write_html(
            os.path.join(output_dir, "tissue_probability_distributions.html")
        )

    def _plot_entropy_along_streamlines(self, metrics: dict, output_dir: str) -> None:
        """Plot entropy distribution along streamlines."""
        import plotly.graph_objects as go

        if "per_point" not in metrics["entropy"]:
            return

        fig = go.Figure()

        fig.add_trace(
            go.Histogram(
                x=metrics["entropy"]["per_point"], name="Point-wise Entropy", nbinsx=50
            )
        )

        fig.update_layout(
            title="Entropy Distribution Along Streamlines",
            xaxis_title="Entropy",
            yaxis_title="Count",
            showlegend=True,
        )

        fig.write_html(os.path.join(output_dir, "entropy_distribution.html"))

    def quality_control(
        self, streamlines: Streamlines, response: dict, tissue_maps: dict | None = None
    ) -> dict:
        """Perform quality control on tractography results.

        Args:
            streamlines: Generated streamlines
            response: Response function dictionary
            tissue_maps: Optional tissue probability maps

        Returns:
            Dictionary containing QC metrics
        """
        # Compute basic metrics
        metrics = {
            "streamlines": self._compute_streamline_metrics(streamlines),
            "response": self._compute_response_metrics(response),
        }

        # Add CMC metrics if tissue maps are available
        if tissue_maps is not None and all(
            k in tissue_maps for k in ["wm_pve", "gm_pve", "csf_pve"]
        ):
            metrics["cmc"] = self._compute_cmc_metrics(streamlines, tissue_maps)

        return metrics

    def _compute_response_metrics(self, response: dict) -> dict:
        """Quality control metrics for response function."""
        metrics = {}

        if "ratio" in response:
            metrics["response_ratio"] = response["ratio"]

        if isinstance(response["response"], ConstrainedSphericalDeconvModel):
            # Check response function shape
            metrics["response_shape"] = response["response"].response.shape

            # Check SH coefficients
            if hasattr(response["response"], "sh_coeff"):
                metrics["sh_coeff_norm"] = np.linalg.norm(response["response"].sh_coeff)

        return metrics

    def visualize_qc(self, qc_metrics: dict, output_dir: str) -> None:
        """
        Generate visualization of quality control metrics.

        Args:
            qc_metrics: Dictionary of QC metrics
            output_dir: Directory to save visualizations
        """
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

        # Plot streamline statistics
        if "streamlines" in qc_metrics:
            self._plot_streamline_stats(qc_metrics["streamlines"], output_dir)

        # Plot tissue metrics if available
        if "tissue" in qc_metrics:
            self._plot_tissue_metrics(qc_metrics["tissue"], output_dir)

        # Save density map if available
        if "density_map" in qc_metrics.get("streamlines", {}):
            self._save_density_map(qc_metrics["streamlines"]["density_map"], output_dir)

    def _plot_streamline_stats(self, metrics: dict, output_dir: str) -> None:
        """Plot streamline statistics."""
        # Create figure
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # Plot length distribution
        axes[0].hist([len(s) for s in metrics.get("streamlines", [])], bins=50)
        axes[0].set_title("Streamline Length Distribution")
        axes[0].set_xlabel("Length")
        axes[0].set_ylabel("Count")

        # Plot other metrics
        if "silhouette_score" in metrics:
            axes[1].bar(["Silhouette Score"], [metrics["silhouette_score"]])
            axes[1].set_title("Streamline Clustering Quality")

        plt.tight_layout()
        plt.savefig(f"{output_dir}/streamline_stats.png")
        plt.close()

    def _plot_tissue_metrics(self, metrics: dict, output_dir: str) -> None:
        """Plot tissue-based metrics."""
        # Create figure
        plt.figure(figsize=(10, 5))

        # Plot mean probabilities
        tissues = [
            k.replace("_mean_prob", "")
            for k in metrics.keys()
            if k.endswith("_mean_prob")
        ]
        means = [metrics[f"{t}_mean_prob"] for t in tissues]
        stds = [metrics[f"{t}_std_prob"] for t in tissues]

        plt.bar(tissues, means, yerr=stds)
        plt.title("Mean Tissue Probabilities Along Streamlines")
        plt.ylabel("Probability")

        plt.tight_layout()
        plt.savefig(f"{output_dir}/tissue_metrics.png")
        plt.close()

    def _save_density_map(self, density: np.ndarray, output_dir: str) -> None:
        """Save streamline density map."""
        # Create NIfTI image
        density_img = nib.Nifti1Image(density, self.affine)
        nib.save(density_img, f"{output_dir}/streamline_density.nii.gz")

        # Create visualization
        renderer = window.Renderer()
        stream_actor = actor.contour_from_roi(density, self.affine, color=(1, 1, 0))
        renderer.add(stream_actor)
        window.record(
            renderer,
            out_path=f"{output_dir}/density_visualization.png",
            size=(800, 800),
        )

    def compare_qc_metrics(
        self,
        qc_metrics_list: list[dict],
        run_names: list[str] | None = None,
        output_dir: str = "qc_comparison",
    ) -> dict[str, Any]:
        """
        Compare QC metrics across multiple pipeline runs.

        Args:
            qc_metrics_list: List of QC metrics dictionaries from different runs
            run_names: Optional list of names for each run
            output_dir: Directory to save comparison results

        Returns:
            Dictionary containing comparison results and statistical tests
        """
        import pandas as pd
        from scipy import stats
        from statsmodels.stats.multitest import multipletests

        # Create output directory
        os.makedirs(output_dir, exist_ok=True)

        # Default run names if not provided
        if run_names is None:
            run_names = [f"Run_{i}" for i in range(len(qc_metrics_list))]

        # Extract metrics for comparison
        comparison_data = {}
        for run_name, qc_metrics in zip(run_names, qc_metrics_list, strict=False):
            # Flatten nested metrics
            flat_metrics = self._flatten_metrics(qc_metrics)
            comparison_data[run_name] = flat_metrics

        # Convert to DataFrame
        df = pd.DataFrame(comparison_data).T

        # Perform statistical tests
        stats_results = {}
        p_values = []
        metrics_list = []

        for metric in df.columns:
            # Skip non-numeric metrics
            if not pd.api.types.is_numeric_dtype(df[metric]):
                continue

            # ANOVA for multiple runs
            if len(run_names) > 2:
                f_stat, p_value = stats.f_oneway(
                    *[df[metric].values for _ in range(len(run_names))]
                )
                stats_results[metric] = {
                    "test": "ANOVA",
                    "statistic": f_stat,
                    "p_value": p_value,
                }
            # t-test for two runs
            else:
                t_stat, p_value = stats.ttest_ind(
                    df[metric].values[0], df[metric].values[1]
                )
                stats_results[metric] = {
                    "test": "t-test",
                    "statistic": t_stat,
                    "p_value": p_value,
                }
            p_values.append(p_value)
            metrics_list.append(metric)

        # Apply multiple comparison correction
        if p_values:
            reject, p_adjusted, _, _ = multipletests(
                p_values, alpha=0.05, method="fdr_bh"
            )

            # Update stats results with corrected p-values
            for metric, p_adj, rej in zip(
                metrics_list, p_adjusted, reject, strict=False
            ):
                stats_results[metric]["p_value_corrected"] = p_adj
                stats_results[metric]["significant"] = rej

        # Generate comparison visualizations
        self._plot_qc_comparison(df, stats_results, output_dir)

        # Save statistical results
        stats_df = pd.DataFrame(stats_results).T
        stats_df.to_csv(f"{output_dir}/statistical_comparison.csv")

        return {"comparison_data": df, "statistical_results": stats_results}

    def _flatten_metrics(self, metrics: dict, prefix: str = "") -> dict:
        """Flatten nested metrics dictionary."""
        flat_metrics = {}
        for key, value in metrics.items():
            new_key = f"{prefix}{key}" if prefix else key
            if isinstance(value, dict):
                flat_metrics.update(self._flatten_metrics(value, f"{new_key}_"))
            else:
                flat_metrics[new_key] = value
        return flat_metrics

    def _plot_qc_comparison(
        self, df: pd.DataFrame, stats_results: dict[str, Any], output_dir: str
    ) -> None:
        """Generate comparison plots for QC metrics."""
        # Create basic metrics comparison
        fig = make_subplots(rows=2, cols=2, subplot_titles=df.columns[:4])

        for i, metric in enumerate(df.columns[:4]):
            row = i // 2 + 1
            col = i % 2 + 1

            # Add box plot
            fig.add_trace(
                go.Box(
                    y=df[metric],
                    x=df.index,
                    name=metric,
                    boxpoints="all",
                    jitter=0.3,
                    pointpos=-1.8,
                ),
                row=row,
                col=col,
            )

            # Add significance annotation
            if metric in stats_results:
                p_value = stats_results[metric].get(
                    "p_value_corrected", stats_results[metric]["p_value"]
                )
                significance = (
                    "***"
                    if p_value < 0.001
                    else "**"
                    if p_value < 0.01
                    else "*"
                    if p_value < 0.05
                    else "ns"
                )
                fig.add_annotation(
                    text=f"p = {p_value:.4f} {significance}",
                    xref=f"x{i + 1}",
                    yref=f"y{i + 1}",
                    x=0.5,
                    y=0.95,
                    showarrow=False,
                    font=dict(size=12),
                )

        fig.update_layout(height=800, width=1200, title_text="QC Metrics Comparison")
        fig.write_html(f"{output_dir}/qc_metrics_comparison.html")

        # Create correlation matrix
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 1:
            corr_matrix = df[numeric_cols].corr()

            fig = go.Figure(
                data=go.Heatmap(
                    z=corr_matrix,
                    x=corr_matrix.columns,
                    y=corr_matrix.columns,
                    colorscale="RdBu",
                    zmid=0,
                    text=np.round(corr_matrix, 2),
                    texttemplate="%{text}",
                    textfont={"size": 10},
                )
            )

            fig.update_layout(title="QC Metrics Correlations", height=800, width=800)
            fig.write_html(f"{output_dir}/qc_metrics_correlations.html")

    def generate_qc_report(
        self, qc_metrics: dict, output_dir: str, format: str = "html"
    ) -> None:
        """
        Generate QC report in specified format.

        Args:
            qc_metrics: Dictionary of QC metrics
            output_dir: Directory to save report
            format: Report format ('html', 'pdf', 'markdown')
        """
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)

        # Flatten metrics for easier processing
        flat_metrics = self._flatten_metrics(qc_metrics)

        if format == "html":
            self._generate_html_report(flat_metrics, output_dir)
        elif format == "pdf":
            self._generate_pdf_report(flat_metrics, output_dir)
        elif format == "markdown":
            self._generate_markdown_report(flat_metrics, output_dir)
        else:
            raise ValueError(f"Unsupported report format: {format}")

    def _generate_html_report(self, metrics: dict, output_dir: str) -> None:
        """Generate HTML QC report."""
        import jinja2

        # Create template
        template = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Tractography QC Report</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; }
                table { border-collapse: collapse; width: 100%; margin-bottom: 20px; }
                th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
                th { background-color: #f2f2f2; }
                .metric-value { font-family: monospace; }
                .section { margin-bottom: 30px; }
                .warning { color: orange; }
                .error { color: red; }
            </style>
        </head>
        <body>
            <h1>Tractography QC Report</h1>
            <div class="section">
                <h2>Summary</h2>
                <table>
                    <tr>
                        <th>Metric</th>
                        <th>Value</th>
                        <th>Status</th>
                    </tr>
                    {% for metric, value in metrics.items() %}
                    <tr>
                        <td>{{ metric }}</td>
                        <td class="metric-value">{{ value }}</td>
                        <td>{{ get_status(metric, value) }}</td>
                    </tr>
                    {% endfor %}
                </table>
            </div>
        </body>
        </html>
        """

        # Define status function
        def get_status(metric, value):
            if "error" in metric.lower():
                return '<span class="error">Error</span>'
            elif "warning" in metric.lower():
                return '<span class="warning">Warning</span>'
            return "OK"

        # Render template
        env = jinja2.Environment()
        env.globals["get_status"] = get_status
        template = env.from_string(template)
        html_content = template.render(metrics=metrics)

        # Save report
        with open(f"{output_dir}/qc_report.html", "w") as f:
            f.write(html_content)

    def _generate_pdf_report(self, metrics: dict, output_dir: str) -> None:
        """Generate PDF QC report."""
        from reportlab.lib import colors
        from reportlab.lib.pagesizes import letter
        from reportlab.platypus import SimpleDocTemplate, Table, TableStyle

        # Create PDF document
        doc = SimpleDocTemplate(
            f"{output_dir}/qc_report.pdf",
            pagesize=letter,
            rightMargin=72,
            leftMargin=72,
            topMargin=72,
            bottomMargin=72,
        )

        # Create table data
        table_data = [["Metric", "Value", "Status"]]
        for metric, value in metrics.items():
            status = "OK"
            if "error" in metric.lower():
                status = "Error"
            elif "warning" in metric.lower():
                status = "Warning"
            table_data.append([metric, str(value), status])

        # Create table
        table = Table(table_data)
        table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.grey),
                    ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
                    ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                    ("FONTSIZE", (0, 0), (-1, 0), 14),
                    ("BOTTOMPADDING", (0, 0), (-1, 0), 12),
                    ("BACKGROUND", (0, 1), (-1, -1), colors.beige),
                    ("TEXTCOLOR", (0, 1), (-1, -1), colors.black),
                    ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
                    ("FONTSIZE", (0, 1), (-1, -1), 12),
                    ("GRID", (0, 0), (-1, -1), 1, colors.black),
                ]
            )
        )

        # Build PDF
        doc.build([table])

    def _generate_markdown_report(self, metrics: dict, output_dir: str) -> None:
        """Generate Markdown QC report."""
        # Create markdown content
        markdown_content = "# Tractography QC Report\n\n"
        markdown_content += "## Summary\n\n"
        markdown_content += "| Metric | Value | Status |\n"
        markdown_content += "|--------|-------|--------|\n"

        for metric, value in metrics.items():
            status = "OK"
            if "error" in metric.lower():
                status = "❌ Error"
            elif "warning" in metric.lower():
                status = "⚠️ Warning"
            markdown_content += f"| {metric} | {value} | {status} |\n"

        # Save report
        with open(f"{output_dir}/qc_report.md", "w") as f:
            f.write(markdown_content)

    def compute_tracking_metrics(
        self,
        streamlines: Streamlines,
        tracking_method: str,
        params: dict,
        fod: dict | None = None,
    ) -> dict[str, Any]:
        """
        Compute comprehensive metrics for tracking results.

        Args:
            streamlines: Generated streamlines
            tracking_method: Tracking method used ('local', 'probabilistic', 'particle', 'eudx')
            params: Tracking parameters used
            fod: Optional FOD dictionary for additional metrics

        Returns:
            Dictionary containing tracking metrics
        """
        metrics = {
            "tracking_method": tracking_method,
            "tracking_params": params,
            "basic_metrics": self._compute_basic_metrics(streamlines),
            "algorithm_specific": self._compute_algorithm_specific_metrics(
                streamlines, tracking_method, params, fod
            ),
        }

        if fod is not None:
            metrics["fod_agreement"] = self._compute_fod_agreement(streamlines, fod)

        return metrics

    def _compute_algorithm_specific_metrics(
        self,
        streamlines: Streamlines,
        tracking_method: str,
        params: dict,
        fod: dict | None = None,
    ) -> dict[str, Any]:
        """Compute metrics specific to each tracking algorithm."""
        metrics = {}

        if tracking_method == "probabilistic":
            metrics.update(self._compute_probabilistic_metrics(streamlines, params))
        elif tracking_method == "particle":
            metrics.update(self._compute_particle_metrics(streamlines, params))
        elif tracking_method == "eudx":
            metrics.update(self._compute_eudx_metrics(streamlines, params))

        return metrics

    def _compute_probabilistic_metrics(
        self, streamlines: Streamlines, params: dict
    ) -> dict[str, Any]:
        """Compute metrics specific to probabilistic tracking."""
        from dipy.tracking.metrics import mean_curvature, mean_orientation

        metrics = {
            "curvature_stats": {
                "mean": np.mean([mean_curvature(s) for s in streamlines]),
                "std": np.std([mean_curvature(s) for s in streamlines]),
            },
            "orientation_stats": {
                "mean": np.mean([mean_orientation(s) for s in streamlines]),
                "std": np.std([mean_orientation(s) for s in streamlines]),
            },
        }

        return metrics

    def _compute_particle_metrics(
        self, streamlines: Streamlines, params: dict
    ) -> dict[str, Any]:
        """Compute metrics specific to particle filtering tracking."""
        from dipy.tracking.metrics import mean_curvature, mean_orientation

        metrics = {
            "particle_stats": {
                "mean_curvature": np.mean([mean_curvature(s) for s in streamlines]),
                "mean_orientation": np.mean([mean_orientation(s) for s in streamlines]),
                "tracking_confidence": self._compute_tracking_confidence(streamlines),
            }
        }

        return metrics

    def _compute_eudx_metrics(
        self, streamlines: Streamlines, params: dict
    ) -> dict[str, Any]:
        """Compute metrics specific to EuDX tracking."""
        from dipy.tracking.metrics import mean_curvature, mean_orientation

        metrics = {
            "eudx_stats": {
                "mean_curvature": np.mean([mean_curvature(s) for s in streamlines]),
                "mean_orientation": np.mean([mean_orientation(s) for s in streamlines]),
                "tracking_quality": self._compute_tracking_quality(streamlines),
            }
        }

        return metrics

    def _compute_tracking_confidence(self, streamlines: Streamlines) -> float:
        """Compute tracking confidence for particle filtering."""
        # Implement confidence metric based on streamline properties
        lengths = np.array([len(s) for s in streamlines])
        curvatures = np.array([mean_curvature(s) for s in streamlines])

        # Normalize metrics
        norm_lengths = (lengths - np.mean(lengths)) / np.std(lengths)
        norm_curvatures = (curvatures - np.mean(curvatures)) / np.std(curvatures)

        # Combine metrics (higher is better)
        confidence = np.mean(norm_lengths) - np.mean(norm_curvatures)
        return float(confidence)

    def _compute_tracking_quality(self, streamlines: Streamlines) -> float:
        """Compute tracking quality for EuDX."""
        # Implement quality metric based on streamline properties
        lengths = np.array([len(s) for s in streamlines])
        orientations = np.array([mean_orientation(s) for s in streamlines])

        # Normalize metrics
        norm_lengths = (lengths - np.mean(lengths)) / np.std(lengths)
        norm_orientations = (orientations - np.mean(orientations)) / np.std(
            orientations
        )

        # Combine metrics (higher is better)
        quality = np.mean(norm_lengths) + np.mean(norm_orientations)
        return float(quality)

    def compare_tracking_methods(
        self, tracking_results: dict[str, dict], output_dir: str | None = None
    ) -> dict[str, Any]:
        """
        Compare tracking results across different methods.

        Args:
            tracking_results: Dictionary of tracking results for each method
            output_dir: Optional directory to save comparison plots

        Returns:
            Dictionary containing comparison metrics and statistical tests
        """
        comparison = {
            "methods": list(tracking_results.keys()),
            "metrics": {},
            "statistical_tests": {},
            "cc_metrics": {},
        }

        # Compare basic metrics
        basic_metrics = [
            "n_streamlines",
            "mean_length",
            "std_length",
            "mean_curvature",
            "std_curvature",
            "mean_orientation",
            "std_orientation",
        ]
        for metric in basic_metrics:
            values = [r["basic_metrics"][metric] for r in tracking_results.values()]
            comparison["metrics"][metric] = {
                "values": values,
                "mean": np.mean(values),
                "std": np.std(values),
            }

            # Perform statistical tests
            if len(values) > 2:
                f_stat, p_value = f_oneway(*values)
                comparison["statistical_tests"][metric] = {
                    "test": "ANOVA",
                    "f_stat": f_stat,
                    "p_value": p_value,
                }
            else:
                t_stat, p_value = ttest_ind(values[0], values[1])
                comparison["statistical_tests"][metric] = {
                    "test": "t-test",
                    "t_stat": t_stat,
                    "p_value": p_value,
                }

        # Compare CC-specific metrics
        if self.cc_mask is not None:
            cc_metrics = [
                "cc_n_streamlines",
                "cc_fraction",
                "cc_mean_density",
                "cc_std_density",
                "cc_mean_curvature",
                "cc_std_curvature",
                "cc_mean_orientation",
                "cc_std_orientation",
                "cc_symmetry_score",
            ]

            for metric in cc_metrics:
                values = []
                for result in tracking_results.values():
                    if "cc_metrics" in result and metric in result["cc_metrics"]:
                        values.append(result["cc_metrics"][metric])
                    else:
                        values.append(None)

                # Only include metrics that are present in all results
                if all(v is not None for v in values):
                    comparison["cc_metrics"][metric] = {
                        "values": values,
                        "mean": np.mean(values),
                        "std": np.std(values),
                    }

                    # Perform statistical tests
                    if len(values) > 2:
                        f_stat, p_value = f_oneway(*values)
                        comparison["statistical_tests"][f"cc_{metric}"] = {
                            "test": "ANOVA",
                            "f_stat": f_stat,
                            "p_value": p_value,
                        }
                    else:
                        t_stat, p_value = ttest_ind(values[0], values[1])
                        comparison["statistical_tests"][f"cc_{metric}"] = {
                            "test": "t-test",
                            "t_stat": t_stat,
                            "p_value": p_value,
                        }

        # Compare algorithm-specific metrics
        algorithm_metrics = {
            "probabilistic": ["curvature_stats", "orientation_stats"],
            "particle": ["particle_stats", "tracking_confidence"],
            "eudx": ["eudx_stats", "tracking_quality"],
        }

        for method, metrics in algorithm_metrics.items():
            if method in tracking_results:
                for metric in metrics:
                    if metric in tracking_results[method]["algorithm_specific"]:
                        comparison["metrics"][f"{method}_{metric}"] = {
                            "values": [
                                tracking_results[method]["algorithm_specific"][metric]
                            ],
                            "mean": tracking_results[method]["algorithm_specific"][
                                metric
                            ],
                            "std": 0.0,
                        }

        # Generate comparison plots if output directory is provided
        if output_dir is not None:
            self._plot_tracking_comparison(tracking_results, output_dir)
            if self.cc_mask is not None:
                self._plot_cc_comparison(tracking_results, output_dir)

        return comparison

    def _plot_tracking_comparison(
        self, tracking_results: dict[str, dict], output_dir: str
    ) -> None:
        """Generate comparison plots for different tracking methods."""
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots

        # Create output directory
        os.makedirs(output_dir, exist_ok=True)

        # Create subplots for basic metrics
        fig = make_subplots(
            rows=2,
            cols=2,
            subplot_titles=(
                "Number of Streamlines",
                "Mean Length",
                "Standard Deviation of Length",
                "Mean Curvature",
            ),
        )

        # Add traces for each metric
        methods = list(tracking_results.keys())
        colors = px.colors.qualitative.Set1[: len(methods)]

        # Number of streamlines
        fig.add_trace(
            go.Bar(
                x=methods,
                y=[
                    r["basic_metrics"]["n_streamlines"]
                    for r in tracking_results.values()
                ],
                marker_color=colors,
            ),
            row=1,
            col=1,
        )

        # Mean length
        fig.add_trace(
            go.Bar(
                x=methods,
                y=[
                    r["basic_metrics"]["mean_length"] for r in tracking_results.values()
                ],
                marker_color=colors,
            ),
            row=1,
            col=2,
        )

        # Standard deviation of length
        fig.add_trace(
            go.Bar(
                x=methods,
                y=[r["basic_metrics"]["std_length"] for r in tracking_results.values()],
                marker_color=colors,
            ),
            row=2,
            col=1,
        )

        # Mean curvature
        fig.add_trace(
            go.Bar(
                x=methods,
                y=[
                    r["basic_metrics"]["mean_curvature"]
                    for r in tracking_results.values()
                ],
                marker_color=colors,
            ),
            row=2,
            col=2,
        )

        # Update layout
        fig.update_layout(
            title="Basic Tracking Metrics Comparison", showlegend=False, height=800
        )

        # Save plot
        fig.write_html(os.path.join(output_dir, "basic_metrics_comparison.html"))

        # Create subplots for algorithm-specific metrics
        fig = make_subplots(
            rows=2,
            cols=2,
            subplot_titles=(
                "Tracking Confidence (Particle)",
                "Tracking Quality (EuDX)",
                "Curvature Stats (Probabilistic)",
                "Orientation Stats (Probabilistic)",
            ),
        )

        # Add traces for algorithm-specific metrics
        for method, result in tracking_results.items():
            if method == "particle":
                fig.add_trace(
                    go.Bar(
                        x=[method],
                        y=[
                            result["algorithm_specific"]["particle_stats"][
                                "tracking_confidence"
                            ]
                        ],
                        marker_color=colors[methods.index(method)],
                    ),
                    row=1,
                    col=1,
                )
            elif method == "eudx":
                fig.add_trace(
                    go.Bar(
                        x=[method],
                        y=[
                            result["algorithm_specific"]["eudx_stats"][
                                "tracking_quality"
                            ]
                        ],
                        marker_color=colors[methods.index(method)],
                    ),
                    row=1,
                    col=2,
                )
            elif method == "probabilistic":
                fig.add_trace(
                    go.Bar(
                        x=[method],
                        y=[result["algorithm_specific"]["curvature_stats"]["mean"]],
                        marker_color=colors[methods.index(method)],
                    ),
                    row=2,
                    col=1,
                )
                fig.add_trace(
                    go.Bar(
                        x=[method],
                        y=[result["algorithm_specific"]["orientation_stats"]["mean"]],
                        marker_color=colors[methods.index(method)],
                    ),
                    row=2,
                    col=2,
                )

        # Update layout
        fig.update_layout(
            title="Algorithm-Specific Metrics Comparison", showlegend=False, height=800
        )

        # Save plot
        fig.write_html(os.path.join(output_dir, "algorithm_metrics_comparison.html"))

    def _plot_cc_comparison(
        self, tracking_results: dict[str, dict], output_dir: str
    ) -> None:
        """Generate comparison plots for corpus callosum metrics."""
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots

        # Create subplots for CC metrics
        fig = make_subplots(
            rows=2,
            cols=2,
            subplot_titles=(
                "CC Streamline Count",
                "CC Fraction",
                "CC Mean Density",
                "CC Symmetry Score",
            ),
        )

        # Add traces for each metric
        methods = list(tracking_results.keys())
        colors = px.colors.qualitative.Set1[: len(methods)]

        # CC Streamline Count
        fig.add_trace(
            go.Bar(
                x=methods,
                y=[
                    r["cc_metrics"]["cc_n_streamlines"]
                    for r in tracking_results.values()
                ],
                marker_color=colors,
            ),
            row=1,
            col=1,
        )

        # CC Fraction
        fig.add_trace(
            go.Bar(
                x=methods,
                y=[r["cc_metrics"]["cc_fraction"] for r in tracking_results.values()],
                marker_color=colors,
            ),
            row=1,
            col=2,
        )

        # CC Mean Density
        fig.add_trace(
            go.Bar(
                x=methods,
                y=[
                    r["cc_metrics"]["cc_mean_density"]
                    for r in tracking_results.values()
                ],
                marker_color=colors,
            ),
            row=2,
            col=1,
        )

        # CC Symmetry Score
        fig.add_trace(
            go.Bar(
                x=methods,
                y=[
                    r["cc_metrics"]["cc_symmetry_score"]
                    for r in tracking_results.values()
                ],
                marker_color=colors,
            ),
            row=2,
            col=2,
        )

        # Update layout
        fig.update_layout(
            title="Corpus Callosum Metrics Comparison", showlegend=False, height=800
        )

        # Save plot
        fig.write_html(os.path.join(output_dir, "cc_metrics_comparison.html"))

        # Create subplots for CC shape metrics
        fig = make_subplots(
            rows=2,
            cols=2,
            subplot_titles=(
                "CC Mean Curvature",
                "CC Std Curvature",
                "CC Mean Orientation",
                "CC Std Orientation",
            ),
        )

        # Add traces for shape metrics
        fig.add_trace(
            go.Bar(
                x=methods,
                y=[
                    r["cc_metrics"]["cc_mean_curvature"]
                    for r in tracking_results.values()
                ],
                marker_color=colors,
            ),
            row=1,
            col=1,
        )

        fig.add_trace(
            go.Bar(
                x=methods,
                y=[
                    r["cc_metrics"]["cc_std_curvature"]
                    for r in tracking_results.values()
                ],
                marker_color=colors,
            ),
            row=1,
            col=2,
        )

        fig.add_trace(
            go.Bar(
                x=methods,
                y=[
                    r["cc_metrics"]["cc_mean_orientation"]
                    for r in tracking_results.values()
                ],
                marker_color=colors,
            ),
            row=2,
            col=1,
        )

        fig.add_trace(
            go.Bar(
                x=methods,
                y=[
                    r["cc_metrics"]["cc_std_orientation"]
                    for r in tracking_results.values()
                ],
                marker_color=colors,
            ),
            row=2,
            col=2,
        )

        # Update layout
        fig.update_layout(
            title="Corpus Callosum Shape Metrics Comparison",
            showlegend=False,
            height=800,
        )

        # Save plot
        fig.write_html(os.path.join(output_dir, "cc_shape_metrics_comparison.html"))

    def compare_with_reference(
        self,
        tracking_results: dict[str, dict],
        reference_data: dict[str, Any],
        output_dir: str | None = None,
    ) -> dict[str, Any]:
        """
        Compare tracking results against reference CC data from atlas.

        Args:
            tracking_results: Dictionary of tracking results for each method
            reference_data: Dictionary containing reference CC data from atlas
            output_dir: Optional directory to save comparison plots

        Returns:
            Dictionary containing comparison metrics and statistical tests
        """
        comparison = {
            "methods": list(tracking_results.keys()),
            "reference_metrics": {},
            "similarity_scores": {},
            "statistical_tests": {},
        }

        # Extract reference metrics
        ref_metrics = self._extract_reference_metrics(reference_data)
        comparison["reference_metrics"] = ref_metrics

        # Compare each tracking method with reference
        for method, result in tracking_results.items():
            if "cc_metrics" in result:
                # Compute similarity scores
                similarity = self._compute_similarity_scores(
                    result["cc_metrics"], ref_metrics
                )
                comparison["similarity_scores"][method] = similarity

                # Perform statistical tests
                stats = self._compare_with_reference_stats(
                    result["cc_metrics"], ref_metrics
                )
                comparison["statistical_tests"][method] = stats

        # Generate comparison plots if output directory is provided
        if output_dir is not None:
            self._plot_reference_comparison(comparison, output_dir)

        return comparison

    def _extract_reference_metrics(
        self, reference_data: dict[str, Any]
    ) -> dict[str, Any]:
        """Extract metrics from reference CC data."""
        metrics = {}

        # Basic metrics
        if "streamlines" in reference_data:
            metrics["n_streamlines"] = len(reference_data["streamlines"])
            metrics["mean_length"] = np.mean(
                [len(s) for s in reference_data["streamlines"]]
            )
            metrics["std_length"] = np.std(
                [len(s) for s in reference_data["streamlines"]]
            )

        # Shape metrics
        if "curvature" in reference_data:
            metrics["mean_curvature"] = np.mean(reference_data["curvature"])
            metrics["std_curvature"] = np.std(reference_data["curvature"])

        if "orientation" in reference_data:
            metrics["mean_orientation"] = np.mean(reference_data["orientation"])
            metrics["std_orientation"] = np.std(reference_data["orientation"])

        # Density metrics
        if "density" in reference_data:
            metrics["mean_density"] = np.mean(
                reference_data["density"][reference_data["density"] > 0]
            )
            metrics["std_density"] = np.std(
                reference_data["density"][reference_data["density"] > 0]
            )

        # Symmetry metrics
        if "symmetry" in reference_data:
            metrics["symmetry_score"] = reference_data["symmetry"]

        return metrics

    def _compute_similarity_scores(
        self, cc_metrics: dict[str, Any], ref_metrics: dict[str, Any]
    ) -> dict[str, float]:
        """Compute similarity scores between tracking results and reference data."""
        similarity = {}

        # Streamline count similarity
        if "cc_n_streamlines" in cc_metrics and "n_streamlines" in ref_metrics:
            similarity["count_similarity"] = 1 - abs(
                cc_metrics["cc_n_streamlines"] - ref_metrics["n_streamlines"]
            ) / max(cc_metrics["cc_n_streamlines"], ref_metrics["n_streamlines"])

        # Length similarity
        if "cc_mean_length" in cc_metrics and "mean_length" in ref_metrics:
            similarity["length_similarity"] = 1 - abs(
                cc_metrics["cc_mean_length"] - ref_metrics["mean_length"]
            ) / max(cc_metrics["cc_mean_length"], ref_metrics["mean_length"])

        # Curvature similarity
        if "cc_mean_curvature" in cc_metrics and "mean_curvature" in ref_metrics:
            similarity["curvature_similarity"] = 1 - abs(
                cc_metrics["cc_mean_curvature"] - ref_metrics["mean_curvature"]
            ) / max(cc_metrics["cc_mean_curvature"], ref_metrics["mean_curvature"])

        # Orientation similarity
        if "cc_mean_orientation" in cc_metrics and "mean_orientation" in ref_metrics:
            similarity["orientation_similarity"] = 1 - abs(
                cc_metrics["cc_mean_orientation"] - ref_metrics["mean_orientation"]
            ) / max(cc_metrics["cc_mean_orientation"], ref_metrics["mean_orientation"])

        # Density similarity
        if "cc_mean_density" in cc_metrics and "mean_density" in ref_metrics:
            similarity["density_similarity"] = 1 - abs(
                cc_metrics["cc_mean_density"] - ref_metrics["mean_density"]
            ) / max(cc_metrics["cc_mean_density"], ref_metrics["mean_density"])

        # Symmetry similarity
        if "cc_symmetry_score" in cc_metrics and "symmetry_score" in ref_metrics:
            similarity["symmetry_similarity"] = 1 - abs(
                cc_metrics["cc_symmetry_score"] - ref_metrics["symmetry_score"]
            )

        # Overall similarity score
        if similarity:
            similarity["overall_similarity"] = np.mean(list(similarity.values()))

        return similarity

    def _compare_with_reference_stats(
        self, cc_metrics: dict[str, Any], ref_metrics: dict[str, Any]
    ) -> dict[str, Any]:
        """Perform statistical tests comparing tracking results with reference data."""
        stats = {}

        # Length comparison
        if "cc_mean_length" in cc_metrics and "mean_length" in ref_metrics:
            t_stat, p_value = ttest_ind(
                [cc_metrics["cc_mean_length"]], [ref_metrics["mean_length"]]
            )
            stats["length_comparison"] = {
                "test": "t-test",
                "t_stat": t_stat,
                "p_value": p_value,
            }

        # Curvature comparison
        if "cc_mean_curvature" in cc_metrics and "mean_curvature" in ref_metrics:
            t_stat, p_value = ttest_ind(
                [cc_metrics["cc_mean_curvature"]], [ref_metrics["mean_curvature"]]
            )
            stats["curvature_comparison"] = {
                "test": "t-test",
                "t_stat": t_stat,
                "p_value": p_value,
            }

        # Orientation comparison
        if "cc_mean_orientation" in cc_metrics and "mean_orientation" in ref_metrics:
            t_stat, p_value = ttest_ind(
                [cc_metrics["cc_mean_orientation"]], [ref_metrics["mean_orientation"]]
            )
            stats["orientation_comparison"] = {
                "test": "t-test",
                "t_stat": t_stat,
                "p_value": p_value,
            }

        return stats

    def _plot_reference_comparison(
        self, comparison: dict[str, Any], output_dir: str
    ) -> None:
        """Generate comparison plots against reference data."""
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots

        # Create output directory
        os.makedirs(output_dir, exist_ok=True)

        # Create subplots for similarity scores
        fig = make_subplots(
            rows=2,
            cols=2,
            subplot_titles=(
                "Count Similarity",
                "Length Similarity",
                "Curvature Similarity",
                "Overall Similarity",
            ),
        )

        # Add traces for each similarity score
        methods = comparison["methods"]
        colors = px.colors.qualitative.Set1[: len(methods)]

        # Count similarity
        fig.add_trace(
            go.Bar(
                x=methods,
                y=[
                    comparison["similarity_scores"][m]["count_similarity"]
                    for m in methods
                ],
                marker_color=colors,
            ),
            row=1,
            col=1,
        )

        # Length similarity
        fig.add_trace(
            go.Bar(
                x=methods,
                y=[
                    comparison["similarity_scores"][m]["length_similarity"]
                    for m in methods
                ],
                marker_color=colors,
            ),
            row=1,
            col=2,
        )

        # Curvature similarity
        fig.add_trace(
            go.Bar(
                x=methods,
                y=[
                    comparison["similarity_scores"][m]["curvature_similarity"]
                    for m in methods
                ],
                marker_color=colors,
            ),
            row=2,
            col=1,
        )

        # Overall similarity
        fig.add_trace(
            go.Bar(
                x=methods,
                y=[
                    comparison["similarity_scores"][m]["overall_similarity"]
                    for m in methods
                ],
                marker_color=colors,
            ),
            row=2,
            col=2,
        )

        # Update layout
        fig.update_layout(
            title="Similarity Scores with Reference Data", showlegend=False, height=800
        )

        # Save plot
        fig.write_html(os.path.join(output_dir, "reference_similarity_comparison.html"))

        # Create subplots for statistical comparisons
        fig = make_subplots(
            rows=2,
            cols=2,
            subplot_titles=(
                "Length Comparison",
                "Curvature Comparison",
                "Orientation Comparison",
                "Statistical Summary",
            ),
        )

        # Add traces for statistical comparisons
        for i, (method, stat) in enumerate(comparison["statistical_tests"].items()):
            # Length comparison
            fig.add_trace(
                go.Bar(
                    x=[method],
                    y=[stat["length_comparison"]["p_value"]],
                    marker_color=colors[i],
                ),
                row=1,
                col=1,
            )

            # Curvature comparison
            fig.add_trace(
                go.Bar(
                    x=[method],
                    y=[stat["curvature_comparison"]["p_value"]],
                    marker_color=colors[i],
                ),
                row=1,
                col=2,
            )

            # Orientation comparison
            fig.add_trace(
                go.Bar(
                    x=[method],
                    y=[stat["orientation_comparison"]["p_value"]],
                    marker_color=colors[i],
                ),
                row=2,
                col=1,
            )

        # Add reference metrics
        ref_metrics = comparison["reference_metrics"]
        fig.add_trace(
            go.Table(
                header=dict(values=["Metric", "Reference Value"]),
                cells=dict(
                    values=[
                        list(ref_metrics.keys()),
                        [f"{v:.4f}" for v in ref_metrics.values()],
                    ]
                ),
            ),
            row=2,
            col=2,
        )

        # Update layout
        fig.update_layout(
            title="Statistical Comparison with Reference Data",
            showlegend=False,
            height=800,
        )

        # Save plot
        fig.write_html(
            os.path.join(output_dir, "reference_statistical_comparison.html")
        )

    def optimize_cmc_parameters(
        self,
        streamlines: Streamlines,
        tissue_maps: dict[str, np.ndarray],
        param_grid: dict[str, list[Any]],
        n_iter: int = 10,
        output_dir: str | None = None,
    ) -> dict[str, Any]:
        """Optimize CMC parameters using grid search.

        Args:
            streamlines: Generated streamlines
            tissue_maps: Dictionary containing tissue probability maps
            param_grid: Dictionary of parameters to optimize with their possible values
            n_iter: Number of iterations for random search
            output_dir: Optional directory to save optimization results

        Returns:
            Dictionary containing optimization results
        """
        # Create parameter combinations
        param_combinations = list(ParameterGrid(param_grid))
        if len(param_combinations) > n_iter:
            param_combinations = np.random.choice(
                param_combinations, n_iter, replace=False
            )

        # Initialize results
        results = {"params": [], "metrics": [], "quality_scores": []}

        # Evaluate each parameter combination
        for params in param_combinations:
            # Compute metrics with current parameters
            metrics = self._compute_cmc_metrics(streamlines, tissue_maps)
            quality_score = metrics["quality_score"]

            # Store results
            results["params"].append(params)
            results["metrics"].append(metrics)
            results["quality_scores"].append(quality_score)

        # Find best parameters
        best_idx = np.argmax(results["quality_scores"])
        best_params = results["params"][best_idx]
        best_metrics = results["metrics"][best_idx]

        # Create optimization summary
        optimization_results = {
            "best_params": best_params,
            "best_quality_score": results["quality_scores"][best_idx],
            "best_metrics": best_metrics,
            "all_results": results,
        }

        # Generate visualizations if output directory is provided
        if output_dir is not None:
            self._visualize_optimization_results(optimization_results, output_dir)

        return optimization_results

    def _visualize_optimization_results(
        self, results: dict[str, Any], output_dir: str
    ) -> None:
        """Generate visualizations for optimization results."""
        import plotly.graph_objects as go

        # Create output directory
        os.makedirs(output_dir, exist_ok=True)

        # Convert results to DataFrame for easier plotting
        df = pd.DataFrame(
            {
                "quality_score": results["all_results"]["quality_scores"],
                **{
                    k: [p[k] for p in results["all_results"]["params"]]
                    for k in results["all_results"]["params"][0].keys()
                },
            }
        )

        # Create parameter importance plot
        param_importance = self._compute_parameter_importance(df)

        fig = go.Figure()
        fig.add_trace(
            go.Bar(
                x=list(param_importance.keys()),
                y=list(param_importance.values()),
                name="Parameter Importance",
            )
        )

        fig.update_layout(
            title="Parameter Importance",
            xaxis_title="Parameter",
            yaxis_title="Importance Score",
            showlegend=True,
        )

        fig.write_html(os.path.join(output_dir, "parameter_importance.html"))

        # Create optimization history plot
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=list(range(len(df))),
                y=df["quality_score"],
                mode="lines+markers",
                name="Quality Score",
            )
        )

        fig.update_layout(
            title="Optimization History",
            xaxis_title="Iteration",
            yaxis_title="Quality Score",
            showlegend=True,
        )

        fig.write_html(os.path.join(output_dir, "optimization_history.html"))

        # Create parameter correlation plot
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 1:
            corr_matrix = df[numeric_cols].corr()

            fig = go.Figure(
                data=go.Heatmap(
                    z=corr_matrix,
                    x=corr_matrix.columns,
                    y=corr_matrix.columns,
                    colorscale="RdBu",
                    zmid=0,
                    text=np.round(corr_matrix, 2),
                    texttemplate="%{text}",
                    textfont={"size": 10},
                )
            )

            fig.update_layout(title="Parameter Correlations", height=800, width=800)

            fig.write_html(os.path.join(output_dir, "parameter_correlations.html"))

        # Create parallel coordinates plot
        fig = px.parallel_coordinates(
            df,
            color="quality_score",
            dimensions=df.columns,
            color_continuous_scale=px.colors.sequential.Viridis,
            title="Parameter Space Exploration",
        )

        fig.write_html(os.path.join(output_dir, "parameter_space.html"))

    def _compute_parameter_importance(self, df: pd.DataFrame) -> dict[str, float]:
        """Compute parameter importance scores using multiple statistical methods in parallel."""
        param_names = [col for col in df.columns if col != "score"]
        importance_scores = {}

        def compute_correlation(param, df):
            return abs(df[param].corr(df["score"]))

        def compute_mutual_info(params, df):
            from sklearn.feature_selection import mutual_info_regression

            return mutual_info_regression(df[params], df["score"])

        def compute_rf_importance(params, df):
            rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
            rf.fit(df[params], df["score"])
            return rf.feature_importances_

        # Parallel computation of correlations
        with ProcessPoolExecutor() as executor:
            correlation_futures = {
                param: executor.submit(compute_correlation, param, df)
                for param in param_names
            }
            for param, future in correlation_futures.items():
                importance_scores[f"{param}_correlation"] = future.result()

        # Parallel computation of mutual information
        with ProcessPoolExecutor() as executor:
            mi_future = executor.submit(compute_mutual_info, param_names, df)
            mi_scores = mi_future.result()
            for param, score in zip(param_names, mi_scores, strict=False):
                importance_scores[f"{param}_mutual_info"] = score

        # Parallel computation of RF importance
        with ProcessPoolExecutor() as executor:
            rf_future = executor.submit(compute_rf_importance, param_names, df)
            rf_scores = rf_future.result()
            for param, score in zip(param_names, rf_scores, strict=False):
                importance_scores[f"{param}_rf_importance"] = score

        return importance_scores

    def _generate_optimization_recommendations(
        self,
        comparison: dict[str, Any],
        optimization_results: dict[str, dict[str, Any]],
    ) -> dict[str, Any]:
        """Generate optimization recommendations based on comparison results."""
        from concurrent.futures import ProcessPoolExecutor

        recommendations = {
            "best_method": None,
            "best_parameters": {},
            "parameter_ranges": {},
            "method_specific_recommendations": {},
            "pipeline_recommendations": [],
        }

        def process_method(method, results):
            method_data = {
                "score": results["best_score"],
                "params": results["best_params"],
                "ranges": {},
                "recommendations": [],
            }

            # Compute parameter ranges
            for param in results["best_params"].keys():
                values = [trial["params"][param] for trial in results["history"]]
                method_data["ranges"][param] = {
                    "min": min(values),
                    "max": max(values),
                    "optimal": results["best_params"][param],
                    "std": np.std(values),
                }

            # Compute parameter importance
            param_importance = self._compute_parameter_importance(
                pd.DataFrame([trial["params"] for trial in results["history"]])
            )

            # Generate method-specific recommendations
            critical_params = []
            for param in results["best_params"].keys():
                importance = max(
                    param_importance.get(f"{param}_correlation", 0),
                    param_importance.get(f"{param}_mutual_info", 0),
                    param_importance.get(f"{param}_rf_importance", 0),
                )
                if importance > 0.5:
                    critical_params.append(param)

            for param in critical_params:
                param_range = method_data["ranges"][param]
                if param_range["std"] < 0.1 * (param_range["max"] - param_range["min"]):
                    method_data["recommendations"].append(
                        f"Parameter {param} is stable around {param_range['optimal']:.2f}"
                    )
                else:
                    method_data["recommendations"].append(
                        f"Parameter {param} is sensitive, recommended range: "
                        f"{param_range['min']:.2f} - {param_range['max']:.2f}"
                    )

            return method, method_data

        # Process methods in parallel
        with ProcessPoolExecutor() as executor:
            method_futures = {
                method: executor.submit(process_method, method, results)
                for method, results in optimization_results.items()
            }

            for method, future in method_futures.items():
                method, method_data = future.result()
                if method_data["score"] > recommendations.get("best_score", -np.inf):
                    recommendations["best_method"] = method
                    recommendations["best_parameters"] = method_data["params"]
                    recommendations["best_score"] = method_data["score"]
                recommendations["parameter_ranges"][method] = method_data["ranges"]
                recommendations["method_specific_recommendations"][method] = (
                    method_data["recommendations"]
                )

        # Generate pipeline-level recommendations
        def analyze_convergence(method, results):
            scores = [trial["score"] for trial in results["history"]]
            if len(scores) > 10:
                convergence_rate = np.mean(np.diff(scores[-5:]))
                if convergence_rate < 0.01:
                    return f"Method {method} shows good convergence, consider reducing optimization iterations"
                elif convergence_rate > 0.1:
                    return f"Method {method} may need more optimization iterations"
            return None

        def analyze_interactions(method, results):
            param_importance = self._compute_parameter_importance(
                pd.DataFrame([trial["params"] for trial in results["history"]])
            )
            high_importance_params = [
                param
                for param in results["best_params"].keys()
                if any(
                    score > 0.7
                    for key, score in param_importance.items()
                    if key.startswith(param)
                )
            ]
            if len(high_importance_params) > 1:
                return f"Method {method} shows strong parameter interactions between {', '.join(high_importance_params)}"
            return None

        # Process convergence and interactions in parallel
        with ProcessPoolExecutor() as executor:
            convergence_futures = {
                method: executor.submit(analyze_convergence, method, results)
                for method, results in optimization_results.items()
            }
            interaction_futures = {
                method: executor.submit(analyze_interactions, method, results)
                for method, results in optimization_results.items()
            }

            for future in convergence_futures.values():
                rec = future.result()
                if rec:
                    recommendations["pipeline_recommendations"].append(rec)

            for future in interaction_futures.values():
                rec = future.result()
                if rec:
                    recommendations["pipeline_recommendations"].append(rec)

        return recommendations

    def _visualize_optimization_comparison(
        self,
        comparison: dict[str, Any],
        optimization_results: dict[str, dict[str, Any]],
        output_dir: str,
    ) -> None:
        """Generate visualizations for optimization comparison results."""
        from concurrent.futures import ProcessPoolExecutor

        import plotly.io as pio

        def generate_parameter_importance_plot(method, results):
            param_importance = self._compute_parameter_importance(
                pd.DataFrame([trial["params"] for trial in results["history"]])
            )

            fig = go.Figure()
            for param in results["best_params"].keys():
                fig.add_trace(
                    go.Bar(
                        name=param,
                        x=["Correlation", "Mutual Info", "RF Importance"],
                        y=[
                            param_importance.get(f"{param}_correlation", 0),
                            param_importance.get(f"{param}_mutual_info", 0),
                            param_importance.get(f"{param}_rf_importance", 0),
                        ],
                    )
                )

            fig.update_layout(
                title=f"Parameter Importance - {method}",
                barmode="group",
                xaxis_title="Importance Metric",
                yaxis_title="Importance Score",
            )

            # Save plot to memory instead of file
            return method, pio.to_json(fig)

        def generate_recommendations_table(recommendations):
            fig = go.Figure()
            fig.add_trace(
                go.Table(
                    header=dict(values=["Category", "Recommendation"]),
                    cells=dict(
                        values=[
                            ["Method"]
                            * len(
                                recommendations["method_specific_recommendations"][
                                    recommendations["best_method"]
                                ]
                            )
                            + ["Pipeline"]
                            * len(recommendations["pipeline_recommendations"]),
                            recommendations["method_specific_recommendations"][
                                recommendations["best_method"]
                            ]
                            + recommendations["pipeline_recommendations"],
                        ]
                    ),
                )
            )
            fig.update_layout(title="Optimization Recommendations")
            return pio.to_json(fig)

        # Generate plots in parallel
        with ProcessPoolExecutor() as executor:
            plot_futures = {
                method: executor.submit(
                    generate_parameter_importance_plot, method, results
                )
                for method, results in optimization_results.items()
            }

            # Get recommendations
            recommendations = self._generate_optimization_recommendations(
                comparison, optimization_results
            )

            # Generate recommendations table
            rec_table_future = executor.submit(
                generate_recommendations_table, recommendations
            )

            # Save all plots
            for method, future in plot_futures.items():
                method, plot_json = future.result()
                with open(
                    os.path.join(output_dir, f"parameter_importance_{method}.html"), "w"
                ) as f:
                    f.write(plot_json)

            rec_table_json = rec_table_future.result()
            with open(
                os.path.join(output_dir, "optimization_recommendations.html"), "w"
            ) as f:
                f.write(rec_table_json)
