import gc
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import imageio
import numpy as np
import vtk
from dipy.segment.clustering import QuickBundles
from dipy.segment.featurespeed import ResampleFeature
from dipy.segment.metricspeed import AveragePointwiseEuclideanMetric
from dipy.tracking.streamline import transform_streamlines
from dipy.viz import actor, window
from dipy.viz import colormap as cmap
from scipy.spatial.transform import Rotation as R

# GIF compression settings
gif_size = (608, 608)
gif_duration = 0.2
gif_palette_size = 64


class TractographyVisualizer:
    """A class for visualizing tractography results with interactive 3D views and comparisons."""

    def __init__(
        self,
        output_dir: str,
        template_dir: str | None = None,
        num_workers: int | None = None,
    ):
        """
        Initialize the visualizer.

        Args:
            output_dir: Directory to save visualizations
            template_dir: Directory containing HTML templates (optional)
            num_workers: Number of parallel workers for processing (default: cpu_count - 1)
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.template_dir = (
            Path(template_dir) if template_dir else Path(__file__).parent
        )

        # Initialize visualization settings
        self.gif_size = (608, 608)
        self.gif_duration = 0.2
        self.gif_palette_size = 64

        # Parallel processing settings
        self.num_workers = (
            num_workers if num_workers is not None else max(1, os.cpu_count() - 1)
        )

        # Initialize QuickBundles for centroid calculation
        feature = ResampleFeature(nb_points=100)
        metric = AveragePointwiseEuclideanMetric(feature)
        self.qb = QuickBundles(np.inf, metric=metric)

        # Initialize pial surface cache
        self._pial_surface = None
        self._pial_surface_loaded = False

    def _get_pial_surface(self):
        """Load and prepare the pial surface for visualization with manual caching."""
        if not self._pial_surface_loaded:
            from nilearn.datasets import load_fsaverage

            fsaverage_meshes = load_fsaverage()
            pial_l = fsaverage_meshes["pial"].parts["left"]
            pial_r = fsaverage_meshes["pial"].parts["right"]

            faces_r_shifted = pial_r.faces + len(pial_l.coordinates)
            vertices = np.vstack((pial_l.coordinates, pial_r.coordinates))
            faces = np.vstack((pial_l.faces, faces_r_shifted))

            colors = np.zeros((vertices.shape[0], 3))
            self._pial_surface = actor.surface(vertices, faces=faces, colors=colors)
            self._pial_surface.GetProperty().SetOpacity(0.06)
            self._pial_surface_loaded = True

            # Clean up references to free memory
            del fsaverage_meshes
            del pial_l
            del pial_r
            gc.collect()

        return self._pial_surface

    def _process_streamline_frame(
        self,
        angle: float,
        streamlines: list[np.ndarray],
        colors: np.ndarray,
        rotation_center: np.ndarray,
        rotation_axis: np.ndarray,
        camera_position: np.ndarray,
        camera_up: np.ndarray,
    ) -> np.ndarray:
        """Process a single frame of the animation."""
        rot_matrix = R.from_rotvec(angle * np.pi / 180 * rotation_axis).as_matrix()
        rotated_streamlines = [
            np.dot(s - rotation_center, rot_matrix.T) + rotation_center
            for s in streamlines
        ]

        stream_actor = actor.line(rotated_streamlines, colors=colors)

        transform_matrix = np.eye(4)
        transform_matrix[:3, :3] = rot_matrix
        transform_matrix[:3, 3] = rotation_center - np.dot(rot_matrix, rotation_center)

        transform = vtk.vtkTransform()
        transform.Concatenate(transform_matrix.flatten())
        self._get_pial_surface().SetUserTransform(transform)

        scene = window.Scene()
        scene.SetBackground(1, 1, 1)
        scene.reset_clipping_range()
        scene.set_camera(
            position=camera_position, focal_point=rotation_center, view_up=camera_up
        )
        scene.add(stream_actor)
        scene.add(self._get_pial_surface())

        frame = window.snapshot(scene, size=self.gif_size)

        # Clean up to prevent memory leaks
        del stream_actor
        del scene
        gc.collect()

        return frame

    def generate_streamline_animation(
        self,
        streamlines: list[np.ndarray],
        name: str,
        colors: np.ndarray | None = None,
        output_format: str = "mp4",
    ) -> str:
        """
        Generate an animated visualization of streamlines with parallel processing.

        Args:
            streamlines: List of streamline arrays
            name: Name for the output file
            colors: Optional colors for the streamlines
            output_format: Output format ('gif' or 'mp4')

        Returns:
            Path to the generated animation file
        """
        if colors is None:
            colors = cmap.line_colors(streamlines)

        # Define flip transformation
        flip_matrix = np.eye(4)
        flip_matrix[0, 0] = -1
        flipped_streamlines = transform_streamlines(streamlines, flip_matrix)

        # Setup camera
        rotation_center = np.array([0, 0, 0])
        brain_bounds = self._get_pial_surface().GetMapper().GetInput().GetBounds()
        brain_center = np.array(
            [
                (brain_bounds[0] + brain_bounds[1]) / 2,
                (brain_bounds[2] + brain_bounds[3]) / 2,
                (brain_bounds[4] + brain_bounds[5]) / 2,
            ]
        )
        brain_size = np.linalg.norm(
            [
                brain_bounds[1] - brain_bounds[0],
                brain_bounds[3] - brain_bounds[2],
                brain_bounds[5] - brain_bounds[4],
            ]
        )

        camera_distance = 2.0 * brain_size
        camera_position = brain_center + camera_distance * np.array([0, 1, 0])
        camera_up = np.array([0, 0, 1])
        rotation_axis = np.array([0, 0, 1])

        # Generate frames in parallel
        angles = np.linspace(0, 360, 60, endpoint=False)
        gif_frames = []

        with ProcessPoolExecutor(max_workers=self.num_workers) as executor:
            futures = {
                executor.submit(
                    self._process_streamline_frame,
                    angle,
                    flipped_streamlines,
                    colors,
                    rotation_center,
                    rotation_axis,
                    camera_position,
                    camera_up,
                ): angle
                for angle in angles
            }

            for future in as_completed(futures):
                frame = future.result()
                gif_frames.append(frame)

        # Sort frames by angle
        gif_frames = [
            frame for _, frame in sorted(zip(angles, gif_frames, strict=False))
        ]

        # Save animation
        gif_path = self.output_dir / f"{name}.gif"
        imageio.mimsave(
            gif_path,
            gif_frames,
            duration=self.gif_duration,
            palettesize=self.gif_palette_size,
        )

        if output_format == "mp4":
            mp4_path = self.output_dir / f"{name}.mp4"
            self._convert_gif_to_mp4(gif_path, mp4_path)
            os.remove(gif_path)
            return str(mp4_path)

        return str(gif_path)

    def _convert_gif_to_mp4(self, gif_path: str, mp4_path: str):
        """Convert GIF to MP4 using FFmpeg with optimized settings."""
        reader = imageio.get_reader(gif_path)
        writer = imageio.get_writer(
            mp4_path,
            format="FFMPEG",
            fps=10,
            codec="libx264",
            quality=10,
            macro_block_size=8,
        )

        for frame in reader:
            writer.append_data(frame)

        writer.close()

    def generate_comparison_report(
        self,
        subject_data: dict[str, dict[str, str]],
        atlas_data: dict[str, str],
        metrics_data: dict[str, dict] | None = None,
    ) -> None:
        """
        Generate an HTML report comparing subject and atlas tractography.

        Args:
            subject_data: Dictionary mapping subject IDs to their tract data
            atlas_data: Dictionary mapping tract names to atlas paths
            metrics_data: Optional dictionary containing metrics for each subject/tract
        """
        # Load templates
        with open(self.template_dir / "subject_template.html") as f:
            subject_template = f.read()

        with open(self.template_dir / "index_template.html") as f:
            index_template = f.read()

        # Generate subject pages in parallel
        def process_subject(
            subject_id: str, tracts: dict[str, str]
        ) -> tuple[str, int, float]:
            comparison_rows = []
            tract_count = 0
            quality_score = 0

            for tract_name, tract_path in tracts.items():
                atlas_path = atlas_data.get(tract_name)
                if atlas_path:
                    tract_count += 1

                    # Get metrics for this tract if available
                    tract_metrics = (
                        metrics_data.get(subject_id, {}).get(tract_name, {})
                        if metrics_data
                        else {}
                    )
                    similarity_score = tract_metrics.get("similarity_score", 0)
                    bundle_volume = tract_metrics.get("bundle_volume", 0)
                    mean_length = tract_metrics.get("mean_length", 0)
                    curvature = tract_metrics.get("curvature", 0)

                    quality_score += similarity_score

                    metrics_html = f"""
                        <div class="small text-muted">
                            Similarity: {similarity_score:.1f}%<br>
                            Volume: {bundle_volume:.0f} mm³<br>
                            Length: {mean_length:.1f} mm<br>
                            Curvature: {curvature:.1f}°
                        </div>
                    """

                    comparison_rows.append(
                        f"<tr class='tract-row'>"
                        f"<td>{tract_name}</td>"
                        f"<td><video class='tract-video' controls><source src='{tract_path}' type='video/mp4'></video></td>"
                        f"<td><video class='tract-video' controls><source src='{atlas_path}' type='video/mp4'></video></td>"
                        f"<td>{metrics_html}</td>"
                        f"</tr>"
                    )

            # Generate subject page
            subject_html = subject_template.replace("{{SUBJECT_ID}}", subject_id)
            subject_html = subject_html.replace(
                "{{COMPARISON_ROWS}}", "\n".join(comparison_rows)
            )

            with open(self.output_dir / f"{subject_id}.html", "w") as f:
                f.write(subject_html)

            return subject_id, tract_count, quality_score

        # Process subjects in parallel
        with ProcessPoolExecutor(max_workers=self.num_workers) as executor:
            futures = {
                executor.submit(process_subject, subject_id, tracts): subject_id
                for subject_id, tracts in subject_data.items()
            }

            total_tract_count = 0
            total_quality_score = 0
            subject_links = []

            for future in as_completed(futures):
                subject_id, tract_count, quality_score = future.result()
                total_tract_count += tract_count
                total_quality_score += quality_score
                subject_links.append(
                    f"<a href='{subject_id}.html' class='subject-item'>{subject_id}</a>"
                )

        # Generate index page
        avg_quality_score = total_quality_score / max(total_tract_count, 1)
        index_html = index_template.replace(
            "{{SUBJECT_LINKS}}", "\n".join(subject_links)
        )
        index_html = index_html.replace("{{TOTAL_SUBJECTS}}", str(len(subject_data)))
        index_html = index_html.replace("{{TOTAL_TRACT_COUNT}}", str(total_tract_count))
        index_html = index_html.replace(
            "{{AVG_QUALITY_SCORE}}", f"{avg_quality_score:.1f}"
        )

        with open(self.output_dir / "index.html", "w") as f:
            f.write(index_html)

    def visualize_metrics(
        self, metrics: dict[str, dict], output_file: str = "metrics_report.html"
    ) -> None:
        """
        Generate an interactive visualization of tractography metrics.

        Args:
            metrics: Dictionary containing metrics data
            output_file: Name of the output HTML file
        """
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots

        # Create figure with subplots
        fig = make_subplots(
            rows=2,
            cols=2,
            subplot_titles=(
                "Similarity Scores",
                "Bundle Volumes",
                "Mean Lengths",
                "Curvature Distribution",
            ),
        )

        # Extract data for plotting
        subjects = []
        similarity_scores = []
        bundle_volumes = []
        mean_lengths = []
        curvatures = []

        for subject_id, subject_metrics in metrics.items():
            for tract_name, tract_metrics in subject_metrics.items():
                subjects.append(f"{subject_id}_{tract_name}")
                similarity_scores.append(tract_metrics.get("similarity_score", 0))
                bundle_volumes.append(tract_metrics.get("bundle_volume", 0))
                mean_lengths.append(tract_metrics.get("mean_length", 0))
                curvatures.append(tract_metrics.get("curvature", 0))

        # Add traces
        fig.add_trace(go.Box(y=similarity_scores, name="Similarity"), row=1, col=1)

        fig.add_trace(go.Box(y=bundle_volumes, name="Volume"), row=1, col=2)

        fig.add_trace(go.Box(y=mean_lengths, name="Length"), row=2, col=1)

        fig.add_trace(go.Box(y=curvatures, name="Curvature"), row=2, col=2)

        # Update layout
        fig.update_layout(
            height=800, showlegend=False, title_text="Tractography Metrics Distribution"
        )

        # Save as HTML
        fig.write_html(str(self.output_dir / output_file))

    def generate_quality_report(
        self, metrics: dict[str, dict], output_file: str = "quality_report.html"
    ) -> None:
        """
        Generate a detailed quality report with statistical analysis.

        Args:
            metrics: Dictionary containing metrics data
            output_file: Name of the output HTML file
        """
        import pandas as pd
        import plotly.express as px

        # Convert metrics to DataFrame
        data = []
        for subject_id, subject_metrics in metrics.items():
            for tract_name, tract_metrics in subject_metrics.items():
                data.append(
                    {"subject": subject_id, "tract": tract_name, **tract_metrics}
                )

        df = pd.DataFrame(data)

        # Create correlation matrix
        corr_matrix = df[
            ["similarity_score", "bundle_volume", "mean_length", "curvature"]
        ].corr()

        # Create figures
        fig_corr = px.imshow(
            corr_matrix,
            title="Metrics Correlation Matrix",
            color_continuous_scale="RdBu",
        )

        fig_similarity = px.box(
            df, x="tract", y="similarity_score", title="Similarity Scores by Tract"
        )

        fig_volume = px.scatter(
            df,
            x="mean_length",
            y="bundle_volume",
            color="tract",
            title="Volume vs Length by Tract",
        )

        # Save as HTML
        with open(self.output_dir / output_file, "w") as f:
            f.write(
                """
            <!DOCTYPE html>
            <html>
            <head>
                <title>Quality Report</title>
                <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
            </head>
            <body>
                <h1>Tractography Quality Report</h1>
                <div id="correlation"></div>
                <div id="similarity"></div>
                <div id="volume"></div>
            </body>
            <script>
                var correlation = """
                + fig_corr.to_json()
                + """;
                var similarity = """
                + fig_similarity.to_json()
                + """;
                var volume = """
                + fig_volume.to_json()
                + """;

                Plotly.newPlot('correlation', correlation.data, correlation.layout);
                Plotly.newPlot('similarity', similarity.data, similarity.layout);
                Plotly.newPlot('volume', volume.data, volume.layout);
            </script>
            </html>
            """
            )

    def compare_pipelines(
        self,
        pipeline_results: dict[str, dict[str, dict]],
        output_file: str = "pipeline_comparison.html",
    ) -> None:
        """
        Generate a comparison report for different tractography pipelines.

        Args:
            pipeline_results: Dictionary mapping pipeline names to their results
            output_file: Name of the output HTML file
        """
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots

        # Extract metrics for comparison
        metrics_data = {}
        for pipeline_name, results in pipeline_results.items():
            metrics_data[pipeline_name] = {
                "n_streamlines": [],
                "mean_length": [],
                "std_length": [],
                "similarity_scores": [],
                "bundle_volumes": [],
                "curvatures": [],
            }

            for _subject_id, subject_results in results.items():
                for _tract_name, tract_metrics in subject_results.items():
                    metrics_data[pipeline_name]["n_streamlines"].append(
                        tract_metrics.get("n_streamlines", 0)
                    )
                    metrics_data[pipeline_name]["mean_length"].append(
                        tract_metrics.get("mean_length", 0)
                    )
                    metrics_data[pipeline_name]["std_length"].append(
                        tract_metrics.get("std_length", 0)
                    )
                    metrics_data[pipeline_name]["similarity_scores"].append(
                        tract_metrics.get("similarity_score", 0)
                    )
                    metrics_data[pipeline_name]["bundle_volumes"].append(
                        tract_metrics.get("bundle_volume", 0)
                    )
                    metrics_data[pipeline_name]["curvatures"].append(
                        tract_metrics.get("curvature", 0)
                    )

        # Create figure with subplots
        fig = make_subplots(
            rows=3,
            cols=2,
            subplot_titles=(
                "Number of Streamlines",
                "Mean Length",
                "Length Standard Deviation",
                "Similarity Scores",
                "Bundle Volumes",
                "Curvature Distribution",
            ),
        )

        # Add box plots for each metric
        for i, (metric_name, _metric_data) in enumerate(
            [
                ("n_streamlines", "Number of Streamlines"),
                ("mean_length", "Mean Length"),
                ("std_length", "Length Standard Deviation"),
                ("similarity_scores", "Similarity Scores"),
                ("bundle_volumes", "Bundle Volumes"),
                ("curvatures", "Curvature"),
            ]
        ):
            row = i // 2 + 1
            col = i % 2 + 1

            for pipeline_name, data in metrics_data.items():
                fig.add_trace(
                    go.Box(
                        y=data[metric_name],
                        name=pipeline_name,
                        boxpoints="all",
                        jitter=0.3,
                        pointpos=-1.8,
                    ),
                    row=row,
                    col=col,
                )

        # Update layout
        fig.update_layout(
            height=1200,
            showlegend=True,
            title_text="Tractography Pipeline Comparison",
            boxmode="group",
        )

        # Save as HTML
        fig.write_html(str(self.output_dir / output_file))

    def generate_pipeline_report(
        self,
        pipeline_results: dict[str, dict[str, dict]],
        output_file: str = "pipeline_report.html",
    ) -> None:
        """
        Generate a detailed report comparing different tractography pipelines.

        Args:
            pipeline_results: Dictionary mapping pipeline names to their results
            output_file: Name of the output HTML file
        """
        import pandas as pd
        import plotly.express as px

        # Convert results to DataFrame
        data = []
        for pipeline_name, results in pipeline_results.items():
            for subject_id, subject_results in results.items():
                for tract_name, tract_metrics in subject_results.items():
                    data.append(
                        {
                            "pipeline": pipeline_name,
                            "subject": subject_id,
                            "tract": tract_name,
                            **tract_metrics,
                        }
                    )

        df = pd.DataFrame(data)

        # Create correlation matrix
        metrics = [
            "n_streamlines",
            "mean_length",
            "similarity_score",
            "bundle_volume",
            "curvature",
        ]
        corr_matrix = df[metrics].corr()

        # Create figures
        fig_corr = px.imshow(
            corr_matrix,
            title="Metrics Correlation Matrix",
            color_continuous_scale="RdBu",
        )

        fig_pipeline = px.box(
            df,
            x="pipeline",
            y="similarity_score",
            color="tract",
            title="Similarity Scores by Pipeline and Tract",
        )

        fig_volume = px.scatter(
            df,
            x="mean_length",
            y="bundle_volume",
            color="pipeline",
            symbol="tract",
            title="Volume vs Length by Pipeline and Tract",
        )

        # Save as HTML
        with open(self.output_dir / output_file, "w") as f:
            f.write(
                """
            <!DOCTYPE html>
            <html>
            <head>
                <title>Pipeline Comparison Report</title>
                <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
                <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
                <style>
                    body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; padding: 20px; }
                    .header { background: linear-gradient(135deg, #6c5ce7, #a29bfe);
                             color: white; padding: 2rem; border-radius: 10px; margin-bottom: 2rem; }
                    .plot-container { background: white; padding: 1.5rem; border-radius: 10px;
                                    margin-bottom: 2rem; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
                </style>
            </head>
            <body>
                <div class="container">
                    <div class="header text-center">
                        <h1>Tractography Pipeline Comparison</h1>
                        <p class="lead">Detailed Analysis of Pipeline Performance</p>
                    </div>

                    <div class="plot-container">
                        <div id="correlation"></div>
                    </div>

                    <div class="plot-container">
                        <div id="pipeline"></div>
                    </div>

                    <div class="plot-container">
                        <div id="volume"></div>
                    </div>
                </div>

                <script>
                    var correlation = """
                + fig_corr.to_json()
                + """;
                    var pipeline = """
                + fig_pipeline.to_json()
                + """;
                    var volume = """
                + fig_volume.to_json()
                + """;

                    Plotly.newPlot('correlation', correlation.data, correlation.layout);
                    Plotly.newPlot('pipeline', pipeline.data, pipeline.layout);
                    Plotly.newPlot('volume', volume.data, volume.layout);
                </script>
            </body>
            </html>
            """
            )
