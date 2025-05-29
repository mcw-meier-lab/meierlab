import glob
import json
import logging
import os
import subprocess
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import cpu_count
from typing import Any

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
from dipy.core.gradients import gradient_table
from dipy.data import default_sphere
from dipy.direction import peaks_from_model
from dipy.io.streamline import save_trk
from dipy.reconst.csdeconv import (
    ConstrainedSphericalDeconvModel,
    auto_response_ssst,
)
from dipy.reconst.dti import TensorModel
from dipy.reconst.shm import CsaOdfModel
from dipy.segment.mask import median_otsu
from dipy.segment.tissue import TissueClassifierHMRF
from dipy.tracking.local_tracking import LocalTracking
from dipy.tracking.stopping_criterion import (
    ActStoppingCriterion,
    BinaryStoppingCriterion,
    CmcStoppingCriterion,
    ThresholdStoppingCriterion,
)
from dipy.tracking.streamline import Streamlines
from scipy.stats import f_oneway, ttest_ind
from sklearn.model_selection import ParameterGrid

from meierlab.postproc.quality.tractography_metrics import TractographyMetrics


class TractographyPipeline:
    """
    A flexible pipeline for experimenting with different tractography parameters.
    """

    def __init__(
        self,
        dwi_data: np.ndarray,
        affine: np.ndarray,
        bvals: np.ndarray,
        bvecs: np.ndarray,
        t1_data: np.ndarray,
        mask: np.ndarray | None = None,
    ):
        """
        Initialize the tractography pipeline.

        Args:
            dwi_data: 4D numpy array of diffusion weighted images
            affine: 4x4 affine transformation matrix
            bvals: 1D array of b-values
            bvecs: 2D array of b-vectors
            t1_data: 3D numpy array of T1-weighted image
            mask: Optional binary mask for the brain. If None, will be generated using FSL's BET.
        """
        self.dwi_data = dwi_data
        self.affine = affine
        self.gtab = gradient_table(bvals, bvecs)
        self.t1_data = t1_data
        self.mask = mask if mask is not None else self._generate_brain_mask()
        self.parameters = {}
        self.logger = logging.getLogger(__name__)

    def _generate_brain_mask(
        self,
        fractional_intensity: float = 0.5,
        vertical_gradient: float = 0.0,
        robust: bool = True,
        temp_dir: str | None = None,
    ) -> np.ndarray:
        """
        Generate a binary brain mask using FSL's BET.

        Args:
            fractional_intensity: Fractional intensity threshold (0-1)
            vertical_gradient: Vertical gradient in fractional intensity threshold (-1 to 1)
            robust: Use robust brain center estimation
            temp_dir: Optional directory for temporary files

        Returns:
            Binary brain mask
        """
        import nibabel as nib

        # Create temporary directory if not provided
        if temp_dir is None:
            temp_dir = tempfile.mkdtemp()

        try:
            # Save b0 image to temporary file
            b0_idx = np.where(self.gtab.bvals == 0)[0][0]
            b0_image = self.dwi_data[..., b0_idx]

            b0_nii = nib.Nifti1Image(b0_image, self.affine)
            b0_path = os.path.join(temp_dir, "b0.nii.gz")
            nib.save(b0_nii, b0_path)

            # Run BET
            mask_path = os.path.join(temp_dir, "brain_mask.nii.gz")
            bet_cmd = [
                "bet",
                b0_path,
                mask_path,
                "-f",
                str(fractional_intensity),
                "-g",
                str(vertical_gradient),
                "-m",  # Generate binary mask
            ]

            if robust:
                bet_cmd.append("-R")

            # Execute BET
            subprocess.run(bet_cmd, check=True)

            # Load mask
            mask_nii = nib.load(mask_path)
            mask = mask_nii.get_fdata().astype(bool)

            return mask

        except subprocess.CalledProcessError as e:
            self.logger.error(f"BET failed: {e!s}")
            raise
        except Exception as e:
            self.logger.error(f"Error generating brain mask: {e!s}")
            raise
        finally:
            # Clean up temporary files
            if temp_dir is not None:
                try:
                    os.remove(b0_path)
                    os.remove(mask_path)
                    if temp_dir == tempfile.gettempdir():
                        os.rmdir(temp_dir)
                except Exception as e:
                    self.logger.warning(f"Error cleaning up temporary files: {e!s}")

    def _compute_csa_response(self, **kwargs) -> dict:
        """Compute response function using CSA method."""
        # Default parameters
        default_params = {
            "sh_order_max": 6,
            "smooth": 0.006,
            "lambda_csa": 0.1,
            "tau": 0.1,
            "mask": self.mask,
        }
        params = {**default_params, **kwargs}

        # Compute CSA response
        try:
            # Initialize CSA model
            csa_model = CsaOdfModel(
                self.gtab,
                sh_order_max=params["sh_order_max"],
                smooth=params["smooth"],
                lambda_csa=params["lambda_csa"],
                tau=params["tau"],
            )

            # Fit model
            csa_fit = csa_model.fit(self.dwi_data, mask=params["mask"])

            # Get response function
            response = csa_fit.response

            # Compute response ratio
            ratio = response[0] / response[1]

            return {
                "response": csa_model,
                "params": params,
                "ratio": ratio,
                "sh_coeff": csa_fit.shm_coeff,
            }

        except Exception as e:
            self.logger.error(f"Error computing CSA response: {e!s}")
            raise

    def set_parameters(self, params: dict) -> None:
        """
        Set or update pipeline parameters.

        Args:
            params: Dictionary of parameters to update
        """
        self.parameters.update(params)

    def compute_response_function(self, method: str = "msmt", **kwargs) -> dict:
        """
        Compute response function using specified method.

        Args:
            method: Method to use ('msmt', 'csd', 'csa')
            **kwargs: Additional parameters for the specific method

        Returns:
            Dictionary containing response function parameters
        """
        if method == "msmt":
            return self._compute_msmt_response(**kwargs)
        elif method == "csd":
            return self._compute_csd_response(**kwargs)
        elif method == "csa":
            return self._compute_csa_response(**kwargs)
        else:
            raise ValueError(f"Unknown response function method: {method}")

    def _compute_csd_response(self, **kwargs) -> dict:
        """Compute response function using CSD method."""
        default_params = {
            "sh_order": 8,
            "peak_thr": 0.01,
            "relative_peak_thr": 0.5,
            "min_separation_angle": 25,
            "npeaks": 5,
            "auto_response": True,
            "wm_pve": None,
        }
        params = {**default_params, **kwargs}

        # Compute response function
        if params["auto_response"]:
            if params["wm_pve"] is not None:
                # Use WM PVE map for better response function estimation
                response, ratio = auto_response_ssst(
                    self.gtab,
                    self.dwi_data,
                    roi_center=None,
                    roi_radius=10,
                    fa_thr=0.7,
                    wm_pve=params["wm_pve"],
                )
            else:
                # Standard response function estimation
                response, ratio = auto_response_ssst(
                    self.gtab, self.dwi_data, roi_center=None, roi_radius=10, fa_thr=0.7
                )

            csd_model = ConstrainedSphericalDeconvModel(
                self.gtab, response, sh_order=params["sh_order"]
            )
        else:
            csd_model = ConstrainedSphericalDeconvModel(
                self.gtab, sh_order=params["sh_order"]
            )
            response = None
            ratio = None

        return {"response": csd_model, "params": params, "ratio": ratio}

    def _compute_msmt_response(self, **kwargs) -> dict:
        """Compute response function using multi-shell multi-tissue (MSMT) method."""
        from dipy.reconst.mcsd import (
            MultiShellDeconvModel,
            auto_response_msmt,
            auto_response_ssst,
        )

        # Default parameters
        default_params = {
            "sh_order": 8,
            "peak_thr": 0.01,
            "relative_peak_thr": 0.5,
            "min_separation_angle": 25,
            "npeaks": 5,
            "wm_pve": None,
            "gm_pve": None,
            "csf_pve": None,
            "use_msmt": None,  # None for auto-detect, True/False to force
        }
        params = {**default_params, **kwargs}

        try:
            # Get tissue probability maps if not provided
            if any(
                pve is None
                for pve in [params["wm_pve"], params["gm_pve"], params["csf_pve"]]
            ):
                tissue_maps = self.segment_tissues()
                params["wm_pve"] = tissue_maps["wm_pve"]
                params["gm_pve"] = tissue_maps["gm_pve"]
                params["csf_pve"] = tissue_maps["csf_pve"]

            # Determine if MSMT should be used
            if params["use_msmt"] is None:
                # Check if we have multiple shells
                unique_bvals = np.unique(self.gtab.bvals)
                params["use_msmt"] = len(unique_bvals) > 2 and np.any(
                    unique_bvals > 1000
                )

            if params["use_msmt"]:
                # Use MSMT response function
                wm_response, gm_response, csf_response = auto_response_msmt(
                    self.gtab,
                    self.dwi_data,
                    params["wm_pve"],
                    params["gm_pve"],
                    params["csf_pve"],
                )
            else:
                # Use SSST response function
                wm_response, gm_response, csf_response = auto_response_ssst(
                    self.gtab,
                    self.dwi_data,
                    roi_center=None,
                    roi_radius=10,
                    fa_thr=0.7,
                    wm_pve=params["wm_pve"],
                    gm_pve=params["gm_pve"],
                    csf_pve=params["csf_pve"],
                )

            # Create MSMT model
            msmt_model = MultiShellDeconvModel(
                self.gtab,
                wm_response,
                gm_response,
                csf_response,
                sh_order=params["sh_order"],
            )

            return {
                "response": msmt_model,
                "params": params,
                "wm_response": wm_response,
                "gm_response": gm_response,
                "csf_response": csf_response,
                "method": "msmt" if params["use_msmt"] else "ssst",
            }

        except Exception as e:
            self.logger.error(f"Error computing MSMT response: {e!s}")
            raise

    def compute_fod(self, response: dict, **kwargs) -> dict:
        """
        Compute Fiber Orientation Distribution (FOD) using MSMT-CSD.

        Args:
            response: Dictionary containing response function parameters
            **kwargs: Additional parameters for FOD computation

        Returns:
            Dictionary containing FODs for each tissue
        """

        # Default parameters
        default_params = {
            "sh_order": 8,
            "mask": self.mask,
            "min_separation_angle": 25,
            "npeaks": 5,
        }
        params = {**default_params, **kwargs}

        try:
            # Get MSMT model
            msmt_model = response["response"]

            # Fit model
            msmt_fit = msmt_model.fit(self.dwi_data, mask=params["mask"])

            # Get FODs
            wm_fod = msmt_fit.wm_fod
            gm_fod = msmt_fit.gm_fod
            csf_fod = msmt_fit.csf_fod

            # Normalize FODs
            wm_fod_norm = self._normalize_fod(wm_fod)
            gm_fod_norm = self._normalize_fod(gm_fod)
            csf_fod_norm = self._normalize_fod(csf_fod)

            return {
                "wm_fod": wm_fod,
                "gm_fod": gm_fod,
                "csf_fod": csf_fod,
                "wm_fod_norm": wm_fod_norm,
                "gm_fod_norm": gm_fod_norm,
                "csf_fod_norm": csf_fod_norm,
                "params": params,
            }

        except Exception as e:
            self.logger.error(f"Error computing FODs: {e!s}")
            raise

    def _normalize_fod(self, fod: np.ndarray) -> np.ndarray:
        """Normalize FOD using MT normalization."""
        from dipy.reconst.mcsd import normalize_data

        # Normalize FOD
        fod_norm = normalize_data(fod)

        return fod_norm

    def track_streamlines(
        self,
        fod: dict,
        method: str = "local",
        act: bool = True,
        sift: bool = True,
        **kwargs,
    ) -> Streamlines:
        """
        Perform streamline tracking with optional ACT and SIFT filtering.

        Args:
            fod: Dictionary containing FODs
            method: Tracking method ('local', 'probabilistic', 'particle', 'eudx')
            act: Whether to use ACT
            sift: Whether to apply SIFT filtering
            **kwargs: Additional parameters for tracking and filtering

        Returns:
            Streamlines object containing the tracked fibers
        """
        from dipy.tracking.stopping_criterion import ActStoppingCriterion

        # Default parameters
        default_params = {
            "step_size": 0.5,
            "max_angle": 30.0,
            "max_length": 250,
            "min_length": 20,
            "seed_mask": None,
            "wm_pve": None,
            "gm_pve": None,
            "csf_pve": None,
        }
        params = {**default_params, **kwargs}

        try:
            # Get tissue probability maps if using ACT
            if act and any(
                pve is None
                for pve in [params["wm_pve"], params["gm_pve"], params["csf_pve"]]
            ):
                tissue_maps = self.segment_tissues()
                params["wm_pve"] = tissue_maps["wm_pve"]
                params["gm_pve"] = tissue_maps["gm_pve"]
                params["csf_pve"] = tissue_maps["csf_pve"]

            # Get seed mask
            if params["seed_mask"] is None:
                params["seed_mask"] = self._generate_gmwm_seed_mask(
                    params["wm_pve"], params["gm_pve"]
                )

            # Get stopping criterion
            if act:
                stopping_criterion = ActStoppingCriterion(
                    params["wm_pve"], params["gm_pve"], params["csf_pve"]
                )
            else:
                stopping_criterion = self._get_fa_stopping_criterion(**params)[0]

            # Track streamlines based on method
            if method == "local":
                streamlines = self._track_local(fod, stopping_criterion, params)
            elif method == "probabilistic":
                streamlines = self._track_probabilistic(fod, stopping_criterion, params)
            elif method == "particle":
                streamlines = self._track_particle(fod, stopping_criterion, params)
            elif method == "eudx":
                streamlines = self._track_eudx(fod, stopping_criterion, params)
            else:
                raise ValueError(f"Unknown tracking method: {method}")

            # Apply SIFT filtering if requested
            if sift:
                streamlines = self.filter_streamlines(
                    streamlines=streamlines, fod=fod, act=act, **params
                )

            return Streamlines(streamlines)

        except Exception as e:
            self.logger.error(f"Error in streamline tracking: {e!s}")
            raise

    def _track_local(
        self, fod: dict, stopping_criterion: object, params: dict
    ) -> Streamlines:
        """Perform local tracking using various algorithms."""
        from dipy.direction import peaks_from_model

        # Get peaks from FOD
        peaks = peaks_from_model(
            fod["wm_fod_norm"], self.dwi_data, self.gtab, mask=self.mask, **params
        )

        # Track streamlines
        streamlines = LocalTracking(
            peaks, stopping_criterion, params["seed_mask"], self.affine, **params
        )

        return streamlines

    def _track_probabilistic(
        self, fod: dict, stopping_criterion: object, params: dict
    ) -> Streamlines:
        """Perform probabilistic tracking."""
        from dipy.direction import ProbabilisticDirectionGetter

        # Create probabilistic direction getter
        pmf = fod["wm_fod_norm"]
        prob_dg = ProbabilisticDirectionGetter.from_pmf(
            pmf, max_angle=params["max_angle"], sphere=default_sphere
        )

        # Track streamlines
        streamlines = LocalTracking(
            prob_dg, stopping_criterion, params["seed_mask"], self.affine, **params
        )

        return streamlines

    def _track_particle(
        self, fod: dict, stopping_criterion: object, params: dict
    ) -> Streamlines:
        """Perform particle filtering tracking."""
        from dipy.tracking.tracker import pft_tracking

        # Get peaks from FOD
        peaks = peaks_from_model(
            fod["wm_fod_norm"], self.dwi_data, self.gtab, mask=self.mask, **params
        )

        # Track streamlines
        pft = pft_tracking(
            peaks, stopping_criterion, params["seed_mask"], self.affine, **params
        )

        return pft

    def _track_eudx(
        self, fod: dict, stopping_criterion: object, params: dict
    ) -> Streamlines:
        """Perform EuDX tracking."""
        from dipy.tracking.tracker import eudx_tracking

        # Get peaks from FOD
        peaks = peaks_from_model(
            fod["wm_fod_norm"], self.dwi_data, self.gtab, mask=self.mask, **params
        )

        # Track streamlines
        eudx = eudx_tracking(
            peaks.peak_values,
            peaks.peak_indices,
            stopping_criterion,
            params["seed_mask"],
            self.affine,
            **params,
        )

        return eudx

    def _generate_gmwm_seed_mask(
        self, wm_pve: np.ndarray, gm_pve: np.ndarray
    ) -> np.ndarray:
        """Generate seed mask from GM and WM probability maps."""
        # Combine GM and WM probability maps
        gmwm_pve = wm_pve + gm_pve

        # Threshold to get seed mask
        seed_mask = gmwm_pve > 0.5

        return seed_mask

    def generate_seed_mask(self, method: str = "wm", **kwargs) -> np.ndarray:
        """
        Generate seed mask for tractography.

        Args:
            method: Method to use ('wm', 'gm', 'custom')
            **kwargs: Additional parameters for the specific method

        Returns:
            Binary seed mask
        """
        if method == "wm":
            return self._generate_wm_seed_mask(**kwargs)
        elif method == "gm":
            return self._generate_gm_seed_mask(**kwargs)
        elif method == "custom":
            return self._generate_custom_seed_mask(**kwargs)
        else:
            raise ValueError(f"Unknown seed mask method: {method}")

    def _generate_wm_seed_mask(self, **kwargs) -> np.ndarray:
        """Generate white matter seed mask.

        Args:
            wm_pve: White matter probability map (optional)
            threshold: Probability threshold for WM (default: 0.5)
            min_volume: Minimum volume in mm³ for connected components (default: 100)

        Returns:
            Binary white matter seed mask
        """
        from scipy.ndimage import binary_closing, label

        # Default parameters
        default_params = {
            "wm_pve": None,
            "threshold": 0.5,
            "min_volume": 100,
            "dilation_radius": 1,
        }
        params = {**default_params, **kwargs}

        try:
            if params["wm_pve"] is None:
                # Generate WM mask from DWI data if not provided
                _, wm_mask = median_otsu(
                    self.dwi_data[..., 0],  # Use b0 image
                    median_radius=2,
                    numpass=2,
                    autocrop=False,
                )
            else:
                # Use provided WM probability map
                wm_mask = params["wm_pve"] > params["threshold"]

            # Remove small connected components
            labeled_array, num_features = label(wm_mask)
            component_sizes = np.bincount(labeled_array.ravel())
            component_sizes[0] = 0  # Ignore background
            small_components = component_sizes < params["min_volume"]
            wm_mask[labeled_array == small_components] = 0

            # Apply morphological closing to fill small holes
            wm_mask = binary_closing(
                wm_mask, structure=np.ones((params["dilation_radius"],) * 3)
            )

            return wm_mask

        except Exception as e:
            self.logger.error(f"Error generating WM seed mask: {e!s}")
            raise

    def _generate_gm_seed_mask(self, **kwargs) -> np.ndarray:
        """Generate gray matter seed mask.

        Args:
            gm_pve: Gray matter probability map (optional)
            threshold: Probability threshold for GM (default: 0.5)
            min_volume: Minimum volume in mm³ for connected components (default: 100)
            exclude_wm: Whether to exclude white matter regions (default: True)

        Returns:
            Binary gray matter seed mask
        """
        from scipy.ndimage import binary_closing, label

        # Default parameters
        default_params = {
            "gm_pve": None,
            "threshold": 0.5,
            "min_volume": 100,
            "dilation_radius": 1,
            "exclude_wm": True,
        }
        params = {**default_params, **kwargs}

        try:
            if params["gm_pve"] is None:
                # Generate GM mask from DWI data if not provided
                _, gm_mask = median_otsu(
                    self.dwi_data[..., 0],  # Use b0 image
                    median_radius=2,
                    numpass=2,
                    autocrop=False,
                )
                # Invert to get GM-like regions
                gm_mask = ~gm_mask
            else:
                # Use provided GM probability map
                gm_mask = params["gm_pve"] > params["threshold"]

            if params["exclude_wm"]:
                # Get WM mask to exclude
                wm_mask = self._generate_wm_seed_mask(**kwargs)
                gm_mask[wm_mask] = 0

            # Remove small connected components
            labeled_array, num_features = label(gm_mask)
            component_sizes = np.bincount(labeled_array.ravel())
            component_sizes[0] = 0  # Ignore background
            small_components = component_sizes < params["min_volume"]
            gm_mask[labeled_array == small_components] = 0

            # Apply morphological closing to fill small holes
            gm_mask = binary_closing(
                gm_mask, structure=np.ones((params["dilation_radius"],) * 3)
            )

            return gm_mask

        except Exception as e:
            self.logger.error(f"Error generating GM seed mask: {e!s}")
            raise

    def _generate_custom_seed_mask(self, **kwargs) -> np.ndarray:
        """Generate custom seed mask based on user-defined criteria.

        Args:
            mask: Pre-defined binary mask (optional)
            roi_coords: List of ROI coordinates in voxel space (optional)
            roi_radius: Radius for spherical ROIs in mm (default: 5)
            combine_method: How to combine multiple ROIs ('union' or 'intersection', default: 'union')
            smooth: Whether to smooth the mask (default: True)
            smooth_sigma: Standard deviation for Gaussian smoothing (default: 1.0)

        Returns:
            Binary custom seed mask
        """
        from scipy.ndimage import gaussian_filter
        from scipy.spatial.distance import cdist

        # Default parameters
        default_params = {
            "mask": None,
            "roi_coords": None,
            "roi_radius": 5,
            "combine_method": "union",
            "smooth": True,
            "smooth_sigma": 1.0,
        }
        params = {**default_params, **kwargs}

        try:
            if params["mask"] is not None:
                # Use provided mask
                seed_mask = params["mask"].astype(bool)
            elif params["roi_coords"] is not None:
                # Generate mask from ROI coordinates
                seed_mask = np.zeros(self.mask.shape, dtype=bool)

                # Convert radius from mm to voxels
                voxel_size = np.sqrt(np.sum(self.affine[:3, :3] ** 2, axis=0))
                radius_voxels = params["roi_radius"] / voxel_size

                # Create spherical ROIs
                for coord in params["roi_coords"]:
                    # Create grid of points
                    x, y, z = np.meshgrid(
                        np.arange(self.mask.shape[0]),
                        np.arange(self.mask.shape[1]),
                        np.arange(self.mask.shape[2]),
                        indexing="ij",
                    )
                    points = np.column_stack((x.ravel(), y.ravel(), z.ravel()))

                    # Calculate distances
                    distances = cdist(points, [coord])
                    distances = distances.reshape(self.mask.shape)

                    # Create spherical ROI
                    roi = distances <= radius_voxels

                    # Combine with existing mask
                    if params["combine_method"] == "union":
                        seed_mask = seed_mask | roi
                    else:  # intersection
                        seed_mask = seed_mask & roi
            else:
                raise ValueError("Either mask or roi_coords must be provided")

            # Apply smoothing if requested
            if params["smooth"]:
                seed_mask = (
                    gaussian_filter(
                        seed_mask.astype(float), sigma=params["smooth_sigma"]
                    )
                    > 0.5
                )

            # Ensure mask is within brain mask
            seed_mask = seed_mask & self.mask

            return seed_mask

        except Exception as e:
            self.logger.error(f"Error generating custom seed mask: {e!s}")
            raise

    def get_stopping_criterion(
        self, method: str = "fa", **kwargs
    ) -> tuple[object, dict]:
        """
        Get stopping criterion for tractography.

        Args:
            method: Method to use ('gfa', 'act', 'cmc', 'binary', 'custom')
            **kwargs: Additional parameters for the specific method

        Returns:
            Tuple of (stopping_criterion, parameters)
        """
        if method == "gfa":
            return self._get_fa_stopping_criterion(**kwargs)
        elif method == "act":
            return self._get_act_stopping_criterion(**kwargs)
        elif method == "cmc":
            return self._get_cmc_stopping_criterion(**kwargs)
        elif method == "binary":
            return self._get_binary_stopping_criterion(**kwargs)
        elif method == "custom":
            return self._get_custom_stopping_criterion(**kwargs)
        else:
            raise ValueError(f"Unknown stopping criterion method: {method}")

    def _get_gfa_stopping_criterion(
        self, **kwargs
    ) -> tuple[ThresholdStoppingCriterion, dict]:
        """Get GFA-based stopping criterion."""
        default_params = {"min_gfa": 0.2, "max_gfa": 1.0}
        params = {**default_params, **kwargs}

        # Compute FA if not provided
        if "gfa" not in kwargs:
            tensor_model = TensorModel(self.gtab)
            tensor_fit = tensor_model.fit(self.dwi_data, mask=self.mask)
            gfa = tensor_fit.gfa
        else:
            gfa = kwargs["gfa"]

        stopping_criterion = ThresholdStoppingCriterion(gfa, params["min_gfa"])
        return stopping_criterion, params

    def _get_act_stopping_criterion(
        self, **kwargs
    ) -> tuple[ActStoppingCriterion, dict]:
        """Get Anatomically Constrained Tractography (ACT) stopping criterion."""
        default_params = {
            "include_map": None,
            "exclude_map": None,
            "wm_map": None,
            "gm_map": None,
            "csf_map": None,
            "wm_pve": None,  # Add WM PVE probability map
            "step_size": 0.5,
        }
        params = {**default_params, **kwargs}

        # Validate required maps
        required_maps = ["wm_map", "gm_map", "csf_map"]
        for map_name in required_maps:
            if params[map_name] is None:
                raise ValueError(f"ACT requires {map_name} to be provided")

        # If WM PVE is provided, use it for more precise tracking
        if params["wm_pve"] is not None:
            # Combine WM mask with PVE for more precise tracking
            wm_map = params["wm_map"] * params["wm_pve"]
        else:
            wm_map = params["wm_map"]

        stopping_criterion = ActStoppingCriterion(
            include_map=params["include_map"],
            exclude_map=params["exclude_map"],
            wm_map=wm_map,
            gm_map=params["gm_map"],
            csf_map=params["csf_map"],
            step_size=params["step_size"],
        )
        return stopping_criterion, params

    def _get_cmc_stopping_criterion(
        self, **kwargs
    ) -> tuple[CmcStoppingCriterion, dict]:
        """Get CMC-based stopping criterion."""
        default_params = {
            "wm_map": None,
            "gm_map": None,
            "csf_map": None,
            "step_size": 0.5,
            "average_voxel_size": 2.0,
        }
        params = {**default_params, **kwargs}

        # Validate required maps
        required_maps = ["wm_map", "gm_map", "csf_map"]
        for map_name in required_maps:
            if params[map_name] is None:
                raise ValueError(f"CMC requires {map_name} to be provided")

        stopping_criterion = CmcStoppingCriterion.from_pve(
            wm_map=params["wm_map"],
            gm_map=params["gm_map"],
            csf_map=params["csf_map"],
            step_size=params["step_size"],
            average_voxel_size=params["average_voxel_size"],
        )
        return stopping_criterion, params

    def _get_binary_stopping_criterion(
        self, **kwargs
    ) -> tuple[BinaryStoppingCriterion, dict]:
        """Get binary mask-based stopping criterion."""
        if "mask" not in kwargs:
            raise ValueError("Binary stopping criterion requires a mask to be provided")

        stopping_criterion = BinaryStoppingCriterion(kwargs["mask"])
        return stopping_criterion, kwargs

    def _get_custom_stopping_criterion(self, **kwargs) -> tuple[object, dict]:
        """Get custom stopping criterion."""
        if "criterion" not in kwargs:
            raise ValueError(
                "Custom stopping criterion requires a criterion object to be provided"
            )

        return kwargs["criterion"], kwargs

    def segment_tissues(self, method: str = "hmrf", **kwargs) -> dict[str, np.ndarray]:
        """
        Generate tissue probability maps from T1 data.

        Args:
            method: Segmentation method ('hmrf', 'fast')
            **kwargs: Additional parameters for segmentation

        Returns:
            Dictionary containing tissue probability maps
        """
        if method == "hmrf":
            return self._segment_hmrf(**kwargs)
        elif method == "fast":
            return self._segment_fast(**kwargs)
        else:
            raise ValueError(f"Unknown segmentation method: {method}")

    def _segment_hmrf(self, **kwargs) -> dict[str, np.ndarray]:
        """Segment tissues using Hidden Markov Random Field (HMRF)."""
        default_params = {"n_iter": 100, "beta": 0.1, "n_classes": 3, "tolerance": 1e-4}
        params = {**default_params, **kwargs}

        # Initialize classifier
        classifier = TissueClassifierHMRF(
            n_iter=params["n_iter"],
            beta=params["beta"],
            n_classes=params["n_classes"],
            tolerance=params["tolerance"],
        )

        # Perform segmentation
        segmentation = classifier.classify(self.t1_data, self.mask)

        # Convert to probability maps
        wm_pve = (segmentation == 2).astype(float)
        gm_pve = (segmentation == 1).astype(float)
        csf_pve = (segmentation == 0).astype(float)

        # Apply smoothing if needed
        if kwargs.get("smooth", True):
            wm_pve = self._smooth_probability_map(wm_pve)
            gm_pve = self._smooth_probability_map(gm_pve)
            csf_pve = self._smooth_probability_map(csf_pve)

        return {"wm_pve": wm_pve, "gm_pve": gm_pve, "csf_pve": csf_pve}

    def _segment_fast(self, **kwargs) -> dict[str, np.ndarray]:
        """Segment tissues using FSL's FAST (FMRIB's Automated Segmentation Tool).

        Args:
            n_classes: Number of tissue classes (default: 3)
            bias_correct: Whether to perform bias field correction (default: True)
            bias_lowpass: Lowpass filter width for bias field (default: 20)
            bias_iters: Number of bias field correction iterations (default: 4)
            output_dir: Directory for temporary files (optional)
            cleanup: Whether to clean up temporary files (default: True)

        Returns:
            Dictionary containing tissue probability maps:
            - 'wm_pve': White matter probability map
            - 'gm_pve': Gray matter probability map
            - 'csf_pve': CSF probability map
        """
        import os

        import nibabel as nib

        # Default parameters
        default_params = {
            "n_classes": 3,
            "bias_correct": True,
            "bias_lowpass": 20,
            "bias_iters": 4,
            "output_dir": None,
            "cleanup": True,
        }
        params = {**default_params, **kwargs}

        try:
            # Create temporary directory if not provided
            if params["output_dir"] is None:
                temp_dir = tempfile.mkdtemp()
            else:
                temp_dir = params["output_dir"]
                os.makedirs(temp_dir, exist_ok=True)

            # Save T1 data to temporary file
            t1_nii = nib.Nifti1Image(self.t1_data, self.affine)
            t1_path = os.path.join(temp_dir, "t1.nii.gz")
            nib.save(t1_nii, t1_path)

            # Build FAST command
            fast_cmd = [
                "fast",
                "-n",
                str(params["n_classes"]),
                "-t",
                "1",  # T1-weighted image
                "-o",
                os.path.join(temp_dir, "fast"),
                t1_path,
            ]

            # Add bias field correction options
            if params["bias_correct"]:
                fast_cmd.extend(
                    [
                        "-b",
                        "-l",
                        str(params["bias_lowpass"]),
                        "-B",
                        str(params["bias_iters"]),
                    ]
                )

            # Run FAST
            self.logger.info("Running FSL FAST segmentation...")
            subprocess.run(fast_cmd, check=True)

            # Load segmentation results
            wm_pve = nib.load(os.path.join(temp_dir, "fast_pve_2.nii.gz")).get_fdata()
            gm_pve = nib.load(os.path.join(temp_dir, "fast_pve_1.nii.gz")).get_fdata()
            csf_pve = nib.load(os.path.join(temp_dir, "fast_pve_0.nii.gz")).get_fdata()

            # Clean up temporary files if requested
            if params["cleanup"]:
                for f in glob.glob(os.path.join(temp_dir, "fast_*")):
                    os.remove(f)
                if params["output_dir"] is None:
                    os.rmdir(temp_dir)

            # Normalize probability maps
            total = wm_pve + gm_pve + csf_pve
            wm_pve = np.divide(
                wm_pve, total, out=np.zeros_like(wm_pve), where=total != 0
            )
            gm_pve = np.divide(
                gm_pve, total, out=np.zeros_like(gm_pve), where=total != 0
            )
            csf_pve = np.divide(
                csf_pve, total, out=np.zeros_like(csf_pve), where=total != 0
            )

            # Apply brain mask
            wm_pve = wm_pve * self.mask
            gm_pve = gm_pve * self.mask
            csf_pve = csf_pve * self.mask

            return {"wm_pve": wm_pve, "gm_pve": gm_pve, "csf_pve": csf_pve}

        except subprocess.CalledProcessError as e:
            self.logger.error(f"FSL FAST failed: {e!s}")
            raise
        except Exception as e:
            self.logger.error(f"Error in FAST segmentation: {e!s}")
            raise
        finally:
            # Clean up temporary files in case of error
            if (
                params["cleanup"]
                and params["output_dir"] is None
                and "temp_dir" in locals()
            ):
                try:
                    for f in glob.glob(os.path.join(temp_dir, "fast_*")):
                        os.remove(f)
                    os.rmdir(temp_dir)
                except Exception as e:
                    self.logger.warning(f"Error cleaning up temporary files: {e!s}")

    def _smooth_probability_map(
        self, prob_map: np.ndarray, sigma: float = 1.0
    ) -> np.ndarray:
        """Smooth probability map using Gaussian filter."""
        from scipy.ndimage import gaussian_filter

        return gaussian_filter(prob_map, sigma=sigma)

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
        # Create metrics computer
        metrics_computer = TractographyMetrics(
            affine=self.affine,
            mask_shape=self.mask.shape,
            cc_mask=self.cc_mask if hasattr(self, "cc_mask") else None,
        )

        # Get all quality control metrics
        return metrics_computer.quality_control(streamlines, response, tissue_maps)

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
        # Create metrics computer
        metrics_computer = TractographyMetrics(
            affine=self.affine,
            mask_shape=self.mask.shape,
            cc_mask=self.cc_mask if hasattr(self, "cc_mask") else None,
        )

        # Run optimization
        return metrics_computer.optimize_cmc_parameters(
            streamlines=streamlines,
            tissue_maps=tissue_maps,
            param_grid=param_grid,
            n_iter=n_iter,
            output_dir=output_dir,
        )

    def visualize_qc(self, qc_metrics: dict, output_dir: str) -> None:
        """Generate visualization of quality control metrics.

        Args:
            qc_metrics: Dictionary of QC metrics
            output_dir: Directory to save visualizations
        """
        # Create metrics computer
        metrics_computer = TractographyMetrics(
            affine=self.affine,
            mask_shape=self.mask.shape,
            cc_mask=self.cc_mask if hasattr(self, "cc_mask") else None,
        )

        # Generate visualizations
        metrics_computer.visualize_qc(qc_metrics, output_dir)

        # Generate QC report
        metrics_computer.generate_qc_report(qc_metrics, output_dir)

    def filter_streamlines(
        self, streamlines: Streamlines, fod: dict, act: bool = True, **kwargs
    ) -> dict[str, Any]:
        """
        Filter streamlines using MRtrix's SIFT approach.

        Args:
            streamlines: Input streamlines to filter
            fod: Dictionary containing FODs
            act: Whether to use ACT for filtering
            **kwargs: Additional parameters for filtering

        Returns:
            Dictionary containing filtered streamlines and quality metrics
        """
        import os

        import nibabel as nib

        # Default parameters
        default_params = {
            "target_number": 1000000,
            "min_length": 20,
            "max_length": 250,
            "step_size": 0.5,
            "wm_pve": None,
            "gm_pve": None,
            "csf_pve": None,
            "compress": True,
            "compute_metrics": True,
            "sift_options": {
                "term_mu": 0.1,
                "term_threshold": 0.1,
                "max_iter": 100,
                "min_iter": 10,
                "lambda": 0.1,
            },
        }
        params = {**default_params, **kwargs}

        try:
            # Create temporary directory
            with tempfile.TemporaryDirectory() as temp_dir:
                # Save streamlines to temporary file
                streamlines_path = os.path.join(temp_dir, "streamlines.tck")
                nib.streamlines.save(streamlines, streamlines_path)

                # Save FOD to temporary file
                fod_path = os.path.join(temp_dir, "fod.mif")
                nib.save(nib.Nifti1Image(fod["wm_fod_norm"], self.affine), fod_path)

                # Save tissue maps if using ACT
                if act and all(
                    pve is not None
                    for pve in [params["wm_pve"], params["gm_pve"], params["csf_pve"]]
                ):
                    wm_path = os.path.join(temp_dir, "wm.mif")
                    gm_path = os.path.join(temp_dir, "gm.mif")
                    csf_path = os.path.join(temp_dir, "csf.mif")

                    nib.save(nib.Nifti1Image(params["wm_pve"], self.affine), wm_path)
                    nib.save(nib.Nifti1Image(params["gm_pve"], self.affine), gm_path)
                    nib.save(nib.Nifti1Image(params["csf_pve"], self.affine), csf_path)

                # Build SIFT command
                sift_cmd = [
                    "tcksift",
                    streamlines_path,
                    fod_path,
                    os.path.join(temp_dir, "sift_weights.txt"),
                ]

                # Add ACT options if using ACT
                if act and all(
                    pve is not None
                    for pve in [params["wm_pve"], params["gm_pve"], params["csf_pve"]]
                ):
                    sift_cmd.extend(
                        ["-act", wm_path, "-act_gm", gm_path, "-act_csf", csf_path]
                    )

                # Add SIFT options
                sift_options = params["sift_options"]
                sift_cmd.extend(
                    [
                        "-term_mu",
                        str(sift_options["term_mu"]),
                        "-term_threshold",
                        str(sift_options["term_threshold"]),
                        "-max_iter",
                        str(sift_options["max_iter"]),
                        "-min_iter",
                        str(sift_options["min_iter"]),
                        "-lambda",
                        str(sift_options["lambda"]),
                    ]
                )

                # Run SIFT
                subprocess.run(sift_cmd, check=True)

                # Load SIFT weights
                with open(os.path.join(temp_dir, "sift_weights.txt")) as f:
                    sift_weights = np.array([float(line.strip()) for line in f])

                # Filter streamlines based on weights
                filtered_streamlines = Streamlines(
                    [
                        s
                        for s, w in zip(streamlines, sift_weights, strict=False)
                        if w > 0
                    ]
                )

                # Compute quality metrics if requested
                metrics = {}
                if params["compute_metrics"]:
                    metrics_computer = TractographyMetrics(
                        affine=self.affine,
                        mask_shape=self.mask.shape,
                        cc_mask=getattr(self, "cc_mask", None),
                    )
                    metrics = metrics_computer.compute_streamline_metrics(
                        streamlines=filtered_streamlines,
                        fod=fod,
                        sift_weights=sift_weights,
                        act=act,
                        params=params,
                    )

                return {
                    "streamlines": filtered_streamlines,
                    "metrics": metrics,
                    "sift_weights": sift_weights,
                }

        except subprocess.CalledProcessError as e:
            self.logger.error(f"MRtrix SIFT failed: {e!s}")
            raise
        except Exception as e:
            self.logger.error(f"Error filtering streamlines: {e!s}")
            raise

    def _compute_streamline_metrics(
        self,
        streamlines: Streamlines,
        fod: dict,
        sift_weights: np.ndarray | None = None,
        act: bool = True,
        params: dict | None = None,
    ) -> dict[str, Any]:
        """Compute comprehensive quality metrics for filtered streamlines."""
        metrics_computer = TractographyMetrics(
            affine=self.affine,
            mask_shape=self.mask.shape,
            cc_mask=getattr(self, "cc_mask", None),
        )
        return metrics_computer.compute_streamline_metrics(
            streamlines=streamlines,
            fod=fod,
            sift_weights=sift_weights,
            act=act,
            params=params,
        )


class PipelineComparison:
    """
    Class for comparing different tractography pipeline configurations.
    """

    def __init__(
        self, base_pipeline: "TractographyPipeline", atlas_dir: str | None = None
    ):
        """
        Initialize pipeline comparison.

        Args:
            base_pipeline: Base pipeline instance to use for comparisons
            atlas_dir: Directory containing reference atlas data
        """
        self.base_pipeline = base_pipeline
        self.comparisons = {}
        self.logger = logging.getLogger(__name__)
        self.atlas_dir = atlas_dir
        self.reference_data = self._load_reference_data() if atlas_dir else None

    def _load_reference_data(self) -> dict[str, Any]:
        """Load reference data from atlas directory."""
        reference_data = {}

        # Load bundle masks
        bundle_dir = os.path.join(self.atlas_dir, "bundles")
        if os.path.exists(bundle_dir):
            for bundle_file in os.listdir(bundle_dir):
                if bundle_file.endswith(".nii.gz"):
                    bundle_name = os.path.splitext(os.path.splitext(bundle_file)[0])[0]
                    bundle_path = os.path.join(bundle_dir, bundle_file)
                    reference_data[f"bundle_{bundle_name}"] = nib.load(
                        bundle_path
                    ).get_fdata()

        # Load ground truth streamlines if available
        gt_dir = os.path.join(self.atlas_dir, "ground_truth")
        if os.path.exists(gt_dir):
            for gt_file in os.listdir(gt_dir):
                if gt_file.endswith(".trk"):
                    gt_name = os.path.splitext(gt_file)[0]
                    gt_path = os.path.join(gt_dir, gt_file)
                    reference_data[f"gt_{gt_name}"] = nib.streamlines.load(
                        gt_path
                    ).streamlines

        return reference_data

    def optimize_parameters(
        self,
        param_grid: dict[str, list[Any]],
        metric: str = "dice_score",
        n_iter: int = 10,
        n_workers: int | None = None,
    ) -> dict[str, Any]:
        """
        Optimize pipeline parameters using parallel grid search.

        Args:
            param_grid: Dictionary of parameters to optimize
            metric: Metric to optimize ('dice_score', 'bundle_similarity', etc.)
            n_iter: Number of iterations for optimization
            n_workers: Number of parallel workers (default: cpu_count - 1)

        Returns:
            Dictionary of optimized parameters and optimization history
        """
        if n_workers is None:
            n_workers = max(1, cpu_count() - 1)

        best_score = -np.inf
        best_params = None
        optimization_history = []

        # Generate parameter combinations
        param_combinations = list(ParameterGrid(param_grid))[:n_iter]

        # Process parameters in parallel
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            future_to_params = {
                executor.submit(self._evaluate_parameters, params, metric): params
                for params in param_combinations
            }

            for future in as_completed(future_to_params):
                params = future_to_params[future]
                try:
                    score = future.result()
                    optimization_history.append({"params": params, "score": score})

                    if score > best_score:
                        best_score = score
                        best_params = params

                except Exception as e:
                    self.logger.error(f"Error evaluating parameters {params}: {e!s}")

        # Save optimization history
        self._save_optimization_history(optimization_history)

        # Generate optimization visualizations
        self._visualize_optimization_results(optimization_history)

        return {
            "best_params": best_params,
            "best_score": best_score,
            "history": optimization_history,
        }

    def _evaluate_parameters(self, params: dict[str, Any], metric: str) -> float:
        """Evaluate a single parameter combination."""
        try:
            config = self._create_config_from_params(params)
            results = self.run_comparison(
                [config], output_dir=f"temp_opt_{hash(str(params))}"
            )
            return self._compute_optimization_score(results, metric)
        except Exception as e:
            self.logger.error(f"Error in parameter evaluation: {e!s}")
            return -np.inf

    def _save_optimization_history(self, history: list[dict[str, Any]]) -> None:
        """Save optimization history to file."""
        # Convert to DataFrame
        df = pd.DataFrame(history)

        # Save to CSV
        df.to_csv("optimization_history.csv", index=False)

        # Save to JSON
        with open("optimization_history.json", "w") as f:
            json.dump(history, f, indent=2)

    def _visualize_optimization_results(self, history: list[dict[str, Any]]) -> None:
        """Generate interactive visualizations of optimization results."""
        # Create output directory
        opt_dir = "optimization_visualizations"
        os.makedirs(opt_dir, exist_ok=True)

        # Convert to DataFrame
        df = pd.DataFrame(history)

        # Extract parameter names
        param_names = list(history[0]["params"].keys())

        # Create parallel coordinates plot
        fig = px.parallel_coordinates(
            df,
            color="score",
            dimensions=[*param_names, "score"],
            color_continuous_scale=px.colors.sequential.Viridis,
            title="Parameter Optimization Landscape",
        )
        fig.write_html(f"{opt_dir}/parallel_coordinates.html")

        # Create scatter plot matrix
        if len(param_names) > 1:
            fig = px.scatter_matrix(
                df,
                dimensions=[*param_names, "score"],
                color="score",
                title="Parameter Relationships",
            )
            fig.write_html(f"{opt_dir}/scatter_matrix.html")

        # Create optimization progress plot
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=list(range(len(history))),
                y=[h["score"] for h in history],
                mode="lines+markers",
                name="Score",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=list(range(len(history))),
                y=[
                    max(h["score"] for h in history[: i + 1])
                    for i in range(len(history))
                ],
                mode="lines",
                name="Best Score",
            )
        )
        fig.update_layout(
            title="Optimization Progress", xaxis_title="Iteration", yaxis_title="Score"
        )
        fig.write_html(f"{opt_dir}/optimization_progress.html")

        # Create parameter importance plot
        if len(param_names) > 1:
            param_importance = self._compute_parameter_importance(df)
            fig = go.Figure(
                data=[
                    go.Bar(
                        x=list(param_importance.keys()),
                        y=list(param_importance.values()),
                    )
                ]
            )
            fig.update_layout(
                title="Parameter Importance",
                xaxis_title="Parameter",
                yaxis_title="Importance Score",
            )
            fig.write_html(f"{opt_dir}/parameter_importance.html")

    def _compute_parameter_importance(self, df: pd.DataFrame) -> dict[str, float]:
        """Compute parameter importance scores."""
        param_names = [col for col in df.columns if col != "score"]
        importance_scores = {}

        for param in param_names:
            # Compute correlation with score
            correlation = df[param].corr(df["score"])
            importance_scores[param] = abs(correlation)

        return importance_scores

    def run_comparison(
        self,
        configs: list[dict[str, Any]],
        output_dir: str,
        n_workers: int | None = None,
    ) -> dict[str, Any]:
        """
        Run comparison of different pipeline configurations in parallel.

        Args:
            configs: List of configuration dictionaries
            output_dir: Directory to save comparison results
            n_workers: Number of parallel workers (default: cpu_count - 1)

        Returns:
            Dictionary containing comparison results
        """
        if n_workers is None:
            n_workers = max(1, cpu_count() - 1)

        results = {}
        os.makedirs(output_dir, exist_ok=True)

        # Process configurations in parallel
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            future_to_config = {
                executor.submit(self._process_configuration, config, i, output_dir): (
                    config,
                    i,
                )
                for i, config in enumerate(configs)
            }

            for future in as_completed(future_to_config):
                config, i = future_to_config[future]
                try:
                    config_name = f"config_{i}"
                    results[config_name] = future.result()
                except Exception as e:
                    self.logger.error(f"Error processing configuration {i}: {e!s}")

        # Generate comparison report
        self._generate_comparison_report(results, output_dir)

        return results

    def _process_configuration(
        self, config: dict[str, Any], config_idx: int, output_dir: str
    ) -> dict[str, Any]:
        """Process a single configuration."""
        config_name = f"config_{config_idx}"
        self.logger.info(f"Running configuration {config_name}")

        # Create pipeline copy with new configuration
        pipeline = self.base_pipeline.__class__(
            dwi_data=self.base_pipeline.dwi_data,
            affine=self.base_pipeline.affine,
            bvals=self.base_pipeline.gtab.bvals,
            bvecs=self.base_pipeline.gtab.bvecs,
            t1_data=self.base_pipeline.t1_data,
            mask=self.base_pipeline.mask,
        )

        # Apply configuration
        if "parameters" in config:
            pipeline.set_parameters(config["parameters"])

        # Run pipeline
        streamlines = self._run_pipeline(pipeline, config)

        # Perform QC
        qc_metrics = pipeline.quality_control(
            streamlines=streamlines,
            response=pipeline.compute_response_function(**config.get("response", {})),
            tissue_maps=config.get("tissue_maps"),
        )

        # Save results
        config_dir = os.path.join(output_dir, config_name)
        os.makedirs(config_dir, exist_ok=True)

        # Save streamlines
        save_trk(f"{config_dir}/streamlines.trk", streamlines, pipeline.affine)

        # Save QC visualizations
        pipeline.visualize_qc(qc_metrics, config_dir)

        # Save configuration
        with open(f"{config_dir}/config.json", "w") as f:
            json.dump(config, f, indent=2)

        return {"config": config, "streamlines": streamlines, "qc_metrics": qc_metrics}

    def _create_pipeline_with_config(
        self, config: dict[str, Any]
    ) -> "TractographyPipeline":
        """Create a new pipeline instance with given configuration."""
        # Create a copy of the base pipeline
        pipeline = self.base_pipeline.__class__(
            dwi_data=self.base_pipeline.dwi_data,
            affine=self.base_pipeline.affine,
            bvals=self.base_pipeline.gtab.bvals,
            bvecs=self.base_pipeline.gtab.bvecs,
            t1_data=self.base_pipeline.t1_data,
            mask=self.base_pipeline.mask,
        )

        # Apply configuration
        if "parameters" in config:
            pipeline.set_parameters(config["parameters"])

        return pipeline

    def _run_pipeline(
        self, pipeline: "TractographyPipeline", config: dict[str, Any]
    ) -> Streamlines:
        """Run pipeline with given configuration."""
        # Generate tissue maps if needed
        if config.get("generate_tissue_maps", False):
            tissue_maps = pipeline.segment_tissues(**config.get("segmentation", {}))
            config["tissue_maps"] = tissue_maps

        # Compute response function
        response = pipeline.compute_response_function(**config.get("response", {}))

        # Generate seed mask
        seed_mask = pipeline.generate_seed_mask(**config.get("seed_mask", {}))

        # Track streamlines
        streamlines = pipeline.track_streamlines(
            response=response, seed_mask=seed_mask, **config.get("tracking", {})
        )

        return streamlines

    def _generate_comparison_report(
        self, results: dict[str, Any], output_dir: str
    ) -> None:
        """Generate comparison report of all configurations."""
        # Create comparison metrics
        comparison_metrics = self._compute_comparison_metrics(results)

        # Save metrics to CSV
        metrics_df = pd.DataFrame(comparison_metrics)
        metrics_df.to_csv(f"{output_dir}/comparison_metrics.csv", index=False)

        # Generate comparison plots
        self._plot_comparison_metrics(comparison_metrics, output_dir)

        # Generate HTML report
        self._generate_html_report(results, comparison_metrics, output_dir)

    def _compute_comparison_metrics(self, results: dict[str, Any]) -> dict[str, list]:
        """Compute comparison metrics across configurations."""
        metrics = {
            "config_name": [],
            "n_streamlines": [],
            "mean_length": [],
            "std_length": [],
            "silhouette_score": [],
            "response_ratio": [],
            "wm_prob_mean": [],
            "gm_prob_mean": [],
            "csf_prob_mean": [],
        }

        for config_name, result in results.items():
            qc = result["qc_metrics"]

            metrics["config_name"].append(config_name)
            metrics["n_streamlines"].append(qc["streamlines"]["n_streamlines"])
            metrics["mean_length"].append(qc["streamlines"]["mean_length"])
            metrics["std_length"].append(qc["streamlines"]["std_length"])
            metrics["silhouette_score"].append(
                qc["streamlines"].get("silhouette_score", None)
            )

            if "response" in qc and "response_ratio" in qc["response"]:
                metrics["response_ratio"].append(qc["response"]["response_ratio"])
            else:
                metrics["response_ratio"].append(None)

            if "tissue" in qc:
                metrics["wm_prob_mean"].append(
                    qc["tissue"].get("wm_pve_mean_prob", None)
                )
                metrics["gm_prob_mean"].append(
                    qc["tissue"].get("gm_pve_mean_prob", None)
                )
                metrics["csf_prob_mean"].append(
                    qc["tissue"].get("csf_pve_mean_prob", None)
                )
            else:
                metrics["wm_prob_mean"].append(None)
                metrics["gm_prob_mean"].append(None)
                metrics["csf_prob_mean"].append(None)

        return metrics

    def _plot_comparison_metrics(
        self, metrics: dict[str, list], output_dir: str
    ) -> None:
        """Generate comparison plots."""
        # Create figure
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))

        # Plot number of streamlines
        axes[0, 0].bar(metrics["config_name"], metrics["n_streamlines"])
        axes[0, 0].set_title("Number of Streamlines")
        axes[0, 0].tick_params(axis="x", rotation=45)

        # Plot mean length
        axes[0, 1].bar(metrics["config_name"], metrics["mean_length"])
        axes[0, 1].set_title("Mean Streamline Length")
        axes[0, 1].tick_params(axis="x", rotation=45)

        # Plot silhouette scores
        axes[1, 0].bar(metrics["config_name"], metrics["silhouette_score"])
        axes[1, 0].set_title("Streamline Clustering Quality")
        axes[1, 0].tick_params(axis="x", rotation=45)

        # Plot tissue probabilities
        x = np.arange(len(metrics["config_name"]))
        width = 0.25
        axes[1, 1].bar(x - width, metrics["wm_prob_mean"], width, label="WM")
        axes[1, 1].bar(x, metrics["gm_prob_mean"], width, label="GM")
        axes[1, 1].bar(x + width, metrics["csf_prob_mean"], width, label="CSF")
        axes[1, 1].set_title("Mean Tissue Probabilities")
        axes[1, 1].set_xticks(x)
        axes[1, 1].set_xticklabels(metrics["config_name"], rotation=45)
        axes[1, 1].legend()

        plt.tight_layout()
        plt.savefig(f"{output_dir}/comparison_plots.png")
        plt.close()

    def _generate_html_report(
        self, results: dict[str, Any], metrics: dict[str, list], output_dir: str
    ) -> None:
        """Generate HTML comparison report with statistical testing."""
        html_content = """
        <html>
        <head>
            <title>Tractography Pipeline Comparison</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; }
                table { border-collapse: collapse; width: 100%; }
                th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
                th { background-color: #f2f2f2; }
                img { max-width: 100%; height: auto; }
            </style>
        </head>
        <body>
            <h1>Tractography Pipeline Comparison</h1>
            <h2>Comparison Metrics</h2>
            <table>
                <tr>
                    <th>Configuration</th>
                    <th># Streamlines</th>
                    <th>Mean Length</th>
                    <th>Silhouette Score</th>
                    <th>Response Ratio</th>
                </tr>
        """

        for i, config_name in enumerate(metrics["config_name"]):
            html_content += f"""
                <tr>
                    <td>{config_name}</td>
                    <td>{metrics["n_streamlines"][i]}</td>
                    <td>{metrics["mean_length"][i]:.2f}</td>
                    <td>{metrics["silhouette_score"][i]:.2f if metrics['silhouette_score'][i] is not None else 'N/A'}</td>
                    <td>{metrics["response_ratio"][i]:.2f if metrics['response_ratio'][i] is not None else 'N/A'}</td>
                </tr>
            """

        html_content += """
            </table>
            <h2>Comparison Plots</h2>
            <img src="comparison_plots.png" alt="Comparison Plots">
            <h2>Individual Configurations</h2>
        """

        for config_name in metrics["config_name"]:
            html_content += f"""
                <h3>{config_name}</h3>
                <img src="{config_name}/streamline_stats.png" alt="Streamline Stats">
                <img src="{config_name}/tissue_metrics.png" alt="Tissue Metrics">
                <img src="{config_name}/density_visualization.png" alt="Density Visualization">
            """

        html_content += """
        </body>
        </html>
        """

        with open(f"{output_dir}/comparison_report.html", "w") as f:
            f.write(html_content)

    def perform_statistical_testing(
        self, results: dict[str, Any], metric: str
    ) -> dict[str, Any]:
        """
        Perform statistical testing between configurations.

        Args:
            results: Dictionary of comparison results
            metric: Metric to test

        Returns:
            Dictionary of statistical test results
        """
        # Extract metric values for each configuration
        metric_values = []
        config_names = []

        for config_name, result in results.items():
            if metric in result["qc_metrics"]:
                metric_values.append(result["qc_metrics"][metric])
                config_names.append(config_name)

        if len(metric_values) < 2:
            return {"error": "Not enough configurations for statistical testing"}

        # Perform ANOVA if more than 2 configurations
        if len(metric_values) > 2:
            f_stat, p_value = f_oneway(*metric_values)
            test_type = "ANOVA"
        else:
            # Perform t-test for 2 configurations
            t_stat, p_value = ttest_ind(metric_values[0], metric_values[1])
            test_type = "t-test"

        return {
            "test_type": test_type,
            "p_value": p_value,
            "configurations": config_names,
            "metric_values": metric_values,
        }

    def visualize_statistical_results(
        self, results: dict[str, Any], output_dir: str
    ) -> None:
        """
        Generate visualizations for statistical results.

        Args:
            results: Dictionary of comparison results
            output_dir: Directory to save visualizations
        """
        # Create output directory
        stats_dir = os.path.join(output_dir, "statistical_analysis")
        os.makedirs(stats_dir, exist_ok=True)

        # Get metrics for statistical analysis
        metrics = ["n_streamlines", "mean_length", "silhouette_score"]
        if self.reference_data:
            metrics.extend(["dice_score", "bundle_similarity"])

        # Create figure for all statistical visualizations
        fig, axes = plt.subplots(len(metrics), 2, figsize=(15, 5 * len(metrics)))

        for i, metric in enumerate(metrics):
            # Perform statistical testing
            test_results = self.perform_statistical_testing(results, metric)

            if "error" not in test_results:
                # Box plot
                sns.boxplot(data=test_results["metric_values"], ax=axes[i, 0])
                axes[i, 0].set_title(f"{metric} Distribution")
                axes[i, 0].set_xticklabels(test_results["configurations"])
                axes[i, 0].tick_params(axis="x", rotation=45)

                # Violin plot
                sns.violinplot(data=test_results["metric_values"], ax=axes[i, 1])
                axes[i, 1].set_title(f"{metric} Distribution with Density")
                axes[i, 1].set_xticklabels(test_results["configurations"])
                axes[i, 1].tick_params(axis="x", rotation=45)

                # Add p-value annotation
                p_value = test_results["p_value"]
                significance = (
                    "***"
                    if p_value < 0.001
                    else "**"
                    if p_value < 0.01
                    else "*"
                    if p_value < 0.05
                    else "ns"
                )
                axes[i, 0].text(
                    0.5,
                    0.95,
                    f"p = {p_value:.4f} {significance}",
                    transform=axes[i, 0].transAxes,
                    ha="center",
                )

        plt.tight_layout()
        plt.savefig(f"{stats_dir}/statistical_analysis.png")
        plt.close()

        # Generate correlation matrix if multiple metrics are available
        if len(metrics) > 1:
            self._plot_correlation_matrix(results, metrics, stats_dir)

        # Generate statistical summary table
        self._generate_statistical_summary(results, metrics, stats_dir)

    def _plot_correlation_matrix(
        self, results: dict[str, Any], metrics: list[str], output_dir: str
    ) -> None:
        """Generate correlation matrix plot for metrics."""
        # Extract metric values
        metric_data = {}
        for metric in metrics:
            values = []
            for _config_name, result in results.items():
                if metric in result["qc_metrics"]:
                    values.append(result["qc_metrics"][metric])
            metric_data[metric] = values

        # Create correlation matrix
        corr_matrix = pd.DataFrame(metric_data).corr()

        # Plot correlation matrix
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            corr_matrix, annot=True, cmap="coolwarm", vmin=-1, vmax=1, square=True
        )
        plt.title("Metric Correlation Matrix")
        plt.tight_layout()
        plt.savefig(f"{output_dir}/correlation_matrix.png")
        plt.close()

    def _generate_statistical_summary(
        self, results: dict[str, Any], metrics: list[str], output_dir: str
    ) -> None:
        """Generate statistical summary table."""
        summary_data = []

        for metric in metrics:
            test_results = self.perform_statistical_testing(results, metric)

            if "error" not in test_results:
                summary_data.append(
                    {
                        "Metric": metric,
                        "Test Type": test_results["test_type"],
                        "p-value": test_results["p_value"],
                        "Significant": "Yes"
                        if test_results["p_value"] < 0.05
                        else "No",
                        "Mean Values": [
                            np.mean(vals) for vals in test_results["metric_values"]
                        ],
                        "Std Values": [
                            np.std(vals) for vals in test_results["metric_values"]
                        ],
                    }
                )

        # Create summary table
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_csv(f"{output_dir}/statistical_summary.csv", index=False)

        # Generate HTML summary
        html_content = """
        <html>
        <head>
            <title>Statistical Summary</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; }
                table { border-collapse: collapse; width: 100%; }
                th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
                th { background-color: #f2f2f2; }
            </style>
        </head>
        <body>
            <h1>Statistical Summary</h1>
            <table>
                <tr>
                    <th>Metric</th>
                    <th>Test Type</th>
                    <th>p-value</th>
                    <th>Significant</th>
                    <th>Mean Values</th>
                    <th>Std Values</th>
                </tr>
        """

        for _, row in summary_df.iterrows():
            html_content += f"""
                <tr>
                    <td>{row["Metric"]}</td>
                    <td>{row["Test Type"]}</td>
                    <td>{row["p-value"]:.4f}</td>
                    <td>{row["Significant"]}</td>
                    <td>{", ".join([f"{v:.2f}" for v in row["Mean Values"]])}</td>
                    <td>{", ".join([f"{v:.2f}" for v in row["Std Values"]])}</td>
                </tr>
            """

        html_content += """
            </table>
        </body>
        </html>
        """

        with open(f"{output_dir}/statistical_summary.html", "w") as f:
            f.write(html_content)
