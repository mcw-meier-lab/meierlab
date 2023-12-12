# -*- coding: utf-8 -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
import numpy as np
import pandas as pd
from nibabel import Nifti1Image, load, save
from nilearn.image import math_img, mean_img
from nipype.utils.filemanip import split_filename

from meierlab.masks import create_avg_mask


def compute_sd_thresholds(mean, sd, threshold=3):
    """Helper function to create Nifti images that are a certain standard deviation above and below the mean, as determined by the given threshold (default 3).

    Parameters
    ----------
    mean : image (:class:`~nibabel.nifti1.Nifti1Image` or file name)
        Mean image of the dataset.
    sd : image (:class:`~nibabel.nifti1.Nifti1Image` or file name)
        Standard deviation image of the dataset.
    threshold : int, optional
        Outlier threshold, by default 3.

    Returns
    -------
    tuple
        `thresh_above` corresponding to the data <threshold> SD above the mean.
        `thresh_below` corresponding to the data <threshold> SD below the mean.
        `upper_cap` containing a Nifti-like image at the upper limit of data.
        `lower_cap` containing a Nifti-like image at the lower limit of data.
    """
    above_str = f"mean_img + {threshold} * sd_img"
    thresh_above = math_img(above_str, mean_img=mean, sd_img=sd)
    below_str = f"mean_img - {threshold} * sd_img"
    thresh_below = math_img(below_str, mean_img=mean, sd_img=sd)
    upper_cap = Nifti1Image(np.full_like(mean.get_fdata(), 200), affine=mean.affine)
    lower_cap = Nifti1Image(np.full_like(mean.get_fdata(), -200), affine=mean.affine)

    return thresh_above, thresh_below, upper_cap, lower_cap


def compute_outlier_nii(
    nii_img, mask, mean, sd, thresh_above, thresh_below, upper_cap, lower_cap
):
    """Helper function to compute an individual outlier image.

    Parameters
    ----------
    nii_img : image (:class:`~nibabel.nifti1.Nifti1Image` or file name)
        Individual Nifti input file.
    mask : image (:class:`~nibabel.nifti1.Nifti1Image` or file name)
        A binary mask image to compute outliers within.
    mean : image (:class:`~nibabel.nifti1.Nifti1Image` or file name)
        Mean image of the dataset.
    sd :image (:class:`~nibabel.nifti1.Nifti1Image` or file name)
        Standard deviation image of the dataset.
    thresh_above : image (:class:`~nibabel.nifti1.Nifti1Image` or file name)
        Image of data above the given threshold (usually 3 SD).
    thresh_below : image (:class:`~nibabel.nifti1.Nifti1Image` or file name)
        Image of data below the given threshold (usually 3 SD).
    upper_cap : image (:class:`~nibabel.nifti1.Nifti1Image` or file name)
        Image of data forming the upper limit.
    lower_cap : image (:class:`~nibabel.nifti1.Nifti1Image` or file name)
        Image of data forming the lower limit.

    Returns
    -------
    :class:`~nibabel.nifti1.Nifti1Image`
        Returns a Nifti image with outlier data.
    """

    ### NOTE: this code is meant to be a python implementation of the following (from Andy Mayer's lab):
    # 3dcalc -a ${outDir}/${region}_mean+tlrc -b ${outDir}/${region}_sd+tlrc \
    #        -c ${subj} -d ${mask} \
    #        -expr 'd*(-1*within(c,-200,a-3*b)+1*within(c,a+3*b,200))' \
    #        -prefix ${outDir}/${s}.${region}_outliers
    outlier_nii = math_img(
        "(-1 * (np.less_equal(bthresh_img,img) & np.less_equal(img,below_img))) + (1*(np.less_equal(above_img,img) & np.less_equal(img,athresh_img)))",
        img=nii_img,
        mean_img=mean,
        mask_img=mask,
        sd_img=sd,
        below_img=thresh_below,
        above_img=thresh_above,
        athresh_img=upper_cap,
        bthresh_img=lower_cap,
    )

    return outlier_nii


def compute_outlier_values(
    nii_images, out_folder, save_m_sd=False, save_prefix=None, save_nii=False, mask=None
):
    """Compute outlier values for a set of nifti images.

    Parameters
    ----------
    nii_images : list
        Nifti images or file paths to compute outliers for.
    out_folder : str or path
        Destination folder.
    save_m_sd : bool, optional
        Option to save the mean and standard deviation images, by default False.
    save_prefix : str, optional
        String prefix to use when (if) saving files, by default None.
    save_nii : bool, optional
        Option to save individual outlier nifti files, by default False.
    mask : image (:class:`~nibabel.nifti1.Nifti1Image` or file name), optional
        Data mask to compute outliers within, by default None. If none given, a binary mask of the average computed from the dataset will be used.

    Returns
    -------
    :class:`~pandas.DataFrame`
        A dataframe containing: subject, total voxels, positive outlier counts and percentages, negative outlier counts and percentages.

    Examples
    --------
    >>> from meierlab import outliers
    >>> nii_images = ['sub-001.nii.gz','sub-002.nii.gz','sub-003.nii.gz']
    >>> out_folder = './outliers'
    >>> outlier_df = compute_outlier_values(nii_images, out_folder, save_m_sd=True)
    >>> outlier_df.to_csv(f'{out_folder}/outliers.csv',index=False)
    """
    out_df = pd.DataFrame(
        columns=[
            "subject",
            "total_voxels",
            "pos_count",
            "pos_pct",
            "neg_count",
            "neg_pct",
        ]
    )

    if not mask:
        mask_nii = create_avg_mask(nii_images)

    mask_nii = load(mask)
    total_voxels = np.count_nonzero(mask_nii.get_fdata())

    mean_nii = mean_img(nii_images)
    sd_nii = math_img("np.std(img,axis=-1)", img=nii_images)
    above, below, thresh_above, thresh_below = compute_sd_thresholds(mean_nii, sd_nii)

    if save_m_sd:
        if not save_prefix:
            save(mean_nii, out_folder / "mean.nii.gz")
            save(sd_nii, out_folder / "sd.nii.gz")
        else:
            save(mean_nii, out_folder / f"{save_prefix}_mean.nii.gz")
            save(sd_nii, out_folder / f"{save_prefix}_sd.nii.gz")

    for nii_img in nii_images:
        _, subject, _ = split_filename(nii_img)
        outlier_nii = compute_outlier_nii(
            nii_img,
            mask_nii,
            mean_nii,
            sd_nii,
            above,
            below,
            thresh_above,
            thresh_below,
        )
        outlier_masked = math_img("mask_img * img", mask_img=mask_nii, img=outlier_nii)
        if save_nii:
            if not save_prefix:
                save(outlier_masked, out_folder / f"{subject}_outliers.nii.gz")
            else:
                save(
                    outlier_masked,
                    out_folder / f"{save_prefix}_{subject}_outliers.nii.gz",
                )

        pos_count = np.count_nonzero(outlier_masked.get_fdata() > 0)
        neg_count = np.count_nonzero(outlier_masked.get_fdata() < 0)
        pos_pct = np.round(pos_count / total_voxels, 5)
        neg_pct = np.round(neg_count / total_voxels, 5)

        subj_out = pd.DataFrame(
            [
                {
                    "subject": subject,
                    "total_voxels": total_voxels,
                    "pos_count": pos_count,
                    "pos_pct": pos_pct,
                    "neg_count": neg_count,
                    "neg_pct": neg_pct,
                }
            ]
        )

        out_df = pd.concat([out_df, subj_out], ignore_index=True)

    return out_df
