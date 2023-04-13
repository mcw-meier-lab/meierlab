# -*- coding: utf-8 -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
import numpy as np
import pandas as pd
from nilearn.image import mean_img, math_img
from nipype.utils.filemanip import split_filename
from nibabel import save, load, Nifti1Image
from pathlib import Path
from meierlab.masks import create_avg_mask


# TODO
# generate_graphs(): plotly interactive dashboard/images


def compute_sd_thresholds(mean, sd, threshold=3):
    above_str = f"mean_img + {threshold} * sd_img"
    above = math_img(above_str,mean_img=mean,sd_img=sd)
    below_str = f"mean_img - {threshold} * sd_img"
    below = math_img(below_str,mean_img=mean,sd_img=sd)
    thresh_above = Nifti1Image(
        np.full_like(mean.get_fdata(),200),affine=mean.affine
    )
    thresh_below = Nifti1Image(
        np.full_like(mean.get_fdata(),-200),affine=mean.affine
    )

    return above, below, thresh_above, thresh_below

def compute_outlier_nii(nii_img,mask,mean,sd,
                        above,below,thresh_above,thresh_below):
    outlier_nii = math_img(
        '(-1 * (np.less_equal(bthresh_img,img) & np.less_equal(img,below_img))) + (1*(np.less_equal(above_img,img) & np.less_equal(img,athresh_img)))',
        img=nii_img,mean_img=mean,mask_img=mask,sd_img=sd,
        below_img=below,above_img=above,
        athresh_img=thresh_above,bthresh_img=thresh_below
    )

    return outlier_nii

def compute_outlier_values(nii_images, out_folder, save_m_sd=False, save_prefix=None, save_nii=False, mask=None):
    out_df = pd.DataFrame(
        columns=["subject","total_voxels","pos_count",
                 "pos_pct","neg_count","neg_pct"]
    )

    if not mask:
        mask_nii = create_avg_mask(nii_images)

    mask_nii = load(mask)
    total_voxels = np.count_nonzero(mask_nii.get_fdata())

    mean_nii = mean_img(nii_images)
    sd_nii = math_img("np.std(img,axis=-1)",img=nii_images)
    above, below, thresh_above, thresh_below = compute_sd_thresholds(mean_nii,sd_nii)

    if save_m_sd:
        if not save_prefix:
            save(mean_nii, out_folder / "mean.nii.gz")
            save(sd_nii, out_folder / "sd.nii.gz")
        else:
            save(mean_nii, out_folder / f"{save_prefix}_mean.nii.gz")
            save(sd_nii, out_folder / f"{save_prefix}_sd.nii.gz")
    
    for nii_img in nii_images:
        _, subject, _ = split_filename(nii_img)
        outlier_nii = compute_outlier_nii(nii_img,mask_nii,
                                          mean_nii,sd_nii,above,below,thresh_above,thresh_below
                                          )
        outlier_masked = math_img('mask_img * img',mask_img=mask_nii,img=outlier_nii)
        if save_nii:
            if not save_prefix:
                save(outlier_masked, out_folder / f"{subject}_outliers.nii.gz")
            else:
                save(outlier_masked, out_folder / f"{save_prefix}_{subject}_outliers.nii.gz")

        pos_count = np.count_nonzero(outlier_masked.get_fdata() > 0)
        neg_count = np.count_nonzero(outlier_masked.get_fdata() < 0)
        pos_pct = np.round(pos_count / total_voxels,5)
        neg_pct = np.round(neg_count / total_voxels,5)

        subj_out = pd.DataFrame([{'subject':subject,'total_voxels':total_voxels,
                    'pos_count':pos_count,'pos_pct':pos_pct,
                    'neg_count':neg_count,'neg_pct':neg_pct
                    }])
        
        out_df = pd.concat([out_df,subj_out], ignore_index=True)

    return out_df
