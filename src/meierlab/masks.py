# -*- coding: utf-8 -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
from nilearn.image import mean_img, binarize_img

def create_avg_mask(nii_images):
   mean_nii = mean_img(nii_images)
   mask_nii = binarize_img(mean_nii)

   return mask_nii