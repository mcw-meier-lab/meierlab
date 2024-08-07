# vi: set ft=python sts=4 ts=4 sw=4 et:
from nilearn.image import binarize_img, mean_img


def create_avg_mask(nii_images):
    """Create a binary mask from the average of subject images.

    Parameters
    ----------
    nii_images : list
       List of image (:class:`~nibabel.nifti1.Nifti1Image`) or paths to files

    Returns
    -------
    :class:`~nibabel.nifti1.Nifti1Image`
       Binary mask Nifti image
    """
    mean_nii = mean_img(nii_images)
    mask_nii = binarize_img(mean_nii)

    return mask_nii
