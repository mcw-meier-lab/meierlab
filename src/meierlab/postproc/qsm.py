from pathlib import Path

import numpy as np
import pydicom as dicom
from nibabel import Nifti1Image, load, save


def load_dcm(real_dcm_dir, imag_dcm_dir):
    scan_meta = {}
    field_map = None

    slice_tag = 0x20, 0x9057
    echo_tag = 0x18, 0x86

    for dcm_file in list(Path(real_dcm_dir).glob("**/*dcm")):
        dcm = dicom.read_file(dcm_file)
        slice_val = (dcm[slice_tag].value) - 1
        echo_val = (dcm[echo_tag].value) - 1
        if slice_val == 0 and echo_val == 0:
            ref_vals = dcm

    # image size
    cols_qsm = int(ref_vals.Columns)
    rows_qsm = int(ref_vals.Rows)
    echoes_qsm = int(ref_vals.EchoTrainLength)
    slices_qsm = int(len(list(Path(real_dcm_dir).glob("**/*dcm"))) / echoes_qsm)

    # resolution
    col_res_qsm = ref_vals.PixelSpacing[0]
    row_res_qsm = ref_vals.PixelSpacing[1]
    slice_res_qsm = ref_vals.SliceThickness
    voxel_size = [col_res_qsm, row_res_qsm, slice_res_qsm]

    # origin
    origin_qsm = np.array(
        ref_vals[0x0020, 0x0032].value
    )  # (0020, 0032) Image Position (Patient)
    origin_qsm = np.asfarray(origin_qsm)
    origin_qsm = -origin_qsm / voxel_size

    B0_dir = [0, 0, 1]
    gyro = 42.58 * 1e6

    scan_meta["B0Dir"] = B0_dir
    scan_meta["gyro"] = gyro
    scan_meta["B0"] = float(ref_vals.MagneticFieldStrength)
    scan_meta["CF"] = float(ref_vals.MagneticFieldStrength) * gyro
    scan_meta["numEchoes"] = int(ref_vals.EchoTrainLength)
    scan_meta["dTE"] = float(ref_vals.LargestImagePixelValue) * 1e-6
    scan_meta["echoTimes"] = None
    scan_meta["matrxiSize"] = [cols_qsm, rows_qsm, slices_qsm]
    scan_meta["voxelSize"] = [col_res_qsm, row_res_qsm, slice_res_qsm]
    scan_meta["orginPos"] = origin_qsm

    # read the dicoms
    real_data = np.zeros((cols_qsm, rows_qsm, slices_qsm, echoes_qsm), dtype=np.float64)
    imag_data = np.zeros((cols_qsm, rows_qsm, slices_qsm, echoes_qsm), dtype=np.float64)

    for dcm_file in list(Path(real_dcm_dir).glob("**/*dcm")):
        dcm = dicom.read_file(dcm_file)
        slice_val = (dcm[slice_tag].value) - 1
        echo_val = (dcm[echo_tag].value) - 1
        real_data[:, :, slice_val, echo_val] = dcm.pixel_array

    for dcm_file in list(Path(imag_dcm_dir).glob("**/*dcm")):
        dcm = dicom.read_file(dcm_file)
        slice_val = (dcm[slice_tag].value) - 1
        echo_val = (dcm[echo_tag].value) - 1
        imag_data[:, :, slice_val, echo_val] = dcm.pixel_array

    # realData = np.flip(np.transpose(realData, [1,0,2,3]), 1) - orig but L/R flipped
    # imagData = np.flip(np.transpose(imagData, [1,0,2,3]), 1) - orig but L/R flipped

    real_data = np.flip(np.flip(np.transpose(real_data, [1, 0, 2, 3]), 1), 0)
    imag_data = np.flip(np.flip(np.transpose(imag_data, [1, 0, 2, 3]), 1), 0)
    field_map = real_data + 1j * imag_data

    return field_map, scan_meta


def get_mag_data(real_dcm_dir, imag_dcm_dir, target_file, out_dir, out_file=None):
    field_map, _ = load_dcm(real_dcm_dir, imag_dcm_dir)

    target = load(target_file)
    out_file = out_file if out_file != None else ""

    # magnitude
    img = np.abs(field_map)
    mag = Nifti1Image(img, affine=target.affine)
    save(mag, Path(out_dir) / f"{out_file}magnitude.nii.gz")

    # get last echo of the magnitude of complex data/field map
    img = np.abs(field_map[:, :, :, -1])
    last_echo = Nifti1Image(img, affine=target.affine)
    save(last_echo, Path(out_dir) / f"{out_file}lastecho.nii.gz")

    # sum of squares of the magnitude/echoes
    # (smoother/some data loss compared to last_echo)
    img = np.sqrt(np.sum((np.abs(field_map)) ** 2, axis=3))
    ss_mag = Nifti1Image(img, affine=target.affine)
    save(ss_mag, Path(out_dir) / f"{out_file}magnitudeSS.nii.gz")

    return
