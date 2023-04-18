
import pydicom as dicom             
import numpy as np

from pathlib import Path
from nibabel import load, save, Nifti1Image

def load_dcm(realDicomLocation,imagDicomLocation):
    scanSetting = {}
    iField = None
    
    slice_tag = 0x20,0x9057
    echo_tag = 0x18,0x86

    for dcm_file in list(Path(realDicomLocation).glob('**/*dcm')):
        dcm = dicom.read_file(dcm_file)
        thisSlice=(dcm[slice_tag].value)-1
        thisEcho=(dcm[echo_tag].value)-1
        if thisSlice == 0 and thisEcho == 0:
            ref_vals = dcm    

    # image size
    cols_qsm = int(ref_vals.Columns)
    rows_qsm = int(ref_vals.Rows)
    echoes_qsm = int(ref_vals.EchoTrainLength)
    slices_qsm = int(len(list(Path(realDicomLocation).glob("**/*dcm")))/echoes_qsm)    
        
    # resolution 
    col_res_qsm = (ref_vals.PixelSpacing[0]) #are these backwards?
    row_res_qsm = (ref_vals.PixelSpacing[1])
    slice_res_qsm = (ref_vals.SliceThickness)  
    voxel_size = [col_res_qsm, row_res_qsm, slice_res_qsm]
        
    # origin
    origin_qsm = np.array(ref_vals[0x0020, 0x0032].value) #(0020, 0032) Image Position (Patient)
    origin_qsm = np.asfarray(origin_qsm) 
    origin_qsm = -origin_qsm/voxel_size        
    
    B0_dir = [0,0,1]
    gyro = 42.58*1e6
    
    scanSetting["B0Dir"] = B0_dir         
    scanSetting["gyro"] = gyro
    scanSetting["B0"] = float(ref_vals.MagneticFieldStrength)
    scanSetting["CF"] = float(ref_vals.MagneticFieldStrength)*gyro
    scanSetting["numEchoes"] = int(ref_vals.EchoTrainLength)
    scanSetting["dTE"] = float(ref_vals.LargestImagePixelValue)*1e-6
    scanSetting["echoTimes"] = None       
    scanSetting["matrxiSize"] = [cols_qsm, rows_qsm, slices_qsm] 
    scanSetting["voxelSize"] = [col_res_qsm, row_res_qsm, slice_res_qsm]
    scanSetting["orginPos"] = origin_qsm
        
    # read the dicoms 
    realData = np.zeros((cols_qsm, rows_qsm, slices_qsm, echoes_qsm), dtype=np.float64)
    imagData = np.zeros((cols_qsm, rows_qsm, slices_qsm, echoes_qsm), dtype=np.float64)
    
    for dcm_file in list(Path(realDicomLocation).glob('**/*dcm')):
        dcm = dicom.read_file(dcm_file)
        thisSlice=(dcm[slice_tag].value)-1
        thisEcho=(dcm[echo_tag].value)-1
        realData[:,:,thisSlice,thisEcho] = dcm.pixel_array

    for dcm_file in list(Path(imagDicomLocation).glob('**/*dcm')):
        dcm = dicom.read_file(dcm_file)
        thisSlice=(dcm[slice_tag].value)-1
        thisEcho=(dcm[echo_tag].value)-1
        imagData[:,:,thisSlice,thisEcho] = dcm.pixel_array

    realData = np.flip(np.transpose(realData, [1,0,2,3]), 1)
    imagData = np.flip(np.transpose(imagData, [1,0,2,3]), 1)
    iField = realData + 1j*imagData
        
    return iField, scanSetting

def get_last_echo(source_dir, out_dir, target_nii, out_file):
    ifield, _ = load_dcm(
        list(Path(source_dir).glob('*Real*/resources/DICOM'))[0],
        list(Path(source_dir).glob('*Imag*/resources/DICOM'))[0]
    )

    img = np.abs(ifield[:,:,:,-1])
    target = load(target_nii)
    last_echo = Nifti1Image(img,affine=target.affine)
    save(last_echo,Path(out_dir) / out_file)

    return

