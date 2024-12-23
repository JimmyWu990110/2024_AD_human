import os
import numpy as np
import SimpleITK as sitk
import cv2

from DataLoader import DataLoader


def get_geometry(img, name):
    spacing = img.GetSpacing()
    origin = img.GetOrigin()
    direction = img.GetDirection()
    print("******", name, "geometry ******")
    print("size:", img.GetSize())
    print("sapcing:", spacing)
    print("origin:", origin)
    print("direction:", direction)
    return spacing, origin, direction

def check_geometry(patient_name):
    base_dir = r"C:\Users\jwu191\Desktop\Projects\AD_fitting\processed_data"
    T1 = sitk.ReadImage(os.path.join(base_dir, patient_name+"_all", "T1_map.nii"))
    T1_reg = sitk.ReadImage(os.path.join(base_dir, patient_name+"_all", "T1_map_reg.nii"))
    M0 = sitk.ReadImage(os.path.join(base_dir, patient_name+"_all", "M0.nii"))
    EMR_single = sitk.ReadImage(os.path.join(base_dir, patient_name+"_all", "EMR_single.nii"))
    EMR = sitk.ReadImage(os.path.join(base_dir, patient_name+"_all", "EMR.nii"))
    get_geometry(T1, "T1")
    get_geometry(T1_reg, "T1_reg")
    get_geometry(M0, "M0")
    get_geometry(EMR_single, "EMR_single")
    get_geometry(EMR, "EMR")

def resize(patient_name):
    base_dir = r"C:\Users\jwu191\Desktop\Projects\AD_fitting\processed_data"
    M0 = sitk.ReadImage(os.path.join(base_dir, patient_name+"_all", "M0.nii"))
    EMR_single = sitk.ReadImage(os.path.join(base_dir, patient_name+"_all", "EMR_single.nii"))
    M0_spacing, _, _ = get_geometry(M0, "M0")
    # tmp = EMR_single[:, :, :, 1]
    # tmp = get_geometry(tmp, "tmp")
    res_list = []
    for i in range(list(EMR_single.GetSize())[-1]): # [224, 224, 1, 56]
        tmp = EMR_single[:, :, :, i]
        resample = sitk.ResampleImageFilter()
        resample.SetInterpolator = sitk.sitkLinear
        resample.SetOutputDirection = tmp.GetDirection()
        resample.SetOutputOrigin = tmp.GetOrigin()
        new_spacing = list(tmp.GetSpacing())
        new_spacing[0:2] = list(M0_spacing)[0:2]
        resample.SetOutputSpacing = tuple(new_spacing)
        resample.SetSize = tuple([256, 256, 1])
        # res = resample.Execute(tmp)
        transform = sitk.AffineTransform()
        res = sitk.Resample(tmp, tuple([256, 256, 1]), transform, sitk.sitkLinear,
                            tmp.GetOrigin(), tuple(new_spacing),tmp.GetDirection())
        res_list.append(res)
        get_geometry(tmp, "tmp")
        get_geometry(res, "res")
    EMR_single_reg = sitk.JoinSeries(res_list)
    get_geometry(EMR_single_reg, "EMR_single_reg")

def geometry_4D_to_3D(spacing, origin, direction):
    new_spacing = np.array(spacing)[0:3]
    new_origin = np.array(origin)[0:3]
    new_direction = np.array(direction)
    new_direction = np.reshape(new_direction, [4, 4])
    new_direction = new_direction[0:3, 0:3]
    new_direction = new_direction.flatten() # [9,]
    return new_spacing, new_origin, new_direction

def reg(patient_name):
    base_dir = r"C:\Users\jwu191\Desktop\Projects\AD_fitting\processed_data"
    M0 = sitk.ReadImage(os.path.join(base_dir, patient_name+"_all", "M0.nii"))
    M0_spacing, _, _ = get_geometry(M0, "M0")
    EMR_single = sitk.ReadImage(os.path.join(base_dir, patient_name+"_all", "EMR_single.nii"))
    spacing, origin, direction = get_geometry(EMR_single, "EMR_single")   
    EMR_single_arr = sitk.GetArrayFromImage(EMR_single) # [56, 1, 224, 224]
    res_arr = np.zeros((EMR_single_arr.shape[0], 256, 256)) # [56, 256, 256]
    for i in range(EMR_single_arr.shape[0]):
        tmp = EMR_single_arr[i][0]
        res_arr[i] = cv2.resize(tmp, (256, 256))
    res = sitk.GetImageFromArray(res_arr)
    spacing = list(spacing)
    spacing[0:2] = M0_spacing[0:2]
    spacing, origin, direction = geometry_4D_to_3D(spacing, origin, direction)
    res.SetSpacing(spacing)
    res.SetOrigin(origin)
    res.SetDirection(direction)
    get_geometry(res, "res")
    sitk.WriteImage(res, os.path.join(base_dir, patient_name+"_all", "EMR_single_reg.nii")) 

base_dir = r"C:\Users\jwu191\Desktop\Projects\AD_fitting\processed_data"
patient_name = "AD_87"

# check_geometry(patient_name)
reg(patient_name)

# print(EMR_single.GetSize())
# get_geometry(EMR_single, "EMR_single")
# EMR_single_reg = resize(EMR_single, np.array([256, 256, 1, 56]))
# get_geometry(EMR_single_reg, "EMR_single_reg")

# factor_1 = 0.8040000200271606/0.703000009059906
# print(256/factor_1)
# factor_2 = 0.828000009059906/0.7030000090599062
# print(256/factor_2)