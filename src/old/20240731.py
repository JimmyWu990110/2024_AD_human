import os
import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt
import multiprocessing
from multiprocessing import Pool
from itertools import repeat
from os import getpid

def check_Zspec(processed_cases_dir, patient_name, nth_slice, x, y):
    base_dir = os.path.join(processed_cases_dir, patient_name+"_all")
    EMR = sitk.ReadImage(os.path.join(base_dir, "EMR.nii"))
    EMR_reg = sitk.ReadImage(os.path.join(base_dir, "EMR_reg.nii"))
    EMR_corrected = sitk.ReadImage(os.path.join(base_dir, "EMR_corrected.nii"))
    offset = np.array([np.inf, 80, 60, 40, 30, 20, 10, 8, 4, -4, 3.5, -3.5, 3.5, -3.5, 
                       3, -3, 2.5, -2.5, 2, -2, 2, -2, 1.5, -1.5])
    Zspec = sitk.GetArrayFromImage(EMR)[:, nth_slice-1, y-1, x-1]
    Zspec_reg = sitk.GetArrayFromImage(EMR_reg)[:, nth_slice-1, y-1, x-1]
    Zspec_corrected = sitk.GetArrayFromImage(EMR_corrected)[:, nth_slice-1, y-1, x-1]
    plt.ylim((0, 1))
    plt.scatter(offset[8:], Zspec[8:]/Zspec[0], marker="x", label="original")
    plt.scatter(offset[8:], Zspec_reg[8:]/Zspec_reg[0], marker="x", label="registered")
    plt.scatter(offset[8:], Zspec_corrected[8:]/Zspec_corrected[0], marker="x", label="corrected")
    plt.gca().invert_xaxis()  # invert x-axis
    plt.legend()
    plt.show()

def combine_4d_img(processed_cases_dir, patient_name, seq):
    assert seq == "EMR" or seq == "WASSR"
    base_dir = os.path.join(processed_cases_dir, patient_name+"_all")
    n = 24
    if seq == "WASSR":
        n = 26
    f_list = []
    for i in range(n):
        f_list.append(os.path.join(base_dir, seq, str(i)+"_reg.nii"))
    res = sitk.JoinSeries([sitk.ReadImage(f) for f in f_list])
    sitk.WriteImage(res, os.path.join(base_dir, seq+"_reg.nii"))
    print(seq, "combined")

def check_reg(processed_cases_dir, patient_name):
    base_dir = os.path.join(processed_cases_dir, patient_name+"_all")
    for f in os.listdir(os.path.join(base_dir, "EMR")):
        if f.endswith("_reg.nii"):
            img = sitk.ReadImage(os.path.join(base_dir, "EMR", f))
            print(f, img.GetOrigin(), img.GetSpacing(), img.GetDirection())
    for f in os.listdir(os.path.join(base_dir, "WASSR")):
        if f.endswith("_reg.nii"):
            img = sitk.ReadImage(os.path.join(base_dir, "WASSR", f))
            print(f, img.GetOrigin(), img.GetSpacing(), img.GetDirection())
    combine_4d_img(processed_cases_dir, patient_name, "WASSR")
    combine_4d_img(processed_cases_dir, patient_name, "EMR")
    
def filter_APTw(patient_name):
    base_dir = os.path.join(r"C:\Users\jwu191\Desktop\backup\AD_fitting\20240730\processed_data", 
                            patient_name+"_all", "Coronal")
    img = sitk.ReadImage(os.path.join(base_dir, "APTw.nii"))
    skull = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(base_dir, "skull.nii.gz")))
    arr = sitk.GetArrayFromImage(img)
    arr[skull == 0] = -5
    new_img = sitk.GetImageFromArray(arr)
    new_img.SetOrigin(img.GetOrigin())
    new_img.SetSpacing(img.GetSpacing())
    new_img.SetDirection(img.GetDirection())
    sitk.WriteImage(new_img, os.path.join(base_dir, "APTw_filtered.nii"))
    
def f(i):
    print("I'm process", getpid())

def test_multi_processing():
    slices = np.arange(1, 5)
    with Pool() as pool:
        pool.map(f, slices)

processed_cases_dir = r"C:\Users\jwu191\Desktop\Projects\AD_fitting\processed_data"
patient_name = "AD_64"
nth_slice = 8
x = 154
y = 142


# print(multiprocessing.cpu_count())
# filter_APTw(patient_name)
# check_reg(processed_cases_dir, patient_name)

test_multi_processing()




