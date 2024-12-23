import os
import shutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import SimpleITK as sitk

from utils import fit_by_pixel, fit_by_ROI_mask, fit_by_ROI_mask_single, fit_by_skull_mask
from funcs_results import (print_paras, save_Zspec, plot_fitting, 
                           compare_EMR_2, compare_EMR_4, generate_map, generate_M0_map)

new_cases_dir = r"C:\Users\jwu191\Desktop\Projects\AD_fitting\raw_data"
processed_cases_dir = r"C:\Users\jwu191\Desktop\Projects\AD_fitting\processed_data"
lut_dir = r"C:\Users\jwu191\Desktop\new_pipeline\idl_rainbow.lut"

method = "BMS_EMR_v2"
patient_list = ["AD_86", "AD_83", "AD_80", "AD_78"]
for patient_name in patient_list:
    for ROI in [1,3]:
        fit_by_ROI_mask_single(patient_name=patient_name, ROI=ROI, method=method)

# patient_name = "AD_45"
# for ROI in [1, 2, 3, 4]:
#     print_paras(patient_name, method, ROI)
#     save_Zspec(patient_name, method, ROI)
    # plot_fitting(patient_name, method, ROI)

# compare_EMR_4(patient_name, method)

# patient_name = "AD_83"
# nth_slice = 5
# fit_by_skull_mask(patient_name=patient_name, nth_slice=nth_slice, method=method)

# filename = patient_name+"_slice_"+str(nth_slice)+"_fitted_paras.xlsx"
# generate_map(patient_name=patient_name, method=method, filename=filename, max_pow=10)
# generate_M0_map(patient_name=patient_name, nth_slice=nth_slice)





