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

""" NOTE: Single voxel tests using phantom and human data given coordinates """
# fit_by_pixel(patient_name="Phantom_20230620", nth_slice=8, x=128, y=128, 
#              method="BMS_EMR")
# fit_by_pixel(patient_name="AD_75", nth_slice=8, x=102, y=136, 
#              method="BMS")

method = "Lorentz_5pool"
# patient_list = ["AD_76", "AD_77", "AD_79", "AD_81", "AD_82", "AD_84", "AD_85"]
# for patient_name in patient_list:
#     for ROI in [1, 3]:
#         fit_by_ROI_mask_single(patient_name=patient_name, ROI=ROI, method=method)

# patient_name = "AD_84"
# single = True
# for ROI in [1]:
#     print_paras(patient_name, method, ROI, single)
#     save_Zspec(patient_name, method, ROI, single)
#     plot_fitting(patient_name, method, ROI, single)

# compare_EMR_4(patient_name, method)

patient_name = "AD_82"
# nth_slice = 4
# fit_by_skull_mask(patient_name=patient_name, nth_slice=nth_slice, method=method)

# filename = patient_name+"_slice_"+str(nth_slice)+"_fitted_paras.xlsx"
filename = "map_single_fitted_paras.xlsx"
generate_map(patient_name=patient_name, method=method, filename=filename, 
             max_pow=10)
# generate_M0_map(patient_name=patient_name, nth_slice=nth_slice)



