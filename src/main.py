import os
import shutil

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import SimpleITK as sitk

from DataPreprocessing import formalize_par_rec, query_par_rec_save_info, sepcify_sequence_ids
from Mapping import get_nifti_for_fitting, get_T1_T2_map, get_highRes, plot_highRes
from MTRasymCalculation import get_M0, cal_MTRasym, strip_skull

from DataLoader import DataLoader
from Visualization import Visualization
from EMR_fitting import EMR_fitting
from Registration import T1_to_M0, Zspec_to_M0

from utils import fit_by_pos_24, fit_by_pos_56, fit_by_ROI_24, fit_by_ROI_56, generate_map, get_vals, analysis, fit_by_mask, check_data, fit_oneslice


new_cases_dir = r"C:\Users\jwu191\Desktop\Projects\AD_fitting\raw_data"
processed_cases_dir = r"C:\Users\jwu191\Desktop\Projects\AD_fitting\processed_data"
lut_dir = r"C:\Users\jwu191\Desktop\new_pipeline\idl_rainbow.lut"
patient_name = "AD_70"
view = "Coronal"

# formalize_par_rec(new_cases_dir, patient_name)
# query_par_rec_save_info(new_cases_dir, processed_cases_dir, patient_name)
# get_nifti_for_fitting(new_cases_dir, processed_cases_dir, patient_name)

# get_highRes(new_cases_dir, processed_cases_dir, patient_name)
# plot_highRes(patient_name)
# get_T1_T2_map(processed_cases_dir, patient_name, view)
# seq_dict = sepcify_sequence_ids(MTR_1p5uT_id=6, WASSR_EMR_id=7, MTR_2uT_id=-1)
# get_M0(processed_cases_dir, patient_name, view, seq_dict)
# cal_MTRasym(processed_cases_dir, patient_name, view, seq_dict)

# strip_skull(processed_cases_dir, patient_name, view)

nth_slice = 8
x = 125
y = 125
lineshape = "SL"

# fit_by_pos_24(patient_name, view, nth_slice, x, y, lineshape)
# fit_by_pos_56(patient_name, nth_slice, x, y, lineshape)
  
# fit_by_ROI_24(patient_name, lineshape, nth_slice)
# fit_by_ROI_56(patient_name, lineshape, nth_slice)

# check_data("AD_60")
patient_list = ["AD_05", "AD_06", "AD_07", "AD_29", "AD_30", "AD_31"]
for patient_name in patient_list:
    for i in range(1, 5):
        fit_by_mask(patient_name, lineshape, ROI=i)

# fit_oneslice(patient_name, lineshape, nth_slice)

# analysis(patient_name, suffix="_two_step.xlsx")
# generate_map(patient_name, nth_slice)
# get_vals(["AD_51", "AD_52", "AD_53", "AD_54", "AD_55", "AD_56", "AD_57",
#           "AD_59", "AD_63", "AD_64", "AD_67", "AD_72", "AD_74"])

# compare_B(patient_name, view)

# T1_to_M0(patient_name, view)
# Zspec_to_M0(patient_name, view, nth_slice)

# dataLoader = DataLoader(patient_name, view)
# dataLoader.test_coordinate(nth_slice)

# fit_by_ROI_24("AD_74", lineshape, 5, 6)







