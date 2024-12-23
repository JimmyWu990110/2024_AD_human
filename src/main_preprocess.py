
from DataPreprocessing import (formalize_par_rec, query_par_rec_save_info,
                               get_nifti_for_fitting, check_data)
from Mapping import get_T1_T2_map
from MTRasymCalculation import cal_B0_shift_map, cal_APTw
# from utils import check_data, png_to_nifti, filter_img

new_cases_dir = r"C:\Users\jwu191\Desktop\Projects\AD_fitting\raw_data"
processed_cases_dir = r"C:\Users\jwu191\Desktop\Projects\AD_fitting\processed_data"
lut_dir = r"C:\Users\jwu191\Desktop\new_pipeline\idl_rainbow.lut"
patient_name = "AD_86"

# formalize_par_rec(new_cases_dir, patient_name)
# query_par_rec_save_info(new_cases_dir, processed_cases_dir, patient_name)
# get_nifti_for_fitting(new_cases_dir, processed_cases_dir, patient_name)

# TODO: organize the folder before running the following steps! 
# (delete Axial T1, T2, copy EMR, WASSR and HighRes to patient_name_all folder)
get_T1_T2_map(processed_cases_dir, patient_name)
# register(processed_cases_dir, patient_name)  # T1,T2 map, WASSR,EMR dynamics, HighRes -> M0
# check_data(processed_cases_dir, patient_name)

# TODO: draw "skull.nii.gz" before running the following steps!
# cal_B0_shift_map(processed_cases_dir, patient_name)
# cal_APTw(processed_cases_dir, patient_name)

# NOTE: The data processing is finished, "processed_data\patient_name_all\Coronal\"
# run check_data to see if the shapes and geometries are correct, may need to mannually
# register T1_map (224*224) to M0 (256*256) using ITK-SNAP

# check_data(patient_name)

# TODO: The between-offset co-registration is to be added
# T1_to_M0(patient_name, view)
# Zspec_to_M0(patient_name, view, nth_slice)



# NOTE: These are optional
# strip_skull(processed_cases_dir, patient_name, view)
# png_to_nifti(patient_name)
# filter_img(patient_name)




