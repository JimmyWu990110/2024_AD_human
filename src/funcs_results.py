import os
import numpy as np
import pandas as pd
import SimpleITK as sitk
import cv2

from DataLoader import DataLoader
from B0_correction import B0_correction
from Fitting import Fitting
from Visualization import Visualization
from funcs import ROI2filename, remove_large_offset

def _ROI_excelname(method, ROI, single=False):
    if "Lorentz" in method:
        return ROI2filename(ROI)+"_single_fitted_paras.xlsx"
    elif single:
        return ROI2filename(ROI)+"_single_SL_fitted_paras.xlsx"
    else:
        return ROI2filename(ROI)+"_SL_fitted_paras.xlsx"

def print_paras(patient_name, method, ROI, single=False):
    """
    Read fitted results (excel file), print mean+-std for each parameters
    Can be used for both EMR and Lorentz fitting, just input the method
    """
    filename = _ROI_excelname(method, ROI, single)
    print("********", patient_name, method, filename, "********")
    df = pd.read_excel(os.path.join(r"C:\Users\jwu191\Desktop\Projects\AD_fitting\results",
                                    patient_name, method, filename))
    data = np.array(df)
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)
    for i in range(1, mean.shape[0]):
        if df.columns[i] == "T1a" or df.columns[i] == "T1b" or df.columns[i] == "T1obs":
            print(df.columns[i]+":", format(mean[i], ".2f"), "+-", format(std[i], ".2f"), "s")
        elif df.columns[i] == "T2a" or df.columns[i] == "T2obs":
            print(df.columns[i]+":", format(1e3*mean[i], ".2f"), "+-", format(1e3*std[i], ".2f"), "ms")
        elif df.columns[i] == "T2b":
            print(df.columns[i]+":", format(1e6*mean[i], ".2f"), "+-", format(1e6*std[i], ".2f"), "us")
        elif df.columns[i] == "noise" or "magnitude" in df.columns[i]:
            print(df.columns[i]+":", format(100*mean[i], ".2f"), "+-", format(100*std[i], ".2f"), "%")
        elif "NRMSE" in df.columns[i] or "#" in df.columns[i]:
            print(df.columns[i]+":", format(mean[i], ".2f"), "+-", format(std[i], ".2f"), "%")
        elif "center" in df.columns[i] or "width" in df.columns[i] or df.columns[i] == "B0 shift":
            print(df.columns[i]+":", format(mean[i], ".2f"), "+-", format(std[i], ".2f"), "ppm")
        else:
            print(df.columns[i]+":", format(mean[i], ".2f"), "+-", format(std[i], ".2f"))

def save_Zspec(patient_name, method, ROI, single=False):
    """
    Read fitted parameters (from excel file) and Zspec (corrected) of corresponding pxiels
    Calculate the EMR curve based on the parameters and save mean Zspec, Zspec_EMR (in .npy)
    """
    base_dir = r"C:\Users\jwu191\Desktop\Projects\AD_fitting\results"
    filename = _ROI_excelname(method, ROI, single)
    print("********", patient_name, method, filename, "********")
    df_info = pd.read_excel(r"C:\Users\jwu191\Desktop\Projects\AD_fitting\info.xlsx")
    dataloader = DataLoader(patient_name)
    df = pd.read_excel(os.path.join(base_dir, patient_name, method, filename))
    if "single" in filename:
        nth_slice = int(df_info[df_info["patient_name"] == patient_name]["nth_slice"])
        EMR = dataloader.read_EMR_single() # [56, 256, 256]
        WASSR = dataloader.read_WASSR()[:, nth_slice-1, :, :] # [26, 256, 256]
        cords = np.array(df[["x", "y"]])
    else:
        EMR = dataloader.read_EMR() # [24, 15, 256, 256]
        WASSR = dataloader.read_WASSR() # [26, 15, 256, 256]
        cords = np.array(df[["z", "x", "y"]])
    if "MT" in method:
        paras = np.array(df[["R*M0b*T1a", "T2b", "R", "T1a/T2a", "noise"]])
    if "BM" in method:
        paras = np.array(df[["T1a", "T2a", "T2b", "R", "M0b", "noise"]])
    if "Lorentz" in method:
        paras = np.array(df[["APT magnitude", "APT center", "APT width", 
                             "CEST2ppm magnitude", "CEST2ppm center", "CEST2ppm width", 
                             "DS magnitude", "DS width", "MT magnitude", "MT center", 
                             "MT width", "NOE magnitude", "NOE center", "NOE width"]])
    Zspec_all = []
    Zspec_EMR_all = []
    offset = None
    for i in range(paras.shape[0]):
        if "single" in filename:
            Zspec = EMR[:, cords[i][0], cords[i][1]]
            WASSR_Spec = WASSR[:, cords[i][0], cords[i][1]]
        else:
            Zspec = EMR[:, cords[i][0], cords[i][1], cords[i][2]]
            WASSR_Spec = WASSR[:, cords[i][0], cords[i][1], cords[i][2]]
        offset, Zspec = dataloader.Zspec_preprocess_onepixel(Zspec, fitting=False)
        correction = B0_correction(offset, Zspec, WASSR_Spec)
        Zspec = correction.correct() # B0 corection
        Zspec_all.append(Zspec)
        fitting = Fitting(B1=1.5, lineshape="SL")
        Zspec_EMR = fitting.generate_Zspec(method, offset, paras[i])
        Zspec_EMR_all.append(Zspec_EMR)
    np.save(os.path.join(base_dir, patient_name, method, filename.replace("fitted_paras.xlsx", "Zspec.npy")), 
            np.mean(Zspec_all, axis=0))
    np.save(os.path.join(base_dir, patient_name, method, filename.replace("fitted_paras.xlsx", "Zspec_EMR.npy")), 
            np.mean(Zspec_EMR_all, axis=0))

def plot_fitting(patient_name, method, ROI, single=False):
    path = os.path.join(r"C:\Users\jwu191\Desktop\Projects\AD_fitting\results",
                        patient_name, method)
    if "Lorentz" in method:
        suffix = "_single_Zspec"
    elif single:
        suffix = "_single_SL_Zspec"
    else:
        suffix = "_SL_Zspec"
    Zspec = np.load(os.path.join(path, ROI2filename(ROI)+suffix+".npy"))
    Zspec_EMR = np.load(os.path.join(path, ROI2filename(ROI)+suffix+"_EMR.npy"))
    dataloader = DataLoader(patient_name)
    offset = dataloader.get_offset(single=single)
    offset_mid, Zspec_mid = remove_large_offset(offset, Zspec)
    offset_mid, Zspec_EMR_mid = remove_large_offset(offset, Zspec_EMR)
    visualization = Visualization()
    visualization.plot_fitting_1_1(offset_mid, Zspec_mid, Zspec_EMR_mid, 
                                   title=patient_name+" "+method+" "+ROI2filename(ROI))
    visualization.plot_Zspec_diff(offset_mid, Zspec_EMR_mid-Zspec_mid,
                                  title=patient_name+" "+method+" "+ROI2filename(ROI))

def compare_EMR_2(patient_name, method):
    path = os.path.join(r"C:\Users\jwu191\Desktop\Projects\AD_fitting\results",
                        patient_name, method)
    fname_1 = "hippocampus_single_SL_Zspec"
    fname_2 = "WM_single_SL_Zspec"
    labels=["hippocampus", "WM"]
    Zspec_1 = np.load(os.path.join(path, fname_1+".npy"))
    Zspec_EMR_1 = np.load(os.path.join(path, fname_1+"_EMR.npy"))
    Zspec_2 = np.load(os.path.join(path, fname_2+".npy"))
    Zspec_EMR_2 = np.load(os.path.join(path, fname_2+"_EMR.npy"))
    dataloader = DataLoader(patient_name)
    offset = dataloader.get_offset_single(fitting=False)
    offset_mid, Zspec_mid_1 = remove_large_offset(offset, Zspec_1)
    offset_mid, Zspec_EMR_mid_1 = remove_large_offset(offset, Zspec_EMR_1)
    offset_mid, Zspec_mid_2 = remove_large_offset(offset, Zspec_2)
    offset_mid, Zspec_EMR_mid_2 = remove_large_offset(offset, Zspec_EMR_2)
    visualization = Visualization()
    visualization.plot_Zspec_diff_2(offset_mid, Zspec_EMR_mid_1-Zspec_mid_1,
                                   Zspec_EMR_mid_2-Zspec_mid_2, 
                                   labels=labels, 
                                   title=patient_name+" "+method)

def compare_EMR_4(patient_name, method):
    path = os.path.join(r"C:\Users\jwu191\Desktop\Projects\AD_fitting\results",
                        patient_name, method)
    suffix = "_SL_Zspec"
    fname_1 = "hippocampus" + suffix
    fname_2 = "thalamus" + suffix
    fname_3 = "WM" + suffix
    fname_4 = "pons" + suffix
    labels=["hippocampus", "thalamus", "WM", "pons"]
    Zspec_1 = np.load(os.path.join(path, fname_1+".npy"))
    Zspec_EMR_1 = np.load(os.path.join(path, fname_1+"_EMR.npy"))
    Zspec_2 = np.load(os.path.join(path, fname_2+".npy"))
    Zspec_EMR_2 = np.load(os.path.join(path, fname_2+"_EMR.npy"))
    Zspec_3 = np.load(os.path.join(path, fname_3+".npy"))
    Zspec_EMR_3 = np.load(os.path.join(path, fname_3+"_EMR.npy"))
    Zspec_4 = np.load(os.path.join(path, fname_4+".npy"))
    Zspec_EMR_4 = np.load(os.path.join(path, fname_4+"_EMR.npy"))
    dataloader = DataLoader(patient_name)
    offset = dataloader.get_offset(fitting=False, suffix=suffix)
    offset_mid, Zspec_mid_1 = remove_large_offset(offset, Zspec_1)
    offset_mid, Zspec_EMR_mid_1 = remove_large_offset(offset, Zspec_EMR_1)
    offset_mid, Zspec_mid_2 = remove_large_offset(offset, Zspec_2)
    offset_mid, Zspec_EMR_mid_2 = remove_large_offset(offset, Zspec_EMR_2)
    offset_mid, Zspec_mid_3 = remove_large_offset(offset, Zspec_3)
    offset_mid, Zspec_EMR_mid_3 = remove_large_offset(offset, Zspec_EMR_3)
    offset_mid, Zspec_mid_4 = remove_large_offset(offset, Zspec_4)
    offset_mid, Zspec_EMR_mid_4 = remove_large_offset(offset, Zspec_EMR_4)
    visualization = Visualization()
    visualization.plot_Zspec_diff_4(offset_mid, Zspec_EMR_mid_1-Zspec_mid_1,
                                    Zspec_EMR_mid_2-Zspec_mid_2,
                                    Zspec_EMR_mid_3-Zspec_mid_3,
                                    Zspec_EMR_mid_4-Zspec_mid_4,
                                    labels=labels, 
                                    title=patient_name+" "+method)

def _read_lookup_table(lut_dir):
    lookup_table = []
    with open(lut_dir, "r") as lut_file:
        lut_lines = lut_file.read().split('\n')
        for line in lut_lines:
            if len(line) > 0:
                line_nums = [int(i) for i in line.split('\t')]
    #             print(line_nums)
                lookup_table.append(line_nums)
    # 4 columns: Gray - R - G - B
    lookup_table = np.array(lookup_table) 
    return lookup_table

def _gray_to_idl(gray_img, lookup_table):
    for i in range(gray_img.shape[0]):
        for j in range(gray_img.shape[1]):
            g_val = gray_img[i,j,2]
            gray_img[i,j,0] = lookup_table[g_val,3] # blue channel
            gray_img[i,j,1] = lookup_table[g_val,2] # green channel
            gray_img[i,j,2] = lookup_table[g_val,1]
    return gray_img

def _generate_map_helper(path, arr, name, scale=None):
    if scale:
        arr[arr > scale[1]] = scale[1]
        arr = np.interp(arr, scale, (0, 255))
    else:
        arr = np.interp(arr, (arr.min(), arr.max()), (0, 255))
    cv2.imwrite(os.path.join(path, name+'.png'), np.flip(arr, 0))
    if "M0" not in name:
        img = cv2.imread(os.path.join(path, name+'.png'), 1)
        lookup_table = _read_lookup_table(r"C:\Users\jwu191\Desktop\Data\idl_rainbow.lut")
        img = _gray_to_idl(img, lookup_table)
        cv2.imwrite(os.path.join(path, name+'.png'), img) 

def generate_map(patient_name, method, filename, max_pow=5):
    """
    Read fitted results (excel file), print mean+-std for each parameters
    Can be used for both EMR and Lorentz fitting, just input the method
    """
    path = os.path.join(r"C:\Users\jwu191\Desktop\Projects\AD_fitting\results", 
                        patient_name, method)
    df = pd.read_excel(os.path.join(path, filename))
    x = np.array(df["x"])
    y = np.array(df["y"])
    if "Lorentz" in method:
        APT_map = np.zeros((256, 256))
        NOE_map = np.zeros((256, 256))
        MT_map = np.zeros((256, 256))
        DS_map = np.zeros((256, 256))
        MT_width_map = np.zeros((256, 256))
        DS_width_map = np.zeros((256, 256))
        APT = np.array(df["APT magnitude"])
        NOE = np.array(df["NOE magnitude"])
        MT = np.array(df["MT magnitude"])
        DS = np.array(df["DS magnitude"])
        MT_width = np.array(df["MT width"])
        DS_width = np.array(df["DS width"])
        for i in range(x.shape[0]):
            APT_map[x[i]][y[i]] = APT[i]
            NOE_map[x[i]][y[i]] = NOE[i]
            MT_map[x[i]][y[i]] = MT[i]
            DS_map[x[i]][y[i]] = DS[i]
            MT_width_map[x[i]][y[i]] = MT_width[i]
            DS_width_map[x[i]][y[i]] = DS_width[i]
        print("APT: (%)", 100*np.min(APT), "~", 100*np.max(APT))
        print("NOE: (%)", 100*np.min(NOE), "~", 100*np.max(NOE))
        print("MT: (%)", 100*np.min(MT), "~", 100*np.max(MT))
        print("DS: (%)", 100*np.min(DS), "~", 100*np.max(DS))
        _generate_map_helper(path, APT_map, patient_name+"_APT_map", scale=(0,0.1))
        _generate_map_helper(path, NOE_map, patient_name+"_NOE_map", scale=(0,0.1))
        _generate_map_helper(path, MT_map, patient_name+"_MT_map", scale=(0,1))
        _generate_map_helper(path, DS_map, patient_name+"_DS_map", scale=(0,1))
        _generate_map_helper(path, MT_width_map, patient_name+"_MT_width_map", 
                             scale=(0,80))
        _generate_map_helper(path, DS_width_map, patient_name+"_DS_width_map", 
                             scale=(0,5))
    elif "MT" in method or "BM" in method:
        min_ = -5
        APT_pow_map = min_ * np.ones((256, 256))
        NOE_pow_map = min_ * np.ones((256, 256))
        APTw_map = min_ * np.ones((256, 256))
        APT_pow = np.array(df["APT#"])
        NOE_pow = np.array(df["NOE#"])
        APTw = np.array(df["APTw"])
        for i in range(x.shape[0]):
            APT_pow_map[x[i]][y[i]] = APT_pow[i]
            NOE_pow_map[x[i]][y[i]] = NOE_pow[i]
            APTw_map[x[i]][y[i]] = APTw[i]
        print("APT#: (%)", np.min(APT_pow), "~", np.max(APT_pow))
        print("NOE#: (%)", np.min(NOE_pow), "~", np.max(NOE_pow))
        print("APTw: (%)", np.min(APTw), "~", np.max(APTw))
        # normalization
        APT_pow_map[APT_pow_map < -5] = -5
        NOE_pow_map[NOE_pow_map < -5] = -5
        APTw_map[APTw_map < -5] = -5
        APT_pow_map[APT_pow_map > max_pow] = max_pow
        NOE_pow_map[NOE_pow_map > max_pow] = max_pow
        APTw_map[APTw_map > 5] = 5
        _generate_map_helper(path, APT_pow_map, filename.replace("_fitted_paras.xlsx", "_APT_pow"))
        _generate_map_helper(path, NOE_pow_map, filename.replace("_fitted_paras.xlsx", "_NOE_pow"))
        _generate_map_helper(path, APTw_map, filename.replace("_fitted_paras.xlsx", "_APTw"))

def generate_M0_map(patient_name, nth_slice):
    dataloader = DataLoader(patient_name)
    M0 = dataloader.read_M0()[nth_slice-1]
    skull = dataloader.read_skull()[nth_slice-1]
    min_ = np.min(M0)
    M0[skull == 0] = min_
    path = os.path.join(r"C:\Users\jwu191\Desktop\Projects\AD_fitting\results", patient_name)
    _generate_map_helper(path, M0, patient_name+"_M0_slice_"+str(nth_slice))
    
    
    
    
    
    
    
    
    




