import os
import shutil
import cv2

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import SimpleITK as sitk

from DataPreprocessing import formalize_par_rec, query_par_rec_save_info, sepcify_sequence_ids
from Mapping import get_nifti_for_fitting, get_T1_T2_map
from MTRasymCalculation import get_M0, cal_MTRasym, strip_skull

from DataLoader import DataLoader
from Visualization import Visualization
from EMR_fitting import EMR_fitting, EMR_fitting_A
from Simulation import Simulation
from Evaluation import Evaluation
from B0Correction import B0Correction 


def select_offset(offset, Zspec, flag=0):
    if flag == 1:
        offset = offset[0:7]
        Zspec = Zspec[0:7]
    elif flag == 2:
        offset = offset[[0, 1, 2, 3, 4, 5, 6, 12]] # 1.5 ppm
        Zspec = Zspec[[0, 1, 2, 3, 4, 5, 6, 12]]
    return [128*offset, Zspec]

def select_offset_56(offset, Zspec, flag=0):
    if flag == 1:
        offset = offset[0:7]
        Zspec = Zspec[0:7]
    elif flag == 2:
        offset = offset[[0, 1, 2, 3, 4, 5, 6, 21]] # 0.5 ppm
        Zspec = Zspec[[0, 1, 2, 3, 4, 5, 6, 21]]
    elif flag == 3:
        offset = offset[[0, 1, 2, 3, 4, 5, 6, 47, 48, 49, 50, 51, 52, 53]]
        Zspec = Zspec[[0, 1, 2, 3, 4, 5, 6, 47, 48, 49, 50, 51, 52, 53]]
    elif flag == 4:
        offset = offset[[0, 1, 2, 3, 4, 5, 6, 23, 30, 47, 48, 49, 50, 51, 52, 53]] # 0.25 ppm
        Zspec = Zspec[[0, 1, 2, 3, 4, 5, 6, 23, 30, 47, 48, 49, 50, 51, 52, 53]]
    elif flag == 5:
        offset = np.concatenate((offset[0:23], offset[31:]))
        Zspec = np.concatenate((Zspec[0:23], Zspec[31:]))
    # print(offset)
    return [128*offset, Zspec]

def fit_by_pos_24(patient_name, view, nth_slice, x, y, lineshape):
    dataLoader = DataLoader(patient_name, view)
    visualization = Visualization()
    evaluation = Evaluation()
    correction = B0Correction()
    Zspec = dataLoader.get_Zspec_by_coordinate_24(nth_slice-1, y-1, x-1)
    offset, Zspec = dataLoader.Zspec_preprocess_onepixel(Zspec)
    # visualization.plot_Zspec(offset[7:], Zspec[7:])
    wassr_spec = dataLoader.read_WASSR()[:, nth_slice-1, round((y-1)/2), round((x-1)/2)]
    B0_shift = correction.cal_B0_shift_onepixel(wassr_spec)
    print("B0 shift (ppm):", B0_shift)
    Zspec = correction.correct_onepixel(offset, Zspec, B0_shift) # WASSR correction
    T1 = dataLoader.read_T1_map()[nth_slice-1, y-1, x-1]
    T2 = dataLoader.read_T2_map()[nth_slice-1, y-1, x-1]
    print("T1, T2, ratio:", T1, T2, T1/T2)
    fitting = EMR_fitting(T1w_obs=T1, T2w_obs=T2)
    fitting.set_lineshape(lineshape)
    fitted_paras, y_estimated = fitting.fit(*select_offset(offset, Zspec, 0), False)
    paras_dict = fitting.cal_paras()
    print(paras_dict)
    y_estimated = fitting.generate_Zpsec(offset*128, fitted_paras)
    visualization.plot_2_Zspec(offset, Zspec, y_estimated, labels=["real", "fitted"])
    visualization.plot_2_Zspec(offset[7:], Zspec[7:], y_estimated[7:], labels=["real", "fitted"])
    visualization.plot_Zspec_diff(offset[7:], y_estimated[7:]-Zspec[7:])
    evaluation.evaluate(Zspec, y_estimated, show=True)
    evaluation.evaluate(Zspec[7:], y_estimated[7:], show=True)
    evaluation.get_metrics(offset, Zspec, y_estimated)
    
    
    """
    first use all the offsets for fitting, then use the fitted parameters to 
    get x0 and lb, ub
    """
    # x0 = fitted_paras
    # lb = 0.1 * x0
    # ub = 10 * x0
    # fitting = EMR_fitting(T1w_obs=paras_dict["T1w"], T2w_obs=paras_dict["T2w"])
    # fitting.set_x0(x0)
    # fitting.set_lb(lb)
    # fitting.set_ub(ub)
    # fitted_paras, y_estimated = fitting.fit(*select_offset(offset, Zspec, 2), True)
    # paras_dict = fitting.cal_paras()
    # print(paras_dict)
    # y_estimated = fitting.generate_Zpsec(offset*128, fitted_paras)
    # visualization.plot_2_Zspec(offset, Zspec, y_estimated, labels=["real", "fitted"])
    # visualization.plot_2_Zspec(offset[7:], Zspec[7:], y_estimated[7:], labels=["real", "fitted"])
    # visualization.plot_Zspec_diff(offset[7:], y_estimated[7:]-Zspec[7:])
    

def fit_by_pos_56(patient_name, nth_slice, x, y, lineshape):
    dataLoader = DataLoader(patient_name, "Coronal")
    visualization = Visualization()
    evaluation = Evaluation()
    correction = B0Correction()
    Zspec = dataLoader.get_Zspec_by_coordinate_56(y-1, x-1)
    offset, Zspec = dataLoader.Zspec_preprocess_onepixel(Zspec, 56)
    # visualization.plot_Zspec(offset, Zspec)
    # visualization.plot_Zspec(offset[7:-7], Zspec[7:-7])
    wassr_spec = dataLoader.read_WASSR()[:, nth_slice-1, round((y-1)/2), round((x-1)/2)]
    B0_shift = correction.cal_B0_shift_onepixel(wassr_spec)
    print("B0 shift (ppm):", B0_shift)
    Zspec = correction.correct_onepixel(offset, Zspec, B0_shift) # WASSR correction
    T1 = dataLoader.read_T1_map()[nth_slice-1, y-1, x-1]
    T2 = dataLoader.read_T2_map()[nth_slice-1, y-1, x-1]
    print("T1, T2, ratio:", T1, T2, T1/T2)
    fitting = EMR_fitting(T1w_obs=T1, T2w_obs=T2)
    fitting.set_lineshape(lineshape)
    fitted_paras, y_estimated = fitting.fit(*select_offset_56(offset, Zspec, 0), True)
    paras_dict = fitting.cal_paras()
    print(paras_dict)
    y_estimated = fitting.generate_Zpsec(offset*128, fitted_paras)
    visualization.plot_2_Zspec(offset, Zspec, y_estimated, labels=["real", "fitted"])
    visualization.plot_2_Zspec(offset[7:-7], Zspec[7:-7], y_estimated[7:-7], labels=["real", "fitted"])
    visualization.plot_Zspec_diff(offset[7:-7], y_estimated[7:-7]-Zspec[7:-7])
    evaluation.evaluate(Zspec, y_estimated, show=True)
    evaluation.evaluate(Zspec[7:-7], y_estimated[7:-7], show=True)
    evaluation.get_metrics(offset, Zspec, y_estimated)

    
def fit_by_ROI_24(patient_name, lineshape, nth_slice):
    dataLoader = DataLoader(patient_name, "Coronal")
    evaluation = Evaluation()
    correction = B0Correction()
    skull = dataLoader.read_skull()
    EMR = dataLoader.read_EMR() # [24, 15, 256, 256]
    T1 = dataLoader.read_T1_map() # [15, 256, 256]
    T2 = dataLoader.read_T2_map() # [15, 256, 256]
    WASSR = dataLoader.read_WASSR() # [26, 15, 128, 128]
    counter = 0
    data = []
    for j in range(skull.shape[1]):
        for k in range(skull.shape[2]):
            if skull[nth_slice-1, j, k] == 0 or T1[nth_slice-1, j, k]*T2[nth_slice-1, j, k] <= 1e-8:
                continue
            Zspec = EMR[:, nth_slice-1, j, k]
            offset, Zspec = dataLoader.Zspec_preprocess_onepixel(Zspec)
            fitting = EMR_fitting(T1w_obs=T1[nth_slice-1, j, k], T2w_obs=T2[nth_slice-1, j, k])
            fitting.set_lineshape(lineshape)   
            wassr_spec = WASSR[:, nth_slice-1, round(j/2), round(k/2)]
            B0_shift = correction.cal_B0_shift_onepixel(wassr_spec)
            Zspec = correction.correct_onepixel(offset, Zspec, B0_shift)
            fitted_paras, _ = fitting.fit(*select_offset(offset, Zspec, 0), False)
            # start two step fitting (coarse to fine)
            paras_dict = fitting.cal_paras()
            x0 = np.array([fitted_paras[0], fitted_paras[1], fitted_paras[3]])
            lb = 0.4 * x0
            ub = 2.5 * x0
            fitting = EMR_fitting_A(T1w=paras_dict["T1w"], T2w=paras_dict["T2w"])
            fitting.set_x0(x0)
            fitting.set_lb(lb)
            fitting.set_ub(ub)
            fitted_paras, _ = fitting.fit(*select_offset(offset, Zspec, 0), True)
            # end of two step fitting (if 1 step, comment the middle lines)
            y_estimated = fitting.generate_Zpsec(offset*128, fitted_paras)
            diff = y_estimated - Zspec
            APT_pow = 100*diff[np.where(offset==3.5)][0] # %
            NOE_pow = 100*diff[np.where(offset==-3.5)][0] # %
            APTw = APT_pow - NOE_pow
            R, NRMSE = evaluation.evaluate(Zspec, y_estimated)
            R_mid, NRMSE_mid = evaluation.evaluate(Zspec[7:], y_estimated[7:])
            # data.append([nth_slice-1, j, k, *fitted_paras, APTw, APT_pow, NOE_pow, 
            #               R, NRMSE, R_mid, NRMSE_mid])
            # For two step only!
            data.append([nth_slice-1, j, k, fitted_paras[0], fitted_paras[1], 
                          paras_dict["T1w"]/paras_dict["T2w"], fitted_paras[2], 
                          APTw, APT_pow, NOE_pow, R, NRMSE, R_mid, NRMSE_mid])
            counter += 1
            if counter % 100 == 0:
                print(counter)
            # if counter > 10:
            #     break # can only break one loop
    columns = ["z", "x", "y", "R", "R*M0m*T1w", "T1w/T2w", "T2m", "APTw", "APT#", "NOE#",
               "R2", "NRMSE", "R2_mid", "NRMSE_mid"]
    df = pd.DataFrame(data=data, columns=columns)
    output_dir = os.path.join(r"C:\Users\jwu191\Desktop\Projects\AD_fitting\results", patient_name)
    os.makedirs(output_dir, exist_ok=True)
    df.to_excel(os.path.join(output_dir, patient_name+"_MTC_24_Unconstrained_G.xlsx"))

def fit_by_ROI_56(patient_name, lineshape, nth_slice):
    dataLoader = DataLoader(patient_name, "Coronal")
    evaluation = Evaluation()
    correction = B0Correction()
    skull = dataLoader.read_skull()
    EMR = dataLoader.read_Zspec_single() # [56, 256, 256]
    T1 = dataLoader.read_T1_map() # [15, 256, 256]
    T2 = dataLoader.read_T2_map() # [15, 256, 256]
    WASSR = dataLoader.read_WASSR() # [26, 15, 128, 128]
    counter = 0
    data = []
    for j in range(skull.shape[1]):
        for k in range(skull.shape[2]):
            if skull[nth_slice-1, j, k] == 0 or T1[nth_slice-1, j, k]*T2[nth_slice-1, j, k] <= 1e-8:
                continue
            Zspec = EMR[:, j, k]
            offset, Zspec = dataLoader.Zspec_preprocess_onepixel(Zspec, 56)
            fitting = EMR_fitting(T1w_obs=T1[nth_slice-1, j, k], T2w_obs=T2[nth_slice-1, j, k])
            fitting.set_lineshape(lineshape)   
            wassr_spec = WASSR[:, nth_slice-1, round(j/2), round(k/2)]
            B0_shift = correction.cal_B0_shift_onepixel(wassr_spec)
            Zspec = correction.correct_onepixel(offset, Zspec, B0_shift)
            fitted_paras, _ = fitting.fit(*select_offset(offset, Zspec, 0), True)
            y_estimated = fitting.generate_Zpsec(offset*128, fitted_paras)
            diff = y_estimated - Zspec
            APT_pow = 100*diff[np.where(offset==3.5)][0] # %
            NOE_pow = 100*diff[np.where(offset==-3.5)][0] # %
            APTw = APT_pow - NOE_pow
            R, NRMSE = evaluation.evaluate(Zspec, y_estimated)
            R_mid, NRMSE_mid = evaluation.evaluate(Zspec[7:-7], y_estimated[7:-7])
            data.append([nth_slice-1, j, k, *fitted_paras, APTw, APT_pow, NOE_pow,
                         R, NRMSE, R_mid, NRMSE_mid])
            counter += 1
            if counter % 100 == 0:
                print(counter)
    columns = ["z", "x", "y", "R", "R*M0m*T1w", "T1w/T2w", "T2m", "APTw", "APT#", "NOE#",
               "R2", "NRMSE", "R2_mid", "NRMSE_mid"]
    df = pd.DataFrame(data=data, columns=columns)
    output_dir = os.path.join(r"C:\Users\jwu191\Desktop\Projects\AD_fitting\results", patient_name)
    os.makedirs(output_dir, exist_ok=True)
    df.to_excel(os.path.join(output_dir, patient_name+"_MTC_56.xlsx"))

def read_lookup_table(lut_dir):
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

def gray_to_idl(gray_img, lookup_table):
    for i in range(gray_img.shape[0]):
        for j in range(gray_img.shape[1]):
            g_val = gray_img[i,j,2]
            # blue channel
            gray_img[i,j,0] = lookup_table[g_val,3]
            # green channel
            gray_img[i,j,1] = lookup_table[g_val,2]
            gray_img[i,j,2] = lookup_table[g_val,1]
    return gray_img

def generate_map_helper(path, arr, name):
    arr = np.interp(arr, (arr.min(), arr.max()), (0, 255))
    cv2.imwrite(os.path.join(path, name+'.png'), np.flip(arr, 0))
    if "M0" not in name:
        img = cv2.imread(os.path.join(path, name+'.png'), 1)
        lookup_table = read_lookup_table(r"C:\Users\jwu191\Desktop\Data\idl_rainbow.lut")
        img = gray_to_idl(img, lookup_table)
        cv2.imwrite(os.path.join(path, name+'.png'), img)  

def generate_map(patient_name, nth_slice):
    filename = "_slice_"+str(nth_slice)+".xlsx" 
    dataLoader = DataLoader(patient_name, "Coronal")
    df = pd.read_excel(os.path.join(r"C:\Users\jwu191\Desktop\Projects\AD_fitting\results",
                                    patient_name, patient_name+filename))
    data = np.array(df)
    print(data.shape)
    APTw = -10*np.ones((256, 256))
    APT_pow = -10*np.ones((256, 256))
    NOE_pow = -10*np.ones((256, 256))
    M0 = np.zeros((256, 256))
    M0_src = dataLoader.read_M0()
    for i in range(data.shape[0]):
        z = int(data[i][1])
        x = int(data[i][2])
        y = int(data[i][3])
        APT_pow[x][y] = data[i][8]
        NOE_pow[x][y] = data[i][9]
        APTw[x][y] = data[i][10]
        M0[x][y] = M0_src[z][x][y]
    APTw[APTw <= -4] = -4
    APTw[APTw >= 4] = 4
    APT_pow[APT_pow <= -2] = -2
    APT_pow[APT_pow >= 6] = 6
    NOE_pow[NOE_pow <= -2] = -2
    NOE_pow[NOE_pow >= 6] = 6
    path = os.path.join(r"C:\Users\jwu191\Desktop\Projects\AD_fitting\results", patient_name)
    generate_map_helper(path, APTw, patient_name+"_APTw")
    generate_map_helper(path, APT_pow, patient_name+"_APT#")
    generate_map_helper(path, NOE_pow, patient_name+"_NOE#")
    generate_map_helper(path, M0, patient_name+"_M0")

def get_vals_helper(patient_name):
    df = pd.read_excel(os.path.join(r"C:\Users\jwu191\Desktop\Projects\AD_fitting\results",
                                    patient_name, patient_name+"_EMR_24.xlsx"))
    dataLoader = DataLoader(patient_name, "Coronal")
    mask = dataLoader.read_mask()
    APTw = []
    APT_pow = []
    NOE_pow = []
    for i in range(mask.shape[0]):
        for j in range(mask.shape[1]):
            for k in range(mask.shape[2]):
                if mask[i][j][k] == 1:
                    item = df[(df.x == j) & (df.y == k)]
                    APTw.append((float(item["APTw"])))
                    APT_pow.append((float(item["APT#"])))
                    NOE_pow.append((float(item["NOE#"])))
    return np.array(APTw), np.array(APT_pow), np.array(NOE_pow)

def get_vals(patient_list):
    data = []
    columns = ["case", "APTw mean", "APTw std", "APT# mean", "APT# std", "NOE# mean", "NOE# std"]
    for patient_name in patient_list:
        APTw, APT_pow, NOE_pow = get_vals_helper(patient_name)
        print("****", patient_name, "****")
        print("APTw:", np.mean(APTw), np.std(APTw))
        print("APT#:", np.mean(APT_pow), np.std(APT_pow))
        print("NOE#:", np.mean(NOE_pow), np.std(NOE_pow))
        data.append([patient_name, np.mean(APTw), np.std(APTw), np.mean(APT_pow), 
                     np.std(APT_pow), np.mean(NOE_pow), np.std(NOE_pow)])
    df = pd.DataFrame(data=data, columns=columns)
    df.to_excel(os.path.join(r"C:\Users\jwu191\Desktop", "statistics.xlsx"))

def analysis(patient_name, suffix):
    path = os.path.join(r"C:\Users\jwu191\Desktop\Projects\AD_fitting\results",
                        patient_name, patient_name+suffix)
    df = pd.read_excel(path)
    data = np.array(df)
    print(data.shape)
    for i in range(4, data.shape[1]):
        tmp = data[:, i]
        print(np.mean(tmp), np.std(tmp))
        
def fit_by_mask_helper(offset, Zspec, T1, T2, B0_shift, lineshape):
    # print("T1, T2, ratio:", T1, T2, T1/T2)
    evaluation = Evaluation()
    visualization = Visualization()
    # first step
    fitting = EMR_fitting(T1w_obs=T1, T2w_obs=T2)
    fitting.set_lineshape(lineshape) 
    fitted_paras, _ = fitting.fit(*select_offset(offset, Zspec, 0), True)
    paras_dict = fitting.cal_paras()
    # print(paras_dict)
    # print(paras_dict["T1w"]/paras_dict["T2w"])
    y_estimated = fitting.generate_Zpsec(offset*128, fitted_paras)
    # visualization.view_24(offset, Zspec, y_estimated)
    evaluation.evaluate_24(Zspec, y_estimated, show=False)
    evaluation.get_metrics(offset, Zspec, y_estimated)
    # second step
    x0 = np.array([fitted_paras[0], fitted_paras[1], fitted_paras[3]])
    lb = 0.4 * x0
    ub = 2.5 * x0
    fitting = EMR_fitting_A(T1w=paras_dict["T1w"], T2w=paras_dict["T2w"])
    fitting.set_x0(x0)
    fitting.set_lb(lb)
    fitting.set_ub(ub)
    fitted_paras, _ = fitting.fit(*select_offset(offset, Zspec, 1), True)
    paras_dict = fitting.cal_paras()
    # print(paras_dict)
    # print(paras_dict["T1w"]/paras_dict["T2w"])
    y_estimated = fitting.generate_Zpsec(offset*128, fitted_paras)
    # visualization.view_24(offset, Zspec, y_estimated)
    NRMSE, NRMSE_out, NRMSE_mid = evaluation.evaluate_24(Zspec, y_estimated, show=False)
    APT_pow, NOE_pow, APTw, MTR_20ppm = evaluation.get_metrics(offset, Zspec, y_estimated)
    return [fitted_paras[0], fitted_paras[1], paras_dict["T1w"]/paras_dict["T2w"], fitted_paras[2], 
            APT_pow, NOE_pow, APTw, MTR_20ppm, NRMSE, NRMSE_out, NRMSE_mid]
            
def fit_by_mask(patient_name, lineshape, ROI):
    dataLoader = DataLoader(patient_name, "Coronal")
    evaluation = Evaluation()
    correction = B0Correction()
    visualization = Visualization()
    mask = dataLoader.read_mask() # [15, 256, 256]
    EMR = dataLoader.read_EMR() # [24, 15, 256, 256]
    T1_map = dataLoader.read_T1_map() # [15, 256, 256]
    T2_map = dataLoader.read_T2_map() # [15, 256, 256]
    WASSR = dataLoader.read_WASSR() # [26, 15, 128, 128]
    data = []
    z = np.where(mask == ROI)[0]
    x = np.where(mask == ROI)[1]
    y = np.where(mask == ROI)[2]
    print("ROI size:", x.shape[0])
    for i in range(x.shape[0]):
        if T1_map[z[i], x[i], y[i]]*T2_map[z[i], x[i], y[i]] == 0:
            continue
        if i % 100 == 0:
            print("********", i, "********")
        Zspec = EMR[:, z[i], x[i], y[i]]
        offset, Zspec = dataLoader.Zspec_preprocess_onepixel(Zspec) 
        wassr_spec = WASSR[:, z[i], round(x[i]/2), round(y[i]/2)]
        B0_shift = correction.cal_B0_shift_onepixel(wassr_spec)
        Zspec = correction.correct_onepixel(offset, Zspec, B0_shift)
        item = fit_by_mask_helper(offset, Zspec, T1_map[z[i], x[i], y[i]], 
                                  T2_map[z[i], x[i], y[i]], B0_shift, lineshape)
        data.append([z[i], x[i], y[i]]+item)
    columns = ["z", "x", "y", "R", "R*M0m*T1w", "T1w/T2w", "T2m", "APT#", "NOE#", "APTw",
               "MTR_20ppm", "NRMSE", "NRMSE_out", "NRMSE_mid"]
    df = pd.DataFrame(data=data, columns=columns)
    output_dir = os.path.join(r"C:\Users\jwu191\Desktop\Projects\AD_fitting\results", patient_name)
    os.makedirs(output_dir, exist_ok=True)
    filename = "_hippocanpus.xlsx"
    if ROI == 2:
        filename = "_thalamus.xlsx"
    if ROI == 3:
        filename = "_WM.xlsx"
    if ROI == 4:
        filename = "_pons.xlsx"
    df.to_excel(os.path.join(output_dir, patient_name+filename))    
    
    # for i in range(skull.shape[0]):
    #     for j in range(skull.shape[1]):
    #         for k in range(skull.shape[2]):


def fit_oneslice(patient_name, lineshape, nth_slice):
    dataLoader = DataLoader(patient_name, "Coronal")
    evaluation = Evaluation()
    correction = B0Correction()
    visualization = Visualization()
    mask = dataLoader.read_skull()[nth_slice-1] # [256, 256]
    EMR = dataLoader.read_EMR()[:, nth_slice-1, :, :] # [24, 256, 256]
    T1_map = dataLoader.read_T1_map()[nth_slice-1] # [256, 256]
    T2_map = dataLoader.read_T2_map()[nth_slice-1] # [256, 256]
    WASSR = dataLoader.read_WASSR()[:, nth_slice-1, :, :] # [26, 128, 128]
    data = []
    x = np.where(mask == 1)[0]
    y = np.where(mask == 1)[1]
    print("ROI size:", x.shape[0])
    for i in range(x.shape[0]):
        if T1_map[x[i], y[i]]*T2_map[ x[i], y[i]] == 0:
            continue
        if i % 100 == 0:
            print("********", i, "********")
        Zspec = EMR[:, x[i], y[i]]
        offset, Zspec = dataLoader.Zspec_preprocess_onepixel(Zspec) 
        wassr_spec = WASSR[:, round(x[i]/2), round(y[i]/2)]
        B0_shift = correction.cal_B0_shift_onepixel(wassr_spec)
        Zspec = correction.correct_onepixel(offset, Zspec, B0_shift)
        item = fit_by_mask_helper(offset, Zspec, T1_map[x[i], y[i]], 
                                  T2_map[x[i], y[i]], B0_shift, lineshape)
        data.append([nth_slice-1, x[i], y[i]]+item)
    columns = ["z", "x", "y", "R", "R*M0m*T1w", "T1w/T2w", "T2m", "APT#", "NOE#", "APTw",
               "MTR_20ppm", "NRMSE", "NRMSE_out", "NRMSE_mid"]
    df = pd.DataFrame(data=data, columns=columns)
    output_dir = os.path.join(r"C:\Users\jwu191\Desktop\Projects\AD_fitting\results", patient_name)
    os.makedirs(output_dir, exist_ok=True)
    filename = "_slice_"+str(nth_slice)+".xlsx"
    df.to_excel(os.path.join(output_dir, patient_name+filename))    
     
                
def check_data(patient_name):
    dataLoader = DataLoader(patient_name, "Coronal")
    EMR = dataLoader.read_EMR() # [24, 15, 256, 256]
    T1_map = dataLoader.read_T1_map() # [15, 256, 256]
    T2_map = dataLoader.read_T2_map() # [15, 256, 256]
    WASSR = dataLoader.read_WASSR() # [26, 15, 128, 128]
    
    
# img = dataLoader.read_EMR()[0, 0, :, :]
# plt.imshow(img, origin="lower")
# plt.show()






