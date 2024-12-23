import os
import numpy as np
import pandas as pd


def get_cols(method):
    if "MT" in method:
        return ["R*M0b*T1a", "T1a/T2a", "T1a", "T1b", "T2a", "T2b", "R", "M0b", "noise", 
                "NRMSE", "APT#", "NOE#", "APTw", "MTR_20ppm", "MTR_60ppm", 
                "T1obs", "T2obs", "z", "x", "y", "B0 shift"]
    if "BM" in method:
        return ["T1a", "T1b", "T2a", "T2b", "R", "M0b", "noise",
                "NRMSE", "APT#", "NOE#", "APTw", "MTR_20ppm", "MTR_60ppm", 
                "T1obs", "T2obs", "z", "x", "y", "B0 shift"]
    if "Lorentz" in method:
        return ["APT magnitude", "APT center", "APT width", "CEST2ppm magnitude", 
                "CEST2ppm center", "CEST2ppm width", "DS magnitude", "DS center", 
                "DS width", "MT magnitude", "MT center", "MT width", "NOE magnitude", 
                "NOE center", "NOE width", "NRMSE", "NRMSE_mid",
                "T1obs", "T2obs", "z", "x", "y", "B0 shift"]
    
def ROI2filename(ROI):
    if ROI == 0:
        filename = "map"
    if ROI == 1:
        filename = "hippocampus"
    if ROI == 2:
        filename = "thalamus"
    if ROI == 3:
        filename = "WM"
    if ROI == 4:
        filename = "pons"
    if ROI == 5:
        filename = "CSF"
    if ROI == 6:
        filename = "ventricle"
    return filename

def get_patients(flag=0):
    list_16 = ["AD_05", "AD_06", "AD_07", "AD_29", "AD_30", "AD_31", "AD_32", 
               "AD_33", "AD_34", "AD_35", "AD_37", "AD_38", "AD_39", "AD_40", 
               "AD_41", "AD_47"] # 16 cases
    list_24 = ["AD_45", "AD_46", "AD_48", "AD_49", "AD_50", "AD_51", "AD_52", 
               "AD_53", "AD_54", "AD_55", "AD_56", "AD_57", "AD_58", "AD_59", 
               "AD_60", "AD_61", "AD_62", "AD_63", "AD_64", "AD_66", "AD_67", 
               "AD_68", "AD_69", "AD_70", "AD_71", "AD_72", "AD_73", "AD_74", 
               "AD_75", "AD_76", "AD_77", "AD_78", "AD_79", "AD_80", "AD_81",
               "AD_82", "AD_83", "AD_84", "AD_85", "AD_86"] # 40 cases
    if flag == 16:
        return list_16
    if flag == 24:
        return list_24   
    return list_16 + list_24

def filter_offset(offset, Zspec, selected_offset):
    # keep only the selected offsets (if exist)
    new_offset = []
    new_Zspec = []
    for x in selected_offset:
        if x in offset:
            new_offset.append(offset[offset==x][0])
            new_Zspec.append(Zspec[offset==x][0])
    return np.array(new_offset), np.array(new_Zspec)

def remove_large_offset(offset, Zspec):
    # remove +-8-80 ppm for visualization
    new_offset = []
    new_Zspec = []
    for i in range(offset.shape[0]):
        if offset[i] > -8 and offset[i] < 8:
            new_offset.append(offset[i])
            new_Zspec.append(Zspec[i])
    return np.array(new_offset), np.array(new_Zspec)

def remove_large_offset_Lorentz(offset, Zspec):
    # remove +-20-80 ppm for visualization
    new_offset = []
    new_Zspec = []
    for i in range(offset.shape[0]):
        if offset[i] > -20 and offset[i] < 20:
            new_offset.append(offset[i])
            new_Zspec.append(Zspec[i])
    return np.array(new_offset), np.array(new_Zspec)

def get_Z_0ppm(offset, Zspec):
    return np.mean(Zspec[offset == 0])

def lorentz_func(x, amplitude, center, width):
    tmp = ((x-center) / width) * ((x-center) / width)
    return amplitude / (1 + 4*tmp)

def cal_MTRasym(offset, Zspec):
    p = np.where(offset >= 0)
    n = np.where(offset <= 0)
    MTRasym = Zspec[n][::-1] - Zspec[p]
    return offset[p], MTRasym  

def save_results(patient_name, method, data, filename):
    df = pd.DataFrame(data=np.array(data), columns=get_cols(method))
    output_dir = os.path.join(r"C:\Users\jwu191\Desktop\Projects\AD_fitting\results", 
                              patient_name, method)
    os.makedirs(output_dir, exist_ok=True)
    df.to_excel(os.path.join(output_dir, filename))

def check_Zspec(offset, Zspec_1, Zspec_2):
    for i in range(offset.shape[0]):
        print(offset[i], 100*Zspec_1[i], 100*Zspec_2[i])

def EMR_offset(freq, Zspec):
    if np.sum(freq == 0) > 0:
        # return np.concatenate((freq[0:7], freq[-7:])), np.concatenate((Zspec[0:7], Zspec[-7:]))
        return freq[0:7], Zspec[0:7]
    else:
        return freq[0:7], Zspec[0:7]




