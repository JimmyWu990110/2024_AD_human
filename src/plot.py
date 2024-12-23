import os
import numpy as np
import pandas as pd

from funcs import ROI2filename, remove_large_offset
from DataLoader import DataLoader
from Visualization import Visualization

def plot_group_EMR(method, ROI, CN_list, AD_list):
    EMR_CN = []
    EMR_AD = []
    dataloader = DataLoader()
    visualization = Visualization()
    for patient_name in CN_list:
        path = os.path.join(r"C:\Users\jwu191\Desktop\Projects\AD_fitting\results",
                            patient_name, method)
        Zspec = np.load(os.path.join(path, ROI2filename(ROI)+"_SL_Zspec.npy"))
        Zspec_EMR = np.load(os.path.join(path, ROI2filename(ROI)+"_SL_Zspec_EMR.npy"))
        EMR_CN.append(Zspec_EMR-Zspec)
    for patient_name in AD_list:
        path = os.path.join(r"C:\Users\jwu191\Desktop\Projects\AD_fitting\results",
                            patient_name, method)
        Zspec = np.load(os.path.join(path, ROI2filename(ROI)+"_SL_Zspec.npy"))
        Zspec_EMR = np.load(os.path.join(path, ROI2filename(ROI)+"_SL_Zspec_EMR.npy"))
        EMR_AD.append(Zspec_EMR-Zspec)
    offset = dataloader.get_offset_by_Zspec(Zspec)
    diff_CN = np.mean(np.array(EMR_CN), axis=0)
    diff_AD = np.mean(np.array(EMR_AD), axis=0)
    std_CN = np.std(np.array(EMR_CN), axis=0)
    std_AD = np.std(np.array(EMR_AD), axis=0)
    offset_mid, diff_mid_CN = remove_large_offset(offset, diff_CN)
    _, diff_mid_AD = remove_large_offset(offset, diff_AD)
    _, std_mid_CN = remove_large_offset(offset, std_CN)
    _, std_mid_AD = remove_large_offset(offset, std_AD)
    visualization.plot_Zspec_diff_2(offset_mid, diff_mid_CN, diff_mid_AD,
                                    std_1=std_mid_CN, std_2=std_mid_AD,
                                    labels=["CN", "AD"], title="Group based EMR anamysis",
                                    lim=[-2.5, 4.25])
    

# exclued 62, 68, 69
CN_list = ["AD_45", "AD_48", "AD_49", "AD_50", "AD_52", "AD_53", "AD_54", "AD_56", 
           "AD_57", "AD_58", "AD_60", "AD_61", "AD_63", "AD_66", "AD_67", 
            "AD_70", "AD_71", "AD_72", "AD_76", "AD_77", "AD_79"]
# CN_list = ["AD_48", "AD_49", "AD_57", "AD_58", "AD_63", "AD_66", "AD_67", "AD_70", 
#            "AD_72", "AD_79"]
AD_list = ["AD_46", "AD_51", "AD_55", "AD_59", "AD_64", "AD_73", "AD_74", "AD_75",
           "AD_78", "AD_80"]

method = "BMS_EMR_v2_noisy"
ROI = 1

plot_group_EMR(method, ROI, CN_list, AD_list)