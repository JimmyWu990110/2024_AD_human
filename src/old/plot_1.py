import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from DataLoader import DataLoader
from B0Correction import B0Correction
from Fitting import Fitting

def plot_Zspec(offset, Zspec_all, patient_name, method, title):
    Zspec_mean = 100 * np.mean(Zspec_all, axis=0)
    Zspec_std = 100 *np.std(Zspec_all, axis=0)
    if offset.shape[0] > 15:
        plt.ylim((20, 100))  # full
    else:
        plt.ylim((20, 80))  # middle
    p = np.where(offset > 0)
    n = np.where(offset < 0)
    plt.errorbar(offset[p], Zspec_mean[p], yerr=Zspec_std[p], color="blue", capsize=2)
    plt.errorbar(offset[n], Zspec_mean[n], yerr=Zspec_std[n], color="blue", capsize=2)
    plt.xlabel("(ppm)")
    plt.ylabel("(%)")
    plt.gca().invert_xaxis()  # invert x-axis
    plt.title(patient_name+"_"+method+"_"+title)
    plt.savefig(os.path.join(r"C:\Users\jwu191\Desktop\Projects\AD_fitting\results",
                             patient_name, method, title+".png"))
    plt.show()

def plot_MTRasym(MTRasym_all, patient_name, method, title):
    MTRasym_mean = 100 * np.mean(MTRasym_all, axis=0)
    MTRasym_std = 100 * np.std(MTRasym_all, axis=0)
    offset = np.array([4, 3.5, 3, 2.5, 2, 1.5])
    plt.ylim((-1.5, 2.5))
    plt.errorbar(offset, MTRasym_mean, yerr=MTRasym_std, color="blue", capsize=2)
    plt.xlabel("(ppm)")
    plt.ylabel("(%)")
    plt.gca().invert_xaxis()  # invert x-axis
    plt.title(patient_name+"_"+method+"_"+title)
    plt.savefig(os.path.join(r"C:\Users\jwu191\Desktop\Projects\AD_fitting\results",
                             patient_name, method, title+".png"))
    plt.show()

def plot_EMR(offset, Zspec_all, Zspec_EMR_all, patient_name, method, title):
    Zspec_mean = 100 * np.mean(Zspec_all, axis=0)
    Zspec_EMR_mean = 100 * np.mean(Zspec_EMR_all, axis=0)
    if offset.shape[0] > 15:
        plt.ylim((20, 100))  # full
    else:
        plt.ylim((20, 80))  # middle
    p = np.where(offset > 0)
    n = np.where(offset < 0)
    plt.plot(offset[p], Zspec_EMR_mean[p], color="blue")
    plt.plot(offset[n], Zspec_EMR_mean[n], color="blue")
    plt.scatter(offset[p], Zspec_mean[p], color="black", marker="x")
    plt.scatter(offset[n], Zspec_mean[n], color="black", marker="x")
    plt.xlabel("(ppm)")
    plt.ylabel("(%)")
    plt.gca().invert_xaxis() # invert x-axis
    plt.title(patient_name+"_"+method+"_"+title)
    plt.savefig(os.path.join(r"C:\Users\jwu191\Desktop\Projects\AD_fitting\results",
                             patient_name, method, title+".png"))
    plt.show()

def plot_EMR_pow(offset, Zspec_all, Zspec_EMR_all, patient_name, method, title):
    pow = 100 * (Zspec_EMR_all - Zspec_all)
    pow_mean = np.mean(pow, axis=0)
    pow_std = np.std(pow, axis=0)
    p = np.where(offset > 0)
    n = np.where(offset < 0)
    plt.ylim((-2.5, 4.5))
    plt.errorbar(offset[p], pow_mean[p], yerr=pow_std[p], color="blue", capsize=2)
    plt.errorbar(offset[n], pow_mean[n], yerr=pow_std[n], color="blue", capsize=2)
    plt.xlabel("(ppm)")
    plt.ylabel("(%)")
    plt.gca().invert_xaxis() # invert x-axis
    plt.title(patient_name+"_"+method+"_"+title)
    plt.savefig(os.path.join(r"C:\Users\jwu191\Desktop\Projects\AD_fitting\results",
                             patient_name, method, title+".png"))
    plt.show()

def MTRasym_helper(offset, Zspec):
    MTRasym = []
    for x in [4, 3.5, 3, 2.5, 2, 1.5]:
        MTRasym.append(Zspec[offset == -x][0] - Zspec[offset == x][0])
    return np.array(MTRasym)

def Zspec_to_excel(patient_name, ROI, method):
    dataLoader = DataLoader(patient_name)
    mask = dataLoader.read_mask() # [15, 256, 256]
    EMR = dataLoader.read_EMR() # [24, 15, 256, 256]
    WASSR = dataLoader.read_WASSR() # [26, 15, 256, 256]
    z = np.where(mask == ROI)[0]
    x = np.where(mask == ROI)[1]
    y = np.where(mask == ROI)[2]
    offset = None
    Zspec_all = []
    for i in range(x.shape[0]):
        Zspec = EMR[:, z[i], x[i], y[i]]
        offset, Zspec = dataLoader.Zspec_preprocess_onepixel(Zspec)
        WASSR_Spec = WASSR[:, z[i], x[i], y[i]]
        correction = B0Correction(offset, Zspec, WASSR_Spec)
        Zspec = correction.correct()
        Zspec_all.append(np.concatenate((Zspec, [z[i], x[i], y[i]])))
    Zspec_all = np.array(Zspec_all)
    cols = np.array([80, 60, 40, 30, 20, 12, 8, 4, 3.5, 3, 2.5, 2, 1.5,
                     -1.5, -2, -2.5, -3, -3.5, -4, "x", "y", "z"])
    df = pd.DataFrame(data=np.array(Zspec_all), columns=cols)
    df.to_excel(os.path.join(r"C:\Users\jwu191\Desktop\Projects\AD_fitting\results",
                             patient_name, method, "hippocampus_Zspec_24.xlsx"))

def get_plots(patient_name, ROI, method):
    base_dir = os.path.join(r"C:\Users\jwu191\Desktop\Projects\AD_fitting\results",
                            patient_name, method)
    df_Zspec = pd.read_excel(os.path.join(base_dir, "hippocampus_Zspec_24.xlsx"))
    df_fitted = pd.read_excel(os.path.join(base_dir, "hippocampus_fitted_24.xlsx"))
    if "MT" in method:
        df_fitted = np.array(df_fitted[["R*M0b*T1a", "T2b", "R", "T1a/T2a"]])
    if "BM" in method:
        df_fitted = np.array(df_fitted[["T1a", "T1b", "T2a", "T2b", "R", "M0b"]])
    df_Zspec = np.array(df_Zspec)
    fitting = Fitting()
    offset = np.array([80, 60, 40, 30, 20, 12, 8, 4, 3.5, 3, 2.5, 2, 1.5, -1.5, -2, -2.5, -3, -3.5, -4])
    Zspec_all = []
    MTRasym_all = []
    Zspec_EMR_all = []
    for i in range(df_Zspec.shape[0]):
        Zspec = df_Zspec[i, 1:-3]
        Zspec_all.append(Zspec)
        MTRasym_all.append(MTRasym_helper(offset, Zspec))
        Zspec_EMR_all.append(fitting.generate_Zspec(method, offset, df_fitted[i]))
    Zspec_all = np.array(Zspec_all)
    MTRasym_all = np.array(MTRasym_all)
    Zspec_EMR_all = np.array(Zspec_EMR_all)
    plot_Zspec(offset, Zspec_all, patient_name, method, "Zspec")
    plot_Zspec(offset[7:], Zspec_all[:, 7:], patient_name, method, "Zspec_middle")
    plot_MTRasym(MTRasym_all, patient_name, method, "MTRasym")
    plot_EMR(offset, Zspec_all, Zspec_EMR_all, patient_name, method, "EMR")
    plot_EMR(offset[7:], Zspec_all[:, 7:], Zspec_EMR_all[:, 7:], patient_name, method, "EMR_middle")
    plot_EMR_pow(offset[7:], Zspec_all[:, 7:], Zspec_EMR_all[:, 7:], patient_name, method, "EMR_pow")
    Zspec_mean = np.mean(Zspec_all, axis=0)
    MTRasym_mean = np.mean(MTRasym_all, axis=0)
    Zspec_EMR_mean = np.mean(Zspec_EMR_all, axis=0)
    np.save(os.path.join(base_dir, "Zspec_mean.npy"), Zspec_mean)
    np.save(os.path.join(base_dir, "MTRasym_mean.npy"), MTRasym_mean)
    np.save(os.path.join(base_dir, "Zspec_EMR_mean.npy"), Zspec_EMR_mean)

def plot_MTRasym_group(CN_list, MCI_list, AD_list, method):
    MTRasym_CN = []
    MTRasym_MCI = []
    MTRasym_AD = []
    for patient_name in CN_list:
        path = os.path.join(r"C:\Users\jwu191\Desktop\Projects\AD_fitting\results",
                            patient_name, method, "MTRasym_mean.npy")
        MTRasym_CN.append(np.load(path))
    for patient_name in MCI_list:
        path = os.path.join(r"C:\Users\jwu191\Desktop\Projects\AD_fitting\results",
                            patient_name, method, "MTRasym_mean.npy")
        MTRasym_MCI.append(np.load(path))
    for patient_name in AD_list:
        path = os.path.join(r"C:\Users\jwu191\Desktop\Projects\AD_fitting\results",
                            patient_name, method, "MTRasym_mean.npy")
        MTRasym_AD.append(np.load(path))
    MTRasym_CN_mean = 100 * np.mean(np.array(MTRasym_CN), axis=0)
    MTRasym_CN_std = 100 * np.std(np.array(MTRasym_CN), axis=0)
    MTRasym_MCI_mean = 100 * np.mean(np.array(MTRasym_MCI), axis=0)
    MTRasym_MCI_std = 100 * np.std(np.array(MTRasym_MCI), axis=0)
    MTRasym_AD_mean = 100 * np.mean(np.array(MTRasym_AD), axis=0)
    MTRasym_AD_std = 100 * np.std(np.array(MTRasym_AD), axis=0)
    offset = np.array([4, 3.5, 3, 2.5, 2, 1.5])
    plt.errorbar(offset, MTRasym_CN_mean, yerr=MTRasym_CN_std, color="blue", capsize=2,
                 label="CN")
    plt.errorbar(offset, MTRasym_MCI_mean, yerr=MTRasym_MCI_std, color="orange", capsize=2,
                 label="MCI")
    plt.errorbar(offset, MTRasym_AD_mean, yerr=MTRasym_AD_std, color="red", capsize=2,
                 label="AD")
    plt.xlabel("(ppm)")
    plt.ylabel("(%)")
    plt.gca().invert_xaxis()  # invert x-axis
    plt.legend()
    plt.title("MTRasym by group")
    plt.show()

def plot_Zspec_group_2(CN_list, MCI_list, AD_list):
    Zspec_CN = []
    Zspec_AD = []
    EMR_CN = []
    EMR_AD = []
    method = "MT_EMR"
    for patient_name in CN_list:
        Zspec_path = os.path.join(r"C:\Users\jwu191\Desktop\Projects\AD_fitting\results",
                            patient_name, method, "Zspec_mean.npy")
        EMR_path = os.path.join(r"C:\Users\jwu191\Desktop\Projects\AD_fitting\results",
                                patient_name, method, "Zspec_EMR_mean.npy")
        Zspec_CN.append(np.load(Zspec_path))
        EMR_CN.append(np.load(EMR_path))
    for patient_name in MCI_list:
        Zspec_path = os.path.join(r"C:\Users\jwu191\Desktop\Projects\AD_fitting\results",
                            patient_name, method, "Zspec_mean.npy")
        EMR_path = os.path.join(r"C:\Users\jwu191\Desktop\Projects\AD_fitting\results",
                                patient_name, method, "Zspec_EMR_mean.npy")
        Zspec_AD.append(np.load(Zspec_path))
        EMR_AD.append(np.load(EMR_path))
    for patient_name in AD_list:
        Zspec_path = os.path.join(r"C:\Users\jwu191\Desktop\Projects\AD_fitting\results",
                            patient_name, method, "Zspec_mean.npy")
        EMR_path = os.path.join(r"C:\Users\jwu191\Desktop\Projects\AD_fitting\results",
                                patient_name, method, "Zspec_EMR_mean.npy")
        Zspec_AD.append(np.load(Zspec_path))
        EMR_AD.append(np.load(EMR_path))
    # Zspec_CN_mean = 100 * np.mean(np.array(Zspec_CN), axis=0)
    # Zspec_CN_std = 100 * np.std(np.array(Zspec_CN), axis=0)
    # Zspec_AD_mean = 100 * np.mean(np.array(Zspec_AD), axis=0)
    # Zspec_AD_std = 100 * np.std(np.array(Zspec_AD), axis=0)
    # offset = np.array([80, 60, 40, 30, 20, 12, 8, 4, 3.5, 3, 2.5, 2, 1.5, 
    #                    -1.5, -2, -2.5, -3, -3.5, -4])
    # plt.ylim((20, 100))
    Zspec_CN_mean = 100 * np.mean(np.array(Zspec_CN), axis=0)[7:]
    Zspec_CN_std = 100 * np.std(np.array(Zspec_CN), axis=0)[7:]
    Zspec_AD_mean = 100 * np.mean(np.array(Zspec_AD), axis=0)[7:]
    Zspec_AD_std = 100 * np.std(np.array(Zspec_AD), axis=0)[7:]
    offset = np.array([4, 3.5, 3, 2.5, 2, 1.5, 
                       -1.5, -2, -2.5, -3, -3.5, -4])
    plt.ylim((20, 60))
    p = np.where(offset > 0)
    n = np.where(offset < 0)
    plt.errorbar(offset[p], Zspec_CN_mean[p], yerr=Zspec_CN_std[p], color="blue", capsize=2,
                 label="CN")
    plt.errorbar(offset[n], Zspec_CN_mean[n], yerr=Zspec_CN_std[n], color="blue", capsize=2)
    plt.errorbar(offset[p], Zspec_AD_mean[p], yerr=Zspec_AD_std[p], color="red", capsize=2,
                 label="AD")
    plt.errorbar(offset[n], Zspec_AD_mean[n], yerr=Zspec_AD_std[n], color="red", capsize=2)
    plt.xlabel("(ppm)")
    plt.ylabel("(%)")
    plt.gca().invert_xaxis()  # invert x-axis
    plt.legend()
    plt.title("Zspec by group")
    plt.show()

def plot_MTRasym_group_2(CN_list, MCI_list, AD_list):
    MTRasym_CN = []
    MTRasym_AD = []
    method = "MT_EMR"
    for patient_name in CN_list:
        path = os.path.join(r"C:\Users\jwu191\Desktop\Projects\AD_fitting\results",
                            patient_name, method, "MTRasym_mean.npy")
        MTRasym_CN.append(np.load(path))
    for patient_name in MCI_list:
        path = os.path.join(r"C:\Users\jwu191\Desktop\Projects\AD_fitting\results",
                            patient_name, method, "MTRasym_mean.npy")
        MTRasym_AD.append(np.load(path))
    for patient_name in AD_list:
        path = os.path.join(r"C:\Users\jwu191\Desktop\Projects\AD_fitting\results",
                            patient_name, method, "MTRasym_mean.npy")
        MTRasym_AD.append(np.load(path))
    MTRasym_CN_mean = 100 * np.mean(np.array(MTRasym_CN), axis=0)
    MTRasym_CN_std = 100 * np.std(np.array(MTRasym_CN), axis=0)
    MTRasym_AD_mean = 100 * np.mean(np.array(MTRasym_AD), axis=0)
    MTRasym_AD_std = 100 * np.std(np.array(MTRasym_AD), axis=0)
    offset = np.array([4, 3.5, 3, 2.5, 2, 1.5])
    plt.ylim((-1.5, 2.5))
    plt.errorbar(offset, MTRasym_CN_mean, yerr=MTRasym_CN_std, color="blue", capsize=2,
                 label="CN")
    plt.errorbar(offset, MTRasym_AD_mean, yerr=MTRasym_AD_std, color="red", capsize=2,
                 label="AD")
    plt.xlabel("(ppm)")
    plt.ylabel("(%)")
    plt.gca().invert_xaxis()  # invert x-axis
    plt.legend()
    plt.title("MTRasym by group")
    plt.show()

def plot_EMR_pow_group(CN_list, MCI_list, AD_list, method):
    pow_CN = []
    pow_MCI = []
    pow_AD = []
    for patient_name in CN_list:
        path = os.path.join(r"C:\Users\jwu191\Desktop\Projects\AD_fitting\results",
                            patient_name, method, "Zspec_mean.npy")
        EMR_path = os.path.join(r"C:\Users\jwu191\Desktop\Projects\AD_fitting\results",
                                patient_name, method, "Zspec_EMR_mean.npy")
        pow_CN.append(np.load(EMR_path) - np.load(path))
    for patient_name in MCI_list:
        path = os.path.join(r"C:\Users\jwu191\Desktop\Projects\AD_fitting\results",
                            patient_name, method, "Zspec_mean.npy")
        EMR_path = os.path.join(r"C:\Users\jwu191\Desktop\Projects\AD_fitting\results",
                                patient_name, method, "Zspec_EMR_mean.npy")
        pow_MCI.append(np.load(EMR_path) - np.load(path))
    for patient_name in AD_list:
        path = os.path.join(r"C:\Users\jwu191\Desktop\Projects\AD_fitting\results",
                            patient_name, method, "Zspec_mean.npy")
        EMR_path = os.path.join(r"C:\Users\jwu191\Desktop\Projects\AD_fitting\results",
                                patient_name, method, "Zspec_EMR_mean.npy")
        pow_AD.append(np.load(EMR_path) - np.load(path))
    pow_CN_mean = 100 * np.mean(np.array(pow_CN), axis=0)[7:]
    pow_CN_std = 100 * np.std(np.array(pow_CN), axis=0)[7:]
    pow_MCI_mean = 100 * np.mean(np.array(pow_MCI), axis=0)[7:]
    pow_MCI_std = 100 * np.std(np.array(pow_MCI), axis=0)[7:]
    pow_AD_mean = 100 * np.mean(np.array(pow_AD), axis=0)[7:]
    pow_AD_std = 100 * np.std(np.array(pow_AD), axis=0)[7:]
    offset = np.array([4, 3.5, 3, 2.5, 2, 1.5, -1.5, -2, -2.5, -3, -3.5, -4])
    p = np.where(offset > 0)
    n = np.where(offset < 0)
    plt.errorbar(offset[p], pow_CN_mean[p], yerr=pow_CN_std[p], color="blue", capsize=2,
                 label="CN")
    plt.errorbar(offset[n], pow_CN_mean[n], yerr=pow_CN_std[n], color="blue", capsize=2)
    plt.errorbar(offset[p], pow_MCI_mean[p], yerr=pow_MCI_std[p], color="orange", capsize=2,
                 label="MCI")
    plt.errorbar(offset[n], pow_MCI_mean[n], yerr=pow_MCI_std[n], color="orange", capsize=2)
    plt.errorbar(offset[p], pow_AD_mean[p], yerr=pow_AD_std[p], color="red", capsize=2,
                 label="AD")
    plt.errorbar(offset[n], pow_AD_mean[n], yerr=pow_AD_std[n], color="red", capsize=2)
    plt.xlabel("(ppm)")
    plt.ylabel("(%)")
    plt.gca().invert_xaxis()  # invert x-axis
    plt.legend()
    plt.title("EMR# ("+method+") by group")
    plt.show()

def plot_EMR_pow_group_2(CN_list, MCI_list, AD_list, method):
    pow_CN = []
    pow_AD = []
    for patient_name in CN_list:
        path = os.path.join(r"C:\Users\jwu191\Desktop\Projects\AD_fitting\results",
                            patient_name, method, "Zspec_mean.npy")
        EMR_path = os.path.join(r"C:\Users\jwu191\Desktop\Projects\AD_fitting\results",
                                patient_name, method, "Zspec_EMR_mean.npy")
        pow_CN.append(np.load(EMR_path) - np.load(path))
    for patient_name in MCI_list:
        path = os.path.join(r"C:\Users\jwu191\Desktop\Projects\AD_fitting\results",
                            patient_name, method, "Zspec_mean.npy")
        EMR_path = os.path.join(r"C:\Users\jwu191\Desktop\Projects\AD_fitting\results",
                                patient_name, method, "Zspec_EMR_mean.npy")
        pow_AD.append(np.load(EMR_path) - np.load(path))
    for patient_name in AD_list:
        path = os.path.join(r"C:\Users\jwu191\Desktop\Projects\AD_fitting\results",
                            patient_name, method, "Zspec_mean.npy")
        EMR_path = os.path.join(r"C:\Users\jwu191\Desktop\Projects\AD_fitting\results",
                                patient_name, method, "Zspec_EMR_mean.npy")
        pow_AD.append(np.load(EMR_path) - np.load(path))
    pow_CN_mean = 100 * np.mean(np.array(pow_CN), axis=0)[7:]
    pow_CN_std = 100 * np.std(np.array(pow_CN), axis=0)[7:]
    pow_AD_mean = 100 * np.mean(np.array(pow_AD), axis=0)[7:]
    pow_AD_std = 100 * np.std(np.array(pow_AD), axis=0)[7:]
    offset = np.array([4, 3.5, 3, 2.5, 2, 1.5, -1.5, -2, -2.5, -3, -3.5, -4])
    p = np.where(offset > 0)
    n = np.where(offset < 0)
    plt.ylim((-2.5, 4.5))
    plt.errorbar(offset[p], pow_CN_mean[p], yerr=pow_CN_std[p], color="blue", capsize=2,
                 label="CN")
    plt.errorbar(offset[n], pow_CN_mean[n], yerr=pow_CN_std[n], color="blue", capsize=2)
    plt.errorbar(offset[p], pow_AD_mean[p], yerr=pow_AD_std[p], color="red", capsize=2,
                 label="AD")
    plt.errorbar(offset[n], pow_AD_mean[n], yerr=pow_AD_std[n], color="red", capsize=2)
    plt.xlabel("(ppm)")
    plt.ylabel("(%)")
    plt.gca().invert_xaxis()  # invert x-axis
    plt.legend()
    plt.title("EMR# ("+method+") by group")
    plt.show()


def plot_by_group(CN_list, MCI_list, AD_list):
    # plot_MTRasym_group(CN_list, MCI_list, AD_list, method)
    # plot_EMR_pow_group(CN_list, MCI_list, AD_list, method)
    plot_Zspec_group_2(CN_list, MCI_list, AD_list)
    plot_MTRasym_group_2(CN_list, MCI_list, AD_list)
    # plot_EMR_pow_group_2(CN_list, MCI_list, AD_list, "MT_EMR")
    # plot_EMR_pow_group_2(CN_list, MCI_list, AD_list, "BMS_EMR")



phantom_list = ["Phantom_20230620"]
CN_list = ["AD_45", "AD_48", "AD_49", "AD_50", "AD_52", "AD_53", "AD_54", "AD_56", 
           "AD_57", "AD_58", "AD_60", "AD_61", "AD_63", "AD_66", "AD_67", "AD_68", 
           "AD_70", "AD_71", "AD_72", "AD_76", "AD_77", "AD_79"] # 22 cases
MCI_list = ["AD_51", "AD_55", "AD_59", "AD_73", "AD_78"]  # 5 cases
AD_list = ["AD_46", "AD_74", "AD_75", "AD_80"]  # 4 cases
CN_selected_list = ["AD_49", "AD_52", "AD_57", "AD_60", "AD_66", "AD_70", 
                    "AD_72", "AD_76", "AD_79"] # 9 of 22 selected
MCIAD_list = MCI_list + AD_list # 9 cases
whole_list = CN_list + MCIAD_list # 31 cases


for patient_name in phantom_list:
    # Zspec_to_excel(patient_name=patient_name, ROI=1, method="MT_EMR")
    # Zspec_to_excel(patient_name=patient_name, ROI=1, method="BMS_EMR")
    get_plots(patient_name=patient_name, ROI=1, method="MT_EMR")
    get_plots(patient_name=patient_name, ROI=1, method="BMS_EMR")
    

# plot_by_group(CN_selected_list, MCI_list, AD_list)


