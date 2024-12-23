import os
import numpy as np
import matplotlib.pyplot as plt

from DataLoader import DataLoader
from funcs import remove_large_offset


def get_data(method, patient_list, phantom_list):
    base_dir = r"C:\Users\jwu191\Desktop\Projects\AD_fitting\results"
    Zspec_phantom = []
    Zspec_EMR_phantom = []
    Zspec_WM = []
    Zspec_EMR_WM = []
    Zspec_hippocampus = []
    Zspec_EMR_hippocampus = []
    for patient_name in phantom_list:
        path = os.path.join(base_dir, patient_name, method)
        Zspec_phantom.append(np.load(os.path.join(path, "thalamus_single_SL_Zspec.npy")))
        Zspec_EMR_phantom.append(np.load(os.path.join(path, "thalamus_single_SL_Zspec_EMR.npy")))
    for patient_name in patient_list:
        path = os.path.join(base_dir, patient_name, method)
        Zspec_WM.append(np.load(os.path.join(path, "WM_single_SL_Zspec.npy")))
        Zspec_EMR_WM.append(np.load(os.path.join(path, "WM_single_SL_Zspec_EMR.npy")))
        Zspec_hippocampus.append(np.load(os.path.join(path, "hippocampus_single_SL_Zspec.npy")))
        Zspec_EMR_hippocampus.append(np.load(os.path.join(path, "hippocampus_single_SL_Zspec_EMR.npy")))
    dataloader = DataLoader()
    offset = dataloader.get_offset(single=True)
    offset_mid, Zspec_mid_phantom = remove_large_offset(offset, np.mean(Zspec_phantom, axis=0))
    _, Zspec_mid_WM = remove_large_offset(offset, np.mean(Zspec_WM, axis=0))
    _, Zspec_mid_hippocampus = remove_large_offset(offset, np.mean(Zspec_hippocampus, axis=0))
    _, Zspec_EMR_mid_phantom = remove_large_offset(offset, np.mean(Zspec_EMR_phantom, axis=0))
    _, Zspec_EMR_mid_WM = remove_large_offset(offset, np.mean(Zspec_EMR_WM, axis=0))
    _, Zspec_EMR_mid_hippocampus = remove_large_offset(offset, np.mean(Zspec_EMR_hippocampus, axis=0))
    return (offset_mid, Zspec_mid_phantom, Zspec_mid_WM, Zspec_mid_hippocampus,
            Zspec_EMR_mid_phantom, Zspec_EMR_mid_WM, Zspec_EMR_mid_hippocampus)


def Fig_1_helper(method):
    base_dir = r"C:\Users\jwu191\Desktop\Projects\AD_fitting\results"
    path_phantom = os.path.join(base_dir, "Phantom_20230503a", method)
    Zspec_phantom = np.load(os.path.join(path_phantom, "hippocampus_single_SL_Zspec.npy"))
    Zspec_EMR_phantom = np.load(os.path.join(path_phantom, "hippocampus_single_SL_Zspec_EMR.npy"))
    patient_name = "AD_86"
    path_WM = os.path.join(base_dir, patient_name, method)
    Zspec_WM = np.load(os.path.join(path_WM, "WM_single_SL_Zspec.npy"))
    Zspec_EMR_WM = np.load(os.path.join(path_WM, "WM_single_SL_Zspec_EMR.npy"))
    path_hippocampus = os.path.join(base_dir, patient_name, method)
    Zspec_hippocampus = np.load(os.path.join(path_hippocampus, "hippocampus_single_SL_Zspec.npy"))
    Zspec_EMR_hippocampus = np.load(os.path.join(path_hippocampus, "hippocampus_single_SL_Zspec_EMR.npy"))
    dataloader = DataLoader()
    offset = dataloader.get_offset(single=True)
    offset_mid, Zspec_mid_phantom = remove_large_offset(offset, Zspec_phantom)
    _, Zspec_mid_WM = remove_large_offset(offset, Zspec_WM)
    _, Zspec_mid_hippocampus = remove_large_offset(offset, Zspec_hippocampus)
    _, Zspec_EMR_mid_phantom = remove_large_offset(offset, Zspec_EMR_phantom)
    _, Zspec_EMR_mid_WM = remove_large_offset(offset, Zspec_EMR_WM)
    _, Zspec_EMR_mid_hippocampus = remove_large_offset(offset, Zspec_EMR_hippocampus)
    # Zspec_EMR_mid_tumor[offset_mid == 0.25] -= 0.005
    # Zspec_EMR_mid_tumor[offset_mid == 0] -= 0.005
    # Zspec_EMR_mid_tumor[offset_mid == -0.25] -= 0.005
    return (offset_mid, Zspec_mid_phantom, Zspec_mid_WM, Zspec_mid_hippocampus,
            Zspec_EMR_mid_phantom, Zspec_EMR_mid_WM, Zspec_EMR_mid_hippocampus)

def Fig_1A(method, patient_list, phantom_list):
    (offset, Zspec_phantom, Zspec_WM, Zspec_hippocampus, Zspec_EMR_phantom, 
     Zspec_EMR_WM, Zspec_EMR_hippocampus) = get_data(method, patient_list, phantom_list)
    labels = ["Phantom", "WM", "Hippocampus"]
    colors = ["black", "blue", "red"]
    scale = 100
    # plt.ylim((0, scale))
    plt.scatter(offset, scale*Zspec_phantom, label=labels[0], color=colors[0], marker="x")
    plt.plot(offset, scale*Zspec_EMR_phantom, color=colors[0])
    plt.scatter(offset, scale*Zspec_WM, label=labels[1], color=colors[1], marker="x")
    plt.plot(offset, scale*Zspec_EMR_WM, color=colors[1])
    plt.scatter(offset, scale*Zspec_hippocampus, label=labels[2], color=colors[2], marker="x")
    plt.plot(offset, scale*Zspec_EMR_hippocampus, color=colors[2])
    plt.gca().invert_xaxis()  # invert x-axis
    plt.legend(fontsize=13)
    plt.xlabel("Frequency offset (ppm)", fontsize=13)
    plt.ylabel("Msat/M0 (%)", fontsize=13)
    plt.xticks(fontsize=13)
    plt.yticks(fontsize=13)
    plt.title(method)
    plt.show() 

def Fig_1B(method, patient_list, phantom_list):
    (offset, Zspec_phantom, Zspec_WM, Zspec_hippocampus, Zspec_EMR_phantom, 
     Zspec_EMR_WM, Zspec_EMR_hippocampus) = get_data(method, patient_list, phantom_list)
    print("APT# phantom:", 100*(Zspec_EMR_phantom-Zspec_phantom)[offset == 3.5][0], "%")
    print("APT# WM:", 100*(Zspec_EMR_WM-Zspec_WM)[offset == 3.5][0], "%")
    print("APT# hippocampus:", 100*(Zspec_EMR_hippocampus-Zspec_hippocampus)[offset == 3.5][0], "%")
    print("NOE# phantom:", 100*(Zspec_EMR_phantom-Zspec_phantom)[offset == -3.5][0], "%")
    print("NOE# WM:", 100*(Zspec_EMR_WM-Zspec_WM)[offset == -3.5][0], "%")
    print("NOE# hippocampus:", 100*(Zspec_EMR_hippocampus-Zspec_hippocampus)[offset == -3.5][0], "%")
    labels = ["Phantom", "WM", "Hippocampus"]
    colors = ["black", "blue", "red"]
    scale = 100
    llim, ulim = -4, 6
    plt.ylim((llim, ulim))
    plt.plot(offset, scale*(Zspec_EMR_phantom-Zspec_phantom), label=labels[0], color=colors[0])
    plt.plot(offset, scale*(Zspec_EMR_WM-Zspec_WM), label=labels[1], color=colors[1])
    plt.plot(offset, scale*(Zspec_EMR_hippocampus-Zspec_hippocampus), label=labels[2], color=colors[2])
    plt.vlines(3.5, llim, ulim, color="black", linestyles="--")
    plt.vlines(-3.5, llim, ulim, color="black", linestyles="--")
    plt.hlines(0, np.min(offset), np.max(offset), color="black", linestyles="--")
    plt.gca().invert_xaxis()  # invert x-axis
    plt.legend(fontsize=13)
    plt.xlabel("Frequency offset (ppm)", fontsize=13)
    plt.ylabel("Z_fitted - Z_real (%)", fontsize=13)
    plt.xticks(fontsize=13)
    plt.yticks(fontsize=13)
    plt.title(method)
    plt.show()


method="BMS_EMR_v2_noisy"
patient_list = ["AD_77", "AD_79", "AD_84", "AD_85"]
# phantom_list = ["Phantom_20230503a", "phantom_0606", "Phantom_20230620"]
phantom_list = ["Phantom_20230503a", "phantom_0606", "Phantom_20230620"]
Fig_1A(method, patient_list, phantom_list)
Fig_1B(method, patient_list, phantom_list)
