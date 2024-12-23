import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from DataLoader import DataLoader
from funcs import *


def plot_Zspec_ROI(patient_name, method, lineshape, middle=False):
    Zspec_tumor = []
    Zspec_edema = []
    Zspec_normal = []
    EMR_tumor = []
    EMR_edema = []
    EMR_normal = []
    base_dir = r"C:\Users\jwu191\Desktop\Projects\Tumor_fitting\results"
    Zspec_path = os.path.join(base_dir, patient_name, "tumor", "Zspec_mean.npy")
    EMR_path = os.path.join(base_dir, patient_name, method, "tumor", "Zspec_EMR_mean.npy")
    Zspec_tumor.append(np.load(Zspec_path))
    EMR_tumor.append(np.load(EMR_path))
    # Zspec_path = os.path.join(base_dir, patient_name, method, "edema", "Zspec_mean.npy")
    # EMR_path = os.path.join(base_dir, patient_name, method, "edema", "Zspec_EMR_mean.npy")
    # Zspec_edema.append(np.load(Zspec_path))
    # EMR_edema.append(np.load(EMR_path))
    Zspec_path = os.path.join(base_dir, patient_name, method, "normal", "Zspec_mean.npy")
    EMR_path = os.path.join(base_dir, patient_name, method, "normal", "Zspec_EMR_mean.npy")
    Zspec_normal.append(np.load(Zspec_path))
    EMR_normal.append(np.load(EMR_path))
    
    if middle:
        Zspec_tumor = 100 * np.mean(np.array(Zspec_tumor), axis=0)[7:]
        # Zspec_edema = 100 * np.mean(np.array(Zspec_edema), axis=0)[7:]
        Zspec_normal = 100 * np.mean(np.array(Zspec_normal), axis=0)[7:]
        EMR_tumor = 100 * np.mean(np.array(EMR_tumor), axis=0)[7:]
        # Zspec_edema = 100 * np.mean(np.array(Zspec_edema), axis=0)[7:]
        EMR_normal = 100 * np.mean(np.array(EMR_normal), axis=0)[7:]
        offset = np.array([4, 3.5, 3, 2.5, 2, 1.5, 
                           -1.5, -2, -2.5, -3, -3.5, -4])
        plt.ylim((20, 60))
    else:
        Zspec_tumor = 100 * np.mean(np.array(Zspec_tumor), axis=0)
        # Zspec_edema = 100 * np.mean(np.array(Zspec_edema), axis=0)
        Zspec_normal = 100 * np.mean(np.array(Zspec_normal), axis=0)
        EMR_tumor = 100 * np.mean(np.array(EMR_tumor), axis=0)
        # Zspec_edema = 100 * np.mean(np.array(Zspec_edema), axis=0)[7:]
        EMR_normal = 100 * np.mean(np.array(EMR_normal), axis=0)
        offset = np.array([80, 60, 40, 30, 20, 12, 8, 4, 3.5, 3, 2.5, 2, 1.5, 
                            -1.5, -2, -2.5, -3, -3.5, -4])
        plt.ylim((20, 100))     
        
    p = np.where(offset > 0)
    n = np.where(offset < 0)
    plt.plot(offset[p], EMR_tumor[p], color="red", label="tumor EMR")
    plt.plot(offset[n], EMR_tumor[n], color="red")
    plt.scatter(offset, Zspec_tumor, color="red", marker="x")
    # plt.plot(offset[p], Zspec_edema[p], color="orange", label="edema")
    # plt.plot(offset[n], Zspec_edema[n], color="orange")
    plt.plot(offset[p], EMR_normal[p], color="blue")
    plt.plot(offset[n], EMR_normal[n], color="blue")
    plt.scatter(offset, Zspec_normal, color="blue", marker="x", label="normal Zspec")
    
    plt.xlabel("(ppm)")
    plt.ylabel("(%)")
    plt.gca().invert_xaxis()  # invert x-axis
    plt.legend()
    plt.title("Zspec by ROI")
    plt.show()

def plot_MTRasym_ROI(patient_name, method):
    MTRasym_tumor = []
    MTRasym_edema = []
    MTRasym_normal = []
    base_dir = r"C:\Users\jwu191\Desktop\Projects\Tumor_fitting\results"
    path = os.path.join(base_dir, patient_name, method, "tumor", "MTRasym_mean.npy")
    MTRasym_tumor.append(np.load(path))
    # path = os.path.join(base_dir, patient_name, method, "edema", "MTRasym_mean.npy")
    # MTRasym_edema.append(np.load(path))
    path = os.path.join(base_dir, patient_name, method, "normal", "MTRasym_mean.npy")
    MTRasym_normal.append(np.load(path))
    
    MTRasym_tumor = 100 * np.mean(np.array(MTRasym_tumor), axis=0)
    # MTRasym_edema = 100 * np.mean(np.array(MTRasym_edema), axis=0)
    MTRasym_normal = 100 * np.mean(np.array(MTRasym_normal), axis=0)
    
    offset = np.array([4, 3.5, 3, 2.5, 2, 1.5])

    plt.plot(offset, MTRasym_tumor, color="red", label="tumor")
    # plt.plot(offset, MTRasym_edema, color="orange", label="edema")
    plt.plot(offset, MTRasym_normal, color="blue", label="normal")
    
    plt.xlabel("(ppm)")
    plt.ylabel("(%)")
    plt.gca().invert_xaxis()  # invert x-axis
    plt.legend()
    plt.title("MTRasym by ROI")
    plt.show()

def plot_EMR_pow_ROI(patient_name, method, lineshape):
    pow_tumor = []
    pow_edema = []
    pow_normal = []
    base_dir = r"C:\Users\jwu191\Desktop\Projects\Tumor_fitting\results"
    path = os.path.join(base_dir, patient_name, method, "tumor", "Zspec_mean.npy")
    EMR_path = os.path.join(base_dir, patient_name, method, "tumor", "Zspec_EMR_mean.npy")
    pow_tumor.append(np.load(EMR_path) - np.load(path))
    # path = os.path.join(base_dir, patient_name, method, "edema", "Zspec_mean.npy")
    # EMR_path = os.path.join(base_dir, patient_name, method, "edema", "Zspec_EMR_mean.npy")
    # pow_edema.append(np.load(EMR_path) - np.load(path))
    path = os.path.join(base_dir, patient_name, method, "normal", "Zspec_mean.npy")
    EMR_path = os.path.join(base_dir, patient_name, method, "normal", "Zspec_EMR_mean.npy")
    pow_normal.append(np.load(EMR_path) - np.load(path))
    
    pow_tumor = 100 * np.mean(np.array(pow_tumor), axis=0)[7:]
    # pow_edema = 100 * np.mean(np.array(pow_edema), axis=0)[7:]
    pow_normal = 100 * np.mean(np.array(pow_normal), axis=0)[7:]
    
    offset = np.array([4, 3.5, 3, 2.5, 2, 1.5, -1.5, -2, -2.5, -3, -3.5, -4])
    p = np.where(offset > 0)
    n = np.where(offset < 0)
    
    plt.plot(offset[p], pow_tumor[p], color="red", label="tumor")
    plt.plot(offset[n], pow_tumor[n], color="red")
    # plt.plot(offset[p], pow_edema[p], color="orange", label="edema")
    # plt.plot(offset[n], pow_edema[n], color="orange")
    plt.plot(offset[p], pow_normal[p], color="blue", label="normal")
    plt.plot(offset[n], pow_normal[n], color="blue")
    
    plt.xlabel("(ppm)")
    plt.ylabel("(%)")
    plt.gca().invert_xaxis()  # invert x-axis
    plt.legend()
    plt.title("EMR# ("+method+") by ROI")
    plt.show()

patient_name = "APT_470"
method = "MT_EMR"
lineshape = "L"

plot_Zspec_ROI(patient_name, method, lineshape)
plot_Zspec_ROI(patient_name, method, lineshape, True)
plot_MTRasym_ROI(patient_name, method)
plot_EMR_pow_ROI(patient_name, method, lineshape)








