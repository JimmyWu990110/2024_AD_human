import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from multiprocessing import Pool
from itertools import repeat
from tqdm import tqdm

from DataLoader import DataLoader
from B0_correction import B0_correction 
from Fitting import Fitting
from Evaluation import Evaluation
from Visualization import Visualization
from funcs import save_results, ROI2filename


def fit_by_pixel(patient_name, nth_slice, x, y, method):
    # NOTE: this function is only used for dubugging
    # The (x, y) are the values directly read from ITK-SNAP
    dataLoader = DataLoader(patient_name, "Coronal")
    evaluation = Evaluation()
    visualization = Visualization()
    Zspec = dataLoader.get_Zspec_by_coordinate_24(nth_slice-1, y-1, x-1)
    offset, Zspec = dataLoader.Zspec_preprocess_onepixel(Zspec)
    WASSR_Spec = dataLoader.read_WASSR()[:, nth_slice-1, round((y-1)/2), round((x-1)/2)]
    correction = B0_correction(offset, Zspec, WASSR_Spec)
    Zspec = correction.correct()
    T1 = dataLoader.read_T1_map()[nth_slice-1, y-1, x-1]
    T2 = dataLoader.read_T2_map()[nth_slice-1, y-1, x-1]
    print("T1 observed:", T1, "s")
    print("T2 observed:", 1000*T2, "ms")
    print("ratio observed:", T1/T2)
    # fitting
    fitting = Fitting(offset, Zspec, T1, T2)
    fitting.fit(method=method, show=True)
    evaluation.evaluate(offset, Zspec, fitting.y_estimated, show=True)
    visualization.plot_2_Zspec(offset, Zspec, fitting.y_estimated)
    visualization.plot_2_Zspec(offset[7:], Zspec[7:], fitting.y_estimated[7:])

def fit_by_skull_mask(patient_name, nth_slice, method):
    print("********", patient_name, method, nth_slice, "********")
    dataLoader = DataLoader(patient_name)
    evaluation = Evaluation()
    skull = dataLoader.read_skull()[nth_slice-1] # [256, 256]
    EMR = dataLoader.read_EMR()[:, nth_slice-1, :, :] # [24, 256, 256]
    T1 = dataLoader.read_T1_map()[nth_slice-1] # [256, 256]
    T2 = dataLoader.read_T2_map()[nth_slice-1] # [256, 256]
    WASSR = dataLoader.read_WASSR()[:, nth_slice-1, :, :] # [26, 256, 256]
    x = np.where(skull == 1)[0]
    y = np.where(skull == 1)[1]
    print("ROI size:", x.shape[0])
    data = []
    filename = patient_name+"_slice_"+str(nth_slice)+"_fitted_paras.xlsx"
    phantom = ("phantom" in patient_name.lower())
    print("is phantom:", phantom)
    for i in tqdm(range(x.shape[0])):
        Zspec = EMR[:, x[i], y[i]]
        WASSR_Spec = WASSR[:, x[i], y[i]]
        offset, Zspec = dataLoader.Zspec_preprocess_onepixel(Zspec)
        correction = B0_correction(offset, Zspec, WASSR_Spec)
        Zspec = correction.correct()
        fitting = Fitting(offset, Zspec, T1[x[i], y[i]], T2[x[i], y[i]], B1=1.5, phantom=phantom)
        if not fitting.is_valid():
                continue
        fitting.fit(method=method, show=False)
        if fitting.success: # fittig succeed
            metrics = evaluation.evaluate(method, offset, Zspec, fitting.y_estimated, show=False)
            coordinates = np.array([T1[x[i], y[i]], T2[x[i], y[i]], nth_slice-1, x[i], y[i]])
            data.append(np.concatenate((fitting.fitted_paras, metrics, coordinates,
                                            [correction.B0_shift])))
        if i > 0 and i % 500 == 0:
                save_results(patient_name, method, data, filename)
    save_results(patient_name, method, data, filename)  

def fit_whole_brain(patient_name, method):
    slices = np.arange(1, 16)
    with Pool(15) as pool:
        pool.starmap(fit_by_skull_mask, zip(repeat(patient_name), slices, repeat(method)))

def fit_by_ROI_mask(patient_name, ROI, method, lineshape="SL"):
    print("********", patient_name, method, ROI2filename(ROI), "********")
    dataLoader = DataLoader(patient_name)
    evaluation = Evaluation()
    mask = dataLoader.read_mask() # [15, 256, 256]
    EMR = dataLoader.read_EMR() # [24, 15, 256, 256]
    T1 = dataLoader.read_T1_map() # [15, 256, 256]
    T2 = dataLoader.read_T2_map() # [15, 256, 256]
    WASSR = dataLoader.read_WASSR() # [26, 15, 256, 256]
    data = []
    if ROI == 0: # fit the whole slice based on skull-stripping mask
        skull = dataLoader.read_skull() # [15, 256, 256]
        z = np.where(skull == 1)[0]
        x = np.where(skull == 1)[1]
        y = np.where(skull == 1)[2]
    else: # ROI-based fitting
        mask = dataLoader.read_mask() # [15, 256, 256]
        z = np.where(mask == ROI)[0]
        x = np.where(mask == ROI)[1]
        y = np.where(mask == ROI)[2]
    if x.shape[0] == 0:
        print("No", ROI2filename(ROI), "!")
        return
    else:
        print("ROI size:", x.shape[0])
    filename = ROI2filename(ROI)+"_"+lineshape+"_fitted_paras.xlsx"
    phantom = ("phantom" in patient_name.lower())
    print("is phantom:", phantom)
    for i in tqdm(range(x.shape[0])):
        Zspec = EMR[:, z[i], x[i], y[i]]
        offset, Zspec = dataLoader.Zspec_preprocess_onepixel(Zspec)
        WASSR_Spec = WASSR[:, z[i], x[i], y[i]]
        correction = B0_correction(offset, Zspec, WASSR_Spec)
        Zspec = correction.correct() # B0 corection
        fitting = Fitting(offset, Zspec, T1[z[i], x[i], y[i]], T2[z[i], x[i], y[i]],
                          B1=1.5, phantom=phantom)
        if not fitting.is_valid():
            continue
        fitting.fit(method=method, show=False)
        if fitting.success: # fittig succeed
            metrics = evaluation.evaluate(method, offset, Zspec, 
                                          fitting.y_estimated, show=False)
            coordinates = np.array([T1[z[i], x[i], y[i]], T2[z[i], x[i], y[i]], 
                                    z[i], x[i], y[i]])
            data.append(np.concatenate((fitting.fitted_paras, metrics, coordinates,
                                        [correction.B0_shift])))
        if i > 0 and i % 500 == 0: # save middle results
            save_results(patient_name, method, data, filename)
    save_results(patient_name, method, data, filename)

def fit_by_ROI_mask_single(patient_name, ROI, method, lineshape="SL"):
    print("********", patient_name, method, ROI2filename(ROI), "********")
    df_info = pd.read_excel(r"C:\Users\jwu191\Desktop\Projects\AD_fitting\info.xlsx")
    nth_slice = int(df_info[df_info["patient_name"] == patient_name]["nth_slice"])
    dataLoader = DataLoader(patient_name)
    evaluation = Evaluation()
    EMR_single = dataLoader.read_EMR_single() # [56, 256, 256]
    T1 = dataLoader.read_T1_map()[nth_slice-1] # [256, 256]
    T2 = dataLoader.read_T2_map()[nth_slice-1] # [256, 256]
    WASSR = dataLoader.read_WASSR()[:, nth_slice-1, :, :] # [26, 256, 256]
    data = []
    if ROI == 0: # fit the whole slice based on skull-stripping mask
        skull = dataLoader.read_skull()[nth_slice-1] # [256, 256]
        x = np.where(skull == 1)[0]
        y = np.where(skull == 1)[1]
    else: # ROI-based fitting
        mask = dataLoader.read_mask_single()[nth_slice-1] # [256, 256]
        x = np.where(mask == ROI)[0]
        y = np.where(mask == ROI)[1]
    if x.shape[0] == 0:
        print("No", ROI2filename(ROI), "!")
        return
    else:
        print("ROI size:", x.shape[0])
    if "Lorentz" in method:
        filename = ROI2filename(ROI)+"_single_fitted_paras.xlsx"
    else:
        filename = ROI2filename(ROI)+"_single_"+lineshape+"_fitted_paras.xlsx"
    phantom = ("phantom" in patient_name.lower())
    print("is phantom:", phantom)
    for i in tqdm(range(x.shape[0])):
        Zspec = EMR_single[:, x[i], y[i]]
        offset, Zspec = dataLoader.Zspec_preprocess_onepixel(Zspec, fitting=True)
        WASSR_Spec = WASSR[:, x[i], y[i]]
        correction = B0_correction(offset, Zspec, WASSR_Spec)
        Zspec = correction.correct() # B0 corection
        fitting = Fitting(offset, Zspec, T1[x[i], y[i]], T2[x[i], y[i]], B1=1.5, 
                          phantom=phantom)
        if not fitting.is_valid():
            continue
        fitting.fit(method=method, show=False)
        if fitting.success: # fittig succeed
            metrics = evaluation.evaluate(method, offset, Zspec, fitting.y_estimated, show=False)
            coordinates = np.array([T1[x[i], y[i]], T2[x[i], y[i]], nth_slice-1, x[i], y[i]])
            data.append(np.concatenate((fitting.fitted_paras, metrics, coordinates,
                                        [correction.B0_shift])))
        if i > 0 and i % 500 == 0: # save middle results
            save_results(patient_name, method, data, filename)
    save_results(patient_name, method, data, filename)
    

def fit_by_ROI_mask_Lorentz(patient_name, ROI, method):
    print("********", patient_name, method, ROI2filename(ROI), "********")
    df_info = pd.read_excel(r"C:\Users\jwu191\Desktop\Projects\AD_fitting\info.xlsx")
    nth_slice = int(df_info[df_info["patient_name"] == patient_name]["nth_slice"])
    dataLoader = DataLoader(patient_name)
    evaluation = Evaluation()
    EMR_single = dataLoader.read_EMR_single() # [56, 256, 256]
    WASSR = dataLoader.read_WASSR()[:, nth_slice-1, :, :] # [26, 256, 256]
    data = []
    if ROI == 0: # fit the whole slice based on skull-stripping mask
        skull = dataLoader.read_skull()[nth_slice-1] # [256, 256]
        x = np.where(skull == 1)[0]
        y = np.where(skull == 1)[1]
    else: # ROI-based fitting
        mask = dataLoader.read_mask_single()[nth_slice-1] # [256, 256]
        x = np.where(mask == ROI)[0]
        y = np.where(mask == ROI)[1]
    if x.shape[0] == 0:
        print("No", ROI2filename(ROI), "!")
        return
    else:
        print("ROI size:", x.shape[0])
    phantom = ("phantom" in patient_name.lower())
    print("is phantom:", phantom)
    filename = ROI2filename(ROI)+"_single_fitted_paras.xlsx"
    for i in tqdm(range(x.shape[0])):
        Zspec = EMR_single[:, x[i], y[i]]
        offset, Zspec = dataLoader.Zspec_preprocess_onepixel(Zspec, fitting=True)
        WASSR_Spec = WASSR[:, x[i], y[i]]
        correction = B0_correction(offset, Zspec, WASSR_Spec)
        correction.get_B0_shift()
        B0_shift = correction.B0_shift
        fitting = Fitting(offset, Zspec, phantom=phantom, B0_shift=B0_shift)
        if not fitting.is_valid():
            continue
        fitting.fit(method=method, show=False)
        if fitting.success: # fittig succeed
            metrics = evaluation.evaluate(method, offset, Zspec, fitting.y_estimated, show=False)
            coordinates = np.array([T1[x[i], y[i]], T2[x[i], y[i]], nth_slice-1, x[i], y[i]])
            data.append(np.concatenate((fitting.fitted_paras, metrics, coordinates,
                                        [correction.B0_shift])))
        if i > 0 and i % 500 == 0: # save middle results
            save_results(patient_name, method, data, filename)
    save_results(patient_name, method, data, filename)



