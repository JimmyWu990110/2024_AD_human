import numpy as np
from scipy.optimize import curve_fit
from scipy import stats
import scipy.integrate as integrate
import matplotlib.pyplot as plt
import SimpleITK as sitk
import os
import cv2
import pandas as pd

def plot_highRes(patient_name):
    path = os.path.join(r"C:\Users\jwu191\Desktop\Projects\AD_fitting\processed_data",
                        patient_name+"_all", "HighRes")
    for f in os.listdir(path):
        if f.endswith(".nii"):
            img = sitk.ReadImage(os.path.join(path, f))
            arr = sitk.GetArrayFromImage(img) # [30, 480, 480]
            print(arr.shape)
            arr = np.interp(arr, (arr.min(), arr.max()), (0., 1.))
            arr *= 255
            for i in range(arr.shape[0]):
                    new_name = "HighRes_" + str(i+1) + ".png"
                    cv2.imwrite(os.path.join(path, new_name), np.flip(arr[i], 0))
    # to tile
    h_patch_list = []
    h_patch = []
    n_slices = 15
    for i in range(n_slices):
        img_path = os.path.join(path, "HighRes_"+str(2*i+1)+".png")
        img = cv2.imread(img_path, 0)
        h_patch.append(img)
        if i % 5 == 4:
            tmp = np.concatenate(h_patch, axis=1)
            h_patch_list.append(np.concatenate(h_patch, axis=1))
            h_patch = []
    final_img = np.concatenate(h_patch_list, axis=0)
    cv2.imwrite(os.path.join(path, "HighRes.png"), final_img)
    print("tile shape:", final_img.shape)

def ReadData_T1(base_dir):
    nii_files = []
    for root, dirs, files in os.walk(base_dir):
        for name in files:
            if name.endswith('.nii'):
                nii_files.append(name)
    if nii_files == []:
        print('No files for reading!')
        return
    else:
        all_images = []
        all_t = []
        for name in nii_files:
            name_parts = name.split('_')
            time = name_parts[-1].split('.')[0][1:]
            all_t.append(int(time))
            image = sitk.ReadImage(os.path.join(base_dir, name))
            image_array = sitk.GetArrayFromImage(image)
            all_images.append(image_array)
        all_images = [all_images for _, all_images in sorted(zip(all_t, all_images))]
        all_t = sorted(all_t)
        all_t = np.array(all_t) * 10 ** -3
        all_images = np.array(all_images)
        # Get image physical information for later use
        spacing = image.GetSpacing()
        origin = image.GetOrigin()
        direction = image.GetDirection()
        imginfo = [spacing, origin, direction]
        print("T1 time:", all_t)
        print("T1 image shape:", all_images.shape)
        return all_t, all_images, imginfo

def ReadData_T2(base_dir):
    nii_files = []
    for root, dirs, files in os.walk(base_dir):
        for name in files:
            if name.endswith('.nii'):
                nii_files.append(name)
    if nii_files == []:
        print('No files for reading!')
        return
    else:
        all_images = []
        all_t = []
        for name in nii_files:
            name_parts = name.split('_')
            time = name_parts[-1].split('.')[0][1:]
            try:
                all_t.append(int(time))
            except ValueError:
                continue
            image = sitk.ReadImage(os.path.join(base_dir, name))
            image_array = sitk.GetArrayFromImage(image)
            all_images.append(image_array)
        # Get image physical information for later use
        spacing = image.GetSpacing()
        origin = image.GetOrigin()
        direction = image.GetDirection()
        imginfo = [spacing, origin, direction]
        # Sort data based on time echo
        all_images = [all_images for _, all_images in sorted(zip(all_t, all_images))]
        all_t = sorted(all_t)
        all_t = np.array(all_t) * 0.03
        all_images = np.array(all_images)       
        print("T2 time:", all_t)
        print("T2 image shape:", all_images.shape)
        return all_t, all_images, imginfo

def removebackground(pid, image):
    processedpath = r"C:\Users\jwu191\Desktop\output"
    apt_mask_name = pid + '_11_apt.nii'
    apt_mask = sitk.ReadImage(os.path.join(processedpath, pid+'_all', pid, '2_nifti', pid, apt_mask_name))
    apt_mask = sitk.GetArrayFromImage(apt_mask)
    apt_mask = (apt_mask == -5)
    image_new = image
    image_new[apt_mask] = 0
    return image_new

def func(t, a, b, t1_star):
    return a - b * np.exp(-t / t1_star)

def calculate_t1(a, b, t1_star):
    return t1_star * (b / a - 1)
    
def t1_map_one_slice(all_images, slice_idx, all_t, flag=0):
    # all_images (n, s, x, y)
    # flag = 0: minimal points keep the same, flag = 1:mininal points become negative
    x_data = all_t
    img = all_images[:, slice_idx, :, :]
    res = np.zeros(img[0].shape)
    for x in range(res.shape[0]):
        for y in range(res.shape[1]):
            try:
                if np.sum(img[:, x, y]) == 0:
                    continue
                else:
                    num_points = np.argmin(img[:, x, y])
                    y_data = np.array(img[:, x, y])
                    y_data[:num_points + flag] = -img[:num_points + flag, x, y]
                    popt, pcov = curve_fit(func, x_data, y_data, method='lm', p0=[3e+05, 8e+05, 1],
                                           maxfev=5000)
                    res[x][y] = calculate_t1(*popt)
            except RuntimeError:
                res[x][y] = 0
    return res

def t1_map_3d(all_images, all_t, flag=0):
    res_3d = []
    for sl in range(all_images.shape[1]):
        print("slice", sl, "processing...")
        res_sl = t1_map_one_slice(all_images, sl, all_t, flag)
        res_3d.append(res_sl)
    return np.array(res_3d)

def T1mapping(base_dir, out_dir, ub):
    all_t, all_images, imginfo = ReadData_T1(base_dir)
    t1_map_all = t1_map_3d(all_images, all_t, 1)

    t1_map_all[t1_map_all < 0] = 0
    t1_map_all[t1_map_all > ub] = ub

    # t1_map_all = removebackground(pid, t1_map_all)

    t1_map_img = sitk.GetImageFromArray(t1_map_all)
    t1_map_img.SetSpacing(imginfo[0])
    t1_map_img.SetOrigin(imginfo[1])
    t1_map_img.SetDirection(imginfo[2])
    save_name = 'T1_map' + '.nii'

    sitk.WriteImage(t1_map_img, os.path.join(out_dir, save_name))
    
def func_t2_fitting(t, a, t2):
    return a * np.exp(-t / t2)

def t2_map_one_slice(all_images, sl, all_t, num_points):
    x_data = all_t[-num_points:]
    y_data = np.log(all_images[-num_points:, sl, :, :])
    res = np.zeros(y_data[0].shape)
    for x in range(res.shape[0]):
        for y in range(res.shape[1]):
            slope, intercept, r_value, p_value, std_err = stats.linregress(x_data, y_data[:, x, y])
            res[x][y] = -1 / slope

    return res

def t2_map_3d(all_images, all_t, num_points):
    res_3d = []
    for sl in range(all_images.shape[1]):
        print("slice", sl, "processing...")
        res_sl = t2_map_one_slice(all_images, sl, all_t, num_points)
        res_3d.append(res_sl)
    return np.array(res_3d)

def T2mapping(base_dir, out_dir, ub):
    all_t, all_images, imginfo = ReadData_T2(base_dir)
    num_points = len(all_t)
    t2_map_all = t2_map_3d(all_images, all_t, num_points)
    t2_map_5_copy = np.copy(t2_map_all)
    t2_map_5_copy[np.isnan(t2_map_5_copy) == True] = 0
    t2_map_5_copy[t2_map_5_copy < 0] = 0
    t2_map_5_copy[t2_map_5_copy > ub] = ub
    # t2_map_5_copy = removebackground(pid, t2_map_5_copy)

    t2_map_img = sitk.GetImageFromArray(t2_map_5_copy)
    t2_map_img.SetSpacing(imginfo[0])
    t2_map_img.SetOrigin(imginfo[1])
    t2_map_img.SetDirection(imginfo[2])
    save_name = 'T2_map' + '.nii'
    sitk.WriteImage(t2_map_img, os.path.join(out_dir, save_name))

def get_T1_T2_map(processed_cases_dir, patient_name):
    out_dir = os.path.join(processed_cases_dir, patient_name+"_all")
    # T1mapping(os.path.join(out_dir, "T1"), out_dir, ub=3)
    T2mapping(os.path.join(out_dir, "T2"), out_dir, ub=5) 




       
        
        