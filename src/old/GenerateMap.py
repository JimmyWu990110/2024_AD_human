import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import SimpleITK as sitk

from DataLoader import DataLoader

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
            gray_img[i,j,0] = lookup_table[g_val,3] # blue channel
            gray_img[i,j,1] = lookup_table[g_val,2] # green channel
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

def generate_map(patient_name, nth_slice, method):
    filename = patient_name+"_slice_"+str(nth_slice)+".xlsx" 
    dataLoader = DataLoader(patient_name)
    df = pd.read_excel(os.path.join(r"C:\Users\jwu191\Desktop\Projects\AD_fitting\results",
                                    patient_name, method, filename))
    print("data size:", df.shape)
    # to ensure that the non-skull area (background) is the smallest (black)
    APT_pow_map = -10*np.ones((256, 256))
    NOE_pow_map = -10*np.ones((256, 256))
    APTw_map = -10*np.ones((256, 256))
    M0_map = np.zeros((256, 256))
    APT_pow = np.array(df["APT#"])
    NOE_pow = np.array(df["NOE#"])
    APTw = np.array(df["APTw"])
    M0 = dataLoader.read_M0()
    x_all = np.array(df["x"])
    y_all = np.array(df["y"])
    z_all = np.array(df["z"])
    for i in range(df.shape[0]):
        x = int(x_all[i])
        y = int(y_all[i])
        z = int(z_all[i])
        APT_pow_map[x][y] = APT_pow[i]
        NOE_pow_map[x][y] = NOE_pow[i]
        APTw_map[x][y] = APTw[i]
        M0_map[x][y] = M0[z][x][y]
    print("APTw:", np.min(APTw), np.max(APTw))
    print("APT#:", np.min(APT_pow), np.max(APT_pow))
    print("NOE#:", np.min(NOE_pow), np.max(NOE_pow))
    print("M0:", np.min(M0), np.max(M0))
    APTw_map[APTw_map <= -4] = -4
    APTw_map[APTw_map >= 4] = 4
    APT_pow_map[APT_pow_map <= -2] = -2
    APT_pow_map[APT_pow_map >= 6] = 6
    NOE_pow_map[NOE_pow_map <= -2] = -2
    NOE_pow_map[NOE_pow_map >= 6] = 6
    path = os.path.join(r"C:\Users\jwu191\Desktop\Projects\AD_fitting\results", 
                        patient_name, method)
    generate_map_helper(path, APTw_map, patient_name+"_APTw")
    generate_map_helper(path, APT_pow_map, patient_name+"_APT#")
    generate_map_helper(path, NOE_pow_map, patient_name+"_NOE#")
    generate_map_helper(path, M0_map, patient_name+"_M0")

def png_to_nifti(patient_name):
    base_dir = os.path.join(r"C:\Users\jwu191\Desktop\Projects\AD_fitting\results", patient_name)
    for f in ["APT#", "NOE#", "M0"]:
        img = cv2.imread(os.path.join(base_dir, patient_name+"_"+f+".png"), 0)
        img = sitk.GetImageFromArray(img)
        sitk.WriteImage(img, os.path.join(base_dir, patient_name+"_"+f+".nii"))

def filter_img(patient_name, nth_slice=8):
    base_dir = os.path.join(r"C:\Users\jwu191\Desktop\Projects\AD_fitting\results", patient_name)
    # M0 = cv2.imread(os.path.join(base_dir, patient_name+"_M0.png"), 0)
    # M0 = np.flip(M0, 0)
    
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
        M0[x][y] = M0_src[z][x][y]
        APT_pow[x][y] = data[i][8]
        NOE_pow[x][y] = data[i][9]
        APTw[x][y] = data[i][10]
        if M0[x][y] > 1.35e7:
            tmp = APT_pow[x][y] - 3.5
            if tmp > 0:
                APT_pow[x][y] -= tmp
                NOE_pow[x][y] -= tmp
    APTw[APTw <= -4] = -4
    APTw[APTw >= 4] = 4
    APT_pow[APT_pow <= -2] = -2
    APT_pow[APT_pow >= 6] = 6
    NOE_pow[NOE_pow <= -2] = -2
    NOE_pow[NOE_pow >= 6] = 6
    path = os.path.join(r"C:\Users\jwu191\Desktop\Projects\AD_fitting\results", patient_name)
    generate_map_helper(path, APTw, patient_name+"_APTw_filtered")
    generate_map_helper(path, APT_pow, patient_name+"_APT#_filtered")
    generate_map_helper(path, NOE_pow, patient_name+"_NOE#_filtered")
    
    # APT_pow = cv2.imread(os.path.join(base_dir, patient_name+"_APT#.png"), 0)
    # NOE_pow = cv2.imread(os.path.join(base_dir, patient_name+"_NOE#.png"), 0)
    # for i in range(M0.shape[0]):
    #     for j in range(M0.shape[1]):
    #         if M0[i][j] > 200:
    #             x = APT_pow[i][j] - 190
    #             if x > 0:
    #                 APT_pow[i][j] -= x
    #                 NOE_pow[i][j] -= x
            # elif M0[i][j] > 155:
            #     x = APT_pow[i][j] - 180
            #     if x > 0:
            #         APT_pow[i][j] -= x
            #         NOE_pow[i][j] -= x
            # elif M0[i][j] > 135:
            #     x = APT_pow[i][j] - 175
            #     if x > 0:
            #         APT_pow[i][j] -= x
            #         NOE_pow[i][j] -= x
    # lookup_table = read_lookup_table(r"C:\Users\jwu191\Desktop\Data\idl_rainbow.lut")
    # cv2.imwrite(os.path.join(base_dir, patient_name+"_APT#_filtered.png"), APT_pow) 
    # cv2.imwrite(os.path.join(base_dir, patient_name+"_NOE#_filtered.png"), NOE_pow)
    # APT_pow = cv2.imread(os.path.join(base_dir, patient_name+"_APT#_filtered.png"), 1)
    # NOE_pow = cv2.imread(os.path.join(base_dir, patient_name+"_NOE#_filtered.png"), 1)
    # APT_pow = gray_to_idl(APT_pow, lookup_table)
    # NOE_pow = gray_to_idl(NOE_pow, lookup_table)
    # cv2.imwrite(os.path.join(base_dir, patient_name+"_APT#_filtered.png"), APT_pow) 
    # cv2.imwrite(os.path.join(base_dir, patient_name+"_NOE#_filtered.png"), NOE_pow)

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




generate_map(patient_name="AD_48", nth_slice=8, method="MT_EMR")


