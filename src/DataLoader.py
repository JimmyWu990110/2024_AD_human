import os
import numpy as np
import pandas as pd
import SimpleITK as sitk
import matplotlib.pyplot as plt

class Offsets:
    def __init__(self):
        self.EMR_offset_16 = np.array([np.inf, 80, 60, 40, 30, 20, 10, 8,
                                       4, -4, 3.5, -3.5, 3.5, -3.5, 3, -3])
        self.EMR_offset_13_unique_old = np.array([80, 60, 40, 30, 20, 10, 8,
                                                  4, -4, 3.5, -3.5, 3, -3])
    # From 2022/03/24, EMR offset 10ppm changed to 12ppm
        self.EMR_offset_13_unique_new = np.array([80, 60, 40, 30, 20, 12, 8,
                                                  4, -4, 3.5, -3.5, 3, -3])
        self.EMR_offset_24 = np.array([np.inf, 80, 60, 40, 30, 20, 12, 8, 
                                       4, -4, 3.5, -3.5, 3.5, -3.5, 3, -3, 
                                       2.5, -2.5, 2, -2, 2, -2, 1.5, -1.5])
        self.EMR_offset_19_unique = np.array([80, 60, 40, 30, 20, 12, 8,
                                              4, -4, 3.5, -3.5, 3, -3, 
                                              2.5, -2.5, 2, -2, 1.5, -1.5])
        self.EMR_offset_56 = np.array([np.inf, 0, 0.25, -0.25, 0.5, -0.5, 0.75, -0.75, 
                                       1, -1, 1.25, -1.25, 1.5, -1.5, 1.75, -1.75, 
                                       2, -2, 2.25, -2.25, 2.5, -2.5, 2.75, -2.75, 
                                       3, -3, 3.25, -3.25, 3.5, -3.5, 3.75, -3.75, 
                                       4, -4, 4.25, -4.25, 4.5, -4.5, 4.75, -4.75, 
                                       5, -5, 8, -8, 12, -12, 20, -20, 30, -30, 
                                       40, -40, 60, -60, 80, -80])
        self.EMR_offset_55_unique = self.EMR_offset_56[1:]
        self.EMR_offset_54_unique = self.EMR_offset_56[2:]
        self.EMR_56_test_v0 = np.array([0, 0.5, -0.5, 1, -1, 1.5, -1.5, 2, -2, 
                                        2.5, -2.5, 3, -3, 3.5, -3.5, 4, -4, 
                                        4.5, -4.5, 5, -5, 8, -8, 12, -12, 
                                        20, -20, 30, -30, 40, -40, 60, -60, 80, -80])
        self.EMR_56_test_v1 = np.array([0, 1.5, -1.5, 2, -2, 2.5, -2.5, 3, -3, 
                                        3.5, -3.5, 4, -4, 8, -8, 12, -12, 20, -20, 
                                        30, -30, 40, -40, 60, -60, 80, -80])


class DataLoader:  
    def __init__(self, patient_name="AD_86"):
        self.offsets = Offsets()
        self.patient_name = patient_name
        self.processed_cases_dir = r"C:\Users\jwu191\Desktop\Projects\AD_fitting\processed_data"
        self.data_dir = os.path.join(self.processed_cases_dir, patient_name+"_all")
    
    def _get_geometry(self, img, show=False):
        spacing = img.GetSpacing()
        origin = img.GetOrigin()
        direction = img.GetDirection()
        if show:
            print("****** get geometry ******")
            print(spacing)
            print(origin)
            print(direction)
        return spacing, origin, direction
    
    def _set_geometry(self, spacing, origin, direction, img):
        img.SetSpacing(spacing)
        img.SetOrigin(origin)
        img.SetDirection(direction)
        return img

    def read_EMR(self):
        print("******** Read EMR ********")
        img = sitk.ReadImage(os.path.join(self.data_dir, "EMR_reg.nii"))
        arr = sitk.GetArrayFromImage(img)  # [24, 15, 256, 256]
        print(arr.shape)
        return arr
    
    def read_EMR_single(self):
        print("******** Read EMR single ********")
        img = sitk.ReadImage(os.path.join(self.data_dir, "EMR_single_reg.nii"))
        arr = np.squeeze(sitk.GetArrayFromImage(img))  # [56, 256, 256]
        print(arr.shape)
        return arr
            
    def read_WASSR(self):
        print("******** Read WASSR ********")
        img = sitk.ReadImage(os.path.join(self.data_dir, "WASSR_reg.nii"))
        arr = sitk.GetArrayFromImage(img)  # [26, 15, 256, 256]
        print(arr.shape)
        return arr

    def read_M0(self):
        print("******** Read M0 ********")
        img = sitk.ReadImage(os.path.join(self.data_dir, "M0.nii"))
        arr = sitk.GetArrayFromImage(img)  # [15, 256, 256]
        print(arr.shape)
        return arr

    def read_APTw(self):
        print("******** Read APTw ********")
        img = sitk.ReadImage(os.path.join(self.data_dir, "APTw.nii"))
        arr = sitk.GetArrayFromImage(img)  # [15, 256, 256]
        print(arr.shape)
        return arr

    def read_T1_map(self):
        print("******** Read T1 map ********")
        img = sitk.ReadImage(os.path.join(self.data_dir, "T1_map_reg.nii"))
        arr = sitk.GetArrayFromImage(img)  # [15, 256, 256]
        print(arr.shape)
        return arr

    def read_T2_map(self):
        print("******** Read T2 map ********")
        img = sitk.ReadImage(os.path.join(self.data_dir, "T2_map_reg.nii"))
        arr = sitk.GetArrayFromImage(img)  # [15, 256, 256]
        print(arr.shape)
        return arr
    
    def read_skull(self):
        print("******** Read skull ********")
        img = sitk.ReadImage(os.path.join(self.data_dir, "skull.nii.gz"))
        arr = sitk.GetArrayFromImage(img)  # [15, 256, 256]
        print(arr.shape)
        return arr
    
    def read_mask(self):
        print("******** Read mask ********")
        img = sitk.ReadImage(os.path.join(self.data_dir, "mask.nii.gz"))
        arr = sitk.GetArrayFromImage(img)  # [15, 256, 256]
        print(arr.shape)
        return arr
    
    def read_mask_single(self):
        print("******** Read mask ********")
        img = sitk.ReadImage(os.path.join(self.data_dir, "mask_single.nii.gz"))
        arr = sitk.GetArrayFromImage(img)  # [15, 256, 256]
        print(arr.shape)
        return arr
    
    def get_Zspec_by_coordinate(self, nth_slice, x, y):
        EMR_img = self.read_EMR()  # [24, 15, 256, 256]
        Zspec = EMR_img[:, nth_slice-1, x, y]  # [24,]
        return Zspec
    
    def get_Zspec_by_coordinate_single(self, x, y):
        EMR_img = self.read_EMR_single()  # [56, 256, 256]
        Zspec = EMR_img[:, x, y]  # [56,]
        return Zspec
    
    def Zspec_preprocess_onepixel(self, Zspec, fitting=False):
        """
        sort Zspec with selected offsets from positive to negative
        then normalize it by M0
        """
        if Zspec.shape[0] == 16:
            scanned_offset = self.offsets.EMR_offset_16
            selected_offset = self.offsets.EMR_offset_13_unique_old # 10ppm
        if Zspec.shape[0] == 24:
            scanned_offset = self.offsets.EMR_offset_24
            selected_offset = self.offsets.EMR_offset_19_unique
            # selected_offset = self.offsets.EMR_offset_13_unique_new # 12ppm
        if Zspec.shape[0] == 56:
            scanned_offset = self.offsets.EMR_offset_56
            if fitting:
                selected_offset = self.offsets.EMR_56_test_v0
                # selected_offset = self.offsets.EMR_offset_19_unique
            else:
                selected_offset = self.offsets.EMR_offset_55_unique
        sort_index = np.argsort(selected_offset)[::-1]
        selected_offset = selected_offset[sort_index]
        processed_Zspec = []
        M0 = Zspec[np.where(scanned_offset == np.inf)][0]
        for x in selected_offset:
            Mx = Zspec[np.where(scanned_offset == x)]
            processed_Zspec.append(np.mean(Mx) / M0)
        return selected_offset, np.array(processed_Zspec)

    def get_offset(self, fitting=False, single=False):
        if single:
            Zspec = self.get_Zspec_by_coordinate_single(128, 128)
            offset, _ = self.Zspec_preprocess_onepixel(Zspec, fitting)
        else:
            Zspec = self.get_Zspec_by_coordinate(7, 128, 128)
            offset, _ = self.Zspec_preprocess_onepixel(Zspec, fitting)
        return offset
    
    def get_offset_by_Zspec(self, Zspec):
        if Zspec.shape[0] == 19:
            Zspec = self.get_Zspec_by_coordinate(7, 128, 128)
            offset, _ = self.Zspec_preprocess_onepixel(Zspec)
            return offset




        




    


  