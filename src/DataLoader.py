import os
import numpy as np
import pandas as pd
import SimpleITK as sitk
import matplotlib.pyplot as plt


class DataLoader:
    EMR_offset_16 = np.array([np.inf, 80, 60, 40, 30, 20, 12, 8, 
                              4, -4, 3.5, -3.5, 3.5, -3.5, 3, -3])
    EMR_offset_13_unique = np.array([80, 60, 40, 30, 20, 12, 8,
                                     4, -4, 3.5, -3.5, 3, -3])
    # From 2022/03/24, EMR offset 10ppm changed to 12ppm
    EMR_offset_24 = np.array([np.inf, 80, 60, 40, 30, 20, 12, 8, 
                              4, -4, 3.5, -3.5, 3.5, -3.5, 3, -3, 
                              2.5, -2.5, 2, -2, 2, -2, 1.5, -1.5])
    EMR_offset_19_unique = np.array([80, 60, 40, 30, 20, 12, 8,
                                     4, -4, 3.5, -3.5, 3, -3, 
                                     2.5, -2.5, 2, -2, 1.5, -1.5])
    EMR_offset_56 = np.array([np.inf, 0, 0.25, -0.25, 0.5, -0.5, 0.75, -0.75, 
                              1, -1, 1.25, -1.25, 1.5, -1.5, 1.75, -1.75, 
                              2, -2, 2.25, -2.25, 2.5, -2.5, 2.75, -2.75, 
                              3, -3, 3.25, -3.25, 3.5, -3.5, 3.75, -3.75, 
                              4, -4, 4.25, -4.25, 4.5, -4.5, 4.75, -4.75, 
                              5, -5, 8, -8, 12, -12, 20, -20, 
                              30, -30, 40, -40, 60, -60, 80, -80])
    EMR_offset_55_unique = EMR_offset_56[1:]
    EMR_offset_54_unique = EMR_offset_56[2:]
    
    def __init__(self, patient_name="AD_70", view="Coronal", B1=1.5):
        self.patient_name = patient_name
        self.view = view
        self.processed_cases_dir = r"C:\Users\jwu191\Desktop\Projects\AD_fitting\processed_data"
        self.data_dir = os.path.join(self.processed_cases_dir, patient_name+"_all", view)
        self.mask_dir = r"C:\Users\jwu191\Desktop\EMR fitting\Zspec_mask"
        self.seq_dict = None
        self.nth_slice = -1
    
    def _get_geometry(self, img):
        print("****** get geometry ******")
        arr = sitk.GetArrayFromImage(img)
        print(arr.shape)
        spacing = img.GetSpacing()
        origin = img.GetOrigin()
        direction = img.GetDirection()
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
        path = os.path.join(self.data_dir, "EMR-nifti")
        for f in os.listdir(path):
            if "EMR" in f and "Zspec" not in f and f.endswith(".nii"):
                img = sitk.ReadImage(os.path.join(path, f)) 
                spacing, origin, direction = self._get_geometry(img)
                return sitk.GetArrayFromImage(img) # [24, 15, 256, 256]
            
    def read_WASSR(self):
        print("******** Read WASSR ********")
        path = os.path.join(self.data_dir, "WASSR-nifti")
        for f in os.listdir(path):
            if "WASSR" in f and f.endswith(".nii"):
                img = sitk.ReadImage(os.path.join(path, f)) 
                spacing, origin, direction = self._get_geometry(img)
                return sitk.GetArrayFromImage(img) # [26, 15, 128, 128]
            
    def read_Zspec_single(self):
        print("******** Read Zspec_single ********")
        path = os.path.join(self.data_dir, "EMR-nifti")
        for f in os.listdir(path):
            if "EMR" in f and "Zspec" in f and f.endswith(".nii"):
                img = sitk.ReadImage(os.path.join(path, f)) 
                spacing, origin, direction = self._get_geometry(img)
                return np.squeeze(sitk.GetArrayFromImage(img)) # [56, 256, 256]

    def read_EMR_56(self):
        print("******** Read EMR_56 ********")
        img = sitk.ReadImage(os.path.join(self.data_dir, "EMR_56.nii"))
        spacing, origin, direction = self._get_geometry(img)
        return np.squeeze(sitk.GetArrayFromImage(img))

    def read_M0(self):
        print("******** Read M0 ********")
        img = sitk.ReadImage(os.path.join(self.data_dir, "M0.nii"))
        spacing, origin, direction = self._get_geometry(img)
        return sitk.GetArrayFromImage(img) # [15, 256, 256]

    def read_APTw(self):
        print("******** Read APTw ********")
        img = sitk.ReadImage(os.path.join(self.data_dir, "APTw.nii"))
        spacing, origin, direction = self._get_geometry(img)
        return sitk.GetArrayFromImage(img) # [15, 256, 256]

    def read_T1_map(self):
        print("******** Read T1 map ********")
        img = sitk.ReadImage(os.path.join(self.data_dir, "T1_map.nii"))
        spacing, origin, direction = self._get_geometry(img)
        return sitk.GetArrayFromImage(img) # [15, 256, 256]

    def read_T2_map(self):
        print("******** Read T2 map ********")
        img = sitk.ReadImage(os.path.join(self.data_dir, "T2_map.nii"))
        spacing, origin, direction = self._get_geometry(img)
        return sitk.GetArrayFromImage(img) # [15, 256, 256]
    
    def read_skull(self):
        print("******** Read skull ********")
        img = sitk.ReadImage(os.path.join(self.data_dir, "skull.nii.gz"))
        return sitk.GetArrayFromImage(img)
    
    def read_mask(self):
        print("******** Read mask ********")
        img = sitk.ReadImage(os.path.join(self.data_dir, "mask.nii.gz"))
        return sitk.GetArrayFromImage(img)
    
    def get_Zspec_by_coordinate_24(self, nth_slice, x, y):
        EMR_img = self.read_EMR() # [24, 15, 256, 256]
        Zspec = EMR_img[:, nth_slice, x, y]
        return Zspec

    def get_Zspec_by_coordinate_56(self, x, y):
        EMR_img = self.read_Zspec_single()
        # Zspec = sitk.GetArrayFromImage(EMR_img)[:, 0, x, y] # [56, 1, 256, 256]
        return EMR_img[:, x, y]
    
    def Zspec_preprocess_onepixel(self, Zspec):
        """
        sort Zspec with selected offsets from positive to negative
        then normalize it by M0
        """
        if Zspec.shape[0] == 16:
            scanned_offset = self.EMR_offset_16
            selected_offset = self.EMR_offset_13_unique
        if Zspec.shape[0] == 24:
            scanned_offset = self.EMR_offset_24
            # selected_offset = self.EMR_offset_19_unique
            selected_offset = self.EMR_offset_13_unique
        if Zspec.shape[0] == 56:
            scanned_offset = self.EMR_offset_56
            selected_offset = self.EMR_offset_54_unique
        sort_index = np.argsort(selected_offset)[::-1]
        selected_offset = selected_offset[sort_index]
        processed_Zspec = []
        M0 = Zspec[np.where(scanned_offset == np.inf)][0]
        for x in selected_offset:
            Mx = Zspec[np.where(scanned_offset == x)]
            processed_Zspec.append(np.mean(Mx) / M0)
        # print(selected_offset, np.array(processed_Zspec))
        return selected_offset, np.array(processed_Zspec)
    
    def _test_coordinate_helper(self, img, title):
        min_ = np.min(img)
        for i in range(190, 200):
            for j in range(90, 100):
                img[j][i] = min_
        plt.imshow(img, origin="lower")
        plt.title(title)
        plt.show()

    def test_coordinate(self, nth_slice):
        M0 = self.read_M0()[nth_slice-1] # [256, 256]
        self._test_coordinate_helper(M0, "M0")
        T1_map = self.read_T1_map()[nth_slice-1] # [224, 224] -> [256, 256]
        self._test_coordinate_helper(T1_map, "T1 map")
        T2_map = self.read_T2_map()[nth_slice-1] # [224, 224] -> [256, 256]
        self._test_coordinate_helper(T2_map, "T2 map")
        single = self.read_Zspec_single()[0] # [256, 256]
        self._test_coordinate_helper(single, "single")
        
        # EMR = self.read_EMR()[0, nth_slice-1, :, :] # [256, 256]
        # self._test_coordinate_helper(EMR)
        # EMR_56 = self.read_EMR_56()[0, :, :] # [256, 256]
        # self._test_coordinate_helper(EMR_56)



















    def get_ROIs_by_Zspec_mask(self):
        self.get_nth_slice_by_Zspec_mask()
        mask_path = os.path.join(self.mask_dir, self.patient_name+".nii.gz")
        mask = sitk.ReadImage(mask_path, sitk.sitkInt32)
        mask_arr = sitk.GetArrayFromImage(mask)[self.nth_slice-1]
        tumor_roi = np.where(mask_arr == 1)
        normal_roi = np.where(mask_arr == 2)
        print("tumor size:", tumor_roi[0].shape[0])
        print("normal size:", normal_roi[0].shape[0])
        return tumor_roi, normal_roi
    
    def get_ROIs_by_phantom_mask(self, label):
        mask_path = os.path.join(self.processed_cases_dir, self.patient_name+"_all",
                                 self.patient_name, "4_coreg2apt", self.patient_name, 
                                 self.patient_name+"_mask.nii.gz")
        mask = sitk.ReadImage(mask_path, sitk.sitkInt32)
        mask_array = sitk.GetArrayFromImage(mask)[self.nth_slice-1]
        roi = None
        if label == 9999:
            roi = np.where(mask_array != 0)
        else:
            roi = np.where(mask_array == label)
        print("phantom size:", roi[0].shape[0])
        return roi
        
    def read_Zspec(self, B1):
        print("******** read Zspec ********")
        EMR_path = os.path.join(self.data_dir, "EMR-nifti")
        B1_str = None
        if B1 == 1.5:
            B1_str = "1p5uT"
        elif B1 == 2:
            B1_str = "2uT"
        else:
            raise Exception("Invalid B1, only supports 1.5 and 2.")
        for f in os.listdir(EMR_path):
            if "ZspecEMR" in f and B1_str in f and f.endswith('.nii'):
                print(f, "readed")
                Zspec = sitk.ReadImage(os.path.join(EMR_path, f))
                Zspec_arr = sitk.GetArrayFromImage(Zspec)
                # [56, 1, 256, 256] -> [56, 256, 256]
                self.Zspec_img = np.squeeze(Zspec_arr)
                print("Zspec img shape:", self.Zspec_img.shape)
                return
        raise Exception("Zspec file not found!")
        
    def get_WASSR_Zspec(self):
        print("******** read WASSR for Zspec ********")
        seq_id = self.seq_dict["WASSR_Zspec_id"]
        base_dir = os.path.join(self.data_dir, "WASSR-nifti")
        files = os.listdir(base_dir)
        for f in files:
            if f.endswith("_"+str(seq_id)+".nii"):
                # return os.path.join(base_dir, f)
                img = sitk.ReadImage(os.path.join(base_dir, f))
                arr = sitk.GetArrayFromImage(img)
                print("read", f, "shape", arr.shape)
                return arr
        raise Exception("WASSR_Zspec file not found!")
    















    def get_tumor_avg_Zspec(self, offset):
        Zspec = []
        for x in offset:
            data = self.Zspec_img[np.where(self.Zspec_56_offset_ppm==x)][0] # 256*256
            Zspec.append(np.mean(data[self.tumor_roi]))
        M0 = self.Zspec_img[np.where(self.Zspec_56_offset_ppm==np.inf)][0] # 256*256
        Zspec /= np.mean(M0[self.tumor_roi])
        return np.array(Zspec)
    
    def get_normal_avg_Zspec(self, offset):
        Zspec = []
        for x in offset:
            data = self.Zspec_img[np.where(self.Zspec_56_offset_ppm==x)][0] # 256*256
            Zspec.append(np.mean(data[self.normal_roi]))
        M0 = self.Zspec_img[np.where(self.Zspec_56_offset_ppm==np.inf)][0] # 256*256
        Zspec /= np.mean(M0[self.tumor_roi])
        return np.array(Zspec)
    
    def get_T1_map(self):
        img = sitk.ReadImage(os.path.join(self.data_dir, "t1_map.nii"))
        self.T1_map = sitk.GetArrayFromImage(img)[self.nth_slice]
        
    def get_T2_map(self):
        img = sitk.ReadImage(os.path.join(self.data_dir, "t2_map.nii"))
        self.T2_map = sitk.GetArrayFromImage(img)[self.nth_slice]







def get_phantom_mask(patient_name, nth_slice, phantom_label):
    mask_path = os.path.join(r"C:\Users\jwu191\Desktop\processed_cases", patient_name+"_all",
                 patient_name, "4_coreg2apt", patient_name, patient_name+"_mask.nii.gz")
    # mask_path = r"C:\Users\jwu191\Desktop\processed_cases\20230309_phantom_all\20230309_phantom\4_coreg2apt\20230309_phantom\20230309_phantom_mask.nii.gz"
    mask = sitk.ReadImage(mask_path, sitk.sitkInt32)
    mask_array = sitk.GetArrayFromImage(mask)[nth_slice-1]
    return np.where(mask_array == phantom_label)

def get_phantom_mask_all(nth_slice):
    mask_path = r"C:\Users\jwu191\Desktop\processed_cases\20230309_phantom_all\20230309_phantom\4_coreg2apt\20230309_phantom\20230309_phantom_mask.nii.gz"
    mask = sitk.ReadImage(mask_path, sitk.sitkInt32)
    mask_array = sitk.GetArrayFromImage(mask)[nth_slice-1]
    return np.where(mask_array == 0)

def read_Zspec_by_id(base_dir, seq_id):
    print("******** read Zspec ********")
    EMR_path = os.path.join(base_dir, "EMR-nifti")
    for f in os.listdir(EMR_path):
        if "ZspecEMR" in f and f.endswith(str(seq_id)+".nii"):
            print("read", f)
            Zspec_img = sitk.ReadImage(os.path.join(EMR_path, f))
            Zspec_array = sitk.GetArrayFromImage(Zspec_img)
            return np.squeeze(Zspec_array)
    raise Exception("Zspec file not found!")

def offset_preprocess(offset):
    offset = np.array(offset)
    sorted_index = np.argsort(offset)[::-1]
    return offset[sorted_index]


        




    


  