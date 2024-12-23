import os
import numpy as np
import SimpleITK as sitk
from scipy import interpolate


def cal_B0_shift_map(processed_cases_dir, patient_name):
    # 26 WASSR frequencies, in hz
    freq = np.array([np.inf, 0, 14, -14, 28, -28, 42,-42, 56, -56, 70, -70, 84, -84, 
                     98, -98, 112, -112, 126, -126, 140, -140, 154, -154, 168, -168])
    base_dir = os.path.join(processed_cases_dir, patient_name+"_all")
    WASSR = sitk.ReadImage(os.path.join(base_dir, "WASSR_reg.nii"))
    WASSR_map = sitk.GetArrayFromImage(WASSR)  # [26, 15, 256, 256]
    B0_shift_map = np.zeros((WASSR_map.shape[1], WASSR_map.shape[2], WASSR_map.shape[3]))
    skull_mask = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(base_dir, "skull.nii.gz")))
    for i in range(B0_shift_map.shape[0]):
        print("processing slice", i, "...")
        for j in range(B0_shift_map.shape[1]):
            for k in range(B0_shift_map.shape[2]):                  
                WASSRspec = WASSR_map[:, i, j, k]
                sort_index = np.argsort(freq)
                x_sorted = freq[sort_index]
                y_sorted = WASSRspec[sort_index]
                m0 = y_sorted[-1]  # normalized by M0, but fitting not include M0
                if skull_mask[i][j][k] == 0 or m0 < 1e-6:
                    continue
                paras = np.polyfit(x_sorted[:-1], y_sorted[:-1]/m0, deg=12)
                p = np.poly1d(paras)
                x_upsampled = np.arange(-168, 169, 1)
                index = np.argmin(p(x_upsampled))
                B0_shift_map[i][j][k] = x_upsampled[index]/128  # in ppm
    B0_shift_img = sitk.GetImageFromArray(B0_shift_map)
    M0 = sitk.ReadImage(os.path.join(base_dir, "M0.nii"))
    B0_shift_img.SetOrigin(M0.GetOrigin())
    B0_shift_img.SetSpacing(M0.GetSpacing())
    B0_shift_img.SetDirection(M0.GetDirection())
    sitk.WriteImage(B0_shift_img, os.path.join(base_dir, "B0_shift_map.nii"))

def linear_interp_twoside(offset, Zspec, B0_shift):
    real_offset = offset - B0_shift
    fp = interpolate.interp1d(x=real_offset[real_offset > 0], y=Zspec[real_offset > 0], 
                              kind="linear", fill_value="extrapolate")
    fn = interpolate.interp1d(x=real_offset[real_offset < 0], y=Zspec[real_offset < 0], 
                              kind="linear", fill_value="extrapolate")
    return np.concatenate((fp(offset[real_offset > 0]), fn(offset[real_offset < 0])))

def cal_APTw(processed_cases_dir, patient_name):
    # 24 EMR offsets, in ppm
    offset = np.array([np.inf, 80, 60, 40, 30, 20, 10, 8, 4, -4, 3.5, -3.5, 3.5, -3.5, 
                       3, -3, 2.5, -2.5, 2, -2, 2, -2, 1.5, -1.5])
    base_dir = os.path.join(processed_cases_dir, patient_name+"_all")
    EMR = sitk.ReadImage(os.path.join(base_dir, "EMR_reg.nii"))
    EMR_map = sitk.GetArrayFromImage(EMR) # [24, 15, 256, 256]
    B0_shift_map = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(base_dir, "B0_shift_map.nii")))
    skull_mask = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(base_dir, "skull.nii.gz")))
    APTw = -5 * np.ones((EMR_map.shape[1], EMR_map.shape[2], EMR_map.shape[3]))
    for i in range(APTw.shape[0]):
        print("processing slice", i, "...")
        for j in range(APTw.shape[1]):
            for k in range(APTw.shape[2]):
                Zspec = EMR_map[:, i, j, k]
                if skull_mask[i][j][k] == 0 or Zspec[0] < 1e-6:
                    continue
                Zspec /= Zspec[0] # normalize by M0
                B0_shift = 128*B0_shift_map[i][j][k] # in hz
                pos_384_hz = np.mean(Zspec[np.where(offset == 3)[0]])
                pos_448_hz = np.mean(Zspec[np.where(offset == 3.5)[0]])
                pos_512_hz = np.mean(Zspec[np.where(offset == 4)[0]])
                x_pos = np.array([384, 448, 512])
                y_pos = np.array([pos_384_hz, pos_448_hz, pos_512_hz])
                x_interp_pos = np.arange(448-168, 448+168+1, 1) # [280, 281, ... 615, 616]
                func_pos = interpolate.interp1d(x_pos, y_pos, "linear", fill_value="extrapolate")
                y_interp_pos = func_pos(x_interp_pos)
                pos_448_hz_corrected = y_interp_pos[np.where(x_interp_pos == 448+B0_shift)][0]
                neg_384_hz = np.mean(Zspec[np.where(offset == -3)[0]])
                neg_448_hz = np.mean(Zspec[np.where(offset == -3.5)[0]])
                neg_512_hz = np.mean(Zspec[np.where(offset == -4)[0]])
                x_neg = np.array([-512, -448, -384])
                y_neg = np.array([neg_512_hz, neg_448_hz, neg_384_hz])
                x_interp_neg = np.arange(-448-168, -448+168+1, 1) # [-616, -615, ... -281, -280]
                func_neg = interpolate.interp1d(x_neg, y_neg, "linear", fill_value="extrapolate")
                y_interp_neg = func_neg(x_interp_neg)
                neg_448_hz_corrected = y_interp_neg[np.where(x_interp_neg == -448+B0_shift)][0]
                APTw[i][j][k] = 100 * (neg_448_hz_corrected-pos_448_hz_corrected) # percent
    APTw[APTw > 5] = 5
    APTw[APTw < -5] = -5
    img = sitk.GetImageFromArray(APTw)
    M0 = sitk.ReadImage(os.path.join(base_dir, "M0.nii"))
    img.SetOrigin(M0.GetOrigin())
    img.SetSpacing(M0.GetSpacing())
    img.SetDirection(M0.GetDirection())
    sitk.WriteImage(img, os.path.join(base_dir, "APTw.nii"))
    

