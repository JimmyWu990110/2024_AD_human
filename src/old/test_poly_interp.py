import numpy as np
import matplotlib.pyplot as plt

from DataLoader import DataLoader
from B0_correction import B0_correction
from funcs import remove_large_offset

def filter_offset(offset, Zspec, selected_offset):
    new_offset = []
    new_Zspec = []
    for i in range(offset.shape[0]):
        if offset[i] in selected_offset:
            new_offset.append(offset[i])
            new_Zspec.append(Zspec[i])
    return np.array(new_offset), np.array(new_Zspec)

patient_name = "Phantom_20230620"
nth_slice = 8
ROI = 1
dataloader = DataLoader(patient_name)
EMR_single = dataloader.read_EMR_single()
WASSR = dataloader.read_WASSR()[:, nth_slice-1, :, :]
mask = dataloader.read_mask()[nth_slice-1]
x = np.where(mask == ROI)[0]
y = np.where(mask == ROI)[1]
print("ROI size:", x.shape[0])
for i in range(100, 101):
    Zspec = EMR_single[:, 133, 171]
    offset_56, Zspec_56 = dataloader.Zspec_preprocess_onepixel(Zspec)
    WASSR_Spec = WASSR[:, 133, 171]
    correction = B0_correction(offset_56, Zspec_56, WASSR_Spec)
    Zspec_56 = correction.correct() # B0 corection
    offset_24, Zspec_24 = filter_offset(offset_56, Zspec_56, dataloader.offsets.EMR_offset_19_unique)
    offset_mid_56, Zspec_mid_56 = remove_large_offset(offset_56, Zspec_56)
    offset_mid_24, Zspec_mid_24 = remove_large_offset(offset_24, Zspec_24)
    paras_24 = np.polyfit(offset_24[7:], Zspec_24[7:], deg=8)
    p_24 = np.poly1d(paras_24) # polynomial function
    paras_56 = np.polyfit(offset_56[7:-7], Zspec_56[7:-7], deg=8)
    p_56 = np.poly1d(paras_56) # polynomial function
    x_upsampled = np.arange(-5, 5, 0.1)
    y_upsampled_24 = p_24(x_upsampled)
    y_upsampled_56 = p_56(x_upsampled)
    print(100*p_24(0))
    print(100*p_56(0))
    print(100*Zspec_56[offset_56 == 0][0])
    plt.ylim((0, 1))
    plt.plot(x_upsampled, y_upsampled_56, color="blue", label="fitted_56")
    plt.plot(x_upsampled, y_upsampled_24, color="red", label="fitted_24")
    plt.scatter(offset_mid_56, Zspec_mid_56, color="blue", label="Zspec_56")
    plt.scatter(offset_mid_24, Zspec_mid_24, color="red", label="Zspec_24")
    plt.gca().invert_xaxis()  # invert x-axis
    plt.legend()
    plt.title("degree=10")
    plt.show()
    



