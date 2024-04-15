import os

import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate

class B0Correction:
    
    def __init__(self):
        # 26 offsets
        self.freq = np.array([np.inf, 0, 14, -14, 28, -28, 42,-42, 56, -56, 70, -70, 84, -84, 98, -98, 
                              112, -112, 126, -126, 140, -140, 154, -154, 168, -168])
  
    def cal_B0_shift_onepixel(self, WASSR_Spec):
        # poly fit the WASSR Spec and find the min point of the upsampled offset
        sort_index = np.argsort(self.freq)
        x_sorted = self.freq[sort_index]
        y_sorted = WASSR_Spec[sort_index]
        m0 = y_sorted[-1]
        # normalized by M0, but fitting not include M0
        paras = np.polyfit(x_sorted[:-1], y_sorted[:-1]/m0, deg=12)
        p = np.poly1d(paras)
        x_upsampled = np.arange(-168, 169, 1)
        index = np.argmin(p(x_upsampled))
        # plt.scatter(x_sorted[:-1], y_sorted[:-1]/m0)
        # plt.plot(x_upsampled, p(x_upsampled))
        # plt.gca().invert_xaxis()  # invert x-axis
        # plt.show()
        # print("B0 shift:", x_upsampled[index])
        return x_upsampled[index]/128 # in ppm
                
    def correct_onepixel(self, offset, Zspec, B0_shift):
        # print("B0 shift:", B0_shift)
        real_offset = offset - B0_shift
        # print(real_offset)
        
        # plt.ylim((0, 1))
        # plt.scatter(offset[8:], Zspec[8:], label="Zspec", color="blue", marker="x")
        # plt.scatter(real_offset[8:], Zspec[8:], label="real Zspec", color="red", marker="x")
        # plt.gca().invert_xaxis()  # invert x-axis
        # plt.legend()
        # plt.show()
        
        f = interpolate.interp1d(x=real_offset, y=Zspec, kind="linear", fill_value="extrapolate")
        Zspec_corrected = f(offset)
        
        # plt.ylim((0, 1))
        # plt.scatter(offset[8:], Zspec[8:], label="Zspec", color="blue", marker="x")
        # plt.scatter(offset[8:], Zspec_corrected[8:], label="Zspec_corrected", color="red", marker="x")
        # plt.gca().invert_xaxis()  # invert x-axis
        # plt.legend()
        # plt.show()
        
        # plt.scatter(offset, Zspec, label="Zspec", color="blue", marker="x")
        # plt.scatter(offset, Zspec_corrected, label="Zspec_corrected", color="red", marker="x")
        # plt.gca().invert_xaxis()  # invert x-axis
        # plt.legend()
        # plt.show()
        
        return Zspec_corrected
        
        
        
        
        
        
        
        
        