import numpy as np
from scipy import interpolate


class B0_correction:
    def __init__(self, offset, Zspec, WASSR_Spec):
        # 26 WASSR frequencies, in hz
        self.freq = np.array([np.inf, 0, 14, -14, 28, -28, 42,-42, 56, -56, 70, -70, 84, -84, 98, -98, 
                              112, -112, 126, -126, 140, -140, 154, -154, 168, -168])
        self.offset = offset
        self.Zspec = Zspec
        self.WASSR_Spec = WASSR_Spec
        self.B0_shift = 0 # in ppm
        self.Zspec_corrected = None
  
    def get_B0_shift(self):
        # poly fit the WASSR Spec and find the min point of the upsampled offset
        sort_index = np.argsort(self.freq)
        x_sorted = self.freq[sort_index]
        y_sorted = self.WASSR_Spec[sort_index]
        # normalized by M0, but fitting not include M0
        m0 = y_sorted[-1]
        paras = np.polyfit(x_sorted[:-1], y_sorted[:-1]/m0, deg=12)
        p = np.poly1d(paras) # polynomial function
        x_upsampled = np.arange(-168, 169, 1)
        index = np.argmin(p(x_upsampled))
        self.B0_shift = x_upsampled[index]/128 # B0 shift, in ppm
        # visualization = Visualization()
        # visualization.plot_WASSR_fitting(x_upsampled, p(x_upsampled))
    
    def linear_interp(self):
        real_offset = self.offset - self.B0_shift
        f = interpolate.interp1d(x=real_offset, y=self.Zspec, kind="linear", fill_value="extrapolate")
        self.Zspec_corrected = f(self.offset)
        # visualization = Visualization()
        # visualization.check_B0_corr(self.offset[7:], self.Zspec[7:], self.Zspec_corrected[7:])

    def linear_interp_twoside(self):
        real_offset = self.offset - self.B0_shift
        positive_offset = real_offset[real_offset > 0]
        negative_offset = real_offset[real_offset < 0]
        fp = interpolate.interp1d(x=real_offset[real_offset > 0], y=self.Zspec[real_offset > 0], 
                                  kind="linear", fill_value="extrapolate")
        fn = interpolate.interp1d(x=real_offset[real_offset < 0], y=self.Zspec[real_offset < 0], 
                                  kind="linear", fill_value="extrapolate")
        self.Zspec_corrected = np.concatenate((fp(self.offset[real_offset > 0]),
                                               fn(self.offset[real_offset < 0])))     
        # visualization = Visualization()
        # visualization.check_B0_corr(self.offset[7:], self.Zspec[7:], self.Zspec_corrected[7:])

    def linear_interp_zeropadded(self):
        offset = np.concatenate((self.offset[self.offset > 0], [self.B0_shift],
                                 self.offset[self.offset < 0]))
        Zspec = np.concatenate((self.Zspec[self.offset > 0], [0], 
                                self.Zspec[self.offset < 0]))
        real_offset = offset - self.B0_shift
        f = interpolate.interp1d(x=real_offset, y=Zspec, kind="linear", fill_value="extrapolate")
        self.Zspec_corrected = f(self.offset)
        # visualization = Visualization()
        # visualization.check_B0_corr(self.offset[7:], self.Zspec[7:], self.Zspec_corrected[7:])

    def correct(self):
        self.get_B0_shift()
        # print("B0 shift:", 128*self.B0_shift, "hz", self.B0_shift, "ppm")
        if np.sum(self.offset == 0) > 0:
            self.linear_interp()
        else:
            self.linear_interp_twoside()
        return self.Zspec_corrected
        
        
        
        
        
        
        
        
        