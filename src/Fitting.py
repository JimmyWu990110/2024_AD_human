import numpy as np
from scipy.optimize import curve_fit

from BM_model import BM_simp_model_5, BM_simp_model_2
from MT_model import MT_simp_model_4, MT_simp_model_3
from Lorentz_model import Lorentz_5pool_model
from Initialization import Initialization
from funcs import filter_offset, remove_large_offset_Lorentz, get_Z_0ppm


class Fitting:
    def __init__(self, offset=None, Zspec=None, T1=1, T2=50e-3, B1=1.5, lineshape="SL", phantom=False, B0_shift=0):
        """ The offset and Zspec passed in are full, for EMR fitting only part of 
        themwill be used. This will be implemented in each fitting methods."""
        self.offset = offset
        self.Zspec = Zspec
        self.T1 = T1
        self.T2 = T2
        self.B1 = B1
        self.lineshape = lineshape
        self.phantom = phantom
        self.constrained = True
        self.fitted_paras = None
        self.y_estimated = None
        self.success = True

    def is_valid(self):
        if self.T1 <= 1e-6 or self.T2 <= 1e-6 or self.T1 <= self.T2:
            print("Invalid T1, T2!", self.T1, self.T2)
            return False
        return True

    def _MTC_onestep(self, selected_offset):
        # 4 parameters: [R*M0b*T1a, T2b, R, T1a/T2a], T1b is fixed to 1s
        model = MT_simp_model_4(B1=self.B1, lineshape=self.lineshape)
        init = Initialization(self.phantom)
        x0, bounds = init.MTC(self.T1/self.T2)
        offset, Zspec = filter_offset(self.offset, self.Zspec, selected_offset)
        freq = 128*offset # 3T
        fitted_paras = None
        try:
            if self.constrained:
                fitted_paras, _ = curve_fit(model.MT_model, freq, Zspec, p0=x0, 
                                            bounds=bounds, method='trf', maxfev=5000)
            else:
                fitted_paras, _ = curve_fit(model.MT_model, freq, Zspec, p0=x0, 
                                            method='lm', maxfev=5000)
        except:
            self.success = False
            return
        # update parameters (with T1a correction)
        # 8 parameters: [R*M0b*T1a, T1a/T2a, T1a, T1b, T2a, T2b, R, M0b]
        self.fitted_paras = model.cal_paras(fitted_paras, self.T1)
        self.y_estimated = model.generate_Zspec(self.offset, fitted_paras)

    def MTC(self):
        selected_offset = self.offset # all offsets
        self._MTC_onestep(selected_offset)

    def MTC_EMR_v0(self):
        selected_offset = [80, 60, 40, 30, 20, 12, 8] # only large offsets
        self._MTC_onestep(selected_offset)

    def MTC_EMR_v1(self):
        selected_offset = [80, 60, 40, 30, 20, 12, 8, 1.5] # large offsets and 0.5ppm
        self._MTC_onestep(selected_offset)

    def MTC_EMR_v2(self, noisy=False):
        # Fitting with all offsets first, then fix T1a/T2a from fitted results
        # and fit with only large offsets
        # 5 parameters: [R*M0b*T1a, T2b, R, T1a/T2a, noise], T1b is fixed to 1s
        model = MT_simp_model_4(B1=self.B1, lineshape=self.lineshape)
        init = Initialization(self.phantom, noisy)
        x0, bounds = init.MTC(self.T1/self.T2)
        freq = 128*self.offset # 3T
        fitted_paras = None
        # first step, fit with all offsets
        try:
            if self.constrained:
                fitted_paras, _ = curve_fit(model.MT_model, freq, self.Zspec, p0=x0, 
                                            bounds=bounds, method='trf', maxfev=5000)
            else:
                fitted_paras, _ = curve_fit(model.MT_model, freq, self.Zspec, p0=x0, 
                                            method='lm', maxfev=5000)
        except:
            self.success = False
            return
        paras = model.cal_paras(fitted_paras, self.T1)
        # second step, fit with large offsets (EMR); T1a, T2a fixed (from previous step)
        model = MT_simp_model_3(B1=self.B1, lineshape=self.lineshape, T1a=paras[2], 
                                T2a=paras[4], noise = paras[-1])
        # 3 parameters: [R*M0b*T1a, T2b, R], T1b and T1a/T2a are fixed
        x0, bounds = init.MTC_2nd_step(fitted_paras)
        selected_offset = [80, 60, 40, 30, 20, 12, 8]
        offset, Zspec = filter_offset(self.offset, self.Zspec, selected_offset)
        freq = 128*offset
        if self.constrained:
            fitted_paras, _ = curve_fit(model.MT_model, freq, Zspec, p0=x0, 
                                        bounds=bounds, method='trf', maxfev=5000)
        else:
            fitted_paras, _ = curve_fit(model.MT_model, freq, Zspec, p0=x0, 
                                        method='lm', maxfev=5000)
        # update parameters (with T1a correction)
        self.fitted_paras = model.cal_paras(fitted_paras)
        self.y_estimated = model.generate_Zspec(self.offset, fitted_paras)

    def _BMS_onestep(self, selected_offset):
        model = BM_simp_model_5(self.B1, 1.5) # t_sat=1.5
        # 5 parameters: [T1a, T2a, T2b, R, M0b], T1b fixed to 1s
        init = Initialization(self.phantom, noisy=False)
        x0, bounds = init.BMS(self.T1, self.T2)
        offset, Zspec = filter_offset(self.offset, self.Zspec, selected_offset)
        freq = 128*offset # 3T
        fitted_paras = None
        try:
            if self.constrained:
                fitted_paras, _ = curve_fit(model.BM_model, freq, Zspec, p0=x0, 
                                            bounds=bounds, method='trf', maxfev=5000)
            else:
                fitted_paras, _ = curve_fit(model.BM_model, freq, Zspec, p0=self.x0, 
                                            method='lm', maxfev=5000)
        except:
            self.success = False
            return
        # 6 parameters: [T1a, T1b, T2a, T2b, R, M0b]
        self.fitted_paras = model.cal_paras(fitted_paras)
        self.y_estimated = model.generate_Zspec(self.offset, fitted_paras)

    def BMS(self):
        selected_offset = self.offset # all offsets
        self._BMS_onestep(selected_offset)

    def BMS_EMR_v0(self):
        selected_offset = [80, 60, 40, 30, 20, 12, 8] # only large offsets
        self._BMS_onestep(selected_offset)

    def BMS_EMR_v1(self):
        selected_offset = [80, 60, 40, 30, 20, 12, 8, 1.5] # large offsets and 0.5ppm
        self._BMS_onestep(selected_offset)

    def BMS_EMR_v2(self, noisy=False):
        # Fitting with all offsets first, then fix T1a/T2a from fitted results
        # and fit with only large offsets
        model = BM_simp_model_5(self.B1, 1.5)
        init = Initialization(self.phantom, noisy)
        # 5/6 parameters: [T1a, T2a, T2b, R, M0b, (noise)], T1b fixed to 1s
        x0, bounds = init.BMS(self.T1, self.T2)
        freq = 128*self.offset # 3T
        # first step, fit with all offsets
        try:
            if self.constrained and not noisy:
                fitted_paras, _ = curve_fit(model.BM_model, freq, self.Zspec, p0=x0, 
                                            bounds=bounds, method='trf', maxfev=5000)
            elif self.constrained and noisy:
                fitted_paras, _ = curve_fit(model.BM_model_noisy, freq, self.Zspec, p0=x0, 
                                            bounds=bounds, method='trf', maxfev=5000)
            elif not self.constrained and noisy:
                fitted_paras, _ = curve_fit(model.BM_model_noisy, freq, self.Zspec, p0=x0, 
                                            method='lm', maxfev=5000)
            else:
                fitted_paras, _ = curve_fit(model.BM_model, freq, self.Zspec, p0=x0, 
                                            method='lm', maxfev=5000)
        except Exception as e:
            print("Exception:", str(e))
            self.success = False
            return
        # 7 parameters: [T1a, T1b, T2a, T2b, R, M0b, noise]
        paras = model.cal_paras(fitted_paras)
        # second step, fit with large offsets (EMR); T1a, T2a fixed (from previous step)
        model = BM_simp_model_2(B1=self.B1, t_sat=1.5, T1a=paras[0], T2a=paras[2], 
                                T2b=paras[3], noise=paras[6])
        # 2 parameters: [R, M0b], T1a, T1b, T2a, T2b, noise are fixed
        x0, bounds = init.BMS_2nd_step(fitted_paras)
        selected_offset = [80, 60, 40, 30, 20, 12, 8]
        offset, Zspec = filter_offset(self.offset, self.Zspec, selected_offset)
        freq = 128*offset
        if self.constrained:
            fitted_paras, _ = curve_fit(model.BM_model, freq, Zspec, p0=x0, 
                                        bounds=bounds, method='trf', maxfev=5000)
        else:
            fitted_paras, _ = curve_fit(model.BM_model, freq, Zspec, p0=x0, 
                                        method='lm', maxfev=5000)
        # update parameters (with T1a correction)
        self.fitted_paras = model.cal_paras(fitted_paras)
        self.y_estimated = model.generate_Zspec(self.offset, fitted_paras)

    def Lorentz_5pool(self):
        model = Lorentz_5pool_model()
        init = Initialization(self.phantom)
        x0, bounds = init.Lorentz_5pool()
        offset, Zspec = remove_large_offset_Lorentz(self.offset, self.Zspec)
        # offset, Zspec = self.offset, self.Zspec
        fitted_paras = None
        try:
            if self.constrained:
                fitted_paras, _ = curve_fit(model.Lorentz_model, offset, Zspec, p0=x0, 
                                            bounds=bounds, method='trf', maxfev=5000)
            else:
                fitted_paras, _ = curve_fit(model.Lorentz_model, offset, Zspec, p0=x0, 
                                            method='lm', maxfev=5000)
        except:
            self.success = False
            return
        # 15 parameters: [APT_A, APT_C, APT_W, CEST_A, CEST_C, CEST_W, DS_A, DS_C, DS_W, 
        #                 MT_A, MT_C, MT_W, NOE_A, NOE_C, NOE_W]
        self.fitted_paras = model.cal_paras(fitted_paras)
        self.y_estimated = model.generate_Zspec(self.offset, fitted_paras)

    def fit(self, method, show=False):
        assert method in ["MTC", "MTC_EMR_v0", "MTC_EMR_v1", "MTC_EMR_v2", "MTC_EMR_v2_noisy",
                          "BMS", "BMS_EMR_v0", "BMS_EMR_v1", "BMS_EMR_v2", "BMS_EMR_v2_noisy",
                          "Lorentz_5pool"]
        if method == "MTC":
            self.MTC()
        if method == "MTC_EMR_v0":
            self.MTC_EMR_v0()
        if method == "MTC_EMR_v1":
            self.MTC_EMR_v1()
        if method == "MTC_EMR_v2":
            self.MTC_EMR_v2()
        if method == "MTC_EMR_v2_noisy":
            self.MTC_EMR_v2(True)
        if method == "BMS":
            self.BMS()
        if method == "BMS_EMR_v0":
            self.BMS_EMR_v0()
        if method == "BMS_EMR_v1":
            self.BMS_EMR_v1()
        if method == "BMS_EMR_v2":
            self.BMS_EMR_v2()
        if method == "BMS_EMR_v2_noisy":
            self.BMS_EMR_v2(True)
        if method == "Lorentz_5pool":
            self.Lorentz_5pool()
        if show:
            self.print_result(method)
        return

    def print_result(self, method):
        print("******** fitted parameters ********")
        if "MT" in method:
            print("R*M0b*T1a:", self.fitted_paras[0])
            print("T1a/T2a:", self.fitted_paras[1])
            print("T1a:", self.fitted_paras[2], "s")
            print("T1b:", self.fitted_paras[3], "s")
            print("T2a:", 1e3*self.fitted_paras[4], "ms")
            print("T2b:", 1e6*self.fitted_paras[5], "us")
            print("R:", self.fitted_paras[6], "s^-1")
            print("M0b:", self.fitted_paras[7])
            print("noise:", 100*self.fitted_paras[8], "%")
        if "BM" in method:
            print("T1a:", self.fitted_paras[0], "s")
            print("T1b:", self.fitted_paras[1], "s")
            print("T2a:", 1e3*self.fitted_paras[2], "ms")
            print("T2b:", 1e6*self.fitted_paras[3], "us")
            print("R:", self.fitted_paras[4], "s^-1")
            print("M0b:", self.fitted_paras[5])
            print("noise:", 100*self.fitted_paras[6], "%")
        if "Lorentz" in method:
            print("APT magnitude:", 100*self.fitted_paras[0], "%")
            print("APT center:", self.fitted_paras[1], "ppm")
            print("APT width:", self.fitted_paras[2], "ppm")
            print("CEST2ppm magnitude:", 100*self.fitted_paras[3], "%")
            print("CEST2ppm center:", self.fitted_paras[4], "ppm")
            print("CEST2ppm width:", self.fitted_paras[5], "ppm")
            print("DS magnitude:", 100*self.fitted_paras[6], "%")
            print("DS center:", self.fitted_paras[7], "ppm")
            print("DS width:", self.fitted_paras[8], "ppm")
            print("MT magnitude:", 100*self.fitted_paras[9], "%")
            print("MT center:", self.fitted_paras[10], "ppm")
            print("MT width:", self.fitted_paras[11], "ppm")
            print("NOE magnitude:", 100*self.fitted_paras[12], "%")
            print("NOE center:", self.fitted_paras[13], "ppm")
            print("NOE width:", self.fitted_paras[14], "ppm")
        print("************************")

    def generate_Zspec(self, method, offset, paras):
        if "MT" in method:
            model = MT_simp_model_4(B1=self.B1, lineshape=self.lineshape)
            return model.generate_Zspec(offset, paras)
        if "BM" in method:
            model = BM_simp_model_5(B1=self.B1)
            return model.generate_Zspec(offset, paras)
        if "Lorentz" in method:
            model = model = Lorentz_5pool_model()
            return model.generate_Zspec(offset, paras)
