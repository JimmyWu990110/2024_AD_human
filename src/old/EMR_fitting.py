
import numpy as np

from BM_model import BM_num_model, BM_ana_model
from MT_model import MT_model_6, MT_model_5, MT_simp_model_4
from Visualization import Visualization


def select_offset(offset, Zspec, flag=0):
    if flag == 1:
        offset = offset[0:7]
        Zspec = Zspec[0:7]
    elif flag == 2:
        offset = offset[[0, 1, 2, 3, 4, 5, 6, 12]] # 1.5 ppm
        Zspec = Zspec[[0, 1, 2, 3, 4, 5, 6, 12]]
    return [128*offset, Zspec]


class EMR_fitting:
    def __init__(self, offset=None, Zspec=None, T1=1, T2=0.04, B1=1.5, lineshape="SL"):
        self.offset = offset
        self.Zspec = Zspec
        self.T1 = T1
        self.T2 = T2
        self.B1 = B1
        self.lineshape = lineshape
        self.y_estimated = None
        self.results = None

    def get_metrics(self):
        diff = self.y_estimated - self.Zspec
        RMSE = np.linalg.norm(diff) / np.sqrt(diff.shape[0])
        NRMSE = np.linalg.norm(diff) / np.linalg.norm(self.Zspec)
        APT_pow = diff[np.where(self.offset==3.5)]
        NOE_pow = diff[np.where(self.offset==-3.5)]
        APTw = APT_pow[0]-NOE_pow[0]
        MTR_20ppm = 1 - self.Zspec[np.where(self.offset==20)]
        MTR_60ppm = 1 - self.Zspec[np.where(self.offset==60)]
        return [100*APT_pow[0], 100*NOE_pow[0], 100*APTw, 100*MTR_20ppm[0], 
                100*MTR_60ppm[0], 100*NRMSE]
    
    def MT_6(self):
        model = MT_model_6(T1a_obs=self.T1, T2a_obs=self.T2, B1=1.5, 
                           lineshape=self.lineshape)
        fitted_paras = model.fit(128*self.offset, self.Zspec, constrained=True)
        self.y_estimated = model.generate_Zspec(128*self.offset, fitted_paras)
        print("fitted parameters:", fitted_paras)
        self.plot()

    def EMR(self):
        model = MT_simp_model_4(T1a_obs=self.T1, T2a_obs=self.T2, B1=1.5, 
                                lineshape=self.lineshape)
        # For 3T, 1ppm = about 128hz
        fitted_paras = model.fit(128*self.offset, self.Zspec, constrained=True)
        # fitted_paras = model.Rfit(128*self.offset, self.Zspec, 
        #                           constrained=True, reg_weight=0.0075)
        self.y_estimated = model.generate_Zspec(128*self.offset, fitted_paras)
        # metrics = self.get_metrics()        
        # self.results = np.array([*fitted_paras, *metrics])
        print("fitted parameters:", fitted_paras)
        self.plot()
    
    def plot(self):
        visualization = Visualization()
        visualization.plot_2_Zspec(self.offset, self.Zspec, self.y_estimated, 
                                   labels=["real", "fitted"])
        visualization.plot_2_Zspec(self.offset[7:], self.Zspec[7:], self.y_estimated[7:], 
                                   labels=["real", "fitted"])
        visualization.plot_Zspec_diff(self.offset[7:], 
                                      self.y_estimated[7:]-self.Zspec[7:])

    def print_results(self):
        names = ["R*M0b*T1a", "T2b", "R", "T1a/T2a", "APT# (%)", "NOE# (%)", 
                 "APTw (%)", "MTR(20ppm) (%)", "MTR(60ppm) (%)", "NRMSE (%)"]
        print("******** fitting results: ********")
        for i in range(self.results.shape[0]):
            print(names[i], ":", self.results[i])