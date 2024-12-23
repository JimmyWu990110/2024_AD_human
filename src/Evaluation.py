import numpy as np
from sklearn.metrics import r2_score

from funcs import remove_large_offset


class Evaluation:
    def __init__(self):
        pass

    def cal_R2(self, Zspec_1, Zspec_2):
        return r2_score(Zspec_1, Zspec_2)
    
    def cal_NRMSE(self, real, estimated):
        return np.linalg.norm(estimated-real)/np.linalg.norm(real)
    
    def evaluate(self, method, offset, real, estimated, show=False):
        NRMSE = self.cal_NRMSE(real, estimated)
        _, real_mid = remove_large_offset(offset, real)
        _, estimated_mid = remove_large_offset(offset, estimated)
        NRMSE_mid = self.cal_NRMSE(real_mid, estimated_mid)
        if "Lorentz" in method:
            if show:
                print("******** metrics ********")
                print("NRMSE (%):", 100*NRMSE)
                print("NRMSE_mid (%):", 100*NRMSE_mid)
                print("************************")
            return 100*np.array([NRMSE, NRMSE_mid])
        diff = estimated - real
        APT_pow = diff[np.where(offset==3.5)]
        NOE_pow = diff[np.where(offset==-3.5)]
        APTw = APT_pow[0] - NOE_pow[0]
        MTR_20ppm = 1 - real[np.where(offset==20)]
        MTR_60ppm = 1 - real[np.where(offset==60)]
        if show:
            print("******** metrics ********")
            print("NRMSE (%):", 100*NRMSE)
            print("APT# (%):", 100*APT_pow[0])
            print("NOE# (%):", 100*NOE_pow[0])
            print("APTw (%):", 100*APTw)
            print("MTR 20ppm (%):", 100*MTR_20ppm[0])
            print("MTR 60ppm (%):", 100*MTR_60ppm[0])
            print("************************")
        return 100*np.array([NRMSE, APT_pow[0], NOE_pow[0], APTw, MTR_20ppm[0], MTR_60ppm[0]])

    def evaluate_16(self, Zspec_1, Zspec_2, show=False):
        R2, NRMSE = self.evaluate(Zspec_1, Zspec_2, show)
        R2_out, NRMSE_out = self.evaluate(Zspec_1[0:7], Zspec_2[0:7], show, "out")
        R2_mid, NRMSE_mid = self.evaluate(Zspec_1[7:], Zspec_2[7:], show, "mid")
        return NRMSE, NRMSE_out, NRMSE_mid
    
    def evaluate_24(self, Zspec_1, Zspec_2, show=False):
        R2, NRMSE = self.evaluate(Zspec_1, Zspec_2, show)
        R2_out, NRMSE_out = self.evaluate(Zspec_1[0:7], Zspec_2[0:7], show, "out")
        R2_mid, NRMSE_mid = self.evaluate(Zspec_1[7:], Zspec_2[7:], show, "mid")
        return NRMSE, NRMSE_out, NRMSE_mid
               
    def get_metrics(self, offset, real, estimated):
        diff = estimated - real
        APT_pow = diff[np.where(offset==3.5)]
        NOE_pow = diff[np.where(offset==-3.5)]
        APTw = APT_pow[0]-NOE_pow[0]
        MTR_20ppm = 1 - real[np.where(offset==20)]
        MTR_60ppm = 1 - real[np.where(offset==60)]
        # print("APT# (%):", 100*APT_pow[0])
        # print("NOE# (%):", 100*NOE_pow[0])
        # print("APTw (%):", 100*APTw)
        # print("MTR 20ppm (%):", 100*MTR_20ppm[0])
        return 100*APT_pow[0], 100*NOE_pow[0], 100*APTw, 100*MTR_20ppm[0], 100*MTR_60ppm[0]

