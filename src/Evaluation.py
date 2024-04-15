
import numpy as np
from sklearn.metrics import r2_score

class Evaluation:
    
    def __init__(self):
        pass

    def cal_R2(self, Zspec_1, Zspec_2):
        return r2_score(Zspec_1, Zspec_2)
    
    def cal_NRMSE(self, Zspec_1, Zspec_2):
        # Zspec_1 is the reference (ground truth)
        diff = Zspec_1 - Zspec_2
        return np.linalg.norm(diff)/np.linalg.norm(Zspec_1)
    
    def evaluate(self, Zspec_1, Zspec_2, show=False, title=""):
        R2 = self.cal_R2(Zspec_1, Zspec_2)
        NRMSE = self.cal_NRMSE(Zspec_1, Zspec_2)
        if show:
            # print("R2 "+title+":", R2)
            print("NRMSE (%) "+title+":", 100*NRMSE)
        return R2, 100*NRMSE

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
        # print("APT# (%):", 100*APT_pow[0])
        # print("NOE# (%):", 100*NOE_pow[0])
        # print("APTw (%):", 100*APTw)
        # print("MTR 20ppm (%):", 100*MTR_20ppm[0])
        return 100*APT_pow[0], 100*NOE_pow[0], 100*APTw, 100*MTR_20ppm[0]

