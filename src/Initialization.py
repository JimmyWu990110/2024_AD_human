from scipy.optimize import Bounds


class Initialization:
    def __init__(self, phantom=False, noisy=False):
        self.phantom = phantom
        self.noisy = noisy

    def MTC(self, ratio):
        # 4 [R*M0b*T1a, T2b, R, T1a/T2a, noise]
        if self.phantom:
            x0 = [2, 10e-6, 5, ratio, 0]
            lb = [0.2, 2e-6, 1, 0.4*ratio, -1e-10]
            ub = [20, 50e-6, 60, 2.5*ratio, 1e-10]
            if self.noisy:
                lb[-1] = -0.1 # -10%
                ub[-1] = 0.1
            bounds = Bounds(lb, ub)
            return x0, bounds
        else:
            x0 = [2, 10e-6, 5, ratio, 0]
            lb = [0.2, 2e-6, 1, 0.4*ratio, -1e-10]
            ub = [20, 50e-6, 60, 2.5*ratio, 1e-10]
            if self.noisy:
                lb[-1] = -0.1 # -10%
                ub[-1] = 0.1
            bounds = Bounds(lb, ub)
            return x0, bounds

    def MTC_2nd_step(self, fitted_paras):
        # 3 [R*M0b*T1a, T2b, R]
        x0 = fitted_paras[0:3]
        lb = [0.5*x0[0], 0.5*x0[1], 0.5*x0[2]]
        ub = [2*x0[0], 2*x0[1], 2*x0[2]]
        bounds = Bounds(lb, ub)
        return x0, bounds
    
    def BMS(self, T1, T2):
        # noisy: 6 paras, [T1a, T2a, T2b, R, M0b, noise]
        # else: 5 paras, [T1a, T2a, T2b, R, M0b]
        if self.phantom:
            if self.noisy:
                x0 = [T1, T2, 5e-6, 3, 0.25, 0.02]
                lb = [0.8*T1, 0.4*T2, 2e-6, 1, 0, 0]
                ub = [1.25*T1, 2.5*T2, 20e-6, 20, 0.5, 0.04]
            else:
                x0 = [T1, T2, 5e-6, 3, 0.25]
                lb = [0.8*T1, 0.4*T2, 2e-6, 1, 0]
                ub = [1.25*T1, 2.5*T2, 20e-6, 20, 0.5]
            bounds = Bounds(lb, ub)
            return x0, bounds
        else:
            if self.noisy:
                x0 = [T1, T2, 10e-6, 10, 0.1, 0.02]
                lb = [0.8*T1, 0.4*T2, 4e-6, 1, 0, 0]
                ub = [1.25*T1, 2.5*T2, 50e-6, 50, 0.5, 0.04]
            else:
                x0 = [T1, T2, 10e-6, 10, 0.1]
                lb = [0.8*T1, 0.4*T2, 4e-6, 1, 0]
                ub = [1.25*T1, 2.5*T2, 50e-6, 50, 0.5]
            bounds = Bounds(lb, ub)
            return x0, bounds
        
    def BMS_2nd_step(self, fitted_paras):
        # 2 [R, M0b]
        x0 = [fitted_paras[3], fitted_paras[4]]
        lb = [0.25*x0[0], 0.5*x0[1]]
        ub = [4*x0[0], 2*x0[1]]
        bounds = Bounds(lb, ub)
        return x0, bounds

    def Lorentz_5pool(self):
        # 14 [APT_A, APT_C, APT_W, CEST_A, CEST_C, CEST_W, DS_A, DS_W, 
        #     MT_A, MT_C, MT_W, NOE_A, NOE_C, NOE_W]
        x0 = [0.01, 3.5, 4, 0.01, 2, 4, 0.4, 4, 0.4, -2, 40, 0.01, -3.6, 5]
        lb = [0, 3.4, 1, 0, 1.8, 1, 0.1, 1, 0.1, -3, 10, 0, -5, 1]
        ub = [0.1, 3.6, 10, 0.1, 2.2, 10, 0.9, 10, 0.9, 0, 150, 0.1, -3.4, 20]
        bounds = Bounds(lb, ub)
        return x0, bounds