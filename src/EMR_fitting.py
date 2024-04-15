import math

import numpy as np
from scipy.optimize import curve_fit
import scipy.integrate as integrate   

    
class EMR_fitting:
    def __init__(self, T1w_obs=1, T2w_obs=0.04, B1=1.5):
        self.B1 = B1
        self.T1w_obs = T1w_obs
        self.T2w_obs = T2w_obs
        self.T1m = 1
        # [R, R*M0m*T1w, T1w/T2w, T2m]
        # self.x0 = [2, 0.8, self.T1w_obs/self.T2w_obs, 5e-6]
        # self.lb = [0.5, 0.2, 0.4*(self.T1w_obs/self.T2w_obs), 2e-6]
        # self.ub = [20, 10, 2.5*(self.T1w_obs/self.T2w_obs), 15e-6]
        self.x0 = [15, 2, self.T1w_obs/self.T2w_obs, 5e-6]
        self.lb = [1, 0.2, 0.4*(self.T1w_obs/self.T2w_obs), 1e-6]
        self.ub = [50, 10, 2.5*(self.T1w_obs/self.T2w_obs), 20e-6]
        self.lineshape = "SL"
        self.R = 20 # for fixed R
        self.fitted_paras = None
   
    def set_x0(self, x0):
        self.x0 = x0
        
    def set_lb(self, lb):
        self.lb = lb
        
    def set_ub(self, ub):
        self.ub = ub
   
    def set_lineshape(self, lineshape):
        if lineshape not in ["G", "L", "SL"]:
            raise Exception("Invalid lineshape! Please use G, L or SL")
        self.lineshape = lineshape
    
    def _func_gaussian(self, x, T2m):
        term = 2 * math.pi * x * T2m
        exp_term = math.exp(-term*term*0.5)
        return (T2m / math.sqrt(2*math.pi)) * exp_term
                
    def _func_lorentz(self, x, T2m):
        term = 2 * math.pi * x * T2m
        return (T2m/math.pi) * (1 / (1 + term*term))

    def _func_super_lorentz(self, t, x, T2m):
        # t is the independent variable for integration, (x, T2m) are parameters
        cos_denominator = abs(3*math.cos(t)*math.cos(t) - 1)
        term = math.sin(t) * math.sqrt(2/math.pi) * (T2m/cos_denominator)
        T2m_numerator = 2 * math.pi * x * T2m 
        power = (-2) * (T2m_numerator/cos_denominator) * (T2m_numerator/cos_denominator)
        return term * math.exp(power) 
    
    def cal_Rrf(self, freq, T2, lineshape):
        result = []
        for x in freq:
            if lineshape == 'G':
                val = self._func_gaussian(x, T2)
                result.append(val)
            elif lineshape == 'L':
                val = self._func_lorentz(x, T2)
                result.append(val)
            elif lineshape == 'SL':
                val, err = integrate.quad(self._func_super_lorentz, 0, math.pi/2, args=(x, T2))
                result.append(val)  
        w1 = 267.522 * self.B1
        return np.array(result) * w1 * w1 * math.pi
    
    def MT_model(self, freq, A, B, C, D):
        # [R, R*M0m*T1w, T1w/T2w, T2m]
        """
        given frequencies and these 4 paras, return Zspec(Mz/M0)
        """
        w1 = 267.522 * self.B1
        Rrfm_line = self.cal_Rrf(freq, D, self.lineshape)
        Zspec = []
        for i in range(freq.shape[0]):
            x = freq[i]
            Rrfm = Rrfm_line[i]
            if x == 0: # deal with singular point
                Zspec.append(0)
                continue    
            # following the formula
            tmp = w1 / (2*math.pi*x)
            numerator = B + (Rrfm + A + 1)
            denominator_1 = B * (Rrfm+1)
            denominator_2 = (1 + tmp*tmp*C) * (Rrfm + A + 1)
            Zspec.append(numerator / (denominator_1+denominator_2))
        return np.array(Zspec)
        
    def fit(self, freq, Zspec, constrained):
        # print("T1 obs, T2 obs, ratio:", self.T1w_obs, self.T2w_obs, self.T1w_obs/self.T2w_obs)     
        # print("offsets used for fitting:", freq/128)   
        if math.isnan(self.T1w_obs/self.T2w_obs) or self.T1w_obs/self.T2w_obs == 0 or math.isinf(self.T1w_obs/self.T2w_obs):
            print("Invalid T1/T2!")
        else: 
            if constrained:
                popt, pcov = curve_fit(self.MT_model, freq, Zspec, p0=self.x0, 
                                       bounds=(self.lb, self.ub), method='trf', maxfev=5000)
            else:
                popt, pcov = curve_fit(self.MT_model, freq, Zspec, p0=self.x0, 
                                       method='lm', maxfev=5000)
            self.fitted_paras = popt
            y_estimated = self.MT_model(freq, *popt)                  
            return popt, y_estimated
      
    def cal_paras(self):
        if self.fitted_paras is None:
            raise Exception("No fitted parameters! Fit first.")            
        R = self.fitted_paras[0]
        R_M0m_T1w = self.fitted_paras[1]
        T1w_T2w_ratio = self.fitted_paras[2]
        # following the formula
        numerator = R_M0m_T1w * (1/self.T1m - 1/self.T1w_obs)
        denominator = (1/self.T1m - 1/self.T1w_obs) + R
        T1w = self.T1w_obs * (1 + numerator/denominator)
        M0m = R_M0m_T1w / (R*T1w)
        T2w = T1w / T1w_T2w_ratio
        para_dict = {"R":R, "R*M0m*T1w":R_M0m_T1w, "T1w/T2w":T1w_T2w_ratio, 
                     "T2m":self.fitted_paras[3],
                     "M0m":M0m, "T1w":T1w, "T2w":T2w}
        return para_dict
   
    def generate_Zpsec(self, freq, paras):
        return self.MT_model(freq, *paras)


class EMR_fitting_A: # fix T1w/T2w
    def __init__(self, T1w, T2w, B1=1.5):
        self.B1 = B1
        self.T1w_obs = T1w
        self.T2w_obs = T2w
        self.T1m = 1
        self.x0 = [2, 0.8, 5e-6]
        self.lb = [0.5, 0.2, 2e-6] # [R, R*M0m*T1w, T2m]
        self.ub = [20, 10, 15e-6]
        self.lineshape = "SL"
        self.fitted_paras = None
   
    def set_x0(self, x0):
        self.x0 = x0
        
    def set_lb(self, lb):
        self.lb = lb
        
    def set_ub(self, ub):
        self.ub = ub
   
    def set_lineshape(self, lineshape):
        if lineshape not in ["G", "L", "SL"]:
            raise Exception("Invalid lineshape! Please use G, L or SL")
        self.lineshape = lineshape
    
    def _func_gaussian(self, x, T2m):
        term = 2 * math.pi * x * T2m
        exp_term = math.exp(-term*term*0.5)
        return (T2m / math.sqrt(2*math.pi)) * exp_term
                
    def _func_lorentz(self, x, T2m):
        term = 2 * math.pi * x * T2m
        return (T2m/math.pi) * (1 / (1 + term*term))

    def _func_super_lorentz(self, t, x, T2m):
        # t is the independent variable for integration, (x, T2m) are parameters
        cos_denominator = abs(3*math.cos(t)*math.cos(t) - 1)
        term = math.sin(t) * math.sqrt(2/math.pi) * (T2m/cos_denominator)
        T2m_numerator = 2 * math.pi * x * T2m 
        power = (-2) * (T2m_numerator/cos_denominator) * (T2m_numerator/cos_denominator)
        return term * math.exp(power) 
    
    def cal_Rrf(self, freq, T2, lineshape):
        result = []
        for x in freq:
            if lineshape == 'G':
                val = self._func_gaussian(x, T2)
                result.append(val)
            elif lineshape == 'L':
                val = self._func_lorentz(x, T2)
                result.append(val)
            elif lineshape == 'SL':
                val, err = integrate.quad(self._func_super_lorentz, 0, math.pi/2, args=(x, T2))
                result.append(val)  
        w1 = 267.522 * self.B1
        return np.array(result) * w1 * w1 * math.pi
    
    def MT_model(self, freq, A, B, D):
        # [R, R*M0m*T1w, T2m]
        """
        given frequencies and these 4 paras, return Zspec(Mz/M0)
        """
        w1 = 267.522 * self.B1
        Rrfm_line = self.cal_Rrf(freq, D, self.lineshape)
        Zspec = []
        for i in range(freq.shape[0]):
            x = freq[i]
            Rrfm = Rrfm_line[i]
            if x == 0: # deal with singular point
                Zspec.append(0)
                continue    
            # following the formula
            tmp = w1 / (2*math.pi*x)
            numerator = B + (Rrfm + A + 1)
            denominator_1 = B * (Rrfm+1)
            C = self.T1w_obs/self.T2w_obs
            denominator_2 = (1 + tmp*tmp*C) * (Rrfm + A + 1)
            Zspec.append(numerator / (denominator_1+denominator_2))
        return np.array(Zspec)
        
    def fit(self, freq, Zspec, constrained):
        # print("T1 obs, T2 obs, ratio:", self.T1w_obs, self.T2w_obs, self.T1w_obs/self.T2w_obs)     
        # print("offsets used for fitting:", freq/128)   
        if math.isnan(self.T1w_obs/self.T2w_obs) or self.T1w_obs/self.T2w_obs == 0 or math.isinf(self.T1w_obs/self.T2w_obs):
            raise Exception("Invalid T1/T2!")
        else: 
            if constrained:
                popt, pcov = curve_fit(self.MT_model, freq, Zspec, p0=self.x0, 
                                       bounds=(self.lb, self.ub), method='trf', maxfev=5000)
            else:
                popt, pcov = curve_fit(self.MT_model, freq, Zspec, p0=self.x0, 
                                       method='lm', maxfev=5000)
            self.fitted_paras = popt
            y_estimated = self.MT_model(freq, *popt)                  
            return popt, y_estimated
      
    def cal_paras(self):
        if self.fitted_paras is None:
            raise Exception("No fitted parameters! Fit first.")            
        R = self.fitted_paras[0]
        R_M0m_T1w = self.fitted_paras[1]
        T1w_T2w_ratio = self.T1w_obs/self.T2w_obs
        # following the formula
        numerator = R_M0m_T1w * (1/self.T1m - 1/self.T1w_obs)
        denominator = (1/self.T1m - 1/self.T1w_obs) + R
        T1w = self.T1w_obs * (1 + numerator/denominator)
        M0m = R_M0m_T1w / (R*T1w)
        T2w = T1w / T1w_T2w_ratio
        para_dict = {"R":R, "R*M0m*T1w":R_M0m_T1w, "T1w/T2w":T1w_T2w_ratio, 
                     "T2m":self.fitted_paras[2],
                     "M0m":M0m, "T1w":T1w, "T2w":T2w}
        return para_dict
   
    def generate_Zpsec(self, freq, paras):
        return self.MT_model(freq, *paras)
        
