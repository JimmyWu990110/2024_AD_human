import math
import numpy as np
from scipy.optimize import Bounds, curve_fit, minimize, least_squares
import scipy.integrate as integrate   

def _gaussian(x, T2):
    term = 2*math.pi*x*T2
    exp_term = math.exp(-0.5*term*term)
    return (T2*exp_term) / math.sqrt(2*math.pi)
                
def _lorentz(x, T2):
    term = 2*math.pi*x*T2
    return T2 / (math.pi*(1+term*term))

def _super_lorentz(t, x, T2):
    # t is the independent variable for integration, (x, T2) are parameters
    cos_term = abs(3*math.cos(t)*math.cos(t) - 1)
    T2_term = 2*math.pi*x*T2
    power = (-2)*(T2_term/cos_term)*(T2_term/cos_term)
    return math.sin(t)*math.sqrt(2/math.pi)*(T2/cos_term)*math.exp(power)

def cal_Rrf(freq, B1, T2, lineshape):
    result = []
    for x in freq: # x in hz, 2*pi*x in rad/s!
        if lineshape == 'G':
            g = _gaussian(x, T2)
            result.append(g)
        elif lineshape == 'L':
            g = _lorentz(x, T2)
            result.append(g)
        elif lineshape == 'SL':
            g, err = integrate.quad(_super_lorentz, 0, math.pi/2, args=(x, T2))
            result.append(g)  
    w1 = 267.522 * B1 # in rad/s!
    return math.pi*w1*w1*np.array(result) # in s^-1
    

class MT_simp_model_4:
    def __init__(self, B1=1.5, lineshape="SL"):
        self.B1 = B1
        self.T1b = 1
        self.lineshape = lineshape
    
    def MT_model(self, freq, RMT, T2b, R, ratio, noise):
        # 5 parameters: [R*M0b*T1a, T2b, R, T1a/T2a, noise], T1b is fixed
        R1b = 1/self.T1b
        w1 = 42.58 * self.B1 # in hz!
        Rrfb_line = cal_Rrf(freq, self.B1, T2b, self.lineshape)
        Zspec = []
        for i in range(freq.shape[0]):
            Rrfb = Rrfb_line[i]
            RrfaT1a = (w1/freq[i])*(w1/freq[i])*ratio # approximate
            numerator = R1b*RMT + Rrfb + R1b + R
            denominator = (R1b+Rrfb)*RMT + (1+RrfaT1a)*(Rrfb+R1b+R)
            Zspec.append(numerator/denominator + noise)
        return np.array(Zspec)
   
    def generate_Zspec(self, offset, paras):
        freq = 128*offset # 3T
        return self.MT_model(freq, *paras)
    
    def cal_paras(self, paras, T1_obs):
        R1_obs = 1/T1_obs
        R1b = 1/self.T1b
        numerator = paras[0]*(R1b - R1_obs)
        denominator = R1b - R1_obs + paras[2]
        T1a = T1_obs * (1 + numerator/denominator)
        T2a = T1a / paras[3]
        M0b = paras[0] / (paras[2]*T1a)
        # [R*M0b*T1a, T1a/T2a, T1a, T1b, T2a, T2b, R, M0b, noise]
        return np.array([paras[0], T1a/T2a, T1a, self.T1b, T2a, paras[1], 
                         paras[2], M0b, paras[-1]])

    
class MT_simp_model_3:
    def __init__(self, B1=1.5, lineshape="SL", T1a=1, T2a=50e-3, noise=0):
        self.B1 = B1
        self.T1b = 1
        self.lineshape = lineshape
        self.T1a = T1a # initial value, will be updated after fitting
        self.T2a = T2a
        self.noise = noise
    
    def MT_model(self, freq, RMT, T2b, R):
        # 3 parameters: [R*M0b*T1a, T2b, R], T1b and T1a/T2a are fixed
        R1b = 1/self.T1b
        w1 = 42.58 * self.B1 # in hz!
        Rrfb_line = cal_Rrf(freq, self.B1, T2b, self.lineshape)
        Zspec = []
        for i in range(freq.shape[0]):
            Rrfb = Rrfb_line[i]
            RrfaT1a = (w1/freq[i])*(w1/freq[i])*(self.T1a/self.T2a) # approximate
            numerator = R1b*RMT + Rrfb + R1b + R
            denominator = (R1b+Rrfb)*RMT + (1+RrfaT1a)*(Rrfb+R1b+R)
            Zspec.append(numerator/denominator + self.noise)
        return np.array(Zspec)
    
    def generate_Zspec(self, offset, paras):
        freq = 128*offset # 3T
        return self.MT_model(freq, *paras)
    
    def cal_paras(self, paras):
        numerator = paras[0] * (1/self.T1b-1/self.T1a)
        denominator = 1/self.T1b - 1/self.T1a + paras[2]
        T1a = self.T1a * (1+numerator/denominator)
        T2a = self.T2a
        M0b = paras[0] / (paras[2]*T1a)
        # [R*M0b*T1a, T1a/T2a, T1a, T1b, T2a, T2b, R, M0b, noise]
        return np.array([paras[0], T1a/T2a, T1a, self.T1b, T2a, paras[1], 
                         paras[2], M0b, self.noise])