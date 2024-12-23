import math
import numpy as np
from scipy.integrate import odeint
from scipy.linalg import expm
import scipy.integrate as integrate
from scipy.stats import norm  

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

def cal_g(x, T2, lineshape):
    g = None
    if lineshape == "G":
        g = _gaussian(x, T2)
    elif lineshape == "L":
        g = _lorentz(x, T2)
    elif lineshape == "SL":
        g, err = integrate.quad(_super_lorentz, 0, math.pi/2, args=(x, T2))
    return g


class BM_simp_model_5:
    def __init__(self, B1, t_sat=1.5):
        self.B1 = B1
        self.t_sat = t_sat

    def ANA(self, dw, T1a, T2a, T2b, R, M0b):
        # NOTE: w1 = 2*pi*42.58*offset, it's in rad/s rather than hz!
        w1 = 2*math.pi*42.58*self.B1
        dwa = 2*math.pi*dw
        dwb = 2*math.pi*dw
        R1a = 1/T1a
        R1b = 1 # fix to 1s
        R2a = 1/T2a
        M0a = 1
        kab = R*M0b 
        kba = R*M0a
        # The lineshape function takes dw in hz!
        Rrfb = math.pi*w1*w1*cal_g(dwb/(2*math.pi), T2b, lineshape="SL")
        y0 = np.array([0, 0, 1, M0b, 1]) #[5,]
        A = np.array([[-R2a, -dwa, 0, 0, 0],
                      [dwa, -R2a, -w1, 0, 0],
                      [0, w1, -R1a-kab, kba, R1a*M0a],
                      [0, 0, kab, -R1b-kba-Rrfb, R1b*M0b],
                      [0, 0, 0, 0, 0]])
        return np.dot(expm(A*self.t_sat), y0) # [5,]
    
    def BM_model(self, freq, T1a, T2a, T2b, R, M0b):
        # 5 parameters: [T1a, T2a, T2b, R, M0b]
        Zspec = []
        for i in range(freq.shape[0]):
            Zspec.append(self.ANA(freq[i], T1a, T2a, T2b, R, M0b)[2])
        return np.array(Zspec)
    
    def BM_model_noisy(self, freq, T1a, T2a, T2b, R, M0b, noise):
        # 6 parameters: [T1a, T2a, T2b, R, M0b, noise]
        Zspec = []
        for i in range(freq.shape[0]):
            tmp = self.ANA(freq[i], T1a, T2a, T2b, R, M0b)[2]
            Zspec.append((1-noise)*tmp + noise)
        return np.array(Zspec)

    def generate_Zspec(self, offset, paras):
        freq = 128*offset # 3T
        if paras.shape[0] == 6: # noisy
            return self.BM_model_noisy(freq, *paras)
        else:
            return self.BM_model(freq, *paras)
    
    def cal_paras(self, paras):
        # always return [T1a, T1b, T2a, T2b, R, M0b, noise] (len=7)
        # if not noisy, set noise=0
        if paras.shape[0] == 6: # noisy
            return np.array([paras[0], 1, paras[1], paras[2], paras[3], paras[4], paras[5]])
        else:
            return np.array([paras[0], 1, paras[1], paras[2], paras[3], paras[4], 0])

class BM_simp_model_2:
    def __init__(self, B1, t_sat=1.5, T1a=1, T2a=50e-3, T2b=10e-6, noise=0):
        self.B1 = B1
        self.t_sat = t_sat
        self.T1a = T1a
        self.T2a = T2a
        self.T2b = T2b
        self.noise = noise

    def ANA(self, dw, R, M0b):
        # NOTE: w1 = 2*pi*42.58*offset, it's in rad/s rather than hz!
        w1 = 2*math.pi*42.58*self.B1
        dwa = 2*math.pi*dw
        dwb = 2*math.pi*dw
        R1a = 1/self.T1a
        R1b = 1
        R2a = 1/self.T2a
        M0a = 1
        kab = R*M0b 
        kba = R*M0a
        # The lineshape function takes dw in hz!
        Rrfb = math.pi*w1*w1*cal_g(dwb/(2*math.pi), self.T2b, lineshape="SL")
        y0 = np.array([0, 0, 1, M0b, 1]) #[5,]
        A = np.array([[-R2a, -dwa, 0, 0, 0],
                      [dwa, -R2a, -w1, 0, 0],
                      [0, w1, -R1a-kab, kba, R1a*M0a],
                      [0, 0, kab, -R1b-kba-Rrfb, R1b*M0b],
                      [0, 0, 0, 0, 0]])
        return np.dot(expm(A*self.t_sat), y0) # [5,]
    
    def BM_model(self, freq, R, M0b):
        Zspec = []
        for i in range(freq.shape[0]): # if noise=0, Zspec[i] = tmp
            tmp = self.ANA(freq[i], R, M0b)[2]
            Zspec.append((1-self.noise)*tmp + self.noise)
        return np.array(Zspec)

    def generate_Zspec(self, offset, paras):
        freq = 128*offset # 3T
        return self.BM_model(freq, *paras)
    
    def cal_paras(self, paras):
        # [T1a, T1b, T2a, T2b, R, M0b, noise]
        return np.array([self.T1a, 1, self.T2a, self.T2b, paras[0], paras[1], self.noise])