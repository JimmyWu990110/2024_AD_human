import numpy as np

def lorentz_func(x, amplitude, center, width):
    tmp = ((x-center) / width) * ((x-center) / width)
    ret = amplitude / (1 + 4*tmp)
    return ret


class Lorentz_5pool_model:
    def __init__(self):
        pass

    def Lorentz_model(self, offset, amplitude_apt, center_apt, width_apt,
                      amplitude_cest2ppm, center_cest2ppm, width_cest2ppm,
                      amplitude_ds, width_ds,
                      amplitude_mt, center_mt, width_mt,
                      amplitude_noe, center_noe, width_noe):
        Zspec = []
        center_ds = 0 # only 14 paras!
        for x in offset:
            apt = lorentz_func(x, amplitude_apt, center_apt, width_apt)
            cest2ppm = lorentz_func(x, amplitude_cest2ppm, center_cest2ppm, width_cest2ppm)
            ds = lorentz_func(x, amplitude_ds, center_ds, width_ds)
            mt = lorentz_func(x, amplitude_mt, center_mt, width_mt)
            noe = lorentz_func(x, amplitude_noe, center_noe, width_noe)
            Zspec.append(1 - (apt+cest2ppm+ds+mt+noe))
        return np.array(Zspec)
    
    def generate_Zspec(self, offset, paras):
        return self.Lorentz_model(offset, *paras)
    
    def cal_paras(self, paras):
        if paras.shape[0] == 14:
            return np.array([*paras[0:7], 0, *paras[7:14]])


class Lorentz_2pool_model:
    def __init__(self):
        pass

    def Lorentz_model(self, offset, amplitude_ds, width_ds, amplitude_mt, center_mt, width_mt):
        Zspec = []
        center_ds = 0 # only 5 paras!
        for x in offset:
            ds = lorentz_func(x, amplitude_ds, center_ds, width_ds)
            mt = lorentz_func(x, amplitude_mt, center_mt, width_mt)
            Zspec.append(1-ds-mt)
        return np.array(Zspec)
    
    def generate_Zspec(self, offset, paras):
        return self.Lorentz_model(offset, *paras)
    
    def cal_paras(self, paras):
        if paras.shape[0] == 5:
            return np.array([*paras[0:2], 0, *paras[2:5]])






