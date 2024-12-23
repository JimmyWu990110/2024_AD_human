
import matplotlib.pyplot as plt

from DataLoader import DataLoader
from B0Correction import B0Correction 
from MT_model import MT_simp_model_4
from EMR_old import EMR_old

def test_model():
    # test given same parameters, the two models can generate the same Zspec
    # passed
    dataloader = DataLoader()
    offset = dataloader.offsets.EMR_offset_19_unique
    paras_old = [1.7, 0.6, 15.6, 7.48e-6]
    paras_new = [paras_old[1], paras_old[3], paras_old[0], paras_old[2]]
    old = EMR_old()
    new = MT_simp_model_4()
    Zspec_old = old.generate_Zspec(128*offset, paras_old)
    Zspec_new = new.generate_Zspec(128*offset, paras_new)
    print(Zspec_old)
    print(Zspec_new)
    plt.plot(offset, Zspec_old, label="old")
    plt.plot(offset, Zspec_new, label="new")
    plt.legend()
    plt.show()

def test_fitting():
    # test given same offset and Zspec, the two models can get consistent fitted parameters
    dataLoader = DataLoader("Phantom_20230620", "Coronal")
    correction = B0Correction()
    nth_slice = 8
    x = 137
    y = 128
    Zspec = dataLoader.get_Zspec_by_coordinate_24(nth_slice-1, y-1, x-1)
    offset, Zspec = dataLoader.Zspec_preprocess_onepixel(Zspec)
    # B0 correction
    wassr_spec = dataLoader.read_WASSR()[:, nth_slice-1, round((y-1)/2), round((x-1)/2)]
    B0_shift = correction.cal_B0_shift_onepixel(wassr_spec)
    Zspec = correction.correct_onepixel(offset, Zspec, B0_shift)
    T1 = dataLoader.read_T1_map()[nth_slice-1, y-1, x-1]
    T2 = dataLoader.read_T2_map()[nth_slice-1, y-1, x-1]
    print("B0 shift (ppm):", B0_shift)
    print("T1, T2, ratio:", T1, T2, T1/T2)
    old = EMR_old(T1w_obs=T1, T2w_obs=T2, B1=1.5)
    popt, y_estimated = old.fit(128*offset, Zspec, True)
    para_dict = old.cal_paras()
    print(para_dict)
    new = MT_simp_model_4(T1a_obs=T1, T2a_obs=T2, B1=1.5)
    fitted_paras = new.fit(128*offset, Zspec, constrained=True)
    print(fitted_paras)




# test_model()
test_fitting()