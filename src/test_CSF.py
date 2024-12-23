
from utils import fit_by_ROI_mask, fit_by_ROI_mask_single
from funcs_results import print_paras, save_Zspec, plot_fitting


patient_name = "AD_77"
method = "BMS_EMR_v2_noisy"
ROI = 5

fit_by_ROI_mask_single(patient_name=patient_name, ROI=ROI, method=method)

print_paras(patient_name, method, ROI, single=True)
save_Zspec(patient_name, method, ROI, single=True)
plot_fitting(patient_name, method, ROI)
