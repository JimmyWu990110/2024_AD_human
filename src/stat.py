import os
import numpy as np
import pandas as pd
from scipy import stats

from funcs import ROI2filename


def _get_stat(df, lorentz=False):
    if lorentz:
        return [100*np.mean(df["APT magnitude"]), np.mean(df["APT center"]), np.mean(df["APT width"]),
                100*np.mean(df["CEST2ppm magnitude"]), np.mean(df["CEST2ppm center"]), np.mean(df["CEST2ppm width"]),
                100*np.mean(df["DS magnitude"]), np.mean(df["DS center"]), np.mean(df["DS width"]),
                100*np.mean(df["MT magnitude"]), np.mean(df["MT center"]), np.mean(df["MT width"]),
                100*np.mean(df["NOE magnitude"]), np.mean(df["NOE center"]), np.mean(df["NOE width"]),]
    else:
        return [np.mean(df["APT#"]), np.std(df["APT#"]), np.quantile(df["APT#"], 0.1),
                np.median(df["APT#"]), np.quantile(df["APT#"], 0.9),
                np.mean(df["NOE#"]), np.std(df["NOE#"]), np.quantile(df["NOE#"], 0.1),
                np.median(df["NOE#"]), np.quantile(df["NOE#"], 0.9),
                np.mean(df["APTw"]), np.std(df["APTw"]), np.quantile(df["APTw"], 0.1),
                np.median(df["APTw"]), np.quantile(df["APTw"], 0.9),
                np.mean(df["MTR_20ppm"]), np.std(df["MTR_20ppm"]), np.quantile(df["MTR_20ppm"], 0.1),
                np.median(df["MTR_20ppm"]), np.quantile(df["MTR_20ppm"], 0.9),
                np.mean(df["MTR_60ppm"]), np.std(df["MTR_60ppm"]), np.quantile(df["MTR_60ppm"], 0.1),
                np.median(df["MTR_60ppm"]), np.quantile(df["MTR_60ppm"], 0.9)]

def _get_cols(lorentz=False):
    if lorentz:
        return ["APT magnitude", "APT center", "APT width",
                "CEST2ppm magnitude", "CEST2ppm center", "CEST2ppm width",
                "DS magnitude", "DS center", "DS width",
                "MT magnitude", "MT center", "MT width",
                "NOE magnitude", "NOE center", "NOE width"]
    else:
        return ["APT# mean", "APT# std", "APT# 10%", "APT# 50%", "APT# 90%",
                "NOE# mean", "NOE# std", "NOE# 10%", "NOE# 50%", "NOE# 90%",
                "APTw mean", "APTw std", "APTw 10%", "APTw 50%", "APTw 90%",
                "MTR_20ppm mean", "MTR_20ppm std", "MTR_20ppm 10%", "MTR_20ppm 50%", "MTR_20ppm 90%",
                "MTR_60ppm mean", "MTR_60ppm std", "MTR_60ppm 10%", "MTR_60ppm 50%", "MTR_60ppm 90%"]
    
def to_excel(method, ROI, patient_list):
    lorentz = False
    if "Lorentz" in method:
        lorentz = True
    base_dir = r"C:\Users\jwu191\Desktop\Projects\AD_fitting\results"
    data = []
    for patient_name in patient_list:
        filename = ROI2filename(ROI)+"_SL_fitted_paras.xlsx"
        if lorentz:
            filename = ROI2filename(ROI)+"_single_fitted_paras.xlsx"
        df = pd.read_excel(os.path.join(base_dir, patient_name, method, filename))
        data.append(_get_stat(df, lorentz))
    df = pd.DataFrame(data=np.array(data), columns=_get_cols(lorentz),
                      index=patient_list)
    output_dir = os.path.join(base_dir, method)
    os.makedirs(output_dir, exist_ok=True)
    df.to_excel(os.path.join(output_dir, ROI2filename(ROI)+".xlsx"))

def group_stat(method, ROI, CN_list, AD_list):
    df = pd.read_excel(os.path.join(r"C:\Users\jwu191\Desktop\Projects\AD_fitting\results",
                                    method, ROI2filename(ROI)+".xlsx"))
    features = ["APT# mean", "NOE# mean", "APTw mean", "MTR_20ppm mean", "MTR_60ppm mean"]
    if "Lorentz" in method:
        features = ["DS magnitude", "DS width", "MT magnitude", "MT width"]
    # features = _get_cols()
    for feature in features:
        print("********", feature, "********")
        CN = np.array(df[df["Unnamed: 0"].isin(CN_list)][feature])
        AD = np.array(df[df["Unnamed: 0"].isin(AD_list)][feature])
        print("CN:", format(np.mean(CN), ".3f"), "+-", format(np.std(CN), ".3f"))
        print("AD:", format(np.mean(AD), ".3f"), "+-", format(np.std(AD), ".3f"))
        print("CN vs AD:", stats.ttest_ind(CN, AD, equal_var=True))
    pass

def get_demongraphic(CN_list, AD_list):
    pass

# 62, 68, 69 exclued 
patient_list = ["AD_45", "AD_46", "AD_48", "AD_49", "AD_50", "AD_51", "AD_52", "AD_53", 
                "AD_54", "AD_55", "AD_56", "AD_57", "AD_58", "AD_59", "AD_60", "AD_61", 
                "AD_63", "AD_64", "AD_66", "AD_67", "AD_70", 
                "AD_71", "AD_72", "AD_73", "AD_74", "AD_75", "AD_76", "AD_77", "AD_78",
                "AD_79", "AD_80", "AD_81", "AD_82", "AD_83", "AD_84", "AD_85", "AD_86"]
# 62, 68, 69 exclued 
# CN_list = ["AD_45", "AD_48", "AD_49", "AD_50", "AD_52", "AD_53", "AD_54", "AD_56", 
#            "AD_57", "AD_58", "AD_60", "AD_61", "AD_63", "AD_66", "AD_67", "AD_70", 
#            "AD_71", "AD_72", "AD_76", "AD_77", "AD_79", "AD_81", "AD_82", "AD_84",
#            "AD_85"]
# CN_list = ["AD_45", "AD_48", "AD_50", "AD_52", "AD_54", "AD_56", "AD_58", "AD_60", 
#            "AD_63", "AD_66", "AD_70", "AD_71", "AD_76", "AD_79"] # n=14
# AD_list = ["AD_46", "AD_51", "AD_55", "AD_59", "AD_64", "AD_73", "AD_74", "AD_75",
#            "AD_78", "AD_80", "AD_83", "AD_86"] # n=12

CN_list = ["AD_76", "AD_77", "AD_79", "AD_81", "AD_82", "AD_84", "AD_85"]
AD_list = ["AD_80", "AD_83", "AD_86"] # remove 78

method = "Lorentz_5pool"
ROI = 1
# patient_list = CN_list + AD_list
to_excel(method, ROI, CN_list+AD_list)
group_stat(method, ROI, CN_list, AD_list)

