import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

from DataLoader import DataLoader


class Analysis:
    def __init__(self):
        pass
    
    def _ROI2filename(self, ROI):
        filename = "_hippocanpus.xlsx"
        if ROI == 2:
            filename = "_thalamus.xlsx"
        if ROI == 3:
            filename = "_WM.xlsx"
        if ROI == 4:
            filename = "_pons.xlsx"
        return filename
        
    def stat_onecase(self, patient_name, ROI):
        base_dir = r"C:\Users\jwu191\Desktop\Projects\AD_fitting\results"
        filename = self._ROI2filename(ROI)
        df = pd.read_excel(os.path.join(base_dir, patient_name, patient_name+filename))
        APT_pow = np.array(df["APT#"])
        NOE_pow = np.array(df["NOE#"])
        APTw = np.array(df["APTw"])
        MTR_20ppm = np.array(df["MTR_20ppm"])
        MTR_60ppm = np.array(df["MTR_60ppm"])
        stat = [np.mean(APT_pow), np.std(APT_pow), np.percentile(APT_pow, 10), np.median(APT_pow), np.percentile(APT_pow, 90),
                np.mean(NOE_pow), np.std(NOE_pow), np.percentile(NOE_pow, 10), np.median(NOE_pow), np.percentile(NOE_pow, 90),
                np.mean(APTw), np.std(APTw), np.percentile(APTw, 10), np.median(APTw), np.percentile(APTw, 90),
                np.mean(MTR_20ppm), np.std(MTR_20ppm), np.percentile(MTR_20ppm, 10), np.median(MTR_20ppm), np.percentile(MTR_20ppm, 90),
                np.mean(MTR_60ppm), np.std(MTR_60ppm), np.percentile(MTR_60ppm, 10), np.median(MTR_60ppm), np.percentile(MTR_60ppm, 90)]
        return np.array(stat)
    
    def stat_all(self, patient_list, ROI):
        data = []
        for patient_name in patient_list:
            data.append(self.stat_onecase(patient_name, ROI))
        columns = ["APT# mean", "APT# std", "APT# 10%", "APT# median", "APT# 90%",
                   "NOE# mean", "NOE# std", "NOE# 10%", "NOE# median", "NOE# 90%",
                   "APTw mean", "APTw std","APTw 10%", "APTw median", "APTw 90%",
                   "MTR_20ppm mean", "MTR_20ppm std", "MTR_20ppm 10%", "MTR_20ppm median", "MTR_20ppm 90%",
                   "MTR_60ppm mean", "MTR_60ppm std", "MTR_60ppm 10%", "MTR_60ppm median", "MTR_60ppm 90%"]
        df = pd.DataFrame(data=np.array(data), index=patient_list, columns=columns)
        df.to_excel(os.path.join(r"C:\Users\jwu191\Desktop\Projects\AD_fitting\results", 
                                 self._ROI2filename(ROI)[1:])) 

    def boxplot(self, CN, MCI, AD):
        plt.boxplot(CN, MCI, AD, labels=["CN", "MCI", "AD"])
        plt.show()
        
    def basic_stat(self, CN_vals, MCI_vals, AD_vals):
        print("CN:", np.mean(CN_vals), CN_vals)
        print("MCI:", np.mean(MCI_vals), MCI_vals)
        print("AD:", np.mean(AD_vals), AD_vals)
        print(stats.ttest_ind(CN_vals, MCI_vals, equal_var=True))
        print(stats.ttest_ind(CN_vals, AD_vals, equal_var=True))
    
    def stat_by_group(self, ROI, feature):
        df_info = pd.read_excel(r"C:\Users\jwu191\Desktop\Projects\AD_fitting\Info.xlsx")
        df = pd.read_excel(os.path.join(r"C:\Users\jwu191\Desktop\Projects\AD_fitting\results", 
                                 self._ROI2filename(ROI)[1:]))
        CN = np.array(df_info[df_info["label"] == "CN"]["patient_name"])
        MCI = np.array(df_info[df_info["label"] == "MCI"]["patient_name"])
        AD = np.array(df_info[df_info["label"] == "AD"]["patient_name"])
        print(MCI, AD)
        CN_vals = np.array(df[df["Unnamed: 0"].isin(CN)][feature])
        MCI_vals = np.array(df[df["Unnamed: 0"].isin(MCI)][feature])
        AD_vals = np.array(df[df["Unnamed: 0"].isin(AD)][feature])
        # self.boxplot(CN_vals, MCI_vals, AD_vals)
        self.basic_stat(CN_vals, MCI_vals, AD_vals)

        
        
patient_name = "AD_75"
# patient_list = ["AD_05", "AD_06", "AD_07", "AD_29", "AD_30", "AD_31", "AD_32", "AD_33", 
#                 "AD_34", "AD_35", "AD_37", "AD_38", "AD_39", "AD_40", "AD_41",
#                 "AD_45", "AD_46", "AD_47", "AD_48", "AD_49", "AD_50", "AD_51", "AD_52", 
#                 "AD_53", "AD_54", "AD_55", "AD_56", "AD_57", "AD_58", "AD_59", "AD_60", 
#                 "AD_61", "AD_62", "AD_63", "AD_64", "AD_66", "AD_67", "AD_68", "AD_69", 
#                 "AD_70", "AD_71", "AD_72", "AD_73", "AD_74", "AD_75", "AD_76"]
patient_list = ["AD_45", "AD_46", "AD_47", "AD_48", "AD_49", "AD_50", "AD_51", "AD_52", 
                "AD_53", "AD_54", "AD_55", "AD_56", "AD_57", "AD_58", "AD_59", "AD_60", 
                "AD_61", "AD_62", "AD_63", "AD_64", "AD_66", "AD_67", "AD_68", "AD_69", 
                "AD_70", "AD_71", "AD_72", "AD_73", "AD_74", "AD_75", "AD_76"]

analysis = Analysis()
# analysis.stat_onecase(patient_name, ROI)
for i in range(1, 5):
    analysis.stat_all(patient_list, i)
# analysis.stat_by_group(ROI, "APT# mean")





