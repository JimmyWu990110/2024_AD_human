import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats


CN_list = ["AD_45", "AD_48", "AD_49", "AD_50", "AD_52", "AD_53", "AD_54", "AD_56", 
           "AD_57", "AD_58", "AD_60", "AD_61", "AD_63", "AD_66", "AD_67", "AD_68", 
           "AD_70", "AD_71", "AD_72", "AD_76", "AD_77", "AD_79"] # 22 cases
MCI_list = ["AD_51", "AD_55", "AD_59", "AD_73", "AD_78"]  # 5 cases
AD_list = ["AD_46", "AD_74", "AD_75", "AD_80"]  # 4 cases
MCIAD_list = MCI_list + AD_list # 9 cases
whole_list = CN_list + MCIAD_list # 31 cases
CN_selected_list = ["AD_49", "AD_52", "AD_57", "AD_60", "AD_66", "AD_70", 
                    "AD_72", "AD_76", "AD_79"] # 9 of 22 selected

class Analysis:
    def __init__(self):
        self.base_dir = r"C:\Users\jwu191\Desktop\Projects\AD_fitting\results"
    
    def _get_cols(self, method):
        if "MT" in method:
            return ["R*M0b*T1a", "T1a/T2a", "T1a", "T1b", "T2a", "T2b", "R", "M0b", 
                    "NRMSE", "APT#", "NOE#", "APTw", "MTR_20ppm", "MTR_60ppm", 
                    "T1obs", "T2obs", "z", "x", "y"]
        if "BM" in method:
            return ["T1a", "T1b", "T2a", "T2b", "R", "M0b", 
                    "NRMSE", "APT#", "NOE#", "APTw", "MTR_20ppm", "MTR_60ppm", 
                    "T1obs", "T2obs", "z", "x", "y"]
    
    def _ROI2filename(self, ROI):
        filename = "_hippocampus_"
        if ROI == 2:
            filename = "_thalamus_"
        if ROI == 3:
            filename = "_WM_"
        if ROI == 4:
            filename = "_pons_"
        return filename
        
    def get_stat_onecase(self, patient_name, ROI, method):
        base_dir = r"C:\Users\jwu191\Desktop\Projects\AD_fitting\results"
        df = pd.read_excel(os.path.join(base_dir, patient_name, method,
                                        "hippocampus_fitted_24.xlsx"))
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
    
    def get_stat(self, patient_list, ROI, method):
        """ 
        Given patient list, generate an excel where each row is a patient and each
        column is a feature
        """
        data = []
        for patient_name in patient_list:
            data.append(self.get_stat_onecase(patient_name, ROI, method))
        columns = ["APT# mean", "APT# std", "APT# 10%", "APT# median", "APT# 90%",
                   "NOE# mean", "NOE# std", "NOE# 10%", "NOE# median", "NOE# 90%",
                   "APTw mean", "APTw std","APTw 10%", "APTw median", "APTw 90%",
                   "MTR_20ppm mean", "MTR_20ppm std", "MTR_20ppm 10%", "MTR_20ppm median", "MTR_20ppm 90%",
                   "MTR_60ppm mean", "MTR_60ppm std", "MTR_60ppm 10%", "MTR_60ppm median", "MTR_60ppm 90%"]
        df = pd.DataFrame(data=np.array(data), index=patient_list, columns=columns)
        df.to_excel(os.path.join(r"C:\Users\jwu191\Desktop\Projects\AD_fitting\results", 
                                 self._ROI2filename(ROI)[1:]+method+".xlsx")) 

    def boxplot(self, CN, MCI, AD):
        plt.boxplot(CN, MCI, AD, labels=["CN", "MCI", "AD"])
        plt.show()
    
    def stat_onefeaure(self, ROI, method, feature, show=False):
        df = pd.read_excel(os.path.join(r"C:\Users\jwu191\Desktop\Projects\AD_fitting\results", 
                                        self._ROI2filename(ROI)[1:]+method+".xlsx"))
        CN = CN_selected_list
        MCI = MCI_list
        AD = AD_list
        CN_vals = np.array(df[df["Unnamed: 0"].isin(CN)][feature])
        MCI_vals = np.array(df[df["Unnamed: 0"].isin(MCI)][feature])
        AD_vals = np.array(df[df["Unnamed: 0"].isin(AD)][feature])
        MCI_AD_vals = np.concatenate((MCI_vals, AD_vals))
        if show:
            print("********", method, feature, "********")
            print("CN:", format(np.mean(CN_vals), ".3f"), "+-", format(np.std(CN_vals), ".3f"))
            # print("MCI:", format(np.mean(MCI_vals), ".3f"), "+-", format(np.std(MCI_vals), ".3f"))
            # print("AD:", format(np.mean(AD_vals), ".3f"), "+-", format(np.std(AD_vals), ".3f"))
            # print("CN vs. MCI:", stats.ttest_ind(CN_vals, MCI_vals, equal_var=True)[1])
            # print("MCI vs. AD:",stats.ttest_ind(MCI_vals, AD_vals, equal_var=True)[1])
            # print("CN vs. AD:",stats.ttest_ind(CN_vals, AD_vals, equal_var=True)[1])
            print("MCI&AD:", format(np.mean(MCI_AD_vals), ".3f"), "+-", format(np.std(MCI_AD_vals), ".3f"))
            print("CN vs. MCI&AD:", stats.ttest_ind(CN_vals, MCI_AD_vals, equal_var=True)[1])
        patients = np.concatenate((CN, MCI, AD))
        vals = np.concatenate((CN_vals, MCI_vals, AD_vals))
        return patients, vals
    
    def stat_by_group(self, ROI, method, feature_list, show):
        patients = None
        data = []
        for feature in feature_list:
            patients, vals = self.stat_onefeaure(ROI, method, feature, show)
            data.append(vals)
        df = pd.DataFrame(data=np.array(data).T, columns=feature_list, index=patients)
        base_dir = r"C:\Users\jwu191\Desktop\Projects\AD_fitting\results\STAT"
        filename = self._ROI2filename(ROI)[1:]+method+".xlsx"
        df.to_excel(os.path.join(base_dir, filename))

    def stat(self, patient_name, method):
        print("********", patient_name, method, "********")
        df = pd.read_excel(os.path.join(self.base_dir, patient_name, method,
                                        "hippocampus_fitted_24.xlsx"))
        print(df.shape)
        columns = self._get_cols(method)[:-3]
        for col in columns:
            data = np.array(df[col])
            if col == "T2a" or col == "T2obs":
                data *= 1000 # ms
            if col == "T2b":
                data *= 1e6 # us
            print(col+":", format(np.mean(data), ".3f"), "+-", format(np.std(data), ".3f"),
                  "("+format(np.min(data), ".3f"), "~", format(np.max(data), ".3f")+")")

    def correlation(self, ROI, method):
        null_list = ["AD_49", "AD_69"]
        patient_list = get_patients(24)[:-5]
        # for patient_name in null_list:
        #     patient_list.remove(patient_name) # 43 cases
        base_dir = r"C:\Users\jwu191\Desktop\Projects\AD_fitting"
        df_info = pd.read_excel(os.path.join(base_dir, "Info.xlsx"))
        filename = self._ROI2filename(ROI)[1:]+method+".xlsx"
        df = pd.read_excel(os.path.join(base_dir, "results", filename))
        CDR = []
        MMSE = []
        APT_pow = []
        NOE_pow = []
        APTw = []
        MTR_20ppm = []
        MTR_60ppm = []
        for patient_name in patient_list:
            CDR.append(float(df_info[df_info["patient_name"] == patient_name]["CDR"]))
            MMSE.append(float(df_info[df_info["patient_name"] == patient_name]["MMSE"]))
            APT_pow.append(float(df[df["Unnamed: 0"] == patient_name]["APT# mean"]))
            NOE_pow.append(float(df[df["Unnamed: 0"] == patient_name]["NOE# mean"]))
            APTw.append(float(df[df["Unnamed: 0"] == patient_name]["APTw mean"]))
            MTR_20ppm.append(float(df[df["Unnamed: 0"] == patient_name]["MTR_20ppm mean"]))
            MTR_60ppm.append(float(df[df["Unnamed: 0"] == patient_name]["MTR_60ppm mean"]))
        # correlation analysis
        print("APT# and CDR", format(stats.pearsonr(APT_pow, CDR)[0], ".3f"))
        print("NOE# and CDR", format(stats.pearsonr(NOE_pow, CDR)[0], ".3f"))
        print("APTw and CDR", format(stats.pearsonr(APTw, CDR)[0], ".3f"))
        print("MTR_20ppm and CDR", format(stats.pearsonr(MTR_20ppm, CDR)[0], ".3f"))
        print("MTR_60ppm and CDR", format(stats.pearsonr(MTR_60ppm, CDR)[0], ".3f"))
        print("APT# and MMSE", format(stats.pearsonr(APT_pow, MMSE)[0], ".3f"))
        print("NOE# and MMSE", format(stats.pearsonr(NOE_pow, MMSE)[0], ".3f"))
        print("APTw and MMSE", format(stats.pearsonr(APTw, MMSE)[0], ".3f"))
        print("MTR_20ppm and MMSE", format(stats.pearsonr(MTR_20ppm, MMSE)[0], ".3f"))
        print("MTR_60ppm and MMSE", format(stats.pearsonr(MTR_60ppm, MMSE)[0], ".3f"))

    def test_corr(self):
        null_list = ["AD_49", "AD_69"]
        patient_list = get_patients()[:-5]
        # for patient_name in null_list:
        #     patient_list.remove(patient_name) # 43 cases
        base_dir = r"C:\Users\jwu191\Desktop\Projects\AD_fitting"
        df_info = pd.read_excel(os.path.join(base_dir, "Info.xlsx"))
        df_info = df_info[df_info["patient_name"].isin(patient_list)]
        CN = df_info[df_info["label"] == "CN"]
        MCI = df_info[df_info["label"] == "MCI"]
        AD = df_info[df_info["label"] == "AD"]
        CDR_CN = np.array(CN["CDR"])
        CDR_MCI = np.array(MCI["CDR"])
        CDR_AD = np.array(AD["CDR"])
        print("******** CDR ********")
        print("CN:", format(np.mean(CDR_CN), ".3f"), "+-", format(np.std(CDR_CN), ".3f"))
        print("MCI:", format(np.mean(CDR_MCI), ".3f"), "+-", format(np.std(CDR_MCI), ".3f"))
        print("AD:", format(np.mean(CDR_AD), ".3f"), "+-", format(np.std(CDR_AD), ".3f"))
        print("CN vs. MCI:", stats.ttest_ind(CDR_CN, CDR_MCI, equal_var=True))
        print("MCI vs. AD:",stats.ttest_ind(CDR_MCI, CDR_AD, equal_var=True))
        print("CN vs. AD:",stats.ttest_ind(CDR_CN, CDR_AD, equal_var=True))
        MMSE_CN = np.array(CN["MMSE"])
        MMSE_MCI = np.array(MCI["MMSE"])
        MMSE_AD = np.array(AD["MMSE"])
        print("******** MMSE ********")
        print("CN:", format(np.mean(MMSE_CN), ".3f"), "+-", format(np.std(MMSE_CN), ".3f"))
        print("MCI:", format(np.mean(MMSE_MCI), ".3f"), "+-", format(np.std(MMSE_MCI), ".3f"))
        print("AD:", format(np.mean(MMSE_AD), ".3f"), "+-", format(np.std(MMSE_AD), ".3f"))
        print("CN vs. MCI:", stats.ttest_ind(MMSE_CN, MMSE_MCI, equal_var=True))
        print("MCI vs. AD:",stats.ttest_ind(MMSE_MCI, MMSE_AD, equal_var=True))
        print("CN vs. AD:",stats.ttest_ind(MMSE_CN, MMSE_AD, equal_var=True))

analysis = Analysis()
# analysis.stat(patient_name="Phantom_20230620", filename="_slice_8_", method="BMS_EMR")
# analysis.stat(patient_name="AD_46", filename="_hippocampus_", method="MT")
# analysis.stat(patient_name="AD_46", filename="_hippocampus_", method="MT_EMR")

# NOTE: AD_65, AD_69 should be deleted; AD_62 bad APT quality

# analysis.get_stat(patient_list=whole_list, ROI=1, method="MT_EMR")
# analysis.get_stat(patient_list=whole_list, ROI=1, method="BMS_EMR")
analysis.stat_onefeaure(ROI=1, method="MT_EMR", feature="APT# mean", show=True)
analysis.stat_onefeaure(ROI=1, method="MT_EMR", feature="NOE# mean", show=True)
analysis.stat_onefeaure(ROI=1, method="BMS_EMR", feature="APT# mean", show=True)
analysis.stat_onefeaure(ROI=1, method="BMS_EMR", feature="NOE# mean", show=True)
analysis.stat_onefeaure(ROI=1, method="MT_EMR", feature="APTw mean", show=True)
analysis.stat_onefeaure(ROI=1, method="MT_EMR", feature="MTR_20ppm mean", show=True)
analysis.stat_onefeaure(ROI=1, method="MT_EMR", feature="MTR_60ppm mean", show=True)

# feature_list = ["APT# mean", "NOE# mean", "APTw mean", "MTR_20ppm mean", "MTR_60ppm mean"]
# analysis.stat_by_group(ROI=1, method=method, feature_list=feature_list, show=True)



