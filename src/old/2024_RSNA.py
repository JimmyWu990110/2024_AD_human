import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns

def ROI2filename(ROI):
    filename = "_hippocanpus.xlsx"
    if ROI == 2:
        filename = "_thalamus.xlsx"
    if ROI == 3:
        filename = "_WM.xlsx"
    if ROI == 4:
        filename = "_pons.xlsx"
    return filename

def analysis(ROI, feature):
    df_info = pd.read_excel(r"C:\Users\jwu191\Desktop\Projects\AD_fitting\Info.xlsx")
    df = pd.read_excel(os.path.join(r"C:\Users\jwu191\Desktop\Projects\AD_fitting\results", 
                                    ROI2filename(ROI)[1:]))
    CN = np.array(df_info[df_info["label"] == "CN"]["patient_name"])
    MCI = np.array(df_info[df_info["label"] == "MCI"]["patient_name"])
    AD = np.array(df_info[df_info["label"] == "AD"]["patient_name"])
    # print(CN, MCI, AD)
    CN_vals = np.array(df[df["Unnamed: 0"].isin(CN)][feature])
    MCI_vals = np.array(df[df["Unnamed: 0"].isin(MCI)][feature])
    AD_vals = np.array(df[df["Unnamed: 0"].isin(AD)][feature])
    print("********", ROI2filename(ROI)[1:-5]+" "+feature[:-5], "********")
    print("CN:", np.mean(CN_vals), np.std(CN_vals))
    print("MCI:", np.mean(MCI_vals), np.std(MCI_vals))
    print("AD:", np.mean(AD_vals), np.std(AD_vals))
    print("CN vs MCI:", stats.ttest_ind(CN_vals, MCI_vals, equal_var=True))
    print("MCI vs AD:", stats.ttest_ind(MCI_vals, AD_vals, equal_var=True))
    print("CN vs AD:", stats.ttest_ind(CN_vals, AD_vals, equal_var=True))
    
    err_params = dict(elinewidth=1.5, ecolor="black", capsize=4)
    plt.bar(["CN", "MCI", "AD"], [np.mean(CN_vals), np.mean(MCI_vals), np.mean(AD_vals)],
            yerr=[np.std(CN_vals), np.std(MCI_vals), np.std(AD_vals)],
            error_kw=err_params, width=0.5,
            color=["blue", "yellow", "red"])
    plt.title(ROI2filename(ROI)[1:-5]+" "+feature[:-5])
    plt.show()
    return [np.mean(CN_vals), np.mean(MCI_vals), np.mean(AD_vals)], [np.std(CN_vals), np.std(MCI_vals), np.std(AD_vals)]
    
def plot():
    # val_1, err_1 = analysis(1, "APT# mean")
    # val_2, err_2 = analysis(1, "NOE# mean")
    df_info = pd.read_excel(r"C:\Users\jwu191\Desktop\Projects\AD_fitting\Info.xlsx")
    df = pd.read_excel(os.path.join(r"C:\Users\jwu191\Desktop\Projects\AD_fitting\results", 
                                    ROI2filename(1)[1:]))
    CN = np.array(df_info[df_info["label"] == "CN"]["patient_name"])
    MCI = np.array(df_info[df_info["label"] == "MCI"]["patient_name"])
    AD = np.array(df_info[df_info["label"] == "AD"]["patient_name"])
    
    CN_APT_pow = np.array(df[df["Unnamed: 0"].isin(CN)]["APT# mean"])
    CN_NOE_pow = np.array(df[df["Unnamed: 0"].isin(CN)]["NOE# mean"])
    CN_APTw = np.array(df[df["Unnamed: 0"].isin(CN)]["APT# mean"])
    CN_MTR = np.array(df[df["Unnamed: 0"].isin(CN)]["MTR_20ppm mean"])
    CN_val = [np.mean(CN_APT_pow), np.mean(CN_NOE_pow), np.mean(CN_APTw), np.mean(CN_MTR)]
 
    MCI_APT_pow = np.array(df[df["Unnamed: 0"].isin(MCI)]["APT# mean"])
    MCI_NOE_pow = np.array(df[df["Unnamed: 0"].isin(MCI)]["NOE# mean"])
    MCI_APTw = np.array(df[df["Unnamed: 0"].isin(MCI)]["APT# mean"])
    MCI_MTR = np.array(df[df["Unnamed: 0"].isin(MCI)]["MTR_20ppm mean"])
    MCI_val = [np.mean(MCI_APT_pow), np.mean(MCI_NOE_pow), np.mean(MCI_APTw), np.mean(MCI_MTR)]   
 
    AD_APT_pow = np.array(df[df["Unnamed: 0"].isin(AD)]["APT# mean"])
    # plt.bar(["APT#", "NOE#", "APTw", "MTR 20ppm"], CN_val, label="CN")
    # plt.bar(["APT#", "NOE#", "APTw", "MTR 20ppm"], MCI_val, label="MCI")
    # plt.boxplot([CN_APT_pow, MCI_APT_pow, AD_APT_pow], labels=["CN", "MCI", "AD"])
    # plt.scatter()
    # plt.legend()
    tmp = ["CN"]*CN_APT_pow.shape[0] + ["MCI"]*MCI_APT_pow.shape[0] + ["AD"]*AD_APT_pow.shape[0]
    vals = np.concatenate((CN_APT_pow, MCI_APT_pow, AD_APT_pow))
    tmp = np.array(tmp)
    vals = vals.reshape((vals.shape[0], 1))
    tmp = tmp.reshape((tmp.shape[0], 1))
    data = np.hstack((tmp, vals.astype(np.float32)))
    # print(data)
    # df = pd.DataFrame(data=data, columns=["group", "value"])
    #df = pd.read_excel(r"C:\Users\jwu191\Desktop\Projects\AD_fitting\results\hippocanpus.xlsx")
    #sns.stripplot(data=df, x="label", y="APT# mean")
    
    err_params = dict(elinewidth=1.5, ecolor="black", capsize=4)
    plt.bar(["CN", "MCI", "AD"], [np.mean(CN_APT_pow), np.mean(MCI_APT_pow), np.mean(AD_APT_pow)],
            yerr=[np.std(CN_APT_pow), np.std(MCI_APT_pow), np.std(AD_APT_pow)],
            error_kw=err_params, width=0.5,
            color=["gainsboro", "silver", "gray"])
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.title("Hippocampus APT#", fontsize=16)
    plt.show()
    # figure, ax = plt.subplots()
    # rects1 = ax.bar(val_1, label)


def cor():
    CDR = [0, 0, 0, 0, 0, 0,
           0, 0, 0, 0, 0, 0,
           0, 0, 0, 0, 0, 0, 
           0.5, 0.5, 0.5, 0.5, 0, 1, 
           1, 0.5]
    MMSE = [30, 30, 30, 29, 28, 30, 
            29, 30, 28, 28, 26, 29, 
            29, 30, 29, 30, 29, 27, 
            30, 29, 28, 30, 29, 22, 
            25, 22]
    APT_pow = [2.858294446, 1.327801607, 2.388357619, 2.835813971, 2.559203228, 2.150307013,
               3.217199323, 2.514572264, 1.914146727, 2.464207374, 3.355591071, 2.042083011,
               3.266665437, 3.560097647, 2.189121234, 2.867261269, 3.528029748, 2.90447961, 
               2.779557432, 2.925357043, 2.493068673, 2.822687508, 2.406284418, 3.59949667, 
               4.114309974, 3.599828174]
    r1 = stats.pearsonr(MMSE, APT_pow)[0]
    print(r1)
    r2 = stats.pearsonr(CDR, APT_pow)[0]
    print(r2)
    
    # plt.scatter(CDR, APT_pow)
    # plt.ylabel("APT#")
    # plt.xlabel("CDR")
    # plt.title("Person's correlation cofficient = 0.496")
    # plt.show()

    plt.scatter(MMSE, APT_pow)
    plt.ylabel("APT#")
    plt.xlabel("MMSE")
    plt.title("Person's correlation cofficient = -0.524")
    plt.show()


    # MMSE = [30, 30, 30, 29, 28, 30, 
    #         29, 30, 28, 28, 26, 29, 
    #         29, 30, 30, 29, 30, 29, 
    #         27, 29, 30, 29, 28, 30, 
    #         28, 29, 22, 25, 22]
    # APT_pow = [2.858294446, 1.327801607, 2.388357619, 2.835813971, 2.559203228, 2.150307013,
    #            3.217199323, 2.514572264, 1.914146727, 2.464207374, 3.355591071, 2.042083011,
    #            3.266665437, 3.670495506, 3.560097647, 2.189121234, 2.867261269, 3.528029748,
    #            2.90447961, 2.065611823, 2.779557432, 2.925357043, 2.493068673, 2.822687508,
    #            5.894819357, 2.406284418, 3.39949667, 4.114309974, 3.599828174]

# analysis(1, "APT# mean")
# plot()
cor()








