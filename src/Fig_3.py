import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats


def plot_metrics(CN_list, AD_list):
    df_info = pd.read_excel(r"C:\Users\jwu191\Desktop\Projects\AD_fitting\info.xlsx")
    df = pd.read_excel(r"C:\Users\jwu191\Desktop\Projects\AD_fitting\results\BMS_EMR_v2_noisy\hippocampus.xlsx")
    CN_APT_pow = np.array(df[df["Unnamed: 0"].isin(CN_list)]["APT# mean"])
    CN_MMSE = np.array(df_info[df_info["patient_name"].isin(CN_list)]["MMSE"])
    CN_CDR = np.array(df_info[df_info["patient_name"].isin(CN_list)]["CDR"])
    AD_APT_pow = np.array(df[df["Unnamed: 0"].isin(AD_list)]["APT# mean"])
    AD_MMSE = np.array(df_info[df_info["patient_name"].isin(AD_list)]["MMSE"])
    AD_CDR = np.array(df_info[df_info["patient_name"].isin(AD_list)]["CDR"])
    print("CN APT#:", format(np.mean(CN_APT_pow), ".3f"), "+-", format(np.std(CN_APT_pow), ".3f"))
    print("AD APT#:", format(np.mean(AD_APT_pow), ".3f"), "+-", format(np.std(AD_APT_pow), ".3f"))
    print("APT# p-value:", stats.ttest_ind(CN_APT_pow, AD_APT_pow, equal_var=True))
    print("CN MMSE:", format(np.mean(CN_MMSE), ".3f"), "+-", format(np.std(CN_MMSE), ".3f"))
    print("AD MMSE:", format(np.mean(AD_MMSE), ".3f"), "+-", format(np.std(AD_MMSE), ".3f"))
    print("MMSE p-value:", stats.ttest_ind(CN_MMSE, AD_MMSE, equal_var=True))
    print("CN CDR:", format(np.mean(CN_CDR), ".3f"), "+-", format(np.std(CN_CDR), ".3f"))
    print("AD CDR:", format(np.mean(AD_CDR), ".3f"), "+-", format(np.std(AD_CDR), ".3f"))
    print("APT# p-value:", stats.ttest_ind(CN_CDR, AD_CDR, equal_var=True))
    groups = ["CN", "AD"]
    # Create a figure with 3 subplots
    fig, axes = plt.subplots(1, 3)  # rows, columns
    # Subplot 1
    axes[0].bar(groups, [np.mean(CN_APT_pow), np.mean(AD_APT_pow)], 
                yerr=[np.std(CN_APT_pow), np.std(AD_APT_pow)], capsize=5, color=["blue", "red"])
    axes[0].set_title("APT# (%)")
    # Subplot 2
    axes[1].bar(groups, [np.mean(CN_MMSE), np.mean(AD_MMSE)], 
                yerr=[np.std(CN_MMSE), np.std(AD_MMSE)], capsize=5, color=["blue", "red"])
    axes[1].set_title("MMSE")
    # Subplot 3
    axes[2].bar(groups, [np.mean(CN_CDR), np.mean(AD_CDR)], 
                yerr=[np.std(CN_CDR), np.std(AD_CDR)], capsize=5, color=["blue", "red"])
    axes[2].set_title("CDR")
    # Adjust layout
    plt.tight_layout()
    plt.show()
    

def plot_corr(CN_list, AD_list, feature):
    df_info = pd.read_excel(r"C:\Users\jwu191\Desktop\Projects\AD_fitting\info.xlsx")
    df = pd.read_excel(r"C:\Users\jwu191\Desktop\Projects\AD_fitting\results\BMS_EMR_v2_noisy\hippocampus.xlsx")
    CN_APT_pow = np.array(df[df["Unnamed: 0"].isin(CN_list)]["APT# mean"])
    CN_val = np.array(df_info[df_info["patient_name"].isin(CN_list)][feature])
    AD_APT_pow = np.array(df[df["Unnamed: 0"].isin(AD_list)]["APT# mean"])
    AD_val = np.array(df_info[df_info["patient_name"].isin(AD_list)][feature])
    APT_pow = np.concatenate((CN_APT_pow, AD_APT_pow))
    val = np.concatenate((CN_val, AD_val))
    print("Correlation:", stats.pearsonr(APT_pow, val))
    print("APT# p-value:", stats.ttest_ind(CN_APT_pow, AD_APT_pow, equal_var=True))
    print(feature+" p-value:", stats.ttest_ind(CN_val, AD_val, equal_var=True))
    plt.scatter(CN_val, CN_APT_pow, label="CN")
    plt.scatter(AD_val, AD_APT_pow, label="AD")
    plt.xlabel(feature)
    plt.ylabel("APT# (%)")
    plt.ylim((0, 6))
    plt.legend()
    plt.title("Corelation analysis")
    plt.show()


CN_list = ["AD_45", "AD_48", "AD_50", "AD_52", "AD_54", "AD_56", "AD_58", "AD_60", 
           "AD_63", "AD_66", "AD_70", "AD_71", "AD_76", "AD_79"] # n=14
AD_list = ["AD_46", "AD_51", "AD_55", "AD_59", "AD_64", "AD_73", "AD_74", "AD_75",
           "AD_78", "AD_80", "AD_83", "AD_86"] # n=12

# plot_metrics(CN_list, AD_list)
plot_corr(CN_list, AD_list, "MMSE")
plot_corr(CN_list, AD_list, "CDR")





