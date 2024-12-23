import os
import numpy as np
import pandas as pd

def stat_onecase(patient_name, method):
    print("********", method, "********")
    base_dir = os.path.join(r"C:\Users\jwu191\Desktop\Projects\AD_fitting\results",
                            patient_name, method)
    df = pd.read_excel(os.path.join(base_dir, "hippocampus_fitted_24.xlsx"))
    mean = np.mean(np.array(df), axis=0)
    std = np.std(np.array(df), axis=0)
    for i in range(1, mean.shape[0]):
        print(df.columns[i], ":", mean[i], "+-", std[i])



for patient_name in ["Phantom_20230620"]:
    stat_onecase(patient_name, "MT_EMR")





