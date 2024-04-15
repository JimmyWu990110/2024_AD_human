import os
import numpy as np
import pandas as pd

from DataLoader import DataLoader
from Visualization import Visualization
from EMR_fitting import EMR_fitting


def build_dict(lineshape):
    dataLoader = DataLoader()
    visualization = Visualization()
    fitting = EMR_fitting()
    fitting.set_lineshape(lineshape)
    offset, _ = dataLoader.Zspec_preprocess_onepixel(np.ones(56), 56)
    
    # 20*12*12*12 = 34560 combinations
    R = np.array([0.5, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 16, 18, 20, 25, 30, 40, 50])
    RM0mT1w = np.array([0.2, 0.35, 0.5, 0.75, 1, 1.5, 2, 2.5, 3, 3.5, 4, 5])
    T1wT2w = np.array([5, 8, 10, 12, 14, 16, 18, 20, 25, 30, 40, 50])
    T2m = 1e-6 * np.array([1, 2.5, 3.5, 5, 6.5, 8.5, 10, 12, 15, 17.5, 20, 25])
    
    data = []
    counter = 0
    for i in range(R.shape[0]):
        for j in range(RM0mT1w.shape[0]):
            for k in range(T1wT2w.shape[0]):
                for m in range(T2m.shape[0]):
                    paras = np.array([R[i], RM0mT1w[j], T1wT2w[k], T2m[m]])
                    Zspec = fitting.generate_Zpsec(128*offset, paras)
                    data.append([*paras, *Zspec])
                    counter += 1
                    if counter % 1000 == 0:
                        print(counter)
                        
    df = pd.DataFrame(data=data)
    df.to_excel(os.path.join(r"C:\Users\jwu191\Desktop", lineshape+"_54.xlsx"))













