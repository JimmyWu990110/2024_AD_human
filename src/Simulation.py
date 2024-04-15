import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from EMR_fitting import EMR_fitting
from DataLoader import DataLoader
from Visualization import Visualization

def offset_preprocess(offset):
    offset = np.array(offset)
    sorted_index = np.argsort(offset)[::-1]
    return offset[sorted_index]

def get_avg_Zspec(path):
    df = pd.read_csv(path) 
    data = np.array(df)
    data = data[:, 3:]
    return np.mean(data, axis=0)

class Simulation:
    def __init__(self, B1=1.5):
        self.B1 = B1
        # self.wide_freq = 128 * offset_preprocess(DataLoader.Zspec_offset.wide_14)
        # self.middle_freq = 128 * offset_preprocess(DataLoader.Zspec_offset.middle_41)
        # self.Zspec_wide = None
        # self.Zspec_middle = None
        
    @staticmethod
    def plotZspec(offset, Zspec, title):
        # TODO: change scale to make the Zspec detail clear
        plt.ylim((0, 1))
        plt.scatter(offset, Zspec, label='Zspec', color='black')
        plt.gca().invert_xaxis()  # invert x-axis
        plt.legend()
        plt.title(title)
        plt.show() 
        
    @staticmethod       
    def compareZspec(offset, Zspec_simulated, Zspec_patient, title):
        plt.ylim((0, 1))
        plt.scatter(offset, Zspec_patient, label='Zspec_patient', color='green')
        plt.scatter(offset, Zspec_simulated, label='Zspec_simulated', color='red')
        plt.gca().invert_xaxis()  # invert x-axis
        plt.legend()
        plt.title(title)
        plt.show() 
        
    def simulate(self, offset, paras, T1=1, T2=0.3, lineshape="SL"):
        dataloader = DataLoader("", "")
        fitting = EMR_fitting(T1, T2)
        fitting.set_lineshape(lineshape)
        # self.plotZspec(offset, Zspec_wide, str(paras))
        return fitting.MT_model(128*offset, *paras)

    def load_patient(self, patient_name):
        dataloader = DataLoader(patient_name)
        dataloader.get_nth_slice_by_Zspec_mask()
        dataloader.get_ROIs_by_Zspec_mask()
        dataloader.read_Zspec(self.B1)
        Zspec_wide_patient = dataloader.get_tumor_avg_Zspec(self.wide_freq/128)
        self.compareZspec(self.wide_freq/128, self.Zspec_wide, 
                          Zspec_wide_patient, "wide")
        Zspec_middle_patient = dataloader.get_tumor_avg_Zspec(self.middle_freq/128)
        self.compareZspec(self.middle_freq/128, self.Zspec_middle, 
                          Zspec_middle_patient, "wide")
        
    def evaluate_R(self):
        paras = np.array([[2, 0.2, 10, 8e-6],
                          [20, 2, 10, 8e-6],
                          [200, 20, 10, 8e-6],
                          [2000, 200, 10, 8e-6],
                          [20000, 2000, 10, 8e-6],
                          [200000, 20000, 10, 8e-6]])
        colors = ["red", "orange", "yellow", "green", "blue", "purple"]
        for i in range(paras.shape[0]):
            Zspec_wide, Zspec_middle = self.simulate(paras[i])
            plt.scatter(self.wide_freq/128, Zspec_wide, color=colors[i])
            # plt.scatter(self.middle_freq/128, Zspec_middle, color=colors[i])
        plt.show()



def compare(patient_name, view, nth_slice, x, y):
    dataLoader = DataLoader(patient_name, view)
    visualization = Visualization()
    Zspec = dataLoader.get_Zspec_by_coordinate(nth_slice, x, y)
    offset, Zspec = dataLoader.Zspec_preprocess_onepixel(Zspec)
    
    simulation = Simulation()
    paras = [11.843, 2.686, 14.682, 2.575e-6]
    Zspec_sim = simulation.simulate(paras, dataLoader.read_T1_map()[nth_slice-1, y-1, x-1],
                                   dataLoader.read_T2_map()[nth_slice-1, y-1, x-1])
    visualization.plot_2_Zspec(offset, Zspec, Zspec_sim, labels=["real", "simulated"])
    visualization.plot_2_Zspec(offset[7:], Zspec[7:], Zspec_sim[7:], labels=["real", "simulated"])
    visualization.plot_Zspec_diff(offset[7:], Zspec_sim[7:]-Zspec[7:])
    
    paras = [2.407, 0.6615, 12.409, 7.154e-6]
    Zspec_sim = simulation.simulate(paras, dataLoader.read_T1_map()[nth_slice-1, y-1, x-1],
                                   dataLoader.read_T2_map()[nth_slice-1, y-1, x-1])
    visualization.plot_2_Zspec(offset, Zspec, Zspec_sim, labels=["real", "simulated"])
    visualization.plot_2_Zspec(offset[7:], Zspec[7:], Zspec_sim[7:], labels=["real", "simulated"])
    visualization.plot_Zspec_diff(offset[7:], Zspec_sim[7:]-Zspec[7:])

def compare_A(patient_name, view):
    dataLoader = DataLoader(patient_name, view)
    visualization = Visualization()
    offset = np.sort(dataLoader.EMR_offset_55_unique)[::-1]
    print(offset)
    simulation = Simulation()
    paras = [11.843, 2.686, 14.682, 2.575e-6]
    Zspec_1 = simulation.simulate(offset, paras)
    paras = [2.407, 0.6615, 12.409, 7.154e-6]
    Zspec_2 = simulation.simulate(offset, paras)
    visualization.plot_2_Zspec(offset[0:28], Zspec_1[0:28], Zspec_2[0:28], labels=["MTC", "EMR"])
    visualization.plot_2_Zspec(offset[8:28], Zspec_1[8:28], Zspec_2[8:28], labels=["MTC", "EMR"])

def compare_B(patient_name, view):
    dataLoader = DataLoader(patient_name, view)
    visualization = Visualization()
    # offset = np.sort(dataLoader.EMR_offset_55_unique)[::-1]
    offset = np.array([120, 110, 100, 90, 80, 70, 60, 50, 40, 30, 20, 12, 8,
              5, 4.75, 4.5, 4.25, 4, 3.75, 3.5, 3.25, 3, 2.75, 2.5,
              2.25, 2, 1.75, 1.5, 1.25, 1, 0.75, 0.5, 0.25, 0])
    # offset = np.arange(200, 0, -10)
    print(offset)
    simulation = Simulation()
    paras = [1.63, 0.58, 15.67, 7.63e-6]
    Zspec_1 = simulation.simulate(offset, paras, lineshape="SL")
    paras = [19.16, 3.08, 18.14, 19.37e-6]
    Zspec_2 = simulation.simulate(offset, paras, lineshape="L")
    paras = [50, 5, 17.84, 21e-6]
    Zspec_3 = simulation.simulate(offset, paras, lineshape="G")
    # paras = [1.63, 0.58, 15.67, 7.63e-6]
    # Zspec_4 = simulation.simulate(offset, paras)
    # visualization.plot_sim_4(offset[0:28], Zspec_1[0:28], Zspec_2[0:28], Zspec_3[0:28], Zspec_4[0:28])
    # visualization.plot_sim_4(offset[8:28], Zspec_1[8:28], Zspec_2[8:28], Zspec_3[8:28], Zspec_4[8:28])
    visualization.plot_sim_3(offset, Zspec_1, Zspec_2, Zspec_3, 
                             labels=["SL","L","G"])
    visualization.plot_sim_3(offset[13:], Zspec_1[13:], Zspec_2[13:], Zspec_3[13:], 
                             labels=["SL","L","G"])

# [R, R_M0m_T1w, T1w_T2w, T2m]
# paras = np.array([[10, 1, 40, 8e-6],
#                   [50, 5, 40, 8e-6],
#                   [500, 50, 40, 8e-6],
#                   [5000, 500, 40, 8e-6],
#                   [50000, 5000, 40, 8e-6]])

# obj = Simulation(B1=1.5)
# obj.simulate([50, 5, 40, 8e-6])
# obj.simulate([20, 2, 10, 8e-6])
# obj.evaluate_R()


# Visualization.plot_2_Zspec(obj.wide_freq/128, Zspec_1_wide, Zspec_2_wide)
# Visualization.plot_2_Zspec(obj.middle_freq/128, Zspec_1_middle, Zspec_2_middle)
# obj.load_patient("APT_479")

# plt.scatter(wide_freq/128, Zspec_wide, label='Zspec_wide', color='blue') 
# plt.scatter(middle_freq/128, Zspec_middle, label='Zspec_middle', color='green') 
# plt.scatter(frequencies / (B0*42.576), Zspec_2, label='Zspec_2', color='yellow') 
# plt.scatter(frequencies / (B0*42.576), Zspec_3, label='Zspec_3', color='purple') 
# plt.scatter(frequencies / (B0*42.576), Zspec_4, label='Zspec_4', color='black') 
 
# Zspec = get_avg_Zspec(r"C:\Users\jwu191\Desktop\EMR fitting\Zspec_1.5uT_56offsets\20220720_JG_normal_Zspec.csv")
# plt.scatter(frequencies / (B0*42.576), Zspec/100, label='case', color='red')
# plt.title("20220720_JG_normal")   
# plt.legend()  
# plt.show()
   
        
        
        