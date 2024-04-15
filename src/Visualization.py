import os

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import SimpleITK as sitk

class Visualization:
    
    def __init__(self):
        pass
    
    def plot(self, x, y, label="", title=""):
        plt.scatter(x, y, label=label, color="blue", marker="x")
        plt.plot(x, y, label=label, color="blue")
        plt.gca().invert_xaxis()  # invert x-axis
        plt.legend()
        plt.title(title)
        plt.show()
    
    def plot_Zspec(self, offset, Zspec, label="Zspec", title=""):
        """
        Plot a single Zspec given the offsets
        Zspec must be normalized to [0, 1]
        """
        plt.ylim((0, 1))
        plt.scatter(offset, Zspec, label=label, color="blue", marker="x")
        plt.plot(offset, Zspec, label=label, color="blue")
        plt.gca().invert_xaxis()  # invert x-axis
        plt.legend()
        plt.title(title)
        plt.show()
    
    def plot_2_Zspec(self, offset, Zspec_1, Zspec_2, labels=["Zspec_1","Zspec_2"], title="", highlight=[3.5, -3.5]):
        scale = 100
        plt.ylim((0, scale))
        plt.scatter(offset, scale*Zspec_1, label=labels[0], color='blue', marker="x")
        # plt.plot(offset, scale*Zspec_1, label=labels[0], color='blue')
        plt.plot(offset, scale*Zspec_2, label=labels[1], color='orange')
        # print(Zspec_1)
        for h in highlight:
            index = np.where(offset==h)
            plt.scatter(offset[index], scale*Zspec_1[index], color='red', marker="x")
        plt.gca().invert_xaxis()  # invert x-axis
        plt.legend()
        plt.xlabel("Frequency offset (ppm)")
        plt.ylabel("Msat/M0 (%)")
        plt.title(title)
        
        plt.axvline(x=3.5, ymin=Zspec_1[np.where(offset==3.5)], ymax=Zspec_2[np.where(offset==3.5)], 
                    color="red", linestyle="--")
        plt.axvline(x=-3.5, ymin=Zspec_1[np.where(offset==-3.5)], ymax=Zspec_2[np.where(offset==-3.5)],
                    color="red", linestyle="--")
        plt.show() 
            
    def plot_multi_Zspec(self, offset, Zspecs, labels=None, title="", highlight=None):
        plt.ylim((0, 1))
        colors = ["black", "purple", "blue", "green", "yellow", "orange", "red"]
        Zspecs = np.array(Zspecs) # shape: [num of Zspecs, num of offsets]
        for i in range(Zspecs.shape[0]):
            plt.scatter(offset, Zspecs[i], label=labels[i], color=colors[i])
        # # index = np.where(x==3.5)
        # # plt.scatter(x[index], y_estimated[index], color='red')
        plt.gca().invert_xaxis()  # invert x-axis
        plt.legend()
        plt.title(title)
        plt.show() 
        
    def plot_Zspec_diff(self, offset, Zspec, title="", highlight=[3.5, -3.5]):
        scale = 100
        plt.ylim((-0.1*scale, 0.2*scale))
        plt.scatter(offset, scale*Zspec, color="blue", marker="x")
        plt.plot(offset, scale*Zspec, color='blue')
        # for h in highlight:
        #     index = np.where(offset==h)
        #     plt.scatter(offset[index], scale*Zspec[index], color='red', marker="x")
        plt.gca().invert_xaxis()  # invert x-axis
        plt.xlabel("Frequency offset (ppm)")
        plt.ylabel("Z_fitted - Z_real (%)")
        plt.title(title)
        plt.axvline(x=3.5, color="red", linestyle="--")
        plt.axvline(x=-3.5, color="red", linestyle="--")
        plt.show() 
        
    def plot_Zspec_all_power(self, offset, Zspec_1, Zspec_2, Zspec_3, Zspec_4, 
                            labels=["Zspec_1","Zspec_2","Zspec_3","Zspec_4"], title=""):
        plt.ylim((0, 1))
        plt.scatter(offset, Zspec_1, label=labels[0], color='blue', marker="x")
        plt.scatter(offset, Zspec_2, label=labels[1], color='green', marker="x")
        plt.scatter(offset, Zspec_3, label=labels[2], color='orange', marker="x")
        plt.scatter(offset, Zspec_4, label=labels[3], color='red', marker="x")
        plt.gca().invert_xaxis()  # invert x-axis
        plt.legend()
        plt.title(title)
        plt.show()
        
    def plot_diff_all_power(self, offset, Zspec_1, Zspec_2, Zspec_3, Zspec_4, 
                            labels=["Zspec_1","Zspec_2","Zspec_3","Zspec_4"], title=""):
        plt.ylim((-0.5, 0.5))
        plt.scatter(offset, Zspec_1, label=labels[0], color='blue')
        plt.scatter(offset, Zspec_2, label=labels[1], color='green')
        plt.scatter(offset, Zspec_3, label=labels[2], color='orange')
        plt.scatter(offset, Zspec_4, label=labels[3], color='red')
        plt.gca().invert_xaxis()  # invert x-axis
        plt.legend()
        plt.title(title)
        plt.show() 
    
    def view_24(self, offset, Zspec, y_estimated):
        self.plot_2_Zspec(offset, Zspec, y_estimated, labels=["real", "fitted"])
        self.plot_2_Zspec(offset[7:], Zspec[7:], y_estimated[7:], labels=["real", "fitted"])
        self.plot_Zspec_diff(offset[7:], y_estimated[7:]-Zspec[7:])
    
    
    
    def _lorentz(self, x, amp, pos, width):
        factor = ((x - pos) / width) * ((x - pos) / width)
        return amp / (1 + 4*factor)
    
    def plot_component_2pool(self, offset, paras, title=""):
        plt.ylim((0, 1))
        mt = []
        ds = []
        for x in offset:
            mt.append(self._lorentz(x, *paras[0:3]))
            ds.append(self._lorentz(x, *paras[3:]))
        plt.scatter(offset, np.array(mt), label="MT", color="purple")
        plt.scatter(offset, np.array(ds), label="DS", color="blue")
        plt.gca().invert_xaxis()  # invert x-axis
        plt.legend()
        plt.title(title)
        plt.show()
        
    def plot_component_5pool(self, offset, paras, title=""):
        plt.ylim((0, 1))
        noe = []
        mt = []
        ds = []
        cest = []
        apt = []
        for x in offset:
            noe.append(self._lorentz(x, *paras[0:3]))
            mt.append(self._lorentz(x, *paras[3:6]))
            ds.append(self._lorentz(x, *paras[6:9]))
            cest.append(self._lorentz(x, *paras[9:12]))
            apt.append(self._lorentz(x, *paras[12:]))
        plt.scatter(offset, np.array(noe), label="NOE", color="green")
        plt.scatter(offset, np.array(mt), label="MT", color="purple")
        plt.scatter(offset, np.array(ds), label="DS", color="blue")
        plt.scatter(offset, np.array(cest), label="CEST", color="yellow")
        plt.scatter(offset, np.array(apt), label="APT", color="red")
        plt.gca().invert_xaxis()  # invert x-axis
        plt.legend()
        plt.title(title)
        plt.show()       
    
    def format_paras(self, paras):
         print("********", "formatted results", "********")
         if paras.shape[0] == 6:
             print("MT-amplitude:", format(100*paras[0], ".3f"), '%',
                   "MT-center:", format(paras[0], ".3f"),
                   "MT-width:", format(paras[2], ".3f"))
             print("DS-amplitude:", format(100*paras[3], ".3f"), "%", 
                   "DS-center:", format(paras[3], ".3f"),
                   "DS-width:", format(paras[3], ".3f"))
         elif paras.shape[0] == 15:
             print("NOE-amplitude:", format(100*paras[0], ".3f"), '%',
                   "NOE-center:", format(paras[1], ".3f"),
                   "NOE-width:", format(paras[2], ".3f"))  
             print("MT-amplitude:", format(100*paras[3], ".3f"), "%",
                   "MT-center:", format(paras[4], ".3f"),
                   "MT-width:", format(paras[5], ".3f"))
             print("DS-amplitude:", format(100*paras[6], ".3f"), "%",
                   "DS-center:", format(paras[7], ".3f"),
                   "DS-width:", format(paras[8], ".3f")) 
             print("CEST-amplitude:", format(100*paras[9], ".3f"), "%",
                   "CEST-center:", format(paras[10], ".3f"), 
                   "CEST-width:", format(paras[11], ".3f"))  
             print("APT-amplitude:", format(100*paras[12], ".3f"), "%",
                   "APT-center:", format(paras[13], ".3f"), 
                   "APT-width:", format(paras[14], ".3f"))  
    
        
        
    def plot_APT(self, offset, y, title=""):
        plt.plot(offset, y)
        plt.scatter(offset, y)
        plt.gca().invert_xaxis()  # invert x-axis
        plt.title(title)
        plt.show()
        
    def show_img_2D(self, img):
        plt.imshow(img, cmap="gray")
        plt.show()
        
    def show_img_pos_2D(self, img, x, y):
        print(np.min(img), np.max(img))
        color_img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        plt.imshow(color_img, cmap="gray")
        plt.show()
    
    def plot_sim_2(self, offset, Zspec_1, Zspec_2, labels=["1","2"], title=""):
        scale = 100
        plt.ylim((0, scale))
        plt.plot(offset, scale*Zspec_1, label=labels[0], color='blue')
        plt.plot(offset, scale*Zspec_2, label=labels[1], color='orange')
        plt.gca().invert_xaxis()  # invert x-axis
        plt.legend()
        plt.xlabel("Frequency offset (ppm)")
        plt.ylabel("Msat/M0 (%)")
        plt.title(title)
        plt.show() 
 
    def plot_sim_3(self, offset, Zspec_1, Zspec_2, Zspec_3, labels=["1","2","3"], title=""):
        scale = 100
        plt.ylim((0, scale))
        plt.plot(offset, scale*Zspec_1, label=labels[0], color='blue')
        plt.plot(offset, scale*Zspec_2, label=labels[1], color='green')
        plt.plot(offset, scale*Zspec_3, label=labels[2], color='orange')
        plt.gca().invert_xaxis()  # invert x-axis
        plt.legend()
        plt.xlabel("Frequency offset (ppm)")
        plt.ylabel("Msat/M0 (%)")
        plt.title(title)
        plt.show() 

    def plot_sim_4(self, offset, Zspec_1, Zspec_2, Zspec_3, Zspec_4, labels=["1","2","3","4"], title=""):
        scale = 100
        plt.ylim((0, scale))
        plt.plot(offset, scale*Zspec_1, label=labels[0], color='blue')
        plt.plot(offset, scale*Zspec_2, label=labels[1], color='green')
        plt.plot(offset, scale*Zspec_3, label=labels[2], color='orange')
        plt.plot(offset, scale*Zspec_4, label=labels[3], color='red')
        plt.gca().invert_xaxis()  # invert x-axis
        plt.legend()
        plt.xlabel("Frequency offset (ppm)")
        plt.ylabel("Msat/M0 (%)")
        plt.title(title)
        plt.show()
        
    






        
        