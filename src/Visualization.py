import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import SimpleITK as sitk

from funcs import lorentz_func

class Visualization:
    def __init__(self):
        pass

    def plot(self, x, y, label="", title=""):
        plt.scatter(x, y, label=label, color="blue", marker="x")
        # plt.plot(x, y, label=label, color="blue")
        plt.gca().invert_xaxis()  # invert x-axis
        plt.legend()
        plt.title(title)
        plt.show()

    def plot_Zspec(self, offset, Zspec, label="Zspec", title=""):
        """
        Plot a single Zspec given the offsets
        Zspec must be normalized to [0, 1]
        """
        scale = 100
        plt.ylim((0, scale))
        p = np.where(offset > 0)
        n = np.where(offset < 0)
        plt.scatter(offset, scale*Zspec, label=label, color="blue", marker="x")
        if np.sum(offset == 0) > 0:
            plt.plot(offset, scale*Zspec, color="blue")
        else:
            plt.plot(offset[p], scale*Zspec[p], color="blue")
            plt.plot(offset[n], scale*Zspec[n], color="blue")
        plt.gca().invert_xaxis()  # invert x-axis
        plt.legend()
        plt.title(title)
        plt.show()

    def plot_Zspec_2(self, offset, Zspec_1, Zspec_2, labels=["",""], title=""):
        """
        Plot a 2 Zspecs given the offsets
        Zspecs must be normalized to [0, 1]
        """
        scale = 100
        plt.ylim((0, scale))
        p = np.where(offset > 0)
        n = np.where(offset < 0)
        plt.scatter(offset, scale*Zspec_1, label=labels[0], color="blue", marker="x")
        plt.scatter(offset, scale*Zspec_2, label=labels[1], color="red", marker="x")
        if np.sum(offset == 0) > 0:
            plt.plot(offset, scale*Zspec_1, color="blue")
            plt.plot(offset, scale*Zspec_2, color="red")
        else:
            plt.plot(offset[p], scale*Zspec_1[p], color="blue")
            plt.plot(offset[n], scale*Zspec_1[n], color="blue")
            plt.plot(offset[p], scale*Zspec_2[p], color="red")
            plt.plot(offset[n], scale*Zspec_2[n], color="red")
        plt.gca().invert_xaxis()  # invert x-axis
        plt.legend()
        plt.title(title)
        plt.show()
    
    def plot_fitting_1_1(self, offset, Zspec, Zspec_fitted, labels=["Zspec","fitted"], title="", highlight=[]):
        """
        Plot Zspec and fitted Zspec given the offsets
        Zspecs must be normalized to [0, 1]
        """
        scale = 100
        plt.ylim((0, scale))
        p = np.where(offset > 0)
        n = np.where(offset < 0)
        plt.scatter(offset, scale*Zspec, label=labels[0], color='blue', marker="x")
        if np.sum(offset == 0) > 0:
            plt.plot(offset, scale*Zspec_fitted, label=labels[1], color='orange')
        else:
            plt.plot(offset[p], scale*Zspec_fitted[p], label=labels[1], color='orange')
            plt.plot(offset[n], scale*Zspec_fitted[n], color='orange')
        plt.gca().invert_xaxis()  # invert x-axis
        plt.legend()
        plt.xlabel("Frequency offset (ppm)")
        plt.ylabel("Msat/M0 (%)")
        plt.title(title)
        plt.show() 

    def plot_fitting_1_2(self, offset, Zspec, Zspec_fitted_1, Zspec_fitted_2, 
                         labels=["Zspec","fitted_1","fitted_2"], title="", highlight=[]):
        """
        Plot Zspec and 2 fitted Zspecs given the offsets
        Zspecs must be normalized to [0, 1]
        """
        scale = 100
        plt.ylim((0, scale))
        p = np.where(offset > 0)
        n = np.where(offset < 0)
        plt.scatter(offset, scale*Zspec, label=labels[0], color='black', marker="x")
        colors = ["blue", "red"]
        if np.sum(offset == 0) > 0:
            plt.plot(offset, scale*Zspec_fitted_1, label=labels[1], color=colors[0])
            plt.plot(offset, scale*Zspec_fitted_2, label=labels[2], color=colors[1])
        else:
            plt.plot(offset[p], scale*Zspec_fitted_1[p], label=labels[1], color=colors[0])
            plt.plot(offset[n], scale*Zspec_fitted_1[n], color=colors[0])
            plt.plot(offset[p], scale*Zspec_fitted_2[p], label=labels[2], color=colors[1])
            plt.plot(offset[n], scale*Zspec_fitted_2[n], color=colors[1])
        plt.gca().invert_xaxis()  # invert x-axis
        plt.legend()
        plt.xlabel("Frequency offset (ppm)")
        plt.ylabel("Msat/M0 (%)")
        plt.title(title)
        plt.show() 

    def plot_3_Zspec(self, offset, Zspec, Zspec_1, Zspec_2, labels=["","",""], title=""):
        if np.max(Zspec) < 1:
            Zspec *= 100
            Zspec_1 *= 100
            Zspec_2 *= 100
        # plt.ylim((0, scale))
        p = np.where(offset > 0)
        n = np.where(offset < 0)
        plt.scatter(offset, Zspec, label=labels[0], color='blue', marker="x")
        if np.sum(offset == 0) > 0:
            plt.plot(offset, Zspec_1, label=labels[1], color='green')
            plt.plot(offset, Zspec_2, label=labels[2], color='orange')
        else:
            plt.plot(offset[p], Zspec_1[p], label=labels[1], color='green')
            plt.plot(offset[n], Zspec_1[n], color='green')
            plt.plot(offset[p], Zspec_2[p], label=labels[1], color='orange')
            plt.plot(offset[n], Zspec_2[n], color='orange')
        plt.gca().invert_xaxis()  # invert x-axis
        plt.legend()
        plt.xlabel("Frequency offset (ppm)")
        plt.ylabel("Msat/M0 (%)")
        plt.title(title)
        plt.show() 

    def plot_4_Zspec(self, offset, Zspec, Zspec_1, Zspec_2, Zspec_3, labels=["","","",""], title=""):
        if np.max(Zspec) < 1:
            Zspec *= 100
            Zspec_1 *= 100
            Zspec_2 *= 100
            Zspec_3 *= 100
        # plt.ylim((0, scale))
        p = np.where(offset > 0)
        n = np.where(offset < 0)
        plt.scatter(offset, Zspec, label=labels[0], color='blue', marker="x")
        if np.sum(offset == 0) > 0:
            plt.plot(offset, Zspec_1, label=labels[1], color='green')
            plt.plot(offset, Zspec_2, label=labels[2], color='orange')
            plt.plot(offset, Zspec_3, label=labels[3], color='red')
        else:
            plt.plot(offset[p], Zspec_1[p], label=labels[1], color='green')
            plt.plot(offset[n], Zspec_1[n], color='green')
            plt.plot(offset[p], Zspec_2[p], label=labels[1], color='orange')
            plt.plot(offset[n], Zspec_2[n], color='orange')
            plt.plot(offset[p], Zspec_3[p], label=labels[1], color='red')
            plt.plot(offset[n], Zspec_3[n], color='red')
        plt.gca().invert_xaxis()  # invert x-axis
        plt.legend()
        plt.xlabel("Frequency offset (ppm)")
        plt.ylabel("Msat/M0 (%)")
        plt.title(title)
        plt.show() 

    def plot_Zspec_diff(self, offset, diff, title="", highlight=[3.5, -3.5]):
        """
        Plot diff (Zspec_fitted - Zspec) given the offsets
        Zspecs must be normalized to [0, 1] before taking the subtraction
        """
        scale = 100
        plt.ylim((-4, 8))
        p = np.where(offset > 0)
        n = np.where(offset < 0)
        if np.sum(offset == 0) > 0:
            plt.plot(offset, scale*diff, color='blue')
        else:
            plt.plot(offset[p], scale*diff[p], color='blue')
            plt.plot(offset[n], scale*diff[n], color='blue')
        plt.hlines(0, np.min(offset), np.max(offset), color="black", linestyles="--")
        plt.gca().invert_xaxis()  # invert x-axis
        plt.xlabel("Frequency offset (ppm)")
        plt.ylabel("Z_fitted - Z_real (%)")
        plt.title(title)
        plt.show() 

    def plot_Zspec_diff_2(self, offset, diff_1, diff_2, std_1=None, std_2=None,
                          labels=["",""], title="", lim=[-4, 8]):
        scale = 100
        plt.ylim(lim)
        colors = ["blue", "red"]
        p = np.where(offset > 0)
        n = np.where(offset < 0)
        if np.sum(offset == 0) > 0:
            plt.plot(offset, scale*diff_1, label=labels[0], color=colors[0])
            plt.plot(offset, scale*diff_2, label=labels[1], color=colors[1])
        else:
            if not std_1.any():
                plt.plot(offset[p], scale*diff_1[p], label=labels[0], color=colors[0])
                plt.plot(offset[n], scale*diff_1[n], color=colors[0])
                plt.plot(offset[p], scale*diff_2[p], label=labels[1], color=colors[1])
                plt.plot(offset[n], scale*diff_2[n], color=colors[1])
            else:
                plt.errorbar(offset[p], scale*diff_1[p], yerr=scale*std_1[p], 
                             label=labels[0], color=colors[0], capsize=2)
                plt.errorbar(offset[n], scale*diff_1[n], yerr=scale*std_1[n], 
                             color=colors[0], capsize=2)
                plt.errorbar(offset[p], scale*diff_2[p], yerr=scale*std_2[p], 
                             label=labels[1], color=colors[1], capsize=2)
                plt.errorbar(offset[n], scale*diff_2[n], yerr=scale*std_2[n], 
                             color=colors[1], capsize=2)
        plt.vlines(3.5, *lim, color="black", linestyles="--")
        plt.vlines(-3.5, *lim, color="black", linestyles="--")
        plt.hlines(0, np.min(offset), np.max(offset), color="black", linestyles="--")
        plt.gca().invert_xaxis()  # invert x-axis
        plt.legend()
        plt.xlabel("Frequency offset (ppm)")
        plt.ylabel("Z_fitted - Z_real (%)")
        plt.title(title)
        plt.show() 

    def plot_Zspec_diff_3(self, offset, diff_1, diff_2, diff_3, labels=["","",""], title=""):
        scale = 100
        plt.ylim((-4, 8))
        colors = ["blue", "orange", "red"]
        p = np.where(offset > 0)
        n = np.where(offset < 0)
        if np.sum(offset == 0) > 0:
            plt.plot(offset, scale*diff_1, label=labels[0], color=colors[0])
            plt.plot(offset, scale*diff_2, label=labels[1], color=colors[1])
            plt.plot(offset, scale*diff_3, label=labels[2], color=colors[2])
        else:
            plt.plot(offset[p], scale*diff_1[p], label=labels[0], color=colors[0])
            plt.plot(offset[n], scale*diff_1[n], label=labels[0], color=colors[0])
            plt.plot(offset[p], scale*diff_2[p], label=labels[1], color=colors[1])
            plt.plot(offset[n], scale*diff_2[n], label=labels[1], color=colors[1])
            plt.plot(offset[p], scale*diff_3[p], label=labels[2], color=colors[2])
            plt.plot(offset[n], scale*diff_3[n], label=labels[2], color=colors[2])
        plt.hlines(0, np.min(offset), np.max(offset), color="black", linestyles="--")
        plt.gca().invert_xaxis()  # invert x-axis
        plt.legend()
        plt.xlabel("Frequency offset (ppm)")
        plt.ylabel("Z_fitted - Z_real (%)")
        plt.title(title)
        plt.show()

    def plot_Zspec_diff_4(self, offset, diff_1, diff_2, diff_3, diff_4,
                          labels=["","","",""], title=""):
        scale = 100
        low, up = -4, 8
        plt.ylim((low, up))
        colors = ["blue", "green", "orange", "red"]
        p = np.where(offset > 0)
        n = np.where(offset < 0)
        if np.sum(offset == 0) > 0:
            plt.plot(offset, scale*diff_1, label=labels[0], color=colors[0])
            plt.plot(offset, scale*diff_2, label=labels[1], color=colors[1])
            plt.plot(offset, scale*diff_3, label=labels[2], color=colors[2])
            plt.plot(offset, scale*diff_4, label=labels[3], color=colors[3])
        else:
            plt.plot(offset[p], scale*diff_1[p], label=labels[0], color=colors[0])
            plt.plot(offset[n], scale*diff_1[n], color=colors[0])
            plt.plot(offset[p], scale*diff_2[p], label=labels[1], color=colors[1])
            plt.plot(offset[n], scale*diff_2[n], color=colors[1])
            plt.plot(offset[p], scale*diff_3[p], label=labels[2], color=colors[2])
            plt.plot(offset[n], scale*diff_3[n], color=colors[2])
            plt.plot(offset[p], scale*diff_4[p], label=labels[3], color=colors[3])
            plt.plot(offset[n], scale*diff_4[n], color=colors[3])
        plt.vlines(3.5, low, up, color="black", linestyles="--")
        plt.vlines(-3.5, low, up, color="black", linestyles="--")
        plt.hlines(0, np.min(offset), np.max(offset), color="black", linestyles="--")
        plt.gca().invert_xaxis()  # invert x-axis
        plt.legend()
        plt.xlabel("Frequency offset (ppm)")
        plt.ylabel("Z_fitted - Z_real (%)")
        plt.title(title)
        plt.show()

    def plot_MTRasym_2(self, offset, MTRasym_1, MTRasym_2, labels=["",""], title=""):
        scale = 100
        colors = ["blue", "red"]
        plt.plot(offset, scale*MTRasym_1, color=colors[0], label=labels[0])
        plt.plot(offset, scale*MTRasym_2, color=colors[1], label=labels[1])
        plt.gca().invert_xaxis()  # invert x-axis
        plt.legend()
        plt.xlabel("Frequency offset (ppm)")
        plt.ylabel("MTR assymmetry (%)")
        plt.title(title)
        plt.show()

    def plot_Lorentz_5pool(self, offset, paras, title=""):
        scale = 100
        APT = []
        CEST2ppm = []
        DS = []
        MT = []
        NOE = []
        for x in offset:
            APT.append(scale*lorentz_func(x, *paras[0:3]))
            CEST2ppm.append(scale*lorentz_func(x, *paras[3:6]))
            DS.append(scale*lorentz_func(x, *paras[6:9]))
            MT.append(scale*lorentz_func(x, *paras[9:12]))
            NOE.append(scale*lorentz_func(x, *paras[12:15]))
        plt.plot(offset, APT, color="red", label="APT")
        plt.plot(offset, CEST2ppm, color="yellow", label="CEST@2ppm")
        plt.plot(offset, DS, color="blue", label="DS")
        plt.plot(offset, MT, color="purple", label="MT")
        plt.plot(offset, NOE, color="green", label="NOE")
        plt.gca().invert_xaxis()  # invert x-axis
        plt.legend()
        plt.xlabel("Frequency offset (ppm)")
        plt.title(title)
        plt.show()


    def check_B0_corr(self, offset, raw, corrected):
        scale = 100
        plt.ylim((0, scale))
        plt.scatter(offset, scale*raw, label="raw", color='blue', marker="x")
        plt.scatter(offset, scale*corrected, label="corrected", color='red', marker="x")
        plt.gca().invert_xaxis()  # invert x-axis
        plt.legend()
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
    
    
    
 
        
        