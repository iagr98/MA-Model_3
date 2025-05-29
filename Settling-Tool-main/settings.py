# -*- coding: utf-8 -*-
"""
@author: luth01
"""
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt

from math import isnan


class Settling_Experiment():
    
    
    def __init__(self):

        #Experimental Parameter
        self.g = 9.81 # acceleration due to gravity [m/s^2]
        self.H_cd = 1e-20 # Hamaker coefficient [Nm] 
        self.H_0 = 0.2 # Hight of the liquids in the settling tank
        self.H_sep_plane = 0 # Hight to seperation plane [m]
        self.t_e = 0 # time when seperation of the phases is finished
        self.t_0 = 0 # time when the stirrer is turned off / offset [s]
        
        #Substance parameters
        self.rho_conti = 0 # Density of conti phase [kg/m³]
        self.rho_disp = 0 # Density of dispers phase [kg/m³]
        self.sigma = 0 # Surface tension [N/m]
        self.eta_conti = 0 # Viscosity conti phase [Pas]
        self.eta_disp = 0 # Viscosity disp phase [Pas]
        self.eta_surf = 0 # Correction viscosity with surface activae substance [Pas]
        
        #Nuermical Parameter
        self.N_t = 500 # Number of time interval
        self.N_h = 500 # Number of hight Elements
        self.epsilon_di = 1 # Hold-up 
        
        #Data
        self.time_exp = []
        self.h_c_exp = []
        self.h_d_exp = []
        
        #Calculated Parameter
        self.epsilon_0 = 0
        self.delta_rho = 0
    
    
          
        
    def update(self,excel_file):
        
        import_data = pd.read_excel(os.path.join('Input', excel_file), 'Parameters', index_col=0)
        
        #Update experimental parameter
        self.H_0 = import_data["H_0"]["Value"] 
        self.H_sep_plane = import_data["H_sep_plane"]["Value"] 
        self.t_e = import_data["t_E"]["Value"]
        self.rho_conti = import_data["ρ_c"]["Value"]
        self.rho_disp = import_data["ρ_d"]["Value"]
        self.sigma = import_data["σ"]["Value"]
        self.eta_conti = import_data["η_c"]["Value"]
        self.eta_disp = import_data["η_d"]["Value"]
        self.eta_surf = import_data["η_v"]["Value"]
        self.t_0 = import_data["t_0"]["Value"]
        
        import_data = pd.read_excel(os.path.join('Input', excel_file), 'Data')
        
        #Update results experiment
        self.time_exp = import_data["t"]
        self.h_c_exp = import_data["h_c"]
        self.h_d_exp = import_data["h_d"]
        
        #update calculated Parameter
        h_c_exp_check = next (x for x in [float(x) for x in list(self.h_c_exp)] if not isnan(x))
        h_d_exp_check = next (x for x in [float(x) for x in list(self.h_d_exp)] if not isnan(x))
        
        if h_c_exp_check < h_d_exp_check:                   # if-Bedingung unterscheidet zwischen o in w und w in o
            self.epsilon_0 = (1-self.H_sep_plane/self.H_0)
        else:
            self.epsilon_0 = self.H_sep_plane/self.H_0
        
        self.delta_rho = abs(self.rho_conti - self.rho_disp)
        
        print("Updated Parameters with Excel")
        
        
    def plot_experiment(self):
        plt.figure(figsize=(8,6))
        plt.scatter( self.time_exp,self.h_c_exp , label = 'Conti Phase',
                    marker ='o', s = 50, color = 'b')
        plt.scatter(self.time_exp,self.h_d_exp, label= 'Disp Phase', 
                    marker = 'o',color='r',s=50)
        
        plt.title('Settling Experiment - Data Visulization',  size = 15)
        plt.xlabel('Time / s', size = 15)
        plt.ylabel('Hight / m', size = 15)
        plt.legend( fontsize="20")
        plt.grid(True)
        plt.show()