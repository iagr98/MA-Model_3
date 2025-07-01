import sim_model as sm
import sim_parameters as sp
import numpy as np
import pandas as pd
import helper_functions as hf

def init_sim(exp, phi_0, dV_ges, eps_0, h_c_0, h_dis_0, N_x):
    if (exp == "ye"):
        filename = "Paraffin_flut_20C.xlsx"
        Set = sp.Settings(N_x=N_x, L=0.56, D=0.15, h_c_0=0.055, h_dis_0=0.04)
    elif(exp == "niba1" or exp == "niba2" or exp == "niba3" or exp == "niba4"):
        Set = sp.Settings(N_x=N_x, L=1.0, D=0.2, h_c_0=0.1, h_dis_0=0.03)
        filename = "niba_V1.xlsx" if exp == "niba1" else \
        "niba_V2.xlsx" if exp == "niba2" else \
        "niba_V3.xlsx" if exp == "niba3" else \
        "niba_V4.xlsx" if exp == "niba4" else None
    elif(exp == "2mmol_21C" or exp == "2mmol_30C" or exp == "5mmol_30C" or exp == "10mmol_21C" or exp == "10mmol_30C" or exp == "15mmol_20C" or exp == "15mmol_30C"):
        Set = sp.Settings(N_x=N_x, L=1.3, D=0.2, h_c_0=h_c_0, h_dis_0=h_dis_0)
        filename = "2mmolNa2CO3_21C.xlsx" if exp == "2mmol_21C" else \
        "2mmolNa2CO3_30C.xlsx" if exp == "2mmol_30C" else \
        "5mmolNa2CO3_30C.xlsx" if exp == "5mmol_30C" else \
        "10mmolNa2CO3_21C.xlsx" if exp == "10mmol_21C" else \
        "10mmolNa2CO3_30C.xlsx" if exp == "10mmol_30C" else \
        "15mmolNa2CO3_20C.xlsx" if exp == "15mmol_20C" else \
        "15mmolNa2CO3_30C.xlsx" if exp == "15mmol_30C" else None
    else:
        print('Test does not belong to either Ye or Niba.')
    SubSys = sp.Substance_System()
    SubSys.update(filename)
    SubSys.phi_0 = phi_0
    SubSys.dV_ges = dV_ges / 3.6 * 1e-6
    SubSys.eps_0 = eps_0
    return sm.Simulation(Set, SubSys)

def init_sims(filename, numberSims):
    Sims = [None] * numberSims
    for i in range(numberSims):
        Sims[i] = init_sim(filename)
    return Sims

def comp_plots(filenames, labels, legend_title='title', title='title', figsize=(6, 4.5)):
    Sims = [None] * len(filenames)
    for i in range(len(filenames)):
        Sims[i] = init_sim(filenames[i])
        Sims[i].calcInitialConditions()
        Sims[i].simulate_upwind()
    sm.plot_comparison(Sims, labels=labels, legend_title=legend_title, title=title, figsize=figsize)

def calc_sensitivity(Sims, p):
    DeltaV1 = Sims[1].calc_Vdis_tot() - Sims[0].calc_Vdis_tot()
    print('+/-10% Sensitivity of parameter 1 is: ' + str(1000*DeltaV1) + ' mL')
    DeltaV2 = Sims[3].calc_Vdis_tot() - Sims[2].calc_Vdis_tot()
    print('+/-10% Sensitivity of parameter 2 is: ' + str(1000 * DeltaV2) + ' mL')
    Q = DeltaV1 / DeltaV2 * p[1] / p[0]
    print('sensitivity ratio between p1 and p2 is: ' + str(Q))
    print('-------------------------------------------')

def run_sim(exp="ye", phi_0=610e-6, dV_ges=240, eps_0=0.2, h_c_0=0.1, h_dis_0=0.05, N_x=201, a_tol=1e-7):
    Sim = init_sim(exp, phi_0, dV_ges, eps_0, h_c_0, h_dis_0, N_x)
    Sim.calcInitialConditions()
    Sim.simulate_ivp(veloConst=False, atol=a_tol)
    return Sim

# run

if __name__ == "__main__":
    # example more examples in file allExamples.py
    # some plotting functions may only work for Simulation with a Sim Object calculated with simulate_upwind() and not ivp!
    # filename = 'Paraffin_flut_20C.xlsx'
    # filename = 'niba_V1.xlsx'
    # filename = 'Hexan_1_1_o_in_w.xlsx'
    # filename = 'Butylacetat_5_6_220.xlsx'

    exp = "5mmol_30C"
    phi_0 = 610e-6
    dV_ges = 240
    eps_0 = 0.2
    h_c_0 = 0.1
    h_dis_0 = 0.05

    Sim = run_sim(exp, phi_0, dV_ges, eps_0, h_c_0, h_dis_0)

    # Sim.calcInitialConditions()
    # # #Sim.simulate_upwind(veloConst=True) # boolean velo defines whether u is assumed constant or not (default:True)
    # Sim.simulate_ivp(veloConst=True) # boolean velo defines whether u is assumed constant or not (default:True)
    # Sims = [Sim]
    # labels = []
    # # #sm.plot_comparison(Sims, title='Hexan o in w, o/w = 1/1, dV_ges=100 L/h')
    # # # Keys for animated Plots: 'velo', 'sauter', 'heights', 'tau'
    plots = ['heights', 'sauter']
    Sim.plot_anim(plots)