import sim_model as sm
import sim_parameters as sp
import numpy as np
import pandas as pd
import helper_functions as hf

def init_sim(filename, N_x=101):
    if (filename == "Paraffin_flut_20C.xlsx"):
        Set = sp.Settings(N_x=N_x, L=0.56, D=0.15, h_c_0=0.055, h_dis_0=0.04)
    elif(filename == "niba_V1.xlsx" or filename == "niba_V2.xlsx" or filename == "niba_V3.xlsx" or filename == "niba_V4.xlsx"):
        Set = sp.Settings(N_x=N_x, L=1.0, D=0.2, h_c_0=0.1, h_dis_0=0.03)
    else:
        print('Test does not belong to either Ye or Niba.')
    SubSys = sp.Substance_System()
    SubSys.update(filename)
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

def run_sim(filename, N_x=101, a_tol=1e-7):
    Sim = init_sim(filename, N_x)
    Sim.calcInitialConditions()
    Sim.simulate_ivp(veloConst=False, atol=a_tol)
    return Sim

# run

if __name__ == "__main__":
    # example more examples in file allExamples.py
    # some plotting functions may only work for Simulation with a Sim Object calculated with simulate_upwind() and not ivp!
    filename = 'Paraffin_flut_20C.xlsx'
    # filename = 'niba_V1.xlsx'
    # filename = 'Hexan_1_1_o_in_w.xlsx'
    # filename = 'Butylacetat_5_6_220.xlsx'

    N_x = 151
    a_tol = 1e-7
    Sim = run_sim(filename, N_x=N_x, a_tol=a_tol)

    # Sim.calcInitialConditions()
    # # #Sim.simulate_upwind(veloConst=True) # boolean velo defines whether u is assumed constant or not (default:True)
    # Sim.simulate_ivp(veloConst=True) # boolean velo defines whether u is assumed constant or not (default:True)
    # Sims = [Sim]
    # labels = []
    # # #sm.plot_comparison(Sims, title='Hexan o in w, o/w = 1/1, dV_ges=100 L/h')
    # # # Keys for animated Plots: 'velo', 'sauter', 'heights', 'tau'
    plots = ['heights', 'velo', 'sauter']
    Sim.plot_anim(plots)