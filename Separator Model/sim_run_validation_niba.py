import sim_model as sm
import sim_parameters as sp
import numpy as np
import pandas as pd
import helper_functions as hf

def init_sim(filename):
    Set = sp.Settings()
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

# run

# example more examples in file allExamples.py
# some plotting functions may only work for Simulation with a Sim Object calculated with simulate_upwind() and not ivp!

file_path = "c:\\Users\\Asus\\OneDrive\\Documentos\\RWTH\\Semester 4 (SS25)\\MA\\W9_Vergleichen_Vdis_sep.Effi\\Versuche AVT.FVT\\Vdis_niba.xlsx"
file_path_results = "c:\\Users\\Asus\\OneDrive\\Documentos\\RWTH\\Semester 4 (SS25)\\MA\W9_Vergleichen_Vdis_sep.Effi\\Versuche AVT.FVT\\Vdis_Model_3_results.xlsx"
filename = ['niba_V1.xlsx', 'niba_V2.xlsx', 'niba_V3.xlsx', 'niba_V4.xlsx']

for i in range(len(filename)):
    Sim = init_sim(filename[i])
    A = np.pi*(Sim.Set.D**2)/4
    if (i==0):
        sheet = "V1"
    elif(i==1):
        sheet = "V2"
    elif(i==2):
        sheet = "V3"
    elif(i==3):
        sheet = "V4"
    df_input = pd.read_excel(file_path, sheet_name=sheet) 
    phi_in0 = df_input['d_32in'].to_numpy()
    dV_ges = df_input['V_ges'].to_numpy()/(3.6 * 1e6)
    eps_0 = df_input['varphi'].to_numpy()
    V_dis = np.zeros(len(phi_in0))

    for j in range(len(phi_in0)):
        Sim.Sub.phi_0 = phi_in0[j]
        Sim.Sub.dV_ges = dV_ges[j]
        Sim.Sub.eps_0 = eps_0[j]
        Sim.calcInitialConditions()
        Sim.simulate_ivp(veloConst=True)
        V_dis[j] = Sim.V_dis_total
    df = pd.DataFrame({
        'd_32in': phi_in0,
        'V_ges': dV_ges * (3.6 * 1e6), # convert dV_ges into L/h
        'varphi': eps_0,
        'V_dis': V_dis
    })

    with pd.ExcelWriter(file_path_results, engine='openpyxl', mode='a') as writer:
        df.to_excel(writer, sheet_name=sheet, index=False)

# Sim.calcInitialConditions()
# # #Sim.simulate_upwind(veloConst=True) # boolean velo defines whether u is assumed constant or not (default:True)
# Sim.simulate_ivp(veloConst=True) # boolean velo defines whether u is assumed constant or not (default:True)
# Sims = [Sim]
# labels = []
# # #sm.plot_comparison(Sims, title='Hexan o in w, o/w = 1/1, dV_ges=100 L/h')
# # # Keys for animated Plots: 'velo', 'sauter', 'heights', 'tau'
# plots = ['heights']
# Sim.plot_anim(plots)