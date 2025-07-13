import numpy as np
import pandas as pd
import joblib
import helper_functions as hf
from sim_run import run_sim

N_CPU = 8

experiment = "sozh" # "main" if ye + niba tests, "sozh" tests from AVT.FVT

df = pd.read_excel("Input/data_main.xlsx", sheet_name=experiment)
exp = df['exp'].tolist()
phi_0 = df['phi_0'].tolist()
dV_ges = df['dV_ges'].tolist()
eps_0 = df['eps_0'].tolist()
if (experiment == "sozh"):
    h_c_0 = df['h_c_0'].tolist() 
    h_dis_0 = df['h_dis_max'].tolist()



def parallel_simulation(params):
    if (experiment == "main"):
        exp, phi_0, dV_ges, eps_0 = params
        print(f"Start simulation with exp={exp}, phi_0={phi_0}, dV_ges={dV_ges}, eps_0={eps_0}")
    elif(experiment == "sozh"):
        exp, phi_0, dV_ges, eps_0, h_c_0, h_dis_0 = params
        print(f"Start simulation with exp={exp}, phi_0={phi_0}, dV_ges={dV_ges}, eps_0={eps_0}, h_c_0={h_c_0}, h_dis_0={h_dis_0}")
    try:
        if (experiment == "main"):
            Sim = run_sim(exp, phi_0, dV_ges, eps_0)
            return {'exp': exp, 'phi_0': phi_0, 'dV_ges': dV_ges, 'eps_0': eps_0,
                'V_dis_total': Sim.V_dis_total,
                'Vol_imbalance [%]': hf.calculate_volume_balance(Sim), 'status': 'success'}
        elif (experiment == "sozh"):
            Sim = run_sim(exp, phi_0, dV_ges, eps_0, h_c_0, h_dis_0)
            return {'exp': exp, 'phi_0': phi_0, 'dV_ges': dV_ges, 'eps_0': eps_0,
                'h_c_0': h_c_0, 'h_dis_0': h_dis_0,
                'V_dis_total': Sim.V_dis_total,
                'Vol_imbalance [%]': hf.calculate_volume_balance(Sim), 'status': 'success'}
    except Exception as e:
        if (experiment == "main"):
            print(f"Simulation failed for exp={exp}, phi_0={phi_0}, dV_ges={dV_ges}, eps_0={eps_0}: {str(e)}")
            return {'exp': exp, 'phi_0': phi_0, 'dV_ges': dV_ges, 'eps_0': eps_0, 'error': str(e), 'status': 'failed'}
        elif(experiment == "sozh"):
            print(f"Simulation failed for exp={exp}, phi_0={phi_0}, dV_ges={dV_ges}, eps_0={eps_0}, h_c_0={h_c_0}, h_dis_0={h_dis_0}: {str(e)}")
            return {'exp': exp, 'phi_0': phi_0, 'dV_ges': dV_ges, 'eps_0': eps_0, 'h_c_0': h_c_0, 'h_dis_0': h_dis_0, 'error': str(e), 'status': 'failed'}

if __name__ == "__main__":

    if (experiment == "main"):
        parameters = [(exp[i], phi_0[i], dV_ges[i], eps_0[i]) for i in range(len(exp))]
    elif(experiment == "sozh"):
        parameters = [(exp[i], phi_0[i], dV_ges[i], eps_0[i], h_c_0[i], h_dis_0[i]) for i in range(len(exp))]
    
    results = joblib.Parallel(n_jobs=N_CPU, backend='multiprocessing')(joblib.delayed(parallel_simulation)(param) for param in parameters)
    
    # Save results
    df_results = pd.DataFrame(results)
    df_results.to_csv('simulation_results_parallel_evaluation_sozh.csv', index=False)
    print("Alle Simulationen abgeschlossen. Ergebnisse gespeichert.")