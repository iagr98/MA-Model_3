import numpy as np
import pandas as pd
import joblib
import helper_functions as hf
from sim_run import run_sim

N_CPU = 2

# experiment = "sozh" # "main" if ye + niba tests, "sozh" tests from AVT.FVT, detail_V_dis for detail analysis
# df = pd.read_excel("Input/data_main.xlsx", sheet_name=experiment)
experiment = 'testing_validation'
df = pd.read_csv('Input/df_te_dpz.csv')
exp = df['exp'].tolist()
phi_0 = df['phi_0'].tolist()
dV_ges = df['dV_ges'].tolist()
eps_0 = df['eps_0'].tolist()
if (experiment == "sozh" or experiment == "detail_V_dis" or experiment == 'testing_validation'):
    h_c_0 = df['h_c_0'].tolist() 
    h_dis_0 = df['h_dis_0'].tolist()

# exponent = 2


def parallel_simulation(params):
    if (experiment == "main"):
        exp, phi_0, dV_ges, eps_0 = params
        print(f"Start simulation with exp={exp}, phi_0={phi_0}, dV_ges={dV_ges}, eps_0={eps_0}")
    elif(experiment == "sozh" or experiment == "detail_V_dis" or experiment == 'testing_validation'):
        exp, phi_0, dV_ges, eps_0, h_c_0, h_dis_0 = params
        print(f"Start simulation with exp={exp}, phi_0={phi_0}, dV_ges={dV_ges}, eps_0={eps_0}, h_c_0={h_c_0}, h_dis_0={h_dis_0}")
    try:
        if (experiment == "main"):
            Sim = run_sim(exp, phi_0, dV_ges, eps_0)
            return {'exp': exp, 'phi_0': phi_0, 'dV_ges': dV_ges, 'eps_0': eps_0,
                'V_dis_total': Sim.V_dis_total,
                'Vol_imbalance [%]': hf.calculate_volume_balance(Sim), 'status': 'success'}
        elif (experiment == "sozh" or experiment == "detail_V_dis" or experiment == 'testing_validation'):
            Sim = run_sim(exp, phi_0, dV_ges, eps_0, h_c_0, h_dis_0)
            return {'exp': exp, 'phi_0': phi_0, 'dV_ges': dV_ges, 'eps_0': eps_0,
                'h_c_0': h_c_0, 'h_dis_0': h_dis_0,
                'V_dis_total': Sim.V_dis_total, 'dpz_flooded': Sim.h_dpz_status,
                'h_dpz': Sim.h_dpz, 'h_c': Sim.h_c,
                'Vol_imbalance [%]': hf.calculate_volume_balance(Sim), 'status': 'success'}
    except Exception as e:
        if (experiment == "main"):
            print(f"Simulation failed for exp={exp}, phi_0={phi_0}, dV_ges={dV_ges}, eps_0={eps_0}: {str(e)}")
            return {'exp': exp, 'phi_0': phi_0, 'dV_ges': dV_ges, 'eps_0': eps_0, 'error': str(e), 'status': 'failed'}
        elif(experiment == "sozh" or experiment == "detail_V_dis" or experiment == 'testing_validation'):
            print(f"Simulation failed for exp={exp}, phi_0={phi_0}, dV_ges={dV_ges}, eps_0={eps_0}, h_c_0={h_c_0}, h_dis_0={h_dis_0}: {str(e)}")
            return {'exp': exp, 'phi_0': phi_0, 'dV_ges': dV_ges, 'eps_0': eps_0, 'h_c_0': h_c_0, 'h_dis_0': h_dis_0, 'error': str(e), 'status': 'failed'}

if __name__ == "__main__":

    if (experiment == "main"):
        parameters = [(exp[i], phi_0[i], dV_ges[i], eps_0[i]) for i in range(len(exp))]
    elif(experiment == "sozh" or experiment == "detail_V_dis"  or experiment == 'testing_validation'):
        parameters = [(exp[i], phi_0[i], dV_ges[i], eps_0[i], h_c_0[i], h_dis_0[i]) for i in range(len(exp))]
    
    results = joblib.Parallel(n_jobs=N_CPU, backend='multiprocessing')(joblib.delayed(parallel_simulation)(param) for param in parameters)
    
    # Save results
    df_results = pd.DataFrame(results)
    h_dpz_columns = pd.DataFrame(df_results['h_dpz'].tolist())   # Convert h_dpz (list of arrays) into separate columns
    h_dpz_columns.columns = [f'h_dpz_{i}' for i in range(h_dpz_columns.shape[1])]
    df_results = df_results.drop(columns=['h_dpz'])
    h_c_columns = pd.DataFrame(df_results['h_c'].tolist())   # Convert h_dpz (list of arrays) into separate columns
    h_c_columns.columns = [f'h_c_{i}' for i in range(h_c_columns.shape[1])]
    df_results = df_results.drop(columns=['h_c'])
    df_results = pd.concat([df_results, h_dpz_columns, h_c_columns], axis=1)  # Concatenate V_dis columns with the main result dataframe
    df_results.to_csv('simulation_results_te_dpz.csv', index=False)
    print("Alle Simulationen abgeschlossen. Ergebnisse gespeichert.")