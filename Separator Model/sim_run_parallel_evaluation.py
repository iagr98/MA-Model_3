import numpy as np
import pandas as pd
import joblib
import helper_functions as hf
from sim_run import run_sim

N_CPU = 7


df = pd.read_excel("Input/data_main.xlsx", sheet_name="main")
exp = df['exp'].tolist()
phi_0 = df['phi_0'].tolist()
dV_ges = df['dV_ges'].tolist()
eps_0 = df['eps_0'].tolist()


def parallel_simulation(params):
    exp, phi_0, dV_ges, eps_0 = params
    print(f"Start simulation with exp={exp}, phi_0={phi_0}, dV_ges={dV_ges}, eps_0={eps_0}")
    try:
        Sim = run_sim(exp, phi_0, dV_ges, eps_0)
        return {'exp': exp, 'phi_0': phi_0, 'dV_ges': dV_ges, 'eps_0': eps_0,
                'V_dis_total': Sim.V_dis_total,
                'Vol_imbalance [%]': hf.calculate_volume_balance(Sim), 'status': 'success'}
    except Exception as e:
        print(f"Simulation failed for exp={exp}, phi_0={phi_0}, dV_ges={dV_ges}, eps_0={eps_0}: {str(e)}")
        return {'exp': exp, 'phi_0': phi_0, 'dV_ges': dV_ges, 'eps_0': eps_0, 'error': str(e), 'status': 'failed'}

if __name__ == "__main__":
    parameters = [(exp[i], phi_0[i], dV_ges[i], eps_0[i]) for i in range(len(exp))]
    
    results = joblib.Parallel(n_jobs=N_CPU)(joblib.delayed(parallel_simulation)(param) for param in parameters)
    
    # Save results
    df_results = pd.DataFrame(results)
    df_results.to_csv('simulation_results_parallel_cos.csv', index=False)
    print("Alle Simulationen abgeschlossen. Ergebnisse gespeichert.")