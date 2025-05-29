import numpy as np
import pandas as pd
import joblib
from sim_run import run_sim

N_CPU = 4
filename = "Paraffin_flut_20C.xlsx"
# filename = "niba_V1.xlsx"
N_x_values = [31, 41, 61, 71]
a_tol_value = 1e-6


def parallel_simulation(params): # Quasi "wrapper function from sozh"
    filename, N_x, a_tol = params
    print(f"Start simulation with N_x={N_x}")
    try:
        Sim = run_sim(filename, N_x=N_x, a_tol=a_tol)
        return {'N_x': N_x, 'V_dis_total': Sim.V_dis_total, 'status': 'success'}
    except Exception as e:
        print(f"Simulation failed by a_tol={a_tol}: {str(e)}")
        return {'N_x': N_x, 'error': str(e), 'status': 'failed'}

if __name__ == "__main__":
    parameters = [(filename, N_x_value, a_tol_value) for N_x_value in N_x_values]
    
    results = joblib.Parallel(n_jobs=N_CPU)(joblib.delayed(parallel_simulation)(param) for param in parameters)
    
    # Save results
    df_results = pd.DataFrame(results)
    df_results.to_csv('simulation_results_parallel.csv', index=False)
    print("Alle Simulationen abgeschlossen. Ergebnisse gespeichert.")