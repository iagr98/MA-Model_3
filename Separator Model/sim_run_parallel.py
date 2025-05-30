import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import helper_functions as hf
from sim_run import run_sim

N_CPU = 8
filename = "Paraffin_flut_20C.xlsx"
# filename = "niba_V2.xlsx"
N_x = [51, 81, 101, 151, 201, 251, 301, 351]
atol = 1e-7

var = 'N_x'


def parallel_simulation(params): 
    filename, N_x, atol = params
    print(f"Start simulation with {var}={N_x}")                                     # Update parameter in second {}
    try:
        Sim = run_sim(filename, N_x=N_x, a_tol=atol)
        return {f"{var}": N_x, 'V_dis_total': Sim.V_dis_total,'Vol_imbalance [%]': hf.calculate_volume_balance(Sim), 'status': 'success'}    # Update parameter in second place
    except Exception as e:
        print(f"Simulation failed by {var}={N_x}: {str(e)}")                        # Update parameter in second {}
        return {f"{var}": N_x, 'error': str(e), 'status': 'failed'}                # Update parameter in second place           

if __name__ == "__main__":
    parameters = [(filename, N_x_value, atol) for N_x_value in N_x]               # Update parameter var_value, var_value & var 
    
    results = joblib.Parallel(n_jobs=N_CPU)(joblib.delayed(parallel_simulation)(param) for param in parameters)
    
    # Save results
    df_results = pd.DataFrame(results)
    df_results.to_csv('simulation_results_parallel.csv', index=False)
    print("Alle Simulationen abgeschlossen. Ergebnisse gespeichert.")

    # Plot results
    df = pd.read_csv("simulation_results_parallel.csv")
    df.columns = df.columns.str.strip()
    plt.figure(figsize=(8, 5))
    plt.plot(df['N_x'], df['V_dis_total'], marker='o')         # Update parameter in first place
    # plt.xscale('log')  # da atol logarithmisch skaliert ist
    plt.yscale('log')  # da atol logarithmisch skaliert ist
    plt.xlabel('N_x')                                          # Change x-label
    plt.ylabel('V_dis')                                         
    plt.title(f'Gitterunabhängigkeitsanalyse ({var})')
    plt.grid(True)
    plt.tight_layout()
    plt.show()