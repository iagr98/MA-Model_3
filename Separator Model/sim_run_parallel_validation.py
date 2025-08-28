import numpy as np
import pandas as pd
import joblib
import helper_functions as hf
from sim_run import run_sim


def parallel_simulation(params):
    # ye
    exp, phi_0, dV_ges, eps_0, exponent = params
    print(f"Start simulation with exp={exp}, phi_0={phi_0}, dV_ges={dV_ges}, eps_0={eps_0}, exponent={exponent}")
    # sz
    # exp, phi_0, dV_ges, eps_0, h_c_0, h_dis_0, exponent = params
    # print(f"Start simulation with exp={exp}, phi_0={phi_0}, dV_ges={dV_ges}, eps_0={eps_0}, h_c_0={h_c_0}, h_dis_0={h_dis_0}, exponent={exponent}")
    try:
        # ye
        Sim = run_sim(exp, phi_0, dV_ges, eps_0, exponent=exponent)
        return {'exp': exp, 'phi_0': phi_0, 'dV_ges': dV_ges, 'eps_0': eps_0,
                'dpz_flooded': Sim.h_dpz_status,  'L_DPZ':Sim.L_DPZ,
                'Vol_imbalance [%]': hf.calculate_volume_balance(Sim), 'status': 'success'} 
        
        #sz
        # Sim = run_sim(exp, phi_0, dV_ges, eps_0, h_c_0, h_dis_0, exponent=exponent)
        # return {'exp': exp, 'phi_0': phi_0, 'dV_ges': dV_ges, 'eps_0': eps_0,
        #         'h_c_0': h_c_0, 'h_dis_0': h_dis_0,
        #         'V_dis_total': Sim.V_dis_total, 'dpz_flooded': Sim.h_dpz_status,
        #         'h_dpz': Sim.h_dpz, 'h_c': Sim.h_c,
        #         'Vol_imbalance [%]': hf.calculate_volume_balance(Sim), 'status': 'success'}
    except Exception as e:
        # ye  
        print(f"Simulation failed for exp={exp}, phi_0={phi_0}, dV_ges={dV_ges}, eps_0={eps_0}: {str(e)}")
        return {'exp': exp, 'phi_0': phi_0, 'dV_ges': dV_ges, 'eps_0': eps_0, 'exponent':exponent, 'error': str(e), 'status': 'failed'}
        
        # sz
        # print(f"Simulation failed for exp={exp}, phi_0={phi_0}, dV_ges={dV_ges}, eps_0={eps_0}, h_c_0={h_c_0}, h_dis_0={h_dis_0}, exponent={exponent}: {str(e)}")
        # return {'exp': exp, 'phi_0': phi_0, 'dV_ges': dV_ges, 'eps_0': eps_0, 'h_c_0': h_c_0, 'h_dis_0': h_dis_0, 'exponent':exponent, 'error': str(e), 'status': 'failed'}

if __name__ == "__main__":

    exponent = 2
    N_CPU = 2

    experiment = "sozh" # "main" if ye + niba tests, "sozh" tests from AVT.FVT, detail_V_dis for detail analysis
    tests = [9, 19]
    df = pd.read_excel("Input/data_main.xlsx", sheet_name=experiment)
    df = df.iloc[tests]  # Select only the tests we want to run
    exp = df['exp'].tolist()
    phi_0 = df['phi_0'].tolist()
    dV_ges = df['dV_ges'].tolist()
    eps_0 = df['eps_0'].tolist()
    h_c_0 = df['h_c_0'].tolist() 
    h_dis_0 = df['h_dis_0'].tolist()

    parameters = [(exp[i], phi_0[i], dV_ges[i], eps_0[i], h_c_0[i], h_dis_0[i], exponent) for i in range(len(exp))]
    results = joblib.Parallel(n_jobs=N_CPU, backend='multiprocessing')(joblib.delayed(parallel_simulation)(param) for param in parameters)
    
    # Save results
    df_results = pd.DataFrame(results)
    df_results.to_csv('simulation_results_parallel_evaluation_detail_new_fit.csv', index=False)
    print("Alle Simulationen abgeschlossen. Ergebnisse gespeichert.")