import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df_sim = pd.read_csv("Output/simulation_results_parallel_cos.csv")
df_lab = pd.read_excel("Input/data_main.xlsx", sheet_name="main")
extra = 0.004   # X-Y limit of plot
data = 34       # [0:34]: Only considers ye data, [0,-1] all data.

V_dis_0 = np.concatenate([[0], df_lab['V_dis'][0:data], [extra]])
V_dis_plus_25 = V_dis_0 * 1.25
V_dis_minus_25 = V_dis_0 * 0.75
V_dis_plus_50 = V_dis_0 * 1.50
V_dis_minus_50 = V_dis_0 * 0.50


mape_sim = (abs(df_sim['V_dis_total'][0:data] - df_lab['V_dis'][0:data]) / df_lab['V_dis'][0:data]).mean() * 100
mape_std_sim = (abs(df_sim['V_dis_total'][0:data] - df_lab['V_dis'][0:data]) / df_lab['V_dis'][0:data]).std() * 100

plt.figure(figsize=(8, 6))
plt.plot(V_dis_0, V_dis_0, color='black', linestyle='--')
plt.scatter(df_lab['V_dis'], df_sim['V_dis_total'], label=f'Modell 3. MAPE: {mape_sim:.2f}%. std. Abw.: {mape_std_sim:.2f}%', color='black')
df_sim_ratio_1 = []
df_sim_ratio_2 = []
df_sim_ratio_3 = []
df_sim_niba = []
df_lab_ratio_1 = []
df_lab_ratio_2 = []
df_lab_ratio_3 = []
df_lab_niba = []
for i in range(len(V_dis_0)-1):
    if (df_lab['ratio'][i] == 0):
        df_sim_niba.append(df_sim['V_dis_total'][i])
        df_lab_niba.append(df_lab['V_dis'][i])
    if (df_lab['ratio'][i] < 0.1):
        df_sim_ratio_1.append(df_sim['V_dis_total'][i])
        df_lab_ratio_1.append(df_lab['V_dis'][i])
    elif(0.1 < df_lab['ratio'][i] < 0.6):
        df_sim_ratio_2.append(df_sim['V_dis_total'][i])
        df_lab_ratio_2.append(df_lab['V_dis'][i])
    elif(df_lab['ratio'][i] > 0.6): 
        df_sim_ratio_3.append(df_sim['V_dis_total'][i])
        df_lab_ratio_3.append(df_lab['V_dis'][i])
plt.scatter(df_lab_ratio_1, df_sim_ratio_1, label='ratio < 0.1', color='red')
plt.scatter(df_lab_ratio_2, df_sim_ratio_2, label='0.1 < ratio < 0.6', color='green')
plt.scatter(df_lab_ratio_3, df_sim_ratio_3, label='ratio > 0.6', color='blue')
plt.scatter(df_lab_niba, df_sim_niba, label='niba', color='orange')
plt.plot(V_dis_0, V_dis_plus_25, color='gray', linestyle=':', alpha=0.5)
plt.plot(V_dis_0, V_dis_minus_25, color='gray', linestyle=':', alpha=0.5)
plt.plot(V_dis_0, V_dis_plus_50, color='gray', linestyle='--', alpha=0.5)
plt.plot(V_dis_0, V_dis_minus_50, color='gray', linestyle='--', alpha=0.5)
plt.xlim([0, extra])
plt.ylim([0, extra])
plt.xlabel('V_dis_exp / m³')
plt.ylabel('V_dis_mod / m³')
plt.title('Parity plot V_dis Model 3')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()