import pandas as pd

df = pd.read_csv('Output\\simulation_results_parallel_evaluation_sozh_opt_2.csv')

filtered_df = df[df['h_dis_0'] > 0.05]
# filtered_df = filtered_df[filtered_df['V_dis_total'] > 0.01]


filtered_df.to_csv('Output\\utils_1.csv', index=False)

