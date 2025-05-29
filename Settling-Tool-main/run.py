"""
@author: luth01
"""

import model as m
import settings as s

def find_time_factor(filename, time_factor):
    Exp = s.Settling_Experiment()
    Exp.update(filename)
    m.linear_regression_sedimentation_observed(Exp, plot=True, factor_time=time_factor)

def full_run(filename, time_factor):
    Exp = s.Settling_Experiment()
    Exp.update(filename)
    Exp.plot_experiment()
    m.linear_regression_sedimentation_observed(Exp, plot=True, factor_time=time_factor)
    Sed = m.Swarm_Sedimentation_Pilhofer_Mewes(Exp)
    Sed.set_factor_time(time_factor)
    Sed.calc_phi_32(full_report=True)
    Hen = m.Settling_Curve_Henschke(Exp, Sed)
    r_s_star = Hen.find_r_s_star(start_value=0.0384, full_report=True)[0]
    h_p_star = Hen.find_h_p_star(r_s_star=r_s_star, full_report=True)[0]
    Hen.plot_Henschke_error_rs(r_s_min=r_s_star / 2, r_s_max=r_s_star * 2, number_steps=25)
    Hen.plot_Henschke_error_hp(h_p_min=h_p_star / 2, h_p_max=h_p_star * 2, number_steps=25, r_s_star=r_s_star)
    Hen.calc_settling_curve(r_s_star)
    Hen.plot_result()
    Hen.calc_settling_curve(r_s_star=r_s_star, simplifiedSauter=True, h_p_star=h_p_star)
    Hen.plot_result()

def init_Hen(filename, time_factor):
    Exp = s.Settling_Experiment()
    Exp.update(filename)
    Exp.plot_experiment()
    m.linear_regression_sedimentation_observed(Exp, plot=True, factor_time=time_factor)
    Sed = m.Swarm_Sedimentation_Pilhofer_Mewes(Exp)
    Sed.set_factor_time(time_factor)
    Sed.calc_phi_32(full_report=True)
    Hen = m.Settling_Curve_Henschke(Exp, Sed)
    return Hen

def find_rs_and_hp_star(Hen):
    r_s_star = Hen.find_r_s_star(full_report=True)[0]
    h_p_star = Hen.find_h_p_star(full_report=True, r_s_star=r_s_star)[0]
    return r_s_star, h_p_star

# %%
if __name__ == '__main__':
    ######## full run and find time_factor
    # filename = 'Henschke__4_1_n-Butylacetat_Water.xlsx'
    # time_factor = 0.175 # If KeyError occurs, reduce time_factor
    # find_time_factor(filename, time_factor) # function to manually find the right time factor
    # full_run(filename, time_factor) # test all plots and calculations


    # example for plot of single result (more examples in allExamples.py)
    # # Butanol
    # filename = "Henschke_n_Butanol_2_1.xlsx"
    # time_factor = 0.55
    # Hen = init_Hen(filename, time_factor)
    # r_s_star, h_p_star = find_rs_and_hp_star(Hen)
    # Hen.calc_settling_curve(r_s_star=r_s_star, simplifiedSauter=True, h_p_star=h_p_star)
    # Hen.plot_result()