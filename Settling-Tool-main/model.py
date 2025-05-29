# -*- coding: utf-8 -*-
"""
@author: luth01
"""
import numpy as np

import settings as settings
import helper_functions as help_fun
import settings as s
import matplotlib.pyplot as plt
import math

from math import isnan
from scipy import optimize, stats

class Settling_Curve_Henschke():

    def __init__(self, Experiment, Sedimentation_model):

        self.Exp = Experiment
        self.Sed_mod = Sedimentation_model
        self.time_calc = []
        self.h_c_calc = []
        self.h_d_calc = []
        self.h_p_calc = []
        self.epsilon_p_calc = []
        self.d_32_calc = []
        self.v_s = self.Sed_mod.v_s
        self.v_rs = self.v_s / (1 - self.Exp.epsilon_0)
        self.phi_32 = Sedimentation_model.calc_phi_32()
        self.t_e_calculated = 0

    def tau(self, r_s_star, h, d_i, ID):
        La_mod = (self.Exp.g * self.Exp.delta_rho / self.Exp.sigma) ** 0.6 \
                 * d_i * h ** 0.2

        R_F = d_i * (1 - (4.7 / (4.7 + La_mod))) ** 0.5
        if ID == "d":
            R_F = 0.3025 * R_F
        else:
            R_F = 0.5240 * R_F

        R_a = 0.5 * d_i * (1 - (1 - 4.7 / (4.7 + La_mod)) ** 0.5)
        tau = 7.65 * self.Exp.eta_conti * (R_a ** (7 / 3) \
                                           / (self.Exp.H_cd ** (1 / 6) * self.Exp.sigma ** (5 / 6) * R_F * r_s_star))

        return tau

    # simplifiedSauter auf True bedeutet, dass das vereinfachte Koaleszenzmodell zur Berechnung verwendet wird
    def calc_settling_curve_light_in_heavy(self, r_s_star, simplifiedSauter=False, no_dd_coal=False, h_p_star=0.01):

        if no_dd_coal:
            simplifiedSauter = True

        # delete old calculation values
        self.time_calc = []
        self.h_c_calc = []
        self.h_d_calc = []
        self.h_p_calc = []
        self.epsilon_p_calc = []
        self.d_32_calc = []

        h_c = 0
        h_d = self.Exp.H_0
        delta_h_d = 0
        h_p = 0
        h_c0 = 0
        h_d0 = self.Exp.H_0
        delta_h = self.Exp.H_0 / self.Exp.N_h

        t = 0
        delta_t = self.Exp.t_e / self.Exp.N_t
        t_plus = 10 * self.Exp.t_e

        epsilon_p = (self.Exp.epsilon_0 + self.Exp.epsilon_di) / 2
        epsilon_p_0 = (self.Exp.epsilon_0 + self.Exp.epsilon_di) / 2

        if simplifiedSauter == False:
            d_32_hight = np.full(self.Exp.N_h + 1, self.phi_32)
        else:
            d_32 = self.phi_32

        # start value to output
        self.time_calc.append(t)
        self.h_c_calc.append(h_c)
        self.h_d_calc.append(h_d)
        self.h_p_calc.append(h_d - h_p)

        while h_d != h_c:
            t = t + delta_t
            h_d = h_d + delta_h_d

            if t < t_plus:
                h_c = h_c + self.v_s * delta_t
                h_p = ((abs(h_c0 - h_c)) * self.Exp.epsilon_0 - \
                       (1 - self.Exp.epsilon_0) * abs(h_d0 - h_d)) / \
                      (epsilon_p_0 - self.Exp.epsilon_0)

                if h_p < 0:
                    h_p = 0

                if h_c >= h_d - h_p:
                    t_plus = t
                    dh_dt = abs(delta_h_d) / delta_t

                    C_1 = ((self.v_s + dh_dt) * epsilon_p_0 ** 2 + epsilon_p_0 * dh_dt) / \
                          ((h_d0 - h_c0) * self.Exp.epsilon_0 - abs(h_d0 - h_d) * \
                           (self.Exp.epsilon_di - epsilon_p_0))
                    C_2 = - C_1 * t_plus - np.log(self.Exp.epsilon_di - epsilon_p_0)

            if t >= t_plus:
                epsilon_p = self.Exp.epsilon_di - np.exp(- C_1 * t - C_2)
                if epsilon_p < epsilon_p_0:
                    epsilon_p = epsilon_p_0
                if epsilon_p > 1:
                    epsilon_p = 1

                h_p = ((abs(h_d0 - h_c0) * self.Exp.epsilon_0) \
                       - abs(h_d0 - h_d)) / epsilon_p

                if h_p < 0:
                    h_p = 0
                h_c = h_d - self.Exp.epsilon_di * h_p

            i_low = self.Exp.N_h - np.round(abs(self.Exp.H_0 - h_d + h_p * epsilon_p) / \
                                            (delta_h * self.Exp.epsilon_0))


            if type(i_low) != int:
                i_low = i_low.astype(int)
                if type(i_low) is np.ndarray:
                    i_low = i_low[0]

            i_high = self.Exp.N_h - np.round(abs(self.Exp.H_0 - h_d) \
                                             / (delta_h * self.Exp.epsilon_0))
            if type(i_high) != int:
                i_high = i_high.astype(int)
                if type(i_high) is np.ndarray:
                    i_high = i_high[0]

            if i_low < 0:
                i_low = 0

            if i_high > self.Exp.N_h or i_high < 1: # GEDANKEN MACHEN
                i_high = self.Exp.N_h

            if simplifiedSauter == False:
                if h_p < d_32_hight[i_high] / 2:
                    h_eff = d_32_hight[i_high] / 2
                else:
                    h_eff = h_p

                tau_di = self.tau(r_s_star, h_eff, d_32_hight[i_high], "i")
                delta_h_d = - 2 * self.Exp.epsilon_di * d_32_hight[i_high] * delta_t \
                        / (3 * tau_di)

                for i in range(i_low, i_high):

                    h_py = (abs(h_d0 - h_d) + h_p * epsilon_p - \
                        (self.Exp.N_h - i) * delta_h * self.Exp.epsilon_0) / epsilon_p
                    if h_py < 0.01 * d_32_hight[i]:
                        h_py = 0.01 * d_32_hight[i]
                    tau_dd = self.tau(r_s_star, h_py, d_32_hight[i], "d")
                    d_32_hight[i] = d_32_hight[i] \
                                    + (delta_t * d_32_hight[i] / (6 * tau_dd))

            else:
                if h_p < d_32 / 2:
                    h_eff = d_32 / 2
                else:
                    h_eff = h_p

                tau_di = self.tau(r_s_star, h_eff, d_32, "i")
                delta_h_d = - 2 * self.Exp.epsilon_di * d_32 * delta_t \
                            / (3 * tau_di)

                if no_dd_coal == False:
                    if h_eff < 0.01 * d_32:
                        h_eff = 0.01 * d_32

                    tau_dd = self.tau(r_s_star, h_p_star*h_eff, d_32, "d")
                    d_32 = d_32 + (delta_t * d_32 / (6 * tau_dd))

            if type(t) is np.ndarray:
                t = t[0]
            self.time_calc.append(t)
            if type(h_c) is np.ndarray:
                h_c = h_c[0]
            self.h_c_calc.append(h_c)
            if type(h_d) is np.ndarray:
                h_d = h_d[0]
            self.h_d_calc.append(h_d)
            if type(h_p) is np.ndarray:
                h_p = h_p[0]
            self.h_p_calc.append(h_d - h_p)

            self.epsilon_p_calc.append(epsilon_p)

            if simplifiedSauter == False:
                self.d_32_calc.append(d_32_hight)
            else:
                self.d_32_calc.append(d_32)

        self.t_e_calculated = t

        return

    def calc_settling_curve_heavy_in_light(self, r_s_star, simplifiedSauter=False, no_dd_coal=False, h_p_star=0.01):

        if no_dd_coal:
            simplifiedSauter = True

        # delete old calculation values
        self.time_calc = []
        self.h_c_calc = []
        self.h_d_calc = []
        self.h_p_calc = []
        self.epsilon_p_calc = []
        self.d_32_calc = []

        self.v_s = -abs(self.v_s)   # v_s needs negative sign

        h_c = self.Exp.H_0
        h_d = 0
        delta_h_d = 0
        h_p = 0
        delta_h = self.Exp.H_0 / self.Exp.N_h

        t = 0
        delta_t = self.Exp.t_e / self.Exp.N_t
        t_plus = 10 * self.Exp.t_e

        epsilon_p = (self.Exp.epsilon_0 + self.Exp.epsilon_di) / 2
        epsilon_p_0 = (self.Exp.epsilon_0 + self.Exp.epsilon_di) / 2
        if simplifiedSauter == False:
            d_32_hight = np.full(self.Exp.N_h + 1, self.phi_32)
        else:
            d_32 = self.phi_32

        while h_c >= h_d:
            t = t + delta_t
            h_d = h_d + delta_h_d

            if t < t_plus:
                h_c = h_c + self.v_s * delta_t
                h_p = ((self.Exp.H_0 - h_c) * self.Exp.epsilon_0 \
                       - (1 - self.Exp.epsilon_0) * h_d) / (epsilon_p_0 - self.Exp.epsilon_0)
                if h_d + h_p >= h_c:
                    t_plus = t
                    dh_d_dt = delta_h_d / delta_t
                    C_1 = ((self.v_s - dh_d_dt) * epsilon_p_0 ** 2 \
                           + epsilon_p_0 * dh_d_dt) / \
                          ((h_d - self.Exp.H_0 * self.Exp.epsilon_0) \
                           * (self.Exp.epsilon_di - epsilon_p_0))
                    C_2 = - C_1 * t_plus - np.log(self.Exp.epsilon_di - epsilon_p_0)
            if t >= t_plus:
                epsilon_p = self.Exp.epsilon_di - np.exp(-C_1 * t - C_2)
                if epsilon_p < epsilon_p_0:
                    epsilon_p = epsilon_p_0
                if epsilon_p > 1:
                    epsilon_p = 1

                h_p = (self.Exp.H_0 * self.Exp.epsilon_0 - h_d) / epsilon_p
                h_c = h_d + h_p

            i_low = round((h_d / (delta_h * self.Exp.epsilon_0)))

            if type(i_low) != int:
                i_low = i_low.astype(int)
                if type(i_low) is np.ndarray:
                    i_low = i_low[0]

            i_high = round(((h_d + h_p * epsilon_p) / (delta_h * self.Exp.epsilon_0)))
            if type(i_high) != int:
                i_high = i_high.astype(int)
                if type(i_high) is np.ndarray:
                    i_high = i_high[0]

            if i_low < 0:
                i_low = 0
            if i_low > self.Exp.N_h:
                i_low = self.Exp.N_h

            if i_high > self.Exp.N_h or i_high < 1:
                i_high = self.Exp.N_h

            if simplifiedSauter == False:
                if h_p < d_32_hight[i_low] / 2:
                    h_eff = d_32_hight[i_low] / 2
                else:
                    h_eff = h_p

                tau_di = self.tau(r_s_star, h_eff, d_32_hight[i_low], "i")
                delta_h_d = 2 * self.Exp.epsilon_di * d_32_hight[i_low] * delta_t \
                        / (3 * tau_di)

                for i in range(i_low, i_high):
                    h_py = (h_d + h_p * epsilon_p - i * delta_h \
                            * self.Exp.epsilon_0) / epsilon_p
                    if h_py < 0.01 * d_32_hight[i]:
                        h_py = 0.01 * d_32_hight[i]
                    tau_dd = self.tau(r_s_star, h_py, d_32_hight[i], "d")
                    d_32_hight[i] = d_32_hight[i] \
                                    + (delta_t * d_32_hight[i] / (6 * tau_dd))
            else:

                if h_p < d_32 / 2:
                    h_eff = d_32 / 2
                else:
                    h_eff = h_p

                tau_di = self.tau(r_s_star, h_eff, d_32, "i")
                delta_h_d = 2 * self.Exp.epsilon_di * d_32 * delta_t \
                            / (3 * tau_di)

                if no_dd_coal == False:
                    if h_eff < 0.01 * d_32:
                        h_eff = 0.01 * d_32

                    tau_dd = self.tau(r_s_star, h_p_star*h_eff, d_32, "d")
                    d_32 = d_32 + (delta_t * d_32 / (6 * tau_dd))

            if type(t) is np.ndarray:
                t = t[0]
            self.time_calc.append(t)
            if type(h_c) is np.ndarray:
                h_c = h_c[0]
            self.h_c_calc.append(h_c)
            if type(h_d) is np.ndarray:
                h_d = h_d[0]
            self.h_d_calc.append(h_d)
            if type(h_p) is np.ndarray:
                h_p = h_p[0]
            self.h_p_calc.append(h_d + h_p)

            self.epsilon_p_calc.append(epsilon_p)

            if simplifiedSauter == False:
                self.d_32_calc.append(d_32_hight)
            else:
                self.d_32_calc.append(d_32)

        self.t_e_calculated = t

        return

    def calc_settling_curve(self, r_s_star, simplifiedSauter=False, h_p_star=0.01):

        no_dd_coal = False
        if h_p_star <= 0:
            h_p_star = 0
            no_dd_coal = True

        h_c_exp_check = next(x for x in [float(x) for x in list(self.Exp.h_c_exp)] if not isnan(x))
        h_d_exp_check = next(x for x in [float(x) for x in list(self.Exp.h_d_exp)] if not isnan(x))

        if h_c_exp_check < h_d_exp_check:
            light_in_heavy = True
        else:
            light_in_heavy = False

        if light_in_heavy:
            self.calc_settling_curve_light_in_heavy(r_s_star, simplifiedSauter=simplifiedSauter,
                                                    h_p_star=h_p_star, no_dd_coal=no_dd_coal)
        else:
            self.calc_settling_curve_heavy_in_light(r_s_star, simplifiedSauter=simplifiedSauter,
                                                    h_p_star=h_p_star, no_dd_coal=no_dd_coal)

    def Henschke_error_func(self):

        exp_h_d = self.Exp.h_d_exp
        time_coal_exp = np.array(self.Exp.time_exp) - self.Exp.t_0
        nan_indicies = np.isnan(exp_h_d)
        exp_h_d = exp_h_d[~nan_indicies]
        time_coal_exp = time_coal_exp[~nan_indicies]

        inter_h_d_sim = np.interp(time_coal_exp, self.time_calc, self.h_d_calc)

        error_coal = help_fun.NRMSE(List_exp=exp_h_d,
                                    List_sim=inter_h_d_sim,
                                    Norm_value=self.Exp.epsilon_0*self.Exp.H_0)


        error_te = ((self.t_e_calculated - self.Exp.t_e) / (2*self.Exp.t_e)) ** 2

        return error_coal, error_te

    def plot_Henschke_error_rs(self, r_s_min=0.01, r_s_max=0.5, number_steps=30):

        r_s_star_vec = np.linspace(r_s_min, r_s_max, number_steps)
        total_error = np.zeros_like(r_s_star_vec)
        error_coal = np.zeros_like(r_s_star_vec)
        error_te = np.zeros_like(r_s_star_vec)

        for i in range(number_steps):
            r_s_star = r_s_star_vec[i]
            self.calc_settling_curve(r_s_star)
            error_coal[i], error_te[i] = self.Henschke_error_func()
            total_error[i] = error_coal[i] + error_te[i]

        plt.clf()
        plt.scatter(r_s_star_vec, total_error, marker='o', label='Q_ges')
        plt.scatter(r_s_star_vec, error_coal, marker='+', label='Q_h')
        plt.scatter(r_s_star_vec, error_te, marker='^', label='Q_te')

        plt.xlabel("r_s_star [-]", size=15)
        plt.ylabel("Henschke Error [-]", size=15)

        plt.legend()
        plt.draw()
        plt.show()

    def plot_Henschke_error_hp(self, h_p_min=0.01, h_p_max=0.5, number_steps=30, r_s_star=0.1):

        h_p_star_vec = np.linspace(h_p_min, h_p_max, number_steps)
        total_error = np.zeros_like(h_p_star_vec)
        error_coal = np.zeros_like(h_p_star_vec)
        error_te = np.zeros_like(h_p_star_vec)

        for i in range(number_steps):
            h_p_star = h_p_star_vec[i]
            self.calc_settling_curve(r_s_star, simplifiedSauter=True, h_p_star=h_p_star)
            error_coal[i], error_te[i] = self.Henschke_error_func()
            total_error[i] = error_coal[i] + error_te[i]

        plt.clf()
        plt.scatter(h_p_star_vec, total_error, marker='o', label='Q_ges')
        plt.scatter(h_p_star_vec, error_coal, marker='+', label='Q_h')
        plt.scatter(h_p_star_vec, error_te, marker='^', label='Q_te')

        plt.xlabel("h_p_star [-]", size=15)
        plt.ylabel("Henschke Error [-]", size=15)

        plt.legend()
        plt.draw()
        plt.show()

    def find_h_p_star(self, r_s_star=0.01, start_value=0.01, bounds=[(1e-13, 1)], full_report=True):

        def func_for_min(h_p_star):
            h_p_star = h_p_star[0]
            if h_p_star <= 0:
                h_p_star = 1e-5
            self.calc_settling_curve(r_s_star, simplifiedSauter=True, h_p_star=h_p_star)
            error_coal, error_te = self.Henschke_error_func()
            total_error = error_coal + error_te
            return total_error

        res = optimize.minimize(func_for_min, x0=np.array(start_value), bounds=bounds, method='Nelder-Mead')

        if full_report:
            print("------------------------------")
            print("Report h_p_star:")
            print(res)
            print("------------------------------")

        return res.x

    def find_r_s_star(self, start_value=0.01, bounds=[(1e-4, 1)], full_report=True):

        def func_for_min(r_s_star):
            self.calc_settling_curve(r_s_star[0])
            error_coal, error_te = self.Henschke_error_func()
            total_error = error_coal + error_te
            return total_error

        res = optimize.minimize(func_for_min, x0=np.array(start_value), bounds=bounds, method='Nelder-Mead')

        if full_report:
            print("------------------------------")
            print("Report r_s_star:")
            print(res)
            print("------------------------------")

        return res.x

    def plot_result(self):
        plt.clf()

        plt.scatter(self.Exp.time_exp, self.Exp.h_c_exp, label='Conti_Phase',
                    color='b', marker='o', s=50)
        plt.plot(np.array(self.time_calc) + self.Exp.t_0, self.h_d_calc,
                 label='disp')

        plt.scatter(self.Exp.time_exp, self.Exp.h_d_exp, label='Disp_phase',
                    color='r', marker='o', s=50, )
        plt.plot(np.array(self.time_calc) + self.Exp.t_0, self.h_p_calc,
                 label=' dpz  ')

        plt.plot(np.array(self.time_calc) + self.Exp.t_0, self.h_c_calc,
                 label='conti ')

        plt.xlim(left=0)
        plt.ylim(bottom=0)
        plt.legend()
        plt.xlabel("Time [s]", size=15)
        plt.ylabel("Height [m]", size=15)

        plt.draw()
        plt.show()

def plot_results(filenames, time_factors, rs_stars, hp_stars, simplie_list, labels, title='Comparison',
                 hp_plot=True, data=True, legend_title=None, figsize=(8,6), butyl=False):
    colors = ['b', 'r', 'g', 'm', 'k', 'y']
    markers = ['o', 'x', '^', '*', 'o', 'o']
    if len(colors) < len(filenames):
        print('add more colors in plot_results function')
        return

    if len(markers) < len(filenames):
        print('add more markers in plot_results function')
        return

    # plt.clf()
    plt.figure(figsize=figsize)

    for i in range(0, len(filenames)):
        file = filenames[i]
        time_fac = time_factors[i]

        Exp = s.Settling_Experiment()
        Exp.update(file)
        Sed = Swarm_Sedimentation_Pilhofer_Mewes(Exp)
        Sed.set_factor_time(time_fac)
        Sed.calc_phi_32()
        Hen = Settling_Curve_Henschke(Exp, Sed)

        simplifiedSauter = simplie_list[i]
        r_s_star = rs_stars[i]
        h_p_star = hp_stars[i]

        if simplifiedSauter:
            Hen.calc_settling_curve(r_s_star=r_s_star, simplifiedSauter=True, h_p_star=h_p_star)
        else:
            Hen.calc_settling_curve(r_s_star=r_s_star)

        print('Sauter in mm: ' + str(Hen.phi_32))

        c = colors[i]
        l = labels[i]
        m = markers[i]

        if data:
            plt.scatter(Exp.time_exp, Exp.h_c_exp, color=c, marker=m, s=30, label=l)
            plt.scatter(Exp.time_exp, Exp.h_d_exp, color=c, marker=m, s=30, )
            plt.plot(np.array(Hen.time_calc) + Exp.t_0, Hen.h_d_calc, color=c)
        else:
            plt.plot(np.array(Hen.time_calc) + Exp.t_0, Hen.h_d_calc, color=c, label=l)
        plt.plot(np.array(Hen.time_calc) + Exp.t_0, Hen.h_c_calc, color=c)
        if hp_plot:
            plt.plot(np.array(Hen.time_calc) + Exp.t_0, Hen.h_p_calc, color=c, linestyle='--')

    plt.xlim(left=0)
    plt.ylim(bottom=0)

    if legend_title == None:
        plt.legend(frameon=False)
    else:
        plt.legend(title=legend_title, frameon=False)
    if butyl:
        plt.xlim(right=90)
        plt.legend(title=legend_title, frameon=False, loc='center right')
        plt.text(50, 0.17, 'o in w')
        plt.text(50, 0.06, 'w in o')
    plt.title(title)
    plt.xlabel("Zeit / s", size=12)
    plt.ylabel("Höhe / m", size=12)
    plt.tick_params(axis='x', top=True, direction='in')
    plt.tick_params(axis='y', right=True, direction='in')
    plt.tight_layout()
    plt.draw()
    plt.show()


def linear_regression_sedimentation_observed(Settling_Experiment_Class, plot=False, factor_time=0.06):
    '''
    ----------
    Settling_Experiment_Class : Object
        data of the experiment and settings are stored in this class
        time_exp and h_c_exp are used for the regression
    plot: boalean
        Default = False -> no plotting
        if True a plot of the Regression is visulized

    Returns
    -------
    TYPE
        v_s is the slop of the regression like Henschke said:
        Sedimentationsgeschwindigkeit v_s = dh/dt
        -> only observed sedimentation velocity

    '''
    exp = Settling_Experiment_Class

    # we should think about a better way of aproximating the Range of the regression

    ratio_time = factor_time  # multiplier of the max time to cap sedimentation zone
    index_of_t_0 = np.where(exp.time_exp ==
                            exp.t_0)[0][0]

    max_time = max(exp.time_exp)

    time_regression = exp.time_exp[index_of_t_0:
                                   round((index_of_t_0 + index_of_t_0 + max_time * ratio_time))]

    Sed_values = exp.h_c_exp[index_of_t_0:
                             round((index_of_t_0 + index_of_t_0 + max_time * ratio_time))]
    nan_indicies = np.isnan(Sed_values)
    Sed_values = Sed_values[~nan_indicies]
    time_regression = time_regression[~nan_indicies]

    def lin_func(time, v_s):
        temp = [float(x) for x in list(exp.h_c_exp)]
        return (time - exp.t_0) * v_s \
            + next(x for x in temp if not isnan(x))

    v_s, covariance = optimize.curve_fit(lin_func, time_regression, Sed_values)

    # Plot function of the Regression!
    if plot == True:
        plt.figure(figsize=(8, 6))
        plt.scatter(exp.time_exp,
                    exp.h_c_exp, s=50, color='b', marker='o',
                    label='Experiment')
        plt.vlines(exp.time_exp[index_of_t_0],
                   min(exp.h_c_exp),
                   max(exp.h_c_exp), 'black',
                   linestyles='solid', label='start regression', linewidth=3)
        plt.vlines(exp.time_exp[round((index_of_t_0 + index_of_t_0 +
                                       max_time * ratio_time))] - 1,
                   min(exp.h_c_exp),
                   max(exp.h_c_exp), 'black',
                   linestyles='-.', label='end regression', linewidth=3)
        # plt.plot(Experiment_1.time_exp, Experiment_1.h_c_exp, 'o', label='Data Experiment')
        temp = [float(x) for x in list(exp.h_c_exp)]
        plt.plot(time_regression, (time_regression - exp.t_0) * v_s \
                 + next(x for x in temp if not isnan(x)),
                 'r', label='Sedimentation Line', linewidth=3)
        plt.legend(fontsize="10")
        plt.xlabel('Time / s', size=15)
        plt.ylabel('Hight / m', size=15)
        plt.show()
    return v_s[0]


class Swarm_Sedimentation_Pilhofer_Mewes():
    '''
    Notation is from Henschk pub 2001!
    Only Valid for Ar > 1 and 0.06 < epsilon_0 < 0.55
    '''

    def __init__(self, Experiment):
        self.g = Experiment.g

        self.Exp = Experiment
        self.delta_rho = Experiment.delta_rho
        self.rho_conti = Experiment.rho_conti
        self.eta_conti = Experiment.eta_conti
        self.rho_disp = Experiment.rho_disp
        self.eta_disp = Experiment.eta_disp

        self.epsilon_0 = Experiment.epsilon_0

        self.phi_32 = 0
        self.factor_time = 0
        self.v_s = abs(linear_regression_sedimentation_observed(Experiment))

    def Ar(self):
        return (self.g * self.phi_32 ** 3 * self.delta_rho * self.rho_conti) \
            / (self.eta_conti ** 2)

    def K_HR(self):
        return (3 * (self.eta_conti + self.eta_disp) / (2 * self.eta_conti + 3 * self.eta_disp))

    def zeta(self):
        return 5 * self.K_HR() ** (-3 / 2) * (self.epsilon_0 / (1 - self.epsilon_0)) ** (0.45)

    def q(self):
        return ((1 - self.epsilon_0) / (2 * self.epsilon_0 * self.K_HR())) * \
            np.exp((2.5 * self.epsilon_0 / (1 - (0.61 * self.epsilon_0))))

    def Re_unendlich(self):
        return 9.72 * ((1 + 0.01 * self.Ar()) ** (4 / 7) - 1)

    def cw(self):
        return self.Ar() / (6 * self.Re_unendlich() ** 2) - 3 / (self.K_HR() * self.Re_unendlich())

    def Re_s(self):
        return self.rho_conti * self.v_rs() * self.phi_32 / (self.eta_conti)  # hier v_rs wieder hinmachen

    def v_rs(self):
        return self.v_s / (1 - self.epsilon_0)

    def v_Stokes(self):
        return self.g * self.phi_32 ** 2 * self.delta_rho / (18 * self.eta_conti)

    def set_factor_time(self, fact):
        self.factor_time = fact
        vs_new = abs(linear_regression_sedimentation_observed(self.Exp, factor_time=self.factor_time))
        self.v_s = vs_new

    def Re_s_Pilhofer_Mewes(self):
        '''
        a = (3 * q * ε_0)/(c_w * ξ * (1- ε_0) )

        b = ((1 + Ar * (cw * ξ * (1 − ε_0)^3) / (54 * q^2 * ε_0^2 )))
        '''
        a = (3 * self.q() * self.epsilon_0) / \
            (self.cw() * self.zeta() * (1 - self.epsilon_0))
        b = ((1 + self.Ar() * (self.cw() * self.zeta() * (1 - self.epsilon_0) ** 3) / \
              (54 * self.q() ** 2 * self.epsilon_0 ** 2)) ** (1 / 2) - 1)

        return a * b

    def calc_phi_32(self, full_report=False):
        def Re_s_itteration(phi_32):
            self.phi_32 = phi_32
            '''
            a = (3 * q * ε_0)/(c_w * ξ * (1- ε_0) )

            b = ((1 + Ar * (cw * ξ * (1 − ε_0)^3) / (54 * q^2 * ε_0^2 )))
            '''
            a = (3 * self.q() * self.epsilon_0) / \
                (self.cw() * self.zeta() * (1 - self.epsilon_0))

            b = ((1 + self.Ar() * (self.cw() * self.zeta() * (1 - self.epsilon_0) ** 3) / \
                  (54 * self.q() ** 2 * self.epsilon_0 ** 2)) ** (1 / 2) - 1)

            return (a * b) - self.Re_s()

        lower_bound = 1e-8
        uper_bound = 10e-3  # 10 mm
        results = optimize.brentq(Re_s_itteration, a=uper_bound, b=lower_bound, full_output=True, xtol=2e-12)
        self.phi_32 = results[0]

        if full_report == True:
            print("------------------------------")
            print("scipy.optimize.brentq is used")
            print("Result:")
            print(f" d_32 = {self.phi_32 * 1000:.3f} mm")
            print(f"infodict = {results[1]}")
            print("------------------------------")
        return self.phi_32