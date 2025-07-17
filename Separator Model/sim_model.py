import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import fsolve
import helper_functions as hf
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from helper_functions import getHeightArray

# Diese Datei beinhaltet alle Berechnungen und Plotting-Funktionen
class Simulation():

    def __init__(self, Settings, Substance_System):   # inputs need to be Objects from classes Settings and Substance_System

        self.Set = Settings
        self.Sub = Substance_System

        self.u0 = self.Sub.dV_ges / self.Set.A  # Leerrohrgeschwindigkeit

        self.y0 = []
        self.sol = []
        self.V_dis = []
        self.V_d = []
        self.V_c = []
        self.u_dis = []
        self.u_d = []
        self.u_c = []
        self.phi_32 = []

        self.sigma_before = []                  # only necessary for dynamic change of sigma plot
        self.rs_before = []                     # only necessary for dynamic change of r_s_star
        self.y00 = []                           # only necessary for dynamic change of phase ratio
        self.H_DPZ = 0
        self.L_DPZ = 0
        self.V_dis_total = 0
        self.u_d_balance = 0
        self.u_c_balance = 0
        self.vol_balance = 0
        self.cfl = 0
        self.h_dpz = []
        self.h_c = []

    def set_dVges(self, dVges):
        self.Sub.dV_ges = dVges * 1e-6 / 3.6
        self.u0 = self.Sub.dV_ges / self.Set.A  # Leerrohrgeschwindigkeit

    def calcInitialConditions(self):

        self.u0 = self.Sub.dV_ges / self.Set.A  # Leerrohrgeschwindigkeit
        A = self.Set.A                          # Cross-sectional Area of the settler [m^2]
        dl = self.Set.dl                        # Length of a discrete [m]
        N_x = self.Set.N_x                      # Number of grid points [-]
        r = self.Set.D / 2                      # Radius of the settler [m]
        # h_dis_0 = self.Sub.h_p_sim[0] /1000     # Start value for the height of the dpz [m]
        h_c_0 = self.Set.h_c_0
        h_dis_0 = self.Set.h_dis_0

        ###################################################################################
        # # Numerische Lösung notwendig, da nicht direkt von Höhe der dicht gepackten Schicht auf das Volumen geschlossen werden kann
        # def initial_conditions(values):
        #     ic = np.zeros_like(values)

        #     h_d = values[0]
        #     h_c = values[1]
        #     A_dis = values[2]
        #     A_d = values[3]
        #     A_c = values[4]

        #     # Gleichungen
        #     ic[0] = h_d + h_dis_0 + h_c - self.Set.D  # Settler geflutet
        #     ic[1] = A_d + A_dis + A_c - A  # Settler geflutet
        #     ic[2] = self.Sub.eps_0 * A - self.Sub.eps_p * A_dis - A_d  # Erhaltung der dispersen Phase
        #     ic[3] = A_d - r ** 2 * np.arccos(1 - h_d / r) + (r - h_d) * np.sqrt(
        #         2 * r * h_d - h_d ** 2)  # Kreissegment-Formel
        #     ic[4] = A_c - r ** 2 * np.arccos(1 - h_c / r) + (r - h_c) * np.sqrt(
        #         2 * r * h_c - h_c ** 2)  # Kreissegmrnt-Formel

        #     return ic

        # # Schätzung für die Anfangswerte
        # initial_guess = [r / 2, r / 2, A / 2, A / 4, A / 4]
        # print(self.Sub.eps_0)
        # # Numerische Lösung für Anfangsbedingung
        # h_d_0, h_c_0, A_dis_0, A_d_0, A_c_0 = fsolve(initial_conditions, initial_guess)

        # Berechnungen von Querschnittsflächen

        h_d_0 = 2 * r - h_c_0 - h_dis_0
        A_d_0 = r**2 * np.arccos((r - h_d_0) / r) - (r - h_d_0) * np.sqrt(2 * r * h_d_0 - h_d_0**2)
        A_c_0 = r**2 * np.arccos((r - h_c_0) / r) - (r - h_c_0) * np.sqrt(2 * r * h_c_0 - h_c_0**2)
        A_dis_0 = A - A_d_0 - A_c_0

        # Anfangsbedingungen für Volumina
        Vdis_0 = A_dis_0 * dl * np.ones(N_x)
        Vd_0 = A_d_0 * dl * np.ones(N_x)
        Vc_0 = A_c_0 * dl * np.ones(N_x)

        # Anfangsbedingung für phi_32
        phi32_0 = self.Sub.phi_0 * np.ones(N_x)

        self.y0 = np.concatenate([Vdis_0, Vd_0, Vc_0, phi32_0])  # Array als Anfangsbedingung
        # Array für Anfangsbedingung für x=0 (nur wichtig für dynamischen Wechsel des Phasenverhältnisses)
        self.y00 = np.array([Vdis_0[0], Vd_0[0], Vc_0[0], phi32_0[0]])

    ##### Für Sprungantworten müssen initial conditions und boundary conditions neu gesetzt werden

    # übernimmt Endbedingungen des alten Simulationsobjekts und nutzt diese als Anfangsbedingung des neuen Objekts
    def getInitialConditions(self, old_Sim):
        self.idx = np.argmin(np.abs(old_Sim.sig - 0.5))
        Vdis_0 = old_Sim.V_dis[:, -1]
        Vd_0 = old_Sim.V_d[:, -1]
        Vc_0 = old_Sim.V_c[:, -1]
        phi32_0 = old_Sim.phi_32[:, -1]
        phi32_0[self.idx:] = 0
        self.y0 = np.concatenate([Vdis_0, Vd_0, Vc_0, phi32_0])

    # legt Boundary Condition fest
    def setBoundaryCondition(self):
        for i in range(4):
            self.y0[i * self.Set.N_x] = self.y00[i]

    # merged 2 simulationsobjekte sodass diese hintereinander geplottet werden können
    def mergeSims(self, Sim1, Sim2):
        self.V_dis = np.concatenate((Sim1.V_dis, Sim2.V_dis), axis=1)
        self.V_d = np.concatenate((Sim1.V_d, Sim2.V_d), axis=1)
        self.V_c = np.concatenate((Sim1.V_c, Sim2.V_c), axis=1)
        self.phi_32 = np.concatenate((Sim1.phi_32, Sim2.phi_32), axis=1)
        self.u_dis = np.concatenate((Sim1.u_dis, Sim2.u_dis), axis=1)
        self.u_d = np.concatenate((Sim1.u_d, Sim2.u_d), axis=1)
        self.u_c = np.concatenate((Sim1.u_c, Sim2.u_c), axis=1)
        self.Set.T = Sim1.Set.T + Sim2.Set.T
        t_1 = Sim1.Set.t
        t_2 = Sim2.Set.t + Sim1.Set.t[-1]
        self.Set.t = np.concatenate((t_1, t_2))
        self.Set.N_x = self.V_dis.shape[0]
        self.Set.x = np.linspace(0, self.Set.L, self.Set.N_x)
        self.V_dis_total = np.sum(self.V_dis[:,-1])
        print('dV_ges=', self.Sub.dV_ges, '. phi_32,0=', self.Sub.phi_0, '. V_dis=', self.V_dis_total)
        print('')

    # Funktion für Koaleszenzzeit
    def tau(self, h, d_32, ID, sigma, r_s_star):

        La_mod = (self.Sub.g * self.Sub.delta_rho / sigma) ** 0.6 * d_32 * h ** 0.2

        R_F = d_32 * (1 - (4.7 / (4.7 + La_mod))) ** 0.5

        if ID == "d":
            R_F = 0.3025 * R_F
        else:
            R_F = 0.5240 * R_F

        R_a = 0.5 * d_32 * (1 - (1 - 4.7 / (4.7 + La_mod)) ** 0.5)

        tau = 7.65 * self.Sub.eta_c * (R_a ** (7 / 3)
            / (self.Sub.H_cd ** (1 / 6) * sigma ** (5 / 6) * R_F * r_s_star))

        return tau

    # Funktion zur Berechnung des Quellterms und der TTK-Zeit nach Henschke (mit vereinfachtem TTK-Modell)
    def henschke(self, V_dis, V_d, V_c, phi_32, sigma, r_s_star):

        D = self.Set.D
        dl = self.Set.dl

        dV = np.zeros_like(V_dis)

        tau_di = 9e9 * np.ones_like(V_dis)  # Koaleszenzzeit hoch gewählt, damit quasi keine stattfindet, wenn V_dis < 0
        tau_dd = tau_di

        for i in range(len(V_dis)):

            if phi_32[i] <= 0:
                phi_32[i] = self.Sub.phi_0 /10
                # print('Sauter kleiner Null')

            if V_dis[i] > 0:
                h_p = max(D - hf.getHeight(V_d[i] / dl, D / 2) - hf.getHeight(V_c[i] / dl, D / 2), 0.0001)
                if (h_p<0):
                    print(h_p)
                tau_di[i] = self.tau(h_p, phi_32[i], 'i', sigma[i], r_s_star[i])
                tau_dd[i] = self.tau(self.Sub.h_p_star * h_p, phi_32[i], 'd', sigma[i], r_s_star[i])
                dV[i] = 2 * self.Sub.eps_di * D * phi_32[i] * dl / (3 * tau_di[i] * self.Sub.eps_p)

        return dV, tau_dd

    # Strategie zur Bestimmung der konvektiven Geschwindigkeiten, funktioniert aber nicht richtig
    # -> keine Verwendung in der Simulation außer wenn veloConst auf False gesetzt wird
    def velocities(self, V_dis, V_d, V_c, t, report=False, balance=False):
        dl = self.Set.dl
        eps_0 = self.Sub.eps_0
        eps_p = self.Sub.eps_p
        dV_ges = self.Sub.dV_ges
        u_0 = (self.Sub.dV_ges / (np.pi * self.Set.D**2 / 4))
        self.u_0 = u_0        
        # u_dis = np.linspace(u_0,0,len(V_dis))                           # Option 1 (Triangle)
        # u_dis = u_0 * (1 - np.linspace(0, 1, len(V_dis))**2)            # Option 2 (Parabola) u_dis''<0
        # u_dis = u_0 * (np.linspace(1, 0, len(V_dis))**2)                # Option 3 (Parabola) u_dis''>0
        u_dis = u_0 * np.cos(np.linspace(0, np.pi/2, self.Set.N_x))     # Option 4 (Cosinus) u_dis''<0
        u_dis[-1] = 0
        A_dis = V_dis / dl
        A_d = V_d / dl
        A_c = V_c / dl
        u_d = u_0 * np.ones(len(V_dis))
        u_c = u_0 * np.ones(len(V_dis))

        if not hasattr(self, "_last_velocities"):
            self._last_velocities = {}

        # if (abs(t % dt) < 1e-3 or t==0):
        #     for i in range(1, len(V_dis)):
        #         if V_dis[i] > 0 and V_dis[i] < V_dis[i - 1]:
        #             hp_before = D - hf.getHeight(V_c[i - 1] / dl, D / 2) - hf.getHeight(V_d[i - 1] / dl, D / 2)
        #             hp = D - hf.getHeight(V_c[i] / dl, D / 2) - hf.getHeight(V_d[i] / dl, D / 2)
        #             delta_hp = hp - hp_before
        #             delta_p_dis = eps_p * (1 - eps_p) * self.Sub.delta_rho * self.Sub.g * delta_hp
        #             dV_dis = -1 * delta_p_dis * hp * D ** 3 \
        #                     / ( dl * ( 11.3 * self.Sub.s * self.Sub.eta_dis + 126 * (self.Sub.eta_d + self.Sub.eta_c) ) )
        #             dV_d = self.Sub.eps_0 * self.Sub.dV_ges - eps_p * dV_dis
        #             dV_c = self.Sub.dV_ges - dV_dis - dV_d

        #             u_d[i] = dV_d * dl / V_d[i]
        #             u_c[i] = dV_c * dl / V_c[i]
        #             u_dis[i] = dV_dis * dl / V_dis[i]
        #     self._last_velocities['u_dis'] = u_dis
        #     self._last_velocities['u_d'] = u_d
        #     self._last_velocities['u_c'] = u_c

        if ((t==0)):
            for i in range(len(V_dis)):
                u_d[i] = (eps_0*dV_ges - u_dis[i]*A_dis[i]*eps_p)/(A_d[i])
                u_c[i] = ((1 - eps_0)*dV_ges - (1 - eps_p)*u_dis[i] * A_dis[i]) / A_c[i]
            self._last_velocities['u_dis'] = u_dis
            self._last_velocities['u_d'] = u_d
            self._last_velocities['u_c'] = u_c
            self.u_dis.append(u_dis)
            self.u_d.append(u_d)
            self.u_c.append(u_c)

        else:
            u_dis = self._last_velocities.get('u_dis', np.zeros_like(V_dis))
            u_d   = self._last_velocities.get('u_d',   np.zeros_like(V_d))
            u_c   = self._last_velocities.get('u_c',   np.zeros_like(V_c))
            self.u_dis.append(u_dis)
            self.u_d.append(u_d)
            self.u_c.append(u_c)

        if (balance):
            u_dis = u_dis[-1]
            u_d = (eps_0*dV_ges)/(A_d[-1])
            u_c = ((1 - eps_0)*dV_ges) / A_c[-1]

        # if report:
        #     print('hp_i: '+str(hp))
        #     print('hp_i-1: '+str(hp_before))
        #     print('delta_hp: ' + str(delta_hp))
        #     print('delta_p_dis: ' + str(delta_p_dis))
        #     print('Volumenstromverhältnis dis: ' + str(dV_dis / self.Sub.dV_ges))
        #     print('Volumenstromverhältnis d: ' + str(dV_d / self.Sub.dV_ges))
        #     print('Volumenstromverhältnis c: ' + str(dV_c / self.Sub.dV_ges))
        #     print('Geschwindigkeitsverhältnis dis: ' +str(u_dis[i] / self.u0))
        #     print('Geschwindigkeitsverhältnis d: ' + str(u_d[i] / self.u0))
        #     print('Geschwindigkeitsverhältnis c: ' + str(u_c[i] / self.u0))
        #     print('-------------------------------------------')

        return u_dis, u_d, u_c

    # Funktion zur Simulation mit solve_ivp
    def simulate_ivp(self, veloConst=True, atol=1e-6):
        
        y = []
        N_x = self.Set.N_x
        dl = self.Set.dl
        eps_p = self.Sub.eps_p
        sigma = self.Sub.sigma * np.ones(N_x)
        r_s_star = self.Sub.r_s_star * np.ones(N_x)
        D = self.Set.D

        a_tol = np.concatenate([atol*np.ones(N_x),              # V_dis
                               atol*np.ones(N_x),               # V_d
                               atol*np.ones(N_x),               # V_c
                               atol*np.ones(N_x),])             # phi_32

        
        r_tol = atol*1e3

        def event(t, y):
            return np.min(y[:N_x])  # event stops integration when V_dis<0
        event.terminal = True

        def fun(t, y):
            V_dis = y[: N_x]
            V_d = y[N_x: 2 * N_x]
            V_c = y[2 * N_x: 3 * N_x]
            phi_32 = y[3 * N_x:]

            dV, tau_dd = self.henschke(V_dis, V_d, V_c, phi_32, sigma, r_s_star)

            if veloConst == False:
                u_dis, u_d, u_c = self.velocities(V_dis, V_d, V_c, t)
            else:
                u_dis = self.u0 * np.ones_like(V_dis)
                u_d = u_dis
                u_c = u_dis
                self.u_dis.append(u_dis)
                self.u_d.append(u_d)
                self.u_c.append(u_c)

            dVdis_dt = - 1 / dl * (u_dis * (V_dis - np.roll(V_dis, 1)) +
                                   V_dis * (u_dis - np.roll(u_dis, 1))) - dV
            # dVd_dt = - 1 / dl * (u_d * (V_d - np.roll(V_d, 1)) +
            #                        V_d * (u_d - np.roll(u_d, 1))) + eps_p * dV
            
            dVc_dt = - 1 / dl * (u_c * (V_c - np.roll(V_c, 1)) +
                                 V_c * (u_c - np.roll(u_c, 1))) + (1 - eps_p) * dV
            dVd_dt = -dVdis_dt - dVc_dt
            dphi32_dt = -u_dis / dl * (phi_32 - np.roll(phi_32, 1)) + (phi_32 / dl) * (np.roll(u_dis, 1) - u_dis) + phi_32 / (6 * tau_dd)


            dVdis_dt[0] = 0
            dVd_dt[0] = 0
            dVc_dt[0] = 0
            dphi32_dt[0] = 0

            return np.concatenate([dVdis_dt, dVd_dt, dVc_dt, dphi32_dt])

        # Lösung des Systems von ODEs
        self.sol = solve_ivp(fun, (0, self.Set.T), self.y0, t_eval=self.Set.t, rtol=r_tol, atol=a_tol, method='RK45', events=event)

        y = self.sol.y
        self.V_dis = y[: N_x]
        self.V_d = y[N_x: 2 * N_x]
        self.V_c = y[2 * N_x: 3 * N_x]
        self.phi_32 = y[3 * N_x:]
        self.Set.t = self.sol.t

        print(self.sol.message)
        print('Simulation ends at t = ' + str(self.Set.t[-1]) + ' s')
        self.vol_balance = hf.calculate_volume_balance(self)
        self.V_dis_total = np.sum(self.V_dis[:, -1])
        self.cfl = hf.calculate_cfl(self)
        print('V_dis_tot =', self.V_dis_total , 'm3', '. Volume imbalance = ',self.vol_balance ,'%')
        print('N_x:', self.Set.N_x, '. CFL number: ', self.cfl)
        h_c = getHeightArray(self.V_c[:, len(self.Set.t) - 1]/self.Set.dl, self.Set.D/2)
        h_c_dis = getHeightArray((self.V_c[:, len(self.Set.t) - 1] + self.V_dis[:, len(self.Set.t) - 1])/self.Set.dl, self.Set.D/2)
        h_dis = (np.max(h_c_dis) - np.min(h_c))
        self.H_DPZ = h_dis
        # print('Height of the DPZ at the end of the simulation: ', 1000 * self.H_DPZ , ' mm')
        a = np.where(np.abs(h_c_dis - h_c) < 1e-3)[0][0] if np.any(np.abs(h_c_dis - h_c) < 1e-3) else -1
        self.L_DPZ = a * self.Set.dl
        # print('Length of the DPZ at the end of the simulation: ', 1000 * self.L_DPZ, ' mm')
        self.h_dpz = h_c_dis
        self.h_c = h_c


    # Simulation mittels Upwind-Verfahren
    # adjust_dl erhöht die Anzahl zeitlicher Gitterpunkte, wenn Instabilität wegen zu hoher CFL-Zahl auftritt
    # sigma und rs_change sind notwendig für die Simulation der Sprungantwort, damit der Sprung entlang der
    # Abscheiderlänge zu unterschiedlichen Zeitpunkten je nach konvektiver Geschwindigkeit stattfindet
    def simulate_upwind(self, veloConst=True, adjust_dl=True, sigmaChange=False, rsChange=False):

        repeat = True

        while repeat:

            if adjust_dl == False:
                repeat = False

            V_dis_calc = []
            V_d_calc = []
            V_c_calc = []
            u_dis_calc = []
            u_d_calc = []
            u_c_calc = []
            phi_32_calc = []

            dl = self.Set.dl
            N_x = self.Set.N_x
            dt = self.Set.dt
            y0 = self.y0
            T = self.Set.T
            eps_p = self.Sub.eps_p
            sigma = self.Sub.sigma * np.ones(N_x)
            r_s_star = self.Sub.r_s_star * np.ones(N_x)

            V_dis_calc.append(y0[: N_x])
            V_d_calc.append(y0[N_x: 2 * N_x])
            V_c_calc.append(y0[2 * N_x: 3 * N_x])
            u_dis_calc.append(self.u0*np.ones(N_x))
            u_d_calc.append(self.u0*np.ones(N_x))
            u_c_calc.append(self.u0*np.ones(N_x))
            phi_32_calc.append(y0[3 * N_x:])

            idx = 0
            t = 0

            while t < T:

                idx += 1
                t += dt

                if rsChange:
                    r_s_star = self.Sub.r_s_star * np.ones(N_x)
                    rs_limit = self.u0 * t
                    if self.Set.x[-1] > rs_limit:
                        idx_rs_change = np.where(self.Set.x >= rs_limit)[0][0]
                        r_s_star[idx_rs_change:] = self.rs_before

                if sigmaChange:
                    sigma = self.Sub.sigma * np.ones(N_x)
                    sigma_limit = self.u0 * t
                    if self.Set.x[-1] > sigma_limit:
                        idx_sigma_change = np.where(self.Set.x >= sigma_limit)[0][0]
                        sigma[idx_sigma_change:] = self.sigma_before

                dV, tau_dd = self.henschke(V_dis_calc[idx-1], V_d_calc[idx-1], V_c_calc[idx-1], phi_32_calc[idx-1],
                                           sigma, r_s_star)

                u_dis = self.u0 * np.ones_like(V_dis_calc[idx-1])
                u_d = u_dis
                u_c = u_dis

                if veloConst == False: # veloConst eigentlich immer True, sonst läuft Simulation nicht
                    u_dis, u_d, u_c = self.velocities(V_dis_calc[idx-1], V_d_calc[idx-1], V_c_calc[idx-1])

                dVdis_dt = - 1 / dl * (u_dis * (V_dis_calc[idx-1] - np.roll(V_dis_calc[idx-1], 1)) +
                                   V_dis_calc[idx-1] * (u_dis - np.roll(u_dis, 1))) - dV
                dVd_dt = - 1 / dl * (u_d * (V_d_calc[idx-1] - np.roll(V_d_calc[idx-1], 1)) +
                                 V_d_calc[idx-1] * (u_d - np.roll(u_d, 1))) + eps_p * dV
                dVc_dt = - 1 / dl * (u_c * (V_c_calc[idx-1] - np.roll(V_c_calc[idx-1], 1)) +
                                 V_c_calc[idx-1] * (u_c - np.roll(u_c, 1))) + (1 - eps_p) * dV

                dphi32_dt = (-u_dis / dl * (phi_32_calc[idx-1] - np.roll(phi_32_calc[idx-1], 1))
                         + phi_32_calc[idx-1] / (6 * tau_dd))

                dVdis_dt[0] = 0
                dVd_dt[0] = 0
                dVc_dt[0] = 0
                dphi32_dt[0] = 0

                V_dis_append = np.zeros_like(dVdis_dt)
                V_d_append = np.zeros_like(dVdis_dt)
                V_c_append = np.zeros_like(dVdis_dt)
                phi_32_append = np.zeros_like(dVdis_dt)
                for i in range(len(dVdis_dt)):
                    V_dis_new = V_dis_calc[idx-1][i] + dt * dVdis_dt[i]
                    if V_dis_new < 1e-8:
                    # if False:
                        V_dis_append[i] = 1e-12
                        V_d_append[i] = V_d_calc[idx-1][i]
                        V_c_append[i] = V_c_calc[idx - 1][i]
                        phi_32_append[i] = phi_32_calc[idx-1][i]
                    else:
                        V_dis_append[i] = V_dis_new
                        V_d_append[i] = V_d_calc[idx - 1][i] + dt * dVd_dt[i]
                        V_c_append[i] = V_c_calc[idx - 1][i] + dt * dVc_dt[i]
                        phi_32_append[i] = phi_32_calc[idx - 1][i] + dt * dphi32_dt[i]

                V_dis_calc.append(V_dis_append)
                V_d_calc.append(V_d_append)
                V_c_calc.append(V_c_append)
                phi_32_calc.append(phi_32_append)

                u_dis_calc.append(u_dis)
                u_d_calc.append(u_d)
                u_c_calc.append(u_c)

            # auf diese Weise wird die Instabilität erkannt, da bei Instabilität ein unphysikalischer Anstieg der
            # Höhe der DGTS auftritt
            decreasing_dpz = np.all(np.diff(V_dis_calc[-1]) <= 0)
            if decreasing_dpz == False and adjust_dl:
                # self.Set.reduce_Nx()
                self.Set.set_Nt(100+self.Set.N_t)
                self.calcInitialConditions()
            else:
                repeat = False

        # Endergebnisse werden auf diese Weise in die gleiche Form wie die Endergebnisse der solve_ivp-Funktion gebracht
        self.V_dis = np.vstack(V_dis_calc).T
        self.V_d = np.vstack(V_d_calc).T
        self.V_c = np.vstack(V_c_calc).T
        self.u_dis = np.vstack(u_dis_calc).T
        self.u_d = np.vstack(u_d_calc).T
        self.u_c = np.vstack(u_c_calc).T
        self.phi_32 = np.vstack(phi_32_calc).T
        self.Set.T = t
        self.Set.t = np.arange(0, t, dt)

        print('Simulation ends at t = '+ str(t) + ' s')
        print('Shape of resulting arrays: '+str(np.shape(self.V_dis)))

    def plot_anim(self, plots): # consists of keys for plotting functions
        from helper_functions import getHeightArray

        # Figure erzeugen
        x = []
        y = []
        if len(plots) > 1:
            fig, axes = plt.subplots(len(plots), 1, figsize=(9, 6))
            for i in range(len(plots)):
                axes[i].plot(x, y)
        else:
            fig, ax = plt.subplots()
            ax.plot(x, y)

        # Variablen definieren
        V_dis = self.V_dis
        V_d = self.V_d
        V_c = self.V_c
        u_dis = self.u_dis
        u_d = self.u_d
        u_c = self.u_c
        dl = self.Set.dl
        D = self.Set.D
        t = self.Set.t
        x = self.Set.x
        h_p_star = self.Sub.h_p_star
        phi_32 = self.phi_32
        light_in_heavy = self.Sub.light_in_heavy

        x*=1000


        ################################################ Plotting-Funktionen (teils sehr kompliziert geschrieben)

        # animierter Plot ausgewählter Variablen (funktioniert, indem eine Liste mit keys (strings) übergeben wird)
        def plot_anim_step(key, ax, frame, i):
            ax.cla()

            if i == 0:
                ax.set_title('Zeit = {:.2f}'.format(t[frame]) + 's')

            if key == 'velo':

                ax.plot(x, u_dis[frame], color='r', label='dpz')
                ax.plot(x, u_d[frame], color='b', label='disp phase')
                ax.plot(x, u_c[frame], color='g', label='conti phase')
                ax.plot(x, self.u0 * np.ones_like(u_dis[frame]), linestyle='--', color='black', label='u0')

                ax.set_xlabel('x in mm')
                ax.set_ylabel('Geschwindigkeit in m/s')
                ax.set_xlim(0, x[-1])

            if key == 'sauter':
                idx_no_dis = self.Set.N_x
                if min(V_dis[:,frame]) < 1e-8:
                    idx_no_dis = np.where(V_dis[:, frame] < 1e-8)[0][0]
                ax.plot(x[:idx_no_dis], phi_32[:idx_no_dis, frame] * 1000, label='t', color='b')

                idx_no_dis = self.Set.N_x
                if min(V_dis[:, 0]) < 1e-8:
                    idx_no_dis = np.where(V_dis[:, 0] < 1e-8)[0][0]
                ax.plot(x[:idx_no_dis], phi_32[:idx_no_dis, 0] * 1000, label='t = 0', color='g', linestyle='--')

                idx_no_dis = self.Set.N_x
                if min(V_dis[:, len(t) - 1]) < 1e-8:
                    idx_no_dis = np.where(V_dis[:, len(t) - 1] < 1e-8)[0][0]
                ax.plot(x[:idx_no_dis], phi_32[:idx_no_dis, len(t) - 1] * 1000, label='t = {:.2f}'.format(t[len(t) - 1]),
                             color='r', linestyle='--')

                ax.set_xlabel('x in mm')
                ax.set_ylabel('Sauterdurchmesser in mm')
                ax.set_ylim(0, np.ceil(1000 * np.max(phi_32)))
                ax.set_xlim(0, x[-1])

            if key == 'heights':

                V_d_dis = V_d + V_dis
                V_tot = V_dis + V_d + V_c
                V_c_dis = V_c + V_dis

                if light_in_heavy:
                    ax.plot(x, 1000 * getHeightArray(V_c[:, 0] / dl, D / 2), color='r', linestyle=':',
                                 label='Interface c, dis; t = 0')
                    ax.plot(x, 1000 * getHeightArray(V_c_dis[:, 0] / dl, D / 2), color='g', linestyle=':',
                                 label='Interface dis, d; t = 0')

                    ax.plot(x, 1000 * getHeightArray(V_c[:, len(t) - 1] / dl, D / 2), color='r',
                                 linestyle='--',
                                 label='Interface c, dis; t = {:.2f}'.format(t[len(t) - 1]))
                    ax.plot(x, 1000 * getHeightArray(V_c_dis[:, len(t) - 1] / dl, D / 2), color='g',
                                 linestyle='--',
                                 label='Interface dis, d; t = {:.2f}'.format(t[len(t) - 1]))

                    ax.plot(x, 1000 * getHeightArray(V_c[:, frame] / dl, D / 2), color='r',
                                 label='Interface c, dis')
                    ax.plot(x, 1000 * getHeightArray(V_c_dis[:, frame] / dl, D / 2), color='g',
                                 label='Interface dis, d')
                else:
                    ax.plot(x, 1000 * getHeightArray(V_d[:, 0] / dl, D / 2), color='r', linestyle=':',
                                 label='Interface d, dis; t = 0')
                    ax.plot(x, 1000 * getHeightArray(V_d_dis[:, 0] / dl, D / 2), color='g', linestyle=':',
                                 label='Interface dis, c; t = 0')

                    ax.plot(x, 1000 * getHeightArray(V_d[:, len(t) - 1] / dl, D / 2), color='r',
                                 linestyle='--',
                                 label='Interface d, dis; t = {:.2f}'.format(t[len(t) - 1]))
                    ax.plot(x, 1000 * getHeightArray(V_d_dis[:, len(t) - 1] / dl, D / 2), color='g',
                                 linestyle='--',
                                 label='Interface dis, c; t = {:.2f}'.format(t[len(t) - 1]))

                    ax.plot(x, 1000 * getHeightArray(V_d[:, frame] / dl, D / 2), color='r',
                                 label='Interface d, dis')
                    ax.plot(x, 1000 * getHeightArray(V_d_dis[:, frame] / dl, D / 2), color='g',
                                 label='Interface dis, c')

                ax.plot(x, 1000 * getHeightArray(V_tot[:, frame] / dl, D / 2), color='b', label='h_tot')

                ax.set_xlabel('x in mm')
                ax.set_ylabel('Höhe in mm')
                ax.set_xlim(0, x[-1])

            if key == 'tau':

                hp = D - getHeightArray(V_d[:, frame] / dl, D / 2) - getHeightArray(V_c[:, frame] / dl, D / 2)

                if np.where(hp[:-1] == hp[1:])[0].size > 0:
                    last_idx = np.where(hp[:-1] == hp[1:])[0][0]
                    for k in range(last_idx, len(x)):
                        hp[k] = hp[k-1]

                tau_di = np.zeros_like(x)
                tau_dd = np.zeros_like(x)
                for k in range(len(x)):
                    if hp[k] < D / 1e5: # Obergrenze, damit keine unendlich großen Koaleszenzzeiten auftreten
                        hp[k] = D / 1e5
                    tau_di[k] = self.tau(hp[k], phi_32[k, frame], 'i')
                    tau_dd[k] = self.tau(h_p_star*hp[k], phi_32[k, frame], 'd')

                ax.plot(x, tau_di, label='tau_di', color='b')
                ax.plot(x, tau_dd, label='tau_dd', color='r')

                ax.set_xlabel('x in mm')
                ax.set_ylabel('Koaleszenzzeit in s')
                ax.set_xlim(0, x[-1])
                ax.set_ylim(0, 10)

        def update(frame):

            if len(plots) > 1:
                for i in range(len(plots)):
                    plot_anim_step(plots[i], axes[i], frame, i)
                    axes[i].legend(loc='upper left', bbox_to_anchor=(1, 1))
            else:
                plot_anim_step(plots[0], ax, frame, 0)
                ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
            plt.tight_layout()

        anim = FuncAnimation(plt.gcf(), update, frames=range(len(t)), interval=10)

        plt.show()

        x /= 1000

    # Plotting-Funktionen für ein zusammengeführtes Simulationsobjekt (Sprungantworten)
    def plot_merged_sim(self, t_plus, times, labels, title='title'):
        from helper_functions import getHeightArray

        fig, axes = plt.subplots(2, 1, figsize=(6.5, 5.5))

        # Variablen definieren
        V_dis = self.V_dis
        V_d = self.V_d
        V_c = self.V_c
        V = self.Set.delta_V
        dl = self.Set.dl
        dt = self.Set.dt
        D = self.Set.D
        t = self.Set.t - t_plus
        x = self.Set.x
        N_x = self.Set.N_x
        h_p_star = self.Sub.h_p_star
        phi_32 = self.phi_32
        light_in_heavy = self.Sub.light_in_heavy

        x *= 1000

        colors = ['r', 'm', 'g', 'orange', 'b']
        V_d_dis = V_d + V_dis
        V_c_dis = V_c + V_dis

        def plot(axes, color, style, x_indices, time_idx, label=None):
            if light_in_heavy:
                axes[0].plot(x[x_indices], 1000 * getHeightArray(V_c[x_indices, time_idx] / dl, D / 2), color=color,
                             linestyle=style, label=label)
                axes[0].plot(x[x_indices], 1000 * getHeightArray(V_c_dis[x_indices, time_idx] / dl, D / 2),
                             linestyle=style, color=color)
            else:
                axes[0].plot(x[x_indices], 1000 * getHeightArray(V_d[x_indices, time_idx] / dl, D / 2), color=color,
                             linestyle=style, label=label)
                axes[0].plot(x[x_indices], 1000 * getHeightArray(V_d_dis[x_indices, time_idx] / dl, D / 2),
                             linestyle=style, color=color)
            axes[1].plot(x[x_indices], phi_32[x_indices, time_idx] * 1000,
                         linestyle=style, label=label, color=color)

        for i in range(len(times)):
            idx = np.where(t >= times[i])[0][0]
            if i > 0 and i < len(times) - 1:
                style = '--'
            else:
                style = '-'

            # Dieses Konstrukt ist dafür da, einen möglichen sekundären Dispersionskeil darzustellen und die Darstellung
            # verschwindend geringer DGTS Volumnina wegzulassen (diese sind nämlich nur der Abbruchbedingung geschuldet)
            change_index1 = 0
            change_index2 = 0
            change_index3 = 0
            mask = V_dis[:, idx] > 1e-8
            diff_mask = np.diff(mask.astype(int))
            if np.where(diff_mask == -1)[0].size > 0:
                change_index1 = np.where(diff_mask == -1)[0][0] + 1  # 1. Wechsel von True (1) auf False (0), V_dis geht auf Null
                if np.where(diff_mask == -1)[0].size > 1:
                    change_index3 = np.where(diff_mask == -1)[0][1] + 1 # V_dis erreicht 2. Mal die Null
                if i == 0:
                    change_index_for_first_plot = change_index1
                    time_idx_for_first_plot = idx
            if np.where(diff_mask == 1)[0].size > 0:
                change_index2 = np.where(diff_mask == 1)[0][0]      # Wechsel von False (0) auf True (1), V_dis geht von Null hoch
            if change_index1 == 0:
                plot(axes, colors[i], style, np.arange(N_x), idx, label=labels[i])
            elif change_index2 == 0:
                plot(axes, colors[i], style, np.arange(N_x)[:change_index1], idx, label=labels[i])
            elif change_index3 == 0:
                plot(axes, colors[i], style, np.arange(N_x)[:change_index1], idx, label=labels[i])
                plot(axes, colors[i], style, np.arange(N_x)[change_index2:], idx)
            else:
                plot(axes, colors[i], style, np.arange(N_x)[:change_index1], idx, label=labels[i])
                plot(axes, colors[i], style, np.arange(N_x)[change_index2:change_index3], idx)

        # t=0s Plot wird wiederholt um als Steady State gestrichelte Linien zu überdecken
        if t_plus > 0:
            plot(axes, colors[0], '-', np.arange(N_x)[:change_index_for_first_plot], time_idx_for_first_plot)

        axes[0].set_ylabel('Höhe / mm', size=12)
        axes[0].set_ylim(bottom=50)
        # axes[0].set_xlim(0, x[-1])
        axes[0].set_xlim(0, 500)
        axes[0].tick_params(axis='x', top=True, direction='in')
        axes[0].tick_params(axis='y', right=True, direction='in')
        # axes[0].legend(loc='upper center', bbox_to_anchor=(0.5, 1.2), ncol=6, frameon=False)

        axes[0].legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=6, frameon=False)

        axes[1].set_xlabel('Länge / mm', size=12)
        axes[1].set_ylabel('Sauterdurchmesser / mm', size=12)
        axes[1].set_ylim(0, np.ceil(1000 * np.max(phi_32)))
        axes[1].set_xlim(0, 500)
        axes[1].tick_params(axis='x', top=True, direction='in')
        axes[1].tick_params(axis='y', right=True, direction='in')
        # axes[1].set_xlim(0, x[-1])
        # axes[1].legend(loc='upper left', bbox_to_anchor=(1, 1))

        plt.suptitle('Durchmesser des Abscheiders = 100$\,$mm')
        plt.tight_layout()

        plt.show()

    # Plottet Dispersionskeillänge
    def plot_separation_length(self):
        fig, ax = plt.subplots(figsize=(3.5, 5.25))
        V_dis = self.V_dis
        x = self.Set.x
        t = self.Set.t
        lengths = []
        for i in range(len(t)):
            sep_idx = len(x)
            if np.where(V_dis[:, i] < 1e-8)[0].size > 0:
                sep_idx = np.where(V_dis[:, i] < 1e-8)[0][0]
            lengths.append(x[sep_idx])
            #ax.scatter(t[i], 1000 * x[sep_idx], marker='x', color='k')
        ax.plot(t, np.array(lengths) * 1000, color='b')
        ax.set_xlabel('Zeit / s', size=12)
        ax.set_xlim(0, 100)
        ax.set_ylim(0, 500)
        ax.set_ylabel('Länge des Dispersionskeils / mm', size=12)
        plt.tick_params(axis='x', top=True, direction='in')
        plt.tick_params(axis='y', right=True, direction='in')
        plt.tight_layout()
        plt.show()

    # Berechnet das gesamte Dispersionsvolumen am Ende der Simulation (Sensitivitätsanalyse)
    def calc_Vdis_tot(self):
        V_dis_end = self.V_dis[:, -1]
        if np.where(V_dis_end < 1e-8)[0].size > 0:
            last_idx = np.where(V_dis_end < 1e-8)[0][0]
            V_dis_end = V_dis_end[:last_idx]
        Vdis_tot = np.sum(V_dis_end) * 1000
        return Vdis_tot

    # berechnet und plottet DGTS-Höhe (für Dispersionskeilplot)
    def calc_comparison(self, ax, i, single, labels, henschkeData):

        from helper_functions import getHeightArray

        colors = ['b', 'r', 'g', 'm', 'k', 'y', 'c']
        markers = ['o', 'x', '^', '*', 'v', 's', '.']

        D = self.Set.D
        dl = self.Set.dl
        x = self.Set.x

        # Array mit h_dis(t_end) bestimmen
        Vd_end = self.V_d[:, -1]
        Vc_end = self.V_c[:, -1]
        hp = D - getHeightArray(Vd_end / dl, D / 2) - getHeightArray(Vc_end / dl, D / 2)

        if np.where(hp < 1e-4)[0].size > 0:
            last_idx = np.where(hp < 1e-4)[0][0]
            hp = hp[:last_idx]
            x = x[:last_idx]

        # Alle Werte ab hp < 1e-5 wegschneiden
        # if np.where(hp[:-1] == hp[1:])[0].size > 0:
        #     last_idx = np.where(hp[:-1] == hp[1:])[0][0]
        #     hp = hp[:last_idx]
        #     x = x[:last_idx]

        # x und hp in mm
        x *= 1000
        hp *= 1000

        # Modellierungsdaten von Henschke
        x_sim = self.Sub.x_sim
        hp_sim = self.Sub.h_p_sim

        # Experimentelle Daten von Henschke
        x_exp = self.Sub.x_exp
        hp_exp = self.Sub.h_p_exp

        # Bestimmung von 'schönen' Achsenwerten
        xmax = max([x_sim[-1], x[-1], x_exp[-1]])
        ymax = max([hp_sim[0], hp_exp[0], hp[0]])
        xmax = np.ceil(xmax / 50) * 50
        ymax = np.ceil(ymax / 5) * 5

        # Plotten der Ergebnisse
        if single:
            ax.plot(x, hp, label='eigene Modellierung', color='r')
            if henschkeData:
                ax.plot(x_sim, hp_sim, label='Henschke Modellierung', color='g')
                ax.scatter(x_exp, hp_exp, label='experimentelle Daten', marker='x', color='b')
        else:
            if henschkeData:
                ax.plot(x, hp, color=colors[i])
                ax.plot(x_sim, hp_sim, linestyle='--', color=colors[i])
                ax.scatter(x_exp, hp_exp, marker=markers[i], color=colors[i], label=labels[i])
            else:
                ax.plot(x, hp, label=labels[i], color=colors[i])

        x /= 1000  # x zurück in m umrechnen

        return xmax, ymax

# Kriegt mehrere Simulationsobjekte in Liste übergeben und ruft einzeln calc_comparison auf, um diese zu plotten
def plot_comparison(Sims, labels=['1','2','3','4', '5'], legend_title=None, title=None,
                    henschkeData=True, xlim=None, figsize=(8,6)):

    fig, ax = plt.subplots(figsize=figsize)
    xmax = 0
    ymax = 0
    single = False
    if len(Sims) == 1:
        single = True
    for i in range(len(Sims)):
        xmax_new, ymax_new = Sims[i].calc_comparison(ax, i, single, labels, henschkeData)
        if xmax_new > xmax:
            xmax = xmax_new
        if ymax_new > ymax:
            ymax = ymax_new
    if title != None:
        ax.set_title(title)
    # ax.set_xlim(0, xmax)
    if xlim == None:
        ax.set_xlim(0, Sims[0].Set.L * 1000)
    else:
        ax.set_xlim(xlim)
    ax.set_ylim(0, ymax)
    ax.set_xlabel('Abscheiderlänge / mm', size=12)
    ax.set_ylabel('Höhe der DGTS / mm', size=12)
    plt.tick_params(axis='x', top=True, direction='in')
    plt.tick_params(axis='y', right=True, direction='in')
    if legend_title == None:
        plt.legend(frameon=False)
    else:
        plt.legend(title = legend_title, frameon=False)

    plt.tight_layout()
    plt.show()

# erstellt den Sensitivitätsplot
def plot_sensitivity(Sims, parameters, title=None, x_label='x-Achse', xlim=None):
    fig, ax = plt.subplots(figsize=(5, 4.4))
    V_tot = np.zeros(len(parameters))
    colors = ['b', 'r', 'g', 'm', 'k', 'y', 'c']
    for i in range(len(Sims)):
        Vdis_end = Sims[i].V_dis[:, -1]
        if np.where(Vdis_end < 1e-8)[0].size > 0:
            last_idx = np.where(Vdis_end < 1e-8)[0][0]
            Vdis_end = Vdis_end[:last_idx]
        Vdis_tot = np.sum(Vdis_end) * 1000
        V_tot[i] = Vdis_tot
        ax.scatter(parameters[i], Vdis_tot, color=colors[i])
    ax.set_xlabel(x_label, size=12)
    ax.set_ylabel('Gesamtvolumen der DGTS / L', size=12)
    if xlim == None:
        ax.set_xlim(left=0)
    else:
        ax.set_xlim(xlim)
    ax.set_ylim(0, 1)
    if title != None:
        ax.set_title(title)
    plt.tick_params(axis='x', direction='in', top=True)
    plt.tick_params(axis='y', right=True, direction='in')
    plt.tight_layout()
    plt.show()