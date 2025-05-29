#### Diese Datei enthält alle Plots aus der Arbeit; in sim_run hineinkopieren!

# Dispersionskeile Butylacetat
filenames = ['Butylacetat_5_6_330.xlsx', 'Butylacetat_5_6_220.xlsx',
             'Butylacetat_1_5_216.xlsx', 'Butylacetat_1_5_144.xlsx']
labels = ['0,455, 330, 1,011', '0,455, 220, 1,213',
          '0,167, 216, 0,710', '0,167, 144, 0,674']
title = r'$r_\mathrm{s}^*=0$,0333, $h_\mathrm{p}^*=0$,05488'
legend_title=r'$\epsilon_0$ / -, $\dot{V}_\mathrm{Ges}$ / $\mathrm{L\,h^{-1}}$, $\Phi_{32,0}$ / $\mathrm{mm}$:'
comp_plots(filenames, labels, legend_title=legend_title, title=title)

# Dispersionskeile MiBK (Aufgeteilt in 2 Plots)
filenames = ['MiBK_1_1_360.xlsx', 'MiBK_1_1_288.xlsx', 'MiBK_1_1_216.xlsx', 'MiBK_1_1_144.xlsx']
labels = ['360, 0,97', '288, 1.04', '216, 1.32', '144, 1.86']
title = r'$r_\mathrm{s}^*=0$,0387, $h_\mathrm{p}^*=0$,00406, $\epsilon_0=0$,50'
legend_title=r'$\dot{V}_\mathrm{Ges}$ / $\mathrm{L\,h^{-1}}$, $\Phi_{32,0}$ / $\mathrm{mm}$:'
figsize = (5,4.4)
comp_plots(filenames, labels, legend_title=legend_title, title=title, figsize=figsize)

filenames = ['MiBK_1_2_324.xlsx', 'MiBK_1_2_270.xlsx', 'MiBK_1_2_216.xlsx']
labels = ['324, 0,98', '270, 1,22', '216, 1,38']
title = r'$r_\mathrm{s}^*=0$,0387, $h_\mathrm{p}^*=0$,00406, $\epsilon_0=0$,33'
legend_title=r'$\dot{V}_\mathrm{Ges}$ / $\mathrm{L\,h^{-1}}$, $\Phi_{32,0}$ / $\mathrm{mm}$:'
figsize = (5,4.4)
comp_plots(filenames, labels, legend_title=legend_title, title=title, figsize=figsize)

# Dispersionskeile Butanol
filenames = ['Butanol_3_2_250.xlsx', 'Butanol_3_2_150.xlsx', 'Butanol_2_1_180.xlsx']
labels = ['0,40, 250, 0,998', '0,40, 150, 1,019', '0,33, 180, 0,885']
title = r'$r_\mathrm{s}^*=0$,2729, $h_\mathrm{p}^*=0$,001812'
legend_title=r'$\epsilon_0$ / -, $\dot{V}_\mathrm{Ges}$ / $\mathrm{L\,h^{-1}}$, $\Phi_{32,0}$ / $\mathrm{mm}$:'
comp_plots(filenames, labels, legend_title=legend_title, title=title)

######## Sensitivitätsplots
# Change of dV plot
Sims = init_sims('Butylacetat_5_6_220.xlsx', 5)
dV_ges = [150, 200, 250, 300, 500]
for i in range(len(Sims)):
    Sims[i].set_dVges(dV_ges[i])
    Sims[i].calcInitialConditions()
    Sims[i].simulate_upwind()
legend_title = r'$\dot{V}_\mathrm{Ges}$ / $\mathrm{L\,h^{-1}}$:'
labels = ['150', '200', '250', '300', '500']
figsize=(5,4.4)
sm.plot_comparison(Sims, labels=labels, legend_title=legend_title, henschkeData=False, figsize=figsize)
x_label=r'Gesamtvolumenstrom / $\mathrm{L\,h^{-1}}$'
sm.plot_sensitivity(Sims, parameters=dV_ges, x_label=x_label)

# Change of r_s_star plot
Sims = init_sims('Butylacetat_5_6_220.xlsx', 5)
r_s_star = [0.02, 0.03, 0.04, 0.05, 0.1]
for i in range(len(Sims)):
    #Sims[i].Set.set_T(75)
    Sims[i].Sub.r_s_star = r_s_star[i]
    Sims[i].calcInitialConditions()
    Sims[i].simulate_upwind(adjust_dl=True)
    # Sims[i].simulate_ivp()
figsize = (5, 4.4)
legend_title = r'$r_\mathrm{s}^*$ / - :'
labels = ['0,02', '0,03', '0,04', '0,05', '0,1']
sm.plot_comparison(Sims, labels=labels, figsize=figsize, legend_title=legend_title, henschkeData=False)
x_label = r'Koaleszenzparameter $r_\mathrm{s}^*$ / -'
sm.plot_sensitivity(Sims, parameters=r_s_star, x_label=x_label)

# Change of h_p_star plot
Sims = init_sims('Butylacetat_5_6_220.xlsx', 5)
h_p_star = [0.001, 0.01, 0.1, 0.5, 1]
for i in range(len(Sims)):
    #Sims[i].Set.set_T(75)
    Sims[i].Sub.h_p_star = h_p_star[i]
    Sims[i].calcInitialConditions()
    Sims[i].simulate_upwind(adjust_dl=True)
legend_title = r'$h_\mathrm{p}^*$ / - :'
labels = ['0,001', '0,01', '0,1', '0,5', '1']
figsize = (5, 4.4)
sm.plot_comparison(Sims, labels=labels, legend_title=legend_title, figsize=figsize, henschkeData=False)
x_label = r'Koaleszenzparameter $h_\mathrm{p}^*$ / -'
sm.plot_sensitivity(Sims, parameters=h_p_star, x_label=x_label)

# Change of sigma plot
Sims = init_sims('Butylacetat_5_6_220.xlsx', 5)
sigma = [0.001, 0.005, 0.01, 0.02, 0.03]
for i in range(len(Sims)):
    Sims[i].Sub.sigma = sigma[i]
    Sims[i].calcInitialConditions()
    Sims[i].simulate_upwind(adjust_dl=True)
    # Sims[i].simulate_ivp()
legend_title = r'$\sigma$ / $\mathrm{mN\,m^{-1}}$:'
labels = ['1', '5', '10', '20', '30']
figsize = (5, 4.4)
sm.plot_comparison(Sims, labels=labels, legend_title=legend_title, figsize=figsize, henschkeData=False)
sigma = 1000*np.array(sigma)
x_label = r'Oberflächenspannung / $\mathrm{mN\,m^{-1}}$'
sm.plot_sensitivity(Sims, parameters=sigma, x_label=x_label)

# Change of phi_0 plot
Sims = init_sims('Butylacetat_5_6_220.xlsx', 5)
phi_0 = [0.0004, 0.0007, 0.001, 0.0013, 0.0016]
for i in range(len(Sims)):
    Sims[i].Sub.phi_0 = phi_0[i]
    Sims[i].calcInitialConditions()
    Sims[i].simulate_upwind(adjust_dl=True)
    # Sims[i].simulate_ivp()
legend_title = r'$\Phi_{32,0}$ / mm:'
labels = ['0,4', '0,7', '1,0', '1,3', '1,6']
figsize=(5, 4.4)
sm.plot_comparison(Sims, labels=labels, legend_title=legend_title, figsize=figsize, henschkeData=False)
x_label = 'Anfangstropfengröße / mm'
phi_0 = 1000*np.array(phi_0)
sm.plot_sensitivity(Sims, parameters=phi_0, x_label=x_label)

# Change of eps_p plot
Sims = init_sims('Butylacetat_5_6_220.xlsx', 5)
eps_p = [0.5, 0.6, 0.7, 0.8, 0.9]
for i in range(len(Sims)):
    Sims[i].calcInitialConditions()     # Initial Conditions zuerst, damit alle Sims gleiche Boundary Condition
    Sims[i].Sub.eps_p = eps_p[i]
    Sims[i].simulate_upwind()
    # Sims[i].simulate_ivp()
legend_title = r'$\overline{\epsilon}_\mathrm{p}$ / - :'
labels = ['0,5', '0,6', '0,7', '0,8', '0,9']
figsize=(5,4.4)
sm.plot_comparison(Sims, labels=labels, legend_title=legend_title, figsize=figsize, henschkeData=False)
x_label='Hold-Up in der DGTS / -'
sm.plot_sensitivity(Sims, parameters=eps_p, x_label=x_label, xlim=(0,1))

###### Sensitivitätsberechnungen im  Steady State

# Sensitivität rs_star
Sims = init_sims('Butylacetat_5_6_220.xlsx', 4)
p = [0.03, 0.1]
p_list = [0.9*p[0], 1.1*p[0], 0.9*p[1], 1.1*p[1]]
for i in range(4):
    Sims[i].Sub.r_s_star = p_list[i]
    Sims[i].calcInitialConditions()
    Sims[i].simulate_upwind(adjust_dl=True)
calc_sensitivity(Sims, p)

# Sensitivität hp_star
Sims = init_sims('Butylacetat_5_6_220.xlsx', 4)
p = [0.01, 0.5]
p_list = [0.9*p[0], 1.1*p[0], 0.9*p[1], 1.1*p[1]]
for i in range(4):
    Sims[i].Sub.h_p_star = p_list[i]
    Sims[i].calcInitialConditions()
    Sims[i].simulate_upwind(adjust_dl=True)
calc_sensitivity(Sims, p)

# Sensitivität dV_ges
Sims = init_sims('Butylacetat_5_6_220.xlsx', 4)
p = [150, 250]
p_list = [0.9*p[0], 1.1*p[0], 0.9*p[1], 1.1*p[1]]
for i in range(4):
    Sims[i].set_dVges(p_list[i])
    Sims[i].calcInitialConditions()
    Sims[i].simulate_upwind(adjust_dl=True)
calc_sensitivity(Sims, p)

# Sensitivität sigma
Sims = init_sims('Butylacetat_5_6_220.xlsx', 4)
p = [0.005, 0.03]
p_list = [0.9*p[0], 1.1*p[0], 0.9*p[1], 1.1*p[1]]
for i in range(4):
    Sims[i].Sub.sigma = p_list[i]
    Sims[i].calcInitialConditions()
    Sims[i].simulate_upwind(adjust_dl=True)
calc_sensitivity(Sims, p)

# Sensitivität phi_0
Sims = init_sims('Butylacetat_5_6_220.xlsx', 4)
p = [0.0004, 0.0016]
p_list = [0.9*p[0], 1.1*p[0], 0.9*p[1], 1.1*p[1]]
for i in range(4):
    Sims[i].Sub.phi_0 = p_list[i]
    Sims[i].calcInitialConditions()
    Sims[i].simulate_upwind(adjust_dl=True)
calc_sensitivity(Sims, p)

# Sensitivität eps_p
Sims = init_sims('Butylacetat_5_6_220.xlsx', 4)
p = [0.5, 0.9]
p_list = [0.9*p[0], 1.1*p[0], 0.9*p[1], 1.1*p[1]]
for i in range(4):
    Sims[i].calcInitialConditions()
    Sims[i].Sub.eps_p = p_list[i]
    Sims[i].simulate_upwind(adjust_dl=True)
calc_sensitivity(Sims, p)

########### Dynamische Betrachtungen

# Beobachtung, wie sich der Steady State mit der Zeit einstellt
Sim = init_sim('Butylacetat_5_6_220.xlsx')
Sim.calcInitialConditions()
Sim.simulate_upwind(adjust_dl=True)
labels = [r'$t_0=0\,$s', r'$t=15\,$s', r'$t=30\,$s', r'$t=45\,$s', r'$t_\mathrm{E}=52\,$s']
times = [0, 15, 30, 45, 52]
Sim.plot_merged_sim(times=times, t_plus=0, labels=labels)

# Sprungantwort Änderung phi_0
Sims = init_sims('Butylacetat_5_6_220.xlsx', 3)
phi_0_new = 0.0005
Sims[0].calcInitialConditions()
Sims[0].simulate_upwind(adjust_dl=True)
Sims[1].Set.set_Nx(Sims[0].Set.N_x)
Sims[1].getInitialConditions(Sims[0])
Sims[1].y0[3*Sims[1].Set.N_x] = phi_0_new
Sims[1].simulate_upwind(adjust_dl=False)
Sims[2].mergeSims(Sims[0], Sims[1])
#### Keys for animated Plots: 'velo', 'sauter', 'heights', 'tau'
# plots = ['heights', 'sauter']
# Sims[0].plot_anim(plots)
# Sims[1].plot_anim(plots)
# Sims[2].plot_anim(plots)
Sims[1].plot_separation_length()
labels = [r'$t_0=0\,$s', r'$t=0,5\,$s', r'$t=25\,$s', r'$t=50\,$s', r'$t_\mathrm{E}=65\,$s']
times = [0, 0.5, 25, 50, 65]
Sims[2].plot_merged_sim(times=times, t_plus=Sims[1].Set.T, labels=labels)

# Sprungantwort Änderung dV_ges
Sims = init_sims('Butylacetat_5_6_220.xlsx', 3)
dV_ges_new = 150
Sims[0].calcInitialConditions()
Sims[0].simulate_upwind(adjust_dl=True)
Sims[1].Set.set_Nx(Sims[0].Set.N_x)
Sims[1].getInitialConditions(Sims[0])
Sims[1].set_dVges(dV_ges_new)
Sims[1].simulate_upwind(adjust_dl=False)
Sims[2].mergeSims(Sims[0], Sims[1])
# Keys for animated Plots: 'velo', 'sauter', 'heights', 'tau'
# plots = ['heights', 'sauter']
# Sims[0].plot_anim(plots)
# Sims[1].plot_anim(plots)
# Sims[2].plot_anim(plots)
Sims[1].plot_separation_length()
labels = [r'$t_0=0\,$s', r'$t=15\,$s', r'$t=30\,$s', r'$t=45\,$s', r'$t_\mathrm{E}=52\,$s']
times = [0, 15, 30, 45, 52]
Sims[2].plot_merged_sim(times=times, t_plus=Sims[0].Set.T, labels=labels)

# Sprungantwort Änderung sigma
Sims = init_sims('Butylacetat_5_6_220.xlsx', 3)
sigma_new = 0.005
Sims[0].calcInitialConditions()
Sims[0].simulate_upwind(adjust_dl=True)
Sims[1].Set.set_Nx(Sims[0].Set.N_x)
Sims[1].sigma_before = Sims[0].Sub.sigma
Sims[1].Sub.sigma = sigma_new
Sims[1].getInitialConditions(Sims[0])
Sims[1].simulate_upwind(adjust_dl=False, sigmaChange=True)
Sims[2].mergeSims(Sims[0], Sims[1])
# Keys for animated Plots: 'velo', 'sauter', 'heights', 'tau'
# plots = ['heights', 'sauter']
# Sims[0].plot_anim(plots)
# Sims[1].plot_anim(plots)
# Sims[2].plot_anim(plots)
Sims[1].plot_separation_length()
labels = [r'$t_0=0\,$s', r'$t=20\,$s', r'$t=40\,$s', r'$t=60\,$s', r'$t_\mathrm{E}=71\,$s']
times = [0, 20, 40, 60, 71]
Sims[2].plot_merged_sim(times=times, t_plus=Sims[0].Set.T, labels=labels)

# Sprungantwort Änderung r_s_star
Sims = init_sims('Butylacetat_5_6_220.xlsx', 3)
r_s_star_new = 0.06
Sims[0].calcInitialConditions()
Sims[0].simulate_upwind(adjust_dl=True)
Sims[1].Set.set_Nx(Sims[0].Set.N_x)
Sims[1].rs_before = Sims[0].Sub.r_s_star
Sims[1].Sub.r_s_star = r_s_star_new
Sims[1].getInitialConditions(Sims[0])
Sims[1].simulate_upwind(adjust_dl=False, rsChange=True)
Sims[2].mergeSims(Sims[0], Sims[1])
# Keys for animated Plots: 'velo', 'sauter', 'heights', 'tau'
# plots = ['heights', 'sauter']
# Sims[0].plot_anim(plots)
# Sims[1].plot_anim(plots)
# Sims[2].plot_anim(plots)
Sims[1].plot_separation_length()
labels = [r'$t_0=0\,$s', r'$t=15\,$s', r'$t=30\,$s', r'$t=40\,$s', r'$t_\mathrm{E}=47\,$s']
times = [0, 15, 30, 40, 47]
Sims[2].plot_merged_sim(times=times, t_plus=Sims[0].Set.T, labels=labels)

##############################################################################################################
# Aussortierte Plots
# Sprungantwort Änderung Phasenverhältnis
Sims = init_sims('Butylacetat_5_6_220.xlsx', 3)
o_w = [5/6, 1/4]
Sims[0].Sub.set_o_to_w(o_w[0])
Sims[0].calcInitialConditions()
Sims[0].Set.set_T(41)
Sims[0].simulate_upwind(adjust_dl=True)
Sims[1].Set.set_Nx(Sims[0].Set.N_x)
Sims[1].Set.set_T(41)
Sims[1].Sub.set_o_to_w(o_w[1])
Sims[1].calcInitialConditions()  # necessary to get boundary consition for x=0
Sims[1].getInitialConditions(Sims[0])   # necessary to set initial condition for the whole settler
Sims[1].setBoundaryCondition()        # necessary to set boundary condition for x=0
Sims[1].simulate_upwind(adjust_dl=False)
Sims[2].mergeSims(Sims[0], Sims[1])
# Keys for animated Plots: 'velo', 'sauter', 'heights', 'tau'
# plots = ['heights', 'sauter']
# Sims[0].plot_anim(plots)
# Sims[1].plot_anim(plots)
# Sims[2].plot_anim(plots)
labels = ['t_0=0s', 't=20s', 't=40s', 't=60s', 't_end=80s']
times = [0, 20, 41, 61, 82]
Sims[2].plot_merged_sim(times=times, labels=labels)

# Change of phase ratio plot
Sims = init_sims('Butylacetat_5_6_220.xlsx', 4)
o_w = [5/6, 1/2, 1/3, 1/4]
for i in range(len(Sims)):
    Sims[i].Sub.set_o_to_w(o_w[i])
    Sims[i].calcInitialConditions()
    Sims[i].simulate_upwind(adjust_dl=True)
    # Sims[i].simulate_ivp()
title = 'Einfluss des Phasenverhältnisses'
legend_title = 'o/w [-]:'
labels = ['5/6', '1/2', '1/3', '1/4']
sm.plot_comparison(Sims, labels=labels, title=title, legend_title=legend_title, henschkeData=False)