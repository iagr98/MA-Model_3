import model as m
import settings as s

# Diese Datei enthält alle Beispiele

######### all filenames and time-factors
# Butanol
filename = "Henschke_n_Butanol_2_1.xlsx"
time_factor = 0.55

# Butylacetat w in o
filename = "Henschke__3_2_n-Butylacetat_Water.xlsx"
time_factor = 0.15

filename = 'Henschke__4_1_n-Butylacetat_Water.xlsx'
time_factor = 0.175

# Butylacetat o in w
filename = "Henschke__2_3_n-Butylacetat_Water.xlsx"
time_factor = 0.2

filename = "Henschke__1_4_n-Butylacetat_Water.xlsx"
time_factor = 0.2

#MiBK o/w = 1/2
filename = "Henschke_5_2_MiBK_400.xlsx"
time_factor = 0.5

filename = "Henschke_5_2_MiBK_200.xlsx"
time_factor = 0.35

#MiBK w in o
filename = 'Henschke_5_5_MiBK_2_3.xlsx'
time_factor = 1.0

filename = 'Henschke_5_5_MiBK_3_2.xlsx'
time_factor = 1.0

filename = 'Henschke_5_5_MiBK_3_1.xlsx'
time_factor = 0.9

#Cyclohexan o in w
filename = "Henschke_5_4_Cyclo_300.xlsx"
time_factor = 0.04

filename = "Henschke_5_4_Cyclo_450.xlsx"
time_factor = 0.1

filename = "Henschke_5_4_Cyclo_600.xlsx"
time_factor = 0.1

# n-Hexan o in w
filename = "Henschke_n_Hexan_1_1.xlsx"
time_factor = 0.1

################ Plots aus der Finalen Arbeit (je nach Fit und Koaleszenzmodell ggf. aus- und einkommentieren)

# Einfluss der Füllhöhe
filename = "Henschke_5_2_MiBK_400.xlsx"
time_factor = 0.5
# filename = "Henschke_5_2_MiBK_200.xlsx"
# time_factor = 0.35
Hen = init_Hen(filename, time_factor)
r_s_star, h_p_star = find_rs_and_hp_star(Hen)
filenames = ['Henschke_5_2_MiBK_400.xlsx', 'Henschke_5_2_MiBK_200.xlsx']
time_factors = [0.5, 0.35]
labels = ['400, 0,625', '200, 0,575']
rs = [r_s_star, r_s_star]
simplies = [True, True]
hp = [h_p_star, h_p_star]
lgnd_title = r'$h_0$ / mm, $\Phi_{32,0}$ / mm:'
title = (r'$r_\mathrm{s}^*=$ ' + str(round(r_s_star, 4)).replace('.', ',')
         + r', $h_\mathrm{p}^*=$' + str(round(h_p_star, 5)).replace('.', ','))
# title = r'$r_\mathrm{s}^*=$ ' + str(0.04).replace('.', ',')
figsize = (4.5, 4)
m.plot_results(filenames=filenames, time_factors=time_factors, rs_stars=rs, simplie_list=simplies,
               hp_stars=hp, labels=labels, title=title, hp_plot=True, data=True,
               legend_title=lgnd_title, figsize=figsize)
# m.plot_results(filenames=filenames, time_factors=time_factors, rs_stars=[0.04, 0.04], hp_stars=hp,
#                simplie_list=[False, False], labels=labels, title=title, hp_plot=True, data=True,
#                legend_title=lgnd_title, figsize=figsize)

# Einfluss der Rührerdrehzahl
filename = "Henschke_5_4_Cyclo_300.xlsx"
time_factor = 0.04
# filename = "Henschke_5_4_Cyclo_450.xlsx"
# time_factor = 0.1
# filename = "Henschke_5_4_Cyclo_600.xlsx"
# time_factor = 0.1
Hen = init_Hen(filename, time_factor)
r_s_star, h_p_star = find_rs_and_hp_star(Hen)
filenames = ['Henschke_5_4_Cyclo_300.xlsx', 'Henschke_5_4_Cyclo_450.xlsx', 'Henschke_5_4_Cyclo_600.xlsx']
time_factors = [0.04, 0.1, 0.1]
labels = ['300, 1,056', '450, 0,623', '600, 0,502']
rs = [r_s_star, r_s_star, r_s_star]
simplies = [True, True, True]
hp = [h_p_star, h_p_star, h_p_star]
lgnd_title = r'$n$ / $\mathrm{min}^{-1}$, $\Phi_{32,0}$ / mm:'
title = (r'$r_\mathrm{s}^*=$ ' + str(round(r_s_star, 4)).replace('.', ',')
         + r', $h_\mathrm{p}^*=$' + '{:.5f}'.format(h_p_star).replace('.', ','))
# title = r'$r_\mathrm{s}^*=$ ' + str(0.03).replace('.', ',')
figsize = (4.5, 4)
m.plot_results(filenames=filenames, time_factors=time_factors, rs_stars=rs, simplie_list=simplies,
               hp_stars=hp, labels=labels, title=title, hp_plot=False, data=True,
               legend_title=lgnd_title, figsize=figsize)
# m.plot_results(filenames=filenames, time_factors=time_factors, rs_stars=[0.03, 0.03, 0.03], hp_stars=hp,
#                simplie_list=[False, False, False], labels=labels, title=title, hp_plot=False, data=True,
#                legend_title=lgnd_title, figsize=figsize)

# Einfluss des Phasenverhältnisses MiBK w in o
# MiBK w in o
filename = 'Henschke_5_5_MiBK_2_3.xlsx'
time_factor = 1.0
# filename = 'Henschke_5_5_MiBK_3_2.xlsx'
# time_factor = 1.0
# filename = 'Henschke_5_5_MiBK_3_1.xlsx'
# time_factor = 0.9
Hen = init_Hen(filename, time_factor)
r_s_star, h_p_star = find_rs_and_hp_star(Hen)
filenames = ['Henschke_5_5_MiBK_2_3.xlsx', 'Henschke_5_5_MiBK_3_2.xlsx', 'Henschke_5_5_MiBK_3_1.xlsx']
time_factors = [1.0, 1.0, 0.9]
labels = ['0,60, 2,012', '0,40, 1,032', '0,25, 0,764']
rs = [r_s_star, r_s_star, r_s_star]
simplies = [True, True, True]
hp = [h_p_star, h_p_star, h_p_star]
lgnd_title = r'$\epsilon_0$ / -, $\Phi_{32,0}$ / mm:'
title = (r'$r_\mathrm{s}^*=$ ' + str(round(r_s_star, 4)).replace('.', ',')
         + r', $h_\mathrm{p}^*=$' + '{:.5f}'.format(h_p_star).replace('.', ','))
# title = r'$r_\mathrm{s}^*=$ ' + str(0.045).replace('.', ',')
figsize = (4.5, 4)
m.plot_results(filenames=filenames, time_factors=time_factors, rs_stars=rs, simplie_list=simplies,
               hp_stars=hp, labels=labels, title=title, hp_plot=False, data=True,
               legend_title=lgnd_title, figsize=figsize)
# m.plot_results(filenames=filenames, time_factors=time_factors, rs_stars=[0.045, 0.045, 0.045], hp_stars=hp,
#                simplie_list=[False, False, False], labels=labels, title=title, hp_plot=False, data=True,
#                legend_title=lgnd_title, figsize=figsize)

# Einfluss des Phasenverhältnisses und der Dispergierrichtung Butylacetat/Wasser
filename = "Henschke__1_4_n-Butylacetat_Water.xlsx"
time_factor = 0.2
# filename = "Henschke__2_3_n-Butylacetat_Water.xlsx"
# time_factor = 0.2
# filename = "Henschke__3_2_n-Butylacetat_Water.xlsx"
# time_factor = 0.15
# filename = 'Henschke__4_1_n-Butylacetat_Water.xlsx'
# time_factor = 0.175
Hen = init_Hen(filename, time_factor)
r_s_star, h_p_star = find_rs_and_hp_star(Hen)
filenames = ['Henschke__1_4_n-Butylacetat_Water.xlsx', 'Henschke__2_3_n-Butylacetat_Water.xlsx',
             'Henschke__3_2_n-Butylacetat_Water.xlsx', 'Henschke__4_1_n-Butylacetat_Water.xlsx']
time_factors = [0.2, 0.2, 0.25, 0.175]
labels = ['0,20, 0,585', '0,40, 1,135', '0,40, 0,672', '0,20, 0,540']
rs = [r_s_star, r_s_star, r_s_star, r_s_star]
simplies = [True, True, True, True]
hp = [h_p_star, h_p_star, h_p_star, h_p_star]
lgnd_title = r'$\epsilon_0$ / -, $\Phi_{32,0}$ / mm:'
title = (r'$r_\mathrm{s}^*=$ ' + str(round(r_s_star, 4)).replace('.', ',')
         + r', $h_\mathrm{p}^*=$' + '{:.5f}'.format(h_p_star).replace('.', ','))
figsize = (5, 4.4)
m.plot_results(filenames=filenames, time_factors=time_factors, rs_stars=rs, simplie_list=simplies,
               hp_stars=hp, labels=labels, title=title, hp_plot=False, data=True,
               legend_title=lgnd_title, figsize=figsize, butyl=True)

# Einfluss von h_p_star auf den Verlauf der Absetzkurven am Beispiel Butylacetat in Wasser
filename = 'Henschke__2_3_n-Butylacetat_Water.xlsx'
time_factor = 0.2
Hen = init_Hen(filename, time_factor)
r_s_star = Hen.find_r_s_star(full_report=True)[0]
hp = [0, 1e-5, 0.01, 1]
filenames = ['Henschke__2_3_n-Butylacetat_Water.xlsx', 'Henschke__2_3_n-Butylacetat_Water.xlsx',
             'Henschke__2_3_n-Butylacetat_Water.xlsx', 'Henschke__2_3_n-Butylacetat_Water.xlsx']
time_factors = [0.2, 0.2, 0.2, 0.2]
lgnd_title = r'$h_\mathrm{p}^*$ / - :'
labels = ['0', '$10^{-5}$', '0,01', '1']
rs = [r_s_star, r_s_star, r_s_star, r_s_star]
simplies = [True, True, True, True]
figsize = (6, 4.5)
title = (r'$r_\mathrm{s}^*=$ ' + str(round(r_s_star, 4)).replace('.', ',')
         + r', $\epsilon_0=0$,40, $\Phi_{32,0}=1$,135$\,$mm')
m.plot_results(filenames=filenames, time_factors=time_factors, rs_stars=rs, simplie_list=simplies,
               hp_stars=hp, labels=labels, title=title, hp_plot=False, data=False,
               legend_title=lgnd_title, figsize=figsize)