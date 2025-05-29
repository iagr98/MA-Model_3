import numpy as np
import matplotlib.pyplot as plt
# Diese Datei erzeugt den Plot der Änderungsrate des Sauterdurchmessers für verschiedene hp_star-Werte

# Parameters
g = 9.81
delta_rho = 115
sigma = 0.013
r_s_star=0.0465
H_cd = 1e-20
eta_c = 0.001012

h = 0.01

def tau(d_32, ID, hp):

    La_mod = (g * delta_rho / sigma) ** 0.6 * d_32 * hp ** 0.2

    R_F = d_32 * (1 - (4.7 / (4.7 + La_mod))) ** 0.5

    if ID == "d":
        R_F = 0.3025 * R_F
    else:
        R_F = 0.5240 * R_F

    R_a = 0.5 * d_32 * (1 - (1 - 4.7 / (4.7 + La_mod)) ** 0.5)

    tau = 7.65 * eta_c * (R_a ** (7 / 3)
                                   / (H_cd ** (1 / 6) * sigma ** (5 / 6) * R_F * r_s_star))

    return tau

plt.figure(figsize=(6,4.5))

hp_star = np.linspace(1e-20, 1, 500)
tau_dd = np.zeros_like(hp_star)

d = np.linspace(0.0005, 0.002, 4)
colors = ['b', 'r', 'g', 'm']

for ind in range(len(d)):
    for i in range(len(hp_star)):
        tau_dd[i] = tau(d[ind], 'd', hp_star[i]*h)

    dphi = d[ind] / (6 * tau_dd) * 1000 # Änderungsrate Sauterdurchmesser in mm/s

    plt.plot(hp_star, dphi, color=colors[ind], label=str(d[ind]*1000))
plt.title(r'$r_\mathrm{s}^*=0$,0465, $h_\mathrm{p}=10\,\mathrm{mm}$')
lgnd_title = r'$\Phi_{32}$ / mm:'
plt.legend(title=lgnd_title, frameon=False)
plt.xlabel(r'Koaleszenzparameter $h_\mathrm{p}^*$ / - (logarithmisch aufgetragen)', size=12)
plt.xscale('log')
plt.xlim(right=1)
plt.ylim(bottom=0)
plt.tick_params(axis='x', top=True, direction='in')
plt.tick_params(axis='y', right=True, direction='in')
plt.ylabel(r'Änderungsrate von $\Phi_{32}$ / $\mathrm{mm\,s^{-1}}$', size=12)
plt.tight_layout()
plt.show()

#  Anderer
# phi_32 = np.linspace(0.0005, 0.002, 50)
# tau_di = np.zeros_like(phi_32)
#
# h_vec = np.array([0.002, 0.005, 0.01, 0.02])
#
# for ind in range(len(h_vec)):
#     for i in range(len(phi_32)):
#         tau_di[i] = 1 / tau(phi_32[i], 'i', h_vec[ind])
#
#     plt.plot(phi_32*1000, tau_di, label='hp='+str(h_vec[ind]*1000)+'mm')
#
# plt.title('TGK-Zeit in Abhängigkeit vom Sauterdurchmesser')
# plt.ylim(bottom=0)
# plt.xlim(phi_32[0]*1000, phi_32[-1]*1000)
# plt.xlabel('phi_32 [mm]')
# plt.ylabel('Tropfen-Grenzflächen-Koaleszenzrate [1/s]')
# plt.legend()
# plt.show()