import numpy as np
from scipy.optimize import newton

# Funktion berechnet Höhe eines Kreissegments auf Basis des Kreisradius r und der Fläche A
def getHeight(A, r):
    eq = lambda h: A - r**2 * np.arccos(1 - h / r) + (r - h) * np.sqrt(2 * r * h - h**2)
    h0 = r / 2
    if A < 0:
        #print('Querschnitt kleiner Null: ' + str(A))
        return 0
    elif A > np.pi * r**2:
        #print('Querschnitt größer als zulässig: ' + str(A))
        return 2*r
    return newton(eq, h0)

def getHeightArray(A, r):
    h = np.zeros_like(A)
    for i in range(len(h)):
        h[i] = getHeight(A[i], r)
    return h

# Funktion berechnet die Fläche eines Kreissegments auf Basis des Kreisradiuses r und der Höhe h des Segments
def getArea(h, r):
    return r**2 * np.arccos(1 - h / r) - (r - h) * np.sqrt(2 * r * h - h**2)

# Funktion berechnet dyn. Viskosität in der dicht gepackten Schicht nach Modell von Yaron und Gal-Or
def yaron(eta_c, eta_d, eps):
    al = eta_c / (eta_d + 23e-3)
    ga = eps ** (1 / 3)
    omega = ((4 * ga ** 7 + 10 - (84 / 11) * ga ** 2 + 4 * al * (1 - ga ** 7)) /
             (10 * (1 - ga ** 10) - 25 * ga ** 3 * (1 - ga ** 4) + 10 * al * (1 - ga ** 3) * (1 - ga ** 7)))
    return eta_c * (1 + 5.5 * omega * eps)

def calculate_volume_balance(Sim):
      """ Berechnet die Volumenbilanz für die Simulation
       Ausgabe in prozent """
      dV_ges = Sim.Sub.dV_ges
      _, u_d, u_c = Sim.velocities(Sim.V_dis[:,-1], Sim.V_d[:,-1], Sim.V_c[:,-1], Sim.Set.T, balance=True)
      A_c = Sim.V_c[-1,-1] / Sim.Set.dl
      A_d = Sim.V_d[-1,-1] / Sim.Set.dl
      return 100*abs(dV_ges - u_c*A_c - u_d*A_d)/dV_ges

def calculate_cfl(Sim):
    u_dis, u_d, u_c = Sim.velocities(Sim.V_dis[:,-1], Sim.V_d[:,-1], Sim.V_c[:,-1], 0, balance=False)
    u = max(max(u_dis), max(u_d), max(u_c))
    return u * Sim.Set.dt / Sim.Set.dl