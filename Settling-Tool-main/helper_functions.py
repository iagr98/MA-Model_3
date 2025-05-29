# -*- coding: utf-8 -*-
"""
@author: luth01
"""

# Gibt Wert der Abweichung der Koaleszenzkurve von der Error-Function zurück
def NRMSE(List_exp, List_sim, Norm_value):
    N = len(List_exp)

    value = ( (sum( ((List_sim - List_exp)/Norm_value) ** 2) ) / N) ** 0.5 # Henschke-Ansatz

    return value