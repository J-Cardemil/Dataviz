import numpy as np
import matplotlib.pyplot as plt
from pyXSteam.XSteam import XSteam
from iapws._iapws import _Sublimation_Pressure, _Melting_Pressure

steam = XSteam(XSteam.UNIT_SYSTEM_MKS)

# Triple and critical points
T_triple, P_triple = 273.16, 611.657      # [K], [Pa]
T_crit, P_crit = 647.096, 22.064e6        # [K], [Pa]

# ---- 1. Vaporization (Liquid–Vapor) ----
T_vap = np.linspace(T_triple, T_crit, 300)
P_vap = [steam.psat_t(T - 273.15) * 1e5 for T in T_vap]  # bar → Pa (input °C)

# ---- 2. Sublimation (Solid–Vapor) ----
T_subl = np.linspace(150, T_triple, 300)
P_subl = [_Sublimation_Pressure(T) * 1e6 for T in T_subl]  # MPa → Pa

# ---- 3. Fusion (Solid–Liquid) ----
T_fus = np.linspace(251.2, T_triple, 300)
P_fus = [_Melting_Pressure(T, ice='Ih') * 1e6 for T in T_fus]  # MPa → Pa

# ---- Plotting ----
plt.figure(figsize=(6, 5))
plt.semilogy(T_vap, P_vap, 'r', label='Vaporization (L–V)')
plt.semilogy(T_subl, P_subl, 'c', label='Sublimation (S–V)')
plt.semilogy(T_fus, P_fus, 'b', label='Fusion (S–L)')
plt.plot(T_triple, P_triple, 'ko', label='Triple Point')
plt.plot(T_crit, P_crit, 'ks', label='Critical Point')

# ---- Labels and Appearance ----
plt.xlabel('Temperature [K]')
plt.ylabel('Pressure [Pa]')
plt.title('Full Phase Diagram of Water (Solid, Liquid, Vapor)')
plt.grid(True, which='both', linestyle='--')
plt.legend()

plt.tight_layout()
plt.savefig('water_phase_diagram.png', dpi=300, bbox_inches='tight', transparent=True)
plt.show()
