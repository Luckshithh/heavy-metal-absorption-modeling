import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#curve fit is used to find the constants
from scipy.optimize import curve_fit

## Applying non-linear regression for plotting equilibrium concentration (Ce) vs adsorption capacity (qe)
df = pd.read_csv('/Users/luckshitg.n/science project/final_isotherm_data.csv')
C_e = df['ce_mg_L'].values
q_e = df['qe_mg_g'].values

#heavy metals form uniform single layer on the biochar suruface
def langmuir_model(C_e, q_m, K_L):
    return (q_m * K_L * C_e) / (1 + K_L * C_e)
#heavy metal can go on top of each other and the biochar surface unequal therefore leading to different binding energies
def freundlich_model(C_e, K_F, n):
    return K_F * (C_e ** (1 / n))


popt_langmuir, _ = curve_fit(langmuir_model, C_e, q_e, p0=[max(q_e)*1.2, 0.1], maxfev=10000)
q_m, K_L = popt_langmuir
q_e_pred_langmuir = langmuir_model(C_e, q_m, K_L)


popt_freundlich, _ = curve_fit(freundlich_model, C_e, q_e, p0=[20, 2], maxfev=10000)
K_F, n = popt_freundlich
q_e_pred_freundlich = freundlich_model(C_e, K_F, n)


def calc_r2(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred)**2)
    ss_tot = np.sum((y_true - np.mean(y_true))**2)
    return 1 - (ss_res / ss_tot)

r2_langmuir = calc_r2(q_e, q_e_pred_langmuir)
r2_freundlich = calc_r2(q_e, q_e_pred_freundlich)


print("Isotherm Model Summary")
print("-" * 75)
print(f"Langmuir Model  : q_m = {q_m:.4f} mg/g, K_L = {K_L:.4f} L/mg, R^2 = {r2_langmuir:.4f}")
print(f"Freundlich Model: K_F = {K_F:.4f} (mg/g)/(mg/L)^(1/n), n = {n:.4f}, R^2 = {r2_freundlich:.4f}")
print("-" * 75)


plt.figure(figsize=(8, 6))
plt.scatter(C_e, q_e, color='black', label='Experimental Data', zorder=5)

C_e_smooth = np.linspace(0, max(C_e) * 1.1, 100)
plt.plot(C_e_smooth, langmuir_model(C_e_smooth, q_m, K_L), 'b--', label='Langmuir Model')
plt.plot(C_e_smooth, freundlich_model(C_e_smooth, K_F, n), 'r-', label='Freundlich Model')

plt.xlabel('Equilibrium Concentration (mg/L)')
plt.ylabel('Adsorption Capacity (mg/g)')
plt.title('Adsorption Isotherms')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)


plot_path = '/Users/luckshitg.n/.gemini/antigravity-ide/brain/23b42a47-30b1-43a9-9b17-1e4b81d498ad/isotherm_plot.png'
plt.savefig(plot_path, dpi=300, bbox_inches='tight')
print(f"\nPlot saved to: {plot_path}")
