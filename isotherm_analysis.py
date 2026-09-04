import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#curve fit is used to find the constants
from scipy.optimize import curve_fit
import os

def calculate_rmse(y_actual, y_predicted):
    return np.sqrt(np.mean((y_actual - y_predicted) ** 2))

## Applying non-linear regression for plotting equilibrium concentration (Ce) vs adsorption capacity (qe)
script_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(script_dir, 'final_isotherm_data.csv')
df = pd.read_csv(data_path)
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

rmse_langmuir = calculate_rmse(q_e, q_e_pred_langmuir)
rmse_freundlich = calculate_rmse(q_e, q_e_pred_freundlich)

print(f"Langmuir RMSE: {rmse_langmuir:.4f} mg/g")
print(f"Freundlich RMSE: {rmse_freundlich:.4f} mg/g")


# Create a 2-panel figure (Top: Fit, Bottom: Residuals)
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8), sharex=True, 
                             gridspec_kw={'height_ratios': [3, 1]})

# --- TOP PANEL: Experimental Data and Fits ---
ax1.scatter(C_e, q_e, color='black', zorder=5, label='Experimental Data')
C_e_smooth = np.linspace(0, max(C_e) * 1.1, 100)
ax1.plot(C_e_smooth, langmuir_model(C_e_smooth, q_m, K_L), label=f'Langmuir Fit ($R^2$ = {r2_langmuir:.4f}, RMSE = {rmse_langmuir:.2f})', color='royalblue', lw=2)
ax1.plot(C_e_smooth, freundlich_model(C_e_smooth, K_F, n), label=f'Freundlich Fit ($R^2$ = {r2_freundlich:.4f}, RMSE = {rmse_freundlich:.2f})', color='crimson', linestyle='--', lw=2)
ax1.set_ylabel('Adsorption Capacity, $q_e$ (mg/g)', fontsize=11)
ax1.set_title('Lead Isotherm Modeling & Residual Analysis', fontsize=13, fontweight='bold')
ax1.legend(loc='lower right', frameon=True)
ax1.grid(True, linestyle=':', alpha=0.6)

# --- BOTTOM PANEL: Residuals ---
# Calculate residuals at the experimental points
res_langmuir = q_e - q_e_pred_langmuir
res_freundlich = q_e - q_e_pred_freundlich

ax2.scatter(C_e, res_langmuir, color='royalblue', marker='o', s=60, label='Langmuir Residuals')
ax2.scatter(C_e, res_freundlich, color='crimson', marker='s', s=60, label='Freundlich Residuals')
ax2.axhline(y=0, color='black', linestyle='-', linewidth=1.5, alpha=0.7)  # Zero-error baseline
ax2.set_xlabel('Equilibrium Concentration, $C_e$ (mg/L)', fontsize=11)
ax2.set_ylabel('Residual (mg/g)', fontsize=11)
ax2.grid(True, linestyle=':', alpha=0.6)
ax2.set_ylim(-max(abs(np.concatenate([res_langmuir, res_freundlich]))) * 1.5, 
             max(abs(np.concatenate([res_langmuir, res_freundlich]))) * 1.5)

# Adjust layout and save
plt.tight_layout()
plot_path = os.path.join(script_dir, 'plots', 'isotherm_plot.png')
plt.savefig(plot_path, dpi=300)
plt.close()
print(f"\nPlot saved to: {plot_path}")
