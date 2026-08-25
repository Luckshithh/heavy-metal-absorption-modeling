import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

my_path = os.path.dirname(os.path.abspath(__file__))
csv_loc = os.path.join(my_path, "..", "data", "final data", "final_isotherm_data.csv")
raw_data = pd.read_csv(csv_loc)

conc = raw_data['ce_mg_L'].values
cap = raw_data['qe_mg_g'].values

def lang_eq(c, qm, kl):
    return (qm * kl * c) / (1 + kl * c)

def freund_eq(c, kf, n_val):
    return kf * (c ** (1 / n_val))

fit_L, cov_L = curve_fit(lang_eq, conc, cap, p0=[max(cap) * 1.2, 0.1], maxfev=10000)
qm_val, kl_val = fit_L
sd_L = np.sqrt(np.diag(cov_L))
qm_sd, kl_sd = sd_L
pred_L = lang_eq(conc, qm_val, kl_val)

fit_F, cov_F = curve_fit(freund_eq, conc, cap, p0=[20.0, 2.0], maxfev=10000)
kf_val, n_coef = fit_F
sd_F = np.sqrt(np.diag(cov_F))
kf_sd, n_sd = sd_F
pred_F = freund_eq(conc, kf_val, n_coef)

def get_metrics(y_real, y_model, p_count=2):
    total_n = len(y_real)
    residual_ss = np.sum((y_real - y_model) ** 2)
    total_ss = np.sum((y_real - np.mean(y_real)) ** 2)
    r_sq = 1 - (residual_ss / total_ss)
    adjusted_r = 1 - ((1 - r_sq) * (total_n - 1) / (total_n - p_count - 1))
    chi_val = np.sum(((y_real - y_model) ** 2) / (y_model + 1e-10))
    return r_sq, adjusted_r, chi_val

r2_l, adj_l, chi_l = get_metrics(cap, pred_L)
r2_f, adj_f, chi_f = get_metrics(cap, pred_F)

print("\n" + "=" * 75)
print("             ADVANCED EQUILIBRIUM ISOTHERM ANALYSIS SUMMARY             ")
print("=" * 75)
print("LANGMUIR ISOTHERM MODEL (Homogeneous Monolayer):")
print(f"  - Maximum Adsorption Capacity (q_m) : {qm_val:.4f} ± {qm_sd:.4f} mg/g")
print(f"  - Adsorption Affinity Constant (K_L): {kl_val:.4f} ± {kl_sd:.4f} L/mg")
print(f"  - Coefficient of Determination (R^2): {r2_l:.4f}")
print(f"  - Adjusted R-Squared (Adj R^2)      : {adj_l:.4f}")
print(f"  - Chi-Squared (X^2) Goodness-of-Fit : {chi_l:.4f}")
print("-" * 75)
print("FREUNDLICH ISOTHERM MODEL (Heterogeneous Multilayer):")
print(f"  - Adsorption Capacity Factor (K_F)  : {kf_val:.4f} ± {kf_sd:.4f} (mg/g)/(mg/L)^(1/n)")
print(f"  - Adsorption Intensity Factor (n)   : {n_coef:.4f} ± {n_sd:.4f}")
print(f"  - Coefficient of Determination (R^2): {r2_f:.4f}")
print(f"  - Adjusted R-Squared (Adj R^2)      : {adj_f:.4f}")
print(f"  - Chi-Squared (X^2) Goodness-of-Fit : {chi_f:.4f}")
print("=" * 75 + "\n")

plt.figure(figsize=(8, 6))
plt.scatter(conc, cap, color='black', marker='o', s=60, label='Experimental Data (Lead)', zorder=5)

x_grid = np.linspace(0, max(conc) * 1.1, 200)
plt.plot(x_grid, lang_eq(x_grid, qm_val, kl_val), 'b--', linewidth=2, label=f'Langmuir Fit ($R^2_{{adj}} = {adj_l:.3f}$)')
plt.plot(x_grid, freund_eq(x_grid, kf_val, n_coef), 'r-', linewidth=2, label=f'Freundlich Fit ($R^2_{{adj}} = {adj_f:.3f}$)')

plt.xlabel('Equilibrium Liquid-Phase Concentration, $C_e$ (mg/L)', fontsize=11)
plt.ylabel('Equilibrium Solid-Phase Capacity, $q_e$ (mg/g)', fontsize=11)
plt.title('Non-Linear Isotherm Fitting for Pb(II) Remediation', fontsize=12, fontweight='bold')
plt.legend(frameon=True, facecolor='white', edgecolor='none')
plt.grid(True, linestyle='--', alpha=0.5)

out_folder = os.path.join(my_path, "..", "plots")
os.makedirs(out_folder, exist_ok=True)
img_dest = os.path.join(out_folder, "isotherm_plot.png")
plt.savefig(img_dest, dpi=300, bbox_inches='tight')
print(f"Successfully generated and saved publication-quality plot to: {img_dest}\n")