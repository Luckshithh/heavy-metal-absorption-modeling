import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

def calculate_rmse(y_actual, y_predicted):
    return np.sqrt(np.mean((y_actual - y_predicted) ** 2))

# trying to test which algorithm is better at modelling the data.
# The Pseudo-first-order (PFO) model says they both are held together through physical forces.
# The Pseudo-second-order (PSO) model says they both are held together through chemical bonding.
import os

#use scipy curve fit function to model both pseudo first order and pseudo second order and plot them on the same graph to see which one is better
script_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(script_dir, 'final_kinetics_data.csv')
df = pd.read_csv(data_path)
t = df['time_min'].values
q_t = df['qt_mg_g'].values

def pfo_model(t, q_e, k_1):
    return q_e * (1 - np.exp(-k_1 * t))

def pso_model(t, q_e, k_2):
    return (k_2 * (q_e**2) * t) / (1 + k_2 * q_e * t)


popt_pfo, _ = curve_fit(pfo_model, t, q_t, p0=[max(q_t), 0.1], maxfev=10000)
q_e_pfo, k_1 = popt_pfo
q_t_pred_pfo = pfo_model(t, q_e_pfo, k_1)


popt_pso, _ = curve_fit(pso_model, t, q_t, p0=[max(q_t), 0.01], maxfev=10000)
q_e_pso, k_2 = popt_pso
q_t_pred_pso = pso_model(t, q_e_pso, k_2)


def calc_r2(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred)**2)
    ss_tot = np.sum((y_true - np.mean(y_true))**2)
    return 1 - (ss_res / ss_tot)

r2_pfo = calc_r2(q_t, q_t_pred_pfo)
r2_pso = calc_r2(q_t, q_t_pred_pso)


print(f"{'Model':<10} | {'q_e (mg/g)':<12} | {'k':<15} | {'R^2':<10}")
print("-" * 55)
print(f"{'PFO':<10} | {q_e_pfo:<12.4f} | {k_1:<15.4e} | {r2_pfo:<10.4f}")
print(f"{'PSO':<10} | {q_e_pso:<12.4f} | {k_2:<15.4e} | {r2_pso:<10.4f}")

rmse_pfo = calculate_rmse(q_t, q_t_pred_pfo)
rmse_pso = calculate_rmse(q_t, q_t_pred_pso)

print(f"PFO RMSE: {rmse_pfo:.4f} mg/g")
print(f"PSO RMSE: {rmse_pso:.4f} mg/g")


# Create a 2-panel figure (Top: Fit, Bottom: Residuals)
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8), sharex=True, 
                             gridspec_kw={'height_ratios': [3, 1]})

# --- TOP PANEL: Kinetics Fit ---
ax1.scatter(t, q_t, color='black', zorder=5, label='Experimental Data')
t_smooth = np.linspace(0, max(t) + 10, 100)
ax1.plot(t_smooth, pfo_model(t_smooth, q_e_pfo, k_1), label=f'Pseudo-First-Order ($R^2$ = {r2_pfo:.4f}, RMSE = {rmse_pfo:.2f})', color='darkorange', lw=2)
ax1.plot(t_smooth, pso_model(t_smooth, q_e_pso, k_2), label=f'Pseudo-Second-Order ($R^2$ = {r2_pso:.4f}, RMSE = {rmse_pso:.2f})', color='forestgreen', linestyle='--', lw=2)
ax1.set_ylabel('Adsorbed Amount, $q_t$ (mg/g)', fontsize=11)
ax1.set_title('Adsorption Kinetics Modeling & Residual Analysis', fontsize=13, fontweight='bold')
ax1.legend(loc='lower right', frameon=True)
ax1.grid(True, linestyle=':', alpha=0.6)

# --- BOTTOM PANEL: Residuals ---
# Calculate residuals at experimental points
res_pfo = q_t - q_t_pred_pfo
res_pso = q_t - q_t_pred_pso

ax2.scatter(t, res_pfo, color='darkorange', marker='o', s=60, label='PFO Residuals')
ax2.scatter(t, res_pso, color='forestgreen', marker='s', s=60, label='PSO Residuals')
ax2.axhline(y=0, color='black', linestyle='-', linewidth=1.5, alpha=0.7)  # Zero-error baseline
ax2.set_xlabel('Time, $t$ (min)', fontsize=11)
ax2.set_ylabel('Residual (mg/g)', fontsize=11)
ax2.grid(True, linestyle=':', alpha=0.6)
ax2.set_ylim(-max(abs(np.concatenate([res_pfo, res_pso]))) * 1.5, 
             max(abs(np.concatenate([res_pfo, res_pso]))) * 1.5)

# Adjust layout and save
plt.tight_layout()
plot_path = os.path.join(script_dir, 'plots', 'kinetics_plot_new.png')
plt.savefig(plot_path, dpi=300)
plt.close()
print(f"\nPlot saved to: {plot_path}")
