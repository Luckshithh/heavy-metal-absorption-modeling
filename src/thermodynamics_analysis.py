import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import linregress

my_path = os.path.dirname(os.path.abspath(__file__))
csv_loc = os.path.join(my_path, "..", "data", "final data", "final_thermodynamics_data.csv")
raw_data = pd.read_csv(csv_loc)

temp_k = raw_data['temp_K'].values
c_eq = raw_data['ce_mg_L'].values
q_eq = raw_data['qe_mg_g'].values

kd_ratio = q_eq / c_eq
recip_t = 1 / temp_k
log_kd = np.log(kd_ratio)

m_slope, c_intercept, r_val, p_val, std_err = linregress(recip_t, log_kd)
r_gas = 8.314

sample_n = len(recip_t)
x_mean = np.mean(recip_t)
dev_x = np.sum((recip_t - x_mean) ** 2)
dev_y = log_kd - (m_slope * recip_t + c_intercept)
ss_residuals = np.sum(dev_y ** 2)
se_regression = np.sqrt(ss_residuals / (sample_n - 2))
m_error = se_regression / np.sqrt(dev_x)
c_error = se_regression * np.sqrt(1/sample_n + (x_mean**2 / dev_x))

h_delta = -m_slope * r_gas / 1000
h_error = m_error * r_gas / 1000

s_delta = c_intercept * r_gas
s_error = c_error * r_gas

g_delta = h_delta - temp_k * (s_delta / 1000)
g_error = np.sqrt((h_error) ** 2 + (temp_k * s_error / 1000) ** 2)

print("\n" + "=" * 75)
print("             ADVANCED THERMODYNAMICS PARAMETERS ANALYSIS             ")
print("=" * 75)
print(f"  - Standard Enthalpy ($\Delta H^0$) : {h_delta:.4f} ± {h_error:.4f} kJ/mol")
print(f"  - Standard Entropy ($\Delta S^0$)  : {s_delta:.4f} ± {s_error:.4f} J/mol*K")
print("-" * 75)
print(f"{'Temperature (K)':<18} | {'Gibbs Free Energy ($\Delta G^0$) (kJ/mol)':<40}")
print("-" * 75)
for t_v, g_v, g_err_v in zip(temp_k, g_delta, g_error):
    print(f"{t_v:<18.2f} | {g_v:<8.4f} ± {g_err_v:<8.4f}")
print("=" * 75 + "\n")

plt.figure(figsize=(8, 6))
plt.scatter(recip_t, log_kd, color='black', marker='o', s=60, label='Experimental Data', zorder=5)

x_grid = np.linspace(recip_t.min() * 0.95, recip_t.max() * 1.05, 100)
y_pred = m_slope * x_grid + c_intercept
plt.plot(x_grid, y_pred, 'r-', linewidth=2, label=f'Linear Fit ($R^2 = {r_val**2:.4f}$)')

plt.xlabel('Reciprocal Temperature, $1/T$ (K$^{-1}$)', fontsize=11)
plt.ylabel('Logarithm of Distribution Coefficient, $\ln(K_d)$', fontsize=11)
plt.title("Van 't Hoff Linear Regression Fit", fontsize=12, fontweight='bold')
plt.legend(frameon=True, facecolor='white', edgecolor='none')
plt.grid(True, linestyle='--', alpha=0.5)

out_folder = os.path.join(my_path, "..", "plots")
os.makedirs(out_folder, exist_ok=True)
img_dest = os.path.join(out_folder, "thermodynamics_plot.png")
plt.savefig(img_dest, dpi=300, bbox_inches='tight')
print(f"Successfully generated and saved publication-quality plot to: {img_dest}\n")