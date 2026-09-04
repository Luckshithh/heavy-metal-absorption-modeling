import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

my_path = os.path.dirname(os.path.abspath(__file__))
csv_loc = os.path.join(my_path, "..", "data", "final data", "final_kinetics_data.csv")
raw_data = pd.read_csv(csv_loc)

time_pts = raw_data['time_min'].values
qt_vals = raw_data['qt_mg_g'].values

def first_order(t, q_eq, rate1):
    return q_eq * (1 - np.exp(-rate1 * t))

def second_order(t, q_eq, rate2):
    return (rate2 * (q_eq**2) * t) / (1 + rate2 * q_eq * t)

fit_1st, cov_1st = curve_fit(first_order, time_pts, qt_vals, p0=[max(qt_vals), 0.1], maxfev=10000)
qe_1st, k1_rate = fit_1st
err_1st = np.sqrt(np.diag(cov_1st))
qe1_err, k1_err = err_1st
pred_1st = first_order(time_pts, qe_1st, k1_rate)

fit_2nd, cov_2nd = curve_fit(second_order, time_pts, qt_vals, p0=[max(qt_vals), 0.01], maxfev=10000)
qe_2nd, k2_rate = fit_2nd
err_2nd = np.sqrt(np.diag(cov_2nd))
qe2_err, k2_err = err_2nd
pred_2nd = second_order(time_pts, qe_2nd, k2_rate)

def compute_goodness(y_real, y_model, p_count=2):
    total_n = len(y_real)
    residual_ss = np.sum((y_real - y_model) ** 2)
    total_ss = np.sum((y_real - np.mean(y_real)) ** 2)
    r_sq = 1 - (residual_ss / total_ss)
    adjusted_r = 1 - ((1 - r_sq) * (total_n - 1) / (total_n - p_count - 1))
    chi_val = np.sum(((y_real - y_model) ** 2) / (y_model + 1e-10))
    return r_sq, adjusted_r, chi_val

r2_1, adj_1, chi_1 = compute_goodness(qt_vals, pred_1st)
r2_2, adj_2, chi_2 = compute_goodness(qt_vals, pred_2nd)

print("\n" + "=" * 75)
print("             ADVANCED ADSORPTION KINETICS ANALYSIS SUMMARY             ")
print("=" * 75)
print("PSEUDO-FIRST-ORDER MODEL (Physisorption Dominant):")
print(f"  - Equil. Capacity Prediction (q_e)  : {qe_1st:.4f} ± {qe1_err:.4f} mg/g")
print(f"  - Adsorption Rate Constant (k_1)    : {k1_rate:.4e} ± {k1_err:.4e} min^-1")
print(f"  - Coefficient of Determination (R^2): {r2_1:.4f}")
print(f"  - Adjusted R-Squared (Adj R^2)      : {adj_1:.4f}")
print(f"  - Chi-Squared (X^2) Goodness-of-Fit : {chi_1:.4f}")
print("-" * 75)
print("PSEUDO-SECOND-ORDER MODEL (Chemisorption Dominant):")
print(f"  - Equil. Capacity Prediction (q_e)  : {qe_2nd:.4f} ± {qe2_err:.4f} mg/g")
print(f"  - Adsorption Rate Constant (k_2)    : {k2_rate:.4e} ± {k2_err:.4e} g/mg*min")
print(f"  - Coefficient of Determination (R^2): {r2_2:.4f}")
print(f"  - Adjusted R-Squared (Adj R^2)      : {adj_2:.4f}")
print(f"  - Chi-Squared (X^2) Goodness-of-Fit : {chi_2:.4f}")
print("=" * 75 + "\n")

plt.figure(figsize=(8, 6))
plt.scatter(time_pts, qt_vals, color='black', marker='o', s=60, label='Experimental Data', zorder=5)

t_grid = np.linspace(0, max(time_pts) * 1.1, 200)
plt.plot(t_grid, first_order(t_grid, qe_1st, k1_rate), 'b--', linewidth=2, label=f'PFO Fit ($R^2_{{adj}} = {adj_1:.3f}$)')
plt.plot(t_grid, second_order(t_grid, qe_2nd, k2_rate), 'r-', linewidth=2, label=f'PSO Fit ($R^2_{{adj}} = {adj_2:.3f}$)')

plt.xlabel('Contact Time, $t$ (min)', fontsize=11)
plt.ylabel('Adsorption Capacity over time, $q_t$ (mg/g)', fontsize=11)
plt.title('Non-Linear Adsorption Kinetics of Pb(II)', fontsize=12, fontweight='bold')
plt.legend(frameon=True, facecolor='white', edgecolor='none')
plt.grid(True, linestyle='--', alpha=0.5)

out_folder = os.path.join(my_path, "..", "plots")
os.makedirs(out_folder, exist_ok=True)
img_dest = os.path.join(out_folder, "kinetics_plot.png")
plt.savefig(img_dest, dpi=300, bbox_inches='tight')
print(f"Successfully generated and saved publication-quality plot to: {img_dest}\n")