import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

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


plt.figure(figsize=(8, 6))
plt.scatter(t, q_t, color='black', label='Experimental Data', zorder=5)

t_smooth = np.linspace(0, max(t) + 10, 100)
plt.plot(t_smooth, pfo_model(t_smooth, q_e_pfo, k_1), 'b--', label='PFO Model')
plt.plot(t_smooth, pso_model(t_smooth, q_e_pso, k_2), 'r-', label='PSO Model')

plt.xlabel('Time (min)')
plt.ylabel('Amount Adsorbed (mg/g)')
plt.title('Adsorption Kinetics')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)

plot_path = os.path.join(script_dir, 'kinetics_plot.png')
plt.savefig(plot_path, dpi=300, bbox_inches='tight')
print(f"\nPlot saved to: {plot_path}")
