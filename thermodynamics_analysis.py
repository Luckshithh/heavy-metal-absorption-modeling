import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress
#calculate line of best fit
#enthaly = m*-R(8.31 gas constant)
#entropy= y intercept * R
# use the main equation to find if the reaction is spontaneous (rmember : negative means spontaneous(opposite of what u usually think))

df = pd.read_csv('/Users/luckshitg.n/science project/final_thermodynamics_data.csv')
T = df['temp_K'].values
C_e = df['ce_mg_L'].values
q_e = df['qe_mg_g'].values


K_d = q_e / C_e
inv_T = 1 / T
ln_Kd = np.log(K_d)


slope, intercept, r_value, p_value, std_err = linregress(inv_T, ln_Kd)


R = 8.314 
dH = -slope * R / 1000 
dS = intercept * R 


dG = dH - T * (dS / 1000)


print("Thermodynamic Parameters Summary")
print("-" * 65)
print(f"Delta H (kJ/mol): {dH:.4f}")
print(f"Delta S (J/mol*K): {dS:.4f}")
print("-" * 65)
print(f"{'T (K)':<10} | {'Delta G (kJ/mol)':<20}")
print("-" * 35)
for t_val, dg_val in zip(T, dG):
    print(f"{t_val:<10} | {dg_val:<20.4f}")
print("-" * 65)


plt.figure(figsize=(8, 6))
plt.scatter(inv_T, ln_Kd, color='black', label='Experimental Data', zorder=5)


inv_T_smooth = np.linspace(min(inv_T)*0.95, max(inv_T)*1.05, 100)
ln_Kd_pred = slope * inv_T_smooth + intercept
plt.plot(inv_T_smooth, ln_Kd_pred, 'r-', label=f'Linear Fit ($R^2 = {r_value**2:.4f}$)')

plt.xlabel('1/T (K$^{-1}$)')
plt.ylabel('ln($K_d$)')
plt.title("Van 't Hoff Plot")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)


plot_path = '/Users/luckshitg.n/.gemini/antigravity-ide/brain/23b42a47-30b1-43a9-9b17-1e4b81d498ad/thermodynamics_plot.png'
plt.savefig(plot_path, dpi=300, bbox_inches='tight')
print(f"\nPlot saved to: {plot_path}")
