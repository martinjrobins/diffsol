import pandas as pd
import matplotlib.pyplot as plt

# load python_results.csv file
df = pd.read_csv('book/src/benchmarks/python_results.csv')
diffrax_low_tol = df[df['tol'] == 1e-4]['diffrax_time']
casadi_low_tol = df[df['tol'] == 1e-4]['casadi_time']
diffsol_low_tol = df[df['tol'] == 1e-4]['diffsol_time']
ngroups_low_tol = df[df['tol'] == 1e-4]['ngroups']
diffrax_high_tol = df[df['tol'] == 1e-8]['diffrax_time']
casadi_high_tol = df[df['tol'] == 1e-8]['casadi_time']
diffsol_high_tol = df[df['tol'] == 1e-8]['diffsol_time']
ngroups_high_tol = df[df['tol'] == 1e-8]['ngroups']

# plot the results from diffrax and casadi scaled by the reference (diffsol)
fig, ax = plt.subplots()
ax.plot(ngroups_low_tol, diffrax_low_tol / diffsol_low_tol, label='diffrax (tol=1e-4)')
ax.plot(ngroups_low_tol, casadi_low_tol / diffsol_low_tol, label='casadi (tol=1e-4)')
ax.plot(ngroups_low_tol, diffrax_high_tol / diffsol_high_tol, label='diffrax (tol=1e-8)')
ax.plot(ngroups_low_tol, casadi_high_tol / diffsol_high_tol, label='casadi (tol=1e-8)')
ax.legend()
ax.set_yscale('log')
ax.set_xscale('log')
ax.set_xlabel('ngroups')
ax.set_ylabel('Time relative to diffsol')

plt.savefig('python_plot.png')
plt.savefig('book/src/benchmarks/images/python_plot.svg')
