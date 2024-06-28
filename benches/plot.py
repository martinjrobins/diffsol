import json
import glob
import matplotlib.pyplot as plt
import numpy as np
from pprint import pprint

bench_dirs = glob.glob('target/criterion/*')
sundials_reference = {}
problems = [
    {
        "name": "robertson", 
        "reference_name": "roberts_dns",
        "arg": None,
        "solvers": ["faer_esdirk34", "faer_tr_bdf2", "faer_bdf", "nalgebra_esdirk34", "nalgebra_tr_bdf2", "nalgebra_bdf"],
    },
    {
        "name": "robertson_ode",
        "reference_name": "robertson_ode_klu",
        "arg": [25, 100, 400, 900],
        "solvers": ["faer_sparse_bdf", "faer_sparse_bdf_klu"],
    },
    {
        "name": "heat2d",
        "reference_name": "heat2d_klu",
        "arg": [5, 10, 20, 30],
        "solvers": ["faer_sparse_esdirk", "faer_sparse_tr_bdf2", "faer_sparse_bdf", "faer_sparse_esdirk_klu", "faer_sparse_tr_bdf2_klu", "faer_sparse_bdf_klu"]
    },
    {
        "name": "foodweb",
        "reference_name": "foodweb_bnd",
        "arg": [5, 10, 20, 30],
        "solvers": ["faer_sparse_esdirk", "faer_sparse_tr_bdf2", "faer_sparse_bdf", "faer_sparse_esdirk_klu", "faer_sparse_tr_bdf2_klu", "faer_sparse_bdf_klu"]
    },
]
estimates = {}
for problem in problems:
    estimates[problem['name']] = {}
    estimates[problem['name']]['reference'] = []
    if problem['arg'] is None:
        reference_dirs = [f"target/criterion/sundials_{problem['reference_name']}"]
    else:
        reference_dirs = [f"target/criterion/sundials_{problem['reference_name']}_{arg}" for arg in problem['arg']]
    for reference_dir in reference_dirs:
        with open(f"{reference_dir}/new/estimates.json") as f:
            bench = json.load(f)
            estimates[problem['name']]['reference'].append(bench["mean"]["point_estimate"])
    for solver in problem['solvers']:
        estimates[problem['name']][solver] = {}
        estimates[problem['name']][solver]['diffsol'] = []
        estimates[problem['name']][solver]['args'] = []
        if problem['arg'] is None:
            diffsol_dirs = [f"target/criterion/{solver}_{problem['name']}"]
        else:
            diffsol_dirs = [f"target/criterion/{solver}_{problem['name']}_{arg}" for arg in problem['arg']]

        for diffsol_dir in diffsol_dirs:
            with open(f"{diffsol_dir}/new/estimates.json") as f:
                bench = json.load(f)
                estimates[problem['name']][solver]['diffsol'].append(bench["mean"]["point_estimate"])

fig = plt.figure(figsize=(12, 12))
((ax1, ax2), (ax3, ax4)) = fig.subplots(2, 2, sharex=True, sharey=True)
for problem in problems:
    reference = np.array(estimates[problem['name']]['reference'])
    for solver in problem['solvers']:
        y = np.array(estimates[problem['name']][solver]['diffsol']) / reference
        label = f"{problem['name']}_{solver}"
        if 'tr_bdf2' in solver:
            ax = ax1
        elif 'esdirk' in solver:
            ax = ax2
        else:
            ax = ax3
        if problem['arg'] is None:
            # plot a horizontal line
            ax.plot([5, 30], [y, y], label=label)
        elif 'robertson_ode' in problem['name']:
            ax.plot(np.sqrt(problem['arg']), y, label=label)
        else:
            ax.plot(problem['arg'], y, label=label)
for ax in [ax1, ax2, ax3]:
    ax.set_yscale('log')
    ax.set_ylabel("Time relative to sundials")
    ax.set_xlabel("Problem size (grid size if applicable)")
    ax.legend()
ax1.set_title("TR-BDF2 solver")
ax2.set_title("ESDIRK solver")
ax3.set_title("BDF solver")
plt.savefig("benches.png")
        
    