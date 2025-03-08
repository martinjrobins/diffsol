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
        "solvers": ["nalgebra_esdirk34", "nalgebra_tr_bdf2", "nalgebra_bdf", "nalgebra_bdf_diffsl"]
    },
    {
        "name": "robertson_ode",
        "reference_name": "robertson_ode_klu",
        "arg": [25, 100, 400, 900],
        #"solvers": ["faer_sparse_bdf_klu"],
        "solvers": ["faer_sparse_bdf"],
    },
    {
        "name": "heat2d",
        "reference_name": "heat2d_klu",
        "arg": [5, 10, 20, 30],
        "solvers": ["faer_sparse_esdirk", "faer_sparse_tr_bdf2", "faer_sparse_bdf", "faer_sparse_bdf_diffsl"]
        #"solvers": ["faer_sparse_esdirk_klu", "faer_sparse_tr_bdf2_klu", "faer_sparse_bdf_klu", "faer_sparse_bdf_klu_diffsl"]
    },
    {
        "name": "foodweb",
        "reference_name": "foodweb_bnd",
        "arg": [5, 10, 20, 30],
        "solvers": ["faer_sparse_esdirk", "faer_sparse_tr_bdf2", "faer_sparse_bdf", "faer_sparse_bdf_diffsl"]
        #"solvers": ["faer_sparse_esdirk_klu", "faer_sparse_tr_bdf2_klu", "faer_sparse_bdf_klu", "faer_sparse_bdf_klu_diffsl"]
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

fig1 = plt.figure(figsize=(12, 4))
(ax1, ax2) = fig1.subplots(1, 2, sharex=True, sharey=True)
fig2 = plt.figure(figsize=(6, 4))
ax3 = fig2.subplots(1, 1)
fig3 = plt.figure(figsize=(6, 4))
ax4 = fig3.subplots(1, 1)
for problem in problems:
    for solver in problem['solvers']:
        reference = np.array(estimates[problem['name']]['reference'])
        #if 'diffsl' in solver:
        #    reference = np.array(estimates[problem['name']][solver.replace("_diffsl", "")]['diffsol'])
        y = np.array(estimates[problem['name']][solver]['diffsol']) / reference
        label = f"{problem['name']}_{solver}"
        if 'tr_bdf2' in solver:
            axs = (ax1,)
        elif 'esdirk' in solver:
            axs = (ax2,)
        elif 'diffsl' in solver:
            axs = (ax4,)
        #elif 'nalgebra_bdf' in solver or 'faer_sparse_bdf' in solver and 'klu' not in solver and 'robertson_ode' not in problem['name']:
        #    axs = (ax3, ax4)
        else:
            axs = (ax3,)
        if 'foodweb' in problem['name']:
            color = 'red'
        elif 'heat2d' in problem['name']:
            color = 'blue'
        elif 'robertson_ode' in problem['name']:
            color = 'green'
        else:
            color = 'orange'

        if problem['arg'] is None:
            # plot a horizontal line
            for ax in axs: 
                ax.plot([5, 30], [y, y], label=label, color=color)
        elif 'robertson_ode' in problem['name']:
            for ax in axs: 
                ax.plot(np.sqrt(problem['arg']), y, label=label, color=color)
        else:
            for ax in axs: 
                ax.plot(problem['arg'], y, label=label, color=color)
for ax in [ax1, ax2, ax3, ax4]:
    ax.set_yscale('log')
    ax.set_yticks([0.1, 1, 10])
    ax.tick_params(axis='y', which='minor')
    ax.set_ylabel("Time relative to sundials")
    ax.set_xlabel("Problem size (grid size if applicable)")
    ax.legend()
ax1.set_title("TR-BDF2 solver")
ax2.set_title("ESDIRK solver")
ax3.set_title("BDF solver")
ax4.set_title("BDF solver + DiffSL")

basedir = "book/src/benchmarks/images"
fig1.savefig(f"{basedir}/bench_tr_bdf2_esdirk.svg")
fig2.savefig(f"{basedir}/bench_bdf.svg")
fig3.savefig(f"{basedir}/bench_bdf_diffsl.svg")
basedir = "."
fig1.savefig(f"{basedir}/bench_tr_bdf2_esdirk.png")
fig2.savefig(f"{basedir}/bench_bdf.png")
fig3.savefig(f"{basedir}/bench_bdf_diffsl.png")
        
    