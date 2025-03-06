"""
Simulate data and reconstruct.
"""

import subprocess
import os
import numpy as np
import shutil

from joblib import Parallel, delayed, cpu_count

r12s = [2.,]  # [.5, 1.5]  # [.5, .75, 1., 1.5, 2.]
reps = 4  #10
l1fs = [.3]  # [.1, .2, .3] [.2, ]  #[.2, .3, .4]
l2s = [1e-2]  #[1e-2, 2e-2, 5e-2, 1e-1,] [1e-3]  # [1e-4, 1e-3, 1e-2, 1e-1]
lfs = [.3]  #[.1, .2, .3]  # [.3, ]  # [.1, 0.2, 0.3, .4]

srfs = [8, ]  # [8, 12, 16]

cwd = "."

njobs = cpu_count()/2  # 1

def run_seed(seed):
    print(f"Running pipeline with seed: {seed}")
    seed_path = os.path.join(r12_path, f"{seed}")
    if not os.path.exists(seed_path):
        os.makedirs(seed_path)

    subprocess.run(['python', os.path.join(cwd, "db_simulate.py"), "--seed", f"{seed:d}",
                    "--srf", str(srf), "--save", seed_path, "--r12", str(r12)], check=True, text=True)

    # saves a list of npz files, some named as blasso_*.npz and some as composite_*_*.npz,
    # where * stands for the actual float value of the regularization parameters.
    # If one argument is missing the associated reconstruction is not performed.
    subprocess.run(['python', os.path.join(cwd, "db_reconstruct.py"),
                    "--l1f", *(str(u) for u in l1fs),
                    "--l2", *(str(u) for u in l2s),
                    "--lf", *(str(u) for u in lfs),
                    "--data_path", seed_path], check=True)

if __name__ == "__main__":

    db_path = os.path.join(cwd, "database")
    if not os.path.exists(db_path):
        os.makedirs(db_path)

    for srf in srfs:
        print(f"Running pipeline with srf: {srf}")
        srf_path = os.path.join(db_path, f"srf_{srf}")
        if not os.path.exists(srf_path):
            os.makedirs(srf_path)
        shutil.copy2(os.path.join(cwd,"db_config.yaml"),
                     os.path.join(srf_path, "db_config.yaml"))
        for r12 in r12s:
            print(f"Running pipeline with r12: {r12:.2f}")
            r12_path = os.path.join(srf_path, f"r12_{r12}")
            if not os.path.exists(r12_path):
                os.makedirs(r12_path)

            seeds = np.random.choice(1_000_000, reps, replace=False)
            Parallel(n_jobs=njobs)(delayed(run_seed)(seed) for seed in seeds)

