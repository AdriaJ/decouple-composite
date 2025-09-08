"""
gt_data.npz keys: "img", "background", "measurements", "noise_meas"
composite_f1_l2.npz keys: "x1", "x2", "t", "lambda1", "lambda2"
blasso_f.npz keys: "x", "t", "lambda_"
"""
import os
import argparse
import re
import yaml


import numpy as np
import matplotlib.pyplot as plt

import pandas as pd

from matplotlib import use
use("Qt5Agg")

db_path = "database/rkhsTk/t_vs_srf" #"database2/rkhsTk" # "database/noOpL2/t_vs_srf" or "database/noOpL2/t_vs_r"
figures_path = "figures/rkhs"
plt.rcParams.update({'font.size': 13})

srf_repr = 4

save_pdf = False

def repr_kernel(kernel_std, srf_repr, Ngrid):
    repr_std = kernel_std / srf_repr
    repr_std_int = np.floor(repr_std * Ngrid).astype(int)
    repr_width = 3 * 2 * repr_std_int + 1  # Length of the Gaussian kernel
    representation_kernel = np.exp(
        -0.5 * ((np.arange(repr_width) - (repr_width - 1) / 2) ** 2) / ((repr_std * Ngrid) ** 2))
    norm_repr = (np.sqrt(2 * np.pi) * repr_std)
    representation_kernel /= norm_repr
    return representation_kernel

def relL2Error(reco, source):
    return np.linalg.norm(reco - source) / np.linalg.norm(source)

def relL1Error(reco, source):
    return np.linalg.norm(reco - source, 1) / np.linalg.norm(source, 1)

def repr(signal, kernel):
    return np.convolve(signal, kernel, mode='same')

def path_error(kernel_std, srf_repr, ord=2):
    path_error = {}
    # browse the folders in db_path
    for srf_dir in next(os.walk(db_path))[1]:
        srf_path = os.path.join(db_path, srf_dir)
        for r12_dir in next(os.walk(srf_path))[1]:
            # assert dir.startswith("r12_")
            r12_path = os.path.join(srf_path, r12_dir)
            for seedtxt in os.listdir(r12_path):
                seed_path = os.path.join(r12_path, seedtxt)
                composite_recos = [f for f in os.listdir(seed_path) if f.startswith("composite")]
                blasso_recos = [f for f in os.listdir(seed_path) if f.startswith("blasso")]
                # Load source and measurements
                gtdata = np.load(os.path.join(seed_path, "gt_data.npz"))
                kernel = repr_kernel(kernel_std, srf_repr, gtdata["img"].shape[0])
                repr_x1gt = repr(gtdata["img"], kernel)
                # Relative l2 error after convolution with representation kernel
                blasso_errors = []
                composite_errors = []
                for comp in composite_recos:
                    compdata = np.load(os.path.join(seed_path, comp))
                    if ord==2:
                        err = relL2Error(repr(compdata["x1"], kernel), repr_x1gt)
                    elif ord==1:
                        err = relL1Error(repr(compdata["x1"], kernel), repr_x1gt)
                    path_error[os.path.join(seed_path, comp)] = err
                for bl in blasso_recos:
                    bldata = np.load(os.path.join(seed_path, bl))
                    if ord==2:
                        err = relL2Error(repr(bldata["x"], kernel), repr_x1gt)
                    elif ord==1:
                        err = relL1Error(repr(bldata["x"], kernel), repr_x1gt)
                    path_error[os.path.join(seed_path, bl)] = err
    return path_error

if __name__ == "__main__":
    with open('db_files/db_config.yaml', 'r') as config_file:
        config = yaml.safe_load(config_file)
    kernel_std = config["meas_model"]["kernel_std"]
    # repr_kernel = repr_kernel(kernel_std, srf_repr, Ngrid)

    list_paths = []
    for dirpath, dirnames, filenames in os.walk(db_path):
        # check if dirpath starts with db_path then a directory with r12
        if len(dirnames) == 0:
            list_paths += [os.path.join(dirpath, f) for f in filenames if f.endswith(".npz")]

    # df = pd.DataFrame(columns=["path", "fgbgR", "seed", "type", "l1f", "l2", "lf"])
    tmp = {"path": [], "srf": [], "r12": [], "seed": [], "type": [], "l1f": [], "l2": [], "lf": []}
    for path in list_paths:
        l = path.split('/')
        srf = int(l[-4][4:])
        r12 = float(l[-3][4:])
        seed = int(l[-2])
        if l[-1].startswith("gt"):
            type = "gt"
            l1f, l2, lf = None, None, None
        else:
            parts = l[-1].split("_")
            if parts[0] == "composite":
                type = "composite"
                l1f, l2 = float(parts[1]), float(parts[2][:-4])
                lf = None
            elif parts[0] == "blasso":
                type = "blasso"
                l1f, l2 = None, None
                lf = float(parts[1][:-4])
        tmp["path"].append(path)
        tmp["srf"].append(srf)
        tmp["r12"].append(r12)
        tmp["seed"].append(seed)
        tmp["type"].append(type)
        tmp["l1f"].append(l1f)
        tmp["l2"].append(l2)
        tmp["lf"].append(lf)
    df = pd.DataFrame(tmp)

    # fill the dataframe with computation time and errors (2 new columns)
    times = []
    ndcp_times = []
    for row in df.itertuples():
        if row[5] == "blasso" or row[5] == "composite":
            data = np.load(row[1])
            times.append(data["t"][0])
            if row[5] == "composite":
                ndcp_times.append(data["ndcp_time"][0])
            else:
                ndcp_times.append(None)
        else:
            times.append(None)
            ndcp_times.append(None)
    df["time"] = times
    df["ndcp_time"] = ndcp_times

    # df[['r12', 'seed', 'type', 'l1f', 'l2', 'lf', 'time', 'ndcp_time']]

    df[f"RelErr_srf{srf_repr}"] = df["path"].map(path_error(kernel_std, srf_repr, ord=2))
    df[f"RelL1Err_srf{srf_repr}"] = df["path"].map(path_error(kernel_std, srf_repr, ord=1))

    # for each srf, each r12 and each seed, extract the minimum error,
    # compare if the minimum error is obtained with the same regularization parameter -> TODO
    # Plot the minimum median and interquartile minimum error
    # Also reconstruction time for best case

    #select reconstructions for L2 error
    df_recos = df[df["type"] != "gt"]
    idx_best = df_recos.groupby(['srf','r12', 'seed', 'type'])[f"RelErr_srf{srf_repr}"].idxmin()
    best_l2 = df.loc[idx_best]
    # select reconstructions for L1 error
    df_recos = df[df["type"] != "gt"]
    L1idx_best = df_recos.groupby(['srf','r12', 'seed', 'type'])[f"RelL1Err_srf{srf_repr}"].idxmin()
    best_l1 = df.loc[L1idx_best]

    times15_comp = best_l2[best_l2['type'] == 'composite'][["r12", 'time', 'ndcp_time']].values
    times15_blasso = best_l2[best_l2['type'] == 'blasso'][["r12", 'time']].values

    if db_path.endswith("vs_r"):
        crit = 'r12'
        xaxis_name = r"Contrast $r_{12}$"
        xticks = [0.5, 1., 2., 4.]
    elif db_path.endswith("vs_srf"):
        crit = 'srf'
        xaxis_name = r"Super-resolution factor $srf$"
        xticks = df['srf'].unique()
    else:
        raise KeyError("Database path must end with 'vs_r' or 'vs_srf'")

    plt.figure()
    blasso = best_l2[best_l2['type'] == 'blasso'].groupby([crit])['time'].agg(['median'])
    plt.plot(blasso.index, blasso['median'], label='BLASSO', color='blue', marker='v')
    plt.fill_between(blasso.index, best_l2[best_l2['type'] == 'blasso'].groupby([crit])['time'].quantile(0.25),
                    best_l2[best_l2['type'] == 'blasso'].groupby([crit])['time'].quantile(0.75), alpha=0.2,
                    color='blue')
    compo = best_l2[best_l2['type'] == 'composite'].groupby([crit])[['time', 'ndcp_time']].agg(['median'])
    plt.plot(compo.index, compo['time']['median'], label='Composite', color='red', marker='^')
    plt.plot(compo.index, compo['ndcp_time']['median'], label='NDCP-Composite', color='red', marker='o')
    plt.fill_between(compo.index, best_l2[best_l2['type'] == 'composite'].groupby([crit])['time'].quantile(0.25),
                    best_l2[best_l2['type'] == 'composite'].groupby([crit])['time'].quantile(0.75), alpha=0.2, color='red')
    plt.fill_between(compo.index, best_l2[best_l2['type'] == 'composite'].groupby([crit])['ndcp_time'].quantile(0.25),
                    best_l2[best_l2['type'] == 'composite'].groupby([crit])['ndcp_time'].quantile(0.75), alpha=0.2, color='red')
    plt.legend(loc="upper right")
    plt.xlabel(xaxis_name)
    plt.xticks(xticks)
    plt.ylabel("Time (s)")
    plt.yscale('log')
    plt.tight_layout()
    plt.subplots_adjust(left=0.140)
    if save_pdf:
        plt.savefig(os.path.join(figures_path, f"time_vs_{crit}.pdf"))
    plt.show()

    fig = plt.figure(figsize=(11, 4))
    axes = fig.subplots(1, 2, sharey=(crit=="srf"))
    ax = axes[0]
    blasso = best_l2[best_l2['type'] == 'blasso'].groupby([crit])[f'RelErr_srf{srf_repr}'].agg(['median'])
    ax.plot(blasso.index, blasso['median'], label='BLASSO', color='blue', marker='+')
    ax.fill_between(blasso.index, best_l2[best_l2['type'] == 'blasso'].groupby([crit])[f'RelErr_srf{srf_repr}'].quantile(0.25),
                    best_l2[best_l2['type'] == 'blasso'].groupby([crit])[f'RelErr_srf{srf_repr}'].quantile(0.75), alpha=0.2, color='blue')
    compo = best_l2[best_l2['type'] == 'composite'].groupby([crit])[f'RelErr_srf{srf_repr}'].agg(['median'])
    ax.plot(compo.index, compo['median'], label='Composite', color='red', marker='+')
    ax.fill_between(compo.index, best_l2[best_l2['type'] == 'composite'].groupby([crit])[f'RelErr_srf{srf_repr}'].quantile(0.25),
                    best_l2[best_l2['type'] == 'composite'].groupby([crit])[f'RelErr_srf{srf_repr}'].quantile(0.75), alpha=0.2, color='red')
    ax.set_xlabel(xaxis_name)
    ax.set_xticks(xticks)
    ax.set_title(r"Relative L2 error")
    ax.legend()
    ax = axes[1]
    blasso = best_l1[best_l1['type'] == 'blasso'].groupby([crit])[f'RelL1Err_srf{srf_repr}'].agg(['median'])
    ax.plot(blasso.index, blasso['median'], label='BLASSO', color='blue', marker='+')
    ax.fill_between(blasso.index, best_l1[best_l1['type'] == 'blasso'].groupby([crit])[f'RelL1Err_srf{srf_repr}'].quantile(0.25),
                     best_l1[best_l1['type'] == 'blasso'].groupby([crit])[f'RelL1Err_srf{srf_repr}'].quantile(0.75), alpha=0.2, color='blue')
    compo = best_l1[best_l1['type'] == 'composite'].groupby([crit])[f'RelL1Err_srf{srf_repr}'].agg(['median'])
    ax.plot(compo.index, compo['median'], label='Composite', color='red', marker='+')
    ax.fill_between(compo.index, best_l1[best_l1['type'] == 'composite'].groupby([crit])[f'RelL1Err_srf{srf_repr}'].quantile(0.25),
                     best_l1[best_l1['type'] == 'composite'].groupby([crit])[f'RelL1Err_srf{srf_repr}'].quantile(0.75), alpha=0.2, color='red')
    ax.set_xlabel(xaxis_name)
    ax.set_title(r"Relative L1 error")
    ax.legend()
    ax.set_xticks(xticks)
    plt.tight_layout()
    plt.subplots_adjust(wspace=0.15)
    if save_pdf:
        plt.savefig(os.path.join(figures_path, f"error_vs_{crit}.pdf"))
    plt.show()

    # Compute average gain in reconstruction time
    comp_df = df[df['type']=='composite']
    # rate_time_r12 = comp_df.groupby("r12")['time'].mean()/comp_df.groupby("r12")['ndcp_time'].mean()
    # (comp_df['time']).mean()/(comp_df['ndcp_time']).mean()
    rate_time_r12 = pd.concat([comp_df['r12'], comp_df['time']/comp_df['ndcp_time']], axis=1, keys=['r12', 'rate']).groupby("r12").mean()
    print(f"Average rate of decoupled time: {rate_time_r12['rate'].mean():.3f}")

