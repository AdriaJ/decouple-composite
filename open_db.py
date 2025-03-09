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

db_path = "database/t_vs_r"
figures_path = "figures"

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
    with open('db_config.yaml', 'r') as config_file:
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

    #select reconstructions
    df_recos = df[df["type"] != "gt"]
    idx_best = df_recos.groupby(['srf','r12', 'seed', 'type'])["RelErr_srf4"].idxmin()
    best_l2 = df.loc[idx_best]
    # idx30 = df_recos.groupby(['r12', 'seed', 'type'])["RelErr_3.0"].idxmin()
    # best_30 = df.loc[idx30]

    # best_15.groupby(['fgbgR', 'type'])['RelErr_srf4'].agg(['mean', 'std', 'median', 'min', 'max'])
    # best_15[best_15['type'] == 'blasso'].groupby(['fgbgR'])['RelErr_srf4'].agg(['median']).plot()
    # best_15.groupby(['fgbgR', 'type'])['RelErr_srf4'].quantile([0.25, 0.75])

    times15_comp = best_l2[best_l2['type'] == 'composite'][["r12", 'time', 'ndcp_time']].values
    times15_blasso = best_l2[best_l2['type'] == 'blasso'][["r12", 'time']].values
    plt.figure()
    # plt.scatter(times15_comp[:, 0], times15_comp[:, 1], color='red', marker='x', alpha=.5, label='Composite')
    # plt.scatter(times15_comp[:, 0], times15_comp[:, 2], color='red', marker='+', alpha=.5, label='NDCP-Composite')
    # plt.scatter(times15_blasso[:, 0], times15_blasso[:, 1], color='blue', marker='x', alpha=.5, label='BLASSO')
    blasso = best_l2[best_l2['type'] == 'blasso'].groupby(['r12'])['time'].agg(['median'])
    plt.plot(blasso.index, blasso['median'], label='BLASSO', color='blue', marker='+')
    plt.fill_between(blasso.index, best_l2[best_l2['type'] == 'blasso'].groupby(['r12'])['time'].quantile(0.25),
                    best_l2[best_l2['type'] == 'blasso'].groupby(['r12'])['time'].quantile(0.75), alpha=0.2,
                    color='blue')
    compo = best_l2[best_l2['type'] == 'composite'].groupby(['r12'])[['time', 'ndcp_time']].agg(['median'])
    plt.plot(compo.index, compo['time']['median'], label='Composite', color='red', marker='x')
    plt.plot(compo.index, compo['ndcp_time']['median'], label='NDCP-Composite', color='red', marker='+')
    plt.fill_between(compo.index, best_l2[best_l2['type'] == 'composite'].groupby(['r12'])['time'].quantile(0.25),
                    best_l2[best_l2['type'] == 'composite'].groupby(['r12'])['time'].quantile(0.75), alpha=0.2, color='red')
    plt.fill_between(compo.index, best_l2[best_l2['type'] == 'composite'].groupby(['r12'])['ndcp_time'].quantile(0.25),
                    best_l2[best_l2['type'] == 'composite'].groupby(['r12'])['ndcp_time'].quantile(0.75), alpha=0.2, color='red')
    plt.legend()
    plt.show()

    fig = plt.figure(figsize=(12, 4))
    axes = fig.subplots(1, 2, sharey=True)
    ax = axes[0]
    blasso = best_l2[best_l2['type'] == 'blasso'].groupby(['r12'])['RelErr_srf4'].agg(['median'])
    ax.plot(blasso.index, blasso['median'], label='BLASSO', color='blue', marker='+')
    ax.fill_between(blasso.index, best_l2[best_l2['type'] == 'blasso'].groupby(['r12'])['RelErr_srf4'].quantile(0.25),
                    best_l2[best_l2['type'] == 'blasso'].groupby(['r12'])['RelErr_srf4'].quantile(0.75), alpha=0.2, color='blue')
    # valb = best_15[best_15['type'] == 'blasso'][["r12", 'RelErr_srf4']].values
    # ax.scatter(valb[:, 0], valb[:, 1], color='blue', marker='x', alpha=.5)
    compo = best_l2[best_l2['type'] == 'composite'].groupby(['r12'])['RelErr_srf4'].agg(['median'])
    ax.plot(compo.index, compo['median'], label='Composite', color='red', marker='+')
    ax.fill_between(compo.index, best_l2[best_l2['type'] == 'composite'].groupby(['r12'])['RelErr_srf4'].quantile(0.25),
                    best_l2[best_l2['type'] == 'composite'].groupby(['r12'])['RelErr_srf4'].quantile(0.75), alpha=0.2, color='red')
    # valc = best_15[best_15['type'] == 'composite'][["r12", 'RelErr_srf4']].values
    # ax.scatter(valc[:, 0], valc[:, 1], color='red', marker='x', alpha=.5)
    ax.set_xlabel("Contrast")
    ax.set_title(r"Relative L2 error, $\sigma=1.5$")
    ax.legend()
    plt.show()

    # ax = axes[1]
    # blasso = best_30[best_30['type'] == 'blasso'].groupby(['r12'])['RelErr_3.0'].agg(['median'])
    # ax.plot(blasso.index, blasso['median'], label='BLASSO', color='blue', marker='+')
    # ax.fill_between(blasso.index, best_30[best_30['type'] == 'blasso'].groupby(['r12'])['RelErr_3.0'].quantile(0.25),
    #                  best_30[best_30['type'] == 'blasso'].groupby(['r12'])['RelErr_3.0'].quantile(0.75), alpha=0.2, color='blue')
    # # valb = best_30[best_30['type'] == 'blasso'][["r12", 'RelErr_3.0']].values
    # # ax.scatter(valb[:, 0], valb[:, 1], color='blue', marker='x', alpha=.5)
    # compo = best_30[best_30['type'] == 'composite'].groupby(['r12'])['RelErr_3.0'].agg(['median'])
    # ax.plot(compo.index, compo['median'], label='Composite', color='red', marker='+')
    # ax.fill_between(compo.index, best_30[best_30['type'] == 'composite'].groupby(['r12'])['RelErr_3.0'].quantile(0.25),
    #                     best_30[best_30['type'] == 'composite'].groupby(['r12'])['RelErr_3.0'].quantile(0.75), alpha=0.2, color='red')
    # # valc = best_30[best_30['type'] == 'composite'][["r12", 'RelErr_3.0']].values
    # # ax.scatter(valc[:, 0], valc[:, 1], color='red', marker='x', alpha=.5)
    # ax.set_xlabel("Contrast")
    # ax.set_title(r"Relative L2 error, $\sigma=3.0$")
    # ax.legend()
    # if save_pdf:
    #     plt.savefig(os.path.join(figures_path, "metrics_rl2.pdf"))
    # plt.show()

    # select reconstructions for L1 error
    df_recos = df[df["type"] != "gt"]
    L1idx_best = df_recos.groupby(['srf','r12', 'seed', 'type'])["RelL1Err_srf4"].idxmin()
    best_l1 = df.loc[L1idx_best]

    # Relative L1 error
    fig = plt.figure(figsize=(12, 4))
    axes = fig.subplots(1, 2, sharey=True)
    ax = axes[0]
    blasso = best_l1[best_l1['type'] == 'blasso'].groupby(['r12'])['RelL1Err_srf4'].agg(['median'])
    ax.plot(blasso.index, blasso['median'], label='BLASSO', color='blue', marker='+')
    ax.fill_between(blasso.index, best_l1[best_l1['type'] == 'blasso'].groupby(['r12'])['RelL1Err_srf4'].quantile(0.25),
                     best_l1[best_l1['type'] == 'blasso'].groupby(['r12'])['RelL1Err_srf4'].quantile(0.75), alpha=0.2, color='blue')
    # valb = best_l1[best_l1['type'] == 'blasso'][["r12", 'RelL1Err_srf4']].values
    # ax.scatter(valb[:, 0], valb[:, 1], color='blue', marker='x', alpha=.5)
    compo = best_l1[best_l1['type'] == 'composite'].groupby(['r12'])['RelL1Err_srf4'].agg(['median'])
    ax.plot(compo.index, compo['median'], label='Composite', color='red', marker='+')
    ax.fill_between(compo.index, best_l1[best_l1['type'] == 'composite'].groupby(['r12'])['RelL1Err_srf4'].quantile(0.25),
                     best_l1[best_l1['type'] == 'composite'].groupby(['r12'])['RelL1Err_srf4'].quantile(0.75), alpha=0.2, color='red')
    # valc = best_l1[best_l1['type'] == 'composite'][["r12", 'RelL1Err_srf4']].values
    # ax.scatter(valc[:, 0], valc[:, 1], color='red', marker='x', alpha=.5)
    ax.set_xlabel("Contrast")
    ax.set_title(r"Relative L1 error, $\sigma=1.5$")
    ax.legend()
    plt.show()

    # ax = axes[1]
    # blasso = L1best_30[L1best_30['type'] == 'blasso'].groupby(['r12'])['RelL1Err_3.0'].agg(['median'])
    # ax.plot(blasso.index, blasso['median'], label='BLASSO', color='blue', marker='+')
    # ax.fill_between(blasso.index, L1best_30[L1best_30['type'] == 'blasso'].groupby(['r12'])['RelL1Err_3.0'].quantile(0.25),
    #                  L1best_30[L1best_30['type'] == 'blasso'].groupby(['r12'])['RelL1Err_3.0'].quantile(0.75), alpha=0.2, color='blue')
    # # valb = L1best_30[L1best_30['type'] == 'blasso'][["r12", 'RelL1Err_3.0']].values
    # # ax.scatter(valb[:, 0], valb[:, 1], color='blue', marker='x', alpha=.5)
    # compo = L1best_30[L1best_30['type'] == 'composite'].groupby(['r12'])['RelL1Err_3.0'].agg(['median'])
    # ax.plot(compo.index, compo['median'], label='Composite', color='red', marker='+')
    # ax.fill_between(compo.index, L1best_30[L1best_30['type'] == 'composite'].groupby(['r12'])['RelL1Err_3.0'].quantile(0.25),
    #                     L1best_30[L1best_30['type'] == 'composite'].groupby(['r12'])['RelL1Err_3.0'].quantile(0.75), alpha=0.2, color='red')
    # # valc = L1best_30[L1best_30['type'] == 'composite'][["r12", 'RelL1Err_3.0']].values
    # # ax.scatter(valc[:, 0], valc[:, 1], color='red', marker='x', alpha=.5)
    # ax.set_xlabel("Contrast")
    # ax.set_title(r"Relative L1 error, $\sigma=3.0$")
    # ax.legend()
    # if save_pdf:
    #     plt.savefig(os.path.join(figures_path, "metrics_l1.pdf"))
    # plt.show()


    # Reconstruction time
    fig = plt.figure(figsize=(15, 7))
    axes = fig.subplots(1, 2, sharey=True)
    ax = axes[0]
    blasso = best_l2[best_l2['type'] == 'blasso'].groupby(['r12'])['time'].agg(['median'])
    ax.plot(blasso.index, blasso['median'], label='BLASSO', color='blue', marker='+')
    ax.fill_between(blasso.index, best_l2[best_l2['type'] == 'blasso'].groupby(['r12'])['time'].quantile(0.25),
                    best_l2[best_l2['type'] == 'blasso'].groupby(['r12'])['time'].quantile(0.75), alpha=0.2, color='blue')
    compo = best_l2[best_l2['type'] == 'composite'].groupby(['r12'])[['time', 'ndcp_time']].agg(['median'])
    ax.plot(compo.index, compo['time']['median'], label='Composite', color='red', marker='+')
    ax.plot(compo.index, compo['ndcp_time']['median'], label='Composite', color='red', marker='x')
    ax.fill_between(compo.index, best_l2[best_l2['type'] == 'composite'].groupby(['r12'])['time'].quantile(0.25),
                    best_l2[best_l2['type'] == 'composite'].groupby(['r12'])['time'].quantile(0.75), alpha=0.2, color='red')
    ax.fill_between(compo.index, best_l2[best_l2['type'] == 'composite'].groupby(['r12'])['ndcp_time'].quantile(0.25),
                    best_l2[best_l2['type'] == 'composite'].groupby(['r12'])['ndcp_time'].quantile(0.75), alpha=0.2,
                    color='red')
    ax.set_xlabel("Contrast")
    ax.set_title(r"Reconstruction time for best case with $\sigma=1.5$")
    ax.legend()
    plt.show()

    # ax = axes[1]
    # blasso = best_30[best_30['type'] == 'blasso'].groupby(['r12'])['time'].agg(['median'])
    # ax.plot(blasso.index, blasso['median'], label='BLASSO', color='blue', marker='+')
    # ax.fill_between(blasso.index, best_30[best_30['type'] == 'blasso'].groupby(['r12'])['time'].quantile(0.25),
    #                  best_30[best_30['type'] == 'blasso'].groupby(['r12'])['time'].quantile(0.75), alpha=0.2, color='blue')
    # compo = best_30[best_30['type'] == 'composite'].groupby(['r12'])[['time', 'ndcp_time']].agg(['median'])
    # ax.plot(compo.index, compo['time']['median'], label='Composite', color='red', marker='+')
    # ax.plot(compo.index, compo['ndcp_time']['median'], label='Composite', color='red', marker='x')
    # ax.fill_between(compo.index, best_30[best_30['type'] == 'composite'].groupby(['r12'])['time'].quantile(0.25),
    #                     best_30[best_30['type'] == 'composite'].groupby(['r12'])['time'].quantile(0.75), alpha=0.2, color='red')
    # ax.fill_between(compo.index, best_30[best_30['type'] == 'composite'].groupby(['r12'])['ndcp_time'].quantile(0.25),
    #                 best_30[best_30['type'] == 'composite'].groupby(['r12'])['ndcp_time'].quantile(0.75), alpha=0.2,
    #                 color='red')
    # ax.set_xlabel("Contrast")
    # ax.set_title(r"Reconstruction time for best case with $\sigma=3.0$")
    # ax.legend()
    # if save_pdf:
    #     plt.savefig(os.path.join(figures_path, "time_comp.pdf"))
    # plt.show()


    # todo: Make the figures for time and accuracy w.r.t. contrast (r12)
    # todo: write the code and make figure for time and accuracy vs srf



