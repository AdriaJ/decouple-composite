"""
Visualize all the figures related to one case.
One case is identified with various parameters: srf, r12, seed.

gt_data.npz keys: "img", "background", "measurements", "noise_meas"
composite_f1_l2.npz keys: "x1", "x2", "x1ndcp", "t", "lambda1", "lambda2", "ndcp_time
blasso_f.npz keys: "x", "t", "lambda_"
"""
import os
import yaml
import numpy as np
import matplotlib

matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt

from utils import relL1Error, relL2Error

srf = 8
r12 = 1.
# r12 = 1.0 :927_539, 891_665
# r12 = 2.0 : 783_710, 784_972
seed = 718494  # 575355 #  117809

srf_repr = 4
save_plots = False
save_ext = ".pdf"  # ".png"  # .pdf

# db_path = "dev/database"
db_path = "database/rkhsTk"
figures_path = "figures"

if __name__ == "__main__":
    case_path = os.path.join(db_path, f"srf_{srf}", f"r12_{r12:.1f}", f"{seed}")
    if save_plots:
        figures_path = os.path.join(figures_path, f"srf{srf}r12{r12:1f}seed{seed}")
        if not os.path.exists(figures_path):
            os.makedirs(figures_path)

    gt = np.load(os.path.join(case_path, "gt_data.npz"))
    blasso_filenames = [ n for n in os.listdir(case_path) if n.startswith("blasso") ]
    blasso_filenames.sort()
    composite_filenames = [ n for n in os.listdir(case_path) if n.startswith("composite") ]
    blasso = [ np.load(os.path.join(case_path, n)) for n in blasso_filenames ]
    composite = [ np.load(os.path.join(case_path, n)) for n in composite_filenames ]

    # extract the regularization parameters
    lfs = [ float(name.split("_")[-1][:-4]) for name in blasso_filenames ]
    l1fsl2 = []
    l1fs = []
    l2s = []
    for name in composite_filenames:
        lambda1f, lambda2 = name.split("_")[1:]
        lambda1f = float(lambda1f)
        lambda2 = float(lambda2[:-4])
        if lambda1f not in l1fs:
            l1fs.append(lambda1f)
        if lambda2 not in l2s:
            l2s.append(lambda2)
        l1fsl2.append((lambda1f, lambda2))
    composite_filenames, composite, l1fsl2 = zip(*sorted(zip(composite_filenames, composite, l1fsl2),
                                                         key=lambda x: x[-1]))
    l1fs.sort()
    l2s.sort()

    # --------------------------------------------------------------
    # Plot the ground truth
    img = gt["img"]
    background = gt["background"]
    y = gt["measurements"]
    Ngrid = img.shape[0]
    locs = np.arange(Ngrid) / Ngrid

    plt.figure(figsize=(15, 4))
    plt.subplot(131)
    plt.stem(np.arange(img.shape[0])[img != 0], img[img != 0], basefmt="C7--", linefmt="C1-")
    plt.stem([0, Ngrid-1], [0, 0], markerfmt='white', basefmt='C7--')
    plt.title("Original image (without background)")
    plt.subplot(132)
    plt.plot(np.arange(Ngrid), background, c='#1f77b4')
    plt.title("Original background)")
    plt.hlines(0, 0, Ngrid-1, ls="--", color='#7f7f7f')
    plt.subplot(133)
    plt.stem(np.arange(img.shape[0])[img != 0], img[img != 0], basefmt="C7--", linefmt="C1-")
    plt.stem([0, Ngrid-1], [0, 0], markerfmt='white', basefmt='C7--')
    plt.plot(np.arange(Ngrid), background, c='#1f77b4')
    plt.title("Original image (with background)")
    if save_plots:
        plt.savefig(os.path.join(figures_path, "gt" + save_ext))
    plt.show()

    # --------------------------------------------------------------
    # representation kernel
    with open(os.path.join(db_path, f"srf_{srf}",'db_config.yaml'), 'r') as config_file:
        config = yaml.safe_load(config_file)
    kernel_std = config["meas_model"]["kernel_std"]
    repr_std = kernel_std / srf_repr
    repr_std_int = np.floor(repr_std * Ngrid).astype(int)
    repr_width = 3 * 2 * repr_std_int + 1  # Length of the Gaussian kernel
    representation_kernel = np.exp(
        -0.5 * ((np.arange(repr_width) - (repr_width - 1) / 2) ** 2) / ((repr_std * Ngrid) ** 2))
    norm_repr = (np.sqrt(2 * np.pi) * repr_std)
    representation_kernel /= norm_repr

    plt.figure()
    repr_source = np.convolve(gt["img"], representation_kernel, mode="same")
    plt.plot(np.arange(repr_source.shape[0]), repr_source, c='#ff7f0e',)
    plt.suptitle(f"FSource foreground merged ($r_{12}$: {r12:.1f}, SRF: {srf:d}, SNR: 10 dB)")
    if save_plots:
        plt.savefig(os.path.join(figures_path, "foreground_merged_source" + save_ext))
    else:
        plt.show()

    # --------------------------------------------------------------
    # Plot measurements
    plt.figure(figsize=(6, 8))
    plt.subplot(211)
    plt.stem(np.arange(img.shape[0])[img != 0], img[img != 0], basefmt="C7--", linefmt="C1-")
    plt.stem([0, Ngrid-1], [0, 0], markerfmt='white', basefmt='C7--')
    plt.plot(np.arange(Ngrid), background, c='#1f77b4')
    plt.title("Original image (with background)")
    plt.subplot(212)
    plt.stem(y, basefmt="C7--", linefmt="C7-", markerfmt='gx')
    # plt.scatter(np.arange(y.shape[0]), y, marker='+')
    # plt.hlines(0, 0, y.shape[0]-1, ls="--", color='#7f7f7f')
    plt.title("Noisy measurements")
    if save_plots:
        plt.savefig(os.path.join(figures_path, "measurements" + save_ext))
    plt.show()

    # --------------------------------------------------------------
    # Plot the sparse component
    Nrows, Ncols = len(l1fs), len(l2s)
    fig, axs = plt.subplots(Nrows, Ncols, sharex=True, sharey=True,
                            figsize=(4 * Ncols + 2 * (Ncols - 1), 4 * Nrows + 2 * (Nrows - 1)))
    i = 0
    for reco, ax in zip(composite, axs.flat):
        # print(reco)
        ax.stem(reco["x1"])
        # ax.set_title(rf"$\lambda_2 = {reco['lambda2'][0]:.2e}, \lambda_1 = {reco['lambda1'][0]:.2e}$")
        ax.set(ylabel=rf"$\lambda_1$ factor: {l1fs[i // len(l1fs)]:.2f}",
               xlabel=rf"$\lambda_2 = {reco['lambda2'][0]:.1f}$")
        # print(rf"$\lambda_1$ factor: {l1fs[i // len(l1fs)]:.2f}", rf"$\lambda_2 = {reco['lambda2'][0]:.1f}$")
        ax.label_outer()
        # ax.set_title(rf"$\lambda_1 = {reco['lambda1'][0]:.2e}$")
        i += 1
    fig.suptitle(f"Foreground ($r_{12}$: {r12:.1f}, SRF: {srf:d}, SNR: 10 dB)")
    # axs = axs.transpose()
    if save_plots:
        plt.savefig(os.path.join(figures_path, "foreground" + save_ext))
    else:
        fig.show()

    # --------------------------------------------------------------
    # Plot the background component
    fig, axs = plt.subplots(Nrows, Ncols, sharex=True, sharey=True,
                            figsize=(3 * Ncols + 2 * (Ncols - 1), 3 * Nrows + 2 * (Nrows - 1)))
    i = 0
    for reco, ax in zip(composite, axs.flat):
        ax.hlines(0, 0, 1, color='black', alpha=.5)
        ax.plot(np.arange(reco["x2"].shape[0])/Ngrid, reco["x2"])
        # ax.set_title(rf"$\lambda_2 = {reco['lambda2'][0]:.2e}, \lambda_1 = {reco['lambda1'][0]:.2e}$")
        ax.set(ylabel=rf"$\lambda_1$ factor: {l1fs[i // len(l1fs)]:.2f}",
               # xlabel=rf"$\lambda_1 = {reco['lambda1'][0]:.2e}  (f: {l1f[i%len(l1f)]:.2f})$",
               xlabel=rf"$\lambda_2 = {reco['lambda2'][0]:.1f}$")
        ax.label_outer()
        i += 1
    # fig.suptitle(f"Background (fgbgR: {fgbgR:.1f}, $r_{12}$: {r12:.1f}, SNR: {snr:.1f} dB)")
    if save_plots:
        plt.savefig(os.path.join(figures_path, "backgrounds" + save_ext))
    fig.show()

    # --------------------------------------------------------------
    # Sparse components convolved
    fig, axs = plt.subplots(Nrows, Ncols, sharex=True, sharey=True,
                            figsize=(3 * Ncols + 2 * (Ncols - 1), 3 * Nrows + 2 * (Nrows - 1)))
    i = 0
    for reco, ax in zip(composite, axs.flat):
        repr_reco = np.convolve(reco["x1"], representation_kernel, mode="same")
        ax.plot(np.arange(repr_reco.shape[0])/Ngrid, repr_reco, c='#ff7f0e', )  # marker='.', markersize=2)
        # ax.set_title(rf"$\lambda_2 = {reco['lambda2'][0]:.2e}, \lambda_1 = {reco['lambda1'][0]:.2e}$")
        ax.set(ylabel=rf"$\lambda_1$ factor: {l1fs[i // len(l1fs)]:.2f}",
               # xlabel=rf"$\lambda_1 = {reco['lambda1'][0]:.2e}    (f: {l1f[i%len(l1f)]:.2f})$",
               xlabel=rf"$\lambda_2 = {reco['lambda2'][0]:.2e}$")
        ax.label_outer()
        i += 1
    # fig.suptitle(f"Merged foreground (fgbgR: {fgbgR:.1f}, $r_{12}$: {r12:.1f}, SNR: {snr:.1f} dB)")
    if save_plots:
        plt.savefig(os.path.join(figures_path, "foreground_merged" + save_ext))
    fig.show()

    # # --------------------------------------------------------------
    # # Reconstruction with the BLASSO
    # Nrows = len(lfs)
    # fig, axs = plt.subplots(Nrows + 1, 1, sharex=True, sharey=True,
    #                         figsize=(6, 4 * Nrows + 2 * (Nrows - 1)))
    # axs[0].stem(gt["img"])
    # axs[0].set_title("Source image")
    # i = 0
    # for reco, ax in zip(blasso, axs.flat[1:]):
    #     ax.stem(reco["x"])
    #     ax.set_title(rf"$\lambda = {reco['lambda_'][0]:.2e}    (f: {lfs[i]:.2f})$")
    #     i += 1
    # fig.suptitle(f"BLASSO foreground ($r_{12}$: {r12:.1f}, SRF: {srf:d}, SNR: 10 dB)")
    # if save_plots:
    #     plt.savefig(os.path.join(figures_path, "blasso" + save_ext))
    # else:
    #     fig.show()
    #
    # # And convolved
    # fig, axs = plt.subplots(2, 2, sharex=True, sharey=True, figsize=(12, 9))
    # repr_source = np.convolve(gt["img"], representation_kernel, mode="same")
    # axs[0, 0].plot(np.arange(repr_source.shape[0])/Ngrid, repr_source, c='#ff7f0e', )
    # axs[0, 0].set_title("Source image")
    # i = 0
    # for reco, ax in zip(blasso[1:], axs.flat[1:]):
    #     repr_reco = np.convolve(reco["x"], representation_kernel, mode="same")
    #     ax.plot(locs, repr_reco, c='#ff7f0e', )
    #     ax.set_title(rf"$\lambda$ factor: {lfs[i+1]:.2f}")
    #     ax.set_ylim(top=1.05 * repr_source.max())
    #     # ax.set_title(rf"$\lambda = {reco['lambda_'][0]:.2e}    (f: {lf[i]:.2f})$")
    #     i += 1
    # # fig.suptitle(f"BLASSO convolved(fgbgR: {fgbgR:.1f}, $r_{12}$: {r12:.1f}, SNR: {snr:.1f} dB)")
    # if save_plots:
    #     plt.savefig(os.path.join(figures_path, "blasso_merged" + save_ext))
    # fig.show()
    #
    # plt.figure()
    # plt.plot(locs, np.convolve(blasso[-1]["x"], representation_kernel, mode="same"))
    # plt.ylim(top=1.05 * repr_source.max())
    # plt.show()

    # --------------------------------------------------------------
    # Metrics
    # --------------------------------------------------------------
    # blasso_conv = [ np.convolve(reco["x"], representation_kernel, mode="same") for reco in blasso ]
    # errors_blasso = np.array([relL2Error(reco, repr_source) for reco in blasso_conv])
    # errors_blasso1 = np.array([relL1Error(reco, repr_source) for reco in blasso_conv])

    composite_conv = [np.convolve(reco["x1"], representation_kernel, mode="same") for reco in composite]
    errors_composite = np.array([relL2Error(reco, repr_source) for reco in composite_conv]).reshape((len(l1fs), len(l2s)))
    errors_composite1 = np.array([relL1Error(reco, repr_source) for reco in composite_conv]).reshape((len(l1fs), len(l2s)))
    errors_bg = np.array([relL2Error(reco["x2"], background) for reco in composite]).reshape((len(l1fs), len(l2s)))
    errors_bg1 = np.array([relL1Error(reco["x2"], background) for reco in composite]).reshape((len(l1fs), len(l2s)))
    times_composite = np.array([reco["t"] for reco in composite]).reshape((len(l1fs), len(l2s)))

    print("Reconstruction times")
    print(times_composite)

    # print("BLASSO")
    # print(errors_blasso)
    # print(errors_blasso1)

    print("Composite")
    print(errors_composite)
    print(errors_composite1)
    import pandas as pd
    cols = [f"{u:.2f}" for u in l2s]
    df_rl2 = pd.DataFrame(errors_composite, index=[f"{u:.2f}" for u in l1fs], columns=cols)
    print(df_rl2.to_latex(index=True, float_format="{:.3f}".format,))

    df_rl1 = pd.DataFrame(errors_composite1, index=[f"{u:.2f}" for u in l1fs], columns=cols)
    print(df_rl1.to_latex(index=True, float_format="{:.3f}".format,))

    dfbg_rl2 = pd.DataFrame(errors_bg, index=[f"{u:.2f}" for u in l1fs], columns=cols)
    print(dfbg_rl2.to_latex(index=True, float_format="{:.3f}".format,))

    # dfbg_rl1 = pd.DataFrame(errors_bg1, index=[f"{u:.2f}" for u in l1fs], columns=cols)
    # print(dfbg_rl1.to_latex(index=True, float_format="{:.3f}".format,))


    # cols = [f"{u:.2f}" for u in lfs]
    # df_blasso = pd.DataFrame(errors_blasso.reshape((1, -1)), columns=cols, index=["L2 error"])
    # df_blasso.loc["L1 error"] = errors_blasso1
    # print(df_blasso.to_latex(index=True, float_format="{:.3f}".format, ))
