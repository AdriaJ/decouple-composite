import time
import numpy as np
from matplotlib import use
use("Qt5Agg")
import matplotlib.pyplot as plt
import pyxu.operator as pxop
import pyxu.opt.solver as pxls

import scipy.fft as sfft
import scipy.signal as sig

from pyxu.opt.stop import RelError, MaxIter
from pyxu.abc import QuadraticFunc


seed = 51_088
# seed = None
srf = 8
Nmeas = 100
Ngrid = int(srf) * Nmeas
k = 10
fgbgR = 10.
ongrid = True

# measurement model
kernel_std = 0.02
kernel_std_bg = 4 * kernel_std
snrdb_meas = 10
r12 = 1.  # rate between l2 norm of fg observations and bg observations

# reconstruction parameters
lambda1_factor = 0.1
lambda2 = 1e-2 * Nmeas * srf  # Ngrid
eps = 1e-5
kernel_std_target = 0.1

blasso_factor = 0.35

srf_repr = 4

article_plots = False
save_pdf = False

# decoupled = True
do_non_decoupled = False
do_blasso = False

if __name__ == "__main__":
    if seed is None:
        seed = np.random.randint(1000)
    rng = np.random.default_rng(seed=seed)

    if save_pdf:
        import os
        figures_path = "figures/rkhs"

    # define the grid-based Gaussian kernels that I will need
    kernel_std_int = np.floor(kernel_std * Ngrid).astype(int)
    kernel_width = 3 * 2 * kernel_std_int + 1  # Length of the Gaussian kernel
    kernel_measurement = np.exp(
        -0.5 * ((np.arange(kernel_width) - (kernel_width - 1) / 2) ** 2) / ((kernel_std * Ngrid) ** 2))
    norm_meas = (np.sqrt(2 * np.pi) * kernel_std)
    kernel_measurement /= norm_meas

    kernel_std_bg_int = np.floor(kernel_std_bg * Ngrid).astype(int)
    kernel_width_bg = 3 * 2 * kernel_std_bg_int + 1
    kernel_bg_1d = np.exp(-0.5 * ((np.arange(kernel_width_bg) - (kernel_width_bg - 1) / 2) ** 2) / ((kernel_std_bg*Ngrid) ** 2))
    norm_bg1d = (np.sqrt(2 * np.pi) * kernel_std_bg)
    kernel_bg_1d /= norm_bg1d

    std_meas_bg2 = kernel_std**2 + kernel_std_bg**2
    std_meas_bg2_int = np.floor(np.sqrt(std_meas_bg2) * Ngrid).astype(int)
    width_meas_bg = 3 * 2 * std_meas_bg2_int + 1
    kernel_meas_bg = (kernel_std * kernel_std_bg * np.sqrt(2 * np.pi / std_meas_bg2) *
                      np.exp(-0.5 * ((np.arange(width_meas_bg) - (width_meas_bg - 1) / 2) ** 2) / (std_meas_bg2 * (Ngrid ** 2))))
    kernel_meas_bg /= (norm_meas * norm_bg1d)
    # Same formula:
    # kernel_meas_bg = np.exp(-0.5 * ((np.arange(width_meas_bg) - (width_meas_bg - 1) / 2) ** 2) / (std_meas_bg2*Ngrid**2))
    # kernel_meas_bg /= (np.sqrt(2 * np.pi) * np.sqrt(std_meas_bg2))

    if ongrid:
        img = np.zeros((Ngrid,))
        Neff = int(.5 * Ngrid)
        # foreground
        idx = rng.choice(Neff, k, replace=False)
        indices = idx + int(.25 * Ngrid)
        img[indices] = rng.uniform(1, fgbgR, k)
        bg_impulses = np.zeros((Ngrid,))
        # background
        kk = 2*k
        idx = rng.choice(Neff, kk, replace=False)
        indices = idx + int(.25 * Ngrid)
        bg_impulses[indices] = 1 + rng.uniform(-.5, .5, kk)
        background = sig.fftconvolve(bg_impulses, kernel_bg_1d, mode='same')

    # Continuous-time convolution and evaluate on the coarse grid
    conv_fg = np.convolve(np.pad(img, (kernel_width//2, kernel_width//2), mode='wrap'), # constant
                          kernel_measurement, mode='valid')
    meas_fg = conv_fg[srf // 2::srf]

    conv_bg = np.convolve(np.pad(bg_impulses, (width_meas_bg//2, width_meas_bg//2), mode='wrap'), # constant
                          kernel_meas_bg, mode='valid')
    meas_bg = conv_bg[srf // 2::srf]

    if r12:
        factor = r12 * np.linalg.norm(meas_bg) / np.linalg.norm(meas_fg)
        img *= factor
        meas_fg *= factor

    x = img + background
    noiseless_y = meas_fg + meas_bg

    # PSNR : 10 * np.log10(max(img)**2 / np.std(noise)**2) = 20 * log10(max(img) / std(noise))
    sigma_noise = np.linalg.norm(noiseless_y)/Nmeas * 10**(-snrdb_meas / 20)
    noise_meas = rng.normal(0, sigma_noise, noiseless_y.shape)
    y = noiseless_y + noise_meas


# -------------------------
# Reconstruction
#--------------------------
    if kernel_std_target is None:
        kernel_std_target = kernel_std_bg
        kernel_target = kernel_bg_1d
    else:
        kernel_std_target_int = np.floor(kernel_std_target * Ngrid).astype(int)
        kernel_width_target = 3 * 2 * kernel_std_target_int + 1
        kernel_target = np.exp(
            -0.5 * ((np.arange(kernel_width_target) - (kernel_width_target - 1) / 2) ** 2) / ((kernel_std_target * Ngrid) ** 2))
        norm_target = (np.sqrt(2 * np.pi) * kernel_std_target)
        kernel_target /= norm_target

    diff_std2 = kernel_std**2 + kernel_std_target**2
    norm_regul = np.sqrt(2 * np.pi * diff_std2)
    diffs = np.arange(0, 4 * np.sqrt(diff_std2) * Nmeas)
    diffs = np.hstack([-diffs[1:][::-1], diffs])
    kernel_regul = np.exp(-0.5 * ((diffs / Nmeas) ** 2) / diff_std2)
    kernel_regul /= norm_regul
    M_kernel = kernel_regul/lambda2
    M_kernel[M_kernel.shape[0]//2] += 1

    regul_width = kernel_regul.shape[0]
    h = np.zeros(Nmeas)
    h[:regul_width] = M_kernel
    h = np.roll(h, -regul_width//2 + 1)
    hm1 = sfft.irfft(1/sfft.rfft(h))

    MlambdaInv = pxop.Convolve(arg_shape=Nmeas, kernel=[hm1,], center=[0,], mode="wrap")
    MlambdaInv.lipschitz = np.abs(hm1).sum()

    fOp = pxop.Convolve(arg_shape=img.shape, kernel=[kernel_measurement,], center=[kernel_width // 2,], mode="wrap")
    fOp.lipschitz = fOp.estimate_lipschitz(method='svd', tol=1e-3)
    ss = pxop.SubSample(Ngrid, slice(srf // 2, Ngrid, srf))
    Hop = ss * fOp

    lambda1max = np.abs(Hop.adjoint(MlambdaInv(y).ravel())).max()
    lambda1 = lambda1_factor * lambda1max

    loss = QuadraticFunc((1, Nmeas), Q=MlambdaInv).asloss(y.ravel()) * Hop
    loss.diff_lipschitz = loss.estimate_diff_lipschitz(method='svd') # could find an upper bound to gain time on the computation

    regul = lambda1 * pxop.PositiveL1Norm(Ngrid)

    stop_crit = RelError(eps=eps, var="x", f=None, norm=2, satisfy_all=True,) & MaxIter(10)

    print("Decoupled solving...")
    pgd = pxls.PGD(loss, g=regul, show_progress=False)
    start = time.time()
    pgd.fit(x0=np.zeros(img.size), stop_crit=stop_crit)
    pgd_time = time.time() - start

    _, hist = pgd.stats()
    x1 = pgd.solution()

    Mresiduals = MlambdaInv(y - Hop(x1))
    tmp = np.zeros(Ngrid)
    tmp[srf // 2::srf] = Mresiduals
    x2 = np.convolve(tmp, kernel_target, mode='same') / lambda2

    if do_non_decoupled:
        print("Non-decoupled composite reconstruction...")
        Top = pxop.Convolve(arg_shape=Nmeas, kernel=[kernel_regul,], center=[kernel_regul.shape[0]//2,], mode="constant")
        ndcp_loss = .5 * pxop.SquaredL2Norm(Nmeas).asloss(y.ravel()) * pxop.hstack([Hop, Top]) + \
            lambda2 * pxop.hstack([pxop.NullFunc(Ngrid), QuadraticFunc((1, Nmeas), Q=Top)])
        start = time.time()
        ndcp_loss.diff_lipschitz = ndcp_loss.estimate_diff_lipschitz(method='svd')
        ndcp_lipschitz_time = time.time() - start
        print(f"Diff Lipschitz constant estimation time (NDCP): {ndcp_lipschitz_time:.2f}s")
        ndcp_regul = lambda1 * pxop.hstack([pxop.PositiveL1Norm(Ngrid), pxop.NullFunc(Nmeas)])
        ndcp_stop = RelError(eps=eps, var="x", f= lambda v: v[:Ngrid], norm=2, satisfy_all=True,) & MaxIter(10)
        ndcp_pgd = pxls.PGD(ndcp_loss, g=ndcp_regul, show_progress=False)
        start = time.time()
        ndcp_pgd.fit(x0=np.zeros(Ngrid + Nmeas), stop_crit=ndcp_stop)
        ndcp_time = time.time() - start

        ndcp_sol, ndcp_hst = ndcp_pgd.stats()

        x1_ndcp = ndcp_sol['x'][:Ngrid]
        x2_innovations_ndcp = ndcp_sol['x'][Ngrid:]
        x2_ndcp = np.zeros(Ngrid)
        x2_ndcp[srf // 2::srf] = x2_innovations_ndcp
        x2_ndcp = np.convolve(np.pad(x2_ndcp, (kernel_width//2, kernel_width//2), mode='wrap'),
                              kernel_target, mode='valid')
        x2_ndcp = np.convolve(x2_ndcp, kernel_target, mode='same')
        # print(np.allclose(x1_ndcp, 0))  # make sure the solution is non null

    # -------------------------
    # Analysis and plotting
    # --------------------------

    plt.figure(figsize=(12, 11))
    plt.suptitle(rf"$\lambda_1$ factor : {lambda1:.2e}, $\lambda_2$ : {lambda2:.2e}")
    ylim = max(img.max(), x1.max())
    plt.subplot(321)
    plt.ylim(top=1.05 * ylim)
    plt.stem(np.arange(img.shape[0])[img != 0]/Ngrid, img[img != 0])
    plt.stem([0, 1 - 1/Ngrid], [0, 0], markerfmt='white')
    plt.title("Source foreground")
    plt.subplot(322)
    plt.ylim(top=1.05 * ylim)
    plt.stem(np.arange(x1.shape[0])[x1 != 0]/Ngrid, x1[x1 != 0])
    plt.stem([0, 1 - 1/Ngrid], [0, 0], markerfmt='white')
    plt.title("Recovered foreground")

    ylim = max(background.max(), x2.max())
    plt.subplot(323)
    plt.ylim(top=1.05 * ylim)
    plt.plot(np.arange(Ngrid)/Ngrid, background, c='orange',)  # marker='.')
    plt.title("Source background")
    plt.subplot(324)
    plt.ylim(top=1.05 * ylim)
    plt.plot(np.arange(Ngrid)/Ngrid, x2, c='orange',)  # marker='.')
    plt.title("Recovered background")

    # measurement fidelity
    measx1 = np.convolve(np.pad(x1, (kernel_width//2, kernel_width//2), mode='constant'), kernel_measurement, mode='valid')
    measx2 = np.convolve(np.pad(x2, (kernel_width//2, kernel_width//2), mode='constant'), kernel_measurement, mode='valid')/Ngrid
    sol_meas = (measx1 + measx2)[srf // 2::srf]
    ylim = max(y.max(), sol_meas.max())
    plt.subplot(325)
    plt.ylim(top=1.05 * ylim)
    plt.stem(y, basefmt="C7--", linefmt="C7-", markerfmt='gx')
    plt.title("Measurements")
    plt.subplot(326)
    plt.ylim(top=1.05 * ylim)
    plt.stem(sol_meas, basefmt="C7--", linefmt="C7-", markerfmt='gx')
    plt.title("Measurements on the solution")
    plt.show()

    # Representation kernel has the same finesse as the gridded image, so very fine
    repr_std = kernel_std / srf_repr
    repr_std_int = np.floor(repr_std * Ngrid).astype(int)
    repr_width = 3 * 2 * repr_std_int + 1  # Length of the Gaussian kernel
    representation_kernel = np.exp(
        -0.5 * ((np.arange(repr_width) - (repr_width - 1) / 2) ** 2) / ((repr_std * Ngrid) ** 2))
    norm_repr = (np.sqrt(2 * np.pi) * repr_std)
    representation_kernel /= norm_repr

    fig = plt.figure(figsize=(12, 6))
    plt.suptitle("Foreground representation: convolution with a narrow Gaussian kernel")
    repr_source = np.convolve(img, representation_kernel, mode='same')
    repr_recovered = np.convolve(x1, representation_kernel, mode='same')
    ylim = max(repr_source.max(), repr_recovered.max())
    axes = fig.subplots(1, 2, sharex=True)
    ax = axes.ravel()[0]
    ax.set_ylim(top=1.05 * ylim)
    ax.plot(np.arange(Ngrid), repr_source, c='orange', marker='.')
    ax.set_title("Source foreground")
    ax = axes.ravel()[1]
    ax.set_ylim(top=1.05 * ylim)
    ax.plot(np.arange(Ngrid), repr_recovered, c='orange', marker='.')
    ax.set_title("Recovered foreground")
    plt.show()

    if do_non_decoupled:
        plt.figure(figsize=(6, 11))
        plt.subplot(211)
        ylim = max(img.max(), x1.max())
        plt.ylim(top=1.05 * ylim)
        plt.stem(np.arange(x1_ndcp.shape[0])[x1_ndcp != 0], x1_ndcp[x1_ndcp != 0])
        plt.stem([0, Ngrid-1], [0, 0], markerfmt='white')
        plt.title("Recovered foreground (non-decoupled)")
        plt.subplot(212)
        ylim = max(background.max(), x2.max())
        plt.ylim(top=1.05 * ylim)
        plt.plot(np.arange(Ngrid), x2_ndcp, c='orange',)  # marker='.')
        plt.title("Recovered background (non-decoupled)")
        plt.show()

    print(f"Reconstruction times:")
    print(f"\tDecoupled: {pgd_time:.2f}s")
    print(f"Relative L2 error on the foreground:")
    print(f"\tComposite: {np.linalg.norm(repr_recovered - repr_source)/np.linalg.norm(repr_source):.2f}")
    print(f"Relative L1 error on the foreground:")
    print(f"\tComposite: {np.linalg.norm(repr_recovered - repr_source, ord=1)/np.linalg.norm(repr_source, ord=1):.2f}")

    l1_value = lambda1 * np.abs(x1).sum()
    print(f"Value of the foreground regularization at convergence: {l1_value:.3e}")
    # l2_value = (np.convolve(x2, kernel_regul, mode='same')**2).sum()
    l2_value = np.linalg.norm(x2)**2 / Ngrid
    print(f"Approximate value of the background regularization at convergence: {lambda2 * l2_value:.3e}")
    data_fid_val = 0.5 * np.linalg.norm(y - sol_meas)**2
    print(f"Approximate value of the data fidelity at convergence: {data_fid_val:.3e}")


    if do_non_decoupled:
        repr_ndcp = np.convolve(x1_ndcp, representation_kernel, mode='same')
        print("Time:")
        print(f"\tNon-decoupled: {ndcp_time:.2f}s")
        print(f"Relative L2 error on the foreground:")
        print(f"\tNon-decoupled: {np.linalg.norm(repr_ndcp - repr_source) / np.linalg.norm(repr_source):.2f}")


    if article_plots:
        # Simulated source: fg, bg, sum
        # Measurements: fg, bg, sum
        # Recovered: fg, bg
        # Convolution: source, reconstruction

        # Plots needed: measurements (with contribution of each comp), simple reco, fg reco (convolved)
        locs = np.arange(Ngrid) / Ngrid

        plt.figure(figsize=(15, 4))
        plt.subplot(131)
        plt.stem(locs[img != 0], img[img != 0], basefmt="C7--", linefmt="C1-")
        plt.stem([0, (Ngrid - 1)/Ngrid], [0, 0], markerfmt='white', basefmt='C7--')
        # plt.title("Original image (without background)")
        plt.subplot(132)
        plt.plot(locs, background, c='#1f77b4')
        # plt.title("Original background)")
        plt.hlines(0, 0, (Ngrid - 1)/Ngrid, ls="--", color='#7f7f7f')
        plt.subplot(133)
        plt.stem(locs[img != 0], img[img != 0], basefmt="C7--", linefmt="C1-")
        plt.stem([0, (Ngrid - 1)/Ngrid], [0, 0], markerfmt='white', basefmt='C7--')
        plt.plot(locs, background, c='#1f77b4')
        # plt.title("Original image (with background)")
        if save_pdf:
            plt.savefig(os.path.join(figures_path, "gt.pdf"))
        plt.show()

        # Sum of the measurements
        yrange = [min(3*y.min(), y.min()-0.05, -.05), 1.05*y.max()]
        plt.figure(figsize=(15, 4))
        plt.subplot(131)
        plt.stem(np.arange(Nmeas)/Nmeas,meas_fg, basefmt="C7--", linefmt="C7-", markerfmt='gx')
        # plt.title("Observations on the foreground")
        plt.ylim(yrange)
        plt.subplot(132)
        plt.stem(np.arange(Nmeas)/Nmeas, meas_bg, basefmt="C7--", linefmt="C7-", markerfmt='gx')
        # plt.title("Observations on the background")
        plt.ylim(yrange)
        plt.subplot(133)
        plt.stem(np.arange(Nmeas)/Nmeas, y, basefmt="C7--", linefmt="C7-", markerfmt='gx')
        # plt.title("Noisy measurements")
        plt.ylim(yrange)
        fig = plt.gcf()
        for ax in fig.axes:
            ax.label_outer()
        if save_pdf:
            plt.savefig(os.path.join(figures_path, "measurements.pdf"))
        plt.show()

        # Best reconstruction
        plt.figure(figsize=(12, 4))
        plt.subplot(121)
        plt.stem(locs[x1 != 0], x1[x1 != 0], basefmt="C7--", linefmt="C1-")
        plt.stem([0, 1], [0, 0], markerfmt='white', basefmt='C7--')
        # plt.title("Recovered foreground (composite model)")
        plt.subplot(122)
        plt.plot(locs, x2, c='#1f77b4')
        # plt.title("Recovered background (composite model)")
        plt.hlines(0, 0, 1, ls="--", color='#7f7f7f')
        if save_pdf:
            plt.savefig(os.path.join(figures_path, "recos.pdf"))
        plt.show()

        # repr_best_reco = np.convolve(best_reco["x1"], representation_kernel, mode="same")
        # repr_source = np.convolve(data["img"], representation_kernel, mode="same")

        # Reconstruction after convolution
        ymax = 1.05 * max(repr_source.max(), repr_recovered.max())
        plt.figure(figsize=(12, 4))
        plt.subplot(121)
        plt.plot(locs, repr_source, c='#ff7f0e', )  # marker='.')
        plt.ylim(top=ymax)
        # plt.title("Source convolved")
        plt.subplot(122)
        plt.plot(locs, repr_recovered, c='#ff7f0e', )  # marker='.')
        # plt.title("Foreground recovered convolved")
        plt.ylim(top=ymax)
        for ax in plt.gcf().axes:
            ax.label_outer()
        if save_pdf:
            plt.savefig(os.path.join(figures_path, "recos_conv.pdf"))
        plt.show()
