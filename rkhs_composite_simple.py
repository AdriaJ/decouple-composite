import time
import numpy as np
# from matplotlib import use
# use("Qt5Agg")
import matplotlib.pyplot as plt
import pyxu.operator as pxop
import pyxu.opt.solver as pxls

import scipy.fft as sfft
import scipy.signal as sig
from matplotlib.pyplot import tight_layout

from pyxu.opt.stop import RelError, MaxIter
from pyxu.abc import QuadraticFunc

from utils import relL2Error

import matplotlib as mpl

mpl.rcParams.update({
    "text.usetex": False,
    "font.family": "sans-serif",
    "font.size": 10,  # match \documentclass[12pt]
    "axes.labelsize": 10,
    "axes.titlesize": 10,
    "legend.fontsize": 9,
    "xtick.labelsize": 8,  # typical hierarchy (slightly smaller)
    "ytick.labelsize": 8,
})


def set_figsize(fraction=1.0, aspect=0.6):
    width_pt = 489.10307
    w = width_pt / 72.27 * fraction
    return (w, w * aspect)


from mpl_toolkits.axes_grid1.inset_locator import mark_inset, inset_axes

seed = 476_613  # 351_930  # 476_613
# seed = None
srf = 8  # 8  # 16
Nmeas = 100
Ngrid = int(srf) * Nmeas
k = 2
fgbgR = 10.
ongrid = True
presence_rate = .5  # .5  # Change percentage of presence

# measurement model
kernel_std = 0.02
kernel_std_bg = 4 * kernel_std #4 * kernel_std
snrdb_meas = 20 # 10
r12 = 1.  # rate between l2 norm of fg observations and bg observations
rep_sigma = 4

# reconstruction parameters
lambda1_factor = 0.02
lambda2 = Nmeas * srf * 5e-4 # 4
eps = 1e-6
kernel_std_target = 0.08  # 0.1

blasso_factor = 0.35

srf_repr = 4

article_plots = True
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
        # figures_path = "figures/rkhs"
        figures_path = "figures/review3"

    # define the grid-based Gaussian kernels that I will need
    kernel_std_int = np.floor(kernel_std * Ngrid).astype(int)
    kernel_width = rep_sigma * 2 * kernel_std_int + 1  # Length of the Gaussian kernel
    kernel_measurement = np.exp(
        -0.5 * ((np.arange(kernel_width) - (kernel_width - 1) / 2) ** 2) / ((kernel_std * Ngrid) ** 2))
    norm_meas = (np.sqrt(2 * np.pi) * kernel_std)
    kernel_measurement /= norm_meas

    kernel_std_bg_int = np.floor(kernel_std_bg * Ngrid).astype(int)
    kernel_width_bg = rep_sigma * 2 * kernel_std_bg_int + 1
    kernel_bg_1d = np.exp(-0.5 * ((np.arange(kernel_width_bg) - (kernel_width_bg - 1) / 2) ** 2) / ((kernel_std_bg*Ngrid) ** 2))
    norm_bg1d = (np.sqrt(2 * np.pi) * kernel_std_bg)
    kernel_bg_1d /= norm_bg1d

    std_meas_bg2 = kernel_std**2 + kernel_std_bg**2
    std_meas_bg2_int = np.floor(np.sqrt(std_meas_bg2) * Ngrid).astype(int)
    width_meas_bg = rep_sigma * 2 * std_meas_bg2_int + 1
    kernel_meas_bg = (kernel_std * kernel_std_bg * np.sqrt(2 * np.pi / std_meas_bg2) *
                      np.exp(-0.5 * ((np.arange(width_meas_bg) - (width_meas_bg - 1) / 2) ** 2) / (std_meas_bg2 * (Ngrid ** 2))))
    kernel_meas_bg /= (norm_meas * norm_bg1d)
    # Same formula:
    # kernel_meas_bg = np.exp(-0.5 * ((np.arange(width_meas_bg) - (width_meas_bg - 1) / 2) ** 2) / (std_meas_bg2*Ngrid**2))
    # kernel_meas_bg /= (np.sqrt(2 * np.pi) * np.sqrt(std_meas_bg2))

    if ongrid:
        img = np.zeros((Ngrid,))
        Neff = int(presence_rate * Ngrid)
        # foreground
        idx = rng.choice(Neff, k, replace=False)
        indices = idx + int((1-presence_rate)/2 * Ngrid)
        img[indices] = rng.uniform(1, fgbgR, k)
        bg_impulses = np.zeros((Ngrid,))
        # background
        kk = 10*k
        idx = rng.choice(Neff, kk, replace=False)
        indices = idx + int((1-presence_rate)/2 * Ngrid)
        bg_impulses[indices] = 1 + rng.uniform(-.5, .5, kk)
        background = sig.fftconvolve(bg_impulses, kernel_bg_1d, mode='same')

    # Continuous-time convolution and evaluate on the coarse grid
    conv_fg = np.convolve(np.pad(img, (kernel_width//2, kernel_width//2), mode='constant'), # constant
                          kernel_measurement, mode='valid')
    meas_fg = conv_fg[srf // 2::srf]

    conv_bg = np.convolve(np.pad(bg_impulses, (width_meas_bg//2, width_meas_bg//2), mode='constant'), # constant
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
    # y = np.clip(y, a_min=0, a_max=None)


# -------------------------
# Reconstruction
#--------------------------
    if kernel_std_target is None:
        kernel_std_target = kernel_std_bg
        kernel_target = kernel_bg_1d
    else:
        kernel_std_target_int = np.floor(kernel_std_target * Ngrid).astype(int)
        kernel_width_target = rep_sigma * 2 * kernel_std_target_int + 1
        kernel_target = np.exp(
            -0.5 * ((np.arange(kernel_width_target) - (kernel_width_target - 1) / 2) ** 2) / ((kernel_std_target * Ngrid) ** 2))
        norm_target = (np.sqrt(2 * np.pi) * kernel_std_target)
        kernel_target /= norm_target

    diff_std2 = kernel_std**2 + kernel_std_target**2
    norm_regul = np.sqrt(2 * np.pi * diff_std2)
    diffs = np.arange(0, 5 * np.sqrt(diff_std2) * Nmeas)  # 4 *
    diffs = np.hstack([-diffs[1:][::-1], diffs])
    kernel_regul = np.exp(-0.5 * ((diffs / Nmeas) ** 2) / diff_std2)
    kernel_regul /= norm_regul
    M_kernel = kernel_regul/lambda2
    M_kernel[M_kernel.shape[0]//2] += 1

    # Mlambda_mat = np.zeros((Nmeas, Nmeas + M_kernel.shape[0]))
    # for i in range(Nmeas):
    #     Mlambda_mat[i, i:M_kernel.shape[0]+i] = M_kernel
    # Mlambda_mat = Mlambda_mat[:, M_kernel.shape[0]//2:-M_kernel.shape[0]//2]
    from scipy.sparse import diags
    K = len(M_kernel)
    offsets = np.arange(-(K // 2), K - K // 2)
    Mlambda_mat = diags(M_kernel, offsets, shape=(Nmeas, Nmeas)).toarray()
    Mlambdam1_mat = np.linalg.inv(Mlambda_mat)
    import pyxu.abc as pxa
    MlambdaInv = pxa.LinOp.from_array(Mlambdam1_mat)
    MlambdaInv.lipschitz = MlambdaInv.estimate_lipschitz()

    # plt.figure()
    # plt.imshow(Mlambda_mat)
    # plt.show()
    # regul_width = kernel_regul.shape[0]
    # h = np.zeros(Nmeas)
    # h[:regul_width] = M_kernel
    # h = np.roll(h, -regul_width//2 + 1)
    # hm1 = sfft.irfft(1/sfft.rfft(h))
    # MlambdaInv = pxop.Convolve(arg_shape=Nmeas, kernel=[hm1,], center=[0,], mode="wrap")
    # MlambdaInv.lipschitz = np.abs(hm1).sum()

    # # test the correct computation of Mlambda Inv:
    # Mlambda = pxop.Convolve(arg_shape=Nmeas, kernel=[M_kernel,], center=[M_kernel.shape[0]//2,], mode="wrap")
    # for _ in range(10):
    #     tmp = np.random.normal(loc=0, scale=1, size=Nmeas)
    #     limit = int(presence_rate*Nmeas)
    #     tmp[:limit] = 0
    #     tmp[-limit:] = 0
    #     print(np.allclose(Mlambda(MlambdaInv(tmp)), tmp))
    #     print(np.allclose(MlambdaInv(Mlambda(tmp)), tmp))

    # Mlambda = np.zeros((Nmeas, Nmeas + 2 * (kernel_regul.shape[0]//2)))
    # for i in range(Nmeas):
    #     Mlambda[i][i:i+kernel_regul.shape[0]] = kernel_regul
    # Mlambda = Mlambda[:, kernel_regul.shape[0]//2:-kernel_regul.shape[0]//2+1]
    # print(np.linalg.cond(Mlambda))
    # start = time.time()
    # matInv = np.linalg.inv(Mlambda)
    # inversion_time = time.time() - start
    # print(f"Inversion time of M lambda matrix: {inversion_time:.2f} s", )
    # ## Compare results with MlambdaInv operator
    # for test_vect in np.eye(5, Nmeas):
    #     print(np.allclose(MlambdaInv(test_vect), matInv@test_vect))
    #     print(np.linalg.norm(MlambdaInv(test_vect) - matInv@test_vect))

    fOp = pxop.Convolve(arg_shape=img.shape, kernel=[kernel_measurement,], center=[kernel_width // 2,], mode="constant")
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

    resid = y - Hop(x1)
    # Mresiduals = MlambdaInv(resid)
    Mresiduals = MlambdaInv(resid)

    # plt.figure()
    # plt.subplot(121)
    # plt.stem(resid)
    # plt.subplot(122)
    # plt.stem(Mresiduals)
    # plt.show()

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
        # x2_ndcp = np.convolve(np.pad(x2_ndcp, (kernel_width_target//2, kernel_width_target//2), mode='constant'),
        #                       kernel_target, mode='valid')
        x2_ndcp = np.convolve(x2_ndcp, kernel_target, mode='same')
        # print(np.allclose(x1_ndcp, 0))  # make sure the solution is non null

    if do_blasso:
        print("BLASSO reconstruction...")
        lambda_max = np.abs(Hop.adjoint(y).ravel()).max()
        lambda_b = blasso_factor * lambda_max
        blasso_regul = lambda_b * pxop.PositiveL1Norm(Ngrid)
        blasso_loss = pxop.SquaredL2Norm(Nmeas).asloss(y) * Hop
        start = time.time()
        blasso_loss.diff_lipschitz = blasso_loss.estimate_diff_lipschitz(method='svd')
        blasso_lipschitz_time = time.time() - start
        print(f"\tLipschitz constant estimation time (BLASSO): {blasso_lipschitz_time:.2f}s")
        pgd = pxls.PGD(blasso_loss, g=blasso_regul, show_progress=False)
        start = time.time()
        pgd.fit(x0=np.zeros(Ngrid), stop_crit=stop_crit)
        blasso_time = time.time() - start

        x_blasso = pgd.solution()

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

    repr_source = np.convolve(img, representation_kernel, mode='same')
    repr_recovered = np.convolve(x1, representation_kernel, mode='same')
    # fig = plt.figure(figsize=(12, 6))
    # plt.suptitle("Foreground representation: convolution with a narrow Gaussian kernel")
    # ylim = max(repr_source.max(), repr_recovered.max())
    # axes = fig.subplots(1, 2, sharex=True)
    # ax = axes.ravel()[0]
    # ax.set_ylim(top=1.05 * ylim)
    # ax.plot(np.arange(Ngrid), repr_source, c='orange', marker='.')
    # ax.set_title("Source foreground")
    # ax = axes.ravel()[1]
    # ax.set_ylim(top=1.05 * ylim)
    # ax.plot(np.arange(Ngrid), repr_recovered, c='orange', marker='.')
    # ax.set_title("Recovered foreground")
    # plt.show()

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

    if do_blasso:
        plt.figure(figsize=(6, 6))
        plt.subplot(211)
        plt.stem(x_blasso)
        plt.title("Reconstruction BLASSO")
        plt.subplot(212)
        repr_blasso = np.convolve(x_blasso, representation_kernel, mode='same')
        plt.plot(np.arange(Ngrid), repr_blasso, c='orange', marker='.')
        plt.title("Recovered BLASSO")
        plt.show()

    print(f"Reconstruction times:")
    print(f"\tDecoupled: {pgd_time:.2f}s")
    print(f"Relative L2 error on the foreground:")
    print(f"\tComposite: {np.linalg.norm(repr_recovered - repr_source)/np.linalg.norm(repr_source):.2f}")
    print(f"Relative L1 error on the foreground:")
    print(f"\tComposite: {np.linalg.norm(repr_recovered - repr_source, ord=1)/np.linalg.norm(repr_source, ord=1):.2f}")

    if do_blasso:
        print("Time:")
        print(f"\tBLASSO: {blasso_time:.2f}s")
        print(f"Relative L2 error on the foreground:")
        print(f"\tBLASSO: {np.linalg.norm(repr_blasso - repr_source)/np.linalg.norm(repr_source):.2f}")
        print(f"Relative L1 error on the fo0.0reground:")
        print(f"\tBLASSO: {np.linalg.norm(repr_blasso - repr_source, ord=1)/np.linalg.norm(repr_source, ord=1):.2f}")

    # l1_value = lambda1 * np.abs(x1).sum()
    # print(f"Value of the foreground regularization at convergence: {l1_value:.3e}")
    # # l2_value = (np.convolve(x2, kernel_regul, mode='same')**2).sum()
    # l2_value = np.linalg.norm(x2)**2 / Ngrid
    # print(f"Approximate value of the background regularization at convergence: {lambda2 * l2_value:.3e}")
    # data_fid_val = 0.5 * np.linalg.norm(y - sol_meas)**2
    # print(f"Approximate value of the data fidelity at convergence: {data_fid_val:.3e}")


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

        fig, axs = plt.subplots(1, 3, figsize=set_figsize(aspect=0.28), sharey=True)
        axs[0].stem(locs[img != 0], img[img != 0], basefmt="white", linefmt="C1-")
        axs[0].stem([0, (Ngrid - 1)/Ngrid], [0, 0], markerfmt='white', basefmt='C7--')
        # plt.title("Original image (without background)")
        axs[1].plot(locs, background, c='#1f77b4')
        axs[1].hlines(0, 0, (Ngrid - 1)/Ngrid, ls="--", color='#7f7f7f')
        # plt.title("Original background)")
        axs[2].stem(locs[img != 0], img[img != 0], basefmt="white", linefmt="C1-")
        axs[2].stem([0, (Ngrid - 1)/Ngrid], [0, 0], markerfmt='white', basefmt='C7--')
        axs[2].plot(locs, background, c='#1f77b4')
        # plt.title("Original image (with background)")
        if save_pdf:
            plt.savefig(os.path.join(figures_path, "gt.pdf"))
        plt.show()

        # # Sum of the measurements  # OLD
        # yrange = [min(3*y.min(), y.min()-0.05, -.05), 1.05*y.max()]
        # plt.figure(figsize=(15, 4))
        # plt.subplot(131)
        # plt.stem(np.arange(Nmeas)/Nmeas,meas_fg, basefmt="C7--", linefmt="C7-", markerfmt='gx')
        # # plt.title("Observations on the foreground")
        # plt.ylim(yrange)
        # plt.subplot(132)
        # plt.stem(np.arange(Nmeas)/Nmeas, meas_bg, basefmt="C7--", linefmt="C7-", markerfmt='gx')
        # # plt.title("Observations on the background")
        # plt.ylim(yrange)
        # plt.subplot(133)
        # plt.stem(np.arange(Nmeas)/Nmeas, y, basefmt="C7--", linefmt="C7-", markerfmt='gx')
        # # plt.title("Noisy measurements")
        # plt.ylim(yrange)
        # fig = plt.gcf()
        # for ax in fig.axes:
        #     ax.label_outer()
        # if save_pdf:
        #     plt.savefig(os.path.join(figures_path, "measurements.pdf"))
        # plt.show()

        yrange = [min(3*y.min(), y.min()-0.05, -.05), 1.05*y.max()]
        plotstyle = {'color':'green', 'marker':'.', 's':10}
        plt.figure(figsize=set_figsize(aspect=0.28))
        plt.subplot(131)
        plt.scatter(np.arange(Nmeas)/Nmeas, meas_fg, **plotstyle)
        plt.ylim(yrange)
        plt.subplot(132)
        plt.scatter(np.arange(Nmeas)/Nmeas, meas_bg, **plotstyle)
        plt.ylim(yrange)
        plt.subplot(133)
        plt.scatter(np.arange(Nmeas)/Nmeas, y, **plotstyle)
        # plt.title("Noisy measurements")
        plt.ylim(yrange)
        fig = plt.gcf()
        for ax in fig.axes:
            ax.label_outer()
        if save_pdf:
            plt.savefig(os.path.join(figures_path, "measurements.pdf"))
        plt.show()

        # plt.figure(figsize=(5, 4))
        # yrange = [min(3*y.min(), y.min()-0.05, -.05), 1.05*y.max()]
        # plt.scatter(np.arange(Nmeas)/Nmeas, meas_fg, color="#ff7f0e", marker='+', s=50)
        # plt.scatter(np.arange(Nmeas)/Nmeas, meas_bg, color='#1f77b4', marker='+', s=50)
        # plt.scatter(np.arange(Nmeas)/Nmeas, y, color='g', marker='+', s=50)
        # plt.ylim(yrange)
        # plt.show()


        # Best reconstruction
        # plt.figure(figsize=(12, 4))
        # plt.subplot(121)
        # plt.stem(locs[x1 != 0], x1[x1 != 0], basefmt="C7--", linefmt="C1-")
        # plt.stem([0, 1], [0, 0], markerfmt='white', basefmt='C7--')
        # # plt.title("Recovered foreground (composite model)")
        # plt.subplot(122)
        # plt.plot(locs, x2, c='#1f77b4')
        # # plt.title("Recovered background (composite model)")
        # plt.hlines(0, 0, 1, ls="--", color='#7f7f7f')
        # if save_pdf:
        #     plt.savefig(os.path.join(figures_path, "recos.pdf"))
        # plt.show()

        # repr_best_reco = np.convolve(best_reco["x1"], representation_kernel, mode="same")
        # repr_source = np.convolve(data["img"], representation_kernel, mode="same")

        # Reconstruction after convolution
        # ymax = 1.05 * max(repr_source.max(), repr_recovered.max())
        # plt.figure(figsize=(12, 4))
        # plt.subplot(121)
        # plt.plot(locs, repr_source, c='#ff7f0e', )  # marker='.')
        # plt.ylim(top=ymax)
        # # plt.title("Source convolved")
        # plt.subplot(122)
        # plt.plot(locs, repr_recovered, c='#ff7f0e', )  # marker='.')
        # # plt.title("Foreground recovered convolved")
        # plt.ylim(top=ymax)
        # for ax in plt.gcf().axes:
        #     ax.label_outer()
        # if save_pdf:
        #     plt.savefig(os.path.join(figures_path, "recos_conv.pdf"))
        # plt.show()


        # For resubmission: superpose the gt and the reconstructions
        fig, axs = plt.subplots(1, 2, figsize=set_figsize(aspect=0.4))
        ax = axs[0]
        ax.stem(locs[img != 0], img[img != 0], basefmt="white", linefmt="C1-", markerfmt="o")
        ax.stem(locs[x1 != 0], x1[x1 != 0], basefmt="white", linefmt="C2:")
        ax.stem(None, None, basefmt="C7--", linefmt="C1-", label="Ground truth", markerfmt="o")
        ax.stem(None, None, basefmt="C7--", linefmt="C2:", label="Reconstruction")
        ax.stem([0, 1], [0, 0], markerfmt='white', basefmt='C7--')
        ax.legend(borderpad=0.5, handleheight=2.0, loc='upper center',
            bbox_to_anchor=(0.5, 1.48),
            ncol=1,
            frameon=True)
        ax = axs[1]
        ax.plot(locs, background, c='#1f77b4', label="Ground truth", alpha=.6)
        ax.plot(locs, x2, c='#1f77b4', ls='--', label="Reconstruction")
        # plt.title("Recovered background (composite model)")
        ax.hlines(0, 0, 1, ls="--", color='#7f7f7f')
        ax.legend(borderpad=0.5, handleheight=2.0, loc='upper center',
            bbox_to_anchor=(0.5, 1.48),
            ncol=1,
            frameon=True)
        plt.subplots_adjust(top=0.72, wspace=0.3)
        if save_pdf:
            plt.savefig(os.path.join(figures_path, "recos.pdf"),)
        plt.show()

        # Reconstruction after convolution

        ls = (0, (5, 1))
        ymax = 1.05 * max(repr_source.max(), repr_recovered.max())
        fig, ax = plt.subplots(figsize=set_figsize(aspect=.5))
        ax.plot(locs, repr_source, c='#ff7f0e', label="Ground truth")
        ax.plot(locs, repr_recovered, c='#663000', ls=ls, zorder=2, label="Reconstruction")
        plt.ylim(top=ymax)
        # Zoom right
        xinf, xsup = 0.56, 0.59
        yinf, ysup = 130, 230
        axins = inset_axes(ax, width="20%", height="40%", bbox_to_anchor=(0.7, 0.15, 1, 1), bbox_transform=ax.transAxes, loc='lower left')
        axins.set_xlim(xinf, xsup)
        axins.set_ylim(yinf, ysup)
        axins.plot(locs, repr_source, c='#ff7f0e', label="Ground truth")
        axins.plot(locs, repr_recovered, c='#663000', ls=ls, zorder=2, label="Reconstruction")
        axins.set_xticks([0.57, 0.59])
        # axins.set_yticks([])
        axins.yaxis.set_ticks_position('right')
        mark_inset(ax, axins, loc1=2, loc2=3, fc="none", ec="0.5")
        # Zoom left
        xinf, xsup = 0.325, 0.355
        yinf, ysup = 420, 520
        axins2 = inset_axes(ax, width="20%", height="40%", bbox_to_anchor=(0.08, 0.4, 1, 1), bbox_transform=ax.transAxes, loc='lower left')
        axins2.set_xlim(xinf, xsup)
        axins2.set_ylim(yinf, ysup)
        axins2.plot(locs, repr_source, c='#ff7f0e', label="Ground truth")
        axins2.plot(locs, repr_recovered, c='#663000', ls=ls, zorder=2, label="Reconstruction")
        mark_inset(ax, axins2, loc1=2, loc2=4, fc="none", ec="0.5")
        axins2.set_xticks([0.33, 0.35])
        # axins2.set_yticks([])
        ax.legend(loc="upper right")
        if save_pdf:
            plt.savefig(os.path.join(figures_path, "recos_conv.pdf"))
        plt.show()


        measx1 = np.convolve(np.pad(x1, (kernel_width // 2, kernel_width // 2), mode='constant'), kernel_measurement,
                             mode='valid')
        measx2 = np.convolve(np.pad(x2, (kernel_width // 2, kernel_width // 2), mode='constant'), kernel_measurement,
                             mode='valid') / Ngrid
        sol_meas = (measx1 + measx2)[srf // 2::srf]

        # # Sum of the measurements
        # marker='+'
        # yrange = [min(3*y.min(), y.min()-0.05, -.05), 1.05*y.max()]
        # plt.figure(figsize=set_figsize(aspect=.3))
        # plt.subplot(131)
        # plt.scatter(np.arange(Nmeas)/Nmeas,meas_fg, marker=marker, color='g') # color='#ff7f0e')
        # # plt.title("Observations on the foreground")
        # plt.ylim(yrange)
        # plt.subplot(132)
        # plt.scatter(np.arange(Nmeas)/Nmeas,meas_bg, marker=marker, color='g') #'#1f77b4')
        # # plt.title("Observations on the background")
        # plt.ylim(yrange)
        # plt.subplot(133)
        # plt.scatter(np.arange(Nmeas)/Nmeas, y, marker=marker, color='g')
        # # plt.title("Noisy measurements")
        # plt.ylim(yrange)
        # fig = plt.gcf()
        # for ax in fig.axes:
        #     ax.label_outer()
        # # if save_pdf:
        # #     plt.savefig(os.path.join(figures_path, "measurements.pdf"))
        # plt.show()

        plt.figure(figsize=set_figsize(fraction=.7, aspect=.5))
        plt.scatter(np.arange(Nmeas)/Nmeas,meas_fg+meas_bg, color='g', marker='+', s=50, label="Noiseless obs.")
        plt.scatter(np.arange(Nmeas)/Nmeas, sol_meas, marker='.', color='#bcbd22', alpha=.7, label="A posteriori obs.")
        plt.ylim(yrange)
        plt.legend()
        if save_pdf:
            plt.savefig(os.path.join(figures_path, "aposteriori_obs.pdf"))
        plt.show()
        print(f"Rel error on noiseless observations and observations on the solution: {relL2Error(sol_meas, meas_fg+meas_bg)}")
        print(np.linalg.norm(sol_meas - meas_fg - meas_bg)**2/Nmeas)


    optimality_plots = True
    if optimality_plots:
        # Sub-optimality evaluation
        # 1. Check optimality of the discrete problem
        # 2. Check optimality of the continuous domain problem
        # (Both need to be evaluated for the foreground and background components)

        Ncontinuous = srf * Ngrid
        ckernel_std_int = np.floor(kernel_std * Ncontinuous).astype(int)
        ckernel_width = 4 * 2 * ckernel_std_int + 1  # Length of the Gaussian kernel
        continuous_kernel = np.exp(
            -0.5 * ((np.arange(ckernel_width) - (ckernel_width - 1) / 2) ** 2) / ((kernel_std * Ncontinuous) ** 2))
        norm_meas = (np.sqrt(2 * np.pi) * kernel_std)
        continuous_kernel /= norm_meas

        # plt.figure()
        # plt.subplot(121)
        # plt.scatter(np.arange(kernel_width)/kernel_width, kernel_measurement)
        # plt.subplot(122)
        # plt.scatter(np.arange(ckernel_width)/ckernel_width, continuous_kernel)
        # plt.show()

        measx1 = np.convolve(np.pad(x1, (kernel_width//2, kernel_width//2), mode='constant'), kernel_measurement, mode='valid')
        measx2 = np.convolve(np.pad(x2, (kernel_width//2, kernel_width//2), mode='constant'), kernel_measurement, mode='valid')/Ngrid
        sol_meas = (measx1 + measx2)[srf // 2::srf]
        residuals = y - sol_meas

        # plt.figure()
        # plt.stem(sol_meas)
        # plt.show()

        cresiduals = np.zeros(srf * Ngrid)
        cresiduals[srf*srf // 2::srf*srf] = residuals

        cont_certif = np.convolve(np.pad(cresiduals, (ckernel_width//2, ckernel_width//2), mode='constant'), continuous_kernel, mode='valid')/lambda1

        local_max1 = np.max(cont_certif[int(0.2 * cont_certif.shape[0]): int(0.4 * cont_certif.shape[0])])
        local_max2 = np.max(cont_certif[int(0.5 * cont_certif.shape[0]): int(0.7 * cont_certif.shape[0])])
        idx_max1 = np.where(cont_certif == local_max1)[0][0]
        idx_max2 = np.argwhere(cont_certif == local_max2)[0][0]

        plt.figure(figsize=set_figsize(aspect=.4))
        plt.hlines([-1, 1], 0, 1, ls='--', lw=1, color='red')
        plt.hlines(0, 0, 1, ls='--', lw=1, color='gray')
        plt.plot(np.arange(Ncontinuous)/Ncontinuous, cont_certif)
        plt.vlines([idx_max1/Ncontinuous, idx_max2/Ncontinuous], -.1, .1, ls='-', lw=1.5, color='k')
        plt.scatter([idx_max1/Ncontinuous, idx_max2/Ncontinuous], [local_max1, local_max2], marker='+', color='k', s=50, zorder=8)
        plt.annotate(f"{local_max1:.3f}", (idx_max1/Ncontinuous, local_max1), xytext=(idx_max1/Ncontinuous, 1.2), color='k',)
        plt.annotate(f"{local_max2:.3f}", (idx_max2/Ncontinuous, local_max2), xytext=(idx_max2/Ncontinuous, 1.2), color='k',)
        plt.ylim([-1.5, 1.5])
        plt.yticks(np.linspace(-1, 1, 5, endpoint=True))
        if save_pdf:
            plt.savefig(os.path.join(figures_path, "composite_certif.pdf"))
        plt.show()
        print(f"Max value of the empirical dual certificate at convergence: {cont_certif.max():.3f}")

        grid_residuals = np.zeros(Ngrid)
        grid_residuals[srf// 2::srf] = residuals
        hilbert_res = np.convolve(grid_residuals, kernel_target, mode='same') - lambda2 * x2

        plt.figure(figsize=set_figsize(aspect=.5))
        plt.plot(np.arange(Ngrid)/Ngrid, hilbert_res)
        plt.hlines(0, 0, 1, ls='--', lw=.5, color='gray')
        plt.show()

        # plt.figure()
        # plt.scatter(np.arange(Nmeas)/Nmeas, sol_meas - MlambdaInv(Hop(x1)))
        # plt.show()
        # tmp_res = sol_meas - MlambdaInv(Hop(x1))
        # padded_res = np.pad(tmp_res, kernel_regul.shape[0]//2, mode='constant')
        # norm_residuals = np.dot(tmp_res, np.convolve(kernel_regul, padded_res, mode='valid'))

        # ckernel_std_target_int = np.floor(kernel_std_target * Ncontinuous).astype(int)
        # ckernel_width_target = 3 * 2 * ckernel_std_target_int + 1
        # ckernel_target = np.exp(
        #     -0.5 * ((np.arange(ckernel_width_target) - (ckernel_width_target - 1) / 2) ** 2) / ((kernel_std_target * Ncontinuous) ** 2))
        # norm_target = (np.sqrt(2 * np.pi) * kernel_std_target)
        # ckernel_target /= norm_target

        # plt.figure()
        # plt.subplot(121)
        # plt.scatter(np.arange(kernel_width_target)/kernel_width_target, kernel_target)
        # plt.subplot(122)
        # plt.scatter(np.arange(ckernel_width_target)/ckernel_width_target, ckernel_target)
        # plt.show()
