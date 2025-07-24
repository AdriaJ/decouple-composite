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


seed = 117_809
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
lambda1_factor = 0.25
lambda2 = 1e-2 * Nmeas * srf  # Ngrid
eps = 1e-5

blasso_factor = 0.35

srf_repr = 4

article_plots = True
save_pdf = True

# decoupled = True
do_non_decoupled = False
do_blasso = True

if __name__ == "__main__":
    if seed is None:
        seed = np.random.randint(1000)
    rng = np.random.default_rng(seed=seed)

    if save_pdf:
        import os
        figures_path = "figures"

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

    locs = np.arange(Ngrid) / Ngrid
    plt.figure(figsize=(15, 4))
    plt.subplot(141)
    plt.stem(locs[bg_impulses != 0], bg_impulses[bg_impulses != 0])
    plt.stem([0, 1 - 1/Ngrid], [0, 0], markerfmt='white')
    plt.subplot(142)
    plt.plot(locs, background, c='orange',)
    plt.subplot(143)
    plt.plot(np.arange(kernel_width) / Ngrid, kernel_measurement)
    plt.xlim([-.5, .5])
    plt.subplot(144)
    plt.stem(np.arange(Nmeas) * srf / Ngrid, meas_bg, basefmt="C7--", linefmt="C7-", markerfmt='gx')
    plt.suptitle("Background measurements")
    plt.show()

    dft = sfft.rfft(background, 3 * background.shape[0], norm="ortho")
    print(dft.shape)
    plt.figure(figsize=(15, 4))
    plt.subplot(131)
    plt.scatter(locs, background, c='orange',)
    plt.subplot(132)
    plt.scatter(np.arange(dft.shape[0]), np.abs(dft), marker='.')
    plt.subplot(133)
    plt.scatter(np.arange(dft.shape[0]), np.abs(dft), marker='.')
    plt.yscale('log')
    plt.show()

    dft_kernel_bg = sfft.rfft(kernel_bg_1d, 3 * kernel_bg_1d.shape[0], norm="ortho")
    plt.figure(figsize=(15, 4))
    plt.subplot(131)
    plt.scatter(np.arange(kernel_bg_1d.shape[0]), kernel_bg_1d, marker='.')
    plt.subplot(132)
    plt.scatter(np.arange(dft_kernel_bg.shape[0]), np.abs(dft_kernel_bg), marker='.')
    plt.subplot(133)
    plt.scatter(np.arange(dft_kernel_bg.shape[0]), np.abs(dft_kernel_bg), marker='.')
    plt.yscale('log')
    plt.show()