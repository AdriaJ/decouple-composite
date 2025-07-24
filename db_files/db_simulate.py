"""
Simulate a sparse continuous-domain foreground and a continuous-domain background.
The foreground is a sum of Dirac impulses, while the background is a sum of Gaussian kernels.
"""

import os
import argparse
import yaml
import numpy as np
import scipy.signal as sig

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, help='Seed', default=None)
    parser.add_argument('--srf', type=int, default=8)
    parser.add_argument('--save', type=str, default=None)
    parser.add_argument('--r12', type=float, default=1.)
    args = parser.parse_args()

    if args.seed:
        seed = int(args.seed)
    else:
        seed = np.random.randint(1000)

    if args.save is None:
        save_path = os.getcwd()
    else:
        save_path = args.save
    with open(os.path.join(save_path, '..', '..', 'db_config.yaml'), 'r') as config_file:
        config = yaml.safe_load(config_file)

    Nmeas = config["meas_model"]["Nmeas"]
    kernel_std = config["meas_model"]["kernel_std"]
    snrdb_meas = config["meas_model"]["snrdb"]
    max_intensity = config["im_model"]["max_intensity"]
    k = config["im_model"]["k"]
    ongrid = config["im_model"]["ongrid"]
    kernel_bg_factor = config["im_model"]["kernel_bg_factor"]
    srf = args.srf
    Ngrid = srf * Nmeas
    kernel_std_bg = kernel_bg_factor * kernel_std

    # define the grid-based Gaussian kernels needed
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

    rng = np.random.default_rng(seed=seed)

    if ongrid:
        img = np.zeros((Ngrid,))
        Neff = int(.5 * Ngrid)
        idx = rng.choice(Neff, k, replace=False)
        indices = idx + int(.25 * Ngrid)
        img[indices] = rng.uniform(1, max_intensity, k)
        bg_impulses = np.zeros((Ngrid,))
        kk = config["im_model"]["bg_k_factor"] * k
        idx = rng.choice(Neff, kk, replace=False)
        indices = idx + int(.25 * Ngrid)
        bg_impulses[indices] = 1 + rng.uniform(-.5, .5, kk)
        background = sig.fftconvolve(bg_impulses, kernel_bg_1d, mode='same')

    conv_fg = np.convolve(np.pad(img, (kernel_width//2, kernel_width//2), mode='wrap'), # constant
                          kernel_measurement, mode='valid')
    meas_fg = conv_fg[srf // 2::srf]

    conv_bg = np.convolve(np.pad(bg_impulses, (width_meas_bg//2, width_meas_bg//2), mode='wrap'), # constant
                          kernel_meas_bg, mode='valid')
    meas_bg = conv_bg[srf // 2::srf]

    # Normalization of the measurements with desired ratio
    factor = args.r12 * np.linalg.norm(meas_bg) / np.linalg.norm(meas_fg)
    img *= factor
    meas_fg *= factor

    noiseless_y = meas_fg + meas_bg

    sigma_noise = np.linalg.norm(noiseless_y)/Nmeas * 10**(-snrdb_meas / 20)
    noise_meas = rng.normal(0, sigma_noise, noiseless_y.shape)
    y = noiseless_y + noise_meas

    np.savez(os.path.join(save_path, "gt_data.npz"),
             img=img,
             background=background,
             measurements=y,
             noise_meas=noise_meas)
