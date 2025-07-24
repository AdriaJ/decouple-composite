import numpy as np
import matplotlib.pyplot as plt

save_pdf = True
figures_path = "/home/jarret/PycharmProjects/decouple-composite/figures"

kernel_std = 0.02  # Gaussian kernel std
Nmeas = 100
srf = 8

snrdb_meas = 10

if __name__=="__main__":
    Ngrid = srf * Nmeas
    # Continuous-time convolution and evaluate on the coarse grid
    # define the grid-based Gaussian kernels that I will need
    kernel_std_int = np.floor(kernel_std * Ngrid).astype(int)
    kernel_width = 3 * 2 * kernel_std_int + 1  # Length of the Gaussian kernel
    kernel_measurement = np.exp(
        -0.5 * ((np.arange(kernel_width) - (kernel_width - 1) / 2) ** 2) / ((kernel_std * Ngrid) ** 2))
    norm_meas = (np.sqrt(2 * np.pi) * kernel_std)
    kernel_measurement /= norm_meas

    X = np.arange(10_000)/9_999
    pos = (70*8-1)/(Ngrid-1)
    continuous_conv = np.exp(-0.5 * ((X - pos) ** 2) / ((kernel_std)** 2)) / (norm_meas)

    img = np.zeros((Ngrid,))
    img[70*8-1] = 1.
    conv_fg = np.convolve(np.pad(img, (kernel_width//2, kernel_width//2), mode='wrap'),
                          kernel_measurement, mode='valid')
    meas_fg = conv_fg[2::srf]

    #todo : add noise
    supp_size = (meas_fg != 0).sum()
    sigma_noise = np.linalg.norm(meas_fg)/supp_size * 10**(-snrdb_meas / 20)
    noise_meas = np.random.normal(0, sigma_noise, meas_fg.shape)
    limited_noise = np.zeros(Nmeas)
    limited_noise[meas_fg != 0] = noise_meas[meas_fg != 0]

    plt.figure(figsize=(8, 4))
    # plt.stem(np.arange(100)/99, img, label="Dirac impulse")
    plt.scatter(np.arange(Ngrid)[2::srf]/(Ngrid-1), np.zeros(100), marker='|', color='k', alpha=.7)
    plt.hlines(0, 0, 1, linestyles='-', color='k', alpha=.7)
    plt.arrow(pos, 0, 0, 10, color='r', width=0.0005,
              head_width=0.006, head_length=1., label="Dirac impulse (height = 1)")
    plt.plot(X, continuous_conv, color='orange', label="Continuous convolution")
    plt.scatter(np.arange(Ngrid)[2::8]/(Ngrid-1), meas_fg, label="Measurements", marker='+', s=70, zorder=10)
    plt.scatter((np.arange(Ngrid)[2::8]/(Ngrid-1))[meas_fg != 0], (meas_fg + limited_noise)[meas_fg != 0], label="Noisy measurements", marker='+', s=70, zorder=10)
    plt.legend()
    plt.xlim([0.53, 0.78])
    if save_pdf:
        plt.savefig(f"{figures_path}/measurement_figure.pdf")
    plt.show()

