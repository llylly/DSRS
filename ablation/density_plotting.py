import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
import matplotlib.pyplot as plt

import numpy as np


def density_generator(k, d, std, range=1.0, samples=2000):
    sigma = np.sqrt(d / (d - k)) * std
    print(sigma)
    x = np.linspace(range / samples, range, num=samples).astype(np.float128) * np.sqrt(d)
    y = np.exp(- x ** 2 / (2.0 * sigma ** 2) + np.log(x) * (d-1-k))
    y = y / max(y)
    x /= np.sqrt(d)
    return x, y

if __name__ == '__main__':

    plt.style.use('seaborn')

    fig = plt.figure(figsize=(4, 3))
    plt.subplots_adjust(left=0.15, bottom=0.18, right=0.99, top=0.99, wspace=0, hspace=0)

    x1, y1 = density_generator(3062, 3072, 0.5)
    x2, y2 = density_generator(0, 3072, 0.5)
    x3, y3 = density_generator(3062, 3072, 0.45)
    x4, y4 = density_generator(0, 3072, 0.45)

    x5, y5 = density_generator(0, 10, 0.5)
    x6, y6 = density_generator(0, 10, 0.45)
    # y_max = max(max(y1), max(y2))
    # y1 /= y_max
    # y2 /= y_max


    # plt.plot(x5, y5, label='Normalized $r_p$')
    # plt.plot(x6, y6, label='Normalized $r_q$')

    # plt.plot(x2, y2, label='Normalized $r_p$')
    # plt.plot(x4, y4, label='Normalized $r_q$')

    plt.plot(x1, y1, label='Normalized $r_p$')
    plt.plot(x3, y3, label='Normalized $r_q$')


    plt.xlabel('$\\frac{x}{\sqrt{d}}$ where $x$ is $\ell_p$ Magnitude')
    plt.ylabel('Normalized $r_p(x)$ or $r_q(x)$')
    plt.legend()

    # plt.show()

    # plt.savefig('ablation/radial-dist-low-dim-gaussian.pdf')
    # plt.savefig('ablation/radial-dist-high-dim-gaussian.pdf')
    plt.savefig('ablation/radial-dist-general-gaussian.pdf')