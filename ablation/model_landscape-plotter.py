import os
import sys
sys.path.append('.')
sys.path.append('..')
sys.path.append('../..')
from time import time

import argparse
# import setGPU

from math import ceil
import numpy as np

import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
from matplotlib import pyplot as plt



# INF = 'data/landscape_sampling/salman-imagenet-0.50.pth/0.0-400.0-41-1000.txt'
# OUTF = 'ablation/model_landscape.pdf'

# # INF = 'data/landscape_sampling/cohen-mnist-1.00.pth/0.0-80.0-50-1000.txt'
# # OUTF = 'ablation/model_landscape_cohen_mnist_1.00.pdf'
#
# # INF = 'data/landscape_sampling/consistency-mnist-1.00.pth/0.0-80.0-50-1000.txt'
# # OUTF = 'ablation/model_landscape_consistency_mnist_1.00.pdf'
#
# INF = 'data/landscape_sampling/cohen-cifar-0.50.pth/0.0-80.0-50-1000.txt'
# OUTF = 'ablation/model_landscape_cohen_cifar_0.50.pdf'
#
# INF = 'data/landscape_sampling/consistency-cifar-0.50.pth/0.0-80.0-50-1000.txt'
# OUTF = 'ablation/model_landscape_consistency_cifar_0.50.pdf'
#
# INF = 'data/landscape_sampling/cohen-imagenet-1.00-orig.pth/0.0-600.0-50-1000.txt'
# OUTF = 'ablation/model_landscape_cohen_imagenet-1.00.pdf'
#
# INF = 'data/landscape_sampling/salman-imagenet-1.00.pth/0.0-600.0-50-1000.txt'
# OUTF = 'ablation/model_landscape_salman_imagenet-1.00.pdf'


INF = 'data/landscape_sampling/cohen-imagenet-75260-1.00.pth/0.0-1200.0-31-100.txt'
OUTF = 'ablation/model_landscape_cohen_75260_imagenet-1.00.pdf'


# INF = 'data/landscape_sampling/cohen-cifar-1530-1.00.pth/0.0-220.0-31-1000.txt'
# OUTF = 'ablation/model_landscape_cohen_1530_cifar-1.00.pdf'



# INF = 'data/landscape_sampling/cohen-mnist-380-1.00.pth/0.0-220.0-62-1000.txt'
# OUTF = 'ablation/model_landscape_cohen_380_mnist-1.00.pdf'

if __name__ == '__main__':
    Xs = list()
    Ys = list()
    last_id = None
    with open(INF, 'r') as f:
        for line in f.readlines():
            tokens = line.split(' ')
            if tokens[0] != 'o':
                continue
            now_id, now_len, now_prob = float(tokens[1]), float(tokens[2]), float(tokens[3])
            if now_id == last_id:
                Xs[-1].append(now_len)
                Ys[-1].append(now_prob)
            else:
                last_id = now_id
                Xs.append([now_len])
                Ys.append([now_prob])

    plt.style.use('seaborn')
    fig, ax1 = plt.subplots(figsize=(5.5,3))
    # fig = plt.figure(figsize=(5,3))
    fig.subplots_adjust(left=0.13, bottom=0.15, right=0.9, top=0.99, wspace=0, hspace=0)

    ax2 = ax1.twinx()
    # ax1.set_ylim(-0.05, 1.05)
    # ax2.set_ylim(-0.05, 1.05)
    ax1.set_ylabel('Prob. of Predicting True Class', color='b', fontsize=14)
    # ax2.set_ylabel('Density of $\ell_2$ Noise Magnitude', color='g', fontsize=14)
    ax1.set_xlabel('$\ell_2$ Length of Perturbation')

    for X, Y in zip(Xs, Ys):
        ax1.plot(X, Y, 'b-', alpha=0.075)
        ax2.plot(X, Y, 'b-', alpha=0.075)

    d = 224 * 224 * 3
    sigma = 1.0
    # d = 32 * 32 * 3
    # sigma = 1.0
    # d = 28 * 28
    # sigma = 1.0
    radialX = np.linspace(0.01, np.max(np.array(Xs)), num=100000)
    radialY = np.log(radialX) * (d - 1.) + (-radialX ** 2. / (2.0 * sigma * sigma))
    radialY = radialY - np.max(radialY)
    radialY = np.exp(radialY)
    # ax1.plot(radialX, radialY, 'g-', linewidth=0.5)
    # ax2.plot(radialX, radialY, 'g-', linewidth=0.5)


    # plt.show()
    plt.savefig(OUTF)

    print(len(Xs))
