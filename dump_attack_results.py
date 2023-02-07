import os
import sys
from utils import read_orig_Rs
import numpy as np

import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
from matplotlib import pyplot as plt

new_radius_dir = 'data/new-radius'
orig_radius_dir = 'data/orig-radius'

# place to store the output human-friendly or TeX-friendly tables and figures
result_folder = 'result/attack'


if not os.path.exists(result_folder):
    os.makedirs(result_folder)

def plot_original_curve(model, disttype, k, sigma, N, alpha):
    raw_orig_Rs = read_orig_Rs(os.path.join(orig_radius_dir, model, f'orig-rad-{disttype}-{k}-{sigma}-{N}-{alpha}.txt'),
                               [])
    slots = [[item[0], item[1]] for item in raw_orig_Rs]
    rads = np.sort(np.array(slots)[:, 1])
    tot = len(slots)
    rads = rads[rads >= 1e-6]
    x = (rads).tolist()
    y = list(np.array(range(len(rads)-1, -1, -1)) / tot)

    return x, y

def plot_improved_curve(model, disttype, k, sigma, betas, N, alpha, print_detail=False):
    orig_disttype = disttype[:-3] if disttype.endswith('-th') else disttype
    raw_orig_Rs = read_orig_Rs(os.path.join(orig_radius_dir, model, f'orig-rad-{orig_disttype}-{k}-{sigma}-{N}-{alpha}.txt'),
                               [])
    slots = [[item[0], item[1]] + [None for _ in betas] for item in raw_orig_Rs]
    slot_idx = dict([(item[0], i) for i, item in enumerate(slots)])
    for beta_i, beta in enumerate(betas):
        fname = f'new-rad-{disttype}-{k}-{sigma}-{beta}-{N}-{alpha}.txt'
        with open(os.path.join(new_radius_dir, model, fname), 'r') as f:
            for line in f.readlines():
                line = line.strip()
                line_fields = line.split(' ')
                no, new_r = int(line_fields[0]), float(line_fields[1])
                slots[slot_idx[no]][2 + beta_i] = new_r
    arr = np.array(slots)[:, 1:]
    arr = arr.max(axis=1)

    rads = np.sort(arr)
    tot = len(rads)
    rads = rads[rads >= 1e-6]
    x = (rads).tolist()
    y = list(np.array(range(len(rads)-1, -1, -1)) / tot)

    return x, y


def plot_attacked_curve(model, disttype, k, sigma, eot_sample, step, pgd=False, start=10):

    files = os.listdir('data/attack/' + model)
    prefix = f'stats-{disttype}-{k}-{sigma}-sample-{eot_sample}-L-2-eps-'
    files = [fm for fm in files if fm.startswith(prefix)]
    if pgd:
        suffix = f'-step-{step}-pgd-start-{start}.txt'
    else:
        suffix = f'-step-{step}.txt'
    files = [fm for fm in files if fm.endswith(suffix)]
    filtered_radius = list()
    for f in files:
        filtered_radius.append(float(f[len(prefix): -len(suffix)]))

    print('filtered radius:', filtered_radius)

    x = []
    y = []
    for i, r in enumerate(filtered_radius):
        with open('data/attack/' + model + f'/{prefix}{r}{suffix}', 'r') as f:
            line = f.readlines()
            if len(line) > 0:
                line = line[0]
            else:
                continue
        N, corN, robN, corAcc, robAcc = line.split(' ')
        if 0. not in x:
            x.append(0.)
            y.append(float(corAcc))
        x.append(float(r))
        y.append(float(robAcc))

    x = np.array(x)
    y = np.array(y)[np.argsort(x)]
    x = x[np.argsort(x)]
    return x, y


if __name__ == '__main__':
    np.set_printoptions(precision=4)

    if not os.path.exists(result_folder):
        os.makedirs(result_folder)

    # generic options
    N = 100000
    alpha = 0.0005
    disttype = 'general-gaussian'

    """ Attacking smoothmix-mnist-380-0.50 """
    model = 'smoothmix'
    dataset = 'mnist'
    k = 380
    sigma = 0.5
    eot_sample = 100
    step = 200

    plt.clf()
    plt.style.use('seaborn')
    plt.ylabel('Robust Accuracy', fontsize=14)
    plt.xlabel('$r$', fontsize=14)
    plt.title('MNIST SmoothMix $\sigma=0.50$, Certified Accuracy vs. Empirical Upper Bound')

    x, y = plot_original_curve(f'{model}-{dataset}-{k}-{sigma:.2f}.pth', 'general-gaussian', k, sigma, N, 0.001)
    plt.plot(x, y, label='Neyman-Pearson Certification')
    x, y = plot_improved_curve(f'{model}-{dataset}-{k}-{sigma:.2f}.pth', 'general-gaussian-th', k, sigma, ['x+'], N//2, 0.0005)
    plt.plot(x, y, label='DSRS Certification')
    x, y = plot_attacked_curve(f'{model}-{dataset}-{k}-{sigma:.2f}.pth', 'general-gaussian', k, sigma, eot_sample, step)
    plt.plot(x, y, '-o', label='Upper bound from I-FGSM Attack')
    x, y = plot_attacked_curve(f'{model}-{dataset}-{k}-{sigma:.2f}.pth', 'general-gaussian', k, sigma, eot_sample, step, pgd=True)
    plt.plot(x, y, '-o', label='Upper bound from PGD Attack')
    plt.legend()
    plt.savefig(result_folder + '/smoothmix_mnist_0.50_380.pdf')
    # plt.show()


    """ Attacking smoothmix-cifar-1530-0.50 """
    model = 'smoothmix'
    dataset = 'cifar'
    k = 1530
    sigma = 0.5
    eot_sample = 100
    step = 200

    plt.clf()
    plt.style.use('seaborn')
    plt.ylabel('Robust Accuracy', fontsize=14)
    plt.xlabel('$r$', fontsize=14)
    plt.title('CIFAR-10 SmoothMix $\sigma=0.50$, Certified Accuracy vs. Empirical Upper Bound')

    x, y = plot_original_curve(f'{model}-{dataset}-{k}-{sigma:.2f}.pth', 'general-gaussian', k, sigma, N, 0.001)
    plt.plot(x, y, label='Neyman-Pearson Certification')
    x, y = plot_improved_curve(f'{model}-{dataset}-{k}-{sigma:.2f}.pth', 'general-gaussian-th', k, sigma, ['x+'], N//2, 0.0005)
    plt.plot(x, y, label='DSRS Certification')
    x, y = plot_attacked_curve(f'{model}-{dataset}-{k}-{sigma:.2f}.pth', 'general-gaussian', k, sigma, eot_sample, step)
    plt.plot(x, y, '-o', label='Upper bound from I-FGSM Attack')
    plt.legend()
    plt.savefig(result_folder + '/smoothmix_cifar_0.50_1530.pdf')
    # plt.show()


    """ Attacking consistency-imagenet-75260-0.50 """
    model = 'consistency'
    dataset = 'imagenet'
    k = 75260
    sigma = 0.5
    eot_sample = 100
    step = 200

    plt.clf()
    plt.style.use('seaborn')
    plt.ylabel('Robust Accuracy', fontsize=14)
    plt.xlabel('$r$', fontsize=14)
    plt.title('ImageNet Consistency $\sigma=0.50$, Certified Accuracy vs. Empirical Upper Bound')

    x, y = plot_original_curve(f'{model}-{dataset}-{k}-{sigma:.2f}.pth', 'general-gaussian', k, sigma, N, 0.001)
    plt.plot(x, y, label='Neyman-Pearson Certification')
    x, y = plot_improved_curve(f'{model}-{dataset}-{k}-{sigma:.2f}.pth', 'general-gaussian-th', k, sigma, ['x+'], N//2, 0.0005)
    plt.plot(x, y, label='DSRS Certification')
    x, y = plot_attacked_curve(f'{model}-{dataset}-{k}-{sigma:.2f}.pth', 'general-gaussian', k, sigma, eot_sample, step)
    plt.plot(x, y, '-o', label='Upper bound from I-FGSM Attack')
    plt.legend()
    plt.savefig(result_folder + '/consistency_imagenet_0.50_75260.pdf')
    # plt.show()


    """ Attacking salman-imagenet-75260-0.50 """
    model = 'salman'
    dataset = 'imagenet'
    k = 75260
    sigma = 0.5
    eot_sample = 100
    step = 200

    plt.clf()
    plt.style.use('seaborn')
    plt.ylabel('Robust Accuracy', fontsize=14)
    plt.xlabel('$r$', fontsize=14)
    plt.title('ImageNet Salman $\sigma=0.50$ Generalized Gaussian Smoothing\n Certified Accuracy vs. Empirical Upper Bound')

    x, y = plot_original_curve(f'{model}-{dataset}-{sigma:.2f}.pth', 'general-gaussian', k, sigma, N, 0.001)
    plt.plot(x, y, label='Neyman-Pearson Certification')
    x, y = plot_improved_curve(f'{model}-{dataset}-{sigma:.2f}.pth', 'general-gaussian-th', k, sigma, ['x2'], N//2, 0.0005)
    plt.plot(x, y, label='DSRS Certification')
    x, y = plot_attacked_curve(f'{model}-{dataset}-{sigma:.2f}.pth', 'general-gaussian', k, sigma, eot_sample, step)
    plt.plot(x, y, '-o', label='Upper bound from I-FGSM Attack')
    plt.legend()
    plt.savefig(result_folder + '/salman_imagenet_0.50_75260.pdf')
    # plt.show()


    """ Attacking salman-imagenet-75260-0.50 """
    model = 'salman'
    dataset = 'imagenet'
    k = None
    sigma = 0.5
    eot_sample = 100
    step = 200

    plt.clf()
    plt.style.use('seaborn')
    plt.ylabel('Robust Accuracy', fontsize=14)
    plt.xlabel('$r$', fontsize=14)
    plt.title('ImageNet Salman $\sigma=0.50$ Standard Gaussian Smoothing\n Certified Accuracy vs. Empirical Upper Bound')

    x, y = plot_original_curve(f'{model}-{dataset}-{sigma:.2f}.pth', 'gaussian', k, sigma, N, 0.001)
    plt.plot(x, y, label='Neyman-Pearson Certification')
    x, y = plot_attacked_curve(f'{model}-{dataset}-{sigma:.2f}.pth', 'gaussian', k, sigma, eot_sample, step)
    plt.plot(x, y, '-o', label='Upper bound from I-FGSM Attack')
    plt.legend()
    plt.savefig(result_folder + '/salman_imagenet_0.50.pdf')
    # plt.show()

    print('Done!')