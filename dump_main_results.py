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
result_folder = 'result/'

def read_improved_radius(model, disttype, k, sigma, betas, N, alpha, print_detail=False):
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
    arr = arr[np.argsort(arr[:, 0])]
    if print_detail:
        counter = [0 for _ in betas + [None]]
        for item in arr:
            if max(item) - item[0] >= 1e-6:
                counter = [counter[i] + 1 if max(item) - item[i] <= 1e-6 else counter[i] for i in range(len(item))]
        # for item in arr:
        #     print(np.array_repr(item).replace('\n', ''))
        print(['Orig'] + betas)
        print(counter)
    return arr.max(axis=1)

def read_original_radius(model, disttype, k, sigma, N, alpha):
    raw_orig_Rs = read_orig_Rs(os.path.join(orig_radius_dir, model, f'orig-rad-{disttype}-{k}-{sigma}-{N}-{alpha}.txt'),
                               [])
    slots = [[item[0], item[1]] for item in raw_orig_Rs]
    arr = np.array(slots)[:, 1]
    arr.sort()
    return arr

def read_original_acr(model, disttype, k, sigma, N, alpha):
    raw_orig_Rs = read_orig_Rs(os.path.join(orig_radius_dir, model, f'orig-rad-{disttype}-{k}-{sigma}-{N}-{alpha}.txt'),
                               [])
    slots = [[item[0], item[1]] for item in raw_orig_Rs]
    ans = np.mean(np.array(slots)[:, 1])
    return ans

def read_improved_acr(model, disttype, k, sigma, betas, N, alpha, print_detail=False):
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
    ans = np.mean(slots)
    return ans

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



def plot_original_curve_series(models, disttype, k, sigmas, N, alpha):
    xs = dict()
    ys = dict()
    for model, sigma in zip(models, sigmas):
        raw_orig_Rs = read_orig_Rs(os.path.join(orig_radius_dir, model, f'orig-rad-{disttype}-{k}-{sigma}-{N}-{alpha}.txt'),
                                   [])
        slots = [[item[0], item[1]] for item in raw_orig_Rs]
        rads = np.sort(np.array(slots)[:, 1])
        tot = len(slots)
        rads = rads[rads >= 1e-6]
        xs[sigma] = (rads).tolist()
        ys[sigma] = list(np.array(range(len(rads)-1, -1, -1)) / tot)

    all_x = list()
    all_y = list()
    for sigma in sigmas:
        all_x += xs[sigma]
    all_x = sorted(list(set(all_x)))
    pointers = dict()
    for sigma in sigmas:
        pointers[sigma] = 0
    for x in all_x:
        now_y = 0.
        for sigma in sigmas:
            while pointers[sigma] < len(xs[sigma]) - 1 and xs[sigma][pointers[sigma] + 1] <= x:
                pointers[sigma] += 1
            if x - xs[sigma][pointers[sigma]] < 1e-6:
                now_y = max(now_y, ys[sigma][pointers[sigma]])
            else:
                # x - xs[sigma][pointers[sigma]] >= 1e-6:
                now_y = max(now_y, ys[sigma][pointers[sigma]] - 1. / tot)
        all_y.append(now_y)

    return all_x, all_y

def plot_improved_curve_series(models, disttype, k, sigmas, betas, N, alpha, print_detail=False):
    xs = dict()
    ys = dict()
    for model, sigma, beta in zip(models, sigmas, betas):
        fname = f'new-rad-{disttype}-{k}-{sigma}-{beta}-{N}-{alpha}.txt'
        with open(os.path.join(new_radius_dir, model, fname), 'r') as f:
            slots = list()
            for line in f.readlines():
                line = line.strip()
                line_fields = line.split(' ')
                no, new_r = int(line_fields[0]), float(line_fields[1])
                slots.append([no, new_r])
        rads = np.sort(np.array(slots)[:, 1])
        tot = len(slots)
        rads = rads[rads >= 1e-6]
        xs[sigma] = (rads).tolist()
        ys[sigma] = list(np.array(range(len(rads)-1, -1, -1)) / tot)

    all_x = list()
    all_y = list()
    for sigma in sigmas:
        all_x += xs[sigma]
    all_x = sorted(list(set(all_x)))
    pointers = dict()
    for sigma in sigmas:
        pointers[sigma] = 0
    for x in all_x:
        now_y = 0.
        for sigma in sigmas:
            while pointers[sigma] < len(xs[sigma]) - 1 and xs[sigma][pointers[sigma] + 1] <= x:
                pointers[sigma] += 1
            if x - xs[sigma][pointers[sigma]] < 1e-6:
                now_y = max(now_y, ys[sigma][pointers[sigma]])
            else:
                # x - xs[sigma][pointers[sigma]] >= 1e-6:
                now_y = max(now_y, ys[sigma][pointers[sigma]] - 1. / tot)
        all_y.append(now_y)

    return all_x, all_y



def nice_print(arr, unit=0.05, infty=False, prefix=' ', pre_decorater='$', suf_decorator='\\%$ &'):
    # for console print
    pre_decorater = ''
    suf_decorator = '%'
    #
    max_u = 0.
    while sum(arr >= max_u - 1e-5) > 0:
        max_u += unit
    max_u -= unit
    tot = len(arr)
    # print(f'tot = {tot}')
    line_1 = ''.join(['R   \t'] + [f'{unit * i if not infty else unit * i * 255.: .2f}\t' for i in range(int(max_u / unit) + 1)])
    line_2 = ''.join(['RAcc\t'] + [prefix + f'{pre_decorater}{sum(arr >= unit * i - 1e-5) / tot * 100.:.1f}{suf_decorator}\t' for i in range(int(max_u / unit) + 1)])
    print(line_1)
    print(line_2)
    print(f'avg ACR = {np.mean(arr):.3f}')


class Unbuffered():

    def __init__(self, stream, filestream):

        self.stream = stream
        self.file_stream = filestream

    def write(self, data):

        self.stream.write(data)
        self.stream.flush()
        self.file_stream.write(data)    # Write the data of stdout here to a text file as well

    def flush(self):
        # this flush method is needed for python 3 compatibility.
        # this handles the flush command by doing nothing.
        # you might want to specify some extra behavior here.
        pass

def dataset_level_print(ds_print_name, ds_internal_name, train_print_name, train_internal_name, k, N, Qsigmas,
                        standard_stdout):

    now_f_out = open(result_folder + f'/{ds_print_name}_{train_print_name}.txt', 'w')
    sys.stdout = Unbuffered(standard_stdout, now_f_out)

    print('R: radius, RAcc: certified accuracy')
    print('  Note: for R=0.00, we report the benign accuracy in the paper instead of 100.0%.\n'
          '        The benign accuracy can be obtained by running benign_sampler.py\n'
          '')
    print('*' * 5, f'Results on {ds_print_name} with {train_print_name}:', '*' * 5)

    print(f'Smoothing distribution: generalized Gaussian with k = {k}, sigma = 0.25')
    model = f'{train_internal_name}-{ds_internal_name}-{k}-0.25.pth'
    sigma = 0.25
    print('Neyman-Pearson certification')
    nice_print(read_original_radius(model, 'general-gaussian', k, sigma, N, 0.001), unit=0.25)
    print('DSRS Certification')
    nice_print(read_improved_radius(model, 'general-gaussian' + ('-th' if ds_internal_name == 'imagenet' else ''), k, sigma, [Qsigmas[0]], N//2, 0.0005), unit=0.25)

    print('')

    print(f'Smoothing distribution: generalized Gaussian with k = {k}, sigma = 0.50')
    model = f'{train_internal_name}-{ds_internal_name}-{k}-0.50.pth'
    sigma = 0.50
    print('Neyman-Pearson certification')
    nice_print(read_original_radius(model, 'general-gaussian', k, sigma, N, 0.001), unit=0.25)
    print('DSRS Certification')
    nice_print(read_improved_radius(model, 'general-gaussian' + ('-th' if ds_internal_name == 'imagenet' else ''), k, sigma, [Qsigmas[1]], N//2, 0.0005), unit=0.25)

    print('')

    print(f'Smoothing distribution: generalized Gaussian with k = {k}, sigma = 1.00')
    model = f'{train_internal_name}-{ds_internal_name}-{k}-1.00.pth'
    sigma = 1.00
    print('Neyman-Pearson certification')
    nice_print(read_original_radius(model, 'general-gaussian', k, sigma, N, 0.001), unit=0.25)
    print('DSRS Certification')
    nice_print(read_improved_radius(model, 'general-gaussian' + ('-th' if ds_internal_name == 'imagenet' else ''), k, sigma, [Qsigmas[2]], N//2, 0.0005), unit=0.25)

    now_f_out.close()

    sys.stdout = standard_stdout

if __name__ == '__main__':
    np.set_printoptions(precision=4)

    if not os.path.exists(result_folder):
        os.makedirs(result_folder)

    # generic options
    N = 100000
    alpha = 0.0005
    disttype = 'general-gaussian'

    """ Script for Figure 2(a) is ablation/less_ideal_sigma.py """

    """ Figure 2(b) """
    print('Output Figure 2(b)...')
    model = 'salman-imagenet-0.50.pth'
    k = 75260
    sigma = 0.5

    plt.clf()
    plt.style.use('seaborn')
    # plt.figure(figsize=(3.6, 3.6))
    # plt.subplots_adjust(left=0.165, bottom=0.3, right=0.99, top=0.88, wspace=0, hspace=0)
    plt.figure(figsize=(8,3.3))
    plt.subplots_adjust(left=0.08, bottom=0.13, right=0.99, top=0.90, wspace=0, hspace=0)
    plt.ylabel('Certified Accuracy', fontsize=14)
    plt.xlabel('$r$', fontsize=14)
    # x, y = plot_original_curve(model, 'gaussian', None, sigma, N, 0.001)
    # plt.plot(x, y, label='Neyman-Pearson Standard Gaussian')
    x, y = plot_original_curve(model, 'general-gaussian', k, sigma, N//2, 0.001)
    plt.plot(x, y, label='Neyman-Pearson ($N=50000$)')
    x, y = plot_original_curve(model, 'general-gaussian', k, sigma, N, 0.001)
    plt.plot(x, y, label='Neyman-Pearson ($N=100000$)')
    x, y = plot_original_curve(model, 'general-gaussian', k, sigma, N*2, 0.001)
    plt.plot(x, y, label='Neyman-Pearson ($N=200000$)')
    x, y = plot_improved_curve(model, 'general-gaussian-th', k, sigma, ['x2'], N//2, 0.0005)
    plt.plot(x, y, '--', label='DSRS ($N=50000+50000$)')
    x, y = plot_improved_curve(model, 'general-gaussian-th', k, sigma, ['x2'], N, 0.0005)
    plt.plot(x, y, '--', label='DSRS ($N=50000+100000$)')
    x, y = plot_improved_curve(model, 'general-gaussian-th', k, sigma, ['x2'], N*2, 0.0005)
    plt.plot(x, y, '--', label='DSRS ($N=50000+200000$)')
    x, y = plot_improved_curve(model, 'general-gaussian-th', k, sigma, ['x2'], N*4, 0.0005)
    plt.plot(x, y, '--', label='DSRS ($N=50000+400000$)')
    x, y = plot_improved_curve(model, 'general-gaussian-th', k, sigma, ['x2'], N*8, 0.0005)
    plt.plot(x, y, '--', label='DSRS ($N=50000+800000$)')
    # x, y = plot_improved_curve(model, 'general-gaussian-th', k, sigma, ['x2'], N*8, 0.005)
    # plt.plot(x, y, label='DSRS General Gaussian ($N=800000$, $\\alpha=0.01$)')
    # plt.legend(bbox_to_anchor=(-0.2,-0.2), loc="upper left", ncol=2, fontsize='x-small')
    # plt.title('ImageNet, smoothadv model from\n (Salmen et al., 2019), $\sigma$=0.50')
    plt.title('ImageNet, smoothadv model from (Salmen et al., 2019), $\sigma$=0.50', fontsize=16)
    plt.xlim([0,2.4])
    plt.legend()
    plt.savefig(result_folder + '/figure_2b.pdf')

    print('=' * 20)

    """ Cache oldstdout """
    standard_stdout = sys.stdout

    """ Output main tables """
    """ Note: the same table is outputted to both stdout and corresponding named txts in results/ folder """
    """ Note: Table 2 in the paper takes the maximum certified accuracy across all three P sigma's, so it is the maximum cell among three corresponding tables respectively. """
    """ Note: the accuracy when radius = 0.0 in the paper corresponds to the benign accuracy, which needs to be obtained via benign_sampler.py """
    """ Note: the result for sigma = 1.00 is used to generate Table 9 """
    print(""">>>>> Table 2 and Table 6 - MNIST - Gaussian Augmentation """)
    dataset_level_print('MNIST', 'mnist', 'Gaussian Augmentation', 'cohen', 380, N, [0.2, 0.4, 0.8],
                        standard_stdout)

    print(""">>>>> Table 2 and Table 6 - MNIST - Consistency """)
    dataset_level_print('MNIST', 'mnist', 'Consistency', 'consistency', 380, N, [0.2, 0.4, 0.8],
                        standard_stdout)

    print(""">>>>> Table 2 and Table 6 - MNIST - SmoothMix """)
    dataset_level_print('MNIST', 'mnist', 'SmoothMix', 'smoothmix', 380, N, [0.2, 0.4, 0.8],
                        standard_stdout)

    print(""">>>>> Table 2 and Table 7 - CIFAR10 - Gaussian Augmentation """)
    dataset_level_print('CIFAR10', 'cifar', 'Gaussian Augmentation', 'cohen', 1530, N, [0.2, 0.4, 0.8],
                        standard_stdout)

    print(""">>>>> Table 2 and Table 7 - CIFAR10 - Consistency """)
    dataset_level_print('CIFAR10', 'cifar', 'Consistency', 'consistency', 1530, N, [0.2, 0.4, 0.8],
                        standard_stdout)

    print(""">>>>> Table 2 and Table 7 - CIFAR10 - SmoothMix """)
    dataset_level_print('CIFAR10', 'cifar', 'SmoothMix', 'smoothmix', 1530, N, [0.2, 0.4, 0.8],
                        standard_stdout)

    print(""">>>>> Table 2 and Table 8 - ImageNet - Gaussian Augmentation """)
    dataset_level_print('ImageNet', 'imagenet', 'Gaussian Augmentation', 'cohen', 75260, N, ['x+', 'x+', 'x+'],
                        standard_stdout)

    print(""">>>>> Table 2 and Table 8 - ImageNet - Consistency """)
    dataset_level_print('ImageNet', 'imagenet', 'Consistency', 'consistency', 75260, N, ['x+', 'x+', 'x+'],
                        standard_stdout)

    print(""">>>>> Table 2 and Table 8 - ImageNet - SmoothMix """)
    dataset_level_print('ImageNet', 'imagenet', 'SmoothMix', 'smoothmix', 75260, N, ['x+', 'x+', 'x+'],
                        standard_stdout)

    """ Output result figure: Figure 8 in the paper """
    dataset_internal_names = ['mnist', 'cifar', 'imagenet']
    dataset_print_names = ['MNIST', 'CIFAR10', 'ImageNet']
    train_method_internal_names = ['cohen', 'consistency', 'smoothmix']
    train_method_print_names = ['Gaussian Augmentation', 'Consistency', 'SmoothMix']
    ks = [380, 1530, 75260]
    sigmas = [0.25, 0.50, 1.00]
    Qsigmas = [0.2, 0.4, 0.8]
    N = 100000

    for ds_indexer in range(3):
        for train_indexer in range(3):
            print(f'  Figure 8 [{ds_indexer + 1}, {train_indexer + 1}] -> result/{dataset_print_names[ds_indexer]}, {train_method_print_names[train_indexer]}.pdf')
            k = ks[ds_indexer]

            plt.clf()
            plt.style.use('seaborn')
            plt.figure(figsize=(4,3))
            plt.subplots_adjust(left=0.13, bottom=0.15, right=0.97, top=0.90, wspace=0, hspace=0)
            plt.ylabel('Certified Accuracy')
            plt.xlabel('Radius $r$')
            x, y = plot_original_curve_series(
                [f'{train_method_internal_names[train_indexer]}-{dataset_internal_names[ds_indexer]}-{k}-0.25.pth',
                 f'{train_method_internal_names[train_indexer]}-{dataset_internal_names[ds_indexer]}-{k}-0.50.pth',
                 f'{train_method_internal_names[train_indexer]}-{dataset_internal_names[ds_indexer]}-{k}-1.00.pth'],
                'general-gaussian', k, [0.25, 0.50, 1.00], N, 0.0010)
            plt.plot(x, y, label='Neyman-Pearson Certification')
            x, y = plot_improved_curve_series(
                [f'{train_method_internal_names[train_indexer]}-{dataset_internal_names[ds_indexer]}-{k}-0.25.pth',
                 f'{train_method_internal_names[train_indexer]}-{dataset_internal_names[ds_indexer]}-{k}-0.50.pth',
                 f'{train_method_internal_names[train_indexer]}-{dataset_internal_names[ds_indexer]}-{k}-1.00.pth'],
                'general-gaussian' + ('-th' if dataset_internal_names[ds_indexer] == 'imagenet' else ''),
                k,
                [0.25, 0.50, 1.00],
                ['x+', 'x+', 'x+'] if dataset_internal_names[ds_indexer] == 'imagenet' else Qsigmas, N//2, 0.0005)
            plt.xlim([0.0, 3.0])
            plt.plot(x, y, label='DSRS Certification')
            plt.legend()
            plt.title(f'{dataset_print_names[ds_indexer]}, {train_method_print_names[train_indexer]}')
            plt.savefig(f'result/{dataset_print_names[ds_indexer]}, {train_method_print_names[train_indexer]}.pdf')
            plt.savefig(f'figures/{dataset_internal_names[ds_indexer]}_{train_method_internal_names[train_indexer]}.pdf')

    print('Done! All result saved to result/ folder')