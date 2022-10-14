import os
import numpy as np
from matplotlib import pyplot as plt

training_methods = ['cohen', 'consistency']

datasets = ['tinyimagenet-4700', 'imagenet-75260']

dims = {'mnist-380': 28 * 28,
        'cifar-1530': 3 * 32 * 32,
        'tinyimagenet-4700': 3 * 64 * 64,
        'imagenet-75260': 3 * 224 * 224}

sigmas = ['0.50']


if __name__ == '__main__':
    for training_method in training_methods:
        for dataset in datasets:
            for sigma in sigmas:
                folder_name = f'data/sampling/{training_method}-{dataset}-{sigma}.pth'
                file_name = f'general-gaussian-{dataset.split("-")[1]}-0.5-100000-0.001.txt'
                print(folder_name)
                paLs = list()
                paRs = list()
                with open(os.path.join(folder_name, file_name), 'r') as f:
                    for line in f.readlines():
                        if line[0] == 'o':
                            line_cells = line.split(' ')
                            if float(line_cells[2]) > 0.5:
                                paLs.append(float(line_cells[2]))
                                paRs.append(float(line_cells[3]))
                plt.style.use('seaborn')
                plt.figure(figsize=(4.5,2))
                plt.subplots_adjust(left=0.1, bottom=0.25, right=0.99, top=0.87, wspace=0, hspace=0)

                plt.hist(- np.array(paLs) + 1., bins=[0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1,0.5], density=True)
                plt.xscale('log')
                if dataset.startswith('tinyimagenet'):
                    datasett = 'TinyImageNet'
                if dataset.startswith('imagenet'):
                    datasett = 'ImageNet'
                if training_method == 'cohen':
                    training_methodt = 'Gaussian Aug.'
                if training_method == 'consistency':
                    training_methodt = 'Consistency'
                plt.title(f'{datasett}($d = {dims[dataset]}$) trained w/ {training_methodt} ')
                plt.xlabel(f'$1-P_A$')
                # plt.show()
                print('# samples =', len(paLs))
                print(f'{np.mean(paLs):.3f}', '+-', f'{np.std(paLs, ddof=1):.3f}')
                print(f'{np.mean(paRs):.3f}', '+-', f'{np.std(paRs, ddof=1):.3f}')
                print('')
                plt.savefig(f'ablation/average-pa-{dataset}-{training_method}.pdf')