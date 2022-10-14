import os
import numpy as np
from matplotlib import pyplot as plt

training_methods = ['cohen', 'consistency', 'smoothmix']

datasets = ['mnist-380', 'cifar-1530', 'imagenet-75260']

dims = {'mnist-380': 28 * 28,
        'cifar-1530': 3 * 32 * 32,
        'tinyimagenet-4700': 3 * 64 * 64,
        'imagenet-75260': 3 * 224 * 224}

sigmas = ['0.25', '0.50', '1.00']


if __name__ == '__main__':
    for training_method in training_methods:
        for sigma in sigmas:
            time_series = list()
            avg_time = list()
            for dataset in datasets:
                folder_name = f'data/new-radius/{training_method}-{dataset}-{sigma}.pth'
                file_name = f'time-new-rad-general-gaussian-th-{dataset.split("-")[1]}-{float(sigma)}-x+-50000-0.0005.txt'
                print(folder_name)
                times = list()
                with open(os.path.join(folder_name, file_name), 'r') as f:
                    for line in f.readlines():
                        times.append(float(line) / 10.)
                plt.style.use('seaborn')
                plt.figure(figsize=(4.5,2))
                plt.subplots_adjust(left=0.1, bottom=0.2, right=0.99, top=0.87, wspace=0, hspace=0)

                times = np.array(times)
                time_series.append(times)
                avg_time.append(np.mean(times))

            plt.hist(time_series)
            datasett = [{'mnist-380': 'MNIST', 'cifar-1530': 'CIFAR', 'imagenet-75260': 'ImageNet'}[dataset] + f' (avg={avg:.2f} s)' for dataset, avg in zip(datasets, avg_time)]
            training_methodt = {'cohen': 'Gaussian Aug.', 'consistency': 'Consistency', 'smoothmix': 'SmoothMix'}[training_method]
            plt.legend(datasett)
            plt.title(f'trained w/ {training_methodt} ($\sigma$={sigma})')
            plt.xlabel(f'$Time (s)$')
            # plt.show()
            plt.savefig(f'ablation/time_hist_{training_method}_{sigma}.pdf')