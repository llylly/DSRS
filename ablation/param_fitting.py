import os
import numpy as np
from matplotlib import pyplot as plt

from scipy.stats import gamma


import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
from matplotlib import pyplot as plt


disttype = 'general-gaussian'

model = 'consistency-cifar-1530-0.50.pth'
k = '1530'

# model = 'cohen-cifar-1530-0.50.pth'
# k = '1530'

# model = 'cohen-mnist-380-1.00.pth'
# k = '380'

# model = 'consistency-mnist-380-1.00.pth'
# k = '380'

# model = 'cohen-imagenet-75260-0.50.pth'
# k = '75260'

sigma = '0.5'
N = '50000'
alpha = '0.0005'

params = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9]
# params = [0.5, 0.6, 0.75, 0.9, 0.95]
# params = [0.5, 0.6]

pAs = dict()

old_radius = dict()

new_radius = dict()

max_radius = dict()

left_param = dict()

right_param = dict()

x = list()
y1 = list()
y2 = list()
y3 = list()

if __name__ == '__main__':
    with open(os.path.join('data/sampling', model, f'{disttype}-{k}-{sigma}-{N}-{alpha}.txt'), 'r') as f:
        for r in f.readlines():
            if r.startswith('o'):
                things = r.split(' ')
                pAs[int(things[1])] = float(things[2])

    with open(os.path.join('data/orig-radius', model, f'orig-rad-{disttype}-{k}-{sigma}-{N}-{alpha}.txt'), 'r') as f:
        for r in f.readlines():
            things = r.split(' ')
            old_radius[int(things[0])] = float(things[1])

    for param in params:
        new_radius[param] = dict()
        with open(os.path.join('data/new-radius', model, f'new-rad-{disttype}-th-{k}-{sigma}-{param}-{N}-{alpha}.txt'), 'r') as f:
            for r in f.readlines():
                things = r.split(' ')
                new_radius[param][int(things[0])] = float(things[1])

    for no in pAs:
        now_max = -1.
        for param in params:
            if new_radius[param][no] > now_max:
                now_max = new_radius[param][no]
        max_radius[no] = now_max

    for no in pAs:
        if max_radius[no] > 1e-5 and max_radius[no] - old_radius[no] > 1e-5:
            for param in params:
                t = new_radius[param][no]
                if max_radius[no] - t < 1e-5:
                    if no not in left_param: left_param[no] = param
                    if no not in right_param: right_param[no] = param
                    left_param[no] = min(left_param[no], param)
                    right_param[no] = max(right_param[no], param)


    for no in pAs:
        if max_radius[no] > 1e-5 and max_radius[no] - old_radius[no] > 1e-5:
            x.append(-np.log(1.0-pAs[no]))
            y1.append(left_param[no])
            y2.append(right_param[no])
            y3.append((left_param[no] + right_param[no]) / 2.)

    plt.figure(figsize=(8, 4), dpi=80)
    # plt.scatter(x, y1)
    # plt.scatter(x, y2)
    plt.errorbar(x, y3, np.array(y2) - np.array(y3), fmt='o')

    plt.xlabel('$-\\ln(1 - P_A)$')

    plt.ylabel('Best Hyperparameter $p$')

    y4 = [max(0.08 * _x + 0.2, 0.5) for _x in sorted(x)]
    # y4 = [1. - np.exp(-_x) - 0.1 for _x in sorted(x)]
    plt.plot(sorted(x), y4)

    plt.savefig('ablation/t_heuristic.pdf')
    # plt.show()

