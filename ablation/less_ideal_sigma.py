import scipy
import numpy as np
from scipy.stats import norm
from scipy.stats import gamma, beta
from multiprocessing.pool import Pool, ThreadPool
from statsmodels.stats.proportion import proportion_confint
from utils import lambertWlog
import time
import json
import os

from distribution import GeneralGaussian

import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
import matplotlib.pyplot as plt

plt.style.use('seaborn')

N = 10
plt.rcParams["axes.prop_cycle"] = plt.cycler("color", plt.cm.viridis(np.linspace(0,1,N)))


# plt.set_cmap('rainbow')
# plt.rcParams['image.cmap'] = 'gray'

from matplotlib import pyplot as plt


stime = time.time()


def radius(d, k, T, Treal, Ninside, method='new', p=0.001):

    sigma = 1.0

    if k == 0:
        Pa = gamma(d / 2.0).cdf(Treal ** 2 / (2.0 * sigma ** 2))
        print('  Pa =', Pa)
        if method == 'old':
            # simulate Pa
            r = sigma * norm.ppf(Pa)
            return r
    else:
        sigma = sigma * np.sqrt(d / (d - 2.0 * k))
        Pa = gamma(d / 2.0 - k).cdf(Treal ** 2 / (2.0 * sigma ** 2))
        print('  Pa =', Pa)
        if method == 'new':
            nu = 1.0 / gamma(d / 2 - k).cdf(T ** 2 / (2.0 * sigma * sigma))

            print('  nu =', nu)

            if Ninside is None:
                # assue Qa = 1
                Qa = 1.
            else:
                Qa = proportion_confint(Ninside, Ninside, alpha=2 * (p / 2.), method="beta")[0]
                print('Qa =', Qa, 'with # sample =', Ninside)

            Prest = Pa - Qa / nu

            print('  Qa =', Qa)
            print('  Prest =', Prest)

            r_l = 0.
            r_r = 10000.

            while r_r - r_l > 1e-6:
                now_r = (r_l + r_r) / 2

                if Prest < 1e-6:
                    u2 = 0.
                else:
                    # binary search to get u2

                    # determine lamb1
                    lamb1_L = - 50.
                    lamb1_R = + 50.
                    while lamb1_R - lamb1_L > 1e-6:
                        lamb1_mid = (lamb1_L + lamb1_R) / 2.0

                        now_P1it = gamma(d / 2.0 - k).expect(lambda t:
                             beta((d - 1.0) / 2.0, (d - 1.0) / 2.0).cdf(
                                 ( (now_r + sigma * np.sqrt(2.0 * t)) ** 2 - 2.0 * k * sigma * sigma * lambertWlog(t / k - lamb1_mid / k + np.log(t / k)) ) /
                                 (4.0 * sigma * now_r * np.sqrt(2.0 * t)))
                             ,lb=T**2 / (2.0 * sigma * sigma), ub=np.inf)

                        if now_P1it > Prest:
                            lamb1_R = lamb1_mid
                        else:
                            lamb1_L = lamb1_mid

                    lamb1 = (lamb1_L + lamb1_R) / 2.0

                    def u2func(t):
                        if lamb1 < -500.:
                            # lamb1 = 0
                            return 0.0
                        else:
                            return max(0.0, beta((d - 1.0) / 2.0, (d - 1.0) / 2.0).cdf(
                                (2.0 * sigma * sigma * k * lambertWlog(t/k + lamb1_L/k + np.log(t/k)) - (sigma * np.sqrt(2.0 * t) - now_r) ** 2)
                                /
                                (4.0 * sigma * now_r * np.sqrt(2.0 * t))
                            ) - beta((d - 1.0) / 2.0, (d - 1.0) / 2.0).cdf(
                                (T**2 - (sigma * np.sqrt(2.0 * t) - now_r) ** 2)
                                /
                                (4.0 * sigma * now_r * np.sqrt(2.0 * t))
                            ))

                    u2 = gamma(d / 2.0 - k).expect(lambda t: u2func(t), lb=0., ub=np.inf)

                if Ninside is None:
                    u1 = gamma(d / 2.0 - k).expect(lambda t: beta((d - 1.0) / 2.0, (d - 1.0) / 2.0).cdf(
                        (T**2 - (sigma * np.sqrt(2.0 * t) - now_r) ** 2) /
                        (4.0 * now_r * sigma * np.sqrt(2.0 * t))
                    ))
                else:
                    # found lamb1 + nu * lamb2 then integrate for u1

                    lamb12_L = - 50.
                    lamb12_R = + 50.
                    while lamb12_R - lamb12_L > 1e-6:
                        lamb12_mid = (lamb12_L + lamb12_R) / 2.0

                        now_Q = nu * gamma(d / 2.0 - k).expect(lambda t:
                                                               beta((d - 1.0) / 2.0, (d - 1.0) / 2.0).cdf(
                                                                   ( (now_r + sigma * np.sqrt(2.0 * t)) ** 2 - 2.0 * k * sigma * sigma * lambertWlog(t / k - lamb12_mid / k + np.log(t / k)) ) /
                                                                   (4.0 * sigma * now_r * np.sqrt(2.0 * t)))
                                                               ,lb=0., ub=T**2 / (2.0 * sigma * sigma))
                        # print(f'     targ Q = {q1}, now_Q = {now_Q} with log(lamb1 + nu * lamb2) = {lamb12_mid}')

                        if now_Q > Qa:
                            lamb12_R = lamb12_mid
                        else:
                            lamb12_L = lamb12_mid

                    lamb12 = (lamb12_L + lamb12_R) / 2.0

                    def u1func(t):
                        return beta((d - 1.0) / 2.0, (d - 1.0) / 2.0).cdf(
                            (min(T ** 2, 2.0 * sigma * sigma * k * lambertWlog(t/k + lamb12_L/k + np.log(t/k))) - (sigma * np.sqrt(2.0 * t) - now_r) ** 2)
                            /
                            (4.0 * sigma * now_r * np.sqrt(2.0 * t))
                        )
                    u1 = gamma(d / 2.0 - k).expect(lambda t: u1func(t), lb=0., ub=np.inf)

                if u1 + u2 > 0.5:
                    r_l = now_r
                else:
                    r_r = now_r
            return r_l
        elif method == 'old':
            gg = GeneralGaussian(d, k, sigma)
            return gg.certify_radius(Pa)


Pa_real=0.6
Pa_cover=0.5

compute = False
plot = True


def compute_all_in_some_dim(d, N_lim=10000000, Pa_real=Pa_real, Pa_cover=Pa_cover):
    k = (d - 10) / 2

    # T = np.sqrt(2. * d / (d - 2. * k) * gamma(d / 2.0 - k).ppf(Pa_cover))
    T = np.sqrt(2. * gamma(d / 2.0).ppf(Pa_cover))
    T_real = np.sqrt(2. * gamma(d / 2.0).ppf(Pa_real))
    # Pa = 0.8

    res = dict()

    r_neyman = radius(d, 0, T, T_real, None, 'old')
    print(f'(d={d}) old =', r_neyman)
    res['o'] = r_neyman

    r = radius(d, k, T, T_real, None)
    print(f'[{time.time() - stime}s]', f'(d={d}) ideal =', r)
    res['i'] = r

    N = 10
    while N <= N_lim:
        r = radius(d, k, T, T_real, N)
        print(f'[{time.time() - stime}s]', f'(d={d}) when sampling number N =', N, ', r =', r)
        res[f'{N}'] = r
        N *= 10

    print(res)

    return res


if __name__ == '__main__':
    # d = 28 * 28
    d = 20
    ds = list()
    while d <= 5000000:
        ds.append(d)
        d = int(d * 1.5 // 2 * 2)

    if compute:

        res = Pool(20).map(compute_all_in_some_dim, ds)
        print(res)
        with open(f'data/theory_simulations/less_ideal_sigma_wrt_dim_real_{Pa_real}_cover_{Pa_cover}.txt', 'a') as f:
            json.dump(res, f, indent=2)

        # k = (d - 10) / 2

        # T = np.sqrt(2. * d / (d - 2. * k) * gamma(d / 2.0 - k).ppf(0.5))
        # T_real = np.sqrt(2. * gamma(d / 2.0).ppf(0.8))
        # # Pa = 0.8
        #
        # r_neyman = radius(d, 0, T, T_real, None, 'old')
        # print('old =', r_neyman)
        #
        # r = radius(d, k, T, T_real, None)
        # print(f'[{time.time() - stime}s]', 'ideal =', r)
        #
        # N = 10
        # while N <= 10000000:
        #     r = radius(d, k, T, T_real, N)
        #     print(f'[{time.time() - stime}s]', 'when sampling number N =', N, ', r =', r)
        #     N *= 10

        print(f'output to data/theory_simulations/less_ideal_sigma_wrt_dim_real_{Pa_real}_cover_{Pa_cover}.txt')


    if plot:
        with open(f'data/theory_simulations/less_ideal_sigma_wrt_dim_real_{Pa_real}_cover_{Pa_cover}.txt', 'r') as f:
        # with open(f'ablation/less_ideal_sigma_wrt_dim_real_{Pa_real}_cover_{Pa_cover}.txt', 'r') as f:
            data = json.load(f)

        xs = sorted([int(d) for d in ds])
        ys = dict()
        for i, x in enumerate(xs):
            for y in data[i]:
                if y not in ys:
                    ys[y] = list()
                ys[y].append(data[i][y])
        print(xs)
        print(ys)

        max_with_i = max([max(l) for l in ys.values()])
        max_wo_i = max([max(l) for k, l in ys.items() if k != 'i'])

        for mode in ['large', 'small', 'small_lie']:
            plt.clf()

            # fig, ax1 = plt.subplots(figsize=(5.5,3))
            if mode != 'small_lie':
                plt.figure(figsize=(5,4))
                plt.subplots_adjust(left=0.11, bottom=0.13, right=0.99, top=0.85, wspace=0, hspace=0)
            else:
                plt.figure(figsize=(8,4))
                plt.subplots_adjust(left=0.08, bottom=0.15, right=0.99, top=0.90, wspace=0, hspace=0)


            plt.ylabel('Certified Radius', fontsize=14)
            plt.xlabel('Input Dimension $d$', fontsize=14)
            plt.plot(xs, ys['o'], '--', label='Neyman-Pearson Standard Gaussian')
            for k, v in ys.items():
                if k != 'o' and k != 'i' and k != '10':
                    plt.plot(xs, v, label=f'DSRS w/ $N={k}, \\alpha=0.1\\%$')
            plt.plot(xs, ys['i'], 'b+-', label='DSRS w/ Deterministic Info. (Ideal)')
            prefix = ''
            if mode == 'small':
                prefix = '($y$-Axis Zoomed-In) '
            if mode != 'small_lie':
                plt.title(prefix + f'Certified Radius with\n $P_A={Pa_real}$ and $(1,0.5)$-Concentration', fontsize=16)
            else:
                plt.title(prefix + f'Certified Radius with $P_A={Pa_real}$ and $(1,0.5)$-Concentration', fontsize=16)

            plt.xscale('log')

            if mode == 'small' or mode == 'small_lie':
                plt.ylim([0, max_wo_i * 1.1])
            else:
                plt.ylim([0, max_with_i * 1.1])

            plt.vlines(28*28, 0., 1000., colors='gray', label='MNIST Input Dim.')
            plt.vlines(32*32*3, 0., 1000., colors='darkred', label='CIFAR-10 Input Dim.')
            plt.vlines(224*224*3, 0., 1000., colors='darkgreen', label='ImageNet Input Dim.')

            plt.legend()
            # plt.show()
            OUTF = f'figures/simulation_under_sampling_error_{mode}.pdf'
            plt.savefig(OUTF)

        print(max_wo_i, max_with_i)

